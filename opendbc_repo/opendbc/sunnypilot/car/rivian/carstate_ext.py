"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from enum import StrEnum

from opendbc.car import Bus, structs
from opendbc.can.parser import CANParser
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.rivian.values import DBC
from opendbc.sunnypilot.car import FOLLOW_DISTANCE_CLOSER_BUTTON, FOLLOW_DISTANCE_FARTHER_BUTTON, TRAFFIC_MODE_BUTTON
from opendbc.sunnypilot.car.rivian.acc_fault_recorder import AccFaultRecorder
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

MAX_SET_SPEED = 85 * CV.MPH_TO_MS
MIN_SET_SPEED = 20 * CV.MPH_TO_MS

# Holding the right thumb control's middle button (pressing the scroll wheel in) this long toggles Traffic Mode.
# carState is updated at 100 Hz (card), so 100 frames is 1.0 s
TRAFFIC_MODE_HOLD_FRAMES = 100

# Right thumb wheel (RightButton_Scroll): a position counter, +1 a notch one way and -1 the other, wrapping through 255
# (= -1: route 00000017 612.36 s, one notch from 0 read 255 and stayed). 0 when the truck wakes, but it keeps counting
# while the truck is awake, so the first frame received is only a baseline.
# The sign of a notch UP is not in any log. The left wheel (volume) climbs in single notches and drops in quick runs, as
# volume up/down would, so +1 is likely; checked on the first drive (one notch up must show "Following Distance: Far")
RIGHT_SCROLL_UP_SIGN = 1
SCROLL_MAX_STEP = 5  # notches in one 50 ms frame (3 seen); more is a glitch or a reset: resync, no request


class CarStateExt:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.set_speed = 10
    self.increase_button = False
    self.decrease_button = False
    self.right_scroll_prev = None
    self.increase_counter = 0
    self.decrease_counter = 0
    self.traffic_mode_button = False
    self.traffic_mode_counter = 0
    self.acc_fault_recorder = AccFaultRecorder()

  def update_longitudinal_upgrade(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_park = can_parsers[Bus.alt]
    cp_adas = can_parsers[Bus.adas]
    cp = can_parsers[Bus.pt]

    prev_increase_button = self.increase_button
    prev_decrease_button = self.decrease_button

    if self.CP.openpilotLongitudinalControl:
      button_events = []

      # the scroll wheel pressed in: the Traffic Mode button below
      self.traffic_mode_button = cp_park.vl["WheelButtons_Fwd"]["RightButton_ScrollClick"] == 2

      # the scroll wheel sets the Following Distance: one event per notch, up = farther (Rivian's own cruise: thumb control
      # up = increase following distance). Counted from the first frame received (the parser reads 0 until then). Not
      # while it is pressed in: pressing it in for Traffic Mode can also roll it a notch
      right_scroll = int(cp_park.vl["WheelButtons_Fwd"]["RightButton_Scroll"])
      if cp_park.ts_nanos["WheelButtons_Fwd"]["RightButton_Scroll"]:
        if self.right_scroll_prev is not None:
          step = ((right_scroll - self.right_scroll_prev + 128) % 256) - 128
          if step != 0 and abs(step) <= SCROLL_MAX_STEP and not self.traffic_mode_button:
            btn = FOLLOW_DISTANCE_FARTHER_BUTTON if step * RIGHT_SCROLL_UP_SIGN > 0 else FOLLOW_DISTANCE_CLOSER_BUTTON
            button_events += [structs.CarState.ButtonEvent(pressed=False, type=btn) for _ in range(abs(step))]
        self.right_scroll_prev = right_scroll

      # button logic for set-speed
      self.increase_button = cp_park.vl["WheelButtons_Fwd"]["RightButton_RightClick"] == 2
      self.decrease_button = cp_park.vl["WheelButtons_Fwd"]["RightButton_LeftClick"] == 2

      self.increase_counter = self.increase_counter + 1 if self.increase_button else 0
      self.decrease_counter = self.decrease_counter + 1 if self.decrease_button else 0

      metric = cp_adas.vl["Cluster"]["Cluster_Unit"] == 0
      conversion = CV.KPH_TO_MS if metric else CV.MPH_TO_MS
      long_press_step = 10.0 if metric else 5.0
      set_speed_converted = self.set_speed * (CV.MS_TO_KPH if metric else CV.MS_TO_MPH)

      if self.increase_button:
        if self.increase_counter % 66 == 0:
          self.set_speed = (int(math.ceil((set_speed_converted + 1) / long_press_step)) * long_press_step) * conversion
        elif not prev_increase_button:
          self.set_speed += conversion

      if self.decrease_button:
        if self.decrease_counter % 66 == 0:
          self.set_speed = (int(math.floor((set_speed_converted - 1) / long_press_step)) * long_press_step) * conversion
        elif not prev_decrease_button:
          self.set_speed -= conversion

      # Traffic Mode: the middle button (the scroll wheel pressed in) held for TRAFFIC_MODE_HOLD_FRAMES sends one toggle
      # when the hold is reached. Holding longer, a shorter press and the release do nothing here. The truck still does
      # whatever it does on that button itself: openpilot only reads it
      self.traffic_mode_counter = self.traffic_mode_counter + 1 if self.traffic_mode_button else 0
      if self.traffic_mode_counter == TRAFFIC_MODE_HOLD_FRAMES:
        button_events.append(structs.CarState.ButtonEvent(pressed=False, type=TRAFFIC_MODE_BUTTON))

      if not ret.cruiseState.enabled:
        self.set_speed = ret.vEgoCluster

      # VDM_UserAdasRequest: 0=IDLE, 1=UP_1, 2=UP_2, 3=DOWN_1, 4=DOWN_2
      stalk_down = int(cp.vl["VDM_AdasSts"]["VDM_UserAdasRequest"]) in (3, 4)
      if stalk_down:
        # Mimic Rivian ACC: stalk down sets speed to current speed (never decreases)
        self.set_speed = max(self.set_speed, ret.vEgoCluster)

      self.set_speed = max(MIN_SET_SPEED, min(self.set_speed, MAX_SET_SPEED))
      ret.cruiseState.speed = self.set_speed
      ret.buttonEvents = button_events

    if self.CP.enableBsm:
      ret.leftBlindspot = cp_park.vl["BSM_BlindSpotIndicator_Fwd"]["BSM_BlindSpotIndicator_Left"] != 0
      ret.rightBlindspot = cp_park.vl["BSM_BlindSpotIndicator_Fwd"]["BSM_BlindSpotIndicator_Right"] != 0

  def update(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    if self.CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE:
      self.update_longitudinal_upgrade(ret, can_parsers)

    # read only: keeps the last half second of the cruise request and the truck's answer, logged on a cruise fault
    self.acc_fault_recorder.update(ret, can_parsers)

  @staticmethod
  def get_parser(CP, CP_SP) -> dict[StrEnum, CANParser]:
    messages = {}

    if CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE:
      messages[Bus.alt] = CANParser(DBC[CP.carFingerprint][Bus.alt], [], 1)

    return messages
