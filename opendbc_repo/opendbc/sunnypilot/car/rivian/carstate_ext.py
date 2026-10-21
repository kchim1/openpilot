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
from opendbc.sunnypilot.car import TRAFFIC_MODE_BUTTON
from opendbc.sunnypilot.car.rivian.acc_fault_recorder import AccFaultRecorder
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type

MAX_SET_SPEED = 85 * CV.MPH_TO_MS
MIN_SET_SPEED = 20 * CV.MPH_TO_MS

# Holding the set speed down button this long toggles Traffic Mode instead of lowering the set speed.
# carState is updated at 100 Hz (card), so 100 frames is 1.0 s
TRAFFIC_MODE_HOLD_FRAMES = 100


class CarStateExt:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP):
    self.CP = CP
    self.CP_SP = CP_SP

    self.set_speed = 10
    self.increase_button = False
    self.decrease_button = False
    self.distance_button = 0
    self.increase_counter = 0
    self.decrease_counter = 0
    self.decrease_press_enabled = False
    self.acc_fault_recorder = AccFaultRecorder()

  def update_longitudinal_upgrade(self, ret: structs.CarState, can_parsers: dict[StrEnum, CANParser]) -> None:
    cp_park = can_parsers[Bus.alt]
    cp_adas = can_parsers[Bus.adas]
    cp = can_parsers[Bus.pt]

    prev_increase_button = self.increase_button
    prev_decrease_button = self.decrease_button
    prev_decrease_counter = self.decrease_counter

    if self.CP.openpilotLongitudinalControl:
      button_events = []

      # distance scroll wheel
      right_scroll = cp_park.vl["WheelButtons_Fwd"]["RightButton_Scroll"]
      if right_scroll != 255:
        if self.distance_button != right_scroll:
          button_events.append(structs.CarState.ButtonEvent(pressed=False, type=ButtonType.gapAdjustCruise))
        self.distance_button = right_scroll

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

      # set speed down: a tap lowers the set speed by 1 when it is released. Held for TRAFFIC_MODE_HOLD_FRAMES, it sends
      # one Traffic Mode toggle when the hold is reached and leaves the set speed alone (no long press steps).
      # Only a press made entirely with cruise enabled lowers it: one that starts with cruise off (set speed following
      # vEgoCluster) must not lower the set speed cruise was engaged at while the button was held
      if self.decrease_button:
        if not prev_decrease_button:
          self.decrease_press_enabled = ret.cruiseState.enabled
        if self.decrease_counter == TRAFFIC_MODE_HOLD_FRAMES:
          button_events.append(structs.CarState.ButtonEvent(pressed=False, type=TRAFFIC_MODE_BUTTON))
      elif prev_decrease_button and prev_decrease_counter < TRAFFIC_MODE_HOLD_FRAMES and self.decrease_press_enabled:
        self.set_speed -= conversion

      if not ret.cruiseState.enabled:
        self.set_speed = ret.vEgoCluster
        self.decrease_press_enabled = False

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
