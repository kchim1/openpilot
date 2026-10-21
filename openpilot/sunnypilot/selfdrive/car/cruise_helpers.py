"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.cereal import custom
from opendbc.car.structs import car
from opendbc.car import structs
from opendbc.sunnypilot.car import TRAFFIC_MODE_BUTTON
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.swaglog import cloudlog

ButtonType = car.CarState.ButtonEvent.Type
EventNameSP = custom.OnroadEventSP.EventName

DISTANCE_LONG_PRESS = 50


class CruiseHelper:
  def __init__(self, CP: structs.CarParams):
    self.CP = CP
    self.params = Params()

    self.button_frame_counts = {ButtonType.gapAdjustCruise: 0}
    self._experimental_mode = False
    self.experimental_mode_switched = False

    # Traffic Mode, re-read by selfdrived's params_thread. The value it starts with is not a change: no alert at startup
    self.traffic_mode = self.read_traffic_mode()
    self._traffic_mode_prev = self.traffic_mode
    self._traffic_mode_cs = None  # the last carState whose button events were handled
    self._traffic_mode_write_error_logged = False

  def update(self, CS, events, experimental_mode) -> None:
    if self.CP.openpilotLongitudinalControl:
      if CS.cruiseState.available:
        self.update_button_frame_counts(CS)

        # toggle experimental mode once on distance button hold
        self.update_experimental_mode(events, experimental_mode)

      self.update_traffic_mode(CS, events)

  def update_button_frame_counts(self, CS) -> None:
    for button in self.button_frame_counts:
      if self.button_frame_counts[button] > 0:
        self.button_frame_counts[button] += 1

    for button_event in CS.buttonEvents:
      button = button_event.type.raw
      if button in self.button_frame_counts:
        self.button_frame_counts[button] = int(button_event.pressed)

  def update_experimental_mode(self, events, experimental_mode) -> None:
    if self.button_frame_counts[ButtonType.gapAdjustCruise] >= DISTANCE_LONG_PRESS and not self.experimental_mode_switched:
      self._experimental_mode = not experimental_mode
      self.params.put_bool("ExperimentalMode", self._experimental_mode)
      events.add(EventNameSP.experimentalModeSwitched)
      self.experimental_mode_switched = True

  def read_traffic_mode(self) -> bool:
    try:
      return self.params.get_bool("TrafficMode")
    except UnknownKeyName:
      return False

  def update_traffic_mode(self, CS, events) -> None:
    # announce every change once, whatever made it: steering wheel, on-device settings or sunnylink
    traffic_mode = self.traffic_mode  # one read: params_thread may update it at any time
    if traffic_mode != self._traffic_mode_prev:
      events.add(EventNameSP.trafficModeOn if traffic_mode else EventNameSP.trafficModeOff)
      self._traffic_mode_prev = traffic_mode

    # one TRAFFIC_MODE_BUTTON event per toggle (Rivian: the right thumb control's middle button held 1 s). Only flip the param, without
    # blocking this 100 Hz loop: the alert above follows once params_thread reads the new value back.
    # Each carState is handled once: when no new one arrives within 20 ms, selfdrived passes the previous one again
    # (the same object, CS_prev), and flipping again would undo the toggle if params_thread has read it back meanwhile.
    # The reference kept here keeps that object alive, so a new carState can never be the same object
    if CS is not self._traffic_mode_cs:
      self._traffic_mode_cs = CS
      if any(be.type == TRAFFIC_MODE_BUTTON and not be.pressed for be in CS.buttonEvents):
        self.write_traffic_mode(not traffic_mode)

  def write_traffic_mode(self, traffic_mode: bool) -> None:
    try:
      self.params.put_bool("TrafficMode", traffic_mode)
    except Exception:
      # params library built without the key: ignore the toggle rather than take selfdrived down
      if not self._traffic_mode_write_error_logged:
        cloudlog.exception("TrafficMode write failed, toggle ignored")
        self._traffic_mode_write_error_logged = True
