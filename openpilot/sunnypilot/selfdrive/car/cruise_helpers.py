"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

from openpilot.cereal import custom
from opendbc.car.structs import car
from opendbc.car import structs
from opendbc.sunnypilot.car import FOLLOW_DISTANCE_CLOSER_BUTTON, FOLLOW_DISTANCE_FARTHER_BUTTON, TRAFFIC_MODE_BUTTON
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.swaglog import cloudlog
from openpilot.sunnypilot.selfdrive.controls.lib import follow_distance

ButtonType = car.CarState.ButtonEvent.Type
EventNameSP = custom.OnroadEventSP.EventName

DISTANCE_LONG_PRESS = 50

FOLLOW_DISTANCE_EVENTS = (EventNameSP.followDistanceCloser, EventNameSP.followDistanceClose, EventNameSP.followDistanceMedium, EventNameSP.followDistanceFar)
# After a wheel step params_thread leaves the stored Following Distance alone for FOLLOW_DISTANCE_WRITE_HOLD, and after that
# until it reads that step's level back, for at most FOLLOW_DISTANCE_WRITE_HOLD_MAX: the write is non-blocking, and reading
# the old value back before it lands would undo the step for a moment (the personality does: route 00000027 2332.3 s, the
# follow time went 1.125, 1.575, 1.125 s within 0.2 s). Then the stored value wins: a level set elsewhere right after a
# step is taken, and so is the old one if the step's write never lands, so selfdrived shows what plannerd uses
FOLLOW_DISTANCE_WRITE_HOLD = 0.5      # s
FOLLOW_DISTANCE_WRITE_HOLD_MAX = 2.0  # s


class CruiseHelper:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP | None = None):
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

    # Rivian Following Distance: the right thumb wheel steps it instead of cycling the personality. Re-read by
    # selfdrived's params_thread; the value it starts with is not a change: no alert at startup
    self.follow_distance_wheel = CP_SP is not None and follow_distance.available(CP, CP_SP)
    self._follow_distance_read_error_logged = False
    self._follow_distance_write_error_logged = False
    self.follow_distance = self.read_follow_distance() if self.follow_distance_wheel else follow_distance.DEFAULT
    self._follow_distance_read = self.follow_distance  # the stored value params_thread last took, or a wheel step wrote
    self._follow_distance_prev = self.follow_distance  # the level last announced
    self._follow_distance_written = None              # the level the last wheel step wrote, until it is read back
    self._follow_distance_written_t = float('-inf')    # time of the last wheel step's write (time.monotonic)
    self._follow_distance_cs = None  # the last carState whose button events were handled

  def update(self, CS, events, experimental_mode, enabled=False) -> None:
    if self.CP.openpilotLongitudinalControl:
      if CS.cruiseState.available:
        self.update_button_frame_counts(CS)

        # toggle experimental mode once on distance button hold
        self.update_experimental_mode(events, experimental_mode)

      self.update_traffic_mode(CS, events)
      if self.follow_distance_wheel:
        self.update_follow_distance(CS, events, enabled)

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

  def read_follow_distance(self) -> int:
    try:
      return follow_distance.sanitize(self.params.get(follow_distance.PARAM, return_default=True))
    except Exception:
      # params library built without the key, or a failed read: Medium rather than take selfdrived down
      if not self._follow_distance_read_error_logged:
        cloudlog.exception(f"{follow_distance.PARAM} read failed, using Medium")
        self._follow_distance_read_error_logged = True
      return follow_distance.DEFAULT

  def read_follow_distance_param(self, now: float) -> None:
    # selfdrived's params_thread: take a level set in the settings or sunnylink. Only a stored value that differs from the
    # one taken last or written by the wheel, and not while a wheel step's write has yet to land (FOLLOW_DISTANCE_WRITE_HOLD)
    if not self.follow_distance_wheel:
      return
    level = self.read_follow_distance()
    if self._follow_distance_written is not None:
      since = now - self._follow_distance_written_t
      if since <= FOLLOW_DISTANCE_WRITE_HOLD or (level != self._follow_distance_written and since <= FOLLOW_DISTANCE_WRITE_HOLD_MAX):
        return
      self._follow_distance_written = None
    if level != self._follow_distance_read:
      self._follow_distance_read = level
      self.follow_distance = level

  def update_follow_distance(self, CS, events, enabled: bool) -> None:
    # announce every change once, whatever made it: steering wheel, on-device settings or sunnylink
    level = self.follow_distance  # one read: params_thread may update it at any time
    announce = level != self._follow_distance_prev

    # one event per notch of the right thumb wheel, summed and clamped to the levels (no wrap: up always means farther).
    # It changes the level while engaged or stopped, as Rivian's own cruise does. Driving without openpilot, or at an
    # end, a notch only shows the level. Each carState once, as for Traffic Mode: when none arrives within 20 ms,
    # selfdrived passes the previous one (the same object) again
    if CS is not self._follow_distance_cs:
      self._follow_distance_cs = CS
      steps = [1 if be.type == FOLLOW_DISTANCE_FARTHER_BUTTON else -1 for be in CS.buttonEvents
               if be.type in (FOLLOW_DISTANCE_FARTHER_BUTTON, FOLLOW_DISTANCE_CLOSER_BUTTON) and not be.pressed]
      if steps:
        new = min(max(level + sum(steps), 0), len(follow_distance.T_FOLLOW) - 1)
        if (enabled or CS.standstill) and new != level and self.write_follow_distance(new):
          level = new
        announce = True

    if announce:
      events.add(FOLLOW_DISTANCE_EVENTS[level])
      self._follow_distance_prev = level

  def write_follow_distance(self, level: int) -> bool:
    try:
      # without blocking this 100 Hz loop; plannerd reads it back within 0.25 s
      self.params.put(follow_distance.PARAM, int(level))
    except Exception:
      # params library built without the key: the step is ignored rather than take selfdrived down
      if not self._follow_distance_write_error_logged:
        cloudlog.exception(f"{follow_distance.PARAM} write failed, step ignored")
        self._follow_distance_write_error_logged = True
      return False
    self._follow_distance_written = level
    self._follow_distance_written_t = time.monotonic()
    self.follow_distance = level
    self._follow_distance_read = level
    return True
