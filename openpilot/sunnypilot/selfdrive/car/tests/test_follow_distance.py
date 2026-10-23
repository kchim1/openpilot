"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Rivian Following Distance on the selfdrived side: the right thumb wheel's notches step the level (clamped, while engaged
or stopped), each change and each notch shows "Following Distance: X", and params_thread never undoes a step by reading
the old value back before the non-blocking write lands.
"""
import time

from openpilot.cereal import custom, log
from opendbc.car.structs import car
from opendbc.sunnypilot.car import FOLLOW_DISTANCE_CLOSER_BUTTON, FOLLOW_DISTANCE_FARTHER_BUTTON, TRAFFIC_MODE_BUTTON
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP
from openpilot.common.params import Params, ParamKeyFlag, UnknownKeyName
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.events import EVENTS
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper, FOLLOW_DISTANCE_EVENTS, FOLLOW_DISTANCE_WRITE_HOLD, \
  FOLLOW_DISTANCE_WRITE_HOLD_MAX
from openpilot.sunnypilot.selfdrive.car.tests.test_traffic_mode import current_alerts
from openpilot.sunnypilot.selfdrive.controls.lib import follow_distance
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventsSP, ET
from openpilot.sunnypilot.selfdrive.selfdrived.events_base import Priority

EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type
# the alert each level shows; FARTHER and CLOSER below are the wheel's notches
SHOW_CLOSER, SHOW_CLOSE, SHOW_MEDIUM, SHOW_FAR = FOLLOW_DISTANCE_EVENTS
TEXTS = ("Following Distance: Closer", "Following Distance: Close", "Following Distance: Medium", "Following Distance: Far")
FARTHER = ButtonEvent(type=FOLLOW_DISTANCE_FARTHER_BUTTON, pressed=False)
CLOSER = ButtonEvent(type=FOLLOW_DISTANCE_CLOSER_BUTTON, pressed=False)
PARAM = "RivianFollowDistance"


def car_params(brand="rivian", openpilot_long=True, harness=True):
  CP = car.CarParams(brand=brand, openpilotLongitudinalControl=openpilot_long)
  CP_SP = custom.CarParamsSP.new_message(flags=int(RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE) if harness else 0)
  return CP, CP_SP


def make_helper(**kwargs):
  return CruiseHelper(*car_params(**kwargs))


def step(helper, button_events=(), enabled=True, standstill=False, CS=None, read_params=False, now=None):
  # what selfdrived does: params_thread re-reads the param, the 100 Hz loop runs CruiseHelper.update with a new carState,
  # or with the same CS object again when none arrived in time
  if read_params:
    helper.read_follow_distance_param(time.monotonic() if now is None else now)
  if CS is None:
    CS = car.CarState(cruiseState={"available": True}, standstill=standstill)
    CS.buttonEvents = list(button_events)
  events = EventsSP()
  helper.update(CS, events, False, enabled)
  return events.names


class FollowDistanceTestCase(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def wait_for_param(self, value):
    # the wheel's write is non-blocking
    for _ in range(200):
      if self.params.get(PARAM) == value:
        return True
      time.sleep(0.01)
    return False


class TestFollowDistanceParam(FollowDistanceTestCase):
  def test_persistent_backed_up_and_medium_by_default(self):
    assert PARAM.encode() in self.params.all_keys(ParamKeyFlag.PERSISTENT)
    assert PARAM.encode() in self.params.all_keys(ParamKeyFlag.BACKUP)
    assert self.params.get(PARAM, return_default=True) == 2
    assert make_helper().follow_distance == 2

  def test_survives_drive_end_and_reboot(self):
    for level in (0, 3):
      self.params.put(PARAM, level, block=True)
      for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                   ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
        self.params.clear_all(flag)
      assert self.params.get(PARAM) == level


class TestFollowDistanceWheel(FollowDistanceTestCase):
  def test_steps_and_stops_at_the_ends_while_engaged(self):
    helper = make_helper()
    assert step(helper) == []
    assert step(helper, [CLOSER]) == [SHOW_CLOSE]
    assert self.wait_for_param(1)
    assert step(helper, [CLOSER]) == [SHOW_CLOSER]
    assert self.wait_for_param(0)
    # at an end a notch shows the level again, and does not wrap round to Far
    assert step(helper, [CLOSER]) == [SHOW_CLOSER]
    assert helper.follow_distance == 0
    assert step(helper, [FARTHER]) == [SHOW_CLOSE]
    assert step(helper, [FARTHER]) == [SHOW_MEDIUM]
    assert step(helper, [FARTHER]) == [SHOW_FAR]
    # nor round to Closer
    assert step(helper, [FARTHER]) == [SHOW_FAR]
    assert helper.follow_distance == 3
    assert self.wait_for_param(3)
    for _ in range(10):
      assert step(helper, read_params=True) == []
    assert helper.follow_distance == 3

  def test_notches_of_one_frame_are_summed_and_clamped(self, mocker):
    helper = make_helper()
    put = mocker.patch.object(helper.params, "put", wraps=helper.params.put)
    # three farther from Medium: Far, written once, one alert
    assert step(helper, [FARTHER] * 3) == [SHOW_FAR]
    assert put.call_count == 1
    assert step(helper, [CLOSER] * 2) == [SHOW_CLOSE]
    assert put.call_count == 2
    # three closer from Close: Closer
    assert step(helper, [CLOSER] * 3) == [SHOW_CLOSER]
    assert put.call_count == 3
    assert self.wait_for_param(0)

  def test_not_engaged_and_moving_only_shows_the_level(self, mocker):
    helper = make_helper()
    put = mocker.patch.object(helper.params, "put", wraps=helper.params.put)
    for notches in ([CLOSER], [FARTHER], [FARTHER, FARTHER], [CLOSER, CLOSER]):
      assert step(helper, notches, enabled=False) == [SHOW_MEDIUM]
    assert helper.follow_distance == 2
    assert put.call_count == 0

  def test_not_engaged_at_a_standstill_changes_it(self):
    helper = make_helper()
    assert step(helper, [FARTHER], enabled=False, standstill=True) == [SHOW_FAR]
    assert self.wait_for_param(3)
    assert step(helper, [CLOSER], enabled=True, standstill=True) == [SHOW_MEDIUM]
    assert self.wait_for_param(2)

  def test_reused_carstate_steps_once(self, mocker):
    # selfdrived passes the previous carState again (the same object) when none arrives within 20 ms
    helper = make_helper()
    put = mocker.patch.object(helper.params, "put", wraps=helper.params.put)
    CS = car.CarState(cruiseState={"available": True}, buttonEvents=[CLOSER])
    alerts = step(helper, CS=CS)
    for _ in range(3):
      alerts += step(helper, CS=CS, read_params=True)
    assert alerts == [SHOW_CLOSE]
    assert put.call_count == 1
    assert helper.follow_distance == 1
    assert self.wait_for_param(1)

  def test_other_button_events_ignored(self):
    helper = make_helper()
    others = [ButtonEvent(type=FOLLOW_DISTANCE_FARTHER_BUTTON, pressed=True), ButtonEvent(type=FOLLOW_DISTANCE_CLOSER_BUTTON, pressed=True),
              ButtonEvent(type=TRAFFIC_MODE_BUTTON, pressed=True), ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    for _ in range(10):
      assert step(helper, others) == []
    assert helper.follow_distance == 2
    assert self.params.get(PARAM) is None

  def test_works_with_cruise_unavailable(self):
    helper = make_helper()
    CS = car.CarState(cruiseState={"available": False}, buttonEvents=[FARTHER])
    assert step(helper, CS=CS) == [SHOW_FAR]
    assert self.wait_for_param(3)


class TestFollowDistanceAnnounce(FollowDistanceTestCase):
  def test_no_alert_at_startup(self):
    for level in (0, 1, 2, 3):
      self.params.put(PARAM, level, block=True)
      helper = make_helper()
      assert helper.follow_distance == level
      for _ in range(10):
        assert step(helper, read_params=True) == []

  def test_one_alert_per_change_made_elsewhere(self):
    # the comma four settings or sunnylink write the param; params_thread takes it and it is shown once
    helper = make_helper()
    assert step(helper, read_params=True) == []
    for level in (3, 0, 1, 2):
      self.params.put(PARAM, level, block=True)
      assert step(helper, read_params=True) == [FOLLOW_DISTANCE_EVENTS[level]]
      for _ in range(10):
        assert step(helper, read_params=True) == []
      assert helper.follow_distance == level


class TestFollowDistanceParamsThread(FollowDistanceTestCase):
  class LaggingParams:
    """Params.put(block=False): the value lands `lag` s later"""
    def __init__(self, clock, lag):
      self.clock, self.lag, self.value, self.pending = clock, lag, 2, []

    def get(self, key, return_default=False):
      landed = [p for p in self.pending if p[0] <= self.clock[0]]
      for _, value in landed:
        self.value = value
      self.pending = [p for p in self.pending if p[0] > self.clock[0]]
      return self.value

    def put(self, key, value):
      self.pending.append((self.clock[0] + self.lag, value))

  def test_a_stale_read_never_undoes_a_step(self, mocker):
    # the personality's race (route 00000027 2332.3 s): params_thread read the old value back before the write landed
    # and put it back for 0.1 s. Notches at 0.30 s (closer), 0.40 s (back to Medium, before the first write may have
    # landed), 1.01 s (two closer: Closer) and 2.02 s (three farther: Far)
    script = {30: [CLOSER], 40: [FARTHER], 101: [CLOSER] * 2, 202: [FARTHER] * 3}
    for lag in (0.005, 0.03, 0.08, 0.3, 0.6, 1.2):
      clock = [0.]
      mocker.patch("openpilot.sunnypilot.selfdrive.car.cruise_helpers.time.monotonic", side_effect=lambda clock=clock: clock[0])
      helper = make_helper()
      helper.params = self.LaggingParams(clock, lag)
      alerts, levels = [], []
      for i in range(400):
        clock[0] = i * DT_CTRL
        if i % 10 == 3:  # params_thread, every 0.1 s
          helper.read_follow_distance_param(clock[0])
        alerts += [(i, e) for e in step(helper, script.get(i, []))]
        levels.append(helper.follow_distance)
      assert alerts == [(30, SHOW_CLOSE), (40, SHOW_MEDIUM), (101, SHOW_CLOSER), (202, SHOW_FAR)], lag
      assert levels == [2] * 30 + [1] * 10 + [2] * 61 + [0] * 101 + [3] * 198, lag

  def test_held_after_a_wheel_step_then_taken(self, mocker):
    # Far then back to Medium; params_thread reads Far, as if the second write had not landed
    helper = make_helper()
    assert step(helper, [FARTHER]) == [SHOW_FAR]
    assert step(helper, [CLOSER]) == [SHOW_MEDIUM]
    assert self.wait_for_param(2)
    mocker.patch.object(helper.params, "get", return_value=3)
    # times from the write, not the clock: a slow first write in the process must not eat into the hold
    written_t = helper._follow_distance_written_t
    for dt in (0., 0.1, 0.4, 0.6, 1.0, FOLLOW_DISTANCE_WRITE_HOLD_MAX - 0.1):
      assert step(helper, read_params=True, now=written_t + dt) == []
    assert helper.follow_distance == 2
    # only for FOLLOW_DISTANCE_WRITE_HOLD_MAX: a value still stored after it was set elsewhere, and is taken
    assert step(helper, read_params=True, now=written_t + FOLLOW_DISTANCE_WRITE_HOLD_MAX + 0.1) == [SHOW_FAR]

  def test_taken_once_the_step_is_read_back(self, mocker):
    helper = make_helper()
    assert step(helper, [FARTHER]) == [SHOW_FAR]
    assert self.wait_for_param(3)
    written_t = helper._follow_distance_written_t
    # read back within FOLLOW_DISTANCE_WRITE_HOLD it is still held; after it, read back, the hold ends
    for dt in (0.1, FOLLOW_DISTANCE_WRITE_HOLD + 0.1):
      assert step(helper, read_params=True, now=written_t + dt) == []
    # so a level then set elsewhere (Closer, in the settings) is taken at once
    mocker.patch.object(helper.params, "get", return_value=0)
    assert step(helper, read_params=True, now=written_t + FOLLOW_DISTANCE_WRITE_HOLD + 0.2) == [SHOW_CLOSER]

  def test_a_level_set_elsewhere_right_after_a_step_is_taken(self, mocker):
    # Far on the wheel and its write lands, then Medium, the level before the step, set in the settings during the hold:
    # taken once the hold ends, so selfdrived shows what plannerd uses
    helper = make_helper()
    assert step(helper, [FARTHER]) == [SHOW_FAR]
    assert self.wait_for_param(3)
    written_t = helper._follow_distance_written_t
    mocker.patch.object(helper.params, "get", return_value=2)
    for dt in (0.1, 0.4, 1.0):
      assert step(helper, read_params=True, now=written_t + dt) == []
    assert step(helper, read_params=True, now=written_t + FOLLOW_DISTANCE_WRITE_HOLD_MAX + 0.1) == [SHOW_MEDIUM]

  def test_a_write_that_never_lands_falls_back_to_the_stored_level(self, mocker):
    helper = make_helper()
    mocker.patch.object(helper.params, "put")  # lost
    assert step(helper, [FARTHER]) == [SHOW_FAR]
    written_t = helper._follow_distance_written_t
    assert step(helper, read_params=True, now=written_t + 1.) == []
    assert step(helper, read_params=True, now=written_t + FOLLOW_DISTANCE_WRITE_HOLD_MAX + 0.1) == [SHOW_MEDIUM]

  def test_unknown_key_or_garbage_reads_medium_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.car.cruise_helpers.cloudlog")
    helper = make_helper()
    helper.params = mocker.MagicMock()
    helper.params.get.side_effect = UnknownKeyName(PARAM.encode())
    for _ in range(3):
      assert helper.read_follow_distance() == 2
    assert cloudlog.exception.call_count == 1
    helper.params.get.side_effect = None
    for stored, level in (("abc", 2), (None, 2), (7, 3), (-3, 0), (0, 0), (3, 3)):
      helper.params.get.return_value = stored
      assert helper.read_follow_distance() == level

  def test_write_failure_ignored_and_logged_once(self, mocker):
    # a params library without the key: the level stays (and is shown), and selfdrived keeps running
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.car.cruise_helpers.cloudlog")
    helper = make_helper()
    helper.params = mocker.MagicMock()
    helper.params.put.side_effect = UnknownKeyName(PARAM.encode())
    for _ in range(3):
      assert step(helper, [FARTHER]) == [SHOW_MEDIUM]
    assert helper.follow_distance == 2
    assert helper.params.put.call_count == 3
    assert cloudlog.exception.call_count == 1


class TestFollowDistanceAlerts(OpenpilotTestCase):
  def test_shown_disengaged_and_never_blocks_engaging(self):
    # one per level, in the levels' order
    assert TEXTS == tuple(f"Following Distance: {name}" for name in follow_distance.NAMES)
    assert len(set(FOLLOW_DISTANCE_EVENTS)) == len(follow_distance.T_FOLLOW)
    for event, text in zip(FOLLOW_DISTANCE_EVENTS, TEXTS, strict=True):
      assert set(EVENTS_SP[event]) == {ET.PERMANENT}
      events = EventsSP()
      events.add(event)
      for et in (ET.NO_ENTRY, ET.SOFT_DISABLE, ET.IMMEDIATE_DISABLE, ET.USER_DISABLE, ET.ENABLE):
        assert not events.contains(et)
      alerts = events.create_alerts([ET.PERMANENT])
      assert len(alerts) == 1
      assert alerts[0].alert_text_1 == text
      assert alerts[0].priority == Priority.LOWER
      assert alerts[0].duration == int(1.5 / DT_CTRL)

  def test_shown_for_one_and_a_half_seconds_engaged_or_not(self):
    duration = int(1.5 / DT_CTRL)
    for alert_types in ([ET.PERMANENT, ET.WARNING], [ET.PERMANENT]):
      for event, text in zip(FOLLOW_DISTANCE_EVENTS, TEXTS, strict=True):
        def add_events(frame, events, events_sp, event=event):
          if frame == 100:
            events_sp.add(event)
        current = current_alerts(alert_types, add_events)
        assert [frame for frame, alert in enumerate(current) if alert.alert_text_1 == text] == list(range(100, 100 + duration + 1))

  def test_a_notch_at_an_end_shows_it_again(self):
    duration = int(1.5 / DT_CTRL)
    for event, text in ((SHOW_CLOSER, TEXTS[0]), (SHOW_FAR, TEXTS[3])):
      def add_events(frame, events, events_sp, event=event):
        if frame in (100, 200):
          events_sp.add(event)
      current = current_alerts([ET.PERMANENT], add_events)
      assert [frame for frame, alert in enumerate(current) if alert.alert_text_1 == text] == list(range(100, 200 + duration + 1))

  def test_the_newest_level_is_shown(self):
    def add_events(frame, events, events_sp):
      if frame in (100, 120, 140, 160):
        events_sp.add(FOLLOW_DISTANCE_EVENTS[(frame - 100) // 20])
    current = current_alerts([ET.PERMANENT], add_events)
    assert current[110].alert_text_1 == TEXTS[0]
    assert current[130].alert_text_1 == TEXTS[1]
    assert current[150].alert_text_1 == TEXTS[2]
    assert current[170].alert_text_1 == TEXTS[3]

  def test_never_hides_an_active_prompt(self):
    take_control = EVENTS[EventName.steerSaturated][ET.WARNING]
    assert take_control.priority == Priority.LOW
    for event in FOLLOW_DISTANCE_EVENTS:
      def add_events(frame, events, events_sp, event=event):
        events.add(EventName.steerSaturated)
        if frame == 100:
          events_sp.add(event)
      assert all(alert is take_control for alert in current_alerts([ET.PERMANENT, ET.WARNING], add_events))


class TestFollowDistanceScope(FollowDistanceTestCase):
  def test_only_rivian_with_the_harness_and_openpilot_long(self):
    # selfdrived skips the personality cycle on gapAdjustCruise exactly when this is set
    assert make_helper().follow_distance_wheel
    for kwargs in ({"harness": False}, {"openpilot_long": False}, {"brand": "toyota"}, {"brand": "honda", "harness": False}):
      assert not make_helper(**kwargs).follow_distance_wheel, kwargs
    assert not CruiseHelper(car.CarParams(brand="rivian", openpilotLongitudinalControl=True)).follow_distance_wheel

  def test_other_brands_ignore_the_wheel_events(self, mocker):
    for kwargs in ({"brand": "toyota"}, {"harness": False}):
      helper = make_helper(**kwargs)
      put = mocker.patch.object(helper.params, "put", wraps=helper.params.put)
      for notches in ([CLOSER], [FARTHER], [CLOSER, CLOSER]):
        assert step(helper, notches) == []
        assert step(helper, notches, enabled=False, standstill=True) == []
      self.params.put(PARAM, 3, block=True)
      assert step(helper, read_params=True) == []
      assert put.call_count == 0
      self.params.remove(PARAM)

  def test_wheel_events_never_touch_the_personality(self):
    self.params.put("LongitudinalPersonality", log.LongitudinalPersonality.aggressive, block=True)
    helper = make_helper()
    for notches in ([FARTHER], [CLOSER], [CLOSER], [FARTHER] * 3):
      step(helper, notches)
    assert self.wait_for_param(3)
    assert self.params.get("LongitudinalPersonality") == log.LongitudinalPersonality.aggressive
    assert not helper.experimental_mode_switched
