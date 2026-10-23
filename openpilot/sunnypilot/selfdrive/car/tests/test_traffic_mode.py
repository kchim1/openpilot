"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

from openpilot.cereal import custom, log
from opendbc.car.structs import car
from opendbc.sunnypilot.car import TRAFFIC_MODE_BUTTON
from openpilot.common.params import Params, ParamKeyFlag, UnknownKeyName
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.selfdrived.alertmanager import AlertManager
from openpilot.selfdrive.selfdrived.events import EVENTS, Events
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventsSP, ET
from openpilot.sunnypilot.selfdrive.selfdrived.events_base import Priority

EventName = log.OnroadEvent.EventName
EventNameSP = custom.OnroadEventSP.EventName
ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type
TRAFFIC_MODE_EVENTS = (EventNameSP.trafficModeOn, EventNameSP.trafficModeOff)
TOGGLE = [ButtonEvent(type=TRAFFIC_MODE_BUTTON, pressed=False)]


def make_helper(openpilot_long=True):
  return CruiseHelper(car.CarParams(openpilotLongitudinalControl=openpilot_long))


def step(helper, cruise_available=True, button_events=None, read_params=True, CS=None):
  # what selfdrived does: params_thread re-reads the param, the 100 Hz loop runs CruiseHelper.update
  # with a new carState, or with the same CS object again when none arrived in time
  if read_params:
    helper.traffic_mode = helper.read_traffic_mode()
  if CS is None:
    CS = car.CarState(cruiseState={"available": cruise_available})
    CS.buttonEvents = button_events or []
  events = EventsSP()
  helper.update(CS, events, False)
  return events.names


def current_alerts(alert_types, add_events, frames=400):
  # selfdrived's update_alerts frame by frame, with the alert types of an engaged ([PERMANENT, WARNING]) or a
  # disengaged ([PERMANENT]) openpilot. add_events(frame, events, events_sp) raises the events of that frame
  AM = AlertManager()
  clear_event_types = set() if ET.WARNING in alert_types else {ET.WARNING}
  current = []
  for frame in range(frames):
    events, events_sp = Events(), EventsSP()
    add_events(frame, events, events_sp)
    AM.add_many(frame, events.create_alerts(alert_types) + events_sp.create_alerts(alert_types))
    AM.process_alerts(frame, clear_event_types)
    current.append(AM.current_alert)
  return current


class TestTrafficModeReset(OpenpilotTestCase):
  """Traffic Mode is off again at the end of every drive and when the device starts, but kept when a drive starts, so the
  home screen T tapped while the truck is off still counts for the next drive."""
  def test_off_at_drive_end_and_device_start(self):
    params = Params()
    # what manager does at the end of every drive (the offroad transition) and at start (manager_init clears every flag)
    for flags in ((ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION,),
                  (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                   ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON)):
      params.put_bool("TrafficMode", True, block=True)
      for flag in flags:
        params.clear_all(flag)
      assert params.get_bool("TrafficMode") is False

  def test_kept_when_a_drive_starts(self):
    params = Params()
    for on in (True, False):
      params.put_bool("TrafficMode", on, block=True)
      # ignition on, then the onroad transition
      for flag in (ParamKeyFlag.CLEAR_ON_IGNITION_ON, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION):
        params.clear_all(flag)
      assert params.get_bool("TrafficMode") is on

  def test_not_persistent_or_backed_up(self):
    assert b"TrafficMode" not in Params().all_keys(ParamKeyFlag.PERSISTENT)
    assert b"TrafficMode" not in Params().all_keys(ParamKeyFlag.BACKUP)

  def test_defaults_off(self):
    params = Params()
    params.remove("TrafficMode")
    assert params.get_bool("TrafficMode") is False


class TestTrafficModeAlert(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def test_no_alert_at_startup(self):
    for on in (False, True):
      self.params.put_bool("TrafficMode", on, block=True)
      helper = make_helper()
      for _ in range(10):
        assert step(helper) == []

  def test_one_alert_per_change(self):
    helper = make_helper()
    assert step(helper) == []
    for on, event in ((True, EventNameSP.trafficModeOn), (False, EventNameSP.trafficModeOff), (True, EventNameSP.trafficModeOn)):
      self.params.put_bool("TrafficMode", on, block=True)
      assert step(helper) == [event]
      for _ in range(10):
        assert step(helper) == []

  def test_announced_with_cruise_unavailable(self):
    helper = make_helper()
    self.params.put_bool("TrafficMode", True, block=True)
    assert step(helper, cruise_available=False) == [EventNameSP.trafficModeOn]

  def test_no_alert_without_openpilot_long(self):
    helper = make_helper(openpilot_long=False)
    for on in (True, False):
      self.params.put_bool("TrafficMode", on, block=True)
      assert step(helper) == []

  def test_unregistered_key_reads_off(self, mocker):
    helper = make_helper()
    helper.params = mocker.MagicMock()
    helper.params.get_bool.side_effect = UnknownKeyName(b"TrafficMode")
    assert helper.read_traffic_mode() is False

  def test_alert_shown_disengaged_and_never_blocks_engaging(self):
    for event, text in zip(TRAFFIC_MODE_EVENTS, ("Traffic Mode On", "Traffic Mode Off"), strict=True):
      assert set(EVENTS_SP[event]) == {ET.PERMANENT}

      events = EventsSP()
      events.add(event)
      for et in (ET.NO_ENTRY, ET.SOFT_DISABLE, ET.IMMEDIATE_DISABLE, ET.USER_DISABLE, ET.ENABLE):
        assert not events.contains(et)

      # a disengaged openpilot only presents PERMANENT alerts
      alerts = events.create_alerts([ET.PERMANENT])
      assert len(alerts) == 1
      assert alerts[0].alert_text_1 == text
      assert alerts[0].duration == int(2. / DT_CTRL)

  def test_alert_shown_for_two_seconds(self):
    duration = int(2. / DT_CTRL)
    for alert_types in ([ET.PERMANENT, ET.WARNING], [ET.PERMANENT]):
      for event, text in zip(TRAFFIC_MODE_EVENTS, ("Traffic Mode On", "Traffic Mode Off"), strict=True):
        def add_events(frame, events, events_sp, event=event):
          if frame == 100:
            events_sp.add(event)
        current = current_alerts(alert_types, add_events)
        # AlertManager keeps an alert up to start + duration
        assert [frame for frame, alert in enumerate(current) if alert.alert_text_1 == text] == list(range(100, 100 + duration + 1))

  def test_alert_never_hides_an_active_prompt(self):
    # personalityChanged's LOWER priority: an active LOW prompt like steerSaturated's Take Control stays up, with its
    # promptRepeat sound, for the 2 s of the announcement
    take_control = EVENTS[EventName.steerSaturated][ET.WARNING]
    assert take_control.priority == Priority.LOW
    for event in TRAFFIC_MODE_EVENTS:
      def add_events(frame, events, events_sp, event=event):
        events.add(EventName.steerSaturated)
        if frame == 100:
          events_sp.add(event)
      assert all(alert is take_control for alert in current_alerts([ET.PERMANENT, ET.WARNING], add_events))


class TestTrafficModeButton(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def wait_for_param(self, value):
    # the toggle is a non-blocking write
    for _ in range(200):
      if self.params.get_bool("TrafficMode") == value:
        return True
      time.sleep(0.01)
    return False

  def test_button_flips_param_and_alert_follows_once(self):
    helper = make_helper()
    for on, event in ((True, EventNameSP.trafficModeOn), (False, EventNameSP.trafficModeOff)):
      # the button only writes the param: no alert until params_thread reads it back
      assert step(helper, button_events=TOGGLE, read_params=False) == []
      assert self.wait_for_param(on)
      assert step(helper) == [event]
      for _ in range(10):
        assert step(helper) == []

  def test_reused_carstate_toggles_once(self, mocker):
    # selfdrived passes the previous carState again (the same object) when none arrives within 20 ms. With params_thread
    # reading the new value back in between, handling its toggle again would turn Traffic Mode straight back off
    helper = make_helper()
    put_bool = mocker.patch.object(helper.params, "put_bool", wraps=helper.params.put_bool)
    CS = car.CarState(cruiseState={"available": True}, buttonEvents=TOGGLE)
    alerts = step(helper, CS=CS, read_params=False)
    assert self.wait_for_param(True)
    for _ in range(3):
      alerts += step(helper, CS=CS)
    for _ in range(10):
      alerts += step(helper)
    assert put_bool.call_count == 1
    assert alerts == [EventNameSP.trafficModeOn]
    assert self.params.get_bool("TrafficMode")

  def test_each_carstate_toggles(self, mocker):
    helper = make_helper()
    put_bool = mocker.patch.object(helper.params, "put_bool", wraps=helper.params.put_bool)
    alerts = []
    for on in (True, False):
      alerts += step(helper, button_events=TOGGLE, read_params=False)
      assert self.wait_for_param(on)
      alerts += step(helper)
    assert put_bool.call_count == 2
    assert alerts == list(TRAFFIC_MODE_EVENTS)

  def test_unregistered_key_write_is_ignored(self, mocker):
    # a params library without the key: the toggle is dropped and logged once, and selfdrived keeps running
    helper = make_helper()
    helper.params = mocker.MagicMock()
    helper.params.put_bool.side_effect = UnknownKeyName(b"TrafficMode")
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.car.cruise_helpers.cloudlog")
    for _ in range(3):
      assert step(helper, button_events=TOGGLE, read_params=False) == []
    assert helper.params.put_bool.call_count == 3
    cloudlog.exception.assert_called_once()

  def test_button_works_with_cruise_unavailable(self):
    helper = make_helper()
    step(helper, cruise_available=False, button_events=TOGGLE, read_params=False)
    assert self.wait_for_param(True)

  def test_other_button_events_ignored(self):
    helper = make_helper()
    others = [ButtonEvent(type=TRAFFIC_MODE_BUTTON, pressed=True), ButtonEvent(type=ButtonType.gapAdjustCruise, pressed=False),
              ButtonEvent(type=ButtonType.decelCruise, pressed=False)]
    for _ in range(10):
      assert step(helper, button_events=others) == []
    assert not self.params.get_bool("TrafficMode")

  def test_no_toggle_without_openpilot_long(self):
    helper = make_helper(openpilot_long=False)
    step(helper, button_events=TOGGLE)
    assert not self.wait_for_param(True)

  def test_toggle_leaves_experimental_mode_alone(self):
    self.params.put_bool("ExperimentalMode", False, block=True)
    helper = make_helper()
    assert step(helper, button_events=TOGGLE, read_params=False) == []
    assert self.wait_for_param(True)
    assert not helper.experimental_mode_switched
    assert not self.params.get_bool("ExperimentalMode")
