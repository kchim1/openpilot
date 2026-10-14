"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time

from openpilot.cereal import custom
from opendbc.car.structs import car
from opendbc.sunnypilot.car import TRAFFIC_MODE_BUTTON
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_CTRL
from openpilot.common.test import OpenpilotTestCase
from openpilot.sunnypilot.selfdrive.car.cruise_helpers import CruiseHelper
from openpilot.sunnypilot.selfdrive.selfdrived.events import EVENTS_SP, EventsSP, ET

EventNameSP = custom.OnroadEventSP.EventName
ButtonEvent = car.CarState.ButtonEvent
ButtonType = car.CarState.ButtonEvent.Type
TRAFFIC_MODE_EVENTS = (EventNameSP.trafficModeOn, EventNameSP.trafficModeOff)
TOGGLE = [ButtonEvent(type=TRAFFIC_MODE_BUTTON, pressed=False)]


def make_helper(openpilot_long=True):
  return CruiseHelper(car.CarParams(openpilotLongitudinalControl=openpilot_long))


def step(helper, cruise_available=True, button_events=None, read_params=True):
  # what selfdrived does: params_thread re-reads the param, the 100 Hz loop runs CruiseHelper.update
  if read_params:
    helper.traffic_mode = helper.read_traffic_mode()
  CS = car.CarState(cruiseState={"available": cruise_available})
  CS.buttonEvents = button_events or []
  events = EventsSP()
  helper.update(CS, events, False)
  return events.names


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
