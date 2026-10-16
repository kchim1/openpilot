import unittest
from types import SimpleNamespace

from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.sunnypilot.car import TRAFFIC_MODE_BUTTON
from opendbc.sunnypilot.car.rivian.carstate_ext import CarStateExt, TRAFFIC_MODE_HOLD_FRAMES
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type

SET_SPEED_MPH = 50
TOGGLE = (TRAFFIC_MODE_BUTTON, False)
SCROLL = (ButtonType.gapAdjustCruise, False)


def button_events(ret):
  return [(be.type, be.pressed) for be in ret.buttonEvents]


class TestSetSpeedButtons(unittest.TestCase):
  """Right thumb control with the longitudinal harness upgrade: left button = set speed down, right button = set speed up."""

  def setUp(self):
    self.cs = self._make()
    self.scroll = 0

  @staticmethod
  def _make(openpilot_long=True):
    CP = structs.CarParams(openpilotLongitudinalControl=openpilot_long)
    CP_SP = structs.CarParamsSP(flags=int(RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE))
    cs = CarStateExt(CP, CP_SP)
    cs.set_speed = SET_SPEED_MPH * CV.MPH_TO_MS
    return cs

  def _frame(self, decrease=False, increase=False, metric=False, cruise_enabled=True):
    wheel = {"RightButton_LeftClick": 2 if decrease else 0, "RightButton_RightClick": 2 if increase else 0, "RightButton_Scroll": self.scroll}
    can_parsers = {
      Bus.alt: SimpleNamespace(vl={"WheelButtons_Fwd": wheel}),
      Bus.adas: SimpleNamespace(vl={"Cluster": {"Cluster_Unit": 0 if metric else 1}}),
      Bus.pt: SimpleNamespace(vl={"VDM_AdasSts": {"VDM_UserAdasRequest": 0}}),
    }
    ret = structs.CarState(vEgoCluster=20.)
    ret.cruiseState.enabled = cruise_enabled
    self.cs.update(ret, can_parsers)
    return ret

  def _hold(self, frames, **kwargs):
    events = []
    for _ in range(frames):
      ret = self._frame(decrease=True, **kwargs)
      events += button_events(ret)
      self.assertAlmostEqual(ret.cruiseState.speed, SET_SPEED_MPH * CV.MPH_TO_MS, places=5)  # never changes while held
    return events

  def _set_speed_mph(self):
    return self.cs.set_speed * CV.MS_TO_MPH

  def test_tap_lowers_set_speed_on_release(self):
    for frames in (1, 5, TRAFFIC_MODE_HOLD_FRAMES - 1):
      with self.subTest(frames=frames):
        self.setUp()
        self.assertEqual(self._hold(frames), [])
        self.assertEqual(button_events(self._frame()), [])
        self.assertAlmostEqual(self._set_speed_mph(), SET_SPEED_MPH - 1)

  def test_tap_metric_lowers_by_one_kph(self):
    self._hold(3, metric=True)
    self._frame(metric=True)
    self.assertAlmostEqual(self.cs.set_speed, SET_SPEED_MPH * CV.MPH_TO_MS - CV.KPH_TO_MS)

  def test_hold_one_second_toggles_once(self):
    events = self._hold(TRAFFIC_MODE_HOLD_FRAMES - 1)
    self.assertEqual(events, [])
    # the toggle goes out on the frame the hold reaches 1 s, while the button is still down
    self.assertEqual(button_events(self._frame(decrease=True)), [TOGGLE])
    self.assertEqual(button_events(self._frame()), [])
    self.assertAlmostEqual(self._set_speed_mph(), SET_SPEED_MPH)

  def test_long_hold_does_not_repeat(self):
    self.assertEqual(self._hold(10 * TRAFFIC_MODE_HOLD_FRAMES), [TOGGLE])
    # no 5 mph long press steps, and the release does nothing
    self.assertEqual(button_events(self._frame()), [])
    self.assertAlmostEqual(self._set_speed_mph(), SET_SPEED_MPH)

  def test_toggle_while_cruise_off(self):
    # set speed follows vEgoCluster as before; the toggle still works
    events = []
    for _ in range(TRAFFIC_MODE_HOLD_FRAMES):
      ret = self._frame(decrease=True, cruise_enabled=False)
      events += button_events(ret)
      self.assertAlmostEqual(ret.cruiseState.speed, 20., places=5)
    self.assertEqual(events, [TOGGLE])

  def test_press_started_with_cruise_off_leaves_set_speed(self):
    # pressed with cruise off (set speed follows vEgoCluster), cruise engaged while the button is still down, then
    # released: cruise keeps the set speed it was engaged at
    for frames_off, frames_on in ((1, 1), (30, 20), (5, TRAFFIC_MODE_HOLD_FRAMES - 10)):
      with self.subTest(frames_off=frames_off, frames_on=frames_on):
        self.setUp()
        for _ in range(frames_off):
          self._frame(decrease=True, cruise_enabled=False)
        for _ in range(frames_on):
          self._frame(decrease=True)
        ret = self._frame()
        self.assertEqual(button_events(ret), [])
        self.assertAlmostEqual(ret.cruiseState.speed, 20., places=5)
        # the next tap with cruise enabled lowers it as usual
        self._frame(decrease=True)
        self.assertAlmostEqual(self._frame().cruiseState.speed, 20. - CV.MPH_TO_MS, places=5)

  def test_press_with_cruise_off_in_between_leaves_set_speed(self):
    # cruise briefly off during the press: the set speed follows vEgoCluster, and the release does not lower it
    for _ in range(10):
      self._frame(decrease=True)
    self._frame(decrease=True, cruise_enabled=False)
    for _ in range(10):
      self._frame(decrease=True)
    self.assertAlmostEqual(self._frame().cruiseState.speed, 20., places=5)

  def test_increase_button_unchanged(self):
    # +1 on press, then each 66 frames held jumps up to the next multiple of 5
    expected = {1: 51, 65: 51, 66: 55, 131: 55, 132: 60}
    for frame in range(1, 133):
      self.assertEqual(button_events(self._frame(increase=True)), [])
      if frame in expected:
        self.assertAlmostEqual(self._set_speed_mph(), expected[frame], msg=f"{frame=}")
    self._frame()
    self.assertAlmostEqual(self._set_speed_mph(), 60)

  def test_scroll_and_toggle_in_same_frame(self):
    self._hold(TRAFFIC_MODE_HOLD_FRAMES - 1)
    self.scroll += 1
    self.assertEqual(button_events(self._frame(decrease=True)), [SCROLL, TOGGLE])

  def test_scroll_alone(self):
    self.scroll += 1
    self.assertEqual(button_events(self._frame()), [SCROLL])
    self.assertEqual(button_events(self._frame()), [])

  def test_no_events_without_openpilot_long(self):
    self.cs = self._make(openpilot_long=False)
    events = []
    for _ in range(2 * TRAFFIC_MODE_HOLD_FRAMES):
      events += button_events(self._frame(decrease=True))
    self.assertEqual(events, [])


if __name__ == "__main__":
  unittest.main()
