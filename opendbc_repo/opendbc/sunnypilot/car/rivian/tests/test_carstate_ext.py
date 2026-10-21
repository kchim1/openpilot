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
  """Right thumb control with the longitudinal harness upgrade: left = set speed down, right = set speed up, middle (the
  scroll wheel pressed in) held 1 s = Traffic Mode toggle."""

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

  def _frame(self, decrease=False, increase=False, middle=False, metric=False, cruise_enabled=True):
    # buttons read 1 at rest and 2 while pressed (routes 0000001d/1e)
    wheel = {"RightButton_LeftClick": 2 if decrease else 1, "RightButton_RightClick": 2 if increase else 1,
             "RightButton_ScrollClick": 2 if middle else 1, "RightButton_Scroll": self.scroll}
    can_parsers = {
      Bus.alt: SimpleNamespace(vl={"WheelButtons_Fwd": wheel}),
      Bus.adas: SimpleNamespace(vl={"Cluster": {"Cluster_Unit": 0 if metric else 1}}),
      Bus.pt: SimpleNamespace(vl={"VDM_AdasSts": {"VDM_UserAdasRequest": 0}}),
    }
    ret = structs.CarState(vEgoCluster=20.)
    ret.cruiseState.enabled = cruise_enabled
    self.cs.update(ret, can_parsers)
    return ret

  def _hold_middle(self, frames, **kwargs):
    events = []
    for _ in range(frames):
      ret = self._frame(middle=True, **kwargs)
      events += button_events(ret)
      if kwargs.get("cruise_enabled", True):
        self.assertAlmostEqual(ret.cruiseState.speed, SET_SPEED_MPH * CV.MPH_TO_MS, places=5)  # never changes the set speed
    return events

  def _set_speed_mph(self):
    return self.cs.set_speed * CV.MS_TO_MPH

  def test_middle_hold_one_second_toggles_once(self):
    self.assertEqual(self._hold_middle(TRAFFIC_MODE_HOLD_FRAMES - 1), [])
    # the toggle goes out on the frame the hold reaches 1 s, while the button is still down
    self.assertEqual(button_events(self._frame(middle=True)), [TOGGLE])
    self.assertEqual(button_events(self._frame()), [])
    self.assertAlmostEqual(self._set_speed_mph(), SET_SPEED_MPH)

  def test_middle_short_press_does_nothing(self):
    for frames in (1, 5, TRAFFIC_MODE_HOLD_FRAMES - 1):
      with self.subTest(frames=frames):
        self.setUp()
        self.assertEqual(self._hold_middle(frames), [])
        self.assertEqual(button_events(self._frame()), [])
        self.assertAlmostEqual(self._set_speed_mph(), SET_SPEED_MPH)

  def test_middle_long_hold_does_not_repeat(self):
    self.assertEqual(self._hold_middle(10 * TRAFFIC_MODE_HOLD_FRAMES), [TOGGLE])
    self.assertEqual(button_events(self._frame()), [])
    # a second hold toggles again
    self.assertEqual(self._hold_middle(TRAFFIC_MODE_HOLD_FRAMES), [TOGGLE])

  def test_middle_toggle_while_cruise_off(self):
    self.assertEqual(self._hold_middle(TRAFFIC_MODE_HOLD_FRAMES, cruise_enabled=False), [TOGGLE])

  def test_decrease_button_steps_down_again(self):
    # -1 on press, then each 66 frames held drops to the next multiple of 5 below; holding it never toggles Traffic Mode
    expected = {1: 49, 65: 49, 66: 45, 131: 45, 132: 40}
    for frame in range(1, 3 * TRAFFIC_MODE_HOLD_FRAMES):
      self.assertEqual(button_events(self._frame(decrease=True)), [])
      if frame in expected:
        self.assertAlmostEqual(self._set_speed_mph(), expected[frame], msg=f"{frame=}")
    self.assertEqual(button_events(self._frame()), [])

  def test_decrease_tap_metric_lowers_by_one_kph(self):
    self._frame(decrease=True, metric=True)
    self._frame(metric=True)
    self.assertAlmostEqual(self.cs.set_speed, SET_SPEED_MPH * CV.MPH_TO_MS - CV.KPH_TO_MS)

  def test_decrease_with_cruise_off_follows_speed(self):
    ret = self._frame(decrease=True, cruise_enabled=False)
    self.assertAlmostEqual(ret.cruiseState.speed, 20., places=5)

  def test_increase_button_unchanged(self):
    # +1 on press, then each 66 frames held jumps up to the next multiple of 5
    expected = {1: 51, 65: 51, 66: 55, 131: 55, 132: 60}
    for frame in range(1, 133):
      self.assertEqual(button_events(self._frame(increase=True)), [])
      if frame in expected:
        self.assertAlmostEqual(self._set_speed_mph(), expected[frame], msg=f"{frame=}")
    self._frame()
    self.assertAlmostEqual(self._set_speed_mph(), 60)

  def test_rolling_while_pressed_in_does_not_change_the_distance(self):
    self._hold_middle(TRAFFIC_MODE_HOLD_FRAMES - 1)
    self.scroll += 1
    self.assertEqual(button_events(self._frame(middle=True)), [TOGGLE])
    # and the roll is not sent later on release either
    self.assertEqual(button_events(self._frame()), [])
    self.scroll += 1
    self.assertEqual(button_events(self._frame()), [SCROLL])

  def test_scroll_alone(self):
    self.scroll += 1
    self.assertEqual(button_events(self._frame()), [SCROLL])
    self.assertEqual(button_events(self._frame()), [])

  def test_no_events_without_openpilot_long(self):
    self.cs = self._make(openpilot_long=False)
    events = []
    for _ in range(2 * TRAFFIC_MODE_HOLD_FRAMES):
      events += button_events(self._frame(middle=True))
    self.assertEqual(events, [])


if __name__ == "__main__":
  unittest.main()
