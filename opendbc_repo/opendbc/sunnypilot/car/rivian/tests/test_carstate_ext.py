import unittest
from types import SimpleNamespace
from unittest import mock

from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.rivian.interface import CarInterface
from opendbc.car.rivian.values import CAR
from opendbc.sunnypilot.car import FOLLOW_DISTANCE_CLOSER_BUTTON, FOLLOW_DISTANCE_FARTHER_BUTTON, TRAFFIC_MODE_BUTTON
from opendbc.sunnypilot.car.rivian.carstate_ext import CarStateExt, SCROLL_MAX_STEP, TRAFFIC_MODE_HOLD_FRAMES
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

ButtonType = structs.CarState.ButtonEvent.Type

SET_SPEED_MPH = 50
TOGGLE = (TRAFFIC_MODE_BUTTON, False)
FARTHER = (FOLLOW_DISTANCE_FARTHER_BUTTON, False)
CLOSER = (FOLLOW_DISTANCE_CLOSER_BUTTON, False)


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

  def _frame(self, decrease=False, increase=False, middle=False, metric=False, cruise_enabled=True, received=True):
    # buttons read 1 at rest and 2 while pressed (routes 0000001d/1e). Until the first WheelButtons_Fwd frame arrives,
    # the parser reads 0 for every signal, with no timestamp
    wheel = {"RightButton_LeftClick": 2 if decrease else 1, "RightButton_RightClick": 2 if increase else 1,
             "RightButton_ScrollClick": 2 if middle else 1, "RightButton_Scroll": self.scroll}
    if not received:
      wheel = dict.fromkeys(wheel, 0.)
    can_parsers = {
      Bus.alt: SimpleNamespace(vl={"WheelButtons_Fwd": wheel},
                               ts_nanos={"WheelButtons_Fwd": dict.fromkeys(wheel, 1 if received else 0)}),
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
    self.assertEqual(button_events(self._frame()), [FARTHER])

  def test_rolling_down_while_pressed_in_does_nothing(self):
    self._frame()
    for scroll in (255, 254):
      self.scroll = scroll
      self.assertEqual(button_events(self._frame(middle=True)), [])
    self.assertEqual(button_events(self._frame()), [])

  def test_no_events_without_openpilot_long(self):
    self.cs = self._make(openpilot_long=False)
    events = []
    for _ in range(2 * TRAFFIC_MODE_HOLD_FRAMES):
      events += button_events(self._frame(middle=True))
    for scroll in (1, 2, 1, 0, 255):
      self.scroll = scroll
      events += button_events(self._frame())
    self.assertEqual(events, [])


class TestFollowDistanceWheel(unittest.TestCase):
  """The right thumb wheel (RightButton_Scroll, a position counter that wraps through 255): one event per notch, farther
  for a notch up, closer for a notch down. The first frame received is only a baseline."""

  def setUp(self):
    self.cs = TestSetSpeedButtons._make()
    self.scroll = 0

  def _frame(self, scroll=None, **kwargs):
    if scroll is not None:
      self.scroll = scroll
    return button_events(TestSetSpeedButtons._frame(self, **kwargs))

  def test_one_notch_up_is_one_farther(self):
    self.assertEqual(self._frame(0), [])
    self.assertEqual(self._frame(1), [FARTHER])
    # the counter holds its position: nothing more on the next frames
    for _ in range(10):
      self.assertEqual(self._frame(), [])

  def test_one_notch_down_is_one_closer(self):
    self.assertEqual(self._frame(6), [])
    self.assertEqual(self._frame(5), [CLOSER])
    self.assertEqual(self._frame(), [])

  def test_wraps_through_255(self):
    # route 00000017 612.36 s: one notch down from 0 read 255, which the old decoding dropped as invalid
    self.assertEqual(self._frame(0), [])
    self.assertEqual(self._frame(255), [CLOSER])
    self.assertEqual(self._frame(), [])
    self.assertEqual(self._frame(0), [FARTHER])
    self.assertEqual(self._frame(254), [CLOSER, CLOSER])
    self.assertEqual(self._frame(1), [FARTHER, FARTHER, FARTHER])

  def test_quick_flick_sends_one_event_per_notch(self):
    # the left wheel was seen moving 3 in one 50 ms frame
    self.assertEqual(self._frame(10), [])
    self.assertEqual(self._frame(13), [FARTHER] * 3)
    self.assertEqual(self._frame(10), [CLOSER] * 3)
    self.assertEqual(self._frame(10 + SCROLL_MAX_STEP), [FARTHER] * SCROLL_MAX_STEP)

  def test_a_jump_is_dropped_and_the_wheel_works_from_the_new_value(self):
    self.assertEqual(self._frame(0), [])
    self.assertEqual(self._frame(40), [])
    self.assertEqual(self._frame(41), [FARTHER])
    self.assertEqual(self._frame(41 + SCROLL_MAX_STEP + 1), [])
    self.assertEqual(self._frame(41 - SCROLL_MAX_STEP), [])  # -7 from 47
    self.assertEqual(self._frame(35), [CLOSER])

  def test_first_frame_is_only_a_baseline(self):
    # the counter keeps counting while the truck is awake, so it can be anywhere when openpilot starts
    for scroll in (6, 3, 255, 128):
      with self.subTest(scroll=scroll):
        self.setUp()
        self.assertEqual(self._frame(scroll), [])
        self.assertEqual(self._frame(), [])

  def test_counted_from_the_first_frame_received(self):
    # before WheelButtons_Fwd first arrives the parser reads 0: that is no baseline, or a counter at 3 would read +3
    for _ in range(5):
      self.assertEqual(self._frame(3, received=False), [])
    self.assertEqual(self._frame(3), [])
    self.assertEqual(self._frame(4), [FARTHER])

  def test_traffic_mode_hold_still_toggles_once(self):
    self.assertEqual(self._frame(0), [])
    events = []
    for _ in range(3 * TRAFFIC_MODE_HOLD_FRAMES):
      events += self._frame(middle=True)
    self.assertEqual(events, [TOGGLE])

  def test_up_sign_reversed(self):
    # if the driveway check shows a notch up reads -1, RIGHT_SCROLL_UP_SIGN = -1 is the only change
    with mock.patch("opendbc.sunnypilot.car.rivian.carstate_ext.RIGHT_SCROLL_UP_SIGN", -1):
      self.assertEqual(self._frame(0), [])
      self.assertEqual(self._frame(1), [CLOSER])
      self.assertEqual(self._frame(255), [FARTHER, FARTHER])

  def test_through_the_real_can_parser(self):
    # card's path: WheelButtons_Fwd (0x131A) on bus 1 parsed by the CarInterface, the counter starting at 3
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CI = CarInterface(CP, structs.CarParamsSP(flags=int(RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE)))
    packer = CANPacker("rivian_park_assist_can")
    events = []
    t = 0
    for scroll, frames in ((None, 5), (3, 5), (4, 5), (2, 5), (255, 5)):
      for _ in range(frames):
        t += 10_000_000
        msgs = []
        if scroll is not None:
          msgs.append(packer.make_can_msg("WheelButtons_Fwd", 1, {"RightButton_Scroll": scroll, "RightButton_ScrollClick": 1,
                                                                  "RightButton_LeftClick": 1, "RightButton_RightClick": 1}))
        ret, _ = CI.update([(t, msgs)])
        events.append(button_events(ret))
    self.assertEqual([e for e in events if e], [[FARTHER], [CLOSER, CLOSER], [CLOSER] * 3])


if __name__ == "__main__":
  unittest.main()
