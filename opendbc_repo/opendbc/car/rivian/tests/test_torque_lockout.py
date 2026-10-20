"""Torque lockouts: a torque frame the panda refuses, and a latched EPAS torque-overlay fault (ToiFlt).

Ported from AdventurePilot (Geomglot). The end-to-end checks against the panda safety model are in
opendbc/safety/tests/test_rivian.py (TestRivianTorqueControllerSafety)."""
import unittest
from collections import deque
from types import SimpleNamespace
from unittest import mock

from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.rivian.carstate import TOI_FAULT_FRAMES
from opendbc.car.rivian.ext_controller import (ExternalController, TOI_BLIP_FRAMES, TOI_CLEAR_LATCH_FRAMES, TOI_CLEAR_QUIET_FRAMES,
                                               TOI_CLEAR_RETRY_FRAMES, TOI_MAX_ANGLE_FRAMES)
from opendbc.car.rivian.interface import CarInterface
from opendbc.car.rivian.torque_rt import HIST_FRAMES, TORQUE_RT_MAX_DELTA, TorqueRtLimiter
from opendbc.car.rivian.values import CAR, DBC, CarControllerParams

# the panda's real-time torque check (opendbc/safety/lateral.h, limits in opendbc/safety/modes/rivian.h)
PANDA_RT_DELTA = 125
PANDA_RT_FRAMES = 26  # its 250 ms interval at 100 Hz: the reference can be any frame sent that long ago


class TestTorqueRtLimiter(unittest.TestCase):
  def test_within_delta_of_every_recent_torque(self):
    lim = TorqueRtLimiter()
    for torque in (0, 60, 120, 180):
      lim.sent(torque)
    # the panda may still hold the 0 as its reference
    self.assertEqual(lim.limit(300), TORQUE_RT_MAX_DELTA)
    self.assertEqual(lim.limit(-300), -TORQUE_RT_MAX_DELTA)
    self.assertEqual(lim.limit(100), 100)

  def test_old_torques_age_out(self):
    lim = TorqueRtLimiter()
    for _ in range(HIST_FRAMES):
      lim.sent(200)
    self.assertEqual(lim.limit(300), 300)
    self.assertEqual(lim.limit(400), 200 + TORQUE_RT_MAX_DELTA)
    self.assertEqual(lim.limit(-300), -TORQUE_RT_MAX_DELTA)

  def test_blip_frame_counts_as_a_reference(self):
    # a blip frame is an ordinary zero-torque frame to the panda, so it can become the reference
    lim = TorqueRtLimiter()
    for _ in range(HIST_FRAMES):
      lim.sent(-200)
    lim.sent(0)
    self.assertEqual(lim.limit(-300), -TORQUE_RT_MAX_DELTA)

  def test_reset_leaves_only_zero(self):
    lim = TorqueRtLimiter()
    for _ in range(HIST_FRAMES):
      lim.sent(250)
    lim.reset()
    self.assertEqual(lim.limit(250), TORQUE_RT_MAX_DELTA)


class TestTorqueRtInController(unittest.TestCase):
  def test_full_demand_reaches_the_torque_limit(self):
    # the limiter must not cap a steady request
    for v_ego, steer_max in ((5.0, 350), (13.0, 300), (25.0, 250)):
      for sign in (1, -1):
        erc = ExternalController()
        erc.torque_active = True
        for _ in range(150):
          erc._update_torque(_cs(v_ego=v_ego), SimpleNamespace(torque=sign * 1.0))
        self.assertEqual(erc.apply_torque_last, sign * steer_max, (v_ego, sign))

  def test_full_demand_between_high_angle_blips(self):
    erc = ExternalController()
    erc.torque_active = True
    torques = []
    for _ in range(300):
      erc._update_torque(_cs(150.0, v_ego=25.0), SimpleNamespace(torque=1.0))
      torques.append(erc.apply_torque_last)
    self.assertIn(0, torques[100:], "blips")
    self.assertEqual(max(torques[100:]), 250)

  def test_faster_ramp_stays_inside_the_panda_real_time_check(self):
    """At this branch's 3/frame up-rate the panda's real-time check cannot bind at 100 Hz, so on the road the limiter never
    acts. Raise the up-rate so it must, and check every frame against the panda's rule: within 125 of any torque it may
    hold as its reference, i.e. one sent in its last interval (a blip's 0 included), or 0 after it refused a frame."""
    with mock.patch.object(CarControllerParams, "STEER_DELTA_UP", 10):
      for angle in (150.0, 30.0):
        for demand in (1.0, -1.0):
          erc = ExternalController()
          erc.torque_active = True
          refs, refused, peak = deque([0], maxlen=PANDA_RT_FRAMES), False, 0
          for i in range(400):
            # reversal at frame 200, refused frames at 120 and 300
            erc._update_torque(_cs(angle, v_ego=5.0, rejected=refused), SimpleNamespace(torque=demand if i < 200 else -demand))
            torque = erc.apply_torque_last
            self.assertLessEqual(torque, max(min(refs), 0) + PANDA_RT_DELTA, (angle, demand, i))
            self.assertGreaterEqual(torque, min(max(refs), 0) - PANDA_RT_DELTA, (angle, demand, i))
            peak = max(peak, abs(torque))
            refused = i in (120, 300)
            if refused:
              refs = deque([0], maxlen=PANDA_RT_FRAMES)  # not sent, and the panda zeroed its reference
            else:
              refs.append(torque)
          self.assertEqual(peak, 350, "still reaches full torque")


def _cs(angle=0.0, v_ego=20.0, toi_fault=False, rejected=False):
  return SimpleNamespace(out=SimpleNamespace(vEgoRaw=v_ego, steeringTorque=0.0, steeringAngleDeg=angle),
                         toi_fault=toi_fault, torque_cmd_rejected=rejected)


def _torque_controller(settle_frames=150, demand=0.5, angle=0.0):
  """torque channel in use, settled at a steady torque"""
  erc = ExternalController()
  erc.torque_active = True
  for _ in range(settle_frames):
    erc._update_torque(_cs(angle), SimpleNamespace(torque=demand))
  return erc


def _run(erc, faults, refused=(), demand=0.5, angle=0.0):
  out = []
  for i, fault in enumerate(faults):
    erc._update_torque(_cs(angle, toi_fault=fault, rejected=i in refused), SimpleNamespace(torque=demand))
    out.append((erc.toi_act_cmd, erc.apply_torque_last))
  return out


def _released(out):
  return [i for i, (toi_act, _) in enumerate(out) if not toi_act]


class TestTorqueRefusalRestart(unittest.TestCase):
  def test_restarts_from_zero_after_a_refused_frame(self):
    # The panda zeroes its torque memory when it refuses a frame and then refuses anything more than one rate step from
    # zero, so carrying on from the refused torque would be refused frame after frame
    erc = _torque_controller()
    steady = erc.apply_torque_last
    self.assertGreater(steady, 100)
    out = _run(erc, [False] * 5, refused={0})
    self.assertEqual([t for _, t in out], [CarControllerParams.STEER_DELTA_UP * (i + 1) for i in range(5)])
    self.assertTrue(all(toi_act for toi_act, _ in out), "a refusal does not drop the request")

  def test_negative_torque_restarts_from_zero(self):
    erc = _torque_controller(demand=-0.5)
    self.assertLess(erc.apply_torque_last, -100)
    out = _run(erc, [False], refused={0}, demand=-0.5)
    self.assertEqual(out[0][1], -CarControllerParams.STEER_DELTA_UP)


class TestToiFaultRelease(unittest.TestCase):
  """A latched ToiFlt never times out on its own but clears ~20 ms after a frame with the TOI request low, so the
  controller releases the request like a high-angle blip: torque 0, then a ramp from 0."""

  def test_latched_fault_is_released_once_then_torque_ramps_from_zero(self):
    erc = _torque_controller()
    out = _run(erc, [True] * TOI_CLEAR_LATCH_FRAMES + [False] * 20)
    rel = _released(out)
    self.assertEqual(rel, list(range(TOI_CLEAR_LATCH_FRAMES - 1, TOI_CLEAR_LATCH_FRAMES - 1 + TOI_BLIP_FRAMES)))
    for i in rel:
      self.assertEqual(out[i][1], 0, "torque is 0 while the request is released")
    self.assertEqual(out[rel[-1] + 1], (True, CarControllerParams.STEER_DELTA_UP), "torque ramps back from 0, like after a blip")

  def test_brief_fault_is_not_released(self):
    erc = _torque_controller()
    out = _run(erc, [True] * (TOI_CLEAR_LATCH_FRAMES - 1) + [False] * 20)
    self.assertEqual(_released(out), [])

  def test_persistent_fault_is_retried_every_half_second(self):
    erc = _torque_controller()
    rel = _released(_run(erc, [True] * 200))
    starts = [i for i in rel if i - 1 not in rel]
    self.assertEqual(len(rel), len(starts) * TOI_BLIP_FRAMES)
    self.assertEqual(starts[0], TOI_CLEAR_LATCH_FRAMES - 1)
    self.assertEqual([b - a for a, b in zip(starts, starts[1:], strict=False)], [TOI_CLEAR_RETRY_FRAMES] * (len(starts) - 1))

  def test_no_release_while_the_panda_is_refusing(self):
    # while 0x120 frames are refused the EPAS keeps missing them and latches again: wait for the refusals to stop
    erc = _torque_controller()
    rel = _released(_run(erc, [True] * 40, refused=set(range(20))))
    self.assertTrue(rel, "released once the refusals stop")
    self.assertEqual(rel[0], 19 + TOI_CLEAR_QUIET_FRAMES)

  def test_release_restarts_the_high_angle_blip_count(self):
    # the release drops the request like a blip, so the next periodic blip is a full interval later
    erc = _torque_controller(settle_frames=40, angle=150.0)
    rel = _released(_run(erc, [True] * TOI_CLEAR_LATCH_FRAMES + [False] * 150, angle=150.0))
    first = TOI_CLEAR_LATCH_FRAMES - 1
    self.assertEqual(rel[:TOI_BLIP_FRAMES], [first, first + 1])
    self.assertGreater(rel[TOI_BLIP_FRAMES] - first, TOI_MAX_ANGLE_FRAMES)

  def test_latch_next_to_a_high_angle_blip(self):
    # the latch lands just before the blip: one run of low frames (the blip, the release, or both), then a ramp from 0
    erc = _torque_controller(settle_frames=TOI_MAX_ANGLE_FRAMES - 2, angle=150.0)  # the blip is 2 frames away
    out = _run(erc, [True] * 20, angle=150.0)
    rel = _released(out)
    self.assertEqual(rel[0], TOI_CLEAR_LATCH_FRAMES - 1)
    self.assertEqual(rel, list(range(rel[0], rel[-1] + 1)), "one run of low frames")
    self.assertLessEqual(len(rel), 2 * TOI_BLIP_FRAMES)
    self.assertEqual(out[rel[-1] + 1], (True, CarControllerParams.STEER_DELTA_UP))

  def test_no_release_without_the_torque_channel(self):
    erc = ExternalController()
    for _ in range(20):
      erc._update_torque(_cs(toi_fault=True), SimpleNamespace(torque=0.5))
      self.assertFalse(erc.toi_act_cmd)
    self.assertEqual((erc.toi_fault_frames, erc.toi_clear_cooldown), (0, 0))


class TestTorqueLockoutCarState(unittest.TestCase):
  """What carstate reads off the bus for the controller and the driver alert"""

  def setUp(self):
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    self.CI = CarInterface(CP, structs.CarParamsSP())
    self.packer = CANPacker(DBC[CAR.RIVIAN_R1][Bus.pt])
    self.frame = 0
    self._update()  # carstate registers the messages it reads on its first update

  def _update(self, toi_flt=0, eac_status=1, eac_error=0, extra=()):
    self.frame += 1
    epas = [
      self.packer.make_can_msg("EPAS_SystemStatus", 0, {"EPAS_SystemStatus_Counter": self.frame % 16, "H_CAN_EPSS_ToiFlt": toi_flt,
                                                        "EPAS_HandsOnLevel": 1}),
      self.packer.make_can_msg("EPAS_AdasStatus", 0, {"EPAS_AdasStatus_Counter": self.frame % 16, "EPAS_EacStatus": eac_status,
                                                      "EPAS_EacErrorCode": eac_error}),
    ]
    ret, _ = self.CI.update([(self.frame * 10_000_000, [*epas, *extra])])
    return ret

  def test_refused_torque_frame_is_reported_once(self):
    self.assertFalse(self.CI.CS.torque_cmd_rejected)
    torque = self.packer.make_can_msg("ACM_lkaHbaCmd", 0, {"ACM_lkaStrToqReq": 120, "ACM_lkaActToi": 1})[:2]
    self._update(extra=[(*torque, 128), (*torque, 192)])  # an accepted frame echoes on 128, a refused one on 192
    self.assertTrue(self.CI.CS.torque_cmd_rejected)
    self.assertFalse(self.CI.CS.long_cmd_rejected_updated)
    self._update(extra=[(*torque, 128)])
    self.assertFalse(self.CI.CS.torque_cmd_rejected)
    for _ in range(100):
      self._update()
    self.assertTrue(self.CI.can_parsers[Bus.loopback].can_valid, "a quiet bus 192 must not raise canError")

  def test_toi_fault_reported_once_it_persists(self):
    # not before 0.3 s, so a latch the controller clears does not raise the warning
    flags = [self._update(toi_flt=1).steerFaultTemporary for _ in range(60)]
    self.assertEqual(flags, [False] * TOI_FAULT_FRAMES + [True] * (60 - TOI_FAULT_FRAMES))
    self.assertFalse(self._update(toi_flt=0).steerFaultTemporary, "cleared with the fault")

  def test_eac_error_is_a_fault_only_while_steering_on_angle(self):
    # the stock ACM shows EAC errors while the angle channel is not in use
    self.assertTrue(self._update(eac_status=2, eac_error=3).steerFaultTemporary)
    self.assertFalse(self._update(eac_status=2, eac_error=0).steerFaultTemporary)
    for eac_status in (0, 1, 3):
      self.assertFalse(self._update(eac_status=eac_status, eac_error=3).steerFaultTemporary, eac_status)

  def test_toi_fault_not_reported_while_steering_on_angle(self):
    for _ in range(80):
      self.assertFalse(self._update(toi_flt=1, eac_status=2).steerFaultTemporary)
    # but the controller still sees it
    self.assertTrue(self.CI.CS.toi_fault)
    self._update(toi_flt=0, eac_status=2)
    self.assertFalse(self.CI.CS.toi_fault)


if __name__ == "__main__":
  unittest.main()
