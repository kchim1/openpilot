"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from opendbc.car import structs
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.realtime import DT_CTRL
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LAUNCH_ASSIST_A_LEAD_TAU, MIN_X_LEAD_FACTOR, \
  ACCEL_MIN, LAUNCH_ASSIST_V_LEAD_K_MIN
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LAUNCH_ASSIST_READ_FRAMES
from openpilot.sunnypilot.selfdrive.controls.lib import launch_assist as la_mod
from openpilot.sunnypilot.selfdrive.controls.lib.launch_assist import LaunchAssist, LaunchAssistState, LongCtrlState, \
  ARM_STANDSTILL_T, LAUNCH_ACCEL, LAUNCH_RAMP_RATE, LAUNCH_MAX_T, LAUNCH_V_END, FADE_RATE, PARAM_READ_FRAMES

LIMITS = (-3.5, 2.0)
ARM_FRAMES = int(round(ARM_STANDSTILL_T / DT_CTRL))
FIRE_FRAMES = int(round(LAUNCH_MAX_T / DT_CTRL))
RAMP_STEP = LAUNCH_RAMP_RATE * DT_CTRL
RAMP_FRAMES = int(np.ceil((LAUNCH_ACCEL - 0.1) / RAMP_STEP - 1e-9))  # fire frame included, from the default 0.1 output


class FakeParams:
  def __init__(self, value=True):
    self.value = value
    self.reads = 0

  def get_bool(self, key):
    assert key == "RivianLaunchAssist"
    self.reads += 1
    return self.value


def cs(standstill=True, v_raw=0., brake=False, v_ego=None):
  return SimpleNamespace(standstill=standstill, vEgoRaw=v_raw, vEgo=v_raw if v_ego is None else v_ego, brakePressed=brake)


def lead(present=True, radar=True, v_lead_k=0., v_lead=0., a_lead_k=0., d_rel=6.8, track_id=1):
  return SimpleNamespace(present=present, radar=radar, vLeadK=v_lead_k, vLead=v_lead, aLeadK=a_lead_k, dRel=d_rel, radarTrackId=track_id)


STOPPED = lead()
DEPARTING = lead(v_lead_k=0.5, v_lead=0.6, a_lead_k=0.4)


class Driver:
  """Calls LaunchAssist.update the way controlsd does, with defaults for a released plan at standstill."""
  def __init__(self, enabled=True):
    self.params = FakeParams(enabled)
    self.la = LaunchAssist(self.params)

  def step(self, accel=0.1, long_active=True, CS=None, ld=STOPPED, state=LongCtrlState.pid, a_target=0.1, should_stop=False, limits=LIMITS):
    return self.la.update(long_active, CS if CS is not None else cs(), ld, state, a_target, should_stop, accel, limits)

  def steps(self, n, **kw):
    return [self.step(**kw) for _ in range(n)]

  def arm(self, state=LongCtrlState.stopping):
    self.steps(ARM_FRAMES, state=state, should_stop=state != LongCtrlState.pid)
    assert self.la.state == LaunchAssistState.armed

  def fire(self, **kw):
    self.arm()
    out = self.step(ld=DEPARTING, **kw)
    assert self.la.state == LaunchAssistState.firing
    return out


class TestLaunchAssistArm(OpenpilotTestCase):
  def test_available(self):
    for brand, op_long, expected in (("rivian", True, True), ("rivian", False, False), ("toyota", True, False)):
      CP = structs.CarParams(brand=brand, openpilotLongitudinalControl=op_long)
      assert LaunchAssist.available(CP) == expected

  def test_arms_after_one_second_with_radar_lead(self):
    d = Driver()
    d.steps(ARM_FRAMES - 1, state=LongCtrlState.stopping, should_stop=True)
    assert d.la.state == LaunchAssistState.idle
    d.step(state=LongCtrlState.stopping, should_stop=True)
    assert d.la.state == LaunchAssistState.armed

  def test_arms_when_already_in_pid_at_standstill(self):
    # long-gap stop: the planner released before the lead moved
    d = Driver()
    d.arm(state=LongCtrlState.pid)
    self.assertAlmostEqual(d.step(ld=DEPARTING), 0.1 + RAMP_STEP)
    assert d.la.state == LaunchAssistState.firing

  @parameterized.expand([(lead(radar=False),), (lead(present=False),)])
  def test_no_arm_without_radar_lead(self, ld):
    d = Driver()
    d.steps(3 * ARM_FRAMES, ld=ld)
    assert d.la.state == LaunchAssistState.idle

  def test_standstill_must_be_continuous(self):
    d = Driver()
    d.steps(ARM_FRAMES - 10)
    d.step(CS=cs(standstill=False))
    d.steps(ARM_FRAMES - 1)
    assert d.la.state == LaunchAssistState.idle
    d.step()
    assert d.la.state == LaunchAssistState.armed

  @parameterized.expand([({"ld": lead(present=False)},), ({"long_active": False},), ({"CS": cs(brake=True)},), ({"CS": cs(standstill=False)},)])
  def test_armed_disarms_and_rearms_after_another_second(self, kw):
    d = Driver()
    d.arm()
    d.step(**kw)
    assert d.la.state == LaunchAssistState.idle
    if kw.get('CS') is not None and not kw['CS'].standstill:
      return
    d.steps(ARM_FRAMES - 1)
    assert d.la.state == LaunchAssistState.idle
    d.step()
    assert d.la.state == LaunchAssistState.armed


  def test_new_track_disarms_and_rearms_after_another_second(self):
    # a different radar track: its first vLeadK is unfiltered, and it is not the car we were stopped behind
    d = Driver()
    d.arm(state=LongCtrlState.pid)
    new_departing = lead(v_lead_k=0.5, v_lead=0.6, a_lead_k=0.4, d_rel=25., track_id=2)
    assert d.step(ld=new_departing) == 0.1
    assert d.la.state == LaunchAssistState.idle
    d.steps(ARM_FRAMES - 1, ld=lead(track_id=2))
    assert d.la.state == LaunchAssistState.idle
    d.step(ld=lead(track_id=2))
    assert d.la.state == LaunchAssistState.armed
    assert d.step(ld=new_departing) > 0.1

  def test_track_change_restarts_the_second(self):
    d = Driver()
    d.steps(ARM_FRAMES - 10)
    d.steps(ARM_FRAMES - 1, ld=lead(track_id=2))
    assert d.la.state == LaunchAssistState.idle
    d.step(ld=lead(track_id=2))
    assert d.la.state == LaunchAssistState.armed


class TestLaunchAssistFire(OpenpilotTestCase):
  def test_no_fire_without_departing_lead(self):
    d = Driver()
    d.arm(state=LongCtrlState.pid)
    for ld in (STOPPED, lead(v_lead_k=0.15), lead(v_lead_k=-0.5), lead(v_lead_k=0.14, v_lead=0.5, a_lead_k=0.5)):
      assert d.steps(200, accel=0.1, ld=ld) == [0.1] * 200
    assert d.la.state == LaunchAssistState.armed

  def test_alt_gate(self, monkeypatch):
    monkeypatch.setattr(la_mod, "LEAD_DEPART_GATE_ALT", True)
    d = Driver()
    d.arm(state=LongCtrlState.pid)
    for ld in (lead(v_lead_k=1.0, v_lead=0.3, a_lead_k=0.5), lead(v_lead_k=1.0, v_lead=0.5, a_lead_k=0.1)):
      assert d.step(ld=ld) == 0.1
    self.assertAlmostEqual(d.step(ld=lead(v_lead_k=0.2, v_lead=0.31, a_lead_k=0.11)), 0.1 + RAMP_STEP)

  @parameterized.expand([(LongCtrlState.stopping, True), (LongCtrlState.stopping, False), (LongCtrlState.off, False), (LongCtrlState.pid, True)])
  def test_no_fire_while_stopping_or_should_stop(self, state, should_stop):
    d = Driver()
    d.arm()
    assert d.steps(100, accel=-0.2, ld=DEPARTING, state=state, should_stop=should_stop) == [-0.2] * 100
    assert d.la.state == LaunchAssistState.armed

  def test_no_fire_when_plan_slows(self):
    d = Driver()
    d.arm()
    assert d.step(accel=-0.05, a_target=-0.05, ld=DEPARTING) == -0.05

  @parameterized.expand([({"long_active": False},), ({"CS": cs(brake=True)},)])
  def test_no_fire_inactive_or_braking(self, kw):
    d = Driver()
    d.arm()
    assert d.step(accel=0.1, ld=DEPARTING, **kw) == 0.1
    assert d.la.state != LaunchAssistState.firing

  def test_never_lowers_output(self):
    d = Driver()
    assert d.fire(accel=0.8) == 0.8
    assert d.steps(FIRE_FRAMES, accel=0.9, ld=DEPARTING) == [0.9] * FIRE_FRAMES

  def test_floor_ramp_duration_and_fade(self):
    d = Driver()
    outs = [d.fire(accel=0.1)] + d.steps(400, accel=0.1, ld=DEPARTING)
    outs = np.array(outs)
    # up from the normal output at LAUNCH_RAMP_RATE, full floor until LAUNCH_MAX_T after the fire, then down at FADE_RATE
    # until the normal output, then the normal output
    assert np.allclose(np.diff(np.concatenate(([0.1], outs[:RAMP_FRAMES - 1]))), RAMP_STEP)
    assert outs[RAMP_FRAMES - 2] < LAUNCH_ACCEL - 1e-6 and np.allclose(outs[RAMP_FRAMES - 1:FIRE_FRAMES], LAUNCH_ACCEL)
    n_full = FIRE_FRAMES
    assert outs[n_full] < LAUNCH_ACCEL
    fade = outs[n_full:]
    fade = fade[fade > 0.1]
    assert len(fade) > 0 and np.allclose(np.diff(np.concatenate(([LAUNCH_ACCEL], fade))), -FADE_RATE * DT_CTRL)
    assert len(fade) == int(np.ceil((LAUNCH_ACCEL - 0.1) / (FADE_RATE * DT_CTRL))) - 1
    assert (outs[n_full + len(fade):] == 0.1).all()
    assert d.la.state == LaunchAssistState.idle

  def test_moving_starts_fade_early(self):
    d = Driver()
    d.fire()
    d.steps(RAMP_FRAMES, ld=DEPARTING)
    out = d.step(ld=DEPARTING, CS=cs(standstill=False, v_raw=LAUNCH_V_END + 0.01))
    assert d.la.state == LaunchAssistState.fading
    assert out == LAUNCH_ACCEL
    self.assertAlmostEqual(d.step(ld=DEPARTING, CS=cs(standstill=False, v_raw=0.3)), LAUNCH_ACCEL - FADE_RATE * DT_CTRL)

  def test_ramp_from_normal_output(self):
    # the floor starts at LongControl's output, so the command never steps
    for accel in (-0.2, 0., 0.3):
      d = Driver()
      outs = [d.fire(accel=accel)] + d.steps(FIRE_FRAMES - 1, accel=accel, ld=DEPARTING)
      steps = np.diff(np.concatenate(([accel], outs)))
      assert (steps >= 0.).all() and (steps <= RAMP_STEP + 1e-9).all()
      assert outs[-1] == LAUNCH_ACCEL

  @parameterized.expand([(False,), (True,)])
  def test_ends_once_as_fast_as_lead(self, fading):
    # lead inching or crawling: stop adding speed once ego has caught up with it
    d = Driver()
    d.fire()
    if fading:
      d.steps(FIRE_FRAMES, ld=DEPARTING)
      assert d.la.state == LaunchAssistState.fading
    else:
      d.steps(5, ld=DEPARTING)
    slow = lead(v_lead_k=0.2, v_lead=0.2)
    assert d.step(accel=0.1, ld=slow, CS=cs(standstill=False, v_raw=0.15, v_ego=0.19)) > 0.1
    assert d.step(accel=0.1, ld=slow, CS=cs(standstill=False, v_raw=0.2, v_ego=0.2)) == 0.1
    assert d.la.state == LaunchAssistState.idle
    # and does not come back while moving
    assert d.steps(100, accel=0.1, ld=DEPARTING, CS=cs(standstill=False, v_raw=0.2)) == [0.1] * 100

  def test_fade_ends_when_normal_output_catches_up(self):
    d = Driver()
    d.fire()
    d.steps(FIRE_FRAMES, ld=DEPARTING)
    assert d.la.state == LaunchAssistState.fading
    assert d.step(accel=0.49, ld=DEPARTING) == 0.49
    assert d.la.state == LaunchAssistState.idle

  def test_clipped_to_limits(self):
    d = Driver()
    outs = [d.fire(accel=0.1, limits=(-3.5, 0.3))] + d.steps(FIRE_FRAMES - 1, accel=0.1, ld=DEPARTING, limits=(-3.5, 0.3))
    assert max(outs) == 0.3 and outs[-1] == 0.3
    d = Driver()
    d.fire(accel=-1.0, limits=(-1.0, 1.0))
    assert d.steps(FIRE_FRAMES - 1, accel=-1.0, ld=DEPARTING, limits=(-1.0, 1.0))[-1] == LAUNCH_ACCEL


CANCELS = [
  {"state": LongCtrlState.stopping, "should_stop": True},  # planner stops again
  {"state": LongCtrlState.pid, "should_stop": True},
  {"long_active": False},                               # disengaged or gas override
  {"CS": cs(brake=True)},
  {"ld": lead(present=False)},
  {"ld": lead(radar=False, v_lead_k=0.3)},
  {"a_target": -0.1},
]


class TestLaunchAssistCancel(OpenpilotTestCase):
  @parameterized.expand([(dict(kw), fading) for kw in CANCELS for fading in (False, True)])
  def test_cancel_immediately(self, kw, fading):
    d = Driver()
    d.fire()
    if fading:
      d.steps(FIRE_FRAMES, ld=DEPARTING)
      assert d.la.state == LaunchAssistState.fading
    kw.setdefault('ld', DEPARTING)
    assert d.step(accel=-0.3, **kw) == -0.3
    assert d.la.state == LaunchAssistState.idle and d.la.floor == 0.
    # no second kick in this standstill once everything qualifies again
    assert d.steps(3 * ARM_FRAMES, accel=0.1, ld=DEPARTING) == [0.1] * (3 * ARM_FRAMES)

  def test_disable_cancels(self):
    d = Driver()
    d.fire()
    d.params.value = False
    d.la._frame = 0  # read on the next frame
    assert d.step(ld=DEPARTING) == 0.1
    assert d.la.state == LaunchAssistState.idle

  def test_once_per_standstill(self):
    d = Driver()
    d.fire()
    # stop/go flicker at standstill after the kick: never a second kick
    for _ in range(10):
      d.steps(20, accel=-0.2, ld=DEPARTING, state=LongCtrlState.stopping, should_stop=True)
      assert d.steps(20, accel=0.1, ld=DEPARTING) == [0.1] * 20
    # moved, then a new standstill: one more kick after another second
    d.step(CS=cs(standstill=False, v_raw=0.5))
    d.arm()
    self.assertAlmostEqual(d.step(ld=DEPARTING), 0.1 + RAMP_STEP)
    assert d.la.state == LaunchAssistState.firing


class TestLaunchAssistToggle(OpenpilotTestCase):
  def test_off_returns_output_unchanged(self):
    d = Driver(enabled=False)
    rng = np.random.default_rng(0)
    for i in range(3000):
      accel = float(rng.uniform(-1, 1))
      st = (LongCtrlState.pid, LongCtrlState.stopping)[(i // 150) % 2]
      ld = DEPARTING if (i // 50) % 3 else STOPPED
      out = d.step(accel=accel, ld=ld, state=st, should_stop=st != LongCtrlState.pid, CS=cs(standstill=(i // 400) % 2 == 0))
      assert out is accel
    assert d.la.state == LaunchAssistState.idle

  def test_param_read_every_second(self):
    d = Driver(enabled=False)
    d.steps(PARAM_READ_FRAMES)
    assert d.params.reads == 1
    d.params.value = True
    d.step()
    assert d.params.reads == 2 and d.la.enabled

  def test_unknown_key_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    params = mocker.MagicMock()
    params.get_bool.side_effect = UnknownKeyName(b"RivianLaunchAssist")
    la = LaunchAssist(params)
    for _ in range(3 * PARAM_READ_FRAMES):
      assert la.update(True, cs(), DEPARTING, LongCtrlState.pid, 0.1, False, 0.1, LIMITS) == 0.1
    assert not la.enabled
    assert cloudlog.exception.call_count == 1

  def test_real_param(self):
    params = Params()
    params.put_bool("RivianLaunchAssist", True, block=True)
    assert LaunchAssist(params).read_param()
    params.put_bool("RivianLaunchAssist", False, block=True)
    assert not LaunchAssist(params).read_param()


def mpc_lead(radar=True, v_lead=0.5, v_lead_k=0.4, a_lead_k=0.5, a_lead_tau=1.5, d_rel=6.8):
  return SimpleNamespace(present=True, radar=radar, dRel=d_rel, vLead=v_lead, vLeadK=v_lead_k, aLeadK=a_lead_k, aLeadTau=a_lead_tau)


class TestLaunchAssistMpc(OpenpilotTestCase):
  def setup_method(self):
    self.mpc = LongitudinalMpc()

  def lead_xv(self, lead, v_ego, on):
    self.mpc.launch_assist = on
    self.mpc.set_cur_state(v_ego, 0.)
    return self.mpc.process_lead(lead)

  def expected(self, lead, a_lead_tau):
    return LongitudinalMpc.extrapolate_lead(lead.dRel, lead.vLead, lead.aLeadK, a_lead_tau)

  def test_departing_radar_lead_keeps_accelerating(self):
    for v_ego in (0., 0.5, 0.99):
      for lead in (mpc_lead(), mpc_lead(v_lead=0.31, v_lead_k=0.16, a_lead_k=0.11), mpc_lead(v_lead=2., v_lead_k=1.8, a_lead_k=1.5, a_lead_tau=0.8)):
        xv = self.lead_xv(lead, v_ego, True)
        assert np.array_equal(xv, self.expected(lead, LAUNCH_ASSIST_A_LEAD_TAU))
        # the lead is predicted further ahead than with radard's tau
        assert xv[-1, 0] > self.lead_xv(lead, v_ego, False)[-1, 0]

  def test_smaller_tau_kept(self):
    lead = mpc_lead(a_lead_tau=0.1)
    assert np.array_equal(self.lead_xv(lead, 0., True), self.expected(lead, 0.1))

  def test_gates(self):
    cases = [
      (mpc_lead(radar=False), 0.),               # vision-only lead
      (mpc_lead(v_lead=0.3), 0.),                # lead not moving away fast enough
      (mpc_lead(v_lead_k=LAUNCH_ASSIST_V_LEAD_K_MIN), 0.),  # only a raw vLead sample moving (noise)
      (mpc_lead(v_lead=0.6, v_lead_k=0.09, a_lead_k=0.21), 0.),
      (mpc_lead(a_lead_k=0.1), 0.),              # lead not accelerating
      (mpc_lead(a_lead_k=-0.5), 0.),
      (mpc_lead(), 1.0),                         # ego already moving
      (mpc_lead(), 5.),
    ]
    for lead, v_ego in cases:
      xv = self.lead_xv(lead, v_ego, True)
      assert np.array_equal(xv, self.lead_xv(lead, v_ego, False)), (lead, v_ego)
      assert np.array_equal(xv, self.expected(lead, lead.aLeadTau))

  def test_off_is_identical(self):
    rng = np.random.default_rng(0)
    for _ in range(500):
      lead = mpc_lead(radar=bool(rng.integers(2)), v_lead=float(rng.uniform(0, 3)), v_lead_k=float(rng.uniform(0, 3)),
                      a_lead_k=float(rng.uniform(-2, 2)),
                      a_lead_tau=float(rng.uniform(0, 1.5)), d_rel=float(rng.uniform(4, 30)))
      v_ego = float(rng.uniform(0, 3))
      self.mpc.set_cur_state(v_ego, 0.)
      # r1t process_lead for a present lead
      min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + lead.vLead) * (v_ego - lead.vLead) / (-ACCEL_MIN * 2)
      stock = LongitudinalMpc.extrapolate_lead(np.clip(lead.dRel, min_x_lead, 1e8), np.clip(lead.vLead, 0.0, 1e8),
                                               np.clip(lead.aLeadK, -10., 5.), lead.aLeadTau)
      assert np.array_equal(self.lead_xv(lead, v_ego, False), stock)

  def test_no_lead(self):
    for on in (False, True):
      self.mpc.launch_assist = on
      self.mpc.set_cur_state(0., 0.)
      assert np.array_equal(self.mpc.process_lead(SimpleNamespace(present=False)), self.mpc.process_lead(None))


class TestLaunchAssistPlanner(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  @staticmethod
  def make_planner(brand="rivian", openpilot_long=True):
    CP = structs.CarParams(brand=brand, openpilotLongitudinalControl=openpilot_long)
    mpc = SimpleNamespace(traffic_mode=False, launch_assist=False)
    return LongitudinalPlannerSP(CP, structs.CarParamsSP(), mpc), mpc

  def test_poll_every_read_frames(self):
    self.params.put_bool("RivianLaunchAssist", True, block=True)
    planner, mpc = self.make_planner()
    planner.update_launch_assist()
    assert mpc.launch_assist
    self.params.put_bool("RivianLaunchAssist", False, block=True)
    for _ in range(LAUNCH_ASSIST_READ_FRAMES - 1):
      planner.update_launch_assist()
      assert mpc.launch_assist
    planner.update_launch_assist()
    assert not mpc.launch_assist

  def test_rivian_openpilot_long_only(self):
    self.params.put_bool("RivianLaunchAssist", True, block=True)
    for brand, op_long in (("rivian", False), ("toyota", True)):
      planner, mpc = self.make_planner(brand, op_long)
      for _ in range(3 * LAUNCH_ASSIST_READ_FRAMES):
        planner.update_launch_assist()
      assert not mpc.launch_assist

  def test_read_error_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner.cloudlog")
    planner, mpc = self.make_planner()
    mpc.launch_assist = True
    planner._params = mocker.MagicMock()
    planner._params.get_bool.side_effect = UnknownKeyName(b"RivianLaunchAssist")
    for _ in range(3 * LAUNCH_ASSIST_READ_FRAMES):
      planner.update_launch_assist()
    assert not mpc.launch_assist
    assert cloudlog.exception.call_count == 1
