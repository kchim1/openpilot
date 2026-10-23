"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from opendbc.car import structs
from openpilot.cereal import log, messaging
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.parameterized import parameterized
from openpilot.common.test import OpenpilotTestCase
from openpilot.common.realtime import DT_CTRL, DT_MDL
from openpilot.selfdrive.controls.lib.longcontrol import LongControl
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LAUNCH_ASSIST_A_LEAD_TAU, MIN_X_LEAD_FACTOR, \
  ACCEL_MIN, LAUNCH_ASSIST_V_LEAD_K_MIN
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, LAUNCH_ASSIST_READ_FRAMES
from openpilot.sunnypilot.selfdrive.controls.lib import launch_assist as la_mod
from openpilot.sunnypilot.selfdrive.controls.lib.launch_assist import LaunchAssist, LaunchAssistState, LongCtrlState, \
  ARM_STANDSTILL_T, LAUNCH_ACCEL_MIN, LAUNCH_ACCEL_MAX, LAUNCH_NO_LEAD, LAUNCH_RAMP_RATE, LAUNCH_DOWN_RATE, LAUNCH_MAX_T, \
  LAUNCH_RISE_FRAMES, NO_LEAD_PLAN_T, FADE_RATE, PARAM_READ_FRAMES, STRENGTH_GENTLE, STRENGTH_STRONG, LAUNCH_LEAD_GAP_MIN, \
  LEAD_MOVING_V_RAW, TRAFFIC_MODE_READ_FRAMES
from openpilot.sunnypilot.selfdrive.controls.lib.stop_behind_lead import GAP_MIN

LIMITS = (-3.5, 2.0)
P = log.LongitudinalPersonality
ARM_FRAMES = int(round(ARM_STANDSTILL_T / DT_CTRL))
FIRE_FRAMES = int(round(LAUNCH_MAX_T / DT_CTRL))
RAMP_STEP = LAUNCH_RAMP_RATE * DT_CTRL
FADE_STEP = FADE_RATE * DT_CTRL
STRONG_CAP = LAUNCH_ACCEL_MAX[STRENGTH_STRONG][0]  # Aggressive, the default personality here
TWO_PLANNER_FRAMES = int(round(2 * DT_MDL / DT_CTRL))  # plannerd's 20 Hz in controlsd frames


class FakeParams:
  def __init__(self, value=True, strength=None, traffic_mode=False):
    self.value = value
    self.strength = strength  # RivianLaunchAssistBoost, None: never written
    self.traffic_mode = traffic_mode
    self.reads = 0
    self.strength_reads = 0
    self.traffic_mode_reads = 0

  def get_bool(self, key):
    if key == "TrafficMode":
      self.traffic_mode_reads += 1
      return self.traffic_mode
    assert key == "RivianLaunchAssist"
    self.reads += 1
    return self.value

  def get(self, key, return_default=False):
    assert key == "RivianLaunchAssistBoost"
    self.strength_reads += 1
    if self.strength is None:
      return STRENGTH_STRONG if return_default else None
    return self.strength


def cs(standstill=True, v_raw=0., brake=False, v_ego=None):
  return SimpleNamespace(standstill=standstill, vEgoRaw=v_raw, vEgo=v_raw if v_ego is None else v_ego, brakePressed=brake)


def lead(present=True, radar=True, v_lead_k=0., v_lead=0., a_lead_k=0., d_rel=6.8, track_id=1):
  return SimpleNamespace(present=present, radar=radar, vLeadK=v_lead_k, vLead=v_lead, aLeadK=a_lead_k, dRel=d_rel, radarTrackId=track_id)


STOPPED = lead()
DEPARTING = lead(v_lead_k=0.5, v_lead=0.6, a_lead_k=0.4)
BRISK = lead(v_lead_k=0.5, v_lead=0.6, a_lead_k=1.0)  # aLeadK past every cap: the floor goes straight to the cap
NO_LEAD = lead(present=False, radar=False, track_id=-1)
VISION_LEAD = lead(radar=False, v_lead_k=0.5, v_lead=0.6, a_lead_k=0.5, track_id=-1)


def model(plan_a=1.0, t_end=1.0):
  """a model plan that averages about plan_a m/s^2 over its first t_end s"""
  t = ModelConstants.T_IDXS
  return SimpleNamespace(velocity=SimpleNamespace(x=list(np.interp(t, [0., t_end, 10.], [0., plan_a * t_end, plan_a * t_end])), t=list(t)))


class Driver:
  """Calls LaunchAssist.update the way controlsd does, with defaults for a released plan at standstill."""
  def __init__(self, enabled=True, strength=None, personality=P.aggressive, traffic_mode=False):
    self.params = FakeParams(enabled, strength, traffic_mode)
    self.la = LaunchAssist(self.params)
    self.personality = personality
    self.v_lead = 0.

  def step(self, accel=0.1, long_active=True, CS=None, ld=STOPPED, state=LongCtrlState.pid, a_target=0.1, should_stop=False, limits=LIMITS,
           md=None):
    return self.la.update(long_active, CS if CS is not None else cs(), ld, state, a_target, should_stop, accel, limits, model=md,
                          personality=self.personality)

  def steps(self, n, **kw):
    return [self.step(**kw) for _ in range(n)]

  def arm(self, state=LongCtrlState.stopping, ld=STOPPED, md=None):
    self.steps(ARM_FRAMES, state=state, should_stop=state != LongCtrlState.pid, ld=ld, md=md)
    assert self.la.state == LaunchAssistState.armed

  def fire(self, **kw):
    self.arm()
    out = self.step(ld=DEPARTING, **kw)
    assert self.la.state == LaunchAssistState.firing
    return out

  def departing(self, a=1.0, track_id=1):
    """a lead pulling away at a m/s^2 (raw speed, and a Kalman pair that lags: past the 0.15 m/s gate, 40 % of the accel)"""
    self.v_lead += a * DT_CTRL
    return lead(v_lead_k=max(self.v_lead, 0.16), v_lead=self.v_lead, a_lead_k=a * 0.4, track_id=track_id)

  def launch(self, frames=30, a=1.5):
    """armed behind a stopped lead, then frames of it pulling away: returns the last output"""
    self.arm()
    out = self.steps(1, ld=self.departing(a))[0]
    for _ in range(frames - 1):
      out = self.step(ld=self.departing(a))
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

  def test_no_arm_behind_a_vision_only_lead(self):
    d = Driver()
    assert d.steps(3 * ARM_FRAMES, ld=VISION_LEAD) == [0.1] * (3 * ARM_FRAMES)
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
    # up from the normal output at LAUNCH_RAMP_RATE to the cap, held until LAUNCH_MAX_T after the fire, then down at
    # FADE_RATE until the normal output, then the normal output
    d = Driver()
    d.arm()
    outs = np.array(d.steps(500, accel=0.1, ld=BRISK))
    ramp_frames = int(np.ceil((STRONG_CAP - 0.1) / RAMP_STEP - 1e-9))  # fire frame included
    assert np.allclose(np.diff(np.concatenate(([0.1], outs[:ramp_frames]))), RAMP_STEP)
    assert np.allclose(outs[ramp_frames - 1:FIRE_FRAMES], STRONG_CAP)
    fade = outs[FIRE_FRAMES:]
    fade = fade[fade > 0.1]
    assert len(fade) > 0 and np.allclose(np.diff(np.concatenate(([STRONG_CAP], fade))), -FADE_STEP)
    assert len(fade) == int(np.ceil((STRONG_CAP - 0.1) / FADE_STEP)) - 1
    assert (outs[FIRE_FRAMES + len(fade):] == 0.1).all()
    assert d.la.state == LaunchAssistState.idle

  def test_a_lead_lift_hands_back_once_moving(self):
    # as v1 did: held on as the truck picked up speed, it ended a queue that creeps and stops up to 1.6 m closer in sim
    d = Driver()
    d.fire()
    d.steps(20, ld=DEPARTING)
    out = d.step(ld=DEPARTING, CS=cs(standstill=False, v_raw=LEAD_MOVING_V_RAW))
    assert d.la.state == LaunchAssistState.firing and out > LAUNCH_ACCEL_MIN
    d.step(ld=DEPARTING, CS=cs(standstill=False, v_raw=0.16))
    assert d.la.state == LaunchAssistState.fading and d.la.fade_reason == "moving"
    # fading out at FADE_RATE, not dropped
    outs = d.steps(3, ld=DEPARTING, CS=cs(standstill=False, v_raw=0.3))
    np.testing.assert_allclose(np.diff(outs), -FADE_STEP)
    # a lift with no car ahead keeps going
    d = Driver()
    d.arm(ld=NO_LEAD, md=model(2.0))
    d.steps(20, ld=NO_LEAD, md=model(2.0))
    d.step(ld=NO_LEAD, md=model(2.0), CS=cs(standstill=False, v_raw=0.3))
    assert d.la.state == LaunchAssistState.firing

  def test_ramp_from_normal_output(self):
    # the floor starts at LongControl's output and moves at its rates, so the command never steps
    for accel in (-0.2, 0., 0.3):
      d = Driver()
      outs = [d.fire(accel=accel)] + d.steps(FIRE_FRAMES - 1, accel=accel, ld=DEPARTING)
      steps = np.diff(np.concatenate(([accel], outs)))
      assert (steps >= -LAUNCH_DOWN_RATE * DT_CTRL - 1e-9).all() and (steps <= RAMP_STEP + 1e-9).all()
      assert max(outs) == STRONG_CAP and outs[-1] == LAUNCH_ACCEL_MIN

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

  def test_end_reason_logged(self, mocker):
    # on-truck logs tell a normal fade end from the speed cap, and what started the fade
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    d = Driver()
    d.fire()
    d.steps(400, ld=DEPARTING)
    d.step(CS=cs(standstill=False, v_raw=0.5))
    d.arm()
    d.step(ld=DEPARTING)
    d.step(ld=lead(v_lead_k=0.2, v_lead=0.2), CS=cs(standstill=False, v_raw=0.2, v_ego=0.2))
    ends = [(c.kwargs["reason"], c.kwargs["fade"], c.kwargs["lead"]) for c in cloudlog.event.call_args_list if c.args[0] == "launch assist end"]
    assert ends == [("fade", "time", True), ("cap", "", True)]

  def test_clipped_to_limits(self):
    d = Driver()
    outs = [d.fire(accel=0.1, limits=(-3.5, 0.3))] + d.steps(FIRE_FRAMES - 1, accel=0.1, ld=DEPARTING, limits=(-3.5, 0.3))
    assert max(outs) == 0.3 and outs[-1] == 0.3
    d = Driver()
    d.fire(accel=-1.0, limits=(-1.0, 1.0))
    assert d.steps(FIRE_FRAMES - 1, accel=-1.0, ld=DEPARTING, limits=(-1.0, 1.0))[-1] == LAUNCH_ACCEL_MIN


CANCELS = [
  {"state": LongCtrlState.stopping, "should_stop": True, "ld": STOPPED},  # planner stops again: the lead stopped
  {"state": LongCtrlState.pid, "should_stop": True, "ld": lead(v_lead_k=0.15, v_lead=0.2)},
  {"long_active": False},                               # disengaged or gas override
  {"CS": cs(brake=True)},
  {"a_target": -0.1},                                   # planner slows
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
      ld = (DEPARTING, STOPPED, NO_LEAD)[(i // 50) % 3]
      out = d.step(accel=accel, ld=ld, state=st, should_stop=st != LongCtrlState.pid, CS=cs(standstill=(i // 400) % 2 == 0), md=model(2.0))
      assert out is accel
    assert d.la.state == LaunchAssistState.idle

  def test_param_read_every_second(self):
    # Traffic Mode every 0.25 s, as plannerd reads it
    assert TRAFFIC_MODE_READ_FRAMES == 25
    d = Driver(enabled=False)
    d.steps(PARAM_READ_FRAMES)
    assert d.params.reads == 1 and d.params.strength_reads == 1 and d.params.traffic_mode_reads == 4
    d.params.value = True
    d.params.traffic_mode = True
    d.step()
    assert d.params.reads == 2 and d.params.strength_reads == 2 and d.params.traffic_mode_reads == 5
    assert d.la.enabled and d.la.traffic_mode

  def test_unknown_key_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    params = mocker.MagicMock()

    def get_bool(key):
      if key == "RivianLaunchAssist":
        raise UnknownKeyName(key.encode())
      return False
    params.get_bool.side_effect = get_bool
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


class TestLaunchAssistLeadGates(OpenpilotTestCase):
  def test_no_lift_behind_a_car_in_traffic_mode(self):
    # Traffic Mode's own launch shift moves it off sooner; a lift on top closed up an inching queue on every inch
    d = Driver(traffic_mode=True)
    d.arm()
    assert d.la.traffic_mode
    assert d.steps(FIRE_FRAMES, ld=d.departing()) == [0.1] * FIRE_FRAMES
    assert d.la.state == LaunchAssistState.armed
    # with no car ahead it still lifts
    d = Driver(traffic_mode=True)
    d.arm(ld=NO_LEAD, md=model(2.0))
    assert max(d.steps(100, ld=NO_LEAD, md=model(2.0))) > 0.1
    # Traffic Mode turned off while armed: lifted from the next read
    d = Driver(traffic_mode=True)
    d.arm()
    assert d.la.traffic_mode and d.step(ld=d.departing()) == 0.1
    d.params.traffic_mode = False
    outs = [d.step(ld=d.departing()) for _ in range(TRAFFIC_MODE_READ_FRAMES)]
    assert not d.la.traffic_mode and outs[-1] > 0.1 and d.la.state == LaunchAssistState.firing

  def test_traffic_mode_turned_on_during_a_lift_cancels_it(self):
    d = Driver()
    d.fire()
    d.steps(10, ld=DEPARTING)
    d.params.traffic_mode = True
    outs = d.steps(TRAFFIC_MODE_READ_FRAMES, ld=DEPARTING)
    assert d.la.traffic_mode and d.la.state == LaunchAssistState.idle and outs[-1] == 0.1
    # a lift with no car ahead goes on
    d = Driver()
    d.arm(ld=NO_LEAD, md=model(2.0))
    d.steps(10, ld=NO_LEAD, md=model(2.0))
    d.params.traffic_mode = True
    d.steps(TRAFFIC_MODE_READ_FRAMES, ld=NO_LEAD, md=model(2.0))
    assert d.la.traffic_mode and d.la.state == LaunchAssistState.firing

  def test_traffic_mode_read_failure_is_on_and_logged_once(self, mocker):
    # no lead lift on top of a Traffic Mode plannerd may have on
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    params = FakeParams()
    real_get_bool = params.get_bool

    def get_bool(key):
      if key == "TrafficMode":
        raise UnknownKeyName(key.encode())
      return real_get_bool(key)
    params.get_bool = get_bool
    la = LaunchAssist(params)
    for _ in range(3 * PARAM_READ_FRAMES):
      la.update(True, cs(), STOPPED, LongCtrlState.stopping, 0., True, 0., LIMITS)
    assert la.enabled and la.traffic_mode and la.state == LaunchAssistState.armed
    assert la.update(True, cs(), DEPARTING, LongCtrlState.pid, 0.1, False, 0.1, LIMITS) == 0.1
    assert cloudlog.exception.call_count == 1

  def test_no_lift_after_a_stop_close_behind_a_car(self):
    # stopped closer than Stop Behind Lead ever aims to: no lift for this launch at all, however far the car then gets (a
    # lift once it was 3.5 m away came with the planner already released, and ended the next stop closer)
    assert LAUNCH_LEAD_GAP_MIN == GAP_MIN == 3.5
    d = Driver()
    d.arm(ld=lead(d_rel=3.4))
    for d_rel in np.linspace(3.4, 6., 100):
      ld = d.departing()
      ld.dRel = float(d_rel)
      assert d.step(ld=ld) == 0.1
    assert d.la.state == LaunchAssistState.armed
    # the closest it came while stopped counts: rolled up to 3.4 m after stopping 4 m back
    d = Driver()
    d.arm(ld=lead(d_rel=4.))
    d.steps(10, ld=lead(d_rel=3.4))
    assert d.step(ld=d.departing()) == 0.1
    # stopped 3.5 m back: lifted
    d = Driver()
    d.arm(ld=lead(d_rel=LAUNCH_LEAD_GAP_MIN))
    assert d.step(ld=d.departing()) > 0.1
    assert d.la.state == LaunchAssistState.firing


class TestLaunchAssistStrength(OpenpilotTestCase):
  # RivianLaunchAssistBoost: Strong (the default) or Gentle, each cap 0.1 m/s^2 lower
  @parameterized.expand([
    (STRENGTH_STRONG, P.aggressive, 0.8, 0.6), (STRENGTH_STRONG, P.standard, 0.7, 0.5), (STRENGTH_STRONG, P.relaxed, 0.6, 0.4),
    (STRENGTH_GENTLE, P.aggressive, 0.7, 0.5), (STRENGTH_GENTLE, P.standard, 0.6, 0.4), (STRENGTH_GENTLE, P.relaxed, 0.5, 0.3),
  ])
  def test_caps_by_strength_and_personality(self, strength, personality, lead_cap, no_lead_cap):
    # behind a car pulling away briskly
    d = Driver(strength=strength, personality=personality)
    d.arm()
    outs = [d.step(ld=d.departing(2.0)) for _ in range(150)]
    assert abs(max(outs) - lead_cap) < 1e-9
    assert (np.diff(np.concatenate(([0.1], outs))) <= RAMP_STEP + 1e-9).all()
    # with no lead, behind a model plan of 2 m/s^2
    d = Driver(strength=strength, personality=personality)
    d.arm(ld=NO_LEAD, md=model(2.0))
    outs = [d.step(ld=NO_LEAD, md=model(2.0)) for _ in range(100)]
    assert abs(max(outs) - no_lead_cap) < 1e-9

  def test_gentle_is_strong_less_a_tenth(self):
    for case in (LAUNCH_ACCEL_MAX, LAUNCH_NO_LEAD):
      assert np.allclose(np.subtract(case[STRENGTH_STRONG], case[STRENGTH_GENTLE]), 0.1)
    assert min(LAUNCH_ACCEL_MAX[STRENGTH_GENTLE]) >= LAUNCH_ACCEL_MIN

  def test_default_is_strong(self):
    # never written: the key's default, also when read from the real params
    assert Driver(strength=None).la.read_strength() == STRENGTH_STRONG
    assert LaunchAssist(Params()).read_strength() == STRENGTH_STRONG
    d = Driver(strength=None)
    d.arm()
    assert abs(max(d.steps(100, ld=BRISK)) - LAUNCH_ACCEL_MAX[STRENGTH_STRONG][0]) < 1e-9

  @parameterized.expand([(0, STRENGTH_GENTLE), (1, STRENGTH_STRONG), (2, STRENGTH_STRONG), (-1, STRENGTH_STRONG)])
  def test_only_zero_is_gentle(self, stored, expected):
    assert Driver(strength=stored).la.read_strength() == expected

  def test_unknown_key_falls_back_to_strong_and_logs_once(self, mocker):
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    params = mocker.MagicMock()
    params.get_bool.side_effect = lambda key: key == "RivianLaunchAssist"  # Traffic Mode off
    params.get.side_effect = UnknownKeyName(b"RivianLaunchAssistBoost")
    la = LaunchAssist(params)
    la.strength = STRENGTH_GENTLE
    for _ in range(ARM_FRAMES):
      la.update(True, cs(), STOPPED, LongCtrlState.stopping, -0.2, True, -0.2, LIMITS)
    outs = [la.update(True, cs(), BRISK, LongCtrlState.pid, 0.1, False, 0.1, LIMITS) for _ in range(3 * PARAM_READ_FRAMES)]
    assert la.enabled and la.strength == STRENGTH_STRONG
    assert abs(max(outs) - LAUNCH_ACCEL_MAX[STRENGTH_STRONG][0]) < 1e-9
    assert cloudlog.exception.call_count == 1

  def test_read_with_the_toggle_every_second(self):
    d = Driver(strength=STRENGTH_STRONG)
    d.step()
    d.params.strength = STRENGTH_GENTLE
    d.steps(PARAM_READ_FRAMES - 1)
    assert d.la.strength == STRENGTH_STRONG
    d.step()
    assert d.la.strength == STRENGTH_GENTLE

  def test_real_param(self):
    params = Params()
    params.put("RivianLaunchAssistBoost", 0, block=True)
    assert LaunchAssist(params).read_strength() == STRENGTH_GENTLE
    params.put("RivianLaunchAssistBoost", 1, block=True)
    assert LaunchAssist(params).read_strength() == STRENGTH_STRONG

  def test_fire_logs_the_strength_and_the_case(self, mocker):
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    d = Driver(strength=STRENGTH_GENTLE)
    d.arm()
    d.step(ld=BRISK)
    d.step(CS=cs(standstill=False, v_raw=0.5))
    d.arm(ld=NO_LEAD, md=model(2.0))
    d.step(ld=NO_LEAD, md=model(2.0))
    fires = [c.kwargs for c in cloudlog.event.call_args_list if c.args[0] == "launch assist fire"]
    assert [(f["lead"], f["strength"], f["cap"]) for f in fires] == [(True, "gentle", 0.7), (False, "gentle", 0.5)]

  def test_personality_from_a_capnp_reader(self):
    # selfdriveState.personality as controlsd hands it over: a capnp enum, compared, never a dict key
    for name, strong, gentle in (("aggressive", 0.8, 0.7), ("standard", 0.7, 0.6), ("relaxed", 0.6, 0.5)):
      msg = messaging.new_message('selfdriveState')
      msg.selfdriveState.personality = name
      pers = msg.selfdriveState.as_reader().personality
      assert la_mod.by_personality(LAUNCH_ACCEL_MAX[STRENGTH_STRONG], pers) == strong
      assert la_mod.by_personality(LAUNCH_ACCEL_MAX[STRENGTH_GENTLE], pers) == gentle


class TestLaunchAssistLeadLaunch(OpenpilotTestCase):
  def run_launch(self, a_lead=1.0, frames=300, personality=P.aggressive, a_target=0.1, accel=0.1):
    d = Driver(personality=personality)
    d.arm()
    outs = [d.step(ld=d.departing(a_lead), a_target=a_target, accel=accel) for _ in range(frames)]
    return d, np.array(outs)

  def test_stopped_lead_no_fire(self):
    d = Driver()
    d.arm()
    assert d.steps(300, ld=lead(v_lead_k=0.1)) == [0.1] * 300
    assert d.la.state == LaunchAssistState.armed

  @parameterized.expand([(P.aggressive, 0.8), (P.standard, 0.7), (P.relaxed, 0.6)])
  def test_floor_follows_a_brisk_lead_up_to_the_personality_max(self, personality, a_max):
    _, outs = self.run_launch(a_lead=1.5, personality=personality)
    assert abs(outs.max() - a_max) < 1e-6
    steps = np.diff(np.concatenate(([0.1], outs[:60])))
    assert (steps <= RAMP_STEP + 1e-9).all()

  def test_slow_lead_gets_the_old_floor(self):
    _, outs = self.run_launch(a_lead=0.2)
    assert abs(outs.max() - LAUNCH_ACCEL_MIN) < 1e-6

  def test_one_noisy_raw_sample_moves_the_floor_by_little(self):
    d = Driver()
    d.arm()
    for i in range(60):
      ld = d.departing(0.3)
      if i == 59:
        ld.vLead += 0.2   # one glitch of 0.2 m/s
      out = d.step(ld=ld)
    assert LAUNCH_ACCEL_MIN - 1e-9 <= out <= LAUNCH_ACCEL_MIN + 1e-9

  def test_hands_back_when_the_planner_catches_up(self):
    d = Driver()
    d.launch(30)
    # the planner now asks for more than the floor: done, its own output goes through
    assert d.step(ld=d.departing(1.5), a_target=0.9, accel=0.9) == 0.9
    assert d.la.state == LaunchAssistState.idle

  def test_only_the_start_is_lifted(self):
    # a lead that only gets going after the first second: the floor does not rise late in the launch
    d = Driver()
    d.arm()
    outs = [d.step(ld=d.departing(0.3 if i < LAUNCH_RISE_FRAMES else 2.0)) for i in range(190)]
    assert max(outs[LAUNCH_RISE_FRAMES:]) <= outs[LAUNCH_RISE_FRAMES - 1] + 1e-9

  def test_lead_stops_pulling_away_ends_it(self):
    d = Driver()
    d.launch(60, a=1.0)
    for _ in range(60):
      d.step(ld=d.departing(-1.0))
    assert d.la.state in (LaunchAssistState.fading, LaunchAssistState.idle)

  def test_never_lowers_the_output(self):
    d = Driver()
    d.arm()
    for _ in range(300):
      assert d.step(ld=d.departing(), accel=1.2, a_target=0.0) >= 1.2


class TestLaunchAssistNoLead(OpenpilotTestCase):
  def test_vision_only_lead_gets_nothing(self):
    d = Driver()
    assert d.steps(3 * ARM_FRAMES, ld=VISION_LEAD, md=model(2.0)) == [0.1] * (3 * ARM_FRAMES)
    assert d.la.state == LaunchAssistState.idle

  @parameterized.expand([(2.0, 0.6), (0.4, None), (0.0, 0.1)])
  def test_floor_is_the_model_plan_up_to_the_personality_value(self, plan_a, expected):
    d = Driver()
    d.arm(ld=NO_LEAD, md=model(plan_a))
    outs = d.steps(100, ld=NO_LEAD, md=model(plan_a))
    if expected is None:
      expected = la_mod.model_plan_accel(model(plan_a), NO_LEAD_PLAN_T)
      assert 0.35 < expected < 0.45
    assert abs(max(outs) - expected) < 1e-6

  def test_no_model_no_floor(self):
    d = Driver()
    d.arm(ld=NO_LEAD)
    assert d.steps(100, ld=NO_LEAD, md=None) == [0.1] * 100

  def test_a_non_finite_plan_is_no_plan(self):
    # NaN would pass min(cap, max(plan, 0.)) as the full cap
    t = list(ModelConstants.T_IDXS)
    for vx in ([float('nan')] * len(t), [float('nan')] + [1.] * (len(t) - 1), [0.] + [float('inf')] * (len(t) - 1)):
      md = SimpleNamespace(velocity=SimpleNamespace(x=vx, t=t))
      assert la_mod.model_plan_accel(md, NO_LEAD_PLAN_T) is None
      d = Driver()
      d.arm(ld=NO_LEAD, md=md)
      assert d.steps(100, ld=NO_LEAD, md=md) == [0.1] * 100

  def test_a_lead_showing_up_cancels(self):
    d = Driver()
    d.arm(ld=NO_LEAD, md=model(2.0))
    d.steps(20, ld=NO_LEAD, md=model(2.0))
    assert d.la.state == LaunchAssistState.firing
    assert d.step(ld=lead(v_lead_k=0.0), md=model(2.0)) == 0.1
    assert d.la.state == LaunchAssistState.idle

  def test_a_planner_stop_cancels_at_once(self):
    # the stop pause is for a radar lead clearly pulling away only
    d = Driver()
    d.arm(ld=NO_LEAD, md=model(2.0))
    d.steps(20, ld=NO_LEAD, md=model(2.0))
    assert d.step(accel=-0.01, ld=NO_LEAD, md=model(2.0), state=LongCtrlState.stopping, should_stop=True) == -0.01
    assert d.la.state == LaunchAssistState.idle


class TestLaunchAssistRearm(OpenpilotTestCase):
  def test_second_launch_after_a_lead_crept_and_stopped(self):
    d = Driver()
    d.launch(30, a=0.5)
    # the lead stops again: the planner stops the launch, the truck never showed any speed
    assert d.step(accel=-0.2, ld=STOPPED, state=LongCtrlState.stopping, should_stop=True) == -0.2
    assert d.la.state == LaunchAssistState.idle
    # not before the planner has held the stop for a full second
    d.steps(ARM_FRAMES - 5, accel=-0.2, ld=STOPPED, state=LongCtrlState.stopping, should_stop=True)
    d.v_lead = 0.
    assert d.step(ld=d.departing()) == 0.1
    d.steps(ARM_FRAMES + 5, accel=-0.2, ld=STOPPED, state=LongCtrlState.stopping, should_stop=True)
    d.v_lead = 0.
    assert d.la.state == LaunchAssistState.armed
    assert d.step(ld=d.departing()) > 0.1
    assert d.la.state == LaunchAssistState.firing

  def test_flicker_never_rearms(self):
    # stop/go flicker shorter than a second after the kick: no second kick
    d = Driver()
    d.launch(30)
    for _ in range(10):
      d.steps(50, accel=-0.2, ld=d.departing(), state=LongCtrlState.stopping, should_stop=True)
      assert [d.step(ld=d.departing()) for _ in range(20)] == [0.1] * 20


class TestLaunchAssistTrack(OpenpilotTestCase):
  # a lead launch keeps to the radar track it fired behind: another car, or none, never inherits the floor
  OTHERS = [
    ("another_track", lead(v_lead_k=2.0, v_lead=2.2, a_lead_k=3.0, d_rel=4.5, track_id=2)),
    ("lost", NO_LEAD),
    ("vision_only", lead(radar=False, v_lead_k=2.0, v_lead=2.2, a_lead_k=3.0, track_id=-1)),
  ]

  @parameterized.expand(OTHERS)
  def test_fades_out(self, _, other):
    d = Driver()
    last = d.launch(25)
    assert last > LAUNCH_ACCEL_MIN
    outs = np.array(d.steps(200, ld=other))
    # from the floor it had, down at FADE_RATE (never up, whatever the new car does), then the normal output
    fade = outs[outs > 0.1]
    assert len(fade) > 0 and np.allclose(np.diff(np.concatenate(([last], fade))), -FADE_STEP)
    assert (outs[len(fade):] == 0.1).all() and d.la.state == LaunchAssistState.idle
    # and nothing more for this standstill, the original car back or not
    assert d.steps(3 * ARM_FRAMES, ld=other) == [0.1] * (3 * ARM_FRAMES)
    assert d.steps(3 * ARM_FRAMES, ld=d.departing(1.5)) == [0.1] * (3 * ARM_FRAMES)

  def test_the_old_track_back_does_not_restart_it(self):
    d = Driver()
    d.launch(25)
    d.step(ld=self.OTHERS[0][1])
    assert d.la.state == LaunchAssistState.fading
    # back on the original track: still fading, at FADE_RATE
    prev = d.la.floor
    d.step(ld=d.departing(1.5))
    assert d.la.state == LaunchAssistState.fading and abs(d.la.floor - (prev - FADE_STEP)) < 1e-9

  def test_a_slower_car_ahead_ends_it_at_once(self):
    # the speed cap holds against whatever car is ahead now
    d = Driver()
    d.launch(25)
    slower = lead(v_lead_k=0.1, v_lead=0.1, track_id=2)
    assert d.step(accel=0.1, ld=slower, CS=cs(standstill=False, v_raw=0.12)) == 0.1
    assert d.la.state == LaunchAssistState.idle

  def test_fade_reason_logged(self, mocker):
    cloudlog = mocker.patch.object(la_mod, "cloudlog")
    reasons = []
    for _, other in self.OTHERS:
      d = Driver()
      d.launch(25)
      d.steps(200, ld=other)
      reasons += [c.kwargs["fade"] for c in cloudlog.event.call_args_list if c.args[0] == "launch assist end"][len(reasons):]
    assert reasons == ["lead changed", "lead lost", "lead changed"]


class TestLaunchAssistStopPulse(OpenpilotTestCase):
  # a shouldStop of under 2 planner frames while the car ahead clearly pulls away (vLeadK > 0.15) only pauses the
  # launch: the stop gate holds the truck while it lasts, and the floor climbs back afterwards. A longer one ends it.
  @staticmethod
  def stop(d, n, ld=None, a_target=-0.1):
    """n frames of the planner's shouldStop, LongControl in stopping, its output from 0 down at 1 m/s^3"""
    return [d.step(accel=-0.01 * (i + 1), ld=ld if ld is not None else d.departing(1.5), state=LongCtrlState.stopping,
                   should_stop=True, a_target=a_target) for i in range(n)]

  @parameterized.expand([(n,) for n in range(1, TWO_PLANNER_FRAMES)])
  def test_a_short_stop_only_pauses_it(self, n):
    d = Driver()
    d.launch(30)
    # the stop gate holds: LongControl's own output at every frame
    assert self.stop(d, n) == [-0.01 * (i + 1) for i in range(n)]
    assert d.la.state == LaunchAssistState.firing
    # released again: from the planner's output back up at the ramp rate, then on to the cap
    outs = [d.step(accel=0.3, a_target=0.3, ld=d.departing(1.5)) for _ in range(60)]
    assert outs[0] == 0.3
    assert (np.diff(np.concatenate(([0.3], outs))) <= RAMP_STEP + 1e-9).all()
    assert abs(max(outs) - STRONG_CAP) < 1e-9 and d.la.state == LaunchAssistState.firing

  def test_one_planner_frame_mid_launch(self):
    # after the rise window too: the launch goes on at the floor it had, not ended by a 50 ms glitch
    d = Driver()
    d.launch(LAUNCH_RISE_FRAMES + 20)
    level = d.la.floor
    self.stop(d, 5)
    outs = [d.step(accel=0.2, a_target=0.2, ld=d.departing(1.5)) for _ in range(30)]
    assert (np.diff(np.concatenate(([0.2], outs))) <= RAMP_STEP + 1e-9).all()
    assert abs(outs[-1] - level) < 1e-9 and d.la.state == LaunchAssistState.firing

  def test_a_stop_of_two_planner_frames_ends_it(self):
    d = Driver()
    d.launch(30)
    assert self.stop(d, TWO_PLANNER_FRAMES) == [-0.01 * (i + 1) for i in range(TWO_PLANNER_FRAMES)]
    assert d.la.state == LaunchAssistState.idle and d.la.floor == 0.
    # and, the planner released after less than a second of stop, nothing more for the rest of this stop
    assert d.steps(3 * ARM_FRAMES, ld=d.departing(1.5)) == [0.1] * (3 * ARM_FRAMES)

  def test_pauses_never_add_to_a_stop(self):
    # shouldStop flickering however it likes: every frame of it gets exactly LongControl's output
    d = Driver()
    d.launch(30)
    rng = np.random.default_rng(0)
    for i in range(1000):
      stop = bool(rng.integers(2))
      accel = float(rng.uniform(-0.5, 0.0)) if stop else float(rng.uniform(0., 0.5))
      out = d.step(accel=accel, ld=d.departing(1.5), state=LongCtrlState.stopping if stop else LongCtrlState.pid, should_stop=stop,
                   a_target=-0.1 if stop else accel)
      if stop:
        assert out == accel, i

  @parameterized.expand([
    ("lead_not_departing", lead(v_lead_k=0.15, v_lead=0.3, a_lead_k=0.2)),
    ("another_track", lead(v_lead_k=2.0, v_lead=2.2, a_lead_k=3.0, track_id=2)),
    ("lead_lost", NO_LEAD),
  ])
  def test_any_other_stop_ends_it_at_once(self, _, ld):
    d = Driver()
    d.launch(30)
    assert self.stop(d, 1, ld=ld) == [-0.01]
    assert d.la.state == LaunchAssistState.idle

  def test_lead_goes_on_red_the_truck_stays_put(self):
    # the car ahead goes, the model holds the truck for the red light: nothing is added for as long as it holds
    d = Driver()
    d.arm()
    for i in range(500):
      assert d.step(accel=-0.5, ld=d.departing(1.5), state=LongCtrlState.stopping, should_stop=True, a_target=-0.3) == -0.5, i
    assert d.la.state == LaunchAssistState.armed
    # the light turns red just after the launch started: held from the first frame of the stop, and it stays so
    d = Driver()
    d.launch(30)
    for i in range(500):
      assert d.step(accel=-0.5, ld=d.departing(1.5), state=LongCtrlState.stopping, should_stop=True, a_target=-0.3) == -0.5, i
    assert d.la.state != LaunchAssistState.firing


class TestLaunchAssistLongInactive(OpenpilotTestCase):
  @parameterized.expand([(False,), (True,)])
  def test_long_inactive_mid_launch_gives_exactly_the_output(self, no_lead):
    d = Driver()
    md = model(2.0)
    d.arm(ld=NO_LEAD if no_lead else STOPPED, md=md)
    for _ in range(40):
      out = d.step(ld=NO_LEAD if no_lead else d.departing(), md=md)
    assert out > 0.4
    # disengaged / gas override: LongControl's output (0 when off) goes through untouched, the floor is exactly 0
    for _ in range(200):
      assert d.step(accel=0., long_active=False, ld=NO_LEAD if no_lead else d.departing(), md=md) == 0.
      assert d.la.floor == 0. and d.la.state == LaunchAssistState.idle

  @parameterized.expand([({"state": LongCtrlState.stopping},), ({"should_stop": True},), ({"a_target": -0.01},), ({"long_active": False},),
                         ({"CS": cs(brake=True)},)])
  def test_never_fires_unless_released_and_engaged(self, kw):
    for no_lead in (False, True):
      d = Driver()
      d.arm(ld=NO_LEAD if no_lead else STOPPED, md=model())
      for _ in range(300):
        assert d.step(accel=-0.2, ld=NO_LEAD if no_lead else d.departing(), md=model(), **kw) == -0.2
      assert d.la.state != LaunchAssistState.firing

  def test_controlsd_output_identical_without_it_when_long_is_off(self):
    # LongControl, then Launch Assist, as controlsd runs them, against LongControl alone: with long not active the sent
    # accel is exactly 0 either way, including frames right after a launch was going
    CP = structs.CarParams(brand="rivian", openpilotLongitudinalControl=True)
    CP.longitudinalTuning.kiBP = [0.]
    CP.longitudinalTuning.kiV = [0.]
    with_la, alone = LongControl(CP, structs.CarParamsSP()), LongControl(CP, structs.CarParamsSP())
    d = Driver()
    rng = np.random.default_rng(1)
    fired = 0
    for i in range(6000):
      active = (i // 250) % 3 != 2
      should_stop = (i // 125) % 2 == 0
      a_target = -0.3 if should_stop else float(rng.uniform(0., 0.6))
      CS = SimpleNamespace(standstill=True, vEgo=0., aEgo=0., vEgoRaw=0., brakePressed=False, cruiseState=SimpleNamespace(standstill=False))
      ld = d.departing(1.5) if (i // 125) % 2 else STOPPED
      if i % 125 == 0:
        d.v_lead = 0.
      accel = with_la.update(active, CS, a_target, should_stop, LIMITS)
      out = d.la.update(active, CS, ld, with_la.long_control_state, a_target, should_stop, accel, LIMITS, model=model(2.0),
                        personality=P.aggressive)
      ref = alone.update(active, CS, a_target, should_stop, LIMITS)
      fired += d.la.state == LaunchAssistState.firing
      if not active:
        assert out == ref == 0. and d.la.floor == 0., i
    assert fired > 100


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
