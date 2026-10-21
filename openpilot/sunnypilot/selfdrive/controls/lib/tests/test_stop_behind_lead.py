"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np
import pytest

from openpilot.cereal import log, messaging
from opendbc.car import structs
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.launch_assist import LEAD_DEPART_V_K
from openpilot.sunnypilot.selfdrive.controls.lib.stop_behind_lead import StopBehindLead, V_ENGAGE, V_RELEASE, LEAD_STOPPED_V, A_STOP, \
  A_MAX, GAP_MIN, GAP_MAX, GAP_MAX_TRAFFIC, RISE_RATE, STOP_V, DROPOUT_FRAMES, PROFILE_V_MAX, PROFILE_A_MAX, PROFILE_STOPPING_V, \
  PROFILE_STOPPING_A, PROFILE_LEAD_DECEL, PROFILE_RELIEF_JERK, PROFILE_RETURN_JERK
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
RISE_STEP = RISE_RATE * DT_MDL
D = GAP_MAX + 1.5   # a stopped lead within reach at walking pace


def lead(d, v_lead_k=0., radar=True, present=True, track=1, a_lead_k=0.):
  return SimpleNamespace(present=present, radar=radar, dRel=d, vLeadK=v_lead_k, radarTrackId=track, aLeadK=a_lead_k)


def settle(sbl, v_ego, ld, frames=100, **kw):
  out = None
  for _ in range(frames):
    out = sbl.update(v_ego, ld, **kw)
  return out


class TestStopBehindLead:
  def setup_method(self):
    self.sbl = StopBehindLead(CP)

  def test_engages_only_at_walking_pace_behind_a_stopped_radar_lead(self):
    assert self.sbl.update(V_ENGAGE + 0.1, lead(D)) is None          # still faster than walking pace: the lead planner's
    assert self.sbl.update(1.0, lead(D, v_lead_k=0.5)) is None       # the lead is still moving
    assert self.sbl.update(1.0, lead(D, radar=False)) is None        # vision only
    assert self.sbl.update(1.0, lead(D, present=False)) is None
    assert self.sbl.update(1.0, lead(30.)) is None                   # too far back: the lead planner closes up first
    assert self.sbl.update(1.0, lead(D), long_off=True) is None      # not commanding
    assert StopBehindLead(structs.CarParams(openpilotLongitudinalControl=False)).update(1.0, lead(D)) is None
    assert not self.sbl.active
    assert self.sbl.update(1.0, lead(D, v_lead_k=LEAD_STOPPED_V - 0.01)) is not None
    assert self.sbl.active

  def test_brakes_at_a_stop_within_range_and_builds_up(self):
    v = 1.0
    d = (GAP_MIN + GAP_MAX) / 2 + v ** 2 / (2 * A_STOP)
    out = self.sbl.update(v, lead(d))
    assert out[0] == -RISE_STEP and out[1] is False                  # does not step in
    out = settle(self.sbl, v, lead(d))
    assert abs(out[0] + A_STOP) < 1e-6                                # stops where A_STOP ends, within range
    assert abs(self.sbl.gap - (d - v ** 2 / (2 * A_STOP))) < 1e-6

  def test_far_out_brakes_less_to_stop_at_gap_max(self):
    v, d = 1.0, GAP_MAX + 2.5
    out = settle(self.sbl, v, lead(d))
    assert self.sbl.gap == GAP_MAX
    assert abs(out[0] + v ** 2 / (2 * (d - GAP_MAX))) < 1e-6 and out[0] > -A_STOP

  def test_too_close_brakes_more_to_stop_at_gap_min_bounded(self):
    out = settle(self.sbl, 1.5, lead(GAP_MIN + 1.))
    assert self.sbl.gap == GAP_MIN
    assert abs(out[0] + 1.5 ** 2 / 2.) < 1e-6 and out[0] < -A_STOP
    sbl = StopBehindLead(CP)
    out = settle(sbl, 1.5, lead(GAP_MIN + 0.1))
    assert out[0] == -A_MAX

  def test_traffic_mode_stops_closer(self):
    d = GAP_MAX_TRAFFIC + 1.5
    settle(self.sbl, 0.5, lead(d), traffic_mode=False)
    assert abs(self.sbl.gap - min(d - 0.5 ** 2 / (2 * A_STOP), GAP_MAX)) < 1e-6
    sbl = StopBehindLead(CP)
    settle(sbl, 0.5, lead(d), traffic_mode=True)
    assert sbl.gap == GAP_MAX_TRAFFIC < self.sbl.gap

  def test_holds_the_stop_until_the_lead_pulls_away(self):
    settle(self.sbl, 1.0, lead(D))
    accel, stop = self.sbl.update(STOP_V - 0.01, lead(D))
    assert stop and accel <= 0.
    accel, stop = settle(self.sbl, 0., lead(D, v_lead_k=0.12))      # radar noise between the two thresholds: held
    assert stop and accel == 0.
    assert self.sbl.update(0., lead(D, v_lead_k=LEAD_DEPART_V_K + 0.01)) is None   # Launch Assist's gate: never later
    assert not self.sbl.active

  def test_lets_go_when_the_truck_speeds_up_or_long_turns_off(self):
    settle(self.sbl, 1.0, lead(D))
    assert self.sbl.update(V_RELEASE + 0.1, lead(D)) is None
    settle(self.sbl, 1.0, lead(D))
    assert self.sbl.update(1.0, lead(D), long_off=True) is None
    assert self.sbl.accel == 0.

  def test_lets_go_when_radar_switches_to_a_car_further_up(self):
    # stopped behind a car, then it changes lanes and radard's lead is the next car, stopped further up
    settle(self.sbl, 0., lead(D))
    assert self.sbl.active
    assert self.sbl.update(0., lead(D + 12., track=2)) is None
    assert not self.sbl.active
    # the same car further out on the same track (no id change) lets go too
    settle(self.sbl, 0., lead(D))
    assert self.sbl.update(0., lead(D + GAP_MAX + 6.)) is None

  def test_keeps_the_same_car_on_a_new_track(self):
    settle(self.sbl, 1.0, lead(D))
    out = self.sbl.update(1.0, lead(D + 0.3, track=7))
    assert out is not None and self.sbl.track_id == 7

  def test_rides_out_a_short_radar_dropout(self):
    accel = settle(self.sbl, 1.0, lead(D))[0]
    for _ in range(DROPOUT_FRAMES):
      assert self.sbl.update(1.0, lead(D, present=False)) == (accel, False)
    assert self.sbl.update(1.0, lead(D, present=False)) is None
    assert not self.sbl.active
    # gas override during a dropout still lets go at once
    settle(self.sbl, 1.0, lead(D))
    assert self.sbl.update(1.0, lead(D, present=False), long_off=True) is None

  def test_braking_goes_away_at_once(self):
    settle(self.sbl, 1.0, lead(D))
    out = self.sbl.update(0.6, lead(D))
    # much less is needed now: no ramp on the way out
    assert abs(out[0] + 0.6 ** 2 / (2 * (D - self.sbl.gap))) < 1e-6


class TestProfileFloor:
  def setup_method(self):
    self.sbl = StopBehindLead(CP)

  def test_even_stop_at_the_gap_behind_a_stopped_car(self):
    v, d = 4., 15.
    assert abs(self.sbl.profile_floor(v, lead(d)) + v ** 2 / (2 * (d - GAP_MAX))) < 1e-9
    assert abs(StopBehindLead(CP).profile_floor(v, lead(d), traffic_mode=True) + v ** 2 / (2 * (d - GAP_MAX_TRAFFIC))) < 1e-9

  def test_behind_a_stopping_car_aims_where_it_stops(self):
    v, d, vl = 4., 15., 3.
    # braking gently: assumed to stop as if at the lead planner's comfort brake; braking harder: sooner
    expect = -v ** 2 / (2 * (d + vl ** 2 / (2 * PROFILE_LEAD_DECEL) - GAP_MAX))
    assert abs(self.sbl.profile_floor(v, lead(d, v_lead_k=vl, a_lead_k=PROFILE_STOPPING_A - 0.2)) - expect) < 1e-9
    hard = -v ** 2 / (2 * (d + vl ** 2 / (2 * 4.) - GAP_MAX))
    assert abs(StopBehindLead(CP).profile_floor(v, lead(d, v_lead_k=vl, a_lead_k=-4.)) - hard) < 1e-9

  def test_not_behind_a_car_rolling_along_or_too_fast(self):
    assert self.sbl.profile_floor(4., lead(15., v_lead_k=2., a_lead_k=0.)) is None             # slow but not braking
    assert self.sbl.profile_floor(4., lead(15., v_lead_k=PROFILE_STOPPING_V + 0.5, a_lead_k=-1.)) is None
    # a gentle stop (well under PROFILE_A_MAX) just above and just below the speed limit
    assert StopBehindLead(CP).profile_floor(PROFILE_V_MAX + 0.5, lead(GAP_MAX + 60.)) is None
    assert StopBehindLead(CP).profile_floor(PROFILE_V_MAX - 0.5, lead(GAP_MAX + 60.)) is not None
    assert self.sbl.profile_floor(4., lead(15., radar=False)) is None
    assert self.sbl.profile_floor(4., lead(15., present=False)) is None
    assert self.sbl.profile_floor(4., lead(15.), long_off=True) is None
    assert StopBehindLead(structs.CarParams(openpilotLongitudinalControl=False)).profile_floor(4., lead(15.)) is None
    assert self.sbl.profile_floor(4., lead(GAP_MAX + 0.4)) is None                             # already at the gap

  def test_only_while_the_stop_is_gentle(self):
    v = 8.
    assert self.sbl.profile_floor(v, lead(GAP_MAX + v ** 2 / (2 * (PROFILE_A_MAX + 0.1)))) is None
    assert self.sbl.profile_floor(v, lead(GAP_MAX + v ** 2 / (2 * (PROFILE_A_MAX - 0.1)))) is not None

  def test_hysteresis_on_the_lead_stopping(self):
    assert self.sbl.profile_floor(2., lead(12., v_lead_k=LEAD_STOPPED_V - 0.01)) is not None
    assert self.sbl.profile_floor(2., lead(12., v_lead_k=0.12)) is not None    # radar noise: kept
    assert self.sbl.profile_floor(2., lead(12., v_lead_k=LEAD_DEPART_V_K + 0.01)) is None

  def relax(self, a_plan, floor, **kw):
    # ACC only: the last output is relax()'s own
    last = self.sbl.relaxed if self.sbl.relaxed is not None else a_plan
    return self.sbl.relax(a_plan, floor, last, **kw)

  def test_relief_builds_gently_never_past_the_floor_and_goes_away(self):
    up, down = PROFILE_RELIEF_JERK * DT_MDL, PROFILE_RETURN_JERK * DT_MDL
    out = [self.relax(-1.5, -0.6) for _ in range(20)]
    assert np.allclose(np.diff(out[:11]), up)                      # eases off gently: 0.9 at `up` a frame takes 12 frames
    assert out[0] == pytest.approx(-1.5 + up) and out[-1] == pytest.approx(-0.6) and max(out) <= -0.6 + 1e-9
    assert self.relax(-1.0, -0.6) == pytest.approx(-0.6)           # the lead planner easing off itself: still the floor
    assert self.relax(-0.4, -0.6) == pytest.approx(-0.4)           # asking for less than the floor: as is
    for _ in range(40):
      self.relax(-1.5, -0.6)
    assert self.relax(-1.5, None) == pytest.approx(-0.6 - down)    # no floor: back at PROFILE_RETURN_JERK
    for _ in range(40):
      self.relax(-1.5, -0.6)
    assert self.relax(-1.5, -0.6, off=True) == -1.5                # FCW or long off: at once

  def test_easing_off_no_faster_than_the_jerk_in_all(self):
    # the lead planner easing off by itself at the same time: the two together still rise at PROFILE_RELIEF_JERK
    self.relax(-1.8, None)
    out = [self.relax(-1.8 + 0.05 * i, -0.6) for i in range(30)]
    assert max(np.diff(out)) <= PROFILE_RELIEF_JERK * DT_MDL + 1e-9
    assert np.allclose(out[15:24], -0.6)                           # at the floor 16 frames in, until the lead planner
    assert out[-1] == pytest.approx(-1.8 + 0.05 * 29)              # itself asks for less
    # braking harder is never held back
    assert self.relax(-2.5, -1.2) == pytest.approx(-1.2)

  def test_no_jump_when_something_braking_harder_lets_go(self):
    # the model (or another candidate) held the output at -1.4 while this candidate was eased: it must not have drifted up
    # to the floor out of sight, or the output would jump there the frame the model lets go
    up = PROFILE_RELIEF_JERK * DT_MDL
    for _ in range(40):
      cand = self.sbl.relax(-1.6, -0.6, -1.4)
    assert cand <= -1.4 + up + 1e-9
    assert self.sbl.relax(-1.6, -0.6, -1.4) <= -1.4 + up + 1e-9

  def test_never_adds_throttle(self):
    for v in np.linspace(0., PROFILE_V_MAX, 21):
      for d in np.linspace(GAP_MAX + 0.6, 60., 30):
        for vl, al in ((0., 0.), (3., -1.), (4.9, -3.)):
          f = self.sbl.profile_floor(v, lead(d, v_lead_k=vl, a_lead_k=al))
          assert f is None or -PROFILE_A_MAX <= f <= 0.


def build_sm(v_ego, standstill, d_rel, v_lead, personality=log.LongitudinalPersonality.aggressive, a_lead=0., model_accel=0.):
  services = {}
  for service in ("vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation", "carControl"):
    services[service] = getattr(messaging.new_message(service), service)
  ss = messaging.new_message('selfdriveState')
  ss.selfdriveState.personality = personality
  services['selfdriveState'] = ss.selfdriveState.as_reader()
  cs = messaging.new_message('controlsState')
  cs.controlsState.longControlState = 'pid'
  services['controlsState'] = cs.controlsState.as_reader()
  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.standstill = standstill
  car_state.carState.vCruise = 60.0
  car_state.carState.vCruiseCluster = 60.0
  services['carState'] = car_state.carState.as_reader()
  radar = messaging.new_message('radarState')
  radar.radarState.leadOne = {"present": True, "radar": True, "dRel": d_rel, "vLead": v_lead, "vLeadK": v_lead, "aLeadK": a_lead,
                              "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()
  m = messaging.new_message('modelV2')
  m.modelV2.velocity.x = [v_ego] * 33
  m.modelV2.position.x = [float(i) for i in range(33)]
  m.modelV2.meta.disengagePredictions.gasPressProbs = [1.] * 6
  m.modelV2.action.desiredAcceleration = model_accel
  services['modelV2'] = m.modelV2.as_reader()
  return MockSubMaster(services)


def roll_up(planner, v0=1.3, d0=11., t=14., creep=True):
  """Kinematic truck that does what the planner asks, half a second late, rolling up behind a stopped car. With creep,
  like the truck below 2 m/s, a braking request comes out 0.1 m/s^2 lighter (routes 0000001d/1e). Once the planner says
  stop, it stops at stopAccel like LongControl's stopping state. Returns the final speed and gap, and whether Stop Behind
  Lead's stop engaged."""
  v, d, q, engaged = v0, d0, [0.] * 10, False
  for _ in range(int(t / DT_MDL)):
    planner.update(build_sm(v, v < 0.01, d, 0.))
    engaged = engaged or planner.stop_behind_lead.active
    q.append(-0.2 if planner.output_should_stop else float(planner.output_a_target))
    a = q.pop(0)
    if creep and 0. < v < 2. and a <= 0. and not planner.output_should_stop:
      a += 0.1
    v = max(v + a * DT_MDL, 0.)
    d -= v * DT_MDL
  return v, d, engaged


class TestStopBehindLeadPlanner(OpenpilotTestCase):
  def test_stops_at_the_gap_without_crawling(self):
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    v, d, engaged = roll_up(planner)
    assert v == 0. and engaged
    assert abs(d - GAP_MAX) < 0.5, d

  def test_without_it_the_truck_crawls_past_the_gap(self):
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    planner.stop_behind_lead.available = False
    v, d, engaged = roll_up(planner)
    assert not engaged
    assert d < GAP_MAX - 0.75, d                      # what it fixes: the creep carries it in well past the gap

  def test_the_even_stop_lands_at_the_gap(self):
    # a truck that delivers exactly what is asked: the lead planner alone would stop about 6 m back
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    _, d, _ = roll_up(planner, creep=False)
    stock = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    stock.stop_behind_lead.profile_floor = lambda *a, **k: None
    _, d_stock, _ = roll_up(stock, creep=False)
    assert abs(d - GAP_MAX) < 0.25 and d_stock - d > 0.4, (d, d_stock)

  def test_launch_is_not_held_up(self):
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    roll_up(planner)
    assert planner.output_should_stop
    planner.update(build_sm(0., True, GAP_MAX, LEAD_DEPART_V_K + 0.05))
    assert not planner.stop_behind_lead.active


class TestProfileFloorPlanner(OpenpilotTestCase):
  V, D_LEAD = 4., 15.
  FLOOR = -V ** 2 / (2 * (D_LEAD - GAP_MAX))

  def plan(self, monkeypatch, mpc_accel, v_lead=0., model_accel=None):
    import openpilot.selfdrive.controls.lib.longitudinal_planner as lp
    monkeypatch.setattr(lp, 'get_accel_from_plan', lambda *a, **k: mpc_accel)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=self.V)
    if model_accel is not None:
      monkeypatch.setattr(planner, 'is_e2e', lambda sm: True)
    sm = build_sm(self.V, False, self.D_LEAD, v_lead, model_accel=model_accel or 0.)
    for _ in range(40):
      planner.update(sm)
    return float(planner.output_a_target), planner.mpc.source

  def test_relaxes_the_lead_planners_extra_braking_to_an_even_stop(self, monkeypatch):
    assert self.plan(monkeypatch, -2.) == (pytest.approx(self.FLOOR), LongitudinalPlanSource.lead0)

  def test_never_less_braking_than_the_lead_planner_asks_for_short_of_it(self, monkeypatch):
    assert self.plan(monkeypatch, self.FLOOR / 2) == (pytest.approx(self.FLOOR / 2), LongitudinalPlanSource.lead0)

  def test_anything_else_braking_harder_still_wins(self, monkeypatch):
    assert self.plan(monkeypatch, -2., model_accel=-1.8) == (pytest.approx(-1.8), LongitudinalPlanSource.e2e)

  def test_not_behind_a_moving_car(self, monkeypatch):
    assert self.plan(monkeypatch, -2., v_lead=8.)[0] == pytest.approx(-2.)
