"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The fast lead acceleration estimate: the slope of the radar's raw lead speed over 8 planner frames, used for leadOne in
place of aLeadK when it shows the car braking clearly harder than aLeadK does, in Traffic Mode and at Closer and Close.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, FAST_LEAD_A_TAU
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.radard import Track, KalmanParams
from openpilot.sunnypilot.selfdrive.controls.lib.fast_lead_accel import FastLeadAccel, WINDOW, A_MIN, A_CLIP, V_EGO_MIN, \
  SLOPE_WEIGHTS
from openpilot.sunnypilot.selfdrive.controls.lib.follow_distance import FAST_LEAD_LEVELS, NAMES
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_follow_distance import PARAM, car_params, long_active_sm
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockPubMaster, build_sm

V_EGO = 8.
TRACK = 7


def lead(v_lead, a_lead_k=-1., track=TRACK, present=True, radar=True):
  return SimpleNamespace(present=present, radar=radar, radarTrackId=track, vLead=v_lead, aLeadK=a_lead_k)


def feed(est, v_leads, a_lead_k=-1., v_ego=V_EGO, long_active=True, traffic_mode=True, close_level=False, track=TRACK):
  """one planner frame per vLead; returns what the MPC got each frame"""
  return [est.update(lead(v, a_lead_k, track), v_ego, long_active, traffic_mode, close_level) for v in v_leads]


def ramp(slope, n=WINDOW, v0=10.):
  return [v0 + slope * i * DT_MDL for i in range(n)]


class TestSlope(OpenpilotTestCase):
  def test_least_squares_slope_of_the_window(self):
    rng = np.random.default_rng(0)
    t = np.arange(WINDOW) * DT_MDL
    for _ in range(20):
      v = 10. + rng.normal(0., 1., WINDOW)
      self.assertAlmostEqual(float(np.dot(SLOPE_WEIGHTS, v)), float(np.polyfit(t, v, 1)[0]), places=9)
    assert WINDOW == 8 and abs(WINDOW * DT_MDL - 0.4) < 1e-9

  def test_braking_lead_after_a_full_window(self):
    est = FastLeadAccel()
    out = feed(est, ramp(-3.), a_lead_k=-1.)
    # nothing until the window is full, then the slope
    assert out[:WINDOW - 1] == [None] * (WINDOW - 1)
    self.assertAlmostEqual(out[-1], -3.)
    assert est.a_lead == out[-1]

  def test_braking_only(self):
    # a car pulling away: radard's aLeadK, in Traffic Mode and at Closer and Close alike
    for v_ego, traffic_mode, close_level in ((V_EGO, True, False), (V_EGO, False, True), (30., False, True)):
      assert feed(FastLeadAccel(), ramp(1.5), a_lead_k=0.4, v_ego=v_ego, traffic_mode=traffic_mode,
                  close_level=close_level) == [None] * WINDOW


class TestRules(OpenpilotTestCase):
  def test_over_a_min_only(self):
    for slope in (-0.45, -0.49):
      assert feed(FastLeadAccel(), ramp(slope), a_lead_k=-0.1)[-1] is None, slope
    self.assertAlmostEqual(feed(FastLeadAccel(), ramp(-0.51), a_lead_k=-0.1)[-1], -0.51)
    assert A_MIN == 0.5

  def test_same_sign_as_a_lead_k(self):
    # aLeadK saying the other way, or exactly 0: not used
    for a_lead_k in (0.3, 0.):
      assert feed(FastLeadAccel(), ramp(-2.), a_lead_k=a_lead_k)[-1] is None

  def test_stronger_than_a_lead_k_only(self):
    # aLeadK already as strong: radard's estimate stays
    assert feed(FastLeadAccel(), ramp(-2.), a_lead_k=-2.5)[-1] is None
    assert feed(FastLeadAccel(), ramp(-2.), a_lead_k=-2.01)[-1] is None
    self.assertAlmostEqual(feed(FastLeadAccel(), ramp(-2.), a_lead_k=-1.99)[-1], -2.)

  def test_clipped(self):
    assert A_CLIP == -6.
    assert feed(FastLeadAccel(), ramp(-9.), a_lead_k=-1.)[-1] == -6.

  def test_radar_lead_noise_does_not_trigger(self):
    # a lead holding 10 m/s with vLead noise within +-0.1 m/s, aLeadK from radard's own Kalman filter on that noise.
    # Radar-like noise (this truck's logs: vLead std 0.02-0.05 m/s about a moving mean, p99 0.05-0.13): sigma 0.033 m/s,
    # clipped to 0.1, for 10 minutes; and the fastest +-0.1 flicker (every frame) for a minute
    rng = np.random.default_rng(0)
    for noise in (np.clip(rng.normal(0., 0.033, 12000), -0.1, 0.1), 0.1 * (-1.) ** np.arange(1200)):
      est, track = FastLeadAccel(), Track(TRACK, 10., KalmanParams(DT_MDL))
      for n in noise:
        track.update(20., 0., n, 10. + n)
        assert est.update(SimpleNamespace(**track.get_RadarState(1.)), V_EGO, True, True, False) is None


class TestGate(OpenpilotTestCase):
  def test_traffic_mode_at_any_speed(self):
    # past 40 mph too: a car that sped up past it and then braked found the shorter town gap still there
    for v_ego, used in ((V_EGO_MIN - 0.01, False), (V_EGO_MIN, True), (13.42, True), (17.88, True), (17.89, True), (25., True),
                        (35., True)):
      assert (feed(FastLeadAccel(), ramp(-3.), v_ego=v_ego)[-1] is not None) == used, v_ego
    # and only with Traffic Mode on
    assert feed(FastLeadAccel(), ramp(-3.), traffic_mode=False)[-1] is None

  def test_closer_and_close_at_any_speed(self):
    for v_ego in (V_EGO_MIN, 10., 17.89, 20., 35.):
      for traffic_mode in (False, True):
        self.assertAlmostEqual(feed(FastLeadAccel(), ramp(-3.), v_ego=v_ego, traffic_mode=traffic_mode, close_level=True)[-1], -3.)
    # not near a stop, where the truck's own speed has a dead band
    assert feed(FastLeadAccel(), ramp(-3.), v_ego=V_EGO_MIN - 0.01, close_level=True)[-1] is None
    assert FAST_LEAD_LEVELS == (0, 1) and [NAMES[level] for level in FAST_LEAD_LEVELS] == ["Closer", "Close"]

  def test_radar_lead_and_long_active_only(self):
    est = FastLeadAccel()
    assert all(est.update(lead(v, radar=False), V_EGO, True, True, True) is None for v in ramp(-3.))
    assert all(est.update(lead(v, present=False), V_EGO, True, True, True) is None for v in ramp(-3.))
    assert feed(FastLeadAccel(), ramp(-3.), long_active=False, close_level=True)[-1] is None


class TestHistory(OpenpilotTestCase):
  def test_a_new_track_starts_a_new_window(self):
    est = FastLeadAccel()
    assert feed(est, ramp(-3.))[-1] is not None
    # leadOne is now another car: its own 8 frames first, never a slope across the two
    out = feed(est, ramp(-3., v0=4.), track=TRACK + 1)
    assert out[:WINDOW - 1] == [None] * (WINDOW - 1)
    self.assertAlmostEqual(out[-1], -3.)

  def test_reset_by_a_missing_lead_a_stop_or_long_off(self):
    def interrupt_then_resume(interrupt):
      est = FastLeadAccel()
      assert feed(est, ramp(-3.))[-1] is not None
      interrupt(est)
      return feed(est, ramp(-3., v0=9.))

    interruptions = (
      lambda est: est.update(lead(10., present=False), V_EGO, True, True, False),          # lead lost for a frame
      lambda est: est.update(lead(10.), V_EGO_MIN - 0.5, True, True, False),                # near a stop
      lambda est: est.update(lead(10.), V_EGO, False, True, False),                         # long off
      lambda est: est.update(lead(10., radar=False), V_EGO, True, True, False),             # vision lead
    )
    for interrupt in interruptions:
      out = interrupt_then_resume(interrupt)
      # the old samples are gone: a full new window before the next estimate
      assert out[:WINDOW - 1] == [None] * (WINDOW - 1)
      self.assertAlmostEqual(out[-1], -3.)


  def test_window_kept_while_the_gate_is_off(self):
    # the window fills with Traffic Mode off at Medium, so the estimate is there the frame the gate opens
    est = FastLeadAccel()
    v = ramp(-3., n=WINDOW + 1)
    assert feed(est, v[:WINDOW], traffic_mode=False) == [None] * WINDOW
    self.assertAlmostEqual(feed(est, v[WINDOW:])[-1], -3.)
    est = FastLeadAccel()
    v = ramp(-3., n=WINDOW + 1)
    assert feed(est, v[:WINDOW], traffic_mode=False) == [None] * WINDOW
    self.assertAlmostEqual(feed(est, v[WINDOW:], traffic_mode=False, close_level=True)[-1], -3.)

class TestMpcAndPlanner(OpenpilotTestCase):
  def test_mpc_plans_lead_one_with_it(self):
    mpc = LongitudinalMpc()
    mpc.set_cur_state(V_EGO, 0.)
    lead_one = SimpleNamespace(present=True, radar=True, dRel=20., vLead=8., vLeadK=8., aLeadK=-1., aLeadTau=1.2, modelProb=1.)
    # in place of aLeadK, with aLeadTau at most FAST_LEAD_A_TAU
    np.testing.assert_allclose(mpc.process_lead(lead_one, -3.), mpc.extrapolate_lead(20., 8., -3., FAST_LEAD_A_TAU))
    np.testing.assert_allclose(mpc.process_lead(lead_one), mpc.extrapolate_lead(20., 8., -1., 1.2))
    lead_one.aLeadTau = 0.2
    np.testing.assert_allclose(mpc.process_lead(lead_one, -3.), mpc.extrapolate_lead(20., 8., -3., 0.2))
    # leadOne only: leadTwo keeps its own aLeadK
    lead_two = SimpleNamespace(**vars(lead_one))
    mpc.a_lead_fast = -3.
    mpc.update(SimpleNamespace(leadOne=SimpleNamespace(**vars(lead_one), radarTrackId=1), leadTwo=lead_two))
    plan_fast = mpc.x_sol[:, 0].copy()
    mpc.reset()
    mpc.set_cur_state(V_EGO, 0.)
    mpc.a_lead_fast = None
    mpc.update(SimpleNamespace(leadOne=SimpleNamespace(**vars(lead_one), radarTrackId=1), leadTwo=lead_two))
    # the harder braking lead brings the plan up shorter
    assert plan_fast[-1] < mpc.x_sol[-1, 0] - 1.

  def test_planner_hands_it_over_and_logs_it(self):
    params = Params()
    params.put("SPFollowingTimeOffset", -10, block=True)
    for traffic_mode, level, used in ((True, 2, True), (False, 2, False), (False, 0, True), (True, 0, True), (False, 1, True),
                                      (False, 3, False)):
      params.put_bool("TrafficMode", traffic_mode, block=True)
      params.put(PARAM, level, block=True)
      planner = LongitudinalPlanner(*car_params(), init_v=V_EGO)
      pm = MockPubMaster()
      sent = []
      for i, v_lead in enumerate(ramp(-3., n=12)):
        sm = long_active_sm(build_sm(V_EGO), True)
        rs = sm['radarState'].as_builder()
        rs.leadOne.radar = True
        rs.leadOne.radarTrackId = TRACK
        rs.leadOne.vLead = v_lead
        rs.leadOne.aLeadK = -0.8
        sm['radarState'] = rs.as_reader()
        planner.update(sm)
        planner.publish(sm, pm)
        sent.append(planner.mpc.a_lead_fast)
        logged = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance.aLeadFast
        self.assertAlmostEqual(logged, 0. if sent[-1] is None else sent[-1], places=5)
        if i < WINDOW - 1:
          assert sent[-1] is None
      if used:
        self.assertAlmostEqual(sent[-1], -3., places=5)
      else:
        assert sent[-1] is None and logged == 0.

  def test_other_cars_in_traffic_mode_too(self):
    # Traffic Mode is any car's with openpilot longitudinal; Closer is Rivian only
    params = Params()
    params.put_bool("TrafficMode", True, block=True)
    planner = LongitudinalPlanner(*car_params(brand="toyota"), init_v=V_EGO)
    assert planner.follow_distance is None
    for v_lead in ramp(-3.):
      sm = long_active_sm(build_sm(V_EGO), True)
      rs = sm['radarState'].as_builder()
      rs.leadOne.radar = True
      rs.leadOne.radarTrackId = TRACK
      rs.leadOne.vLead = v_lead
      rs.leadOne.aLeadK = -0.8
      sm['radarState'] = rs.as_reader()
      planner.update(sm)
    self.assertAlmostEqual(planner.mpc.a_lead_fast, -3., places=5)
