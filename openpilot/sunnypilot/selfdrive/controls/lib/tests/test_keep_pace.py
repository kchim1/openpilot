"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log, messaging
from opendbc.car import structs
from openpilot.common.params import Params, ParamKeyFlag, UnknownKeyName
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_max_accel, get_cruise_accel, \
  A_CRUISE_MAX_VALS, A_FOLLOW_MAX_VALS
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import get_safe_obstacle_distance, get_stopped_equivalence_factor
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, KEEP_PACE_READ_FRAMES, \
  KEEP_PACE_V_MIN, KEEP_PACE_EXCESS_BP, KEEP_PACE_OFF_FRAMES, keep_pace_share
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

Personality = log.LongitudinalPersonality
AGGRESSIVE, STANDARD, RELAXED = Personality.aggressive, Personality.standard, Personality.relaxed
CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
DT = 0.05


T_FOLLOW = 1.25
V_EGO = 5.


def desired(v_lead=8.):
  return get_safe_obstacle_distance(V_EGO, T_FOLLOW) - get_stopped_equivalence_factor(v_lead)


def lead(present=True, radar=True, v_lead=8., excess=KEEP_PACE_EXCESS_BP[1]):
  # excess: meters past the distance the lead planner wants
  return SimpleNamespace(present=present, radar=radar, vLead=v_lead, dRel=desired(v_lead) + excess)


def settle_cruise_accel(v_ego, follow_share, angle=0., frames=200):
  a = 0.
  for _ in range(frames):
    a = get_cruise_accel(False, 40., v_ego, a, angle, CP, DT, -0.3, True, follow_share)
  return a


class TestKeepPaceCap(OpenpilotTestCase):
  def test_cap_values(self):
    # stock, then Aggressive (all of the step), Standard (half) and Relaxed (a quarter)
    for v_ego, stock, aggressive, standard, relaxed in ((0., 1.6, 2.0, 1.8, 1.7), (10., 1.2, 2.0, 1.6, 1.4),
                                                        (25., 0.8, 1.4, 1.1, 0.95), (40., 0.6, 1.0, 0.8, 0.7)):
      self.assertAlmostEqual(get_max_accel(v_ego), stock)
      for personality, cap in ((AGGRESSIVE, aggressive), (STANDARD, standard), (RELAXED, relaxed)):
        self.assertAlmostEqual(get_max_accel(v_ego, keep_pace_share(personality)), cap)
    assert all(f >= s for f, s in zip(A_FOLLOW_MAX_VALS, A_CRUISE_MAX_VALS, strict=True))
    assert keep_pace_share(AGGRESSIVE) > keep_pace_share(STANDARD) > keep_pace_share(RELAXED) > 0.

  def test_share_from_a_selfdrive_state_message(self):
    # the planner passes selfdriveState.personality, a capnp enum
    for personality, share in ((AGGRESSIVE, 1.), (STANDARD, 0.5), (RELAXED, 0.25)):
      msg = messaging.new_message('selfdriveState')
      msg.selfdriveState.personality = personality
      assert keep_pace_share(msg.selfdriveState.as_reader().personality) == share

  def test_cruise_candidate_reaches_the_higher_cap_on_a_straight_road(self):
    # stock: the cap, and below 20 m/s the 1.7 total-accel limit; following: 2.0 (the turn limit is raised to match)
    self.assertAlmostEqual(settle_cruise_accel(5., 0.), 1.4)
    self.assertAlmostEqual(settle_cruise_accel(5., 1.), 2.0)
    self.assertAlmostEqual(settle_cruise_accel(5., 0.5), 1.7)
    self.assertAlmostEqual(settle_cruise_accel(18., 1.), np.interp(18., [10., 25.], [2.0, 1.4]))

  def test_curves_keep_the_stock_friction_circle(self):
    # from 1 m/s^2 lateral the total limit is the stock 1.7 again (the higher straight-line cap can still apply inside
    # it); the reviewed failure was 1.05 m/s^2 allowed at 1.7 lateral where stock allows none
    for a_y in (1.0, 1.5, 1.7, 1.9, 3.0):
      angle = a_y * CP.steerRatio * CP.wheelbase / 15. ** 2 / (np.pi / 180.)
      assert settle_cruise_accel(15., 1., angle=angle) <= np.sqrt(max(1.7 ** 2 - a_y ** 2, 0.)) + 1e-6
    for a_y in (1.7, 1.9, 3.0):
      angle = a_y * CP.steerRatio * CP.wheelbase / 15. ** 2 / (np.pi / 180.)
      assert settle_cruise_accel(15., 1., angle=angle) == 0.
    self.assertAlmostEqual(settle_cruise_accel(15., 1.), np.interp(15., [10., 25.], [2.0, 1.4]))

  def test_e2e_unchanged(self):
    for follow_share in (0., 1.):
      a = 0.
      for _ in range(200):
        a = get_cruise_accel(True, 40., 5., a, 0., CP, DT, -0.3, True, follow_share)
      self.assertAlmostEqual(a, 2.0)


class TestKeepPaceGate(OpenpilotTestCase):
  @staticmethod
  def make(enabled=True, openpilot_long=True):
    planner = LongitudinalPlannerSP(structs.CarParams(openpilotLongitudinalControl=openpilot_long), structs.CarParamsSP(),
                                    SimpleNamespace(traffic_mode=False, t_follow=T_FOLLOW))
    planner._keep_pace_enabled = enabled
    return planner

  def test_only_behind_a_moving_radar_lead(self):
    for ld in (lead(present=False), lead(radar=False), lead(v_lead=KEEP_PACE_V_MIN - 0.1)):
      assert self.make().keep_pace(ld, AGGRESSIVE, V_EGO) == 0.
    assert self.make(enabled=False).keep_pace(lead(), AGGRESSIVE, V_EGO) == 0.

  def test_only_while_the_gap_is_past_the_desired_one(self):
    # none at or inside the desired distance, building to all of it 10 m past, none once the lead is 40 m past
    planner = self.make()
    for excess, share in ((-5., 0.), (0., 0.), (5., 0.5), (10., 1.), (30., 1.), (35., 0.5), (45., 0.)):
      self.assertAlmostEqual(planner.keep_pace(lead(excess=excess), AGGRESSIVE, V_EGO), share)

  def test_half_by_highway_speed(self):
    planner = self.make()
    for v_ego, share in ((5., 1.), (10., 1.), (15., 0.75), (20., 0.5), (30., 0.5)):
      ld = SimpleNamespace(present=True, radar=True, vLead=v_ego + 3.,
                           dRel=get_safe_obstacle_distance(v_ego, T_FOLLOW) - get_stopped_equivalence_factor(v_ego + 3.) + 20.)
      self.assertAlmostEqual(planner.keep_pace(ld, AGGRESSIVE, v_ego), share)

  def test_share_follows_the_personality(self):
    planner = self.make()
    for personality, share in ((AGGRESSIVE, 1.), (STANDARD, 0.5), (RELAXED, 0.25)):
      assert planner.keep_pace(lead(), personality, V_EGO) == share

  def test_stays_on_briefly_after_the_lead_drops(self):
    planner = self.make()
    planner.keep_pace(lead(), AGGRESSIVE, V_EGO)
    for _ in range(KEEP_PACE_OFF_FRAMES - 1):
      assert planner.keep_pace(lead(present=False), AGGRESSIVE, V_EGO) == 1.
    assert planner.keep_pace(lead(present=False), AGGRESSIVE, V_EGO) == 0.

  def test_setting_read_every_second(self):
    params = Params()
    params.put_bool("KeepPaceWithTraffic", False, block=True)
    planner = LongitudinalPlannerSP(CP, structs.CarParamsSP(), SimpleNamespace(traffic_mode=False, t_follow=T_FOLLOW))
    planner.update_keep_pace()
    assert not planner._keep_pace_enabled
    params.put_bool("KeepPaceWithTraffic", True, block=True)
    for _ in range(KEEP_PACE_READ_FRAMES - 1):
      planner.update_keep_pace()
      assert not planner._keep_pace_enabled
    planner.update_keep_pace()
    assert planner._keep_pace_enabled

  def test_not_read_without_openpilot_long(self):
    Params().put_bool("KeepPaceWithTraffic", True, block=True)
    planner = LongitudinalPlannerSP(structs.CarParams(openpilotLongitudinalControl=False), structs.CarParamsSP(),
                                    SimpleNamespace(traffic_mode=False, t_follow=T_FOLLOW))
    for _ in range(3 * KEEP_PACE_READ_FRAMES):
      planner.update_keep_pace()
    assert not planner._keep_pace_enabled

  def test_read_error_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner.cloudlog")
    planner = self.make()
    planner._params = mocker.MagicMock()
    planner._params.get_bool.side_effect = UnknownKeyName(b"KeepPaceWithTraffic")
    for _ in range(3 * KEEP_PACE_READ_FRAMES):
      planner.update_keep_pace()
    assert not planner._keep_pace_enabled
    assert cloudlog.exception.call_count == 1

  def test_default_on_and_kept(self):
    params = Params()
    # manager writes the default for unset keys at startup
    assert params.get_default_value("KeepPaceWithTraffic") is True
    params.put_bool("KeepPaceWithTraffic", False, block=True)
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
    assert params.get_bool("KeepPaceWithTraffic") is False
    assert b"KeepPaceWithTraffic" in params.all_keys(ParamKeyFlag.BACKUP)


def build_sm(v_ego, d_rel, v_lead, personality=AGGRESSIVE):
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
  car_state.carState.vCruise = 120.0
  car_state.carState.vCruiseCluster = 120.0
  services['carState'] = car_state.carState.as_reader()
  radar = messaging.new_message('radarState')
  radar.radarState.leadOne = {"present": True, "radar": True, "dRel": d_rel, "vLead": v_lead, "vLeadK": v_lead, "aLeadK": 2.,
                              "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()
  m = messaging.new_message('modelV2')
  m.modelV2.orientationRate.z = [0.01] * 33
  m.modelV2.velocity.x = [v_ego] * 33
  m.modelV2.position.x = [float(i) for i in range(33)]
  m.modelV2.meta.disengagePredictions.gasPressProbs = [1.] * 6
  services['modelV2'] = m.modelV2.as_reader()
  return MockSubMaster(services)


class TestKeepPacePlanner(OpenpilotTestCase):
  def run_planner(self, enabled, personality=AGGRESSIVE):
    Params().put_bool("KeepPaceWithTraffic", enabled, block=True)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=5.)
    # a lead 25 m ahead pulling away at 9 m/s while the truck does 5 m/s: the lead planner wants all it can get
    sm = build_sm(5., 25., 9., personality)
    for _ in range(100):
      planner.update(sm)
    return planner

  def test_lead_pulling_away_gets_the_higher_cap(self):
    # the cruise candidate is the cap here (set speed far above): stock 1.4 at 5 m/s, 2.0 while following
    stock = self.run_planner(False)
    self.assertAlmostEqual(stock.a_cruise, get_max_accel(5.), places=3)
    keep = self.run_planner(True)
    self.assertAlmostEqual(keep.a_cruise, 2.0, places=3)
    assert keep.output_a_target <= 2.0 + 1e-6
    for personality, cap in ((STANDARD, 1.7), (RELAXED, 1.55)):
      self.assertAlmostEqual(self.run_planner(True, personality).a_cruise, cap, places=3)
