"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import log, messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, COMFORT_BRAKE, APPROACH_MARGIN_RATE
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import BRAKE_SOONER_COMFORT_BRAKE, APPROACH_ONSET_JERK, \
  APPROACH_ONSET_JERK_URGENT, APPROACH_ONSET_A_BP, APPROACH_ONSET_V_LEAD_MAX, APPROACH_ONSET_A_LEAD, APPROACH_ONSET_OTHER_MARGIN, \
  APPROACH_ONSET_RESET_HOLD
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

AGGRESSIVE = log.LongitudinalPersonality.aggressive
CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
V_EGO = 15.6  # 35 mph
MORE = BRAKE_SOONER_COMFORT_BRAKE


def margin_for(v_ego, v_lead, comfort_brake=MORE):
  return (1 / (2 * comfort_brake) - 1 / (2 * COMFORT_BRAKE)) * (v_ego**2 - v_lead**2)


def lead(d_rel, v_lead=0., present=True):
  return SimpleNamespace(present=present, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=0., aLeadTau=1.5, modelProb=1., radar=True)


def run_mpc(comfort_brake, d_rel, v_lead=0., v_ego=V_EGO, frames=80, present=True):
  """the same lead and speed every frame. Returns the MPC and the largest margin applied each frame"""
  mpc = LongitudinalMpc()
  mpc.approach_comfort_brake = comfort_brake
  radarstate = SimpleNamespace(leadOne=lead(d_rel, v_lead, present), leadTwo=lead(0., present=False))
  margins = []
  for _ in range(frames):
    mpc.set_cur_state(v_ego, 0.)
    mpc.update(radarstate, AGGRESSIVE)
    margins.append(mpc.approach_margin_last[0])
  return mpc, margins


class TestApproachMargin(OpenpilotTestCase):
  def test_off_is_stock(self):
    stock, margins = run_mpc(COMFORT_BRAKE, 115.)
    assert max(margins) == 0.
    plain = LongitudinalMpc()
    radarstate = SimpleNamespace(leadOne=lead(115.), leadTwo=lead(0., present=False))
    for _ in range(80):
      plain.set_cur_state(V_EGO, 0.)
      plain.update(radarstate, AGGRESSIVE)
    np.testing.assert_array_equal(stock.a_solution, plain.a_solution)

  def test_stopped_car_seen_early(self):
    # a gentle stop is still possible (0.98 m/s^2): all of the margin, built up at APPROACH_MARGIN_RATE
    _, margins = run_mpc(MORE, 130.)
    self.assertAlmostEqual(margins[0], APPROACH_MARGIN_RATE * 0.05)
    self.assertAlmostEqual(margins[9], 10 * APPROACH_MARGIN_RATE * 0.05)
    self.assertAlmostEqual(margins[-1], margin_for(V_EGO, 0.), delta=0.1)

  def test_brakes_sooner(self):
    # 115 m from a stopped car at 35 mph: the plan brakes harder now, so the truck starts braking sooner
    stock, _ = run_mpc(COMFORT_BRAKE, 115.)
    sooner, _ = run_mpc(MORE, 115.)
    assert sooner.a_solution[4] < stock.a_solution[4] - 0.2
    stronger = [run_mpc(cb, 115.)[0].a_solution[4] for cb in (COMFORT_BRAKE, 2.0, 1.75, MORE)]
    assert all(a > b for a, b in zip(stronger, stronger[1:], strict=False))

  def test_slow_lead_fades_out(self):
    self.assertAlmostEqual(run_mpc(MORE, 120., v_lead=2.)[1][-1], margin_for(V_EGO, 2.), delta=0.1)
    self.assertAlmostEqual(run_mpc(MORE, 120., v_lead=5.5)[1][-1], 0.5 * margin_for(V_EGO, 5.5), delta=0.3)
    assert max(run_mpc(MORE, 120., v_lead=8.)[1]) == 0.

  def test_car_seen_late_gets_the_stock_plan(self):
    # stopping 6 m short of a car 70 m ahead from 35 mph takes 1.9 m/s^2: no margin
    stock, _ = run_mpc(COMFORT_BRAKE, 70.)
    late, margins = run_mpc(MORE, 70.)
    assert max(margins) == 0.
    np.testing.assert_allclose(late.a_solution, stock.a_solution, atol=1e-6)

  def test_partial_between(self):
    # 100 m: 1.29 m/s^2 to stop, part of the margin
    margin = run_mpc(MORE, 100.)[1][-1]
    assert 0.5 * margin_for(V_EGO, 0.) < margin < margin_for(V_EGO, 0.)

  def test_no_lead(self):
    assert max(run_mpc(MORE, 120., present=False)[1]) == 0.

  def test_lead_lost_restarts_the_build_up(self):
    mpc, _ = run_mpc(MORE, 130.)
    assert mpc.approach_margin_last[0] > 15.
    mpc.update(SimpleNamespace(leadOne=lead(0., present=False), leadTwo=lead(0., present=False)), AGGRESSIVE)
    assert mpc.approach_margin_last[0] == 0.
    mpc.update(SimpleNamespace(leadOne=lead(130.), leadTwo=lead(0., present=False)), AGGRESSIVE)
    self.assertAlmostEqual(mpc.approach_margin_last[0], APPROACH_MARGIN_RATE * 0.05)

  def test_launch_behind_a_lead_nearly_unchanged(self):
    # truck at 1 m/s, lead pulling away at 2 m/s: the plan catches up a little late in the horizon, under a meter
    assert max(run_mpc(MORE, 10., v_lead=2., v_ego=1.)[1]) < 1.0


def build_sm(v_ego, d_rel, v_cruise=None, long_off=False, a_ego=0.):
  services = {}
  for service in ("vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation", "carControl"):
    services[service] = getattr(messaging.new_message(service), service)
  ss = messaging.new_message('selfdriveState')
  ss.selfdriveState.personality = AGGRESSIVE
  services['selfdriveState'] = ss.selfdriveState.as_reader()
  cs = messaging.new_message('controlsState')
  cs.controlsState.longControlState = 'off' if long_off else 'pid'
  services['controlsState'] = cs.controlsState.as_reader()
  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.aEgo = a_ego
  car_state.carState.vCruise = (v_cruise or v_ego) * 3.6
  car_state.carState.vCruiseCluster = (v_cruise or v_ego) * 3.6
  services['carState'] = car_state.carState.as_reader()
  radar = messaging.new_message('radarState')
  radar.radarState.leadOne = {"present": True, "radar": True, "dRel": d_rel, "vLead": 0., "vLeadK": 0., "aLeadK": 0.,
                              "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()
  m = messaging.new_message('modelV2')
  m.modelV2.orientationRate.z = [0.01] * 33
  m.modelV2.velocity.x = [v_ego] * 33
  m.modelV2.position.x = [float(i) for i in range(33)]
  m.modelV2.meta.disengagePredictions.gasPressProbs = [1.] * 6
  services['modelV2'] = m.modelV2.as_reader()
  return MockSubMaster(services)




class TestOnsetJerk(OpenpilotTestCase):
  @staticmethod
  def planner(brake_sooner=True):
    Params().put_bool("BrakeSoonerForStops", brake_sooner, block=True)
    p = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=V_EGO)
    p.dec._brake_sooner = brake_sooner
    return p

  def test_eases_in_while_a_gentle_stop_is_possible(self):
    p = self.planner()
    # 0.98 m/s^2 stops 6 m short of a car stopped 130 m ahead at 35 mph: full ease-in
    assert p.approach_onset_jerk(lead(130.), V_EGO) == APPROACH_ONSET_JERK
    # 2.0 m/s^2 (half way to urgent), 3 m/s^2 (urgent)
    self.assertAlmostEqual(p.approach_onset_jerk(lead(6. + V_EGO ** 2 / 4.), V_EGO),
                           (APPROACH_ONSET_JERK + APPROACH_ONSET_JERK_URGENT) / 2)
    assert p.approach_onset_jerk(lead(6. + V_EGO ** 2 / 6.), V_EGO) is None

  def test_only_the_start_of_braking(self):
    p = self.planner()
    assert p.approach_onset_jerk(lead(130.), V_EGO, APPROACH_ONSET_A_BP[0]) == APPROACH_ONSET_JERK
    self.assertAlmostEqual(p.approach_onset_jerk(lead(130.), V_EGO, sum(APPROACH_ONSET_A_BP) / 2),
                           (APPROACH_ONSET_JERK + APPROACH_ONSET_JERK_URGENT) / 2)
    self.assertAlmostEqual(p.approach_onset_jerk(lead(130.), V_EGO, APPROACH_ONSET_A_BP[1] + 1e-6), APPROACH_ONSET_JERK_URGENT, places=3)
    assert p.approach_onset_jerk(lead(130.), V_EGO, APPROACH_ONSET_A_BP[1]) is None

  def test_no_limit_when_urgent_or_already_close(self):
    p = self.planner()
    # 2.5 m/s^2 or more needed: stock braking at once
    assert p.approach_onset_jerk(lead(6. + V_EGO ** 2 / 5.), V_EGO) is None
    assert p.approach_onset_jerk(lead(6. + V_EGO ** 2 / 6.), V_EGO) is None
    # already within the stop distance
    assert p.approach_onset_jerk(lead(6.5), 1.5) is None
    assert p.approach_onset_jerk(lead(4.), 1.5) is None
    # braking already past the eased part
    assert p.approach_onset_jerk(lead(130.), V_EGO, APPROACH_ONSET_A_BP[1]) is None

  def test_car_seen_late_gets_stock_braking_at_once(self):
    # 50 mph, a stopped car first seen 80 m out (3.4 m/s^2 needed): the same requests with the setting on or off
    out = {}
    for on in (False, True):
      p = self.planner(on)
      for _ in range(20):
        p.update(build_sm(22.4, 400., v_cruise=22.4))
      out[on] = []
      for _ in range(10):
        p.update(build_sm(22.4, 80., v_cruise=22.4))
        out[on].append(float(p.output_a_target))
    assert out[True][0] < 0.
    np.testing.assert_allclose(out[True], out[False], atol=0.05)

  def test_gas_release_is_not_eased(self):
    # the driver on the gas at +1.5 m/s^2 (long control off) toward a car stopped 100 m ahead, then lets go: the planner's
    # own requests, not a ramp down from the driver's throttle, until the plan first gets to 0
    def run(no_ease):
      p = self.planner()
      if no_ease:
        p.approach_onset_jerk = lambda *a, **k: None   # the same planner and plan, without the ease-in
      for _ in range(40):
        p.update(build_sm(15., 130., v_cruise=25., long_off=True, a_ego=1.5))
      out = []
      for _ in range(30):
        p.update(build_sm(15., 100., v_cruise=25.))
        out.append(float(p.output_a_target))
      return out
    eased, plain = run(False), run(True)
    held = min(next((i for i, a in enumerate(plain) if a <= 0.), len(plain)), int(APPROACH_ONSET_RESET_HOLD / 0.05)) + 1
    assert plain[0] > 0.           # the plan right after the release still had throttle in it
    np.testing.assert_allclose(eased[:held], plain[:held], atol=1e-3)

  def test_braking_for_something_else_is_not_eased(self):
    # a slow radar car 90 m ahead (a gentle stop) while the winning request is much firmer, e.g. the model stopping for a
    # light in front of it: it goes through at once
    p = self.planner()
    for _ in range(40):
      p.update(build_sm(V_EGO, 400., v_cruise=V_EGO + 10.))
    a_prev = float(p.output_a_target)
    p.approach_onset_jerk(lead(90.), V_EGO, a_prev)
    need = p.approach_need
    assert 0. < need < 1.5
    orig = p.mpc.update
    def firm(*a, **k):   # the lead planner (standing in for any candidate) asks for far more than the lead needs
      orig(*a, **k)
      p.mpc.a_solution[:] = -(need + APPROACH_ONSET_OTHER_MARGIN + 1.5)
      p.mpc.v_solution[:] = np.maximum(V_EGO + np.cumsum(p.mpc.a_solution) * 0.2, 0.)
    p.mpc.update = firm
    p.update(build_sm(V_EGO, 90., v_cruise=V_EGO + 10.))
    assert float(p.output_a_target) < a_prev - 2 * APPROACH_ONSET_JERK * 0.05

  def test_only_for_a_stopped_or_slow_radar_lead_with_the_setting_on(self):
    p = self.planner()
    assert p.approach_onset_jerk(lead(130., v_lead=APPROACH_ONSET_V_LEAD_MAX + 0.1), V_EGO) is None
    braking = lead(130., v_lead=5.)
    braking.aLeadK = APPROACH_ONSET_A_LEAD - 0.1
    assert p.approach_onset_jerk(braking, V_EGO) is None
    vision = lead(130.)
    vision.radar = False
    assert p.approach_onset_jerk(vision, V_EGO) is None
    assert p.approach_onset_jerk(lead(130., present=False), V_EGO) is None
    assert self.planner(brake_sooner=False).approach_onset_jerk(lead(130.), V_EGO) is None

  def test_the_swing_into_braking_eases_in_and_lets_go_smoothly(self):
    # accelerating at the cruise cap toward a set speed 10 m/s higher when a car stopped 90 m ahead shows up
    p = self.planner()
    for _ in range(40):
      p.update(build_sm(V_EGO, 400., v_cruise=V_EGO + 10.))
    out = [float(p.output_a_target)]
    for _ in range(60):
      p.update(build_sm(V_EGO, 90., v_cruise=V_EGO + 10.))
      out.append(float(p.output_a_target))
    for prev, cur in zip(out[:-1], out[1:], strict=True):
      jerk = p.approach_onset_jerk(lead(90.), V_EGO, prev)
      if jerk is not None:
        assert prev - cur <= jerk * 0.05 + 1e-6
    assert min(out) < APPROACH_ONSET_A_BP[1]       # and it still brakes past the eased part

  def test_setting_sets_the_lead_planners_comfort_brake(self):
    for on, cb in ((True, MORE), (False, COMFORT_BRAKE)):
      p = self.planner(on)
      p.update(build_sm(V_EGO, 130.))
      assert p.mpc.approach_comfort_brake == cb


class TestBrakeSoonerPlanner(OpenpilotTestCase):
  def run_planner(self, on):
    Params().put_bool("BrakeSoonerForStops", on, block=True)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=V_EGO)
    planner.dec._brake_sooner = on
    sm = build_sm(V_EGO, 115.)
    for _ in range(60):
      planner.update(sm)
    return planner

  def test_setting_reaches_the_plan(self):
    # cruising at the set speed toward a stopped car 115 m ahead
    off, on = self.run_planner(False), self.run_planner(True)
    assert off.mpc.approach_comfort_brake == COMFORT_BRAKE and on.mpc.approach_comfort_brake == MORE
    assert on.output_a_target < off.output_a_target - 0.1
