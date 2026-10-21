"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The speed-cap invariant in LongitudinalPlannerSP.update_targets().

A source wins the arbitration on the lowest SPEED target, but the a_target that comes back becomes the planner's
acceleration state (output_a_target): it seeds the MPC and is integrated into v_desired_filter. Every source in the
contest except `cruise` is a speed CAP, so none of them may ever ask for MORE acceleration than that state already is.
Without that clamp, curve control holding a cap the truck is already below (because a lead is braking it under it), or
SCC-V leaving a turn, hands the planner an acceleration the truck is not making, v_desired_filter drifts above the real
speed, and the MPC brakes for a truck going faster than it is.

update_targets() is arbitration plus arithmetic, so the sub-controllers are faked rather than driven. That keeps each
case pinned to one published (v_target, a_target) pair, which is what the invariant is about; the controllers have
their own tests. The later cases drive the real LongitudinalPlanner: where the clamped value goes, Curve Speed Control's
throttle fade on the cruise accel cap, and the real curve controller and governor publishing into the plan.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import custom, messaging
from opendbc.car import structs
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, A_CRUISE_MAX_BP, J_CRUISE_VALS, \
  get_cruise_accel, get_max_accel
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.curve_speed_controller import A_LAT_MAX
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster, MockPubMaster

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource
CurveState = custom.LongitudinalPlanSP.SmartCruiseControl.CurveState

UNSET = 1e3  # far above any real v_target, so an unset source never wins the min-by-speed contest


class FakeSource:
  def __init__(self, v_target=UNSET, a_target=0.0):
    self.output_v_target = v_target
    self.output_a_target = a_target

  def update(self, *args):
    pass


class FakeGovernor(FakeSource):
  def __init__(self, v_target=UNSET, a_target=0.0, throttle_scale=1.0):
    super().__init__(v_target, a_target)
    self._throttle_scale = throttle_scale

  def throttle_scale(self):
    return self._throttle_scale


class FakeSCC:
  def __init__(self, vision=None, scc_map=None, curve=None, governor=None):
    self.vision = vision or FakeSource()
    self.map = scc_map or FakeSource()
    self.curve = curve or FakeSource()
    self.governor = governor or FakeGovernor()

  def update(self, *args):
    pass


class FakeResolver:
  speed_limit_valid = False
  speed_limit_last_valid = False
  speed_limit = 0.0
  speed_limit_final_last = 0.0
  distance = 0.0

  def update(self, *args):
    pass


def _targets(v_ego, a_ego, v_cruise, vision=None, scc_map=None, sla=None, curve=None, governor=None):
  """Run update_targets() with the sub-controllers faked. Returns (source, v_target, a_target)."""
  planner = object.__new__(LongitudinalPlannerSP)
  planner.events_sp = SimpleNamespace()  # handed to the faked Speed Limit Assist, unused
  planner.resolver = FakeResolver()
  planner.sla = sla or FakeSource()
  planner.scc = FakeSCC(vision, scc_map, curve, governor)
  planner.source = LongitudinalPlanSource.cruise
  planner.output_v_target = 0.0
  planner.output_a_target = 0.0

  sm = {
    'carState': SimpleNamespace(vCruiseCluster=v_cruise * 3.6),
    'carControl': SimpleNamespace(enabled=True, cruiseControl=SimpleNamespace(override=False)),
  }
  v_target, a_target = planner.update_targets(sm, v_ego, a_ego, v_cruise)
  return planner.source, v_target, a_target


class TestSpeedCapInvariant:
  def test_cap_still_wins_the_speed_contest(self):
    # the clamp must not disturb arbitration: the lowest v_target still selects the source
    source, v_target, _ = _targets(v_ego=25.0, a_ego=0.0, v_cruise=30.0,
                                   vision=FakeSource(v_target=18.0, a_target=-1.0))
    assert source == LongitudinalPlanSource.sccVision
    assert v_target == 18.0

  def test_phantom_acceleration_is_dropped(self):
    # SCC-V leaving a turn asks for +0.5 m/s^2 while a lead has braked the truck below its cap. It still wins on
    # speed, but that positive a_target must not become the planner's acceleration state
    _, _, a_target = _targets(v_ego=15.0, a_ego=-2.5, v_cruise=30.0,
                              vision=FakeSource(v_target=20.0, a_target=+0.5))
    assert a_target == -2.5

  def test_a_cap_own_decel_is_preserved(self):
    # the clamp is one-way. A cap braking harder than the truck currently is must pass through untouched, or SCC-V
    # entering a turn could no longer slow the truck at all
    _, _, a_target = _targets(v_ego=25.0, a_ego=-0.5, v_cruise=30.0,
                              vision=FakeSource(v_target=18.0, a_target=-1.0))
    assert a_target == -1.0

  def test_accel_out_is_bounded_by_a_ego(self):
    # accelerating out of a turn is left to the MPC and the cruise accel; a cap may ask for at most what the planner
    # is already asking for
    _, _, a_target = _targets(v_ego=20.0, a_ego=0.2, v_cruise=30.0,
                              vision=FakeSource(v_target=22.0, a_target=+0.5))
    assert a_target == 0.2

  def test_cruise_source_is_unaffected(self):
    # cruise publishes a_ego as its own a_target, so min(a_target, a_ego) is a no-op for it. Pins that the clamp did
    # not cap normal driving
    source, _, a_target = _targets(v_ego=20.0, a_ego=0.8, v_cruise=25.0)
    assert source == LongitudinalPlanSource.cruise
    assert a_target == 0.8

  def test_every_cap_source_is_clamped(self):
    # SCC-M and Speed Limit Assist publish a_ego today, so the clamp is a no-op for them; it must still hold if either
    # ever publishes an acceleration of its own
    for kwargs, expected_source in (({'vision': FakeSource(v_target=20.0, a_target=+1.0)}, LongitudinalPlanSource.sccVision),
                                    ({'scc_map': FakeSource(v_target=20.0, a_target=+1.0)}, LongitudinalPlanSource.sccMap),
                                    ({'sla': FakeSource(v_target=20.0, a_target=+1.0)}, LongitudinalPlanSource.speedLimitAssist),
                                    ({'curve': FakeSource(v_target=20.0, a_target=+1.0)}, LongitudinalPlanSource.curveSpeed),
                                    ({'governor': FakeGovernor(v_target=20.0, a_target=+1.0)}, LongitudinalPlanSource.curveSpeed)):
      source, v_target, a_target = _targets(v_ego=18.0, a_ego=-1.5, v_cruise=30.0, **kwargs)
      assert source == expected_source
      assert v_target == 20.0
      assert a_target == -1.5

  def test_curve_control_phantom_acceleration_is_dropped(self):
    # the reported bug: curve control wins on speed while asking to accelerate, because a lead has braked the truck
    # below its cap. That positive a_target must not reach the planner's acceleration state
    source, _, a_target = _targets(v_ego=15.0, a_ego=-2.5, v_cruise=30.0,
                                   curve=FakeSource(v_target=20.0, a_target=+1.2))
    assert source == LongitudinalPlanSource.curveSpeed
    assert a_target == -2.5

  def test_curve_control_own_decel_is_preserved(self):
    _, _, a_target = _targets(v_ego=25.0, a_ego=-0.5, v_cruise=30.0,
                              curve=FakeSource(v_target=18.0, a_target=-1.8))
    assert a_target == -1.8

  def test_governor_overrides_curve_and_cannot_accelerate(self):
    # when the governor wants a lower speed than the curve profile it replaces it, and its a_target is floored at 0
    # before the clamp ever sees it
    source, v_target, a_target = _targets(v_ego=25.0, a_ego=-0.2, v_cruise=30.0,
                                          curve=FakeSource(v_target=22.0, a_target=+0.5),
                                          governor=FakeGovernor(v_target=17.0, a_target=+0.5))
    assert source == LongitudinalPlanSource.curveSpeed
    assert v_target == 17.0
    assert a_target == -0.2

  def test_governor_keeps_the_profile_decel(self):
    # the governor's lower cap replaces the profile's speed, but the profile's own decel toward its curve stays
    _, v_target, a_target = _targets(v_ego=25.0, a_ego=-0.2, v_cruise=30.0, curve=FakeSource(v_target=22.0, a_target=-0.9),
                                     governor=FakeGovernor(v_target=17.0))
    assert v_target == 17.0
    assert a_target == -0.9

  def test_governor_alone_takes_the_curve_slot(self):
    # no curve ahead (the profile is unset and hands back a_ego), the measured load still caps the speed
    source, v_target, a_target = _targets(v_ego=22.0, a_ego=0.3, v_cruise=30.0, curve=FakeSource(a_target=0.3),
                                          governor=FakeGovernor(v_target=20.0))
    assert source == LongitudinalPlanSource.curveSpeed
    assert v_target == 20.0
    assert a_target == 0.0

  def test_governor_above_the_curve_speed_leaves_the_profile(self):
    _, v_target, a_target = _targets(v_ego=25.0, a_ego=-0.5, v_cruise=30.0,
                                     curve=FakeSource(v_target=18.0, a_target=-1.5),
                                     governor=FakeGovernor(v_target=21.0))
    assert v_target == 18.0
    assert a_target == -1.5

  def test_the_throttle_fade_leaves_the_acceleration_state_alone(self):
    # the fade caps the cruise accel (get_cruise_accel), not the state handed back here: scaling that would only make
    # v_desired_filter lag the real speed, and in this planner it would not cut the cruise throttle anyway
    for kwargs in ({}, {'curve': FakeSource(v_target=22.0, a_target=+0.6)}):
      _, _, a_target = _targets(v_ego=20.0, a_ego=0.8, v_cruise=30.0, governor=FakeGovernor(throttle_scale=0.0), **kwargs)
      assert a_target == (0.8 if not kwargs else 0.6)

  def test_a_cap_publishing_a_ego_is_unchanged(self):
    # what SCC-M and Speed Limit Assist do: a_target is a_ego, whatever its sign
    for a_ego in (-2.0, 0.0, 0.7):
      _, _, a_target = _targets(v_ego=18.0, a_ego=a_ego, v_cruise=30.0, scc_map=FakeSource(v_target=20.0, a_target=a_ego))
      assert a_target == a_ego


def settle_cruise_accel(v_ego, e2e=False, throttle_scale=1., allow_throttle=True, a_cruise=0., frames=200):
  CP = structs.CarParams(steerRatio=15., wheelbase=3.4)
  for _ in range(frames):
    a_cruise = get_cruise_accel(e2e, 40., v_ego, a_cruise, 0., CP, DT_MDL, -0.3, allow_throttle, 0., throttle_scale)
  return a_cruise


class TestThrottleFadeCruiseAccel:
  def test_scales_the_cap(self):
    for v_ego in (5., 15., 25.):
      stock = settle_cruise_accel(v_ego)
      assert stock > 0.5
      for scale in (0.75, 0.5, 0.):
        assert np.isclose(settle_cruise_accel(v_ego, throttle_scale=scale), scale * stock)

  def test_end_to_end_cap_too(self):
    # the driving model's own throttle is capped by this candidate too (the planner takes the lowest)
    assert np.isclose(settle_cruise_accel(15., e2e=True, throttle_scale=0.5), 0.5 * ACCEL_MAX)

  def test_never_raises_a_coast_limit(self):
    # with the model predicting no gas the cap is the coast accel, -0.3 here: scaling must not lift it toward 0
    for scale in (1., 0.5, 0.):
      assert np.isclose(settle_cruise_accel(15., allow_throttle=False, throttle_scale=scale), -0.3)

  def test_never_asks_for_braking(self):
    assert settle_cruise_accel(15., throttle_scale=0., a_cruise=1.0) == 0.


V_EGO = 20.0


def build_sm(v_ego, lead_d_rel=None, lead_v=0., lead_a=0., model_yaw_rate=0.01, model_speed=None, lat_accel=0., angle=False):
  services = {}
  for service in ("vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation"):
    services[service] = getattr(messaging.new_message(service), service)

  # the measured lateral state as the Rivian has it: controlsState.curvature from the steering angle in both modes,
  # torqueState's lateral accels while steering on torque, angleState (no lateral accel) on angle, no carState.yawRate
  controls_state = messaging.new_message('controlsState')
  controls_state.controlsState.curvature = lat_accel / max(v_ego, 1.) ** 2
  if angle:
    controls_state.controlsState.lateralControlState.init('angleState')
  else:
    torque_state = controls_state.controlsState.lateralControlState.init('torqueState')
    torque_state.actualLateralAccel = torque_state.desiredLateralAccel = lat_accel
  services['controlsState'] = controls_state.controlsState.as_reader()

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
  services['carState'] = car_state.carState.as_reader()

  selfdrive_state = messaging.new_message('selfdriveState')
  selfdrive_state.selfdriveState.enabled = True
  services['selfdriveState'] = selfdrive_state.selfdriveState.as_reader()

  car_control = messaging.new_message('carControl')
  car_control.carControl.enabled = True
  services['carControl'] = car_control.carControl.as_reader()

  radar = messaging.new_message('radarState')
  if lead_d_rel is not None:
    radar.radarState.leadOne = {"present": True, "radar": True, "dRel": lead_d_rel, "vLead": lead_v, "vLeadK": lead_v,
                                "aLeadK": lead_a, "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()

  model = messaging.new_message('modelV2')
  model.modelV2.orientationRate.z = [model_yaw_rate] * 33  # nonzero: a straight path divides by zero in SCC vision
  model.modelV2.velocity.x = [v_ego if model_speed is None else model_speed] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0] * 6
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def build_planner(vision=None, curve=None, governor=None):
  CP = structs.CarParams(openpilotLongitudinalControl=False, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
  CP_SP = custom.CarParamsSP.new_message().as_reader()
  planner = LongitudinalPlanner(CP, CP_SP, init_v=V_EGO)
  planner.scc = FakeSCC(vision, curve=curve, governor=governor)
  return planner


class TestSpeedCapInPlanner:
  def test_the_mpc_is_seeded_with_the_clamped_target(self):
    # SCC-V wins (22 m/s against a 27.8 m/s set speed) asking for +0.5 while the planner's own target is -1.0
    planner = build_planner(vision=FakeSource(v_target=22.0, a_target=+0.5))
    planner.output_a_target = -1.0
    seeds = []
    set_cur_state = planner.mpc.set_cur_state
    planner.mpc.set_cur_state = lambda v, a: (seeds.append(a), set_cur_state(v, a))
    planner.update(build_sm(V_EGO))
    assert planner.source == LongitudinalPlanSource.sccVision
    assert seeds == [-1.0]

  def test_no_speed_drift_while_a_cap_wins_and_a_lead_brakes(self):
    # a lead braking at 2 m/s^2 from 20 m/s, 40 m ahead, with SCC-V holding a 22 m/s cap and asking for +0.5 the
    # whole time. The truck does what the planner asks; the planner's idea of its speed must stay on the real one
    self._no_drift(vision=FakeSource(v_target=22.0, a_target=+0.5), source=LongitudinalPlanSource.sccVision)

  def test_no_speed_drift_while_curve_control_wins_and_a_lead_brakes(self):
    # AdventurePilot's report: curve control holding a 22 m/s curve speed, asking for its full +1.2 to get back up to it
    self._no_drift(curve=FakeSource(v_target=22.0, a_target=+1.2), source=LongitudinalPlanSource.curveSpeed)

  def test_throttle_fade_caps_the_planner_output(self):
    # no lead, accelerating from 15 m/s to a 27.8 m/s set speed: the output is the cruise accel, so the fade must cap
    # it there, at the scale times the stock cap for the speed, and with the scale at 0 the truck gets no throttle
    def run(scale):
      planner = build_planner(governor=FakeGovernor(throttle_scale=scale))
      v_ego, out, caps = 15., [], []
      for _ in range(int(4. / DT_MDL)):
        planner.update(build_sm(v_ego))
        out.append(float(planner.output_a_target))
        caps.append(get_max_accel(v_ego))
        v_ego += out[-1] * DT_MDL
      return np.array(out), np.array(caps)
    stock, caps = run(1.0)
    assert abs(stock[-1] - caps[-1]) < 0.05  # stock reaches the cap
    for scale in (0.5, 0.2, 0.0):
      faded, caps = run(scale)
      assert (faded <= scale * caps + 1e-3).all(), (scale, faded.max())
      assert abs(faded[-1] - scale * caps[-1]) < 0.05, (scale, faded[-1])
      assert faded.min() >= 0.  # the fade never brakes

  def test_throttle_fade_ramps_at_the_cruise_jerk(self):
    # the fade comes on mid-acceleration: the throttle comes off no faster than the cruise jerk, and back the same way
    planner = build_planner(governor=FakeGovernor(throttle_scale=1.0))
    v_ego, out = 15., []
    for i in range(int(6. / DT_MDL)):
      planner.scc.governor._throttle_scale = 0.0 if int(2. / DT_MDL) <= i < int(4. / DT_MDL) else 1.0
      planner.update(build_sm(v_ego))
      out.append(float(planner.output_a_target))
      v_ego += out[-1] * DT_MDL
    j_cruise = float(np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS))
    assert np.abs(np.diff(out)).max() <= 1.2 * j_cruise * DT_MDL + 1e-3
    assert out[int(4. / DT_MDL) - 1] <= 1e-3  # all of it off within 2 s
    assert out[-1] > 0.9

  @staticmethod
  def _no_drift(source, **scc):
    planner = build_planner(**scc)
    v_ego, d_rel, v_lead = V_EGO, 40., V_EGO
    drift = []
    for _ in range(int(6. / DT_MDL)):
      planner.update(build_sm(v_ego, lead_d_rel=d_rel, lead_v=v_lead, lead_a=-2.0 if v_lead > 0. else 0.))
      a = float(planner.output_a_target)
      v_ego = max(v_ego + a * DT_MDL, 0.)
      v_lead = max(v_lead - 2.0 * DT_MDL, 0.)
      d_rel += (v_lead - v_ego) * DT_MDL
      drift.append(planner.v_desired_filter.x - v_ego)
    assert planner.source == source
    assert max(np.abs(drift)) < 0.5, max(drift)


class TestCurveSpeedInPlanner:
  # the real curve controller and governor behind the real planner
  def setup_method(self):
    Params().put_bool("CurveSpeedControl", True, block=True)

  def teardown_method(self):
    Params().put_bool("CurveSpeedControl", False, block=True)

  @staticmethod
  def run(frames, v_ego=24., **sm_kwargs):
    CP = structs.CarParams(openpilotLongitudinalControl=False, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
    planner = LongitudinalPlanner(CP, custom.CarParamsSP.new_message().as_reader(), init_v=v_ego)
    pm = MockPubMaster()
    for _ in range(frames):
      sm = build_sm(v_ego, **sm_kwargs)
      planner.update(sm)
      planner.publish(sm, pm)
    return planner, pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP

  def test_a_curve_ahead_caps_the_speed_and_is_published(self):
    # a sustained 0.07 rad/s at 15 m/s through the model horizon: kappa 0.0047 1/m, so the curve speed is
    # sqrt(2.2 / 0.0047) = 21.7 m/s against the 27.8 m/s set speed, and the truck at 24 m/s starts slowing
    planner, plan = self.run(20, model_yaw_rate=0.07, model_speed=15.)
    curve = plan.smartCruiseControl.curve
    assert plan.longitudinalPlanSource == LongitudinalPlanSource.curveSpeed
    assert curve.enabled and curve.active and not curve.governorActive
    assert curve.state == CurveState.slowing
    assert np.isclose(curve.vTarget, (A_LAT_MAX / (0.07 / 15.)) ** 0.5, rtol=0.02)
    assert np.isclose(curve.latAccelTarget, A_LAT_MAX)
    assert plan.vTarget == curve.vTarget
    assert curve.aTarget < 0. and planner.output_a_target < 0.

  def test_the_measured_load_is_published_and_trims_the_speed(self):
    # 3.0 m/s^2 measured on torque steering, past the 2.8 ceiling: the governor caps the speed on a road the model
    # sees as straight
    planner, plan = self.run(20, lat_accel=3.0)
    curve = plan.smartCruiseControl.curve
    assert plan.longitudinalPlanSource == LongitudinalPlanSource.curveSpeed
    assert curve.governorActive and not curve.active
    assert np.isclose(curve.latAccelMeasured, 3.0, atol=0.01)
    assert plan.vTarget < 24.
    assert planner.scc.governor.throttle_scale() == 0.

  def test_the_measured_load_trims_the_speed_on_angle_steering(self):
    # the Extreme kit's default: angleState, the 3.0 m/s^2 ceiling, the load from controlsState.curvature alone
    planner, plan = self.run(20, lat_accel=3.3, angle=True)
    curve = plan.smartCruiseControl.curve
    assert plan.longitudinalPlanSource == LongitudinalPlanSource.curveSpeed
    assert curve.governorActive
    assert np.isclose(curve.latAccelMeasured, 3.3, atol=0.01)
    assert plan.vTarget < 24.
    assert planner.scc.governor.throttle_scale() == 0.

  def test_the_truck_reaches_the_curve_near_the_curve_speed(self):
    # closed loop, no lead: 27 m/s toward a curve 200 m ahead whose speed is 18 m/s, the model seeing it from 10 s out
    # and the truck doing what the planner asks. The cap is tracked by the cruise candidate (at most -1.2 m/s^2, and
    # only in proportion to the speed error), so the backward pass must plan at a decel it can follow: planned at 1.8
    # the truck got there at 23.7 m/s, 3.8 m/s^2 in the curve
    CP = structs.CarParams(openpilotLongitudinalControl=False, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
    planner = LongitudinalPlanner(CP, custom.CarParamsSP.new_message().as_reader(), init_v=27.)
    t_idxs = np.array(ModelConstants.T_IDXS)
    v_curve, v_ego, d_curve, a_min, sources = 18., 27., 200., 0., set()
    while d_curve > 0.:
      sm = build_sm(v_ego)
      model = messaging.new_message('modelV2')
      model.modelV2.velocity.x = [v_ego] * 33
      model.modelV2.orientationRate.z = [float(y) for y in np.where(v_ego * t_idxs >= d_curve, A_LAT_MAX / v_curve ** 2 * v_ego, 0.)]
      model.modelV2.position.x = [float(i) for i in range(33)]
      model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0] * 6
      sm['modelV2'] = model.modelV2.as_reader()
      planner.update(sm)
      a_min = min(a_min, float(planner.output_a_target))
      sources.add(planner.source)
      v_ego += float(planner.output_a_target) * DT_MDL
      d_curve -= v_ego * DT_MDL
    assert LongitudinalPlanSource.curveSpeed in sources
    assert v_ego < v_curve + 1.5, v_ego
    assert a_min >= -1.2 - 1e-6  # the cruise floor, no harder

  def test_off_by_default(self):
    Params().put_bool("CurveSpeedControl", False, block=True)
    _, plan = self.run(20, model_yaw_rate=0.07, model_speed=15., lat_accel=3.0)
    curve = plan.smartCruiseControl.curve
    assert plan.longitudinalPlanSource == LongitudinalPlanSource.cruise
    assert not (curve.enabled or curve.active or curve.governorActive)
    assert curve.state == CurveState.disabled
