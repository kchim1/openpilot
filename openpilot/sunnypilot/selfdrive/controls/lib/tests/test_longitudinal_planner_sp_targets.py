"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

The speed-cap invariant in LongitudinalPlannerSP.update_targets().

A source wins the arbitration on the lowest SPEED target, but the a_target that comes back becomes the planner's
acceleration state (output_a_target): it seeds the MPC and is integrated into v_desired_filter. Every source in the
contest except `cruise` is a speed CAP, so none of them may ever ask for MORE acceleration than that state already is.
Without that clamp, SCC-V leaving a turn (+0.5 m/s^2) while a lead brakes the truck below its cap hands the planner an
acceleration the truck is not making, v_desired_filter drifts above the real speed, and the MPC brakes for a truck
going faster than it is.

update_targets() is arbitration plus arithmetic, so the sub-controllers are faked rather than driven. That keeps each
case pinned to one published (v_target, a_target) pair, which is what the invariant is about; the controllers have
their own tests. The last cases drive the real LongitudinalPlanner to check where the clamped value goes.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import custom, messaging
from opendbc.car import structs
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource

UNSET = 1e3  # far above any real v_target, so an unset source never wins the min-by-speed contest


class FakeSource:
  def __init__(self, v_target=UNSET, a_target=0.0):
    self.output_v_target = v_target
    self.output_a_target = a_target

  def update(self, *args):
    pass


class FakeSCC:
  def __init__(self, vision=None, scc_map=None):
    self.vision = vision or FakeSource()
    self.map = scc_map or FakeSource()

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


def _targets(v_ego, a_ego, v_cruise, vision=None, scc_map=None, sla=None):
  """Run update_targets() with the sub-controllers faked. Returns (source, v_target, a_target)."""
  planner = object.__new__(LongitudinalPlannerSP)
  planner.events_sp = SimpleNamespace()  # handed to the faked Speed Limit Assist, unused
  planner.resolver = FakeResolver()
  planner.sla = sla or FakeSource()
  planner.scc = FakeSCC(vision, scc_map)
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
                                    ({'sla': FakeSource(v_target=20.0, a_target=+1.0)}, LongitudinalPlanSource.speedLimitAssist)):
      source, v_target, a_target = _targets(v_ego=18.0, a_ego=-1.5, v_cruise=30.0, **kwargs)
      assert source == expected_source
      assert v_target == 20.0
      assert a_target == -1.5

  def test_a_cap_publishing_a_ego_is_unchanged(self):
    # what SCC-M and Speed Limit Assist do: a_target is a_ego, whatever its sign
    for a_ego in (-2.0, 0.0, 0.7):
      _, _, a_target = _targets(v_ego=18.0, a_ego=a_ego, v_cruise=30.0, scc_map=FakeSource(v_target=20.0, a_target=a_ego))
      assert a_target == a_ego


V_EGO = 20.0


def build_sm(v_ego, lead_d_rel=None, lead_v=0., lead_a=0.):
  services = {}
  for service in ("controlsState", "vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation"):
    services[service] = getattr(messaging.new_message(service), service)

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
  model.modelV2.orientationRate.z = [0.01] * 33  # nonzero: a straight path divides by zero in SCC vision
  model.modelV2.velocity.x = [v_ego] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  model.modelV2.meta.disengagePredictions.gasPressProbs = [1.0] * 6
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def build_planner(vision=None):
  CP = structs.CarParams(openpilotLongitudinalControl=False, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
  CP_SP = custom.CarParamsSP.new_message().as_reader()
  planner = LongitudinalPlanner(CP, CP_SP, init_v=V_EGO)
  planner.scc = FakeSCC(vision)
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
    planner = build_planner(vision=FakeSource(v_target=22.0, a_target=+0.5))
    v_ego, d_rel, v_lead = V_EGO, 40., V_EGO
    drift = []
    for _ in range(int(6. / DT_MDL)):
      planner.update(build_sm(v_ego, lead_d_rel=d_rel, lead_v=v_lead, lead_a=-2.0 if v_lead > 0. else 0.))
      a = float(planner.output_a_target)
      v_ego = max(v_ego + a * DT_MDL, 0.)
      v_lead = max(v_lead - 2.0 * DT_MDL, 0.)
      d_rel += (v_lead - v_ego) * DT_MDL
      drift.append(planner.v_desired_filter.x - v_ego)
    assert planner.source == LongitudinalPlanSource.sccVision
    assert max(np.abs(drift)) < 0.5, max(drift)
