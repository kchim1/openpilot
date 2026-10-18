"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from openpilot.cereal import messaging
from opendbc.car import structs
from openpilot.common.params import Params, ParamKeyFlag, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.red_light_stop import RedLightStop, clip_offset, A_FLOOR, EXTRA_DECEL_MAX, \
  EXTRA_RISE_RATE, EXTRA_FALL_RATE, HOLD_V, HOLD_MARGIN, GO_DIST, GO_FRAMES, CREEP_A, CREEP_FRAMES, LEAD_MARGIN, LOW_SPEED_V, \
  MIN_TARGET_DIST, OFFSET_MAX, PARAM_READ_FRAMES, STOP_ARM_V, STOP_RELEASE_V
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

RISE_STEP = EXTRA_RISE_RATE * DT_MDL
FALL_STEP = EXTRA_FALL_RATE * DT_MDL
NO_LEAD = SimpleNamespace(present=False, dRel=0.)


class FakeParams:
  def __init__(self, value=2):
    self.value = value
    self.reads = 0

  def get(self, key, return_default=False):
    assert key == "RedLightStopDistance"
    self.reads += 1
    return self.value


def model(stop_dist=20., end_v=0.):
  return SimpleNamespace(position=SimpleNamespace(x=[0.] * 32 + [stop_dist]), velocity=SimpleNamespace(x=[5.] + [1.] * 31 + [end_v]))


def target_extra(v_ego, stop_dist, offset):
  return min(max(v_ego ** 2 / 2 * (1 / max(stop_dist - offset, MIN_TARGET_DIST) - 1 / max(stop_dist, MIN_TARGET_DIST)), 0.), EXTRA_DECEL_MAX)


class Drive:
  """Feeds RedLightStop the way the planner does, with the model's stop point fixed on the road while the truck moves."""
  def __init__(self, offset=2, openpilot_long=True):
    self.rls = RedLightStop(structs.CarParams(openpilotLongitudinalControl=openpilot_long), FakeParams(offset))
    self.rls.update_offset()
    self.pos = 0.

  def step(self, v_ego, a_e2e, stop_point=20., end_v=0., lead=NO_LEAD, is_e2e=True, long_off=False, stop_dist=None):
    if stop_dist is None:
      stop_dist = stop_point - self.pos
    out = self.rls.update(is_e2e, v_ego, a_e2e, model(stop_dist, end_v), lead, long_off=long_off)
    self.pos += v_ego * DT_MDL
    return out

  def steps(self, n, **kw):
    return [self.step(**kw) for _ in range(n)]

  def hold(self, offset_margin=0.5):
    # arrive nearly stopped within reach of the shortened stop point
    self.step(3., -1., stop_dist=10.)
    assert self.rls.armed
    for _ in range(40):  # the stop point is smoothed over ~0.3 s
      self.step(0., -0.1, stop_dist=self.rls.offset + offset_margin)
    assert self.rls.holding


class TestRedLightStopGates(OpenpilotTestCase):
  def test_off_or_not_e2e_adds_nothing(self):
    for offset, is_e2e in ((0, True), (2, False)):
      dr = Drive(offset)
      for _ in range(50):
        assert dr.step(8., -1.5, is_e2e=is_e2e) is None
      assert not dr.rls.armed and not dr.rls.holding and dr.rls.extra == 0.

  def test_not_read_without_openpilot_long(self):
    dr = Drive(5, openpilot_long=False)
    assert dr.rls._params.reads == 0 and dr.rls.offset == 0.
    assert dr.step(8., -1.5) is None

  def test_setting_reread_every_second(self):
    dr = Drive(2)
    dr.rls._params.value = 4
    for _ in range(PARAM_READ_FRAMES - 1):
      dr.rls.update_offset()
      assert dr.rls.offset == 2.
    dr.rls.update_offset()
    assert dr.rls.offset == 4.

  def test_offset_clipped(self):
    for value, expected in ((-3, 0.), (0, 0.), (1, 1.), (OFFSET_MAX, OFFSET_MAX), (9, OFFSET_MAX), (None, 0.), ("x", 0.), (float("nan"), 0.)):
      assert clip_offset(value) == expected

  def test_read_error_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.red_light_stop.cloudlog")
    dr = Drive(2)
    dr.rls._params = mocker.MagicMock()
    dr.rls._params.get.side_effect = UnknownKeyName(b"RedLightStopDistance")
    for _ in range(3 * PARAM_READ_FRAMES):
      dr.rls.update_offset()
    assert dr.rls.offset == 0.
    assert cloudlog.exception.call_count == 1

  def test_arms_only_for_a_stop_it_plans_to_hold(self):
    dr = Drive(2)
    # the model plans to keep moving (slowly), or plans a stop but is not braking yet
    for end_v, a_e2e in ((STOP_ARM_V + 0.2, -1.), (0., 0.1)):
      assert dr.step(10., a_e2e, stop_point=60., end_v=end_v) is None
      assert not dr.rls.armed
    dr.step(10., -0.5, stop_point=60., end_v=STOP_ARM_V - 0.1)
    assert dr.rls.armed
    # between the thresholds it stays armed, above the release speed it lets go
    dr.step(10., 0.2, stop_point=60., end_v=(STOP_ARM_V + STOP_RELEASE_V) / 2)
    assert dr.rls.armed
    dr.step(10., 0.2, stop_point=60., end_v=STOP_RELEASE_V + 0.1)
    assert not dr.rls.armed

  def test_not_for_a_car_stopped_at_the_stop_point(self):
    for d_rel, armed in ((15., False), (20. + LEAD_MARGIN - 0.1, False), (20. + LEAD_MARGIN + 0.1, True)):
      dr = Drive(2)
      dr.step(8., -1., stop_dist=20., lead=SimpleNamespace(present=True, dRel=d_rel))
      assert dr.rls.armed == armed

  def test_lead_appearing_disarms(self):
    dr = Drive(2)
    dr.steps(20, v_ego=8., a_e2e=-1.)
    assert dr.rls.armed and dr.rls.extra > 0.
    dr.step(8., -1., lead=SimpleNamespace(present=True, dRel=10.))
    assert not dr.rls.armed

  def test_bad_numbers(self):
    dr = Drive(2)
    dr.steps(20, v_ego=8., a_e2e=-1.)
    extra = dr.rls.extra
    assert extra > 0.
    # a non-finite stop point: disarm and fade, never a NaN request
    out = dr.step(8., -1., stop_dist=float("nan"))
    assert not dr.rls.armed and np.isfinite(dr.rls.extra) and out is not None and np.isfinite(out[0])
    self.assertAlmostEqual(dr.rls.extra, extra - FALL_STEP)
    # a non-finite model request, or empty lists
    assert dr.step(8., float("nan")) is None
    empty = SimpleNamespace(position=SimpleNamespace(x=[]), velocity=SimpleNamespace(x=[]))
    assert dr.rls.update(True, 8., -1., empty, NO_LEAD) is None


class TestRedLightStopBraking(OpenpilotTestCase):
  def test_extra_builds_at_the_rise_rate_toward_the_target(self):
    dr = Drive(2)
    for i in range(1, 30):
      accel, should_stop = dr.step(8., -1., stop_point=40.)
      d = 40. - (dr.pos - 8. * DT_MDL)
      target = target_extra(8., d, 2)
      assert 0. < target < EXTRA_DECEL_MAX
      self.assertAlmostEqual(dr.rls.extra, min(i * RISE_STEP, target), places=4)
      self.assertAlmostEqual(accel, -1. - dr.rls.extra)
      assert not should_stop

  def test_small_far_out_large_near_the_line(self):
    assert target_extra(18., 90., 2) < 0.05
    assert target_extra(8., 15., 2) > 0.3
    assert target_extra(LOW_SPEED_V + 0.1, 3., 2) == EXTRA_DECEL_MAX

  def test_low_speed_overrules_a_model_creeping_to_its_own_point(self):
    dr = Drive(2)
    dr.step(5., -1., stop_dist=10.)
    # at 2 m/s, 5 m from the model's point (3 m from the short one), the model eases off to creep
    for _ in range(40):
      accel, _ = dr.step(2., 0.2, stop_dist=5.)
    # a stop point that moves along with the truck reads a little closer once smoothed
    assert 4. < dr.rls.stop_dist < 5.
    # braking for the short point, within the cap on what is added
    self.assertAlmostEqual(accel, max(-2. ** 2 / (2 * (dr.rls.stop_dist - 2.)), 0.2 - EXTRA_DECEL_MAX), places=4)
    assert accel < -0.6

  def test_low_speed_lets_it_creep_up_from_far_short(self):
    dr = Drive(2)
    dr.step(5., -1., stop_dist=10.)
    # stopped well short of the short point: a slow creep toward it is not braked
    for _ in range(40):
      out = dr.step(0.2, 0.3, stop_dist=10.)
    assert out is None or out[0] > 0.29

  def test_only_adds_braking_within_the_cap_and_floor(self):
    rng = np.random.default_rng(0)
    dr = Drive(5)
    for _ in range(5000):
      a_e2e = rng.uniform(-3.5, 1.0)
      out = dr.step(rng.uniform(0., 25.), a_e2e, stop_dist=rng.uniform(-5., 120.), end_v=rng.uniform(0., 2.))
      if out is None:
        continue
      accel, holding = out
      assert accel <= a_e2e + 1e-9
      assert accel >= min(a_e2e, A_FLOOR) - 1e-9
      assert accel >= a_e2e - EXTRA_DECEL_MAX - 1e-9
      if holding:
        assert accel <= 0.

  def test_model_braking_past_the_floor_is_left_alone(self):
    dr = Drive(5)
    for _ in range(50):
      accel, _ = dr.step(6., -2.5, stop_dist=4.)
      assert accel == -2.5

  def test_extra_fades_after_the_model_plans_to_go(self):
    dr = Drive(3)
    dr.steps(100, v_ego=5., a_e2e=-1., stop_dist=6.)
    assert dr.rls.extra == EXTRA_DECEL_MAX
    extra = dr.rls.extra
    while dr.rls.extra > 0.:
      out = dr.step(5., 0.5, stop_dist=60., end_v=10.)
      self.assertAlmostEqual(dr.rls.extra, max(extra - FALL_STEP, 0.))
      extra = dr.rls.extra
      assert out is None or out[0] == 0.5 - dr.rls.extra
    assert dr.step(5., 0.5, stop_dist=60., end_v=10.) is None

  def test_setting_off_or_leaving_e2e_fades_instead_of_dropping(self):
    for change in ("off", "not_e2e"):
      dr = Drive(3)
      dr.steps(100, v_ego=5., a_e2e=-1., stop_dist=6.)
      extra = dr.rls.extra
      if change == "off":
        dr.rls.offset = 0.
        out = dr.step(5., -1., stop_dist=6.)
        self.assertAlmostEqual(dr.rls.extra, extra - FALL_STEP)
        self.assertAlmostEqual(out[0], -1. - dr.rls.extra)
      else:
        # the e2e candidate is gone too: nothing to add to
        assert dr.step(5., -1., stop_dist=6., is_e2e=False) is None
        self.assertAlmostEqual(dr.rls.extra, extra - FALL_STEP)

  def test_driver_override_resets(self):
    dr = Drive(3)
    dr.steps(100, v_ego=5., a_e2e=-1., stop_dist=6.)
    assert dr.step(5., -1., stop_dist=6., long_off=True) is None
    assert not dr.rls.armed and not dr.rls.holding and dr.rls.extra == 0.


class TestRedLightStopHold(OpenpilotTestCase):
  def test_holds_short_of_the_model_stop_point(self):
    dr = Drive(2)
    dr.step(3., -1., stop_dist=10.)
    # still rolling, or the model's stop point is further than offset + margin: no hold
    for v_ego, stop_dist in ((HOLD_V + 0.1, 2.5), (0.2, 2 + HOLD_MARGIN + 0.5)):
      dr.step(v_ego, -0.3, stop_dist=stop_dist)
      assert not dr.rls.holding
    for _ in range(40):
      accel, should_stop = dr.step(0.2, 0.1, stop_dist=2 + HOLD_MARGIN - 0.3)
    assert should_stop and accel <= 0.
    # held at a standstill while the model's plan drifts, as long as it does not plan to go
    for stop_dist, end_v in ((2 + GO_DIST - 1., 1.), (5., 2.5), (3., 0.)):
      for _ in range(40):
        accel, should_stop = dr.step(0., 0.1, stop_dist=stop_dist, end_v=end_v)
        assert should_stop and accel <= 0.

  def test_green_light_releases_after_two_frames(self):
    dr = Drive(2)
    dr.hold()
    # one odd frame does not release it
    accel, should_stop = dr.step(0., 0.8, stop_dist=2 + GO_DIST + 40., end_v=8.)
    assert should_stop
    dr.step(0., -0.1, stop_dist=2.5)
    assert dr.rls.holding
    for _ in range(GO_FRAMES):
      out = dr.step(0., 0.8, stop_dist=2 + GO_DIST + 40., end_v=8.)
    # released with nothing added (no extra braking is left at a standstill)
    assert out is None and not dr.rls.holding and not dr.rls.armed

  def test_creeping_model_is_let_go(self):
    dr = Drive(2)
    dr.hold()
    for _ in range(CREEP_FRAMES - 1):
      _, should_stop = dr.step(0., CREEP_A + 0.1, stop_dist=4.)
      assert should_stop
    assert dr.step(0., CREEP_A + 0.1, stop_dist=4.) is None
    assert not dr.rls.holding

  def test_moving_or_override_drops_the_hold(self):
    # the reviewed failure: held, the driver uses the gas, and on release the hold must not cap the braking
    dr = Drive(3)
    dr.hold()
    assert dr.step(0.5, 1.0, stop_dist=3., long_off=True) is None
    assert not dr.rls.holding
    dr = Drive(3)
    dr.hold()
    out = dr.step(HOLD_V + 0.1, -0.5, stop_dist=2.)
    assert not dr.rls.holding and (out is None or not out[1])


class TestRedLightStopParam(OpenpilotTestCase):
  def test_defaults_to_one_meter_and_survives_drives_and_reboots(self):
    params = Params()
    params.remove("RedLightStopDistance")
    assert params.get("RedLightStopDistance", return_default=True) == 1
    params.put("RedLightStopDistance", 3, block=True)
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
    assert params.get("RedLightStopDistance") == 3
    assert b"RedLightStopDistance" in params.all_keys(ParamKeyFlag.PERSISTENT)
    assert b"RedLightStopDistance" in params.all_keys(ParamKeyFlag.BACKUP)


def build_sm(v_ego, a_e2e, stop_dist, end_v=0., experimental=True, lead_d=None, long_state='pid'):
  services = {}
  for service in ("vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation", "carControl"):
    services[service] = getattr(messaging.new_message(service), service)
  cs = messaging.new_message('controlsState')
  cs.controlsState.longControlState = long_state  # engaged unless 'off', when the planner resets its state
  services['controlsState'] = cs.controlsState.as_reader()
  ss = messaging.new_message('selfdriveState')
  ss.selfdriveState.experimentalMode = experimental
  services['selfdriveState'] = ss.selfdriveState.as_reader()
  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
  services['carState'] = car_state.carState.as_reader()
  radar = messaging.new_message('radarState')
  if lead_d is not None:
    radar.radarState.leadOne = {"present": True, "dRel": lead_d, "vLead": 0., "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()
  m = messaging.new_message('modelV2')
  m.modelV2.orientationRate.z = [0.01] * 33  # nonzero: a straight path divides by zero in SCC vision
  m.modelV2.velocity.x = [v_ego] * 32 + [end_v]
  m.modelV2.position.x = [stop_dist * i / 32 for i in range(33)]
  m.modelV2.action.desiredAcceleration = a_e2e
  services['modelV2'] = m.modelV2.as_reader()
  return MockSubMaster(services)


class TestRedLightStopPlanner(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()
    self.params.put_bool("DynamicExperimentalControl", False, block=True)

  def run_planner(self, offset, frames=40, **kw):
    self.params.put("RedLightStopDistance", offset, block=True)
    CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=kw.get('v_ego', 8.))
    sm = build_sm(**{'v_ego': 8., 'a_e2e': -1.0, 'stop_dist': 15., **kw})
    for _ in range(frames):
      planner.update(sm)
    return planner

  def test_adds_braking_to_the_e2e_plan(self):
    off = self.run_planner(0)
    self.assertAlmostEqual(off.output_a_target, -1.0, places=5)
    on = self.run_planner(3)
    rls = on.red_light_stop
    assert rls.armed and 0. < rls.extra <= EXTRA_DECEL_MAX
    self.assertAlmostEqual(on.output_a_target, -1.0 - rls.extra, places=5)
    assert on.mpc.source == LongitudinalPlanSource.e2e

  def test_reset_while_long_control_is_off(self):
    planner = self.run_planner(3, long_state='off')
    assert not planner.red_light_stop.armed and planner.red_light_stop.extra == 0.

  def test_nothing_added_outside_e2e_or_for_a_lead(self):
    for kw in ({"experimental": False}, {"lead_d": 12.}):
      planner = self.run_planner(5, **kw)
      assert planner.red_light_stop.extra == 0.
      assert not planner.red_light_stop.armed

  def test_holds_at_standstill(self):
    planner = self.run_planner(3, frames=5, v_ego=0.2, a_e2e=-0.2, stop_dist=3.5)
    assert planner.output_should_stop
    assert planner.output_a_target <= 0.
