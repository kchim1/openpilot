"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import defaultdict
from types import SimpleNamespace

from openpilot.cereal import log, messaging
from opendbc.car import structs
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, get_T_FOLLOW, get_jerk_factor, \
  get_traffic_mode_T_FOLLOW, DANGER_ZONE_COST, LIMIT_COST
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, TRAFFIC_MODE_READ_FRAMES

Personality = log.LongitudinalPersonality
PERSONALITIES = (Personality.aggressive, Personality.standard, Personality.relaxed)
V_EGO = 2.5


class MockSubMaster(dict):
  def __init__(self, services: dict):
    super().__init__(services)
    self.valid = dict.fromkeys(services, True)
    self.logMonoTime = dict.fromkeys(services, 0)
    self.updated = dict.fromkeys(services, True)
    self.recv_frame = dict.fromkeys(services, 1)

  def all_checks(self, service_list=None) -> bool:
    return True


class MockPubMaster:
  def __init__(self):
    self.sent = defaultdict(list)

  def send(self, name, msg):
    self.sent[name].append(msg)


def build_sm(v_ego):
  services = {}
  for service in ("controlsState", "vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation",
                  "selfdriveState", "carControl"):
    services[service] = getattr(messaging.new_message(service), service)

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
  services['carState'] = car_state.carState.as_reader()

  radar = messaging.new_message('radarState')
  radar.radarState.leadOne = {"present": True, "dRel": 10., "vLead": v_ego, "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()

  model = messaging.new_message('modelV2')
  model.modelV2.orientationRate.z = [0.01] * 33  # nonzero: a straight path divides by zero in SCC vision
  model.modelV2.velocity.x = [v_ego] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def radar_state():
  lead = SimpleNamespace(present=True, dRel=20., vLead=10., aLeadK=0., aLeadTau=1.5, modelProb=1.)
  return SimpleNamespace(leadOne=lead, leadTwo=lead)


def run_mpc(mpc, v_ego, personality):
  # same call order as LongitudinalPlanner.update
  mpc.set_weights(True, personality=personality)
  mpc.set_cur_state(v_ego, 0.)
  mpc.update(radar_state(), personality=personality)


def record_weights(mpc):
  calls = []
  mpc.set_cost_weights = lambda cost_weights, constraint_cost_weights: calls.append((list(cost_weights), list(constraint_cost_weights)))
  return calls


class TestTrafficModeMpc(OpenpilotTestCase):
  def test_follow_time_curve(self):
    for v_ego, t_follow in ((0., 0.5), (1., 0.6), (2.5, 0.75), (5., 1.0), (6., 1.0), (15., 1.0), (40., 1.0)):
      self.assertAlmostEqual(get_traffic_mode_T_FOLLOW(v_ego), t_follow)

  def test_mpc_uses_curve_and_ignores_offset(self):
    mpc = LongitudinalMpc()
    mpc.traffic_mode = True
    for offset in (-10, 0, 10):
      mpc.t_follow_offset_pct = offset
      for personality in PERSONALITIES:
        for v_ego in (0., 2.5, 5., 20.):
          run_mpc(mpc, v_ego, personality)
          expected = get_traffic_mode_T_FOLLOW(v_ego)
          assert (mpc.params[:, 4] == expected).all(), (offset, personality, v_ego)
          assert mpc.t_follow == expected
          assert mpc.get_t_follow_offset_pct() == 0.
          assert mpc.jerk_factor == 0.5
    # the offset setting itself is left alone
    assert mpc.t_follow_offset_pct == 10

  def test_off_restores_personality_and_offset(self):
    mpc = LongitudinalMpc()
    mpc.t_follow_offset_pct = -10
    for personality in PERSONALITIES:
      mpc.traffic_mode = True
      run_mpc(mpc, 2., personality)
      mpc.traffic_mode = False
      run_mpc(mpc, 2., personality)
      expected = get_T_FOLLOW(personality) * (1 + -10 / 100)
      assert (mpc.params[:, 4] == expected).all()
      assert mpc.t_follow == expected
      assert mpc.get_t_follow_offset_pct() == -10.
      assert mpc.jerk_factor == get_jerk_factor(personality)

  def test_only_jerk_factor_changes_in_weights(self):
    mpc = LongitudinalMpc()
    calls = record_weights(mpc)
    mpc.set_weights(True, personality=Personality.aggressive)
    for personality in PERSONALITIES:
      mpc.traffic_mode = True
      mpc.set_weights(True, personality=personality)
    # every personality gets Aggressive's weights, and the danger zone cost is unchanged
    assert all(c == calls[0] for c in calls)
    assert calls[0][1] == [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]


class TestTrafficModePlanner(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  @staticmethod
  def make_planner(openpilot_long=True):
    CP = structs.CarParams(openpilotLongitudinalControl=openpilot_long)
    mpc = SimpleNamespace(traffic_mode=False)
    return LongitudinalPlannerSP(CP, structs.CarParamsSP(), mpc), mpc

  def test_poll_every_read_frames(self):
    self.params.put_bool("TrafficMode", True, block=True)
    planner, mpc = self.make_planner()
    planner.update_traffic_mode()
    assert mpc.traffic_mode

    # switched off mid-drive: applied within TRAFFIC_MODE_READ_FRAMES planner frames (0.25 s at 20 Hz)
    self.params.put_bool("TrafficMode", False, block=True)
    for _ in range(TRAFFIC_MODE_READ_FRAMES - 1):
      planner.update_traffic_mode()
      assert mpc.traffic_mode
    planner.update_traffic_mode()
    assert not mpc.traffic_mode

  def test_not_read_without_openpilot_long(self):
    self.params.put_bool("TrafficMode", True, block=True)
    planner, mpc = self.make_planner(openpilot_long=False)
    for _ in range(3 * TRAFFIC_MODE_READ_FRAMES):
      planner.update_traffic_mode()
    assert not mpc.traffic_mode

  def test_read_error_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner.cloudlog")
    planner, mpc = self.make_planner()
    mpc.traffic_mode = True
    planner._params = mocker.MagicMock()
    planner._params.get_bool.side_effect = UnknownKeyName(b"TrafficMode")
    for _ in range(3 * TRAFFIC_MODE_READ_FRAMES):
      planner.update_traffic_mode()
    assert not mpc.traffic_mode
    assert cloudlog.exception.call_count == 1

  def test_offset_param_never_written(self):
    self.params.put("SPFollowingTimeOffset", -6, block=True)
    planner, mpc = self.make_planner()
    for on in (True, False, True, False):
      self.params.put_bool("TrafficMode", on, block=True)
      for _ in range(TRAFFIC_MODE_READ_FRAMES):
        planner.update_traffic_mode()
      assert mpc.traffic_mode == on
      assert mpc.t_follow_offset_pct == -6.
    assert self.params.get("SPFollowingTimeOffset") == -6

  def test_planner_passes_follow_time_and_logs_it(self):
    CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=2.7, longitudinalActuatorDelay=0.2)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=V_EGO)
    planner.mpc.t_follow_offset_pct = 10
    sm, pm = build_sm(V_EGO), MockPubMaster()

    for on in (True, False, True):
      self.params.put_bool("TrafficMode", on, block=True)
      for _ in range(TRAFFIC_MODE_READ_FRAMES):
        planner.update(sm)
        planner.publish(sm, pm)

      # personality is Aggressive (selfdriveState default)
      t_follow = planner.mpc.params[0, 4]
      if on:
        assert t_follow == get_traffic_mode_T_FOLLOW(planner.mpc.x0[1])
        assert 0.5 < t_follow < 1.0
      else:
        assert t_follow == get_T_FOLLOW(Personality.aggressive) * 1.1

      fd = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance
      assert fd.trafficMode == on
      self.assertAlmostEqual(fd.tFollow, t_follow, places=6)
      assert fd.offsetPct == (0. if on else 10.)
      assert fd.jerkFactor == 0.5
