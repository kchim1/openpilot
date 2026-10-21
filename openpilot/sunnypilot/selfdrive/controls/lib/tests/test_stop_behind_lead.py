"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

from openpilot.cereal import log, messaging
from opendbc.car import structs
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import STOP_DISTANCE
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.launch_assist import LEAD_DEPART_V_K
from openpilot.sunnypilot.selfdrive.controls.lib.stop_behind_lead import StopBehindLead, V_ENGAGE, V_RELEASE, LEAD_STOPPED_V, A_STOP, \
  A_MAX, GAP_MIN, GAP_MAX, GAP_MAX_TRAFFIC, RISE_RATE, STOP_V, DROPOUT_FRAMES
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)
RISE_STEP = RISE_RATE * DT_MDL


def lead(d, v_lead_k=0., radar=True, present=True, track=1):
  return SimpleNamespace(present=present, radar=radar, dRel=d, vLeadK=v_lead_k, radarTrackId=track)


def settle(sbl, v_ego, ld, frames=100, **kw):
  out = None
  for _ in range(frames):
    out = sbl.update(v_ego, ld, **kw)
  return out


class TestStopBehindLead:
  def setup_method(self):
    self.sbl = StopBehindLead(CP)

  def test_engages_only_at_walking_pace_behind_a_stopped_radar_lead(self):
    assert self.sbl.update(V_ENGAGE + 0.1, lead(8.)) is None          # still faster than walking pace: the lead planner's
    assert self.sbl.update(1.0, lead(8., v_lead_k=0.5)) is None       # the lead is still moving
    assert self.sbl.update(1.0, lead(8., radar=False)) is None        # vision only
    assert self.sbl.update(1.0, lead(8., present=False)) is None
    assert self.sbl.update(1.0, lead(30.)) is None                    # too far back: the lead planner closes up first
    assert self.sbl.update(1.0, lead(8.), long_off=True) is None      # not commanding
    assert StopBehindLead(structs.CarParams(openpilotLongitudinalControl=False)).update(1.0, lead(8.)) is None
    assert not self.sbl.active
    assert self.sbl.update(1.0, lead(8., v_lead_k=LEAD_STOPPED_V - 0.01)) is not None
    assert self.sbl.active

  def test_brakes_at_a_stop_within_range_and_builds_up(self):
    v = 1.2
    out = self.sbl.update(v, lead(8.))
    assert out[0] == -RISE_STEP and out[1] is False                   # does not step in
    out = settle(self.sbl, v, lead(8.))
    assert abs(out[0] + A_STOP) < 1e-6                                 # stops where A_STOP ends: 8 - 1.44 m, within range
    assert abs(self.sbl.gap - (8. - v ** 2 / (2 * A_STOP))) < 1e-6

  def test_far_out_brakes_less_to_stop_at_gap_max(self):
    v, d = 1.5, GAP_MAX + 2.5
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
    assert abs(self.sbl.gap - (d - 0.5 ** 2 / (2 * A_STOP))) < 1e-6
    sbl = StopBehindLead(CP)
    settle(sbl, 0.5, lead(d), traffic_mode=True)
    assert sbl.gap == GAP_MAX_TRAFFIC

  def test_holds_the_stop_until_the_lead_pulls_away(self):
    settle(self.sbl, 1.0, lead(8.))
    accel, stop = self.sbl.update(STOP_V - 0.01, lead(8.))
    assert stop and accel <= 0.
    accel, stop = settle(self.sbl, 0., lead(8., v_lead_k=0.12))      # radar noise between the two thresholds: held
    assert stop and accel == 0.
    assert self.sbl.update(0., lead(8., v_lead_k=LEAD_DEPART_V_K + 0.01)) is None   # Launch Assist's gate: never later
    assert not self.sbl.active

  def test_lets_go_when_the_truck_speeds_up_or_long_turns_off(self):
    settle(self.sbl, 1.0, lead(8.))
    assert self.sbl.update(V_RELEASE + 0.1, lead(8.)) is None
    settle(self.sbl, 1.0, lead(8.))
    assert self.sbl.update(1.0, lead(8.), long_off=True) is None
    assert self.sbl.accel == 0.

  def test_lets_go_when_radar_switches_to_a_car_further_up(self):
    # stopped 8 m behind, then the car ahead changes lanes and radard's lead is the next car, stopped 20 m ahead
    settle(self.sbl, 0., lead(8.))
    assert self.sbl.active
    assert self.sbl.update(0., lead(20., track=2)) is None
    assert not self.sbl.active
    # the same car further out on the same track (no id change) lets go too
    settle(self.sbl, 0., lead(8.))
    assert self.sbl.update(0., lead(8. + GAP_MAX + 6.)) is None

  def test_keeps_the_same_car_on_a_new_track(self):
    settle(self.sbl, 1.0, lead(8.))
    out = self.sbl.update(1.0, lead(8.3, track=7))
    assert out is not None and self.sbl.track_id == 7 and abs(out[0] + A_STOP) < 1e-6

  def test_rides_out_a_short_radar_dropout(self):
    accel = settle(self.sbl, 1.0, lead(8.))[0]
    for _ in range(DROPOUT_FRAMES):
      assert self.sbl.update(1.0, lead(8., present=False)) == (accel, False)
    assert self.sbl.update(1.0, lead(8., present=False)) is None
    assert not self.sbl.active
    # gas override during a dropout still lets go at once
    settle(self.sbl, 1.0, lead(8.))
    assert self.sbl.update(1.0, lead(8., present=False), long_off=True) is None

  def test_braking_goes_away_at_once(self):
    settle(self.sbl, 1.2, lead(8.))
    out = self.sbl.update(0.6, lead(8.))
    # much less is needed now: no ramp on the way out
    assert abs(out[0] + 0.6 ** 2 / (2 * (8. - self.sbl.gap))) < 1e-6


def build_sm(v_ego, standstill, d_rel, v_lead, personality=log.LongitudinalPersonality.aggressive):
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
  radar.radarState.leadOne = {"present": True, "radar": True, "dRel": d_rel, "vLead": v_lead, "vLeadK": v_lead, "aLeadK": 0.,
                              "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()
  m = messaging.new_message('modelV2')
  m.modelV2.velocity.x = [v_ego] * 33
  m.modelV2.position.x = [float(i) for i in range(33)]
  m.modelV2.meta.disengagePredictions.gasPressProbs = [1.] * 6
  services['modelV2'] = m.modelV2.as_reader()
  return MockSubMaster(services)


def roll_up(planner, v0=1.3, d0=11., t=12.):
  """Kinematic truck that does what the planner asks, half a second late, rolling up behind a stopped car. Once the planner
  says stop, it stops at stopAccel like LongControl's stopping state."""
  v, d, q = v0, d0, [0.] * 10
  for _ in range(int(t / DT_MDL)):
    planner.update(build_sm(v, v < 0.01, d, 0.))
    q.append(-0.2 if planner.output_should_stop else float(planner.output_a_target))
    a = q.pop(0)
    v = max(v + a * DT_MDL, 0.)
    d -= v * DT_MDL
  return v, d


class TestStopBehindLeadPlanner(OpenpilotTestCase):
  def test_stops_near_where_the_crawl_begins(self):
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    v, d = roll_up(planner)
    assert v == 0.
    assert d > GAP_MAX - 2.5, d                      # stopped well short of the lead planner's own standstill gap

  def test_without_it_the_planner_crawls_in(self):
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    planner.stop_behind_lead.available = False
    v, d = roll_up(planner)
    assert d < STOP_DISTANCE + 1., d                  # what it fixes: rolled most of the way to ~6 m

  def test_launch_is_not_held_up(self):
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=1.3)
    roll_up(planner)
    assert planner.output_should_stop
    planner.update(build_sm(0., True, 8., LEAD_DEPART_V_K + 0.05))
    assert not planner.stop_behind_lead.active
