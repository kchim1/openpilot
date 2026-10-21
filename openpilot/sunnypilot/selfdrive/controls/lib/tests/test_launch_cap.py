"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import numpy as np

from openpilot.cereal import log, messaging
from opendbc.car import structs
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_cruise_accel, get_max_accel, get_launch_cap, \
  apply_launch_cap, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS, A_FOLLOW_MAX_VALS, LAUNCH_CAP_BP, LAUNCH_CAP_AGGRESSIVE, LAUNCH_CAP_STANDARD, \
  LAUNCH_CAP_RELAXED
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockSubMaster

Personality = log.LongitudinalPersonality
AGGRESSIVE, STANDARD, RELAXED = Personality.aggressive, Personality.standard, Personality.relaxed
CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)


def settled_cap(e2e, v_ego, personality, follow_share=0.):
  # set speed far above, straight wheel, throttle allowed: the cruise candidate ramps up to its cap
  a = 0.
  for _ in range(200):
    a = get_cruise_accel(e2e, 40., v_ego, a, 0., CP, DT_MDL, ACCEL_MAX, True, follow_share, 1., personality)
  return a


class TestLaunchCap:
  def test_personality_caps(self):
    assert get_launch_cap(AGGRESSIVE) == LAUNCH_CAP_AGGRESSIVE
    assert get_launch_cap(STANDARD) == LAUNCH_CAP_STANDARD
    assert get_launch_cap(RELAXED) == LAUNCH_CAP_RELAXED
    assert LAUNCH_CAP_RELAXED < LAUNCH_CAP_STANDARD < LAUNCH_CAP_AGGRESSIVE < A_CRUISE_MAX_VALS[0]

  def test_holds_then_fades_into_the_stock_cap_and_never_raises_it(self):
    for v in np.linspace(0., 20., 41):
      for p in (AGGRESSIVE, STANDARD, RELAXED):
        for stock in (ACCEL_MAX, float(np.interp(v, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)), 0.3):
          cap = apply_launch_cap(stock, v, p)
          assert cap <= stock + 1e-9
          if v <= LAUNCH_CAP_BP[0]:
            assert abs(cap - min(stock, get_launch_cap(p))) < 1e-9
          if v >= LAUNCH_CAP_BP[1]:
            assert cap == stock

  def test_launch_from_a_stop_is_capped_end_to_end_and_on_radar_cruise(self):
    for e2e in (True, False):
      for p in (AGGRESSIVE, STANDARD, RELAXED):
        assert abs(settled_cap(e2e, 0., p) - get_launch_cap(p)) < 1e-6
    # stock before: 2.0 end to end, 1.6 on radar cruise
    assert abs(settled_cap(True, 0., None) - ACCEL_MAX) < 1e-6
    assert abs(settled_cap(False, 0., None) - get_max_accel(0.)) < 1e-6

  def test_unchanged_at_speed(self):
    for e2e in (True, False):
      for v in (LAUNCH_CAP_BP[1], 20., 30.):
        assert abs(settled_cap(e2e, v, AGGRESSIVE) - settled_cap(e2e, v, None)) < 1e-6

  def test_keep_up_with_traffic_still_lifts_it(self):
    for v in (0., 3., 8.):
      follow = float(np.interp(v, A_CRUISE_MAX_BP, A_FOLLOW_MAX_VALS))
      for p in (AGGRESSIVE, STANDARD):
        base = apply_launch_cap(get_max_accel(v), v, p)
        assert abs(settled_cap(False, v, p, 0.5) - (base + 0.5 * (follow - base))) < 1e-6
        assert abs(settled_cap(False, v, p, 1.0) - follow) < 1e-6
      assert abs(settled_cap(True, v, AGGRESSIVE, 1.0) - ACCEL_MAX) < 1e-6


def build_sm(v_ego, standstill, personality, experimental, model_accel):
  services = {}
  for service in ("vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation", "carControl", "radarState"):
    services[service] = getattr(messaging.new_message(service), service)
  ss = messaging.new_message('selfdriveState')
  ss.selfdriveState.personality = personality
  ss.selfdriveState.experimentalMode = experimental
  services['selfdriveState'] = ss.selfdriveState.as_reader()
  cs = messaging.new_message('controlsState')
  cs.controlsState.longControlState = 'pid'
  services['controlsState'] = cs.controlsState.as_reader()
  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.standstill = standstill
  car_state.carState.vCruise = 70.0
  car_state.carState.vCruiseCluster = 70.0
  services['carState'] = car_state.carState.as_reader()
  m = messaging.new_message('modelV2')
  m.modelV2.velocity.x = [v_ego] * 33
  m.modelV2.position.x = [float(i) for i in range(33)]
  m.modelV2.action.desiredAcceleration = model_accel
  m.modelV2.meta.disengagePredictions.gasPressProbs = [1.] * 6
  services['modelV2'] = m.modelV2.as_reader()
  return MockSubMaster(services)


class TestLaunchCapPlanner(OpenpilotTestCase):
  def launch(self, personality, experimental):
    """No one ahead, the light turns green: the model asks for 1.9 m/s^2 (route 0000001d 7:24). Returns the peak request
    while the full launch cap holds (up to LAUNCH_CAP_BP[0])."""
    planner = LongitudinalPlanner(CP, structs.CarParamsSP())
    v, peak = 0., 0.
    while v < LAUNCH_CAP_BP[0]:
      planner.update(build_sm(v, v < 0.01, personality, experimental, 1.9))
      peak = max(peak, float(planner.output_a_target))
      v += max(float(planner.output_a_target), 0.) * DT_MDL
    return peak, v

  def test_no_lead_launch_is_softer_by_personality(self):
    for experimental in (True, False):
      peaks = [self.launch(p, experimental)[0] for p in (AGGRESSIVE, STANDARD, RELAXED)]
      for peak, p in zip(peaks, (AGGRESSIVE, STANDARD, RELAXED), strict=True):
        assert peak <= get_launch_cap(p) + 1e-6, (experimental, p, peak)
      assert peaks[0] > peaks[1] > peaks[2]
