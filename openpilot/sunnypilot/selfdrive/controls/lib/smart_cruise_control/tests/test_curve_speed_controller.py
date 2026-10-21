"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from unittest.mock import patch

import numpy as np

from opendbc.car import structs
from openpilot.cereal import messaging, custom, log
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.car.interfaces import _cleanup_unsupported_params
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import curve_speed_controller
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.curve_speed_controller import (
  CurveSpeedController, A_LAT_MAX, A_DECEL, A_ACCEL, V_TARGET_FLOOR)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import lateral_load_governor
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.lateral_load_governor import (
  LateralLoadGovernor, A_LAT_CEILING_TORQUE, A_LAT_CEILING_ANGLE, SETPOINT, TAPER_START, LOAD_LP, UNDERSTEER_TH)
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl

CurveState = custom.LongitudinalPlanSP.SmartCruiseControl.CurveState
N = len(ModelConstants.T_IDXS)


def make_model(yaw_rate, speed):
  model = messaging.new_message('modelV2')
  orientationRate = log.XYZTData.new_message()
  orientationRate.z = [float(y) for y in np.broadcast_to(yaw_rate, N)]
  model.modelV2.orientationRate = orientationRate
  velocity = log.XYZTData.new_message()
  velocity.x = [float(speed) for _ in range(N)]
  model.modelV2.velocity = velocity
  return model


def make_controls_state(curvature=0.0, saturated=False, actual=0.0, desired=0.0, angle=False):
  # what controlsd publishes: angleState while the Rivian kit steers on the angle channel, torqueState otherwise, and
  # the curvature from the steering angle in both
  cs = messaging.new_message('controlsState')
  cs.controlsState.curvature = float(curvature)
  if angle:
    st = cs.controlsState.lateralControlState.init('angleState')
    st.saturated = bool(saturated)
  else:
    st = cs.controlsState.lateralControlState.init('torqueState')
    st.saturated = bool(saturated)
    st.actualLateralAccel = float(actual)
    st.desiredLateralAccel = float(desired)
  return cs


def make_car_state(v_ego=20.0, yaw_rate=0.0):
  cs = messaging.new_message('carState')
  cs.carState.vEgo = float(v_ego)
  cs.carState.yawRate = float(yaw_rate)
  return cs


def sm_for(yaw_rate, speed, curvature=None, saturated=False, actual=0.0, desired=0.0, v_ego=20.0, angle=False, car_yaw_rate=0.0):
  # yaw_rate and speed are the model's. By default the measured state is the Rivian's: controlsState.curvature is what
  # torqueState.actualLateralAccel is built from (curvature * v^2), and carState.yawRate is never filled
  if curvature is None:
    curvature = actual / max(v_ego, 1.0) ** 2
  return {
    'modelV2': make_model(yaw_rate, speed).modelV2,
    'controlsState': make_controls_state(curvature, saturated, actual, desired, angle).controlsState,
    'carState': make_car_state(v_ego, car_yaw_rate).carState,
  }


class TestCurveSpeedController:
  def setup_method(self):
    Params().put_bool("CurveSpeedControl", True, block=True)
    self.scc = CurveSpeedController()
    self.scc.enabled = True

  def teardown_method(self):
    Params().put_bool("CurveSpeedControl", False, block=True)

  def test_initial_state(self):
    assert self.scc.state == CurveState.disabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_reads_the_toggle(self):
    assert CurveSpeedController().enabled
    Params().put_bool("CurveSpeedControl", False, block=True)
    assert not CurveSpeedController().enabled

  def test_straight_inactive(self):
    sm = sm_for(yaw_rate=0.001, speed=25.0, v_ego=25.0)
    self.scc.update(sm, True, False, 25.0, 0.0, 30.0)
    assert not self.scc.is_active
    assert self.scc.state == CurveState.cruise
    assert self.scc.output_v_target == V_CRUISE_UNSET
    assert self.scc.output_a_target == 0.0  # a_ego when not active

  def test_curve_slows(self):
    # sustained yaw rate -> a curve through the horizon -> target below set speed, slowing
    sm = sm_for(yaw_rate=0.07, speed=15.0, curvature=0.0047, v_ego=24.0)
    self.scc.update(sm, True, False, 24.0, 0.0, 30.0)
    assert self.scc.is_active
    assert self.scc.output_v_target < 30.0
    # physics check: v_target ~= sqrt(a_lat_max / kappa) for the sustained curve
    kappa = 0.07 / 15.0
    assert np.isclose(self.scc.v_target, (A_LAT_MAX / kappa) ** 0.5, rtol=0.05)
    assert self.scc.state == CurveState.slowing
    # asks to slow down, never harder than the comfortable decel
    assert -A_DECEL <= self.scc.output_a_target < 0.

  def test_holds_the_curve_speed_once_there(self):
    kappa = 0.07 / 15.0
    v_curve = (A_LAT_MAX / kappa) ** 0.5
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=v_curve)
    self.scc.update(sm, True, False, v_curve, 0.0, 30.0)
    assert self.scc.state == CurveState.curve
    assert self.scc.is_active

  def test_brakes_early_for_a_curve_ahead(self):
    # straight for the first ~2 s (~44 m), a tight curve after: the backward pass already caps the speed now, at what
    # a comfortable decel gets down to the curve speed by the time it arrives
    t_idxs = np.array(ModelConstants.T_IDXS)
    yaw = np.where(t_idxs < 2.0, 0.0, 0.1)
    sm = sm_for(yaw_rate=yaw, speed=20.0, v_ego=26.0)
    self.scc.update(sm, True, False, 26.0, 0.0, 30.0)
    assert self.scc.state == CurveState.slowing
    v_curve = (A_LAT_MAX / (0.1 / 20.0)) ** 0.5
    d_curve = 20.0 * t_idxs[t_idxs >= 2.0][0]
    assert np.isclose(self.scc.v_target, np.sqrt(v_curve ** 2 + 2 * A_DECEL * d_curve), rtol=0.01)
    assert v_curve < self.scc.v_target < 26.0

  def test_a_target_is_half_the_speed_error(self):
    # the a_target handed to the planner: half the speed error, so a cap slightly under the truck asks for a little decel
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=23.0)
    self.scc.update(sm, True, False, 23.0, 0.0, 30.0)
    assert -A_DECEL < self.scc.output_a_target < 0.
    assert np.isclose(self.scc.output_a_target, (self.scc.v_target - 23.0) / 2.0)

  def test_a_hot_entry_asks_for_at_most_the_planned_decel(self):
    # 25 m/s into a curve whose speed is 13.0 m/s: the speed error alone would ask for -6.0 m/s^2; it is bounded at the
    # backward pass's decel (the planner would otherwise seed the MPC with it)
    sm = sm_for(yaw_rate=0.195, speed=15.0, v_ego=25.0)
    self.scc.update(sm, True, False, 25.0, 0.0, 30.0)
    assert self.scc.v_target < 25.0 - 2 * A_DECEL
    assert self.scc.output_a_target == -A_DECEL

  def test_accel_out_is_bounded(self):
    # under its cap (a lead held the truck back) it asks for at most A_ACCEL; the planner clamps even that to its state
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=10.0)
    self.scc.update(sm, True, False, 10.0, 0.0, 30.0)
    assert self.scc.state == CurveState.curve
    assert self.scc.output_a_target == A_ACCEL

  def test_a_set_speed_just_over_the_curve_speed_is_not_curve_control_working(self):
    # within 0.5 m/s of the set speed the cap is not constraining: no flicker into the curve state on a curve taken at
    # about the set speed
    v_curve = (A_LAT_MAX / (0.07 / 15.0)) ** 0.5
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=v_curve)
    for set_speed, active in ((v_curve + 0.4, False), (v_curve + 0.6, True)):
      self.scc.update(sm, True, False, v_curve, 0.0, set_speed)
      assert self.scc.is_active == active, set_speed

  def test_force_decel_set_speed_is_not_curve_control_working(self):
    # controlsd's forceDecel (driver monitoring, soft disable) hands the planner a set speed of 0: cruise wins with it,
    # and curve control must not report itself working with a 148 m/s straight-road profile
    sm = sm_for(yaw_rate=0.001, speed=20.0, v_ego=20.0)
    self.scc.update(sm, True, False, 20.0, 0.0, 0.0)
    assert not self.scc.is_active
    assert self.scc.state == CurveState.cruise
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_never_plans_below_the_floor(self):
    # a hairpin sharper than 2.2 m/s^2 allows even at a crawl: the profile is floored, never a stop
    sm = sm_for(yaw_rate=5.0, speed=6.0, v_ego=6.0)
    self.scc.update(sm, True, False, 6.0, 0.0, 30.0)
    assert self.scc.v_target == V_TARGET_FLOOR

  def test_set_speed_below_the_curve_speed_is_not_constrained(self):
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=10.0)
    self.scc.update(sm, True, False, 10.0, 0.0, 10.0)
    assert not self.scc.is_active

  def test_toggle_off_disables(self):
    self.scc.enabled = False
    sm = sm_for(yaw_rate=0.07, speed=15.0, curvature=0.0047, v_ego=24.0)
    self.scc.update(sm, True, False, 24.0, 0.0, 30.0)
    assert self.scc.state == CurveState.disabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_long_disabled_inactive(self):
    sm = sm_for(yaw_rate=0.07, speed=15.0, curvature=0.0047, v_ego=24.0)
    self.scc.update(sm, False, False, 24.0, 0.0, 30.0)
    assert not self.scc.is_active
    assert self.scc.state == CurveState.disabled

  def test_override_lets_the_driver_have_it(self):
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=24.0)
    self.scc.update(sm, True, True, 24.0, 0.0, 30.0)
    assert not self.scc.is_active
    assert not self.scc.is_enabled
    assert self.scc.output_v_target == V_CRUISE_UNSET

  def test_below_min_speed_inactive(self):
    sm = sm_for(yaw_rate=0.07, speed=5.0, v_ego=5.0)
    self.scc.update(sm, True, False, 5.0, 0.0, 30.0)
    assert not self.scc.is_active

  def test_short_model_output_is_ignored(self):
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=24.0)
    model = messaging.new_message('modelV2')
    model.modelV2.velocity.x = [15.0] * 5
    model.modelV2.orientationRate.z = [0.07] * 5
    sm['modelV2'] = model.modelV2
    self.scc.update(sm, True, False, 24.0, 0.0, 30.0)
    assert not self.scc.is_active

  def test_toggle_is_read_again_while_driving(self):
    sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=24.0)
    Params().put_bool("CurveSpeedControl", False, block=True)
    period = int(PARAMS_UPDATE_PERIOD / DT_MDL)
    for _ in range(period + 1):
      self.scc.update(sm, True, False, 24.0, 0.0, 30.0)
    assert not self.scc.enabled
    assert self.scc.state == CurveState.disabled

  def test_unreadable_toggle_is_off_and_logged_once(self):
    with patch.object(Params, 'get_bool', side_effect=UnknownKeyName("CurveSpeedControl")), \
         patch.object(curve_speed_controller.cloudlog, 'exception') as log_exception:
      scc = CurveSpeedController()
      assert not scc.enabled
      sm = sm_for(yaw_rate=0.07, speed=15.0, v_ego=24.0)
      for _ in range(3 * int(PARAMS_UPDATE_PERIOD / DT_MDL)):
        scc.update(sm, True, False, 24.0, 0.0, 30.0)
      assert not scc.enabled
      assert scc.state == CurveState.disabled
      assert log_exception.call_count == 1


class TestLateralLoadGovernor:
  def setup_method(self):
    self.gov = LateralLoadGovernor()
    self.gov.enabled = True

  def test_low_load_passthrough(self):
    sm = sm_for(yaw_rate=0.02, speed=20.0, actual=1.0, desired=1.0, v_ego=20.0)
    self.gov.update(sm, True, False, 20.0)
    assert self.gov.output_v_target == V_CRUISE_UNSET
    assert self.gov.throttle_scale() == 1.0

  def test_overload_caps_speed(self):
    over = A_LAT_CEILING_TORQUE * 1.1
    sm = sm_for(yaw_rate=0.05, speed=24.0, actual=over, desired=over, v_ego=24.0)
    for _ in range(20):  # let the EMA converge
      self.gov.update(sm, True, False, 24.0)
    assert self.gov.is_active
    # the speed that brings the load back to the setpoint: a_lat goes with v^2
    assert np.isclose(self.gov.output_v_target, 24.0 * np.sqrt(SETPOINT / 1.1), rtol=0.01)
    assert self.gov.throttle_scale() == 0.0  # past the ceiling: no throttle

  def test_interlock_zeros_throttle_when_running_wide(self):
    sm = sm_for(yaw_rate=0.05, speed=24.0, saturated=True, actual=2.0, desired=3.0, v_ego=24.0)
    self.gov.update(sm, True, False, 24.0)
    assert self.gov.throttle_scale() == 0.0  # saturated + desired>actual => running wide => no throttle

  def test_throttle_fades_between_taper_start_and_the_ceiling(self):
    for frac in (TAPER_START, 0.8, 0.9):
      gov = LateralLoadGovernor()
      gov.enabled = True
      a_lat = frac * A_LAT_CEILING_TORQUE
      sm = sm_for(yaw_rate=0.05, speed=24.0, actual=a_lat, desired=a_lat, v_ego=24.0)
      for _ in range(40):
        gov.update(sm, True, False, 24.0)
      assert np.isclose(gov.throttle_scale(), (1. - frac) / (1. - TAPER_START), atol=1e-3)

  def test_yaw_rate_counts_on_a_car_that_reports_one(self):
    # the Rivian never fills carState.yawRate, but on a car that does, v * yaw rate wins over lower steering readings
    over = A_LAT_CEILING_TORQUE * 1.1
    sm = sm_for(yaw_rate=0.05, speed=24.0, actual=0.5, desired=0.5, v_ego=24.0, car_yaw_rate=over / 24.0)
    for _ in range(20):
      self.gov.update(sm, True, False, 24.0)
    assert self.gov.is_active

  def test_angle_steering_uses_the_curvature_clip_as_its_ceiling(self):
    # controlsd publishes angleState while the kit steers on angle: the binding limit there is the 3.0 m/s^2
    # curvature clip, and the load comes from controlsState.curvature (angleState has no lateral accel, and the Rivian
    # no yaw rate)
    a_lat = 2.7  # 96% of the torque ceiling, 90% of the angle one
    torque, angle = LateralLoadGovernor(), LateralLoadGovernor()
    torque.enabled = angle.enabled = True
    for _ in range(40):
      torque.update(sm_for(yaw_rate=0.05, speed=24.0, actual=a_lat, desired=a_lat, v_ego=24.0), True, False, 24.0)
      angle.update(sm_for(yaw_rate=0.05, speed=24.0, v_ego=24.0, angle=True, curvature=a_lat / 24.0 ** 2), True, False, 24.0)
    assert torque.ceiling == A_LAT_CEILING_TORQUE
    assert angle.ceiling == A_LAT_CEILING_ANGLE == 3.0
    assert np.isclose(torque.load, a_lat / A_LAT_CEILING_TORQUE, atol=1e-3)
    assert np.isclose(angle.load, a_lat / A_LAT_CEILING_ANGLE, atol=1e-3)
    assert torque.is_active and angle.is_active
    assert angle.output_v_target > torque.output_v_target
    assert np.isclose(angle.throttle_scale(), (1. - a_lat / A_LAT_CEILING_ANGLE) / (1. - TAPER_START), atol=1e-3)

  def test_angle_steering_sees_the_load_without_a_yaw_rate(self):
    # as the truck logs it: angleState, carState.yawRate 0, only controlsState.curvature. 2.4 m/s^2 is 80% of the angle
    # ceiling: no cap yet, the throttle fading
    for a_lat, active in ((2.4, False), (2.7, True)):
      gov = LateralLoadGovernor()
      gov.enabled = True
      sm = sm_for(yaw_rate=0.05, speed=24.0, v_ego=24.0, angle=True, curvature=-a_lat / 24.0 ** 2, car_yaw_rate=0.0)
      for _ in range(40):
        gov.update(sm, True, False, 24.0)
      assert np.isclose(gov.load, a_lat / A_LAT_CEILING_ANGLE, atol=1e-3)  # either direction of the curve
      assert gov.is_active == active
      assert np.isclose(gov.throttle_scale(), (1. - a_lat / A_LAT_CEILING_ANGLE) / (1. - TAPER_START), atol=1e-3)

  def test_a_frame_on_the_other_controller_keeps_the_load(self):
    # the cooperative torque fallback flips controlsd between torqueState and angleState for a frame or two: the load,
    # measured from the same curvature in both, must not drop toward zero on the angle frames
    a_lat = 0.9 * A_LAT_CEILING_TORQUE
    for _ in range(40):
      self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, actual=a_lat, desired=a_lat, v_ego=24.0), True, False, 24.0)
    before = self.gov.load
    self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, v_ego=24.0, angle=True, curvature=a_lat / 24.0 ** 2), True, False, 24.0)
    assert self.gov.load > 0.95 * before  # a zero fed in for that frame would cut it by 40%

  def test_a_nan_sample_is_skipped(self):
    a_lat = 0.5 * A_LAT_CEILING_TORQUE
    for _ in range(30):
      self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, actual=a_lat, desired=a_lat, v_ego=24.0), True, False, 24.0)
    before = self.gov.load
    self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, actual=float('nan'), desired=0.0, v_ego=24.0), True, False, 24.0)
    assert self.gov.load == before
    self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, actual=a_lat, desired=a_lat, v_ego=24.0), True, False, 24.0)
    assert np.isfinite(self.gov.load) and np.isfinite(self.gov.throttle_scale())

  def test_a_one_frame_spike_is_filtered(self):
    base = 0.5 * A_LAT_CEILING_TORQUE
    for _ in range(30):
      self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, actual=base, desired=base, v_ego=24.0), True, False, 24.0)
    spike = 1.2 * A_LAT_CEILING_TORQUE
    self.gov.update(sm_for(yaw_rate=0.05, speed=24.0, actual=spike, desired=spike, v_ego=24.0), True, False, 24.0)
    assert np.isclose(self.gov.load, 0.5 + LOAD_LP * (1.2 - 0.5), atol=1e-3)
    assert not self.gov.is_active
    assert self.gov.throttle_scale() > 0.5

  def test_angle_steering_saturated_is_running_wide(self):
    # angle control has no lateral accel to compare: its saturation (angle can't track) is the running-wide signal
    sm = sm_for(yaw_rate=0.05, speed=24.0, v_ego=24.0, angle=True, saturated=True, curvature=1.0 / 24.0 ** 2)
    self.gov.update(sm, True, False, 24.0)
    assert self.gov.running_wide
    assert self.gov.throttle_scale() == 0.0

  def test_torque_saturated_without_understeer_is_not_running_wide(self):
    sm = sm_for(yaw_rate=0.05, speed=24.0, saturated=True, actual=2.0, desired=2.0, v_ego=24.0)
    self.gov.update(sm, True, False, 24.0)
    assert not self.gov.running_wide

  def test_understeer_threshold(self):
    for short_by, wide in ((UNDERSTEER_TH - 0.02, False), (UNDERSTEER_TH + 0.02, True)):
      sm = sm_for(yaw_rate=0.05, speed=24.0, saturated=True, actual=2.0, desired=2.0 + short_by, v_ego=24.0)
      self.gov.update(sm, True, False, 24.0)
      assert self.gov.running_wide == wide, short_by

  def test_the_load_starts_over_after_an_override(self):
    # the driver's gas override (or the setting off) drops what was measured before: the cap does not come straight
    # back from an old curve when the driver lets go
    over = A_LAT_CEILING_TORQUE * 1.1
    sm = sm_for(yaw_rate=0.05, speed=24.0, actual=over, desired=over, v_ego=24.0)
    for _ in range(20):
      self.gov.update(sm, True, False, 24.0)
    assert self.gov.is_active
    self.gov.update(sm, True, True, 24.0)
    assert self.gov.load == 0.
    self.gov.update(sm, True, False, 24.0)
    assert np.isclose(self.gov.load, LOAD_LP * 1.1)
    assert not self.gov.is_active

  def test_the_cap_is_floored(self):
    # just above the 20 km/h gate with an absurd load: the speed cap stops at the floor, never a stop
    huge = A_LAT_CEILING_TORQUE * 20.
    sm = sm_for(yaw_rate=0.05, speed=6.0, actual=huge, desired=huge, v_ego=6.0)
    for _ in range(40):
      self.gov.update(sm, True, False, 6.0)
    assert self.gov.is_active
    assert self.gov.output_v_target == lateral_load_governor.V_TARGET_FLOOR

  def test_off_override_or_long_off_is_inert(self):
    over = A_LAT_CEILING_TORQUE * 1.1
    sm = sm_for(yaw_rate=0.05, speed=24.0, saturated=True, actual=over, desired=over + 1., v_ego=24.0)
    for enabled, long_enabled, override in ((False, True, False), (True, False, False), (True, True, True)):
      gov = LateralLoadGovernor()
      gov.enabled = enabled
      for _ in range(20):
        gov.update(sm, long_enabled, override, 24.0)
      assert not gov.is_active
      assert gov.output_v_target == V_CRUISE_UNSET
      assert gov.throttle_scale() == 1.0

  def test_below_min_speed_does_not_cap(self):
    over = A_LAT_CEILING_TORQUE * 1.1
    sm = sm_for(yaw_rate=0.3, speed=4.0, actual=over, desired=over, v_ego=4.0)
    for _ in range(20):
      self.gov.update(sm, True, False, 4.0)
    assert not self.gov.is_active


class TestTune:
  def test_the_tuned_values_are_pinned(self):
    # a retune has to be a deliberate change here. A_DECEL is planned at what this branch's planner delivers (see
    # curve_speed_controller.py); the others are AdventurePilot's final tune
    assert A_LAT_MAX == 2.2
    assert A_DECEL == 1.0
    assert SETPOINT == 0.85
    assert TAPER_START == 0.70
    assert A_LAT_CEILING_TORQUE == 2.8
    assert A_LAT_CEILING_ANGLE == 3.0


class TestSmartCruiseControl:
  def teardown_method(self):
    Params().put_bool("CurveSpeedControl", False, block=True)

  def test_governor_shares_the_curve_toggle(self):
    over = A_LAT_CEILING_TORQUE * 1.1
    sm = sm_for(yaw_rate=0.05, speed=24.0, actual=over, desired=over, v_ego=24.0)
    for enabled in (True, False):
      Params().put_bool("CurveSpeedControl", enabled, block=True)
      scc = SmartCruiseControl()
      for _ in range(20):
        scc.update(sm, True, False, 24.0, 0.0, 30.0)
      assert scc.governor.enabled == enabled
      assert scc.governor.is_active == enabled


class TestForgottenWithoutLongitudinal:
  # like SCC-V and SCC-M: with openpilot longitudinal off, the car's own cruise drives and curve control can do nothing
  def teardown_method(self):
    Params().put_bool("CurveSpeedControl", False, block=True)

  def test_cleared_at_car_start_without_openpilot_longitudinal(self):
    for long, kept in ((True, True), (False, False)):
      Params().put_bool("CurveSpeedControl", True, block=True)
      _cleanup_unsupported_params(structs.CarParams(openpilotLongitudinalControl=long), structs.CarParamsSP(pcmCruiseSpeed=True))
      assert Params().get_bool("CurveSpeedControl") == kept

  def test_cleared_with_icbm_but_no_openpilot_longitudinal(self):
    # ICBM keeps SCC-V and SCC-M, but every Curve Speed Control toggle is shown with openpilot longitudinal only: left on
    # here, it could not be turned off anywhere
    Params().put_bool("CurveSpeedControl", True, block=True)
    _cleanup_unsupported_params(structs.CarParams(openpilotLongitudinalControl=False),
                                structs.CarParamsSP(pcmCruiseSpeed=False, intelligentCruiseButtonManagementAvailable=True))
    assert not Params().get_bool("CurveSpeedControl")

  def test_a_params_library_without_the_key_does_not_stop_the_car_starting(self):
    remove = Params.remove

    def remove_unknown(self, key):
      if key == "CurveSpeedControl":
        raise UnknownKeyName(key)
      return remove(self, key)

    with patch.object(Params, 'remove', remove_unknown):
      _cleanup_unsupported_params(structs.CarParams(openpilotLongitudinalControl=False), structs.CarParamsSP(pcmCruiseSpeed=True))
