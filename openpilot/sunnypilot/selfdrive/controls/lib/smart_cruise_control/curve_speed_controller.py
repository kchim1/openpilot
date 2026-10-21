"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Curve Speed Control: a speed profile over the road ahead (feedforward + backward pass).

Unlike SCC-V (which reduces the model's curvature to one 97th-percentile number and reacts with a state machine),
this builds a full speed profile over the distance ahead and back-propagates a comfortable deceleration, so the
car slows early, holds a steady speed through the curve, and powers out, planning for lateral accel near a chosen
budget. Vision only: the model horizon (~100-150 m) is well beyond the ~40 m of lookahead canyon curves need, so
no map dependency. It lowers the entry speed rather than guaranteeing the budget: over 28 road curves on this truck's
logs the model's curvature at the apex was a median 32% low 3 s before it and 50% low 8 s before, so a curve is
often only fully seen once the truck is nearly in it.

Pairs with LateralLoadGovernor (lateral_load_governor.py), the reactive backstop that keeps the measured load
inside the steering's real limit when this feedforward under-reads a curve.
"""
import numpy as np

from openpilot.cereal import messaging, custom
from openpilot.common.params import Params
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_UNSET
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.sunnypilot import PARAMS_UPDATE_PERIOD
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control import MIN_V

CurveState = custom.LongitudinalPlanSP.SmartCruiseControl.CurveState

T_IDXS = np.array(ModelConstants.T_IDXS)  # 33-pt model horizon, 0..10 s (quadratic spacing)

# Lateral-accel budget the profile plans to. Same value for R1T and R1S (same vehicle family). This is the
# "feel" knob = how hard the truck loads the steering in curves; keep it a margin below the steering's lateral ceiling.
# 2.2: dropped 2.4->2.0 on AdventurePilot's 0000003b for overshoot headroom, then nudged to 2.2: canyon test 00000006
# showed 2.0 over-slows vs the human's ~4 m/s^2; split the difference (the governor still backstops the peak).
A_LAT_MAX = 2.2         # m/s^2
# Backward-pass decel: how fast the planned speed may fall toward a curve ahead, so braking starts early. Planned at
# what this planner delivers, not at AdventurePilot's 1.8: the speed cap is tracked by the cruise candidate
# (get_cruise_accel), which brakes at most A_CRUISE_MIN = -1.2 m/s^2 and in proportion to the speed error, so it runs
# about a second behind a falling target. In sim from 100 km/h with no lead, a 100 m radius curve planned for 14.8 m/s
# was entered at 22.4 m/s (5.0 m/s^2 lateral) with 1.8, and at 16.0 m/s (2.5) with 1.0.
A_DECEL = 1.0           # m/s^2
# The a_target this hands the planner (half the speed error, within these bounds) becomes its acceleration state when
# curve control wins: it seeds the MPC, which only matters with a lead (the set speed and this cap are tracked by the
# cruise candidate), and it is integrated into v_desired_filter. The planner clamps it to the incoming state, so it can
# only ever ask for less acceleration; do not add a positive a_target path that bypasses that clamp.
A_ACCEL = 1.2           # m/s^2, accel-out bound
V_TARGET_FLOOR = 2.0    # m/s, never command a crawl below this
KAPPA_FLOOR = 1e-4      # 1/m, ignore curvature below this (treat as straight)


class CurveSpeedController:
  def __init__(self):
    self.params = Params()
    self.frame = -1
    self._read_error_logged = False
    self.enabled = self.read_enabled()

    self.long_enabled = False
    self.long_override = False
    self.is_enabled = False
    self.is_active = False
    self.state = CurveState.disabled

    self.v_ego = 0.
    self.a_ego = 0.
    self.v_cruise_setpoint = V_CRUISE_UNSET
    self.v_target = V_CRUISE_UNSET
    self.a_lat_max = A_LAT_MAX
    self.lat_accel_target = 0.

    self.output_v_target = V_CRUISE_UNSET
    self.output_a_target = 0.

  def read_enabled(self) -> bool:
    try:
      return self.params.get_bool("CurveSpeedControl")
    except Exception:
      # params library built without the key, or a failed read: off rather than take plannerd down
      if not self._read_error_logged:
        cloudlog.exception("CurveSpeedControl read failed, treating it as off")
        self._read_error_logged = True
      return False

  def _update_params(self) -> None:
    if self.frame % int(PARAMS_UPDATE_PERIOD / DT_MDL) == 0:
      self.enabled = self.read_enabled()

  def _plan(self, sm: messaging.SubMaster) -> None:
    """Build kappa(s) ahead from the model, cap speed by a_lat_max, back-propagate a_decel, emit v_target."""
    vel = np.asarray(sm['modelV2'].velocity.x, dtype=np.float64)
    yaw = np.abs(np.asarray(sm['modelV2'].orientationRate.z, dtype=np.float64))
    if len(vel) != len(T_IDXS) or len(yaw) != len(T_IDXS):
      self.v_target = V_CRUISE_UNSET
      return

    # distance to each horizon point: trapezoidal integral of predicted speed over T_IDXS
    s = np.concatenate([[0.0], np.cumsum(0.5 * (vel[:-1] + vel[1:]) * np.diff(T_IDXS))])
    kappa = yaw / np.maximum(vel, 0.1)
    kappa = np.maximum(kappa, KAPPA_FLOOR)

    v_allow = np.minimum(np.sqrt(self.a_lat_max / kappa), self.v_cruise_setpoint)

    # backward pass (far -> near): only let speed drop as fast as a_decel allows -> brake early, smooth entry
    for i in range(len(v_allow) - 2, -1, -1):
      ds = s[i + 1] - s[i]
      if ds <= 0:
        continue
      v_allow[i] = min(v_allow[i], float(np.sqrt(v_allow[i + 1] ** 2 + 2.0 * A_DECEL * ds)))

    self.v_target = max(float(v_allow[0]), V_TARGET_FLOOR)

  def _update_state(self) -> tuple[bool, bool]:
    # a set speed of 0 (controlsd's forceDecel) is a real one: the profile is under it, so curve control is not working
    enabled = self.long_enabled and self.enabled and not self.long_override
    constraining = enabled and self.v_ego > MIN_V and self.v_target < self.v_cruise_setpoint - 0.5
    if not self.long_enabled or not self.enabled:
      self.state = CurveState.disabled
    elif self.long_override:
      self.state = CurveState.disabled  # manual override; let the driver have it
    elif not constraining:
      self.state = CurveState.cruise
    elif self.v_target < self.v_ego - 0.5:
      self.state = CurveState.slowing
    else:
      self.state = CurveState.curve
    active = self.state in (CurveState.slowing, CurveState.curve)
    return enabled, active

  def get_v_target_from_control(self) -> float:
    return self.v_target if self.is_active else V_CRUISE_UNSET

  def get_a_target_from_control(self) -> float:
    if self.is_active:
      return float(np.clip((self.v_target - self.v_ego) / 2.0, -A_DECEL, A_ACCEL))
    return self.a_ego

  def update(self, sm: messaging.SubMaster, long_enabled: bool, long_override: bool, v_ego: float, a_ego: float,
             v_cruise: float) -> None:
    self.long_enabled = long_enabled
    self.long_override = long_override
    self.v_ego = v_ego
    self.a_ego = a_ego
    self.v_cruise_setpoint = v_cruise

    self._update_params()
    self.a_lat_max = A_LAT_MAX
    if self.long_enabled and self.enabled:
      self._plan(sm)
    else:
      self.v_target = V_CRUISE_UNSET

    self.is_enabled, self.is_active = self._update_state()
    self.lat_accel_target = self.a_lat_max
    self.output_v_target = self.get_v_target_from_control()
    self.output_a_target = self.get_a_target_from_control()
    self.frame += 1
