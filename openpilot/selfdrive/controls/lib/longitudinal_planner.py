#!/usr/bin/env python3
import math
import numpy as np

import openpilot.cereal.messaging as messaging
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.constants import CV
from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.selfdrive.modeld.constants import ModelConstants
from openpilot.selfdrive.controls.lib.longcontrol import LongCtrlState
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, LongitudinalPlanSource
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import T_IDXS as T_IDXS_MPC
from openpilot.selfdrive.controls.lib.drive_helpers import CONTROL_N, get_accel_from_plan, should_stop
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX, V_CRUISE_UNSET
from openpilot.common.swaglog import cloudlog

from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, APPROACH_ONSET_OTHER_MARGIN, \
  APPROACH_ONSET_RESET_HOLD

A_CRUISE_MAX_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MAX_BP = [0., 10.0, 25., 40.]
# Keep Up With Traffic: the cap while following a moving radar lead, so the lead planner can hold the gap behind a car
# that pulls away briskly. The MPC still decides how much of it to use; 2.0 is also the Rivian request limit.
# follow_share (0 to 1, set by the sunnypilot planner from the personality and how far the gap is past the desired one)
# is how much of the step from the stock cap it gets. Tuned with the Rivian long integral term removed: with it, a lead
# that pulls away and then brakes hard left too little gap in sim
A_FOLLOW_MAX_VALS = [2.0, 2.0, 1.4, 1.0]
A_FOLLOW_TOTAL_MAX = 2.0
J_CRUISE_VALS = [1.6, 1.2, 0.8, 0.6]
A_CRUISE_MIN = -1.2
# Launch: the throttle cap from a stop and at low speed, by driving personality. The stock caps (1.6 m/s^2 on radar
# cruise, 2.0 end to end, where the driving model launches at ~1.9) felt too strong pulling away from a red light with no
# one ahead (route 0000001d 7:24: 2.1 m/s^2 measured, as the truck delivers ~12% more than it is asked for). The cap holds
# up to LAUNCH_CAP_BP[0], fades into the normal one by LAUNCH_CAP_BP[1], and never raises it. Keep Up With Traffic still
# lifts it toward its own cap behind a lead that pulls away
LAUNCH_CAP_BP = [4., 12.]
LAUNCH_CAP_AGGRESSIVE = 1.4
LAUNCH_CAP_STANDARD = 1.2
LAUNCH_CAP_RELAXED = 1.0
# Traffic Mode: gentler acceleration where it follows closer, TRAFFIC_MODE_ACCEL_CAP up to 30 mph, rising to the request
# limit (no cap) by 40 mph. After Keep Up With Traffic, so that can't lift the cruise accel past it, and the cruise candidate
# is always in the min() of the planner's candidates, so it caps the planner's output too. A min: it never raises a lower cap
TRAFFIC_MODE_ACCEL_CAP = 1.5                # m/s^2
TRAFFIC_MODE_ACCEL_CAP_BP = [13.41, 17.88]  # m/s
CONTROL_N_T_IDX = ModelConstants.T_IDXS[:CONTROL_N]
ALLOW_THROTTLE_THRESHOLD = 0.4
MIN_ALLOW_THROTTLE_SPEED = 2.5

# Lookup table for turns
_A_TOTAL_MAX_V = [1.7, 3.2]
_A_TOTAL_MAX_BP = [20., 40.]

def get_max_accel(v_ego, follow_share=0.):
  max_accel = np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS)
  if follow_share > 0.:
    max_accel += follow_share * (np.interp(v_ego, A_CRUISE_MAX_BP, A_FOLLOW_MAX_VALS) - max_accel)
  return max_accel

def get_launch_cap(personality) -> float:
  # compared like get_T_FOLLOW does: selfdriveState.personality is a capnp enum, not a dict key
  if personality == log.LongitudinalPersonality.aggressive:
    return LAUNCH_CAP_AGGRESSIVE
  elif personality == log.LongitudinalPersonality.relaxed:
    return LAUNCH_CAP_RELAXED
  return LAUNCH_CAP_STANDARD

def apply_launch_cap(max_accel, v_ego, personality):
  w = float(np.interp(v_ego, LAUNCH_CAP_BP, [1., 0.]))
  return min(max_accel, max_accel + w * (get_launch_cap(personality) - max_accel))

def get_coast_accel(pitch):
  return np.sin(pitch) * -5.65 - 0.3  # fitted from data using xx/projects/allow_throttle/compute_coast_accel.py

def get_cruise_accel(e2e, v_cruise, v_ego, a_cruise_prev, angle_steers, CP, dt, accel_coast, allow_throttle, follow_share=0.,
                     throttle_scale=1., personality=None, traffic_mode=False):
  if personality is None:
    max_accel = ACCEL_MAX if e2e else get_max_accel(v_ego, follow_share)
  else:
    stock = ACCEL_MAX if e2e else float(np.interp(v_ego, A_CRUISE_MAX_BP, A_CRUISE_MAX_VALS))
    follow = ACCEL_MAX if e2e else float(np.interp(v_ego, A_CRUISE_MAX_BP, A_FOLLOW_MAX_VALS))
    base = apply_launch_cap(stock, v_ego, personality)
    max_accel = base + follow_share * (follow - base) if follow_share > 0. else base

  if not e2e:
    a_total_max = np.interp(v_ego, _A_TOTAL_MAX_BP, _A_TOTAL_MAX_V)
    a_y = v_ego ** 2 * angle_steers * CV.DEG_TO_RAD / (CP.steerRatio * CP.wheelbase)
    if follow_share > 0.:
      # the higher total is for nearly straight roads only: gone by 1 m/s^2 lateral, so curves keep the stock limit
      a_total_max += follow_share * max(A_FOLLOW_TOTAL_MAX - a_total_max, 0.) * float(np.interp(abs(a_y), [0.5, 1.0], [1., 0.]))
    a_x_allowed = math.sqrt(max(a_total_max ** 2 - a_y ** 2, 0.))
    max_accel = min(max_accel, a_x_allowed)
    if not allow_throttle:
      clipped_accel_coast = max(accel_coast, ACCEL_MIN)
      coast_limit = np.interp(v_ego, [MIN_ALLOW_THROTTLE_SPEED, MIN_ALLOW_THROTTLE_SPEED*2], [max_accel, clipped_accel_coast])
      max_accel = min(max_accel, coast_limit)

  # Traffic Mode's cap, on top of all the above including Keep Up With Traffic's
  if traffic_mode:
    max_accel = min(max_accel, float(np.interp(v_ego, TRAFFIC_MODE_ACCEL_CAP_BP, [TRAFFIC_MODE_ACCEL_CAP, ACCEL_MAX])))

  # Curve Speed Control's throttle-fade interlock (lateral_load_governor.py): no added throttle while the steering is near
  # or at its lateral limit. Every other candidate only lowers the target, so this caps the planner's throttle, through the
  # cruise jerk limit. Only a positive cap is scaled: it never raises a coast limit, and never asks for braking
  if max_accel > 0.:
    max_accel *= throttle_scale

  target_accel = np.clip(v_cruise - v_ego, A_CRUISE_MIN, max_accel)
  j_cruise = np.interp(v_ego, A_CRUISE_MAX_BP, J_CRUISE_VALS)
  target_accel = float(np.clip(target_accel, a_cruise_prev - j_cruise * dt, a_cruise_prev + j_cruise * dt))

  return target_accel


class LongitudinalPlanner(LongitudinalPlannerSP):
  def __init__(self, CP, CP_SP, init_v=0.0, init_a=0.0, dt=DT_MDL):
    self.CP = CP
    self.mpc = LongitudinalMpc(dt=dt)
    LongitudinalPlannerSP.__init__(self, self.CP, CP_SP, self.mpc)
    self.fcw = False
    self.dt = dt
    self.allow_throttle = True
    self.e2e_prev = False
    self.onset_hold = int(round(APPROACH_ONSET_RESET_HOLD / self.dt))

    self.v_desired_filter = FirstOrderFilter(init_v, 2.0, self.dt)
    self.a_cruise = init_a
    self.output_a_target = init_a
    self.output_should_stop = False

    self.v_desired_trajectory = np.zeros(CONTROL_N)
    self.a_desired_trajectory = np.zeros(CONTROL_N)
    self.j_desired_trajectory = np.zeros(CONTROL_N)

  def update(self, sm):
    LongitudinalPlannerSP.update(self, sm)

    if len(sm['carControl'].orientationNED) == 3:
      accel_coast = get_coast_accel(sm['carControl'].orientationNED[1])
    else:
      accel_coast = ACCEL_MAX

    v_ego = sm['carState'].vEgo
    v_cruise_kph = min(sm['carState'].vCruise, V_CRUISE_MAX)
    v_cruise = v_cruise_kph * CV.KPH_TO_MS
    if sm['controlsState'].forceDecel:
      v_cruise = 0.0

    long_control_off = sm['controlsState'].longControlState == LongCtrlState.off

    # Reset current state when not engaged, or user is controlling the speed
    reset_state = long_control_off if self.CP.openpilotLongitudinalControl else not sm['selfdriveState'].enabled
    # PCM cruise speed may be updated a few cycles later, check if initialized
    v_cruise_initialized = sm['carState'].vCruise != V_CRUISE_UNSET
    reset_state = reset_state or not v_cruise_initialized

    throttle_probs = sm['modelV2'].meta.disengagePredictions.gasPressProbs
    throttle_prob = throttle_probs[1] if len(throttle_probs) > 1 else 1.0
    self.allow_throttle = throttle_prob > ALLOW_THROTTLE_THRESHOLD or v_ego <= MIN_ALLOW_THROTTLE_SPEED

    steer_angle_without_offset = sm['carState'].steeringAngleDeg - sm['vehicleParameters'].angleOffsetDeg

    if reset_state:
      self.v_desired_filter.x = v_ego
      self.output_a_target = np.clip(sm['carState'].aEgo, ACCEL_MIN, ACCEL_MAX)
      self.a_cruise = self.output_a_target

    # Prevent divergence, smooth in current v_ego
    self.v_desired_filter.x = max(0.0, self.v_desired_filter.update(v_ego))

    # No change cost when user is controlling the speed, or when standstill
    prev_accel_constraint = not (reset_state or sm['carState'].standstill)

    # Get new v_cruise and a_target from Smart Cruise Control and Speed Limit Assist
    v_cruise, self.output_a_target = LongitudinalPlannerSP.update_targets(self, sm, self.v_desired_filter.x, self.output_a_target, v_cruise)

    self.mpc.set_weights(prev_accel_constraint, personality=sm['selfdriveState'].personality)
    self.mpc.set_cur_state(self.v_desired_filter.x, self.output_a_target)
    self.mpc.update(sm['radarState'], personality=sm['selfdriveState'].personality)

    self.v_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.v_solution)
    self.a_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC, self.mpc.a_solution)
    self.j_desired_trajectory = np.interp(CONTROL_N_T_IDX, T_IDXS_MPC[:-1], self.mpc.j_solution)

    # TODO counter is only needed because radar is glitchy, remove once radar is gone
    self.fcw = self.mpc.crash_cnt > 2 and not sm['carState'].standstill
    if self.fcw:
      cloudlog.info("FCW triggered")

    # Save starting point for next iteration
    a_prev = self.output_a_target

    action_t =  self.CP.longitudinalActuatorDelay + DT_MDL
    output_a_target_mpc = get_accel_from_plan(self.v_desired_trajectory, self.a_desired_trajectory, CONTROL_N_T_IDX,
                                              action_t=action_t)
    output_should_stop_mpc = should_stop(v_ego, output_a_target_mpc)
    output_a_target_e2e = sm['modelV2'].action.desiredAcceleration
    output_should_stop_e2e = sm['modelV2'].action.shouldStop

    is_e2e = self.is_e2e(sm)
    # Handing back from end to end to ACC (DEC, or Experimental Mode turned off): while e2e, the cruise accel can build up
    # to ACCEL_MAX behind the min() because the e2e candidate is lower. Restart it from the last target (at least 0) so it
    # builds up again at the cruise jerk to the ACC cap, instead of jumping there in one step
    if self.e2e_prev and not is_e2e:
      self.a_cruise = min(self.a_cruise, max(a_prev, 0.))
    self.e2e_prev = is_e2e

    self.a_cruise = get_cruise_accel(is_e2e, v_cruise, v_ego,
                                     self.a_cruise, steer_angle_without_offset, self.CP, self.dt,
                                     accel_coast, self.allow_throttle,
                                     self.keep_pace(sm['radarState'].leadOne, sm['selfdriveState'].personality, v_ego),
                                     self.scc.governor.throttle_scale(), sm['selfdriveState'].personality, self.mpc.traffic_mode)
    cruise_should_stop = should_stop(v_ego, self.a_cruise)

    # Behind a car that has stopped or is stopping, the lead planner's braking beyond one even stop at the gap eases off
    # to that stop. Its own braking only, before the min(): anything else asking for more still wins
    floor = self.stop_behind_lead.profile_floor(v_ego, sm['radarState'].leadOne, long_off=reset_state,
                                                traffic_mode=self.mpc.traffic_mode)
    output_a_target_mpc = self.stop_behind_lead.relax(output_a_target_mpc,
                                                      floor if self.mpc.source == LongitudinalPlanSource.lead0 else None,
                                                      a_prev, off=self.fcw or reset_state)

    candidates = [(output_a_target_mpc, self.mpc.source, output_should_stop_mpc),
                  (self.a_cruise, LongitudinalPlanSource.cruise, cruise_should_stop)]
    if is_e2e:
      candidates.append((output_a_target_e2e, LongitudinalPlanSource.e2e, output_should_stop_e2e))
    # Red Light Stop Distance: extra braking on top of the model's to stop short of its stop point at a red light
    red_light = self.red_light_stop.update(is_e2e, v_ego, output_a_target_e2e, sm['modelV2'], sm['radarState'].leadOne,
                                           long_off=reset_state)
    if red_light is not None:
      candidates.append((red_light[0], LongitudinalPlanSource.e2e, red_light[1]))
    # Stop behind a stopped car at walking pace instead of crawling up to the standstill gap
    stop_behind = self.stop_behind_lead.update(v_ego, sm['radarState'].leadOne, long_off=reset_state,
                                               traffic_mode=self.mpc.traffic_mode)
    if stop_behind is not None:
      candidates.append((stop_behind[0], LongitudinalPlanSource.lead0, stop_behind[1]))

    output_a_target, self.mpc.source, _ = min(candidates, key=lambda c: c[0])
    self.output_should_stop = any(should_stop for _, _, should_stop in candidates)
    # Brake Sooner for Stops: ease into braking for a stopped or slow car while a gentle stop is still possible. Only the
    # braking that car calls for: a request for clearly more is for something else and goes through. Not after a reset
    # (gas override, engaging) until the plan first gets to 0, since a_prev is then the truck's own accel
    onset_jerk = self.approach_onset_jerk(sm['radarState'].leadOne, v_ego, a_prev)
    if onset_jerk is not None and not self.fcw and self.onset_hold == 0 and \
       output_a_target >= -(self.approach_need + APPROACH_ONSET_OTHER_MARGIN):
      output_a_target = max(output_a_target, a_prev - onset_jerk * self.dt)
    if reset_state:
      self.onset_hold = int(round(APPROACH_ONSET_RESET_HOLD / self.dt))
    elif self.onset_hold > 0:
      self.onset_hold = 0 if output_a_target <= 0. else self.onset_hold - 1
    self.output_a_target = np.clip(output_a_target, ACCEL_MIN, ACCEL_MAX)

    self.v_desired_filter.x = self.v_desired_filter.x + self.dt * (self.output_a_target + a_prev) / 2.0

  def publish(self, sm, pm):
    plan_send = messaging.new_message('longitudinalPlan')

    plan_send.valid = sm.all_checks()

    longitudinalPlan = plan_send.longitudinalPlan
    longitudinalPlan.modelMonoTime = sm.logMonoTime['modelV2']
    longitudinalPlan.processingDelay = (plan_send.logMonoTime / 1e9) - sm.logMonoTime['modelV2']
    longitudinalPlan.solverExecutionTime = self.mpc.solve_time

    longitudinalPlan.speeds = self.v_desired_trajectory.tolist()
    longitudinalPlan.accels = self.a_desired_trajectory.tolist()
    longitudinalPlan.jerks = self.j_desired_trajectory.tolist()

    longitudinalPlan.hasLead = sm['radarState'].leadOne.present
    longitudinalPlan.longitudinalPlanSource = self.mpc.source
    longitudinalPlan.fcw = self.fcw

    longitudinalPlan.aTarget = float(self.output_a_target)
    longitudinalPlan.shouldStop = bool(self.output_should_stop)
    longitudinalPlan.allowBrake = True
    longitudinalPlan.allowThrottle = bool(self.allow_throttle)

    pm.send('longitudinalPlan', plan_send)

    self.publish_longitudinal_plan_sp(sm, pm)
