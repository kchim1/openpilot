"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""

from openpilot.cereal import messaging, custom, log
from opendbc.car import structs
from openpilot.common.constants import CV
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.car.cruise import V_CRUISE_MAX
import numpy as np

from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import COMFORT_BRAKE, STOP_DISTANCE, clip_t_follow_offset_pct, \
  get_safe_obstacle_distance, get_stopped_equivalence_factor
from openpilot.sunnypilot.selfdrive.controls.lib import follow_distance
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController
from openpilot.sunnypilot.selfdrive.controls.lib.e2e_alerts_helper import E2EAlertsHelper
from openpilot.sunnypilot.selfdrive.controls.lib.fast_lead_accel import FastLeadAccel
from openpilot.sunnypilot.selfdrive.controls.lib.red_light_stop import RedLightStop
from openpilot.sunnypilot.selfdrive.controls.lib.stop_behind_lead import StopBehindLead
from openpilot.sunnypilot.selfdrive.controls.lib.smart_cruise_control.smart_cruise_control import SmartCruiseControl
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_assist import SpeedLimitAssist
from openpilot.sunnypilot.selfdrive.controls.lib.speed_limit.speed_limit_resolver import SpeedLimitResolver
from openpilot.sunnypilot.selfdrive.selfdrived.events import EventsSP
from openpilot.sunnypilot.models.helpers import get_active_bundle

DecState = custom.LongitudinalPlanSP.DynamicExperimentalControl.DynamicExperimentalControlState
LongitudinalPlanSource = custom.LongitudinalPlanSP.LongitudinalPlanSource

# plannerd runs at 20 Hz: re-read TrafficMode every 0.25 s so it can be switched while driving
TRAFFIC_MODE_READ_FRAMES = 5
# and RivianLaunchAssist every 1 s
LAUNCH_ASSIST_READ_FRAMES = 20
# Keep Up With Traffic: re-read every 1 s. Raise the cruise accel cap while a moving radar-tracked lead is further away
# than the lead planner wants it: none at the desired gap (so a lead that brakes right after pulling away finds the
# truck no closer than stock), all of it from KEEP_PACE_EXCESS_BP[1] past, and none once the lead is too far ahead to
# matter. Kept KEEP_PACE_OFF_FRAMES after the lead track drops so a flickering track does not toggle it
KEEP_PACE_READ_FRAMES = 20
KEEP_PACE_V_MIN = 1.0      # m/s
KEEP_PACE_EXCESS_BP = [0., 10., 30., 40.]  # m past the desired following distance
KEEP_PACE_EXCESS_V = [0., 1., 1., 0.]
# and half of it by highway speed: in sim, a lead that pulls away and then brakes at -4.5 m/s^2 left 3.0 m at 15 m/s with
# Traffic Mode (8.2 m stock), 4.8 m with this fade, while launches keep nearly all the gain (peak gap 79 -> 38 m vs 35 m)
KEEP_PACE_SPEED_BP = [10., 20.]  # m/s
KEEP_PACE_SPEED_V = [1., 0.5]
KEEP_PACE_OFF_FRAMES = 10  # 0.5 s at 20 Hz
# share of the step from the stock cap up to the follow cap, per driving personality
KEEP_PACE_SHARE_AGGRESSIVE = 1.0
KEEP_PACE_SHARE_STANDARD = 0.5
KEEP_PACE_SHARE_RELAXED = 0.25

# Brake Sooner for Stops, in the lead planner too (radar cruise or end to end, DEC on or off): a stopped car the radar
# picks up far ahead used to bring an abrupt swing from the cruise throttle into braking. Route 0000001d 7:36: 133 m out
# from 40 mph while accelerating at +1 m/s^2, the request fell to -0.3 in one frame (3.2 m/s^3 felt over half a second)
# and peaked at -1.8 (2.1 felt).
# - Closing on a stopped or slow car, the lead planner plans as if its comfort brake were this instead of 2.5, so braking
#   starts sooner and peaks lower (long_mpc.py approach_margin, which keeps it to cars at 8 m/s or less and to stops that
#   can still be gentle).
BRAKE_SOONER_COMFORT_BRAKE = 1.5
# - And the start of braking eases in at APPROACH_ONSET_JERK while the constant decel that stops STOP_DISTANCE short of
#   where the car would stop is under the first APPROACH_ONSET_NEED_BP, the limit rising to APPROACH_ONSET_JERK_URGENT
#   toward the second, and from the second on there is no limit at all: a car seen late gets the stock braking at once.
#   Only the start: from the first APPROACH_ONSET_A_BP to the second the limit rises too, and from the second there is
#   none, so it lets go without a step. None behind a car braking harder than APPROACH_ONSET_A_LEAD, within the stop
#   distance already, or on the frame after a reset (gas override, engaging), whose state is the truck's own accel.
#   On 7:36 in closed loop: 1.7 m/s^3 felt at the start instead of 3.2, the peak about the same (2.14 vs 2.11)
APPROACH_ONSET_JERK = 1.5         # m/s^3
APPROACH_ONSET_JERK_URGENT = 5.0  # m/s^3
APPROACH_ONSET_NEED_BP = [1.5, 2.5]
APPROACH_ONSET_A_BP = [-0.5, -1.5]  # m/s^2
APPROACH_ONSET_V_LEAD_MAX = 8.    # m/s
APPROACH_ONSET_OTHER_MARGIN = 0.5 # m/s^2: braking asked for beyond what the lead needs by more than this is for something
                                  # else (the model stopping for a light in front of it, a cut-in): not eased
APPROACH_ONSET_RESET_HOLD = 1.0   # s: after a reset (gas override, engaging) the state is the truck's own accel: no ease-in
                                  # until the plan first gets to 0, or for this long
APPROACH_ONSET_A_LEAD = -2.       # m/s^2


def keep_pace_share(personality) -> float:
  # compared like get_T_FOLLOW does: selfdriveState.personality is a capnp enum, not a dict key
  if personality == log.LongitudinalPersonality.aggressive:
    return KEEP_PACE_SHARE_AGGRESSIVE
  elif personality == log.LongitudinalPersonality.relaxed:
    return KEEP_PACE_SHARE_RELAXED
  return KEEP_PACE_SHARE_STANDARD


class LongitudinalPlannerSP:
  def __init__(self, CP: structs.CarParams, CP_SP: structs.CarParamsSP, mpc):
    self.events_sp = EventsSP()
    self.resolver = SpeedLimitResolver()
    self.dec = DynamicExperimentalController(CP, mpc)
    self.scc = SmartCruiseControl()
    self.resolver = SpeedLimitResolver()
    self.sla = SpeedLimitAssist(CP, CP_SP)
    self.generation = int(model_bundle.generation) if (model_bundle := get_active_bundle()) else None
    self.source = LongitudinalPlanSource.cruise
    self.e2e_alerts_helper = E2EAlertsHelper()

    self.output_v_target = 0.
    self.output_a_target = 0.

    self._mpc = mpc
    self._params = Params()
    self._traffic_mode_available = CP.openpilotLongitudinalControl
    self._traffic_mode_frame = 0
    self._traffic_mode_error_logged = False
    self._launch_assist_available = CP.brand == 'rivian' and CP.openpilotLongitudinalControl
    self._launch_assist_frame = 0
    self._launch_assist_error_logged = False
    self.red_light_stop = RedLightStop(CP, self._params)
    self.approach_need = 0.
    self.stop_behind_lead = StopBehindLead(CP)
    self._keep_pace_available = CP.openpilotLongitudinalControl
    self._keep_pace_enabled = False
    self._keep_pace_read_frame = 0
    self._keep_pace_error_logged = False
    self._keep_pace_frames = 0
    self._keep_pace_share = 0.
    # Rivian Following Distance (the right thumb wheel), in place of the personality's follow time. None everywhere else
    self.follow_distance = follow_distance.FollowDistance(self._params) if follow_distance.available(CP, CP_SP) else None
    # leadOne's acceleration from the radar's raw lead speed, sooner than aLeadK: in Traffic Mode and at Closer
    self.fast_lead_accel = FastLeadAccel()

    # Following distance offset, read once per drive (plannerd restarts on every onroad transition)
    if CP.openpilotLongitudinalControl:
      try:
        mpc.t_follow_offset_pct = clip_t_follow_offset_pct(self._params.get("SPFollowingTimeOffset", return_default=True))
      except UnknownKeyName:
        # params library built without the key: keep the stock gap rather than take plannerd down
        cloudlog.exception("SPFollowingTimeOffset not registered, using 0")
        mpc.t_follow_offset_pct = 0.

  def read_traffic_mode(self) -> bool:
    try:
      return self._params.get_bool("TrafficMode")
    except Exception:
      # params library built without the key, or a failed read: keep the normal gap rather than take plannerd down
      if not self._traffic_mode_error_logged:
        cloudlog.exception("TrafficMode read failed, treating it as off")
        self._traffic_mode_error_logged = True
      return False

  def update_traffic_mode(self) -> None:
    if self._traffic_mode_available and self._traffic_mode_frame % TRAFFIC_MODE_READ_FRAMES == 0:
      self._mpc.traffic_mode = self.read_traffic_mode()
    self._traffic_mode_frame += 1

  def read_launch_assist(self) -> bool:
    try:
      return self._params.get_bool("RivianLaunchAssist")
    except Exception:
      # params library built without the key, or a failed read: off rather than take plannerd down
      if not self._launch_assist_error_logged:
        cloudlog.exception("RivianLaunchAssist read failed, treating it as off")
        self._launch_assist_error_logged = True
      return False

  def update_launch_assist(self) -> None:
    if self._launch_assist_available and self._launch_assist_frame % LAUNCH_ASSIST_READ_FRAMES == 0:
      self._mpc.launch_assist = self.read_launch_assist()
    self._launch_assist_frame += 1

  def read_keep_pace(self) -> bool:
    try:
      return self._params.get_bool("KeepPaceWithTraffic")
    except Exception:
      # params library built without the key, or a failed read: stock caps rather than take plannerd down
      if not self._keep_pace_error_logged:
        cloudlog.exception("KeepPaceWithTraffic read failed, treating it as off")
        self._keep_pace_error_logged = True
      return False

  def update_keep_pace(self) -> None:
    if self._keep_pace_available and self._keep_pace_read_frame % KEEP_PACE_READ_FRAMES == 0:
      self._keep_pace_enabled = self.read_keep_pace()
    self._keep_pace_read_frame += 1

  def keep_pace(self, lead, personality, v_ego: float) -> float:
    """Share of the higher accel cap to use, 0 to 1: with Keep Up With Traffic on and a moving radar-tracked lead, the
    personality's share, scaled by how far the gap is past what the lead planner wants (it runs before this)."""
    if self._keep_pace_enabled and lead.present and lead.radar and lead.vLead > KEEP_PACE_V_MIN:
      desired = get_safe_obstacle_distance(v_ego, self._mpc.t_follow) - get_stopped_equivalence_factor(lead.vLead)
      excess = float(np.interp(lead.dRel - desired, KEEP_PACE_EXCESS_BP, KEEP_PACE_EXCESS_V))
      speed = float(np.interp(v_ego, KEEP_PACE_SPEED_BP, KEEP_PACE_SPEED_V))
      self._keep_pace_share = keep_pace_share(personality) * excess * speed
      self._keep_pace_frames = KEEP_PACE_OFF_FRAMES
    elif self._keep_pace_frames > 0:
      self._keep_pace_frames -= 1
    return self._keep_pace_share if self._keep_pace_frames > 0 else 0.

  def approach_onset_jerk(self, lead, v_ego: float, a_prev: float = 0.) -> float | None:
    """Brake Sooner for Stops: how fast braking may build up from a_prev (m/s^3), or None for no limit."""
    if not self.dec.brake_sooner() or not (lead.present and lead.radar):
      return None
    if lead.vLeadK > APPROACH_ONSET_V_LEAD_MAX or lead.aLeadK < APPROACH_ONSET_A_LEAD:
      return None
    stop_room = lead.dRel + get_stopped_equivalence_factor(max(lead.vLeadK, 0.)) - STOP_DISTANCE
    if stop_room <= 1.:
      return None
    need = v_ego ** 2 / (2. * stop_room)
    self.approach_need = need
    if need >= APPROACH_ONSET_NEED_BP[1] or a_prev <= APPROACH_ONSET_A_BP[1]:
      return None
    jerk = float(np.interp(need, APPROACH_ONSET_NEED_BP, [APPROACH_ONSET_JERK, APPROACH_ONSET_JERK_URGENT]))
    return float(np.interp(a_prev, APPROACH_ONSET_A_BP[::-1], [APPROACH_ONSET_JERK_URGENT, jerk]))

  def is_e2e(self, sm: messaging.SubMaster) -> bool:
    experimental_mode = sm['selfdriveState'].experimentalMode
    if not self.dec.active():
      return experimental_mode

    return experimental_mode and self.dec.mode() == "blended"

  def update_targets(self, sm: messaging.SubMaster, v_ego: float, a_ego: float, v_cruise: float) -> tuple[float, float]:
    CS = sm['carState']
    v_cruise_cluster_kph = min(CS.vCruiseCluster, V_CRUISE_MAX)
    v_cruise_cluster = v_cruise_cluster_kph * CV.KPH_TO_MS

    long_enabled = sm['carControl'].enabled
    long_override = sm['carControl'].cruiseControl.override

    # Smart Cruise Control
    self.scc.update(sm, long_enabled, long_override, v_ego, a_ego, v_cruise)

    # Speed Limit Resolver
    self.resolver.update(v_ego, sm)

    # Speed Limit Assist
    has_speed_limit = self.resolver.speed_limit_valid or self.resolver.speed_limit_last_valid
    self.sla.update(long_enabled, long_override, v_ego, a_ego, v_cruise_cluster, self.resolver.speed_limit,
                    self.resolver.speed_limit_final_last, has_speed_limit, self.resolver.distance, self.events_sp)

    # Curve Speed Control: the feedforward profile, with the reactive lateral-load governor folded in as a tighter speed
    # cap (whichever wants the lower speed). Both return V_CRUISE_UNSET when not constraining.
    curve_v, curve_a = self.scc.curve.output_v_target, self.scc.curve.output_a_target
    if self.scc.governor.output_v_target < curve_v:
      curve_v, curve_a = self.scc.governor.output_v_target, min(curve_a, 0.0)

    targets = {
      LongitudinalPlanSource.cruise: (v_cruise, a_ego),
      LongitudinalPlanSource.sccVision: (self.scc.vision.output_v_target, self.scc.vision.output_a_target),
      LongitudinalPlanSource.sccMap: (self.scc.map.output_v_target, self.scc.map.output_a_target),
      LongitudinalPlanSource.speedLimitAssist: (self.sla.output_v_target, self.sla.output_a_target),
      LongitudinalPlanSource.curveSpeed: (curve_v, curve_a),
    }

    self.source = min(targets, key=lambda k: targets[k][0])
    self.output_v_target, self.output_a_target = targets[self.source]

    # A source wins on the lowest SPEED target, but the a_target handed back becomes the planner's acceleration state:
    # it seeds the MPC and is integrated into v_desired_filter (a_ego here is that state from the last frame, not the
    # measured acceleration). So a source that wins the speed contest while asking to ACCELERATE (SCC-V turning or
    # leaving a turn, up to +0.5 m/s^2, or curve control under its curve speed, up to +1.2, holding a cap a lead has
    # braked the truck below) would inject an acceleration the truck is not making: v_desired_filter drifts above the
    # true speed and the MPC plans braking for a truck going faster than it really is.
    #
    # Every source but cruise is a speed CAP: it may only ever ask for less acceleration, never more. Clamping to the
    # incoming state keeps every cap able to slow the truck (SCC-V's and curve control's own decel still pass through)
    # while leaving the accelerate-out to the MPC and the cruise accel. Cruise, SCC-M and Speed Limit Assist hand back
    # a_ego itself, so this is a no-op for them.
    self.output_a_target = min(self.output_a_target, a_ego)
    return self.output_v_target, self.output_a_target

  def update(self, sm: messaging.SubMaster) -> None:
    self.events_sp.clear()
    self.update_traffic_mode()
    if self.follow_distance is not None:
      self._mpc.follow_t_base = self.follow_distance.update(self._params, sm['carControl'].longActive, DT_MDL)
    close_level = self.follow_distance is not None and self.follow_distance.level in follow_distance.FAST_LEAD_LEVELS
    self._mpc.a_lead_fast = self.fast_lead_accel.update(sm['radarState'].leadOne, sm['carState'].vEgo, sm['carControl'].longActive,
                                                        self._mpc.traffic_mode, close_level)
    self.update_launch_assist()
    self.red_light_stop.update_offset()
    self.update_keep_pace()
    self.dec.update(sm)
    self._mpc.approach_comfort_brake = BRAKE_SOONER_COMFORT_BRAKE if self.dec.brake_sooner() else COMFORT_BRAKE
    self.e2e_alerts_helper.update(sm, self.events_sp)

  def publish_longitudinal_plan_sp(self, sm: messaging.SubMaster, pm: messaging.PubMaster) -> None:
    plan_sp_send = messaging.new_message('longitudinalPlanSP')

    plan_sp_send.valid = sm.all_checks(service_list=['carState', 'controlsState'])

    longitudinalPlanSP = plan_sp_send.longitudinalPlanSP
    longitudinalPlanSP.longitudinalPlanSource = self.source
    longitudinalPlanSP.vTarget = float(self.output_v_target)
    longitudinalPlanSP.aTarget = float(self.output_a_target)
    longitudinalPlanSP.events = self.events_sp.to_msg()

    # Dynamic Experimental Control
    dec = longitudinalPlanSP.dec
    dec.state = DecState.blended if self.dec.mode() == 'blended' else DecState.acc
    dec.enabled = self.dec.enabled()
    dec.active = self.dec.active()

    # Smart Cruise Control
    smartCruiseControl = longitudinalPlanSP.smartCruiseControl
    # Vision Control
    sccVision = smartCruiseControl.vision
    sccVision.state = self.scc.vision.state
    sccVision.vTarget = float(self.scc.vision.output_v_target)
    sccVision.aTarget = float(self.scc.vision.output_a_target)
    sccVision.currentLateralAccel = float(self.scc.vision.current_lat_acc)
    sccVision.maxPredictedLateralAccel = float(self.scc.vision.max_pred_lat_acc)
    sccVision.enabled = self.scc.vision.is_enabled
    sccVision.active = self.scc.vision.is_active
    # Map Control
    sccMap = smartCruiseControl.map
    sccMap.state = self.scc.map.state
    sccMap.vTarget = float(self.scc.map.output_v_target)
    sccMap.aTarget = float(self.scc.map.output_a_target)
    sccMap.enabled = self.scc.map.is_enabled
    sccMap.active = self.scc.map.is_active
    # Curve Speed Control (feedforward profile + reactive lateral-load governor)
    curve = smartCruiseControl.curve
    curve.state = self.scc.curve.state
    curve.vTarget = float(self.scc.curve.output_v_target)
    curve.aTarget = float(self.scc.curve.output_a_target)
    curve.latAccelTarget = float(self.scc.curve.lat_accel_target)
    curve.latAccelMeasured = float(self.scc.governor.load * self.scc.governor.ceiling)
    curve.governorActive = self.scc.governor.is_active
    curve.enabled = self.scc.curve.is_enabled
    curve.active = self.scc.curve.is_active

    # Speed Limit
    speedLimit = longitudinalPlanSP.speedLimit
    resolver = speedLimit.resolver
    resolver.speedLimit = float(self.resolver.speed_limit)
    resolver.speedLimitLast = float(self.resolver.speed_limit_last)
    resolver.speedLimitFinal = float(self.resolver.speed_limit_final)
    resolver.speedLimitFinalLast = float(self.resolver.speed_limit_final_last)
    resolver.speedLimitValid = self.resolver.speed_limit_valid
    resolver.speedLimitLastValid = self.resolver.speed_limit_last_valid
    resolver.speedLimitOffset = float(self.resolver.speed_limit_offset)
    resolver.distToSpeedLimit = float(self.resolver.distance)
    resolver.source = self.resolver.source
    assist = speedLimit.assist
    assist.state = self.sla.state
    assist.enabled = self.sla.is_enabled
    assist.active = self.sla.is_active
    assist.vTarget = float(self.sla.output_v_target)
    assist.aTarget = float(self.sla.output_a_target)

    # E2E Alerts
    e2eAlerts = longitudinalPlanSP.e2eAlerts
    e2eAlerts.greenLightAlert = self.e2e_alerts_helper.green_light_alert
    e2eAlerts.leadDepartAlert = self.e2e_alerts_helper.lead_depart_alert

    # Following distance: what the MPC used this frame
    followDistance = longitudinalPlanSP.followDistance
    followDistance.trafficMode = bool(self._mpc.traffic_mode)
    followDistance.tFollow = float(self._mpc.t_follow)
    followDistance.offsetPct = float(self._mpc.get_t_follow_offset_pct())
    followDistance.jerkFactor = float(self._mpc.jerk_factor)
    followDistance.trafficModeShift = float(self._mpc.launch_shift)
    followDistance.aLeadFast = float(self._mpc.a_lead_fast or 0.)
    if self.follow_distance is not None:
      followDistance.level = self.follow_distance.level
      followDistance.tFollowBase = float(self.follow_distance.t_base)

    pm.send('longitudinalPlanSP', plan_sp_send)
