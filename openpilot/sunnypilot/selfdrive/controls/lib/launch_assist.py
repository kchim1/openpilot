"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

import numpy as np

from opendbc.car.structs import car
from openpilot.cereal import log
from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog

LongCtrlState = car.CarControl.Actuators.LongControlState
Personality = log.LongitudinalPersonality

# Rivian Launch Assist: the start of a launch from a stop. The truck itself is quick (the VDM drops its brake hold and
# propulsion follows ~0.3 s after the first positive request, then the request with ~0.1 s lag, at the IMU), but the plan
# behind a car pulling away from a light rises slowly: the lead planner follows radard's aLeadK, which lags such a car by
# ~0.5 s, and the end-to-end target ramps at ~0.6 m/s^3. On route 00000028 the truck pulled at 0.45-0.72 m/s^2 for the
# first 1.5 s and took 1.9 s from the release to 1 m/s (the driver's own launches: 1.2-2.0 m/s^2, 1 m/s 1.1 s after the gas).
# While a launch is assisted the request is held at least at a floor:
# - behind a stopped radar lead that measurably pulls away: as hard as it pulls away (the larger of aLeadK and the rise of
#   its raw speed over LEAD_SLOPE_T), between LAUNCH_ACCEL_MIN and the personality's LAUNCH_ACCEL_MAX. The floor may only
#   rise during LAUNCH_RISE_T after the fire, follows a lead that eases off down at LAUNCH_DOWN_RATE, and stops once the
#   lead's speed stops rising. If that radar track changes or is lost, it fades out: another car never inherits the floor.
# - with no lead at all: LAUNCH_NO_LEAD by personality, but never more than the driving model's own plan averages over the
#   next NO_LEAD_PLAN_T (a planned creep stays a creep).
# It hands back as soon as the planner's own target reaches the floor, or after LAUNCH_MAX_T, fading at FADE_RATE.
# RivianLaunchAssistBoost sets the caps: Strong (the default) or Gentle, 0.1 m/s^2 lower.
# A lift behind a car hands back once the truck reads moving (LEAD_MOVING_V_RAW, as v1 did): held on as the truck
# picked up speed, it ended a queue that creeps and stops up to 1.6 m closer than today in sim. No lift behind a car in
# Traffic Mode, whose own launch shift already moves it off sooner, and none after a stop closer than LAUNCH_LEAD_GAP_MIN
# behind it: in an inching queue either one ended each stop a little closer than the last, down to contact in sim. A
# launch with no car ahead is lifted in Traffic Mode too.
ARM_STANDSTILL_T = 1.0      # s stopped (behind the same radar lead, or with no lead) before a launch may be assisted
STRENGTH_GENTLE = 0         # RivianLaunchAssistBoost values
STRENGTH_STRONG = 1
STRENGTH_NAMES = ("gentle", "strong")
LAUNCH_ACCEL_MIN = 0.5      # m/s^2 floor behind a lead pulling away slowly (v1's fixed floor)
# m/s^2 floor caps by strength, for Aggressive, Standard, Relaxed: behind a lead pulling away briskly, and with no lead
LAUNCH_ACCEL_MAX = {STRENGTH_STRONG: (0.8, 0.7, 0.6), STRENGTH_GENTLE: (0.7, 0.6, 0.5)}
LAUNCH_NO_LEAD = {STRENGTH_STRONG: (0.6, 0.5, 0.4), STRENGTH_GENTLE: (0.5, 0.4, 0.3)}
LAUNCH_RAMP_RATE = 5.0      # m/s^3 floor rise from the normal output (~0.1 s to 0.5), not a one-frame step
LAUNCH_DOWN_RATE = 1.0      # m/s^3 the floor follows a lead that eases off
LAUNCH_RISE_T = 1.0         # s after the fire the floor may still rise: only the start of the launch is lifted
LAUNCH_MAX_T = 2.0          # s at most from the fire to the fade
LEAD_SLOPE_T = 0.5          # s window for the rise of the lead's raw speed (aLeadK lags a car pulling away by ~0.5 s)
NO_LEAD_PLAN_T = 1.0        # s: with no lead, at most the model plan's average accel over this long
LEAD_EASED_SLOPE = -0.1     # m/s^2: the lead's raw speed falling at least this fast: it stopped pulling away
LAUNCH_LEAD_GAP_MIN = 3.5   # m: no lift after a stop closer than this to the car (Stop Behind Lead's GAP_MIN)
LEAD_MOVING_V_RAW = 0.15    # m/s vEgoRaw: a lift behind a car hands back once the truck reads this fast
FADE_RATE = 1.0             # m/s^3 floor fade back to the normal output, rather than dropping at once
STOP_PULSE_T = 0.1          # s, 2 planner frames: a shorter planner stop while the lead clearly pulls away only pauses it

# Departing-lead gate. Standstill radar noise measured std <= 0.05 m/s on vLead, |aLeadK| <= 0.12; vLeadK of a stationary
# lead stays <= 0.02 in real Rivian logs. A higher threshold delays real launches (real logs: lead start to gate median
# 0.45 s at 0.15, 0.60 s at 0.2)
LEAD_DEPART_V_K = 0.15      # primary: radarState.leadOne.vLeadK above this
LEAD_DEPART_ALT_V = 0.3     # alternative: vLead above this and aLeadK above LEAD_DEPART_ALT_A
LEAD_DEPART_ALT_A = 0.1
LEAD_DEPART_GATE_ALT = False

PARAM_READ_FRAMES = 100     # controlsd runs at 100 Hz: re-read the toggle and the strength every 1 s
TRAFFIC_MODE_READ_FRAMES = 25  # and Traffic Mode every 0.25 s, as plannerd does
ARM_FRAMES = int(round(ARM_STANDSTILL_T / DT_CTRL))
LAUNCH_MAX_FRAMES = int(round(LAUNCH_MAX_T / DT_CTRL))
LAUNCH_RISE_FRAMES = int(round(LAUNCH_RISE_T / DT_CTRL))
LEAD_SLOPE_FRAMES = int(round(LEAD_SLOPE_T / DT_CTRL))
STOP_PULSE_FRAMES = int(round(STOP_PULSE_T / DT_CTRL))
_SLOPE_T = np.arange(LEAD_SLOPE_FRAMES + 1) * DT_CTRL
_SLOPE_W = (_SLOPE_T - _SLOPE_T.mean()) / np.sum((_SLOPE_T - _SLOPE_T.mean()) ** 2)  # least-squares slope weights


class LaunchAssistState(IntEnum):
  idle = 0
  armed = 1
  firing = 2
  fading = 3


def is_radar_lead(lead) -> bool:
  return bool(lead.present and lead.radar)


def lead_departing(lead) -> bool:
  if LEAD_DEPART_GATE_ALT:
    return lead.vLead > LEAD_DEPART_ALT_V and lead.aLeadK > LEAD_DEPART_ALT_A
  return lead.vLeadK > LEAD_DEPART_V_K


def by_personality(values, personality) -> float:
  # compared like get_T_FOLLOW does: selfdriveState.personality is a capnp enum, not a dict key
  if personality == Personality.aggressive:
    return values[0]
  elif personality == Personality.relaxed:
    return values[2]
  return values[1]


def model_plan_accel(model, t: float) -> float | None:
  """The driving model plan's average accel over the next t s, or None without a full, finite plan."""
  if model is None:
    return None
  vx, tx = model.velocity.x, model.velocity.t
  if len(vx) < 2 or len(vx) != len(tx):
    return None
  accel = float((np.interp(t, tx, vx) - vx[0]) / t)
  # NaN would pass min(cap, max(plan, 0.)) as the cap
  return accel if np.isfinite(accel) else None


class LaunchAssist:
  def __init__(self, params):
    self.params = params
    self.enabled = False
    self.strength = STRENGTH_STRONG
    self.traffic_mode = False
    self._frame = 0
    self._error_logged = False
    self._strength_error_logged = False
    self._traffic_mode_error_logged = False

    self.state = LaunchAssistState.idle
    self.standstill_frames = 0  # stopped behind the same radar lead (or with no lead) with a launch qualifying
    self.lead_id = -1
    self.stop_gap = None        # the closest the truck has come to that radar lead during this standstill (m)
    self.spent = False          # fired during this standstill
    self.rearm_frames = 0       # after the planner stopped an assisted launch at a standstill: frames it has held the stop
    self.fire_frames = 0
    self.floor = 0.
    self.lead_mode = True       # this launch is behind a radar lead (False: no lead at all)
    self.fire_lead_id = -1      # the radar track a lead launch fired behind
    self.pause_frames = 0       # frames of a planner stop that so far only pauses the launch
    self.resume = None          # after a pause: the floor climbing back from the planner's output
    self.fade_reason = ""
    self.v_lead_hist: list[float] = []  # raw vLead of the current radar track, last LEAD_SLOPE_FRAMES + 1 frames

  @staticmethod
  def available(CP) -> bool:
    return CP.brand == 'rivian' and CP.openpilotLongitudinalControl

  def read_param(self) -> bool:
    try:
      return self.params.get_bool("RivianLaunchAssist")
    except Exception:
      # params library built without the key, or a failed read: off rather than take controlsd down
      if not self._error_logged:
        cloudlog.exception("RivianLaunchAssist read failed, treating it as off")
        self._error_logged = True
      return False

  def read_strength(self) -> int:
    try:
      gentle = self.params.get("RivianLaunchAssistBoost", return_default=True) == STRENGTH_GENTLE
      return STRENGTH_GENTLE if gentle else STRENGTH_STRONG
    except Exception:
      # params library built without the key, or a failed read: Strong, the default
      if not self._strength_error_logged:
        cloudlog.exception("RivianLaunchAssistBoost read failed, using Strong")
        self._strength_error_logged = True
      return STRENGTH_STRONG

  def read_traffic_mode(self) -> bool:
    try:
      return self.params.get_bool("TrafficMode")
    except Exception:
      # a failed read: on, so no lead lift on top of a Traffic Mode plannerd may have on, rather than take controlsd down
      if not self._traffic_mode_error_logged:
        cloudlog.exception("TrafficMode read failed, treating it as on")
        self._traffic_mode_error_logged = True
      return True

  def update_param(self) -> None:
    if self._frame % PARAM_READ_FRAMES == 0:
      self.enabled = self.read_param()
      self.strength = self.read_strength()
    if self._frame % TRAFFIC_MODE_READ_FRAMES == 0:
      self.traffic_mode = self.read_traffic_mode()
    self._frame += 1

  def lead_lift_allowed(self) -> bool:
    return not self.traffic_mode and self.stop_gap is not None and self.stop_gap >= LAUNCH_LEAD_GAP_MIN

  def reset(self) -> None:
    self.state = LaunchAssistState.idle
    self.standstill_frames = 0
    self.floor = 0.
    self.pause_frames = 0
    self.resume = None
    self.fade_reason = ""

  def _log_end(self, reason, CS, lead) -> None:
    cloudlog.event("launch assist end", reason=reason, fade=self.fade_reason, fire_frames=self.fire_frames, v_ego=float(CS.vEgo),
                   v_lead_k=float(lead.vLeadK), d_rel=float(lead.dRel), lead=self.lead_mode, strength=STRENGTH_NAMES[self.strength],
                   floor=float(self.floor))

  def lead_slope(self) -> float | None:
    """How fast the lead's raw speed rose over the last LEAD_SLOPE_T: a least-squares slope, so one noisy radar sample
    (std ~0.05 m/s at a standstill) moves it by little"""
    h = self.v_lead_hist
    if len(h) != LEAD_SLOPE_FRAMES + 1:
      return None
    return float(np.dot(_SLOPE_W, h))

  def update(self, long_active, CS, lead, long_control_state, a_target, should_stop, accel, accel_limits, model=None,
             personality=Personality.aggressive):
    """accel: LongControl's output this frame. Returns it, or a higher floor while a launch is assisted."""
    self.update_param()

    radar_lead = is_radar_lead(lead)
    no_lead = not lead.present
    # the planner owns the launch: never act in stopping/off, with shouldStop, or when it asks to slow down
    released = long_control_state == LongCtrlState.pid and not should_stop and a_target >= 0.
    engaged = self.enabled and long_active and not CS.brakePressed
    usable = engaged and (radar_lead or no_lead)

    # a new radar track restarts the second (its first vLeadK is unfiltered) and the speed history
    same_lead = lead.radarTrackId == self.lead_id if radar_lead else no_lead and self.lead_id == -1
    self.lead_id = lead.radarTrackId if radar_lead else -1
    if radar_lead:
      if not same_lead:
        self.v_lead_hist = []
      self.v_lead_hist.append(float(lead.vLead))
      del self.v_lead_hist[:-LEAD_SLOPE_FRAMES - 1]
    else:
      self.v_lead_hist = []

    if CS.standstill and radar_lead:
      self.stop_gap = min(self.stop_gap, float(lead.dRel)) if same_lead and self.stop_gap is not None else float(lead.dRel)
    elif not CS.standstill:
      self.stop_gap = None

    if CS.standstill:
      self.standstill_frames = (self.standstill_frames + 1 if same_lead else 1) if usable else 0
      # an assisted launch the planner stopped again before the truck showed any speed (a lead that crept and stopped):
      # once the planner has held the stop for a full second, a later launch from this standstill may be assisted
      if self.spent and self.rearm_frames > 0:
        self.rearm_frames = self.rearm_frames + 1 if should_stop else 1
        if self.rearm_frames > ARM_FRAMES:
          self.spent = False
          self.rearm_frames = 0
    else:
      self.standstill_frames = 0
      self.spent = False
      self.rearm_frames = 0

    if self.state in (LaunchAssistState.firing, LaunchAssistState.fading):
      # a lead launch keeps to the radar track it fired behind
      fired_lead = self.lead_mode and radar_lead and lead.radarTrackId == self.fire_lead_id
      if engaged and should_stop and fired_lead and lead_departing(lead) and self.pause_frames + 1 < STOP_PULSE_FRAMES:
        # the planner stops while that car clearly pulls away: its stop holds the truck (nothing is added), and one
        # shorter than STOP_PULSE_T only pauses the launch, which then climbs back from the planner's output
        self.pause_frames += 1
        self.resume = accel
        return accel
      if not (engaged and released and (self.lead_mode or no_lead)) or (self.lead_mode and self.traffic_mode):
        # (Traffic Mode turned on during a lift behind a car: its launch shift takes over)
        cloudlog.event("launch assist cancel", enabled=self.enabled, long_active=bool(long_active), brake=bool(CS.brakePressed),
                       radar_lead=radar_lead, no_lead=no_lead, lead=self.lead_mode, long_control_state=int(long_control_state),
                       should_stop=bool(should_stop), a_target=float(a_target), fire_frames=self.fire_frames,
                       pause_frames=self.pause_frames, fade=self.fade_reason, strength=STRENGTH_NAMES[self.strength],
                       traffic_mode=self.traffic_mode)
        self.reset()
        if CS.standstill and should_stop:
          self.rearm_frames = 1
      else:
        self.pause_frames = 0
        if self.lead_mode and not fired_lead and self.state == LaunchAssistState.firing:
          # that car is gone or another radar track took its place: hand back with the fade, no other car inherits the floor
          self.state = LaunchAssistState.fading
          self.fade_reason = "lead changed" if lead.present else "lead lost"
        if lead.present and CS.vEgo >= lead.vLeadK:
          # already as fast as the car ahead: more would only close the gap (lead inching or crawling, then stopping)
          self._log_end("cap", CS, lead)
          self.reset()

    elif self.state == LaunchAssistState.armed:
      if self.standstill_frames < ARM_FRAMES:
        # moved, lead lost or changed, not engaged, braking
        self.reset()
      elif released and ((radar_lead and lead_departing(lead) and self.lead_lift_allowed()) or (no_lead and a_target > 0.)):
        cap = by_personality((LAUNCH_ACCEL_MAX if radar_lead else LAUNCH_NO_LEAD)[self.strength], personality)
        cloudlog.event("launch assist fire", lead=radar_lead, strength=STRENGTH_NAMES[self.strength], cap=float(cap),
                       track=int(lead.radarTrackId), v_lead_k=float(lead.vLeadK), v_lead=float(lead.vLead),
                       a_lead_k=float(lead.aLeadK), d_rel=float(lead.dRel), accel=float(accel))
        self.state = LaunchAssistState.firing
        self.lead_mode = radar_lead
        self.fire_lead_id = lead.radarTrackId if radar_lead else -1
        self.spent = True
        self.fire_frames = 0
        self.floor = accel

    elif self.state == LaunchAssistState.idle:
      # arms in stopping or, after a long-gap stop the planner already released, in pid
      if not self.spent and self.standstill_frames >= ARM_FRAMES:
        self.state = LaunchAssistState.armed

    if self.state == LaunchAssistState.firing:
      slope = self.lead_slope()
      if self.lead_mode:
        a_max = by_personality(LAUNCH_ACCEL_MAX[self.strength], personality)
        target = min(max(lead.aLeadK, slope if slope is not None else 0., LAUNCH_ACCEL_MIN), a_max)
      else:
        target = by_personality(LAUNCH_NO_LEAD[self.strength], personality)
        plan = model_plan_accel(model, NO_LEAD_PLAN_T)
        target = min(target, max(plan, 0.)) if plan is not None else 0.
      if self.fire_frames >= LAUNCH_RISE_FRAMES:
        target = min(target, self.floor)  # only the start is lifted: later the floor may only come down
      if self.floor < target:
        self.floor = min(self.floor + LAUNCH_RAMP_RATE * DT_CTRL, target)
      else:
        self.floor = max(self.floor - LAUNCH_DOWN_RATE * DT_CTRL, target)
      self.fire_frames += 1
      caught_up = self.floor >= target - 1e-6 and a_target >= self.floor
      eased = self.lead_mode and slope is not None and slope < LEAD_EASED_SLOPE
      moving = self.lead_mode and CS.vEgoRaw > LEAD_MOVING_V_RAW
      if self.fire_frames >= LAUNCH_MAX_FRAMES or caught_up or eased or moving:
        # out of time, the planner caught up, the lead stopped pulling away, or behind a car the truck is under way: the
        # planner takes it from here
        self.state = LaunchAssistState.fading
        self.fade_reason = "planner" if caught_up else ("lead eased" if eased else ("moving" if moving else "time"))
    elif self.state == LaunchAssistState.fading:
      self.floor -= FADE_RATE * DT_CTRL

    if self.state not in (LaunchAssistState.firing, LaunchAssistState.fading):
      return accel

    floor = self.floor
    if self.resume is not None:
      # after a pause: back up to the floor from the planner's output at the ramp rate, not in one step
      self.resume += LAUNCH_RAMP_RATE * DT_CTRL
      if self.resume < floor:
        floor = self.resume
      else:
        self.resume = None
    floor = min(max(floor, accel_limits[0]), accel_limits[1])
    if self.state == LaunchAssistState.fading and floor <= accel:
      # back at the normal output: done until the next standstill
      self._log_end("fade", CS, lead)
      self.reset()
      return accel
    return max(accel, floor)
