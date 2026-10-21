"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.launch_assist import LEAD_DEPART_V_K

# Stop behind a stopped car without the crawl, about GAP_MAX back. The lead planner's gap shrinks with speed
# (t_follow * v + 6 m), so at walking pace behind a car that has stopped it asks for almost no braking and rolls up to the
# 6 m standstill gap; at such small requests the truck can hold its creep speed (routes 0000001d/1e: 12 m -> 3 m at 1.3 m/s
# for 5 s; 0.4 m/s for 8 s). Two parts, both only for a radar lead:
# - Below walking pace behind a stopped lead, a braking candidate (the lowest still wins) stops the truck at a constant
#   A_STOP, aiming where that ends, kept between GAP_MIN and GAP_MAX, and holds it until the lead pulls away.
# - Behind a lead that has stopped, or is slower than PROFILE_STOPPING_V and braking, below PROFILE_V_MAX: one even stop
#   at GAP_MAX behind where that car stops (assuming it brakes at least as hard as the lead planner assumes, 2.5 m/s^2),
#   used as a floor on the lead planner's own braking, which otherwise slowed the truck to a walk 10-15 m back and
#   trickled in (route 00000024: stops 7.3 / 8.5 / 9.0 m back, the driver twice closing to ~5 m with the gas). Only while
#   that stop needs at most PROFILE_A_MAX: firmer, the lead planner as is.
V_ENGAGE = 1.6             # m/s: walking pace. Below this behind a stopped lead, stop rather than crawl
V_RELEASE = 2.5            # m/s: faster than this (the driver's gas, or the lead planner pulling away) lets go
LEAD_STOPPED_V = 0.1       # m/s vLeadK: the lead counts as stopped below this (standstill radar noise is <= 0.05)
A_STOP = 0.5               # m/s^2: the stop's decel, about what the truck's own brake hold feels like
A_MAX = 1.5                # m/s^2: never asks for more than this, the lead planner's own braking can still go further
GAP_MIN = 3.5              # m: never aims closer than this to the lead (radar dRel, from the bumper)
GAP_MAX = 5.0              # m: the stopping gap (the driver closed to 4.5-5.3 m by hand on route 00000024)
GAP_MAX_TRAFFIC = 4.0      # m: with Traffic Mode's shorter gaps
PROFILE_V_MAX = 10.        # m/s: the even stop only below this
PROFILE_A_MAX = 1.5        # m/s^2: and only while it needs at most this
PROFILE_STOPPING_V = 5.    # m/s: a lead slower than this
PROFILE_STOPPING_A = -0.3  # m/s^2: and braking harder than this counts as stopping
PROFILE_LEAD_DECEL = 2.5   # m/s^2: where a stopping lead stops, assuming it brakes at least this hard (COMFORT_BRAKE)
PROFILE_RELIEF_JERK = 1.5  # m/s^3: easing off toward the even stop, the braking goes no faster than this (in all, with
                           # the lead planner's own easing off)
PROFILE_RETURN_JERK = 5.0  # m/s^3: and comes back this fast once the even stop no longer applies
ENGAGE_MARGIN = 2.0        # m: starts once the stop at A_STOP would end within GAP_MAX + this
RISE_RATE = 1.0            # m/s^3: the braking builds no faster than this, so it does not step in
STOP_V = 0.3               # m/s: below this the stop is held (should_stop), as long as the lead stays stopped
RELEASE_MARGIN = 3.0       # m: lets go once the stop would end this far past where it would have started, e.g. radard
                           # switching to a car further up the queue (the one ahead changed lanes), so it moves up
DROPOUT_FRAMES = 5         # a radar lead missing for up to this many frames (0.25 s) keeps the braking as it was
SAME_CAR_DIST = 2.0        # m: a new radar track this close to the last one's distance is the same car (track churn)


class StopBehindLead:
  def __init__(self, CP):
    self.available = bool(CP.openpilotLongitudinalControl)
    self.active = False
    self.accel = 0.          # m/s^2, the braking currently asked for (<= 0)
    self.gap = 0.            # m, where it is stopping
    self.track_id = -1
    self.last_d = 0.
    self.profile_on = False
    self.relief = 0.         # m/s^2, how much of the lead planner's braking is currently eased off (>= 0)
    self.relaxed: float | None = None   # m/s^2, the last accel relax() returned
    self.dropout = 0

  def profile_floor(self, v_ego: float, lead, long_off: bool = False, traffic_mode: bool = False) -> float | None:
    """The braking an even stop at gap_max behind a stopped or stopping radar lead needs (<= 0), or None."""
    if long_off or not self.available or not (lead.present and lead.radar) or not math.isfinite(v_ego) or \
       not math.isfinite(lead.dRel):
      self.profile_on = False
      return None
    stopped = lead.vLeadK < (LEAD_DEPART_V_K if self.profile_on else LEAD_STOPPED_V)
    stopping = lead.vLeadK < PROFILE_STOPPING_V and lead.aLeadK < PROFILE_STOPPING_A
    gap_max = GAP_MAX_TRAFFIC if traffic_mode else GAP_MAX
    lead_stop = max(lead.vLeadK, 0.) ** 2 / (2. * max(-lead.aLeadK, PROFILE_LEAD_DECEL)) if stopping else 0.
    room = lead.dRel + lead_stop - gap_max
    stopped = stopped or stopping
    if not stopped or v_ego > PROFILE_V_MAX or room <= 0.5:
      self.profile_on = False
      return None
    need = max(v_ego, 0.) ** 2 / (2. * room)
    if need > PROFILE_A_MAX:
      self.profile_on = False
      return None
    self.profile_on = True
    return -need

  def relax(self, a_plan: float, floor: float | None, a_prev: float, off: bool = False) -> float:
    """The lead planner's accel a_plan with its braking beyond the floor eased off, never past the floor. Easing off, the
    result rises no faster than PROFILE_RELIEF_JERK above the last output a_prev (the floor can appear in one frame: the
    lead slowing below PROFILE_STOPPING_V; and while something else brakes harder this candidate must not drift up out of
    sight); with no floor the relief goes away at PROFILE_RETURN_JERK, and at once when off (FCW, long off)."""
    if off:
      self.relief = 0.
    elif floor is not None:
      last = a_plan if self.relaxed is None else min(self.relaxed, a_prev)
      self.relief = max(min(floor, last + PROFILE_RELIEF_JERK * DT_MDL) - a_plan, 0.)
    else:
      self.relief = max(self.relief - PROFILE_RETURN_JERK * DT_MDL, 0.)
    self.relaxed = a_plan + self.relief
    return self.relaxed

  def reset(self) -> None:
    self.active = False
    self.accel = 0.
    self.dropout = 0

  def update(self, v_ego: float, lead, long_off: bool = False, traffic_mode: bool = False) -> tuple[float, bool] | None:
    """Returns an (accel, should_stop) planner candidate, or None when there is nothing to add."""
    radar_lead = bool(lead.present and lead.radar)
    if self.active and not long_off and not radar_lead and self.dropout < DROPOUT_FRAMES and math.isfinite(v_ego) and \
       v_ego <= V_RELEASE:
      # a radar dropout of a frame or two: keep the braking as it was rather than drop it and build it up again
      self.dropout += 1
      return float(self.accel), v_ego < STOP_V
    if long_off or not self.available or not radar_lead or not math.isfinite(v_ego) or not math.isfinite(lead.dRel):
      self.reset()
      return None
    self.dropout = 0

    v_ego = max(v_ego, 0.)
    d = float(lead.dRel)
    gap_max = GAP_MAX_TRAFFIC if traffic_mode else GAP_MAX
    stop_dist = v_ego ** 2 / (2. * A_STOP)   # how far the truck goes stopping at A_STOP from here

    if self.active:
      if lead.radarTrackId != self.track_id and abs(d - self.last_d) < SAME_CAR_DIST:
        self.track_id = lead.radarTrackId   # the same car on a new track
      # the lead pulls away (Launch Assist's own gate, so a launch is never held up), the truck speeds up, or the lead
      # is now a different car, or the same track much further out: let the lead planner move up (re-engages near it)
      if lead.vLeadK > LEAD_DEPART_V_K or v_ego > V_RELEASE or lead.radarTrackId != self.track_id or \
         d - stop_dist > gap_max + ENGAGE_MARGIN + RELEASE_MARGIN:
        self.reset()
        return None
    elif v_ego < V_ENGAGE and lead.vLeadK < LEAD_STOPPED_V and d - stop_dist < gap_max + ENGAGE_MARGIN:
      self.active = True
      self.accel = 0.
      self.track_id = lead.radarTrackId
    else:
      return None

    # stop where A_STOP would stop it, kept between GAP_MIN and gap_max: further out it brakes less (the lead planner
    # closes up), too close it brakes more
    self.last_d = d
    self.gap = min(max(d - stop_dist, GAP_MIN), gap_max)
    target = -min(v_ego ** 2 / (2. * max(d - self.gap, 0.2)), A_MAX) if v_ego > 0. else 0.
    self.accel = max(target, self.accel - RISE_RATE * DT_MDL)
    return float(self.accel), v_ego < STOP_V
