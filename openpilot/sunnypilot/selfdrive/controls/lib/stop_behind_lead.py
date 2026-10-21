"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

from openpilot.common.realtime import DT_MDL
from openpilot.sunnypilot.selfdrive.controls.lib.launch_assist import LEAD_DEPART_V_K

# Stop behind a stopped car without the crawl. The lead planner's gap shrinks with speed (t_follow * v + 6 m), so once the
# truck is down to walking pace behind a car that has stopped, it keeps asking for almost no braking and rolls up to the
# 6 m standstill gap at walking pace, and at such small requests the truck holds its creep speed: on route 0000001d it
# rolled 12 m -> 3 m at ~1.3 m/s for 5 s, on 0000001e it hovered at ~0.4 m/s for 8 s. Here, below walking pace behind a
# stopped radar lead, the truck comes to a stop at a gentle constant decel instead, near where the crawl would have begun.
# Adds braking on top of the planner's, never removes any.
V_ENGAGE = 1.6             # m/s: walking pace. Below this behind a stopped lead, stop rather than crawl
V_RELEASE = 2.5            # m/s: faster than this (the driver's gas, or the lead planner pulling away) lets go
LEAD_STOPPED_V = 0.1       # m/s vLeadK: the lead counts as stopped below this (standstill radar noise is <= 0.05)
A_STOP = 0.5               # m/s^2: the stop's decel, about what the truck's own brake hold feels like
A_MAX = 1.5                # m/s^2: never asks for more than this, the lead planner's own braking can still go further
GAP_MIN = 4.0              # m: never aims closer than this to the lead (radar dRel, from the bumper)
GAP_MAX = 9.0              # m: and not farther back than this: further out, it lets the lead planner close up first
GAP_MAX_TRAFFIC = 7.0      # m: with Traffic Mode's shorter gaps
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
    self.dropout = 0

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
