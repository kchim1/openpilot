"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import deque

import numpy as np

from openpilot.common.realtime import DT_MDL

# Fast lead acceleration: radard's aLeadK, a Kalman filter on the lead's speed, takes about 0.5 to 1 s to show the car ahead
# braking, the main delay in responding to it. The least-squares slope of the radar's raw lead speed (vLead) over the last
# WINDOW planner frames shows it sooner. When that slope is braking harder than A_MIN and than aLeadK, and aLeadK is
# braking too, the MPC plans leadOne with it (down to A_CLIP) instead of aLeadK, kept up longer (long_mpc.py
# FAST_LEAD_A_TAU). The window is a radar lead that has been leadOne, the same track, for all of it, at V_EGO_MIN or faster
# (vLead is the truck's speed plus the radar's relative speed, and the truck's speed has a dead band near a stop). It is
# used where the gap can be shorter than before: in Traffic Mode and at the Closer and Close following distances, at any
# speed (gated at 40 mph in Traffic Mode, it was off just when a car that had sped up past 40 mph braked hard while the
# shorter town gap still held, 0.3-0.9 m closer than today in sim). Braking only: a pulling-away estimate, kept up as long,
# had the plan close in harder on a car that sped up and then braked hard (1.4-1.7 m less room at Closer at 50-60 mph in
# sim), and it rarely reached the plan anyway: radard's leadTwo is mostly leadOne's own track, and its obstacle, on aLeadK,
# won the min(). In sim the braking request comes about 0.3 s sooner behind a car braking hard, and at Closer a -4 m/s^2
# stop from 70 mph leaves 6.5 m instead of 0.3 m
WINDOW = 8                  # planner frames: 0.4 s at 20 Hz
A_MIN = 0.5                 # m/s^2
A_CLIP = -6.                # m/s^2: at most this hard
V_EGO_MIN = 2.0             # m/s

# the least-squares slope of WINDOW samples DT_MDL apart, as a weighted sum
_T = np.arange(WINDOW) * DT_MDL
SLOPE_WEIGHTS = (_T - _T.mean()) / np.sum((_T - _T.mean()) ** 2)


class FastLeadAccel:
  def __init__(self):
    self.v_lead = deque(maxlen=WINDOW)
    self.track_id = -1
    self.a_lead = None  # leadOne's acceleration for the MPC this frame (m/s^2), None: aLeadK

  def update(self, lead, v_ego: float, long_active: bool, traffic_mode: bool, close_level: bool) -> float | None:
    tracking = long_active and lead.present and lead.radar and v_ego >= V_EGO_MIN
    if not tracking or lead.radarTrackId != self.track_id:
      # a new window: off, or another car
      self.v_lead.clear()
      self.track_id = lead.radarTrackId if tracking else -1

    self.a_lead = None
    if tracking:
      # the window fills whatever the mode, so turning Traffic Mode on or stepping to a closer level uses it at once
      self.v_lead.append(lead.vLead)
      if len(self.v_lead) == WINDOW and (traffic_mode or close_level):
        slope = float(np.dot(SLOPE_WEIGHTS, self.v_lead))
        if slope < -A_MIN and lead.aLeadK < 0. and slope < lead.aLeadK:
          self.a_lead = max(slope, A_CLIP)
    return self.a_lead
