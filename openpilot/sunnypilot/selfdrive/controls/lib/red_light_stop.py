"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np

from openpilot.common.filter_simple import FirstOrderFilter
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog

# Red Light Stop Distance: stop this many meters short of where the driving model plans to stop for a red light.
# Only with end to end longitudinal (Experimental Mode, or DEC in blended), the only mode that stops for lights.
# Adds braking on top of the model's own, never removes any. Above LOW_SPEED_V the extra is the difference between the
# constant decel that stops in (d - offset) and the one that stops in d, so it is tiny far out and grows near the line.
STOP_ARM_V = 0.3           # m/s: the model's plan ends (10 s out) below this, so it plans to stop and stay stopped
STOP_RELEASE_V = 1.5       # m/s: before the stop, a plan ending above this disarms (a green light, or no stop after all)
LOW_SPEED_V = 3.0          # m/s: below this, also brake enough to stop at the shortened point, since the model may be
LOW_SPEED_A_MIN = 0.3      # m/s^2  creeping up to its own: once that takes more than this
MIN_TARGET_DIST = 1.0      # m: never aim closer than this
EXTRA_DECEL_MAX = 1.0      # m/s^2: most braking added on top of the model's
A_FLOOR = -2.0             # m/s^2: adds no braking past this, the model's own can still go further
EXTRA_RISE_RATE = 1.5      # m/s^3: the added braking builds no faster than this
EXTRA_FALL_RATE = 3.0      # m/s^3: and goes away no faster than this, rather than dropping at once
STOP_FILTER_TAU = 0.3      # s: the model's stop point is smoothed where it is on the road, so it does not lag the truck

HOLD_V = 0.5               # m/s: below this, hold the stop once within offset + HOLD_MARGIN of the model's stop point
HOLD_MARGIN = 1.0          # m
GO_DIST = 20.              # m: a held stop lets go once the model's plan reaches this far past the offset
GO_V = 4.0                 # m/s: or ends this fast (the recorded green light: 3 m to 69 m in one frame)
GO_FRAMES = 2              # in a row, so one bad frame does not drop it
CREEP_A = 0.2              # m/s^2: and lets go if the model keeps asking for more than this
CREEP_T = 3.0              # s: for this long, so a model that only creeps (a stop sign) is never stuck
LEAD_MARGIN = 8.0          # m: a car this close past the model's stop point is what it is stopping for, not a light.
                           # Radar dRel is from the bumper, the model's x from the device ~2.5 m behind it, and once
                           # stopped the model's x[-1] is near 0 while the car ahead is 3-6 m away

OFFSET_MAX = 5             # m, the setting's range is 0 to 5
PARAM_READ_FRAMES = 20     # plannerd runs at 20 Hz: re-read the setting every 1 s
CREEP_FRAMES = int(round(CREEP_T / DT_MDL))


def clip_offset(value) -> float:
  try:
    value = float(value)
  except (TypeError, ValueError):
    return 0.
  return float(np.clip(value, 0, OFFSET_MAX)) if math.isfinite(value) else 0.


class RedLightStop:
  def __init__(self, CP, params):
    self.available = bool(CP.openpilotLongitudinalControl)
    self._params = params
    self._frame = 0
    self._error_logged = False

    self.offset = 0.
    self.armed = False
    self.holding = False
    self.extra = 0.         # m/s^2 of braking currently added
    self.stop_dist = 0.     # m, the model's smoothed stop point ahead of the truck
    self._odo = 0.          # m driven, to smooth the stop point where it is on the road
    self._stop_point = FirstOrderFilter(0., STOP_FILTER_TAU, DT_MDL)
    self._go_frames = 0
    self._creep_frames = 0

  def read_offset(self) -> float:
    try:
      return clip_offset(self._params.get("RedLightStopDistance", return_default=True))
    except Exception:
      # params library built without the key, or a failed read: off rather than take plannerd down
      if not self._error_logged:
        cloudlog.exception("RedLightStopDistance read failed, treating it as off")
        self._error_logged = True
      return 0.

  def update_offset(self) -> None:
    if self.available and self._frame % PARAM_READ_FRAMES == 0:
      self.offset = self.read_offset()
    self._frame += 1

  def _disarm(self) -> None:
    self.armed = False
    self.holding = False

  def _candidate(self, a_e2e: float) -> tuple[float, bool] | None:
    if self.extra <= 0. and not self.holding:
      return None
    accel = max(a_e2e - self.extra, min(a_e2e, A_FLOOR))
    if self.holding:
      accel = min(accel, 0.)
    return float(accel), self.holding

  def update(self, is_e2e: bool, v_ego: float, a_e2e: float, model, lead, long_off: bool = False) -> tuple[float, bool] | None:
    """Returns an (accel, should_stop) planner candidate, or None when there is nothing to add."""
    if long_off or not math.isfinite(a_e2e) or not math.isfinite(v_ego):
      # nothing is being commanded (driver override, disengaged), or no usable speed: start clean next time
      self._disarm()
      self.extra = 0.
      self._odo = 0.
      return None

    v_ego = max(v_ego, 0.)
    self._odo += v_ego * DT_MDL
    x, v = model.position.x, model.velocity.x
    valid = self.available and is_e2e and self.offset > 0. and len(x) > 0 and len(v) > 0 and \
            math.isfinite(x[-1]) and math.isfinite(v[-1])
    if not valid:
      # setting off, not end to end, or no usable plan: let any added braking fade rather than drop at once
      self._disarm()
      self.extra = max(self.extra - EXTRA_FALL_RATE * DT_MDL, 0.)
      return self._candidate(a_e2e) if is_e2e else None

    stop_dist, plan_end_v = float(x[-1]), float(v[-1])

    # a car stopped at or just past the model's stop point is a lead stop, left to the model and the lead planner
    if lead.present and lead.dRel < stop_dist + LEAD_MARGIN:
      self._disarm()
    elif self.holding:
      going = stop_dist > self.offset + GO_DIST or plan_end_v > GO_V
      self._go_frames = self._go_frames + 1 if going else 0
      self._creep_frames = self._creep_frames + 1 if a_e2e > CREEP_A else 0
      if v_ego > HOLD_V or self._go_frames >= GO_FRAMES or self._creep_frames >= CREEP_FRAMES:
        self._disarm()
    elif self.armed:
      if plan_end_v > STOP_RELEASE_V:
        self._disarm()
    elif plan_end_v < STOP_ARM_V and a_e2e < 0.:
      # the model plans to stop and stay stopped (a red light) and has started braking for it
      self.armed = True
      self._stop_point.x = self._odo + stop_dist

    self._stop_point.update(self._odo + stop_dist)
    self.stop_dist = d = self._stop_point.x - self._odo

    target_extra = 0.
    if self.armed and not self.holding:
      d_short = d - self.offset
      target_extra = v_ego ** 2 / 2. * (1. / max(d_short, MIN_TARGET_DIST) - 1. / max(d, MIN_TARGET_DIST))
      if v_ego < LOW_SPEED_V:
        a_short = v_ego ** 2 / (2. * max(d_short, 0.5))
        if a_short > LOW_SPEED_A_MIN:
          target_extra = max(target_extra, a_e2e + a_short)
      target_extra = min(max(target_extra, 0.), EXTRA_DECEL_MAX)

      # once nearly stopped within reach of the shortened stop point, stay stopped rather than creep up to the model's
      if v_ego < HOLD_V and d <= self.offset + HOLD_MARGIN:
        self.holding = True
        self._go_frames = 0
        self._creep_frames = 0
    self.extra = float(np.clip(target_extra, self.extra - EXTRA_FALL_RATE * DT_MDL, self.extra + EXTRA_RISE_RATE * DT_MDL))

    return self._candidate(a_e2e)
