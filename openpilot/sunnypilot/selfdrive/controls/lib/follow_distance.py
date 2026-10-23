"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP
from openpilot.common.swaglog import cloudlog

# Rivian Following Distance: with the longitudinal harness, the right thumb wheel sets how far behind the car ahead
# sunnypilot follows, instead of cycling the driving personality. Each level is a follow time before the Following
# Distance Offset, which still applies on top: Medium is Aggressive's 1.25 s (at a -10% offset, the 1.125 s gap of
# before), Far is Standard's 1.45 s, Close 0.15 s under Medium and Closer 0.1 s under Close. The personality keeps the
# gas and brake feel. selfdrived steps the level on the wheel and writes the param, plannerd reads it from there. Light
# imports only: selfdrived imports this too, and must not load the MPC solver
PARAM = "RivianFollowDistance"
NAMES = ("Closer", "Close", "Medium", "Far")
T_FOLLOW = (1.00, 1.10, 1.25, 1.45)  # s
DEFAULT = 2                          # Medium
FAST_LEAD_LEVELS = (0, 1)            # Closer and Close, under Medium's gap: plannerd also uses its faster estimate of the lead's
                                     # acceleration (fast_lead_accel.py)
RAMP_RATE = 0.1                      # s of follow time per s while openpilot controls the speed: a level in 1 to 2 s
READ_FRAMES = 5                      # plannerd runs at 20 Hz: re-read the level every 0.25 s


def available(CP, CP_SP) -> bool:
  """Rivian with openpilot longitudinal and the harness upgrade, the only case card reads the wheel"""
  return CP.brand == "rivian" and bool(CP.openpilotLongitudinalControl) and \
    bool(CP_SP.flags & RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE)


def sanitize(value) -> int:
  """the stored level, clipped to the ones there are; anything that is not a number is Medium"""
  try:
    return min(max(int(value), 0), len(T_FOLLOW) - 1)
  except (TypeError, ValueError, OverflowError):
    return DEFAULT


class FollowDistance:
  """plannerd: the level, and the follow time before the offset (t_base), eased toward the level's"""
  def __init__(self, params):
    self._frame = 0
    self._error_logged = False
    self.level = self.read(params)
    self.t_base = T_FOLLOW[self.level]  # at the level from the start: nothing to ease in

  def read(self, params) -> int:
    try:
      return sanitize(params.get(PARAM, return_default=True))
    except Exception:
      # params library built without the key, or a failed read: Medium rather than take plannerd down
      if not self._error_logged:
        cloudlog.exception(f"{PARAM} read failed, using Medium")
        self._error_logged = True
      return DEFAULT

  def update(self, params, long_active: bool, dt: float) -> float:
    if self._frame % READ_FRAMES == 0:
      self.level = self.read(params)
    self._frame += 1

    # a change eases in at RAMP_RATE while openpilot controls the speed, so the plan never jumps; at once otherwise
    target = T_FOLLOW[self.level]
    step = RAMP_RATE * dt
    if not long_active or abs(target - self.t_base) < step + 1e-9:  # within float error of a step: land on it exactly
      self.t_base = target
    else:
      self.t_base += step if target > self.t_base else -step
    return self.t_base
