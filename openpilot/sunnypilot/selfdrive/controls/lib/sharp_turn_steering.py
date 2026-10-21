"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import numpy as np

from openpilot.cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import ModelConstants

LaneChangeState = log.LaneChangeState

# Sharp Turn Steering: in tight low-speed turns the driving model's request (modelV2.action.desiredCurvature, its action
# output) asks for too little curvature. Over the 92 sharp turns in 149 segments of this truck's drives (4 builds, one
# model: the bundled default, CD210), the request peaked at a median 0.62 of the curvature driven (0.77 in road turns,
# R >= 10 m), less the tighter and slower the turn, the same in every build. openpilot's vehicle model is right on
# this truck (gyro yaw rate / speed against its curvature from the steering angle: 0.94-1.02 in every angle bin), and
# across 62 public trucks the steering geometry varies about +-3% within each body (R1T and R1S differ only in wheelbase;
# paramsd learns the rest), so the shortfall is the model's, not the truck's. The model's own planned path turns harder,
# so this moves the request part of the way (BLEND) toward that plan's curvature half a second out, only when the plan
# turns harder the same way. That plan point mostly follows the truck's own turn (about 0.9x it, from manual turns), so
# with openpilot steering what it adds partly comes back through the plan: a loop gain of about 0.35, bounded, and in
# closed loop the gain is smaller than replayed (about 0.84 of the curvature driven in road turns, 0.88 open loop).
# REQUEST_CAP is the backstop: never more than that times the model's own request, which does not follow the truck.
# The plan's curvature uses the truck's speed, not the plan's, which reads 2-17% low in curves. Only with the model it
# was measured on: another model may not under-ask (an older plan-based one asked 1.1-2.5x on a comma CI R1T log), so
# off unless the bundled default is in use and off on any frame the big (Chestnut) model drives. Never during a lane
# change. Its timing is the model's (half of its own peak about 0.1 s ahead of the truck); being bigger it reaches a
# given curvature sooner, as a driver turning harder would. No setting: how much the model under-asks is not a matter
# of taste.
# All of it below SPEED_BP[0] and in turns tighter than CURV_BP[1] (R 50 m), none from SPEED_BP[1], on curves gentler
# than CURV_BP[0] (R 125 m), or below LOW_SPEED_BP[0]. Lane keeping and highway driving are untouched.
BLEND = 0.4                        # the share of the way to the plan's curvature
REQUEST_CAP = 1.35                 # and never more than this times the model's own request
PLAN_IDX = 7                       # the plan half a second out (ModelConstants.T_IDXS[7] = 0.48 s)
SPEED_BP = [9., 14.]               # m/s: 20 -> 31 mph
LOW_SPEED_BP = [1.5, 3.]           # m/s
CURV_BP = [0.008, 0.02]            # 1/m: R 125 m -> 50 m
V_MIN = 0.5                        # m/s
MEASURED_MODEL = "CD210"           # the bundled default model the shortfall was measured on


def measured_model_active(params) -> bool:
  """The driving model picked is the one the correction was measured on: the build's bundled default (no model
  picked), or that same model picked by ref. Whether the big (Chestnut) model drives instead is checked per frame."""
  try:
    from openpilot.sunnypilot.models.helpers import get_active_bundle
    from openpilot.sunnypilot.models.model_name import DEFAULT_MODEL, DEFAULT_MODEL_REF
    if DEFAULT_MODEL != MEASURED_MODEL:
      return False
    bundle = get_active_bundle(params, chestnut=False)
    return bundle is None or bundle.ref == DEFAULT_MODEL_REF
  except Exception:
    cloudlog.exception("Sharp Turn Steering: could not tell the active driving model, staying off")
    return False


def plan_curvature(model, v_ego: float) -> float:
  yaw_rate = model.orientationRate.z
  if len(yaw_rate) != ModelConstants.IDX_N:
    return math.nan
  return float(yaw_rate[PLAN_IDX]) / max(v_ego, V_MIN)


def sharpen(curvature: float, plan: float, v_ego: float, blend: float) -> float:
  if blend <= 0. or not all(math.isfinite(x) for x in (curvature, plan, v_ego)):
    return curvature
  if plan * curvature <= 0. or abs(plan) <= abs(curvature):
    return curvature
  w = float(np.interp(v_ego, SPEED_BP, [1., 0.])) * float(np.interp(v_ego, LOW_SPEED_BP, [0., 1.])) * \
      float(np.interp(abs(plan), CURV_BP, [0., 1.]))
  out = curvature + blend * w * (plan - curvature)
  return math.copysign(min(abs(out), REQUEST_CAP * abs(curvature)), curvature)


class SharpTurnSteering:
  def __init__(self, params=None, blend: float = BLEND, enabled: bool | None = None):
    self.blend = blend
    self.enabled = measured_model_active(params) if enabled is None else enabled

  def apply(self, curvature: float, v_ego: float, model) -> float:
    if not self.enabled or model.big or model.meta.laneChangeState != LaneChangeState.off:
      return curvature
    return sharpen(curvature, plan_curvature(model, v_ego), v_ego, self.blend)
