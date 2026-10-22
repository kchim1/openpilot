"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import hashlib
import math
import os

import numpy as np

from openpilot.cereal import log
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.modeld.constants import ModelConstants

LaneChangeState = log.LaneChangeState

# Sharp Turn Steering: in tight low-speed turns the driving model's request (modelV2.action.desiredCurvature, its action
# output) asks for too little curvature. Over the 92 sharp turns in 149 segments of this truck's drives (4 builds, one
# model: the bundled big model, Cinque Terre Model v2, on the Chestnut, on every frame), the request peaked at a median
# 0.62 of the curvature driven (0.77 in road turns: R >= 10 m at 3 m/s and up), less the tighter and slower the turn, the same in every
# build. openpilot's vehicle model is right on this truck (gyro yaw rate / speed against its curvature from the steering
# angle: 0.94-1.02 in every angle bin), and across 62 public trucks the steering geometry varies about +-3% within each
# body (R1T and R1S differ only in wheelbase; paramsd learns the rest), so the shortfall is the model's, not the
# truck's. The model's own planned path turns harder, so this moves the request part of the way (BLEND) toward that
# plan's curvature half a second out, only when the plan turns harder the same way. That plan point mostly follows the
# truck's own turn (about 0.9x it, from manual turns), so with openpilot steering what it adds partly comes back through
# the plan: a loop gain of about 0.35, bounded, and in closed loop the gain is smaller than replayed (about 0.84 of the
# curvature driven in road turns, 0.88 open loop). REQUEST_CAP is the backstop: never more than that times the model's
# own request, which does not follow the truck. The plan's curvature uses the truck's speed, not the plan's, which reads
# 2-17% low in curves. Only with the model it was measured on: another model may not under-ask (an older plan-based one
# asked 1.1-2.5x on a comma CI R1T log), so off unless nothing is picked for the Chestnut (its bundled big model
# drives), and off on any frame the small model drives (no Chestnut, or the big model failed). Never during a lane
# change. Its timing is the model's (half of its own peak about 0.1 s ahead of the truck); being bigger it reaches a
# given curvature sooner, as a driver turning harder would. No setting: how much the model under-asks is not a matter of
# taste.
# All of it below SPEED_BP[0] and in turns tighter than CURV_BP[1] (R 50 m), none from SPEED_BP[1], on curves gentler
# than CURV_BP[0] (R 125 m), or below LOW_SPEED_BP[0]. Lane keeping and highway driving are untouched.
BLEND = 0.4                        # the share of the way to the plan's curvature
REQUEST_CAP = 1.35                 # and never more than this times the model's own request
PLAN_IDX = 7                       # the plan half a second out (ModelConstants.T_IDXS[7] = 0.48 s)
SPEED_BP = [9., 14.]               # m/s: 20 -> 31 mph
LOW_SPEED_BP = [1.5, 3.]           # m/s
CURV_BP = [0.008, 0.02]            # 1/m: R 125 m -> 50 m
V_MIN = 0.5                        # m/s
# model_fingerprint() of the big model it was measured on: big_driving_tinygrad.pkl, LFS oid 76cc0a9b, 776634338 bytes
MEASURED_FINGERPRINT = "b205523eddbe7196de2af2ad1495e4bf04f18bf16bc260f1974ebf29e2b7f3d3"
FP_SLICES, FP_SLICE_BYTES = 1024, 4096


def model_parts(path: str) -> list[tuple[str, int]] | None:
  """A model file's parts and sizes, whole or split into chunks (common/file_chunker.py), or None when it is not there."""
  from openpilot.common.file_chunker import get_chunk_name, get_manifest_path
  if os.path.isfile(path):
    return [(path, os.path.getsize(path))]
  manifest = get_manifest_path(path)
  if not os.path.isfile(manifest):
    return None
  with open(manifest) as f:
    n = int(f.read().strip())
  return [(c, os.path.getsize(c)) for c in (get_chunk_name(path, i, n) for i in range(n))]


def read_at(parts: list[tuple[str, int]], pos: int, n: int) -> bytes:
  out = b""
  for part, size in parts:
    if pos >= size:
      pos -= size
      continue
    with open(part, "rb") as f:
      f.seek(pos)
      out += f.read(min(n - len(out), size - pos))
    pos = 0
    if len(out) >= n:
      break
  return out


def model_fingerprint(path: str, slices: int = FP_SLICES, slice_bytes: int = FP_SLICE_BYTES) -> str | None:
  """sha256 of a model file's size and of slices spread evenly across it. Hashing the whole file (776 MB) would take
  seconds; this reads 4 MB, and sees any change wider than the 0.76 MB between slices. The two past big models of the
  same size (773082315 bytes) differ in only 4% of the file, in 54 places (21 of them wider than that gap): 41 of the
  slices land on a change."""
  parts = model_parts(path)
  if parts is None:
    return None
  size = sum(s for _, s in parts)
  h = hashlib.sha256(str(size).encode())
  for k in range(slices):
    h.update(read_at(parts, max(size - slice_bytes, 0) * k // max(slices - 1, 1), slice_bytes))
  return h.hexdigest()


def measured_model_active(params) -> bool:
  """The Chestnut runs the model the correction was measured on. comma's modeld runs the build's big model file when
  nothing is picked for the Chestnut, and that file must be the one measured: Cinque Terre Model v2,
  big_driving_tinygrad.pkl with LFS oid 76cc0a9b, in every logged build. The file is checked, not the model's name,
  which upstream has left stale across a model swap. Anything picked from the catalog stays off, even this model picked
  by ref, since sunnypilot's modeld computes the request another way. Whether the big model drives is checked per
  frame."""
  try:
    from openpilot.selfdrive.modeld.helpers import modeld_pkl_path
    from openpilot.sunnypilot.models.helpers import get_active_bundle
    if get_active_bundle(params, chestnut=True) is not None:
      cloudlog.warning("Sharp Turn Steering: off, a model is picked for the Chestnut")
      return False
    fingerprint = model_fingerprint(str(modeld_pkl_path(chestnut=True)))
    if fingerprint != MEASURED_FINGERPRINT:
      cloudlog.warning(f"Sharp Turn Steering: off, the big model file is not the one measured ({fingerprint})")
      return False
    cloudlog.warning("Sharp Turn Steering: on, the big model file is the one measured")
    return True
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
    if not self.enabled or not model.big or model.meta.laneChangeState != LaneChangeState.off:
      return curvature
    return sharpen(curvature, plan_curvature(model, v_ego), v_ego, self.blend)
