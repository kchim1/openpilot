"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
import os
import subprocess
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants
import openpilot.sunnypilot.selfdrive.controls.lib.sharp_turn_steering as sts_mod
from openpilot.sunnypilot.selfdrive.controls.lib.sharp_turn_steering import SharpTurnSteering, BLEND, REQUEST_CAP, PLAN_IDX, \
  SPEED_BP, LOW_SPEED_BP, V_MIN, LaneChangeState, measured_model_active, model_fingerprint, model_parts, plan_curvature, sharpen

V = 5.
MEASURED_OID = "76cc0a9bc3af7a318889483dcbe126337f8d338f5abcbe664a8988c9b18b6639"  # big_driving_tinygrad.pkl, every logged build
MEASURED_SIZE = 776634338


def model(plan_curv, v=V, lane_change=LaneChangeState.off, plan_v=None, big=True):
  yaw = [0.] * ModelConstants.IDX_N
  yaw[PLAN_IDX] = plan_curv * v
  return SimpleNamespace(orientationRate=SimpleNamespace(z=yaw),
                         velocity=SimpleNamespace(x=[plan_v if plan_v is not None else v] * ModelConstants.IDX_N),
                         meta=SimpleNamespace(laneChangeState=lane_change), big=big)


def expected(req, plan):
  return math.copysign(min(abs(req + BLEND * (plan - req)), REQUEST_CAP * abs(req)), req)


class TestSharpen:
  def test_bounds(self):
    assert 0. < BLEND < 1. and 1. < REQUEST_CAP < 1.5

  def test_plan_curvature_uses_the_trucks_speed(self):
    # the plan's own speed reads low in curves; the truck's speed is used
    assert abs(plan_curvature(model(0.07, plan_v=4.), V) - 0.07) < 1e-9
    assert abs(plan_curvature(model(0.07), 0.) - 0.07 * V / V_MIN) < 1e-9
    assert math.isnan(plan_curvature(SimpleNamespace(orientationRate=SimpleNamespace(z=[])), V))

  def test_moves_part_of_the_way_to_a_harder_turning_plan(self):
    for sign in (1., -1.):
      assert abs(sharpen(sign * 0.04, sign * 0.07, V, BLEND) - sign * (0.04 + BLEND * 0.03)) < 1e-9

  def test_never_more_than_the_cap_times_the_request(self):
    # a plan far beyond the request (it partly echoes the truck's own turn): capped
    for req, plan in ((0.02, 0.2), (-0.03, -0.15), (0.01, 0.05)):
      out = sharpen(req, plan, V, BLEND)
      assert abs(out) <= REQUEST_CAP * abs(req) + 1e-12
      assert abs(out - expected(req, plan)) < 1e-12

  def test_never_straightens_or_reverses(self):
    assert sharpen(0.05, 0.03, V, BLEND) == 0.05
    assert sharpen(0.05, -0.07, V, BLEND) == 0.05
    assert sharpen(0., 0.07, V, BLEND) == 0.

  def test_only_slow_and_tight(self):
    assert sharpen(0.04, 0.07, SPEED_BP[1], BLEND) == 0.04
    assert sharpen(0.04, 0.07, 25., BLEND) == 0.04
    assert sharpen(0.04, 0.07, LOW_SPEED_BP[0], BLEND) == 0.04
    assert sharpen(0.004, 0.0079, V, BLEND) == 0.004
    full = sharpen(0.04, 0.05, V, BLEND)
    assert abs(sharpen(0.04, 0.05, sum(SPEED_BP) / 2, BLEND) - (0.04 + (full - 0.04) / 2)) < 1e-9
    assert abs(sharpen(0.04, 0.05, sum(LOW_SPEED_BP) / 2, BLEND) - (0.04 + (full - 0.04) / 2)) < 1e-9

  def test_bad_inputs_are_a_no_op(self):
    assert sharpen(0.04, 0.07, V, 0.) == 0.04
    for bad in ((0.04, math.nan, V), (0.04, 0.07, math.inf)):
      assert sharpen(*bad, BLEND) == bad[0]
    assert math.isnan(sharpen(math.nan, 0.07, V, BLEND))


class TestSharpTurnSteering:
  def test_on_with_the_measured_model(self):
    sts = SharpTurnSteering(enabled=True)
    assert abs(sts.apply(0.04, V, model(0.05)) - (0.04 + BLEND * 0.01)) < 1e-9
    assert sts.apply(0.04, 20., model(0.05)) == 0.04

  def test_off_with_any_other_model(self):
    assert SharpTurnSteering(enabled=False).apply(0.04, V, model(0.05)) == 0.04
    # the small model driving this frame (no Chestnut, or the big model failed)
    assert SharpTurnSteering(enabled=True).apply(0.04, V, model(0.05, big=False)) == 0.04

  def test_not_during_a_lane_change(self):
    sts = SharpTurnSteering(enabled=True)
    for state in (LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing):
      assert sts.apply(0.04, V, model(0.05, lane_change=state)) == 0.04


class TestMeasuredModel:
  @staticmethod
  def patch(monkeypatch, bundle=None, fingerprint=sts_mod.MEASURED_FINGERPRINT, raises=False):
    import openpilot.sunnypilot.models.helpers as helpers
    def active_bundle(*a, **k):
      if raises:
        raise OSError("params")
      assert k.get("chestnut") is True   # the Chestnut's slot, whether or not it is plugged in at this moment
      return bundle
    monkeypatch.setattr(helpers, "get_active_bundle", active_bundle)
    monkeypatch.setattr(sts_mod, "model_fingerprint", lambda path: fingerprint)

  def test_bundled_big_model_with_nothing_picked(self, monkeypatch):
    self.patch(monkeypatch)
    assert measured_model_active(None)
    assert SharpTurnSteering().apply(0.04, V, model(0.05)) > 0.04

  @pytest.mark.parametrize("kw", [{"bundle": SimpleNamespace(ref="another-model")}, {"fingerprint": "0" * 64}, {"fingerprint": None},
                                  {"raises": True}])
  def test_off_otherwise(self, monkeypatch, kw):
    logged = []
    monkeypatch.setattr(sts_mod, "cloudlog", SimpleNamespace(exception=logged.append, warning=logged.append))
    self.patch(monkeypatch, **kw)
    assert not measured_model_active(None)
    assert len(logged) == 1 and ("off" in logged[0] or "could not tell" in logged[0])   # why it is off, once
    assert SharpTurnSteering().apply(0.04, V, model(0.05)) == 0.04

  def test_same_model_picked_from_the_catalog(self, monkeypatch):
    # sunnypilot's modeld runs a picked model and computes the request another way
    from openpilot.sunnypilot.models.model_name import DEFAULT_BIG_MODEL_REF
    self.patch(monkeypatch, bundle=SimpleNamespace(ref=DEFAULT_BIG_MODEL_REF))
    assert not measured_model_active(None)

  def test_the_bundled_big_model_is_the_one_measured(self):
    # comma's modeld loads this file on the Chestnut; the shortfall was measured with exactly this one (and
    # MEASURED_FINGERPRINT is of it). A new big model must be measured again before this pin moves.
    from openpilot.selfdrive.modeld.helpers import modeld_pkl_path
    path = modeld_pkl_path(chestnut=True)
    pointer = subprocess.check_output(["git", "show", f":./{path.name}"], cwd=path.parent, text=True).splitlines()
    assert "oid sha256:" + MEASURED_OID in pointer and f"size {MEASURED_SIZE}" in pointer


class TestModelFingerprint:
  SLICES, SLICE = 64, 512

  @staticmethod
  def write(path, data):
    with open(path, "wb") as f:
      f.write(data)
    return str(path)

  def fp(self, path):
    return model_fingerprint(path, self.SLICES, self.SLICE)

  def test_whole_or_in_chunks(self, tmp_path, monkeypatch):
    import openpilot.common.file_chunker as fc
    data = os.urandom(5 * 2**20 + 123)
    whole = self.write(tmp_path / "whole.pkl", data)
    chunked = self.write(tmp_path / "chunked.pkl", data)
    monkeypatch.setattr(fc, "CHUNK_SIZE", 2**20)
    fc.chunk_file(chunked, fc.get_chunk_targets(chunked, len(data)))
    assert not os.path.exists(chunked) and len(model_parts(chunked)) == 6
    assert self.fp(whole) == self.fp(chunked) is not None

  def test_a_change_wider_than_the_gap_is_seen(self, tmp_path):
    data = bytearray(os.urandom(2**20))
    base = self.fp(self.write(tmp_path / "a.pkl", data))
    gap = (len(data) - self.SLICE) // (self.SLICES - 1)
    for start in range(0, len(data) - gap, len(data) // 37):
      changed = bytearray(data)
      changed[start:start + gap] = os.urandom(gap)
      assert self.fp(self.write(tmp_path / "b.pkl", changed)) != base
    assert self.fp(self.write(tmp_path / "c.pkl", data + b"\0")) != base   # the size counts too

  def test_missing(self, tmp_path):
    assert model_fingerprint(str(tmp_path / "none.pkl")) is None


@pytest.mark.parametrize("v", [0., 2., 5., 9., 12., 20., 35.])
def test_monotonic_in_the_plan(v):
  prev = None
  for i in range(60):
    out = sharpen(0.03, 0.03 + i * 0.002, v, BLEND)
    assert out >= 0.03 and (prev is None or out >= prev - 1e-12)
    prev = out
