"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math
from types import SimpleNamespace

import pytest

from openpilot.selfdrive.modeld.constants import ModelConstants
import openpilot.sunnypilot.selfdrive.controls.lib.sharp_turn_steering as sts_mod
from openpilot.sunnypilot.selfdrive.controls.lib.sharp_turn_steering import SharpTurnSteering, BLEND, REQUEST_CAP, PLAN_IDX, \
  SPEED_BP, LOW_SPEED_BP, V_MIN, LaneChangeState, measured_model_active, plan_curvature, sharpen

V = 5.


def model(plan_curv, v=V, lane_change=LaneChangeState.off, plan_v=None, big=False):
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
    # the big (Chestnut) model driving this frame
    assert SharpTurnSteering(enabled=True).apply(0.04, V, model(0.05, big=True)) == 0.04

  def test_not_during_a_lane_change(self):
    sts = SharpTurnSteering(enabled=True)
    for state in (LaneChangeState.preLaneChange, LaneChangeState.laneChangeStarting, LaneChangeState.laneChangeFinishing):
      assert sts.apply(0.04, V, model(0.05, lane_change=state)) == 0.04


class TestMeasuredModel:
  @staticmethod
  def patch(monkeypatch, bundle=None, default="CD210", raises=False):
    import openpilot.sunnypilot.models.helpers as helpers
    import openpilot.sunnypilot.models.model_name as model_name
    def active_bundle(*a, **k):
      if raises:
        raise OSError("params")
      assert k.get("chestnut") is False   # the qcom slot: the big model is checked per frame
      return bundle
    monkeypatch.setattr(helpers, "get_active_bundle", active_bundle)
    monkeypatch.setattr(model_name, "DEFAULT_MODEL", default)

  def test_default_model_with_nothing_picked(self, monkeypatch):
    self.patch(monkeypatch)
    assert measured_model_active(None)

  def test_the_same_model_picked_by_ref(self, monkeypatch):
    from openpilot.sunnypilot.models.model_name import DEFAULT_MODEL_REF
    self.patch(monkeypatch, bundle=SimpleNamespace(ref=DEFAULT_MODEL_REF))
    assert measured_model_active(None)

  @pytest.mark.parametrize("kw", [{"bundle": SimpleNamespace(ref="another-model")}, {"default": "SomeNewModel"}, {"raises": True}])
  def test_off_otherwise(self, monkeypatch, kw):
    monkeypatch.setattr(sts_mod, "cloudlog", SimpleNamespace(exception=lambda *a, **k: None))
    self.patch(monkeypatch, **kw)
    assert not measured_model_active(None)
    assert SharpTurnSteering().apply(0.04, V, model(0.05)) == 0.04


@pytest.mark.parametrize("v", [0., 2., 5., 9., 12., 20., 35.])
def test_monotonic_in_the_plan(v):
  prev = None
  for i in range(60):
    out = sharpen(0.03, 0.03 + i * 0.002, v, BLEND)
    assert out >= 0.03 and (prev is None or out >= prev - 1e-12)
    prev = out
