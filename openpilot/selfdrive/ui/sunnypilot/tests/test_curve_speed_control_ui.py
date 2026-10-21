"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Curve Speed Control on the screens: when the comma four's C badge shows, where, and for how long; when the comma 3/3X
CURVE pill blinks; and the UI forgetting the setting without openpilot longitudinal. The badge runs through the real
HudRendererSP with raylib's draw calls recorded, so no window is needed.
"""
from types import SimpleNamespace

import pytest
import pyray as rl

from openpilot.cereal import custom, messaging
from opendbc.car.structs import car
import openpilot.selfdrive.ui.sunnypilot.mici.onroad.hud_renderer as hud_renderer
import openpilot.selfdrive.ui.sunnypilot.mici.widgets.curve_speed_badge as curve_speed_badge
import openpilot.selfdrive.ui.sunnypilot.mici.widgets.traffic_mode_badge as traffic_mode_badge
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.sunnypilot.onroad.smart_cruise_control import SmartCruiseControlRenderer
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.sunnypilot.lib.utils import AlertFadeAnimator

CurveState = custom.LongitudinalPlanSP.SmartCruiseControl.CurveState

ROAD_VIEW = rl.Rectangle(0, 0, 476, 240)  # the comma four's camera area
C_COLOR = (0, 255, 0)
T_COLOR = (201, 34, 49)


class FakeSM(dict):
  def __init__(self):
    super().__init__()
    self.updated = {"longitudinalPlanSP": False, "carControl": False}


def plan_sp(state=CurveState.cruise, governor=False, active=None):
  msg = messaging.new_message("longitudinalPlanSP")
  curve = msg.longitudinalPlanSP.smartCruiseControl.curve
  curve.enabled = True
  curve.state = state
  curve.active = state in (CurveState.slowing, CurveState.curve) if active is None else active
  curve.governorActive = governor
  return msg.longitudinalPlanSP.as_reader()


class Badges:
  """the real HudRendererSP's badge code, its draw calls recorded as {letter: (left, right, top, bottom)}"""
  def __init__(self, monkeypatch):
    self.now = 100.0
    self.circles = []
    self.sm = FakeSM()
    monkeypatch.setattr(rl, "get_time", lambda: self.now)
    monkeypatch.setattr(rl, "draw_circle_v", lambda center, r, color: self.circles.append((center.x, center.y, r, color)))
    monkeypatch.setattr(rl, "draw_text_ex", lambda *args: None)
    for module in (curve_speed_badge, traffic_mode_badge):
      monkeypatch.setattr(module, "measure_text_cached", lambda font, text, size: rl.Vector2(size * 0.6, size))
    monkeypatch.setattr(HudRenderer, "_update_state", lambda self: None)
    monkeypatch.setattr(hud_renderer, "lateral_mode", SimpleNamespace(update=lambda: None, wheel_tint=None))
    monkeypatch.setattr(ui_state, "sm", self.sm)
    monkeypatch.setattr(ui_state, "has_longitudinal_control", True)
    monkeypatch.setattr(ui_state, "traffic_mode", False)

    self.hud = object.__new__(hud_renderer.HudRendererSP)
    self.hud.blind_spot_indicators = SimpleNamespace(update=lambda: None)
    self.hud._curve_working_time = -hud_renderer.CURVE_BADGE_HOLD
    self.hud._font_bold = None
    self.hud._can_draw_top_icons = True

  def frame(self, plan=None, dt=1 / 60, traffic_mode=False, long=True, alert=False):
    self.sm.updated["longitudinalPlanSP"] = plan is not None
    if plan is not None:
      self.sm["longitudinalPlanSP"] = plan
    ui_state.traffic_mode = traffic_mode
    ui_state.has_longitudinal_control = long
    self.hud._can_draw_top_icons = not alert
    self.now += dt
    self.circles.clear()
    self.hud._update_state()
    self.hud._draw_traffic_mode_badge(ROAD_VIEW)
    self.hud._draw_curve_speed_badge(ROAD_VIEW)
    shown = {}
    for x, y, r, color in self.circles:
      letter = {C_COLOR: "C", T_COLOR: "T"}[(color.r, color.g, color.b)]
      shown[letter] = (x - r, x + r, y - r, y + r)
    return shown


@pytest.fixture
def badges(monkeypatch):
  return Badges(monkeypatch)


class TestCurveBadge:
  def test_shows_while_slowing_or_trimming(self, badges):
    assert badges.frame(plan_sp()) == {}
    assert set(badges.frame(plan_sp(CurveState.slowing))) == {"C"}
    badges.frame(plan_sp(), dt=2.)
    assert set(badges.frame(plan_sp(governor=True))) == {"C"}

  def test_not_while_only_holding_a_cap(self, badges):
    # state curve: the cap is at or above the truck's speed, which includes a truck kept 10 m/s under it by a lead
    assert badges.frame(plan_sp(CurveState.curve)) == {}
    assert badges.frame(plan_sp(CurveState.disabled)) == {}

  def test_left_of_the_traffic_mode_badge_never_on_it(self, badges):
    alone = badges.frame(plan_sp(CurveState.slowing))
    both = badges.frame(plan_sp(CurveState.slowing), traffic_mode=True)
    assert set(both) == {"C", "T"}
    assert both["C"][1] < both["T"][0]  # C's right edge left of T's left edge
    assert both["C"][2:] == both["T"][2:]  # same row
    assert alone["C"] == both["T"]  # alone, it takes the T's corner

  def test_hidden_under_an_alert_and_without_openpilot_longitudinal(self, badges):
    assert badges.frame(plan_sp(CurveState.slowing), alert=True, traffic_mode=True) == {}
    assert badges.frame(plan_sp(CurveState.slowing), long=False) == {}

  def test_held_one_second(self, badges):
    badges.frame(plan_sp(CurveState.slowing))
    assert set(badges.frame(plan_sp(CurveState.curve), dt=0.5)) == {"C"}
    assert badges.frame(plan_sp(CurveState.curve), dt=0.55) == {}

  def test_a_message_that_is_not_updated_does_not_hold_it(self, badges):
    # plannerd stopped: the last message still says slowing, but only a new one restarts the hold
    badges.frame(plan_sp(CurveState.slowing))
    badges.frame(dt=0.6)
    assert badges.frame(dt=0.5) == {}


class TestCurvePill:
  @staticmethod
  def pill_after(plan):
    sm = FakeSM()
    sm.updated["longitudinalPlanSP"] = True
    sm["longitudinalPlanSP"] = plan
    pill = object.__new__(SmartCruiseControlRenderer)
    pill.vision_active = pill.map_active = False
    pill._vision_fade = AlertFadeAnimator(20)
    pill._map_fade = AlertFadeAnimator(20)
    pill._curve_fade = AlertFadeAnimator(20)
    ui_state_sm = ui_state.sm
    ui_state.sm = sm
    try:
      pill.update()
    finally:
      ui_state.sm = ui_state_sm
    return pill

  def test_shown_while_on_and_blinks_while_slowing_or_trimming(self):
    for plan, blinks in ((plan_sp(), False), (plan_sp(CurveState.curve), False), (plan_sp(CurveState.slowing), True),
                         (plan_sp(governor=True), True)):
      pill = self.pill_after(plan)
      assert pill.curve_enabled
      assert pill.curve_active == blinks


class TestForgottenInTheUI:
  @staticmethod
  def enforce(params, long, icbm_available, icbm_on):
    CP = car.CarParams.new_message(openpilotLongitudinalControl=long).as_reader()
    CP_SP = custom.CarParamsSP.new_message(intelligentCruiseButtonManagementAvailable=icbm_available).as_reader()
    state = SimpleNamespace(params=params, CP=CP, CP_SP=CP_SP, has_longitudinal_control=long, has_icbm=icbm_on)
    UIStateSP._enforce_constraints(state)

  def test_kept_with_openpilot_longitudinal_and_cleared_without_it_icbm_or_not(self):
    params = ui_state.params
    try:
      for long, icbm, kept in ((True, False, True), (False, False, False), (False, True, False)):
        params.put_bool("CurveSpeedControl", True, block=True)
        self.enforce(params, long, icbm, icbm)
        assert params.get_bool("CurveSpeedControl") == kept, (long, icbm)
    finally:
      params.put_bool("CurveSpeedControl", False, block=True)
