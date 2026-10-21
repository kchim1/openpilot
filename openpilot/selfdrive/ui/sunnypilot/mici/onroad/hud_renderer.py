"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.cereal import custom
from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.sunnypilot.onroad.blind_spot_indicators import BlindSpotIndicators
from openpilot.selfdrive.ui.sunnypilot.onroad.lateral_mode import lateral_mode
from openpilot.selfdrive.ui.sunnypilot.mici.widgets.curve_speed_badge import draw_curve_speed_badge
from openpilot.selfdrive.ui.sunnypilot.mici.widgets.traffic_mode_badge import TRAFFIC_MODE_BADGE_SIZE, draw_traffic_mode_badge
from openpilot.selfdrive.ui.ui_state import ui_state

CurveState = custom.LongitudinalPlanSP.SmartCruiseControl.CurveState

# Traffic Mode badge in the top right corner of the road view
TRAFFIC_MODE_BADGE_MARGIN = 20  # the MAX box's visible top and side inset, and the blind spot indicators' side margin
# Curve Speed Control badge: in that corner, or just left of the Traffic Mode badge while it shows
CURVE_BADGE_GAP = 8
# kept this long after the planner last reported curve control working, so a state that flickers for a frame or two
# does not blink the badge, and it still goes once plannerd stops reporting
CURVE_BADGE_HOLD = 1.0  # s


class HudRendererSP(HudRenderer):
  def __init__(self):
    super().__init__()
    self.blind_spot_indicators = BlindSpotIndicators()
    self._curve_working_time = -CURVE_BADGE_HOLD

  def _update_state(self) -> None:
    super()._update_state()
    self.blind_spot_indicators.update()
    lateral_mode.update()
    self.wheel_tint = lateral_mode.wheel_tint

    sm = ui_state.sm
    if sm.updated["longitudinalPlanSP"]:
      curve = sm["longitudinalPlanSP"].smartCruiseControl.curve
      # working: slowing for a curve (its speed target under the truck's), or the governor trimming speed. Not while it
      # only holds a cap the truck is at or under: that includes a truck kept well below it by a car ahead
      if curve.state == CurveState.slowing or curve.governorActive:
        self._curve_working_time = rl.get_time()

  def _render(self, rect: rl.Rectangle) -> None:
    super()._render(rect)
    self.blind_spot_indicators.render(rect)
    self._draw_traffic_mode_badge(rect)
    self._draw_curve_speed_badge(rect)

  def _traffic_mode_badge_shown(self) -> bool:
    # the planner only applies Traffic Mode with openpilot longitudinal. Hidden while an alert is up, like the MAX box:
    # the HUD is drawn after the alerts and would otherwise cover them
    return bool(ui_state.traffic_mode and ui_state.has_longitudinal_control and self._can_draw_top_icons)

  def _draw_traffic_mode_badge(self, rect: rl.Rectangle) -> None:
    if not self._traffic_mode_badge_shown():
      return

    radius = TRAFFIC_MODE_BADGE_SIZE / 2
    center = rl.Vector2(rect.x + rect.width - TRAFFIC_MODE_BADGE_MARGIN - radius, rect.y + TRAFFIC_MODE_BADGE_MARGIN + radius)
    draw_traffic_mode_badge(self._font_bold, center)

  def _draw_curve_speed_badge(self, rect: rl.Rectangle) -> None:
    # curve control only acts with openpilot longitudinal. Hidden while an alert is up, like the Traffic Mode badge, and
    # never on top of it
    working = rl.get_time() - self._curve_working_time < CURVE_BADGE_HOLD
    if not (working and ui_state.has_longitudinal_control and self._can_draw_top_icons):
      return

    radius = TRAFFIC_MODE_BADGE_SIZE / 2
    x = rect.x + rect.width - TRAFFIC_MODE_BADGE_MARGIN - radius
    if self._traffic_mode_badge_shown():
      x -= TRAFFIC_MODE_BADGE_SIZE + CURVE_BADGE_GAP
    draw_curve_speed_badge(self._font_bold, rl.Vector2(x, rect.y + TRAFFIC_MODE_BADGE_MARGIN + radius))

  def _has_blind_spot_detected(self) -> bool:

    return self.blind_spot_indicators.detected
