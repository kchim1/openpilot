"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.selfdrive.ui.mici.onroad.hud_renderer import HudRenderer
from openpilot.selfdrive.ui.sunnypilot.onroad.blind_spot_indicators import BlindSpotIndicators
from openpilot.selfdrive.ui.sunnypilot.onroad.lateral_mode import lateral_mode
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Traffic Mode badge: a solid red circle with a white "T" in the top right corner of the road view
TRAFFIC_MODE_BADGE_COLOR = rl.Color(201, 34, 49, 255)  # openpilot's red (sidebar danger, lead chevron)
TRAFFIC_MODE_BADGE_SIZE = 56  # between the 50 px steering wheel and the 60 px driver monitoring icon
TRAFFIC_MODE_BADGE_MARGIN = 20  # the MAX box's visible top and side inset, and the blind spot indicators' side margin
TRAFFIC_MODE_FONT_SIZE = 40  # "T" about half as tall as the badge


class HudRendererSP(HudRenderer):
  def __init__(self):
    super().__init__()
    self.blind_spot_indicators = BlindSpotIndicators()

  def _update_state(self) -> None:
    super()._update_state()
    self.blind_spot_indicators.update()
    lateral_mode.update()
    self.wheel_tint = lateral_mode.wheel_tint

  def _render(self, rect: rl.Rectangle) -> None:
    super()._render(rect)
    self.blind_spot_indicators.render(rect)
    self._draw_traffic_mode_badge(rect)

  def _draw_traffic_mode_badge(self, rect: rl.Rectangle) -> None:
    # the planner only applies Traffic Mode with openpilot longitudinal. Hidden while an alert is up, like the MAX box:
    # the HUD is drawn after the alerts and would otherwise cover them
    if not (ui_state.traffic_mode and ui_state.has_longitudinal_control and self._can_draw_top_icons):
      return

    radius = TRAFFIC_MODE_BADGE_SIZE / 2
    center = rl.Vector2(rect.x + rect.width - TRAFFIC_MODE_BADGE_MARGIN - radius, rect.y + TRAFFIC_MODE_BADGE_MARGIN + radius)
    rl.draw_circle_v(center, radius, TRAFFIC_MODE_BADGE_COLOR)

    # Inter's capitals sit in the middle of the line box, so centering the measured text centers the "T"
    text_size = measure_text_cached(self._font_bold, "T", TRAFFIC_MODE_FONT_SIZE)
    rl.draw_text_ex(self._font_bold, "T", rl.Vector2(center.x - text_size.x / 2, center.y - text_size.y / 2),
                    TRAFFIC_MODE_FONT_SIZE, 0, rl.WHITE)

  def _has_blind_spot_detected(self) -> bool:

    return self.blind_spot_indicators.detected
