"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.system.ui.lib.text_measure import measure_text_cached

# Traffic Mode badge: a solid red circle with a white "T" when on (road view and offroad home screen),
# a grey circle with a grey "T" when off (offroad home screen only)
TRAFFIC_MODE_BADGE_COLOR = rl.Color(201, 34, 49, 255)  # openpilot's red (sidebar danger, lead chevron)
TRAFFIC_MODE_BADGE_OFF_COLOR = rl.Color(60, 60, 60, 255)
TRAFFIC_MODE_TEXT_OFF_COLOR = rl.Color(150, 150, 150, 255)
TRAFFIC_MODE_BADGE_SIZE = 56  # between the 50 px steering wheel and the 60 px driver monitoring icon
TRAFFIC_MODE_FONT_SIZE = 40  # "T" about half as tall as the badge


def draw_traffic_mode_badge(font: rl.Font, center: rl.Vector2, on: bool = True) -> None:
  rl.draw_circle_v(center, TRAFFIC_MODE_BADGE_SIZE / 2, TRAFFIC_MODE_BADGE_COLOR if on else TRAFFIC_MODE_BADGE_OFF_COLOR)

  # Inter's capitals sit in the middle of the line box, so centering the measured text centers the "T"
  text_size = measure_text_cached(font, "T", TRAFFIC_MODE_FONT_SIZE)
  rl.draw_text_ex(font, "T", rl.Vector2(center.x - text_size.x / 2, center.y - text_size.y / 2),
                  TRAFFIC_MODE_FONT_SIZE, 0, rl.WHITE if on else TRAFFIC_MODE_TEXT_OFF_COLOR)
