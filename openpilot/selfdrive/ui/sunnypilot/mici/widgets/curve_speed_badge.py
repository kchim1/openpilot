"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import pyray as rl

from openpilot.selfdrive.ui.sunnypilot.mici.widgets.traffic_mode_badge import TRAFFIC_MODE_BADGE_SIZE, TRAFFIC_MODE_FONT_SIZE
from openpilot.system.ui.lib.text_measure import measure_text_cached

# Curve Speed Control badge: a green circle with a black "C", the colors of the CURVE pill on comma 3/3X, the size and
# letter style of the Traffic Mode badge. Drawn only while curve control is slowing for a curve or the lateral-load
# governor is trimming speed
CURVE_BADGE_COLOR = rl.Color(0, 255, 0, 255)
CURVE_BADGE_TEXT_COLOR = rl.Color(0, 0, 0, 255)


def draw_curve_speed_badge(font: rl.Font, center: rl.Vector2) -> None:
  rl.draw_circle_v(center, TRAFFIC_MODE_BADGE_SIZE / 2, CURVE_BADGE_COLOR)

  # centered with the measured text size, like the "T"
  text_size = measure_text_cached(font, "C", TRAFFIC_MODE_FONT_SIZE)
  rl.draw_text_ex(font, "C", rl.Vector2(center.x - text_size.x / 2, center.y - text_size.y / 2),
                  TRAFFIC_MODE_FONT_SIZE, 0, CURVE_BADGE_TEXT_COLOR)
