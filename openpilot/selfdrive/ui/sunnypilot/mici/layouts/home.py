"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import math

import pyray as rl

from openpilot.common.params import UnknownKeyName
from openpilot.common.swaglog import cloudlog
from openpilot.selfdrive.ui.mici.layouts.home import MiciHomeLayout
from openpilot.selfdrive.ui.sunnypilot.mici.widgets.traffic_mode_badge import TRAFFIC_MODE_BADGE_SIZE, draw_traffic_mode_badge
from openpilot.selfdrive.ui.ui_state import ui_state, ChestnutState
from openpilot.system.ui.lib.application import FontWeight, MousePos, gui_app
from openpilot.system.ui.widgets.label import UnifiedLabel

# Traffic Mode button on the right, between the title and the status bar. Taps within this distance of the badge count
TRAFFIC_MODE_BUTTON_MARGIN = 20  # same right inset as the road view badge
TRAFFIC_MODE_BUTTON_CENTER_Y = 130
TRAFFIC_MODE_TOUCH_PADDING = 16


class MiciHomeLayoutSP(MiciHomeLayout):
  def __init__(self):
    super().__init__()
    self._openpilot_label = UnifiedLabel("sunnypilot", font_size=88, font_weight=FontWeight.AUDIOWIDE, max_width=480, wrap_text=False)

    self._traffic_mode_font = gui_app.font(FontWeight.BOLD)
    self._traffic_mode_press_on_button = False

  def _set_chestnut_visibility(self):
    usb_connected = ui_state.usb_connected
    usb_unknown = ui_state.usb_unknown
    chestnut_state = ui_state.chestnut_state
    loading = chestnut_state == ChestnutState.LOADING

    self._usb_icon.set_visible(usb_connected and usb_unknown)
    self._chestnut_loading_icon.set_opacity(0.35 + 0.65 * (0.5 - 0.5 * math.cos(rl.get_time() * 6.0)))
    self._chestnut_loading_icon.set_visible(not usb_unknown and loading)
    self._chestnut_icon.set_visible(not usb_unknown and not loading and
                                    chestnut_state in (ChestnutState.READY, ChestnutState.ACTIVE))
    self._chestnut_failed_icon.set_visible(not usb_unknown and chestnut_state in (ChestnutState.UNCOMPILED, ChestnutState.FAILED))

  # ***** Traffic Mode button: offroad with the ignition off, so it can never be switched by touch while driving *****
  # (onroad, a tap on the road view brings this screen up. openpilot can also be offroad while the car is driven:
  # Always Offroad, an overheating device, a panda timeout. The ignition covers those, for Rivian it is
  # VDM_EpasPowerMode == Drive_On, and it keeps its last value if the panda stops reporting)
  @staticmethod
  def _traffic_mode_button_visible() -> bool:
    return not ui_state.started and not ui_state.ignition and ui_state.has_longitudinal_control and not ui_state.is_body

  def _traffic_mode_center(self) -> rl.Vector2:
    return rl.Vector2(self.rect.x + self.rect.width - TRAFFIC_MODE_BUTTON_MARGIN - TRAFFIC_MODE_BADGE_SIZE / 2,
                      self.rect.y + TRAFFIC_MODE_BUTTON_CENTER_Y)

  def _on_traffic_mode_button(self, mouse_pos: MousePos) -> bool:
    if not self._traffic_mode_button_visible():
      return False
    center = self._traffic_mode_center()
    return math.hypot(mouse_pos.x - center.x, mouse_pos.y - center.y) <= TRAFFIC_MODE_BADGE_SIZE / 2 + TRAFFIC_MODE_TOUCH_PADDING

  def _toggle_traffic_mode(self) -> None:
    traffic_mode = not ui_state.traffic_mode
    try:
      ui_state.params.put_bool("TrafficMode", traffic_mode, block=True)
    except UnknownKeyName:
      # a prebuilt whose params library does not know the setting yet
      cloudlog.exception("TrafficMode write failed, tap ignored")
      return
    ui_state.traffic_mode = traffic_mode

  def _handle_mouse_press(self, mouse_pos: MousePos):
    super()._handle_mouse_press(mouse_pos)
    self._traffic_mode_press_on_button = self._on_traffic_mode_button(mouse_pos)

  def _handle_mouse_release(self, mouse_pos: MousePos):
    # a tap that starts and ends on the button toggles Traffic Mode instead of opening settings.
    # A long press anywhere still toggles experimental mode, as before
    on_button = self._traffic_mode_press_on_button and self._on_traffic_mode_button(mouse_pos)
    self._traffic_mode_press_on_button = False
    if on_button and not self._did_long_press:
      self._toggle_traffic_mode()
      return
    super()._handle_mouse_release(mouse_pos)

  def _render(self, rect):
    super()._render(rect)
    if self._traffic_mode_button_visible():
      draw_traffic_mode_badge(self._traffic_mode_font, self._traffic_mode_center(), ui_state.traffic_mode)
