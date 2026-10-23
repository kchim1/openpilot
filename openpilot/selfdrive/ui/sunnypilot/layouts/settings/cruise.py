"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from openpilot.selfdrive.ui.sunnypilot.layouts.settings.cruise_sub_layouts.speed_limit_settings import SpeedLimitSettingsLayout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp, option_item_sp, simple_button_item_sp
from openpilot.system.ui.widgets import Widget
from openpilot.system.ui.widgets.scroller_tici import Scroller


class PanelType(IntEnum):
  CRUISE = 0
  SLA = 1


ICBM_DESC = tr_noop("When enabled, sunnypilot will attempt to manage the built-in cruise control buttons " +
                    "by emulating button presses for limited longitudinal control.")
ICMB_UNAVAILABLE = tr_noop("Intelligent Cruise Button Management is currently unavailable on this platform.")
ICMB_UNAVAILABLE_LONG_AVAILABLE = tr_noop("Disable the sunnypilot Longitudinal Control (alpha) toggle to allow Intelligent Cruise Button Management.")
ICMB_UNAVAILABLE_LONG_UNAVAILABLE = tr_noop("sunnypilot Longitudinal Control is the default longitudinal control for this platform.")

ACC_ENABLED_DESCRIPTION = tr_noop("Enable custom Short & Long press increments for cruise speed increase/decrease.")
ACC_NOLONG_DESCRIPTION = tr_noop("This feature can only be used with sunnypilot longitudinal control enabled.")
ACC_PCMCRUISE_DISABLED_DESCRIPTION = tr_noop("This feature is not supported on this platform due to vehicle limitations.")
ONROAD_ONLY_DESCRIPTION = tr_noop("Start the vehicle to check vehicle compatibility.")
FOLLOWING_TIME_OFFSET_DESCRIPTION = tr_noop("Adjusts the gap to the car ahead in every driving personality and Following Distance, " +
                                            "from -10% (closer) to +10% (further). In Experimental mode the driving model " +
                                            "can keep its own gap, so the change may be smaller. Takes effect on the next drive.")
TRAFFIC_MODE_DESCRIPTION = tr_noop("Shorter gap for stop-and-go traffic: about 0.6 s behind the car ahead when nearly stopped, " +
                                   "0.8 s from about 11 to 20 mph, 0.9 s at 30 mph and 1.0 s from 40 mph up, as before " +
                                   "(Aggressive and Following Distance Medium are 1.25 s). Stopped behind a car, it moves off sooner " +
                                   "when that car creeps away and follows a car that only inches forward, and it still stops about " +
                                   "4 m (13 ft) behind a stopped car. It accelerates more gently, at most 1.5 m/s² up to 30 mph, and " +
                                   "it reacts sooner when the car ahead brakes. Its own gap ignores Following Distance Offset, but " +
                                   "on a Rivian it never follows farther back than Following Distance with the offset. Press and " +
                                   "hold the right scroll wheel on the steering wheel (its middle button) for 1 second to toggle. " +
                                   "Stays on or off until you change it.")
CURVE_SPEED_CONTROL_DESCRIPTION = tr_noop("Slow down ahead of curves and power smoothly through them. Also slows and eases off the " +
                                          "gas when cornering nears what the steering can hold. Uses the vision path; no map data " +
                                          "required.")


def _following_time_offset_label(value: int) -> str:
  if value == 0:
    return tr("Default")
  return f"+{value}%" if value > 0 else f"{value}%"


class CruiseLayout(Widget):
  def __init__(self):
    super().__init__()
    self._current_panel = PanelType.CRUISE
    self._speed_limit_layout = SpeedLimitSettingsLayout(lambda: self._set_current_panel(PanelType.CRUISE))

    items = self._initialize_items()
    self._scroller = Scroller(items, line_separator=True, spacing=0)

  def _initialize_items(self):

    self.icbm_toggle = toggle_item_sp(
      title=tr("Intelligent Cruise Button Management (ICBM) (Alpha)"),
      description="",
      param="IntelligentCruiseButtonManagement")

    self.scc_v_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Vision"),
      description=tr("Use vision path predictions to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlVision")

    self.scc_m_toggle = toggle_item_sp(
      title=tr("Smart Cruise Control - Map"),
      description=tr("Use map data to estimate the appropriate speed to drive through turns ahead."),
      param="SmartCruiseControlMap")

    # hidden, and not tied to the param, on a prebuilt whose params library does not know the setting yet
    self._curve_speed_known = b"CurveSpeedControl" in ui_state.params.all_keys()
    self.curve_speed_toggle = toggle_item_sp(
      title=tr("Curve Speed Control"),
      description=tr(CURVE_SPEED_CONTROL_DESCRIPTION),
      param="CurveSpeedControl" if self._curve_speed_known else None)
    self.curve_speed_toggle.set_visible(self._curve_speed_known)

    self.custom_acc_toggle = toggle_item_sp(
      title=tr("Custom ACC Speed Increments"),
      description="",
      param="CustomAccIncrementsEnabled",
      callback=self._on_custom_acc_toggle)

    self.custom_acc_short_increment = option_item_sp(
      title=tr("Short Press Increment"),
      param="CustomAccShortPressIncrement",
      min_value=1, max_value=10, value_change_step=1,
      inline=True)

    self.custom_acc_long_increment = option_item_sp(
      title=tr("Long Press Increment"),
      param="CustomAccLongPressIncrement",
      value_map={1: 1, 2: 5, 3: 10},
      min_value=1, max_value=3, value_change_step=1,
      inline=True)

    self.following_time_offset = option_item_sp(
      title=tr("Following Distance Offset"),
      description=tr(FOLLOWING_TIME_OFFSET_DESCRIPTION),
      param="SPFollowingTimeOffset",
      min_value=-10, max_value=10, value_change_step=2,
      label_callback=_following_time_offset_label,
      inline=True)

    # hidden, and not tied to the param, on a prebuilt whose params library does not know the setting yet
    traffic_mode_known = b"TrafficMode" in ui_state.params.all_keys()
    self.traffic_mode_toggle = toggle_item_sp(
      title=tr("Traffic Mode"),
      description=tr(TRAFFIC_MODE_DESCRIPTION),
      param="TrafficMode" if traffic_mode_known else None)
    self.traffic_mode_toggle.set_visible(traffic_mode_known)
    self._traffic_mode = ui_state.traffic_mode

    self.sla_settings_button = simple_button_item_sp(
      button_text=lambda: tr("Speed Limit"),
      button_width=800,
      callback=lambda: self._set_current_panel(PanelType.SLA)
    )

    self.dec_toggle = toggle_item_sp(
      title=tr("Enable Dynamic Experimental Control"),
      description=tr("Enable toggle to allow the model to determine when to use sunnypilot ACC or sunnypilot End to End Longitudinal."),
      param="DynamicExperimentalControl")

    items = [
      self.icbm_toggle,
      self.dec_toggle,
      self.scc_v_toggle,
      self.scc_m_toggle,
      self.curve_speed_toggle,
      self.custom_acc_toggle,
      self.custom_acc_short_increment,
      self.custom_acc_long_increment,
      self.following_time_offset,
      self.traffic_mode_toggle,
      self.sla_settings_button,
    ]
    return items

  def _render(self, rect):
    if self._current_panel == PanelType.SLA:
      self._speed_limit_layout.render(rect)
    else:
      self._scroller.render(rect)

  def show_event(self):
    self._set_current_panel(PanelType.CRUISE)
    self._scroller.show_event()
    self.icbm_toggle.show_description(True)
    self.custom_acc_toggle.show_description(True)

  def _set_current_panel(self, panel: PanelType):
    self._current_panel = panel
    if panel == PanelType.SLA:
      self._speed_limit_layout.show_event()

  def _update_state(self):
    super()._update_state()

    if ui_state.CP is not None and ui_state.CP_SP is not None:
      has_icbm = ui_state.has_icbm
      has_long = ui_state.has_longitudinal_control

      if ui_state.CP_SP.intelligentCruiseButtonManagementAvailable and not has_long:
        self.icbm_toggle.action_item.set_enabled(ui_state.is_offroad())
        self.icbm_toggle.set_description(tr(ICBM_DESC))
      else:
        ui_state.params.remove("IntelligentCruiseButtonManagement")
        self.icbm_toggle.action_item.set_enabled(False)

        long_desc = ICMB_UNAVAILABLE
        if has_long:
          if ui_state.CP.alphaLongitudinalAvailable:
            long_desc += " " + ICMB_UNAVAILABLE_LONG_AVAILABLE
          else:
            long_desc += " " + ICMB_UNAVAILABLE_LONG_UNAVAILABLE

        new_desc = "<b>" + tr(long_desc) + "</b>\n\n" + tr(ICBM_DESC)
        if self.icbm_toggle.description != new_desc:
          self.icbm_toggle.set_description(new_desc)
          self.icbm_toggle.show_description(True)

      if has_long or has_icbm:
        self.custom_acc_toggle.action_item.set_enabled(((has_long and not ui_state.CP.pcmCruise) or has_icbm) and ui_state.is_offroad())
        self.dec_toggle.action_item.set_enabled(has_long)
        self.scc_v_toggle.action_item.set_enabled(True)
        self.scc_m_toggle.action_item.set_enabled(True)
        # Curve Speed Control only feeds the longitudinal planner, so it only does anything with openpilot longitudinal
        # active: gated like the DEC toggle (and like the comma four toggle, which is hidden without it)
        self.curve_speed_toggle.action_item.set_enabled(has_long)
        self.following_time_offset.action_item.set_enabled(has_long and ui_state.is_offroad())
        self.traffic_mode_toggle.action_item.set_enabled(has_long)
      else:
        ui_state.params.remove("CustomAccIncrementsEnabled")
        ui_state.params.remove("DynamicExperimentalControl")
        ui_state.params.remove("SmartCruiseControlVision")
        ui_state.params.remove("SmartCruiseControlMap")
        if self._curve_speed_known:
          ui_state.params.remove("CurveSpeedControl")
        self.custom_acc_toggle.action_item.set_enabled(False)
        self.dec_toggle.action_item.set_enabled(False)
        self.scc_v_toggle.action_item.set_enabled(False)
        self.scc_m_toggle.action_item.set_enabled(False)
        self.curve_speed_toggle.action_item.set_enabled(False)
        self.following_time_offset.action_item.set_enabled(False)
        self.traffic_mode_toggle.action_item.set_enabled(False)

    else:
      has_icbm = has_long = False
      self.icbm_toggle.action_item.set_enabled(False)
      self.following_time_offset.action_item.set_enabled(False)
      self.traffic_mode_toggle.action_item.set_enabled(False)
      self.icbm_toggle.set_description(tr(ONROAD_ONLY_DESCRIPTION))

    # mirror Traffic Mode changes made elsewhere: steering wheel, sunnylink
    if ui_state.traffic_mode != self._traffic_mode:
      self._traffic_mode = ui_state.traffic_mode
      self.traffic_mode_toggle.action_item.set_state(self._traffic_mode)

    show_custom_acc_desc = False

    if ui_state.is_offroad():
      new_custom_acc_desc = tr(ONROAD_ONLY_DESCRIPTION)
      show_custom_acc_desc = True
    else:
      if has_long or has_icbm:
        if has_long and ui_state.CP.pcmCruise:
          new_custom_acc_desc = tr(ACC_PCMCRUISE_DISABLED_DESCRIPTION)
          show_custom_acc_desc = True
        else:
          new_custom_acc_desc = tr(ACC_ENABLED_DESCRIPTION)
      else:
        new_custom_acc_desc = tr(ACC_NOLONG_DESCRIPTION)
        show_custom_acc_desc = True
        self.custom_acc_toggle.action_item.set_state(False)

    if self.custom_acc_toggle.description != new_custom_acc_desc:
      self.custom_acc_toggle.set_description(new_custom_acc_desc)
      if show_custom_acc_desc:
        self.custom_acc_toggle.show_description(True)

    self._on_custom_acc_toggle(self.custom_acc_toggle.action_item.get_state())

  def _on_custom_acc_toggle(self, state):
    self.custom_acc_short_increment.set_visible(state)
    self.custom_acc_long_increment.set_visible(state)
    self.custom_acc_short_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
    self.custom_acc_long_increment.action_item.set_enabled(self.custom_acc_toggle.action_item.enabled)
