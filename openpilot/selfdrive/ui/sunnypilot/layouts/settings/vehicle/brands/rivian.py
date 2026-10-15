"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp

LAUNCH_ASSIST_DESCRIPTION = tr_noop("Starts the truck moving sooner when the car ahead pulls away from a stop. After a radar lead has been " +
                                    "stopped ahead for a second and then starts moving away, sunnypilot briefly requests 0.5 m/s^2 " +
                                    "(at most 0.7 s) to get past the truck's standstill dead zone, then eases back to the normal plan. " +
                                    "Once moving it accelerates as before. Can be changed while driving.")


class RivianSettings(BrandSettings):
  def __init__(self):
    super().__init__()
    # left out on a prebuilt whose params library does not know the setting yet
    self.launch_assist_toggle = None
    if b"RivianLaunchAssist" in ui_state.params.all_keys():
      self.launch_assist_toggle = toggle_item_sp(tr("Launch Assist"), tr(LAUNCH_ASSIST_DESCRIPTION), param="RivianLaunchAssist")
      self._launch_assist = ui_state.params.get_bool("RivianLaunchAssist")
      self.items = [self.launch_assist_toggle]

  def update_settings(self):
    if self.launch_assist_toggle is None:
      return

    long_enabled = ui_state.has_longitudinal_control
    self.launch_assist_toggle.action_item.set_enabled(long_enabled)
    desc = tr(LAUNCH_ASSIST_DESCRIPTION)
    if not long_enabled:
      desc = f"<b>{tr('This feature is unavailable because sunnypilot Longitudinal Control (Alpha) is not enabled.')}</b><br><br>{desc}"
    self.launch_assist_toggle.set_description(desc)

    # mirror changes made in sunnylink while this page is open
    launch_assist = ui_state.params.get_bool("RivianLaunchAssist")
    if launch_assist != self._launch_assist:
      self._launch_assist = launch_assist
      self.launch_assist_toggle.action_item.set_state(launch_assist)
