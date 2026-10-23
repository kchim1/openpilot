"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from openpilot.selfdrive.ui.sunnypilot.layouts.settings.vehicle.brands.base import BrandSettings
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.multilang import tr, tr_noop
from openpilot.system.ui.sunnypilot.widgets.list_view import toggle_item_sp

LAUNCH_ASSIST_DESCRIPTION = tr_noop("A quicker start in the first second of a launch from a stop. Once the truck has been stopped " +
                                    "behind a radar-tracked car for a second and that car pulls away, sunnypilot asks for at least as much " +
                                    "as that car pulls away with, from 0.5 m/s^2 up to 0.8, 0.7 or 0.6 m/s^2 (Aggressive, Standard, " +
                                    "Relaxed), 0.1 less on Gentle Launch Assist Strength (set on sunnylink or the comma four), until " +
                                    "the truck is under way, then eases back into the normal plan; sooner once the plan asks for as " +
                                    "much or when the radar loses that car or switches to another. It never takes the truck faster " +
                                    "than that car. With no car ahead it adds a smaller lift, never more than the driving model plans. " +
                                    "Behind a car it doesn't lift a launch in Traffic Mode, which moves off sooner on its own when that " +
                                    "car creeps away, or after stopping closer than 3.5 m (11 ft) to it. sunnypilot's own stop always " +
                                    "wins. If that car pulls away and then stops at once, the truck can end up a little closer to it, " +
                                    "up to about 0.7 m. Can be changed while driving.")


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
