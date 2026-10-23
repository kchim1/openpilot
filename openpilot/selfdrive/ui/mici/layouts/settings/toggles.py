from collections.abc import Callable

from openpilot.cereal import log
from openpilot.common.params import UnknownKeyName

from openpilot.system.ui.widgets.scroller import NavScroller
from openpilot.selfdrive.ui.mici.widgets.button import BigParamControl, BigMultiParamToggle, BigToggle, GreyBigButton
from openpilot.selfdrive.ui.mici.widgets.dialog import BigConfirmationCircleButton
from openpilot.system.ui.lib.application import gui_app
from openpilot.selfdrive.ui.layouts.settings.common import restart_needed_callback
from openpilot.selfdrive.ui.ui_state import ui_state

PERSONALITY_TO_INT = log.LongitudinalPersonality.schema.enumerants


class ExperimentalModeConfirmPage(NavScroller):
  def __init__(self, on_confirm: Callable[[], None]):
    super().__init__()

    accept = BigConfirmationCircleButton("enable\nexperimental mode",
                                         gui_app.texture("icons_mici/setup/driver_monitoring/dm_check.png", 64, 64),
                                         lambda: self.dismiss(on_confirm))

    self._scroller.add_widgets([
      GreyBigButton("enabling\nexperimental mode", "scroll to continue",
                    gui_app.texture("icons_mici/setup/warning.png", 64, 64)),
      GreyBigButton("", "openpilot defaults to driving in chill mode."),
      GreyBigButton("", "Experimental mode enables alpha-level features that aren't ready for chill mode."),
      GreyBigButton("End-to-End Longitudinal Control"),
      GreyBigButton("", "Let the driving model control the gas and brakes."),
      GreyBigButton("", "openpilot will drive as it thinks a human would, including stopping for red lights and stop signs."),
      GreyBigButton("", "The set speed will only act as an upper bound."),
      GreyBigButton("", "This is an alpha quality feature; mistakes should be expected."),
      GreyBigButton("New Driving Visualization"),
      GreyBigButton("", "The path will change colors to communicate acceleration intent."),
      GreyBigButton("", "Red for braking, green for acceleration, and gray for coasting."),
      accept,
    ])


class TogglesLayoutMici(NavScroller):
  def __init__(self):
    super().__init__()

    self._personality_toggle = BigMultiParamToggle("driving personality", "LongitudinalPersonality", ["aggressive", "standard", "relaxed"],
                                                   description="Standard is recommended.\n" +
                                                               "Aggressive follows closer, with firmer gas and braking.\n" +
                                                               "Relaxed leaves more space.\n" +
                                                               "Use the steering wheel distance button on supported cars.\n" +
                                                               "On a Rivian, following distance sets the gap;\n" +
                                                               "this sets only the gas and brake feel.")
    self._experimental_btn = BigToggle("experimental mode", description_icon=gui_app.texture("icons_mici/experimental_mode.png", 64, 64),
                                       initial_state=ui_state.params.get_bool("ExperimentalMode"), toggle_callback=self._on_experimental_mode,
                                       description="Let the driving model control gas and brakes.\n" +
                                                   "Includes stopping for red lights and stop signs.\n" +
                                                   "Set speed is a maximum, not a target.\n" +
                                                   "These are alpha features. Expect mistakes.\n" +
                                                   "The path colors show acceleration and braking.")
    # left out on a prebuilt whose params library does not know the setting yet
    try:
      self._follow_distance_toggle = BigMultiParamToggle("following distance", "RivianFollowDistance", ["closer", "close", "medium", "far"],
                                                         description="How far behind the car ahead it follows.\n" +
                                                                     "Medium is the old Aggressive gap.\n" +
                                                                     "Closer and Close also react faster to braking.\n" +
                                                                     "Roll the right scroll wheel up for farther,\n" +
                                                                     "down for closer, while engaged or stopped.\n" +
                                                                     "Rivian with sunnypilot longitudinal only.")
      self._follow_distance_toggle.set_visible(False)  # shown once CarParams say Rivian with longitudinal control
    except UnknownKeyName:
      self._follow_distance_toggle = None
    self._follow_distance = ui_state.follow_distance
    try:
      self._traffic_mode_toggle = BigParamControl("traffic mode", "TrafficMode",
                                                  description="Shorter gap for stop-and-go traffic.\n" +
                                                              "About 0.6 s when nearly stopped, 0.8 s at 11-20 mph,\n" +
                                                              "0.9 s at 30 mph, 1.0 s from 40 mph up as before.\n" +
                                                              "Moves off sooner when the car ahead creeps away\n" +
                                                              "and follows one that only inches forward.\n" +
                                                              "Accelerates gently (1.5 m/s^2 up to 30 mph).\n" +
                                                              "Reacts sooner when the car ahead brakes.\n" +
                                                              "Never farther back than following distance.\n" +
                                                              "Hold the right scroll wheel in 1 s to toggle,\n" +
                                                              "or tap the T on the home screen when off-road.\n" +
                                                              "Stays on or off until you change it.")
    except UnknownKeyName:
      self._traffic_mode_toggle = None
    self._traffic_mode = ui_state.traffic_mode
    try:
      self._launch_assist_toggle = BigParamControl("launch assist", "RivianLaunchAssist",
                                                   description="Starts moving sooner when the stopped car ahead pulls away.\n" +
                                                               "Briefly requests 0.5 m/s^2 to clear the dead zone.\n" +
                                                               "Then eases back to the normal plan.\n" +
                                                               "Acceleration is unchanged once moving.\n" +
                                                               "May also follow a car that only inches forward.\n" +
                                                               "Rivian with sunnypilot longitudinal only.")
      self._launch_assist_toggle.set_visible(False)  # shown once CarParams say Rivian with longitudinal control
    except UnknownKeyName:
      self._launch_assist_toggle = None
    try:
      self._curve_speed_toggle = BigParamControl("curve speed control", "CurveSpeedControl",
                                                 description="Slows down ahead of curves from the model's path.\n" +
                                                             "Holds a steady speed through the bend, then speeds up.\n" +
                                                             "Slows and eases off the gas when cornering\n" +
                                                             "nears what the steering can hold.\n" +
                                                             "A green C shows on the road view while it slows.\n" +
                                                             "Can be changed while driving.")
    except UnknownKeyName:
      self._curve_speed_toggle = None
    is_metric_toggle = BigParamControl("use metric units", "IsMetric")
    ldw_toggle = BigParamControl("lane departure warnings", "IsLdwEnabled",
                                 description="Warn when you drift across a detected lane line.\n" +
                                             "Only above 31 mph (50 km/h), with no turn signal.")
    always_on_dm_toggle = BigParamControl("always-on driver monitor", "AlwaysOnDM", description="Monitor the driver even when sunnypilot is not engaged.")
    record_front = BigParamControl("record & upload cabin camera", "RecordFront",
                                   description_icon=gui_app.texture("icons_mici/settings/device/cameras.png", 64, 64),
                                   toggle_callback=restart_needed_callback, description="Upload cabin camera data to help improve driver monitoring.")
    record_mic = BigParamControl("record & upload mic audio", "RecordAudio", description_icon=gui_app.texture("icons_mici/microphone.png", 64, 64),
                                 toggle_callback=restart_needed_callback,
                                 description="Record microphone audio while driving.\n" +
                                             "Audio is included in dashcam videos in sunnylink.")
    enable_openpilot = BigParamControl("enable sunnypilot", "OpenpilotEnabledToggle", toggle_callback=restart_needed_callback,
                                       description="Enable to use sunnypilot driver assistance.\n" +
                                                   "Disable to use your car's stock driver assistance.")


    self._scroller.add_widgets([
      self._personality_toggle,
      self._experimental_btn,
      *([self._follow_distance_toggle] if self._follow_distance_toggle is not None else []),
      *([self._traffic_mode_toggle] if self._traffic_mode_toggle is not None else []),
      *([self._launch_assist_toggle] if self._launch_assist_toggle is not None else []),
      *([self._curve_speed_toggle] if self._curve_speed_toggle is not None else []),
      is_metric_toggle,
      ldw_toggle,
      always_on_dm_toggle,
      record_front,
      record_mic,
      enable_openpilot,
    ])

    # Toggle lists
    self._refresh_toggles = (
      ("ExperimentalMode", self._experimental_btn),
      ("IsMetric", is_metric_toggle),
      ("IsLdwEnabled", ldw_toggle),
      ("AlwaysOnDM", always_on_dm_toggle),
      ("RecordFront", record_front),
      ("RecordAudio", record_mic),
      ("OpenpilotEnabledToggle", enable_openpilot),
    ) + ((("TrafficMode", self._traffic_mode_toggle),) if self._traffic_mode_toggle is not None else ()) \
      + ((("RivianLaunchAssist", self._launch_assist_toggle),) if self._launch_assist_toggle is not None else ()) \
      + ((("CurveSpeedControl", self._curve_speed_toggle),) if self._curve_speed_toggle is not None else ())

    enable_openpilot.set_enabled(lambda: not ui_state.engaged)
    record_front.set_enabled(False if ui_state.params.get_bool("RecordFrontLock") else (lambda: not ui_state.engaged))
    record_mic.set_enabled(lambda: not ui_state.engaged)

    if ui_state.params.get_bool("ShowDebugInfo"):
      gui_app.set_show_touches(True)
      gui_app.set_show_fps(True)

    ui_state.add_engaged_transition_callback(self._update_toggles)

  def _update_state(self):
    super()._update_state()

    if ui_state.sm.updated["selfdriveState"]:
      personality = PERSONALITY_TO_INT[ui_state.sm["selfdriveState"].personality]
      if personality != ui_state.personality and ui_state.started:
        self._personality_toggle.set_value(self._personality_toggle._options[personality])
      ui_state.personality = personality

    # mirror Traffic Mode changes made elsewhere while this page is open: steering wheel, sunnylink
    if self._traffic_mode_toggle is not None and ui_state.traffic_mode != self._traffic_mode:
      self._traffic_mode = ui_state.traffic_mode
      self._traffic_mode_toggle.set_checked(self._traffic_mode)

    # and Following Distance changes: the right scroll wheel, sunnylink
    if self._follow_distance_toggle is not None and ui_state.follow_distance != self._follow_distance:
      self._follow_distance = ui_state.follow_distance
      self._follow_distance_toggle.set_value(self._follow_distance_toggle._options[self._follow_distance])

  def show_event(self):
    super().show_event()
    self._update_toggles()

  def _update_toggles(self):
    ui_state.update_params()

    # CP gating for experimental mode
    if ui_state.CP is not None:
      if ui_state.has_longitudinal_control:
        self._experimental_btn.set_visible(True)
        self._personality_toggle.set_visible(True)
      else:
        # no long for now
        self._experimental_btn.set_visible(False)
        self._experimental_btn.set_checked(False)
        self._personality_toggle.set_visible(False)
        ui_state.params.remove("ExperimentalMode")
      if self._traffic_mode_toggle is not None:
        self._traffic_mode_toggle.set_visible(ui_state.has_longitudinal_control)
      if self._follow_distance_toggle is not None:
        self._follow_distance_toggle.set_visible(ui_state.has_longitudinal_control and ui_state.CP.brand == "rivian")
      if self._launch_assist_toggle is not None:
        self._launch_assist_toggle.set_visible(ui_state.has_longitudinal_control and ui_state.CP.brand == "rivian")
      if self._curve_speed_toggle is not None:
        self._curve_speed_toggle.set_visible(ui_state.has_longitudinal_control)

    # Refresh toggles from params to mirror external changes
    for key, item in self._refresh_toggles:
      item.set_checked(ui_state.params.get_bool(key))
    if self._follow_distance_toggle is not None:
      self._follow_distance = ui_state.follow_distance
      self._follow_distance_toggle.set_value(self._follow_distance_toggle._options[self._follow_distance])

  def _on_experimental_mode(self, state: bool):
    if state and not ui_state.params.get_bool("ExperimentalModeConfirmed"):
      # Don't show enabled state until confirm
      self._experimental_btn.set_checked(False)

      def on_confirm():
        ui_state.params.put_bool("ExperimentalModeConfirmed", True)
        ui_state.params.put_bool("ExperimentalMode", True)
        self._experimental_btn.set_checked(True)

      gui_app.push_widget(ExperimentalModeConfirmPage(on_confirm))
    else:
      ui_state.params.put_bool("ExperimentalMode", state)
