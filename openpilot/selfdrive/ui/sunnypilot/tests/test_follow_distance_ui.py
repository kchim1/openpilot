"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Following Distance on the comma four toggles page: shown only for a Rivian with openpilot longitudinal, left out on a
params library without the key, mirrors changes made with the wheel or sunnylink, and a tap steps closer, close, medium,
far.
The page is built headless: textures and fonts are stubbed and nothing is drawn.
"""
import time
from types import SimpleNamespace

import pyray as rl

from opendbc.car.structs import car
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.mici.layouts.settings import toggles as toggles_layout
from openpilot.selfdrive.ui.sunnypilot.ui_state import UIStateSP
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.sunnypilot.selfdrive.controls.lib import follow_distance
from openpilot.system.ui.lib.application import MousePos, gui_app
from openpilot.system.ui.widgets.scroller import NavScroller

PARAM = "RivianFollowDistance"


class StaleParams:
  """a params library built before the key: every other key reads as usual"""
  def __init__(self, params):
    self._params = params

  def __getattr__(self, name):
    return getattr(self._params, name)

  def get(self, key, *args, **kwargs):
    if key == PARAM:
      raise UnknownKeyName(key.encode())
    return self._params.get(key, *args, **kwargs)


def read_params(params):
  # the UI's params refresh worker (UIStateSP.update_params), on a stand-in so the shared ui_state keeps its other values
  state = SimpleNamespace(params=params, _enforce_constraints=lambda: None, chestnut_present=False, chestnut_active=None,
                          chestnut_loading=False, is_offroad=lambda: True, chestnut_compiled=False, _sp_initialized=True)
  UIStateSP.update_params(state)
  return state.follow_distance


class TestFollowDistanceUIState(OpenpilotTestCase):
  def test_reads_the_level(self):
    params = Params()
    assert read_params(params) == 2
    for stored in (0, 1, 2, 3):
      params.put(PARAM, stored, block=True)
      assert read_params(params) == stored

  def test_medium_on_a_params_library_without_the_key(self):
    assert read_params(StaleParams(Params())) == 2


class TestFollowDistanceToggle(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def build(self, monkeypatch, brand="rivian", long=True, stale=False):
    monkeypatch.setattr(gui_app, "texture", lambda *args, **kwargs: rl.Texture(0, 64, 64, 1, 7))
    monkeypatch.setattr(gui_app, "font", lambda *args, **kwargs: rl.Font())
    monkeypatch.setattr(rl, "get_time", lambda: 0.)
    monkeypatch.setattr(NavScroller, "_update_state", lambda self: None)
    monkeypatch.setattr(ui_state, "params", self.params)
    monkeypatch.setattr(ui_state, "sm", SimpleNamespace(updated={"selfdriveState": False}))
    monkeypatch.setattr(ui_state, "_engaged_transition_callbacks", [])
    monkeypatch.setattr(ui_state, "CP", car.CarParams(brand=brand, openpilotLongitudinalControl=long))
    monkeypatch.setattr(ui_state, "has_longitudinal_control", long)
    monkeypatch.setattr(ui_state, "follow_distance", ui_state.follow_distance)
    monkeypatch.setattr(ui_state, "update_params", self.refresh)
    if stale:
      real = toggles_layout.BigMultiParamToggle

      def stale_toggle(text, param, *args, **kwargs):
        if param == PARAM:
          raise UnknownKeyName(param.encode())
        return real(text, param, *args, **kwargs)
      monkeypatch.setattr(toggles_layout, "BigMultiParamToggle", stale_toggle)
    page = toggles_layout.TogglesLayoutMici()
    page.show_event()
    return page

  def refresh(self):
    ui_state.follow_distance = read_params(self.params)

  def wait_for_param(self, value):
    for _ in range(200):
      if self.params.get(PARAM) == value:
        return True
      time.sleep(0.01)
    return False

  def test_shown_only_for_rivian_with_openpilot_longitudinal(self, monkeypatch):
    assert self.build(monkeypatch)._follow_distance_toggle.is_visible
    for kwargs in ({"brand": "toyota"}, {"long": False}):
      assert not self.build(monkeypatch, **kwargs)._follow_distance_toggle.is_visible, kwargs

  def test_next_to_the_rivian_longitudinal_toggles(self, monkeypatch):
    page = self.build(monkeypatch)
    widgets = page._scroller._items
    i = widgets.index(page._follow_distance_toggle)
    assert widgets[i + 1] is page._traffic_mode_toggle
    assert widgets[i + 2] is page._launch_assist_toggle

  def test_left_out_on_a_params_library_without_the_key(self, monkeypatch):
    page = self.build(monkeypatch, stale=True)
    assert page._follow_distance_toggle is None
    page._update_state()
    page._update_toggles()

  def test_four_choices_in_the_levels_order(self, monkeypatch):
    toggle = self.build(monkeypatch)._follow_distance_toggle
    assert toggle._options == [name.lower() for name in follow_distance.NAMES] == ["closer", "close", "medium", "far"]

  def test_shows_the_stored_level(self, monkeypatch):
    for stored, shown in ((None, "medium"), (0, "closer"), (1, "close"), (3, "far")):
      if stored is not None:
        self.params.put(PARAM, stored, block=True)
      assert self.build(monkeypatch)._follow_distance_toggle.value == shown, stored

  def test_a_stored_value_out_of_range_shows_the_nearest_level(self, monkeypatch):
    # a hand edit, a raw sunnylink write or a backup from a later build: the page still builds (indexing with it raised at
    # UI start), and shows the level the planner uses
    for stored, shown in ((4, "far"), (7, "far"), (255, "far"), (-5, "closer")):
      self.params.put(PARAM, stored, block=True)
      assert self.build(monkeypatch)._follow_distance_toggle.value == shown, stored
    # the driving personality, on the same widget
    self.params.put("LongitudinalPersonality", 3, block=True)
    assert self.build(monkeypatch)._personality_toggle.value == "relaxed"

  def test_mirrors_changes_made_elsewhere(self, monkeypatch):
    page = self.build(monkeypatch)
    toggle = page._follow_distance_toggle
    for stored, shown in ((3, "far"), (0, "closer"), (1, "close"), (2, "medium")):
      # the right scroll wheel or sunnylink writes the param; the UI's refresh worker reads it
      self.params.put(PARAM, stored, block=True)
      self.refresh()
      page._update_state()
      assert toggle.value == shown

  def test_tap_steps_closer_close_medium_far(self, monkeypatch):
    self.params.put(PARAM, 0, block=True)
    toggle = self.build(monkeypatch)._follow_distance_toggle
    assert toggle.value == "closer"
    for stored, shown in ((1, "close"), (2, "medium"), (3, "far"), (0, "closer")):
      toggle._handle_mouse_release(MousePos(0, 0))
      assert toggle.value == shown
      assert self.wait_for_param(stored)
