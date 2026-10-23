"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Launch Assist Strength on the comma four toggles page: right after Launch Assist, shown only for a Rivian with openpilot
longitudinal, enabled only while Launch Assist is on, left out on a params library without the key, Strong unless set to
Gentle, and a tap switches between the two.
The page is built headless: textures and fonts are stubbed and nothing is drawn.
"""
import time
from types import SimpleNamespace

import pyray as rl

from opendbc.car.structs import car
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.ui.mici.layouts.settings import toggles as toggles_layout
from openpilot.selfdrive.ui.ui_state import ui_state
from openpilot.system.ui.lib.application import MousePos, gui_app
from openpilot.system.ui.widgets.scroller import NavScroller

PARAM = "RivianLaunchAssistBoost"


class TestLaunchAssistStrengthToggle(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def build(self, monkeypatch, brand="rivian", long=True, stale=()):
    monkeypatch.setattr(gui_app, "texture", lambda *args, **kwargs: rl.Texture(0, 64, 64, 1, 7))
    monkeypatch.setattr(gui_app, "font", lambda *args, **kwargs: rl.Font())
    monkeypatch.setattr(rl, "get_time", lambda: 0.)
    monkeypatch.setattr(NavScroller, "_update_state", lambda self: None)
    monkeypatch.setattr(ui_state, "params", self.params)
    monkeypatch.setattr(ui_state, "sm", SimpleNamespace(updated={"selfdriveState": False}))
    monkeypatch.setattr(ui_state, "_engaged_transition_callbacks", [])
    monkeypatch.setattr(ui_state, "CP", car.CarParams(brand=brand, openpilotLongitudinalControl=long))
    monkeypatch.setattr(ui_state, "has_longitudinal_control", long)
    monkeypatch.setattr(ui_state, "update_params", lambda: None)
    # a params library built before these keys
    for name in ("BigMultiParamToggle", "BigParamControl"):
      real = getattr(toggles_layout, name)

      def stale_toggle(text, param, *args, _real=real, **kwargs):
        if param in stale:
          raise UnknownKeyName(param.encode())
        return _real(text, param, *args, **kwargs)
      monkeypatch.setattr(toggles_layout, name, stale_toggle)
    page = toggles_layout.TogglesLayoutMici()
    page.show_event()
    return page

  def wait_for_param(self, value):
    for _ in range(200):
      if self.params.get(PARAM) == value:
        return True
      time.sleep(0.01)
    return False

  def test_shown_only_for_rivian_with_openpilot_longitudinal(self, monkeypatch):
    assert self.build(monkeypatch)._launch_boost_toggle.is_visible
    for kwargs in ({"brand": "toyota"}, {"long": False}):
      assert not self.build(monkeypatch, **kwargs)._launch_boost_toggle.is_visible, kwargs

  def test_right_after_launch_assist(self, monkeypatch):
    page = self.build(monkeypatch)
    widgets = page._scroller._items
    assert widgets[widgets.index(page._launch_assist_toggle) + 1] is page._launch_boost_toggle

  def test_left_out_on_a_params_library_without_the_key(self, monkeypatch):
    for stale in ((PARAM,), ("RivianLaunchAssist",)):
      page = self.build(monkeypatch, stale=stale)
      assert page._launch_boost_toggle is None, stale
      assert page._launch_assist_toggle is not None or stale == ("RivianLaunchAssist",)
      page._update_toggles()

  def test_gentle_and_strong(self, monkeypatch):
    assert self.build(monkeypatch)._launch_boost_toggle._options == ["gentle", "strong"]

  def test_shows_the_stored_strength(self, monkeypatch):
    for stored, shown in ((None, "strong"), (0, "gentle"), (1, "strong")):
      if stored is not None:
        self.params.put(PARAM, stored, block=True)
      assert self.build(monkeypatch)._launch_boost_toggle.value == shown, stored

  def test_a_stored_value_out_of_range_shows_strong(self, monkeypatch):
    # a hand edit, a raw sunnylink write or a backup from a later build: the page still builds (indexing with it raised at
    # UI start), and shows what Launch Assist uses, Strong unless 0
    for stored in (2, 7, -3):
      self.params.put(PARAM, stored, block=True)
      assert self.build(monkeypatch)._launch_boost_toggle.value == "strong", stored

  def test_enabled_only_while_launch_assist_is_on(self, monkeypatch):
    self.params.put_bool("RivianLaunchAssist", False, block=True)
    page = self.build(monkeypatch)
    assert not page._launch_boost_toggle.enabled
    page._launch_assist_toggle._handle_mouse_release(MousePos(0, 0))
    assert page._launch_boost_toggle.enabled
    # and turned on elsewhere (sunnylink): the page picks it up when shown again
    self.params.put_bool("RivianLaunchAssist", False, block=True)
    page.show_event()
    assert not page._launch_boost_toggle.enabled
    self.params.put_bool("RivianLaunchAssist", True, block=True)
    page.show_event()
    assert page._launch_boost_toggle.enabled

  def test_tap_switches_and_writes(self, monkeypatch):
    self.params.put_bool("RivianLaunchAssist", True, block=True)
    toggle = self.build(monkeypatch)._launch_boost_toggle
    assert toggle.value == "strong"
    for stored, shown in ((0, "gentle"), (1, "strong")):
      toggle._handle_mouse_release(MousePos(0, 0))
      assert toggle.value == shown
      assert self.wait_for_param(stored)
