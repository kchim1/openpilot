"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.

Rivian Following Distance in the planner: the level's follow time with the Following Distance Offset on top, eased in
while openpilot controls the speed, Traffic Mode never further back than it, and every other car exactly as before.
"""
import json
from types import SimpleNamespace

from openpilot.cereal import custom, log, messaging
from opendbc.car import structs
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, clip_t_follow_offset_pct, get_T_FOLLOW, \
  get_traffic_mode_T_FOLLOW
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib import follow_distance
from openpilot.sunnypilot.selfdrive.controls.lib.follow_distance import FollowDistance, RAMP_RATE, READ_FRAMES, T_FOLLOW
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, TRAFFIC_MODE_READ_FRAMES
from openpilot.sunnypilot.selfdrive.controls.lib.tests.test_traffic_mode import MockPubMaster, build_sm, run_mpc
from openpilot.sunnypilot.sunnylink.tools.compile_settings_ui import DEFAULT_OUT as SETTINGS_UI

Personality = log.LongitudinalPersonality
PERSONALITIES = (Personality.aggressive, Personality.standard, Personality.relaxed)
PARAM = "RivianFollowDistance"
V_EGO = 2.5


def car_params(brand="rivian", openpilot_long=True, harness=True):
  CP = structs.CarParams(brand=brand, openpilotLongitudinalControl=openpilot_long, steerRatio=15., wheelbase=3.08,
                         longitudinalActuatorDelay=0.25)
  CP_SP = structs.CarParamsSP(flags=int(RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE) if harness else 0)
  return CP, CP_SP


def old_get_t_follow(mpc, personality):
  # get_t_follow before the Following Distance, for comparison
  if mpc.traffic_mode:
    return get_traffic_mode_T_FOLLOW(mpc.x0[1])
  return get_T_FOLLOW(personality) * (1 + mpc.get_t_follow_offset_pct() / 100)


def long_active_sm(sm, long_active):
  cc = messaging.new_message('carControl')
  cc.carControl.longActive = long_active
  sm['carControl'] = cc.carControl.as_reader()
  return sm


class TestLevels(OpenpilotTestCase):
  def test_medium_and_far_are_aggressive_and_standard(self):
    assert T_FOLLOW == (1.00, 1.10, 1.25, 1.45)
    assert T_FOLLOW[2] == get_T_FOLLOW(Personality.aggressive)
    assert T_FOLLOW[3] == get_T_FOLLOW(Personality.standard)
    assert follow_distance.NAMES == ("Closer", "Close", "Medium", "Far")
    assert follow_distance.DEFAULT == 2

  def test_sanitize(self):
    for stored, level in ((0, 0), (1, 1), (2, 2), (3, 3), (4, 3), (-1, 0), (None, 2), ("abc", 2), (float('nan'), 2), (2.7, 2)):
      assert follow_distance.sanitize(stored) == level, stored

  def test_sunnylink_choices_are_the_levels(self):
    with open(SETTINGS_UI) as f:
      cruise = next(panel for panel in json.load(f)["panels"] if panel["id"] == "cruise")
    item = next(item for section in cruise["sections"] for item in section["items"] if item["key"] == PARAM)
    assert [(option["value"], option["label"]) for option in item["options"]] == list(enumerate(follow_distance.NAMES))

  def test_available_only_on_rivian_with_the_harness_and_openpilot_long(self):
    assert follow_distance.available(*car_params())
    for kwargs in ({"harness": False}, {"openpilot_long": False}, {"brand": "toyota"}, {"brand": "hyundai", "harness": False}):
      assert not follow_distance.available(*car_params(**kwargs)), kwargs
    # selfdrived and plannerd hand it capnp CarParams
    CP = structs.CarParams(brand="rivian", openpilotLongitudinalControl=True).as_reader()
    CP_SP = custom.CarParamsSP.new_message(flags=int(RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE)).as_reader()
    assert follow_distance.available(CP, CP_SP)


class TestFollowTime(OpenpilotTestCase):
  def test_offset_on_top_of_the_level(self):
    mpc = LongitudinalMpc()
    for offset in (-10, -4, 0, 10):
      mpc.t_follow_offset_pct = offset
      for t in T_FOLLOW:
        mpc.follow_t_base = t
        for personality in PERSONALITIES:
          run_mpc(mpc, 20., personality)
          expected = t * (1 + offset / 100)
          assert mpc.t_follow == expected
          assert (mpc.params[:, 4] == expected).all()

  def test_medium_at_minus_ten_is_the_old_aggressive_gap_for_every_personality(self):
    mpc = LongitudinalMpc()
    mpc.t_follow_offset_pct = -10
    old = get_T_FOLLOW(Personality.aggressive) * (1 + -10 / 100)
    assert abs(old - 1.125) < 1e-12
    for level, expected in ((0, 0.90), (1, 0.99), (2, 1.125), (3, 1.305)):
      mpc.follow_t_base = T_FOLLOW[level]
      for personality in PERSONALITIES:
        t_follow = mpc.get_t_follow(personality)
        assert abs(t_follow - expected) < 1e-12, (level, personality)
        if level == 2:
          assert t_follow == old

  def test_traffic_mode_never_further_back_than_the_level(self):
    mpc = LongitudinalMpc()
    mpc.traffic_mode = True
    mpc.t_follow_offset_pct = -10
    # Closer, Close, Medium and Far at each speed: Closer (0.90 s) is under the curve from 4 m/s (9 mph), Close (0.99 s)
    # from 4.9 m/s (11 mph)
    for v_ego, expected in ((0., (0.5, 0.5, 0.5, 0.5)), (2.5, (0.75, 0.75, 0.75, 0.75)), (4.5, (0.9, 0.95, 0.95, 0.95)),
                            (5., (0.9, 0.99, 1.0, 1.0)), (30., (0.9, 0.99, 1.0, 1.0))):
      for t, t_expected in zip(T_FOLLOW, expected, strict=True):
        mpc.follow_t_base = t
        run_mpc(mpc, v_ego, Personality.standard)
        assert abs(mpc.t_follow - t_expected) < 1e-12, (v_ego, t)
        assert mpc.jerk_factor == 0.5
    # with no offset no level is under the curve (Closer is its 1.0 s): Traffic Mode as before
    mpc.t_follow_offset_pct = 0
    for t in T_FOLLOW:
      mpc.follow_t_base = t
      for v_ego in (0., 2.5, 5., 30.):
        mpc.x0[1] = v_ego
        assert mpc.get_t_follow(Personality.aggressive) == get_traffic_mode_T_FOLLOW(v_ego)

  def test_without_the_level_exactly_as_before(self):
    mpc = LongitudinalMpc()
    assert mpc.follow_t_base is None
    for traffic_mode in (False, True):
      mpc.traffic_mode = traffic_mode
      for offset in (-10, -6, 0, 4, 10, 25, float('nan')):
        mpc.t_follow_offset_pct = offset
        for personality in PERSONALITIES:
          for v_ego in (0., 1., 3., 5., 13.4, 30.):
            mpc.x0[1] = v_ego
            assert mpc.get_t_follow(personality) == old_get_t_follow(mpc, personality), (traffic_mode, offset, personality, v_ego)
    assert clip_t_follow_offset_pct(float('nan')) == 0.


class TestFollowDistanceEase(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def test_starts_at_the_stored_level(self):
    for level in (0, 1, 2, 3):
      self.params.put(PARAM, level, block=True)
      fd = FollowDistance(self.params)
      assert fd.level == level
      assert fd.t_base == T_FOLLOW[level]
      assert fd.update(self.params, True, DT_MDL) == T_FOLLOW[level]

  def test_eases_in_while_long_active(self):
    fd = FollowDistance(self.params)
    self.params.put(PARAM, 3, block=True)
    t_base = [fd.update(self.params, True, DT_MDL) for _ in range(60)]
    step = RAMP_RATE * DT_MDL
    # 0.1 s of follow time per s: Medium to Far (0.2 s) in 2 s, never past it
    assert t_base[0] == T_FOLLOW[2] + step
    assert all(abs((b - a) - step) < 1e-9 for a, b in zip(t_base[:39], t_base[1:40], strict=True))
    assert t_base[39:] == [T_FOLLOW[3]] * 21
    # and back to Closer (0.45 s) in 4.5 s
    self.params.put(PARAM, 0, block=True)
    t_base = [fd.update(self.params, True, DT_MDL) for _ in range(100)]
    assert t_base[88] > T_FOLLOW[0]
    assert t_base[90:] == [T_FOLLOW[0]] * 10
    assert all(0 < a - b < step + 1e-9 for a, b in zip(t_base[:89], t_base[1:90], strict=True))

  def test_at_once_when_not_controlling(self):
    fd = FollowDistance(self.params)
    self.params.put(PARAM, 0, block=True)
    assert fd.update(self.params, False, DT_MDL) == T_FOLLOW[0]
    # an ease in progress finishes at once when openpilot stops controlling the speed (gas pressed, disengaged)
    self.params.put(PARAM, 3, block=True)
    for _ in range(READ_FRAMES):
      fd.update(self.params, True, DT_MDL)
    assert T_FOLLOW[0] < fd.t_base < T_FOLLOW[3]
    assert fd.update(self.params, False, DT_MDL) == T_FOLLOW[3]

  def test_read_every_read_frames(self):
    fd = FollowDistance(self.params)
    fd.update(self.params, False, DT_MDL)
    self.params.put(PARAM, 0, block=True)
    for _ in range(READ_FRAMES - 1):
      assert fd.update(self.params, False, DT_MDL) == T_FOLLOW[2]
    assert fd.update(self.params, False, DT_MDL) == T_FOLLOW[0]
    assert fd.level == 0

  def test_read_error_is_medium_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.follow_distance.cloudlog")
    params = mocker.MagicMock()
    params.get.side_effect = UnknownKeyName(PARAM.encode())
    fd = FollowDistance(params)
    for _ in range(3 * READ_FRAMES):
      assert fd.update(params, True, DT_MDL) == T_FOLLOW[2]
    assert fd.level == 2
    assert cloudlog.exception.call_count == 1


class TestFollowDistancePlanner(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  def test_only_rivian_with_the_harness_gets_it(self):
    assert LongitudinalPlannerSP(*car_params(), SimpleNamespace(traffic_mode=False)).follow_distance is not None
    for kwargs in ({"harness": False}, {"openpilot_long": False}, {"brand": "toyota"}):
      assert LongitudinalPlannerSP(*car_params(**kwargs), SimpleNamespace(traffic_mode=False)).follow_distance is None, kwargs

  def test_planner_follows_the_level_and_logs_it(self):
    self.params.put("SPFollowingTimeOffset", -10, block=True)
    self.params.put(PARAM, 1, block=True)
    planner = LongitudinalPlanner(*car_params(), init_v=V_EGO)
    sm, pm = long_active_sm(build_sm(V_EGO), True), MockPubMaster()
    for level, expected in ((0, 0.90), (3, 1.305), (2, 1.125), (1, 0.99)):
      self.params.put(PARAM, level, block=True)
      # read within 0.25 s, then eased in at 0.1 s per s (at most 0.45 s: 4.5 s)
      for _ in range(READ_FRAMES + int(4.5 / DT_MDL) + 1):
        planner.update(sm)
        planner.publish(sm, pm)
      assert abs(planner.mpc.params[0, 4] - expected) < 1e-9
      fd = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance
      assert fd.level == level
      assert abs(fd.tFollowBase - T_FOLLOW[level]) < 1e-6
      assert abs(fd.tFollow - expected) < 1e-6
      assert fd.offsetPct == -10.

  def test_personality_no_longer_sets_the_gap(self):
    planner = LongitudinalPlanner(*car_params(), init_v=V_EGO)
    for personality in PERSONALITIES:
      planner.mpc.set_cur_state(V_EGO, 0.)
      assert planner.mpc.get_t_follow(personality) == planner.mpc.get_t_follow(Personality.aggressive)
      planner.update(build_sm(V_EGO))
      assert planner.mpc.params[0, 4] == T_FOLLOW[2]

  def test_traffic_mode_on_each_level(self):
    self.params.put("SPFollowingTimeOffset", -10, block=True)
    self.params.put_bool("TrafficMode", True, block=True)
    # at 0, 2.5, 4.5 and 10 m/s: Traffic Mode's curve, never above Closer's 0.90 s or Close's 0.99 s
    for level, expected in enumerate(((0.5, 0.75, 0.9, 0.9), (0.5, 0.75, 0.95, 0.99), (0.5, 0.75, 0.95, 1.0), (0.5, 0.75, 0.95, 1.0))):
      self.params.put(PARAM, level, block=True)
      for v_ego, t_expected in zip((0., 2.5, 4.5, 10.), expected, strict=True):
        planner = LongitudinalPlanner(*car_params(), init_v=v_ego)
        sm, pm = build_sm(v_ego), MockPubMaster()
        for _ in range(TRAFFIC_MODE_READ_FRAMES):
          planner.update(sm)
          planner.publish(sm, pm)
        # the curve at the planner's own speed
        t_follow = planner.mpc.params[0, 4]
        assert t_follow == min(get_traffic_mode_T_FOLLOW(planner.mpc.x0[1]), T_FOLLOW[level] * (1 + -10 / 100))
        assert abs(t_follow - t_expected) < 0.01, (level, v_ego)
        fd = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance
        assert fd.trafficMode and fd.level == level

  def test_other_cars_unchanged_and_not_logged(self):
    for kwargs in ({"brand": "toyota"}, {"harness": False}):
      self.params.put(PARAM, 0, block=True)
      self.params.put("SPFollowingTimeOffset", -10, block=True)
      planner = LongitudinalPlanner(*car_params(**kwargs), init_v=V_EGO)
      sm, pm = build_sm(V_EGO), MockPubMaster()
      for _ in range(2 * READ_FRAMES):
        planner.update(sm)
        planner.publish(sm, pm)
      assert planner.follow_distance is None and planner.mpc.follow_t_base is None
      # personality is Aggressive (selfdriveState default)
      assert planner.mpc.params[0, 4] == get_T_FOLLOW(Personality.aggressive) * (1 + -10 / 100)
      fd = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance
      assert fd.level == -1
      assert fd.tFollowBase == 0.
