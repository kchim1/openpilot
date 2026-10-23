"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from collections import defaultdict
from types import SimpleNamespace
from unittest import mock

import numpy as np

from openpilot.cereal import log, messaging
from opendbc.car import structs
from opendbc.car.interfaces import ACCEL_MAX
from openpilot.common.params import Params, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib import long_mpc
from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.long_mpc import LongitudinalMpc, get_T_FOLLOW, get_jerk_factor, \
  get_traffic_mode_T_FOLLOW, DANGER_ZONE_COST, LIMIT_COST, STOP_DISTANCE, T_IDXS, TRAFFIC_MODE_LAUNCH_SHIFT, \
  TRAFFIC_MODE_LAUNCH_SHIFT_GAP_BP, TRAFFIC_MODE_LAUNCH_SHIFT_MARGIN, TRAFFIC_MODE_LAUNCH_SHIFT_RATE, \
  TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, get_cruise_accel, TRAFFIC_MODE_ACCEL_CAP, \
  TRAFFIC_MODE_ACCEL_CAP_BP
from openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlannerSP, TRAFFIC_MODE_READ_FRAMES
from openpilot.sunnypilot.selfdrive.controls.lib.stop_behind_lead import GAP_MIN

Personality = log.LongitudinalPersonality
PERSONALITIES = (Personality.aggressive, Personality.standard, Personality.relaxed)
V_EGO = 2.5
CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=3.4, longitudinalActuatorDelay=0.25)


class MockSubMaster(dict):
  def __init__(self, services: dict):
    super().__init__(services)
    self.valid = dict.fromkeys(services, True)
    self.logMonoTime = dict.fromkeys(services, 0)
    self.updated = dict.fromkeys(services, True)
    self.recv_frame = dict.fromkeys(services, 1)

  def all_checks(self, service_list=None) -> bool:
    return True


class MockPubMaster:
  def __init__(self):
    self.sent = defaultdict(list)

  def send(self, name, msg):
    self.sent[name].append(msg)


def build_sm(v_ego):
  services = {}
  for service in ("controlsState", "vehicleParameters", "carStateSP", "liveMapDataSP", "gpsLocationExternal", "gpsLocation",
                  "selfdriveState", "carControl"):
    services[service] = getattr(messaging.new_message(service), service)

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = v_ego
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
  services['carState'] = car_state.carState.as_reader()

  radar = messaging.new_message('radarState')
  radar.radarState.leadOne = {"present": True, "dRel": 10., "vLead": v_ego, "aLeadTau": 1.5, "modelProb": 1.}
  services['radarState'] = radar.radarState.as_reader()

  model = messaging.new_message('modelV2')
  model.modelV2.orientationRate.z = [0.01] * 33  # nonzero: a straight path divides by zero in SCC vision
  model.modelV2.velocity.x = [v_ego] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def radar_state():
  lead = SimpleNamespace(present=True, dRel=20., vLead=10., aLeadK=0., aLeadTau=1.5, modelProb=1., radar=True, radarTrackId=1)
  return SimpleNamespace(leadOne=lead, leadTwo=lead)


def lead(d_rel=20., v_lead=0., present=True, radar=True, track=1, a_lead=0.):
  return SimpleNamespace(present=present, dRel=d_rel, vLead=v_lead, vLeadK=v_lead, aLeadK=a_lead, aLeadTau=1.5, modelProb=1.,
                         radar=radar, radarTrackId=track)


def leads(lead_one, lead_two=None):
  return SimpleNamespace(leadOne=lead_one, leadTwo=lead_two or SimpleNamespace(present=False))


def launch_shifts(mpc, frames, v_ego=0., lead_one=None):
  # leadOne's launch shift each frame, the plan at v_ego
  mpc.x0[1] = v_ego
  return [mpc.traffic_mode_launch_shift(lead_one or lead(3.7)) for _ in range(frames)]


def stopped_mpc(d_rel=3.7, traffic_mode=True):
  # stopped behind a stopped radar lead long enough for the shift to build up
  mpc = LongitudinalMpc()
  mpc.traffic_mode = traffic_mode
  launch_shifts(mpc, 60, lead_one=lead(d_rel))
  return mpc


def closing_fcw(v_ego, d_rel, v_lead, frames=80):
  """FCW (the planner's crash_cnt > 2) each frame, closing on a slower car that appears d_rel ahead after following a car far
  ahead, the truck driving the plan"""
  mpc = LongitudinalMpc()
  mpc.traffic_mode = True
  for _ in range(60):
    mpc.set_weights(True)
    mpc.set_cur_state(v_ego, 0.)
    mpc.update(leads(lead(80., v_ego)))
  fcw, a_ego = [], 0.
  for _ in range(frames):
    mpc.set_weights(True)
    mpc.set_cur_state(v_ego, a_ego)
    mpc.update(leads(lead(d_rel, v_lead)))
    fcw.append(mpc.crash_cnt > 2)
    a_ego = float(np.interp(DT_MDL, T_IDXS, mpc.a_solution))
    v_next = max(v_ego + a_ego * DT_MDL, 0.)
    d_rel += (v_lead - (v_ego + v_next) / 2) * DT_MDL
    v_ego = v_next
  return np.array(fcw)


def settle_cruise_accel(e2e, v_ego, follow_share=0., traffic_mode=True, personality=Personality.aggressive):
  # set speed far above, straight wheel, throttle allowed: the cruise candidate ramps up to its cap; returns it and its peak
  a, peak = 0., -1e9
  for _ in range(200):
    a = get_cruise_accel(e2e, 40., v_ego, a, 0., CP, DT_MDL, ACCEL_MAX, True, follow_share, 1., personality, traffic_mode)
    peak = max(peak, a)
  return a, peak


def run_mpc(mpc, v_ego, personality):
  # same call order as LongitudinalPlanner.update
  mpc.set_weights(True, personality=personality)
  mpc.set_cur_state(v_ego, 0.)
  mpc.update(radar_state(), personality=personality)


def record_weights(mpc):
  calls = []
  mpc.set_cost_weights = lambda cost_weights, constraint_cost_weights: calls.append((list(cost_weights), list(constraint_cost_weights)))
  return calls


class TestTrafficModeMpc(OpenpilotTestCase):
  def test_follow_time_curve(self):
    # 0.6 s at a stop, 0.8 s from 11 to 20 mph, 0.9 s at 30 mph and 1.0 s from 40 mph up, as before there
    for v_ego, t_follow in ((0., 0.6), (2.5, 0.7), (5., 0.8), (7., 0.8), (8.94, 0.8), (11.175, 0.85), (13.41, 0.9), (15.645, 0.95),
                            (17.88, 1.0), (25., 1.0), (40., 1.0)):
      self.assertAlmostEqual(get_traffic_mode_T_FOLLOW(v_ego), t_follow)

  def test_mpc_uses_curve_and_ignores_offset(self):
    mpc = LongitudinalMpc()
    mpc.traffic_mode = True
    for offset in (-10, 0, 10):
      mpc.t_follow_offset_pct = offset
      for personality in PERSONALITIES:
        for v_ego in (0., 2.5, 5., 20.):
          run_mpc(mpc, v_ego, personality)
          expected = get_traffic_mode_T_FOLLOW(v_ego)
          assert (mpc.params[:, 4] == expected).all(), (offset, personality, v_ego)
          assert mpc.t_follow == expected
          assert mpc.get_t_follow_offset_pct() == 0.
          assert mpc.jerk_factor == 0.5
    # the offset setting itself is left alone
    assert mpc.t_follow_offset_pct == 10

  def test_off_restores_personality_and_offset(self):
    mpc = LongitudinalMpc()
    mpc.t_follow_offset_pct = -10
    for personality in PERSONALITIES:
      mpc.traffic_mode = True
      run_mpc(mpc, 2., personality)
      mpc.traffic_mode = False
      run_mpc(mpc, 2., personality)
      expected = get_T_FOLLOW(personality) * (1 + -10 / 100)
      assert (mpc.params[:, 4] == expected).all()
      assert mpc.t_follow == expected
      assert mpc.get_t_follow_offset_pct() == -10.
      assert mpc.jerk_factor == get_jerk_factor(personality)

  def test_only_jerk_factor_changes_in_weights(self):
    mpc = LongitudinalMpc()
    calls = record_weights(mpc)
    mpc.set_weights(True, personality=Personality.aggressive)
    for personality in PERSONALITIES:
      mpc.traffic_mode = True
      mpc.set_weights(True, personality=personality)
    # every personality gets Aggressive's weights, and the danger zone cost is unchanged
    assert all(c == calls[0] for c in calls)
    assert calls[0][1] == [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]


class TestTrafficModeLaunchShift(OpenpilotTestCase):
  def test_only_once_stopped_behind_a_radar_lead_in_traffic_mode(self):
    assert TRAFFIC_MODE_LAUNCH_SHIFT == 2. and TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP == [0.1, 1.0]
    # stopped 3.7 m back (Stop Behind Lead's gap): all of it
    assert stopped_mpc().launch_shift == 2.
    # none while following or closing in, at any speed: stops plan as without it
    for v_ego in (0.5, 2., 5., 15.):
      mpc = LongitudinalMpc()
      mpc.traffic_mode = True
      assert launch_shifts(mpc, 60, v_ego) == [0.] * 60
    # none without Traffic Mode, for a vision-only lead (Stop Behind Lead doesn't backstop it) or with no lead
    assert stopped_mpc(traffic_mode=False).launch_shift == 0.
    for lead_one in (lead(3.7, radar=False), lead(3.7, present=False), None):
      mpc = LongitudinalMpc()
      mpc.traffic_mode = True
      mpc.x0[1] = 0.
      assert [mpc.traffic_mode_launch_shift(lead_one) for _ in range(60)] == [0.] * 60

  def test_the_room_left_short_of_the_standstill_distance(self):
    # the obstacle at most STOP_DISTANCE - margin away: it can't pull the truck toward a car that stays put. And less for a
    # truck stopped under Stop Behind Lead's GAP_MIN, none by 1 m closer: the next stop can't end closer still
    assert TRAFFIC_MODE_LAUNCH_SHIFT_GAP_BP == [GAP_MIN - 1., GAP_MIN]
    for d_rel, shift in ((2., 0.), (2.5, 0.), (3., 1.), (3.5, 2.), (3.7, 2.), (4., 1.7), (5., 0.7), (5.7, 0.), (8., 0.)):
      held = stopped_mpc(d_rel).launch_shift
      np.testing.assert_allclose(held, shift, atol=1e-9, err_msg=str(d_rel))
      assert d_rel + held <= max(STOP_DISTANCE - TRAFFIC_MODE_LAUNCH_SHIFT_MARGIN, d_rel) + 1e-9

  def test_the_gap_is_the_closest_while_stopped(self):
    # the car pulling away doesn't take it back
    mpc = stopped_mpc(4.)
    held = mpc.launch_shift
    self.assertAlmostEqual(held, 1.7)
    assert launch_shifts(mpc, 20, lead_one=lead(4.5, 0.5)) == [held] * 20
    # the truck rolling closer after its speed reads 0 does
    np.testing.assert_allclose(launch_shifts(mpc, 1, lead_one=lead(3., 0.)), [1.])

  def test_stopped_truck_stays_put_behind_a_car_that_does(self):
    # stopped 3 to 8 m behind a stopped car, the plan closes up no more than without Traffic Mode
    for d_rel in (3., 4., 5., 6., 8.):
      ends = []
      for traffic_mode in (False, True):
        mpc = LongitudinalMpc()
        mpc.traffic_mode = traffic_mode
        for _ in range(100):
          mpc.set_weights(True)
          mpc.set_cur_state(0., 0.)
          mpc.update(leads(lead(d_rel)))
        ends.append(mpc.x_sol[-1, 0])
      assert ends[1] < ends[0] + 0.1, (d_rel, ends)
      if d_rel < STOP_DISTANCE:
        assert ends[1] < 0.1, (d_rel, ends)

  def test_moves_off_sooner_behind_a_car_pulling_away(self):
    # stopped 3.7 m behind a car that pulls away at 0.5 m/s^2 after 3 s, in Traffic Mode with and without the shift: the
    # first frame the plan asks to go (as should_stop sees it)
    first_go = []
    for shift in (0., TRAFFIC_MODE_LAUNCH_SHIFT):
      with mock.patch.object(long_mpc, 'TRAFFIC_MODE_LAUNCH_SHIFT', shift):
        mpc = LongitudinalMpc()
        mpc.traffic_mode = True
        for i in range(200):
          t = max(i - 60, 0) * DT_MDL
          mpc.set_weights(True)
          mpc.set_cur_state(0., 0.)
          mpc.update(leads(lead(3.7 + 0.25 * t ** 2, 0.5 * t, a_lead=0.5 if t > 0 else 0.)))
          if np.interp(0.3, T_IDXS, mpc.a_solution) >= 0.1:
            break
        first_go.append(i)
    assert first_go[1] < 60 + 40 and first_go[1] + 10 <= first_go[0] < 199, first_go

  def test_builds_up_at_1_m_per_s_and_lets_go_at_once(self):
    step = TRAFFIC_MODE_LAUNCH_SHIFT_RATE * DT_MDL
    mpc = LongitudinalMpc()
    mpc.traffic_mode = True
    # stopped: 2 m in 2 s
    shifts = launch_shifts(mpc, 50)
    np.testing.assert_allclose(shifts[:40], step * np.arange(1, 41))
    np.testing.assert_allclose(shifts[40:], 2.)
    # Traffic Mode off: none from the next frame (the obstacle closer is the cautious way); on again, built up again
    mpc.traffic_mode = False
    assert launch_shifts(mpc, 1) == [0.]
    mpc.traffic_mode = True
    np.testing.assert_allclose(launch_shifts(mpc, 3), step * np.arange(1, 4))
    # the lead lost, or a vision-only one: none at once, then built up again
    launch_shifts(mpc, 60)
    assert launch_shifts(mpc, 1, lead_one=lead(3.7, present=False)) == [0.]
    np.testing.assert_allclose(launch_shifts(mpc, 3), step * np.arange(1, 4))
    launch_shifts(mpc, 60)
    assert launch_shifts(mpc, 1, lead_one=lead(3.7, radar=False)) == [0.]
    np.testing.assert_allclose(launch_shifts(mpc, 3), step * np.arange(1, 4))

  def test_another_car_starts_over(self):
    # a car cutting in while the lead stays present (another radar track): built up again from none, to its own stop gap's
    mpc = stopped_mpc(5.)
    np.testing.assert_allclose(mpc.launch_shift, 0.7)
    cut_in = lead(3.6, track=2)
    np.testing.assert_allclose(launch_shifts(mpc, 1, lead_one=cut_in), [TRAFFIC_MODE_LAUNCH_SHIFT_RATE * DT_MDL])
    np.testing.assert_allclose(launch_shifts(mpc, 60, lead_one=cut_in)[-1], 2.)

  def test_once_moving_it_only_comes_down(self):
    mpc = stopped_mpc()
    # spent by the plan's own travel and faded by its speed: (2 - travel) * fade
    shifts = launch_shifts(mpc, 20, v_ego=0.2)
    travel = 0.2 * DT_MDL * np.arange(1, 21)
    np.testing.assert_allclose(shifts, (2. - travel) * np.interp(0.2, TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP, [1., 0.]))
    # slowing again without stopping: it doesn't grow back
    held = mpc.launch_shift
    assert launch_shifts(mpc, 20, v_ego=0.12) == [held] * 20
    # up to speed, the stop is over: slowing again without stopping gets none
    assert launch_shifts(mpc, 1, v_ego=TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP[1]) == [0.]
    assert launch_shifts(mpc, 20, v_ego=0.5) == [0.] * 20

  def test_every_stop_measures_its_gap_again(self):
    # an inching queue: moved off a little, stopped again. The shift is the new stop's, not the first one's, whether the
    # new stop is closer (3 m: 1 m) or farther (4.5 m: 1.2 m, not the first stop's 2 m, which would pull toward the car)
    for d_rel, shift in ((3., 1.), (4.5, 1.2)):
      mpc = stopped_mpc(3.7)
      launch_shifts(mpc, 10, v_ego=0.3, lead_one=lead(3.4))
      shifts = launch_shifts(mpc, 60, lead_one=lead(d_rel))
      np.testing.assert_allclose(shifts[-1], shift, err_msg=str(d_rel))
      assert d_rel + max(shifts) <= max(STOP_DISTANCE - TRAFFIC_MODE_LAUNCH_SHIFT_MARGIN, d_rel) + 1e-9

  def test_none_from_1_m_per_s_on_a_gradual_move_off(self):
    # the plan picking up speed at 0.5 m/s^2 after the stop: none from TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP[1] on, for good
    mpc = stopped_mpc()
    for v_ego in np.arange(0.1, 3., 0.5 * DT_MDL):
      shift = launch_shifts(mpc, 1, v_ego=float(v_ego))[0]
      if v_ego >= TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP[1]:
        assert shift == 0., v_ego
    assert mpc.launch_shift_gap is None

  def test_reset_clears_it(self):
    mpc = stopped_mpc()
    launch_shifts(mpc, 5, v_ego=0.3)
    mpc.reset()
    assert (mpc.launch_shift, mpc.launch_shift_track, mpc.launch_shift_gap, mpc.launch_shift_travel) == (0., -1, None, 0.)

  def test_lead_two_only_on_lead_ones_track(self):
    obstacles = {}
    for name, lead_two in (('same car', lead(3.7)), ('another car', lead(3.7, track=2)), ('vision copy', lead(3.7, radar=False)),
                           ('absent', SimpleNamespace(present=False))):
      mpc = LongitudinalMpc()
      mpc.traffic_mode = True
      for _ in range(60):
        mpc.set_cur_state(0., 0.)
        mpc.update(leads(lead(3.7), lead_two))
      obstacles[name] = mpc.params[0, 2]
    # min() over the two obstacles: leadTwo on another track, or a vision copy, keeps its own (the cautious way)
    np.testing.assert_allclose(obstacles['same car'], obstacles['absent'])
    np.testing.assert_allclose(obstacles['same car'] - obstacles['another car'], 2.)
    np.testing.assert_allclose(obstacles['same car'] - obstacles['vision copy'], 2.)
    # leadOne gone: no shift for leadTwo either
    mpc = stopped_mpc()
    mpc.set_cur_state(0., 0.)
    mpc.update(leads(SimpleNamespace(present=False), lead(3.7)))
    assert mpc.launch_shift == 0.

  def test_moves_the_obstacle_by_the_shift(self):
    obstacles = []
    for traffic_mode in (False, True):
      mpc = stopped_mpc(traffic_mode=traffic_mode)
      mpc.set_cur_state(0., 0.)
      mpc.update(leads(lead(3.7)))
      obstacles.append(mpc.params[:, 2].copy())
    np.testing.assert_allclose(obstacles[1] - obstacles[0], 2.)

  def test_fcw_checks_the_lead_itself(self):
    # the same plan, ending 0.1 m short of a stopped car, warns with the shift as without it: the crash check is on the
    # car, not on its obstacle, which the solver got 2 m farther on
    d_rel = 3.7
    for traffic_mode in (False, True):
      mpc = stopped_mpc(d_rel, traffic_mode)

      def run(mpc=mpc):
        mpc.x_sol[:, 0] = np.minimum(5. * T_IDXS, d_rel - 0.1)
      mpc.run = run
      mpc.set_cur_state(0., 0.)
      mpc.update(leads(lead(d_rel)))
      np.testing.assert_allclose(mpc.params[:, 2], d_rel + (2. if traffic_mode else 0.))
      assert mpc.crash_cnt == 1

  def test_fcw_unchanged_while_moving(self):
    # closing on a slower or stopped car (a cut-in, a car seen late): no shift while moving, so FCW fires on the same frames
    fired = 0
    for v_ego, d_rel, v_lead in ((5., 6.5, 0.), (5., 9.5, 0.), (8., 8., 2.), (8., 12., 0.), (10., 12.5, 2.), (12., 20., 0.)):
      with_shift = closing_fcw(v_ego, d_rel, v_lead)
      with mock.patch.object(long_mpc, 'TRAFFIC_MODE_LAUNCH_SHIFT', 0.):
        without = closing_fcw(v_ego, d_rel, v_lead)
      np.testing.assert_array_equal(with_shift, without)
      fired += int(without.any())
    assert fired >= 3


class TestTrafficModeAccelCap(OpenpilotTestCase):
  def test_keep_up_with_traffic_cannot_lift_it(self):
    assert TRAFFIC_MODE_ACCEL_CAP == 1.5 and TRAFFIC_MODE_ACCEL_CAP_BP == [13.41, 17.88]
    for e2e in (False, True):
      for v_ego in (0., 3., 8., 13.41):
        a, peak = settle_cruise_accel(e2e, v_ego, follow_share=1.)
        assert peak <= TRAFFIC_MODE_ACCEL_CAP + 1e-9
        self.assertAlmostEqual(a, TRAFFIC_MODE_ACCEL_CAP)
        # without Traffic Mode it lifts the cruise accel past it
        assert settle_cruise_accel(e2e, v_ego, follow_share=1., traffic_mode=False)[0] > TRAFFIC_MODE_ACCEL_CAP + 0.3

  def test_fades_out_by_40_mph(self):
    for e2e in (False, True):
      # half way, half of the step up to the request limit
      self.assertAlmostEqual(settle_cruise_accel(e2e, sum(TRAFFIC_MODE_ACCEL_CAP_BP) / 2, follow_share=1.)[0],
                             (TRAFFIC_MODE_ACCEL_CAP + ACCEL_MAX) / 2)
      for v_ego in (TRAFFIC_MODE_ACCEL_CAP_BP[1], 20., 30.):
        for follow_share in (0., 1.):
          assert settle_cruise_accel(e2e, v_ego, follow_share)[0] == settle_cruise_accel(e2e, v_ego, follow_share, traffic_mode=False)[0]

  def test_never_raises_a_lower_cap(self):
    # every cap only comes down (launch cap, turn limit, coast limit, Curve Speed Control's throttle scale), and without
    # Traffic Mode it is exactly as before
    for e2e in (False, True):
      for v_ego in np.linspace(0., 20., 11):
        for personality in PERSONALITIES + (None,):
          for follow_share in (0., 0.5, 1.):
            for allow_throttle, coast in ((True, ACCEL_MAX), (False, -0.3), (False, 0.8)):
              for throttle_scale in (1., 0.5, 0.):
                for a_prev in (-1., 0., 1., 2.):
                  args = (e2e, 40., v_ego, a_prev, 0., CP, DT_MDL, coast, allow_throttle, follow_share, throttle_scale, personality)
                  assert get_cruise_accel(*args, True) <= get_cruise_accel(*args, False) + 1e-12
                  assert get_cruise_accel(*args) == get_cruise_accel(*args, False)
    # under it they stay as they are: Relaxed's 1.0 launch cap, the coast limit, the throttle scale
    for e2e in (False, True):
      self.assertAlmostEqual(settle_cruise_accel(e2e, 2., personality=Personality.relaxed)[0], 1.)
    for args in ((False, 40., 6., 0., 0., CP, DT_MDL, -0.3, False, 1., 1., Personality.aggressive),
                 (True, 40., 2., 0., 0., CP, DT_MDL, ACCEL_MAX, True, 0., 0.5, Personality.relaxed)):
      a_tm, a = args[3], args[3]
      for _ in range(100):
        a_tm = get_cruise_accel(*args[:3], a_tm, *args[4:], True)
        a = get_cruise_accel(*args[:3], a, *args[4:], False)
      assert a_tm == a < TRAFFIC_MODE_ACCEL_CAP


class TestTrafficModePlanner(OpenpilotTestCase):
  def setup_method(self):
    self.params = Params()

  @staticmethod
  def make_planner(openpilot_long=True):
    CP = structs.CarParams(openpilotLongitudinalControl=openpilot_long)
    mpc = SimpleNamespace(traffic_mode=False)
    return LongitudinalPlannerSP(CP, structs.CarParamsSP(), mpc), mpc

  def test_poll_every_read_frames(self):
    self.params.put_bool("TrafficMode", True, block=True)
    planner, mpc = self.make_planner()
    planner.update_traffic_mode()
    assert mpc.traffic_mode

    # switched off mid-drive: applied within TRAFFIC_MODE_READ_FRAMES planner frames (0.25 s at 20 Hz)
    self.params.put_bool("TrafficMode", False, block=True)
    for _ in range(TRAFFIC_MODE_READ_FRAMES - 1):
      planner.update_traffic_mode()
      assert mpc.traffic_mode
    planner.update_traffic_mode()
    assert not mpc.traffic_mode

  def test_not_read_without_openpilot_long(self):
    self.params.put_bool("TrafficMode", True, block=True)
    planner, mpc = self.make_planner(openpilot_long=False)
    for _ in range(3 * TRAFFIC_MODE_READ_FRAMES):
      planner.update_traffic_mode()
    assert not mpc.traffic_mode

  def test_read_error_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.longitudinal_planner.cloudlog")
    planner, mpc = self.make_planner()
    mpc.traffic_mode = True
    planner._params = mocker.MagicMock()
    planner._params.get_bool.side_effect = UnknownKeyName(b"TrafficMode")
    for _ in range(3 * TRAFFIC_MODE_READ_FRAMES):
      planner.update_traffic_mode()
    assert not mpc.traffic_mode
    assert cloudlog.exception.call_count == 1

  def test_offset_param_never_written(self):
    self.params.put("SPFollowingTimeOffset", -6, block=True)
    planner, mpc = self.make_planner()
    for on in (True, False, True, False):
      self.params.put_bool("TrafficMode", on, block=True)
      for _ in range(TRAFFIC_MODE_READ_FRAMES):
        planner.update_traffic_mode()
      assert mpc.traffic_mode == on
      assert mpc.t_follow_offset_pct == -6.
    assert self.params.get("SPFollowingTimeOffset") == -6

  def test_planner_passes_follow_time_and_logs_it(self):
    CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=2.7, longitudinalActuatorDelay=0.2)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=V_EGO)
    planner.mpc.t_follow_offset_pct = 10
    sm, pm = build_sm(V_EGO), MockPubMaster()

    for on in (True, False, True):
      self.params.put_bool("TrafficMode", on, block=True)
      for _ in range(TRAFFIC_MODE_READ_FRAMES):
        planner.update(sm)
        planner.publish(sm, pm)

      # personality is Aggressive (selfdriveState default)
      t_follow = planner.mpc.params[0, 4]
      if on:
        assert t_follow == get_traffic_mode_T_FOLLOW(planner.mpc.x0[1])
        assert 0.5 < t_follow < 1.0
      else:
        assert t_follow == get_T_FOLLOW(Personality.aggressive) * 1.1

      fd = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance
      assert fd.trafficMode == on
      self.assertAlmostEqual(fd.tFollow, t_follow, places=6)
      assert fd.offsetPct == (0. if on else 10.)
      assert fd.jerkFactor == 0.5
      # no launch shift (moving, behind a vision lead) and no fast estimate (a vision lead)
      assert fd.trafficModeShift == planner.mpc.launch_shift == 0.
      assert fd.aLeadFast == 0.

  def test_planner_logs_the_launch_shift(self):
    # stopped 3.7 m behind a radar lead in Traffic Mode: the shift builds up to 2 m and is logged
    self.params.put_bool("TrafficMode", True, block=True)
    CP = structs.CarParams(openpilotLongitudinalControl=True, steerRatio=15., wheelbase=2.7, longitudinalActuatorDelay=0.2)
    planner = LongitudinalPlanner(CP, structs.CarParamsSP(), init_v=0.)
    sm, pm = build_sm(0.), MockPubMaster()
    radar = messaging.new_message('radarState')
    radar.radarState.leadOne = {"present": True, "radar": True, "radarTrackId": 4, "dRel": 3.7, "vLead": 0., "aLeadTau": 1.5,
                                "modelProb": 1.}
    sm['radarState'] = radar.radarState.as_reader()
    shifts = []
    for _ in range(TRAFFIC_MODE_READ_FRAMES + 60):
      planner.update(sm)
      planner.publish(sm, pm)
      fd = pm.sent['longitudinalPlanSP'][-1].longitudinalPlanSP.followDistance
      self.assertAlmostEqual(fd.trafficModeShift, planner.mpc.launch_shift, places=6)
      shifts.append(fd.trafficModeShift)
    assert fd.trafficMode
    # building up, then all of it
    assert 0. < shifts[10] < 1.
    self.assertAlmostEqual(shifts[-1], 2., places=6)
