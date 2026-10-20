"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from types import SimpleNamespace

import numpy as np

from opendbc.car import structs
from openpilot.common.params import Params, ParamKeyFlag, UnknownKeyName
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner
from openpilot.sunnypilot.selfdrive.controls.lib.dec.constants import WMACConstants
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

READ_FRAMES = int(1. / DT_MDL)

# Route 539e9b3496be3ce9/00000017 at 5:15.00-5:21.50, every 0.25 s (the first model frame at or after each time):
# Experimental + DEC, 50 mph, the model slowing for a red light. vEgo (m/s) and the model's 10 s plan endpoint (m). The
# shipped DEC switched to blended at 5:20.01 with the model already asking for -0.91 m/s^2; the model first asked for -0.3
# at 5:18.10. The radar lead appears at 5:21.26, after both switches.
ROUTE_V = [22.3, 22.3, 22.2, 22.3, 22.2, 22.3, 22.3, 22.2, 22.3, 22.2, 22.3, 22.2, 22.2, 22.3, 22.2, 22.3, 22.3, 22.3, 22.2,
           22.2, 22.1, 21.9, 21.6, 21.3, 20.9, 20.5, 20.0]
ROUTE_END_X = [197.5, 199.1, 196.4, 196.4, 195.0, 192.2, 191.5, 185.9, 184.2, 175.8, 173.9, 173.6, 167.8, 162.1, 158.6, 162.2,
               157.1, 154.1, 142.5, 142.8, 138.0, 131.8, 127.9, 121.8, 118.1, 112.6, 105.4]
ROUTE_SHIPPED_SWITCH = 5.01  # s after 5:15.00


class FakeParams:
  def __init__(self, brake_sooner=True):
    self.values = {"DynamicExperimentalControl": True, "BrakeSoonerForStops": brake_sooner}
    self.reads = 0

  def get_bool(self, key):
    if key == "BrakeSoonerForStops":
      self.reads += 1
    return self.values[key]


def sm_for(v_ego, end_x, lead=False):
  # the model's plan: evenly spaced to end_x (DEC only looks at the endpoint and the trajectory length)
  model = SimpleNamespace(position=SimpleNamespace(x=list(np.linspace(0., end_x, 33))), orientation=SimpleNamespace(x=[0.] * 33))
  return {'carState': SimpleNamespace(vEgo=v_ego, vCruise=v_ego * 3.6 + 10., standstill=False),
          'radarState': SimpleNamespace(leadOne=SimpleNamespace(present=float(lead))),
          'modelV2': model,
          'selfdriveState': SimpleNamespace(experimentalMode=True)}


def make_dec(brake_sooner=True):
  return DynamicExperimentalController(SimpleNamespace(radarUnavailable=False), SimpleNamespace(crash_cnt=0),
                                       params=FakeParams(brake_sooner))


def run(dec, frames):
  """frames: (vEgo, endpoint) per 20 Hz frame. Returns the time (s) of the first blended frame, or None"""
  for i, (v, end_x) in enumerate(frames):
    dec.update(sm_for(v, end_x))
    if dec.mode() == 'blended':
      return i * DT_MDL
  return None


def cruise(v_ego, ratio=0.97, seconds=3.):
  return [(v_ego, ratio * v_ego * 10.)] * int(seconds / DT_MDL)


def route_frames():
  t = np.arange(0., (len(ROUTE_V) - 1) * 0.25 + 1e-6, DT_MDL)
  ts = np.arange(len(ROUTE_V)) * 0.25
  return list(zip(np.interp(t, ts, ROUTE_V), np.interp(t, ts, ROUTE_END_X), strict=True))


class TestBrakeSoonerCurve(OpenpilotTestCase):
  def test_unchanged_up_to_60_kph(self):
    for kph in np.arange(0., 60.01, 0.5):
      assert np.interp(kph, WMACConstants.SLOW_DOWN_BP_BRAKE_SOONER, WMACConstants.SLOW_DOWN_DIST_BRAKE_SOONER) == \
             np.interp(kph, WMACConstants.SLOW_DOWN_BP, WMACConstants.SLOW_DOWN_DIST)

  def test_grows_with_speed_to_100_kph_then_holds(self):
    for kph in (70., 80., 90., 100.):
      dist = np.interp(kph, WMACConstants.SLOW_DOWN_BP_BRAKE_SOONER, WMACConstants.SLOW_DOWN_DIST_BRAKE_SOONER)
      assert abs(dist - 0.92 * kph / 3.6 * 10.) < 0.1
    assert np.all(np.diff(WMACConstants.SLOW_DOWN_DIST_BRAKE_SOONER) > 0)
    for kph in (110., 130., 160.):
      assert np.interp(kph, WMACConstants.SLOW_DOWN_BP_BRAKE_SOONER, WMACConstants.SLOW_DOWN_DIST_BRAKE_SOONER) == 255.6


class TestBrakeSoonerSwitch(OpenpilotTestCase):
  def test_route_red_light_switches_sooner(self):
    frames = cruise(22.3) + route_frames()
    lead_in = len(cruise(22.3)) * DT_MDL
    shipped = run(make_dec(brake_sooner=False), frames)
    sooner = run(make_dec(brake_sooner=True), frames)
    # off reproduces what the truck did; on hands over to the model about 2.5 s sooner, as it starts to slow
    assert shipped is not None and abs(shipped - lead_in - ROUTE_SHIPPED_SWITCH) < 0.3, shipped
    assert sooner is not None and shipped - sooner > 2.0, (sooner, shipped)
    assert sooner - lead_in > 1.5  # not before the model starts slowing (endpoint 182 m at 2.0 s)

  def test_steady_cruise_stays_acc(self):
    # cruising plans end at 93-99% of the 10 s distance (route 00000017); 90% is a margin below that
    for kph in (65., 70., 80., 90., 100., 110., 130.):
      v = kph / 3.6
      for ratio in (0.97, 0.93, 0.90):
        assert run(make_dec(), cruise(v, ratio, 30.)) is None, (kph, ratio)

  def test_holds_the_100_kph_distance_above_it(self):
    # above 100 km/h it keeps 255.6 m: still sooner than stock's 165 m, but by less than at 100 km/h
    for kph, both_acc, new_only in ((110., 0.82, 0.75), (130., 0.71, 0.64)):
      v = kph / 3.6
      for ratio, new_blended in ((both_acc, False), (new_only, True)):
        frames = cruise(v) + [(v, ratio * v * 10.)] * int(20. / DT_MDL)
        assert run(make_dec(brake_sooner=False), frames) is None, (kph, ratio)
        assert (run(make_dec(brake_sooner=True), frames) is not None) == new_blended, (kph, ratio)

  def test_same_as_stock_up_to_60_kph(self):
    for kph in (20., 40., 55., 60.):
      v = kph / 3.6
      frames = cruise(v) + [(v, v * 10. * r) for r in np.linspace(0.97, 0.3, 100)]
      stock, new = make_dec(brake_sooner=False), make_dec(brake_sooner=True)
      for v_ego, end_x in frames:
        stock.update(sm_for(v_ego, end_x))
        new.update(sm_for(v_ego, end_x))
        assert stock.mode() == new.mode() and stock._urgency == new._urgency, kph

  def test_radar_lead_still_means_acc(self):
    dec = make_dec()
    for v, end_x in cruise(22.3) + route_frames():
      dec.update(sm_for(v, end_x, lead=True))
      assert dec.mode() == 'acc'


class TestBrakeSoonerSetting(OpenpilotTestCase):
  def test_read_every_second(self):
    dec = make_dec(brake_sooner=False)
    assert not dec._brake_sooner
    dec._params.values["BrakeSoonerForStops"] = True
    dec.update(sm_for(20., 190.))  # frame 0 reads
    assert dec._brake_sooner
    dec._params.values["BrakeSoonerForStops"] = False
    for _ in range(READ_FRAMES - 1):
      dec.update(sm_for(20., 190.))
      assert dec._brake_sooner
    dec.update(sm_for(20., 190.))
    assert not dec._brake_sooner
    assert dec._params.reads == 3  # init, frame 0, frame 20

  def test_read_error_is_off_and_logged_once(self, mocker):
    cloudlog = mocker.patch("openpilot.sunnypilot.selfdrive.controls.lib.dec.dec.cloudlog")
    params = FakeParams(brake_sooner=True)
    dec = DynamicExperimentalController(SimpleNamespace(radarUnavailable=False), SimpleNamespace(crash_cnt=0), params=params)
    dec.update(sm_for(20., 190.))
    assert dec._brake_sooner
    # a missing key (UnknownKeyName) and a failed read in the C layer (RuntimeError) both mean off, logged once
    errors = iter([UnknownKeyName(b"BrakeSoonerForStops"), RuntimeError("read failed")] * 3 * READ_FRAMES)
    good = params.get_bool
    params.get_bool = lambda key: (_ for _ in ()).throw(next(errors)) if key == "BrakeSoonerForStops" else good(key)
    for _ in range(3 * READ_FRAMES):
      dec.update(sm_for(20., 190.))
    assert not dec._brake_sooner and dec.enabled()
    assert len(cloudlog.mock_calls) == 1 and cloudlog.exception.call_count == 1, cloudlog.mock_calls

  def test_default_on_and_kept(self):
    params = Params()
    # manager writes the default for unset keys at startup
    assert params.get_default_value("BrakeSoonerForStops") is True
    params.put_bool("BrakeSoonerForStops", False, block=True)
    for flag in (ParamKeyFlag.CLEAR_ON_MANAGER_START, ParamKeyFlag.CLEAR_ON_ONROAD_TRANSITION,
                 ParamKeyFlag.CLEAR_ON_OFFROAD_TRANSITION, ParamKeyFlag.CLEAR_ON_IGNITION_ON):
      params.clear_all(flag)
    assert params.get("BrakeSoonerForStops") is False  # None if it was cleared
    assert b"BrakeSoonerForStops" in params.all_keys(ParamKeyFlag.BACKUP)

  def test_planner_dec_reads_the_setting(self):
    params = Params()
    for value in (True, False):
      params.put_bool("BrakeSoonerForStops", value, block=True)
      planner = LongitudinalPlanner(structs.CarParams(openpilotLongitudinalControl=True), structs.CarParamsSP())
      assert planner.dec._brake_sooner is value
