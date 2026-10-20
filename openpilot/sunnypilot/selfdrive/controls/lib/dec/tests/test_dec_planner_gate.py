"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from typing import cast

import numpy as np

from openpilot.cereal import custom, messaging
from opendbc.car import structs
from openpilot.common.realtime import DT_MDL
from openpilot.common.test import OpenpilotTestCase
from openpilot.selfdrive.controls.lib.longitudinal_planner import LongitudinalPlanner, LongitudinalPlanSource, A_CRUISE_MAX_BP, \
  J_CRUISE_VALS, get_max_accel
from openpilot.sunnypilot.selfdrive.controls.lib.dec.dec import DynamicExperimentalController

V_EGO = 20.0
E2E_ACCEL = -3.0  # low enough that e2e wins the min() whenever it is a candidate


class MockDec:
  def __init__(self, active: bool, mode: str):
    self._active = active
    self._mode = mode

  def update(self, sm):
    pass

  def active(self) -> bool:
    return self._active

  def mode(self) -> str:
    return self._mode

  def enabled(self) -> bool:
    return True


class MockSubMaster(dict):
  def __init__(self, services: dict):
    super().__init__(services)
    self.valid = dict.fromkeys(services, True)
    self.logMonoTime = dict.fromkeys(services, 0)
    self.updated = dict.fromkeys(services, True)
    self.recv_frame = dict.fromkeys(services, 1)

  def all_checks(self, service_list=None) -> bool:
    return True


def build_sm(experimental_mode: bool, e2e_accel: float = E2E_ACCEL) -> MockSubMaster:
  services = {}
  for service in ("radarState", "controlsState", "vehicleParameters", "carStateSP",
                  "liveMapDataSP", "gpsLocationExternal", "gpsLocation"):
    services[service] = getattr(messaging.new_message(service), service)

  car_state = messaging.new_message('carState')
  car_state.carState.vEgo = V_EGO
  car_state.carState.vCruise = 100.0
  car_state.carState.vCruiseCluster = 100.0
  services['carState'] = car_state.carState.as_reader()

  selfdrive_state = messaging.new_message('selfdriveState')
  selfdrive_state.selfdriveState.experimentalMode = experimental_mode
  selfdrive_state.selfdriveState.enabled = True
  services['selfdriveState'] = selfdrive_state.selfdriveState.as_reader()

  car_control = messaging.new_message('carControl')
  car_control.carControl.enabled = True
  services['carControl'] = car_control.carControl.as_reader()

  model = messaging.new_message('modelV2')
  model.modelV2.orientationRate.z = [0.01] * 33  # nonzero: a straight path divides by zero in SCC vision
  model.modelV2.velocity.x = [V_EGO] * 33
  model.modelV2.position.x = [float(i) for i in range(33)]
  model.modelV2.action.desiredAcceleration = e2e_accel
  services['modelV2'] = model.modelV2.as_reader()

  return MockSubMaster(services)


def build_planner(dec_active: bool, dec_mode: str) -> LongitudinalPlanner:
  CP = structs.CarParams()
  CP.steerRatio = 15.0
  CP.wheelbase = 2.7
  CP.longitudinalActuatorDelay = 0.2
  CP_SP = custom.CarParamsSP.new_message().as_reader()

  planner = LongitudinalPlanner(CP, CP_SP, init_v=V_EGO)
  planner.dec = cast(DynamicExperimentalController, MockDec(dec_active, dec_mode))
  return planner


class TestDecPlannerGate(OpenpilotTestCase):
  """The e2e candidate must be gated on is_e2e(), not raw experimentalMode."""

  def _source(self, experimental_mode: bool, dec_active: bool, dec_mode: str) -> LongitudinalPlanSource:
    planner = build_planner(dec_active, dec_mode)
    planner.update(build_sm(experimental_mode))
    return planner.mpc.source

  def test_no_e2e_when_experimental_mode_off(self):
    assert self._source(False, False, 'acc') != LongitudinalPlanSource.e2e

  def test_e2e_when_dec_inactive(self):
    # DEC off: behavior must match upstream
    assert self._source(True, False, 'acc') == LongitudinalPlanSource.e2e

  def test_e2e_when_dec_blended(self):
    assert self._source(True, True, 'blended') == LongitudinalPlanSource.e2e

  def test_no_e2e_when_dec_holds_acc(self):
    # the regression
    assert self._source(True, True, 'acc') != LongitudinalPlanSource.e2e

  def test_no_surge_when_dec_hands_back_to_acc(self):
    # 72 km/h, set speed 100 km/h, no lead. While the model is in control its candidate wins the min(), and the cruise
    # accel behind it builds up to ACCEL_MAX. Handing back to ACC must not let that through in one step: the target
    # builds up again from where it was, no faster than the cruise jerk, to the ACC cap
    planner = build_planner(True, 'blended')
    for _ in range(int(3. / DT_MDL)):
      planner.update(build_sm(True, e2e_accel=0.2))
    a_blended = float(planner.output_a_target)
    assert abs(a_blended - 0.2) < 1e-3
    planner.dec = cast(DynamicExperimentalController, MockDec(True, 'acc'))
    targets = []
    for _ in range(int(3. / DT_MDL)):
      planner.update(build_sm(True, e2e_accel=0.2))
      targets.append(float(planner.output_a_target))
    j_cruise = float(np.interp(V_EGO, A_CRUISE_MAX_BP, J_CRUISE_VALS))
    assert max(np.diff([a_blended] + targets)) <= j_cruise * DT_MDL + 1e-6, targets[:5]
    assert abs(targets[-1] - get_max_accel(V_EGO)) < 1e-3
