import copy
import unittest
from types import SimpleNamespace

from opendbc.car import Bus, structs
from opendbc.car.rivian.ext_controller import ExternalController, MIN_TORQUE_FRAMES, HANDS_OFF_EXIT_FRAMES
from opendbc.car.rivian.fingerprints import FW_VERSIONS
from opendbc.car.rivian.carcontroller import ACM_ENGAGE_TIMEOUT_FRAMES
from opendbc.car.rivian.interface import CarInterface
from opendbc.car.rivian.values import CAR, FW_QUERY_CONFIG, WMI, ModelLine, ModelYear

GearShifter = structs.CarState.GearShifter


class TestRivian(unittest.TestCase):
  def test_custom_fuzzy_fingerprinting(self):
    for platform in CAR:
      with self.subTest(platform=platform.name):
        for wmi in WMI:
          for line in ModelLine:
            for year in ModelYear:
              for bad in (True, False):
                vin = ["0"] * 17
                vin[:3] = wmi
                vin[3] = line.value
                vin[9] = year.value
                if bad:
                  vin[3] = "Z"
                vin = "".join(vin)

                matches = FW_QUERY_CONFIG.match_fw_to_car_fuzzy({}, vin, FW_VERSIONS)
                should_match = year in platform.config.years and not bad
                assert (matches == {platform}) == should_match, "Bad match"


class TestExternalControllerForceTorque(unittest.TestCase):
  @staticmethod
  def _run(erc, frames, lat_active=True):
    actuators = SimpleNamespace(steeringAngleDeg=0.0, torque=0.3)
    states = []
    for _ in range(frames):
      # hands off, wheel settled on the commanded angle; the EPAS reports ready, then active once it is steering on angle
      out = SimpleNamespace(steeringAngleDeg=0.0, steeringRateDeg=0.0, steeringTorque=0.0, steeringPressed=False,
                            vEgo=25.0, vEgoRaw=25.0, aEgo=0.0)
      CS = SimpleNamespace(out=out, hands_on_level=0, eac_status=2 if erc.angle_active else 1, eac_error_code=0,
                           sccm_wheel_touch={"SCCM_WheelTouch_Calibration": 1000, "SCCM_WheelTouch_CapacitiveValue": 0})
      erc.update(CS, lat_active, actuators)
      states.append((erc.torque_active, erc.angle_active, erc.apply_torque_last))
    return states

  def test_default_steers_on_angle(self):
    erc = ExternalController()
    assert not erc.force_torque
    for torque_active, angle_active, apply_torque in self._run(erc, 300):
      assert not torque_active and angle_active and apply_torque == 0

  def test_force_torque_never_hands_back(self):
    erc = ExternalController()
    erc.force_torque = True
    # well past every hand-back condition: hands off, EPAS ready, wheel settled on the commanded angle
    states = self._run(erc, 4 * (MIN_TORQUE_FRAMES + HANDS_OFF_EXIT_FRAMES))
    for torque_active, angle_active, _ in states:
      assert torque_active and not angle_active
    assert states[-1][2] > 0 and erc.toi_act_cmd

    # lateral off: everything idle, same as the normal mode
    for torque_active, angle_active, apply_torque in self._run(erc, 5, lat_active=False):
      assert not torque_active and not angle_active and apply_torque == 0
    assert not erc.toi_act_cmd


class TestBlockedEngagePress(unittest.TestCase):
  """The ACC stalk is also the gear selector: only a stalk-down in Drive may be reported as a blocked ACC press."""

  @staticmethod
  def _run(seq, engageable=False):
    # seq: (gear, VDM_UserAdasRequest) per frame. Returns the frames that raised engage_request_blocked.
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CI = CarInterface(CP, structs.CarParamsSP())
    CC, CS = CI.CC, CI.CS
    CC.openpilot_engageable = engageable
    CS.acm_lka_hba_cmd = copy.copy(CI.can_parsers[Bus.cam].vl["ACM_lkaHbaCmd"])
    CS.sccm_wheel_touch = copy.copy(CI.can_parsers[Bus.pt].vl["SCCM_WheelTouch"])
    adas_status = copy.copy(CI.can_parsers[Bus.pt].vl["VDM_AdasSts"])
    car_control = structs.CarControl().as_reader()

    raised = []
    for frame, (gear, request) in enumerate(seq):
      CS.out = structs.CarState(gearShifter=gear, vEgo=0.3, vEgoRaw=0.3)
      CS.vdm_adas_status = [dict(adas_status, VDM_UserAdasRequest=request)]
      CC.update(car_control, structs.CarControlSP(), CS, 0)
      if CC.engage_request_blocked:
        raised.append(frame)
        CC.engage_request_blocked = False
    return raised

  def test_gear_changes_are_not_engage_presses(self):
    D, R, P = GearShifter.drive, GearShifter.reverse, GearShifter.park
    self.assertEqual(self._run([(D, 0), (D, 2), (R, 2), (R, 0)]), [])  # D -> R, stalk up
    self.assertEqual(self._run([(R, 0), (R, 4), (D, 4), (D, 0)]), [])  # R -> D, stalk down
    self.assertEqual(self._run([(P, 0), (P, 3), (D, 3), (D, 4), (D, 0)]), [])  # P -> D
    self.assertEqual(self._run([(D, 0), (D, 1), (D, 0)]), [])  # cancel

  def test_acc_press_in_drive_is_reported(self):
    D = GearShifter.drive
    self.assertEqual(self._run([(D, 0), (D, 3), (D, 4), (D, 0)]), [1])
    self.assertEqual(self._run([(D, 0), (D, 4), (D, 0), (D, 4), (D, 0)]), [1, 3])
    # openpilot can engage: the press goes to the ACM, nothing to report
    self.assertEqual(self._run([(D, 0), (D, 4), (D, 0)], engageable=True), [])


class TestDeclinedEngagePress(unittest.TestCase):
  """An ACC-on press in Drive that the stock ACM does not take within 0.5 s is reported once, with a reason."""

  @staticmethod
  def _run(gear, v_ego, request=4, acc_engages_at=None, fault_supervisor_state=0, gear_after=None, shift_at=None):
    # one stalk press (VDM_UserAdasRequest 0 -> request -> 0), then frames until past the timeout.
    # Returns every engage_request_refused reason raised
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CI = CarInterface(CP, structs.CarParamsSP())
    CC, CS = CI.CC, CI.CS
    CC.openpilot_engageable = True
    CS.acm_lka_hba_cmd = copy.copy(CI.can_parsers[Bus.cam].vl["ACM_lkaHbaCmd"])
    CS.sccm_wheel_touch = copy.copy(CI.can_parsers[Bus.pt].vl["SCCM_WheelTouch"])
    CS.acm_fault_supervisor_state = fault_supervisor_state
    adas_status = copy.copy(CI.can_parsers[Bus.pt].vl["VDM_AdasSts"])
    car_control = structs.CarControl().as_reader()

    refused = []
    for frame in range(ACM_ENGAGE_TIMEOUT_FRAMES + 20):
      enabled = acc_engages_at is not None and frame >= acc_engages_at
      gear_now = gear_after if shift_at is not None and frame >= shift_at else gear
      CS.out = structs.CarState(gearShifter=gear_now, vEgo=v_ego, vEgoRaw=v_ego, cruiseState={"enabled": enabled})
      CS.vdm_adas_status = [dict(adas_status, VDM_UserAdasRequest=request if 1 <= frame <= 5 else 0)]
      CC.update(car_control, structs.CarControlSP(), CS, 0)
      if CC.engage_request_refused is not None:
        refused.append(CC.engage_request_refused)
        CC.engage_request_refused = None
    return refused

  def test_declined_below_20_mph_in_drive(self):
    D = GearShifter.drive
    for request in (3, 4):
      self.assertEqual(self._run(D, 7.5, request), ["lowSpeed"])  # 16.8 mph, route 539e9b3496be3ce9/00000008
    self.assertEqual(self._run(D, 0.0), ["lowSpeed"])

  def test_not_ready_takes_precedence(self):
    self.assertEqual(self._run(GearShifter.drive, 7.5, fault_supervisor_state=2), ["notReady"])

  def test_accepted_press_is_not_reported(self):
    self.assertEqual(self._run(GearShifter.drive, 7.5, acc_engages_at=4), [])

  def test_declined_at_20_mph_and_above_stays_silent(self):
    self.assertEqual(self._run(GearShifter.drive, 9.0), [])

  def test_stalk_down_out_of_drive_is_a_gear_change(self):
    for gear in (GearShifter.park, GearShifter.reverse, GearShifter.neutral):
      self.assertEqual(self._run(gear, 0.0), [])
      self.assertEqual(self._run(gear, 3.0), [])

  def test_shifting_out_of_drive_cancels_the_report(self):
    D = GearShifter.drive
    for gear_after in (GearShifter.park, GearShifter.neutral, GearShifter.reverse):
      self.assertEqual(self._run(D, 0.0, gear_after=gear_after, shift_at=20), [])


class TestCruiseAccelBias:
  def test_only_cruising_at_highway_speed(self):
    from opendbc.car.rivian.carcontroller import cruise_accel_bias, CRUISE_ACCEL_BIAS
    assert cruise_accel_bias(10., 0.) == 0.
    assert cruise_accel_bias(25., 0.3) == CRUISE_ACCEL_BIAS
    assert cruise_accel_bias(25., -0.2) == CRUISE_ACCEL_BIAS
    assert 0. < cruise_accel_bias(18.5, 0.) < CRUISE_ACCEL_BIAS
    # braking is left alone, with a fade rather than a step
    assert cruise_accel_bias(25., -1.5) == 0.
    assert 0. < cruise_accel_bias(25., -0.75) < CRUISE_ACCEL_BIAS


class TestLongRequestWhileOverridden(unittest.TestCase):
  """With the gas pressed the panda drops any long request but an exact 0, and a run of dropped requests faults the ACM
  ("Cruise Fault: Restart the Car", routes 00000015 and 00000016)."""

  @staticmethod
  def _sent_accel(v_ego, accel, long_active, gas_pressed):
    # raw ACM_AccelerationRequest of the 0x160 frame, decoded the way the panda safety does (opendbc/safety/modes/rivian.h)
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CI = CarInterface(CP, structs.CarParamsSP())
    CC, CS = CI.CC, CI.CS
    CS.acm_lka_hba_cmd = copy.copy(CI.can_parsers[Bus.cam].vl["ACM_lkaHbaCmd"])
    CS.sccm_wheel_touch = copy.copy(CI.can_parsers[Bus.pt].vl["SCCM_WheelTouch"])
    CS.vdm_adas_status = [copy.copy(CI.can_parsers[Bus.pt].vl["VDM_AdasSts"])]
    CS.out = structs.CarState(gearShifter=GearShifter.drive, vEgo=v_ego, vEgoRaw=v_ego, gasPressed=gas_pressed,
                              cruiseState={"enabled": True})
    car_control = structs.CarControl(enabled=True, longActive=long_active, actuators={"accel": accel}).as_reader()
    _, can_sends = CC.update(car_control, structs.CarControlSP(), CS, 0)
    data = next(m[1] for m in can_sends if m[0] == 0x160)
    return ((data[2] << 3) | (data[3] >> 5)) - 1024

  def test_gas_override_sends_exactly_zero(self):
    # controlsd zeroes the accel while the gas overrides; the cruise bias must not turn that into a small request
    for v_ego in (0., 10., 15.5, 18.5, 22., 30., 40.):
      self.assertEqual(self._sent_accel(v_ego, 0., long_active=False, gas_pressed=True), 0, v_ego)

  def test_bias_applies_while_openpilot_controls_speed(self):
    self.assertEqual(self._sent_accel(30., 0., long_active=True, gas_pressed=False), -8)  # 0.01 m/s^2 per bit
    self.assertEqual(self._sent_accel(10., 0., long_active=True, gas_pressed=False), 0)
