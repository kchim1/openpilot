import copy
import unittest
from types import SimpleNamespace

from opendbc.car import Bus, structs
from opendbc.car.rivian.ext_controller import ExternalController, MIN_TORQUE_FRAMES, HANDS_OFF_EXIT_FRAMES
from opendbc.car.rivian.fingerprints import FW_VERSIONS
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
