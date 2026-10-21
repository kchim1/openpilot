#!/usr/bin/env python3
import unittest
from types import SimpleNamespace
import numpy as np

from opendbc.can import CANParser

from opendbc.car import Bus, structs
from opendbc.car.lateral import get_max_angle_delta_vm, get_max_angle_vm
from opendbc.car.rivian.carcontroller import get_safety_CP
from opendbc.car.rivian.ext_controller import ExternalController, TOI_CLEAR_LATCH_FRAMES
from opendbc.car.rivian.interface import CarInterface
from opendbc.car.rivian.values import CAR, CarControllerParams, RivianSafetyFlags
from opendbc.car.rivian.riviancan import checksum as _checksum
from opendbc.car.structs import CarParams
from opendbc.car.vehicle_model import VehicleModel
from opendbc.safety import ALTERNATIVE_EXPERIENCE
from opendbc.safety.tests.libsafety import libsafety_py
import opendbc.safety.tests.common as common
from opendbc.safety.tests.common import CANPackerSafety


def checksum(msg):
  addr, dat, bus = msg
  ret = bytearray(dat)

  # ESP_Status
  if addr == 0x208:
    ret[0] = _checksum(ret[1:], 0x1D, 0xB1)
  elif addr == 0x150:
    ret[0] = _checksum(ret[1:], 0x1D, 0x9A)

  return addr, ret, bus


class TestRivianSafetyBase(common.CarSafetyTest, common.AngleSteeringSafetyTest, common.DriverTorqueSteeringSafetyTest,
                           common.LongitudinalAccelSafetyTest):

  TX_MSGS = [[0x100, 0], [0x110, 0], [0x120, 0], [0x321, 2], [0x162, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x100, 0x110, 0x120), 2: (0x321, 0x162)}
  FWD_BLACKLISTED_ADDRS = {0: [0x321, 0x162], 2: [0x100, 0x110, 0x120]}

  # Torque limits
  MAX_TORQUE_LOOKUP = [9, 17], [350, 250]
  DYNAMIC_MAX_TORQUE = True
  MAX_RATE_UP = 3
  MAX_RATE_DOWN = 5
  MAX_RT_DELTA = 125
  DRIVER_TORQUE_ALLOWANCE = 100
  DRIVER_TORQUE_FACTOR = 2

  # Angle limits (VM-based, no simple breakpoint rates)
  STEER_ANGLE_MAX = 360
  DEG_TO_CAN = 10
  ANGLE_RATE_BP = None
  ANGLE_RATE_UP = None
  ANGLE_RATE_DOWN = None
  LATERAL_FREQUENCY = 100

  cnt_speed = 0
  cnt_speed_2 = 0
  cnt_angle_cmd = 0

  def _get_steer_cmd_angle_max(self, speed):
    return get_max_angle_vm(max(speed, 1), self.VM, CarControllerParams)

  def _torque_driver_msg(self, torque):
    values = {"EPAS_TorsionBarTorque": torque / 100.0}
    return self.packer.make_can_msg_safety("EPAS_SystemStatus", 0, values)

  def _torque_cmd_msg(self, torque, steer_req=1):
    values = {"ACM_lkaStrToqReq": torque, "ACM_lkaActToi": steer_req}
    return self.packer.make_can_msg_safety("ACM_lkaHbaCmd", 0, values)

  def _angle_cmd_msg(self, angle: float, enabled: bool, increment_timer: bool = True):
    values = {"ACM_SteeringAngleRequest": angle, "ACM_EacEnabled": enabled}
    if increment_timer:
      self.safety.set_timer(self.cnt_angle_cmd * int(1e6 / self.LATERAL_FREQUENCY))
      self.__class__.cnt_angle_cmd += 1
    return self.packer.make_can_msg_safety("ACM_SteeringControl", 0, values)

  def _angle_meas_msg(self, angle: float):
    values = {"EPAS_InternalSas": angle}
    return self.packer.make_can_msg_safety("EPAS_AdasStatus", 0, values)

  def _speed_msg(self, speed, quality_flag=True):
    values = {"ESP_Vehicle_Speed": speed * 3.6, "ESP_Status_Counter": self.cnt_speed % 15,
              "ESP_Vehicle_Speed_Q": 1 if quality_flag else 0}
    self.__class__.cnt_speed += 1
    return self.packer.make_can_msg_safety("ESP_Status", 0, values, fix_checksum=checksum)

  def _speed_msg_2(self, speed, quality_flag=True):
    # Rivian has a dynamic max torque limit based on speed, so it checks two sources
    return self._user_gas_msg(0, speed, quality_flag)

  def _user_brake_msg(self, brake):
    values = {"iBESP2_BrakePedalApplied": brake}
    return self.packer.make_can_msg_safety("iBESP2", 0, values)

  def _user_gas_msg(self, gas, speed=0, quality_flag=True):
    values = {"VDM_AcceleratorPedalPosition": gas, "VDM_VehicleSpeed": speed * 3.6,
              "VDM_PropStatus_Counter": self.cnt_speed_2 % 15, "VDM_VehicleSpeedQ": 1 if quality_flag else 0}
    self.__class__.cnt_speed_2 += 1
    return self.packer.make_can_msg_safety("VDM_PropStatus", 0, values, fix_checksum=checksum)

  def _pcm_status_msg(self, enable):
    values = {"ACM_FeatureStatus": enable, "ACM_Unkown1": 1}
    return self.packer.make_can_msg_safety("ACM_Status", 2, values)

  def _accel_msg(self, accel: float):
    values = {"ACM_AccelerationRequest": accel}
    return self.packer.make_can_msg_safety("ACM_longitudinalRequest", 0, values)

  def test_angle_cmd_when_enabled(self):
    # VM-based limits tested in test_lateral_accel_limit and test_lateral_jerk_limit
    pass

  def _can_to_deg(self, can_val):
    return can_val / self.DEG_TO_CAN

  @staticmethod
  def _round_speed(speed):
    """Round speed through CAN encoding to match what safety computes after fudge"""
    speed_kph_can = round((speed + 1) * 3.6 / 0.01) * 0.01
    stored = round(speed_kph_can / 3.6 * 1000)
    return max(stored / 1000.0 - 1.0, 1.0)

  def test_lateral_accel_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = max(self._round_speed(speed), 1)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)

        # safety: max_angle_can = (max_angle_deg * DEG_TO_CAN) + 1
        max_angle_can = int(get_max_angle_vm(speed, self.VM, CarControllerParams) * self.DEG_TO_CAN) + 1
        max_angle_can = min(max_angle_can, self.STEER_ANGLE_MAX * self.DEG_TO_CAN)

        # at limit
        self.safety.set_desired_angle_last(max_angle_can * sign)
        self.assertTrue(self._tx(self._angle_cmd_msg(self._can_to_deg(max_angle_can) * sign, True)))

        # 1 unit above limit
        above_can = max_angle_can + 1
        above_deg = self._can_to_deg(above_can) * sign
        self._tx(self._angle_cmd_msg(above_deg, True))
        should_tx = above_can > self.STEER_ANGLE_MAX * self.DEG_TO_CAN
        self.assertEqual(should_tx, self._tx(self._angle_cmd_msg(above_deg, True)))

  def test_lateral_jerk_limit(self):
    for speed in np.linspace(0, 40, 100):
      speed = max(self._round_speed(speed), 1)
      for sign in (-1, 1):
        self.safety.set_controls_allowed(True)
        self._reset_speed_measurement(speed + 1)
        self._tx(self._angle_cmd_msg(0, True))

        # safety: max_delta_can = (max_delta_deg * DEG_TO_CAN) + 1
        max_delta_can = int(get_max_angle_delta_vm(speed, self.VM, CarControllerParams) * self.DEG_TO_CAN) + 1

        # within limits
        self.assertTrue(self._tx(self._angle_cmd_msg(self._can_to_deg(max_delta_can) * sign, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(self._can_to_deg(max_delta_can) * sign, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

        # too high rate
        above_can = max_delta_can + 1
        self.assertFalse(self._tx(self._angle_cmd_msg(self._can_to_deg(above_can) * sign, True)))

        # recover
        self.safety.set_desired_angle_last(round(self._can_to_deg(above_can) * sign * self.DEG_TO_CAN))
        self.assertTrue(self._tx(self._angle_cmd_msg(self._can_to_deg(above_can) * sign, True)))
        self.assertFalse(self._tx(self._angle_cmd_msg(0, True)))
        self.assertTrue(self._tx(self._angle_cmd_msg(0, True)))

  def test_wheel_touch(self):
    # For hiding hold wheel alert on engage
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      values = {
        "SCCM_WheelTouch_HandsOn": 1 if controls_allowed else 0,
        "SCCM_WheelTouch_CapacitiveValue": 100 if controls_allowed else 0,
        "SCCM_WheelTouch_Calibration": 100,
      }
      self.assertTrue(self._tx(self.packer.make_can_msg_safety("SCCM_WheelTouch", 2, values)))

  def test_rx_hook(self):
    # checksum, counter, and quality flag checks
    for quality_flag in (True, False):
      for msg_type in ("speed", "speed_2"):
        self.safety.set_controls_allowed(True)
        # send multiple times to verify counter checks
        for _ in range(10):
          if msg_type == "speed":
            msg = self._speed_msg(0, quality_flag=quality_flag)
          elif msg_type == "speed_2":
            msg = self._speed_msg_2(0, quality_flag=quality_flag)

          self.assertEqual(quality_flag, self._rx(msg))
          self.assertEqual(quality_flag, self.safety.get_controls_allowed())

        # Mess with checksum to make it fail
        msg[0].data[0] = 0xff
        self.assertFalse(self._rx(msg))
        self.assertFalse(self.safety.get_controls_allowed())


class TestRivianStockSafety(TestRivianSafetyBase):

  LONGITUDINAL = False

  def setUp(self):
    self.VM = VehicleModel(get_safety_CP())
    self.packer = CANPackerSafety("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, 0)
    self.safety.init_tests()

  def test_adas_status(self):
    # For canceling stock ACC
    for controls_allowed in (True, False):
      self.safety.set_controls_allowed(controls_allowed)
      for interface_status in range(4):
        values = {"VDM_AdasInterfaceStatus": interface_status}
        self.assertTrue(self._tx(self.packer.make_can_msg_safety("VDM_AdasSts", 2, values)))


class TestRivianLongitudinalSafety(TestRivianSafetyBase):

  TX_MSGS = [[0x100, 0], [0x110, 0], [0x120, 0], [0x321, 2], [0x160, 0], [0x162, 2]]
  RELAY_MALFUNCTION_ADDRS = {0: (0x100, 0x110, 0x120, 0x160), 2: (0x321, 0x162)}
  FWD_BLACKLISTED_ADDRS = {0: [0x321, 0x162], 2: [0x100, 0x110, 0x120, 0x160]}

  def setUp(self):
    self.VM = VehicleModel(get_safety_CP())
    self.packer = CANPackerSafety("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, RivianSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()


class TestRivianTorqueControllerSafety(unittest.TestCase):
  """The real cooperative-torque controller (ext_controller) driven frame by frame (10 ms) through the panda safety model,
  so each is checked against the other instead of against a hand-copied limit. At a high wheel angle the controller
  blips the TOI request every ~0.9 s: torque 0 with the request low, then a ramp from 0."""

  SPEED = 11.4  # m/s
  T0 = 1_000_000  # us
  cnt_speed = 0

  def setUp(self):
    self.packer = CANPackerSafety("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, 0)
    self.safety.init_tests()

  def _rx(self, name, values):
    self.assertTrue(self.safety.safety_rx_hook(self.packer.make_can_msg_safety(name, 0, values, fix_checksum=checksum)))

  def _tx_torque(self, torque, steer_req):
    msg = self.packer.make_can_msg_safety("ACM_lkaHbaCmd", 0, {"ACM_lkaStrToqReq": torque, "ACM_lkaActToi": steer_req})
    return self.safety.safety_tx_hook(msg)

  def _setup(self, angle, timer_offset_frames=0):
    self.erc = ExternalController()
    self.erc.torque_active = True
    self.cs = SimpleNamespace(out=SimpleNamespace(vEgoRaw=self.SPEED, steeringTorque=0.0, steeringAngleDeg=angle),
                              toi_fault=False, torque_cmd_rejected=False)
    self.safety.init_tests()
    self.safety.set_controls_allowed(True)
    for _ in range(6):
      self._rx("ESP_Status", {"ESP_Vehicle_Speed": self.SPEED * 3.6, "ESP_Status_Counter": self.cnt_speed % 15, "ESP_Vehicle_Speed_Q": 1})
      self.__class__.cnt_speed += 1
      self._rx("EPAS_SystemStatus", {"EPAS_TorsionBarTorque": 0})
    # the panda's 250 ms real-time interval starts at its first torque frame: send one now and start the controller
    # timer_offset_frames later, so the interval is out of step with the controller's blip cycle
    self.safety.set_timer(self.T0)
    self.assertTrue(self._tx_torque(0, 1))
    self.frame = max(1, timer_offset_frames)

  def _frame(self, demand, toi_fault=False, refuse=False):
    """one controller frame through the panda; returns whether the panda let it through. refuse: the panda's lateral
    permission drops for this one frame, so it refuses a torque request and zeroes its torque memory"""
    self.safety.set_timer(self.T0 + self.frame * 10_000)
    self.frame += 1
    self.cs.toi_fault = toi_fault
    self.erc._update_torque(self.cs, SimpleNamespace(torque=demand))
    self.safety.set_controls_allowed(not refuse)
    ok = self._tx_torque(self.erc.apply_torque_last, int(self.erc.toi_act_cmd))
    self.safety.set_controls_allowed(True)
    # the refused frame comes back on bus 192 and reaches carstate by the next frame
    self.cs.torque_cmd_rejected = not ok
    return ok

  def test_torque_through_the_blip_is_never_refused(self):
    """A full-rate ramp, and a reversal, straddling the TOI blip at every phase of the panda's 250 ms real-time interval.
    On this branch a blip frame is an ordinary zero-torque frame to the panda, so it can never leave the real-time
    reference stale (AdventurePilot's safety code skips it, which made this fail there)."""
    blocked = []
    for sign in (1.0, -1.0):
      for offset in range(0, 26, 5):  # the panda's real-time interval against the blip cycle
        for start in range(0, 91, 10):  # where in the blip cycle the ramp starts
          self._setup(150.0, offset)
          for _ in range(start):
            self._frame(0.0)
          for i in range(240):
            if not self._frame(sign if i < 120 else -sign):
              blocked.append((sign, offset, start, i))
              break
    self.assertEqual(blocked, [], f"panda refused the controller at (sign, timer offset, ramp start, frame): {blocked[:6]}")

  def test_torque_recovers_after_a_refused_frame_at_every_blip_phase(self):
    """A refused frame zeroes the panda's torque memory, after which it refuses anything more than one rate step from
    zero. The controller must restart from zero on the next frame. It used to carry on and be refused until the next
    blip zeroed it (up to 0.9 s), or, below 90 deg where there is no blip, until the torque channel ended."""
    stuck = []
    for angle in (150.0, 30.0):
      for phase in range(2, 89, 3):  # where in the blip cycle (torque ramping from 0) the refusal lands, not on the blip
        for offset in (0, 13):
          self._setup(angle, offset)
          for _ in range(phase):
            self.assertTrue(self._frame(0.8))
          self.assertFalse(self._frame(0.8, refuse=True))
          refused = sum(not self._frame(0.8) for _ in range(100))
          if refused:
            stuck.append((angle, phase, offset, refused))
    self.assertEqual(stuck, [], f"frames refused after the first, at (angle, blip phase, timer offset, count): {stuck[:6]}")

  def test_toi_fault_release_is_never_refused(self):
    """Releasing the TOI request to clear a latched EPAS ToiFlt, and the ramp back from 0, must pass the panda wherever
    the release lands: mid-ramp, next to a blip, at any phase of its real-time interval, and with no blip at all."""
    blocked, late = [], []
    for angle in (150.0, 30.0):
      for sign in (1.0, -1.0):
        for offset in range(0, 26, 8):
          # where in the ramp / blip cycle the latch lands; the blip is on frames 89 and 90
          for start in (*range(0, 91, 10), 86, 87, 88, 89, 91):
            self._setup(angle, offset)
            low_at = None
            for i in range(200):
              # like the EPAS: latched from start until two frames after the first frame sent with the TOI request low
              fault = start <= i and (low_at is None or i < low_at + 2)
              if not self._frame(sign, toi_fault=fault):
                blocked.append((angle, sign, offset, start, i))
                break
              if low_at is None and start <= i and not self.erc.toi_act_cmd:
                low_at = i
            if low_at is None or low_at - start > TOI_CLEAR_LATCH_FRAMES - 1:
              late.append((angle, sign, offset, start, low_at))
    self.assertEqual(blocked, [], f"panda refused around a ToiFlt release at (angle, sign, timer offset, latch start, frame): {blocked[:6]}")
    self.assertEqual(late, [], f"latch not released on its 3rd frame at (angle, sign, timer offset, latch start, low frame): {late[:6]}")


class TestRivianLongRequestSafety(unittest.TestCase):
  """The real CarController's 0x160 frame through the panda safety model on the frame the panda first sees a pedal, or the
  ACM leaving cruise, while carControl still says openpilot controls the speed. A dropped frame is a hole in the request
  stream, and a run of them faults the ACM ("Cruise Fault: Restart the Car")."""

  SPEED = 20.0  # m/s

  def setUp(self):
    self.packer = CANPackerSafety("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, RivianSafetyFlags.LONG_CONTROL)
    self.safety.init_tests()
    self.cnt = 0

  def _rx(self, name, bus, values):
    self.assertTrue(self.safety.safety_rx_hook(self.packer.make_can_msg_safety(name, bus, values, fix_checksum=checksum)))

  def _rx_car(self, gas, brake, feature_status):
    self.cnt += 1
    self._rx("ESP_Status", 0, {"ESP_Vehicle_Speed": self.SPEED * 3.6, "ESP_Status_Counter": self.cnt % 15, "ESP_Vehicle_Speed_Q": 1})
    self._rx("VDM_PropStatus", 0, {"VDM_AcceleratorPedalPosition": 20 if gas else 0, "VDM_VehicleSpeed": self.SPEED * 3.6,
                                   "VDM_PropStatus_Counter": self.cnt % 15, "VDM_VehicleSpeedQ": 1})
    self._rx("iBESP2", 0, {"iBESP2_BrakePedalApplied": brake})
    self._rx("ACM_Status", 2, {"ACM_FeatureStatus": feature_status, "ACM_Unkown1": 1})

  def _controller_frame(self, gas, brake, cruise):
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CI = CarInterface(CP, structs.CarParamsSP())
    CS = CI.CS
    CS.acm_lka_hba_cmd = dict(CI.can_parsers[Bus.cam].vl["ACM_lkaHbaCmd"])
    CS.sccm_wheel_touch = dict(CI.can_parsers[Bus.pt].vl["SCCM_WheelTouch"])
    CS.vdm_adas_status = [dict(CI.can_parsers[Bus.pt].vl["VDM_AdasSts"])]
    CS.out = structs.CarState(gearShifter=structs.CarState.GearShifter.drive, vEgo=self.SPEED, vEgoRaw=self.SPEED,
                              gasPressed=gas, brakePressed=brake, cruiseState={"enabled": cruise})
    car_control = structs.CarControl(enabled=True, longActive=True, actuators={"accel": 1.0}).as_reader()
    _, can_sends = CI.CC.update(car_control, structs.CarControlSP(), CS, 0)
    addr, dat, bus = next(m for m in can_sends if m[0] == 0x160)
    return libsafety_py.make_CANPacket(addr, bus, dat)

  def test_request_is_not_dropped_when_a_pedal_appears(self):
    for gas, brake, cruise in ((True, False, True), (False, True, True), (False, False, False)):
      with self.subTest(gas=gas, brake=brake, cruise=cruise):
        self.safety.init_tests()
        for feature_status in (0, 1):  # the ACM enters cruise: the panda allows longitudinal
          self._rx_car(False, False, feature_status)
        self.assertTrue(self.safety.get_longitudinal_allowed())
        self.assertTrue(self.safety.safety_tx_hook(self._controller_frame(False, False, True)))
        # the panda sees the pedal or the ACM leaving cruise on the same CAN this frame's CarState came from
        self._rx_car(gas, brake, 1 if cruise else 0)
        self.assertFalse(self.safety.get_longitudinal_allowed())
        self.assertTrue(self.safety.safety_tx_hook(self._controller_frame(gas, brake, cruise)))


MADS_DISENGAGE = ALTERNATIVE_EXPERIENCE.ENABLE_MADS | ALTERNATIVE_EXPERIENCE.MADS_DISENGAGE_LATERAL_ON_BRAKE
MADS_PAUSE = ALTERNATIVE_EXPERIENCE.ENABLE_MADS | ALTERNATIVE_EXPERIENCE.MADS_PAUSE_LATERAL_ON_BRAKE
MADS_REMAIN_ACTIVE = ALTERNATIVE_EXPERIENCE.ENABLE_MADS
NO_MADS = ALTERNATIVE_EXPERIENCE.DEFAULT


class TestRivianBrakeLateralSafety(unittest.TestCase):
  """The real CarController's steering frames (0x110 angle, 0x120 cooperative torque) through the panda safety model,
  from the frame the panda first sees the brake, while carControl still asks for steering for a few frames. The panda
  ends lateral on the press unless MADS is set to "Remain active", and refuses every frame that still steers. The rack
  sees refused frames as a gap in the message counter and latches EPAS_Angle_Control_Cntr_Err ("LKAS Fault")."""

  FRAME_US = 10_000

  def setUp(self):
    self.packer = CANPackerSafety("rivian_primary_actuator")
    self.safety = libsafety_py.libsafety
    self.safety.set_safety_hooks(CarParams.SafetyModel.rivian, RivianSafetyFlags.LONG_CONTROL)

  def _start(self, alternative_experience, torque=False, speed=12.0, brake=False):
    self.safety.init_tests()
    self.safety.set_alternative_experience(alternative_experience)
    self.safety.mads_apply_alternative_experience(alternative_experience)
    self.safety.set_heartbeat_engaged_mads(True)
    self.t = 1_000_000
    self.cnt = 0
    self.speed = speed

    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CP.alternativeExperience = alternative_experience
    self.CI = CarInterface(CP, structs.CarParamsSP())
    CS = self.CI.CS
    CS.acm_lka_hba_cmd = dict(self.CI.can_parsers[Bus.cam].vl["ACM_lkaHbaCmd"])
    CS.sccm_wheel_touch = {**self.CI.can_parsers[Bus.pt].vl["SCCM_WheelTouch"], "SCCM_WheelTouch_Calibration": 100}
    CS.vdm_adas_status = [dict(self.CI.can_parsers[Bus.pt].vl["VDM_AdasSts"])]
    self.CI.CC.erc.force_torque = torque
    self.eac_status = 1  # the rack is ready for angle control, and steers on angle once asked to

    # the car (enough to fill the panda's speed and angle samples), then the ACM entering cruise: the panda allows
    # control, and with MADS lateral too
    self.safety.set_desired_angle_last(0)
    for feature_status in (0, 0, 0, 0, 0, 0, 1):
      self._rx_car(brake, feature_status)

  def _rx(self, name, bus, values):
    self.assertTrue(self.safety.safety_rx_hook(self.packer.make_can_msg_safety(name, bus, values, fix_checksum=checksum)))

  def _rx_car(self, brake, feature_status):
    self.cnt += 1
    kph = self.speed * 3.6
    self._rx("ESP_Status", 0, {"ESP_Vehicle_Speed": kph, "ESP_Status_Counter": self.cnt % 15, "ESP_Vehicle_Speed_Q": 1})
    self._rx("VDM_PropStatus", 0, {"VDM_AcceleratorPedalPosition": 0, "VDM_VehicleSpeed": kph,
                                   "VDM_PropStatus_Counter": self.cnt % 15, "VDM_VehicleSpeedQ": 1})
    self._rx("EPAS_AdasStatus", 0, {"EPAS_InternalSas": 0.0, "EPAS_EacStatus": self.eac_status})
    self._rx("EPAS_SystemStatus", 0, {"EPAS_TorsionBarTorque": 0.0})
    self._rx("iBESP2", 0, {"iBESP2_BrakePedalApplied": int(brake)})
    self._rx("ACM_Status", 2, {"ACM_FeatureStatus": feature_status, "ACM_Unkown1": 1})

  def _frame(self, lat_active, brake, cruise=True):
    """One 10 ms frame: the panda reads the car, then the controller's frame from the same CAN goes out.
    Returns whether 0x110 and 0x120 were let through, and whether the frame was steering."""
    self.t += self.FRAME_US
    self.safety.set_timer(self.t)
    self._rx_car(brake, 1 if cruise else 0)
    CS = self.CI.CS
    CS.eac_status = self.eac_status
    CS.out = structs.CarState(gearShifter=structs.CarState.GearShifter.drive, vEgo=self.speed, vEgoRaw=self.speed,
                              brakePressed=brake, standstill=self.speed == 0, cruiseState={"enabled": cruise})
    car_control = structs.CarControl(enabled=lat_active, latActive=lat_active, longActive=lat_active and cruise and not brake,
                                     actuators={"steeringAngleDeg": 2.0, "torque": 0.5}).as_reader()
    _, can_sends = self.CI.CC.update(car_control, structs.CarControlSP(), CS, 0)
    self.can_sends = can_sends
    ok, steering = True, False
    for addr, dat, bus in can_sends:
      if addr in (0x110, 0x120):
        ok &= self.safety.safety_tx_hook(libsafety_py.make_CANPacket(addr, bus, dat))
        steering |= bool((dat[1] >> 4) & 1) if addr == 0x110 else ((dat[2] << 3) | (dat[3] >> 5)) != 1024
    if self.CI.CC.erc.angle_active:
      self.eac_status = 2
    return ok, steering

  def _engage(self, frames=40, brake=False, cruise=True):
    # lateral engaged long enough for the torque ramp to be well under way; every frame is let through
    for _ in range(frames):
      self.assertEqual(self._frame(True, brake, cruise), (True, True))

  def _reengage(self, brake=False):
    # carControl has dropped lateral; the ACM leaves cruise and enters it again, which gives the panda control back
    for _ in range(3):
      self.assertEqual(self._frame(False, brake, cruise=False), (True, False))
    self.eac_status = 1  # the rack is back to ready once angle control has ended
    self._engage(brake=brake)

  def test_brake_press_ends_steering_on_the_same_frame(self):
    for alt_exp in (MADS_DISENGAGE, MADS_PAUSE, NO_MADS):
      for torque in (False, True):
        for speed in (12.0, 0.0):
          with self.subTest(alternative_experience=alt_exp, torque=torque, speed=speed):
            self._start(alt_exp, torque, speed)
            self._engage()
            # the panda reads the press with this frame's CAN; carControl still asks for steering for three more frames
            for lag in range(4):
              self.assertEqual(self._frame(True, True), (True, False), f"frame {lag} of the press")
            self.assertEqual(self._frame(False, True), (True, False))

  def test_remain_active_keeps_steering_through_the_brake(self):
    for torque in (False, True):
      for speed in (12.0, 0.0):
        for cruise in (True, False):  # with the ACM in cruise, and lateral only (MADS engaged, cruise off)
          with self.subTest(torque=torque, speed=speed, cruise=cruise):
            self._start(MADS_REMAIN_ACTIVE, torque, speed)
            self._engage(cruise=cruise)
            self.assertTrue(self.safety.get_controls_allowed_lateral())
            for _ in range(10):
              self.assertEqual(self._frame(True, True, cruise), (True, True))
            self.assertEqual(self._frame(True, False, cruise), (True, True))

  def test_lateral_only_ends_on_the_brake(self):
    # MADS engaged with the ACM out of cruise: lateral rides on controls_allowed_lateral alone, which the press ends
    for alt_exp in (MADS_DISENGAGE, MADS_PAUSE):
      with self.subTest(alternative_experience=alt_exp):
        self._start(alt_exp)
        self._engage(cruise=False)
        self.assertFalse(self.safety.get_controls_allowed())
        self.assertTrue(self.safety.get_controls_allowed_lateral())
        for _ in range(3):
          self.assertEqual(self._frame(True, True, cruise=False), (True, False))

  def test_short_press_does_not_bring_steering_back_early(self):
    # released before carControl has caught up: still refused by the panda, unless MADS pauses, which gives it back
    for alt_exp in (MADS_DISENGAGE, NO_MADS, MADS_PAUSE):
      with self.subTest(alternative_experience=alt_exp):
        self._start(alt_exp)
        self._engage()
        self.assertEqual(self._frame(True, True), (True, False))
        self.assertEqual(self._frame(True, True), (True, False))
        back = alt_exp == MADS_PAUSE
        self.assertEqual(self._frame(True, False), (True, back))
        self.assertEqual(self._frame(True, False), (True, back))
        if not back:
          self.assertEqual(self._frame(False, False), (True, False))

  def test_brake_held_through_engagement_is_not_a_press(self):
    # at a standstill with the brake already held when cruise engages, the panda keeps control: so does the controller
    for alt_exp in (NO_MADS, MADS_DISENGAGE):
      with self.subTest(alternative_experience=alt_exp):
        self._start(alt_exp, speed=0.0, brake=True)
        self.assertEqual(self._frame(False, True), (True, False))
        self._engage(brake=True)
        self.assertTrue(self.safety.get_controls_allowed())

  def test_brake_held_ends_steering_once_moving(self):
    # held from a standstill through the engagement, then the truck creeps: without MADS the panda ends control on a held
    # brake while moving, and selfdrived disengages on it either way
    for alt_exp in (NO_MADS, MADS_DISENGAGE):
      for torque in (False, True):
        with self.subTest(alternative_experience=alt_exp, torque=torque):
          self._start(alt_exp, torque, speed=0.0, brake=True)
          self.assertEqual(self._frame(False, True), (True, False))
          self._engage(brake=True)
          self.speed = 0.5
          for lag in range(3):
            self.assertEqual(self._frame(True, True), (True, False), f"frame {lag} moving")
          self.assertEqual(self._frame(False, True), (True, False))

  def test_steering_returns_after_reengaging_and_the_next_press_ends_it(self):
    for alt_exp in (MADS_DISENGAGE, NO_MADS):
      for torque in (False, True):
        for speed in (12.0, 0.0):
          with self.subTest(alternative_experience=alt_exp, torque=torque, speed=speed):
            self._start(alt_exp, torque, speed)
            self._engage()
            self.assertEqual(self._frame(True, True), (True, False))
            self.assertEqual(self._frame(False, True), (True, False))
            self.assertEqual(self._frame(False, False), (True, False))
            self._reengage()
            self.assertEqual(self._frame(True, True), (True, False), "second press")
            self.assertEqual(self._frame(False, True), (True, False))

  def test_resume_at_a_stop_with_the_brake_still_held(self):
    # pressed at a standstill and held; resuming with the brake still held steers again, as the panda allows
    for alt_exp in (MADS_DISENGAGE, NO_MADS):
      with self.subTest(alternative_experience=alt_exp):
        self._start(alt_exp, speed=0.0)
        self._engage()
        self.assertEqual(self._frame(True, True), (True, False))
        self.assertEqual(self._frame(False, True), (True, False))
        self._reengage(brake=True)

  def test_acm_feature_status_follows_on_the_press_frame(self):
    self._start(MADS_DISENGAGE)
    self._engage()
    self.assertEqual(self._frame(True, True), (True, False))
    parser = CANParser("rivian_primary_actuator", [("ACM_Status", 0)], 0)
    parser.update([(self.t * 1000, [m for m in self.can_sends if m[0] == 0x100])])
    self.assertEqual(parser.vl["ACM_Status"]["ACM_FeatureStatus"], 0)


class TestRivianIgnition(unittest.TestCase):
  TX_MSGS: list = []

  def setUp(self):
    self.safety = libsafety_py.libsafety
    self.safety.init_tests()
    self.packer = CANPackerSafety("rivian_primary_actuator")

  def _msg(self, counter, mode):
    return self.packer.make_can_msg_safety("VDM_OutputSignals", 0,
                                           {"VDM_OutputSigs_Counter": counter,
                                            "VDM_EpasPowerMode": mode})

  # VDM_EpasPowerMode_Drive_On=1
  def test_ignition_on(self):
    for i in range(15):
      self.safety.init_tests()
      self.safety.ignition_can_hook(self._msg(i, 1))
      self.assertFalse(self.safety.get_ignition_can())
      self.safety.ignition_can_hook(self._msg((i + 1) % 15, 1))
      self.assertTrue(self.safety.get_ignition_can())

  def test_ignition_off(self):
    self.safety.ignition_can_hook(self._msg(0, 1))
    self.safety.ignition_can_hook(self._msg(1, 1))
    self.assertTrue(self.safety.get_ignition_can())
    self.safety.ignition_can_hook(self._msg(2, 0))
    self.safety.ignition_can_hook(self._msg(3, 0))
    self.assertFalse(self.safety.get_ignition_can())


if __name__ == "__main__":
  unittest.main()
