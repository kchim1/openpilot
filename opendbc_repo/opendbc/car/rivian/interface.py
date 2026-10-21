from opendbc.car import get_safety_config, scale_rot_inertia, scale_tire_stiffness, structs
from opendbc.car.interfaces import CarInterfaceBase
from opendbc.car.rivian.carcontroller import CarController
from opendbc.car.rivian.carstate import CarState
from opendbc.car.rivian.radar_interface import RadarInterface
from opendbc.car.rivian.values import ModelLine, RivianFlags, RivianSafetyFlags, R1S_WHEELBASE
from opendbc.car.vin import Vin
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP


class CarInterface(CarInterfaceBase):
  CarState = CarState
  CarController = CarController
  RadarInterface = RadarInterface

  @staticmethod
  def _get_params(ret: structs.CarParams, candidate, fingerprint, car_fw, alpha_long, is_release, docs) -> structs.CarParams:
    ret.brand = "rivian"

    ret.safetyConfigs = [get_safety_config(structs.CarParams.SafetyModel.rivian)]

    # GEN2 (2025+) doesn't have SCCM_WheelTouch on the bus
    if 0x321 not in fingerprint[0]:
      ret.flags |= RivianFlags.GEN2.value

    # no angle upgrade installed
    if 0x1310 not in fingerprint[1]:
      ret.dashcamOnly = True

    ret.steerActuatorDelay = 0.1
    ret.steerAtStandstill = True
    ret.steerLimitTimer = 0.8  # 0.4 flagged normal low-speed turn-in lag as steerSaturated
    CarInterfaceBase.configure_torque_tune(candidate, ret.lateralTuning)

    # torque is the primary channel, the angle command comes in via actuators.steeringAngleDeg
    ret.steerControlType = structs.CarParams.SteerControlType.torque
    ret.radarUnavailable = True

    # TODO: pending finding/handling missing set speed
    ret.alphaLongitudinalAvailable = False
    if alpha_long:
      ret.openpilotLongitudinalControl = True
      ret.safetyConfigs[0].safetyParam |= RivianSafetyFlags.LONG_CONTROL.value

    ret.longitudinalActuatorDelay = 0.25
    ret.stopAccel = -0.2
    # no integral term, like upstream: the truck tracks the accel request itself. Braking from speed it overshot the
    # request by ~0.3 m/s^2, an integrator learned to ask for that much less, and the stale offset then roughly halved
    # the braking over the last few mph to a red light, so the truck crept ~2.5 m past where the model meant to stop

    return ret

  @staticmethod
  def _get_params_sp(stock_cp: structs.CarParams, ret: structs.CarParamsSP, candidate, fingerprint: dict[int, dict[int, int]],
                     car_fw: list[structs.CarParams.CarFw], alpha_long: bool, is_release_sp: bool, docs: bool) -> structs.CarParamsSP:
    # The body from the VIN (set by get_car before this runs): an R1S has the R1T's steering rack on a shorter wheelbase.
    # Across 62 public trucks the steering angle per curvature was 47.9 rad*m on R1Ss and 53.9 on R1Ts (1.126x, the
    # wheelbase ratio 1.122). An unknown VIN keeps the R1T's. paramsd keeps what it learned (steer ratio x wheelbase)
    # across a wheelbase change, so a truck that has learned steers the same either way.
    if Vin(stock_cp.carVin).vds[:1] == ModelLine.R1S:
      stock_cp.centerToFront *= R1S_WHEELBASE / stock_cp.wheelbase
      stock_cp.wheelbase = R1S_WHEELBASE
      stock_cp.rotationalInertia = scale_rot_inertia(stock_cp.mass, stock_cp.wheelbase)
      stock_cp.tireStiffnessFront, stock_cp.tireStiffnessRear = scale_tire_stiffness(stock_cp.mass, stock_cp.wheelbase,
                                                                                     stock_cp.centerToFront,
                                                                                     stock_cp.tireStiffnessFactor)

    if 0x131a in fingerprint[1]:
      ret.flags |= RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE.value
      stock_cp.radarUnavailable = False
      stock_cp.enableBsm = True
      stock_cp.alphaLongitudinalAvailable = True

    if alpha_long and stock_cp.alphaLongitudinalAvailable:
      stock_cp.openpilotLongitudinalControl = True
      stock_cp.safetyConfigs[0].safetyParam |= RivianSafetyFlags.LONG_CONTROL.value

    return ret
