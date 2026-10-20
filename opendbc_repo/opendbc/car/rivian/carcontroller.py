import numpy as np
from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.common.conversions import Conversions as CV
from opendbc.car.interfaces import CarControllerBase
from opendbc.car.rivian.ext_controller import ExternalController, get_safety_CP  # noqa: F401
from opendbc.car.rivian.riviancan import (create_angle_steering, create_lka_steering, create_longitudinal, create_wheel_touch,
                                          create_adas_status, create_acm_status)
from opendbc.car.rivian.values import CarControllerParams, RivianFlags
from opendbc.safety import ALTERNATIVE_EXPERIENCE

from opendbc.sunnypilot.car.rivian.mads import MadsCarController


GearShifter = structs.CarState.GearShifter

# MADS steering modes in which a brake press ends lateral at the panda (opendbc/safety/sunnypilot/mads.h)
MADS_LATERAL_OFF_ON_BRAKE = ALTERNATIVE_EXPERIENCE.MADS_DISENGAGE_LATERAL_ON_BRAKE | ALTERNATIVE_EXPERIENCE.MADS_PAUSE_LATERAL_ON_BRAKE


def panda_drops_lateral_on_brake(alternative_experience: int) -> bool:
  """Whether a brake press ends lateral at the panda. Without MADS steering is allowed by controls_allowed, which a press
  clears (opendbc/safety/safety.h). With MADS it is also allowed by controls_allowed_lateral, which a press clears only
  when MADS is set to disengage or pause steering on the brake, not with "Remain active"."""
  if not alternative_experience & ALTERNATIVE_EXPERIENCE.ENABLE_MADS:
    return True
  return bool(alternative_experience & MADS_LATERAL_OFF_ON_BRAKE)

# The stock ACM answers an ACC stalk press within ~30 ms when it takes it. If it hasn't after this long, it declined.
ACM_ENGAGE_TIMEOUT_FRAMES = 50  # 0.5 s
# The stock ACM has been seen to decline ACC below this speed, even with a car ahead
ACM_MIN_ENGAGE_SPEED = 20 * CV.MPH_TO_MS

# Cruising at highway speed the truck delivers ~0.08 m/s^2 more than requested, the same on grades from -2% to +4% (routes
# 00000008 and 0000000e). With no long integrator to cancel it, leads would be followed ~10-15% closer, so ask for that
# much less there. Faded in from 15 to 22 m/s, and out from -0.5 to -1.0 m/s^2 so braking is untouched.
CRUISE_ACCEL_BIAS = 0.08


def cruise_accel_bias(v_ego: float, accel: float) -> float:
  return CRUISE_ACCEL_BIAS * float(np.interp(v_ego, [15., 22.], [0., 1.])) * float(np.interp(accel, [-1.0, -0.5], [0., 1.]))


class CarController(CarControllerBase, MadsCarController):
  def __init__(self, dbc_names, CP, CP_SP):
    CarControllerBase.__init__(self, dbc_names, CP, CP_SP)
    MadsCarController.__init__(self)
    self.apply_torque_last = 0
    self.packer = CANPacker(dbc_names[Bus.pt])

    self.cancel_frames = 0
    self.long_cmd_counter = 0
    # set by card: False while selfdrived has a NO_ENTRY event (openpilot would refuse to engage)
    self.openpilot_engageable = True
    # set here when a driver ACC engage press was hidden from the ACM; card reads and clears it
    self.engage_request_blocked = False
    self.engage_request_prev = False
    # set here when the ACM declined a driver ACC engage press that openpilot passed on; card reads and clears it.
    # 'notReady': the ACM was not ready (ACM_FaultSupervisorState != 0), 'lowSpeed': declined below ACM_MIN_ENGAGE_SPEED
    self.engage_request_refused = None
    self.engage_press_frames = 0
    self.engage_press_not_ready = False
    self.engage_press_low_speed = False
    self.erc = ExternalController()
    # lateral held off from a brake press the panda ends lateral on until carControl catches up, see update_brake_lateral
    self.brake_lat_off = False
    self.brake_pressed_prev = False

  def update_brake_lateral(self, CC, CS) -> None:
    """Stop steering on the frame a brake press shows up in CarState, when the panda ends lateral on it.

    The panda reads the pedal off the bus and ends lateral at once, while carControl asks for steering until selfdrived
    and controlsd have seen the press (one frame on this truck). Every angle (0x110) or cooperative torque (0x120) frame
    that still steers in between is refused, and the rack sees refused frames as a gap in the message counter: four in a
    row on an AdventurePilot truck latched EPAS_Angle_Control_Cntr_Err ("LKAS Fault: Restart the car to engage").

    On a press, or the brake held while moving, as at the panda (safety.h) and in selfdrived: a brake already held at a
    standstill when lateral engaged leaves it on until the truck moves. Steering then stays off until carControl drops
    lateral too, so a press shorter than that lag cannot turn it back on while the panda still refuses it, except with
    MADS set to pause, where the panda gives lateral back on release.
    """
    alt_exp = self.CP.alternativeExperience
    brake = CS.out.brakePressed
    press = brake and (not self.brake_pressed_prev or not CS.out.standstill)
    if press and self.mads.lat_active and panda_drops_lateral_on_brake(alt_exp):
      self.brake_lat_off = True
    self.brake_pressed_prev = brake

    paused = bool(alt_exp & ALTERNATIVE_EXPERIENCE.ENABLE_MADS and alt_exp & ALTERNATIVE_EXPERIENCE.MADS_PAUSE_LATERAL_ON_BRAKE)
    if not CC.latActive or (paused and not brake):
      self.brake_lat_off = False
    if self.brake_lat_off:
      self.mads = self.mads._replace(lat_active=False)

  def update(self, CC, CC_SP, CS, now_nanos):
    MadsCarController.update(self, CC, CC_SP, CS)
    self.update_brake_lateral(CC, CS)
    actuators = CC.actuators
    can_sends = []

    apply_torque = 0
    steer_max = round(float(np.interp(CS.out.vEgoRaw, CarControllerParams.STEER_MAX_LOOKUP[0],
                                      CarControllerParams.STEER_MAX_LOOKUP[1])))

    self.erc.update(CS, self.mads.lat_active, actuators)
    apply_torque = self.erc.apply_torque_last

    # send steering command, torque is 0 and toi_act_cmd low during a blip
    self.apply_torque_last = apply_torque
    can_sends.append(create_lka_steering(self.packer, self.frame, CS.acm_lka_hba_cmd, apply_torque, CC.enabled, self.erc.toi_act_cmd, self.mads))

    can_sends.append(create_angle_steering(self.packer, self.frame, self.erc.apply_angle_last, self.erc.angle_active))
    feature_status = (1 if self.erc.torque_active else 2) if self.mads.lat_active else 0
    can_sends.append(create_acm_status(self.packer, self.frame, feature_status))

    if self.frame % 5 == 0 and not (self.CP.flags & RivianFlags.GEN2):
      can_sends.append(create_wheel_touch(self.packer, CS.sccm_wheel_touch, self.mads.lat_active))

    # Stock ACC cancel: openpilot declined (noEntry) or dropped (soft/immediate disable) an engagement while
    # the ACM is in ACC. Spoof a driver cancel press to the ACM so it exits ACC cleanly, instead of sitting in
    # ACC with no enabled long controller behind it and timing out into an ACC fault.
    # The ACM needs to see "available" before it will accept "unavailable"; the VDM takes a few frames to ack.
    interface_status = None
    if CC.cruiseControl.cancel:
      interface_status = 1 if self.cancel_frames < 5 else 0
      self.cancel_frames += 1
    else:
      self.cancel_frames = 0

    # Longitudinal control
    if self.CP.openpilotLongitudinalControl:
      # CC.enabled lags the ACM clearing its feature status by the selfdrived/controlsd round trip
      long_enabled = CC.enabled and CS.out.cruiseState.enabled
      # The panda passes no long request but an exact 0 unless the ACM's feature status is 1 (cruise enabled), the gas is
      # not pressed and the brake has not been pressed (opendbc/safety/modes/rivian.h). It reads them straight off the bus,
      # a frame or two before carControl reacts, and a run of blocked requests faults the ACM ("Cruise Fault: Restart the
      # Car", routes 00000015 and 00000016). So send exactly 0 whenever this frame's CarState shows it would block the
      # request, the cruise bias included.
      long_allowed = CC.longActive and CS.out.cruiseState.enabled and not CS.out.gasPressed and not CS.out.brakePressed
      accel = 0.0
      if long_allowed:
        accel = actuators.accel - cruise_accel_bias(CS.out.vEgo, actuators.accel)
        accel = float(np.clip(accel, CarControllerParams.ACCEL_MIN, CarControllerParams.ACCEL_MAX))
      # a rejected frame never reached the bus, resend its counter if nothing newer went out
      resend = CS.long_cmd_rejected_updated and CS.long_cmd_rejected_counter == self.long_cmd_counter % 15
      if not resend:
        self.long_cmd_counter += 1
      can_sends.append(create_longitudinal(self.packer, self.long_cmd_counter, accel, long_enabled))

      # If openpilot would refuse to engage (NO_ENTRY), hide the driver's ACC engage request from the ACM so it never
      # enters ACC. Once the ACM is in ACC with nothing accepting its long request, it latches a fault that only
      # clears when the car sleeps; a spoofed cancel is not enough to prevent it. Cancel requests still pass through.
      block_engage = not self.openpilot_engageable and not CS.out.cruiseState.enabled

      # flag the rising edge of a blocked engage press so openpilot can tell the driver why nothing happened
      if CS.vdm_adas_status:
        engage_request = any(msg["VDM_UserAdasRequest"] not in (0, 1) for msg in CS.vdm_adas_status)
        # VDM_UserAdasRequest: 0=IDLE, 1=UP_1, 2=UP_2, 3=DOWN_1, 4=DOWN_2. The ACC stalk is also the gear selector,
        # so only a stalk-down while already in Drive is an ACC press. Stalk-up, or stalk-down out of R/N/P, is a
        # gear change and must not be reported as an engage attempt.
        acc_on_request = any(msg["VDM_UserAdasRequest"] in (3, 4) for msg in CS.vdm_adas_status)
        in_drive = CS.out.gearShifter == GearShifter.drive
        if block_engage and acc_on_request and in_drive and not self.engage_request_prev:
          self.engage_request_blocked = True

        # an ACC-on press (stalk down in Drive) that openpilot passes to the ACM: watch whether the ACM takes it
        if acc_on_request and in_drive and not self.engage_request_prev and not block_engage and not CS.out.cruiseState.enabled:
          self.engage_press_frames = 1
          self.engage_press_not_ready = False
          self.engage_press_low_speed = CS.out.vEgo < ACM_MIN_ENGAGE_SPEED
        self.engage_request_prev = engage_request

      # the ACM declined the press: tell the driver the likely reason, the truck only says ACC is not available
      if self.engage_press_frames > 0:
        # the ACM took it, or the driver shifted out of Drive: nothing to report
        if CS.out.cruiseState.enabled or CS.out.gearShifter != GearShifter.drive:
          self.engage_press_frames = 0
        else:
          self.engage_press_not_ready |= CS.acm_fault_supervisor_state != 0
          self.engage_press_frames += 1
          if self.engage_press_frames > ACM_ENGAGE_TIMEOUT_FRAMES:
            self.engage_press_frames = 0
            if self.engage_press_not_ready:
              self.engage_request_refused = 'notReady'
            elif self.engage_press_low_speed:
              self.engage_request_refused = 'lowSpeed'

      # keep the stock ACM from winding up its unactuated request
      for msg in CS.vdm_adas_status:
        can_sends.append(create_adas_status(self.packer, msg, interface_status, 0 if CC.enabled else None, block_engage))
    else:
      for msg in CS.vdm_adas_status:
        can_sends.append(create_adas_status(self.packer, msg, interface_status))

    new_actuators = actuators.as_builder()
    if self.CP.openpilotLongitudinalControl:
      new_actuators.accel = accel  # what went on the bus, cruise bias included
    new_actuators.torque = apply_torque / steer_max
    new_actuators.torqueOutputCan = apply_torque
    new_actuators.steeringAngleDeg = self.erc.apply_angle_last

    self.frame += 1
    return new_actuators, can_sends
