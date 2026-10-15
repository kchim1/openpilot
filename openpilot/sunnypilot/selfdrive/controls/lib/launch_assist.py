"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
from enum import IntEnum

from opendbc.car.structs import car
from openpilot.common.realtime import DT_CTRL
from openpilot.common.swaglog import cloudlog

LongCtrlState = car.CarControl.Actuators.LongControlState

# Rivian Launch Assist: after the planner releases a stop, its request rises slowly from ~+0.1 m/s^2 while the
# VDM does not move the wheels below ~0.2-0.35 m/s^2 at standstill, which costs ~0.6 s before the truck starts
# following. When a stopped radar lead measurably pulls away, hold a short floor above that dead zone.
ARM_STANDSTILL_T = 1.0      # s stopped behind a radar lead before a launch may be assisted
LAUNCH_ACCEL = 0.5          # m/s^2 floor while firing
LAUNCH_RAMP_RATE = 5.0      # m/s^3 floor rise from the normal output to LAUNCH_ACCEL (~0.1 s), not a one-frame step
LAUNCH_MAX_T = 0.7          # s at most from the fire to the fade
LAUNCH_V_END = 0.15         # m/s vEgoRaw: moving, start the fade (ESP speed reads 0 below ~0.125)
FADE_RATE = 0.75            # m/s^3 floor fade back to the normal output, rather than dropping at once

# Departing-lead gate. Standstill radar noise measured std <= 0.05 m/s on vLead, |aLeadK| <= 0.12; vLeadK of a stationary
# lead stays <= 0.02 in real Rivian logs. A higher threshold delays real launches (real logs: lead start to gate median
# 0.45 s at 0.15, 0.60 s at 0.2)
LEAD_DEPART_V_K = 0.15      # primary: radarState.leadOne.vLeadK above this
LEAD_DEPART_ALT_V = 0.3     # alternative: vLead above this and aLeadK above LEAD_DEPART_ALT_A
LEAD_DEPART_ALT_A = 0.1
LEAD_DEPART_GATE_ALT = False

PARAM_READ_FRAMES = 100     # controlsd runs at 100 Hz: re-read the toggle every 1 s
ARM_FRAMES = int(round(ARM_STANDSTILL_T / DT_CTRL))
LAUNCH_MAX_FRAMES = int(round(LAUNCH_MAX_T / DT_CTRL))


class LaunchAssistState(IntEnum):
  idle = 0
  armed = 1
  firing = 2
  fading = 3


def is_radar_lead(lead) -> bool:
  return bool(lead.present and lead.radar)


def lead_departing(lead) -> bool:
  if LEAD_DEPART_GATE_ALT:
    return lead.vLead > LEAD_DEPART_ALT_V and lead.aLeadK > LEAD_DEPART_ALT_A
  return lead.vLeadK > LEAD_DEPART_V_K


class LaunchAssist:
  def __init__(self, params):
    self.params = params
    self.enabled = False
    self._frame = 0
    self._error_logged = False

    self.state = LaunchAssistState.idle
    self.standstill_frames = 0  # stopped behind the same radar lead with a launch qualifying
    self.lead_id = -1
    self.spent = False          # fired during this standstill
    self.fire_frames = 0
    self.floor = 0.

  @staticmethod
  def available(CP) -> bool:
    return CP.brand == 'rivian' and CP.openpilotLongitudinalControl

  def read_param(self) -> bool:
    try:
      return self.params.get_bool("RivianLaunchAssist")
    except Exception:
      # params library built without the key, or a failed read: off rather than take controlsd down
      if not self._error_logged:
        cloudlog.exception("RivianLaunchAssist read failed, treating it as off")
        self._error_logged = True
      return False

  def update_param(self) -> None:
    if self._frame % PARAM_READ_FRAMES == 0:
      self.enabled = self.read_param()
    self._frame += 1

  def reset(self) -> None:
    self.state = LaunchAssistState.idle
    self.standstill_frames = 0
    self.floor = 0.

  def update(self, long_active, CS, lead, long_control_state, a_target, should_stop, accel, accel_limits):
    """accel: LongControl's output this frame. Returns it, or a higher floor while a launch is assisted."""
    self.update_param()

    radar_lead = is_radar_lead(lead)
    # the planner owns the launch: never act in stopping/off, with shouldStop, or when it asks to slow down
    released = long_control_state == LongCtrlState.pid and not should_stop and a_target >= 0.
    usable = self.enabled and long_active and not CS.brakePressed and radar_lead

    # a new radar track restarts the second (its first vLeadK is unfiltered)
    same_lead = lead.radarTrackId == self.lead_id
    self.lead_id = lead.radarTrackId
    if CS.standstill:
      self.standstill_frames = (self.standstill_frames + 1 if same_lead else 1) if usable else 0
    else:
      self.standstill_frames = 0
      self.spent = False

    if self.state in (LaunchAssistState.firing, LaunchAssistState.fading):
      if not (usable and released):
        cloudlog.event("launch assist cancel", enabled=self.enabled, long_active=bool(long_active), brake=bool(CS.brakePressed),
                       radar_lead=radar_lead, long_control_state=int(long_control_state), should_stop=bool(should_stop),
                       a_target=float(a_target), fire_frames=self.fire_frames)
        self.reset()
      elif CS.vEgo >= lead.vLeadK:
        # already as fast as the lead: more would only close the gap (lead inching or crawling, then stopping)
        self.reset()

    elif self.state == LaunchAssistState.armed:
      if self.standstill_frames < ARM_FRAMES:
        # moved, lead lost or changed, not engaged, braking
        self.reset()
      elif released and lead_departing(lead):
        cloudlog.event("launch assist fire", v_lead_k=float(lead.vLeadK), v_lead=float(lead.vLead), a_lead_k=float(lead.aLeadK),
                       d_rel=float(lead.dRel), accel=float(accel))
        self.state = LaunchAssistState.firing
        self.spent = True
        self.fire_frames = 0
        self.floor = accel

    elif self.state == LaunchAssistState.idle:
      # arms in stopping or, after a long-gap stop the planner already released, in pid
      if not self.spent and self.standstill_frames >= ARM_FRAMES:
        self.state = LaunchAssistState.armed

    if self.state == LaunchAssistState.firing:
      self.floor = min(self.floor + LAUNCH_RAMP_RATE * DT_CTRL, LAUNCH_ACCEL)
      self.fire_frames += 1
      if self.fire_frames >= LAUNCH_MAX_FRAMES or CS.vEgoRaw > LAUNCH_V_END:
        self.state = LaunchAssistState.fading
    elif self.state == LaunchAssistState.fading:
      self.floor -= FADE_RATE * DT_CTRL

    if self.state not in (LaunchAssistState.firing, LaunchAssistState.fading):
      return accel

    floor = min(max(self.floor, accel_limits[0]), accel_limits[1])
    if self.state == LaunchAssistState.fading and floor <= accel:
      # back at the normal output: done until the next standstill
      self.reset()
      return accel
    return max(accel, floor)
