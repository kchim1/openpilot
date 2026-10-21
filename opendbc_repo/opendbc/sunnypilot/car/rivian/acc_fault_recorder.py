"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import time
from collections import deque

from opendbc.car import Bus, structs

GearShifter = structs.CarState.GearShifter

# Rows kept before and after the trigger, at the 100 Hz CarState rate. The run of refused requests behind the cruise
# fault in route 00000015 spanned 90 ms, so 0.5 s before and 0.25 s after is generous. Bounded because the dump goes out
# as one log line.
PRE_FRAMES = 50
POST_FRAMES = 25

# at most this many dumps per drive, so a fault that keeps re-arming cannot fill the log
MAX_DUMPS = 4

# ACM_FaultSupervisorState: 0 once the ACM is ready to take cruise. It is 2 or 3 in Park while the truck wakes up and
# shuts down, but in Drive it is 0 on every healthy drive logged here. 3 held in Drive is the ACM having shut itself
# down, which leaves the truck without cruise until it is restarted (route 00000015 from 3:51 to the end of the drive).
ACM_SUPERVISOR_SHUT_DOWN = 3
LATCH_FRAMES = 100  # held this long (1 s) in Drive to count as shut down rather than a blip

# The panda's echo of a refused 0x160 frame comes back on bus 192 one to three frames after it was sent (route 00000015).
# It is matched to its row by the frame's counter, looking back this many rows.
REFUSAL_LOOKBACK = 8

# one row per CarState frame; accel, en, allowed and ctr (the 0x160 counter) are filled in by the CarController
# (record_command), refused once the panda's echo of that frame comes back
COLUMNS = ("dt_ms", "accel", "en", "allowed", "ctr", "refused", "vdm_fault", "vdm_iface", "vdm_mode", "vdm_user",
           "vdm_accel_lim", "acm_feat", "acm_fault", "acm_sup", "v_ego", "brake", "gas")
CTR, REFUSED = COLUMNS.index("ctr"), COLUMNS.index("refused")


class AccFaultRecorder:
  """The last half second of the cruise request and the truck's answer, written to the log when cruise faults.

  A cruise fault ("Cruise Fault: Restart the Car") has only been explainable from rlogs, by decoding the CAN traffic of
  a whole drive by hand. This keeps a short ring buffer of what openpilot asked for (0x160), whether the panda refused
  it, and what the VDM and the ACM answered, and dumps it through cloudlog with error=True when the truck reports a
  cruise fault in Drive. Error-level log lines go out as errorLogMessage, which is kept whole in the qlog, so the
  record survives on the small log that is always uploaded. It also notes when the ACM shuts itself down in Drive and
  stays down, and when it comes back.

  Read only: nothing here changes what is sent to the truck, and no alert is shown. A healthy drive emits nothing.
  Any error in here disables the recorder for the rest of the drive instead of reaching the car's control loop.
  """

  def __init__(self, clock=time.monotonic):
    self.clock = clock
    self.buf: deque = deque(maxlen=PRE_FRAMES + 1 + POST_FRAMES)
    self.dumps = 0
    self.suppressed = 0
    self.notes = 0
    self.unmatched_refusals = 0  # refusal echoes with no recent row of that counter
    self.disabled = False
    self._reason: str | None = None
    self._countdown = 0
    self._last_t: float | None = None
    self._prev_fault = False
    self._latch_frames = 0   # consecutive frames with the ACM shut down, counted in Drive until it is reported
    self._latched = False
    # lazy openpilot import: opendbc must stay importable without openpilot (the safety tests, other users of opendbc)
    try:
      from openpilot.common.swaglog import cloudlog
      self._log = cloudlog
    except Exception:
      self._log = None

  def record_command(self, accel: float, enabled: bool, allowed: bool, counter: int | None = None) -> None:
    """What the CarController put in this frame's 0x160: the accel request, ACM_longInterfaceEnable, whether the
    request was allowed to be non-zero, and the frame's counter. Called after update() in the same card loop, so it
    completes the newest row."""
    if self.disabled or not self.buf:
      return
    try:
      self.buf[-1][1:5] = [round(float(accel), 3), int(enabled), int(allowed), None if counter is None else int(counter)]
    except Exception:
      self._disable()

  def update(self, ret: structs.CarState, can_parsers) -> None:
    """One CarState frame, from the CAN CarState was just updated with."""
    if self.disabled:
      return
    try:
      # counters of our 0x160 frames the panda refused, echoed on bus 192 since the last frame
      loopback = can_parsers.get(Bus.loopback)
      refused = loopback.vl_all["ACM_longitudinalRequest"]["ACM_longitudinalRequest_Counter"] if loopback is not None else ()
      self._update(ret, can_parsers[Bus.pt].vl["VDM_AdasSts"], can_parsers[Bus.cam].vl["ACM_Status"], refused)
    except Exception:
      self._disable()

  def _mark_refused(self, counter) -> None:
    # the newest row sent with that counter that is not already marked: a resend reuses the counter of the refused frame
    for i in range(len(self.buf) - 1, max(len(self.buf) - 1 - REFUSAL_LOOKBACK, -1), -1):
      row = self.buf[i]
      if row[CTR] == counter and not row[REFUSED]:
        row[REFUSED] = 1
        return
    self.unmatched_refusals += 1

  def _update(self, ret: structs.CarState, vdm, acm, refused) -> None:
    # The echoes refer to frames sent in earlier rows, which are all complete by now
    for counter in refused:
      self._mark_refused(int(counter))

    # Dump before appending this frame, so every row in it has had its command filled in by the CarController
    if self._reason is not None and self._countdown <= 0:
      self._emit(ret)

    now = self.clock()
    dt_ms = (now - self._last_t) * 1e3 if self._last_t is not None else 0.
    self._last_t = now

    acm_sup = int(acm["ACM_FaultSupervisorState"])
    self.buf.append([round(dt_ms, 1), None, None, None, None, 0,
                     int(vdm["VDM_AdasFaultStatus"]), int(vdm["VDM_AdasInterfaceStatus"]), int(vdm["VDM_AdasDriverModeStatus"]),
                     int(vdm["VDM_UserAdasRequest"]), round(float(vdm["VDM_AdasAccelLimit"]), 2),
                     int(acm["ACM_FeatureStatus"]), int(acm["ACM_FaultStatus"]), acm_sup,
                     round(ret.vEgo, 2), int(ret.brakePressed), int(ret.gasPressed)])

    in_drive = ret.gearShifter == GearShifter.drive

    # The ACM shut down: counted in Drive only, where it is never 3 on a healthy drive. Once reported it stays reported
    # through gear changes until the ACM leaves the state, which is noted as the recovery.
    if acm_sup == ACM_SUPERVISOR_SHUT_DOWN and (in_drive or self._latched):
      self._latch_frames += 1
    else:
      if self._latched and self.notes < MAX_DUMPS:
        self.notes += 1
        self._event("rivian_acc_latch_cleared", held_frames=self._latch_frames, acm_sup=acm_sup, gear=str(ret.gearShifter))
      self._latch_frames = 0
      self._latched = False

    # A fault in Drive is the one that drops cruise. The VDM also flags one for a single frame in Park as the truck
    # wakes up on some drives (routes 0000000e and 00000017), which is not worth a record. A fault carried into Drive
    # counts from the frame it is in Drive.
    fault = ret.accFaulted and in_drive
    fault_edge = fault and not self._prev_fault
    latch_edge = not self._latched and self._latch_frames >= LATCH_FRAMES
    if latch_edge:
      self._latched = True

    if self._reason is not None:
      self._countdown -= 1
      if fault_edge or latch_edge:
        self.suppressed += 1  # inside the record already being written
    elif fault_edge or latch_edge:
      if self.dumps < MAX_DUMPS:
        self._reason = "acc_faulted" if fault_edge else "acc_latched"
        self._countdown = POST_FRAMES
      else:
        self.suppressed += 1

    self._prev_fault = fault

  def _event(self, name: str, **kwargs) -> None:
    if self._log is None:
      return
    try:
      # error level: goes out as errorLogMessage, which the qlog keeps whole
      self._log.event(name, error=True, **kwargs)
    except Exception:
      pass

  def _emit(self, ret: structs.CarState) -> None:
    reason, self._reason = self._reason, None
    self.dumps += 1
    rows = list(self.buf)
    # columns rather than rows: about a third of the characters, and each one reads as a time series
    cols = [list(c) for c in zip(*rows, strict=True)] if rows else [[] for _ in COLUMNS]
    self._event("rivian_acc_fault_snapshot", reason=reason, frames=len(rows), trigger_row=len(rows) - 1 - POST_FRAMES,
                dump=self.dumps, suppressed=self.suppressed, unmatched_refusals=self.unmatched_refusals, gear=str(ret.gearShifter),
                cruise_enabled=int(ret.cruiseState.enabled), **dict(zip(COLUMNS, cols, strict=True)))

  def _disable(self) -> None:
    self.disabled = True
    self.buf.clear()
    if self._log is not None:
      try:
        self._log.exception("rivian_acc_fault_recorder_error")
      except Exception:
        pass
