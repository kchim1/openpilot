import os
import subprocess
import sys
import textwrap
import unittest
from types import SimpleNamespace

from opendbc.can import CANPacker
from opendbc.car import Bus, structs
from opendbc.car.rivian.interface import CarInterface
from opendbc.car.rivian.riviancan import create_longitudinal
from opendbc.car.rivian.values import CAR, RivianFlags
from opendbc.sunnypilot.car.rivian.acc_fault_recorder import (AccFaultRecorder, COLUMNS, LATCH_FRAMES, MAX_DUMPS, POST_FRAMES,
                                                              PRE_FRAMES, REFUSED)
from opendbc.sunnypilot.car.rivian.values import RivianFlagsSP

GearShifter = structs.CarState.GearShifter
D, P, R = GearShifter.drive, GearShifter.park, GearShifter.reverse
SNAPSHOT = "rivian_acc_fault_snapshot"
CLEARED = "rivian_acc_latch_cleared"


class Capture:
  def __init__(self):
    self.events = []

  def event(self, name, **kwargs):
    self.events.append((name, kwargs))

  def exception(self, name):
    self.events.append(("exception", {"name": name}))


def can_parsers(vdm_fault=1, iface=1, sup=0, feature=0, acm_fault=0, refused=()):
  # the resting values on this truck: VDM_AdasFaultStatus 1, ACM_FaultSupervisorState 0 once the ACM is ready.
  # refused: counters of our 0x160 frames whose bus 192 echo came in with this frame
  vdm = {"VDM_AdasFaultStatus": vdm_fault, "VDM_AdasInterfaceStatus": iface, "VDM_AdasDriverModeStatus": 0,
         "VDM_UserAdasRequest": 0, "VDM_AdasAccelLimit": 9.5}
  acm = {"ACM_FeatureStatus": feature, "ACM_FaultStatus": acm_fault, "ACM_FaultSupervisorState": sup}
  loopback = {"ACM_longitudinalRequest": {"ACM_longitudinalRequest_Counter": list(refused)}}
  return {Bus.pt: SimpleNamespace(vl={"VDM_AdasSts": vdm}), Bus.cam: SimpleNamespace(vl={"ACM_Status": acm}),
          Bus.loopback: SimpleNamespace(vl_all=loopback)}


class TestAccFaultRecorder(unittest.TestCase):
  def setUp(self):
    self.t = 0.
    self.rec = AccFaultRecorder(clock=lambda: self.t)
    self.log = Capture()
    self.rec._log = self.log
    self.frame = 0

  def _run(self, frames=1, gear=D, fault=False, sup=0, cruise=False, refused=(), refuse_prev=False, command=True, ctr=None,
           vdm_fault=None):
    """frames of CarState, each followed by the CarController's command like in card; returns the events they caused.
    Frame f sends counter f % 15 unless ctr is given. refused: echoed counters arriving with each frame; refuse_prev: the
    previous frame's request comes back refused with each frame."""
    n = len(self.log.events)
    for _ in range(frames):
      self.t += 0.01
      self.frame += 1
      ret = structs.CarState(gearShifter=gear, vEgo=self.frame / 100., accFaulted=fault)
      ret.cruiseState.enabled = cruise
      echoes = list(refused) + ([(self.frame - 1) % 15] if refuse_prev else [])
      parsers = can_parsers(vdm_fault=vdm_fault if vdm_fault is not None else (3 if fault else 1), sup=sup, feature=int(cruise),
                            refused=echoes)
      self.rec.update(ret, parsers)
      if command:
        self.rec.record_command(-self.frame / 1000., cruise, cruise, self.frame % 15 if ctr is None else ctr)
    return self.log.events[n:]

  def test_ring_buffer_keeps_the_last_frames(self):
    self._run(500, cruise=True)
    self.assertEqual(len(self.rec.buf), PRE_FRAMES + 1 + POST_FRAMES)
    self.assertEqual(self._run(POST_FRAMES + 1, fault=True, cruise=True), [])  # the trigger, then the frames after it
    trigger_frame = 501
    (name, dump), = self._run(1, cruise=True)
    self.assertEqual(name, SNAPSHOT)
    self.assertEqual(dump["reason"], "acc_faulted")
    self.assertEqual(dump["frames"], PRE_FRAMES + 1 + POST_FRAMES)
    self.assertEqual(dump["trigger_row"], PRE_FRAMES)
    for col in COLUMNS:
      self.assertEqual(len(dump[col]), dump["frames"], col)
    # one row per frame, oldest first, ending POST_FRAMES after the trigger, each with its own command
    frames = list(range(trigger_frame - PRE_FRAMES, trigger_frame + POST_FRAMES + 1))
    self.assertEqual(dump["v_ego"], [round(f / 100., 2) for f in frames])
    self.assertEqual(dump["accel"], [round(-f / 1000., 3) for f in frames])
    self.assertEqual(dump["v_ego"][dump["trigger_row"]], trigger_frame / 100.)
    self.assertEqual(dump["vdm_fault"][PRE_FRAMES - 1:PRE_FRAMES + 1], [1, 3])
    self.assertEqual(set(dump["dt_ms"]), {10.0})
    self.assertEqual(set(dump["en"]), {1})

  def test_command_completes_the_newest_row(self):
    self.rec.record_command(1.0, True, True)  # nothing recorded yet: nothing to complete, no error
    self.assertEqual(len(self.rec.buf), 0)
    self._run(1, command=False)
    self.assertEqual(self.rec.buf[-1][1:4], [None, None, None])  # openpilot not controlling the speed
    self.rec.record_command(0.5, True, False, 3)
    self._run(1, command=False)
    self.assertEqual([row[1:5] for row in self.rec.buf], [[0.5, 1, 0, 3], [None, None, None, None]])

  def test_refused_requests_are_marked_on_the_rows_that_sent_them(self):
    # route 00000015: nine frames in a row refused before the fault. Each echo comes back one to three frames after its
    # frame went out, and is matched to it by the counter
    self._run(20, cruise=True)
    start = self.frame + 1
    echoes: dict[int, list[int]] = {}
    for i, lag in enumerate((1, 1, 1, 2, 1, 1, 3, 1, 1)):
      echoes.setdefault(start + i + lag, []).append((start + i) % 15)
    for f in range(start, start + 12):
      self._run(1, cruise=True, refused=echoes.get(f, ()))
    self._run(POST_FRAMES + 1, fault=True)
    (_, dump), = self._run(1)
    trigger = dump["trigger_row"]
    self.assertEqual(dump["refused"][trigger - 13:trigger + 1], [0] + [1] * 9 + [0] * 4)
    self.assertEqual(dump["ctr"][trigger - 12:trigger - 3], [f % 15 for f in range(start, start + 9)])
    self.assertEqual(dump["unmatched_refusals"], 0)

  def test_a_refused_resend_is_marked_on_its_own_row(self):
    # the CarController resends the counter of a refused frame when nothing newer went out
    self._run(20, cruise=True)
    k = self.frame + 1
    self._run(1, cruise=True)
    self._run(1, cruise=True, refused=[k % 15], ctr=k % 15)  # the echo of frame k; the resend, refused too
    self._run(1, cruise=True, refused=[k % 15])
    self._run(1, cruise=True)
    self.assertEqual([row[REFUSED] for row in list(self.rec.buf)[-4:]], [1, 1, 0, 0])

  def test_an_echo_with_no_recent_frame_is_counted_not_marked(self):
    self._run(20, cruise=True)
    self._run(1, cruise=True, refused=[(self.frame - 10) % 15])  # older than the look-back
    self.assertEqual(sum(row[REFUSED] for row in self.rec.buf), 0)
    self.assertEqual(self.rec.unmatched_refusals, 1)

  def test_fault_in_drive_is_recorded_once_per_edge(self):
    self._run(100, cruise=True)
    events = self._run(300, fault=True)  # held: one record, POST_FRAMES after the edge
    self.assertEqual([(name, kw["reason"], kw["dump"]) for name, kw in events], [(SNAPSHOT, "acc_faulted", 1)])
    events = self._run(100) + self._run(50, fault=True) + self._run(50)
    self.assertEqual([(name, kw["reason"], kw["dump"]) for name, kw in events], [(SNAPSHOT, "acc_faulted", 2)])

  def test_dumps_are_capped(self):
    events = []
    for _ in range(MAX_DUMPS + 3):
      events += self._run(50) + self._run(50, fault=True)
    events += self._run(50)
    self.assertEqual([kw["dump"] for _, kw in events], list(range(1, MAX_DUMPS + 1)))
    self.assertEqual(self.rec.suppressed, 3)

  def test_fault_carried_into_drive_is_recorded(self):
    self._run(100, gear=P)
    events = self._run(5, gear=P, fault=True) + self._run(POST_FRAMES + 2, fault=True)
    self.assertEqual([(name, kw["reason"]) for name, kw in events], [(SNAPSHOT, "acc_faulted")])
    self.assertEqual(events[0][1]["v_ego"][events[0][1]["trigger_row"]], 106 / 100.)

  def test_fault_out_of_drive_is_not_recorded(self):
    # the VDM flags a fault for one frame in Park as the truck wakes up (routes 0000000e and 00000017)
    for gear in (P, R):
      self.assertEqual(self._run(100, gear=gear) + self._run(1, gear=gear, fault=True) + self._run(200, gear=gear), [])
    self.assertEqual(self.rec.dumps, 0)

  def test_acm_shut_down_in_drive_and_recovery(self):
    self._run(200, cruise=True)
    self.assertEqual(self._run(LATCH_FRAMES - 1, sup=3) + self._run(10), [])  # a blip
    events = self._run(LATCH_FRAMES + POST_FRAMES + 1, sup=3)
    self.assertEqual([(name, kw["reason"]) for name, kw in events], [(SNAPSHOT, "acc_latched")])
    self.assertEqual(events[0][1]["acm_sup"][events[0][1]["trigger_row"]], 3)
    # still down while the driver shifts to Reverse and Park: nothing more, it has not recovered
    self.assertEqual(self._run(300, sup=3) + self._run(100, gear=R, sup=3) + self._run(100, gear=P, sup=3), [])
    events = self._run(1, gear=P)
    self.assertEqual([name for name, _ in events], [CLEARED])
    self.assertEqual(events[0][1]["held_frames"], LATCH_FRAMES + POST_FRAMES + 1 + 500)
    self.assertEqual(events[0][1]["gear"], "park")
    # the ACM back at 3 in Park, as when the truck shuts down: not a shut-down in Drive
    self.assertEqual(self._run(500, gear=P, sup=3), [])

  def test_shut_down_inside_a_fault_record_is_counted(self):
    # the ACM already down for most of a second when the fault comes: its latch lands inside the fault's record
    self._run(200, cruise=True)
    events = self._run(LATCH_FRAMES - 10, sup=3) + self._run(POST_FRAMES + 1, fault=True, sup=3) + self._run(1, sup=3)
    self.assertEqual([(name, kw["reason"]) for name, kw in events], [(SNAPSHOT, "acc_faulted")])
    self.assertEqual(self.rec.suppressed, 1)

  def test_recovery_noted_once_and_the_next_shut_down_reported(self):
    self._run(200, cruise=True)
    self._run(LATCH_FRAMES + POST_FRAMES + 1, sup=3)
    self.assertEqual([name for name, _ in self._run(1)], [CLEARED])
    self.assertEqual(self._run(300), [])
    self.assertEqual([(n, kw.get("reason")) for n, kw in self._run(LATCH_FRAMES + POST_FRAMES + 1, sup=3)], [(SNAPSHOT, "acc_latched")])

  def test_recovery_notes_are_capped(self):
    events = []
    for _ in range(MAX_DUMPS + 3):
      events += self._run(LATCH_FRAMES + POST_FRAMES + 1, sup=3) + self._run(10)
    self.assertEqual([name for name, _ in events].count(CLEARED), MAX_DUMPS)
    self.assertEqual(self.rec.dumps, MAX_DUMPS)

  def test_record_shape(self):
    # in plain numbers, so a retune of the module's constants shows up here
    self._run(200, cruise=True)
    self.assertEqual(self._run(50, sup=3) + self._run(10), [])  # 0.5 s of the ACM down in Drive is a blip
    self.assertEqual(self._run(26, fault=True), [])
    (_, dump), = self._run(1)
    self.assertEqual((dump["frames"], dump["trigger_row"]), (76, 50))
    for _ in range(6):
      self._run(50)
      self._run(50, fault=True)
    self.assertEqual(self.rec.dumps, 4)

  def test_every_column_is_its_signal(self):
    vdm = {"VDM_AdasFaultStatus": 3, "VDM_AdasInterfaceStatus": 1, "VDM_AdasDriverModeStatus": 2, "VDM_UserAdasRequest": 4,
           "VDM_AdasAccelLimit": 7.25}
    acm = {"ACM_FeatureStatus": 1, "ACM_FaultStatus": 2, "ACM_FaultSupervisorState": 3}
    parsers = can_parsers()
    parsers[Bus.pt].vl["VDM_AdasSts"] = vdm
    parsers[Bus.cam].vl["ACM_Status"] = acm
    self.rec.update(structs.CarState(gearShifter=D, vEgo=12.5, brakePressed=True, gasPressed=False), parsers)
    self.rec.record_command(-0.5, True, False, 7)
    self.rec.update(structs.CarState(gearShifter=D), can_parsers(refused=[7]))
    row = dict(zip(COLUMNS, self.rec.buf[0], strict=True))
    self.assertEqual(row, {"dt_ms": 0., "accel": -0.5, "en": 1, "allowed": 0, "ctr": 7, "refused": 1, "vdm_fault": 3, "vdm_iface": 1,
                           "vdm_mode": 2, "vdm_user": 4, "vdm_accel_lim": 7.25, "acm_feat": 1, "acm_fault": 2, "acm_sup": 3,
                           "v_ego": 12.5, "brake": 1, "gas": 0})

  def test_fault_then_shut_down(self):
    # route 00000015: the VDM faults, the ACM leaves cruise and shuts down 4 frames later and stays down
    self._run(200, cruise=True)
    events = self._run(3, fault=True) + self._run(4) + self._run(1000, sup=3) + self._run(1)
    self.assertEqual([(name, kw.get("reason")) for name, kw in events],
                     [(SNAPSHOT, "acc_faulted"), (SNAPSHOT, "acc_latched"), (CLEARED, None)])

  def test_healthy_drive_is_silent(self):
    events = []
    # the truck wakes up in Park: the ACM not ready yet, and the VDM flags a fault for a frame
    events += self._run(300, gear=P, sup=2) + self._run(100, gear=P, sup=3) + self._run(1, gear=P, fault=True)
    events += self._run(200, gear=P) + self._run(300, gear=R) + self._run(500)
    # cruise on and off, with a few refused requests on a pedal press
    for _ in range(5):
      events += self._run(1000, cruise=True) + self._run(2, cruise=True, refuse_prev=True) + self._run(500)
    # parked, the ACM shuts down with the truck
    events += self._run(100, gear=P) + self._run(1000, gear=P, sup=3)
    self.assertEqual(events, [])
    self.assertEqual((self.rec.dumps, self.rec.suppressed, self.rec.disabled), (0, 0, False))

  def test_an_error_never_reaches_the_car(self):
    self._run(10)
    self.rec.update(structs.CarState(), {Bus.pt: SimpleNamespace(vl={})})  # missing messages
    self.assertTrue(self.rec.disabled)
    self.assertEqual(self.log.events, [("exception", {"name": "rivian_acc_fault_recorder_error"})])
    # off for the rest of the drive, quietly
    self._run(500, fault=True)
    self.rec.record_command(1.0, True, True)
    self.assertEqual(len(self.log.events), 1)
    self.assertEqual(len(self.rec.buf), 0)

  def test_errors_in_the_command_or_the_log_never_reach_the_car(self):
    self._run(1)
    self.rec.record_command("not a number", True, True)
    self.assertTrue(self.rec.disabled)

    class BrokenLog:
      def event(self, *a, **k):
        raise RuntimeError

      def exception(self, *a, **k):
        raise RuntimeError
    rec = AccFaultRecorder()
    rec._log = BrokenLog()
    rec.update(structs.CarState(), {Bus.pt: SimpleNamespace(vl={})})
    self.assertTrue(rec.disabled)

  def test_importable_and_silent_without_openpilot(self):
    # opendbc is used without openpilot (the panda safety tests, other projects): openpilot is only imported lazily
    code = textwrap.dedent("""
      import sys
      sys.modules["openpilot"] = None  # any import of openpilot now fails
      from opendbc.car import structs
      from opendbc.car.rivian.interface import CarInterface
      from opendbc.sunnypilot.car.rivian.acc_fault_recorder import AccFaultRecorder
      from opendbc.sunnypilot.car.rivian.tests.test_acc_fault_recorder import can_parsers
      CI = CarInterface(CarInterface.get_non_essential_params("RIVIAN_R1"), structs.CarParamsSP())
      rec = CI.CS.acc_fault_recorder
      assert isinstance(rec, AccFaultRecorder) and rec._log is None
      for i in range(200):
        ret = structs.CarState(gearShifter=structs.CarState.GearShifter.drive, accFaulted=50 <= i < 60)
        rec.update(ret, can_parsers(sup=3 if i >= 60 else 0))
      assert rec.dumps == 2 and not rec.disabled
      print("ok")
    """)
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, env=os.environ, timeout=120)
    self.assertEqual(out.stdout.strip(), "ok", out.stderr[-2000:])


class TestAccFaultRecorderWiring(unittest.TestCase):
  """The truck's side comes from CarState, openpilot's from the CarController, in the same card loop."""

  def _interface(self):
    CP = CarInterface.get_non_essential_params(CAR.RIVIAN_R1)
    CP.openpilotLongitudinalControl = True
    CP.flags &= ~RivianFlags.GEN2.value  # this truck: Gen 1, with the wheel touch sensor on the bus
    CI = CarInterface(CP, structs.CarParamsSP(flags=int(RivianFlagsSP.LONGITUDINAL_HARNESS_UPGRADE)))
    CI.CS.acc_fault_recorder._log = Capture()
    return CI

  def test_carstate_row_completed_by_the_controller(self):
    CI = self._interface()
    rec = CI.CS.acc_fault_recorder

    # the panda refused our first 0x160 frame (counter 1): its echo comes back on bus 192 with the next frame, which
    # resends that counter
    refused = create_longitudinal(CANPacker("rivian_primary_actuator"), 1, 1.0, True)
    for t, frames in ((10_000_000, []), (20_000_000, [(refused[0], refused[1], 192)]), (30_000_000, [])):
      CI.update([(t, frames)])
      CI.CS.out.cruiseState.enabled = True
      CI.CS.out.vEgo = CI.CS.out.vEgoRaw = 10.
      car_control = structs.CarControl(enabled=True, longActive=True, actuators={"accel": 1.0}).as_reader()
      CI.CC.update(car_control, structs.CarControlSP(), CI.CS, 0)

    self.assertEqual([row[1:6] for row in rec.buf], [[1.0, 1, 1, 1, 1], [1.0, 1, 1, 1, 0], [1.0, 1, 1, 2, 0]])
    self.assertFalse(rec.disabled)
    self.assertEqual(rec._log.events, [])

  def test_records_what_went_out(self):
    # highway speed with the cruise bias; the gas pressed with cruise on; the ACM out of cruise before carControl
    CI = self._interface()
    for i, (cruise, gas) in enumerate(((True, False), (True, True), (False, False))):
      CI.update([(10_000_000 * (i + 1), [])])
      CI.CS.out.cruiseState.enabled = cruise
      CI.CS.out.gasPressed = gas
      CI.CS.out.vEgo = CI.CS.out.vEgoRaw = 25.
      car_control = structs.CarControl(enabled=True, longActive=True, actuators={"accel": 1.0}).as_reader()
      CI.CC.update(car_control, structs.CarControlSP(), CI.CS, 0)
    self.assertEqual([row[1:4] for row in CI.CS.acc_fault_recorder.buf], [[0.92, 1, 1], [0.0, 1, 0], [0.0, 0, 0]])

if __name__ == "__main__":
  unittest.main()
