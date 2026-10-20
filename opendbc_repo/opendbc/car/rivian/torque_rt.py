from collections import deque

# Panda real-time torque check (opendbc/safety/lateral.h): a torque request may be at most max_rt_delta (125) from a
# reference torque, and the reference is the request that happened to be sent when the last 250 ms interval ran out.
# Any torque sent in roughly the last interval can be that reference, a TOI blip frame (torque 0) included: on this
# branch's safety code a blip frame is an ordinary zero-torque frame. If the panda refuses a frame it zeroes the
# reference and its rate-limit memory, and refuses everything that is not one rate step from zero.
#
# With STEER_DELTA_UP 3 a ramp covers at most 78 counts in one 26-frame interval, so at 100 Hz this never binds today.
# It keeps the requests inside the check if the rates, the blip or the frame timing ever change.
TORQUE_RT_MAX_DELTA = 120  # panda allows 125
HIST_FRAMES = 30           # the reference is at most one 250 ms interval (26 frames) old, plus margin


class TorqueRtLimiter:
  def __init__(self):
    self.hist: deque[int] = deque([0], maxlen=HIST_FRAMES)  # every torque sent, 0 on blip frames

  def reset(self):
    """the panda's reference is 0: the torque channel is idle, or the panda refused a frame and zeroed it"""
    self.hist.clear()
    self.hist.append(0)

  def limit(self, torque: int) -> int:
    upper = max(min(self.hist), 0) + TORQUE_RT_MAX_DELTA
    lower = min(max(self.hist), 0) - TORQUE_RT_MAX_DELTA
    return max(lower, min(upper, torque))

  def sent(self, torque: int):
    self.hist.append(torque)
