#!/usr/bin/env python3
import os
import time
import numpy as np
from openpilot.cereal import log
from opendbc.car.interfaces import ACCEL_MIN, ACCEL_MAX
from openpilot.common.realtime import DT_MDL
from openpilot.common.swaglog import cloudlog
# WARNING: imports outside of constants will not trigger a rebuild
from openpilot.selfdrive.modeld.constants import index_function
from openpilot.selfdrive.controls.radard import _LEAD_ACCEL_TAU

if __name__ == '__main__':  # generating code
  from acados.acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
else:
  from openpilot.selfdrive.controls.lib.longitudinal_mpc_lib.c_generated_code.acados_ocp_solver_pyx import AcadosOcpSolverCython

from casadi import SX, vertcat

MODEL_NAME = 'long'
LONG_MPC_DIR = os.path.dirname(os.path.abspath(__file__))
EXPORT_DIR = os.path.join(LONG_MPC_DIR, "c_generated_code")
JSON_FILE = os.path.join(LONG_MPC_DIR, "acados_ocp_long.json")

LongitudinalPlanSource = log.LongitudinalPlan.LongitudinalPlanSource
MPC_SOURCES = (LongitudinalPlanSource.lead0, LongitudinalPlanSource.lead1)

X_DIM = 3
U_DIM = 1
PARAM_DIM = 6
COST_E_DIM = 5
COST_DIM = COST_E_DIM + 1
CONSTR_DIM = 4

X_EGO_OBSTACLE_COST = 3.
X_EGO_COST = 0.
V_EGO_COST = 0.
A_EGO_COST = 0.
J_EGO_COST = 5.
A_CHANGE_COST = 200.
DANGER_ZONE_COST = 100.
CRASH_DISTANCE = .25
LEAD_DANGER_FACTOR = 0.75
LIMIT_COST = 1e6
ACADOS_SOLVER_TYPE = 'SQP_RTI'

# Fewer timestamps don't hurt performance and lead to
# much better convergence of the MPC with low iterations
N = 12
MAX_T = 10.0
T_IDXS_LST = [index_function(idx, max_val=MAX_T, max_idx=N) for idx in range(N+1)]

T_IDXS = np.array(T_IDXS_LST)
FCW_IDXS = T_IDXS < 5.0
T_DIFFS = np.diff(T_IDXS, prepend=[0.])
COMFORT_BRAKE = 2.5
STOP_DISTANCE = 6.0
MIN_X_LEAD_FACTOR = 0.5

def get_jerk_factor(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.0
  elif personality==log.LongitudinalPersonality.standard:
    return 1.0
  elif personality==log.LongitudinalPersonality.aggressive:
    return 0.5
  else:
    raise NotImplementedError("Longitudinal personality not supported")


def get_T_FOLLOW(personality=log.LongitudinalPersonality.standard):
  if personality==log.LongitudinalPersonality.relaxed:
    return 1.75
  elif personality==log.LongitudinalPersonality.standard:
    return 1.45
  elif personality==log.LongitudinalPersonality.aggressive:
    return 1.25
  else:
    raise NotImplementedError("Longitudinal personality not supported")

# Following distance offset limits, in percent of the personality's follow time.
# Enforced here so no setting, sunnylink push or caller can scale the gap past them.
T_FOLLOW_OFFSET_PCT_MIN = -10
T_FOLLOW_OFFSET_PCT_MAX = 10


def clip_t_follow_offset_pct(offset_pct):
  try:
    offset_pct = float(offset_pct)
  except (TypeError, ValueError):
    return 0.
  if not np.isfinite(offset_pct):
    return 0.
  return float(np.clip(offset_pct, T_FOLLOW_OFFSET_PCT_MIN, T_FOLLOW_OFFSET_PCT_MAX))

# Traffic Mode: a shorter gap for stop-and-go traffic, switched live by the sunnypilot planner.
# 0.6 s at standstill, 0.8 s from 5 m/s (11 mph) to 8.94 m/s (20 mph), 0.9 s at 13.41 m/s (30 mph) and 1.0 s from 17.88 m/s
# (40 mph) up, as before there (Aggressive is 1.25 s). It replaces the personality's follow time and ignores the offset,
# except that a Rivian never follows farther back than its Following Distance with the offset (get_t_follow). From a stop
# it moves off sooner (TRAFFIC_MODE_LAUNCH_SHIFT) and below 40 mph the planner caps its acceleration (longitudinal_planner.py);
# COMFORT_BRAKE, LEAD_DANGER_FACTOR, DANGER_ZONE_COST and the solver's accel limits do not change.
TRAFFIC_MODE_T_FOLLOW_BP = [0., 5., 8.94, 13.41, 17.88]
TRAFFIC_MODE_T_FOLLOW_V = [0.6, 0.8, 0.8, 0.9, 1.0]
TRAFFIC_MODE_JERK_FACTOR = 0.5  # same as Aggressive
# Traffic Mode moves off sooner behind a car that pulls away from a stop. Stop Behind Lead stops the truck about 4 m behind
# a stopped car, short of the solver's STOP_DISTANCE, so the plan would first wait for that car to open up the difference.
# Once the plan has stopped (its speed under TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP[0]) behind a radar leadOne, that lead's
# obstacle moves farther by the room the truck stopped short of STOP_DISTANCE - TRAFFIC_MODE_LAUNCH_SHIFT_MARGIN, at most
# TRAFFIC_MODE_LAUNCH_SHIFT: it can't pull the truck toward a car that stays put. The gap is the closest the truck comes
# while stopped, and the room tapers to none for a gap under TRAFFIC_MODE_LAUNCH_SHIFT_GAP_BP[1] (Stop Behind Lead's
# GAP_MIN), so a truck that ended up close gets less head start, or none. Once the plan
# moves off it only comes down: the truck's own travel since the stop spends it, and it fades out with the plan's speed,
# none from TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP[1]. Each stop measures its gap again. It builds up at most
# TRAFFIC_MODE_LAUNCH_SHIFT_RATE. Another radar track as leadOne, or a vision-only one, gets none until the truck stops
# behind it; leadTwo gets the same only while it is leadOne's own track. Shifting while moving too ended town stops 1-2 m
# closer than without it (the truck under-brakes at a crawl and Stop Behind Lead only backstops 4 m of real gap), and a
# shift kept through a crawl, or the full one after a short stop, closed up an inching queue a little more on every inch,
# down to contact with Launch Assist, in closed-loop sims. The FCW check stays on the lead itself
TRAFFIC_MODE_LAUNCH_SHIFT = 2.0                  # m
TRAFFIC_MODE_LAUNCH_SHIFT_MARGIN = 0.3           # m
TRAFFIC_MODE_LAUNCH_SHIFT_GAP_BP = [2.5, 3.5]    # m: none for a truck stopped this close, all of it from here
TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP = [0.1, 1.0]  # m/s
TRAFFIC_MODE_LAUNCH_SHIFT_RATE = 1.0             # m/s


def get_traffic_mode_T_FOLLOW(v_ego):
  return float(np.interp(v_ego, TRAFFIC_MODE_T_FOLLOW_BP, TRAFFIC_MODE_T_FOLLOW_V))

# Rivian Launch Assist, set by the sunnypilot planner: while the planned speed is nearly zero, a radar lead that is
# pulling away keeps its acceleration longer in the prediction (smaller aLeadTau), so the plan starts following sooner.
LAUNCH_ASSIST_V_EGO_MAX = 1.0
LAUNCH_ASSIST_V_LEAD_MIN = 0.3
LAUNCH_ASSIST_V_LEAD_K_MIN = 0.15  # filtered speed too, so one noisy raw vLead sample can't trigger it
LAUNCH_ASSIST_A_LEAD_MIN = 0.1
LAUNCH_ASSIST_A_LEAD_TAU = 0.3

# Fast lead acceleration, set by the sunnypilot planner (fast_lead_accel.py): when it sees leadOne braking sooner than
# radard's aLeadK does, the MPC plans with its estimate instead, kept up longer (aLeadTau at most this)
FAST_LEAD_A_TAU = 0.3

# Brake Sooner for Stops, set by the sunnypilot planner: closing on a stopped or slow lead, plan as if the comfort
# brake were approach_comfort_brake instead of COMFORT_BRAKE, so braking starts sooner and peaks lower. The solver's
# desired distance is fixed at build time, so the lead's obstacle moves closer by the difference in its v_ego^2 term (at
# the speeds the last plan expected) less the same term at the lead's speed: none at the lead's speed, none at a stop.
# All of it for a lead at 3 m/s or less, none from 8 m/s, so following moving traffic is unchanged. Only while a gentle
# stop is still possible (the constant decel to stop STOP_DISTANCE short of where the lead stops: all of it up to
# 1.2 m/s^2, none from 1.7), so a car seen late gets the stock plan instead of harder braking. Built up at most
# APPROACH_MARGIN_RATE so a new lead or a lead switch doesn't jolt the plan.
APPROACH_V_LEAD_BP = [3., 8.]
APPROACH_DECEL_BP = [1.2, 1.7]
APPROACH_MARGIN_RATE = 10.  # m/s
# None of it below APPROACH_V_EGO_BP[0], all of it from [1]: at town speeds the extra caution slowed the truck well short
# of the car (route 00000024, 3:09: about 5 m of extra margin at 7 m/s, 31 m back); the onset at speed is unchanged
APPROACH_V_EGO_BP = [6., 10.]


def get_stopped_equivalence_factor(v_lead):
  return (v_lead**2) / (2 * COMFORT_BRAKE)

def get_safe_obstacle_distance(v_ego, t_follow):
  return (v_ego**2) / (2 * COMFORT_BRAKE) + t_follow * v_ego + STOP_DISTANCE

def gen_long_model():
  model = AcadosModel()
  model.name = MODEL_NAME

  # states
  x_ego, v_ego, a_ego = SX.sym('x_ego'), SX.sym('v_ego'), SX.sym('a_ego')
  model.x = vertcat(x_ego, v_ego, a_ego)

  # controls
  j_ego = SX.sym('j_ego')
  model.u = vertcat(j_ego)

  # xdot
  x_ego_dot = SX.sym('x_ego_dot')
  v_ego_dot = SX.sym('v_ego_dot')
  a_ego_dot = SX.sym('a_ego_dot')
  model.xdot = vertcat(x_ego_dot, v_ego_dot, a_ego_dot)

  # runtime parameters
  a_min = SX.sym('a_min')
  a_max = SX.sym('a_max')
  x_obstacle = SX.sym('x_obstacle')
  a_prev = SX.sym('a_prev')
  lead_t_follow = SX.sym('lead_t_follow')
  lead_danger_factor = SX.sym('lead_danger_factor')
  model.p = vertcat(a_min, a_max, x_obstacle, a_prev, lead_t_follow, lead_danger_factor)

  # dynamics model
  f_expl = vertcat(v_ego, a_ego, j_ego)
  model.f_impl_expr = model.xdot - f_expl
  model.f_expl_expr = f_expl
  return model

def gen_long_ocp():
  ocp = AcadosOcp()
  ocp.model = gen_long_model()

  Tf = T_IDXS[-1]

  # set dimensions
  ocp.dims.N = N

  # set cost module
  ocp.cost.cost_type = 'NONLINEAR_LS'
  ocp.cost.cost_type_e = 'NONLINEAR_LS'

  QR = np.zeros((COST_DIM, COST_DIM))
  Q = np.zeros((COST_E_DIM, COST_E_DIM))

  ocp.cost.W = QR
  ocp.cost.W_e = Q

  x_ego, v_ego, a_ego = ocp.model.x[0], ocp.model.x[1], ocp.model.x[2]
  j_ego = ocp.model.u[0]

  a_min, a_max = ocp.model.p[0], ocp.model.p[1]
  x_obstacle = ocp.model.p[2]
  a_prev = ocp.model.p[3]
  lead_t_follow = ocp.model.p[4]
  lead_danger_factor = ocp.model.p[5]

  ocp.cost.yref = np.zeros((COST_DIM, ))
  ocp.cost.yref_e = np.zeros((COST_E_DIM, ))

  desired_dist_comfort = get_safe_obstacle_distance(v_ego, lead_t_follow)

  # The main cost in normal operation is how close you are to the "desired" distance
  # from an obstacle at every timestep. This obstacle can be a lead car
  # or other object. In e2e mode we can use x_position targets as a cost
  # instead.
  costs = [((x_obstacle - x_ego) - (desired_dist_comfort)) / (v_ego + 10.),
           x_ego,
           v_ego,
           a_ego,
           a_ego - a_prev,
           j_ego]
  ocp.model.cost_y_expr = vertcat(*costs)
  ocp.model.cost_y_expr_e = vertcat(*costs[:-1])

  # Constraints on speed, acceleration and desired distance to
  # the obstacle, which is treated as a slack constraint so it
  # behaves like an asymmetrical cost.
  constraints = vertcat(v_ego,
                        (a_ego - a_min),
                        (a_max - a_ego),
                        ((x_obstacle - x_ego) - lead_danger_factor * (desired_dist_comfort)) / (v_ego + 10.))
  ocp.model.con_h_expr = constraints

  x0 = np.zeros(X_DIM)
  ocp.constraints.x0 = x0
  ocp.parameter_values = np.array([-1.2, 1.2, 0.0, 0.0, get_T_FOLLOW(), LEAD_DANGER_FACTOR])


  # We put all constraint cost weights to 0 and only set them at runtime
  cost_weights = np.zeros(CONSTR_DIM)
  ocp.cost.zl = cost_weights
  ocp.cost.Zl = cost_weights
  ocp.cost.Zu = cost_weights
  ocp.cost.zu = cost_weights

  ocp.constraints.lh = np.zeros(CONSTR_DIM)
  ocp.constraints.uh = 1e4*np.ones(CONSTR_DIM)
  ocp.constraints.idxsh = np.arange(CONSTR_DIM)

  # The HPIPM solver can give decent solutions even when it is stopped early
  # Which is critical for our purpose where compute time is strictly bounded
  # We use HPIPM in the SPEED_ABS mode, which ensures fastest runtime. This
  # does not cause issues since the problem is well bounded.
  ocp.solver_options.qp_solver = 'PARTIAL_CONDENSING_HPIPM'
  ocp.solver_options.hessian_approx = 'GAUSS_NEWTON'
  ocp.solver_options.integrator_type = 'ERK'
  ocp.solver_options.nlp_solver_type = ACADOS_SOLVER_TYPE
  ocp.solver_options.qp_solver_cond_N = 1

  # More iterations take too much time and less lead to inaccurate convergence in
  # some situations. Ideally we would run just 1 iteration to ensure fixed runtime.
  ocp.solver_options.qp_solver_iter_max = 10
  ocp.solver_options.qp_tol = 1e-3

  # set prediction horizon
  ocp.solver_options.tf = Tf
  ocp.solver_options.shooting_nodes = T_IDXS

  ocp.code_export_directory = EXPORT_DIR
  return ocp


class LongitudinalMpc:
  def __init__(self, dt=DT_MDL):
    self.dt = dt
    # Following distance offset in percent, set by the sunnypilot planner at the start of a drive
    self.t_follow_offset_pct = 0.
    # Traffic Mode, set by the sunnypilot planner while driving. Must exist before reset(), which calls set_weights()
    self.traffic_mode = False
    # Rivian Launch Assist, set by the sunnypilot planner while driving
    self.launch_assist = False
    # Brake Sooner for Stops, set by the sunnypilot planner while driving (COMFORT_BRAKE: off)
    self.approach_comfort_brake = COMFORT_BRAKE
    # Rivian Following Distance, set by the sunnypilot planner while driving: the follow time before the offset, in
    # place of the personality's. None: the personality's
    self.follow_t_base = None
    # Fast lead acceleration, set by the sunnypilot planner while driving: leadOne's acceleration to plan with in place of
    # aLeadK. None: aLeadK
    self.a_lead_fast = None
    # follow time and jerk factor last passed to the solver, for logging
    self.t_follow = get_T_FOLLOW()
    self.jerk_factor = get_jerk_factor()
    self.solver = AcadosOcpSolverCython(MODEL_NAME, ACADOS_SOLVER_TYPE, N)
    self.reset()
    self.source = LongitudinalPlanSource.cruise

  def reset(self):
    self.solver.reset()

    self.x_sol = np.zeros((N+1, X_DIM))
    self.u_sol = np.zeros((N, 1))
    self.approach_margin_last = [0., 0.]  # largest approach margin applied last frame, per lead
    self.launch_shift = 0.          # Traffic Mode's launch shift applied last frame, to leadOne's obstacle
    self.launch_shift_track = -1    # leadOne's radar track it is for (-1: none)
    self.launch_shift_gap = None    # the gap the truck stopped at behind that track (None: not stopped behind it)
    self.launch_shift_travel = 0.   # how far the plan has moved since that stop (m)
    self.v_solution = np.zeros(N+1)
    self.a_solution = np.zeros(N+1)
    self.j_solution = np.zeros(N)
    self.a_prev = np.array(self.a_solution)
    self.yref = np.zeros((N+1, COST_DIM))

    for i in range(N):
      self.solver.cost_set(i, "yref", self.yref[i])
    self.solver.cost_set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params = np.zeros((N+1, PARAM_DIM))
    for i in range(N+1):
      self.solver.set(i, 'x', np.zeros(X_DIM))

    self.last_cloudlog_t = 0
    self.crash_cnt = 0.0
    self.solution_status = 0
    # timers
    self.solve_time = 0.0
    self.x0 = np.zeros(X_DIM)
    self.set_weights()

  def set_cost_weights(self, cost_weights, constraint_cost_weights):
    W = np.asfortranarray(np.diag(cost_weights))
    for i in range(N):
      # TODO don't hardcode A_CHANGE_COST idx
      # reduce the cost on (a-a_prev) later in the horizon.
      W[4,4] = cost_weights[4] * np.interp(T_IDXS[i], [0.0, 1.0, 2.0], [1.0, 1.0, 0.0])
      self.solver.cost_set(i, 'W', W)
    # Setting the slice without the copy make the array not contiguous,
    # causing issues with the C interface.
    self.solver.cost_set(N, 'W', np.copy(W[:COST_E_DIM, :COST_E_DIM]))

    # Set L2 slack cost on lower bound constraints
    Zl = np.array(constraint_cost_weights)
    for i in range(N):
      self.solver.cost_set(i, 'Zl', Zl)

  def set_weights(self, prev_accel_constraint=True, personality=log.LongitudinalPersonality.standard):
    jerk_factor = TRAFFIC_MODE_JERK_FACTOR if self.traffic_mode else get_jerk_factor(personality)
    self.jerk_factor = jerk_factor
    a_change_cost = A_CHANGE_COST if prev_accel_constraint else 0
    cost_weights = [X_EGO_OBSTACLE_COST, X_EGO_COST, V_EGO_COST, A_EGO_COST, jerk_factor * a_change_cost, jerk_factor * J_EGO_COST]
    constraint_cost_weights = [LIMIT_COST, LIMIT_COST, LIMIT_COST, DANGER_ZONE_COST]
    self.set_cost_weights(cost_weights, constraint_cost_weights)

  def set_cur_state(self, v, a):
    v_prev = self.x0[1]
    self.x0[1] = v
    self.x0[2] = a
    if abs(v_prev - v) > 2.:  # probably only helps if v < v_prev
      for i in range(N+1):
        self.solver.set(i, 'x', self.x0)

  @staticmethod
  def extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau):
    a_lead_traj = a_lead * np.exp(-a_lead_tau * (T_IDXS**2)/2.)
    v_lead_traj = np.clip(v_lead + np.cumsum(T_DIFFS * a_lead_traj), 0.0, 1e8)
    x_lead_traj = x_lead + np.cumsum(T_DIFFS * v_lead_traj)
    lead_xv = np.column_stack((x_lead_traj, v_lead_traj))
    return lead_xv

  def process_lead(self, lead, a_lead_fast=None):
    v_ego = self.x0[1]
    if lead is not None and lead.present:
      x_lead = lead.dRel
      v_lead = lead.vLead
      a_lead = lead.aLeadK
      a_lead_tau = lead.aLeadTau
      if self.launch_assist and lead.radar and v_ego < LAUNCH_ASSIST_V_EGO_MAX and v_lead > LAUNCH_ASSIST_V_LEAD_MIN and \
         lead.vLeadK > LAUNCH_ASSIST_V_LEAD_K_MIN and a_lead > LAUNCH_ASSIST_A_LEAD_MIN:
        a_lead_tau = min(a_lead_tau, LAUNCH_ASSIST_A_LEAD_TAU)
      if a_lead_fast is not None:
        a_lead = a_lead_fast
        a_lead_tau = min(a_lead_tau, FAST_LEAD_A_TAU)
    else:
      # Fake a fast lead car, so mpc can keep running in the same mode
      x_lead = 50.0
      v_lead = v_ego + 10.0
      a_lead = 0.0
      a_lead_tau = _LEAD_ACCEL_TAU

    # MPC will not converge if immediate crash is expected
    # Clip lead distance to what is still possible to brake for
    min_x_lead = MIN_X_LEAD_FACTOR * (v_ego + v_lead) * (v_ego - v_lead) / (-ACCEL_MIN * 2)
    x_lead = np.clip(x_lead, min_x_lead, 1e8)
    v_lead = np.clip(v_lead, 0.0, 1e8)
    a_lead = np.clip(a_lead, -10., 5.)
    lead_xv = self.extrapolate_lead(x_lead, v_lead, a_lead, a_lead_tau)
    return lead_xv

  def plan_speeds(self):
    # the solver's v_ego is symbolic: use the last plan's speeds, or the current speed if that plan is stale
    v_ego = self.x0[1]
    return self.x_sol[:,1] if abs(self.x_sol[0,1] - v_ego) < 1. else np.full(N+1, v_ego)

  def approach_margin(self, idx, lead, lead_xv):
    """Brake Sooner for Stops: how much closer to put this lead's obstacle at each step of the horizon"""
    k = 1 / (2 * self.approach_comfort_brake) - 1 / (2 * COMFORT_BRAKE)
    if k <= 0. or lead is None or not lead.present or not lead.radar:
      self.approach_margin_last[idx] = 0.
      return np.zeros(N+1)
    v_ego = self.x0[1]
    v_plan = self.plan_speeds()
    margin = k * np.clip(v_plan**2 - lead_xv[:,1]**2, 0., None)
    d_stop = max(lead_xv[0,0] + get_stopped_equivalence_factor(lead_xv[0,1]) - STOP_DISTANCE, 0.1)
    margin *= np.interp(lead_xv[0,1], APPROACH_V_LEAD_BP, [1., 0.]) * np.interp(v_ego**2 / (2 * d_stop), APPROACH_DECEL_BP, [1., 0.]) * \
              np.interp(v_ego, APPROACH_V_EGO_BP, [0., 1.])
    target = float(np.max(margin))
    applied = min(target, self.approach_margin_last[idx] + APPROACH_MARGIN_RATE * self.dt)
    self.approach_margin_last[idx] = applied
    return margin * (applied / target) if target > 1e-3 else np.zeros(N+1)

  def traffic_mode_launch_shift(self, lead):
    """Traffic Mode: how much farther to put leadOne's obstacle, from a stop behind it until the truck has moved off"""
    track = lead.radarTrackId if self.traffic_mode and lead is not None and lead.present and lead.radar else -1
    if track != self.launch_shift_track:
      # Traffic Mode off, no radar lead, or another car: none until the truck stops behind this one
      self.launch_shift = 0.
      self.launch_shift_track = track
      self.launch_shift_gap = None
    if track == -1:
      return 0.
    v_ego = self.x0[1]
    if v_ego < TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP[0]:
      if self.launch_shift_gap is None or self.launch_shift_travel > 0.:
        # a new stop behind this car
        self.launch_shift_gap = float(lead.dRel)
        self.launch_shift_travel = 0.
      # the closest it comes while stopped: the truck can still roll a little once its speed reads 0
      self.launch_shift_gap = min(self.launch_shift_gap, float(lead.dRel))
      self.launch_shift = min(self.launch_shift_room(), self.launch_shift + TRAFFIC_MODE_LAUNCH_SHIFT_RATE * self.dt)
    elif self.launch_shift_gap is not None:
      # moving off: only comes down, spent by the truck's own travel and faded out with the plan's speed
      self.launch_shift_travel += v_ego * self.dt
      fade = float(np.interp(v_ego, TRAFFIC_MODE_LAUNCH_SHIFT_V_EGO_BP, [1., 0.]))
      self.launch_shift = min(self.launch_shift, max(self.launch_shift_room() - self.launch_shift_travel, 0.) * fade)
      if fade == 0.:
        self.launch_shift_gap = None
    return self.launch_shift

  def launch_shift_room(self):
    gap = self.launch_shift_gap
    return float(np.clip(STOP_DISTANCE - TRAFFIC_MODE_LAUNCH_SHIFT_MARGIN - gap, 0., TRAFFIC_MODE_LAUNCH_SHIFT)) * \
           float(np.interp(gap, TRAFFIC_MODE_LAUNCH_SHIFT_GAP_BP, [0., 1.]))

  def get_t_follow_offset_pct(self):
    # Traffic Mode ignores the offset (the stored setting is untouched and applies again once it is off), except where a
    # Rivian Following Distance with it is closer than Traffic Mode's curve (get_t_follow)
    offset_pct = clip_t_follow_offset_pct(self.t_follow_offset_pct)
    if self.traffic_mode:
      level_closer = self.follow_t_base is not None and \
                     self.follow_t_base * (1 + offset_pct / 100) < get_traffic_mode_T_FOLLOW(self.x0[1])
      return offset_pct if level_closer else 0.
    return offset_pct

  def get_t_follow(self, personality=log.LongitudinalPersonality.standard):
    if self.follow_t_base is not None:
      # Rivian Following Distance: the level's follow time with the offset. Traffic Mode keeps its curve, but never
      # follows further back than that (Closer and Close at a -10% offset are 0.90 and 0.99 s, under its 1.0 s)
      t_follow = self.follow_t_base * (1 + clip_t_follow_offset_pct(self.t_follow_offset_pct) / 100)
      return min(get_traffic_mode_T_FOLLOW(self.x0[1]), t_follow) if self.traffic_mode else t_follow
    if self.traffic_mode:
      # x0 holds the speed the planner just passed to set_cur_state()
      return get_traffic_mode_T_FOLLOW(self.x0[1])
    return get_T_FOLLOW(personality) * (1 + self.get_t_follow_offset_pct() / 100)

  def update(self, radarstate, personality=log.LongitudinalPersonality.standard):
    # The follow time also moves the danger zone (LEAD_DANGER_FACTOR * desired distance), as a personality change does
    t_follow = self.get_t_follow(personality)
    self.t_follow = t_follow

    lead_xv_0 = self.process_lead(radarstate.leadOne, self.a_lead_fast)
    lead_xv_1 = self.process_lead(radarstate.leadTwo)

    # To estimate a safe distance from a moving lead, we calculate how much stopping
    # distance that lead needs as a minimum. We can add that to the current distance
    # and then treat that as a stopped car/obstacle at this new distance.
    launch_shift = self.traffic_mode_launch_shift(radarstate.leadOne)
    lead_two = radarstate.leadTwo
    # leadTwo gets it only while it is leadOne's own radar track
    launch_shift_two = launch_shift if launch_shift > 0. and lead_two.present and lead_two.radar and \
                       lead_two.radarTrackId == self.launch_shift_track else 0.
    lead_0_obstacle = lead_xv_0[:,0] + get_stopped_equivalence_factor(lead_xv_0[:,1]) - self.approach_margin(0, radarstate.leadOne, lead_xv_0) + \
                      launch_shift
    lead_1_obstacle = lead_xv_1[:,0] + get_stopped_equivalence_factor(lead_xv_1[:,1]) - self.approach_margin(1, radarstate.leadTwo, lead_xv_1) + \
                      launch_shift_two

    x_obstacles = np.column_stack([lead_0_obstacle, lead_1_obstacle])
    self.source = MPC_SOURCES[np.argmin(x_obstacles[0])]

    self.yref[:,:] = 0.0
    for i in range(N):
      self.solver.set(i, "yref", self.yref[i])
    self.solver.set(N, "yref", self.yref[N][:COST_E_DIM])

    self.params[:,0] = ACCEL_MIN
    self.params[:,1] = ACCEL_MAX
    self.params[:,2] = np.min(x_obstacles, axis=1)
    self.params[:,3] = np.copy(self.a_prev)
    self.params[:,4] = t_follow
    self.params[:,5] = LEAD_DANGER_FACTOR

    self.run()
    # on the lead's own predicted path (with the fast estimate of its acceleration when that is in use), not its obstacle
    # (Traffic Mode's launch shift moves that)
    if (np.any(lead_xv_0[FCW_IDXS,0] - self.x_sol[FCW_IDXS,0] < CRASH_DISTANCE) and
            radarstate.leadOne.modelProb > 0.9):
      self.crash_cnt += 1
    else:
      self.crash_cnt = 0

  def run(self):
    for i in range(N+1):
      self.solver.set(i, 'p', self.params[i])
    self.solver.constraints_set(0, "lbx", self.x0)
    self.solver.constraints_set(0, "ubx", self.x0)

    self.solution_status = self.solver.solve()
    self.solve_time = float(self.solver.get_stats('time_tot')[0])

    for i in range(N+1):
      self.x_sol[i] = self.solver.get(i, 'x')
    for i in range(N):
      self.u_sol[i] = self.solver.get(i, 'u')

    self.v_solution = self.x_sol[:,1]
    self.a_solution = self.x_sol[:,2]
    self.j_solution = self.u_sol[:,0]

    self.a_prev = np.interp(T_IDXS + self.dt, T_IDXS, self.a_solution)

    t = time.monotonic()
    if self.solution_status != 0:
      if t > self.last_cloudlog_t + 5.0:
        self.last_cloudlog_t = t
        cloudlog.warning(f"Long mpc reset, solution_status: {self.solution_status}")
      self.reset()


if __name__ == "__main__":
  ocp = gen_long_ocp()
  AcadosOcpSolver.generate(ocp, json_file=JSON_FILE)
