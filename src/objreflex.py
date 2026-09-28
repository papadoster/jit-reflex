"""C1-E2: an object-anchored reflex between calls of a frozen chunking VLA (pure numpy, no torch/LIBERO).

Spec: docs/superpowers/specs/2026-09-28-c1-e2-object-reflex-design.md. Positions are metres in the LIBERO
world frame; actions are raw LIBERO delta-EEF commands in [-1, 1], xyz = action[:3], gripper = action[6]
(> 0 closes). scripts/c1e2_run.py feeds one Episode's observations to one Agent per step.
"""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

G_POS = 0.011  # m of EEF motion per action unit per step (integrator, docs/c1/scouting.md §3)
EPS = 0.005  # G dead band, m
T0_THR = 0.01  # T0 re-queries at |U| >= 1 cm
T_THR = 0.05  # the T switch re-queries at |U| > 5 cm
R_GRIP = 0.04  # PPC reset radius (gripper half-span), m
PPC_K = 2  # PPC minimum execution steps (paper default)
PPC_V_MIN = 0.001  # object speed counted by PPC, m/step
H = 50  # SmolVLA chunk length
SMOOTH_STEPS = 20

# cell: (s = call period, d = inference delay)
CELLS = {"A": (10, 0), "B": (25, 0), "C": (50, 0), "D": (10, 10), "E": (40, 10), "F": (10, 20), "Gp": (30, 20)}
TASKS = ([("libero_spatial", i) for i in range(10)] + [("libero_object", i) for i in range(10)]
         + [("libero_goal", i) for i in (1, 2, 4, 6, 8, 9)])
MAG_CLASSES = ((0.01, 0.02), (0.03, 0.04), (0.05, 0.06))


@dataclass(frozen=True)
class Episode:
    suite: str
    task: int
    init: int
    kind: str  # "step", "smooth" or "control" (spec §4)
    seed: int
    r: float  # trigger radius, m
    mag: float  # shift length, m (0 for control)
    angle: float  # shift direction in the table plane, rad
    mag_class: int  # 0, 1, 2 for 1-2 / 3-4 / 5-6 cm; -1 for smooth and control

    @property
    def delta(self):
        return self.mag * np.array([math.cos(self.angle), math.sin(self.angle), 0.0])


def make_episode(suite, task, init, kind):
    """Everything drawn from the episode seed; identical for all methods and cells (spec §4)."""
    assert kind in ("step", "smooth", "control"), kind
    seed = int.from_bytes(hashlib.sha256(f"c1e2/{suite}/{task}/{init}/{kind}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    r, angle = float(rng.uniform(0.08, 0.20)), float(rng.uniform(0.0, 2 * math.pi))
    if kind == "step":
        cls = (50 * TASKS.index((suite, task)) + init) % 3  # classes cycle over the episode number
        mag = float(rng.uniform(*MAG_CLASSES[cls]))
    elif kind == "smooth":
        cls, mag = -1, float(rng.uniform(0.03, 0.06))
    else:
        cls, mag = -1, 0.0
    return Episode(suite, task, init, kind, seed, r, mag, angle, cls)


class Perturbation:
    """When and how the object is moved (spec §4): once, at the first step with no close command yet and
    |EEF - object| < r; a step shift at once, a smooth one over SMOOTH_STEPS steps,
    stopped by the first close command."""

    def __init__(self, ep):
        self.ep, self.t_fire, self.p_pre = ep, None, None

    def dq(self, t, eef, obj, grasp_started):
        """Shift (m) to apply before this step's physics, or None."""
        if self.ep.kind == "control":
            return None
        if self.t_fire is None:
            if grasp_started or np.linalg.norm(eef - obj) >= self.ep.r:
                return None
            self.t_fire, self.p_pre = t, np.array(obj, dtype=float)
        k = t - self.t_fire
        if self.ep.kind == "step":
            return self.ep.delta if k == 0 else None
        return self.ep.delta / SMOOTH_STEPS if k < SMOOTH_STEPS and not grasp_started else None


class Schedule:
    """Brain calls every s steps; the answer to a call observed at t arrives at t + d (spec §4). The first call
    of an episode arrives at once. A triggered call resets the schedule; no call while one is in flight."""

    def __init__(self, s, d):
        # with one call in flight, a chunk is used up to index max(s, d) + d - 1
        assert s >= 1 and d >= 0 and max(s, d) + d <= H, (s, d)
        self.s, self.d = s, d
        self.next_call, self.in_flight = 0, None  # in_flight: (t_obs, arrival, chunk)
        self.chunk, self.t_obs = None, None
        self.n_sched = self.n_trig = 0

    def wants_call(self, t, trigger):
        return self.in_flight is None and (t >= self.next_call or trigger)

    def issue(self, t, chunk):
        """Issue a call observed at t; it counts as scheduled if t reached next_call, else as triggered."""
        assert self.in_flight is None
        scheduled = t >= self.next_call
        self.n_sched += scheduled
        self.n_trig += not scheduled
        self.in_flight = (t, t + (0 if self.chunk is None else self.d), chunk)
        self.next_call = t + self.s

    def arrive(self, t):
        """Install the answer if it arrives at t; True when a new chunk starts being used."""
        if self.in_flight is None or self.in_flight[1] != t:
            return False
        self.t_obs, _, self.chunk = self.in_flight
        self.in_flight = None
        return True

    def action(self, t):
        return np.array(self.chunk[t - self.t_obs], dtype=float)


def ppc_profile(k):
    """PPC path weights (1 - F_{2j+1} / F_{2k+1}), j = 0..k-1 (Fibonacci numbers, F_1 = F_2 = 1)."""
    f = [0, 1]
    while len(f) <= 2 * k + 1:
        f.append(f[-1] + f[-2])
    return [1 - f[2 * j + 1] / f[2 * k + 1] for j in range(k)]


def ppc_pace(v, dp, s):
    """PPC pace: alpha* = 1 + |v| cos(theta) / |dp| and the execution horizon max(K, min(ceil(s / alpha), s))."""
    ndp = np.linalg.norm(dp)
    if ndp < 1e-9:
        return 1.0, s
    alpha = 1 + float(v @ (dp / ndp)) / ndp  # |v| cos(theta) / |dp|
    k_exec = s if alpha <= 1 else max(PPC_K, min(math.ceil(s / alpha), s))
    return alpha, k_exec


class Agent:
    """One episode's controller: the call schedule plus the method's reflex (spec §5, G in command form, Task 0).

    U(t): shift of the object that the executing plan does not know: p(t) - p(t_obs), or p(t) - p_pre for
    Gpost once the perturbation fired. E(t) = C(t) - C(t_obs): extra displacement already commanded since that
    plan's observation. G adds e = K_p (U - E), |e| <= |U| / T_ramp, converted to action units by G_POS.
    """

    METHODS = ("none", "T0", "G", "GT", "PPC", "Gpost", "GJ")

    def __init__(self, method, s, d, t_ramp=5, k_p=1.0):
        assert method in self.METHODS, method
        self.method, self.sched = method, Schedule(s, d)
        self.t_ramp, self.k_p = t_ramp, k_p
        self.p_hist, self.c_hist = {}, {}  # object position and cumulative extra (m) by step
        self.c = np.zeros(3)
        self.p_pre = None  # set by the runner when the perturbation fires (Gpost anchor)
        self.grasp_started = False
        # G engaged at least once (|U| > EPS), even if clipping sent nothing (report: false triggers in control)
        self.g_on = False
        self.ppc_next, self.ppc_off = math.inf, {}
        self.j_corr, self.o_raw = None, {}  # GJ: callable(t, k) -> J_k (o_t - o_hat_k); raw states by step

    def on_new_chunk(self):
        """Called when a new chunk starts being used (G's anchor moves with sched.t_obs by itself)."""
        self.ppc_next, self.ppc_off = math.inf, {}  # PPC: the next chunk starts unbiased (delta_K = 0)

    def observe(self, t, eef, obj):
        self.x, self.p_hist[t], self.c_hist[t] = np.array(eef, float), np.array(obj, float), self.c.copy()

    def unknown_shift(self, t):
        if self.method == "Gpost" and self.p_pre is not None:
            return self.p_hist[t] - self.p_pre
        return self.p_hist[t] - self.p_hist[self.sched.t_obs]

    def trigger(self, t):
        """An unscheduled brain call at step t (T0, the T switch of GT, PPC's shorter horizon)."""
        if self.grasp_started or self.sched.chunk is None:
            return False
        if self.method == "PPC":
            return t >= self.ppc_next
        u = np.linalg.norm(self.unknown_shift(t))
        if self.method == "T0":
            return u >= T0_THR
        if self.method == "GT":
            return u > T_THR
        return False

    def act(self, t, a):
        a = np.array(a, dtype=float)
        if not self.grasp_started:
            if self.method == "GJ" and self.j_corr is not None:
                a = np.clip(a + np.clip(self.j_corr(t, t - self.sched.t_obs), -1, 1), -1, 1)
            if self.method in ("G", "GT", "Gpost", "GJ"):
                a = self._g(t, a)
            if self.method == "PPC":
                a = self._ppc(t, a)
        if a[6] > 0:
            self.grasp_started = True
        return a

    def _g(self, t, a):
        u = self.unknown_shift(t)
        if np.linalg.norm(u) <= EPS:
            return a
        self.g_on = True
        e = self.k_p * (u - (self.c - self.c_hist[self.sched.t_obs]))
        cap = np.linalg.norm(u) / self.t_ramp
        if np.linalg.norm(e) > cap:
            e *= cap / np.linalg.norm(e)
        base = np.clip(a[:3], -1, 1)  # what the arm gets from the plan alone
        new = np.clip(base + e / G_POS, -1, 1)
        self.c = self.c + G_POS * (new - base)
        a[:3] = new
        return a

    def _ppc(self, t, a):
        """Our PPC (paper §3, arXiv 2605.11459): velocity v(t) = p(t) - p(t-1) from the oracle tracker; pace
        shortens the executing chunk (an earlier call), path adds (1 - F_{2k+1}/F_{2K+1}) v_perp over the next K
        steps. Our interpretation: dp is the step of the clipped plan action (as in G); the offsets are re-planned
        at each velocity measurement (the latest one wins, no summing) and cleared at a new chunk. Reset (no
        correction) within R_GRIP of the object. The 2-EMA latch stabiliser is not implemented."""
        p, x = self.p_hist[t], self.x
        if np.linalg.norm(x - p) < R_GRIP:
            self.ppc_next, self.ppc_off = math.inf, {}
            return a
        if t - 1 in self.p_hist:
            v = p - self.p_hist[t - 1]
            if np.linalg.norm(v) > PPC_V_MIN:
                dp = G_POS * np.clip(a[:3], -1, 1)
                _, k_exec = ppc_pace(v, dp, self.sched.s)
                self.ppc_next = min(self.ppc_next, self.sched.t_obs + k_exec)
                ndp = np.linalg.norm(dp)
                v_perp = v - (v @ dp) / ndp**2 * dp if ndp > 1e-9 else v
                self.ppc_off = {t + j: w * v_perp for j, w in enumerate(ppc_profile(PPC_K))}
        off = self.ppc_off.pop(t, None)
        if off is not None:
            a[:3] = np.clip(a[:3] + off / G_POS, -1, 1)
        return a
