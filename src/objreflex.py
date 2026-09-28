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
        assert s >= 1 and d >= 0 and s + d <= H, (s, d)
        self.s, self.d = s, d
        self.next_call, self.in_flight = 0, None  # in_flight: (t_obs, arrival, chunk)
        self.chunk, self.t_obs = None, None
        self.n_sched = self.n_trig = 0

    def wants_call(self, t, trigger):
        return self.in_flight is None and (t >= self.next_call or trigger)

    def issue(self, t, chunk):
        """Issue a call observed at t; it counts as scheduled if t reached next_call, else as triggered."""
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
