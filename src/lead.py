"""C1-E4 part 2, block B: constant-velocity Kalman eyes, the lead of G-auto with a confidence gate and tracking until the
fingers close, and the smooth constant-speed motion of the object (pure numpy).

Spec: docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md (block B)."""

import hashlib
import math
from dataclasses import dataclass

import numpy as np

import gauto
import handoff as hf

HZ = 20
V_MIN = 0.005 / HZ  # m/step: below 0.5 cm/s no lead
GATE_M, GATE_TOL, GATE_SD = 6, 0.3, 0.5  # oracle window; relative tolerance; velocity sd <= GATE_SD |v|
LEAD_MAX = 0.05  # m
SPEEDS = (0.02, 0.04, 0.06)  # m/s, classes cycling over the episode number
R_MOVE = (0.15, 0.25)  # m, trigger radius of the motion
MAX_PATH = 0.15  # m


class CVKalman:
    """Per-axis constant-velocity Kalman filter on hf.Tracker's noise model (generator [seed, 3], WARM still frames,
    lag, sigma, dropped frames; both draws every step). kf(p_true) -> p_hat; .v velocity (m/step); .sd its sd."""

    WARM = hf.Tracker.WARM

    def __init__(self, seed, sigma, lag, q_drop, q_acc, r_floor=1e-4):
        self.rng = np.random.default_rng([seed, 3])
        self.sigma, self.lag, self.q = sigma, lag, q_drop
        self.R = max(sigma, r_floor) ** 2
        self.Qm = q_acc * np.array([[0.25, 0.5], [0.5, 1.0]])
        self.true, self.x, self.P = [], None, None  # x: (3, 2) position and velocity per axis; P shared by the axes

    @classmethod
    def oracle(cls, q_acc=1e-8):
        return cls(0, 0.0, 0, 0.0, q_acc)

    @property
    def v(self):
        return self.x[:, 1].copy()

    @property
    def sd(self):
        return math.sqrt(self.P[1, 1])

    def _step(self, src, may_drop=True):
        F = np.array([[1.0, 1.0], [0.0, 1.0]])
        self.x, self.P = self.x @ F.T, F @ self.P @ F.T + self.Qm
        n, drop = self.rng.normal(0.0, 1.0, 3), self.rng.random()
        if may_drop and drop < self.q:
            return
        k = self.P[:, 0] / (self.P[0, 0] + self.R)
        self.x = self.x + np.outer(src + self.sigma * n - self.x[:, 0], k)
        self.P = self.P - np.outer(k, self.P[0, :])

    def __call__(self, p):
        p = np.array(p, float)
        if self.x is None:
            self.x, self.P = np.stack([p, np.zeros(3)], axis=1), np.diag([self.R, 1e-6])
            for i in range(self.WARM):
                self._step(p, may_drop=i > 0)
        self.true.append(p)
        self._step(self.true[max(len(self.true) - 1 - self.lag, 0)])
        return self.x[:, 0].copy()


def lead_gate(kf, seen, m=GATE_M):
    """lambda: the speed is above V_MIN, the velocity confident (sd <= GATE_SD |v|) and the motion uniform: over the last
    m seen positions both halves moved by ~ v m / 2. A step moves only one half, so the gate stays shut."""
    v = kf.v[:2]
    sp = float(np.linalg.norm(v))
    if sp < V_MIN or len(seen) <= m or kf.sd > GATE_SD * sp:
        return False
    h = np.asarray(seen[-1 - m:])[:, :2]
    half, want = m // 2, v * (m // 2)
    tol = GATE_TOL * float(np.linalg.norm(want))
    return all(float(np.linalg.norm(d - want)) <= tol for d in (h[half] - h[0], h[m] - h[half]))


class LeadAgent(gauto.GAgent):
    """G-auto (law Z1) with the lead: U_lead(t) = U(t) + lambda(t) clip(v_xy(t) tau_lead, LEAD_MAX), z of the lead 0,
    lambda = lead_gate on the same eyes, and tracking until the fingers close: G's law runs tau_close steps from the
    first close command on (that step included), the gripper untouched. The velocity comes from the CV-Kalman eyes when
    they are the tracker (one filter step per env step), else from an oracle CVKalman on the seen positions."""

    def __init__(self, s, d, kbar, tau_lead, tau_close, tracker=None, gate_m=GATE_M, **kw):
        super().__init__("Gauto", s, d, kbar=kbar, tracker=tracker, **kw)
        self.tau_lead, self.tau_close, self.gate_m = tau_lead, tau_close, gate_m
        self.kf = tracker if isinstance(tracker, CVKalman) else CVKalman.oracle()
        self.t_close = None

    def observe(self, t, eef, obj, contact=False):
        super().observe(t, eef, obj, contact)
        if self.kf is not self.tracker:
            self.kf(self.p_hist[t])

    def lead(self, t):
        """The lead at the latest observed step t (the filter holds only its current state)."""
        seen = [self.p_hist[k] for k in range(max(0, t - self.gate_m), t + 1)]
        if not lead_gate(self.kf, seen, self.gate_m):
            return np.zeros(3)
        add = np.append(self.kf.v[:2] * self.tau_lead, 0.0)
        n = float(np.linalg.norm(add))
        return add if n <= LEAD_MAX else add * LEAD_MAX / n

    def unknown_shift(self, t):
        return super().unknown_shift(t) + self.lead(t)

    def act(self, t, a):
        # the close step itself goes through G-auto's act (G's law, grasp_started); the next tau_close - 1 through _g
        if self.t_close is not None and t < self.t_close + self.tau_close:
            return self._g(t, np.array(a, dtype=float))
        a = super().act(t, a)
        if self.t_close is None and self.grasp_started:
            self.t_close = t
        return a


@dataclass(frozen=True)
class MoveEpisode:
    suite: str
    task: int
    init: int
    kind: str  # "move"
    seed: int
    speed: float  # m/s, one of SPEEDS
    r: float  # trigger radius, m
    direction: float  # direction of the motion in the table plane, rad


def make_move_episode(suite, task, init):
    """As hf.make_episode with prefix "c1e4b" and kind "move": r ~ U[R_MOVE], direction ~ U[0, 2 pi); the speed class
    cycles over the episode number as the shift classes do."""
    seed = int.from_bytes(hashlib.sha256(f"c1e4b/{suite}/{task}/{init}/move".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    r, angle = float(rng.uniform(*R_MOVE)), float(rng.uniform(0.0, 2 * math.pi))
    cls = (50 * hf.ALL_TASKS.index((suite, task)) + init) % 3
    return MoveEpisode(suite, task, init, "move", seed, SPEEDS[cls], r, angle)


class MovePerturbation:
    """Fires at the first step with no close command yet and |EEF - object| < r; from then on the object is moved by
    speed / HZ along the direction every step (before that step's physics), up to tau_close steps from the first close
    command on (that step included) and while the path stays within MAX_PATH."""

    def __init__(self, ep, tau_close):
        self.ep, self.tau_close = ep, tau_close
        self.t_fire = self.p_pre = self.t_close = None
        self.path = 0.0
        self.step = ep.speed / HZ * np.array([math.cos(ep.direction), math.sin(ep.direction), 0.0])

    def dq(self, t, eef, obj, close_cmd):
        """Shift (m) to apply before this step's physics, or None."""
        if close_cmd and self.t_close is None:
            self.t_close = t
        if self.t_fire is None:
            if self.t_close is not None or hf.box_dist(eef, obj) >= self.ep.r:
                return None
            self.t_fire, self.p_pre = t, np.array(obj, dtype=float)
        n = self.ep.speed / HZ
        if (self.t_close is not None and t >= self.t_close + self.tau_close) or self.path + n > MAX_PATH + 1e-9:
            return None
        self.path += n
        return self.step.copy()
