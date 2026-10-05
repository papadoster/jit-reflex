"""C1-E4 part 2, block B: constant-velocity Kalman eyes, the lead of G-auto with a confidence gate and tracking until the
fingers close, and the object's constant-speed motion on its commanded track (pure numpy).

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
T_FIRE = (5, 15)  # policy steps from the episode start, uniform integer: the motion starts by time
MAX_PATH = 0.15  # m
STILL = 1e-4  # m per step: the fingers have stopped (the tau_close rule of scripts/c1e4b_offline.py)


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
        if self.x is None:  # prior: centred on the true position with variance R, velocity 0
            self.x, self.P = np.stack([p, np.zeros(3)], axis=1), np.diag([self.R, 1e-6])
            for i in range(self.WARM):
                self._step(p, may_drop=i > 0)
        self.true.append(p)
        self._step(self.true[max(len(self.true) - 1 - self.lag, 0)])
        return self.x[:, 0].copy()


def lead_gate(kf, seen, m=GATE_M):
    """lambda: the speed is above V_MIN, the velocity confident (sd <= GATE_SD |v|) and the motion uniform: over the last
    m seen positions both halves moved by ~ v m / 2. A step moves only one half, so the gate stays shut. m even: an odd
    m's second half spans m - m // 2 steps against v (m // 2) and shuts the gate on uniform motion."""
    assert m % 2 == 0, m
    v = kf.v[:2]
    sp = float(np.linalg.norm(v))
    if sp < V_MIN or len(seen) <= m or kf.sd > GATE_SD * sp:
        return False
    h = np.asarray(seen[-1 - m:])[:, :2]
    half, want = m // 2, v * (m // 2)
    tol = GATE_TOL * float(np.linalg.norm(want))
    return all(float(np.linalg.norm(d - want)) <= tol for d in (h[half] - h[0], h[m] - h[half]))


class LeadAgent(gauto.GAgent):
    """G-auto (law Z1) on a virtual object: the parent sees p_v(t) = p_seen(t) + L(t), the lead L = lambda(t)
    clip(v_xy(t) tau_lead, LEAD_MAX), z 0, lambda = lead_gate. So p_hist, the engagement, Delta_N, S_N and Z1's share
    all run on the virtual track and the lead is sent once, not again under every plan (Task 6 Mac check: added to U,
    with E restarting at each plan, a barely reacting brain got the full lead per plan). The gate and the velocity use
    the true seen positions (seen_hist): the oracle's, or the CV-Kalman eyes' output (one filter step per env step,
    run here; the parent's tracker is off); without eyes an oracle CVKalman gives the velocity. Approximation: the
    brain sees the real object, so Z1 credits it with kappa Delta of the virtual shift where it handles kappa
    Delta_real; the error is <= kappa |L|. Tracking until the fingers close: G's law runs tau_close steps from the
    first close command on (that step included), the gripper untouched; a plan arriving in that window gets Z1's kappa
    (HAgent resets kappa to 1 after the close), its kappa_log entry tagged post_close."""

    def __init__(self, s, d, kbar, tau_lead, tau_close, tracker=None, gate_m=GATE_M, **kw):
        assert gate_m >= 2 and gate_m % 2 == 0, gate_m
        assert tracker is None or isinstance(tracker, CVKalman), tracker
        super().__init__("Gauto", s, d, kbar=kbar, **kw)
        self.tau_lead, self.tau_close, self.gate_m = tau_lead, tau_close, gate_m
        self.kf = tracker or CVKalman.oracle()
        self.eyes, self.seen_hist, self.t_close = tracker, {}, None

    def observe(self, t, eef, obj, contact=False):
        self.t_now = t
        seen = self.kf(obj) if self.eyes else np.array(obj, float)
        if not self.eyes:
            self.kf(seen)
        self.seen_hist[t] = seen
        super().observe(t, eef, seen + self.lead(t), contact)

    def on_new_chunk(self):
        if self.t_close is None or self.t_now >= self.t_close + self.tau_close:
            return super().on_new_chunk()
        n = len(self.kappa_log)
        self.grasp_started = False  # only HAgent.on_new_chunk reads it here; t_e is set by observe alone
        try:
            super().on_new_chunk()
        finally:
            self.grasp_started = True
        for e in self.kappa_log[n:]:
            e["post_close"] = True

    def lead(self, t):
        """The lead at the latest observed step t (the filter holds only its current state). Its horizon is tau_lead
        until the close command, then the time left until the fingers close: tau_lead - (t - t_close), down to
        tau_lead - tau_close (the arm's lag)."""
        seen = [self.seen_hist[k] for k in range(max(0, t - self.gate_m), t + 1)]
        if not lead_gate(self.kf, seen, self.gate_m):
            return np.zeros(3)
        h = self.tau_lead - (0 if self.t_close is None else min(t - self.t_close, self.tau_close))
        add = np.append(self.kf.v[:2] * h, 0.0)
        n = float(np.linalg.norm(add))
        return add if n <= LEAD_MAX else add * LEAD_MAX / n

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
    t_fire: int  # policy step the motion starts at, from the episode start
    direction: float  # direction of the motion in the table plane, rad


def make_move_episode(suite, task, init):
    """As hf.make_episode with prefix "c1e4b" and kind "move": t_fire ~ U{T_FIRE}, direction ~ U[0, 2 pi); the speed
    class cycles over the episode number as the shift classes do. By time, not by distance: on SmolVLA spatial the
    15-25 cm radius came 4-5 steps before the close (Task 6 Mac check)."""
    seed = int.from_bytes(hashlib.sha256(f"c1e4b/{suite}/{task}/{init}/move".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    t_fire, angle = int(rng.integers(T_FIRE[0], T_FIRE[1] + 1)), float(rng.uniform(0.0, 2 * math.pi))
    cls = (50 * hf.ALL_TASKS.index((suite, task)) + init) % 3
    return MoveEpisode(suite, task, init, "move", seed, SPEEDS[cls], t_fire, angle)


def held(contact, w, t_close):
    """The fingers hold the object at the latest step t = len(w) - 1: robot-object contact, and the fingers have
    stopped closing: their last move since the first close command t_close (|w(j) - w(j-1)| >= STILL, t_close < j <= t)
    closed them, and now they are still (|w(t) - w(t-1)| < STILL), the tau_close rule (fingers move, then still)
    online. w: finger widths up to t, m. A touch with the fingers still open is no hold (Task 6 Mac check: 52 or 31 mm
    open), nor fingers that reopened under a flickering gripper command and paused (the re-check: 79 mm)."""
    if not contact or t_close is None:
        return False
    d = np.diff(np.asarray(w[t_close:], float))
    moves = d[np.abs(d) >= STILL]
    return bool(len(moves) and moves[-1] < 0 and abs(d[-1]) < STILL)


class MovePerturbation:
    """Fires at the episode's t_fire if no close command came yet (else never); from then on the object is put on its
    commanded track p_pre + speed / HZ (k + 1) along the direction (k steps since the fire) every step, before that
    step's physics, the runner also setting its velocity to speed along the direction: the drift of the physics step
    is corrected at the next one, so the mean speed is the class speed. It stops for good at the earliest of: tau_close
    steps from the first close command on (that step included), "tau"; the first step from that close on at which the
    fingers hold it (moving a held object would fight the grasp), "held"; the path reaching MAX_PATH, "path".
    .stop: that reason, None while moving or never fired."""

    def __init__(self, ep, tau_close):
        self.ep, self.tau_close = ep, tau_close
        self.t_fire = self.p_pre = self.t_close = self.stop = None
        self.path = 0.0
        self.step = ep.speed / HZ * np.array([math.cos(ep.direction), math.sin(ep.direction), 0.0])

    def dq(self, t, obj, close_cmd, held=False):
        """Shift (m) to apply before this step's physics, or None. obj: the object now; held: held(contact, widths,
        t_close) at this step, contact with the fingers moved since the first close command and now stopped."""
        if close_cmd and self.t_close is None:
            self.t_close = t
        if self.t_fire is None:
            if self.t_close is not None or t < self.ep.t_fire:
                return None
            self.t_fire, self.p_pre = t, np.array(obj, dtype=float)
        n = self.ep.speed / HZ
        if self.stop is None:
            if self.t_close is not None and t >= self.t_close + self.tau_close:
                self.stop = "tau"
            elif held and self.t_close is not None:
                self.stop = "held"
            elif self.path + n > MAX_PATH + 1e-9:
                self.stop = "path"
        if self.stop is not None:
            return None
        self.path += n
        return self.p_pre + self.step * (t - self.t_fire + 1) - np.asarray(obj, float)
