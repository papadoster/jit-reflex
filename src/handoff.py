"""C1-E3: hand-off of the object reflex between plans, a synthetic seeing brain, a push filter and noisy eyes
(pure numpy, no torch/LIBERO).

Spec: docs/superpowers/specs/2026-09-29-c1-e3-handoff-reflex-design.md. Builds on src/objreflex.py (C1-E2): the same
units (metres in the LIBERO world frame; raw delta-EEF actions, xyz = action[:3], gripper = the last column, > 0
closes), the Schedule and G's command law. scripts/c1e3_run.py feeds one episode's observations to one HAgent per step.
"""

import hashlib
import math

import numpy as np

import objreflex as orx
from objreflex import G_POS

TASKS10 = [("libero_10", i) for i in (0, 1, 3, 5, 7, 8, 9)]  # spec §4 and journal: the target is grasped first
ALL_TASKS = orx.TASKS + TASKS10
R_RANGE = {"step": (0.08, 0.20), "control": (0.08, 0.20), "close": (0.03, 0.08)}  # trigger radius r, m (spec §4)
KS = (10, 20, 30)  # fallback horizon candidates (spec §7)
# noisy tracker (spec §4, §6 rows 20-23, 26): (sigma m, lag steps, dropout probability); sweeps vary one factor alone
NOISE = {"real": (0.01, 3, 0.10), "s0.5": (0.005, 0, 0.0), "s1": (0.01, 0, 0.0), "s2": (0.02, 0, 0.0),
         "L3": (0.0, 3, 0.0), "L6": (0.0, 6, 0.0), "d10": (0.0, 0, 0.10)}
SWEEP = ("s0.5", "s1", "s2", "L3", "L6", "d10")
SIGMA_SWEEP = ("s0.5", "s1", "s2")
LIFT = 0.03  # lifted: >= 3 cm above the object's height at the close command (spec §5, G-R + return)


def make_episode(suite, task, init, kind):
    """spec §4: everything from the episode seed, identical for all methods and cells. kind: step; close (the shift
    fires within r ~ U[3, 8] cm); control (no shift: the pseudo-shift moment and its direction are only recorded)."""
    assert kind in R_RANGE, kind
    seed = int.from_bytes(hashlib.sha256(f"c1e3/{suite}/{task}/{init}/{kind}".encode()).digest()[:4], "little")
    rng = np.random.default_rng(seed)
    r, angle = float(rng.uniform(*R_RANGE[kind])), float(rng.uniform(0.0, 2 * math.pi))
    if kind == "control":
        cls, mag = -1, 0.0
    else:
        cls = (50 * ALL_TASKS.index((suite, task)) + init) % 3  # classes cycle over the episode number
        mag = float(rng.uniform(*orx.MAG_CLASSES[cls]))
    return orx.Episode(suite, task, init, kind, seed, r, mag, angle, cls)


class Perturbation:
    """spec §4: fires once, at the first step with no close command yet and |EEF - object| < r. A step shift (kinds step
    and close) is applied before that step's physics, so the observation at t_fire is still pre-shift. In control only
    t_fire and p_pre are recorded (pseudo-shift)."""

    def __init__(self, ep):
        self.ep, self.t_fire, self.p_pre = ep, None, None

    def dq(self, t, eef, obj, grasp_started):
        """Shift (m) to apply before this step's physics, or None."""
        if self.t_fire is None:
            if grasp_started or np.linalg.norm(np.asarray(eef, float) - obj) >= self.ep.r:
                return None
            self.t_fire, self.p_pre = t, np.array(obj, dtype=float)
        return self.ep.delta if t == self.t_fire and self.ep.kind != "control" else None


def plan_path(x_obs, chunk):
    """w_P(k) = x(t_obs) + G_POS sum_{j<k} clip(P[j, :3]), k = 0..len(P): where the plan means the arm to be."""
    steps = G_POS * np.clip(np.asarray(chunk, float)[:, :3], -1, 1)
    return np.vstack([np.zeros(3), np.cumsum(steps, 0)]) + np.asarray(x_obs, float)


def close_index(chunk):
    """Index of the plan's first close command (gripper = last column > 0), or None."""
    c = np.flatnonzero(np.asarray(chunk, float)[:, -1] > 0)
    return int(c[0]) if len(c) else None


def target_diff(new, ref, K):
    """g_N - g_R (spec §5): the plans' grasp points (their paths at the first close command); if either plan has no
    close command, their paths at the common time t_obs_N + K (the reference's index clamped to its end).
    new, ref: (t_obs, x_obs, chunk). Returns (difference, fallback used)."""
    (tn, xn, cn), (tr, xr, cr) = new, ref
    wn, wr = plan_path(xn, cn), plan_path(xr, cr)
    kn, kr = close_index(cn), close_index(cr)
    if kn is not None and kr is not None:
        return wn[kn] - wr[kr], False
    return wn[min(K, len(cn))] - wr[min(tn + K - tr, len(cr))], True


def kappa_hat(m, nd, tau):
    """kappa_hat = clip(m / |Delta|, 0, 1), m := 0 inside the dead band |m| < tau (spec §5); 0 when |Delta| ~ 0."""
    if nd < 1e-9 or abs(m) < tau:
        return 0.0
    return float(np.clip(m / nd, 0.0, 1.0))


def early(t_fire, s, d, t_ramp=5):
    """spec §8 early chunks: the first plan observed after the shift (t_obs >= t_fire + 1) on the fixed schedule (calls
    at multiples of s) arrives by t_fire + t_ramp. None without a shift."""
    if t_fire is None:
        return None
    return -(-(t_fire + 1) // s) * s + d <= t_fire + t_ramp


class Tracker:
    """Noisy eyes on the object (spec §4): measurement p(t - lag) + N(0, sigma^2 I), none with probability q_drop (the
    estimate holds), smoothed by EMA beta (1 = none). Warm: it has run WARM steps on the still object before the
    episode (spec journal), so the first plan's reference is smoothed too. Both random draws happen every step, so
    settings with the same seed share their noise stream."""

    WARM = 20

    def __init__(self, seed, sigma, lag, q_drop, beta=1.0):
        self.rng = np.random.default_rng([seed, 3])
        self.sigma, self.lag, self.q, self.beta = sigma, lag, q_drop, beta
        self.true, self.est = [], None

    def __call__(self, p):
        p = np.array(p, float)
        if self.est is None:
            for _ in range(self.WARM):
                self._update(p)
        self.true.append(p)
        return self._update(self.true[max(len(self.true) - 1 - self.lag, 0)])

    def _update(self, src):
        n, drop = self.rng.normal(0.0, 1.0, 3), self.rng.random()
        if self.est is not None and drop < self.q:
            return self.est.copy()
        meas = src + self.sigma * n
        self.est = meas if self.est is None else self.beta * meas + (1 - self.beta) * self.est
        return self.est.copy()


def surr_push(p, p0, eef, eef0, r_c, direction):
    """Surrogate push flag for the step p0 -> p (spec §4): the arm is within r_c of the object's last position and, with
    direction, the object moved the way the arm did."""
    return bool(np.linalg.norm(eef - p0) < r_c and (not direction or (p - p0) @ (eef - eef0) > 0))


class PushFilter:
    """Contact filter (spec §4): object increments on steps marked as pushes add up in P; the reflex sees p - P.
    mode "contact": MuJoCo robot-object contact at t - 1 or t (the increment p(t) - p(t-1) comes from the physics
    between the two observations); "surr": surr_push. flag: the last step's mark."""

    def __init__(self, mode, r_c=0.05, direction=False):
        assert mode in ("contact", "surr"), mode
        self.mode, self.r_c, self.dir = mode, r_c, direction
        self.prev, self.P, self.flag = None, np.zeros(3), False

    def __call__(self, p, eef, contact):
        p, eef = np.array(p, float), np.array(eef, float)
        self.flag = False
        if self.prev is not None:
            p0, e0, c0 = self.prev
            self.flag = (c0 or bool(contact)) if self.mode == "contact" else surr_push(p, p0, eef, e0, self.r_c, self.dir)
            if self.flag:
                self.P = self.P + (p - p0)
        self.prev = (p, eef, bool(contact))
        return p - self.P


def ramp_step(base, rem, cap):
    """One step of G's command form (alpha_topup, the return): send rem (m), at most cap, on top of the clipped plan
    action base, clipped to +-1. Returns the new action xyz and what is still to send."""
    n = np.linalg.norm(rem)
    new = np.clip(base + (rem if n <= cap else rem * cap / n) / G_POS, -1, 1)
    return new, rem - G_POS * (new - base)


def alpha_topup(chunk, add, start, t_ramp):
    """G's command law inside a chunk (spec §4 SmolVLA-alpha): add the displacement `add` (m) from index start, at most
    |add| / t_ramp per step, each action clipped to +-1, up to the chunk's first close command."""
    ch, rem = np.array(chunk, float), np.array(add, float)
    cap, end = np.linalg.norm(rem) / t_ramp, close_index(chunk)
    for j in range(start, len(ch) if end is None else end):
        if np.linalg.norm(rem) < 1e-12:
            break
        ch[j, :3], rem = ramp_step(np.clip(ch[j, :3], -1, 1), rem, cap)
    return ch


class SeeingBrain:
    """Synthetic seeing brain SmolVLA-alpha (spec §4, oracle): a chunk observed after the shift (t_obs >= t_fire + 1)
    and before the episode's first close command gets A = alpha (D - S). D = p(t_obs) - p_pre, the true shift; S, the
    extra displacement commanded since the shift against the raw brain (reflex and earlier top-ups), is summed by
    executed(). With alpha = 0 it only keeps the raw chunks and S (kappa_model in the report)."""

    def __init__(self, alpha, t_ramp=5):
        self.alpha, self.t_ramp = alpha, t_ramp
        self.S, self.raw, self.s_at = np.zeros(3), {}, {}  # raw chunks and S at each call, by t_obs

    def chunk(self, t, raw, pert, p_true, start, grasp_started):
        """The chunk the brain answers to a call observed at t; start: its first executed index (0 or d). Always a
        copy: an in-place edit of the executed chunk must not reach the raw chunk that S is counted against."""
        self.raw[t], self.s_at[t] = np.asarray(raw, float), self.S.copy()
        if (self.alpha == 0 or pert.ep.kind == "control" or pert.t_fire is None or t < pert.t_fire + 1
                or grasp_started):
            return self.raw[t].copy()
        return alpha_topup(raw, self.alpha * (np.asarray(p_true, float) - pert.p_pre - self.S), start, self.t_ramp)

    def executed(self, t, t_obs, a, pert):
        """Count the executed action a at step t (plan observed at t_obs) into S, from the shift step on."""
        if pert.t_fire is not None and t >= pert.t_fire:
            raw = self.raw[t_obs][t - t_obs]
            self.S = self.S + G_POS * (np.clip(a[:3], -1, 1) - np.clip(raw[:3], -1, 1))


class HAgent(orx.Agent):
    """C1-E3 controller (spec §5): the kappa family on top of C1-E2's Agent.

    Engagement t_e: the first step with |p(t) - p(t_obs)| > eps against the plan executing at observe time. R: the
    latest plan observed before t_e (the one in flight at d > 0, else the executing one); p_R = p(t_obs_R). For a plan
    N observed at or after t_e: Delta_N = p(t_obs_N) - p_R and U(t) = (1 - kappa) Delta_N + (p(t) - p(t_obs_N));
    for earlier plans kappa = 1 (G's U). G: kappa = 1; Gkeep: kappa = 0 (U = p(t) - p_R); GR: kappa_hat from the
    plans' grasp points. The T variants re-query when the raw shift since the executing plan's observation exceeds
    T_THR. GRret: GR, and after the lift it sends back the reflex's accumulated extra (_return).
    p is what the reflex sees: the oracle or a Tracker, minus pushes when a PushFilter is on. t_e and R are recorded
    for every method (the runner logs t_engage for all arms); only the kappa family acts on them."""

    FAMILY = {"G": 1.0, "GT": 1.0, "Gkeep": 0.0, "GkeepT": 0.0, "GR": None, "GRT": None, "GRret": None}
    METHODS = ("none", "T0", "PPC", *FAMILY)

    def __init__(self, method, s, d, t_ramp=5, k_p=1.0, eps=orx.EPS, tau_k=0.01, K=20, tracker=None, push=None,
                 ppc_v_min=orx.PPC_V_MIN):
        super().__init__(method, s, d, t_ramp=t_ramp, k_p=k_p)
        self.eps, self.ppc_v_min, self.tau_k, self.K = eps, ppc_v_min, tau_k, K
        self.tracker, self.push = tracker, push
        self.x_hist = {}
        self.t_e = self.ref = self.p_ref = None
        self.kappa, self.delta_n, self.kappa_log = 1.0, np.zeros(3), []
        self.z_close = self.t_lift = self.ret_left = self.ret_cap = None  # GRret
        self.released = False

    def observe(self, t, eef, obj, contact=False):
        p = self.tracker(obj) if self.tracker else np.array(obj, float)
        if self.push:
            p = self.push(p, eef, contact)
        self.x_hist[t] = np.array(eef, float)
        super().observe(t, eef, p)
        sc = self.sched
        if (self.t_e is None and sc.chunk is not None and not self.grasp_started
                and np.linalg.norm(p - self.p_hist[sc.t_obs]) > self.eps):
            self.t_e = t
            t_r, _, ch = sc.in_flight or (sc.t_obs, None, sc.chunk)
            self.ref, self.p_ref = (t_r, self.x_hist[t_r], ch), self.p_hist[t_r]

    def on_new_chunk(self):
        super().on_new_chunk()
        t_obs = self.sched.t_obs
        self.kappa = 1.0
        if self.method not in self.FAMILY or self.t_e is None or t_obs < self.t_e or self.grasp_started:
            return
        delta = self.p_hist[t_obs] - self.p_ref
        nd = float(np.linalg.norm(delta))
        u = delta / nd if nd > 1e-9 else np.zeros(3)
        new = (t_obs, self.x_hist[t_obs], self.sched.chunk)
        diffs = {k: target_diff(new, self.ref, k) for k in sorted({*KS, self.K})}
        m = {k: float(v[0] @ u) for k, v in diffs.items()}
        k_hat, fixed = kappa_hat(m[self.K], nd, self.tau_k), self.FAMILY[self.method]
        self.kappa, self.delta_n = (k_hat if fixed is None else fixed), delta
        self.kappa_log.append({"t": t_obs, "fb": diffs[self.K][1], "m": {str(k): round(v, 6) for k, v in m.items()},
                               "nd": round(nd, 6), "d": np.round(delta, 6).tolist(), "k_hat": round(k_hat, 4),
                               "k": round(self.kappa, 4)})

    def unknown_shift(self, t):
        return (1 - self.kappa) * self.delta_n + (self.p_hist[t] - self.p_hist[self.sched.t_obs])

    def trigger(self, t):
        if self.method in ("GT", "GkeepT", "GRT"):
            return (not self.grasp_started and self.sched.chunk is not None
                    and np.linalg.norm(self.p_hist[t] - self.p_hist[self.sched.t_obs]) > orx.T_THR)
        return super().trigger(t)

    def act(self, t, a):
        a = np.array(a, dtype=float)
        if not self.grasp_started:
            if self.method in self.FAMILY:
                a = self._g(t, a)
            elif self.method == "PPC":
                a = self._ppc(t, a)
            if a[6] > 0:
                self.grasp_started, self.z_close = True, self.p_hist[t][2]
        elif self.method == "GRret":
            a = self._return(t, a)
        return a

    def _return(self, t, a):
        """G-R + return (spec §5): once the seen object is LIFT above its height at the close command, send back the
        reflex's accumulated extra C in G's command form (<= |C| / T_ramp per step, clip +-1), until the first open
        command after the close."""
        if self.released or a[6] <= 0:
            self.released = True
            return a
        if self.ret_left is None:
            if self.p_hist[t][2] - self.z_close < LIFT:
                return a
            self.t_lift, self.ret_left, self.ret_cap = t, -self.c.copy(), np.linalg.norm(self.c) / self.t_ramp
        if np.linalg.norm(self.ret_left) < 1e-12:
            return a
        a[:3], self.ret_left = ramp_step(np.clip(a[:3], -1, 1), self.ret_left, self.ret_cap)
        return a


def parse_arm(label):
    """A grid arm label -> (method, filter, noise): "GR+surr" -> ("GR", "surr", ""), "PPC@real" -> ("PPC", "", "real").
    GRret is oracle only (spec rows 27, 28): a push filter freezes the seen object while it is carried, so the lift
    never fires, and noise crosses LIFT on a still object."""
    base, _, noise = label.partition("@")
    method, _, filt = base.partition("+")
    assert method in HAgent.METHODS and filt in ("", "surr", "contact") and (not noise or noise in NOISE), label
    assert method != "GRret" or not (filt or noise), f"{label}: the return runs on the oracle only"
    return method, filt, noise
