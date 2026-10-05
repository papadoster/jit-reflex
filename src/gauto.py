"""C1-E4: the call conveyor, law Z1 of the self-calibrating reflex G-auto, its shadow sample and the GLR + jump-Kalman
eyes (pure numpy, no torch/LIBERO).

Spec: docs/superpowers/specs/2026-10-03-c1-e4-design.md. Builds on src/handoff.py (C1-E3) and src/objreflex.py (C1-E2):
the same units (metres in the LIBERO world frame; raw delta-EEF actions, xyz = action[:3], gripper = action[6], > 0
closes) and G's command law. scripts/c1e4_run.py feeds one episode's observations to one GAgent per step.
"""

import numpy as np

import handoff as hf
import objreflex as orx

# spec §4.2: cell -> (s = call period, d = answer delay), steps; s + d <= H (§4.3)
CELLS = {"A": (10, 0), "A10": (10, 10), "A20": (10, 20), "A40": (10, 40), "C": (50, 0)}
# spec §4.1: brain -> (Hugging Face repo, pinned revision); SmolVLA: the snapshot of the C1-E2 / C1-E3 pods
BRAINS = {"pi05": ("lerobot/pi05_libero_finetuned_v044", "8e174154ef5f6c60a8da12ae99c303d8963138c1"),
          "smolvla": ("HuggingFaceVLA/smolvla_libero", "6721902bc4d61e50a3bfdb11dfb4cb626f05d102")}
KAPPA0 = {"pi05": 0.355, "smolvla": 0.038}  # spec §5.2: C1-E4a's median headroom, frozen before data (G-kappa0)
K = 20  # spec §5.2-5.3: the fallback horizon of G-R's kappa_hat and of the shadow sample
ND_MIN = 1e-3  # spec §5.3: s_N := 0 when |Delta_N| < 1 mm
NOISE = hf.NOISE["real"]  # spec §4.5: the realistic point, sigma 1 cm, lag 3 steps, 10% dropped frames
EMA_BETA, EMA_EPS = 0.2, 0.02  # spec §4.5: the old pipeline (report row 12)


class Conveyor(orx.Schedule):
    """spec §4.3: a scheduled call every s steps, even while earlier answers are in flight; the answer to a call
    observed at t arrives at t + d (the first call of an episode at once) and is executed by step index (t' - t) until
    the next answer arrives. At most one unscheduled (triggered) call in flight; any call moves the next scheduled one
    to its own step + s. When the executing chunk is used up the arm holds: zero position and rotation deltas, the
    gripper keeps the chunk's last command (the reflex still acts on top). With d = 0, or with s > d and no triggers,
    this is orx.Schedule's timing."""

    def __init__(self, s, d):
        assert s >= 1 and d >= 0 and s + d <= orx.H, (s, d)
        self.s, self.d = s, d
        self.next_call, self.queue = 0, []  # queue: [t_obs, arrival, chunk, scheduled] in arrival order
        self.chunk, self.t_obs = None, None
        self.n_sched = self.n_trig = 0

    @property
    def in_flight(self):
        """(t_obs, arrival, chunk) of the latest call still in flight, else None: HAgent's reference R (spec §4.3)."""
        return tuple(self.queue[-1][:3]) if self.queue else None

    def wants_call(self, t, trigger):
        return t >= self.next_call or (trigger and all(q[3] for q in self.queue))

    def issue(self, t, chunk):
        scheduled = t >= self.next_call
        self.n_sched += scheduled
        self.n_trig += not scheduled
        first = self.chunk is None and not self.queue
        self.queue.append([t, t if first else t + self.d, chunk, scheduled])
        self.next_call = t + self.s

    def arrive(self, t):
        """Install the answer arriving at t; True when a new chunk starts being used."""
        if not self.queue or self.queue[0][1] != t:
            return False
        self.t_obs, _, self.chunk, _ = self.queue.pop(0)
        return True

    def action(self, t):
        k = t - self.t_obs
        if k < len(self.chunk):
            return np.array(self.chunk[k], dtype=float)
        hold = np.zeros(len(self.chunk[-1]))
        hold[-1] = self.chunk[-1][-1]
        return hold


class GLRKalman:
    """spec §4.5 noisy eyes, the default pipeline: C1-E3d's GLR alarm and jump-Kalman following, online
    (scripts/c1e3_detectors.py stats() "glr", follow() "kfjump", same arithmetic).

    Frames as hf.Tracker's: generator [seed, 3], WARM updates on the still object first, measurement p(t - lag) +
    N(0, sigma^2 I), dropped with probability q_drop (never the first), the same draws in the same order, so @glr and
    @ema share their noise. The GLR statistic at step t, over the frames after the executing plan's observation c
    (set t_obs before each call): max over k in [max(c + 1, t - GLR_W + 1), t] of n |mean_k - r_c|^2 / (2 sig^2), n the
    delivered frames in k..t, r_c the mean of the last REF_N delivered frames at c, sig = max(sigma, 1 mm). Until the
    first alarm (statistic > GLR_H) the output stays at the warm-up mean: the reflex sees no shift. At the alarm a Kalman
    filter restarts at the maximising k (k_hat) from r_c with prior P0 (process noise Q, measurement variance sig^2)
    and catches up on the frames since k_hat; the output is then the warm-up mean + (estimate - r_c). One alarm per
    episode (ponytail: a false alarm before the shift leaves the slow steady-state filter to catch the real one;
    C1-E3d: 0.3% false alarms on the control, so it stays)."""

    WARM, GLR_W, REF_N = hf.Tracker.WARM, 30, 20
    GLR_H = 17.45740309874882  # results/c1-e3-det/detectors.json points/real/glr/thr: 1% false on calibration noise
    Q, P0, SIG_FLOOR = 1e-3 ** 2, 0.05 ** 2, 1e-3

    def __init__(self, seed, sigma, lag, q_drop):
        self.rng = np.random.default_rng([seed, 3])
        self.sigma, self.lag, self.q = sigma, lag, q_drop
        self.sig = max(sigma, self.SIG_FLOOR)
        self.true, self.meas, self.deliv, self.pos = [], [], [], []  # pos: stream positions of delivered frames
        self.cs, self.cnt = [np.zeros(3)], [0]  # running sums of delivered frames and their count (c1e3_detectors)
        self.t_obs = None  # the executing plan's observation step, set by GAgent.observe before each call
        self.t_alarm = self.k_hat = self.ref = self.x = self.P = self.base0 = None

    def _frame(self, src):
        z, u = self.rng.normal(0.0, 1.0, 3), self.rng.random()
        m, d = src + self.sigma * z, not self.meas or u >= self.q
        self.meas.append(m)
        self.deliv.append(d)
        self.cs.append(self.cs[-1] + m * d)
        self.cnt.append(self.cnt[-1] + d)
        if d:
            self.pos.append(len(self.meas) - 1)

    def _ref(self, i):
        """Mean of the last REF_N delivered frames up to stream position i (c1e3_detectors.last_mean)."""
        c = self.cnt[i + 1]
        start = self.pos[max(c - self.REF_N, 0)]
        return (self.cs[i + 1] - self.cs[start]) / (c - self.cnt[start])

    def _kf(self, i):
        self.P = self.P + self.Q
        if self.deliv[i]:
            k = self.P / (self.P + self.sig ** 2)
            self.x, self.P = self.x + k * (self.meas[i] - self.x), (1 - k) * self.P

    def __call__(self, p):
        p = np.array(p, float)
        if not self.meas:
            for _ in range(self.WARM):
                self._frame(p)
            self.base0 = self.cs[-1] / self.cnt[-1]
        self.true.append(p)
        self._frame(self.true[max(len(self.true) - 1 - self.lag, 0)])
        W, t = self.WARM, len(self.true) - 1
        if self.t_alarm is not None:
            self._kf(W + t)
        elif self.t_obs is not None:
            c, i = self.t_obs, W + t
            rc, best, kb = self._ref(W + c), 0.0, None
            for k in range(max(c + 1, t - self.GLR_W + 1), t + 1):
                n = self.cnt[i + 1] - self.cnt[W + k]
                if n > 0:
                    g = n * (((self.cs[i + 1] - self.cs[W + k]) / n - rc) ** 2).sum() / (2 * self.sig ** 2)
                    if g > best:
                        best, kb = g, k
            if best > self.GLR_H:
                self.t_alarm, self.k_hat, self.ref, self.x, self.P = t, kb, rc, rc.copy(), self.P0
                for j in range(kb, t + 1):
                    self._kf(W + j)
        return self.base0.copy() if self.t_alarm is None else self.base0 + (self.x - self.ref)


def share(S, delta):
    """spec §5.3: s_N = clip(S_N . Delta_N / |Delta_N|^2, 0, 1), the share of the shift the reflex had already carried at
    the plan's observation; 0 when |Delta_N| < 1 mm."""
    nd = float(np.linalg.norm(delta))
    return 0.0 if nd < ND_MIN else float(np.clip(np.asarray(S, float) @ delta / nd ** 2, 0.0, 1.0))


def shadow_sample(before, after, p0, p1):
    """spec §5.3: the shadow paired call's log entry. before, after: (t_obs, x_obs, chunk) of the two calls with one noise,
    on the cached pre-shift observation and on the current one; p0, p1: the object as the reflex saw it at those steps.
    "sample" = (g_after - g_before) . Delta / |Delta|^2 (grasp points by hf.target_diff, fallback horizon K), Delta =
    p1 - p0; None when |Delta| < 1 mm (the pseudo-shift of a control episode)."""
    diff, fb = hf.target_diff(after, before, K)
    delta = np.asarray(p1, float) - np.asarray(p0, float)
    nd = float(np.linalg.norm(delta))
    return {"t0": int(before[0]), "t1": int(after[0]), "diff": np.round(diff, 6).tolist(), "fb": bool(fb),
            "d": np.round(delta, 6).tolist(), "nd": round(nd, 6),
            "sample": None if nd < ND_MIN else round(float(diff @ delta) / nd ** 2, 6)}


class GAgent(hf.HAgent):
    """C1-E4 controller (spec §5): HAgent (C1-E3) on the call conveyor, plus law Z1 and the calibration arm.

    For a plan N observed at or after the engagement t_e and before the close (HAgent.on_new_chunk), every kappa-family
    arm logs s_N = share(S_N, Delta_N), S_N = C(t_obs_N) - C(t_e) the extra displacement the reflex commanded since
    the engagement. Gk0, Gauto and their T variants: kappa_N = s_N + kbar (1 - s_N), kbar given (G-kappa0: C1-E4a's
    kappa0; G-auto: kbar_30 frozen in the trial). Gcal acts as G (kappa = 1); the runner makes the shadow paired call
    at its engagement. The T variants re-query when the raw shift since the executing plan's observation exceeds T_THR.
    GautoU: Gauto plus the carry unwind after the close (part-2 spec §unwind, _unwind)."""

    LAW = ("Gk0", "Gk0T", "Gauto", "GautoT", "GautoU")
    # LAW: 1.0 placeholder, on_new_chunk sets Z1's kappa
    FAMILY = {"G": 1.0, "GT": 1.0, "Gkeep": 0.0, "GR": None, **dict.fromkeys(LAW, 1.0), "Gcal": 1.0}
    METHODS = ("none", "T0", "PPC", *FAMILY)

    def __init__(self, method, s, d, kbar=None, tau_close=None, **kw):
        assert (method in self.LAW) == (kbar is not None), f"{method}: kbar {kbar}"
        assert (method == "GautoU") == (tau_close is not None), f"{method}: tau_close {tau_close}"
        super().__init__(method, s, 0, **kw)  # orx.Schedule would assert max(s, d) + d <= H; the conveyor replaces it
        self.sched, self.kbar, self.tau_close = Conveyor(s, d), kbar, tau_close
        # GautoU: the close step, the runner's log (acting steps, plans acted under, the open that ended the window)
        self.t_c, self.unwind_cap, self.unwind_steps, self.unwind_plans, self.unwind_stop = None, None, 0, [], None

    def observe(self, t, eef, obj, contact=False):
        if isinstance(self.tracker, GLRKalman):
            self.tracker.t_obs = self.sched.t_obs  # the plan executing at observe time, as the C1-E3 agent compares
        super().observe(t, eef, obj, contact)

    def on_new_chunk(self):
        n = len(self.kappa_log)
        super().on_new_chunk()
        if len(self.kappa_log) == n:
            return
        S = self.c_hist[self.sched.t_obs] - self.c_hist[self.t_e]
        s_n = share(S, self.delta_n)
        if self.method in self.LAW:
            self.kappa = s_n + self.kbar * (1 - s_n)
        self.kappa_log[-1] |= {"s": round(s_n, 4), "S": np.round(S, 6).tolist(), "k": round(self.kappa, 4)}

    def act(self, t, a):
        a = super().act(t, a)
        if self.method != "GautoU" or not self.grasp_started:
            return a
        if self.t_c is None:  # the close step: G's law already ran on it (HAgent.act)
            self.t_c = t
            return a
        return self._unwind(t, a)

    def _unwind(self, t, a):
        """part-2 spec §unwind: the object is put down at the release, so the reflex acts only when the executing plan N,
        observed before t_c, commands the release (first open row from t on) before the next arrival on the known
        schedule; it then drives E_N = C(t) - C(t_obs_N), the extra N does not know, to 0 in G's command form, at most
        |E_N| / min(T_ramp, t_rel - t) a step (fixed at its first acting step under N), clip +-1, C counts what was sent.
        Window: t_c + tau_close up to the first open command in it, then off for good; elsewhere GautoU is Gauto. A stop
        at the first fresh plan left the unwind in flight unknown to later plans (review C1)."""
        sc = self.sched
        if self.unwind_stop is not None or t < self.t_c + self.tau_close:
            return a
        if a[6] <= 0:
            self.unwind_stop = t
            return a
        if sc.t_obs >= self.t_c:
            return a
        k = t - sc.t_obs
        rel = np.flatnonzero(np.asarray(sc.chunk, float)[k:, 6] <= 0)  # the executing chunk only: in-flight ones unseen
        t_rel = sc.t_obs + k + int(rel[0]) if len(rel) else np.inf
        # no trigger after the close (grasp_started), so the next arrival is queue[0]'s, else the next scheduled call's
        t_next = sc.queue[0][1] if sc.queue else sc.next_call + sc.d
        err = self.c - self.c_hist[sc.t_obs]  # observe wrote c_hist[t] before act: E from self.c
        if t_rel >= t_next or np.linalg.norm(err) < 1e-12:
            return a
        if not self.unwind_plans or self.unwind_plans[-1] != sc.t_obs:
            self.unwind_plans.append(sc.t_obs)
            self.unwind_cap = np.linalg.norm(err) / max(1, min(self.t_ramp, t_rel - t))
        base = np.clip(a[:3], -1, 1)
        new, _ = hf.ramp_step(base, -err, self.unwind_cap)
        self.c = self.c + orx.G_POS * (new - base)
        a[:3] = new
        self.unwind_steps += 1
        return a

    def trigger(self, t):
        if self.method in ("Gk0T", "GautoT"):
            return (not self.grasp_started and self.sched.chunk is not None
                    and np.linalg.norm(self.p_hist[t] - self.p_hist[self.sched.t_obs]) > orx.T_THR)
        return super().trigger(t)


def parse_arm(label):
    """A grid arm label -> (method, noise): "G@glr" -> ("G", "glr"), "Gauto" -> ("Gauto", ""). Noise: glr (default noisy
    eyes, GLRKalman) on G and PPC (part 2: PPC on the same frames as G@glr); ema (the old pipeline, hf.Tracker with
    EMA_BETA and EMA_EPS) on G only (spec §5.2); cv (lead.CVKalman, part 2 block B) on G, Gauto, PPC and Glead
    (lead.LeadAgent, built by the runner, not by GAgent)."""
    method, _, noise = label.partition("@")
    assert method in (*GAgent.METHODS, "Glead") and noise in ("", "glr", "ema", "cv"), label
    assert noise != "glr" or method in ("G", "PPC"), f"{label}: glr eyes run on G and PPC"
    assert noise != "ema" or method == "G", f"{label}: ema eyes run on G only"
    assert noise != "cv" or method in ("G", "Gauto", "PPC", "Glead"), f"{label}: cv eyes run on G, Gauto, PPC, Glead"
    return method, noise
