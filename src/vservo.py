"""C1-E4 PBVS: position-based visual servoing next to the reflex, the reference from the policy's plan (pure numpy).

Spec: docs/superpowers/specs/2026-10-20-c1-e4-pbvs-design.md. Builds on src/gauto.py (the call conveyor, HAgent's
engagement t_e, x_hist / p_hist, the first-close stop) and G's action conversion (objreflex.G_POS, clip +-1).
"""

import numpy as np

import gauto
from objreflex import G_POS

LAMBDAS = {f"PBVS{k}": k for k in (0.1, 0.3, 1.0)}  # spec §4: the gain grid, the share of e removed per step


def parse_arm(label):
    """gauto.parse_arm plus the PBVS labels (oracle eyes only): "PBVS0.3" -> ("PBVS0.3", "")."""
    return (label, "") if label in LAMBDAS else gauto.parse_arm(label)


class PBVSAgent(gauto.GAgent):
    """spec §2, PBVS-B. Built as GAgent "none" (no reflex, no extra brain calls) and relabelled: no method check of the
    parents matches a PBVS label, so they keep only the conveyor, t_e, x_hist / p_hist and the first-close stop.

    Anchor (b): a plan N starts executing at its arrival t_a (on_new_chunk, right after observe(t_a)); its path on xy
    w(t) = x(t_a) + sum_{j=t_a}^{t-1} D_j, D_j = G clip(row_j) + B D_{j-1}, D_{t_a-1} = x(t_a) - x(t_a-1) (0 at t = 0),
    row_j = sched.action(j)[:2] the plan's raw row (the conveyor's step -> row mapping, its hold past the end); B = 0 is
    the integrator. From t_e up to the first close (G's window): e = (x(t) - w(t)) - (p(t) - p(t_obs_N)) on xy; |e| <=
    delta: the plan's action, else a_xy = clip(clip(a_xy) - lambda e / G_POS, +-1). z, rotation and gripper untouched;
    no ramp, no kappa, no E (c stays 0). band = {"model", "G": [gx, gy], "B": [bx, by], "delta"} (spec §3, V0)."""

    def __init__(self, method, s, d, band, **kw):
        super().__init__("none", s, d, **kw)
        self.method, self.lam = method, LAMBDAS[method]
        self.G, self.B, self.delta = np.asarray(band["G"], float), np.asarray(band["B"], float), float(band["delta"])
        self.w = self.dx = self.j = None  # the path at step j, its last increment D_{j-1}
        self.u_hat, self.sgn = np.zeros(2), 0.0  # flips: the engagement's shift direction, the last acting sign
        self.pbvs_stats = {"steps_window": 0, "steps_acting": 0, "sum_e": 0.0, "max_e": 0.0, "flips": 0}

    def observe(self, t, eef, obj, contact=False):
        super().observe(t, eef, obj, contact)
        if self.t_e == t:  # the shift the engagement saw, against the plan executing at observe time
            u = (self.p_hist[t] - self.p_hist[self.sched.t_obs])[:2]
            self.u_hat = u / max(float(np.linalg.norm(u)), 1e-12)

    def on_new_chunk(self):
        super().on_new_chunk()
        t = next(reversed(self.x_hist))  # t_a: the arrival step, the last observed one
        self.w, self.j = self.x_hist[t][:2], t
        self.dx = self.w - self.x_hist[t - 1][:2] if t - 1 in self.x_hist else np.zeros(2)

    def act(self, t, a):
        a = np.array(a, dtype=float)
        if self.t_e is not None and not self.grasp_started:  # t >= t_e: observe(t_e) set it
            a = self._pbvs(t, a)
        return super().act(t, a)  # the first-close bookkeeping (HAgent.act; no PBVS label matches its methods)

    def _pbvs(self, t, a):
        while self.j < t:  # the path up to t: one increment per step (a catch-up once, at the first window step)
            self.dx = self.G * np.clip(self.sched.action(self.j)[:2], -1, 1) + self.B * self.dx
            self.w, self.j = self.w + self.dx, self.j + 1
        e = (self.x_hist[t][:2] - self.w) - (self.p_hist[t] - self.p_hist[self.sched.t_obs])[:2]
        ne, st = float(np.linalg.norm(e)), self.pbvs_stats
        st["steps_window"] += 1
        st["sum_e"] += ne
        st["max_e"] = max(st["max_e"], ne)
        if ne <= self.delta:
            return a
        st["steps_acting"] += 1
        sg = float(np.sign(e @ self.u_hat))
        if sg:
            st["flips"] += self.sgn == -sg
            self.sgn = sg
        a[:2] = np.clip(np.clip(a[:2], -1, 1) - self.lam * e / G_POS, -1, 1)
        return a
