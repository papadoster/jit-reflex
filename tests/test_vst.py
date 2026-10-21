"""src/vservo.py: VSTAgent (PBVS spec journal "2026-10-21, 11:00 UTC — ветка VS-T гоняется") on a toy arm: the decay
to the jump, the reference at the close, the dead band, the window, the axes, s_star None, lambda, the counters."""

import numpy as np
import pytest

import gauto
import vservo
from objreflex import EPS, G_POS, H

P0 = np.array([0.10, 0.0, 0.0])
D36 = np.array([0.03, -0.02, 0.0])  # 36 mm: engages (> 5 mm); 0.3 e / G_POS stays under one action unit per axis
D10 = np.array([0.008, -0.006, 0.0])  # 10 mm: e / G_POS under one action unit per axis (lambda = 1, no clip)
S0 = (-P0)[:2]  # the still arm's offset to the unmoved object (the arm at the origin): the pre-jump grasp offset


def plan(t_obs, close_at=None, v=(0.0, 0.0), stop=np.inf):
    """A plan observed at t_obs: xy rows v up to absolute step stop, then 0; a raw z beyond -1 (the runner clips it, the
    servo leaves it), a rotation; the gripper open, closing for absolute steps >= close_at."""
    t, ch = t_obs + np.arange(H), np.zeros((H, 7))
    ch[:, :2] = np.where((t < stop)[:, None], v, 0.0)
    ch[:, 2], ch[:, 3:6] = -1.2, [0.02, -0.03, 0.01]
    ch[:, 6] = np.where(t >= (np.inf if close_at is None else close_at), 1.0, -1.0)
    return ch


def drive(ag, delta=D36, shift_at=12, T=60, **kw):
    """The runner's order per step (observe, arrive, call, arrive, act); the object jumps by delta at shift_at; the
    arm moves exactly by G_POS clip(a). Returns the plan's rows, the sent actions, and x(t), p(t) as observed."""
    x, rows, acts, xs, ps = np.zeros(3), [], [], [], []
    for t in range(T):
        p = P0 + (np.asarray(delta, float) if t >= shift_at else 0.0)
        ag.observe(t, x, p)
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, plan(t, **kw))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        row = ag.sched.action(t)
        acts.append(ag.act(t, row))
        rows.append(row)
        xs.append(x.copy())
        ps.append(p)
        x = x + G_POS * np.clip(acts[-1][:3], -1, 1)
    return np.array(rows), np.array(acts), np.array(xs), np.array(ps)


def sent(rows, acts):
    """The correction in metres per step (xy)."""
    return G_POS * (acts[:, :2] - np.clip(rows[:, :2], -1, 1))


@pytest.mark.parametrize("arm, dlt", [("VST0.1", D36), ("VST0.3", D36), ("VST1.0", D10)])
def test_decay_to_the_jump(arm, dlt):
    """An exact arm on plans without xy motion, s_star its pre-jump grasp offset; the object jumps by dlt at 12 (t_e =
    12). e(12 + j) = -(1 - lambda)^j dlt: the correction lambda (1 - lambda)^j dlt a step while |e| > 5 mm, so it sums
    to dlt but the residual inside the dead band."""
    lam = vservo.VST[arm]
    ag = vservo.VSTAgent(arm, 10, 0, S0)
    rows, acts, xs, ps = drive(ag, delta=dlt)
    m, e = sent(rows, acts), (xs - ps)[:, :2] - S0
    n = next(j for j in range(200) if (1 - lam) ** j * np.linalg.norm(dlt) <= EPS)
    assert ag.t_e == 12 and n == {"VST0.1": 19, "VST0.3": 6, "VST1.0": 1}[arm]
    assert m[12:12 + n] == pytest.approx(lam * (1 - lam) ** np.arange(n)[:, None] * dlt[:2], abs=1e-12)
    assert not m[:12].any() and not m[12 + n:].any() and ag.vst_stats["steps_acting"] == n
    assert e[12 + n] == pytest.approx(-(1 - lam) ** n * dlt[:2], abs=1e-12) and np.linalg.norm(e[-1]) <= EPS
    assert m.sum(0) == pytest.approx(dlt[:2] + e[-1], abs=1e-12)


@pytest.mark.parametrize("arm", list(vservo.VST))
def test_moving_plans_land_on_the_reference(arm):
    """Plans move the arm 0.1 unit a step on x and y up to step 15, then hold, and close from 30. s_star: the offset at
    the first close of the same plans without the jump (the paired Base). With the jump VS-T pulls the arm toward the
    reference grasp ahead of the plan's path; at the close the offset is s_star within the dead band."""
    kw = {"v": (0.1, 0.1), "stop": 15, "close_at": 30, "T": 40}
    _, acts, xs, ps = drive(vservo.VSTAgent(arm, 10, 0, None), delta=np.zeros(3), **kw)
    tc = int(np.argmax(acts[:, 6] > 0))
    s = (xs[tc] - ps[tc])[:2]
    ag = vservo.VSTAgent(arm, 10, 0, s)
    rows, acts, xs, ps = drive(ag, delta=D10, **kw)
    assert tc == 30 and int(np.argmax(acts[:, 6] > 0)) == 30 and ag.vst_stats["steps_acting"] >= 1
    assert np.linalg.norm((xs[30] - ps[30])[:2] - s) <= EPS < np.linalg.norm((xs[12] - ps[12])[:2] - s)


def test_the_correction_starts_from_the_clipped_plan():
    """Plan rows beyond +-1 on xy (1.5, -1.5): the correction starts from the clipped plan action, as G's and PBVS's,
    a_xy = clip(clip(row) - lambda e / G_POS, +-1), not clip(row - lambda e / G_POS). s_star set so that e(12) = (8, -8)
    mm."""
    kw = {"v": (1.5, -1.5), "T": 14}
    _, _, xs, ps = drive(vservo.VSTAgent("VST1.0", 10, 0, None), **kw)
    e12 = np.array([0.008, -0.008])
    ag = vservo.VSTAgent("VST1.0", 10, 0, (xs[12] - ps[12])[:2] - e12)
    rows, acts, _, _ = drive(ag, **kw)
    want = np.clip(np.clip(rows[12, :2], -1, 1) - e12 / G_POS, -1, 1)
    assert ag.t_e == 12 and acts[12, :2] == pytest.approx(want, abs=1e-12) and np.array_equal(acts[:12], rows[:12])
    assert np.abs(want - np.clip(rows[12, :2] - e12 / G_POS, -1, 1)).min() > 0.4


def test_dead_band():
    """|e| <= 5 mm (orx.EPS): the plan's action. The 10 mm jump engages, s_star 4.9 mm from the jumped target: no
    correction; 5.1 mm: one (lambda = 1)."""
    assert vservo.VSTAgent.DELTA == EPS == 0.005
    for off, n in ((0.0049, 0), (0.0051, 1)):
        ag = vservo.VSTAgent("VST1.0", 10, 0, (-P0 - D10)[:2] + off * np.array([0.6, 0.8]))
        rows, acts, _, _ = drive(ag, delta=D10)
        st = ag.vst_stats
        assert ag.t_e == 12 and st["steps_acting"] == n and np.array_equal(acts, rows) == (n == 0), off
        assert st["max_e"] == pytest.approx(off, abs=1e-12)


def test_window_axes_and_calls():
    """Plans close from 28; s_star 5 cm off (e stays above the band): VS-T acts from t_e = 12 through the close step 28
    (G's window), never before or after; z, rotation and gripper are the plan's on every step (the raw z -1.2 too); no
    E, no extra brain call. A 4 mm jump does not engage: the plan's actions throughout."""
    s = S0 + [0.05, 0.0]
    ag = vservo.VSTAgent("VST0.1", 10, 0, s)
    rows, acts, _, _ = drive(ag, T=50, close_at=28)
    moved = [t for t in range(50) if not np.array_equal(acts[t], rows[t])]
    assert ag.t_e == 12 and ag.grasp_started and moved == list(range(12, 29))
    assert np.array_equal(acts[:, 2:], rows[:, 2:]) and not ag.c.any() and ag.vst_stats["steps_window"] == 17
    assert (ag.sched.n_sched, ag.sched.n_trig) == (5, 0) and not any(ag.trigger(t) for t in range(50))
    ag = vservo.VSTAgent("VST0.1", 10, 0, s)
    rows, acts, _, _ = drive(ag, delta=[0.004, 0.0, 0.0], T=50, close_at=28)
    assert ag.t_e is None and np.array_equal(acts, rows) and ag.vst_stats["steps_window"] == 0


def test_no_target_is_the_plan_bitwise():
    """s_star None (the paired Base never closed): the plan's actions bitwise, GAgent "none"'s; its window steps count
    as no_target."""
    ag = vservo.VSTAgent("VST1.0", 10, 0, None)
    rows, acts, _, _ = drive(ag, T=50, close_at=28)
    _, acts0, _, _ = drive(gauto.GAgent("none", 10, 0), T=50, close_at=28)
    assert np.array_equal(acts, rows) and np.array_equal(acts, acts0) and ag.t_e == 12
    assert ag.vst_stats == {"steps_window": 17, "steps_acting": 0, "sum_e": 0.0, "max_e": 0.0, "flips": 0,
                            "no_target": 17}


def test_counters():
    """The record's counters, as PBVS's: the window, the acting steps, sum and max |e|, flips of e . u_hat (the engaged
    jump's direction) over acting steps only (the in-band step of the other sign is no flip), no_target 0. The arm
    placed by hand (plans without xy motion): e = (m - 1) DLT with s_star = -P0."""
    dlt = np.array([0.006, -0.005, 0.0])  # 7.8 mm
    ag = vservo.VSTAgent("VST0.3", 10, 0, S0)
    m = {13: 1.2, 15: 2.0}  # e / dlt at 12-16: -1, +0.2 (in the band), -1, +1, -1
    for t in range(17):
        ag.observe(t, m.get(t, 0.0) * dlt, P0 + (dlt if t >= 12 else 0.0))
        if ag.sched.wants_call(t, False):
            ag.sched.issue(t, plan(t))
            ag.sched.arrive(t)
            ag.on_new_chunk()
        ag.act(t, ag.sched.action(t))
    st, n = ag.vst_stats, np.linalg.norm(dlt)
    assert (st["steps_window"], st["steps_acting"], st["flips"], st["no_target"]) == (5, 4, 2, 0)
    assert type(st["flips"]) is int and st["sum_e"] == pytest.approx(4.2 * n) and st["max_e"] == pytest.approx(n)


def test_labels_and_lambda():
    """VST0.1 / 0.3 / 1.0 with their lambda; parse_arm keeps every earlier label; oracle eyes only."""
    assert vservo.VST == {"VST0.1": 0.1, "VST0.3": 0.3, "VST1.0": 1.0}
    assert vservo.LAMBDAS == {"PBVS0.1": 0.1, "PBVS0.3": 0.3, "PBVS1.0": 1.0}
    for a in ("VST0.1", "VST0.3", "VST1.0", "PBVS0.3"):
        assert vservo.parse_arm(a) == (a, "")
    for a in ("G@glr", "Gauto", "PPCL@glr", "none", "VS1.0"):
        assert vservo.parse_arm(a) == gauto.parse_arm(a)
    for bad in ("VST0.3@glr", "VST0.5", "VST", "VST1"):
        with pytest.raises(AssertionError):
            vservo.parse_arm(bad)
    ag = vservo.VSTAgent("VST0.1", 10, 20, [0.01, 0.02, 0.03])
    assert ag.method == "VST0.1" and ag.lam == 0.1 and ag.sched.d == 20 and ag.s_star.tolist() == [0.01, 0.02]
