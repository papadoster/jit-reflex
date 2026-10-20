"""src/vservo.py: PBVSAgent (PBVS spec §2) on a toy arm, spec §9 items 1-7, and the record's counters."""

import numpy as np
import pytest

import vservo
from objreflex import G_POS, H

P0 = np.array([0.10, 0.0, 0.0])
DLT = np.array([0.006, -0.005, 0.0])  # 7.8 mm: G's engagement (> 5 mm), under one action unit
EXACT = {"model": "I", "G": [G_POS, G_POS], "B": [0.0, 0.0], "delta": 0.005}  # the toy arm's own model
ARX = {"model": "A", "G": [0.0025, 0.003], "B": [0.79, 0.6], "delta": 1e-9}


def plan(t_obs, close_at=None):
    """A plan observed at t_obs: xy rows change with the row index k and with t_obs (old and new plans differ), a raw z
    beyond -1 (the runner clips it, the regulator leaves it), a rotation; the gripper open, closing for absolute steps
    >= close_at."""
    k, ch = np.arange(H), np.zeros((H, 7))
    ch[:, 0], ch[:, 1], ch[:, 2] = 0.2 + 0.01 * t_obs + 0.005 * k, -0.1 + 0.004 * k, -1.2
    ch[:, 3:6] = [0.02, -0.03, 0.01]
    ch[:, 6] = np.where(t_obs + k >= (np.inf if close_at is None else close_at), 1.0, -1.0)
    return ch


def drive(ag, delta=DLT, shift_at=12, T=40, close_at=None, gain=(G_POS, G_POS), b=(0.0, 0.0)):
    """The runner's order per step (observe, arrive, call, arrive, act); the object jumps by delta at shift_at; the
    arm's xy moves by D = gain clip(a) + b D_prev (default: EXACT's integrator), z by G_POS clip(a). Returns the plan's
    rows, the sent actions and per step (the anchor t_a, x(t), the agent's path w(t) after act)."""
    x, dx, rows, acts, log, ta = np.zeros(3), np.zeros(2), [], [], [], None
    for t in range(T):
        ag.observe(t, x, P0 + (np.asarray(delta) if t >= shift_at else 0.0))
        if ag.sched.arrive(t):
            ag.on_new_chunk()
            ta = t
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, plan(t, close_at))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
                ta = t
        row = ag.sched.action(t)
        a = ag.act(t, row)
        rows.append(row)
        acts.append(a)
        log.append((ta, x.copy(), None if ag.j != t else ag.w.copy()))
        dx = np.asarray(gain) * np.clip(a[:2], -1, 1) + np.asarray(b) * dx
        x = x + np.append(dx, G_POS * np.clip(a[2], -1, 1))
    return np.array(rows), np.array(acts), log


def sent(rows, acts):
    """The correction in metres per step (xy)."""
    return G_POS * (acts[:, :2] - np.clip(rows[:, :2], -1, 1))


def test_identity():
    """§9 item 1 (§2 identity): the arm on the model, the object jumps by DLT at 12 (s 10, d 0: t_e = 12, plan 10 saw
    it before). PBVS1.0 sends DLT on the first acting step, then the arm is on the object: nothing more. PBVS0.3 (band
    0.1 mm) removes 0.3 of e a step, e(12 + k) = -0.7^k DLT, until plan 20 (it saw the jump) takes over at 20."""
    ag = vservo.PBVSAgent("PBVS1.0", 10, 0, EXACT)
    rows, acts, _ = drive(ag)
    m = sent(rows, acts)
    assert ag.t_e == 12 and m[12] == pytest.approx(DLT[:2], abs=1e-12) and not m[:12].any() and not m[13:].any()
    ag = vservo.PBVSAgent("PBVS0.3", 10, 0, {**EXACT, "delta": 1e-4})
    rows, acts, _ = drive(ag)
    m = sent(rows, acts)
    assert m[12:20] == pytest.approx(0.3 * 0.7 ** np.arange(8)[:, None] * DLT[:2], abs=1e-12)
    assert not m[:12].any() and not m[20:].any() and ag.pbvs_stats["steps_acting"] == 8
    assert ag.pbvs_stats["flips"] == 0 and ag.pbvs_stats["max_e"] == pytest.approx(np.linalg.norm(DLT))


@pytest.mark.parametrize("d", [0, 10, 20])
@pytest.mark.parametrize("band", [EXACT, ARX], ids=["I", "A"])
def test_still_object_and_anchor_b(d, band):
    """§9 items 2 and 6: the object still, the arm exactly on the model (I, or A from its own first step). Unforced, no
    engagement and the plan's actions. With the window forced open (band 1e-9 m), e = 0 on every step:
    anchor (b) starts each plan's path at x(t_a) with its rows from t_a - t_obs (d), so the old / new plan mismatch over
    the delay window (the plans differ with t_obs and k) is no error."""
    band = {**band, "delta": 1e-9}
    ag = vservo.PBVSAgent("PBVS1.0", 10, d, band)
    rows, acts, _ = drive(ag, delta=np.zeros(3), T=60, gain=band["G"], b=band["B"])
    assert ag.t_e is None and np.array_equal(acts, rows)
    ag = vservo.PBVSAgent("PBVS1.0", 10, d, band)
    ag.t_e = -1  # the window forced open (no engagement step)
    rows, acts, log = drive(ag, delta=np.zeros(3), T=60, gain=band["G"], b=band["B"])
    st = ag.pbvs_stats
    assert np.array_equal(acts, rows) and st["steps_window"] == 60 and st["steps_acting"] == 0 and st["max_e"] < 1e-12
    assert log[10 + d][0] == 10 + d and np.array_equal(rows[10 + d], plan(10)[d])  # plan 10: from row d at its arrival


def test_dead_band():
    """§9 item 3: |e| <= delta, the plan's action. The 7.8 mm jump engages (> 5 mm) inside an 8 mm band; a band just
    under |DLT| acts at once."""
    ag = vservo.PBVSAgent("PBVS1.0", 10, 0, {**EXACT, "delta": 0.008})
    rows, acts, _ = drive(ag)
    st = ag.pbvs_stats
    assert ag.t_e == 12 and np.array_equal(acts, rows) and st["steps_acting"] == 0 and st["steps_window"] == 28
    assert st["sum_e"] == pytest.approx(8 * np.linalg.norm(DLT))  # 12-19 under plan 10, then plan 20 knows
    ag = vservo.PBVSAgent("PBVS1.0", 10, 0, {**EXACT, "delta": np.linalg.norm(DLT) - 1e-6})
    rows, acts, _ = drive(ag)
    assert not np.array_equal(acts[12], rows[12]) and ag.pbvs_stats["steps_acting"] == 1


def test_window_and_axes():
    """§9 items 4, 5: an arm 20 % short of the model, so e != 0 before t_e too; the jump at 12, the plans close from
    28. Before t_e and after the close step (G's window: it acts on the close step, then stops) the actions are the
    plan's bitwise; z, rotation and gripper are the plan's on every step; no E (c stays 0)."""
    ag = vservo.PBVSAgent("PBVS0.3", 10, 0, {**EXACT, "delta": 1e-4})
    rows, acts, _ = drive(ag, T=50, close_at=28, gain=(0.8 * G_POS, 0.8 * G_POS))
    moved = [t for t in range(50) if not np.array_equal(acts[t], rows[t])]
    assert ag.t_e == 12 and ag.grasp_started and moved[0] == 12 and moved[-1] == 28
    assert np.array_equal(acts[:12], rows[:12]) and np.array_equal(acts[29:], rows[29:])
    assert np.array_equal(acts[:, 2:], rows[:, 2:]) and not ag.c.any() and ag.pbvs_stats["steps_window"] == 17


def test_arx_path_is_the_recursion():
    """§9 item 7: the path against a hand-written loop over the executed rows, D from the arm's actual step before each
    arrival, at d = 10 with the arm off the model (the band never acts); B = 0 is the integrator x(t_a) + G sum clip."""
    for B in (ARX["B"], [0.0, 0.0]):
        G, B = np.array(ARX["G"]), np.array(B)
        ag = vservo.PBVSAgent("PBVS0.3", 10, 10, {**ARX, "B": B, "delta": 1.0})
        ag.t_e = -1  # the window forced open (no engagement step)
        rows, _, log = drive(ag, delta=np.zeros(3), T=60)
        xs = [x for _, x, _ in log]
        for t, (ta, _, w) in enumerate(log):
            D, want = (xs[ta] - xs[ta - 1])[:2] if ta else np.zeros(2), xs[ta][:2]
            for j in range(ta, t):
                D = G * np.clip(rows[j, :2], -1, 1) + B * D
                want = want + D
            assert w == pytest.approx(want, abs=1e-15), t
            if not B.any():
                assert w == pytest.approx(xs[ta][:2] + G * np.clip(rows[ta:t, :2], -1, 1).sum(0), abs=1e-15), t


def test_counters():
    """The record's counters: the window, the acting steps, sum and max |e|, flips of e . u_hat (the engaged jump's
    direction) over acting steps only (the in-band step of the other sign is no flip). The arm placed by hand on plans
    without xy motion: e = (m - 1) DLT."""
    ag = vservo.PBVSAgent("PBVS0.3", 10, 0, EXACT)
    ch = plan(0)
    ch[:, :2] = 0.0
    m = {13: 1.2, 15: 2.0}  # e / DLT at 12-16: -1, +0.2 (in the band), -1, +1, -1
    for t in range(17):
        ag.observe(t, m.get(t, 0.0) * DLT, P0 + (DLT if t >= 12 else 0.0))
        if ag.sched.wants_call(t, False):
            ag.sched.issue(t, ch)
            ag.sched.arrive(t)
            ag.on_new_chunk()
        ag.act(t, ag.sched.action(t))
    st, n = ag.pbvs_stats, np.linalg.norm(DLT)
    assert (st["steps_window"], st["steps_acting"], st["flips"]) == (5, 4, 2) and type(st["flips"]) is int
    assert st["sum_e"] == pytest.approx(4.2 * n) and st["max_e"] == pytest.approx(n)


def test_labels():
    assert vservo.LAMBDAS == {"PBVS0.1": 0.1, "PBVS0.3": 0.3, "PBVS1.0": 1.0}
    assert vservo.parse_arm("PBVS0.3") == ("PBVS0.3", "") and vservo.parse_arm("G@glr") == ("G", "glr")
    for bad in ("PBVS0.3@glr", "PBVS0.5", "PBVS"):
        with pytest.raises(AssertionError):
            vservo.parse_arm(bad)
    ag = vservo.PBVSAgent("PBVS0.1", 10, 20, EXACT)
    assert ag.method == "PBVS0.1" and ag.lam == 0.1 and ag.sched.d == 20 and not ag.trigger(0)
