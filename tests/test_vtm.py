"""src/vservo.py: VTMAgent (PBVS spec journal "2026-10-22 — ветка VT-m: формула и правила") on a toy arm: a_N by the
formula (d 0 and d > 0, the rows the delay ate skipped), its path the live one, no close row ahead, the plans before
t_e (a_floor), s_star None, the counters, the labels; and PBVSAgent bitwise across the _e hook."""

import numpy as np
import pytest

import vservo
from objreflex import G_POS, H
from test_vservo import ARX, DLT, EXACT, P0, plan

S = np.array([-0.08, 0.01])  # an s* off every toy plan's aim: a != 0
FINE = {**EXACT, "delta": 1e-4}
BAND_A = {**ARX, "delta": 1e-9}  # always acting once e != 0


def drive(ag, chunk, T, x=None, delta=DLT, shift_at=12, gain=EXACT["G"], b=EXACT["B"]):
    """The runner's order per step (observe, arrive, call, arrive, act); chunk(t_obs): the plan of a call; the object
    jumps by delta at shift_at; the arm placed by hand at x(t), else moved by the model (xy D = gain clip(a) + b D_prev,
    z G_POS clip(a)). Returns the sent actions, {t_obs: (t_a, a, aim)} at each arrival and the path w(t) after act
    (None off the window)."""
    xt, dx, acts, arr, ws = np.zeros(3), np.zeros(2), [], {}, []

    def installed():
        ag.on_new_chunk()
        a = getattr(ag, "a", None)
        arr[ag.sched.t_obs] = (t, None if a is None else a.copy(), getattr(ag, "aim", None))

    for t in range(T):
        xt = xt if x is None else x(t)
        ag.observe(t, xt, P0 + (np.asarray(delta) if t >= shift_at else 0.0))
        if ag.sched.arrive(t):
            installed()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, chunk(t))
            if ag.sched.arrive(t):
                installed()
        acts.append(ag.act(t, ag.sched.action(t)))
        ws.append(ag.w.copy() if ag.j == t else None)
        dx = np.asarray(gain) * np.clip(acts[-1][:2], -1, 1) + np.asarray(b) * dx
        xt = xt + np.append(dx, G_POS * np.clip(acts[-1][2], -1, 1))
    return np.array(acts), arr, ws


def rows(t_obs, close=()):
    """A plan whose xy rows change with k and t_obs, x beyond +1 from k = 6 (clipped); the gripper closing at the row
    indices close."""
    k, ch = np.arange(H), np.zeros((H, 7))
    ch[:, 0], ch[:, 1] = 0.5 + 0.1 * k, -0.3 - 0.05 * k + 0.01 * t_obs
    ch[:, 6] = np.where(np.isin(k, close), 1.0, -1.0)
    return ch


def x_hand(t):
    """The arm placed by hand, off any model: its last step differs from step to step."""
    return np.array([0.002 * t + 1e-4 * t * t, -0.003 * t, 0.3])


def test_aim_by_the_formula_d0():
    """Integrator band, d 0 (t_a = t_obs), the object still (no t_e, no correction, z = 0): a_N = x(t_a) + G sum_{k <
    k_c} clip(row_k) - p(t_obs) - s*. Plan 0 closes from row 30 (ahead, never reached), plan 10 from row 7, plan 20
    from row 0 (t_c = t_a: the anchor itself)."""
    G = np.array([0.01, 0.02])
    close = {0: range(30, H), 10: range(7, H), 20: range(H)}
    ag = vservo.VTMAgent("VTm", 10, 0, {"model": "I", "G": G, "B": [0.0, 0.0], "delta": 1.0}, S)
    _, arr, _ = drive(ag, lambda t: rows(t, close[t]), 21, x=x_hand, delta=np.zeros(3))
    assert ag.t_e is None and sorted(arr) == [0, 10, 20]
    for t, kc in ((0, 30), (10, 7), (20, 0)):
        ta, a, aim = arr[t]
        want = x_hand(t)[:2] + G * np.clip(rows(t)[:kc, :2], -1, 1).sum(0) - P0[:2] - S
        assert ta == t and not aim and a == pytest.approx(want, abs=1e-15), t
    assert np.abs(rows(10)[:7, 0]).max() > 1  # the clip matters


def test_aim_by_the_formula_delay_skips_the_eaten_rows():
    """ARX band, s 20, d 10: plan N starts at t_a = t_obs + 10 from row k0 = 10, D_{t_a-1} the arm's last step (no
    correction: z = 0). Plan 20 closes at row 9 (eaten by the delay) and from row 15: k_c = 15, t_c = 35, w_hat by the
    recursion over rows 10..14; plan 40 closes from row 10 = k0: t_c = t_a, w_hat = x(50); plan 60 only at row 9: a
    None; plan 0 never closes: a None."""
    G, B = np.array(ARX["G"]), np.array(ARX["B"])
    close = {0: (), 20: [9, *range(15, H)], 40: range(10, H), 60: [9]}
    ag = vservo.VTMAgent("VTm", 20, 10, {**ARX, "delta": 1.0}, S)
    _, arr, _ = drive(ag, lambda t: rows(t, close[t]), 71, x=x_hand, delta=np.zeros(3))
    assert {t: v[0] for t, v in arr.items()} == {0: 0, 20: 30, 40: 50, 60: 70} and ag.t_e is None
    D, w = (x_hand(30) - x_hand(29))[:2], x_hand(30)[:2]
    for k in range(10, 15):
        D = G * np.clip(rows(20)[k, :2], -1, 1) + B * D
        w = w + D
    assert arr[20][1] == pytest.approx(w - P0[:2] - S, abs=1e-15)
    assert arr[40][1] == pytest.approx(x_hand(50)[:2] - P0[:2] - S, abs=1e-15)
    assert arr[0][1] is None and arr[60][1] is None


@pytest.mark.parametrize("shift_at", [12, 25])
def test_the_aim_is_the_live_path_at_the_close(shift_at):
    """The same recursion as the live path, its anchor with z: an ARX arm on ARX's model, s 20, d 10, the jump at 12
    or 25 (t_e) and corrections sent from it (z != 0 at plan 20's arrival 30); plans close from step 45. The path w(45)
    after act(45), plan 20 still executing, is w_hat_20(45) = a_20 + p(20) + s*, p(20) the object plan 20 saw (the jump
    at 25: before it, p(30) after it; plan 20 then runs without its a)."""
    ag = vservo.VTMAgent("VTm", 20, 10, BAND_A, S)
    _, arr, ws = drive(ag, lambda t: plan(t, 45), 50, shift_at=shift_at, gain=ARX["G"], b=ARX["B"])
    ta, a, aim = arr[20]
    assert ag.t_e == shift_at and ta == 30 and aim == (shift_at <= 20) and np.abs(ag.z[30] - ag.z[20]).max() > 1e-4
    assert ws[45] == pytest.approx(a + (P0 + (DLT if shift_at <= 20 else 0))[:2] + S, abs=1e-15) and ag.grasp_started


@pytest.mark.parametrize("d", [0, 10])
def test_no_close_row_ahead_is_pbvs(d):
    """Plans without a close row: every a None (no clamp to the plan's end, no skipped step); VT-m is PBVS0.3 on the
    same band bitwise, its window steps under plans observed at or after t_e = 12 count as no_aim."""
    ag, pb = vservo.VTMAgent("VTm", 10, d, FINE, S), vservo.PBVSAgent("PBVS0.3", 10, d, FINE)
    acts, arr, _ = drive(ag, plan, 40)
    acts0, _, _ = drive(pb, plan, 40)
    st = ag.vtm_stats
    assert all(v[1] is None for v in arr.values()) and np.array_equal(acts, acts0) and ag.t_e == 12
    assert (st["steps_window"], st["steps_aim"], st["no_aim"]) == (28, 0, 40 - (20 + d)) and st["steps_acting"] > 0
    assert {k: st[k] for k in pb.pbvs_stats} == pb.pbvs_stats and ag.a_floor is None and ag.a_first is None


def test_plans_before_t_e_run_without_their_aim():
    """s 10, d 10, the jump at 12: plan 0 executes at t_e (its a is a_floor), plan 10 (observed at 10 < t_e) arrives at
    20, after t_e: neither applies its a, so the actions are PBVS0.3's bitwise up to 29; plan 20 (the first observed
    after t_e, a_first) applies its a from its arrival 30 to the close at 40."""
    ag, pb = vservo.VTMAgent("VTm", 10, 10, FINE, S), vservo.PBVSAgent("PBVS0.3", 10, 10, FINE)
    acts, arr, _ = drive(ag, lambda t: plan(t, 40), 50)
    acts0, _, _ = drive(pb, lambda t: plan(t, 40), 50)
    st = ag.vtm_stats
    assert ag.t_e == 12 and [arr[t][2] for t in (0, 10, 20)] == [False, False, True]
    assert all(arr[t][1] is not None for t in (0, 10, 20)) and np.abs(arr[10][1]).max() > 0.01
    assert np.array_equal(acts[:30], acts0[:30]) and not np.array_equal(acts[30], acts0[30])
    assert (st["steps_window"], st["steps_aim"], st["no_aim"]) == (29, 11, 0)
    assert np.array_equal(ag.a_floor, arr[0][1]) and np.array_equal(ag.a_first, arr[20][1])


@pytest.mark.parametrize("close_at", [29, 30])
def test_a_first_arrived_before_the_close(close_at):
    """The journal's "уточнения": a_first counts only a plan that arrived before the first close was executed. s 10,
    d 10, the jump at 12: plan 20, the first observed after t_e, arrives at 30. Plans close from close_at: at 30 plan
    20 executes the close itself (it arrived before it: a_first = a_20); at 29 plan 10 closes first, so plan 20 and
    every later one arrived after the close: a_first None, a_floor still plan 0's."""
    ag = vservo.VTMAgent("VTm", 10, 10, FINE, S)
    acts, arr, _ = drive(ag, lambda t: plan(t, close_at), 50)
    assert ag.t_e == 12 and int(np.argmax(acts[:, 6] > 0)) == close_at and arr[20][0] == 30 and arr[20][2]
    assert all(arr[t][1] is not None for t in (0, 10, 20, 30)) and np.array_equal(ag.a_floor, arr[0][1])
    assert (ag.a_first is None) if close_at == 29 else np.array_equal(ag.a_first, arr[20][1])


@pytest.mark.parametrize("d", [0, 10])
def test_a_floor_is_the_plan_acting_at_t_e(d):
    """The journal's "уточнения": a_floor is the a of the plan whose action runs at step t_e. s 10, the jump at 20 (t_e
    20 against plan 0 at d 10, plan 10 at d 0). d 10: plan 10, observed before t_e, arrives exactly at 20, after
    observe(20): the floor is its a, not plan 0's. d 0: plan 20, observed at t_e, arrives at 20: it saw the shift, its
    a is applied (a_first), and the floor stays plan 10's, executing at observe(20)."""
    ag = vservo.VTMAgent("VTm", 10, d, FINE, S)
    _, arr, _ = drive(ag, lambda t: plan(t, 45), 50, shift_at=20)
    assert ag.t_e == 20 and arr[10][0] == {0: 10, 10: 20}[d] and arr[20][0] == 20 + d
    assert not np.array_equal(arr[0][1], arr[10][1]) and not np.array_equal(arr[10][1], arr[20][1])
    assert np.array_equal(ag.a_floor, arr[10][1]) and np.array_equal(ag.a_first, arr[20][1])
    assert [arr[t][2] for t in (10, 20)] == [False, True]


@pytest.mark.parametrize("d", [0, 10, 20])
@pytest.mark.parametrize("band", [FINE, BAND_A], ids=["I", "A"])
@pytest.mark.parametrize("close_at", [28, None])
def test_no_s_star_is_pbvs03_bitwise(d, band, close_at):
    """s_star None (the paired Base never closed): every a None, the actions and PBVS's five counters are PBVS0.3's on
    the same band bitwise; no_aim the window steps under plans observed at or after t_e."""
    kw = {"gain": band["G"], "b": band["B"]}
    ag = vservo.VTMAgent("VTm", 10, d, band, None)
    acts, arr, _ = drive(ag, lambda t: plan(t, close_at), 50, **kw)
    pb = vservo.PBVSAgent("PBVS0.3", 10, d, band)
    acts0, _, _ = drive(pb, lambda t: plan(t, close_at), 50, **kw)
    st, tc = ag.vtm_stats, next((t for t in range(50) if acts[t, 6] > 0), 49)  # the window: t_e = 12 up to the close
    under = [t for t in range(12, tc + 1) if max(o for o in arr if arr[o][0] <= t) >= 12]
    assert np.array_equal(acts, acts0) and {k: st[k] for k in pb.pbvs_stats} == pb.pbvs_stats
    assert st["steps_aim"] == 0 and st["no_aim"] == len(under) and ag.a_floor is None and ag.a_first is None


def test_counters():
    """The record's counters by hand: the arm still at the origin, plans without xy motion closing at step 40 (never
    reached), s* = -p(20) - A. Plan 10 (before t_e = 12): e = -DLT on 12-19 (a_10 = DLT + A, the floor, not applied);
    plan 20: e = a_20 = A (6 mm) on 20-24. Acting on all 13 window steps, one flip (e . DLT changes sign at 20)."""
    A = np.array([0.006, 0.0])
    ag = vservo.VTMAgent("VTm", 10, 0, EXACT, -(P0 + DLT)[:2] - A)

    def flat(t):
        ch = plan(t, 40)
        ch[:, :2] = 0.0
        return ch

    drive(ag, flat, 25, x=lambda t: np.zeros(3))
    st, n = ag.vtm_stats, np.linalg.norm(DLT)
    assert st is ag.pbvs_stats and ag.t_e == 12
    assert (st["steps_window"], st["steps_acting"], st["flips"], st["steps_aim"], st["no_aim"]) == (13, 13, 1, 5, 0)
    assert all(type(st[k]) is int for k in ("steps_window", "steps_acting", "flips", "steps_aim", "no_aim"))
    assert st["sum_e"] == pytest.approx(8 * n + 5 * 0.006) and st["max_e"] == pytest.approx(n)
    assert ag.a_floor == pytest.approx(DLT[:2] + A, abs=1e-15) and ag.a_first == pytest.approx(A, abs=1e-15)


def test_labels():
    """VTm, lambda 0.3 inherited; oracle eyes only; PBVSAgent's machinery with its own label and s*."""
    assert vservo.VTM == {"VTm": 0.3} and vservo.parse_arm("VTm") == ("VTm", "")
    for bad in ("VTm@glr", "VTm0.3", "VTM", "VT-m"):
        with pytest.raises(AssertionError):
            vservo.parse_arm(bad)
    ag = vservo.VTMAgent("VTm", 10, 20, ARX, [0.01, 0.02, 0.03])
    assert ag.method == "VTm" and ag.lam == 0.3 and ag.sched.d == 20 and ag.s_star.tolist() == [0.01, 0.02]
    assert ag.G.tolist() == ARX["G"] and ag.delta == ARX["delta"] and not ag.trigger(0)
    assert vservo.VTMAgent("VTm", 10, 0, ARX).s_star is None


class PBVSBefore(vservo.PBVSAgent):
    """PBVSAgent._pbvs before the _e hook (c70ed29), e inline."""

    def _pbvs(self, t, a):
        while self.j < t:
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


@pytest.mark.parametrize("lam", list(vservo.LAMBDAS))
@pytest.mark.parametrize("d", [0, 10, 20])
@pytest.mark.parametrize("band", [FINE, BAND_A], ids=["I", "A"])
def test_pbvs_is_bitwise_across_the_hook(lam, d, band):
    """PBVSAgent's actions, path and counters are those before the _e hook, bitwise."""
    kw = {"gain": band["G"], "b": band["B"]}
    ag, old = vservo.PBVSAgent(lam, 10, d, band), PBVSBefore(lam, 10, d, band)
    (acts, _, ws), (acts0, _, ws0) = (drive(a, lambda t: plan(t, 28), 50, **kw) for a in (ag, old))
    assert np.array_equal(acts, acts0) and ag.pbvs_stats == old.pbvs_stats and ag.pbvs_stats["steps_acting"] > 0
    assert all((w is None and w0 is None) or np.array_equal(w, w0) for w, w0 in zip(ws, ws0))
