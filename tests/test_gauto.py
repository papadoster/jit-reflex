import importlib
import sys
from pathlib import Path

import numpy as np
import pytest

import gauto
import handoff as hf
import objreflex as orx

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def _marked(t, n=orx.H):
    """A chunk whose row k reads (t, k, ...): action(t') tells which plan runs and at which index."""
    ch = np.zeros((n, 7))
    ch[:, 0], ch[:, 1], ch[:, 6] = t, np.arange(n), -1.0
    return ch


def _drive(sc, T, trig=lambda t: False, n=orx.H):
    """The runner's order per step: arrive, then (trigger, call, arrive); the executing (t_obs, index) per step."""
    out = []
    for t in range(T):
        sc.arrive(t)
        if sc.wants_call(t, trig(t)):
            sc.issue(t, _marked(t, n))
            sc.arrive(t)
        a = sc.action(t)
        out.append((sc.t_obs, a[0], a[1], a[6]))
    return out, sc.n_sched, sc.n_trig


def test_make_episode_prefix():
    a, b = hf.make_episode("libero_object", 3, 7, "step"), hf.make_episode("libero_object", 3, 7, "step", prefix="c1e4")
    assert a == hf.make_episode("libero_object", 3, 7, "step", prefix="c1e3")  # C1-E3's seeds unchanged
    assert b.seed != a.seed and b.mag_class == a.mag_class and 0.08 <= b.r <= 0.20  # classes cycle as before
    assert hf.make_episode("libero_object", 3, 7, "control", prefix="c1e4").mag == 0.0


@pytest.mark.parametrize("cell", ["A", "B", "C", "D", "E", "F"])
def test_conveyor_is_the_old_schedule(cell):
    s, d = orx.CELLS[cell]
    rng = np.random.default_rng(1)
    trig = (lambda t: False) if d else (lambda t, f=rng.random(400) < 0.15: bool(f[t]))  # triggers only at d = 0
    assert _drive(gauto.Conveyor(s, d), 400, trig) == _drive(orx.Schedule(s, d), 400, trig)


def test_conveyor_long_delay():
    out, n_sched, n_trig = _drive(gauto.Conveyor(10, 40), 120)
    assert out[0][:3] == (0, 0, 0) and out[49][:3] == (0, 0, 49)  # the first answer at once, used up to index 49
    assert out[50][:3] == (10, 10, 40) and out[59][:3] == (10, 10, 49)  # then each plan runs indices d .. d + s - 1
    assert out[60][:3] == (20, 20, 40) and n_sched == 12 and n_trig == 0


def test_conveyor_triggers():
    sc = gauto.Conveyor(10, 20)
    out, n_sched, n_trig = _drive(sc, 40, lambda t: t in (12, 13, 25))
    # 0: at once; 10: scheduled; 12: triggered while 10 is in flight; 13 and 25: refused while 12 is in flight;
    # the scheduled calls follow 12 by s: 22, 32
    assert out[30][:3] == (10, 10, 20) and out[32][:3] == (12, 12, 20) and out[39][:3] == (12, 12, 27)
    assert (n_sched, n_trig) == (4, 1) and sc.in_flight[0] == 32


def test_conveyor_hold():
    out, _, _ = _drive(gauto.Conveyor(50, 0), 60, n=30)
    assert out[29][1:3] == (0, 29) and out[30][1:] == (0.0, 0.0, -1.0)  # zero deltas, the gripper keeps its command
    sc = gauto.Conveyor(50, 0)
    ch = _marked(0, 5)
    ch[-1, 6] = 1.0
    sc.issue(0, ch)
    sc.arrive(0)
    assert list(sc.action(7)) == [0, 0, 0, 0, 0, 0, 1.0]


# ------------------------------------------------------------------ GLR + jump Kalman against scripts/c1e3_detectors.py

def test_glr_kalman_matches_c1e3d():
    det = importlib.import_module("c1e3_detectors")
    T, tf, sig, lag, q = 90, 33, *gauto.NOISE
    tob = det.tobs("A", T)
    alarms = 0
    for seed in range(40):
        for mag in (0.0, 0.015, 0.04):
            P = np.zeros((1, T, 3))
            P[0, tf + 1:] = mag * np.array([0.6, 0.8, 0.0])
            z, u = det.noise([seed], T, 3)
            meas, deliv = det.measure(P, z, u, sig, lag, q)
            st, est, ref = det.stats(meas, deliv, tob, sig)
            stat, kh = st["glr"]
            a = det.first(stat, gauto.GLRKalman.GLR_H, np.ones(1, int), np.array([T - 1]))[0]
            tr, out = gauto.GLRKalman(seed, sig, lag, q), []
            for t in range(T):
                tr.t_obs = None if tob[t] < 0 else int(tob[t])
                out.append(tr(P[0, t]))
            assert np.array_equal(np.array(tr.meas), meas[0]) and tr.deliv == list(deliv[0])  # hf.Tracker's frames
            assert tr.t_alarm == (None if a < 0 else a), (seed, mag)
            alarms += a >= 0
            if a >= 0:
                assert tr.k_hat == kh[0, a]
            ta = det.first(stat, gauto.GLRKalman.GLR_H, np.ones(1, int), np.array([tf + det.LAG_N]))
            lag_m = det.follow("kfjump", meas, deliv, est, ref, tob, P, np.array([tf]), ta, kh[0, np.maximum(ta, 0)],
                                sig)[0][0]
            st_ = tf + 1 + np.arange(det.LAG_N)
            C = np.array(out)[st_] - tr.base0  # the applied correction: 0 before the alarm
            assert abs(np.linalg.norm((P[0, st_] - P[0, tf] - C)[:, :2], axis=1).mean() - lag_m) < 1e-12
    assert alarms >= 40  # the 4 cm shifts are found


def test_glr_kalman_still_object_outputs_constant():
    tr = gauto.GLRKalman(5, *gauto.NOISE)
    outs = []
    for t in range(60):
        tr.t_obs = 10 * (t // 10) if t else None
        outs.append(tr(np.array([0.1, 0.2, 0.9])))
    if tr.t_alarm is None:
        assert all(np.array_equal(o, outs[0]) for o in outs)
    assert np.allclose(outs[0], [0.1, 0.2, 0.9], atol=0.01)  # the warm-up mean


# ------------------------------------------------------------------ law Z1, the calibration arm, the shadow sample

P0 = np.array([0.10, 0.0, 0.0])


def _run(agent, delta, shift_at=12, T=40):
    """A still object shifted by delta at shift_at; the brain plans no motion (open gripper), the arm integrates."""
    eef, acts = np.zeros(3), []
    for t in range(T):
        agent.observe(t, eef, P0 + (np.asarray(delta) if t >= shift_at else 0.0))
        if agent.sched.arrive(t):
            agent.on_new_chunk()
        if agent.sched.wants_call(t, agent.trigger(t)):
            ch = np.zeros((orx.H, 7))
            ch[:, 6] = -1.0
            agent.sched.issue(t, ch)
            if agent.sched.arrive(t):
                agent.on_new_chunk()
        a = agent.act(t, agent.sched.action(t))
        eef = eef + orx.G_POS * np.clip(a[:3], -1, 1)
        acts.append(a)
    return np.array(acts)


def test_share():
    d = np.array([0.03, 0.0, 0.0])
    assert gauto.share(0.5 * d, d) == pytest.approx(0.5)
    assert gauto.share(2 * d, d) == 1.0 and gauto.share(-d, d) == 0.0
    assert gauto.share(d, np.array([0.0009, 0.0, 0.0])) == 0.0  # |Delta| < 1 mm


def test_law_z1_partial_share():
    ag = gauto.GAgent("Gauto", 10, 0, kbar=0.3, t_ramp=20)  # a slow ramp: 1.5 mm a step, 12 mm by the plan at 20
    _run(ag, [0.03, 0.0, 0.0])
    assert ag.t_e == 12
    rec = ag.kappa_log[0]
    S = ag.c_hist[20] - ag.c_hist[12]
    s = float(S @ ag.delta_n / (ag.delta_n @ ag.delta_n))
    assert rec["t"] == 20 and 0.3 < s < 0.5 and rec["s"] == pytest.approx(s, abs=1e-4)
    assert rec["k"] == pytest.approx(s + 0.3 * (1 - s), abs=1e-4)


def test_law_z1_t_plan_holds_one_minus_kbar():
    ag = gauto.GAgent("GautoT", 10, 0, kbar=0.3)
    _run(ag, [0.06, 0.0, 0.0])  # 6 cm > T_THR: the T call is observed at the engagement step, before any carry
    rec = ag.kappa_log[0]
    assert rec["t"] == 12 and rec["s"] == 0.0 and rec["k"] == pytest.approx(0.3)
    # under the T plan the reflex carries (1 - kbar) Delta = 4.2 cm, so the next plan (22 = 12 + s) sees s = 0.7
    assert ag.kappa_log[1]["t"] == 22 and ag.kappa_log[1]["s"] == pytest.approx(0.7, abs=1e-3)


def test_gk0_is_gauto_with_its_number_and_gcal_is_g():
    a = _run(gauto.GAgent("Gk0", 10, 0, kbar=0.355, t_ramp=20), [0.03, 0.0, 0.0])
    b = _run(gauto.GAgent("Gauto", 10, 0, kbar=0.355, t_ramp=20), [0.03, 0.0, 0.0])
    g, cal = gauto.GAgent("G", 10, 0, t_ramp=20), gauto.GAgent("Gcal", 10, 0, t_ramp=20)
    assert np.array_equal(a, b) and np.array_equal(_run(g, [0.03, 0, 0]), _run(cal, [0.03, 0, 0]))
    assert g.kappa_log[0]["k"] == 1.0 and 0 < g.kappa_log[0]["s"] < 1  # s logged for every family arm
    with pytest.raises(AssertionError):
        gauto.GAgent("G", 10, 0, kbar=0.3)
    with pytest.raises(AssertionError):
        gauto.GAgent("Gauto", 10, 0)


def test_reference_is_the_plan_in_flight():
    ag = gauto.GAgent("Gauto", 10, 40, kbar=0.3)  # A40: no orx.Schedule assertion
    _run(ag, [0.03, 0.0, 0.0], T=70)
    assert ag.ref[0] == 10 and ag.p_ref == pytest.approx(P0)  # observed at 10, still in flight at t_e = 12
    assert ag.kappa_log[0]["t"] == 20 and ag.kappa_log[0]["d"] == pytest.approx([0.03, 0.0, 0.0])


def test_shadow_sample():
    def walk(n_move):
        ch = np.zeros((orx.H, 7))
        ch[:n_move, 0], ch[:, 6], ch[n_move:, 6] = 0.5, -1.0, 1.0
        return ch

    x0 = np.zeros(3)
    x1 = x0 + [orx.G_POS, 0.0, 0.0]
    e = gauto.shadow_sample((5, x0, walk(10)), (6, x1, walk(12)), P0, P0 + [0.02, 0.0, 0.0])
    assert e["t0"] == 5 and e["t1"] == 6 and not e["fb"]
    assert e["diff"][0] == pytest.approx(orx.G_POS * (1 + 0.5 * 2)) and e["sample"] == pytest.approx(0.022 / 0.02)
    assert gauto.shadow_sample((5, x0, walk(10)), (6, x1, walk(12)), P0, P0)["sample"] is None


def test_parse_arm():
    assert gauto.parse_arm("G@glr") == ("G", "glr") and gauto.parse_arm("Gauto") == ("Gauto", "")
    for bad in ("Gauto@glr", "GRT", "G@real", "nope"):
        with pytest.raises(AssertionError):
            gauto.parse_arm(bad)
