import math

import numpy as np
import pytest

import gauto
import handoff as hf
import lead
import objreflex as orx

P0 = np.array([0.10, 0.0, 0.0])


def _run(agent, path, T=60, close_at=None):
    """The runner's order per step on a toy arm (x += G_POS clip(a)); the brain plans no motion, the gripper closes for
    absolute steps >= close_at (row k of a call observed at t is step t + k). Returns the executed actions."""
    eef, acts = np.zeros(3), []
    for t in range(T):
        agent.observe(t, eef, path(t))
        if agent.sched.arrive(t):
            agent.on_new_chunk()
        if agent.sched.wants_call(t, agent.trigger(t)):
            ch = np.zeros((orx.H, 7))
            ch[:, 6] = np.where(close_at is not None and t + np.arange(orx.H) >= (close_at or 0), 1.0, -1.0)
            agent.sched.issue(t, ch)
            if agent.sched.arrive(t):
                agent.on_new_chunk()
        a = agent.act(t, agent.sched.action(t))
        eef = eef + orx.G_POS * np.clip(a[:3], -1, 1)
        acts.append(a)
    return np.array(acts)


def _eef(acts):
    """_run's toy arm at each observation t (x(0) = 0): the actions before t."""
    return np.vstack([np.zeros(3), np.cumsum(orx.G_POS * np.clip(acts[:, :3], -1, 1), 0)])[:-1]


def step(t):
    return P0 + (np.array([0.045, 0.02, 0.0]) if t >= 12 else 0.0)


def move(v):
    return lambda t: P0 + np.array([v, 0.0, 0.0]) * max(t - 12, 0)


def test_lead_is_gauto_on_a_step():
    a = _run(gauto.GAgent("Gauto", 10, 0, kbar=0.2), step)
    b = _run(lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6), step)
    assert np.array_equal(a, b)


def test_lead_term_on_constant_speed_and_cap():
    v = 0.002  # m/step = 4 cm/s
    ag = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6)
    _run(ag, move(v), T=40)
    assert ag.lead(39) == pytest.approx([v * 8, 0.0, 0.0], rel=0.05, abs=1e-6)
    # the lead acts through a virtual object: G-auto sees p_seen + lead; the gate and the velocity read the true p_seen
    assert np.allclose(ag.p_hist[39] - ag.seen_hist[39], ag.lead(39)) and np.allclose(ag.seen_hist[39], move(v)(39))
    fast = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6)
    _run(fast, move(0.01), T=40)  # 20 cm/s x 8 steps = 8 cm > LEAD_MAX
    assert np.linalg.norm(fast.lead(39)) == pytest.approx(lead.LEAD_MAX)


def test_gate_closed_on_still_and_step_open_on_motion():
    for path, want in ((lambda t: P0, False), (step, False)):
        kf, seen = lead.CVKalman.oracle(), []
        for t in range(80):
            seen.append(path(t))
            kf(path(t))
            assert lead.lead_gate(kf, seen, 6) is want
    kf, seen = lead.CVKalman.oracle(), []
    for t in range(40):
        seen.append(move(0.001)(t))
        kf(seen[-1])
    assert lead.lead_gate(kf, seen, 6)


def test_noisy_eyes_share_the_stream_and_feed_the_lead():
    eyes = (lead.CVKalman(7, 0.01, 3, 0.1, 1e-8), hf.Tracker(7, 0.01, 3, 0.1), gauto.GLRKalman(7, 0.01, 3, 0.1))
    for t in range(50):
        for e in eyes:
            e(move(0.002)(t))
    st = [e.rng.bit_generator.state for e in eyes]
    assert st[0] == st[1] == st[2]  # the same draws in the same order: branches with one seed share the noise
    kf = lead.CVKalman(7, 0.01, 3, 0.1, 1e-8)
    ag = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6, tracker=kf, gate_m=16)
    _run(ag, move(0.003), T=20)
    # the shared eyes give both the seen position and the velocity, one filter step per env step (the parent's tracker
    # is off: it would see the virtual object)
    twin = lead.CVKalman(7, 0.01, 3, 0.1, 1e-8)
    assert ag.kf is kf and len(kf.true) == 20 and ag.tracker is None
    assert all(np.array_equal(twin(move(0.003)(t)), ag.seen_hist[t]) for t in range(20))


def test_lead_does_not_run_away_on_a_brain_that_does_not_react():
    # Task 6 Mac check: with the lead added to U and E restarting at every plan, a barely reacting brain got the full
    # lead again under each new plan (C 22.5 cm along the motion against the object's 9.9 cm)
    v, tau = 0.002, 15
    L = v * tau
    acts = _run(lead.LeadAgent(10, 0, kbar=0.038, tau_lead=tau, tau_close=13), move(v), T=60)  # zero chunks
    off = _eef(acts)[:, 0] - np.array([move(v)(t)[0] - P0[0] for t in range(60)])  # arm ahead of the object along v
    assert (off[40:] >= 0.5 * L).all() and (off[40:] <= 1.5 * L).all(), off[40:]
    assert abs(off[59] - off[40]) < 0.25 * L  # no growth from plan to plan


def test_cv_eyes_follow_the_lagged_position():
    kf, v = lead.CVKalman(0, 0.0, 3, 0.0, 1e-8), 0.001
    for t in range(60):
        est = kf(move(v)(t))
    # on uniform motion the CV filter has no steady-state error, so it sits on the lagged frame p(t - 3), 3v behind p(t)
    assert np.allclose(est, move(v)(59 - 3), atol=1e-6) and np.allclose(kf.v, [v, 0.0, 0.0], atol=1e-7)


def test_even_gate_window():
    kf, seen = lead.CVKalman.oracle(), []
    for t in range(40):
        seen.append(move(0.001)(t))
        kf(seen[-1])
    with pytest.raises(AssertionError):
        lead.lead_gate(kf, seen, 7)
    with pytest.raises(AssertionError):
        lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6, gate_m=7)


def test_lead_is_gauto_on_a_step_with_a_close():
    a = _run(gauto.GAgent("Gauto", 10, 0, kbar=0.2), step, close_at=30)
    b = _run(lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6), step, close_at=30)
    assert np.array_equal(a[:30], b[:30]) and np.array_equal(a[36:], b[36:])
    assert np.array_equal(a, b)  # the reflex has carried the whole step by the close: Z1's U is 0 in the window


def test_z1_for_a_plan_inside_the_tracking_window(monkeypatch):
    monkeypatch.setattr(lead, "lead_gate", lambda *a: False)  # no lead: U is Z1's target alone
    ag = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6)
    _run(ag, move(0.002), T=40, close_at=25)  # window 25..30, the plan observed at 30 arrives inside it
    assert ag.t_close == 25 and ag.t_e == 15 and [e["t"] for e in ag.kappa_log] == [20, 30]
    assert "post_close" not in ag.kappa_log[0] and ag.kappa_log[-1]["post_close"] is True
    # Z1 at 30: kappa 0.93 < 1 and U(30) = (1 - kappa) Delta_30 = 2.6 mm (the old reset: kappa 1, U 0); the reflex had
    # carried most of the shift, so U(30) is inside G's dead band and the action at 30 stays 0
    u = ag.unknown_shift(30)
    assert ag.kappa_log[-1]["k"] < 1 and np.allclose(u, (1 - ag.kappa) * ag.delta_n) and 0 < np.linalg.norm(u) < orx.EPS
    late = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6)
    acts = _run(late, move(0.002), T=40, close_at=29)  # window 29..34: U(32) = 2.6 + 4 mm leaves the dead band
    assert acts[32, :3].any()  # the old reset: U(32) = 4 mm, no action before 33


def test_tracking_until_the_fingers_close(monkeypatch):
    # the lead off: with it, a constant speed and the shrinking horizon hold the virtual object still in the window
    monkeypatch.setattr(lead, "lead_gate", lambda *a: False)
    g = _run(gauto.GAgent("Gauto", 10, 0, kbar=0.2), move(0.002), T=45, close_at=30)
    ld = _run(lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6), move(0.002), T=45, close_at=30)
    assert not g[31:].any(axis=0)[:3].any()  # G-auto stops at the first close (the plan moves nothing)
    assert np.abs(ld[32:36, :3]).sum(axis=1).min() > 0  # tracking goes on tau_close steps (at 31: 2 mm < G's dead band)
    assert not ld[36:, :3].any() and (ld[30:, 6] > 0).all()


def test_lead_shrinks_in_the_tracking_window():
    # at the close command the fingers finish tau_close steps later: the lead's horizon is the time left, tau_lead - j
    # at close + j, down to tau_lead - tau_close (the arm's lag); a constant tau_lead put the arm v tau_close ahead of
    # the object when the fingers closed
    v, tau, tc = 0.002, 15, 6
    ag = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=tau, tau_close=tc)
    _run(ag, move(v), T=45, close_at=30)
    off = {t: np.linalg.norm(ag.p_hist[t] - ag.seen_hist[t]) for t in range(30, 45)}
    for t, h in ((30, tau), (31, tau - 1), (35, tau - 5), (36, tau - tc), (44, tau - tc)):
        assert off[t] == pytest.approx(v * h, rel=0.05), (t, off[t] / v)


def test_move_episode_classes_and_stop():
    eps = [lead.make_move_episode("libero_spatial", 0, i) for i in range(3)]
    assert sorted(e.speed for e in eps) == list(lead.SPEEDS)
    assert {lead.make_move_episode("libero_spatial", 0, i).t_fire for i in range(200)} == set(range(5, 16))
    assert all(e.kind == "move" for e in eps) and eps[0].seed == lead.make_move_episode("libero_spatial", 0, 0).seed
    ep = eps[0]
    pt = lead.MovePerturbation(ep, tau_close=6)
    obj = np.array([0.0, 0.1, 1.0])
    dqs = [pt.dq(t, obj, close_cmd=t >= 20) for t in range(40)]
    # the motion starts by time, at the episode's t_fire, and moves to tau_close steps after the first close at 20
    assert pt.t_fire == ep.t_fire and [t for t, d in enumerate(dqs) if d is not None] == list(range(ep.t_fire, 26))
    assert np.linalg.norm(dqs[ep.t_fire]) == pytest.approx(ep.speed / lead.HZ) and dqs[ep.t_fire][2] == 0.0
    early = lead.MovePerturbation(ep, tau_close=6)  # the brain closes before t_fire: the motion never fires
    assert all(early.dq(t, obj, close_cmd=t >= ep.t_fire - 1) is None for t in range(40))
    assert pt.stop == "tau" and early.t_fire is None and early.stop is None


def test_move_follows_its_commanded_track():
    # Task 6 Mac check: a constant step with the qvel set ran 6% fast; the shift to the track corrects the drift
    for i in range(3):
        ep = lead.make_move_episode("libero_spatial", 0, i)
        pt, obj, step = lead.MovePerturbation(ep, tau_close=6), np.array([0.0, 0.1, 1.0]), ep.speed / lead.HZ
        u = np.array([math.cos(ep.direction), math.sin(ep.direction), 0.0])
        err = []
        for t in range(50):  # under the path cap at 6 cm/s (50 steps)
            d = pt.dq(t, obj, close_cmd=False)
            if d is not None:
                obj = obj + 0.94 * d  # an env that applies only 94% of each shift
                err.append(np.linalg.norm(obj - (pt.p_pre + step * (t - pt.t_fire + 1) * u)))
        assert len(err) == 50 - ep.t_fire
        assert max(err) < 0.065 * step  # bounded at 0.06 / 0.94 of a step, never accumulating
        assert ep.speed > 0.02 or max(err) < 1e-4  # 2 cm/s: within 0.1 mm


def test_held_is_contact_once_the_fingers_closed_and_stopped():
    w = [0.08, 0.08, 0.079, 0.07, 0.06, 0.06]  # first close command at 1: the fingers move from 2 and stop at 5
    assert not lead.held(True, w[:2], 1)  # the close step: the fingers have not moved yet
    assert not lead.held(True, w[:5], 1)  # still closing
    assert lead.held(True, w, 1) and not lead.held(False, w, 1) and not lead.held(True, w, None)
    # Task 6 Mac check: a touch with the fingers still open (52 or 31 mm) stopped the motion
    assert not lead.held(True, [0.08] * 6, 1)
    # the brain's gripper flickers (close, open, close): fingers that reopened and paused have not stopped closing
    # (the fix re-check, SmolVLA libero_object:0 init 48: 0.7 mm closed, reopened, paused at 79 mm)
    assert not lead.held(True, [0.08, 0.08, 0.0793, 0.0786, 0.0796, 0.07975, 0.07983], 1)
    assert lead.held(True, [0.08, 0.08, 0.0793, 0.0796, 0.0790, 0.0790], 1)  # closing again, then stopped


def test_move_stops_when_held_or_at_the_path_cap():
    ep = lead.make_move_episode("libero_spatial", 0, 0)
    obj = np.array([0.0, 0.1, 1.0])
    pt = lead.MovePerturbation(ep, tau_close=6)
    # held before the close is ignored; from the close on it stops the motion for good (steps t_fire..21 move)
    dqs = [pt.dq(t, obj, close_cmd=t >= 20, held=t in (ep.t_fire + 1, 22)) for t in range(40)]
    assert pt.stop == "held" and [d is not None for d in dqs] == [ep.t_fire <= t < 22 for t in range(40)]
    capped = lead.MovePerturbation(ep, tau_close=6)
    moved = sum(capped.dq(t, obj, close_cmd=False) is not None for t in range(400))
    assert capped.stop == "path" and moved == round(lead.MAX_PATH / (ep.speed / lead.HZ))
    assert capped.path == pytest.approx(lead.MAX_PATH)


def test_parse_arm_accepts_the_lead_and_cv_eyes():
    assert gauto.parse_arm("Glead") == ("Glead", "")
    assert gauto.parse_arm("Glead@cv") == ("Glead", "cv") and gauto.parse_arm("PPC@cv") == ("PPC", "cv")
    assert gauto.parse_arm("Gauto@cv") == ("Gauto", "cv")
    with pytest.raises(AssertionError):
        gauto.parse_arm("Glead@ema")
