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
    assert np.allclose(ag.unknown_shift(39) - gauto.GAgent.unknown_shift(ag, 39), ag.lead(39))
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
    ag = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6, tracker=lead.CVKalman(7, 0.01, 3, 0.1, 1e-8),
                        gate_m=16)
    _run(ag, move(0.003), T=20)
    assert ag.kf is ag.tracker  # the shared eyes give both the seen position and the velocity


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


def test_tracking_until_the_fingers_close():
    g = _run(gauto.GAgent("Gauto", 10, 0, kbar=0.2), move(0.002), T=45, close_at=30)
    ld = _run(lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6), move(0.002), T=45, close_at=30)
    assert not g[31:].any(axis=0)[:3].any()  # G-auto stops at the first close (the plan moves nothing)
    assert np.abs(ld[31:36, :3]).sum(axis=1).min() > 0  # the lead keeps tracking tau_close steps
    assert not ld[36:, :3].any() and (ld[30:, 6] > 0).all()


def test_move_episode_classes_and_stop():
    eps = [lead.make_move_episode("libero_spatial", 0, i) for i in range(3)]
    assert sorted(e.speed for e in eps) == list(lead.SPEEDS) and all(0.15 <= e.r <= 0.25 for e in eps)
    assert all(e.kind == "move" for e in eps) and eps[0].seed == lead.make_move_episode("libero_spatial", 0, 0).seed
    ep = eps[0]
    pt = lead.MovePerturbation(ep, tau_close=6)
    eef, obj = np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.1, 1.0])
    dqs = [pt.dq(t, eef, obj, close_cmd=t >= 20) for t in range(40)]
    moving = [d for d in dqs if d is not None]
    assert pt.t_fire == 0 and len(moving) == 26  # steps 0..25: tau_close steps after the first close at 20
    assert np.linalg.norm(moving[0]) == pytest.approx(ep.speed / lead.HZ) and moving[0][2] == 0.0
    far = lead.MovePerturbation(ep, tau_close=6)
    assert far.dq(0, eef, np.array([0.0, 0.3, 1.0]), close_cmd=False) is None and far.t_fire is None


def test_parse_arm_accepts_the_lead_and_cv_eyes():
    assert gauto.parse_arm("Glead") == ("Glead", "")
    assert gauto.parse_arm("Glead@cv") == ("Glead", "cv") and gauto.parse_arm("PPC@cv") == ("PPC", "cv")
    assert gauto.parse_arm("Gauto@cv") == ("Gauto", "cv")
    with pytest.raises(AssertionError):
        gauto.parse_arm("Glead@ema")
