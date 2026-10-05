import numpy as np
import pytest

import gauto
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
    a, b = lead.CVKalman(7, 0.01, 3, 0.1, 1e-8), lead.CVKalman(7, 0.01, 3, 0.1, 1e-8)
    assert np.array_equal(a(P0), b(P0 + 0.0))
    ag = lead.LeadAgent(10, 0, kbar=0.2, tau_lead=8, tau_close=6, tracker=lead.CVKalman(7, 0.01, 3, 0.1, 1e-8),
                        gate_m=16)
    _run(ag, move(0.003), T=20)
    assert ag.kf is ag.tracker  # the shared eyes give both the seen position and the velocity


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
