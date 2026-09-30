import dataclasses
import hashlib
from types import SimpleNamespace

import numpy as np
import pytest

import handoff as hf
import objreflex as orx

P0 = np.array([0.10, 0.0, 0.0])  # the object before the shift; the toy arm starts at the origin


def _walk(x, target, n=50, v=0.5):
    """A chunk that walks the arm to target at <= v action units per axis and step, then closes the gripper."""
    ch, xx = np.zeros((n, 7)), np.array(x, float)
    ch[:, 6] = -1.0
    for j in range(n):
        if np.linalg.norm(target - xx) < 1e-9:
            ch[j:, 6] = 1.0
            break
        a = np.clip((target - xx) / orx.G_POS, -v, v)
        ch[j, :3], xx = a, xx + orx.G_POS * a
    return ch


TRAJ = _walk(np.zeros(3), P0, n=100)  # a memorised motion from the start
BRAINS = {  # brain(x, p, t) -> chunk; they only differ after the shift
    "blind_abs": lambda x, p, t: _walk(x, P0),  # aims at the old place from wherever the arm is
    "blind_rel": lambda x, p, t: TRAJ[t: t + 50],  # replays its delta commands: the arm's offset carries over
    "seeing": lambda x, p, t: _walk(x, p),  # aims at the object where it is
}
FAR = np.array([0.30, 0.0, 0.0])  # a far object (close at step 55): at d > 0, plans observed after the shift still count
TRAJ_FAR = _walk(np.zeros(3), FAR, n=150)
FAR_BRAINS = {
    "blind_abs": lambda x, p, t: _walk(x, FAR),
    "blind_rel": lambda x, p, t: TRAJ_FAR[t: t + 50],
    "blind_rel_open": lambda x, p, t: np.where(np.arange(7) == 6, -1.0, TRAJ_FAR[t: t + 50]),  # never closes
}


def _episode(method, brain, s=10, d=0, alpha=0.0, moves=None, t_fire=5, steps=45, cls=hf.HAgent, p0=P0, **kw):
    """Toy arm (x += G_POS a) and object (at p0) in the runner's order. moves {t: dp}: object moves applied before step
    t's physics, so observe(t + 1) is the first to see them (as the shift in LIBERO). Stops after the first close.
    brain: a BRAINS key or a brain(x, p, t)."""
    moves = {5: np.array([0.0, 0.03, 0.0])} if moves is None else moves
    brain = BRAINS[brain] if isinstance(brain, str) else brain
    ag = cls(method, s, d, **kw)
    sb = hf.SeeingBrain(alpha)
    pert = SimpleNamespace(ep=SimpleNamespace(kind="step"), t_fire=t_fire, p_pre=p0.copy())
    x, p, acts = np.zeros(3), p0.copy(), []
    for t in range(steps):
        ag.observe(t, x, p) if cls is orx.Agent else ag.observe(t, x, p, False)
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            start = 0 if ag.sched.chunk is None else d
            ag.sched.issue(t, sb.chunk(t, brain(x, p, t), pert, p, start, ag.grasp_started))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        a = np.clip(ag.act(t, ag.sched.action(t)), -1, 1)
        acts.append(a)
        sb.executed(t, ag.sched.t_obs, a, pert)
        p = p + moves.get(t, 0.0)
        x = x + orx.G_POS * a[:3]
        if a[6] > 0:
            break
    return x, p, ag, np.array(acts), sb


def _far(method, brain, steps=150, **kw):
    return _episode(method, FAR_BRAINS[brain], p0=FAR, steps=steps, **kw)


def test_make_episode_kinds_and_seeds():
    e = hf.make_episode("libero_spatial", 3, 7, "step")
    assert 0.08 <= e.r < 0.20 and e.mag_class == (50 * 3 + 7) % 3
    assert orx.MAG_CLASSES[e.mag_class][0] <= e.mag < orx.MAG_CLASSES[e.mag_class][1]
    assert e.seed != orx.make_episode("libero_spatial", 3, 7, "step").seed  # new c1e3 seeds
    c = hf.make_episode("libero_spatial", 3, 7, "close")
    assert 0.03 <= c.r < 0.08 and c.mag > 0
    k = hf.make_episode("libero_spatial", 3, 7, "control")
    assert k.mag == 0 and k.mag_class == -1 and 0 <= k.angle < 2 * np.pi
    t10 = hf.make_episode("libero_10", 9, 0, "step")
    assert t10.mag_class == (50 * 32 + 0) % 3  # libero_10 task 9 is the 33rd of ALL_TASKS
    assert hf.make_episode("libero_10", 9, 0, "step") == t10
    with pytest.raises(AssertionError):
        hf.make_episode("libero_spatial", 0, 0, "smooth")


def test_perturbation_close_shifts_once_and_control_only_records():
    for kind in ("close", "control"):
        ep = hf.make_episode("libero_spatial", 0, 0, kind)
        pt, far, near = hf.Perturbation(ep, np.zeros(6)), np.zeros(3), np.array([ep.r - 1e-3, 0, 0])  # a point box
        assert pt.dq(0, far + 1.0, far, False) is None and pt.t_fire is None
        out = [pt.dq(t, near, np.zeros(3), False) for t in (1, 2)]
        assert pt.t_fire == 1 and np.array_equal(pt.p_pre, np.zeros(3))
        assert (out[0] is None) == (kind == "control") and out[1] is None
    pt = hf.Perturbation(hf.make_episode("libero_spatial", 0, 0, "control"))
    assert pt.dq(0, np.zeros(3), np.zeros(3), True) is None and pt.t_fire is None  # not after a close command


BOX = [0.0, 0.0, 0.026, 0.078, 0.079, 0.027]  # a bowl's box (Mac scratch): centre 2.6 cm above p, top at 5.3 cm


def test_box_dist_inside_outside_and_no_box():
    p = np.array([0.1, 0.2, 0.9])
    assert hf.box_dist(p + [0.05, -0.07, 0.04], p, BOX) == 0.0  # inside
    assert hf.box_dist(p + [0.10, 0.0, 0.10], p, BOX) == pytest.approx(np.hypot(0.10 - 0.078, 0.10 - 0.053))
    assert hf.box_dist(p + [0.0, 0.0, -0.02], p, BOX) == pytest.approx(0.019)  # below the bottom at p - 1 mm
    assert hf.box_dist(p + [0.03, 0.04, 0.0], p, None) == pytest.approx(0.05)  # no box: the origin


def test_close_shift_fires_by_the_box_step_and_control_by_the_origin():
    """Journal item 14: the same arm 8 cm above the object's origin, 2.7 cm above its box, radius 6 cm: only the close
    shift fires. A close shift without the box is an error."""
    p, eef = np.array([0.1, 0.2, 0.9]), np.array([0.1, 0.2, 0.98])
    fired = {}
    for kind in ("close", "step", "control"):
        pt = hf.Perturbation(dataclasses.replace(hf.make_episode("libero_spatial", 0, 0, kind), r=0.06), BOX)
        dq = pt.dq(0, eef, p, False)
        fired[kind] = pt.t_fire == 0
        assert (dq is not None) == (kind == "close")
    assert fired == {"close": True, "step": False, "control": False}
    with pytest.raises(AssertionError):
        hf.Perturbation(hf.make_episode("libero_spatial", 0, 0, "close"))
    hf.Perturbation(hf.make_episode("libero_spatial", 0, 0, "step"))  # step and control need no box


def test_surrogate_and_push_filter_measure_to_the_box_at_the_last_position():
    """The arm 8 cm above p(t - 1), 2.7 cm above its box: a push at r_c = 5 cm with the box, none without; the box
    sits at p(t - 1), not at p(t)."""
    p0, eef = np.array([0.1, 0.2, 0.9]), np.array([0.1, 0.2, 0.98])
    p1 = p0 + [0.004, 0, 0]
    assert hf.surr_push(p1, p0, eef, eef, 0.05, False, BOX) and not hf.surr_push(p1, p0, eef, eef, 0.05, False)
    assert not hf.surr_push(p1, p0 - [0, 0, 0.03], eef, eef, 0.05, False, BOX)  # box top 5.7 cm below the arm
    for box, flag in ((BOX, True), (None, False)):
        f = hf.PushFilter("surr", 0.05, False, box)
        f(p0, eef, False)
        seen = f(p1, eef, False)
        assert f.flag == flag and seen == pytest.approx(p0 if flag else p1)


def test_plan_path_close_index_and_target_diff():
    ch = np.zeros((50, 7))
    ch[:, 0], ch[:, 6] = 0.5, -1.0
    ch[10:, 6] = 1.0
    w = hf.plan_path([0.1, 0, 0], ch)
    assert w.shape == (51, 3) and w[10] == pytest.approx([0.1 + 10 * 0.5 * orx.G_POS, 0, 0])
    assert hf.close_index(ch) == 10 and hf.close_index(ch[:, [0, 1, 2, 6]]) == 10  # trace format: grip last
    assert hf.close_index(np.zeros((5, 7))) is None
    n = ch.copy()
    n[8:, 6], n[:8, 6] = 1.0, -1.0
    diff, fb = hf.target_diff((10, [0.1, 0.02, 0], n), (0, [0.1, 0, 0], ch), 20)
    assert not fb and diff == pytest.approx([(8 - 10) * 0.5 * orx.G_POS, 0.02, 0])
    r = ch.copy()
    r[:, 6] = -1.0  # no close in R: common time t_obs_N + K
    diff, fb = hf.target_diff((10, [0.1, 0.02, 0], n), (0, [0.1, 0, 0], r), 20)
    assert fb and diff == pytest.approx([(20 - 30) * 0.5 * orx.G_POS, 0.02, 0])
    diff, _ = hf.target_diff((45, [0.1, 0, 0], n), (0, [0.1, 0, 0], r), 30)  # R's index clamped to its end (50)
    assert diff[0] == pytest.approx((30 - 50) * 0.5 * orx.G_POS)


def test_kappa_hat_dead_band_and_clip():
    assert hf.kappa_hat(0.005, 0.03, 0.01) == 0.0  # inside the dead band
    assert hf.kappa_hat(0.015, 0.03, 0.01) == pytest.approx(0.5)
    assert hf.kappa_hat(0.05, 0.03, 0.01) == 1.0 and hf.kappa_hat(-0.02, 0.03, 0.01) == 0.0
    assert hf.kappa_hat(0.02, 0.0, 0.01) == 0.0


def test_early_chunks():
    assert hf.early(None, 10, 0) is None
    assert hf.early(7, 10, 0) and hf.early(5, 10, 0) and not hf.early(4, 10, 0)  # first plan after at 10
    assert not hf.early(10, 10, 0)  # the call at 10 observes pre-shift; the next is at 20
    assert not hf.early(19, 20, 10)  # D: arrives at 30 > 24


def test_tracker_lag_noise_dropout_and_ema():
    ps = [np.array([0.01 * t, 0, 0]) for t in range(10)]
    tr = hf.Tracker(1, 0.0, 3, 0.0)
    out = [tr(p) for p in ps]
    assert out[1] == pytest.approx(ps[0]) and out[5] == pytest.approx(ps[2])  # p(t - 3), clamped at 0
    a, b = hf.Tracker(7, 0.01, 0, 0.0), hf.Tracker(7, 0.02, 0, 0.0)  # one noise stream across sigma
    for p in ps:
        assert b(p) - p == pytest.approx(2 * (a(p) - p))
    d = hf.Tracker(7, 0.0, 0, 1.0)  # always dropped after the first: holds
    assert all(d(p) == pytest.approx(ps[0]) for p in ps)
    e = hf.Tracker(7, 0.0, 0, 0.0, beta=0.5)
    e(np.zeros(3))
    assert e(np.array([0.02, 0, 0]))[0] == pytest.approx(0.01)
    # warm start: the first estimate is already smoothed, so the first plan's reference is not a raw noisy read
    err = lambda beta: np.mean([np.linalg.norm(hf.Tracker(s, 0.01, 0, 0.0, beta)(np.zeros(3))) for s in range(200)])
    assert err(0.2) < 0.5 * err(1.0)


def test_push_filter_surrogate_and_contact():
    f = hf.PushFilter("surr", r_c=0.05, direction=True)
    f(np.zeros(3), np.array([-0.02, 0, 0]), False)
    seen = f(np.array([0.003, 0, 0]), np.array([-0.018, 0, 0]), False)  # pushed the way the arm moves, arm near
    assert f.flag and seen == pytest.approx(np.zeros(3))
    seen = f(np.array([0.003, 0.03, 0]), np.array([-0.016, 0, 0]), False)  # external jump sideways: kept
    assert not f.flag and seen == pytest.approx([0, 0.03, 0])
    g = hf.PushFilter("surr", r_c=0.05, direction=False)
    g(np.zeros(3), np.array([-0.02, 0, 0]), False)
    g(np.array([0, 0.03, 0]), np.array([-0.018, 0, 0]), False)
    assert g.flag  # without the direction condition the same jump is swallowed
    c = hf.PushFilter("contact")
    c(np.zeros(3), np.zeros(3), True)
    c(np.array([0.01, 0, 0]), np.zeros(3), False)  # contact at t - 1 marks the increment
    assert c.flag and c.P == pytest.approx([0.01, 0, 0])
    c(np.array([0.02, 0, 0]), np.zeros(3), False)
    assert not c.flag


def test_alpha_topup_ramp_clip_and_close():
    ch = np.zeros((50, 7))
    ch[:, 0], ch[:, 6] = 0.8, -1.0
    ch[20:, 6] = 1.0
    out = hf.alpha_topup(ch, [0.0, 0.03, 0.0], 2, 5)
    extra = orx.G_POS * (out[:, :3] - np.clip(ch[:, :3], -1, 1))
    assert np.allclose(extra[:2], 0) and extra[:, 1].sum() == pytest.approx(0.03)
    assert extra[:, 1].max() <= 0.03 / 5 + 1e-12 and np.allclose(extra[:, 0], 0)
    out = hf.alpha_topup(ch, [0.06, 0.0, 0.0], 0, 5)  # x has 0.2 units of room per step for 20 steps: 4.4 of 6 cm
    assert np.abs(out[:, :3]).max() <= 1 and np.allclose(out[20:], ch[20:])
    assert orx.G_POS * (out[:20, 0] - 0.8).sum() == pytest.approx(20 * 0.2 * orx.G_POS)


@pytest.mark.parametrize("method", ["none", "T0", "G", "GT", "PPC"])
def test_hagent_matches_c1e2_agent_on_common_methods(method):
    kw = dict(moves={5: np.array([0.0, 0.06, 0.0])})
    _, _, _, a_new, _ = _episode(method, "blind_abs", **kw)
    _, _, _, a_old, _ = _episode(method, "blind_abs", cls=orx.Agent, **kw)
    assert np.array_equal(a_new, a_old)
    kw |= dict(s=25, d=20)  # cell F, far object: the scheduled and triggered (T0, GT) plans arrive 20 steps late
    assert np.array_equal(_far(method, "blind_abs", **kw)[3], _far(method, "blind_abs", cls=orx.Agent, **kw)[3])


def test_noisy_eye_dead_bands_are_per_agent():
    """eps and PPC's speed threshold come from the row (spec §7 choice 4); a 8 mm shift passes below 1 cm."""
    kw = dict(moves={5: np.array([0.0, 0.008, 0.0])})
    none = _episode("none", "blind_abs", **kw)[3]
    for m in ("G", "GR"):
        assert not np.array_equal(_episode(m, "blind_abs", **kw)[3], none)
        x, _, ag, a, _ = _episode(m, "blind_abs", eps=0.01, **kw)
        assert np.array_equal(a, none) and ag.t_e is None
    assert not np.array_equal(_episode("PPC", "blind_abs", **kw)[3], none)
    assert np.array_equal(_episode("PPC", "blind_abs", ppc_v_min=0.01, **kw)[3], none)


def test_gr_lands_under_every_brain_and_g_gkeep_do_not():
    new = P0 + [0, 0.03, 0]
    miss = {b: {m: np.linalg.norm(_episode(m, b)[0] - new) for m in ("G", "Gkeep", "GR")} for b in BRAINS}
    assert all(miss[b]["GR"] < 1e-6 for b in BRAINS)
    assert miss["blind_abs"]["G"] == pytest.approx(0.03) and miss["blind_rel"]["G"] == pytest.approx(0.006)
    assert miss["blind_abs"]["Gkeep"] < 1e-6
    assert miss["blind_rel"]["Gkeep"] == pytest.approx(0.024) and miss["seeing"]["Gkeep"] == pytest.approx(0.03)


def test_gr_reference_is_the_plan_in_flight_and_kappa_matches_the_carried_share():
    _, _, ag, _, sb = _episode("GR", "blind_rel")  # s = 10, d = 0: R is the executing plan
    assert ag.t_e == 6 and ag.ref[0] == 0
    k = ag.kappa_log[0]
    s = sb.s_at[k["t"]]  # G's extra delivered on plan 0 before plan 10 arrived: 4 of 5 ramp steps
    assert k["t"] == 10 and not k["fb"] and k["k_hat"] == pytest.approx(0.8) and s[1] == pytest.approx(0.024)
    x, _, ag, _, _ = _episode("GR", "blind_rel", s=10, d=5, moves={12: np.array([0, 0.03, 0])}, t_fire=12)
    assert ag.t_e == 13 and ag.ref[0] == 10  # observed at 10, in flight until 15
    assert ag.kappa_log == [] and np.linalg.norm(x - P0 - [0, 0.03, 0]) < 1e-6  # plan 10 is G's; closed before 25


def test_seeing_brain_alpha_one_turns_none_into_seeing_and_adds_nothing_after_g():
    new = P0 + [0, 0.03, 0]
    assert np.linalg.norm(_episode("none", "blind_rel")[0] - new) == pytest.approx(0.03)
    assert np.linalg.norm(_episode("none", "blind_rel", alpha=1.0)[0] - new) < 1e-6
    assert np.linalg.norm(_episode("none", "blind_rel", alpha=0.5)[0] - new) == pytest.approx(0.015)
    for m in ("G", "GR"):
        assert np.linalg.norm(_episode(m, "blind_rel", alpha=1.0)[0] - new) < 1e-6
    x, _, ag, _, _ = _episode("GR", "blind_rel", alpha=1.0)
    assert ag.kappa_log[0]["k_hat"] == pytest.approx(1.0)
    assert np.linalg.norm(_episode("Gkeep", "blind_rel", alpha=1.0)[0] - new) == pytest.approx(0.03)


def test_t_variants_query_once_on_the_raw_shift():
    kw = dict(s=50, moves={5: np.array([0.0, 0.06, 0.0])})  # s = 50: the triggered plan runs to its close
    new = P0 + [0, 0.06, 0]
    x, _, ag, _, _ = _episode("GRT", "blind_abs", **kw)
    assert ag.sched.n_trig == 1 and np.linalg.norm(x - new) < 1e-6
    x, _, ag, _, _ = _episode("GT", "blind_abs", **kw)
    assert ag.sched.n_trig == 1 and np.linalg.norm(x - P0) < 1e-6  # G re-anchored on the triggered plan: T's harm
    assert _episode("GkeepT", "blind_abs", **kw)[2].sched.n_trig == 1
    assert _episode("GRT", "blind_abs")[2].sched.n_trig == 0  # 3 cm: no switch


def test_filter_ignores_the_arm_pushing_and_noise_delays_engagement():
    push = {t: np.array([0.003, 0.0, 0.0]) for t in (3, 4, 5)}  # 9 mm, the way the arm walks, arm within 10 cm
    ag = hf.HAgent("GR", 10, 0, push=hf.PushFilter("surr", r_c=0.10, direction=True))
    bare = hf.HAgent("GR", 10, 0)
    x, p = np.zeros(3), P0.copy()
    for t in range(12):
        for a in (ag, bare):
            a.observe(t, x, p, False)
            if a.sched.wants_call(t, False):
                a.sched.issue(t, _walk(x, P0, v=0.1))
                a.sched.arrive(t)
        p = p + push.get(t, 0.0)
        x = x + orx.G_POS * 0.1 * np.array([1.0, 0, 0])
    assert bare.t_e == 5 and ag.t_e is None  # 6 mm seen at 5 without the filter; pushes subtracted with it
    oracle = _episode("GR", "blind_abs")[2].t_e
    lagged = _episode("GR", "blind_abs", tracker=hf.Tracker(0, 0.0, 3, 0.0))[2].t_e
    assert (oracle, lagged) == (6, 9)


def _carry(method, shift=0.03, dz=0.0, open_at=47):
    """Pick and place with a replaying brain: walk to P0, close, lift 4.4 cm, carry 5.5 cm along x, open at open_at.
    The shift (shift along y, dz up) lands before step 5's physics; after the close the object moves with the gripper.
    Returns the object's offset from the place target at the first open command, and the agent."""
    traj = np.zeros((100, 7))
    traj[:, 6] = -1.0
    traj[:20] = _walk(np.zeros(3), P0, n=20)  # reaches P0 at index 18, closes at 19
    traj[19:open_at, 6] = 1.0
    traj[20:28, 2], traj[28:38, 0] = 0.5, 0.5
    ag, x, p, held = hf.HAgent(method, 10, 0), np.zeros(3), P0.copy(), None
    for t in range(60):
        ag.observe(t, x, p, False)
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, traj[t: t + 50])
            ag.sched.arrive(t)
            ag.on_new_chunk()
        a = np.clip(ag.act(t, ag.sched.action(t)), -1, 1)
        if held is not None and a[6] <= 0:
            return p - (P0 + orx.G_POS * np.array([10 * 0.5, 0, 8 * 0.5])), ag  # the memorised place point
        if t == 5:
            p = p + [0, shift, dz]
        x = x + orx.G_POS * a[:3]
        if a[6] > 0:
            held = p - x if held is None else held
            p = x + held


def test_return_sends_back_the_reflex_extra_after_the_lift():
    off, ag = _carry("GR")
    assert off == pytest.approx([0, 0.03, 0], abs=1e-6)  # the replayed transport carries the arm's offset
    off, ag = _carry("GRret")
    assert off == pytest.approx([0, 0, 0], abs=1e-6) and ag.t_lift == 19 + 1 + 6  # 6 lift steps = 3.3 cm >= 3 cm
    assert np.linalg.norm(ag.c) == pytest.approx(0.03)
    off0, ag0 = _carry("GRret", shift=0.0)
    assert off0 == pytest.approx([0, 0, 0], abs=1e-9) and np.linalg.norm(ag0.c) == 0  # nothing to return


def test_parse_arm():
    assert hf.parse_arm("GR") == ("GR", "", "")
    assert hf.parse_arm("GR+surr") == ("GR", "surr", "")
    assert hf.parse_arm("PPC@real") == ("PPC", "", "real")
    for bad in ("GR@loud", "GRret+surr", "GRret+contact", "GRret@real"):  # the return: oracle only (spec rows 27, 28)
        with pytest.raises(AssertionError):
            hf.parse_arm(bad)


def test_engagement_on_an_arrival_step_is_checked_against_the_executing_plan():
    """Journal item 3: the shift lands before step 9's physics, so observe(10) sees it first, on the step plan 10 is
    observed and arrives (d = 0). Engagement is checked before plan 10 is installed, against plan 0: t_e = 10, R =
    plan 0, and plan 10 is the first N. Against plan 10, U would be 0 and GR would never engage."""
    kw = dict(moves={9: np.array([0.0, 0.03, 0.0])}, t_fire=9)
    new = P0 + [0, 0.03, 0]
    x, _, ag, _, _ = _episode("GR", "blind_rel", **kw)
    assert ag.t_e == 10 and ag.ref[0] == 0 and np.array_equal(ag.p_ref, P0)
    k = ag.kappa_log[0]
    assert k["t"] == 10 and not k["fb"] and k["k_hat"] == 0.0 and k["d"] == pytest.approx([0, 0.03, 0])
    assert np.linalg.norm(x - new) < 1e-6  # plan 10 replays the old motion from an unshifted arm: kappa_hat = 0
    x, _, ag, _, _ = _episode("G", "blind_rel", **kw)
    assert ag.t_e == 10 and np.linalg.norm(x - new) == pytest.approx(0.03)  # G re-anchors on plan 10: U = 0


def test_kappa_at_d_positive_with_the_reference_in_flight_and_its_fallback():
    """s = 10, d = 5, shift before step 12's physics: t_e = 13 on plan 0, R = plan 10 (observed pre-shift, in flight
    until 15), N = plan 20 (arrives at 25, before the close at 55). A replaying brain carries G's 3 cm on the arm:
    kappa_hat = 1 from the grasp points, and from the common time t_obs_N + K (every K) when no plan closes."""
    kw = dict(s=10, d=5, moves={12: np.array([0.0, 0.03, 0.0])}, t_fire=12)
    x, p, ag, _, _ = _far("GR", "blind_rel", **kw)
    k = ag.kappa_log[0]
    assert ag.t_e == 13 and ag.ref[0] == 10 and np.array_equal(ag.p_ref, FAR)
    assert k["t"] == 20 and not k["fb"] and k["k_hat"] == 1.0 and np.linalg.norm(x - p) < 1e-6
    x, p, ag, _, _ = _far("GR", "blind_rel_open", steps=60, **kw)
    k = ag.kappa_log[0]
    assert ag.t_e == 13 and ag.ref[0] == 10 and k["t"] == 20 and k["fb"] and k["k_hat"] == 1.0
    assert k["m"] == {"10": 0.03, "20": 0.03, "30": 0.03} and np.linalg.norm(x - p) < 1e-6


def test_journal_13_plan_path_counts_the_steps_the_arm_never_runs():
    """Journal item 13, first effect (kept as in spec §5): the plan path starts at index 0, but at d = 5 the arm never
    runs indices 0-4. The absolute-target brain pulls back to the old point there, so kappa_hat = 0 while the executed
    part already carries the shift, and GR adds it again: G misses by -0.25 cm, GR and Gkeep overshoot by +2.25 cm."""
    kw = dict(s=10, d=5, moves={12: np.array([0.0, 0.03, 0.0])}, t_fire=12)
    y = {}
    for m in ("none", "G", "GR", "Gkeep"):
        x, p, ag, _, _ = _far(m, "blind_abs", **kw)
        y[m] = (x - p)[1]
        assert all(k["k_hat"] == 0.0 and not k["fb"] for k in ag.kappa_log)
    assert y == pytest.approx({"none": -0.03, "G": -0.0025, "GR": 0.0225, "Gkeep": 0.0225}, abs=1e-9)


def test_journal_13_seeing_brain_with_t_counts_the_shift_twice():
    """Journal item 13, second effect (kept): s = 25, d = 20, alpha = 1, 6 cm shift before step 12's physics. T
    re-queries at 13 with S = 0, so the seeing brain adds the whole shift from index d = 20; meanwhile the reflex
    delivers it on plan 0, and plan 13 arrives at 33, before the close at 55: every T variant overshoots by 6 cm.
    Without T the reflex and the brain land (Gkeep overshoots anyway: kappa = 0 re-adds what a seeing plan carries).
    GRT's kappa_hat is 0: plan 0 has no close, and the fallback at K <= d compares only steps the arm never runs."""
    kw = dict(s=25, d=20, alpha=1.0, moves={12: np.array([0.0, 0.06, 0.0])}, t_fire=12)
    y, runs = {}, {}
    for m in ("none", "T0", "G", "GR", "Gkeep", "GT", "GRT", "GkeepT"):
        runs[m] = _far(m, "blind_rel", **kw)
        y[m] = (runs[m][0] - runs[m][1])[1]
    assert y == pytest.approx({"none": 0, "T0": 0, "G": 0, "GR": 0, "Gkeep": 0.06, "GT": 0.06, "GRT": 0.06,
                               "GkeepT": 0.06}, abs=1e-9)
    _, _, ag, _, sb = runs["GRT"]
    k = ag.kappa_log[0]
    assert ag.sched.n_trig == 1 and k["t"] == 13 and k["fb"] and k["k_hat"] == 0.0 and not sb.s_at[13].any()
    assert k["m"] == {"10": 0.0, "20": 0.0, "30": 0.06}


def test_seeing_brain_gating_and_copy():
    """spec §4, journal item 8: a plan observed at t_fire is still pre-shift (no top-up); from t_fire + 1 the top-up
    starts at the answer's first executed index (d on a running schedule, 0 on the first call); none after the first
    close command or in control. chunk() hands out a copy: editing it in place cannot corrupt S."""
    pert = SimpleNamespace(ep=SimpleNamespace(kind="step"), t_fire=10, p_pre=P0.copy())
    raw, seen, sb = np.zeros((50, 7)), P0 + [0, 0.03, 0], hf.SeeingBrain(1.0)
    raw[:, 6] = -1.0
    out = sb.chunk(10, raw, pert, seen, 0, False)
    a0 = out[0].copy()  # the arm gets the brain's action ...
    out[0, :3] = 0.5  # ... and the handed-out chunk is edited in place afterwards
    sb.executed(10, 10, a0, pert)
    assert np.array_equal(a0, raw[0]) and not sb.S.any()
    extra = orx.G_POS * (sb.chunk(11, raw, pert, seen, 5, False)[:, :3] - raw[:, :3])
    assert np.allclose(extra[:5], 0) and extra[5, 1] == pytest.approx(0.006) and extra[:, 1].sum() == pytest.approx(0.03)
    assert orx.G_POS * sb.chunk(12, raw, pert, seen, 0, False)[0, 1] == pytest.approx(0.006)
    assert np.array_equal(sb.chunk(13, raw, pert, seen, 0, True), raw)  # after the first close command
    ctl = SimpleNamespace(ep=SimpleNamespace(kind="control"), t_fire=10, p_pre=P0.copy())
    assert np.array_equal(sb.chunk(14, raw, ctl, seen, 0, False), raw)


def test_return_edge_cases():
    """Open before the lift: no lift, nothing returned (as GR). Open mid-return (lift at 26, open at 27): one 6 mm ramp
    step went back, 2.4 cm stay unsent and nothing more is sent. The lift counts from the object's height at the close
    command: an object 2 cm lower (slid off its stand at the shift) lifts at 26; from the start height it never would
    (4.4 cm of lift reach 2.4 cm)."""
    off, ag = _carry("GRret", open_at=22)
    assert ag.t_lift is None and ag.ret_left is None and ag.ret_cap is None and ag.released
    assert off == pytest.approx(_carry("GR", open_at=22)[0])
    off, ag = _carry("GRret", open_at=27)
    assert ag.t_lift == 26 and ag.released and ag.ret_left == pytest.approx([0, -0.024, 0])
    a = np.r_[np.zeros(6), 1.0]
    assert np.array_equal(ag.act(28, a.copy()), a) and ag.ret_left == pytest.approx([0, -0.024, 0])
    off, ag = _carry("GRret", dz=-0.02)
    assert ag.z_close == pytest.approx(-0.02) and ag.t_lift == 26 and off == pytest.approx([0, 0, 0], abs=1e-6)


def test_tracker_dropout_holds_the_estimate_and_lag_with_warm_start():
    ps = [np.array([0.01 * t, 0, 0]) for t in range(20)]
    out = [float(v[0]) for v in map(hf.Tracker(3, 0.0, 0, 0.5, beta=0.5), ps)]
    held = [t for t in range(1, 20) if out[t] == out[t - 1]]  # the object moves: an update always changes the estimate
    assert 0 < len(held) < 19 and all(out[t] != ps[t][0] for t in held)  # a drop holds the estimate, not p
    assert all(out[t] == pytest.approx(0.5 * ps[t][0] + 0.5 * out[t - 1]) for t in range(1, 20) if t not in held)
    tr = hf.Tracker(0, 0.0, 3, 0.0, 0.5)  # warm on p(0), lag clamped at the first step, then EMA of p(t - 3)
    assert [float(tr(p)[0]) for p in ps[:8]] == pytest.approx([0, 0, 0, 0, 0.005, 0.0125, 0.02125, 0.030625])
    lag, ref = hf.Tracker(5, 0.01, 3, 0.1, 0.3), hf.Tracker(5, 0.01, 0, 0.1, 0.3)  # one noise stream
    assert all(np.array_equal(lag(p), ref(ps[max(t - 3, 0)])) for t, p in enumerate(ps))


def test_push_filter_contact_at_t_far_arm_and_distance_to_the_last_position():  # no box: the origin p(t - 1)
    c = hf.PushFilter("contact")
    c(np.zeros(3), np.zeros(3), False)
    c(np.array([0.01, 0, 0]), np.zeros(3), True)  # contact at t alone marks the increment
    assert c.flag and c.P == pytest.approx([0.01, 0, 0])
    f = hf.PushFilter("surr", r_c=0.05)
    f(np.zeros(3), np.array([0.20, 0, 0]), False)
    assert not f.flag and f(np.array([0.01, 0, 0]), np.array([0.20, 0, 0]), False) == pytest.approx([0.01, 0, 0])
    f(np.array([0.03, 0, 0]), np.array([0.07, 0, 0]), False)  # 6 cm from p(t - 1) = 0.01, 4 cm from p(t): far
    assert not f.flag
    f(np.array([-0.01, 0, 0]), np.array([0.07, 0, 0]), False)  # 4 cm from p(t - 1) = 0.03, 8 cm from p(t): near
    assert f.flag and f.P == pytest.approx([-0.04, 0, 0])


def test_episode_seed_string_and_control_radius():
    e = hf.make_episode("libero_goal", 4, 2, "close")
    seed = int.from_bytes(hashlib.sha256(b"c1e3/libero_goal/4/2/close").digest()[:4], "little")
    assert e.seed == seed == 2813226981
    rs = [hf.make_episode("libero_spatial", 0, i, "control").r for i in range(50)]
    assert all(0.08 <= r < 0.20 for r in rs) and min(rs) < 0.10 and max(rs) > 0.18
