import math

import numpy as np
import pytest

import objreflex as orx


def test_episode_is_deterministic_and_shared():
    a = orx.make_episode("libero_spatial", 3, 5, "step")
    b = orx.make_episode("libero_spatial", 3, 5, "step")
    assert a == b and a.seed == 2082101548
    assert 0.08 <= a.r <= 0.20 and 0.0 <= a.angle < 2 * math.pi
    lo, hi = orx.MAG_CLASSES[a.mag_class]
    assert lo <= a.mag <= hi
    assert abs(np.linalg.norm(a.delta) - a.mag) < 1e-12 and a.delta[2] == 0.0


def test_magnitude_classes_balanced_over_the_grid():
    classes = [orx.make_episode(s, t, i, "step").mag_class for s, t in orx.TASKS for i in range(10)]
    counts = np.bincount(classes, minlength=3)
    assert counts.sum() == 260 and counts.max() - counts.min() <= 1


def test_control_and_smooth():
    c = orx.make_episode("libero_goal", 8, 0, "control")
    assert c.mag == 0.0 and c.mag_class == -1
    sm = orx.make_episode("libero_goal", 8, 0, "smooth")
    assert 0.03 <= sm.mag <= 0.06 and sm.mag_class == -1
    assert orx.Perturbation(c).dq(0, np.zeros(3), np.zeros(3), grasp_started=False) is None


def test_perturbation_fires_once_when_close_and_open():
    ep = orx.make_episode("libero_object", 0, 1, "step")
    p = orx.Perturbation(ep)
    obj = np.zeros(3)
    far = obj + np.array([ep.r + 0.01, 0, 0])
    near = obj + np.array([ep.r - 0.01, 0, 0])
    assert p.dq(0, far, obj, grasp_started=False) is None
    assert p.dq(1, near, obj, grasp_started=True) is None  # blocked while the caller's (sticky) close flag is set
    dq = p.dq(2, near, obj, grasp_started=False)
    assert np.allclose(dq, ep.delta) and p.t_fire == 2 and np.allclose(p.p_pre, obj)
    assert p.dq(3, near, obj, grasp_started=False) is None


def test_smooth_perturbation_spreads_over_20_steps():
    ep = orx.make_episode("libero_object", 0, 1, "smooth")
    p = orx.Perturbation(ep)
    total, n = np.zeros(3), 0
    for t in range(40):
        dq = p.dq(t, np.zeros(3), np.zeros(3), grasp_started=False)
        if dq is not None:
            total, n = total + dq, n + 1
    assert np.allclose(total, ep.delta) and n == orx.SMOOTH_STEPS


def test_smooth_perturbation_stops_at_the_first_close_command():
    p = orx.Perturbation(orx.make_episode("libero_object", 0, 1, "smooth"))
    assert p.dq(0, np.zeros(3), np.zeros(3), grasp_started=False) is not None
    assert p.dq(1, np.zeros(3), np.zeros(3), grasp_started=True) is None


def test_schedule_without_delay():
    s = orx.Schedule(10, 0)
    assert s.wants_call(0, False)
    s.issue(0, np.arange(50)[:, None] * np.ones((1, 7)))
    assert s.arrive(0) and s.action(0)[0] == 0 and s.action(9)[0] == 9
    assert not s.wants_call(9, False) and s.wants_call(10, False)
    assert (s.n_sched, s.n_trig) == (1, 0)


def test_schedule_with_delay_keeps_old_chunk_until_arrival():
    s = orx.Schedule(10, 5)
    s.issue(0, np.zeros((50, 7)))
    assert s.arrive(0)  # the first call arrives at once
    s.issue(10, np.ones((50, 7)))
    for t in range(10, 15):
        assert not s.arrive(t) and s.action(t)[0] == 0  # old chunk, index t - 0
    assert s.arrive(15) and s.t_obs == 10 and s.action(15)[0] == 1  # new chunk from index 5
    assert s.next_call == 20


def test_triggered_call_resets_schedule_and_counts():
    s = orx.Schedule(50, 0)
    s.issue(0, np.zeros((50, 7)))
    s.arrive(0)
    assert s.wants_call(7, True)
    s.issue(7, np.zeros((50, 7)))
    assert s.arrive(7) and s.next_call == 57 and (s.n_sched, s.n_trig) == (1, 1)


def test_no_second_call_in_flight():
    s = orx.Schedule(10, 20)
    s.issue(0, np.zeros((50, 7)))
    s.arrive(0)
    s.issue(10, np.zeros((50, 7)))
    assert not s.wants_call(11, True)


def test_schedule_rejects_chunk_overrun():
    with pytest.raises(AssertionError):
        orx.Schedule(40, 20)
    with pytest.raises(AssertionError):
        orx.Schedule(10, 30)


def test_schedule_saturated_brain_when_delay_exceeds_period():
    s = orx.Schedule(10, 20)
    calls, max_idx = [], 0
    for t in range(80):  # the runner's order
        s.arrive(t)
        if s.wants_call(t, False):
            s.issue(t, np.zeros((50, 7)))
            calls.append(t)
            s.arrive(t)
        s.action(t)
        max_idx = max(max_idx, t - s.t_obs)
        if t > 10:
            assert not s.wants_call(t, True)  # a trigger cannot get through: the brain is saturated
    assert calls == [0, 10, 30, 50, 70] and s.n_trig == 0 and s.n_sched == 5
    assert max_idx == 39


def _brain(x, p_seen, n=50):  # a chunk that walks the arm straight to p_seen, 1 cm per step at most
    ch, xx = np.zeros((n, 7)), x.copy()
    for j in range(n):
        a = np.clip((p_seen - xx) / orx.G_POS, -1, 1)
        ch[j, :3], ch[j, 6] = a, -1.0
        xx = xx + orx.G_POS * a
    return ch


def _run(method, s=50, d=0, shift=np.array([0.03, 0.0, 0.0]), t_shift=5, steps=50, t_ramp=5, k_p=1.0):
    ag = orx.Agent(method, s, d, t_ramp=t_ramp, k_p=k_p)
    x, p, acts = np.zeros(3), np.array([0.10, 0.0, 0.0]), []
    for t in range(steps):
        if t == t_shift:
            ag.p_pre = p.copy()
            p = p + shift
        ag.observe(t, x, p)
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, _brain(x, p))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        a = ag.act(t, ag.sched.action(t))
        acts.append(a)
        x = x + orx.G_POS * a[:3]
    return x, p, np.array(acts), ag


def test_none_goes_to_the_old_place_and_g_to_the_new_one():
    x_none, p, _, _ = _run("none")
    x_g, _, _, ag = _run("G")
    assert abs(x_none[0] - 0.10) < 1e-6  # the brain never re-looks (s = 50) and misses by 3 cm
    assert np.linalg.norm(x_g - p) < 1e-3  # geometry follows the shift
    assert ag.sched.n_trig == 0


def test_g_is_identity_without_shift_and_inside_dead_band():
    _, _, a_none, _ = _run("none", shift=np.zeros(3))
    _, _, a_g, _ = _run("G", shift=np.zeros(3))
    assert np.array_equal(a_none, a_g)
    _, _, a_g4, _ = _run("G", shift=np.array([0.004, 0, 0]))
    _, _, a_n4, _ = _run("none", shift=np.array([0.004, 0, 0]))
    assert np.array_equal(a_n4, a_g4)


def test_g_ramp_limits_the_step_and_delivers_the_whole_shift():
    # s = 50: both runs execute the same chunk, so a_G - a_none is exactly G's extra command
    _, _, a_none, _ = _run("none", t_ramp=10, k_p=1.0)
    _, _, a_g, _ = _run("G", t_ramp=10, k_p=1.0)
    extra = orx.G_POS * (a_g[:, 0] - a_none[:, 0])
    assert extra.max() <= 0.03 / 10 + 1e-9  # at most |U| / T_ramp per step
    assert extra.sum() == pytest.approx(0.03, abs=1e-6)


def test_t0_requeries_and_then_reaches_the_new_place():
    x, p, _, ag = _run("T0")
    assert ag.sched.n_trig == 1 and np.linalg.norm(x - p) < 1e-3


def test_gt_switch_only_for_big_shifts():
    _, _, _, small = _run("GT", shift=np.array([0.03, 0, 0]))
    _, _, _, big = _run("GT", shift=np.array([0.06, 0, 0]))
    assert small.sched.n_trig == 0 and big.sched.n_trig == 1


def test_g_off_after_close_command():
    ag = orx.Agent("G", 50, 0, t_ramp=1, k_p=1.0)
    ag.observe(0, np.zeros(3), np.zeros(3))
    ag.sched.issue(0, np.zeros((50, 7)))
    ag.sched.arrive(0)
    ag.act(0, np.array([0, 0, 0, 0, 0, 0, 1.0]))  # close
    ag.observe(1, np.zeros(3), np.array([0.03, 0, 0]))
    assert np.array_equal(ag.act(1, np.zeros(7)), np.zeros(7))


def test_gpost_keeps_shifting_new_chunks_and_g_does_not():
    # the brain re-looks every 10 steps but "ignores" the shift: it keeps aiming at the pre-shift place
    def run(method):
        ag = orx.Agent(method, 10, 0, t_ramp=1, k_p=1.0)
        x, p0 = np.zeros(3), np.array([0.10, 0.0, 0.0])
        p = p0
        for t in range(60):
            if t == 5:
                ag.p_pre, p = p.copy(), p0 + np.array([0.03, 0, 0])
            ag.observe(t, x, p)
            if ag.sched.arrive(t):
                ag.on_new_chunk()
            if ag.sched.wants_call(t, ag.trigger(t)):
                ag.sched.issue(t, _brain(x, p0))
                if ag.sched.arrive(t):
                    ag.on_new_chunk()
            x = x + orx.G_POS * ag.act(t, ag.sched.action(t))[:3]
        return x, p
    xg, p = run("G")
    xp, _ = run("Gpost")
    assert np.linalg.norm(xg - np.array([0.10, 0, 0])) < 2e-3  # G re-anchors and follows the stale brain
    assert np.linalg.norm(xp - p) < 2e-3


def test_g_counts_only_the_extra_actually_sent_under_saturation():
    ag = orx.Agent("G", 50, 0, t_ramp=1, k_p=1.0)
    x, target = np.zeros(3), np.array([0.30, 0.0, 0.0])  # far: the plan runs at +1 for ~27 steps
    p = target.copy()
    for t in range(50):
        if t == 5:
            ag.p_pre, p = p.copy(), target + np.array([0.03, 0, 0])
        ag.observe(t, x, p)
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, _brain(x, target))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        a = ag.act(t, ag.sched.action(t))
        if t == 10:
            assert ag.c[0] == 0.0  # saturated at +1: nothing sent, nothing counted in E
        x = x + orx.G_POS * a[:3]
    assert np.linalg.norm(x - p) < 1e-3  # the whole shift arrives once the plan leaves saturation


def test_g_does_not_count_phantom_extra_beyond_the_action_limit():
    ag = orx.Agent("G", 50, 0, t_ramp=1, k_p=1.0)
    p = np.array([0.10, 0.0, 0.0])
    ag.observe(0, np.zeros(3), p)
    ag.sched.issue(0, np.tile([1.3, 0, 0, 0, 0, 0, -1.0], (50, 1)))  # the brain emits beyond the +1 limit
    ag.sched.arrive(0)
    ag.act(0, ag.sched.action(0))
    ag.p_pre, p = p.copy(), p + np.array([0.03, 0, 0])
    for t in range(1, 5):
        ag.observe(t, np.zeros(3), p)
        ag.act(t, ag.sched.action(t))
    assert ag.c[0] == 0.0  # nothing sent beyond +1, nothing counted


def test_triggers_off_after_close_command():
    ag = orx.Agent("T0", 50, 0)
    ag.observe(0, np.zeros(3), np.zeros(3))
    ag.sched.issue(0, np.zeros((50, 7)))
    ag.sched.arrive(0)
    ag.act(0, np.array([0, 0, 0, 0, 0, 0, 1.0]))  # close
    ag.observe(1, np.zeros(3), np.array([0.03, 0, 0]))
    assert ag.trigger(1) is False


def test_g_leaves_rotation_and_gripper_untouched():
    ag = orx.Agent("G", 50, 0, t_ramp=1, k_p=1.0)
    x, p = np.zeros(3), np.array([0.10, 0.0, 0.0])
    ch = _brain(x, p)
    ch[:, 3:6] = [0.2, -0.3, 0.4]
    for t in range(20):
        if t == 5:
            ag.p_pre, p = p.copy(), p + np.array([0.03, 0, 0])
        ag.observe(t, x, p)
        if t == 0:
            ag.sched.issue(0, ch)
            ag.sched.arrive(0)
        plan = ag.sched.action(t)
        a = ag.act(t, plan)
        assert np.array_equal(a[3:7], plan[3:7])
        x = x + orx.G_POS * a[:3]
    assert ag.g_on


def test_g_carries_its_extra_across_a_delayed_reanchor():
    # the shift lands while a pre-shift call is in flight; the brain never moves, so the arm moves only by G's extra
    for s, d in ((10, 10), (10, 20)):
        ag, p = orx.Agent("G", s, d, t_ramp=5, k_p=1.0), np.array([0.10, 0, 0])
        for t in range(60):
            if t == 12:
                p = p + np.array([0.03, 0, 0])
            ag.observe(t, np.zeros(3), p)
            if ag.sched.arrive(t):
                ag.on_new_chunk()
            if ag.sched.wants_call(t, ag.trigger(t)):
                ag.sched.issue(t, np.zeros((50, 7)))
                if ag.sched.arrive(t):
                    ag.on_new_chunk()
            ag.act(t, ag.sched.action(t))
        assert ag.c[0] == pytest.approx(0.03)  # sent once, not again after the chunk observed at 10 arrives


def test_ppc_fibonacci_profile_and_pace():
    assert orx.ppc_profile(2) == pytest.approx([0.8, 0.6])
    alpha, k_exec = orx.ppc_pace(v=np.array([0.02, 0, 0]), dp=np.array([0.01, 0, 0]), s=50)
    assert alpha == pytest.approx(3.0) and k_exec == 17  # ceil(50 / 3)
    alpha, k_exec = orx.ppc_pace(v=np.array([-0.02, 0, 0]), dp=np.array([0.01, 0, 0]), s=50)
    assert k_exec == 50  # moving against the plan: no pace change


def _ppc_agent(p0, x=np.zeros(3), method="PPC"):  # a PPC agent at step 0 with a chunk moving the arm +x (0.7)
    ag = orx.Agent(method, 50, 0)
    ag.observe(0, x, p0)
    ag.sched.issue(0, np.tile([0.7, 0, 0, 0, 0, 0, -1.0], (50, 1)))
    ag.sched.arrive(0)
    ag.trigger(0)
    ag.act(0, ag.sched.action(0))
    return ag


def test_ppc_calls_earlier_and_offsets_sideways():
    x, p = np.zeros(3), np.array([0.10, 0.0, 0.0])
    ag = _ppc_agent(p, x)
    ag.observe(1, x, p + np.array([0.02, 0.02, 0.0]))  # a jump with a component along the plan and across it
    ag.trigger(1)
    a = ag.act(1, ag.sched.action(1))
    assert ag.ppc_next == 14 and a[1] > 0  # an earlier call (ceil(50 / alpha*), alpha* = 1 + 0.02 / 0.0077)
    assert ag.trigger(ag.ppc_next)


def test_ppc_keeps_correcting_near_the_object_until_grasp():
    x, p = np.zeros(3), np.array([0.02, 0.0, 0.0])  # the arm 2 cm from the object: no reset
    ag = _ppc_agent(p, x)
    ag.observe(1, x, p + np.array([0.0, 0.01, 0.0]))  # a sideways jump
    ag.trigger(1)
    assert ag.act(1, ag.sched.action(1))[1] != ag.sched.action(1)[1]
    ag.observe(2, x, p + np.array([0.0, 0.01, 0.0]))
    ag.trigger(2)
    ag.act(2, np.array([0.7, 0, 0, 0, 0, 0, 1.0]))  # close
    ag.observe(3, x, p + np.array([0.0, 0.03, 0.0]))  # a further jump
    assert not ag.trigger(3)
    assert np.array_equal(ag.act(3, ag.sched.action(3)), ag.sched.action(3))


def test_ppc_new_chunk_clears_offsets():
    x, p = np.zeros(3), np.array([0.10, 0.0, 0.0])
    ag = _ppc_agent(p, x)
    ag.observe(1, x, p + np.array([0.0, 0.02, 0.0]))
    ag.trigger(1)
    assert ag.ppc_off
    ag.on_new_chunk()
    assert not ag.ppc_off and np.array_equal(ag.act(1, ag.sched.action(1)), ag.sched.action(1))


def test_ppc_early_call_is_not_a_step_late():
    # s = 15, a 3 cm shift along the plan at t = 5: alpha* = 1 + 0.03 / 0.011, K_exec = ceil(15 / alpha*) = 5,
    # so the call is due at t_obs + 5 = 5, the shift step itself, the same step as T0
    for method in ("PPC", "T0"):
        _, _, _, ag = _run(method, s=15, steps=6)
        assert ag.sched.n_trig == 1 and ag.sched.t_obs == 5, method
    # s = 50: K_exec = ceil(50 / alpha*) = 14, so PPC calls at t_obs + 14, later than T0 (paper's pace)
    _, _, _, ag = _run("PPC", s=50, steps=15)
    assert ag.sched.n_trig == 1 and ag.sched.t_obs == 14


def test_ppc_offsets_are_replanned_not_summed():
    x, p = np.zeros(3), np.array([0.10, 0.0, 0.0])
    ag = _ppc_agent(p, x)
    for t in (1, 2, 3):  # the object moves +2 mm/step across the plan
        ag.observe(t, x, p + np.array([0.0, 0.002 * t, 0.0]))
        ag.trigger(t)
        a = ag.act(t, ag.sched.action(t))
        assert a[1] == pytest.approx(0.8 * 0.002 / orx.G_POS)  # the latest 0.8 v_perp, not 0.8 + 0.6


def test_ppc_does_not_read_past_the_chunk_at_the_scheduled_call():
    ag, x = orx.Agent("PPC", 50, 0), np.zeros(3)
    for t in range(61):  # the runner's order
        ag.observe(t, x, np.array([0.10, 0.002 * min(max(t - 47, 0), 5), 0.0]))  # +2 mm/step in y at t = 48..52
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, np.tile([0.7, 0, 0, 0, 0, 0, -1.0], (50, 1)))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        ag.act(t, ag.sched.action(t))
    assert ag.sched.n_sched == 2


def test_ppc9_profile_over_k_exec_under_constant_sideways_motion():
    x, p = np.zeros(3), np.array([0.10, 0.0, 0.0])
    ag = _ppc_agent(p, x, "PPC9")
    for t in (1, 2, 3, 4):  # +2 mm/step across the plan: alpha* = 1, K_exec = s = 50
        ag.observe(t, x, p + np.array([0.0, 0.002 * t, 0.0]))
        ag.trigger(t)
        a = ag.act(t, ag.sched.action(t))
        assert a[1] == pytest.approx(orx.ppc_profile(50)[0] * 0.002 / orx.G_POS)  # (1 - F_1 / F_101) v_perp


def test_ppc9_still_object_clears_the_tail_of_a_jump():
    x, p = np.zeros(3), np.array([0.10, 0.0, 0.0])
    ag = _ppc_agent(p, x, "PPC9")
    ag.observe(1, x, p + np.array([0.0, 0.02, 0.0]))  # a single jump
    ag.trigger(1)
    assert ag.act(1, ag.sched.action(1))[1] > 0
    ag.observe(2, x, p + np.array([0.0, 0.02, 0.0]))  # then still
    ag.trigger(2)
    assert np.array_equal(ag.act(2, ag.sched.action(2)), ag.sched.action(2))


def test_episode_metrics():
    xs = np.array([[0, 0, 0], [0.01, 0, 0], [0.02, 0, 0], [0.02, 0.01, 0]], float)
    acts = np.array([[1, 0, 0, 0, 0, 0, -1], [1, 0, 0, 0, 0, 0, -1], [0, 1, 0, 0, 0, 0, 1], [0, 1, 0, 0, 0, 0, 1]], float)
    objs = np.array([[0.03, 0, 0.90]] * 3 + [[0.03, 0, 0.95]], float)
    m = orx.episode_metrics(xs, acts, objs)
    assert m["path_m"] == pytest.approx(0.03)
    assert m["jerk"] == pytest.approx(2 + 4)  # |a1-a0|^2 = 0, |a2-a1|^2 = 1+1+4, |a3-a2|^2 = 0
    assert m["grasp_miss_m"] == pytest.approx(0.01)  # at the first close command (step 2): xy distance
    assert m["grasp_ok"]  # lifted 5 cm with the gripper closed
    assert m["t_close"] == 2
    m = orx.episode_metrics(xs, np.where(np.arange(7) == 6, -1.0, acts), objs)  # never closes
    assert m["t_close"] is None and m["grasp_miss_m"] is None and not m["grasp_ok"]


def test_headroom_ratio():
    ch0 = np.zeros((50, 7))
    ch1 = np.zeros((50, 7))
    ch1[:20, 0] = 0.5  # the fresh plan moves 20 * 0.5 * 11 mm = 11 cm along +x over 20 steps
    ch1[:, 1] = 0.3  # sideways to the shift: projected out
    r = orx.headroom(ch0, ch1, np.array([0.11, 0, 0]))
    assert r[10] == pytest.approx(0.5) and r[25] == pytest.approx(1.0) and r[50] == pytest.approx(1.0)
    ch1 = np.zeros((50, 7))
    ch1[:10, 0] = 1.5  # clipped to 1: 10 * 1 * 11 mm = 11 cm
    assert orx.headroom(ch0, ch1, np.array([0.11, 0, 0]))[10] == pytest.approx(1.0)


def test_grasp_ok_needs_the_lift_at_a_closed_step():
    xs = np.zeros((3, 3))
    acts = np.zeros((3, 7))
    acts[:, 6] = [-1, 1, -1]  # closes at step 1, opens again at step 2
    objs = np.array([[0.0, 0, 0.90], [0.0, 0, 0.90], [0.0, 0, 0.95]])  # pushed up 5 cm with the gripper open
    assert not orx.episode_metrics(xs, acts, objs)["grasp_ok"]
