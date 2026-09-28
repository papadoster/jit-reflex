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
