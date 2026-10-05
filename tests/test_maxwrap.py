import numpy as np
import pytest

import maxwrap as mw

NAMES = {"libero_goal": [f"open_the_drawer_{k}" for k in range(7)] + ["open_the_drawer_1_and_close_it"]}


def case(i, task, ctype="target_relocation", name=None, variant=None):
    c = {"case_id": f"c{i:04d}", "task_suite_name": "libero_goal", "task_index": i, "init_state_index": i % 50,
         "policy_seed": 195, "task_name": name or f"open_the_drawer_{task}_table_{i}",
         "scenario": {"change_type": ctype, "change": {"distance_m": 0.06 if i % 2 else 0.12}}}
    if variant:
        c["substrate_variant"] = {"init_states_file": f"libero_goal/{variant}.pruned_init"}
    return c


def toy(n_tasks=7, per=20):
    return [case(t * per + k, t) for t in range(n_tasks) for k in range(per)]


def test_base_task_longest_prefix_and_pro_variant():
    assert mw.base_task(case(0, 3), NAMES) == ("libero_goal", "open_the_drawer_3")
    c = case(1, 1, name="open_the_drawer_1_and_close_it_tb_2")
    assert mw.base_task(c, NAMES) == ("libero_goal", "open_the_drawer_1_and_close_it")
    c = case(2, 0, name="LIVING_ROOM_whatever", variant="open_the_drawer_5")
    assert mw.base_task(c, NAMES) == ("libero_goal", "open_the_drawer_5")
    with pytest.raises(KeyError):
        mw.base_task(case(3, 0, name="close_the_door"), NAMES)


def test_case_order_round_robin_and_deterministic():
    cs = toy()
    a, b = mw.case_order(cs, NAMES, seed=7), mw.case_order(cs, NAMES, seed=7)
    assert [c["case_id"] for c in a] == [c["case_id"] for c in b]
    assert sorted(c["case_id"] for c in a) == sorted(c["case_id"] for c in cs)
    for r in range(3):  # one case of every task per round
        assert sorted(mw.base_task(c, NAMES)[1] for c in a[7 * r:7 * r + 7]) == sorted(NAMES["libero_goal"][:7])
    assert [c["case_id"] for c in mw.case_order(cs, NAMES, seed=8)] != [c["case_id"] for c in a]


def test_split_cases_and_extension():
    cs = toy()
    cal, ev = mw.split_cases(cs, NAMES, seed=7, n_cal=30)
    assert len(cal) == 30 and len(ev) == 110
    assert not {c["case_id"] for c in cal} & {c["case_id"] for c in ev}
    cal2, ev2 = mw.split_cases(cs, NAMES, seed=7, n_cal=30, extra=5)
    assert [c["case_id"] for c in cal2[:30]] == [c["case_id"] for c in cal] and len(cal2) == 35
    assert [c["case_id"] for c in ev2] == [c["case_id"] for c in ev[5:]]


def test_n_extra_rule():
    assert mw.n_extra([1] * 50) == 0  # 50 valid samples >= N_MIN
    assert mw.n_extra([1] * 40 + [None] * 20, more=[1, None, 1, 1, 1, 1, 1]) == 6  # 40 + 5 valid in 6 more cases


def test_connection_sample_balanced_over_events():
    cs = [case(i, i % 5, ctype=f"e{i % 8}") for i in range(800)]
    s = mw.connection_sample(cs, n=300, seed=3)
    per = {}
    for c in s:
        per[c["scenario"]["change_type"]] = per.get(c["scenario"]["change_type"], 0) + 1
    assert len(s) == 300 and set(per) == {f"e{k}" for k in range(8)} and max(per.values()) - min(per.values()) <= 1
    assert [c["case_id"] for c in s] == [c["case_id"] for c in mw.connection_sample(cs, n=300, seed=3)]


@pytest.mark.parametrize("q,e", [(5, 1), (5, 5), (5, 7), (10, 23), (8, 16)])
def test_schedule_check(q, e):
    plans = [(t, t) for t in range(0, 200, q)]  # (t_obs, arrival) of the native schedule at d = 0
    ok, last, first = mw.schedule_check(e, q, plans)
    assert ok and last == q * ((e - 1) // q) and first == q * -(-e // q)
    assert not mw.schedule_check(e, q, [p for p in plans if p[0] != last])[0]
    assert not mw.schedule_check(e, q, [p for p in plans if p[0] != first])[0]


def test_chunk_replay_serves_recorded_chunks_until_event():
    rec = {0: np.zeros((50, 7)), 5: np.ones((50, 7))}
    rp = mw.ChunkReplay(rec, event_step=7)
    assert rp.replaying(4) and rp.replaying(6) and not rp.replaying(7)
    assert np.array_equal(rp.chunk(5), np.ones((50, 7)))
    with pytest.raises(KeyError):
        rp.chunk(3)


def test_kbar_line_band():
    rng = np.random.default_rng(0)
    xs = list(rng.normal(0.33, 0.4, 70)) + [None, None]
    k, lo, hi, n = mw.kbar_line(xs, n_boot=2000, seed=0)
    good = [x for x in xs if x is not None]
    assert n == 70 and k == pytest.approx(float(np.median(good)))
    assert lo < k < hi and hi - lo < 0.4
    assert mw.kbar_line(xs, n_boot=2000, seed=0) == (k, lo, hi, n)
