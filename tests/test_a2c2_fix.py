import sys

sys.path.insert(0, "src")
import a2c2_fix  # noqa: E402
import probe  # noqa: E402


def test_experts_cover_every_level_and_match_the_mac_diagnostic():
    assert sorted(a2c2_fix.EXPERTS) == sorted(probe.LEVELS)
    # docs/results/b2b5-checks.md (a2) used these: the spec's rule must pick the same experts
    assert a2c2_fix.EXPERTS["worlds/l/trampoline.json"] == (2, 840)
    assert a2c2_fix.EXPERTS["worlds/l/mjc_walker.json"] == (7, 580)
    assert a2c2_fix.EXPERTS["worlds/l/car_launch.json"] == (0, 920)


def test_expert_path_and_url():
    p, u = a2c2_fix.expert_path("worlds/l/trampoline.json"), a2c2_fix.expert_url("worlds/l/trampoline.json")
    assert p == "checkpoints/expert/seed_2_step_840_worlds_l_trampoline.pkl"
    assert u == "https://storage.googleapis.com/rtc-assets/expert/seed_2/840/policies/worlds_l_trampoline.pkl"


def test_latency_writes_head_numbers(tmp_path):
    a2c2_fix.latency(out_dir=str(tmp_path), repeats=3, warmup=1)
    info = __import__("json").loads((tmp_path / "head_latency.json").read_text())
    assert info["in_dim"] == 4 * 679 + 6 + 6 + 2
    assert info["head_ms_cpu"] > 0 and info["head_ms_gpu"] > 0
    # 2 x (2730 x 512 + 512 x 512 + 512 x 6) multiply-adds, LayerNorm and ReLU on top: about 3.3 MFLOP
    assert 0.003 < info["head_gflop"] < 0.004


import numpy as np
import pandas as pd


def _frame(values):
    """results.csv rows: values {(method, predictor): {level: solved}} at every cell of b2b5.ALL16 and seeds 20-22."""
    import b2b5
    rows = []
    for (m, p), per_level in values.items():
        for lv, v in per_level.items():
            for d, s in b2b5.ALL16:
                for sd in b2b5.SEEDS:
                    rows.append({"method": m, "predictor": p, "delay": d, "execute_horizon": s, "seed": sd,
                                 "level": lv, "returned_episode_solved": v})
    return pd.DataFrame(rows)


def test_diff_splits_development_and_held_out_levels():
    levels = list(a2c2_fix.EXPERTS)
    new = {lv: (0.9 if lv in a2c2_fix.DEV else 0.6) for lv in levels}
    df = _frame({("a2c2_paper", "-"): new, ("realtime", "-"): {lv: 0.5 for lv in levels}})
    cell = a2c2_fix.cells_of(df)
    held = [lv for lv in levels if lv not in a2c2_fix.DEV]
    r = a2c2_fix.diff(cell, ("a2c2_paper", "-"), ("realtime", "-"), [(3, 5)], held)
    assert abs(r["pooled_pp"] - 10.0) < 1e-9 and r["per_seed_pp"] == [10.0, 10.0, 10.0]
    r = a2c2_fix.diff(cell, ("a2c2_paper", "-"), ("realtime", "-"), [(3, 5)], list(a2c2_fix.DEV))
    assert abs(r["pooled_pp"] - 40.0) < 1e-9
    r = a2c2_fix.diff(cell, ("a2c2_paper", "-"), ("realtime", "-"), [(3, 5)], levels)
    assert abs(r["pooled_pp"] - 17.5) < 1e-9  # (3 x 40 + 9 x 10) / 12


def test_diff_is_none_without_common_data():
    df = _frame({("a2c2_paper", "-"): {"worlds/l/trampoline.json": 0.9}})
    cell = a2c2_fix.cells_of(df)
    assert a2c2_fix.diff(cell, ("a2c2_paper", "-"), ("t3", "learned"), [(3, 5)], list(a2c2_fix.EXPERTS)) is None


def test_placement_adds_one_when_the_head_misses_the_tact():
    lat = pd.DataFrame([{"method": "naive", "delay": 1, "execute_horizon": 1, "ms": 1.128}])
    info = {"realtime_ms": 1.804}
    fast = a2c2_fix.head_cells(lat, info, {"head_ms_cpu": 0.1})
    slow = a2c2_fix.head_cells(lat, info, {"head_ms_cpu": 0.7})  # the tact at base 3 is 1.804 / 3 = 0.60 ms
    assert fast[3] == [(2, 2), (2, 3), (2, 4), (2, 5), (2, 6)]
    assert slow[3] == [(3, 3), (3, 4), (3, 5)]
    assert fast[1] == [(1, s) for s in range(1, 8)]
