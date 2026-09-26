import json
import time

import pandas as pd
import pytest

import b2b5
import eval_flow
import probe


def test_b2b5_methods_are_registered():
    m = eval_flow.METHODS
    assert m["t3"].cand == "T3" and m["t3"].j_delay == 0
    assert m["m3"].cand == "M3"
    assert [m[f"late{k}"].j_delay for k in range(1, 5)] == [1, 2, 3, 4]
    assert all(m[f"late{k}"].cand == "T3" for k in range(1, 5))
    assert eval_flow.FLOW_STEPS == {"realtime10": 10}
    assert isinstance(m["realtime10"], eval_flow.RealtimeMethodConfig)


def test_cells_parse_in_order():
    assert eval_flow.parse_cells(["3,5", "4,4", "1,7"]) == [(3, 5), (4, 4), (1, 7)]


def test_resume_keeps_only_complete_configs(tmp_path):
    cols = ["seed", "delay", "execute_horizon", "method", "predictor", "level"]
    rows = [(0, 1, 1, "naive", "-", lv) for lv in ("a", "b")] + [(0, 2, 2, "t3", "oracle", "a")]  # 2nd: partial
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "results.csv", index=False)
    old, done = eval_flow.load_done(tmp_path, 2)
    assert done == {(0, 1, 1, "naive", "-")} and list(old["level"]) == ["a", "b"]
    assert eval_flow.load_done(tmp_path / "missing", 2)[1] == set()


def test_grid_has_the_spec_configs():
    c = b2b5.configs()
    assert len(c) == 135
    assert all(s >= d and d + s <= 8 for _, _, d, s in c)
    assert ("late1", "learned", 4, 4) in c and ("late2", "learned", 4, 4) in c
    assert ("realtime10", "-", 1, 1) in c and ("a2c2_distill", "-", 4, 4) in c
    assert ("pred", "phys0.2", 3, 5) in c and ("rtc_reflex", "oracle", 1, 6) in c


def test_commands_cover_every_block_of_a_worker():
    lines = b2b5.command_lines("B", out_dir="x")
    assert all("--output-dir x/eval_B" in ln for ln in lines)
    assert any("--methods late2" in ln and "--cells 4,4" in ln for ln in lines)
    assert all("--seeds 20 21 22" in ln for ln in lines)


def test_lock_detects_a_changed_input(tmp_path, monkeypatch):
    f = tmp_path / "in.bin"
    f.write_bytes(b"a")
    monkeypatch.setattr(b2b5, "inputs", lambda: [str(f)])
    lock = tmp_path / "lock.json"
    b2b5.lock(write=True, path=str(lock))
    b2b5.lock(check=True, path=str(lock))
    w = tmp_path / "w"
    w.mkdir()
    (w / "x.pkl").write_bytes(b"w")
    b2b5.lock(add=[str(w)], path=str(lock))
    assert str(w / "x.pkl") in json.loads(lock.read_text())["created"]
    f.write_bytes(b"b")
    with pytest.raises(SystemExit):
        b2b5.lock(check=True, path=str(lock))


def test_lock_refuses_to_rerecord_a_changed_created_file(tmp_path, monkeypatch):
    monkeypatch.setattr(b2b5, "inputs", lambda: [])
    lock = str(tmp_path / "lock.json")
    b2b5.lock(write=True, path=lock)
    (tmp_path / "x.pkl").write_bytes(b"w")
    b2b5.lock(add=[str(tmp_path)], path=lock)
    b2b5.lock(add=[str(tmp_path)], path=lock)  # the same bytes: a no-op
    (tmp_path / "x.pkl").write_bytes(b"v")
    with pytest.raises(SystemExit):
        b2b5.lock(add=[str(tmp_path)], path=lock)
    with pytest.raises(SystemExit):
        b2b5.lock(check=True, path=lock)


def _lat(pred=1.15, t3=1.55, m3=1.30, reflex=1.85):
    rows = [("realtime", 1, s, 1.0) for s in (1, 4, 7)] + [("naive", 1, s, 0.63) for s in (1, 4, 7)]
    rows += [("realtime10", 1, s, 2.0) for s in (1, 4, 7)]
    for m, r in (("pred", pred), ("t3", t3), ("m3", m3), ("reflex", reflex), ("rtc_reflex", 2.2)):
        rows += [(m, d, s, r) for d, s in b2b5.R9 + [(1, 1), (1, 4)]]
    return pd.DataFrame(rows, columns=["method", "delay", "execute_horizon", "ms"])


INFO = {"realtime_ms": 1.0, "kappa": 1.0, "kappa_j": 1.0, "head_ms_cpu": 0.01, "head_ms_gpu": 0.05,
        "gflop_per_eval": 0.001, "head_gflop": 0.0006}


def test_placement_at_base_3():
    pl = b2b5.placement(_lat(), INFO, 3)
    assert pl["naive"] == [(2, s) for s in range(2, 7)] and pl["a2c2"] == pl["naive"]
    assert pl["realtime"] == [(3, 3), (3, 4), (3, 5)] and pl["realtime10"] == []
    assert pl["pred"] == [(4, 4)] and pl["m3"] == [(4, 4)] and pl["t3"] == [] and pl["reflex"] == []
    assert [(c["d"], c["s"], c["delta"], c["method"]) for c in pl["late"]] == [(4, 4, 1, "late1")]


def test_placement_kappa_head_and_edges():
    assert b2b5.placement(_lat(), INFO | {"kappa": 1.2}, 3)["late"] == []  # 1.2 * 1.15 * 3 = 4.14 -> 5
    assert b2b5.placement(_lat(), INFO | {"kappa": 1.1, "kappa_j": 1.2}, 3)["late"][0]["delta"] == 1  # §5.2: free
    assert b2b5.placement(_lat(m3=1.36), INFO, 3)["m3"] == []  # 4.08 -> 5
    assert b2b5.placement(_lat(), INFO | {"head_ms_cpu": 0.5}, 3)["a2c2"] == [(3, 3), (3, 4), (3, 5)]
    pl2 = b2b5.placement(_lat(), INFO, 2)
    assert pl2["pred"] == [(3, 3), (3, 4), (3, 5)] and pl2["t3"] == [(4, 4)]
    assert sorted((c["d"], c["s"], c["method"]) for c in pl2["late"]) == [(3, s, "late1") for s in (3, 4, 5)]


def test_placement_stops_on_a_late_delta_eval_flow_lacks():
    with pytest.raises(ValueError, match="late5"):  # (1, 6): x_j = 5.5 -> delta 5 < s; (1, 5): delta >= s is pred
        b2b5.placement(_lat(pred=0.9, t3=5.5), INFO, 1)


def test_other_side_of_a_latency_edge():
    assert b2b5.other_side(3.98) == 5 and b2b5.other_side(4.02) == 4 and b2b5.other_side(3.5) is None


def test_hours_left_counts_per_seed_runs():
    rows = [("naive", "-", 1, 1, sd, t) for sd, t in ((20, 100.0), (21, 10.0), (22, 10.0))]  # done
    rows += [("naive", "-", 1, 2, 20, 100.0)]  # started: 2 later seeds x 10 s
    df = pd.DataFrame([(*r, lv) for r in rows for lv in "ab"], columns=[*b2b5.CFG, "seed", "seconds", "level"])
    todo = {("naive", "-", 1, 1), ("naive", "-", 1, 2), ("naive", "-", 2, 2), ("pred", "learned", 2, 2)}
    # not started: naive 100 + 2 x 10 s; pred (no runs) the worker's first-run mean 100 + 2 x 100 s
    assert b2b5.hours_left(df, todo) == (1, pytest.approx((20 + 120 + 300) / 3600))
    empty = pd.DataFrame(columns=[*b2b5.CFG, "seed", "seconds"])
    assert b2b5.hours_left(empty, {("naive", "-", 2, 2)}) == (0, pytest.approx(900 / 3600))


def test_concurrent_fails_loudly_when_the_load_dies():
    n = [0]

    def fb():
        n[0] += 1
        if n[0] > 3:  # the 3 solo calls pass, the background ones fail
            raise MemoryError("oom")

    with pytest.raises(RuntimeError, match="background load died"):
        b2b5._concurrent(lambda: time.sleep(0.01), fb, (), 3, 0)


def _write_results(tmp_path, value, seeds=b2b5.SEEDS):
    """results.csv for every grid config and seed; value(method, predictor, d, s, seed) -> solve rate."""
    rows = [
        {"method": m, "predictor": p, "delay": d, "execute_horizon": s, "seed": sd, "level": lv,
         "returned_episode_solved": value(m, p, d, s, sd), "seconds": 1.0}
        for m, p, d, s in b2b5.configs() for sd in seeds for lv in probe.LEVELS
    ]
    (tmp_path / "eval_A").mkdir(exist_ok=True)
    pd.DataFrame(rows).to_csv(tmp_path / "eval_A" / "results.csv", index=False)
    _lat().to_csv(tmp_path / "latency.csv", index=False)
    (tmp_path / "latency.json").write_text(json.dumps(INFO))


def test_summarize_rules(tmp_path):
    def value(m, p, d, s, sd):
        if m == "late1" and (d, s) == (4, 4):
            return 0.80  # +5 pp over the best rival in every seed
        if m == "t3" and (d, s) == (3, 5):
            return 0.70 + (0.02 if sd == 20 else -0.02)  # pooled -0.67 pp against reflex's 0.70 -> KEEPS
        if m == "reflex":
            return 0.70
        return 0.75 if m == "realtime" else 0.60

    _write_results(tmp_path, value)
    v = b2b5.summarize(out_dir=str(tmp_path))
    assert v["B2-R1 t3"]["verdict"] == "KEEPS"
    assert v["B5-R1 late"]["verdict"] == "WIN" and abs(v["B5-R1 late"]["pooled_pp"] - 5.0) < 1e-6
    assert v["B5-R2 m3"]["verdict"] == "LOSE"  # m3 0.60 vs realtime 0.75
    assert v["B5-R2 m3"]["edge"]["other_d"] == 5 and v["B5-R2 m3"]["edge"]["verdict_other_d"] == "NOT-FEASIBLE"
    assert v["B5-R3 t3 (3,5)"]["verdict"] == "FAIL"
    assert (tmp_path / "extension.txt").read_text() == ""


def test_summarize_gray_writes_the_extension(tmp_path):
    def value(m, p, d, s, sd):
        if m == "late1" and (d, s) == (4, 4):
            return 0.75 + (0.02 if sd == 20 else 0.0)  # pooled +0.67 pp, not > 0 in every seed -> GRAY
        return 0.75 if m == "realtime" else 0.60

    _write_results(tmp_path, value)
    v = b2b5.summarize(out_dir=str(tmp_path))
    assert v["B5-R1 late"]["verdict"] == "GRAY"
    ext = (tmp_path / "extension.txt").read_text().splitlines()
    assert any(ln.startswith("B --methods late1") and "--seeds 23 24 25" in ln for ln in ext)
    assert any(ln.startswith("A --methods realtime ") for ln in ext)


def test_summarize_strict_stops_on_a_missing_seed(tmp_path):
    _write_results(tmp_path, lambda *a: 0.5, seeds=(20, 21))
    with pytest.raises(SystemExit):
        b2b5.summarize(out_dir=str(tmp_path))


def test_summarize_partial_grid_without_strict(tmp_path):
    _write_results(tmp_path, lambda m, *a: 0.75 if m == "realtime" else 0.60)
    f = tmp_path / "eval_A" / "results.csv"
    df = pd.read_csv(f)
    df[df.method.isin(["naive", "realtime", "pred", "m3"])].to_csv(f, index=False)  # 53 of the 135 configs
    _lat(m3=1.36).to_csv(tmp_path / "latency.csv", index=False)  # r_m3 x 3 = 4.08: d' = 5, but 4 within 3%
    v = b2b5.summarize(out_dir=str(tmp_path), strict=False)
    assert v["B2-R1 t3"]["verdict"] == "MISSING" and v["B5-R1 late"]["verdict"] == "MISSING"
    assert v["B5-R2 m3"]["verdict"] == "NOT-FEASIBLE"
    assert v["B5-R2 m3"]["edge"]["other_d"] == 4 and v["B5-R2 m3"]["edge"]["verdict_other_d"] == "LOSE"
    res = json.loads((tmp_path / "b2b5.json").read_text())
    assert res["missing"] == 3 * (135 - 53) and res["hypotheses"]["H2 spearman"] is None
