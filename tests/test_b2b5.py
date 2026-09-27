import json
import pickle
import time

import flax.nnx as nnx
import jax
import numpy as np
import pandas as pd
import pytest

import a2c2
import b2b5
import eval_flow
import model
import probe


def test_b2b5_methods_are_registered():
    m = eval_flow.METHODS
    assert m["t3"].cand == "T3" and m["t3"].j_delay == 0
    assert m["m3"].cand == "M3"
    assert [m[f"late{k}"].j_delay for k in range(1, 7)] == [1, 2, 3, 4, 5, 6]  # delta < s <= 7 with H = 8
    assert all(m[f"late{k}"].cand == "T3" for k in range(1, 7))
    assert "late6" in eval_flow.HIST and "late6" in b2b5.REFLEX
    assert eval_flow.FLOW_STEPS == {"realtime10": 10}
    assert isinstance(m["realtime10"], eval_flow.RealtimeMethodConfig)


def test_cells_parse_in_order():
    assert eval_flow.parse_cells(["3,5", "4,4", "1,7"]) == [(3, 5), (4, 4), (1, 7)]


def test_resume_keeps_only_complete_configs(tmp_path):
    cols = ["seed", "delay", "execute_horizon", "method", "predictor", "level"]
    rows = [(0, 1, 1, "naive", "-", lv) for lv in ("a", "b")] + [(0, 2, 2, "t3", "oracle", "a")]  # 2nd: partial
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "results.csv", index=False)
    old, done = eval_flow.load_done(tmp_path, ["a", "b"])
    assert done == {(0, 1, 1, "naive", "-")} and list(old["level"]) == ["a", "b"]
    assert eval_flow.load_done(tmp_path / "missing", ["a", "b"])[1] == set()


def test_resume_keeps_the_rows_of_other_level_sets(tmp_path):
    cols = ["seed", "delay", "execute_horizon", "method", "predictor", "level"]
    rows = [(0, 3, 5, "naive", "-", lv) for lv in probe.LEVELS] + [(0, 3, 5, "a2c2", "-", probe.LEVELS[1])]
    pd.DataFrame(rows, columns=cols).to_csv(tmp_path / "results.csv", index=False)
    old, done = eval_flow.load_done(tmp_path, [probe.LEVELS[1]])  # a one-level call keeps the 12-level config
    assert done == {(0, 3, 5, "naive", "-"), (0, 3, 5, "a2c2", "-")} and len(old) == 13
    old, done = eval_flow.load_done(tmp_path, list(probe.LEVELS))  # the 12-level call reruns a2c2: its row goes
    assert done == {(0, 3, 5, "naive", "-")} and len(old) == 12 and set(old["method"]) == {"naive"}


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
    a = b2b5.command_lines("A", out_dir="x")  # a missing A2C2 head fails only its own eval_flow call (spec §14)
    assert [ln.split(" --predictors")[0] for ln in a[:2]] == ["--methods naive realtime realtime10",
                                                              "--methods a2c2 a2c2_distill"]


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
        "gflop_per_eval": 0.001, "head_gflop": 0.0006, "level": probe.LEVELS[0]}


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


def test_placement_reaches_late5_and_late6():
    for t3, k in ((5.5, 5), (6.5, 6)):  # x_j = t3 at base 1: delta = k at d = 1; (1, 5) with delta >= s is pred
        late = {(c["d"], c["s"], c["method"]) for c in b2b5.placement(_lat(pred=0.9, t3=t3), INFO, 1)["late"]}
        assert (1, 7, f"late{k}") in late and (1, 5, "pred") in late
    with pytest.raises(ValueError, match="late7"):  # impossible with H = 8 (s <= 7), but loud
        b2b5.late_method(7, 8)


def _lat44(pred, t3):
    """_lat() with other r_pred and r_t3 at (4,4) only: B5-R1 reads them, the per-cell placement does not."""
    lat = _lat()
    lat.loc[(lat.method == "pred") & (lat.delay == 4), "ms"] = pred
    lat.loc[(lat.method == "t3") & (lat.delay == 4), "ms"] = t3
    return lat


def test_r1_cell_comes_from_the_44_measurement():
    lat = _lat44(0.95, 1.2)  # x_nom = 2.85 -> d' 3, cell (3,5); x_j = 3.6 -> 4, delta 1
    c = b2b5.r1_cell(lat, INFO)
    assert (c["d"], c["s"], c["delta"], c["method"]) == (3, 5, 1, "late1")
    assert not [c for c in b2b5.placement(lat, INFO, 3)["late"] if c["s"] == 8 - c["d"]]  # per-cell r: none
    c = b2b5.r1_cell(lat, INFO | {"kappa": 1.2})  # 1.2 x 0.95 x 3 = 3.42 -> 4; kappa_j 1.0: 3.6 -> 4, delta 0
    assert (c["d"], c["s"], c["method"]) == (4, 4, "t3")
    assert b2b5.r1_cell(lat, INFO | {"kappa": 1.1, "kappa_j": 2.0})["method"] == "late1"  # kappa <= 1.10: free
    assert b2b5.r1_cell(_lat(pred=1.5), INFO)["method"] is None  # 4.5 -> 5: NOT-FEASIBLE
    assert b2b5.r1_cell(lat, INFO, d=4)["method"] == "t3"  # the other d' of the edge: delta 4 - 4


def test_place_adds_the_r1_cell_and_its_edge_neighbour(tmp_path):
    _lat44(2.02 / 3, 1.2).to_csv(tmp_path / "latency.csv", index=False)  # x_nom 2.02: d' 3, edge d' 2
    (tmp_path / "latency.json").write_text(json.dumps(INFO))
    b2b5.place(out_dir=str(tmp_path))  # R1: late1 (3,5) is in the grid; the edge: delta 4 - 2 at (2,6) is not
    assert json.loads((tmp_path / "extra_blocks.json").read_text()) == [["B", ["late2"], ["learned"], [[2, 6]]]]
    _lat44(2.5 / 3, 1.5).to_csv(tmp_path / "latency.csv", index=False)  # R1: d' 3, x_j 4.5 -> delta 2; no edge
    b2b5.place(out_dir=str(tmp_path))
    assert json.loads((tmp_path / "extra_blocks.json").read_text()) == [["B", ["late2"], ["learned"], [[3, 5]]]]


def test_flops_are_counted_on_cpu_copies():
    O, A = 679, 6
    sd = nnx.state(model.FlowPolicy(obs_dim=O, action_dim=A, config=model.ModelConfig(), rngs=nnx.Rngs(0)))
    g, hg = b2b5.cpu_gflop(sd.to_pure_dict(), jax.numpy.zeros((1, O)), jax.numpy.zeros((1, 8, A)))
    assert 0.02 < g < 0.1 and hg == pytest.approx(0.000621, rel=0.01)  # the rehearsal: 0.052 and 0.000621


def test_a_missing_head_drops_only_its_method(tmp_path, capsys):
    (tmp_path / "a2c2").mkdir()
    with (tmp_path / "a2c2" / "worlds_l_catapult.pkl").open("wb") as f:
        pickle.dump(nnx.state(a2c2.Head(4, 2, rngs=nnx.Rngs(0))).to_pure_dict(), f)
    heads, ms = eval_flow.load_head_sets(["naive", "a2c2", "a2c2_distill"], str(tmp_path), ["worlds/l/catapult.json"])
    assert list(heads) == ["a2c2"] and ms == ["naive", "a2c2"]
    assert "!!! a2c2_distill: heads missing" in capsys.readouterr().out


def _covered(lines) -> set:
    """The (method, predictor, d, s) configs eval_flow runs for these command lines."""
    out = set()
    for ln in lines:
        a = {k: v for k, *v in (p.split() for p in ln.lstrip("-").split(" --"))}
        out |= {(m, p if m in b2b5.REFLEX else "-", *map(int, c.split(",")))
                for m in a["methods"] for p in a["predictors"] for c in a["cells"]}
    return out


def test_cut_drops_the_spec_steps_in_order_without_touching_the_grid():
    sizes = [len(b2b5.cut_configs(n)) for n in range(5)]
    assert sizes == [0, 8, 10, 12, 18]  # (2,2) (2,6) x 4; phys0.2 x 2; t3 m3 oracle; a2c2_distill x 6
    assert ("a2c2_distill", "-", 1, 4) in b2b5.cut_configs(4) and ("a2c2_distill", "-", 2, 3) not in b2b5.cut_configs(4)
    assert not any(c[0] == "realtime10" for c in b2b5.cut_configs(4))
    for n in range(5):
        lines = b2b5.command_lines("A", "x", cut=n) + b2b5.command_lines("B", "x", cut=n)
        assert _covered(lines) == b2b5.configs() - b2b5.cut_configs(n)
    assert b2b5.command_lines("B", "x", cut=0) == b2b5.command_lines("B", "x")
    assert b2b5.grid_sha() == json.loads(open("results/b2b5/lock.json").read())["grid"]
    with pytest.raises(ValueError):
        b2b5.cut_configs(5)


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


def test_dominated_flags():
    rows = [("v", "naive", 2, 4, 0.60, 1.0),  # realtime costs as much and solves more: dominated
            ("v", "realtime", 3, 3, 0.70, 1.0),
            ("v", "t3", 4, 4, 0.80, 5.0),
            ("v", "t3", 4, 4, 0.80, 5.0),  # the same config again (a late row): not a rival of its twin
            ("v", "pred", 4, 4, np.nan, 0.5),  # no data: no flag
            ("w", "naive", 1, 1, 0.99, 0.1)]  # another view: would dominate everything above
    fr = pd.DataFrame(rows, columns=["view", "method", "delay", "execute_horizon", "P", "fe_step"]).assign(base=3)
    assert b2b5.dominated(fr, "fe_step") == [True, False, False, False, None, False]


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
    hist = [{"seed": sd, "delay": 3, "execute_horizon": 5, "method": "t3", "predictor": "learned", "level": "x", "k": 3,
             "bin": 9, "count": 10} for sd in (20, 99)]  # seed 99 is not a main seed: left out
    (tmp_path / "eval_B").mkdir()
    pd.DataFrame(hist).to_csv(tmp_path / "eval_B" / "hist.csv", index=False)
    (tmp_path / "a2c2_distill").mkdir()
    pd.DataFrame([{"level": "x", "transitions": t, "steps": 10, "mse": m, "mse_base": 0.2}  # a rerun: last row
                  for t, m in ((1, 0.5), (1000, 0.1))]).to_csv(tmp_path / "a2c2_distill" / "train_log.csv", index=False)
    v = b2b5.summarize(out_dir=str(tmp_path))
    assert v["B2-R1 t3"]["verdict"] == v["B2-R1 t3"]["final"] == "KEEPS"
    assert v["B5-R1 late"]["verdict"] == "WIN" and abs(v["B5-R1 late"]["pooled_pp"] - 5.0) < 1e-6
    assert v["B5-R2 m3"]["verdict"] == "LOSE"  # m3 0.60 vs realtime 0.75
    assert v["B5-R2 m3"]["edge"]["other_d"] == 5 and v["B5-R2 m3"]["edge"]["verdict_other_d"] == "NOT-FEASIBLE"
    assert v["B5-R3 t3 (3,5)"]["verdict"] == "FAIL"
    assert (tmp_path / "extension.txt").read_text() == ""
    res = json.loads((tmp_path / "b2b5.json").read_text())
    a = res["a2c2"]["A2C2 (вариант bt-kinetix) − t3"]["D3 (3,5)"]  # a2c2 0.60, t3 0.72 / 0.68 / 0.68
    assert a["per_seed_pp"] == pytest.approx([-12, -8, -8]) and res["hypotheses"]["E1 A2C2-t3 D3 pp"] == a["pooled_pp"]
    assert res["a2c2_flag"] == "реализация под сомнением"  # a2c2 0.60 < realtime 0.75 at D3
    assert res["hypotheses"]["E2 late1-pred (4,4) pp"] == pytest.approx(20)
    assert res["hypotheses"]["E2 late2-pred (4,4) pp"] == pytest.approx(0)
    assert res["edges base 3 (report)"]["realtime10"]["other_d"] == 7  # r 2.0 x 3 = 6.0
    fr = pd.read_csv(tmp_path / "frontier.csv")
    assert {"A2C2 (bt-kinetix variant)", "A2C2-distill"} <= set(fr.label) and fr.dominated_fe_step.any()
    wm = 2 * (685 * 256 + 256 * 256 + 256 * 679) / 1e9 / INFO["gflop_per_eval"]  # one world-model step in FE
    assert fr[fr.method == "t3"].query("delay == 3 and execute_horizon == 5").wm_fe_step.iloc[0] == pytest.approx(
        7 / 5 * wm)
    assert fr[fr.method == "naive"].wm_fe_step.isna().all()
    assert pd.read_csv(tmp_path / "hist_summary.csv")["count"].tolist() == [10]
    tc = pd.read_csv(tmp_path / "training_compute.csv").query("method == 'a2c2_distill'")
    g = INFO["gflop_per_eval"]
    assert tc.train_gflop.tolist() == pytest.approx([10 * 512 * 3 * INFO["head_gflop"] + 5 * g * 1000 * 9 / 8 * 1.1])
    assert res["head MSE < base MSE (§9)"] == {"a2c2_distill (held-out)": {"x": True}}


def test_summarize_gray_writes_the_extension(tmp_path):
    def value(m, p, d, s, sd):
        if m == "late1" and (d, s) == (4, 4):
            return 0.75 + (0.02 if sd == 20 else 0.0)  # pooled +0.67 pp, not > 0 in every seed -> GRAY
        return 0.75 if m == "realtime" else 0.60

    _write_results(tmp_path, value)
    v = b2b5.summarize(out_dir=str(tmp_path))
    assert v["B5-R1 late"]["verdict"] == v["B5-R1 late"]["final"] == "GRAY"
    ext = (tmp_path / "extension.txt").read_text().splitlines()
    assert any(ln.startswith("B --methods late1") and "--seeds 23 24 25" in ln for ln in ext)
    assert any(ln.startswith("A --methods realtime ") for ln in ext)
    _write_results(tmp_path, value, seeds=(*b2b5.SEEDS, *b2b5.EXT_SEEDS))  # the extension ran: GRAY again
    r = b2b5.summarize(out_dir=str(tmp_path))["B5-R1 late"]
    assert r["verdict_6_seeds"] == "GRAY" and r["final"] == "не определён"
    assert len(r["ci95_pp"]) == len(r["ci95_6_pp"]) == 2 and (tmp_path / "extension.txt").read_text() == ""


def test_summarize_strict_stops_on_a_missing_seed(tmp_path):
    _write_results(tmp_path, lambda *a: 0.5, seeds=(20, 21))
    with pytest.raises(SystemExit):
        b2b5.summarize(out_dir=str(tmp_path))


def test_summarize_strict_accepts_cut_configs(tmp_path):
    _write_results(tmp_path, lambda *a: 0.5)
    f = tmp_path / "eval_A" / "results.csv"
    df = pd.read_csv(f)
    cut = b2b5.cut_configs(4)
    df[[c not in cut for c in df[b2b5.CFG].itertuples(index=False, name=None)]].to_csv(f, index=False)
    with pytest.raises(SystemExit):
        b2b5.summarize(out_dir=str(tmp_path))
    b2b5.summarize(out_dir=str(tmp_path), cut=4)
    res = json.loads((tmp_path / "b2b5.json").read_text())
    assert res["missing"] == 0 and len(res["cut"]) == 18 and ["pred", "phys0.2", 3, 5] in res["cut"]


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
    res = json.loads((tmp_path / "b2b5.json").read_text(), parse_constant=pytest.fail)  # strict JSON: no NaN
    assert res["missing"] == 3 * (135 - 53) and res["hypotheses"]["H2 spearman"] is None
    assert res["a2c2"]["A2C2-distill − t3"]["D1 (s 5-7)"]["pooled_pp"] is None and res["a2c2_flag"] is None
    _lat(m3=1.0).to_csv(tmp_path / "latency.csv", index=False)  # x = 3.0: m3's rule cell is (4,4) either way
    e = b2b5.summarize(out_dir=str(tmp_path), strict=False)["B5-R2 m3"]["edge"]
    assert e["other_d"] == 4 and "verdict_other_d" not in e
