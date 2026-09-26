import json

import pandas as pd
import pytest

import b2b5
import eval_flow


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
