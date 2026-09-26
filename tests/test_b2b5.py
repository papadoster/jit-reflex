import pandas as pd

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
