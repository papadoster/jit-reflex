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
