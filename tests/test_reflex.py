import eval_flow


def test_horizons_for():
    assert eval_flow.horizons_for(0, 8, (), False) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert eval_flow.horizons_for(2, 8, (), False) == [2, 3, 4, 5, 6]
    assert eval_flow.horizons_for(2, 8, (), True) == [2, 6]
    assert eval_flow.horizons_for(4, 8, (), True) == [4]
    assert eval_flow.horizons_for(3, 8, (1, 3, 5, 7), False) == [3, 5]
