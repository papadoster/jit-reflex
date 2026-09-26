import numpy as np

import video


def test_pick_is_the_first_seed_where_hero_solves_and_rival_fails():
    solved = {"h": np.array([False, True, True, True]), "r": np.array([False, True, False, False])}
    assert video.pick(solved, "h", "r") == 2
    assert video.pick(solved, "r", "h") is None


def test_footer_numbers_and_level_claims_match_b1():
    t = video.b1_table()
    assert t.shape == (12, 11)
    assert t.loc["mjc_swimmer", ["realtime", "pred:learned", "reflex:learned"]].round(1).tolist() == [41.4, 42.2, 71.2]
    assert t.loc["mjc_swimmer", ["pred:phys0.2", "reflex:phys0.2"]].round(1).tolist() == [44.0, 67.6]
    assert t.loc["catapult", ["realtime", "pred:learned", "reflex:learned"]].round(1).tolist() == [41.0, 28.9, 29.6]
    assert round((t["reflex:learned"] - t["realtime"]).mean(), 1) == 7.7  # report section 8
    assert round((t["reflex:phys0.2"] - t["realtime"]).mean(), 1) == 3.3
    j = {p: t[f"reflex:{p}"] - t[f"pred:{p}"] for p in ("learned", "phys0.2")}  # the claims in PRESETS[...]["why"]
    assert j["learned"].idxmax() == "mjc_swimmer"  # a: strongest for J
    assert j["phys0.2"].rank(ascending=False)["mjc_swimmer"] == 4  # b: 4th of 12
    assert (t["reflex:learned"] - t["realtime"]).idxmin() == "catapult"  # c: worst against RTC
