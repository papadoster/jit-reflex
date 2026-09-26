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


def test_presets_are_consistent_with_demo_and_b1():
    levels = video.b1_table().index
    for p in video.PRESETS.values():
        assert p["hero"] in p["panels"] and p["rival"] in p["panels"] and p["hero"] != p["rival"]
        assert p["panels"][-1].startswith("reflex:")  # the footer takes the predictor from the last panel
        assert p["level"] in levels


def test_panel_freezes_after_the_episode_and_keeps_its_input():
    from PIL import ImageFont

    video_in = np.arange(6 * 4 * 4 * 3, dtype=np.uint8).reshape(6, 4, 4, 3)
    before = video_in.copy()
    out = video._panel(video_in, "x", False, 3, ImageFont.load_default(size=8))
    assert out.shape == (6, 512, 512, 3)
    assert (out[3:] == out[3]).all() and not (out[3] == out[2]).all()  # frozen on frame 2, with the outcome caption
    assert (video_in == before).all()
