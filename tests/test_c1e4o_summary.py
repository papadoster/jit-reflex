"""scripts/c1e4o_summary.py (cell О12, PPCL on the OFT brains at 12 cm): its selftest (the plan on the real split, a
broken 12 cm filter refused, the reading on a synthetic pod); the plan's lines; its 12 cm cases are part 2's distance_m
0.12 cases of eval300 for both brains (part 2's records, read only)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4c_summary as cc  # noqa: E402
import c1e4o_summary as co  # noqa: E402

SPLIT = co.maxwrap.load_split(co.cs.SPLIT)


def test_selftest():
    co.selftest()


def test_plan():
    for line, b, kbar in (("O12oftplus", "oftplus", 0.121), ("O12oft", "oft", 0.088)):
        lns = co.plan(line, SPLIT)
        assert [(ln["id"], ln["source"], cc.n_of(ln), len(co.cv.ids(ln, SPLIT))) for ln in lns] == [
            (f"O_{b}_eval300_plus_d0", "plus", 297, 99), (f"O_{b}_eval300_pro_d0", "pro", 174, 58)]
        for ln in lns:
            assert ln["phase"] == line.lower() and ln["arms"] == ["PPCL", "PPC", "Gauto"] and ln["prefix"]
            assert ln["args"].endswith(f" --kbar {kbar} --base-prefix") and "--d 0 " in ln["args"]
            assert len(ln["keys"]) == 4 * len(co.cv.ids(ln, SPLIT))  # the cut Base and three arms per case


def test_the_12cm_cases_are_part2s():
    """The reading's check "cm12" on part 2's own records: the same 157 cases for OFT+ (M1, all of eval) and OFT (M4)."""
    e300, c157 = set(co.run.set_ids(SPLIT, "eval300")), set(co.cases(SPLIT))
    p2 = co.load_p2()
    for b in ("oftplus", "oft"):
        far = {r["case_id"] for r in p2 if r["brain"] == b and r["case_id"] in e300 and r["distance_m"] == 0.12}
        assert far == c157, b
