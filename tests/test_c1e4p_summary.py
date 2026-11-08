"""scripts/c1e4p_summary.py (the placebo cell, docs/superpowers/specs/2026-11-08-c1-e4-placebo-design.md): its selftest
(the plan on the real split, the smoke's check, the reading on a synthetic pod: every rule branch, both slope sentences,
each check alone, drops by reason); the plan's lines against the controls lines C2 / C2_nv and C3 d10 as they ran."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4c_summary as cc  # noqa: E402
import c1e4p_summary as cp  # noqa: E402

SPLIT = cp.maxwrap.load_split(cp.cs.SPLIT)


def test_selftest():
    cp.selftest()


def test_plan_mirrors_the_controls_lines():
    """spec §3: per cell the four arms, part 2's kbar 0.306 and --base-prefix, one runner line per controls line (C2,
    C2_nv; C3 d10 Plus), their 187 Plus cases in their order and their groups of 6 (the Base batches)."""
    ctl = {ln["id"]: ln for L in ("2", "3") for ln in cc.plan(L, cp.cv.CMODE, cp.cv.CK, SPLIT)}
    for L, pairs in (("P0", [("P0_pi05_eval300_plus_d0", "C2_pi05_eval300_plus_d0"),
                             ("P0nv_pi05_eval300_plus_d0", "C2_nv_pi05_eval300_plus_d0")]),
                     ("P10", [("P10_pi05_eval300_plus_d10", "C3_pi05_eval300_plus_d10")])):
        lns = cp.plan(L, SPLIT)
        assert [ln["id"] for ln in lns] == [p for p, _ in pairs]
        for ln, (_, c) in zip(lns, pairs):
            ids = cp.cv.ids(ln, SPLIT)
            assert ids == cp.cv.ids(ctl[c], SPLIT) and all(not i.startswith("pro-") for i in ids)
            assert cp.cv.groups([ln], SPLIT) == cp.cv.groups([ctl[c]], SPLIT)
            assert ln["arms"] == ["none", "none2", "Gauto", "GautoSham"] and ln["prefix"] and ctl[c]["prefix"]
            assert ln["args"].endswith(" --kbar 0.306 --base-prefix") and ln["phase"] == L.lower()
            assert ln["args"].split(" --arms")[0] == ctl[c]["args"].split(" --arms")[0]  # brain, source, set
            assert ("--cases " in ln["args"]) == ("--cases " in ctl[c]["args"])
            assert len(ln["keys"]) == 5 * len(ids) and cc.n_of(ln) == 4 * len(ids)  # the cut Base and four arms
        assert sum(len(cp.cv.ids(ln, SPLIT)) for ln in lns) == 187
