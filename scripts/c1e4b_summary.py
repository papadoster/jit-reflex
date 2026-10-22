"""C1-E4 VT-m (B): PBVS's machinery with VS-T's target (docs/superpowers/specs/2026-10-20-c1-e4-pbvs-design.md, its
journal "2026-10-22 — ветка VT-m: формула и правила (до кода и до данных B)"), on pod C after a clean kgrid
(scripts/gpu_c1e4m.sh POD=C mgrid): the runner plan per line, for now. The reading (the pairs, the noise floor's rule,
VT-m - G-auto with B1 and B2) is added to this script before it is read, as the journal requires. Reuses
scripts/c1e4v_summary.py (the controls mode and k, groups) and through it scripts/c1e4c_summary.py; run from the
repository root.
    python scripts/c1e4b_summary.py --plan --line L
        L: Md0 Md10, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4b_summary.py --selftest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import P, maxwrap  # noqa: E402

ARM = "VTm"  # vservo.VTM: lambda 0.3 inherited
BAND = f" --band {cc.OUT}/band_eps.json"  # the journal: model A of band.json, delta 5 mm
# the journal's cells -> (source, d, the controls line it pairs with): d 0 PRO (VT2's, line 2's), d 10 Plus (VT3d10's,
# line 3's)
LINES = {"Md0": ("pro", 0, "2"), "Md10": ("plus", 10, "3")}


def plan(line, split=None):
    """The journal's lines, phase line.lower(), in c1e4c_summary.mlines's format: VTm alone on eval300's cases of one
    source and d, the Base full (no --base-prefix: s* is its first close) in the same process, --band band_eps.json;
    the controls line's cases, order and Base groups of that source and d (asserted), as VT2 PRO and VT3d10 Plus."""
    split = split or maxwrap.load_split(cs.SPLIT)
    src, d, cl = LINES[line]
    one = lambda lns: [ln for ln in lns if ln["source"] == src and f"--d {d} " in ln["args"] + " "]  # noqa: E731
    lns = [ln | {"args": ln["args"] + BAND} for ln in one(cc.mlines(line.lower(), "M", P, "eval300", (ARM,), split, d,
                                                                       prefix=False))]
    ctl = one(cc.plan(cl, cv.CMODE, cv.CK, split))
    assert len(lns) == 1 and cv.groups(lns, split) == cv.groups(ctl, split), \
        f"{line}: not controls line {cl}'s {src} cases, order and groups at d {d}"
    return lns


def selftest():
    """The plan on the real split: one line per cell, VTm alone, no --base-prefix, no kbar, --band band_eps.json, the
    controls line's cases in its groups (every Base full: 113 PRO / 187 Plus cases, two episodes each)."""
    split = maxwrap.load_split(cs.SPLIT)
    for line, ident, src, d, n in (("Md0", "M_pi05_eval300_pro_d0", "pro", 0, 226),
                                   ("Md10", "M_pi05_eval300_plus_d10", "plus", 10, 374)):
        (ln,) = plan(line, split)
        assert ln["id"] == ident and ln["phase"] == line.lower() and ln["arms"] == ["VTm"] and ln["source"] == src, ln
        assert ln["args"] == (f"--brain pi05 --source {src} --set eval300 --arms=VTm --d {d} "
                              "--band results/c1-e4/controls/pod_C/band_eps.json"), ln
        assert not ln["prefix"] and cc.n_of(ln) == n, ln
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--line", choices=LINES)
    a = ap.parse_args()
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    selftest() if a.selftest else print(cc.tsv(plan(a.line)))
