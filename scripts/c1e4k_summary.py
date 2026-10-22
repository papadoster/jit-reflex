"""C1-E4 kappa-law ablation (C): G-auto - G(k = 1.0) along the delay axis (docs/superpowers/specs/
2026-10-20-c1-e4-pbvs-design.md, its journal "2026-10-22 — рецензия: задачи A, B, C", C), on pod C after a clean vtgrid
(scripts/gpu_c1e4m.sh POD=C kgrid): the runner plan per line. G(k = 1.0) is VS1.0, the controls spec §5's k_π
(k_select.json), G by code (tests/test_gauto.py test_vs1_is_g_by_executed_code). Reuses scripts/c1e4v_summary.py (the
controls mode and k, groups) and through it scripts/c1e4c_summary.py; run from the repository root.
    python scripts/c1e4k_summary.py --plan --line L
        L: Kd10 Kd20, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4k_summary.py --selftest
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import P, maxwrap  # noqa: E402

ARM = "VS1.0"  # the controls spec §5's k_π (k_select.json on pod C): G with kappa fixed at 1.0
LINES = {"Kd10": 10, "Kd20": 20}  # the journal: Plus only; d = 0 is line 2's frozen file (G-auto and VS1.0 together)


def plan(line, split=None):
    """The journal's lines, phase line.lower(), in c1e4c_summary.mlines's format: VS1.0 alone on eval300's Plus cases
    at d 10 / 20 with --base-prefix, controls line 3's Plus cases, order and Base groups of that d (asserted): the pairs
    with its G-auto, as V3's PBVS."""
    split = split or maxwrap.load_split(cs.SPLIT)
    d = LINES[line]
    plus = lambda lns: [ln for ln in lns if ln["source"] == "plus" and f"--d {d} " in ln["args"]]  # noqa: E731
    lns = plus(cc.mlines(line.lower(), "K", P, "eval300", (ARM,), split, d))
    ctl = plus(cc.plan("3", cv.CMODE, cv.CK, split))
    assert len(lns) == 1 and cv.groups(lns, split) == cv.groups(ctl, split), \
        f"{line}: not controls line 3's Plus cases, order and groups"
    return lns


def selftest():
    """The plan on the real split: one Plus line per d, VS1.0 alone, --base-prefix, no kbar, line 3's 187 Plus cases
    in its groups (187 episodes: the Bases are cut at the event), ids K_pi05_eval300_plus_d10 / _d20."""
    split = maxwrap.load_split(cs.SPLIT)
    for line, d in LINES.items():
        (ln,) = plan(line, split)
        assert ln["id"] == f"K_{P}_eval300_plus_d{d}" and ln["phase"] == line.lower() and ln["arms"] == [ARM], ln
        assert "--base-prefix" in ln["args"] and "--kbar" not in ln["args"] and cc.n_of(ln) == 187, ln
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
