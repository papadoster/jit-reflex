"""scripts/c1e4r_summary.py (cell R, docs/superpowers/specs/2026-11-17-c1-e4-delay-trend-design.md): its selftest
(the plan on the real split, the smoke's check, R0's gate, the reading on a synthetic pod: both rules, the noise floor's
gate, each check alone, drops by reason); the cells' plans against each other; the CLI's failing exits; the live-stage
guard of scripts/gpu_c1e4m.sh's rgrid stages; the old 300's pairs against the VS-T reading's Т1 (pod C)."""

import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import c1e4c_summary as cc  # noqa: E402
import c1e4r_summary as cr  # noqa: E402

SPLIT = cr.maxwrap.load_split(cr.cs.SPLIT)


def test_selftest():
    cr.selftest()


def test_cells_run_the_same_cases():
    """spec §2: R0 and R20 the same 640 cases, sources, order and groups, only d apart; the smoke R0's first group."""
    r0, r20 = cr.plan("R0", SPLIT), cr.plan("R20", SPLIT)
    assert [ln["args"].replace(" --d 0 ", " --d 20 ") for ln in r0] == [ln["args"] for ln in r20]
    assert {g for _, g in cr.cv.groups(r0, SPLIT)} == {g for _, g in cr.cv.groups(r20, SPLIT)}
    assert sum(cc.n_of(ln) for ln in r0 + r20) == 6400  # spec §2: 640 x (Base + 4) per cell
    (sm,) = cr.plan("Rsmoke", SPLIT)
    assert cr.cv.ids(sm, SPLIT) == cr.cv.ids(r0[0], SPLIT)[:6] and sm["args"].split(" --cases ")[0] == \
        r0[0]["args"].split(" --kbar")[0]


def test_cli_fails_on_missing_records(tmp_path):
    f = tmp_path / "empty.jsonl"
    f.write_text("")
    for args, last in ((["--smoke-check", str(f)], "C1-E4 delay smoke (code only): FAIL"),
                       (["--complete", str(f), "--line", "R0"], "C1-E4 delay R0 complete: FAIL")):
        p = subprocess.run([sys.executable, "scripts/c1e4r_summary.py", *args], cwd=ROOT, capture_output=True,
                           text=True)
        assert p.returncode == 3 and p.stdout.strip().splitlines()[-1] == last, (p.stdout, p.stderr)


def live(stages):
    """scripts/gpu_c1e4m.sh's live_re for these stages, from the script itself."""
    sh = "eval \"$(sed -n '/^live_re() {/,/^}/p' scripts/gpu_c1e4m.sh)\"; live_re \"$1\""
    return subprocess.run(["bash", "-c", sh, "-", stages], cwd=ROOT, capture_output=True, text=True,
                          check=True).stdout.strip()


def test_live_guard_matches_processes_not_their_mentions():
    """pgrid refused on pod P because the tmux server's argv held "gpu_c1e4m.sh setup"
    (results/c1-e4/placebo/console_P/pgrid_P_refused.log): the R stages' guard matches a stage only as the process that
    runs the script (bash, the script, the stage) and a runner; grep -E is pgrep's POSIX ERE."""
    st = "setup|rsmoke|rgrid20"  # rgrid0's view: every other stage
    hits = ["bash ./scripts/gpu_c1e4m.sh setup", "/usr/bin/bash scripts/gpu_c1e4m.sh rsmoke",
            "bash /workspace/jit-reflex/scripts/gpu_c1e4m.sh rgrid20",
            "venv/bin/python scripts/c1e4m_run.py --device cuda --brain pi05",
            "bash -o pipefail -c \"${@:2}\" wd log env X=1 venv/bin/python scripts/c1e4m_run.py --brain pi05"]
    miss = ["tmux new -d -s c1e4m cd /workspace && git clone -q https://github.com/papadoster/jit-reflex.git && cd "
            "jit-reflex && git checkout -q c1-e4 && git log --oneline -1 && POD=R ./scripts/gpu_c1e4m.sh setup 2>&1 | "
            "tee /workspace/setup_R.log; exec bash",
            "bash -c cd /workspace/jit-reflex && POD=R ./scripts/gpu_c1e4m.sh setup 2>&1 | tee setup_R.log",
            "sh -c POD=R ./scripts/gpu_c1e4m.sh rsmoke", "bash ./scripts/gpu_c1e4m.sh rgrid0",
            "tail -f results/c1-e4/delay/pod_R/logs/rgrid0.log", "bash ./scripts/gpu_c1e4m.sh setups"]
    p = subprocess.run(["grep", "-E", live(st)], input="\n".join(hits + miss) + "\n", capture_output=True, text=True)
    assert p.stdout.splitlines() == hits, p.stdout


@pytest.mark.skipif(not (ROOT / cr.CPOD / "grid_all_T.jsonl").exists(), reason="pod C's records not here")
def test_old300_pairs_are_t1s():
    """spec §5: the old 300's pairs (VT2 / VT3d20 against lines 2 / 3) are the VS-T reading's Т1 at d 0 and d 20: the
    same pairs and estimate as summary_T.json (code only: the values are compared, never printed)."""
    names = pytest.importorskip("libero.libero") and cr.cs.libero_names()
    old = cr.old300(ROOT / cr.CPOD, names, SPLIT)
    t1 = json.loads((ROOT / cr.CPOD / "summary_T.json").read_text())["measurements"]
    for d in (0, 20):
        p, bad = old[d]
        e = cr.e2.rnd(cr.ci([(v[0], [1, float(v[1]) - float(v[2])]) for v in p.values()], cr.MEAN,
                            cr.resamples({v[0] for v in p.values()})))
        m = t1[f"Т1/d{d}"]
        assert (e["diff_pp"], e["pairs"], e["clusters"], bad) == (m["diff_pp"], m["pairs"], m["clusters"],
                                                                  m["invalid_pairs"]), d
