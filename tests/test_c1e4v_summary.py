"""scripts/c1e4v_summary.py (PBVS spec §9 item 9): its selftest, and the band's conventions on a Base record the runner
writes itself (scripts/c1e4m_run.py --log-traj on its toy env)."""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4v_summary as cv  # noqa: E402
import objreflex as orx  # noqa: E402
import test_c1e4m_run as tr  # noqa: E402  (the runner's toy env and brain)


def test_selftest():
    cv.selftest()


class Wavy(tr.Brain):
    """The toy brain with rows that change along the chunk, some beyond +-1 (the runner clips them)."""

    def __call__(self, *a):
        out = super().__call__(*a)
        out[:, :, :2] += 1.5 * np.sin(0.7 * np.arange(orx.H))[None, :, None]
        return out


def test_band_errors_on_a_runner_base_record(monkeypatch):
    """The toy env's arm is the integrator G_POS on every axis: on the runner's own Base record (x per step before its
    action, the executed clipped action, plans_in, t_close) the path of model I with G_POS is the arm's to the traj's
    1e-6 rounding, over plans whose rows change step by step; one step off in the action's index errs by millimetres."""
    monkeypatch.setitem(tr.run.MAX_STEPS, "toy", 60)  # as test_c1e4m_run's fixture
    (base,) = tr.run.run_cases([tr.CASE], [], tr.conf(), [tr.Toy()], Wavy(), log_traj=True)
    e = cv.errors(base, [orx.G_POS] * 2, [0.0, 0.0])
    a = np.asarray(base["traj"]["a"])[:, :2]
    assert len(base["plans_in"]) > 3 and len(e) == base["steps"] and np.abs(np.diff(a, axis=0)).max() > 0.1
    assert max(e) < 3e-6, max(e)
    shifted = base | {"traj": base["traj"] | {"a": base["traj"]["a"][1:] + base["traj"]["a"][-1:]}}
    assert max(cv.errors(shifted, [orx.G_POS] * 2, [0.0, 0.0])) > 1e-3


def test_v2_runs_line_2s_cases_and_batches():
    """The real split: V2's first 108 Plus cases are the frozen VS1.0 file's (line 2's C2 line), and V2, Nglr and V2eps
    run line 2's runner groups, V3 line 3's."""
    split = cv.maxwrap.load_split(cv.cs.SPLIT)
    pk, c2 = {"arm": "PBVS1.0"}, cv.cc.plan("2", cv.CMODE, cv.CK, split)
    v2 = cv.plan("V2", pk, split=split)
    assert cv.ids(v2[0], split)[:108] == cv.ids(c2[0], split) and c2[0]["id"] == "C2_pi05_eval300_plus_d0"
    for L in ("V2", "Nglr", "V2eps"):
        assert cv.groups(cv.plan(L, pk, {"delta": 0.03}, split), split) == cv.groups(c2, split), L
    assert cv.groups(cv.plan("V3", pk, split=split), split) == cv.groups(cv.cc.plan("3", cv.CMODE, cv.CK, split), split)
