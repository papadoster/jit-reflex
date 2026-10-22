"""scripts/c1e4b_summary.py (VT-m, B): its selftest (the plan on the real split, the noise floor's rule, the reading on
a synthetic pod); a cell against another controls line is refused; the reading's s* check on records the runner writes
itself (scripts/c1e4m_run.py on its toy env)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4b_summary as cb  # noqa: E402
import test_c1e4m_run as tr  # noqa: E402  (the runner's toy env and brains)


def test_selftest():
    cb.selftest()


def test_plan_asserts_the_controls_groups(monkeypatch):
    for cell in (("pro", 0, "3"), ("plus", 10, "2")):  # no d 0 line in line 3, no d 10 one in line 2
        monkeypatch.setitem(cb.LINES, "Md0", cell)
        with pytest.raises(AssertionError, match="not controls line"):
            cb.plan("Md0")


def test_s_star_and_floor_on_the_runners_records(monkeypatch):
    """The runner's VTm record and its full Base pass the reading's s* check, with or without a close in the Base; a
    Base said to close elsewhere, or a record without "vtm", does not."""
    monkeypatch.setitem(tr.run.MAX_STEPS, "toy", 60)  # as test_c1e4m_run's fixture
    for brain, closes in ((tr.Closing(), True), (tr.Brain(), False)):
        base, dyn = tr.by_arm(tr.run.run_cases([tr.CASE], ["VTm"], {**tr.conf(), "pbvs": tr.BAND},
                                               [tr.Toy(), tr.Toy()], brain))
        r = dyn["VTm"]
        assert (base["t_close"] is not None) == closes and cb.ref_ok(r, base), closes
        moved = base | ({"grasp_miss_m": base["grasp_miss_m"] + 2e-5} if closes
                        else {"t_close": 40, "grasp_miss_m": 0.01})
        assert not cb.ref_ok(r, moved) and not cb.ref_ok({k: v for k, v in r.items() if k != "vtm"}, base), closes
