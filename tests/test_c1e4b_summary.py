"""scripts/c1e4b_summary.py (VT-m, B): its selftest, the plan on the real split; a cell against another controls line
is refused."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4b_summary as cb  # noqa: E402


def test_selftest():
    cb.selftest()


def test_plan_asserts_the_controls_groups(monkeypatch):
    for cell in (("pro", 0, "3"), ("plus", 10, "2")):  # no d 0 line in line 3, no d 10 one in line 2
        monkeypatch.setitem(cb.LINES, "Md0", cell)
        with pytest.raises(AssertionError, match="not controls line"):
            cb.plan("Md0")
