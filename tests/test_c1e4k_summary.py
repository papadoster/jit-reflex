"""scripts/c1e4k_summary.py (the kappa law's ablation, C): its selftest, the plan on the real split."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4k_summary as ck  # noqa: E402


def test_selftest():
    ck.selftest()
