import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4c_summary as cc  # noqa: E402


def test_selftest():
    cc.selftest()
