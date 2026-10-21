"""scripts/c1e4t_summary.py (VS-T): its selftest, and its s* check on records the runner writes itself
(scripts/c1e4m_run.py on its toy env)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4t_summary as ct  # noqa: E402
import test_c1e4m_run as tr  # noqa: E402  (the runner's toy env and brains)


def test_selftest():
    ct.selftest()


def test_s_star_check_on_the_runners_records(monkeypatch):
    """The runner's VST record and its Base pass the reading's s* check (|s*| rounded to 1e-6 against the Base's grasp
    miss), with or without a close in the Base; a Base said to close elsewhere does not."""
    monkeypatch.setitem(tr.run.MAX_STEPS, "toy", 60)  # as test_c1e4m_run's fixture
    for brain, closes in ((tr.Closing(), True), (tr.Brain(), False)):
        base, dyn = tr.by_arm(tr.run.run_cases([tr.CASE], ["VST0.3"], tr.conf(), [tr.Toy(), tr.Toy()], brain))
        assert (base["t_close"] is not None) == closes and ct.ref_ok(dyn["VST0.3"], base), closes
        moved = base | ({"grasp_miss_m": base["grasp_miss_m"] + 2e-5} if closes
                        else {"t_close": 40, "grasp_miss_m": 0.01})
        assert not ct.ref_ok(dyn["VST0.3"], moved), closes
        assert not ct.ref_ok(dyn["VST0.3"] | {"vst": None}, base)
