"""C1-E4 part 2, block B: the bench calibration pairs of G-auto's kbar (kind step, inits 44-47, cell A, every task of
orx.TASKS) in maxwrap.case_order; the first maxwrap.N_CAL_CASES are calibration ("calib"), the rest follow in the same
order ("more": further pairs while N < maxwrap.N_MIN). Frozen with a SHA-256 as maxwrap.freeze_split (load_split checks
it); scripts/c1e4_run.py --pairs-file reads "calib". Not data: no brain, no env. Task names: hf-libero's benchmark.
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e4b_calib_pairs.py
"""

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import maxwrap  # noqa: E402
import objreflex as orx  # noqa: E402
from libero.libero import benchmark  # noqa: E402

INITS = [44, 45, 46, 47]
OUT = ROOT / "results/c1-e4/part2/bench_calib_pairs.json"

bench = benchmark.get_benchmark_dict()
names = {s: bench[s]().get_task_names() for s in sorted({s for s, _ in orx.TASKS})}
cases = [{"case_id": f"{s}/{t}/{i}", "task_suite_name": s, "task_name": names[s][t], "init_state_index": i,
          "pair": {"suite": s, "task": t, "init": i}} for s, t in orx.TASKS for i in INITS]
order = [c["pair"] for c in maxwrap.case_order(cases, names)]
assert len({tuple(q.values()) for q in order}) == len(cases) == 4 * len(orx.TASKS)
n = maxwrap.N_CAL_CASES
d = {"seed": maxwrap.SEED, "kind": "step", "cell": "A", "inits": INITS, "n_cal": n, "calib": order[:n],
     "more": order[n:], "numpy": np.__version__}
d["sha256"] = maxwrap._sha(d)
OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(d, indent=1) + "\n")
assert maxwrap.load_split(OUT) == d
print(f"{OUT.relative_to(ROOT)}: {n} calib + {len(order) - n} more pairs, sha256 {d['sha256']}")
