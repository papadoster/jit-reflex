"""C1-E4 part 2 on LIBERO-MAX: case selection, the schedule check (g), the Base prefix replayed as recorded chunks and
the calibration line (pure numpy; no LIBERO / LIBERO-MAX import, the runner passes their manifest's cases and LIBERO's
40 task names in).

Spec: docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md."""

import hashlib
import json
from pathlib import Path, PurePosixPath

import numpy as np

SEED = "c1e4m"
N_CAL_CASES = 60  # spec: calibration cases (the same for every brain), first in case_order
N_MIN = 45  # spec: at least 45 valid samples (part-1 memo: n90 ~ 45 for pi0.5), else more cases in order


def base_task(case, names):
    """(suite, original LIBERO task): the longest of LIBERO's task names in the suite that prefixes the case's task
    name (LIBERO-PRO: the init-states file's stem). LIBERO-plus's task_index numbers variants, not tasks."""
    suite = case["task_suite_name"]
    assert len(names[suite]) == 10, (f"{suite}: {len(names[suite])} names; pass standard LIBERO's 40 task names "
                                     "(hf-libero), not a LIBERO-plus overlay's variant list")
    v = case.get("substrate_variant")
    name = PurePosixPath(v["init_states_file"]).name.split(".")[0] if v else case["task_name"]
    hits = [n for n in names[suite] if name.startswith(n)]
    if not hits:
        raise KeyError(f"{case['case_id']}: {name} matches no {suite} task")
    return suite, max(hits, key=len)


def _seed(seed):
    return seed if isinstance(seed, int) else int.from_bytes(seed.encode(), "big") % 2**63


def case_order(cases, names, seed=SEED):
    """A seeded random order with equal task shares: original tasks in a random order, each task's cases in a random
    order, then one case of every task per round while the task has cases left."""
    rng = np.random.default_rng(_seed(seed))
    by = {}
    for c in sorted(cases, key=lambda c: c["case_id"]):
        by.setdefault(base_task(c, names), []).append(c)
    tasks = sorted(by)
    tasks = [tasks[i] for i in rng.permutation(len(tasks))]
    queues = [[by[t][i] for i in rng.permutation(len(by[t]))] for t in tasks]
    out = []
    for r in range(max(map(len, queues))):
        out += [q[r] for q in queues if r < len(q)]
    return out


def split_cases(cases, names, seed=SEED, n_cal=N_CAL_CASES, extra=0):
    """(calibration, evaluation): the first n_cal + extra cases of case_order, then the rest in the same order."""
    order = case_order(cases, names, seed)
    return order[:n_cal + extra], order[n_cal + extra:]


def _sha(d):
    body = json.dumps({k: v for k, v in d.items() if k != "sha256"}, sort_keys=True)
    return hashlib.sha256(body.encode()).hexdigest()


def freeze_split(cases, names, seed=SEED, n_cal=N_CAL_CASES, extra=0, n_conn=300, all_cases=None):
    """The split as case_ids, frozen on the Mac: NumPy does not promise the same Generator stream across versions (the
    pod runs in another venv) and under the LIBERO-plus overlay get_task_names() gives variant names, so the runner
    reads this (load_split) and does not recompute it. all_cases: the whole manifest, for the connection sample."""
    cal, ev = split_cases(cases, names, seed, n_cal, extra)
    d = {"seed": seed, "n_cal": n_cal, "extra": extra, "calib": [c["case_id"] for c in cal],
         "eval": [c["case_id"] for c in ev],
         "connect": [c["case_id"] for c in connection_sample(cases if all_cases is None else all_cases, n_conn, seed)],
         "numpy": np.__version__}
    return {**d, "sha256": _sha(d)}


def load_split(path):
    """freeze_split's dict from its JSON file; ValueError if its SHA-256 does not match."""
    d = json.loads(Path(path).read_text())
    if d.get("sha256") != _sha(d):
        raise ValueError(f"{path}: SHA-256 mismatch")
    return d


def n_extra(samples, more=(), n_min=N_MIN):
    """How many further cases (in order; their samples in `more`, None = no valid sample) bring the valid samples of
    the calibration cases up to n_min. 0 when there are enough already. The runner passes the maximum over brains to
    split_cases(extra=...) / freeze_split(extra=...), so the added cases drop out of evaluation for every brain."""
    have = sum(x is not None for x in samples)
    for k, x in enumerate(more):
        if have >= n_min:
            return k
        have += x is not None
    if have < n_min:
        raise ValueError(f"{have} valid samples < {n_min} after {len(more)} more cases")
    return len(more)


def connection_sample(cases, n, seed=SEED):
    """n cases spread evenly over the change types (Base does not depend on the event), seeded random within a type."""
    rng = np.random.default_rng(_seed(seed))
    by = {}
    for c in sorted(cases, key=lambda c: c["case_id"]):
        by.setdefault(c["scenario"]["change_type"], []).append(c)
    types = sorted(by)
    share = [n // len(types) + (i < n % len(types)) for i in range(len(types))]
    return [by[t][j] for t, k in zip(types, share) for j in sorted(rng.choice(len(by[t]), size=k, replace=False))]


def schedule_check(e, q, plans_in, steps, d=0):
    """Check (g), defined at the native schedule (s = Q, d = 0; not used for d > 0): the last plan observed before the
    event step e is at Q floor((e - 1) / Q) and the first one at or after e at Q ceil(e / Q), as LIBERO-MAX's Dynamic
    (their action queue is not cleared at the event); at d = 0 every plan arrives at its observation step.
    Time: t is LIBERO-MAX's post-warm-up policy step (total_env_steps - warmup_steps); e is their
    cosmos_query_boundary_step, the first policy step whose observation is already post-event (the event is applied
    inside the previous step() call, after physics). On our own bench e = t_fire + 1 (handoff.Perturbation: the
    observation at t_fire is pre-shift).
    plans_in: (t_obs, arrival) per installed plan; steps: the episode's policy steps actually run. The first plan is
    required when steps > Q ceil(e / Q); None only when the episode ended before that step.
    (ok, last t_obs before e, first t_obs at or after e)."""
    before = [t for t, _ in plans_in if t < e]
    after = [t for t, _ in plans_in if t >= e]
    last, first = (max(before) if before else None), (min(after) if after else None)
    q_first = q * -(-e // q)
    on_time = d != 0 or all(a == t for t, a in plans_in)
    return (last == q * ((e - 1) // q) and (first == q_first or first is None and steps <= q_first)
            and on_time), last, first


class ChunkReplay:
    """Dynamic's prefix: until the event the runner gives the agent Base's recorded chunk of each call instead of calling
    the brain, so the agent's state at the event is Base's (the observations are bitwise the same up to it).
    t, event_step: as in schedule_check (LIBERO-MAX's post-warm-up policy step; event_step = their
    cosmos_query_boundary_step, the first step with a post-event observation; on our bench t_fire + 1)."""

    def __init__(self, recorded, event_step):
        self.recorded, self.event_step = recorded, event_step

    def replaying(self, t):
        return t < self.event_step

    def chunk(self, t):
        assert self.replaying(t), f"step {t} >= event step {self.event_step}: the brain plans from the event on"
        return self.recorded[t]


def kbar_line(samples, n_boot=2000, seed=0):
    """(kbar, band 5%, band 95%, N): the median of the valid samples (None dropped) and the percentile bootstrap band
    of that median (resampling the N samples with replacement)."""
    x = np.asarray([s for s in samples if s is not None], float)
    assert len(x) > 0, "no valid sample"
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(len(x), size=(n_boot, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 5)), float(np.percentile(meds, 95)), len(x)
