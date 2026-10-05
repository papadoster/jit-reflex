"""C1-E4 part 2 on LIBERO-MAX: case selection, the schedule check (g), the Base prefix replayed as recorded chunks and
the calibration line (pure numpy; no LIBERO / LIBERO-MAX import, the runner passes their manifest's cases and LIBERO's
40 task names in).

Spec: docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md."""

from pathlib import PurePosixPath

import numpy as np

SEED = "c1e4m"
N_CAL_CASES = 60  # spec: calibration cases (the same for every brain), first in case_order
N_MIN = 45  # spec: at least 45 valid samples (part-1 memo: n90 ~ 45 for pi0.5), else more cases in order


def base_task(case, names):
    """(suite, original LIBERO task): the longest of LIBERO's task names in the suite that prefixes the case's task
    name (LIBERO-PRO: the init-states file's stem). LIBERO-plus's task_index numbers variants, not tasks."""
    suite = case["task_suite_name"]
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


def n_extra(samples, more=(), n_min=N_MIN):
    """How many further cases (in order; their samples in `more`, None = no valid sample) bring the valid samples of
    the calibration cases up to n_min. 0 when there are enough already."""
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


def schedule_check(e, q, plans_in):
    """Check (g): with the native schedule (s = Q, d = 0) the last plan observed before the event step e is at
    Q floor((e - 1) / Q) and the first one at or after e at Q ceil(e / Q), as LIBERO-MAX's Dynamic (their action queue
    is not cleared at the event). plans_in: (t_obs, arrival) per installed plan; an episode that ends before
    Q ceil(e / Q) has no first. (ok, last t_obs before e, first t_obs at or after e)."""
    before = [t for t, _ in plans_in if t < e]
    after = [t for t, _ in plans_in if t >= e]
    last, first = (max(before) if before else None), (min(after) if after else None)
    return last == q * ((e - 1) // q) and first in (None, q * -(-e // q)), last, first


class ChunkReplay:
    """Dynamic's prefix: until the event the runner gives the agent Base's recorded chunk of each call instead of calling
    the brain, so the agent's state at the event is Base's (the observations are bitwise the same up to it)."""

    def __init__(self, recorded, event_step):
        self.recorded, self.event_step = recorded, event_step

    def replaying(self, t):
        return t < self.event_step

    def chunk(self, t):
        return self.recorded[t]


def kbar_line(samples, n_boot=2000, seed=0):
    """(kbar, band 5%, band 95%, N): the median of the valid samples (None dropped) and the percentile bootstrap band
    of that median (resampling the N samples with replacement)."""
    x = np.asarray([s for s in samples if s is not None], float)
    rng = np.random.default_rng(seed)
    meds = np.median(x[rng.integers(len(x), size=(n_boot, len(x)))], axis=1)
    return float(np.median(x)), float(np.percentile(meds, 5)), float(np.percentile(meds, 95)), len(x)
