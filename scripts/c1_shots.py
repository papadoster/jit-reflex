"""Report only (owner's check, after the data): objects 'shot' by the simulator at the shift in the C1-E2 and C1-E3
grids, by cell and class, and the main rules (C1-E2 R1, R4; C1-E3 Q2a, Q5a) with and without those episodes.

The grids keep no traces, so a shot is found by proxy. The owner's definition (> 20 cm within <= 5 steps, no contact)
is checked on the C1-E3 pilot traces; there every shot has the 'unstable' flag and grasp miss > 15 cm at the close.
A shot happens at the shift, before any method acts, so it is shared by all arms of the episode; an arm knocking
the object is not. Proxy: 'unstable' and grasp miss > 15 cm in EVERY arm recorded for the episode (in a rule: in
both arms of the pair).
    ~/Desktop/M2R-c1-env/bin/python scripts/c1_shots.py
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import c1e2_summary as e2  # noqa: E402

MISS = 0.15


def far(r):
    return bool(r.get("unstable")) and r.get("grasp_miss_m") is not None and r["grasp_miss_m"] > MISS


def pilot():
    """Owner's definition on the C1-E3 pilot traces, and how the single-record proxy compares."""
    out = {"n": 0, "shots": 0, "proxy": 0, "proxy_and_shot": 0}
    for f in ("results/c1-e3/pilot.jsonl", "results/c1-e3/pilot_b.jsonl"):
        for line in open(f):
            r = json.loads(line)
            if not r.get("trace") or r["kind"] == "control" or r["t_fire"] is None:
                continue
            p, c, tf = np.array(r["trace"]["p"]), np.array(r["trace"]["c"]), r["t_fire"]
            delta = r["mag"] * np.array([np.cos(r["angle"]), np.sin(r["angle"]), 0.0])
            shot = bool(np.linalg.norm(p[tf + 1:tf + 6] - (p[tf] + delta), axis=1).max() > 0.20
                    and not c[tf:tf + 6].any())
            out["n"] += 1
            out["shots"] += shot
            out["proxy"] += far(r)
            out["proxy_and_shot"] += shot and far(r)
    return out


def load(path, key):
    idx = defaultdict(dict)
    for line in open(path):
        r = json.loads(line)
        if r["suite"] != "libero_10":
            idx[key(r)][r["suite"], r["task"], r["init"]] = r
    return idx


def shares(idx, kinds, arm_of):
    """Share of shot episodes (proxy in every recorded arm, >= 2 arms) by cell and class, fired shifts only."""
    eps = defaultdict(list)
    for k, recs in idx.items():
        if k[0] in kinds and arm_of(k) is not None:
            for e, r in recs.items():
                if r["t_fire"] is not None:
                    eps[k[0], k[1], e].append(r)
    by = defaultdict(lambda: [0, 0])
    for (kind, cell, e), rs in eps.items():
        if len(rs) < 2:
            continue
        hit = all(far(r) for r in rs)
        for key in (f"{kind} all", f"{kind} cell {cell}", f"{kind} class {rs[0]['mag_class']}"):
            by[key][0] += hit
            by[key][1] += 1
    return {k: [v[0], v[1], round(100 * v[0] / v[1], 2)] for k, v in sorted(by.items())}


def rule(idx, kind, cells, a, b, inits=None, cls=None, key=lambda kind, c, m: (kind, c, m)):
    """The rule's difference on all pairs and without pairs where both records carry the proxy."""
    out = {}
    for name, drop in (("all", False), ("no_shots", True)):
        prs, n_drop = [], 0
        for c in cells:
            x, y = idx.get(key(kind, c, a), {}), idx.get(key(kind, c, b), {})
            for e in sorted(x.keys() & y.keys()):
                if (inits is not None and e[2] not in inits) or (cls is not None and y[e]["mag_class"] not in cls):
                    continue
                if drop and far(x[e]) and far(y[e]):
                    n_drop += 1
                    continue
                prs.append((e[:2], x[e]["success"], y[e]["success"]))
        out[name] = e2.rnd(e2.diff_pp(prs)) | {"dropped": n_drop}
    return out


def main():
    i2 = load("results/c1-e2/grid_all.jsonl", lambda r: (r["kind"], r["cell"], r["method"]))
    i3 = load("results/c1-e3/grid_all.jsonl", lambda r: (r["kind"], r["cell"], r["arm"], r["alpha"]))
    k3 = lambda kind, c, m: (kind, c, m, 0.0)  # noqa: E731
    out = {"pilot": pilot(),
           "c1e2_shares": shares(i2, ("step",), lambda k: k[2]),
           "c1e3_shares": shares(i3, ("step", "close"), lambda k: k[2] if k[3] == 0.0 else None),
           "c1e2_R1": rule(i2, "step", e2.CELLS, "G", "none"),
           "c1e2_R4": rule(i2, "step", ["F", "Gp"], "GT", "T0"),
           "c1e3_Q2a": rule(i3, "step", "ABC", "GRT", "GT", range(20), (2,), k3),
           "c1e3_Q5a": rule(i3, "step", "ACF", "GR@real", "none", range(20), None, k3)}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
