"""C1-E4 part 2, a measurement on recorded data (part-2 spec journal "2026-11-08 — доля доведённых захватов и
разложение разрыва Base − G-auto при d = 20", written and pushed before the computation). A measurement, not a rule.

pi0.5, d = 20, part 2's D line (pod A), Base, none, PPC, Gauto on the cases where all four ran and are valid:
- c_X = #(grasp_ok and success) / #grasp_ok, the share of grasps carried to success; g_X grasp share, s_X success,
  r_X success without grasp_ok;
- the gap s_Base - s_Gauto = (s_Base - S*) + (S* - s_Gauto), S* = g_Gauto c_Base + r_Gauto: the loss at the grasp and
  the loss after it.
Intervals: percentile bootstrap over the source-task clusters (maxwrap.base_task), 2000 resamples, default_rng(0), one
cluster draw for every quantity, each a ratio of sums over the drawn clusters.

Run from the repository root (the Mac venv: hf-libero for the 40 task names):
    python scripts/c1e4_grasp_split.py [--out results/c1-e4/part2/grasp_split_d20.json]
    python scripts/c1e4_grasp_split.py --selftest
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
from c1e4m_summary import e2, maxwrap  # noqa: E402

FILES = [f"results/c1-e4/part2/pod_A/D_pi05_eval300_{s}_d20.jsonl" for s in ("plus", "pro")]
ARMS = ("Base", "none", "PPC", "Gauto")
FIELDS = ("n", "g", "sg", "s", "r")  # per cluster and arm: episodes, grasps, grasped successes, successes, ungrasped ones


def table(recs):
    """case -> {arm: record} for the cases where all four arms ran and are valid; (table, dropped case ids)."""
    by = defaultdict(dict)
    for r in recs:
        if r.get("brain") != "pi05" or r.get("d") != 20:
            continue
        a = "Base" if r["kind"] == "base" else r["arm"]
        if a not in ARMS:
            continue
        old = by[r["case_id"]].get(a)
        if old is not None:
            if (old["success"], old["grasp_ok"]) != (r["success"], r["grasp_ok"]):
                raise SystemExit(f"{r['case_id']} {a}: recorded twice with different outcomes")
            continue
        by[r["case_id"]][a] = r
    bad = lambda r: (r["skipped_reason"] is not None or r["success"] is None  # noqa: E731
                     or (r["kind"] == "dynamic" and cs.invalid(r)))
    keep = {c: v for c, v in by.items() if len(v) == len(ARMS) and not any(bad(r) for r in v.values())}
    return keep, sorted(set(by) - set(keep))


def counts(tab, clu):
    """{field: array (clusters, arms)} of sums, clusters in sorted order."""
    keys = sorted({clu[c] for c in tab})
    out = {f: np.zeros((len(keys), len(ARMS))) for f in FIELDS}
    for c, v in tab.items():
        k = keys.index(clu[c])
        for j, a in enumerate(ARMS):
            g, s = bool(v[a]["grasp_ok"]), bool(v[a]["success"])
            for f, x in zip(FIELDS, (1, g, g and s, s, s and not g)):
                out[f][k, j] += x
    return out, keys


def measures(x):
    """x: {field: (..., arms) sums} -> the shares (%) and the split (pp), broadcast over leading axes."""
    n, g, sg, s, r = (x[f] for f in FIELDS)
    with np.errstate(invalid="ignore", divide="ignore"):
        c = sg / g
    sh = {"c": 100 * c, "g": 100 * g / n, "s": 100 * s / n, "r": 100 * r / n}
    B, G = ARMS.index("Base"), ARMS.index("Gauto")
    s_star = g[..., G] / n[..., G] * c[..., B] + r[..., G] / n[..., G]
    sp = {"gap": sh["s"][..., B] - sh["s"][..., G], "S_star": 100 * s_star,
          "after_grasp": 100 * s_star - sh["s"][..., G], "at_grasp": sh["s"][..., B] - 100 * s_star}
    return sh, sp


def compute(tab, clu):
    x, keys = counts(tab, clu)
    pick = np.random.default_rng(0).integers(len(keys), size=(e2.N_BOOT, len(keys)))
    sh, sp = measures({f: v.sum(0) for f, v in x.items()})
    bsh, bsp = measures({f: v[pick].sum(1) for f, v in x.items()})
    # a resample whose drawn clusters hold no grasp of an arm leaves its c undefined: the band is over the defined
    # resamples and their number is reported (with 33 clusters it should not happen)
    ci = lambda v: [round(float(q), 3) for q in np.nanpercentile(v, [2.5, 97.5])]  # noqa: E731
    tot = {f: x[f].sum(0) for f in FIELDS}
    arms = {a: {"n": int(tot["n"][j]), "grasps": int(tot["g"][j]), "grasped_successes": int(tot["sg"][j]),
                "successes": int(tot["s"][j]), "ungrasped_successes": int(tot["r"][j]),
                **{k: {"est": round(float(sh[k][j]), 3), "ci95": ci(bsh[k][:, j])} for k in ("c", "g", "s", "r")},
                "c_undefined_resamples": int(np.isnan(bsh["c"][:, j]).sum())}
            for j, a in enumerate(ARMS)}
    split = {k: {"est": round(float(sp[k]), 3), "ci95": ci(bsp[k])} for k in sp}
    split["undefined_resamples"] = int(np.isnan(bsp["S_star"]).sum())
    split["after_share_of_gap"] = round(float(sp["after_grasp"] / sp["gap"]), 4) if sp["gap"] else None
    return {"measurement": "измерение, не правило (part-2 spec journal 2026-11-08)", "cases": len(tab),
            "clusters": len(keys), "resamples": e2.N_BOOT, "arms": arms, "split_base_minus_gauto": split}


def selftest():
    # two clusters; per arm and cluster: (grasp_ok, success) of each case
    plan = {"k0": {"Base": [(1, 1), (1, 1), (1, 0), (0, 0)], "none": [(0, 0), (1, 0), (0, 0), (0, 1)],
                   "PPC": [(1, 0), (0, 0), (0, 0), (0, 0)], "Gauto": [(1, 1), (1, 0), (1, 0), (0, 0)]},
            "k1": {"Base": [(1, 1), (0, 1), (0, 0), (0, 0)], "none": [(0, 0), (0, 0), (0, 0), (0, 0)],
                   "PPC": [(1, 1), (0, 0), (0, 0), (0, 0)], "Gauto": [(1, 0), (1, 1), (0, 0), (0, 1)]}}
    recs, clu = [], {}
    for k, arms in plan.items():
        for i in range(4):
            c = f"{k}-c{i}"
            clu[c] = k
            for a, v in arms.items():
                g, s = v[i]
                recs.append({"brain": "pi05", "d": 20, "case_id": c, "kind": "base" if a == "Base" else "dynamic",
                             "arm": "none" if a == "Base" else a, "grasp_ok": bool(g), "success": bool(s),
                             "skipped_reason": None, "prefix_ok": True, "replay_miss": None})
    # a dropped case: an invalid Dynamic record; a duplicate with the same outcome is kept once
    recs += [dict(recs[0]), {**recs[1], "case_id": "k1-bad", "prefix_ok": False},
             *[{**r, "case_id": "k1-bad"} for r in recs[2:5]]]
    clu["k1-bad"] = "k1"
    tab, dropped = table(recs)
    assert len(tab) == 8 and dropped == ["k1-bad"], (len(tab), dropped)
    out = compute(tab, clu)
    A, S = out["arms"], out["split_base_minus_gauto"]
    # Base: grasps 4, grasped successes 3, successes 4, ungrasped 1; Gauto: grasps 5, grasped 2, successes 3, ungrasped 1
    assert (A["Base"]["grasps"], A["Base"]["grasped_successes"], A["Base"]["ungrasped_successes"]) == (4, 3, 1)
    assert (A["Gauto"]["grasps"], A["Gauto"]["successes"], A["Gauto"]["ungrasped_successes"], A["Base"]["n"]) \
        == (5, 3, 1, 8)
    assert abs(A["Base"]["c"]["est"] - 75.0) < 1e-9 and abs(A["Gauto"]["c"]["est"] - 40.0) < 1e-9
    assert abs(A["none"]["c"]["est"] - 0.0) < 1e-9 and A["none"]["ungrasped_successes"] == 1
    # S* = 5/8 * 0.75 + 1/8 = 59.375 %; gap 50 - 37.5; after 59.375 - 37.5; at the grasp (4/8 - 5/8) 0.75 + (1/8 - 1/8)
    for k, v in (("gap", 12.5), ("S_star", 59.375), ("after_grasp", 21.875), ("at_grasp", -9.375)):
        assert abs(S[k]["est"] - v) < 1e-9, (k, S[k])
    x, _ = counts(tab, clu)
    pick = np.random.default_rng(1).integers(2, size=(50, 2))
    _, bsp = measures({f: v[pick].sum(1) for f, v in x.items()})
    ok = np.isfinite(bsp["gap"])
    assert np.allclose(bsp["gap"][ok], (bsp["after_grasp"] + bsp["at_grasp"])[ok]), "the split must sum to the gap"
    try:
        table(recs + [{**recs[0], "success": False}])
        raise AssertionError("a duplicate with a different outcome must stop")
    except SystemExit:
        pass
    print("selftest ok")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out", default="results/c1-e4/part2/grasp_split_d20.json")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    recs = [json.loads(ln) for f in FILES for ln in open(f) if ln.strip()]
    tab, dropped = table(recs)
    names = cs.libero_names()
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": v["Base"]["suite"], "task_name": v["Base"]["task_name"],
                                 "substrate_variant": v["Base"]["substrate_variant"]}, names) for c, v in tab.items()}
    out = compute(tab, {c: "/".join(k) for c, k in clu.items()}) | {"dropped_cases": dropped, "files": FILES}
    Path(a.out).write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n")
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
