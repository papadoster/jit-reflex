"""C1-E4 cell R, the descriptive 'none' branch (spec 2026-11-17-c1-e4-delay-trend-design.md, journal 2026-11-19: the
owner's decision after the spec closed; outside Q8 and Q9, read separately after cell R). Not a rule: no sentence, no
gate. Per cell (d = 0, 20): the success of 'none' (shift, no correction) on its valid cases, then on the cases valid in
both runs the reflex arms (pod R's R0/R20 pairs, as c1e4r_summary.read) against 'none' (pod R's N0/N20, a Base of its
own, cut at the event): a CROSS-RUN pair by case, not one process on one Base. M (VT-m) stays unread: cell R's noise
floor gate is closed. Bootstrap: c1e4r_summary's (the same 33 task clusters, 8000 draws, default_rng(0)). Run from the
repository root:
    python scripts/c1e4r_none.py [--out results/c1-e4/delay/pod_R/summary_N.json]
"""
import argparse
import json
from pathlib import Path

import c1e4r_summary as cr
from c1e4r_summary import ARMS, CELLS, SIDE, cs, e2, maxwrap, run

R = Path("results/c1-e4/delay/pod_R")
NCELL = {"R0": "N0", "R20": "N20"}
READ = ("G", "T", "V5")  # M: not read (cell R's gate)


def main(out=None):
    names, split = cs.libero_names(), maxwrap.load_split(cs.SPLIT)
    cases = run.set_ids(split, "evalrest")
    rec = [r for r in e2.load(R / "grid_all_R.jsonl") if "kind" in r]
    nrec = [r for f in sorted(R.glob("N*_pi05_evalrest_*.jsonl")) for r in e2.load(f) if "kind" in r]
    clu = {r["case_id"]: maxwrap.base_task({"case_id": r["case_id"], "task_suite_name": r["suite"],
                                            "task_name": r["task_name"], "substrate_variant": r["substrate_variant"]},
                                           names) for r in rec}
    bs = cr.resamples(set(clu.values()))
    res = {"note": "описательно, не правило; пары «ветка − нет» межпрогонные (свой Base у «нет»); M не читается"}
    for L, d in CELLS.items():
        ok = lambda r: r["d"] == d and not r.get("skipped_reason")  # noqa: E731
        base = {r["case_id"]: r for r in rec if ok(r) and r["kind"] == "base"}
        arm = {a: {r["case_id"]: r for r in rec if ok(r) and r["kind"] == "dynamic" and r["arm"] == a} for a in ARMS}
        nb = {r["case_id"]: r for r in nrec if ok(r) and r["kind"] == "base"}
        nn = {r["case_id"]: r for r in nrec if ok(r) and r["kind"] == "dynamic" and r["arm"] == "none"}
        pr = {c: {SIDE[a]: arm[a][c] for a in ARMS} for c in cases if c in base and base[c]["e"] is not None
              and all(c in arm[a] and not cs.invalid(arm[a][c]) for a in ARMS)}  # c1e4r_summary.read's pair
        nv = {c: nn[c] for c in cases if c in nb and nb[c]["e"] is not None and c in nn and not cs.invalid(nn[c])}
        both = [c for c in cases if c in pr and c in nv]
        prs = [(c, clu[c], {**{k: float(pr[c][k]["success"]) for k in READ}, "none": float(nv[c]["success"])})
               for c in both]
        src = {s: [p for p in prs if p[0] in {c for c in both if base[c]["source"] == s}] for s in ("plus", "pro")}
        res[L] = {
            "none_valid": {"success": round(sum(r["success"] for r in nv.values()) / len(nv), 4), "cases": len(nv),
                           "drops": {"missing": sum(c not in nb or c not in nn for c in cases),  # disjoint, in order
                                     "no_event": sum(c in nb and c in nn and nb[c]["e"] is None for c in cases),
                                     "invalid": sum(c in nb and c in nn and nb[c]["e"] is not None
                                                    and cs.invalid(nn[c]) for c in cases)}},
            "cross_run": {"cases": len(both),
                          "success": {k: round(sum(s[k] for *_, s in prs) / len(prs), 4) for k in (*READ, "none")},
                          **{f"{k}-none": e2.rnd(cr.dpp(prs, k, "none", bs)) for k in READ}},
            "e_equal_to_R_base": {"equal": sum(nb[c]["e"] == base[c]["e"] for c in both), "total": len(both)},
            "by_source_descriptive": {s: {"cases": len(p), **{k: round(sum(x[k] for *_, x in p) / len(p), 4)
                                                             for k in ("G", "none")}} for s, p in src.items() if p},
            "G_check": [round(sum(float(q["G"]["success"]) for q in pr.values()) / len(pr), 4), len(pr)]}
    s = json.loads((R / "summary_R.json").read_text())["report"]["cells"]
    for L in CELLS:  # the pairs here are summary_R's: G's success on them matches its reading
        assert res[L].pop("G_check") == s[L]["success"]["G"], (L, s[L]["success"]["G"])
    print(e2.dump(res))
    if out:
        Path(out).write_text(e2.dump(res) + "\n")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out")
    main(ap.parse_args().out)
