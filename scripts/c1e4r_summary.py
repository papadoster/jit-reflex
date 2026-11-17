"""C1-E4 cell R: the delay trend on the same cases (docs/superpowers/specs/2026-11-17-c1-e4-delay-trend-design.md,
"spec"), on pod R (scripts/gpu_c1e4m.sh POD=R rsmoke, rgrid0, rgrid20): the runner plan per line, the smoke's code
check, R0's completeness before R20 and the reading. Reuses scripts/c1e4c_summary.py (mlines, lk, lkeys, KBAR, tsv,
mp), scripts/c1e4v_summary.py (ids, unique, controls), scripts/c1e4t_summary.py (the VT lines), scripts/c1e4b_summary.py
(floor, BAND) and scripts/c1e4k_summary.py (REPS: e2.diff_pp's bootstrap at 8000); pod C's records are read only; run
from the repository root. R = results/c1-e4/delay/pod_R:
    python scripts/c1e4r_summary.py --plan --line L
        L: Rsmoke R0 R20, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4r_summary.py --smoke-check R/smoke/Rsmoke_*.jsonl   the code only (§8), exit 0, or 3
    python scripts/c1e4r_summary.py --complete --line R0 R/R0_*.jsonl   every planned record there (skipped ones
        too), exit 0, or 3: rgrid20's gate
    python scripts/c1e4r_summary.py [read] R/grid_all_R.jsonl [--c results/c1-e4/controls/pod_C] [--out ...]
        the checks before reading (§6), then the rules Q8, Q9 (§4) and the report (§5); a failed check prints only the
        checks, exit 3. --c: pod C's grid_all_C.jsonl, grid_all_T.jsonl and vst_k.json (Q8 on the old 300, report
        only; --c '' leaves it out). Needs hf-libero (the task clusters)
    python scripts/c1e4r_summary.py --selftest
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4b_summary as cb  # noqa: E402
import c1e4c_summary as cc  # noqa: E402
import c1e4k_summary as ck  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4t_summary as ct  # noqa: E402
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import P, TOL, e2, maxwrap, run  # noqa: E402

ARMS = ("Gauto", "VST0.3", "VTm", "PBVS0.3")  # spec §2: one process, one full Base (s* of VS-T and VT-m)
SIDE = dict(zip(ARMS, ("G", "T", "M", "V5")))  # spec's letters
CELLS = {"R0": 0, "R20": 20}  # spec §8: R0 first
LINES = ("Rsmoke", *CELLS)
N_SRC = {"plus": 471, "pro": 169}  # spec §2: the 640 evalrest cases
KBAR = cc.KBAR["m", P]
BANDF = cc.REPO / cc.OUT / "band_eps.json"  # cb.BAND's file: one 5 mm band on V5 and M
DELTA = 0.005
REPS = ck.REPS  # spec §4: 8000 cluster resamples, default_rng(0)
CPOD = "results/c1-e4/controls/pod_C"
# spec §4, word for word
RULES = {"Q8": ("T", "G", "преимущество привилегированной ветки убывает с задержкой"),
         "Q9": ("M", "V5", "преимущество цели из показа при одном законе управления убывает с задержкой")}
NOT_SHOWN = "убывание не показано"
DIFFS = (("T", "G"), ("M", "V5"), ("M", "G"), ("T", "V5"), ("V5", "G"))  # spec §5, per cell
LAM = {"R0": ((0, 4), (5, 9)), "R20": ((20, 24), (25, 29))}  # spec §5: the fresh-plan lag's bands, both ends in
NO_Q9_OLD = ("Q9 на старых 300 нет: M там прогнан только при d = 0 PRO и d = 10 Plus, при d = 20 — нет")


def lines(ph, tag, d, split, cases=None):
    """The four arms with part 2's kbar and band_eps.json on evalrest, every Base full (no --base-prefix)."""
    return [ln | {"args": ln["args"] + cb.BAND}
            for ln in cc.mlines(ph, tag, P, "evalrest", ARMS, split, d, cases, KBAR, prefix=False)]


def plan(line, split=None):
    """spec §2, §9: a cell's runner lines (phase line.lower()), one per source, the set's order (the runner's groups of
    6); Rsmoke: R0's first 6 Plus cases, the smoke's own line."""
    split = split or maxwrap.load_split(cs.SPLIT)
    if line == "Rsmoke":
        six = cs.sources(run.set_ids(split, "evalrest"))["plus"][:6]
        (ln,) = lines("rsmoke", "Rsmoke", 0, split, set(six))
        return [ln | {"id": "Rsmoke_pi05"}]
    out = lines(line.lower(), line, CELLS[line], split)
    assert {ln["source"]: len(cv.ids(ln, split)) for ln in out} == N_SRC, f"{line}: not the 640 evalrest cases"
    return out


def band():
    b = json.loads(BANDF.read_text())
    assert b["delta"] == DELTA, b["delta"]
    return b


def present(recs, line, split=None):
    """{planned key: record} of a line's records (skipped ones too) and the planned keys."""
    want = cc.lkeys(plan(line, split))
    return {x: r for x, r in cv.unique([r for r in recs if "kind" in r], cc.lk).items() if x in want}, want


def complete(recs, line, split=None):
    """rgrid20's gate (spec §8: R0 whole first): every planned record of the line there, a skipped one too."""
    have, want = present(recs, line, split)
    return len(have) == len(want), {"line": line, "expected": len(want), "missing": len(want - have.keys())}


def arm_checks(rs, bj):
    """spec §6: kbar 0.306 on G's records, band_eps.json (delta 5 mm) in conf["pbvs"] of V5's and M's."""
    k = sum(r["conf"].get("kbar") != KBAR for r in rs if r["arm"] == "Gauto")
    b = sum(r["conf"].get("pbvs") != bj for r in rs if r["arm"] in ("VTm", "PBVS0.3"))
    return {"kbar": {"want": KBAR, "off": k, "pass": not k}, "band": {"delta": bj["delta"], "off": b, "pass": not b}}


def smoke_check(recs, split=None, bj=None):
    """spec §8: rsmoke's records checked for the code only, never an outcome: its 6 full Bases and 24 Dynamic episodes,
    none skipped; every Dynamic one valid (prefix_ok, no replay miss) with its plans_in; conf as planned (one conf:
    pi0.5, d 0, the four arms, kbar, band_eps.json); VS-T's record with s_star, VT-m's with s_star, a_floor and a_first,
    PBVS's with its counters; no Base cut. (pass, details)."""
    split, bj = split or maxwrap.load_split(cs.SPLIT), bj or band()
    have, want = present(recs, "Rsmoke", split)
    have = {x: r for x, r in have.items() if not r.get("skipped_reason")}
    dyn = [r for r in have.values() if r["kind"] == "dynamic"]
    conf = {json.dumps(r["conf"], sort_keys=True) for r in have.values()}
    c = json.loads(next(iter(conf))) if len(conf) == 1 else {}
    need = {"VST0.3": ("vst", ("s_star",)), "VTm": ("vtm", ("s_star", "a_floor", "a_first")),
            "PBVS0.3": ("pbvs", ("steps_window",))}
    fld = sum(a in need and not set(need[a][1]) <= set(r.get(need[a][0]) or {}) for r in dyn for a in [r["arm"]])
    out = {"complete": {"expected": len(want), "missing": len(want - have.keys())},
           "prefix": {"dynamic": len(dyn), "invalid": sum(r.get("prefix_ok") is not True or cs.invalid(r)
                                                          or not r.get("plans_in") for r in dyn)},
           "conf": {"distinct": len(conf), "pass": len(conf) == 1 and c.get("brain") == P and c.get("d") == 0
                    and c.get("arms") == list(ARMS) and c.get("kbar") == KBAR},
           "fields": {"off": fld, "pass": not fld},
           "base_full": {"cut": sum(r.get("base_cut") is not False for r in have.values() if r["kind"] == "base")},
           **arm_checks(dyn, bj)}
    out["complete"]["pass"] = out["complete"]["missing"] == 0
    out["prefix"]["pass"] = out["prefix"]["invalid"] == 0
    out["base_full"]["pass"] = out["base_full"]["cut"] == 0
    return all(v["pass"] for v in out.values()), out


# ------------------------------------------------------------------ the bootstrap (spec §4)

def resamples(clusters):
    """spec §4: one set of resamples, REPS draws of whole clusters (sorted) from default_rng(0)."""
    keys = sorted(clusters)
    return keys, np.random.default_rng(0).integers(len(keys), size=(REPS, len(keys)))


def ci(rows, f, bs, name="diff_pp"):
    """rows (cluster, values): f of the values summed over the pairs (the estimate) and over each resample's drawn
    clusters (bs, resamples': a cluster without rows adds nothing); percentiles 2.5 and 97.5 (95 %), 5 and 95
    (one-sided) over the resamples where f is defined (finite), their number undefined. Raw floats; None: no rows."""
    if not rows:
        return None
    keys, pick = bs
    ix = {k: j for j, k in enumerate(keys)}
    S = np.zeros((len(keys), len(rows[0][1])))
    np.add.at(S, [ix[c] for c, _ in rows], np.array([v for _, v in rows], float))
    with np.errstate(divide="ignore", invalid="ignore"):
        est, b = f(S.sum(0)), f(S[pick].sum(1))
    ok = np.isfinite(b)
    q = [float(v) for v in np.percentile(b[ok], [2.5, 5, 95, 97.5])] if ok.any() else [None] * 4
    return {name: float(est) if np.isfinite(est) else None, "lo2.5": q[0], "lo5_one_sided": q[1],
            "hi95_one_sided": q[2], "hi97.5": q[3], "pairs": len(rows), "clusters": len({c for c, _ in rows}),
            "undefined_resamples": int((~ok).sum())}


MEAN = lambda T: 100 * T[..., 1] / T[..., 0]  # noqa: E731  rows (cluster, [1, x - y]): pp
RATIO = lambda T: np.log(T[..., 1] / T[..., 2])  # noqa: E731  rows (cluster, [1, s_X, s_Y])
RTREND = lambda T: np.log(T[..., 1] / T[..., 2]) - np.log(T[..., 3] / T[..., 4])  # noqa: E731


def dpp(prs, x, y, bs):
    """X - Y in pp over the pairs (case, cluster, successes)."""
    return ci([(cl, [1, s[x] - s[y]]) for _, cl, s in prs], MEAN, bs)


def trend(both, x, y, bs):
    """spec §4: per case q = (X - Y)_R0 - (X - Y)_R20 in pp, the mean over cases. both: (case, cluster, s0, s20)."""
    return ci([(cl, [1, (a[x] - a[y]) - (b[x] - b[y])]) for _, cl, a, b in both], MEAN, bs)


def sentence(est, text):
    """spec §4: the rule's sentence by the raw one-sided lower bound lo5 > TOL."""
    return text if est and est["lo5_one_sided"] is not None and est["lo5_one_sided"] > TOL else NOT_SHOWN


def lam(r):
    """spec §5: the arrival step of the first plan observed at or after e (plans_in: [t_obs, arrival]), minus e."""
    return next((t - r["e"] for o, t in r["plans_in"] if o >= r["e"]), None)


# ------------------------------------------------------------------ the reading

def read(new, old, names, split, bj=None):
    """spec §3-§6. The checks before reading (§6: each runner line's Dynamic episodes present on >= 99 %; invalid pairs
    <= 2 % per cell, a missing record, a skipped case or one without an event not one; conf equal per cell but arms;
    arm_checks per cell; both plans the same 640 cases, no record of either cell on another case), then Q8, Q9 and the
    report. new: grid_all_R.jsonl; old: old300()'s rows or None. A pair: one case in one cell, its full Base and four
    Dynamic episodes there, the Base with an event, every Dynamic one valid (cs.invalid: prefix_ok false or a replay
    miss); a trend case: a pair in both cells (the others counted in trend_cases by the cell that dropped them).
    Clusters: maxwrap.base_task. Records off the plan are counted, never read."""
    bj = bj or band()
    lns = {L: plan(L, split) for L in CELLS}
    want = {x: L for L, ln in lns.items() for x in cc.lkeys(ln)}
    by = cv.unique(new, cc.lk)
    used = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    skip = {(want[x], r["case_id"]) for x, r in by.items() if x in want and r.get("skipped_reason")}
    of = lambda L, kind, a: {r["case_id"]: r for x, r in used.items()  # noqa: E731
                             if want[x] == L and r["kind"] == kind and r["arm"] == a}
    comp = {}
    for ln in [ln for v in lns.values() for ln in v]:  # spec §6: per runner line
        exp = {x for x in cc.lkeys([ln]) if x[3] == "dynamic"}
        miss = len(exp - used.keys())
        comp[ln["id"]] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    cases = run.set_ids(split, "evalrest")
    # spec §6: both plans the same 640 ids, no record of a cell's d and arms on another case (a missing case is the
    # completeness's, and the trend's drop)
    foreign = sorted({r["case_id"] for r in by.values() if "case_id" in r and r["d"] in CELLS.values()
                      and tuple(r["conf"]["arms"]) == ARMS and r["case_id"] not in set(cases)})
    same = {"planned": len(cases), "foreign_cases": foreign[:10], "foreign": len(foreign),
            "pass": all(set(cv.ids(a, split)) == set(cv.ids(b, split)) for a, b in zip(*lns.values())) and not foreign}
    one = {r["case_id"]: r for r in used.values()}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    pairs, drops = {}, {}
    for L in CELLS:  # spec §3
        out, drop, base, arm = {}, Counter(), of(L, "base", "none"), {a: of(L, "dynamic", a) for a in ARMS}
        for c in cases:
            q = {SIDE[a]: arm[a].get(c) for a in ARMS}
            why = ("skipped" if (L, c) in skip else "missing" if c not in base or None in q.values() else
                   "no_event" if base[c]["e"] is None else "invalid" if any(map(cs.invalid, q.values())) else None)
            if why:
                drop[why] += 1
            else:
                out[c] = q
        pairs[L], drops[L] = out, dict(drop)
    conf = defaultdict(set)
    for x, r in used.items():  # spec §6: conf equal per cell but arms
        conf[want[x]].add(json.dumps({f: v for f, v in r["conf"].items() if f != "arms"}, sort_keys=True))
    mixed = {L: sorted(v) for L, v in conf.items() if len(v) > 1}
    inv = {L: drops[L].get("invalid", 0) for L in CELLS}
    checks = {"records": len(new), "read": len(used), "skipped": len(skip),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for x, r in by.items() if x not in want)),
              "complete": comp,
              "invalid_pairs": {L: {"pairs": len(pairs[L]) + inv[L], "invalid": inv[L],
                                    "pass": inv[L] <= 0.02 * (len(pairs[L]) + inv[L]) + TOL} for L in CELLS},
              "conf": {"except": ["arms"], "mixed": mixed, "pass": not mixed},
              "arms": {L: arm_checks([r for x, r in used.items() if want[x] == L and r["kind"] == "dynamic"], bj)
                       for L in CELLS},
              "same_cases": same}
    checks["pass"] = bool(all(v["pass"] for f in ("complete", "invalid_pairs") for v in checks[f].values())
                          and checks["conf"]["pass"] and same["pass"]
                          and all(v["pass"] for a in checks["arms"].values() for v in a.values()))
    if not checks["pass"]:
        return {"checks": checks}

    bs = resamples(set(clu.values()))  # spec §4: one set of resamples for everything on the 640
    succ = lambda q: {k: float(r["success"]) for k, r in q.items()}  # noqa: E731
    cell = {L: [(c, clu[c], succ(q)) for c, q in p.items()] for L, p in pairs.items()}
    tc = [c for c in cases if c in pairs["R0"] and c in pairs["R20"]]
    both = [(c, clu[c], succ(pairs["R0"][c]), succ(pairs["R20"][c])) for c in tc]
    nos = {c for c in tc if any(pairs[L][c]["T"]["vst"]["s_star"] is None for L in CELLS)}
    vtm = [r["vtm"] for L in CELLS for r in of(L, "dynamic", "VTm").values()  # spec §4: both cells' valid M episodes
           if r.get("prefix_ok") is True and r.get("replay_miss") is None]  # (cb.read's)
    fl = cb.floor(vtm)
    gate = None if fl["informative"] else f"не читается: {fl['why']}"
    m = lambda *s: gate if gate and "M" in s else None  # noqa: E731  spec §4: M's outcomes behind the gate

    rules = {}
    for n, (x, y, text) in RULES.items():
        if m(x, y):
            rules[n] = m(x, y)
            continue
        est = trend(both, x, y, bs)
        rules[n] = {"test": f"pi05, evalrest 640, per case ({x} - {y})_R0 - ({x} - {y})_R20, pp", **e2.rnd(est),
                    "sentence": sentence(est, text)}
    rules["Q8"]["without_s_star"] = e2.rnd(trend([b for b in both if b[0] not in nos], "T", "G", bs)) | {
        "cases_out": len(nos), "note": "вторичная строка: без случаев, где у T нет s* хотя бы в одной клетке"}
    rep = {"trend_cases": {"both_valid": len(tc), **{f"out: valid in {a} only, dropped in {b}": len(set(pairs[a]) - set(
        pairs[b])) for a, b in (("R0", "R20"), ("R20", "R0"))}},
           "drops": drops, "cells": {}, "scale": {}, "lambda": {}}
    for L, prs in cell.items():
        rep["cells"][L] = {f"{x}-{y}": m(x, y) or e2.rnd(dpp(prs, x, y, bs)) for x, y in DIFFS} | {"success": {
            k: m(k) or [round(float(np.mean([s[k] for _, _, s in prs])), 4), len(prs)] for k in SIDE.values()}}
    for n, (x, y, _) in RULES.items():  # spec §5: L_d = ln(s_X / s_Y) on the trend cases, its trend L_0 - L_20
        rep["scale"][n] = m(x, y) or {
            **{L: e2.rnd(ci([(cl, [1, ss[j][x], ss[j][y]]) for _, cl, *ss in both], RATIO, bs, "L"))
               for j, L in enumerate(CELLS)},
            "L0-L20": e2.rnd(ci([(cl, [1, a[x], a[y], b[x], b[y]]) for _, cl, a, b in both], RTREND, bs, "L"))}
    rep["lambda"] = lam_bands(pairs, cell, bs, m)
    rep["supplementary_Q8"] = supplementary(old, both, clu)
    rep["Q9_old300"] = NO_Q9_OLD
    rep["counters"] = counters(pairs, fl)
    for L in CELLS:  # spec §5: every read Dynamic record's e against its Base's
        b = of(L, "base", "none")
        e = [r["e"] == b[c]["e"] for a in ARMS for c, r in of(L, "dynamic", a).items() if c in b]
        rep["counters"][L]["base_e"] = {"equal": sum(e), "total": len(e)}
    rep["gpu"] = sorted({str(r["conf"].get("gpu")) for r in used.values()})
    return {"checks": checks, "rules": rules, "report": rep}


def lam_bands(pairs, cell, bs, m):
    """spec §5: per valid pair the fresh-plan lag of each arm's record; checked against (-e mod s) + d and across the
    case's arms (mismatches listed); then T - G and M - V5 per band of the cell, on the pairs whose arms agree."""
    out = {}
    for L, p in pairs.items():
        lams, mis, none = {}, [], 0
        for c, q in p.items():
            v = {k: lam(r) for k, r in q.items()}
            if set(v.values()) == {None}:  # every arm's episode ended before a fresh plan arrived: out of the bands
                none += 1
                continue
            r = q["G"]
            f = -r["e"] % r["conf"]["s"] + r["d"]
            if set(v.values()) != {f}:
                mis.append({"case": c, "e": r["e"], "formula": f, "lambda": v})
            if len(set(v.values())) == 1:
                lams[c] = v["G"]
        bands = {}
        for lo, hi in LAM[L]:
            prs = [x for x in cell[L] if x[0] in lams and lo <= lams[x[0]] <= hi]
            bands[f"[{lo}, {hi}]"] = {f"{x}-{y}": m(x, y) or e2.rnd(dpp(prs, x, y, bs)) for x, y in (("T", "G"),
                                                                                                     ("M", "V5"))}
        out[L] = {"bands": bands, "mismatches": len(mis), "mismatch_list": mis[:20],
                  "no_fresh_plan_before_the_end": none,
                  "outside_bands": sum(not any(lo <= v <= hi for lo, hi in LAM[L]) for v in lams.values())}
    return out


def counters(pairs, fl):
    """spec §5, over each cell's valid pairs: the share of T without s*; the noise floor; M's aim-step shares (per
    episode with a window: steps_aim, no_aim over steps_window); brain calls per episode per arm; the Dynamic records'
    e against their Base's (prefix_ok holds it: a check)."""
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    out = {"noise_floor": fl}
    for L, p in pairs.items():
        q = list(p.values())
        w = [x["M"]["vtm"] for x in q if x["M"]["vtm"].get("steps_window")]
        out[L] = {"T_without_s_star": mean([x["T"]["vst"]["s_star"] is None for x in q]),
                  "M_steps_aim_share": mean([v["steps_aim"] / v["steps_window"] for v in w if "steps_aim" in v]),
                  "M_no_aim_share": mean([v["no_aim"] / v["steps_window"] for v in w if "no_aim" in v]),
                  "calls_per_episode": {k: mean([x[k]["calls_sched"] + x[k]["calls_trig"] for x in q])
                                        for k in SIDE.values()}}
    return out


def old300(cdir, names, split):
    """spec §5: the old 300 on pod C: T from VT2 (d 0) / VT3d20, G from controls lines 2 (C2, C2_nv) and 3 (d 20), vst_k
    .json's arm; paired by case as Т1 (cc.mp: both valid, the two e equal; not bitwise). {d: ({case: (cluster, T, G)},
    invalid pairs)}."""
    cdir = Path(cdir)
    vk = json.loads((cdir / "vst_k.json").read_text())
    cused, _, _ = cv.controls(e2.load(cdir / "grid_all_C.jsonl"), cv.CMODE, cv.CK, split)
    tw = cc.lkeys(ct.plan("VT2", vk, split) + ct.plan("VT3d20", vk, split))
    tused = [r for x, r in cv.unique([r for r in e2.load(cdir / "grid_all_T.jsonl") if "kind" in r], cc.lk).items()
             if x in tw and not r.get("skipped_reason")]
    rs = list(cused.values()) + tused
    clu = {r["case_id"]: maxwrap.base_task({"case_id": r["case_id"], "task_suite_name": r["suite"],
                                            "task_name": r["task_name"], "substrate_variant": r["substrate_variant"]},
                                           names) for r in rs}
    mi = cs.mindex([r for r in rs if r["kind"] == "dynamic"])
    e300 = set(run.set_ids(split, "eval300"))
    return {d: cc.mp(mi, clu, P, vk["arm"], "Gauto", d, e300) for d in (0, 20)}


def supplementary(old, both, clu):
    """spec §5, not a rule: Q8 on the old 300 (pod C; its per-d T - G on all its pairs beside) and on the 940 (the old
    300's trend cases and the new 640's, one bootstrap over their clusters)."""
    if old is None:
        return "нет: записи пода C не переданы (--c)"
    (p0, b0), (p20, b20) = old[0], old[20]
    rows = [(p0[c][0], [1, (p0[c][1] - p0[c][2]) - (p20[c][1] - p20[c][2])]) for c in sorted(p0.keys() & p20.keys())]
    new = [(cl, [1, (a["T"] - a["G"]) - (b["T"] - b["G"])]) for _, cl, a, b in both]
    note = "не правило: на старых 300 вопрос и возник"
    one = lambda p: [(v[0], [1, float(v[1]) - float(v[2])]) for v in p.values()]  # noqa: E731
    return {"old300": {**e2.rnd(ci(rows, MEAN, resamples({c for c, _ in rows}))), "note": note,
                       "per_d": {f"d{d}": e2.rnd(ci(one(p), MEAN, resamples({v[0] for v in p.values()})))
                                 | {"invalid_pairs": b} for d, p, b in ((0, p0, b0), (20, p20, b20))}},
            "940": {**e2.rnd(ci(rows + new, MEAN, resamples({c for c, _ in rows + new}))), "note": note}}


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 delay checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


# ------------------------------------------------------------------ selftest

def selftest():
    """The plan on the real split: per cell two runner lines (471 Plus, 169 PRO evalrest cases in the set's order,
    groups of 6), the four arms, part 2's kbar, band_eps.json, no --base-prefix; Rsmoke on R0's first 6 Plus cases;
    the reading on a synthetic pod."""
    split = maxwrap.load_split(cs.SPLIT)
    rest = run.set_ids(split, "evalrest")
    src = cs.sources(rest)
    tail = " --kbar 0.306 --band results/c1-e4/controls/pod_C/band_eps.json"
    for L, d in CELLS.items():
        lns = plan(L, split)
        assert [ln["id"] for ln in lns] == [f"{L}_pi05_evalrest_{s}_d{d}" for s in N_SRC], lns
        for ln in lns:
            assert ln["args"] == (f"--brain pi05 --source {ln['source']} --set evalrest "
                                  f"--arms=Gauto,VST0.3,VTm,PBVS0.3 --d {d}{tail}"), ln
            assert ln["phase"] == L.lower() and not ln["prefix"] and "--base-prefix" not in ln["args"], ln
            assert cv.ids(ln, split) == src[ln["source"]] and cc.n_of(ln) == 5 * N_SRC[ln["source"]], ln
        assert cv.groups(lns, split) == {(d, tuple(c[j:j + 6])) for c in src.values() for j in range(0, len(c), 6)}
    (sm,) = plan("Rsmoke", split)
    assert (sm["id"], sm["phase"], cc.n_of(sm), cv.ids(sm, split)) == ("Rsmoke_pi05", "rsmoke", 30, src["plus"][:6])
    assert sm["args"] == (f"--brain pi05 --source plus --set evalrest --arms=Gauto,VST0.3,VTm,PBVS0.3 --d 0 --cases "
                          f"{','.join(src['plus'][:6])}{tail}"), sm
    assert cc.tsv([sm]).split("\t")[:6] == ["rsmoke", "c1e4m", P, "plus", "Rsmoke_pi05", "30"]
    assert band()["delta"] == DELTA and cb.BAND == f" --band {cc.OUT}/band_eps.json"
    selftest_read(split, rest)
    print("selftest ok")


def selftest_read(split, rest):
    """The smoke check, R0's gate and the reading on a synthetic pod: success by the case's place j among the 640 in
    cycles of 8 (W: the residues that succeed per cell and arm), so every difference is exact; e = 15 + j mod 37 and
    plans_in on the schedule (s = 10), so lambda = (-e mod 10) + d; no s* on j = 0 mod 50; both aims on j = 0 mod 3;
    task clusters from fake LIBERO names as c1e4p_summary's selftest; pod C's old 300 written to a temporary dir."""
    import tempfile
    from unittest import mock
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    bj, i = band(), {c: j for j, c in enumerate(rest)}
    S8 = lambda *k: frozenset(k)  # noqa: E731
    # sizes R0: G 1, T 4, M 7, V5 2; R20: G 1, T 3, M 5, V5 4 (every difference of DIFFS apart within a cell)
    W = {("R0", "Gauto"): S8(0), ("R0", "VST0.3"): S8(0, 1, 2, 3), ("R0", "VTm"): S8(0, 1, 2, 3, 4, 5, 6),
         ("R0", "PBVS0.3"): S8(0, 7), ("R20", "Gauto"): S8(5), ("R20", "VST0.3"): S8(0, 5, 6),
         ("R20", "VTm"): S8(1, 2, 3, 4, 5), ("R20", "PBVS0.3"): S8(0, 1, 2, 7)}
    win0 = lambda L, a, c, W=W: i[c] % 8 in W[L, a]  # noqa: E731
    A = {v: k for k, v in SIDE.items()}

    def rec(L, c, kind, a, win=win0, aim=cb.S):
        h, j, d = sum(map(ord, c)), i[c], CELLS[L]
        su = list(run.MAX_STEPS)[h % 4]
        cf = {"brain": P, "revision": "r", "precision": "fp32", "s": 10, "d": d, "kbar": KBAR, "t_ramp": 5, "k_p": 1.0,
              "tau_k": 0.01, "R": 360, "arms": list(ARMS), "split": split["sha256"], "gpu": "RTX 4090", "pbvs": bj}
        r = {"brain": P, "case_id": c, "kind": kind, "arm": a, "method": a, "d": d, "policy_seed": 195, "conf": cf,
             "suite": su, "task_name": f"{su}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None, "e": 15 + j % 37,
             "prefix_ok": None if kind == "base" else True, "replay_miss": None, "skipped_reason": None,
             "success": bool(win(L, a, c)) if kind == "dynamic" else j % 2 == 0, "calls_sched": 30,
             "calls_trig": 0, "plans_in": [[t, t + d * (t > 0)] for t in range(0, 300, 10)]}
        s_star = None if j % 50 == (0 if L == "R0" else 25) else [0.001, 0.0]  # T without s*: 26 cases
        r |= ({"base_cut": False} if kind == "base" else {"vst": {"s_star": s_star, "steps_window": 10}}
              if a == "VST0.3" else {"pbvs": {"steps_window": 10}} if a == "PBVS0.3" else {
            "vtm": {"s_star": s_star, "steps_window": 10, "steps_aim": 4, "no_aim": 6, "a_first": aim,
                    "a_floor": cb.F if j % 3 == 0 else None}} if a == "VTm" else {})
        return r

    world = lambda win=win0, aim=cb.S: [rec(L, x[2], x[3], x[4], win, aim) for L in CELLS  # noqa: E731
                                        for x in sorted(cc.lkeys(plan(L, split)))]
    new = world()
    hand = lambda L, x, y, out=(): round(100 * sum(win0(L, A[x], c) - win0(L, A[y], c) for c in rest  # noqa: E731
                                                   if c not in out) / (640 - len(out)), 3)

    # pod C's old 300: G from lines 2 and 3 (d 20), T from VT2 and VT3d20; success by the place k among eval300 in
    # cycles of 4; one d-20 pair invalid (prefix_ok false), one with the two e apart
    e300, vk = run.set_ids(split, "eval300"), {"arm": "VST0.3", "lambda": 0.3}
    k = {c: j for j, c in enumerate(e300)}
    WO = {(0, "Gauto"): {0}, (0, "VST0.3"): {0, 1, 2}, (20, "Gauto"): {0, 1}, (20, "VST0.3"): {1, 2}}
    bad_p, bad_e = e300[7], e300[11]

    def orec(x):
        c, d = x[2], x[5]
        h = sum(map(ord, c))
        su = list(run.MAX_STEPS)[h % 4]
        return {"brain": P, "case_id": c, "kind": x[3], "arm": x[4], "d": d, "policy_seed": 195,
                "conf": {"arms": list(x[7])}, "suite": su, "task_name": f"{su}_t{h // 4 % 10}_v{h % 7}",
                "substrate_variant": None, "e": 30 + (x[4] == "VST0.3" and c == bad_e and d == 20),
                "prefix_ok": not (c == bad_p and d == 20 and x[4] == "Gauto"), "replay_miss": None,
                "skipped_reason": None, "success": k[c] % 4 in WO.get((d, x[4]), ())}
    ckeys = [x for x in cc.lkeys([ln for L in "23" for ln in cc.plan(L, cv.CMODE, cv.CK, split)]) if x[5] in (0, 20)]
    tkeys = cc.lkeys(ct.plan("VT2", vk, split) + ct.plan("VT3d20", vk, split))
    with tempfile.TemporaryDirectory() as tmp:
        for f, ks in (("grid_all_C.jsonl", ckeys), ("grid_all_T.jsonl", tkeys)):
            Path(tmp, f).write_text("".join(json.dumps(orec(x)) + "\n" for x in sorted(ks)))
        Path(tmp, "vst_k.json").write_text(json.dumps(vk))
        old = old300(tmp, names, split)
    (p0, b0), (p20, b20) = old[0], old[20]
    assert (len(p0), b0, len(p20), b20) == (300, 0, 298, 2), (len(p0), b0, len(p20), b20)
    tq = lambda c, d: (k[c] % 4 in WO[d, "VST0.3"]) - (k[c] % 4 in WO[d, "Gauto"])  # noqa: E731
    qold = [tq(c, 0) - tq(c, 20) for c in e300 if c not in (bad_p, bad_e)]
    res = read(new, old, names, split)
    ch, ru, rp = res["checks"], res["rules"], res["report"]
    assert ch["pass"] and ch["read"] == len(new) == 2 * 5 * 640 and ch["same_cases"]["pass"], ch
    assert ch["invalid_pairs"] == {L: {"pairs": 640, "invalid": 0, "pass": True} for L in CELLS}, ch
    # the rules and the cells, exact
    assert ru["Q8"]["diff_pp"] == hand("R0", "T", "G") - hand("R20", "T", "G") == 12.5, ru["Q8"]
    assert ru["Q9"]["diff_pp"] == hand("R0", "M", "V5") - hand("R20", "M", "V5") == 50.0, ru["Q9"]
    assert ru["Q8"]["sentence"] == RULES["Q8"][2] and ru["Q9"]["sentence"] == RULES["Q9"][2], ru
    assert ru["Q8"]["pairs"] == 640 and ru["Q8"]["without_s_star"]["pairs"] == 640 - 26 and ru["Q8"][
        "without_s_star"]["cases_out"] == 26 and ru["Q8"]["without_s_star"]["diff_pp"] == round(100 * sum(
            tq_ for c in rest if i[c] % 50 not in (0, 25) for tq_ in [(win0("R0", "VST0.3", c) - win0("R0", "Gauto", c))
                                                      - (win0("R20", "VST0.3", c) - win0("R20", "Gauto", c))])
        / 614, 3), ru["Q8"]
    for L in CELLS:
        got = {n: v["diff_pp"] for n, v in rp["cells"][L].items() if n != "success"}
        assert got == {f"{x}-{y}": hand(L, x, y) for x, y in DIFFS} and len(set(got.values())) == 5, (L, got)
        assert rp["cells"][L]["success"] == {s: [round(len(W[L, A[s]]) / 8, 4), 640] for s in SIDE.values()}
    # the bootstrap: Q8 is the K reading's T = D(R0) - D(R20) (ck.boot, the clusters drawn jointly); a cell's
    # difference e2.diff_pp's at 8000 resamples; hi95 by hand; the printed numbers are these rounded
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": None}, names) for r in new for c in [r["case_id"]]}
    bs = resamples(set(clu.values()))
    sc = lambda L, c: {s: float(win0(L, A[s], c)) for s in SIDE.values()}  # noqa: E731
    both = [(c, clu[c], sc("R0", c), sc("R20", c)) for c in rest]
    q8 = trend(both, "T", "G", bs)
    kb = ck.boot(*((sg, [(c, clu[c], win0(L, "VST0.3", c), win0(L, "Gauto", c)) for c in rest])
                   for sg, L in ((1, "R0"), (-1, "R20"))))
    assert all(abs(q8[f] - kb[f]) < 1e-9 for f in ("diff_pp", "lo2.5", "lo5_one_sided", "hi97.5")), (q8, kb)
    assert e2.rnd(q8) | {"test": ru["Q8"]["test"], "sentence": RULES["Q8"][2], "without_s_star": ru["Q8"][
        "without_s_star"]} == ru["Q8"] and q8["lo5_one_sided"] > 0 and q8["undefined_resamples"] == 0, (q8, ru)
    keys, pick = bs
    per = defaultdict(list)
    for c, cl, a, b in both:
        per[cl].append((a["T"] - a["G"]) - (b["T"] - b["G"]))
    st = [100 * sum(sum(per[keys[j]]) for j in row) / sum(len(per[keys[j]]) for j in row) for row in pick]
    assert abs(q8["hi95_one_sided"] - float(np.percentile(st, 95))) < 1e-9 and len(pick) == REPS == 8000
    with mock.patch.object(e2, "N_BOOT", REPS):
        ed = e2.diff_pp([(clu[c], win0("R20", "VTm", c), win0("R20", "Gauto", c)) for c in rest])
    md = dpp([(c, clu[c], sc("R20", c)) for c in rest], "M", "G", bs)
    assert all(abs(md[f] - ed[f]) < 1e-9 for f in ed if f != "pairs") and e2.rnd(md) == rp["cells"]["R20"]["M-G"]
    # the relative scale, exact: ln(s_X / s_Y) per cell on the 640 and its trend
    sq = rp["scale"]
    for n, (x, y, _) in RULES.items():
        lx = {L: math.log(len(W[L, A[x]]) / len(W[L, A[y]])) for L in CELLS}
        assert {L: sq[n][L]["L"] for L in CELLS} == {L: round(v, 3) for L, v in lx.items()}, sq[n]
        assert sq[n]["L0-L20"]["L"] == round(lx["R0"] - lx["R20"], 3) and sq[n]["L0-L20"]["pairs"] == 640, sq[n]
    # the fresh-plan lag: (-e mod 10) + d on every record, bands by hand
    lm = rp["lambda"]
    for L, d in CELLS.items():
        assert lm[L]["mismatches"] == 0 and lm[L]["outside_bands"] == 0, lm[L]
        for lo, hi in LAM[L]:
            cs_ = [c for c in rest if lo <= -(15 + i[c] % 37) % 10 + d <= hi]
            b = lm[L]["bands"][f"[{lo}, {hi}]"]
            assert b["T-G"]["pairs"] == len(cs_) > 100 and b["M-V5"]["diff_pp"] == round(100 * sum(
                win0(L, "VTm", c) - win0(L, "PBVS0.3", c) for c in cs_) / len(cs_), 3), (L, lo, b)
    # the old 300 and the 940, report only
    sp = rp["supplementary_Q8"]
    assert sp["old300"]["diff_pp"] == round(100 * sum(qold) / 298, 3) and sp["old300"]["pairs"] == 298, sp
    assert sp["old300"]["per_d"]["d0"]["diff_pp"] == 50.0 and sp["old300"]["per_d"]["d20"]["invalid_pairs"] == 2
    q640 = [(win0("R0", "VST0.3", c) - win0("R0", "Gauto", c)) - (win0("R20", "VST0.3", c) - win0("R20", "Gauto", c))
            for c in rest]
    assert sp["940"]["diff_pp"] == round(100 * (sum(qold) + sum(q640)) / 938, 3) and sp["940"]["pairs"] == 938, sp
    assert rp["Q9_old300"] == NO_Q9_OLD and read(new, None, names, split)["report"]["supplementary_Q8"].startswith(
        "нет")
    # counters
    co = rp["counters"]
    assert co["noise_floor"]["informative"] and co["noise_floor"]["n"] == 2 * len([c for c in rest if i[c] % 3 == 0])
    assert co["R0"]["T_without_s_star"] == round(13 / 640, 4) and co["R0"]["M_steps_aim_share"] == 0.4 and co["R20"][
        "calls_per_episode"] == {"G": 30.0, "T": 30.0, "M": 30.0, "V5": 30.0} and co["R20"]["base_e"] == {
        "equal": 2560, "total": 2560}, co
    cc.exits(lambda: read(new + [new[3] | {"success": not new[3]["success"]}], old, names, split), 2)

    # the sentences by the raw lo5 alone; a world without the drop; the noise floor's gate
    e = lambda lo5: {"diff_pp": 1.0, "lo5_one_sided": lo5}  # noqa: E731
    assert [sentence(e(v), "x") for v in (0.0, 1e-10, 2e-9, 0.0004, -1.0, None)] == [NOT_SHOWN] * 2 + ["x"] * 2 + [
        NOT_SHOWN] * 2 and sentence(None, "x") == NOT_SHOWN
    r = read(world(lambda L, a, c: i[c] % 8 in {**W, ("R20", "VST0.3"): S8(0, 1, 5, 6)}[L, a]), None, names, split)
    assert r["rules"]["Q8"]["diff_pp"] == 0.0 and r["rules"]["Q8"]["sentence"] == NOT_SHOWN, r["rules"]["Q8"]
    r = read(world(aim=[0.0006, 0.0007]), None, names, split)  # 1.5 mm against 1 mm: not informative
    no = "не читается: знание утонуло в невязке модели"
    assert r["rules"]["Q9"] == no and r["rules"]["Q8"]["diff_pp"] == 12.5 and r["report"]["scale"]["Q9"] == no
    assert all(r["report"]["cells"][L][n] == no for L in CELLS for n in ("M-V5", "M-G")) and all(
        r["report"]["cells"][L]["success"]["M"] == no and r["report"]["cells"][L]["T-V5"]["pairs"] == 640
        for L in CELLS) and r["report"]["lambda"]["R0"]["bands"]["[0, 4]"]["M-V5"] == no, r["report"]
    r = read(world(lambda L, a, c: win0(L, a, c) and not (L, a) == ("R20", "Gauto")), None, names, split)
    u = r["report"]["scale"]["Q8"]  # no G success at R20: L there and the trend undefined, every resample
    assert u["R20"]["L"] is None and u["R20"]["undefined_resamples"] == REPS and u["L0-L20"]["L"] is None, u

    # drops by reason, per cell; the trend on the cases valid in both
    c1, c2, c3, c4, c5, c6, c7, c8, c9 = rest[1:10]  # c6: every R0 arm a step late; c7: its R0 Base's e off by one;
    # c9: at R20 no fresh plan before the end in any arm; c4: also an invalid VT-m episode with an outlying a_floor
    ed = lambda r: (r | {"skipped_reason": "x"} if (r["case_id"], r["d"]) == (c1, 0) else  # noqa: E731
                    r | {"plans_in": [[t, t + 1] for t in range(0, 300, 10)]}
                    if (r["case_id"], r["d"], r["kind"]) == (c6, 0, "dynamic") else
                    r | {"e": r["e"] + 1} if (r["case_id"], r["d"], r["kind"]) == (c7, 0, "base") else
                    r | {"e": None} if (r["case_id"], r["d"]) == (c3, 0) else
                    r | {"replay_miss": 12} if (r["case_id"], r["d"], r["arm"]) == (c4, 20, "PBVS0.3") else
                    r | {"prefix_ok": False, "vtm": r["vtm"] | {"a_floor": [1.0, 0.0]}}
                    if (r["case_id"], r["d"], r["arm"]) == (c4, 20, "VTm") else
                    r | {"plans_in": [[0, 0]]} if (r["case_id"], r["d"], r["kind"]) == (c9, 20, "dynamic") else
                    r | {"plans_in": [[t, t + 21 * (t > 0)] for t in range(0, 300, 10)]}
                    if (r["case_id"], r["d"], r["arm"]) == (c5, 20, "VST0.3") else r)
    r = read([ed(x) for x in new if (x["case_id"], x["d"], x["arm"], x["kind"]) not in (
        (c2, 20, "VTm", "dynamic"), (c8, 20, "none", "base"))], None, names, split)
    assert r["checks"]["pass"] and r["report"]["drops"] == {"R0": {"skipped": 1, "no_event": 1},
                                                            "R20": {"missing": 2, "invalid": 1}}, r["report"]["drops"]
    out = (c1, c2, c3, c4, c8)
    assert r["rules"]["Q8"]["pairs"] == 635 and r["report"]["trend_cases"] == {
        "both_valid": 635, "out: valid in R0 only, dropped in R20": 3, "out: valid in R20 only, dropped in R0": 2}
    assert r["rules"]["Q8"]["diff_pp"] == round(100 * sum(q for c, q in zip(rest, q640) if c not in out) / 635, 3)
    assert r["report"]["cells"]["R0"]["T-G"]["diff_pp"] == hand("R0", "T", "G", (c1, c3)) and r["checks"][
        "invalid_pairs"]["R20"] == {"pairs": 638, "invalid": 1, "pass": True}
    lm = r["report"]["lambda"]["R20"]
    assert lm["mismatches"] == 1 and lm["mismatch_list"][0]["case"] == c5 and lm["mismatch_list"][0]["lambda"][
        "T"] == lm["mismatch_list"][0]["formula"] + 1 and sum(b["T-G"]["pairs"] for b in lm["bands"].values()) == 635
    assert lm["no_fresh_plan_before_the_end"] == 1 and r["report"]["lambda"]["R0"]["no_fresh_plan_before_the_end"] == 0
    fl = r["report"]["counters"]["noise_floor"]  # c4's invalid VT-m episode not in the floor
    assert fl["n"] == 2 * len([c for c in rest if i[c] % 3 == 0]) and fl["median_floor_m"] == 0.001, fl
    lm = r["report"]["lambda"]["R0"]  # c6: its arms agree, off the formula: listed, in its band
    assert lm["mismatches"] == 1 and lm["mismatch_list"][0]["case"] == c6 and set(lm["mismatch_list"][0][
        "lambda"].values()) == {lm["mismatch_list"][0]["formula"] + 1} and sum(
        b["T-G"]["pairs"] for b in lm["bands"].values()) + lm["outside_bands"] == 638, lm
    assert r["report"]["counters"]["R0"]["base_e"] == {"equal": 2552, "total": 2556}, r["report"]["counters"]["R0"]

    # the checks before reading, each alone
    six = cs.sources(rest)["plus"]
    dyn = lambda L, a: [x for x in new if x["d"] == CELLS[L] and x["arm"] == a and x["kind"] == "dynamic"  # noqa
                        and x["case_id"] in six]
    sub = lambda f: [f(x) for x in new]  # noqa: E731
    gone = lambda n: [x for x in new if x not in dyn("R0", "Gauto")[:n]]  # noqa: E731  (of 1884 in R0 Plus)
    inv = lambda n: sub(lambda x: x | {"prefix_ok": False} if x in dyn("R20", "PBVS0.3")[:n] else x)  # noqa: E731
    for why, rs, ok in (("complete", gone(19), False), ("complete", gone(18), True),
                        ("complete", [x for x in new if x["d"] != 20], False),
                        ("invalid_pairs", inv(13), False), ("invalid_pairs", inv(12), True),
                        ("conf", sub(lambda x: x | {"conf": x["conf"] | {"t_ramp": 4}} if x is new[7] else x), False),
                        ("arms/kbar", sub(lambda x: x | {"conf": x["conf"] | {"kbar": 0.2}} if x["d"] == 20 else x),
                         False),
                        ("arms/band", sub(lambda x: x | {"conf": x["conf"] | {"pbvs": bj | {"delta": 0.026}}}
                                          if x["d"] == 0 else x), False),
                        ("same_cases", [x for x in new if (x["case_id"], x["d"]) != (rest[9], 20)], True),  # a drop
                        ("same_cases", new + [new[0] | {"case_id": run.set_ids(split, "eval300")[0]}], False)):
        r = read(rs, None, names, split)
        f, _, k_ = why.partition("/")
        c = {L: r["checks"][f][L][k_] for L in CELLS} if k_ else r["checks"][f]
        flat_ = [v["pass"] for v in c.values()] if "pass" not in c else [c["pass"]]
        assert r["checks"]["pass"] == ok == ("rules" in r) == all(flat_), (why, ok, c)
        if not ok:
            cc.exits(lambda: finish(r), 3)

    # the smoke's code check and R0's gate
    sm = [rec("R0", x[2], x[3], x[4]) for x in sorted(cc.lkeys(plan("Rsmoke", split)))]
    ok, det = smoke_check(sm, split)
    assert ok and det["complete"] == {"expected": 30, "missing": 0, "pass": True} and det["prefix"]["dynamic"] == 24, \
        det
    for bad in ([x for x in sm if x["arm"] != "VTm" or x["case_id"] != six[0]],
                [x | {"prefix_ok": False} if x["arm"] == "Gauto" and x["case_id"] == six[5] else x for x in sm],
                [x | {"plans_in": []} if x["arm"] == "PBVS0.3" else x for x in sm],
                [x | {"conf": x["conf"] | {"kbar": 0.2}} for x in sm],
                [x | {"conf": x["conf"] | {"pbvs": bj | {"delta": 0.026}}} for x in sm],
                [x | {"conf": x["conf"] | {"d": 20}} for x in sm],
                [x | {"vst": {"steps_window": 1}} if x["arm"] == "VST0.3" else x for x in sm],
                [x | {"vtm": {k_: v for k_, v in x["vtm"].items() if k_ != "a_floor"}} if x["arm"] == "VTm" else x
                 for x in sm],
                [x | {"base_cut": True} if x["kind"] == "base" and x["case_id"] == six[2] else x for x in sm],
                [x | {"skipped_reason": "x"} if x["case_id"] == six[1] else x for x in sm]):
        assert not smoke_check(bad, split)[0], smoke_check(bad, split)[1]
    r0 = [x for x in new if x["d"] == 0]
    assert complete(r0, "R0", split)[0] and complete([x | {"skipped_reason": "x"} if x is r0[0] else x for x in r0],
                                                     "R0", split)[0]
    assert complete(r0[1:], "R0", split) == (False, {"line": "R0", "expected": 3200, "missing": 1})
    assert not complete([x for x in new if x["d"] == 20], "R0", split)[0]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--smoke-check", nargs="+", metavar="JSONL", help="rsmoke's records: the code check, exit 0 or 3")
    g.add_argument("--complete", nargs="+", metavar="JSONL", help="--line's records all there: exit 0, or 3")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_R.jsonl: checks, rules, report")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--c", default=CPOD, help="pod C's records (read only): grid_all_C, grid_all_T, vst_k.json")
    ap.add_argument("--out", help="the reading: JSON too")
    a = ap.parse_args()
    a.read = a.read[a.read[:1] == ["read"]:]  # "read" before the files is optional
    if bool(a.read) + a.plan + a.selftest + bool(a.smoke_check) + bool(a.complete) != 1:
        ap.error("one mode: --plan, --smoke-check, --complete, --selftest, or [read] JSONL...")
    if (a.plan or a.complete) and not a.line:
        ap.error("--plan and --complete need --line")
    if a.selftest:
        selftest()
    elif a.plan:
        print(cc.tsv(plan(a.line)))
    elif a.smoke_check or a.complete:
        recs = [x for f in a.smoke_check or a.complete for x in e2.load(f)]
        ok, det = smoke_check(recs) if a.smoke_check else complete(recs, a.line)
        print(e2.dump(det))
        what = "smoke (code only)" if a.smoke_check else f"{a.line} complete"
        print(f"C1-E4 delay {what}: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    else:
        names, split = cs.libero_names(), maxwrap.load_split(cs.SPLIT)
        old = old300(a.c, names, split) if a.c else None  # --c '': no old 300
        finish(read([x for f in a.read for x in e2.load(f) if "kind" in x], old, names, split), a.out)
