"""C1-E4 placebo cell: the direction of the correction, or the perturbation itself
(docs/superpowers/specs/2026-11-08-c1-e4-placebo-design.md, "spec"), on pod P (scripts/gpu_c1e4m.sh POD=P psmoke,
pgrid): the runner plan per line, the smoke's code check and the reading. Reuses scripts/c1e4c_summary.py (mlines,
plan, lk, lkeys, KBAR, tsv), scripts/c1e4v_summary.py (CMODE, CK, ids, groups, unique, pairs, est) and
scripts/c1e4m_summary.py (e2.diff_pp's bootstrap); the controls' and part 2's records are read only; run
from the repository root. R = results/c1-e4/placebo/pod_P:
    python scripts/c1e4p_summary.py --plan --line L
        L: Psmoke P0 P10, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4p_summary.py --smoke-check R/smoke/Psmoke_*.jsonl   the code only (§10), exit 0, or 3
    python scripts/c1e4p_summary.py [read] R/grid_all_P.jsonl [--c results/c1-e4/controls/pod_C/grid_all_C.jsonl]
            [--p2 results/c1-e4/part2] [--out R/summary_P.json]
        the checks before reading (§8), then the rule (§6), the measurements and the angle dependence (§5) and the
        report (§7); a failed check prints only the checks, exit 3. Needs hf-libero (the task clusters)
    python scripts/c1e4p_summary.py --selftest
"""

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import P, TOL, e2, maxwrap, run  # noqa: E402

ARMS = ("none", "none2", "Gauto", "GautoSham")  # spec §3: one process, one Base cut at the event
SIDE = dict(zip(ARMS, ("N", "N2", "G", "P")))  # spec §2's letters (N2: N')
CELLS = {"P0": 0, "P10": 10}  # spec §3: P0 first
CTL = {"P0": "2", "P10": "3"}  # the controls line whose Plus cases, order, Base groups and --base-prefix a cell runs
LINES = ("Psmoke", *CELLS)
N_PLUS, KBAR = 187, cc.KBAR["m", P]
# spec §5: per pair x - y, each side a weighted sum of the arms' successes; Q = (P - N) - (G - N) / 2 = P - (G + N) / 2
DIFF = {"Q": ({"P": 1}, {"G": 0.5, "N": 0.5}), "G-N": ({"G": 1}, {"N": 1}), "P-N": ({"P": 1}, {"N": 1}),
        "G-P": ({"G": 1}, {"P": 1}), "N'-N": ({"N2": 1}, {"N": 1}), "G-N'": ({"G": 1}, {"N2": 1})}
BANDS = ((0, 60), (60, 120), (120, 180))  # spec §5: phi in degrees, [0, 60), [60, 120), [120, 180]
# spec §7: part 2's "none" through the pod (pod A), per cell
P2F = {"P0": "pod_A/M1M2_pi05_eval300_plus_d0.jsonl", "P10": "pod_A/D_pi05_eval300_plus_d10.jsonl"}
CTRL = "results/c1-e4/controls/pod_C/grid_all_C.jsonl"
# spec §6, word for word
RULE = {"П0": "наш выигрыш над «нет» на этом поде не показан; правило плацебо не читается",
        "П1": "фиктивная поправка той же величины дала над «нет» меньше половины нашего выигрыша: выигрыш "
              "приписывается направлению поправки, а не самому возмущению действий",
        "П2": "фиктивная поправка той же величины дала над «нет» не меньше половины нашего выигрыша: результат "
              "приписывается самому возмущению действий, а не направлению поправки",
        "П3": "не разрешено: на этих данных вклад направления не отделяется от самого возмущения"}
BOUND = "фиктивная ветвь даёт над «нет» не более {:.1f} п.п. при нашем выигрыше {:.1f} п.п."
NUMS = "Q = {}, P − N = {}, G − N = {} (п.п., 95%)"
# spec §5, the angle dependence's sentence by its bound
SLOPE = {True: "успех плацебо падает с ростом ошибки, которую оставляет поправка: β = {:.1f} п.п. на единицу r "
               "[{:.1f}; {:.1f}]",
         False: "зависимость успеха от угла не показана"}


def ctl(line, split):
    """The controls line's Plus runner lines a cell mirrors: line 2 as it ran (C2: the first VS_STOP Plus cases, C2_nv:
    the rest), line 3 at d 10."""
    d = CELLS[line]
    return [ln for ln in cc.plan(CTL[line], cv.CMODE, cv.CK, split)
            if ln["source"] == "plus" and f"--d {d} " in ln["args"] + " "]


def plan(line, split=None):
    """spec §3, §10: the runner lines of a cell, phase line.lower(), in c1e4c_summary.mlines's format: the four arms
    with part 2's kbar and --base-prefix on the controls line's Plus cases, one runner line per controls line (P0: P0
    for C2's cases, P0nv for C2_nv's; P10 as C3 d10 Plus, its whole source), so its cases, order and Base groups are
    the controls' (asserted, 187 cases). Psmoke: P0's first 6 cases, the smoke's own line."""
    split = split or maxwrap.load_split(cs.SPLIT)
    if line == "Psmoke":
        six = [c for ln in plan("P0", split) for c in cv.ids(ln, split)][:6]
        (ln,) = cc.mlines("psmoke", "Psmoke", P, "eval300", ARMS, split, cases=set(six), kbar=KBAR)
        return [ln | {"id": "Psmoke_pi05"}]
    out, d, c = [], CELLS[line], ctl(line, split)
    for x in c:
        tag = line + "nv" * x["id"].startswith("C2_nv")
        cases = set(cv.ids(x, split)) if " --cases " in x["args"] else None
        out += [ln for ln in cc.mlines(line.lower(), tag, P, "eval300", ARMS, split, d, cases, KBAR)
                if ln["source"] == "plus"]
    ids = [i for ln in out for i in cv.ids(ln, split)]
    assert len(ids) == len(set(ids)) == N_PLUS and ids == [i for x in c for i in cv.ids(x, split)] and \
        cv.groups(out, split) == cv.groups(c, split), f"{line}: not controls line {CTL[line]}'s Plus cases and groups"
    return out


def phi(theta):
    """spec §5: phi = min(theta, 2 pi - theta) in [0, pi], r = 2 sin(phi / 2), the band of phi (BANDS)."""
    f = min(theta, 2 * math.pi - theta)
    return f, 2 * math.sin(f / 2), next(j for j, (_, hi) in enumerate(BANDS) if f < math.radians(hi) or hi == 180)


def theta_ok(r):
    """spec §8: a P record's theta in [0, 2 pi), the one sham_theta gives its case."""
    th = (r.get("sham") or {}).get("theta")
    return type(th) is float and 0 <= th < 2 * math.pi and th == run.sham_theta(run.case_seed(r["case_id"]))


def arm_checks(rs):
    """spec §8 on the Dynamic records of one cell (or of the smoke): kbar part 2's on G and P; every P record's theta
    its case's; every N' record with its seed shift; N''s act_hash off N's in at least half the episodes with an event
    (none of them: fails)."""
    by = {(r["case_id"], r["arm"]): r for r in rs}
    th, n2 = [r for r in rs if r["arm"] == "GautoSham"], [r for r in rs if r["arm"] == "none2"]
    ev = [(by[c, "none"], r) for (c, a), r in by.items() if a == "none2" and (c, "none") in by
          and by[c, "none"]["e"] is not None]
    k = sum(r["conf"].get("kbar") != KBAR for r in rs if r["arm"] in ("Gauto", "GautoSham"))
    t, s = sum(not theta_ok(r) for r in th), sum(r.get("seed_shift") != run.SEED2 for r in n2)
    dif = sum(a["act_hash"] != b["act_hash"] for a, b in ev)
    return {"kbar": {"want": KBAR, "off": k, "pass": not k}, "theta": {"records": len(th), "off": t, "pass": not t},
            "seed_shift": {"records": len(n2), "off": s, "pass": not s},
            "act_hash_N'_vs_N": {"events": len(ev), "differ": dif, "pass": bool(ev) and dif >= 0.5 * len(ev) - TOL}}


def smoke_check(recs, split=None):
    """spec §10: psmoke's records checked for the code only, never an outcome: the line complete (its 6 cut Bases and
    24 Dynamic episodes, none skipped), every Dynamic one valid (prefix_ok, no replay miss), arm_checks. (pass,
    details)."""
    split = split or maxwrap.load_split(cs.SPLIT)
    want = cc.lkeys(plan("Psmoke", split))
    by = cv.unique([r for r in recs if "kind" in r], cc.lk)
    have = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    dyn = [r for r in have.values() if r["kind"] == "dynamic"]
    out = {"complete": {"expected": len(want), "missing": len(want - have.keys()), "off_plan": len(by.keys() - want)},
           "prefix": {"dynamic": len(dyn), "invalid": sum(r.get("prefix_ok") is not True or cs.invalid(r)
                                                          for r in dyn)},
           **arm_checks(dyn)}
    out["complete"]["pass"] = out["complete"]["missing"] == 0
    out["prefix"]["pass"] = out["prefix"]["invalid"] == 0
    return all(v["pass"] for v in out.values()), out


def side(s, w):
    return sum(v * s[k] for k, v in w.items())


def prs(pairs, name):
    """(case, cluster, x, y) of a DIFF on the pairs (case, cluster, successes, theta)."""
    x, y = DIFF[name]
    return [(c, cl, side(s, x), side(s, y)) for c, cl, s, _ in pairs]


def slope(pts):
    """spec §5, the angle dependence: the OLS slope of y on r over pts (cluster, r, y), every pair weighs one; the
    cluster bootstrap of e2.diff_pp (clusters sorted, e2.N_BOOT resamples of whole clusters, generator seed 0), the
    slope recomputed in each from the drawn clusters' sums; percentiles 2.5 and 97.5 (95 %), 5 and 95 (one-sided).
    Raw floats."""
    if not pts:
        return None
    cl = defaultdict(list)
    for c, r, y in pts:
        cl[c].append((r, y))
    keys = sorted(cl)
    S = np.array([[len(v), sum(r for r, _ in v), sum(y for _, y in v), sum(r * r for r, _ in v),
                   sum(r * y for r, y in v)] for v in (cl[k] for k in keys)])
    b = lambda T: (T[..., 0] * T[..., 4] - T[..., 1] * T[..., 2]) / (T[..., 0] * T[..., 3] - T[..., 1] ** 2)  # noqa
    pick = np.random.default_rng(0).integers(len(keys), size=(e2.N_BOOT, len(keys)))
    lo2, lo5, hi95, hi = np.percentile(b(S[pick].sum(1)), [2.5, 5, 95, 97.5])
    return {"beta_pp_per_r": float(b(S.sum(0))), "lo2.5": float(lo2), "lo5_one_sided": float(lo5),
            "hi95_one_sided": float(hi95), "hi97.5": float(hi), "pairs": len(pts), "clusters": len(keys)}


def est(prs):
    """spec §5: the paired difference x - y in pp of (case, cluster, x, y), e2.diff_pp's estimate and percentiles, and
    hi95_one_sided, the 95th percentile of the same draws (clusters sorted, e2.N_BOOT, default_rng(0)), raw: the
    rule tests the raw bounds (cv.upper95 rounds to 1e-3, so a bound in [-0.0005, 0) would read as 0)."""
    p = [q[1:] for q in prs]
    cl = defaultdict(list)
    for c, a, b in p:
        cl[c].append(float(a) - float(b))
    keys = sorted(cl)
    s, n = np.array([sum(cl[k]) for k in keys]), np.array([len(cl[k]) for k in keys])
    pick = np.random.default_rng(0).integers(len(keys), size=(e2.N_BOOT, len(keys)))
    return e2.diff_pp(p) | {"hi95_one_sided": float(np.percentile(100 * s[pick].sum(1) / n[pick].sum(1), 95))}


def slope_sentence(sl):
    """spec §5: the angle dependence's sentence by the raw one-sided bound hi95(beta) < 0 (< -TOL), its numbers as
    printed."""
    r = e2.rnd(sl)
    return (SLOPE[True].format(r["beta_pp_per_r"], r["lo2.5"], r["hi97.5"]) if sl["hi95_one_sided"] < -TOL
            else SLOPE[False])


def rule(m):
    """spec §6 on the pooled raw estimates m (est's): П0 the precondition lo5(G - N) > 0, then П1 hi95(Q) < 0, П2
    lo5(Q) >= 0 (equality passes), else П3, each against TOL (strict: beyond it; equality: within it); the sentence,
    П1's bound (pooled hi95(P - N) and G - N), П3's numbers, printed rounded."""
    gn, q, pn = m["G-N"], m["Q"], m["P-N"]
    b = ("П0" if not gn["lo5_one_sided"] > TOL else "П1" if q["hi95_one_sided"] < -TOL else
         "П2" if q["lo5_one_sided"] >= -TOL else "П3")
    out, (gn, q, pn) = {"branch": b, "sentence": RULE[b]}, (e2.rnd(x) for x in (gn, q, pn))  # printed as rounded
    ci = lambda d: f"{d['diff_pp']:.1f} [{d['lo2.5']:.1f}; {d['hi97.5']:.1f}]"  # noqa: E731
    if b == "П1":
        out["bound"] = BOUND.format(pn["hi95_one_sided"], gn["diff_pp"])
    if b == "П3":
        out["numbers"] = NUMS.format(ci(q), ci(pn), ci(gn))
    return out


def load_p2(root=cc.P2):
    """Part 2's pi0.5 records (pod A) of the cells' d, Plus: M1 d 0 and D d 10."""
    return [x for f in P2F.values() for x in e2.load(Path(root) / f) if "kind" in x]


def read(new, ctrl, p2, names, split):
    """spec §4-§8. The checks before reading (each runner line's Dynamic episodes present on >= 99 %, its cut Bases not
    counted; invalid pairs <= 2 % per cell, a missing record or a skipped case not one; conf equal per cell but arms;
    arm_checks per cell), then the rule on the pooled cells, the measurements pooled and per cell, the angle dependence
    and the report. new: grid_all_P.jsonl; ctrl: the controls' records (C2, C2_nv, C3 among them); p2: load_p2(). A
    pair: one case in one cell, its four Dynamic episodes all there and valid (cs.invalid: prefix_ok false or a replay
    miss), on one Base in one process; a case skipped in a cell drops there. Clusters: maxwrap.base_task. Records off
    the plan are counted, never read."""
    lns = {L: plan(L, split) for L in CELLS}
    want = {x: L for L, ln in lns.items() for x in cc.lkeys(ln)}
    by = cv.unique(new, cc.lk)
    used = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    skip = {(want[x], r["case_id"]) for x, r in by.items() if x in want and r.get("skipped_reason")}
    of = lambda L, kind, a: {r["case_id"]: r for x, r in used.items()  # noqa: E731
                             if want[x] == L and r["kind"] == kind and r["arm"] == a}
    comp = {}
    for ln in [ln for v in lns.values() for ln in v]:  # spec §8: per runner line
        exp = {x for x in cc.lkeys([ln]) if x[3] == "dynamic"}
        miss = len(exp - used.keys())
        comp[ln["id"]] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    cases = {L: [c for ln in v for c in cv.ids(ln, split)] for L, v in lns.items()}
    cby = cv.unique(ctrl, cc.lk)
    cw = {x: L for L in CELLS for x in cc.lkeys(ctl(L, split))}
    cof = lambda L, kind, a: {r["case_id"]: r for x, r in cby.items()  # noqa: E731
                              if cw.get(x) == L and r["kind"] == kind and r["arm"] == a}
    pby = cv.unique(p2, cs.key)
    pof = lambda L, kind, a: {r["case_id"]: r for r in pby.values() if r["brain"] == P  # noqa: E731
                              and r["d"] == CELLS[L] and r["kind"] == kind and r["arm"] == a}
    one = {r["case_id"]: r for r in list(pby.values()) + [r for r in cby.values() if "case_id" in r]
           + list(used.values())}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    pairs, drops = {}, {}
    for L in CELLS:
        out, drop, arm = [], Counter(), {a: of(L, "dynamic", a) for a in ARMS}
        for c in cases[L]:
            q = {SIDE[a]: arm[a].get(c) for a in ARMS}
            if (L, c) in skip:
                drop["skipped"] += 1
            elif None in q.values():
                drop["missing"] += 1
            elif any(cs.invalid(r) for r in q.values()):
                drop["invalid"] += 1
            else:
                out.append((c, clu[c], {k: r["success"] for k, r in q.items()}, q["P"]))
        pairs[L], drops[L] = out, dict(drop)
    conf = defaultdict(set)
    for x, r in used.items():  # spec §8: conf equal per cell but arms
        conf[want[x]].add(json.dumps({f: v for f, v in r["conf"].items() if f != "arms"}, sort_keys=True))
    mixed = {L: sorted(v) for L, v in conf.items() if len(v) > 1}
    checks = {"records": len(new), "read": len(used), "skipped": len(skip),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for x, r in by.items() if x not in want)),
              "complete": comp,
              "invalid_pairs": {L: {"pairs": len(pairs[L]) + drops[L].get("invalid", 0),
                                    "invalid": drops[L].get("invalid", 0),
                                    "pass": drops[L].get("invalid", 0) <= 0.02 * (len(pairs[L]) + drops[L].get(
                                        "invalid", 0)) + TOL} for L in CELLS},
              "conf": {"except": ["arms"], "mixed": mixed, "pass": not mixed},
              "arms": {L: arm_checks([r for x, r in used.items() if want[x] == L and r["kind"] == "dynamic"])
                       for L in CELLS}}
    checks["pass"] = bool(all(v["pass"] for f in ("complete", "invalid_pairs") for v in checks[f].values())
                          and checks["conf"]["pass"]
                          and all(v["pass"] for a in checks["arms"].values() for v in a.values()))
    if not checks["pass"]:
        return {"checks": checks}

    pool = pairs["P0"] + pairs["P10"]  # spec §5: every case x cell weighs one, a cluster carries both cells' pairs
    raw = {g: {n: est(prs(p, n)) for n in DIFF} for g, p in (("pooled", pool), *pairs.items())}
    drops["pooled"] = dict(sum((Counter(drops[L]) for L in CELLS), Counter()))
    meas = {g: {n: e2.rnd(v) for n, v in m.items()} | {"drops": drops[g]} for g, m in raw.items()}
    geo = [(c, cl, s, phi(r["sham"]["theta"])) for c, cl, s, r in pool]
    sl = slope([(cl, f[1], 100 * (float(s["P"]) - float(s["G"]))) for c, cl, s, f in geo])
    angle = {"test": "pi05, P0 + P10 pooled: OLS slope of (P - G) pp on r = 2 sin(phi / 2), phi = min(theta, 2 pi - "
                     "theta)", **e2.rnd(sl), "sentence": slope_sentence(sl),
             "bands_P-N": {f"[{lo}, {hi}{']' if hi == 180 else ')'}": e2.rnd(e2.diff_pp(
                 [(cl, side(s, {"P": 1}), side(s, {"N": 1})) for c, cl, s, f in geo if f[2] == j]))
                 for j, (lo, hi) in enumerate(BANDS)}}
    return {"checks": checks, "rule": rule(raw["pooled"]), "measurements": meas, "angle": angle,
            "report": report(pairs, geo, raw, of, cof, pof, clu, cases, used)}


def report(pairs, geo, raw, of, cof, pof, clu, cases, used):
    """spec §7, only a report: success per arm and cell on the valid pairs; brain calls per episode and the g_on share
    of G and P; phi's band shares and the mean r over the pooled pairs; rho = (P - N) / (G - N), a point; through the
    pod, by case (cv.pairs: the same seed, both valid, the Bases' e equal; not bitwise): G - G-auto of controls lines
    2 / 3 (pod C) and N - part 2's "none" (pod A); the Bases' e against the controls lines'."""
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    rep = defaultdict(dict)
    for L, p in pairs.items():
        rep["success"][L] = {a: [mean([s[SIDE[a]] for _, _, s, _ in p]), len(p)] for a in ARMS}
        for a in ARMS:
            g = list(of(L, "dynamic", a).values())
            rep["calls"][f"{L}/{a}"] = {"n": len(g), "sched": mean([r["calls_sched"] for r in g]),
                                        "trig": mean([r["calls_trig"] for r in g]),
                                        "per_episode": mean([r["calls_sched"] + r["calls_trig"] for r in g])}
            if a in ("Gauto", "GautoSham"):
                rep["g_on"][f"{L}/{a}"] = mean([bool(r["g_on"]) for r in g])
        for name, a, o in ((f"G - G-auto of controls line {CTL[L]} (pod C)", "Gauto", cof),
                           ("N - part 2's none (pod A)", "none", pof)):
            p, dr = cv.pairs(of(L, "dynamic", a), o(L, "dynamic", a), of(L, "base", "none"), o(L, "base", "none"), clu,
                             cases[L])
            rep["through_pod"][f"{L}: {name}"] = cv.est(p) | {"drops": dr}
        b, cbs = of(L, "base", "none"), cof(L, "base", "none")
        rep["base_e_controls"][L] = {"equal": sum(cbs.get(c, {}).get("e", "-") == r["e"] for c, r in b.items()),
                                     "total": len(b)}
    rep["phi"] = {"bands": {f"{lo}-{hi}": mean([f[2] == j for *_, f in geo]) for j, (lo, hi) in enumerate(BANDS)},
                  "mean_r": mean([f[1] for *_, f in geo]), "pairs": len(geo)}
    for g, m in raw.items():
        gn = m["G-N"]["diff_pp"]
        rep["rho_point"][g] = None if abs(gn) < TOL else round(m["P-N"]["diff_pp"] / gn, 4)
    rep["gpu"] = sorted({str(r["conf"].get("gpu")) for r in used.values()})
    return dict(rep)


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 placebo checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


# ------------------------------------------------------------------ selftest

def selftest():
    """The plan on the real split: P0 in two runner lines (C2's 108 Plus cases, then C2_nv's 79), P10 one (C3 d10 Plus,
    its whole source), the four arms with part 2's kbar and --base-prefix, the controls' cases, order and Base groups;
    Psmoke on P0's first 6; the smoke's check; the reading on a synthetic pod."""
    split = maxwrap.load_split(cs.SPLIT)
    plus = cs.sources(run.set_ids(split, "eval300"))["plus"]
    a4 = "--arms=none,none2,Gauto,GautoSham"
    for L, ids, d in (("P0", ["P0_pi05_eval300_plus_d0", "P0nv_pi05_eval300_plus_d0"], 0),
                      ("P10", ["P10_pi05_eval300_plus_d10"], 10)):
        lns = plan(L, split)
        assert [ln["id"] for ln in lns] == ids and [c for ln in lns for c in cv.ids(ln, split)] == plus, lns
        assert all(ln["phase"] == L.lower() and ln["arms"] == list(ARMS) and ln["prefix"] and ln["source"] == "plus"
                   and ln["args"].endswith(f" --kbar {KBAR} --base-prefix") and f" --d {d} " in ln["args"]
                   for ln in lns), lns
        assert cv.groups(lns, split) == cv.groups(ctl(L, split), split) and sum(map(cc.n_of, lns)) == 4 * N_PLUS
    p0, (p10,) = plan("P0", split), plan("P10", split)
    assert [len(cv.ids(ln, split)) for ln in p0] == [108, 79] and p0[0]["args"] == (
        f"--brain pi05 --source plus --set eval300 {a4} --d 0 --cases {','.join(plus[:108])} --kbar 0.306 "
        "--base-prefix"), p0[0]["args"]
    assert p10["args"] == f"--brain pi05 --source plus --set eval300 {a4} --d 10 --kbar 0.306 --base-prefix", p10
    (sm,) = plan("Psmoke", split)
    assert (sm["id"], sm["phase"], cc.n_of(sm), cv.ids(sm, split)) == ("Psmoke_pi05", "psmoke", 24, plus[:6]), sm
    assert sm["args"] == (f"--brain pi05 --source plus --set eval300 {a4} --d 0 --cases {','.join(plus[:6])} "
                          "--kbar 0.306 --base-prefix"), sm
    assert cc.tsv([sm]).split("\t")[:6] == ["psmoke", "c1e4m", P, "plus", "Psmoke_pi05", "24"]
    selftest_read(split, plus)
    print("selftest ok")


def selftest_read(split, plus):
    """The smoke check and the reading on a synthetic pod, the controls and part 2: success by the case's place i among
    the 187 in cycles of 8 (W: the residues that succeed, by side, cell and arm), so every difference is exact; P's
    theta sham_theta's of the case; task clusters from fake LIBERO names as c1e4o_summary's selftest."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    i = {c: j for j, c in enumerate(plus)}
    S8 = lambda *k: frozenset(k)  # noqa: E731
    # the П1 world, the cells apart (pooling one cell alone or a swap of arms changes every number)
    W = {("pod", "P0", "none"): S8(0, 1), ("pod", "P0", "none2"): S8(0, 2, 3),
         ("pod", "P0", "Gauto"): S8(0, 1, 2, 3, 4),
         ("pod", "P0", "GautoSham"): S8(1, 7), ("pod", "P10", "none"): S8(0), ("pod", "P10", "none2"): S8(1, 6),
         ("pod", "P10", "Gauto"): S8(0, 1, 2, 3, 4, 5), ("pod", "P10", "GautoSham"): S8(0),
         ("C", "P0", "Gauto"): S8(0, 1, 2), ("C", "P10", "Gauto"): S8(0, 1, 2, 3, 4, 5, 6),
         ("A", "P0", "none"): S8(5, 6), ("A", "P10", "none"): S8(0, 1, 2)}
    win0 = lambda s, L, a, c, W=W: i[c] % 8 in W.get((s, L, a), ())  # noqa: E731
    th = {c: run.sham_theta(run.case_seed(c)) for c in plus}

    def rec(s, L, c, kind, a, arms, win):
        h = sum(map(ord, c))
        su = list(run.MAX_STEPS)[h % 4]
        d = CELLS[L]
        cf = {"brain": P, "revision": "r", "precision": "fp32", "s": 10, "d": d, "kbar": KBAR, "t_ramp": 5, "k_p": 1.0,
              "tau_k": 0.01, "R": 360, "arms": list(arms), "split": split["sha256"]}
        r = {"brain": P, "case_id": c, "kind": kind, "arm": a, "method": a, "d": d, "policy_seed": 195,
             "conf": cf | ({"gpu": "RTX 4090"} if s != "A" else {}), "suite": su,
             "task_name": f"{su}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None, "e": 30,
             "prefix_ok": None if kind == "base" else True, "replay_miss": None, "skipped_reason": None,
             "success": None if kind == "base" else bool(win(s, L, a, c)), "act_hash": f"{s}/{L}/{c}/{kind}/{a}",
             "calls_sched": 30, "calls_trig": int(a == "GautoSham"),
             "g_on": a in ("Gauto", "GautoSham") and i[c] % 3 > 0}
        if a == "GautoSham":
            r["sham"] = {"theta": th[c]}
        if a == "none2":
            r["seed_shift"] = run.SEED2
        return r

    def world(win=win0):
        new = [rec("pod", L, x[2], x[3], x[4], x[7], win) for L in CELLS for x in sorted(cc.lkeys(plan(L, split)))]
        ctrl = [rec("C", L, x[2], x[3], x[4], x[7], win) for L in CELLS for x in sorted(cc.lkeys(ctl(L, split)))]
        p2 = [rec("A", L, c, k, a, ("none", "Gauto", "PPC"), win) for L in CELLS for c in plus
              for k, a in [("dynamic", a) for a in ("none", "Gauto", "PPC")] + [("base", "none")]]
        return new, ctrl, p2

    def want(Ls, x, y, s="pod", sy="pod", win=win0, out=()):  # a difference by hand, pp: the mean over the pairs of
        # the cells Ls ("P0", "P10" or both), out (case, cell) dropped; x, y: the arms' weights
        ps = [(L, c) for L in Ls.split("+") for c in plus if (c, L) not in out]
        v = lambda side_, L, w, c: sum(k * win(side_, L, a, c) for a, k in w.items())  # noqa: E731
        return round(100 * sum(v(s, L, x, c) - v(sy, L, y, c) for L, c in ps) / len(ps), 3)

    A = {v: k for k, v in SIDE.items()}
    dw = lambda n: tuple({A[k]: w for k, w in part.items()} for part in DIFF[n])  # noqa: E731  (arms' weights)
    new, ctrl, p2 = world()
    rd = lambda n=new, c=ctrl, p=p2: read(n, c, p, names, split)  # noqa: E731
    probe = rec("pod", "P0", plus[0], "dynamic", "G", ("G",), win0)  # off the plan
    res = rd(new + [probe] + new[:5], ctrl + ctrl[:3])  # [:5]: identical copies
    ch, ms, ru, an, rp = (res[k] for k in ("checks", "measurements", "rule", "angle", "report"))
    assert ch["pass"] and ch["off_plan"] == {f"{P}/G": 1} and ch["read"] == len(new) == 2 * 5 * N_PLUS, ch
    assert ch["complete"] == {"P0_pi05_eval300_plus_d0": {"expected": 432, "missing": 0, "pass": True},
                              "P0nv_pi05_eval300_plus_d0": {"expected": 316, "missing": 0, "pass": True},
                              "P10_pi05_eval300_plus_d10": {"expected": 748, "missing": 0, "pass": True}}, ch
    assert ch["invalid_pairs"] == {L: {"pairs": 187, "invalid": 0, "pass": True} for L in CELLS}, ch
    assert all(v["act_hash_N'_vs_N"] == {"events": 187, "differ": 187, "pass": True} for v in ch["arms"].values())
    # exact differences per cell and pooled (every case x cell weighs one: the mean of the two cells' means here)
    for L in CELLS:
        got = {n: v["diff_pp"] for n, v in ms[L].items() if n in DIFF}
        assert got == {n: want(L, *dw(n)) for n in DIFF} and ms[L]["drops"] == {}, (L, got)
    pool = {n: want("P0+P10", *dw(n)) for n in DIFF}
    assert {n: ms["pooled"][n]["diff_pp"] for n in DIFF} == pool, (ms["pooled"], pool)
    assert pool["Q"] == round(-9450 / 374, 3), pool
    assert pool["N'-N"] != 0 and len({ms[g]["Q"]["diff_pp"] for g in ms}) == 3
    assert all(ms["pooled"][n]["pairs"] == 374 and ms["pooled"][n]["lo2.5"] <= ms["pooled"][n]["lo5_one_sided"]
               <= ms["pooled"][n]["diff_pp"] <= ms["pooled"][n]["hi95_one_sided"] <= ms["pooled"][n]["hi97.5"]
               for n in DIFF), ms["pooled"]
    # the bootstrap of Q, by hand: clusters sorted, 2000 resamples of whole clusters, seed 0, hi95 of the same draws
    q, rows, q95 = boot_by_hand(new, names, lambda s: s["P"] - (s["G"] + s["N"]) / 2)
    assert {k: ms["pooled"]["Q"][k] for k in q} == q, (ms["pooled"]["Q"], q)
    h = est([(None, cl, y, 0.0) for cl, _, y in rows])["hi95_one_sided"]
    assert h == q95 and round(h, 3) != h, (h, q95)  # raw, the rule's: not the printed 1e-3
    assert ru == {"branch": "П1", "sentence": RULE["П1"], "bound": BOUND.format(
        ms["pooled"]["P-N"]["hi95_one_sided"], ms["pooled"]["G-N"]["diff_pp"])}, ru
    # the angle: P independent of theta here, the slope not shown; by hand it is the same bootstrap with OLS per draw
    b, _, _ = boot_by_hand(new, names, lambda s: 100 * (s["P"] - s["G"]), slope_=True)
    assert {k: round(an[k], 3) for k in b} == b and an["sentence"] == SLOPE[False] and an["pairs"] == 374, (an, b)
    phis = [phi(th[c]) for c in plus]
    for j, (lo, hi) in enumerate(BANDS):
        band = an["bands_P-N"][f"[{lo}, {hi}{']' if hi == 180 else ')'}"]
        k = [c for c, f in zip(plus, phis) if f[2] == j]
        assert band["pairs"] == 2 * len(k) > 0 and band["diff_pp"] == round(100 * sum(
            win0("pod", L, "GautoSham", c) - win0("pod", L, "none", c) for L in CELLS for c in k) / (2 * len(k)), 3)
        assert rp["phi"]["bands"][f"{lo}-{hi}"] == round(len(k) / N_PLUS, 4), rp["phi"]
    assert rp["phi"]["mean_r"] == round(float(np.mean([f[1] for f in phis])), 4) and rp["phi"]["pairs"] == 374
    assert phi(0.0)[1:] == (0.0, 0) and phi(math.pi)[2] == 2 and phi(math.pi / 3)[2] == 1 and abs(
        phi(math.pi / 3)[1] - 1) < 1e-12 and abs(phi(2 * math.pi - 0.5)[0] - 0.5) < 1e-12
    # the report: success on the pairs, calls, g_on, rho, through the pod, the Bases' e
    assert rp["success"]["P0"]["GautoSham"] == [round(sum(win0("pod", "P0", "GautoSham", c) for c in plus) / 187, 4),
                                                187] and rp["calls"]["P10/GautoSham"]["per_episode"] == 31.0, rp
    assert rp["g_on"] == {f"{L}/{a}": round(sum(i[c] % 3 > 0 for c in plus) / 187, 4) for L in CELLS
                          for a in ("Gauto", "GautoSham")}, rp["g_on"]
    assert rp["rho_point"]["pooled"] == round(pool["P-N"] / pool["G-N"], 4), rp["rho_point"]
    for L in CELLS:
        tp = rp["through_pod"]
        assert tp[f"{L}: G - G-auto of controls line {CTL[L]} (pod C)"]["diff_pp"] == want(
            L, {"Gauto": 1}, {"Gauto": 1}, sy="C") and tp[f"{L}: N - part 2's none (pod A)"]["diff_pp"] == want(
            L, {"none": 1}, {"none": 1}, sy="A"), tp
        assert rp["base_e_controls"][L] == {"equal": 187, "total": 187}, rp
    cc.exits(lambda: rd(new + [new[3] | {"success": not new[3]["success"]}]), 2)  # one key, other content

    # every branch of the rule, both of the slope's sentence
    branch = lambda w: rd(*world(lambda s, L, a, c: i[c] % 8 in w.get((L, a), W.get((s, L, a), ()))))  # noqa: E731
    G6 = {(L, "Gauto"): S8(0, 1, 2, 3, 4, 5) for L in CELLS}
    for w, b_ in (({**{(L, "Gauto"): S8(0, 1, 2) for L in CELLS}, **{(L, "none"): S8(0, 1, 7) for L in CELLS}}, "П0"),
                  ({**G6, **{(L, "none"): S8(0, 1) for L in CELLS}, **{(L, "GautoSham"): S8(0, 1, 2, 3, 4, 5)
                                                                       for L in CELLS}}, "П2"),
                  ({**G6, **{(L, "none"): S8(0, 1) for L in CELLS}, **{(L, "GautoSham"): S8(0, 1, 2, 3)
                                                                       for L in CELLS}}, "П3")):
        r = branch(w)
        assert r["rule"]["branch"] == b_ and r["rule"]["sentence"] == RULE[b_], (b_, r["rule"], r["measurements"])
        assert b_ != "П0" or r["measurements"]["pooled"]["G-N"]["diff_pp"] > 0  # П0 by its bound, not the estimate
    assert "numbers" in r["rule"] and r["rule"]["numbers"].startswith("Q = ") and "bound" not in r["rule"], r["rule"]
    # the bounds' edges, on estimates by hand: lo5(G - N) = 0 is no precondition (П0, also with a positive estimate);
    # hi95(Q) = 0 is no П1, one just below 0 inside the 1e-3 rounding is; lo5(Q) = 0 is П2 (equality passes); П1's
    # bound sentence; the slope's sentence by its raw hi95 alone
    e = lambda d, lo2, lo5, hi95, hi: {"diff_pp": d, "lo2.5": lo2, "lo5_one_sided": lo5,  # noqa: E731
                                       "hi95_one_sided": hi95, "hi97.5": hi}
    gn, pn = e(14.0, 6.0, 8.0, 20.0, 22.0), e(2.0, -1.0, 0.0, 3.44, 4.0)
    for g_, q_, b_ in ((e(14.0, -1.0, 0.0, 20.0, 22.0), e(-5.0, -9.0, -8.0, -1.0, -0.5), "П0"),
                       (e(3.0, -2.0, -1.0, 6.0, 7.0), e(-5.0, -9.0, -8.0, -1.0, -0.5), "П0"),
                       (gn, e(-5.0, -9.0, -8.0, 0.0, 1.0), "П3"), (gn, e(3.0, -1.0, 0.0, 6.0, 7.0), "П2"),
                       (gn, e(1.0, -1.0, -1e-6, 6.0, 7.0), "П3"), (gn, e(-5.0, -9.0, -8.0, -1e-6, 0.2), "П1"),
                       (gn, e(-1.0, -3.0, -2.0, -0.0004, 0.2), "П1"), (gn, e(-1.0, -3.0, -2.0, 0.0, 0.2), "П3"),
                       (e(14.0, -1.0, 1e-10, 20.0, 22.0), e(-5.0, -9.0, -8.0, -1.0, -0.5), "П0")):
        assert rule({"G-N": g_, "Q": q_, "P-N": pn})["branch"] == b_, (g_, q_, b_)
    assert rule({"G-N": gn, "Q": e(-5.0, -9.0, -8.0, -1.0, 0.2), "P-N": pn})["bound"] == (
        "фиктивная ветвь даёт над «нет» не более 3.4 п.п. при нашем выигрыше 14.0 п.п."), "П1's bound"
    x = e(-5.0, -9.0, -8.0, -0.0004, 0.5) | {"beta_pp_per_r": -5.0}
    assert slope_sentence(x) == SLOPE[True].format(-5.0, -9.0, 0.5) and slope_sentence(
        x | {"hi95_one_sided": 0.0, "hi97.5": -0.5}) == SLOPE[False], "the slope's sentence"
    near = lambda s, L, a, c: (phi(th[c])[1] < 1) if (s, a) == ("pod", "GautoSham") else win0(s, L, a, c)  # noqa
    r = rd(*world(near))  # P succeeds where the turn leaves less than the shift itself (phi < 60 deg)
    a2 = r["angle"]
    assert a2["hi95_one_sided"] < 0 and a2["sentence"] == SLOPE[True].format(
        a2["beta_pp_per_r"], a2["lo2.5"], a2["hi97.5"]) and a2["bands_P-N"]["[0, 60)"]["diff_pp"] > 0, a2

    # drops by reason: a skipped case, a missing record, an invalid one (prefix_ok false; a replay miss), per cell
    c1, c2, c3, c4 = plus[1], plus[150], plus[3], plus[4]  # c2: a P0nv case (the lines' 1 % apart)
    ed = lambda r: (r | {"skipped_reason": "x"} if (r["case_id"], r["d"]) == (c1, 0) else  # noqa: E731
                    r | {"prefix_ok": False} if (r["case_id"], r["d"], r["arm"]) == (c3, 0, "none2") else
                    r | {"replay_miss": 12} if (r["case_id"], r["d"], r["arm"]) == (c4, 10, "Gauto") else r)
    r = rd([ed(x) for x in new if (x["case_id"], x["d"], x["arm"], x["kind"]) != (c2, 0, "GautoSham", "dynamic")])
    assert r["checks"]["pass"] and r["checks"]["skipped"] == 1 and r["measurements"]["P0"]["drops"] == {
        "skipped": 1, "missing": 1, "invalid": 1} and r["measurements"]["P10"]["drops"] == {"invalid": 1}, r["checks"]
    assert r["measurements"]["pooled"]["drops"] == {"skipped": 1, "missing": 1, "invalid": 2}
    assert r["measurements"]["P0"]["Q"]["diff_pp"] == want("P0", *dw("Q"), out=[(c, "P0") for c in (c1, c2, c3)]) \
        and r["measurements"]["pooled"]["Q"]["diff_pp"] == want("P0+P10", *dw("Q"), out=[
            (c1, "P0"), (c2, "P0"), (c3, "P0"), (c4, "P10")]) and r["measurements"][
        "P0"]["Q"]["pairs"] == 184 and r["checks"]["invalid_pairs"]["P0"] == {"pairs": 185, "invalid": 1, "pass": True}
    assert [v["missing"] for v in r["checks"]["complete"].values()] == [4, 1, 0]  # c1's 4 skipped, c2's P missing

    # the checks before reading, each alone
    dyn = lambda L, a: [x for x in new if x["d"] == CELLS[L] and x["arm"] == a and x["kind"] == "dynamic"]  # noqa
    sub = lambda f, rs: [f(x) for x in rs]  # noqa: E731
    inv = lambda n: sub(lambda x: x | {"prefix_ok": False} if x in dyn("P10", "GautoSham")[:n] else x, new)  # noqa
    gone = lambda L, a, n: [x for x in new if x not in [y for y in dyn(L, a) if i[y["case_id"]] < 108][:n]]  # noqa
    # (the P0 line's cases: C2's first 108)
    same = lambda n: sub(lambda x: x | {"act_hash": f"pod/P0/{x['case_id']}/dynamic/none"} if x in dyn(
        "P0", "none2")[:n] else x, new)  # noqa: E731
    for why, rs, ok in (("complete", gone("P0", "Gauto", 5), False), ("complete", gone("P0", "Gauto", 4), True),
                        ("complete", [x for x in new if x["d"] != 10], False),  # a line not there at all
                        ("invalid_pairs", inv(4), False), ("invalid_pairs", inv(3), True),
                        ("conf", sub(lambda x: x | {"conf": x["conf"] | {"t_ramp": 4}} if x is new[7] else x, new),
                         False),
                        ("arms/kbar", sub(lambda x: x | {"conf": x["conf"] | {"kbar": 0.2}} if x["d"] == 10 else x,
                                          new), False),  # the whole cell: conf stays equal, only the kbar check fails
                        ("arms/theta", sub(lambda x: x | {"sham": {"theta": x["sham"]["theta"] + 1e-9}}
                                           if x is dyn("P0", "GautoSham")[5] else x, new), False),
                        ("arms/theta", sub(lambda x: {k: v for k, v in x.items() if k != "sham"}
                                           if x is dyn("P10", "GautoSham")[0] else x, new), False),
                        ("arms/seed_shift", sub(lambda x: x | {"seed_shift": 0} if x is dyn("P0", "none2")[9] else x,
                                                new), False),
                        ("arms/act_hash_N'_vs_N", same(94), False), ("arms/act_hash_N'_vs_N", same(93), True)):
        r = rd(rs)
        f, _, k = why.partition("/")
        c = {L: r["checks"][f][L][k] for L in CELLS} if k else r["checks"][f]
        flat = [v["pass"] for v in c.values()] if "pass" not in c else [c["pass"]]
        assert r["checks"]["pass"] == ok == ("rule" in r) == all(flat), (why, ok, c)
        if not ok:
            cc.exits(lambda: finish(r), 3)

    # the smoke's code check (spec §10): its 6 cases, the four arms and the cut Bases
    sm = [rec("pod", "P0", x[2], x[3], x[4], x[7], win0) for x in sorted(cc.lkeys(plan("Psmoke", split)))]
    ok, det = smoke_check(sm, split)
    assert ok and det["complete"] == {"expected": 30, "missing": 0, "off_plan": 0, "pass": True} and det["prefix"] == {
        "dynamic": 24, "invalid": 0, "pass": True}, det
    for bad in ([x for x in sm if x["arm"] != "none2" or x["case_id"] != plus[0]],
                [x | {"prefix_ok": False} if x["arm"] == "Gauto" and x["case_id"] == plus[5] else x for x in sm],
                [x | {"sham": {"theta": 1.0}} if x["arm"] == "GautoSham" and x["case_id"] == plus[2] else x
                 for x in sm],
                [{k: v for k, v in x.items() if k != "seed_shift"} for x in sm],
                [x | {"conf": x["conf"] | {"kbar": 0.2}} if x["arm"] == "GautoSham" else x for x in sm],  # P's own
                [x | {"act_hash": "same"} if x["arm"] in ("none", "none2") else x for x in sm]):
        assert not smoke_check(bad, split)[0]


def boot_by_hand(new, names, f, slope_=False):
    """The pooled bootstrap written out, for the selftest: per valid pair (both cells, the plan's records) its cluster
    and f(successes) (slope_: and r), clusters sorted, e2.N_BOOT draws of whole clusters from default_rng(0); the mean
    (or np.polyfit's slope per draw). Rounded as the reading; the rows and the raw hi95."""
    rows = []
    for r in new:
        if r["kind"] == "dynamic" and r["arm"] == "GautoSham":
            s = {SIDE[x["arm"]]: x["success"] for x in new if x["kind"] == "dynamic" and x["case_id"] == r["case_id"]
                 and x["d"] == r["d"]}
            cl = maxwrap.base_task({"case_id": r["case_id"], "task_suite_name": r["suite"],
                                    "task_name": r["task_name"], "substrate_variant": None}, names)
            rows.append((cl, phi(r["sham"]["theta"])[1], float(f(s))))
    keys = sorted({c for c, _, _ in rows})
    pick = np.random.default_rng(0).integers(len(keys), size=(e2.N_BOOT, len(keys)))
    by = {k: [(x, y) for c, x, y in rows if c == k] for k in keys}
    if slope_:
        st = [np.polyfit(*np.array([p for j in row for p in by[keys[j]]]).T, 1)[0] for row in pick]
        full = np.polyfit(*np.array([(x, y) for _, x, y in rows]).T, 1)[0]
        lo2, lo5, hi95, hi = np.percentile(st, [2.5, 5, 95, 97.5])
        return {"beta_pp_per_r": round(float(full), 3), "lo2.5": round(float(lo2), 3),
                "lo5_one_sided": round(float(lo5), 3), "hi95_one_sided": round(float(hi95), 3),
                "hi97.5": round(float(hi), 3)}, rows, float(hi95)
    st = [100 * sum(y for j in row for _, y in by[keys[j]]) / sum(len(by[keys[j]]) for j in row) for row in pick]
    lo2, lo5, hi95, hi = np.percentile(st, [2.5, 5, 95, 97.5])
    return {"diff_pp": round(100 * sum(y for *_, y in rows) / len(rows), 3), "lo2.5": round(float(lo2), 3),
            "lo5_one_sided": round(float(lo5), 3), "hi95_one_sided": round(float(hi95), 3),
            "hi97.5": round(float(hi), 3), "clusters": len(keys)}, rows, float(hi95)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--smoke-check", nargs="+", metavar="JSONL", help="psmoke's records: the code check, exit 0 or 3")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_P.jsonl: checks, rule, measurements")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--c", default=CTRL, help="the controls' records (read only): lines 2 and 3 among them")
    ap.add_argument("--p2", default=str(cc.P2), help="part 2's records (read only): pod_A's M1 and D files")
    ap.add_argument("--out", help="the reading: JSON too")
    a = ap.parse_args()
    a.read = a.read[a.read[:1] == ["read"]:]  # "read" before the files is optional
    if bool(a.read) + a.plan + a.selftest + bool(a.smoke_check) != 1:
        ap.error("one mode: --plan, --smoke-check, --selftest, or [read] JSONL...")
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    if a.selftest:
        selftest()
    elif a.plan:
        print(cc.tsv(plan(a.line)))
    elif a.smoke_check:
        ok, det = smoke_check([x for f in a.smoke_check for x in e2.load(f)])
        print(e2.dump(det))
        print(f"C1-E4 placebo smoke (code only): {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    else:
        finish(read([x for f in a.read for x in e2.load(f) if "kind" in x], e2.load(a.c), load_p2(a.p2),
                    cs.libero_names(), maxwrap.load_split(cs.SPLIT)), a.out)
