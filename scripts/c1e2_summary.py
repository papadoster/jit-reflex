"""C1-E2 verdict: applies section 8 of docs/superpowers/specs/2026-09-28-c1-e2-object-reflex-design.md to the JSON
lines of scripts/c1e2_run.py (no manual decisions) and builds the section 9 report. Writes summary.json next to the
input.
    python scripts/c1e2_summary.py results/c1-e2/grid.jsonl --baseline results/c1-e2/baseline.json
    python scripts/c1e2_summary.py results/c1-e2/grid.jsonl --baseline ... --override-gate "<journal entry>"
    python scripts/c1e2_summary.py --selftest
If the read gate fails, only the gate is returned (results are not read), unless the owner's deviation, recorded in the
spec journal before reading, is passed with --override-gate.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import objreflex as orx  # noqa: E402

CELLS, METHODS, KINDS = list(orx.CELLS), orx.Agent.METHODS, ("step", "control", "smooth")
SUITES = tuple(dict.fromkeys(s for s, _ in orx.TASKS))
N_BOOT, TOL = 2000, 1e-9  # TOL: equality to a threshold passes (spec §8) despite float rounding
# spec §6 episode table: (kind, cell, method, init states) x 26 tasks; the coefficient pilot (inits 44-47) is not in it
EXPECTED = ([("step", c, m, range(40 if (c, m) in (("A", "none"), ("C", "GT")) else 10))  # R3 extension: inits 10-39
             for c in CELLS for m in ("none", "T0", "G", "GT", "PPC")]
            + [("control", c, m, range(40, 44)) for c in CELLS for m in ("none", "G", "T0", "PPC")]
            + [("step", "C", "Gpost", range(10)), ("step", "A", "GJ", range(4))]
            + [("smooth", "C", m, range(4)) for m in ("none", "G", "T0", "PPC", "PPC9")])

# per-episode report metrics (spec §9); a field that does not apply is None and is skipped
EPISODE = [(k, itemgetter(k)) for k in ("success", "grasp_ok", "steps_to_success", "calls_sched", "calls_trig",
                                        "path_m", "jerk", "grasp_miss_m", "grasp_miss_along_m")] + \
          [("no_close_command", lambda r: r["t_close"] is None)]
SHARES = [
    ("no_fire", lambda r: None if r["kind"] == "control" else r["t_fire"] is None),
    ("fire_at_step_0", lambda r: None if r["kind"] == "control" else r["t_fire"] == 0),
    # smooth increments k = 0..19 run at t_fire + k unless a close command came earlier: cut when t_close < t_fire + 19
    ("smooth_cut_by_close", lambda r: None if r["kind"] != "smooth" or r["t_fire"] is None
     else r["t_close"] is not None and r["t_close"] - r["t_fire"] < orx.SMOOTH_STEPS - 1),
    ("pushed_before_fire", itemgetter("pushed_before_fire")),
    ("false_trigger", lambda r: (r["g_on"] or r["calls_trig"] > 0) if r["kind"] == "control" else None),
    ("unstable_of_fired", lambda r: r["unstable"] if r["kind"] == "step" and r["t_fire"] is not None else None),
    ("bad_qacc", lambda r: r["bad_qacc"] > 0),
]
# spec §8 predicates on diff_pp output; "> 0" is strict
RULES = {"R1": lambda x: x["diff_pp"] >= 5 - TOL and x["lo2.5"] > 0,
         "R2": lambda x: x["lo5_one_sided"] >= -2 - TOL,
         "R3": lambda x: x["lo5_one_sided"] >= -5 - TOL,
         "R4": lambda x: x["diff_pp"] >= 5 - TOL and x["lo2.5"] > 0}


def r5_category(lo):
    return None if lo is None else "ЛУЧШЕ" if lo > 0 else "НЕ ХУЖЕ" if lo >= -2 - TOL else "ХУЖЕ"


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def agg(rs, f, stat=np.mean):
    """[stat, n] over f(r) where it applies: a missing field (KeyError) or None skips the record."""
    x = []
    for r in rs:
        try:
            v = f(r)
        except KeyError:
            continue
        if v is not None:
            x.append(float(v))
    return [round(float(stat(x)), 4), len(x)] if x else None


def profile(rs, metrics, stat=np.mean):
    return {name: v for name, f in metrics if (v := agg(rs, f, stat))}


def nest(rs, dims, leaf):
    """{v1: {v2: ... leaf(records)}} over dims = [(field, values), ...]; empty entries dropped."""
    if not dims:
        return leaf(rs)
    (key, vals), rest = dims[0], dims[1:]
    out = {v: nest([r for r in rs if r.get(key) == v], rest, leaf) for v in vals}
    return {v: x for v, x in out.items() if x}


def pairs(idx, kind, ab):
    """(task cluster, success a, success b) on the same episode seed; ab = [((cell, method), (cell, method)), ...]."""
    out = []
    for a, b in ab:
        x, y = idx[(kind, *a)], idx[(kind, *b)]
        out += [(k[:2], x[k]["success"], y[k]["success"]) for k in x.keys() & y.keys()]
    return out


def same(cells, a, b):
    return [((c, a), (c, b)) for c in cells]


def diff_pp(prs):
    """Paired success difference a - b in pp, pooled; percentile bootstrap over the 26 task clusters (spec §8)."""
    if not prs:
        return None
    cl = defaultdict(list)
    for c, a, b in prs:
        cl[c].append(float(a) - float(b))
    keys = sorted(cl)  # fixed cluster order: the bootstrap is reproducible
    s, n = np.array([sum(cl[k]) for k in keys]), np.array([len(cl[k]) for k in keys])
    pick = np.random.default_rng(0).integers(len(keys), size=(N_BOOT, len(keys)))
    lo2, lo5, hi = np.percentile(100 * s[pick].sum(1) / n[pick].sum(1), [2.5, 5, 97.5])
    return {"diff_pp": float(100 * s.sum() / n.sum()), "lo2.5": float(lo2), "lo5_one_sided": float(lo5),
            "hi97.5": float(hi), "pairs": int(n.sum()), "clusters": len(keys)}


def rnd(d):
    return d and {k: round(v, 3) if isinstance(v, float) else v for k, v in d.items()}


def versus(idx, cell, a, b):
    """A report-only method against G on their common episodes (step shift)."""
    x, y = idx["step", cell, a], idx["step", cell, b]
    common = sorted(x.keys() & y.keys())
    return {f"{a}-{b}": rnd(diff_pp(pairs(idx, "step", [((cell, a), (cell, b))]))),
            a: profile([x[k] for k in common], EPISODE), b: profile([y[k] for k in common], EPISODE)}


def summarize(raw, baseline, override=None):
    key = lambda r: (r["kind"], r["cell"], r["method"], r["suite"], r["task"], r["init"])  # noqa: E731
    succ = defaultdict(set)
    for r in raw:
        succ[key(r)].add(r["success"])
    if conflict := [k for k, v in succ.items() if len(v) > 1]:
        raise SystemExit(f"{len(conflict)} episodes recorded twice with different success, e.g. {conflict[0]}")
    have, want, missing = {key(r) for r in raw}, set(), {}
    for k, c, m, inits in EXPECTED:
        group = {(k, c, m, s, t, i) for s, t in orx.TASKS for i in inits}
        want |= group
        if n := len(group - have):
            missing[f"{k}/{c}/{m}"] = n
    rs = list({key(r): r for r in raw if key(r) in want}.values())  # the §6 grid only, one record per episode
    coef = sorted({(r.get("t_ramp"), r.get("k_p")) for r in rs}, key=str)
    if len(coef) > 1:
        raise SystemExit(f"grid records mix coefficients (t_ramp, k_p): {coef}")
    idx = defaultdict(dict)
    for r in rs:
        idx[r["kind"], r["cell"], r["method"]][r["suite"], r["task"], r["init"]] = r
    outside = sum(key(r) not in want for r in raw)
    ctrl = [r["success"] for r in idx["control", "A", "none"].values()]
    rate, base, n_miss = (float(np.mean(ctrl)) if ctrl else None), baseline.get("s10"), sum(missing.values())
    gate = {"records": len(raw), "ignored_outside_grid": outside, "duplicates": len(raw) - outside - len(rs),
            "expected": len(want), "missing": n_miss, "missing_by_group": missing,
            "complete": n_miss <= 0.01 * len(want) + TOL,
            "control_A_none": rate, "control_A_none_n": len(ctrl), "baseline_s10": base,
            "baseline_ok": rate is not None and base is not None and abs(rate - base) <= 0.15 + TOL}
    gate["pass"] = gate["complete"] and gate["baseline_ok"]
    res = {"read_gate": gate}
    if not gate["pass"]:
        if not override:
            return res  # spec §8: results are not read
        gate["override"] = override  # a deviation recorded in the spec journal before reading

    d = {"R1": diff_pp(pairs(idx, "step", same(CELLS, "G", "none"))),
         "R2": diff_pp(pairs(idx, "control", same(CELLS, "G", "none"))),
         "R3": diff_pp(pairs(idx, "step", [(("C", "GT"), ("A", "none"))])),
         "R4": diff_pp(pairs(idx, "step", same(["F", "Gp"], "GT", "T0"))),
         "R5": diff_pp(pairs(idx, "step", same(CELLS, "GT", "PPC")))}
    test = {"R1": "step, 7 cells: G - none >= +5 pp and lo2.5 > 0",
            "R2": "control, 7 cells: G - none, one-sided lo5 >= -2 pp",
            "R3": "step: C-GT - A-none (inits 0-39), one-sided lo5 >= -5 pp",
            "R4": "step, F and Gp: GT - T0 >= +5 pp and lo2.5 > 0",
            "R5": "step, 7 cells: GT - PPC (our implementation); lo2.5 > 0 ЛУЧШЕ, >= -2 pp НЕ ХУЖЕ, else ХУЖЕ"}
    rules = {r: {"test": test[r], **rnd(d[r] or {}), "pass": d[r] is not None and RULES[r](d[r])} for r in RULES}
    rules["R5"] = {"test": test["R5"], **rnd(d["R5"] or {}), "category": r5_category(d["R5"] and d["R5"]["lo2.5"])}
    rules["R3"]["rates"] = {"C-GT": agg(idx["step", "C", "GT"].values(), itemgetter("success")),
                            "A-none": agg(idx["step", "A", "none"].values(), itemgetter("success"))}
    p = {r: v.get("pass") for r, v in rules.items()}
    verdict = ("НЕТ ДЕМО: рефлекс по геометрии в этой постановке не работает (R1 не прошёл)" if not p["R1"]
               else "НЕТ ДЕМО: R2 (без вреда) не прошёл" if not p["R2"]
               else "ДЕМО ЕСТЬ, ГЛАВНАЯ ОСЬ ПОДТВЕРЖДЕНА" if p["R3"] and p["R4"] else "ДЕМО ЕСТЬ")
    if "override" in gate:
        verdict = "ОТСТУПЛЕНИЕ (проверка перед чтением не пройдена, см. read_gate.override): " + verdict
    res |= {"verdict": verdict, "rules": rules, "t_ramp_k_p": coef, "report": report(rs, idx)}
    return res


def report(rs, idx):
    core = [r for r in rs if not (r["kind"] == "step" and r["init"] >= 10)]  # inits 10-39 only feed R3
    step = [r for r in core if r["kind"] == "step"]
    hr = [dict(r, stage="near" if r["headroom"]["dist_m"] <= 0.10 else "far")  # spec §9: cells A and C, method none
          for r in step if r.get("headroom") and r["cell"] in ("A", "C")]
    ratio = [(h, lambda r, h=h: r["headroom"]["ratio"][h]) for h in ("10", "25", "50")]
    stage = ("stage", ("near", "far"))
    success = lambda x: agg(x, itemgetter("success"))  # noqa: E731
    return {
        # smooth set (spec §9): by_cell.smooth.C.<method>, plus shares.smooth.<method>.smooth_cut_by_close
        "by_cell": nest(core, [("kind", KINDS), ("cell", CELLS), ("method", METHODS)], lambda x: profile(x, EPISODE)),
        "success_by_class": nest(step, [("mag_class", (0, 1, 2)), ("method", METHODS)], success),
        "success_by_suite": nest(core, [("kind", KINDS), ("suite", SUITES), ("method", METHODS)], success),
        "gt_minus_t0_by_d": {dd: rnd(diff_pp(pairs(idx, "step", same(cs, "GT", "T0"))))
                             for dd, cs in ((0, ["A", "B", "C"]), (10, ["D", "E"]), (20, ["F", "Gp"]))},
        "shares": nest(core, [("kind", KINDS), ("method", METHODS)], lambda x: profile(x, SHARES)),
        "headroom_median": nest(hr, [stage], lambda x: profile(x, ratio, np.median)),
        "headroom_median_by_class": nest(hr, [stage, ("mag_class", (0, 1, 2))], lambda x: profile(x, ratio, np.median)),
        "headroom_median_by_cell": nest(hr, [("cell", ("A", "C")), stage], lambda x: profile(x, ratio, np.median)),
        "headroom_G": "≈ 1 after T_ramp by construction (spec §9); not measured",
        "gpost_vs_g_C": versus(idx, "C", "Gpost", "G"),
        "gj_vs_g_A": versus(idx, "A", "GJ", "G"),
    }


def dump(res):
    """JSON with innermost lists ([value, n]) on one line; real newlines only, so string values stay intact."""
    text = json.dumps(res, indent=1, ensure_ascii=False)
    return re.sub(r"\[\n\s+([^\[\]{}]*?)\n\s+\]", lambda m: "[" + re.sub(r"\n\s+", " ", m.group(1)) + "]", text)


def selftest():
    def grid(p):
        """Every §6 episode; success when a fixed per-seed number u < p[(kind, cell, method)] or p[(kind, method)]
        (default 0.5): a coupling that keeps each paired difference close to p_a - p_b."""
        rs = []
        for k, c, m, inits in EXPECTED:
            for s, t in orx.TASKS:
                for i in inits:
                    ok = ((i + 3 * t) * 0.618034) % 1 < p.get((k, c, m), p.get((k, m), 0.5))
                    rs.append({"suite": s, "task": t, "init": i, "kind": k, "cell": c, "method": m, "success": ok,
                               "mag_class": i % 3 if k == "step" else -1, "grasp_ok": ok,
                               "t_fire": None if k == "control" or i % 7 == 6 else i % 4,
                               "t_close": 10 + i if i % 5 else None, "steps_to_success": 120 if ok else None,
                               "calls_sched": 10, "calls_trig": int(m in ("T0", "GT")), "g_on": m == "G" and i == 41,
                               "path_m": 1.0, "jerk": 0.5, "grasp_miss_m": 0.01, "grasp_miss_along_m": 0.002,
                               "pushed_before_fire": i == 3, "unstable": i == 2, "bad_qacc": 0,
                               "headroom": {"ratio": {"10": 0.03, "25": 0.1, "50": 0.17}, "dist_m": 0.06 + 0.01 * i}
                               if k == "step" and m == "none" and i < 10 else None})
        return rs

    # rule thresholds, directly: equality passes, beyond fails, "> 0" strict
    x = lambda d=0.0, lo2=-9.0, lo5=-9.0: {"diff_pp": d, "lo2.5": lo2, "lo5_one_sided": lo5}  # noqa: E731
    for r, t in (("R2", -2.0), ("R3", -5.0)):
        assert RULES[r](x(lo5=t)) and not RULES[r](x(lo5=t - 0.01)), r
    for r in ("R1", "R4"):
        assert [RULES[r](x(d, lo2)) for d, lo2 in ((5.0, -9.0), (5.0, 0.1), (4.99, 0.1), (5.0, 0.0))] == \
               [False, True, False, False], r
    assert [r5_category(lo) for lo in (-2.0, -2.01, 0.0, 0.01)] == ["НЕ ХУЖЕ", "ХУЖЕ", "НЕ ХУЖЕ", "ЛУЧШЕ"]

    base = {"s10": 0.5}
    pg = {("step", "none"): 0.3, ("step", "T0"): 0.35, ("step", "PPC"): 0.4, ("step", "G"): 0.6, ("step", "GT"): 0.6}
    good = grid(pg)
    res = summarize(good, base)
    g, rl = res["read_gate"], res["rules"]
    assert g["pass"] and g["missing"] == 0 and g["expected"] == len(good) == 14456, g
    assert all(rl[r]["pass"] for r in ("R1", "R2", "R3", "R4")) and rl["R5"]["category"] == "ЛУЧШЕ", rl
    assert [rl[r]["pairs"] for r in ("R1", "R2", "R3", "R4", "R5")] == [1820, 728, 1040, 520, 1820], rl
    assert res["verdict"] == "ДЕМО ЕСТЬ, ГЛАВНАЯ ОСЬ ПОДТВЕРЖДЕНА", res["verdict"]
    rep = res["report"]
    assert rep["headroom_median"]["near"]["10"][0] == 0.03 and "PPC9" in rep["by_cell"]["smooth"]["C"], rep
    assert set(rep["headroom_median_by_cell"]) == {"A", "C"}, rep["headroom_median_by_cell"]
    json.dumps(res)

    # the other verdict branches: only R2 fails; only R4 fails; only R3 fails
    for p, fails, verdict in (({("control", "G"): 0.4}, {"R2"}, "НЕТ ДЕМО: R2 (без вреда) не прошёл"),
                              ({("step", "F", "GT"): 0.35, ("step", "Gp", "GT"): 0.35}, {"R4"}, "ДЕМО ЕСТЬ"),
                              ({("step", "C", "GT"): 0.2}, {"R3"}, "ДЕМО ЕСТЬ")):
        res = summarize(grid(pg | p), base)
        assert {r for r in RULES if not res["rules"][r]["pass"]} == fails and res["verdict"] == verdict, res["rules"]

    bad = grid({("step", "none"): 0.5, ("step", "A", "none"): 0.65, ("step", "G"): 0.53, ("step", "GT"): 0.45,
                ("step", "T0"): 0.45, ("step", "PPC"): 0.45, ("control", "G"): 0.4})
    res = summarize(bad, base)
    rl = res["rules"]
    assert not any(rl[r]["pass"] for r in ("R1", "R2", "R3", "R4")) and rl["R5"]["category"] == "НЕ ХУЖЕ", rl
    assert res["verdict"].startswith("НЕТ ДЕМО") and "R1" in res["verdict"], res["verdict"]

    # R1's interval arm: +5.8 pp carried by 3 of 26 tasks -> the lower 2.5% bound is 0, not > 0; R5 ХУЖЕ
    few = grid({("step", "GT"): 0.45, ("step", "PPC"): 0.6})
    for r in few:
        if (r["kind"], r["method"]) == ("step", "G") and (r["suite"], r["task"]) in orx.TASKS[:3]:
            r["success"] = True
    rl = summarize(few, base)["rules"]
    assert rl["R1"]["diff_pp"] >= 5 and rl["R1"]["lo2.5"] <= 0 and not rl["R1"]["pass"], rl["R1"]
    assert rl["R5"]["category"] == "ХУЖЕ", rl["R5"]

    # gate, baseline: equality to ±15 pp passes, beyond fails; gate failure returns only the gate
    rate = summarize(good, base)["read_gate"]["control_A_none"]
    assert summarize(good, {"s10": rate - 0.15})["read_gate"]["baseline_ok"]
    assert set(summarize(good, {"s10": rate + 0.1501})) == {"read_gate"}
    assert set(summarize(good, {})) == {"read_gate"}
    # gate, completeness: 1.1% missing fails, 0.9% passes; an override is recorded and the rules are read
    drop90, drop110 = [r for j, r in enumerate(good) if j % 90], [r for j, r in enumerate(good) if j % 110]
    g = summarize(drop90, base)["read_gate"]
    assert not g["complete"] and not g["pass"] and g["missing"] == len(good) - len(drop90), g
    assert summarize(drop110, base)["read_gate"]["pass"]
    note = "journal 2026-09-29: см. [ §8 ,   п. 2 ]"
    res = summarize(drop90, base, note)
    assert res["read_gate"]["override"] == note and "rules" in res and res["verdict"] == \
        "ОТСТУПЛЕНИЕ (проверка перед чтением не пройдена, см. read_gate.override): ДЕМО ЕСТЬ, ГЛАВНАЯ ОСЬ ПОДТВЕРЖДЕНА"
    assert json.loads(dump(res)) == json.loads(json.dumps(res))  # compaction leaves strings (the note) intact

    # integrity stops: the same episode with another success; mixed coefficients
    for raw, msg in ((good + [dict(good[0], success=not good[0]["success"])], "different success"),
                     (good[1:] + [dict(good[0], t_ramp=8)], "mix coefficients")):
        try:
            summarize(raw, base)
            raise AssertionError(msg)
        except SystemExit as e:
            assert msg in str(e), e

    # report on records with only the required fields, plus one duplicate and one record outside the grid
    need = ("suite", "task", "init", "kind", "cell", "method", "success")
    bare = [{k: r[k] for k in need} for r in good]
    res = summarize(bare + [dict(bare[0]), dict(bare[0], init=44)], base)
    assert res["read_gate"]["duplicates"] == 1 and res["read_gate"]["ignored_outside_grid"] == 1, res["read_gate"]
    assert res["rules"]["R1"]["pass"] and res["report"]["headroom_median"] == {}
    json.dumps(res)
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="?", help="JSON lines of scripts/c1e2_run.py")
    ap.add_argument("--baseline", help='JSON {"s1": rate, "s10": rate} from the pod\'s stock lerobot-eval')
    ap.add_argument("--override-gate", help="the spec journal entry that records the owner's deviation")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        raise SystemExit
    if not (a.grid and a.baseline):
        ap.error("grid and --baseline are required")
    src = Path(a.grid)
    text = dump(summarize(load(src), json.loads(Path(a.baseline).read_text()), a.override_gate))
    (src.parent / "summary.json").write_text(text + "\n")
    print(text)
