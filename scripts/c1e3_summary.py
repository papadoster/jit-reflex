"""C1-E3 verdicts: applies §8 of docs/superpowers/specs/2026-09-29-c1-e3-handoff-reflex-design.md to the JSON lines of
scripts/c1e3_run.py (no manual decisions) and builds the §9 report. Writes summary.json next to the input.
    python scripts/c1e3_summary.py results/c1-e3/grid_all.jsonl --baseline results/c1-e2/baseline.json
    python scripts/c1e3_summary.py results/c1-e3/grid_all.jsonl --baseline ... --override-gate "<journal entry>"
    python scripts/c1e3_summary.py --q3a results/c1-e3/grid_b.jsonl    the pod's gate: exit 0 if Q3a passes, else 3
    python scripts/c1e3_summary.py --selftest
If the read gate fails, only the gate is returned (results are not read), unless the owner's deviation, recorded in the
spec journal before reading, is passed with --override-gate. Helpers (load, agg, profile, nest, diff_pp, rnd, dump) are
C1-E2's.
"""

import argparse
import json
import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import numpy as np

import c1e2_summary as e2  # noqa: E402  (also puts src/ on sys.path)
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402

TOL, N_BOOT = e2.TOL, e2.N_BOOT
T26, T10 = orx.TASKS, hf.TASKS10
SUITES = ("libero_spatial", "libero_object", "libero_goal")
I10, I20, I30, I40, ALL, C56 = range(10), range(20), range(30), range(40), (0, 1, 2), (2,)
# spec §6 grid: (row, kind, cells, arms, alpha, inits, shift classes, tasks); row 6 went into row 3 (journal 2026-09-30);
# "25x" is row 25's B and F on inits 10-19, the part the §11 fuse trims first; row 2 at 0-39: Q1 on 0-29, Q1b on 0-39
ROWS = [("1", "step", "ACF", ["none"], 0.0, I20, ALL, T26),
        ("2", "step", "AB", ["G", "GR"], 0.0, I40, ALL, T26),
        ("3", "step", "CF", ["GR"], 0.0, I30, ALL, T26),
        ("4", "step", "AB", ["Gkeep"], 0.0, I10, ALL, T26),
        ("5", "step", "ABC", ["GRT"], 0.0, I30, C56, T26),
        ("7", "step", "ABC", ["GT"], 0.0, I20, C56, T26),
        ("8", "step", "ABC", ["GkeepT"], 0.0, I10, C56, T26),
        ("8", "step", "C", ["Gkeep"], 0.0, I10, C56, T26),
        ("8", "step", "C", ["G"], 0.0, I20, C56, T26),
        ("9", "step", "ACF", ["PPC"], 0.0, I10, ALL, T26),
        ("10", "step", "C", ["none", "T0"], 1.0, I20, ALL, T26),
        ("11", "step", "AB", ["G", "GR"], 1.0, I20, ALL, T26),
        ("12", "step", "A", ["Gkeep"], 1.0, I20, ALL, T26),
        ("13", "step", "BF", ["GRT", "T0"], 1.0, I20, ALL, T26),
        ("14", "step", "D", ["GRT", "T0"], 1.0, I10, ALL, T26),
        ("15", "step", "A", ["G", "Gkeep", "GR"], 0.5, I10, ALL, T26),
        ("16", "control", "ACF", ["none", "GR", "GR+surr"], 0.0, I10, ALL, T26),
        ("17", "control", "C", ["GR+contact"], 0.0, I10, ALL, T26),
        ("18", "close", "ACF", ["GR", "GR+surr"], 0.0, I10, ALL, T26),
        ("19", "close", "C", ["none", "GR+contact"], 0.0, I10, ALL, T26),
        ("20", "step", "ACF", ["GR@real"], 0.0, I20, ALL, T26),
        ("21", "control", "ACF", ["GR@real"], 0.0, I10, ALL, T26),
        ("22", "step", "ACF", ["PPC@real"], 0.0, I10, ALL, T26),
        ("23", "step", "C", [f"GR@{s}" for s in hf.SWEEP], 0.0, I10, ALL, T26),
        ("24", "step", "A", ["none", "G", "GR"], 0.0, I10, ALL, T10),
        ("25", "step", "BF", ["none"], 1.0, I10, ALL, T26),
        ("25", "step", "D", ["none"], 1.0, I10, ALL, T26),
        ("25x", "step", "BF", ["none"], 1.0, range(10, 20), ALL, T26),
        ("26", "step", "C", [f"PPC@{s}" for s in hf.SIGMA_SWEEP], 0.0, I10, ALL, T26),
        ("27", "step", "ACF", ["GRret"], 0.0, I30, ALL, T26),
        ("28", "control", "ACF", ["GRret"], 0.0, I10, ALL, T26)]
Q3A_ROW, HELD = "10", ("11", "12", "13", "14", "15", "25", "25x")  # spec §8: Q3a fails -> the others wait for the owner
REPORT_ROWS = ("4", "8", "9", "14", "15", "17", "19", "22", "23", "24", "25", "25x", "26")  # the only rows §11 may cut
# two pod lanes: rule rows first, report rows in reverse cut order (the first to cut runs last); lane B opens with Q3a;
# row 9 in lane B keeps the lanes even (journal 16)
LANES = {"A": "2 3 27 1 5 7 4 8 17 19 23 24 22 26".split(), "B": "10 11 12 13 16 21 28 18 20 25 14 25x 9 15".split()}


def row_keys(row):
    """The row's episodes as (kind, cell, arm, alpha, suite, task, init); the runner keeps control or the classes."""
    _, kind, cells, arms, alpha, inits, cls, tasks = row
    keep = [(s, t, i) for s, t in tasks for i in inits
            if kind == "control" or hf.make_episode(s, t, i, kind).mag_class in cls]
    return {(kind, c, a, alpha, *e) for c in cells for a in arms for e in keep}


EXPECTED = {row[0]: set() for row in ROWS}
for _row in ROWS:
    EXPECTED[_row[0]] |= row_keys(_row)
N_GRID = sum(map(len, EXPECTED.values()))
KEY = itemgetter("kind", "cell", "arm", "alpha", "suite", "task", "init")


def index(rs):
    idx = defaultdict(dict)
    for r in rs:
        idx[r["kind"], r["cell"], r["arm"], r["alpha"]][r["suite"], r["task"], r["init"]] = r
    return idx


def prs(idx, kind, alpha, cells, a, b, inits=None, cls=None, keep=None):
    """(task cluster, success a, success b) on the same episode seed, pooled over cells; inits, shift classes and
    keep(record of b) narrow the pairs."""
    out = []
    for c in cells:
        x, y = idx.get((kind, c, a, alpha), {}), idx.get((kind, c, b, alpha), {})
        for k in sorted(x.keys() & y.keys()):
            if ((inits is None or k[2] in inits) and (cls is None or y[k]["mag_class"] in cls)
                    and (keep is None or keep(y[k]))):
                out.append((k[:2], x[k]["success"], y[k]["success"]))
    return out


def early_prs(idx):
    """Q1b's pairs (journal 16): (GR, G) at alpha 0, step, A + B, inits 0-39, in the §8 early-chunk subset: the first
    plan observed after the shift arrives by t_fire + T_ramp. Chosen by the schedule and G's t_fire, as the §9 report
    (journal 17): pair noise or a pre-shift push can move G-R's t_fire; those pairs stay and are counted apart."""
    return prs(idx, "step", 0.0, AB, "GR", "G", I40, keep=early_rec)


def early_rec(r):
    """The record is in the §8 early-chunk subset (its own t_fire, cell and t_ramp)."""
    return bool(hf.early(r["t_fire"], *orx.CELLS[r["cell"]], r["conf"]["t_ramp"]))


def did(pf, pb):
    """Difference of paired differences, F - B, in pp; each bootstrap rep draws one set of task clusters for both."""
    keys = sorted({c for c, _, _ in pf + pb})

    def sums(p):
        s, n = defaultdict(float), defaultdict(int)
        for c, a, b in p:
            s[c] += float(a) - float(b)
            n[c] += 1
        return np.array([s[k] for k in keys]), np.array([n[k] for k in keys])

    if not (pf and pb):
        return None
    (sf, nf), (sb, nb) = sums(pf), sums(pb)
    pick = np.random.default_rng(0).integers(len(keys), size=(N_BOOT, len(keys)))
    with np.errstate(invalid="ignore", divide="ignore"):  # a rep without one side's clusters drops out (nan)
        boot = 100 * (sf[pick].sum(1) / nf[pick].sum(1) - sb[pick].sum(1) / nb[pick].sum(1))
    lo2, lo5, hi = np.nanpercentile(boot, [2.5, 5, 97.5])
    return {"diff_pp": float(100 * (sf.sum() / nf.sum() - sb.sum() / nb.sum())), "lo2.5": float(lo2),
            "lo5_one_sided": float(lo5), "hi97.5": float(hi), "pairs": len(pf) + len(pb), "clusters": len(keys)}


def sup(thr):  # "diff >= thr and lower > 0"
    return lambda x: x["diff_pp"] >= thr - TOL and x["lo2.5"] > 0


def ni(thr):  # one-sided lower 95% bound >= thr
    return lambda x: x["lo5_one_sided"] >= thr - TOL


ACF, AB, ABC = "ACF", "AB", "ABC"
# spec §8: rule -> (test, pairs(idx), predicate)
RULES = {
    "Q1": ("alpha 0, step, A + B, 0-29: one-sided (GR - G) >= -2",
           lambda i: prs(i, "step", 0.0, AB, "GR", "G", I30), ni(-2)),
    "Q1b": ("alpha 0, step, A + B, 0-39, early chunks (the first plan observed after the shift arrives by "
            "t_fire + T_ramp): (GR - G) >= +2 and lo2.5 > 0", early_prs, sup(2)),
    "Q2a": ("alpha 0, class 5-6 cm, A + B + C, 0-19: (GRT - GT) >= +5 and lo2.5 > 0",
            lambda i: prs(i, "step", 0.0, ABC, "GRT", "GT", I20, C56), sup(5)),
    "Q2b": ("alpha 0, class 5-6 cm, A + B + C, 0-29: one-sided (GRT - GR) >= -4",
            lambda i: prs(i, "step", 0.0, ABC, "GRT", "GR", I30, C56), ni(-4)),
    "Q3a": ("alpha 1, C, 0-19: (T0 - none) >= +4 and lo2.5 > 0",
            lambda i: prs(i, "step", 1.0, "C", "T0", "none"), sup(4)),
    "Q3b_gate": ("alpha 1, A, 0-19: (Gkeep - G) hi97.5 < 0",
                 lambda i: prs(i, "step", 1.0, "A", "Gkeep", "G"), lambda x: x["hi97.5"] < 0),
    "Q3b": ("alpha 1, A + B, 0-19: one-sided (GR - G) >= -3; read only if Q3b_gate passes",
            lambda i: prs(i, "step", 1.0, AB, "GR", "G"), ni(-3)),
    "Q3c": ("alpha 1, 0-19: ((GRT - T0)[F] - (GRT - T0)[B]) >= +5 and lo2.5 > 0 (same task draw per rep)", None, sup(5)),
    "Q4a": ("control, A + C + F, 0-9: one-sided (GR+surr - none) >= -1",
            lambda i: prs(i, "control", 0.0, ACF, "GR+surr", "none"), ni(-1)),
    "Q4b": ("close shift, A + C + F, 0-9: one-sided (GR+surr - GR) >= -2",
            lambda i: prs(i, "close", 0.0, ACF, "GR+surr", "GR"), ni(-2)),
    "Q5a": ("step, A + C + F, 0-19: (GR@real - none) >= +5 and lo2.5 > 0",
            lambda i: prs(i, "step", 0.0, ACF, "GR@real", "none"), sup(5)),
    "Q5b": ("control, A + C + F, 0-9: one-sided (GR@real - none) >= -2",
            lambda i: prs(i, "control", 0.0, ACF, "GR@real", "none"), ni(-2)),
    "Q6a": ("alpha 0, step, A + C + F, 0-29, all 26 tasks: (GRret - GR) >= +3 and lo2.5 > 0",
            lambda i: prs(i, "step", 0.0, ACF, "GRret", "GR", I30), sup(3)),
    "Q6b": ("control, A + C + F, 0-9: one-sided (GRret - GR) >= -2",
            lambda i: prs(i, "control", 0.0, ACF, "GRret", "GR"), ni(-2)),
}
VERDICTS = {"ПЕРЕДАЧА РАБОТАЕТ": ("Q1", "Q1b", "Q2a", "Q2b"),
            "ДЛЯ ЗРЯЧЕГО МОЗГА ТОЖЕ": ("Q3a", "Q3b_gate", "Q3b"),
            "ЦЕНА ЗАДЕРЖКИ ИЗМЕРЕНА": ("Q3a", "Q3c"),
            "ФИЛЬТР КОНТАКТА РАБОТАЕТ": ("Q4a", "Q4b"),
            "РЕАЛИСТИЧНЫЕ ГЛАЗА": ("Q5a", "Q5b"),
            "ПЕРЕНОС ПОСЛЕ ЗАХВАТА ЧИНИТСЯ": ("Q6a", "Q6b")}


def rule(name, idx):
    test, pairs, ok = RULES[name]
    d = (did(prs(idx, "step", 1.0, "F", "GRT", "T0"), prs(idx, "step", 1.0, "B", "GRT", "T0")) if name == "Q3c"
         else e2.diff_pp(pairs(idx)))
    res = {"test": test, **(e2.rnd(d) or {}), "pass": d is not None and bool(ok(d))}
    if name == "Q1b":  # journal 17, report only: G-R's t_fire differs inside Q1b; G and G-R on two sides of the cut
        pairs = [(x[k], y[k]) for c in AB for x, y in [(idx.get(("step", c, "GR", 0.0), {}), idx.get(("step", c, "G", 0.0), {}))]
                 for k in x.keys() & y.keys() if k[2] in I40]
        res["t_fire_mismatch"] = sum(early_rec(g) and gr["t_fire"] != g["t_fire"] for gr, g in pairs)
        res["early_differs"] = sum(early_rec(g) != early_rec(gr) for gr, g in pairs)
    return res


def q3a(raw):
    """The pod's gate after row 10 (spec §8): (pass, detail); an incomplete row 10 (> 1% missing) does not pass."""
    have = {KEY(r) for r in raw}
    miss = len(EXPECTED[Q3A_ROW] - have)
    res = rule("Q3a", index([r for r in raw if KEY(r) in EXPECTED[Q3A_ROW]]))
    ok = res["pass"] and miss <= 0.01 * len(EXPECTED[Q3A_ROW]) + TOL
    return ok, {"Q3a": res, "row10_missing": miss}


def runner_args(row):
    """The scripts/c1e3_run.py arguments that make one ROWS entry."""
    _, kind, cells, arms, alpha, inits, cls, tasks = row
    return (f"--kinds {kind} --cells {','.join(cells)} --methods {','.join(arms)} --alpha {alpha:g} "
            f"--inits {inits[0]}-{inits[-1]}" + (" --classes 2" if cls == C56 else "")
            + (" --tasks libero_10" if tasks is T10 else ""))


def plan(lane, cut=(), hold=False):
    """[(row, episodes, runner args)] of a pod lane in run order, without the cut rows (report rows only) and, with
    hold, without the alpha-brain rows the Q3a gate holds."""
    if bad := [r for r in cut if r not in REPORT_ROWS]:
        raise SystemExit(f"cut {bad}: not report rows (spec §11: rule rows are never cut; report rows {REPORT_ROWS})")
    cut = set(cut) | ({"25x"} if "25" in cut else set())
    return [(n, len(row_keys(row)), runner_args(row)) for n in LANES[lane] if n not in cut and not (hold and n in HELD)
            for row in ROWS if row[0] == n]


def summarize(raw, baseline, override=None):
    succ = defaultdict(set)
    for r in raw:
        succ[KEY(r)].add(r["success"])
    if conflict := [k for k, v in succ.items() if len(v) > 1]:
        raise SystemExit(f"{len(conflict)} episodes recorded twice with different success, e.g. {conflict[0]}")
    want = set().union(*EXPECTED.values())
    rs = list({KEY(r): r for r in raw if KEY(r) in want}.values())  # the §6 grid only, one record per episode
    conf = sorted({json.dumps(r.get("conf"), sort_keys=True) for r in rs})
    if len(conf) > 1:
        raise SystemExit(f"grid records mix settings (conf): {conf}")
    # rules and report on the 26 tasks; LIBERO-10 (row 24) only in its own report entry
    rs26, have = [r for r in rs if r["suite"] in SUITES], {KEY(r) for r in rs}
    idx, idx10 = index(rs26), index([r for r in rs if r["suite"] == "libero_10"])
    q3a_pass, q3a_det = q3a(rs)  # the pod's own gate, so the rows held here are the rows the pod held
    rows = {n: e for n, e in EXPECTED.items() if q3a_pass or n not in HELD}
    need = set().union(*rows.values())
    missing = {n: len(e - have) for n, e in rows.items() if e - have}
    ctrl = [r["success"] for r in idx["control", "A", "none", 0.0].values()]
    rate, base, n_miss = (float(np.mean(ctrl)) if ctrl else None), baseline.get("s10"), sum(missing.values())
    outside = sum(KEY(r) not in want for r in raw)
    gate = {"records": len(raw), "ignored_outside_grid": outside, "duplicates": len(raw) - outside - len(rs),
            "expected": len(need), "missing": n_miss, "missing_by_row": missing,
            "held_alpha_rows": [] if q3a_pass else list(HELD), "complete": n_miss <= 0.01 * len(need) + TOL,
            "control_A_none": rate, "control_A_none_n": len(ctrl), "baseline_s10": base,
            "baseline_ok": rate is not None and base is not None and abs(rate - base) <= 0.15 + TOL}
    gate["pass"] = gate["complete"] and gate["baseline_ok"]
    res = {"read_gate": gate}
    if not gate["pass"]:
        if not override:
            return res  # spec §8: results are not read
        gate["override"] = override  # a deviation recorded in the spec journal before reading

    rules = {n: rule(n, idx) for n in RULES}
    if rules["Q3a"]["pass"] and not q3a_pass:  # passes on what ran, but row 10 is > 1% incomplete: the pod held
        rules["Q3a"] |= {"pass": False, "note": f"строка 10 неполная (> 1%): не хватает {q3a_det['row10_missing']}"}
    if not rules["Q3a"]["pass"]:  # spec §8: the seeing brain is broken, Q3b and Q3c are not read
        for n in ("Q3b_gate", "Q3b", "Q3c"):
            rules[n] = {"test": RULES[n][0], "pass": None, "note": "не читается: Q3a не прошло"}
    elif not rules["Q3b_gate"]["pass"]:
        rules["Q3b"] |= {"pass": None, "note": "не читается: заслонка Q3b не прошла"}
    # each verdict needs all its rules; a rule not read (None) does not pass
    verdicts = {v: "ДА" if all(rules[n]["pass"] for n in need_rules) else "НЕТ" for v, need_rules in VERDICTS.items()}
    if "override" in gate:
        verdicts = {v: "ОТСТУПЛЕНИЕ (проверка перед чтением не пройдена, см. read_gate.override): " + x
                    for v, x in verdicts.items()}
    return res | {"verdicts": verdicts, "rules": rules, "conf": json.loads(conf[0]) if conf else None,
                  "report": report(rs26, idx, idx10, json.loads(conf[0]) if conf and conf[0] != "null" else {})}


# ---------------------------------------------------------------- §9 report (only reported, never a rule)

EPISODE = [(k, itemgetter(k)) for k in ("success", "grasp_ok", "steps_to_success", "calls_sched", "calls_trig",
                                        "path_m", "jerk", "grasp_miss_m", "grasp_miss_along_m")] + \
          [("no_close_command", lambda r: r["t_close"] is None)]
# journal item 3: an engagement counts only strictly before the first close (observe can set t_e at the close step)
ENGAGED = lambda r: r["t_engage"] is not None and (r["t_close"] is None or r["t_engage"] < r["t_close"])  # noqa: E731
SHARES = [("no_fire", lambda r: None if r["kind"] == "control" else r["t_fire"] is None),
          ("engaged_before_close", lambda r: ENGAGED(r) if r["method"] in hf.HAgent.FAMILY else None),
          ("pushed_before_fire", itemgetter("pushed_before_fire")),
          ("unstable_of_fired", lambda r: r["unstable"] if r["kind"] != "control" and r["t_fire"] is not None else None),
          ("contact_at_shift", itemgetter("contact_at_shift")),
          ("bad_qacc", lambda r: r["bad_qacc"] > 0)]
REL = lambda f: lambda r: None if r.get("release") is None else f(r["release"])  # noqa: E731
RELEASE = [("along_m", REL(itemgetter("along"))), ("abs_along_m", REL(lambda x: abs(x["along"]))),
           ("across_m", REL(itemgetter("across"))), ("dz_m", REL(itemgetter("dz"))),
           ("lifted", REL(itemgetter("lifted"))), ("at_success", REL(lambda x: x["at"] == "success")),
           ("success", itemgetter("success"))]


def d(p):
    return e2.rnd(e2.diff_pp(p))


def report(rs, idx, idx10, conf):
    arms = sorted({r["arm"] for r in rs})
    by = lambda x, dims, m, stat=np.mean: e2.nest(x, dims, lambda y: e2.profile(y, m, stat))  # noqa: E731
    kinds, cells, alphas = ("kind", ("step", "control", "close")), ("cell", tuple(orx.CELLS)), ("alpha", (0.0, 0.5, 1.0))
    a0 = [r for r in rs if r["alpha"] == 0.0]
    step0 = [r for r in a0 if r["kind"] == "step"]
    out = {
        "by_cell": by(rs, [kinds, alphas, cells, ("arm", arms)], EPISODE),
        "shares": by(rs, [kinds, alphas, ("arm", arms)], SHARES),
        "success_by_class": e2.nest(step0, [("mag_class", ALL), ("arm", arms)],
                                    lambda x: e2.agg(x, itemgetter("success"))),
        "success_by_suite": e2.nest(a0, [kinds, ("suite", SUITES), ("arm", arms)],
                                    lambda x: e2.agg(x, itemgetter("success"))),
    }
    # share of the lost success won back, inits 0-9 (control and shift paired by task and init)
    rec = {}
    for c in "ACF":
        ctrl, none = idx.get(("control", c, "none", 0.0), {}), idx.get(("step", c, "none", 0.0), {})
        for arm in ("GR", "PPC", "GR@real") + (("G",) if c == "A" else ()):
            m = idx.get(("step", c, arm, 0.0), {})
            ks = [k for k in m.keys() & none.keys() & ctrl.keys() if k[2] < 10]
            if ks:
                sm, sn, sc = (float(np.mean([x[k]["success"] for k in ks])) for x in (m, none, ctrl))
                rec[f"{c}/{arm}"] = [round((sm - sn) / (sc - sn), 3) if sc != sn else None, len(ks)]
    out["loss_recovered"] = rec
    out["kappa"] = kappa_report(rs, conf)
    # early chunks (spec §8 subsample): A and B, the first plan observed after the shift arrives by t_fire + T_ramp
    early = lambda r: hf.early(r["t_fire"], *orx.CELLS[r["cell"]], conf.get("t_ramp", 5))  # noqa: E731
    out["early_chunks"] = {
        name: {"GR-G": d(prs(idx, "step", 0.0, AB, "GR", "G", keep=f)),
               **{a: e2.profile([r for c in AB for r in idx.get(("step", c, a, 0.0), {}).values() if f(r)],
                                [EPISODE[0], EPISODE[-2]]) for a in ("G", "GR")}}
        for name, f in (("early", lambda r: bool(early(r))), ("rest", lambda r: r["t_fire"] is not None
                                                               and not early(r)))}
    out["who_is_right_A_0_9"] = {
        str(al): {a: e2.profile([r for k, r in idx.get(("step", "A", a, al), {}).items() if k[2] < 10],
                                [EPISODE[0], EPISODE[-2]]) for a in ("G", "Gkeep", "GR")} for al in (0.0, 0.5, 1.0)}
    out["t_harm_class_5_6"] = {"GT-G (0-19)": d(prs(idx, "step", 0.0, ABC, "GT", "G", I20, C56)),
                               "GkeepT-Gkeep (0-9)": d(prs(idx, "step", 0.0, ABC, "GkeepT", "Gkeep", I10, C56))}
    out["gr_vs_ppc"] = {"oracle ACF 0-9": d(prs(idx, "step", 0.0, ACF, "GR", "PPC", I10)),
                        "real ACF 0-9": d(prs(idx, "step", 0.0, ACF, "GR@real", "PPC@real", I10)),
                        **{f"{s} C 0-9": d(prs(idx, "step", 0.0, "C", f"GR@{s}", f"PPC@{s}", I10))
                           for s in hf.SIGMA_SWEEP}}
    ctrl_arms = ("GR", "GR+surr", "GR+contact", "GR@real", "GRret")
    marks = np.array([r["push_steps"] for r in rs if r["arm"] == "none" and r["kind"] != "step"
                      and r.get("push_steps")] or np.zeros((1, 3)))
    s_, c_, b_ = marks.sum(0)
    out["filter"] = {
        "false_engagement_control": e2.nest([r for r in rs if r["kind"] == "control"],
                                            [("cell", ("A", "C", "F")), ("arm", ctrl_arms)],
                                            lambda x: e2.agg(x, ENGAGED)),
        "surrogate_vs_contact_steps": {"precision": round(b_ / s_, 3) if s_ else None,
                                       "recall": round(b_ / c_, 3) if c_ else None, "steps": [int(s_), int(c_), int(b_)]},
        "close_shift": {"GR-none C": d(prs(idx, "close", 0.0, "C", "GR", "none")),
                        "GR+contact-none C": d(prs(idx, "close", 0.0, "C", "GR+contact", "none")),
                        "GR+surr-GR by cell": {c: d(prs(idx, "close", 0.0, c, "GR+surr", "GR")) for c in "ACF"}},
        "touch_after_shift": e2.nest([r for r in rs if r["arm"] == "none"], [kinds],
                                     lambda x: e2.agg(x, itemgetter("contact_at_shift"))),
    }
    sweep = ["GR"] + [f"GR@{s}" for s in hf.SWEEP] + ["GR@real"]
    out["noise"] = {"success_C_0_9": {a: e2.profile([r for k, r in idx.get(("step", "C", a, 0.0), {}).items()
                                                     if k[2] < 10], [EPISODE[0], ("jerk", itemgetter("jerk"))])
                                      for a in sweep}}
    out["delay_law_alpha1"] = delay_law(idx)
    out["t_for_seeing_brain_B"] = d(prs(idx, "step", 1.0, "B", "GRT", "GR"))
    out["libero_10"] = {"success": {a: e2.agg(idx10.get(("step", "A", a, 0.0), {}).values(), itemgetter("success"))
                                    for a in ("none", "G", "GR")},
                        **{f"{a}-{b}": d(prs(idx10, "step", 0.0, "A", a, b))
                           for a, b in (("G", "none"), ("GR", "none"), ("GR", "G"))}}
    rel = [r for r in rs if r.get("release")]
    out["release"] = {"all": by(rel, [kinds, alphas, ("arm", arms)], RELEASE),
                      "lifted": by([r for r in rel if r["release"]["lifted"]], [kinds, alphas, ("arm", arms)], RELEASE),
                      "step_alpha0_by_class": by([r for r in rel if r["kind"] == "step" and r["alpha"] == 0.0],
                                                 [("mag_class", ALL), ("arm", arms)], RELEASE),
                      "step_alpha0_by_suite": by([r for r in rel if r["kind"] == "step" and r["alpha"] == 0.0],
                                                 [("suite", SUITES), ("arm", arms)], RELEASE)}
    out["return"] = return_report(rs, idx)
    out["pair_noise"] = pair_noise(idx)
    return out


def kappa_report(rs, conf):
    """kappa_hat against kappa_model = alpha + (1 - alpha) (S . Delta^) / |Delta| and against alpha, per logged plan of
    the oracle G-R arms; shares of the dead zone, the fallback and the clips at 0 and 1."""
    K, tau, pts = str(conf.get("K", 20)), conf.get("tau_k", 0.01), []
    for r in rs:
        if r["arm"] not in ("GR", "GRT", "GRret") or r["kind"] == "control":  # control has no shift: its plans see pushes
            continue
        for k in r.get("kappa_log") or []:
            if k["nd"] < 1e-9:
                continue
            u, m = np.array(k["d"]) / k["nd"], k["m"][K]
            model = r["alpha"] + (1 - r["alpha"]) * float(np.array(k["S"]) @ u) / k["nd"]
            dead = abs(m) < tau
            pts.append({"cell": r["cell"], "alpha": r["alpha"], "mag_class": r["mag_class"],
                        "err_model": abs(k["k_hat"] - model), "err_alpha": abs(k["k_hat"] - r["alpha"]),
                        "dead": dead, "fb": k["fb"], "clip0": not dead and m / k["nd"] < 0,
                        "clip1": not dead and m / k["nd"] > 1})
    med = [(f, itemgetter(f)) for f in ("err_model", "err_alpha")]
    shares = [(f, itemgetter(f)) for f in ("dead", "fb", "clip0", "clip1")]
    al = ("alpha", (0.0, 0.5, 1.0))
    return {"median_err_by_cell": e2.nest(pts, [al, ("cell", tuple(orx.CELLS))], lambda x: e2.profile(x, med, np.median)),
            "median_err_by_class": e2.nest(pts, [al, ("mag_class", ALL)], lambda x: e2.profile(x, med, np.median)),
            "shares": e2.nest(pts, [al], lambda x: e2.profile(x, shares))}


def delay_law(idx):
    """alpha 1: (GRT - T0) and (T0 - none) by tau = t_close - t_fire of "none" on the same episode and cell, against
    d = 0 / 10 / 20 (B, D, F)."""
    buckets = {"<=10": lambda t: t is not None and t <= 10, "11-20": lambda t: t is not None and 10 < t <= 20,
               ">20": lambda t: t is not None and t > 20, "no close": lambda t: t is None}
    out = {}
    for c in "BDF":
        none = idx.get(("step", c, "none", 1.0), {})
        tau = {k: None if r["t_close"] is None else r["t_close"] - r["t_fire"]
               for k, r in none.items() if r["t_fire"] is not None}
        out[f"{c} d={orx.CELLS[c][1]}"] = {
            b: {"GRT-T0": d([p for k, p in zip(*keyed(idx, c, "GRT", "T0")) if k in tau and f(tau[k])]),
                "T0-none": d([p for k, p in zip(*keyed(idx, c, "T0", "none")) if k in tau and f(tau[k])])}
            for b, f in buckets.items()}
    return out


def keyed(idx, c, a, b):
    x, y = idx.get(("step", c, a, 1.0), {}), idx.get(("step", c, b, 1.0), {})
    ks = sorted(x.keys() & y.keys())
    return ks, [(k[:2], x[k]["success"], y[k]["success"]) for k in ks]


def return_report(rs, idx):
    """Q6a by suite and class; G-R's release offset along the shift against its accumulated extra C . u; the win by
    G-R's |offset along|; how often the return acted in control."""
    q6 = lambda keep: d(prs(idx, "step", 0.0, ACF, "GRret", "GR", I30, None, keep))  # noqa: E731
    gr = [r for c in ACF for r in idx.get(("step", c, "GR", 0.0), {}).values()
          if r.get("release") and r["release"]["lifted"] and r["t_fire"] is not None]
    x = np.array([np.array(r["c_extra"][:2]) @ [np.cos(r["angle"]), np.sin(r["angle"])] for r in gr])
    y = np.array([r["release"]["along"] for r in gr])
    link = ({"n": len(gr), "slope": round(float(np.polyfit(x, y, 1)[0]), 3), "corr": round(float(np.corrcoef(x, y)[0, 1]), 3)}
            if len(gr) > 2 and x.std() > 1e-9 and y.std() > 1e-9 else {"n": len(gr)})
    band = {"<1cm": (0, 0.01), "1-3cm": (0.01, 0.03), ">=3cm": (0.03, np.inf)}
    ctrl = [r for c in ACF for r in idx.get(("control", c, "GRret", 0.0), {}).values()]
    return {"Q6a_by_suite": {s: q6(lambda r, s=s: r["suite"] == s) for s in SUITES},
            "Q6a_by_class": {str(k): q6(lambda r, k=k: r["mag_class"] == k) for k in ALL},
            "GR_release_along_vs_C": link,
            "Q6a_by_GR_abs_along": {b: q6(lambda r, lo=lo, hi=hi: r.get("release") is not None
                                          and lo <= abs(r["release"]["along"]) < hi) for b, (lo, hi) in band.items()},
            "control_return_acted": e2.agg(ctrl, lambda r: r["t_lift"] is not None
                                           and float(np.linalg.norm(r["c_extra"])) > 1e-9)}


def pair_noise(idx):
    """Logically identical pairs (the reflex never acted) that differ in success, and their share of equal actions."""
    groups = [("control", c, a, "none", lambda r: r["t_engage"] is None) for c in ACF for a in ("GR", "GR+surr", "GR@real")] + \
             [("step", c, "GRret", "GR", lambda r: r["t_lift"] is None or float(np.linalg.norm(r["c_extra"])) <= 1e-9)
              for c in ACF]
    n = diff = same = 0
    for kind, c, a, b, quiet in groups:
        x, y = idx.get((kind, c, a, 0.0), {}), idx.get((kind, c, b, 0.0), {})
        for k in x.keys() & y.keys():
            if quiet(x[k]):
                n += 1
                diff += x[k]["success"] != y[k]["success"]
                same += x[k].get("act_hash") == y[k].get("act_hash")
    return {"pairs": n, "success_differs": round(diff / n, 4) if n else None,
            "actions_equal": round(same / n, 4) if n else None}


# ---------------------------------------------------------------- self-test

def selftest():
    assert N_GRID == 32190, N_GRID  # spec §6: ≈ 32.2k (journal 16: row 2 at 0-39)

    def grid(p, drop=()):
        """Every §6 episode; success when a fixed per-seed number u < p[(kind, alpha, cell, arm)] or p[(kind, alpha,
        arm)] or p[arm] (default 0.5; a callable gets (cell, t_fire)): a coupling that keeps each paired difference
        close to p_a - p_b."""
        rs = []
        for key in sorted(set().union(*EXPECTED.values())):
            kind, c, arm, al, s, t, i = key
            if key[:4] in drop:
                continue
            u = (hash_u(s, t, i, kind))
            pr = p.get((kind, al, c, arm), p.get((kind, al, arm), p.get(arm, 0.5)))
            tf = None if i % 11 == 10 else 30 + i
            ok = u < (pr(c, tf) if callable(pr) else pr)
            ep = hf.make_episode(s, t, i, kind)
            fam, cx = hf.parse_arm(arm)[0] in hf.HAgent.FAMILY, 0.01 * (1 + i % 3)
            rs.append({"suite": s, "task": t, "init": i, "kind": kind, "cell": c, "arm": arm, "alpha": al,
                       "method": hf.parse_arm(arm)[0], "success": ok, "mag_class": ep.mag_class, "angle": ep.angle,
                       "conf": {"K": 20, "tau_k": 0.01, "t_ramp": 5}, "grasp_ok": ok,
                       "t_fire": tf, "t_close": None if i % 7 == 6 else 40 + i + 3 * t,
                       "t_engage": 31 + i if fam and kind != "control" or (fam and i % 10 == 3) else None,
                       "t_lift": 60 if arm == "GRret" and i % 4 else None,
                       "c_extra": [cx * np.cos(ep.angle), cx * np.sin(ep.angle), 0] if fam else [0, 0, 0],
                       "kappa_log": [{"t": 40, "fb": i % 5 == 0, "m": {"10": 0.01, "20": 0.012, "30": 0.012},
                                      "nd": 0.03, "d": [0.03, 0, 0], "k_hat": 0.4, "k": 0.4, "S": [0.01, 0, 0]}]
                       if arm == "GR" else [],
                       "release": {"t": 90, "at": "open", "along": cx if fam else 0.002, "across": 0.001,
                                   "dz": 0.05, "lifted": i % 3 > 0} if i % 7 != 6 else None,
                       "push_steps": [3, 2, 2], "contact_at_shift": None if kind == "control" else i % 9 == 0,
                       "act_hash": "h" + str(i) if fam else "h" + str(i), "steps_to_success": 120 if ok else None,
                       "calls_sched": 10, "calls_trig": 0, "path_m": 1.0, "jerk": 0.5, "grasp_miss_m": 0.01,
                       "grasp_miss_along_m": 0.002, "pushed_before_fire": i == 3, "unstable": i == 2, "bad_qacc": 0})
        return rs

    x = lambda dd=0.0, lo2=-9.0, lo5=-9.0, hi=9.0: {"diff_pp": dd, "lo2.5": lo2, "lo5_one_sided": lo5, "hi97.5": hi}  # noqa: E731
    for n, t in (("Q1", -2), ("Q2b", -4), ("Q3b", -3), ("Q4a", -1), ("Q4b", -2), ("Q5b", -2), ("Q6b", -2)):
        assert RULES[n][2](x(lo5=t)) and not RULES[n][2](x(lo5=t - 0.01)), n
    for n, t in (("Q1b", 2), ("Q2a", 5), ("Q3a", 4), ("Q3c", 5), ("Q5a", 5), ("Q6a", 3)):
        assert [RULES[n][2](x(dd, lo2)) for dd, lo2 in ((t, -1), (t, 0.1), (t - 0.01, 0.1), (t, 0.0))] == \
               [False, True, False, False], n
    assert RULES["Q3b_gate"][2](x(hi=-0.01)) and not RULES["Q3b_gate"][2](x(hi=0.0))

    base = {"s10": 0.8}
    pg = {"none": 0.3, "T0": 0.3, "G": 0.6, "GT": 0.6, "Gkeep": 0.6, "GkeepT": 0.6, "GR": 0.6, "GRT": 0.7,
          "GR+surr": 0.6, "GR+contact": 0.6, "PPC": 0.5, "GR@real": 0.6, "PPC@real": 0.5, "GRret": 0.7,
          ("control", 0.0, "none"): 0.8, ("control", 0.0, "GR"): 0.8, ("control", 0.0, "GR+surr"): 0.8,
          ("control", 0.0, "GR+contact"): 0.8, ("control", 0.0, "GR@real"): 0.8, ("control", 0.0, "GRret"): 0.8,
          ("step", 0.0, "GT"): 0.45, ("step", 0.0, "GR"): 0.65,
          ("step", 1.0, "T0"): 0.5, ("step", 1.0, "Gkeep"): 0.4, ("step", 1.0, "B", "GRT"): 0.55,
          ("step", 1.0, "F", "GRT"): 0.7, ("step", 1.0, "F", "T0"): 0.4}
    good = grid(pg)
    res = summarize(good, base)
    g, rl = res["read_gate"], res["rules"]
    assert g["pass"] and g["missing"] == 0 and g["expected"] == len(good) == N_GRID, g
    assert all(v["pass"] for v in rl.values()), {k: (v.get("diff_pp"), v.get("lo2.5"), v["pass"]) for k, v in rl.items()}
    assert set(res["verdicts"].values()) == {"ДА"}, res["verdicts"]
    assert rl["Q1"]["pairs"] == 1560 and rl["Q3a"]["pairs"] == 520 and rl["Q6a"]["pairs"] == 2340, rl
    # Q1b: t_fire = 30 + i (none when i % 11 == 10), T_ramp 5. A (s 10): the next call after t_fire is <= 5 steps away
    # when i % 10 is 5-9, 20 of inits 0-39; B (s 25): only t_fire 45-49, inits 15-19. (20 + 5) x 26 tasks = 650
    assert rl["Q1b"]["pairs"] == 650, rl["Q1b"]
    # a pair outside the early subset never enters Q1b; Q1 keeps its pairs
    late = index([r for r in good if not hf.early(r["t_fire"], *orx.CELLS[r["cell"]], 5)])
    assert not early_prs(late) and prs(late, "step", 0.0, AB, "GR", "G", I30)
    assert rl["Q3c"]["pairs"] == 1040 and rl["Q3c"]["clusters"] == 26 and rl["Q6b"]["pairs"] == 780, rl["Q3c"]
    assert abs(rl["Q3c"]["diff_pp"] - (did_direct(good, "F") - did_direct(good, "B"))) < 1e-3  # rnd: 3 decimals
    pf = prs(index(good), "step", 1.0, "F", "GRT", "T0")  # one task draw for F and B: identical sets give exactly 0
    z = did(pf, pf)
    assert pf and z["lo2.5"] == z["hi97.5"] == 0, z
    rep = res["report"]
    assert rep["kappa"]["shares"][0.0]["fb"][0] == 0.2 and rep["libero_10"]["success"]["GR"][1] == 70, rep
    assert rep["kappa"]["shares"][0.0]["fb"][1] == sum(r["arm"] == "GR" and r["alpha"] == 0.0 and r["kind"] != "control"
                                                       and r["suite"] in SUITES for r in good), rep["kappa"]["shares"]
    assert not ENGAGED({"t_engage": 40, "t_close": 40}) and ENGAGED({"t_engage": 39, "t_close": 40}) \
        and ENGAGED({"t_engage": 5, "t_close": None}) and not ENGAGED({"t_engage": None, "t_close": 40})
    assert rep["return"]["GR_release_along_vs_C"]["slope"] == 1.0 and rep["pair_noise"]["pairs"] > 0, rep["return"]
    assert set(rep["delay_law_alpha1"]) == {"B d=0", "D d=10", "F d=20"}
    assert json.loads(e2.dump(res)) == json.loads(json.dumps(res))

    # Q3a fails: Q3b, Q3c not read; the held alpha rows may be missing without failing the gate
    held = {(k[:4]) for n in HELD for k in EXPECTED[n]}
    res = summarize(grid(pg | {("step", 1.0, "T0"): 0.3}, drop=held), base)
    assert res["read_gate"]["pass"] and res["read_gate"]["held_alpha_rows"] == list(HELD), res["read_gate"]
    assert res["rules"]["Q3b"]["pass"] is None and res["rules"]["Q3c"]["pass"] is None, res["rules"]
    assert res["verdicts"]["ЦЕНА ЗАДЕРЖКИ ИЗМЕРЕНА"] == "НЕТ" and res["verdicts"]["ПЕРЕДАЧА РАБОТАЕТ"] == "ДА"
    ok, det = q3a(grid(pg | {("step", 1.0, "T0"): 0.3}))
    assert not ok and det["row10_missing"] == 0, det
    assert q3a(good)[0] and not q3a([r for r in good if r["arm"] != "T0"])[0]
    # row 10 > 1% missing, Q3a passing on the rest: the pod's check fails, so the summary holds the same rows
    raw = [r for r in grid(pg, drop=held) if not (r["alpha"] == 1.0 and r["arm"] == "T0" and r["init"] == 0)]
    ok, det = q3a(raw)
    assert not ok and det["Q3a"]["pass"] and det["row10_missing"] == 26, det
    res = summarize(raw, base)
    assert res["read_gate"]["pass"] and res["read_gate"]["held_alpha_rows"] == list(HELD), res["read_gate"]
    assert res["read_gate"]["missing"] == 26 and res["rules"]["Q3a"]["pass"] is False, res["rules"]["Q3a"]
    assert all(res["rules"][n]["pass"] is None for n in ("Q3b_gate", "Q3b", "Q3c")), res["rules"]
    assert all(v["pass"] for n, v in res["rules"].items() if n[:2] != "Q3"), res["rules"]
    # gate Q3b fails: Q3b not read, verdict НЕТ; the others stay
    res = summarize(grid(pg | {("step", 1.0, "Gkeep"): 0.6}), base)
    assert res["rules"]["Q3b"]["pass"] is None and res["verdicts"]["ДЛЯ ЗРЯЧЕГО МОЗГА ТОЖЕ"] == "НЕТ", res["rules"]
    assert res["verdicts"]["ЦЕНА ЗАДЕРЖКИ ИЗМЕРЕНА"] == "ДА"
    # each rule fails alone (the coupling makes each paired difference exact)
    # Q1: G-R loses 15 pp on the late chunks, wins 5 on the early ones; Q1b: G-R = G
    q1 = lambda c, tf: 0.65 if hf.early(tf, *orx.CELLS[c], 5) else 0.45  # noqa: E731
    for p, fail in (({("step", 0.0, "GR"): q1}, "Q1"), ({("step", 0.0, "GR"): 0.6}, "Q1b"),
                    ({("step", 0.0, "GT"): 0.67}, "Q2a"),
                    ({("step", 0.0, "GRT"): 0.55}, "Q2b"), ({("step", 1.0, "F", "GRT"): 0.58, ("step", 1.0, "F", "T0"): 0.5}, "Q3c"),
                    ({("control", 0.0, "GR+surr"): 0.7}, "Q4a"), ({("close", 0.0, "GR+surr"): 0.5}, "Q4b"),
                    ({("step", 0.0, "GR@real"): 0.32}, "Q5a"), ({("control", 0.0, "GR@real"): 0.7}, "Q5b"),
                    ({("step", 0.0, "GRret"): 0.61}, "Q6a"), ({("control", 0.0, "GRret"): 0.7}, "Q6b")):
        res = summarize(grid(pg | p), base)
        assert {n for n, v in res["rules"].items() if not v["pass"]} == {fail}, (p, res["rules"])
        assert [v for v, need in VERDICTS.items() if res["verdicts"][v] == "НЕТ"] == \
               [v for v, need in VERDICTS.items() if fail in need], res["verdicts"]

    # gate: the baseline at exactly ±15 pp passes, beyond fails; 1.1% missing fails, 0.9% passes; override recorded
    rate = summarize(good, base)["read_gate"]["control_A_none"]
    assert summarize(good, {"s10": rate - 0.15})["read_gate"]["baseline_ok"]
    assert set(summarize(good, {"s10": rate + 0.1501})) == {"read_gate"}
    drop90, drop110 = [r for j, r in enumerate(good) if j % 90], [r for j, r in enumerate(good) if j % 110]
    assert not summarize(drop90, base)["read_gate"]["pass"] and summarize(drop110, base)["read_gate"]["pass"]
    note = "journal 2026-10-01: см. [ §8 ]"
    res = summarize(drop90, base, note)
    assert res["read_gate"]["override"] == note and all(v.startswith("ОТСТУПЛЕНИЕ") for v in res["verdicts"].values())
    j = next(j for j, r in enumerate(good) if KEY(r)[:4] == ("step", "B", "GR", 0.0) and r["t_fire"] is not None
             and early_rec(r) and early_rec(dict(r, t_fire=r["t_fire"] + 1)))  # in Q1b, stays early when moved
    # journal 17: a G-R record whose t_fire differs from G's stays in Q1b (chosen by G's t_fire) and is counted
    moved = good[:j] + good[j + 1:] + [dict(good[j], t_fire=good[j]["t_fire"] + 1)]
    q1b = summarize(moved, base)["rules"]["Q1b"]
    assert q1b["t_fire_mismatch"] == 1 and q1b["early_differs"] == 0, q1b
    assert q1b["pairs"] == summarize(good, base)["rules"]["Q1b"]["pairs"], q1b
    late = good[:j] + good[j + 1:] + [dict(good[j], t_fire=None)]  # G-R never fired: outside the cut, G still in it
    assert summarize(late, base)["rules"]["Q1b"]["early_differs"] == 1
    # integrity stops
    for raw, msg in ((good + [dict(good[0], success=not good[0]["success"])], "different success"),
                     (good[1:] + [dict(good[0], conf={"K": 10})], "mix settings")):
        try:
            summarize(raw, base)
            raise AssertionError(msg)
        except SystemExit as e:
            assert msg in str(e), e
    # the pod plan makes exactly the §6 grid, each episode once (a copy of the runner's episode enumeration)
    def runner_keys(args):
        ap = argparse.ArgumentParser()
        for f, dflt in (("--tasks", "all"), ("--kinds", "step"), ("--cells", ""), ("--methods", ""), ("--inits", ""),
                        ("--classes", "0,1,2")):
            ap.add_argument(f, default=dflt)
        ap.add_argument("--alpha", type=float, default=0.0)
        x = ap.parse_args(args.split())
        lo, _, hi = x.inits.partition("-")
        tasks, cls = (T10 if x.tasks == "libero_10" else T26), {int(c) for c in x.classes.split(",")}
        return {(k, c, m, x.alpha, s, t, i) for s, t in tasks for k in x.kinds.split(",") for c in x.cells.split(",")
                for m in x.methods.split(",") for i in range(int(lo), int(hi or lo) + 1)
                if k == "control" or hf.make_episode(s, t, i, k).mag_class in cls}
    lines = plan("A") + plan("B")
    made = [runner_keys(a) for _, _, a in lines]
    assert sum(map(len, made)) == len(set().union(*made)) == N_GRID and set().union(*made) == set().union(*EXPECTED.values())
    assert all(len(m) == n for m, (_, n, _) in zip(made, lines)) and {r for r, _, _ in lines} == set(EXPECTED)
    na, nb = (sum(n for _, n, _ in plan(x)) for x in "AB")
    assert (na, nb) == (16070, 16120) and plan("B")[0][0] == Q3A_ROW, (na, nb)
    assert {r for r, _, _ in plan("B", hold=True)} == set(LANES["B"]) - set(HELD)
    assert {r for r, _, _ in plan("B", cut=["25"])} == set(LANES["B"]) - {"25", "25x"}
    try:
        plan("A", cut=["26", "2"])
        raise AssertionError("a rule row was cut")
    except SystemExit as e:
        assert "rule rows are never cut" in str(e)
    print("selftest ok")


def hash_u(s, t, i, kind):
    """A fixed number in [0, 1) per episode, shared by all arms and cells (the selftest's coupling)."""
    return ((i + 3 * t + 7 * SUITES.index(s) if s in SUITES else i + 3 * t) * 0.618034 + 0.1 * (kind == "close")) % 1


def did_direct(rs, c):
    x = {(r["suite"], r["task"], r["init"]): r["success"] for r in rs if (r["cell"], r["arm"], r["alpha"]) == (c, "GRT", 1.0)}
    y = {(r["suite"], r["task"], r["init"]): r["success"] for r in rs if (r["cell"], r["arm"], r["alpha"]) == (c, "T0", 1.0)}
    return 100 * float(np.mean([x[k] - y[k] for k in x.keys() & y.keys()]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="?", help="JSON lines of scripts/c1e3_run.py")
    ap.add_argument("--baseline", help='JSON {"s1": rate, "s10": rate}: C1-E2\'s, or this pod\'s stock lerobot-eval')
    ap.add_argument("--override-gate", help="the spec journal entry that records the owner's deviation")
    ap.add_argument("--q3a", nargs="+", metavar="JSONL", help="row 10's records: exit 0 if Q3a passes, else 3")
    ap.add_argument("--plan", choices=("A", "B"), help='the pod lane\'s runner lines: "row<TAB>episodes<TAB>args"')
    ap.add_argument("--cut", default="", help='with --plan: report rows cut by the §11 fuse, e.g. "26 22"')
    ap.add_argument("--hold", action="store_true", help="with --plan: without the alpha rows Q3a holds")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
        raise SystemExit
    if a.plan:
        for row, n, args in plan(a.plan, a.cut.split(), a.hold):
            print(f"{row}\t{n}\t{args}")
        raise SystemExit
    if a.q3a:
        ok, det = q3a([r for f in a.q3a for r in e2.load(f)])
        print(json.dumps(det, ensure_ascii=False))
        raise SystemExit(0 if ok else 3)
    if not (a.grid and a.baseline):
        ap.error("grid and --baseline are required")
    src = Path(a.grid)
    text = e2.dump(summarize(e2.load(src), json.loads(Path(a.baseline).read_text()), a.override_gate))
    (src.parent / "summary.json").write_text(text + "\n")
    print(text)
