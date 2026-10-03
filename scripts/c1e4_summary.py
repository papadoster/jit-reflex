"""C1-E4 part 1: the grid (§6), the pod's gates and trial choices (§7, §8) and the verdicts, applied mechanically to the JSON
lines of scripts/c1e4_run.py (spec docs/superpowers/specs/2026-10-03-c1-e4-design.md). Pure numpy, no torch/LIBERO.
    python scripts/c1e4_summary.py --plan [--cut "12 16"] [--hold]     the pod's runner lines: phase, lane, id, n, args
    python scripts/c1e4_summary.py --control results/c1-e4/pod/grid_pi05.jsonl  pi control check: exit 0, or 3
    python scripts/c1e4_summary.py --q0 results/c1-e4/pod/grid_pi05.jsonl       the Q0 gate: exit 0, or 3
    python scripts/c1e4_summary.py --precision FP32_S FP32_ENVS BF16_S BF16_FITS BITWISE   the §4.1 rule (smoke)
    python scripts/c1e4_summary.py --trial P1 P2 P3 --choice precision_choice.txt         the §7 choices and journal line
    python scripts/c1e4_summary.py --forecast PI_S S_S                 hours and dollars, RTX 4090 vs a 48 GB card
    python scripts/c1e4_summary.py results/c1-e4/pod/grid_all.jsonl --baseline results/c1-e2/baseline.json
    python scripts/c1e4_summary.py --selftest
If the read gate fails only the gate is returned, unless the owner's deviation, journaled in the spec before reading, is
passed with --override-gate. Bootstrap and rule helpers are C1-E2's and C1-E3's; tau_kappa is C1-E3's pilot choice.
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from operator import itemgetter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e2_summary as e2  # noqa: E402  (also puts src/ on sys.path)
import c1e3_pilot as pilot  # noqa: E402
import c1e3_summary as s3  # noqa: E402
import gauto  # noqa: E402
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402

TOL = e2.TOL
P, S = "pi05", "smolvla"
I10, I20, I30, I40, ALL, C56 = range(10), range(20), range(30), range(40), (0, 1, 2), (2,)
DELAY = ("A20", "A40")
# spec §6: id -> (row, brain, kind, cells, arms, inits, shift classes); one id is one runner line. Suffixes split a row:
# "d" its A10 / A20 / A40 part, "a" / "b" its halves on the two SmolVLA lanes
GRID = {
    "6": ("6", P, "control", ("A",), ("none", "G@glr"), I10, ALL),
    "1": ("1", P, "step", ("C",), ("none", "T0"), I40, ALL),
    "2": ("2", P, "step", ("A",), ("none", "G", "Gkeep", "Gk0", "Gauto"), I30, ALL),
    "3": ("3", P, "step", ("A", "C"), ("GT", "GautoT"), I40, C56),
    "5": ("5", P, "step", ("A",), ("G@glr",), I20, ALL),
    "4": ("4", P, "step", DELAY, ("none", "G"), I20, ALL),
    "5d": ("5", P, "step", DELAY, ("G@glr",), I20, ALL),
    "6d": ("6", P, "control", DELAY, ("none", "G@glr"), I10, ALL),
    "7a": ("7", S, "step", ("A",), ("none", "G", "Gkeep"), I30, ALL),
    "7b": ("7", S, "step", ("A",), ("Gk0", "Gauto"), I30, ALL),
    "8": ("8", S, "control", ("A",), ("none",), I10, ALL),
    "17": ("17", P, "step", ("A10",), ("none", "G"), I20, ALL),
    "13": ("13", P, "control", ("C",), ("none",), I10, ALL),
    "14a": ("14", S, "step", ("A20",), ("none", "G"), I20, ALL),
    "14b": ("14", S, "step", ("A40",), ("none", "G"), I20, ALL),
    "9": ("9", P, "step", ("A", "C"), ("Gk0T",), I20, C56),
    "10": ("10", P, "step", ("A",), ("GR",), I20, ALL),
    "15a": ("15", S, "step", ("A",), ("GT", "GautoT"), I20, C56),
    "15b": ("15", S, "step", ("C",), ("GT", "GautoT"), I20, C56),
    "11": ("11", P, "step", ("A",), ("T0", "PPC"), I10, ALL),
    "11d": ("11", P, "step", DELAY, ("T0", "PPC"), I10, ALL),
    "16a": ("16", S, "step", ("C",), ("none",), I20, ALL),
    "16b": ("16", S, "step", ("C",), ("T0",), I20, ALL),
    "12": ("12", P, "step", ("A",), ("G@ema",), I10, ALL),
    "12d": ("12", P, "step", DELAY, ("G@ema",), I10, ALL),
}
# spec §12 order: phases one after another, the lanes of a phase in parallel, a lane's ids in turn. pi0.5 runs alone
# (lane p), SmolVLA in two lanes (a, b). "control" ends with the pi control check, "q0" with the Q0 gate; report rows
# last, in reverse cut order (the first to cut runs last)
PHASES = [("control", {"p": ["6"]}), ("q0", {"p": ["1"]}), ("pi", {"p": ["2", "3", "5", "4", "5d", "6d"]}),
          ("smolvla", {"a": ["7a"], "b": ["7b", "8"]}), ("r1", {"p": ["17", "13"]}), ("r2", {"a": ["14a"], "b": ["14b"]}),
          ("r3", {"p": ["9", "10"]}), ("r4", {"a": ["15a"], "b": ["15b"]}), ("r5", {"p": ["11", "11d"]}),
          ("r6", {"a": ["16a"], "b": ["16b"]}), ("r7", {"p": ["12", "12d"]})]
LANE_BRAIN = {"p": P, "a": S, "b": S}
HELD = ("4", "5d", "6d", "17", "14a", "14b", "11d", "12d")  # spec §8: Q0 fails -> the delay-axis ids do not run
CUT_ORDER = ("12", "16", "11", "15", "10", "9", "14", "13", "17")  # spec §12 fuse: report rows only
RULE_ROWS = ("1", "2", "3", "4", "5", "6", "7", "8")
N_CAL, N_TRIAL, CONTROL_MIN, FLOOR = 30, 104, 0.90, 0.50  # spec §5.3, §7, §8
PRICE_4090, CARD48 = (0.6, 0.7), {"price": (1.3, 1.5), "pi_speed": 2.0}  # spec §4.1, §12: $/h and a 48 GB card (guess)

assert {i for _, lanes in PHASES for ids in lanes.values() for i in ids} == set(GRID)
assert all(GRID[i][1] == LANE_BRAIN[ln] for _, lanes in PHASES for ln, ids in lanes.items() for i in ids)


def row_keys(gid):
    """The id's episodes as (brain, kind, cell, arm, suite, task, init); the runner keeps control or the classes."""
    _, brain, kind, cells, arms, inits, cls = GRID[gid]
    keep = [(s, t, i) for s, t in orx.TASKS for i in inits
            if kind == "control" or hf.make_episode(s, t, i, kind, prefix="c1e4").mag_class in cls]
    return {(brain, kind, c, a, *e) for c in cells for a in arms for e in keep}


EXPECTED = {gid: row_keys(gid) for gid in GRID}
KEY = itemgetter("brain", "kind", "cell", "arm", "suite", "task", "init")


def index(rs):
    idx = defaultdict(dict)
    for r in rs:
        idx[r["brain"], r["kind"], r["cell"], r["arm"]][r["suite"], r["task"], r["init"]] = r
    return idx


def prs(idx, brain, kind, cells, a, b, inits, cls=None):
    """(task cluster, success a, success b) on the same episode seed, pooled over cells, on inits (and classes)."""
    out = []
    for c in cells:
        x, y = idx.get((brain, kind, c, a), {}), idx.get((brain, kind, c, b), {})
        for k in sorted(x.keys() & y.keys()):
            if k[2] in inits and (cls is None or y[k]["mag_class"] in cls):
                out.append((k[:2], x[k]["success"], y[k]["success"]))
    return out


def rate(idx, brain, kind, cell, arm):
    xs = [r["success"] for r in idx.get((brain, kind, cell, arm), {}).values()]
    return (float(np.mean(xs)) if xs else None), len(xs)


# spec §8: rule -> (test, pairs(idx, delay cells), predicate); "delay cells" is A20 + A40, or A20 under the control floor
RULES = {
    "Q0": ("pi05, step, C, 0-39: (T0 - none) >= +4 and lo2.5 > 0",
           lambda i, dc: prs(i, P, "step", ("C",), "T0", "none", I40), s3.sup(4)),
    "Q1": ("pi05, step: (G - none)[A20 + A40, 0-19] - (G - none)[A, 0-29] >= +5 and lo2.5 > 0 (same task draw per rep)",
           None, s3.sup(5)),
    "Q2b": ("pi05, step, A, 0-29: one-sided (Gauto - Gk0) >= -2",
            lambda i, dc: prs(i, P, "step", ("A",), "Gauto", "Gk0", I30), s3.ni(-2)),
    "Q2c": ("pi05, step, A, 0-29: (Gauto - Gkeep) >= +3 and lo2.5 > 0",
            lambda i, dc: prs(i, P, "step", ("A",), "Gauto", "Gkeep", I30), s3.sup(3)),
    "Q2d": ("pi05, step, class 5-6 cm, A + C, 0-39: (GautoT - GT) >= +5 and lo2.5 > 0",
            lambda i, dc: prs(i, P, "step", ("A", "C"), "GautoT", "GT", I40, C56), s3.sup(5)),
    "Q2e": ("pi05, step, A, 0-29: one-sided (Gauto - G) >= -3",
            lambda i, dc: prs(i, P, "step", ("A",), "Gauto", "G", I30), s3.ni(-3)),
    "Q2f": ("smolvla, step, A, 0-29: one-sided (Gauto - G) >= -3",
            lambda i, dc: prs(i, S, "step", ("A",), "Gauto", "G", I30), s3.ni(-3)),
    "Q3a": ("pi05, step, A + A20 + A40, 0-19: (G@glr - none) >= +4 and lo2.5 > 0",
            lambda i, dc: prs(i, P, "step", ("A", *dc), "G@glr", "none", I20), s3.sup(4)),
    "Q3b": ("pi05, control, A + A20 + A40, 0-9: one-sided (G@glr - none) >= -2",
            lambda i, dc: prs(i, P, "control", ("A", *dc), "G@glr", "none", I10), s3.ni(-2)),
}
Q2A = "|kbar_30 - kappa0| <= 0.10 for both brains (kbar_30 frozen in the trial, read from the records' conf)"
VERDICTS = {"МОЗГ САМ ОТРАБАТЫВАЕТ СДВИГ": ("Q0",),
            "ОСЬ ЗАДЕРЖКИ НА НАСТОЯЩЕЙ VLA": ("Q0", "Q1"),
            "САМОНАСТРОЙКА РАБОТАЕТ": ("Q2a", "Q2b", "Q2c", "Q2d", "Q2e", "Q2f"),
            "РЕАЛИСТИЧНЫЕ ГЛАЗА НА ЗРЯЧЕМ МОЗГЕ": ("Q3a", "Q3b")}


def rule(name, idx, dc=DELAY):
    test, pairs, ok = RULES[name]
    d = (s3.did(prs(idx, P, "step", dc, "G", "none", I20), prs(idx, P, "step", ("A",), "G", "none", I30))
         if name == "Q1" else e2.diff_pp(pairs(idx, dc)))
    return {"test": test, **(e2.rnd(d) or {}), "pass": d is not None and bool(ok(d))}


def q2a(rs):
    out = {"test": Q2A}
    for b in (P, S):
        kb = {r["conf"]["kbar"] for r in rs if r["brain"] == b}
        out[b] = {"kbar_30": sorted(kb, key=str), "kappa0": gauto.KAPPA0[b]}
    ok = all(len(out[b]["kbar_30"]) == 1 and out[b]["kbar_30"][0] is not None
             and abs(out[b]["kbar_30"][0] - gauto.KAPPA0[b]) <= 0.10 + TOL for b in (P, S))
    return out | {"pass": ok}


def gate_row(raw, gid, pred):
    """A pod gate on one id: (pass, detail); an incomplete id (> 1% missing) does not pass."""
    have = {KEY(r) for r in raw}
    miss = len(EXPECTED[gid] - have)
    rs = list({KEY(r): r for r in raw if KEY(r) in EXPECTED[gid]}.values())
    ok, det = pred(index(rs))
    return ok and miss <= 0.01 * len(EXPECTED[gid]) + TOL, det | {f"id{gid}_missing": miss}


def control(raw):
    """spec §8: the pi control check before Q0, success of "none" in control A (id 6, 0-9) >= 90%."""
    def pred(idx):
        x, n = rate(idx, P, "control", "A", "none")
        return x is not None and x >= CONTROL_MIN - TOL, {"control_A_none": x, "n": n, "min": CONTROL_MIN}
    return gate_row(raw, "6", pred)


def q0(raw):
    def pred(idx):
        res = rule("Q0", idx)
        return res["pass"], {"Q0": res}
    return gate_row(raw, "1", pred)


def plan(cut=(), hold=False):
    """[(phase, lane, id, episodes, runner args)] in run order, without the cut report rows and, with hold, without the
    delay-axis ids the Q0 gate holds."""
    if bad := [r for r in cut if r not in CUT_ORDER]:
        raise SystemExit(f"cut {bad}: not report rows (spec §12: rule rows are never cut; report rows {CUT_ORDER})")
    out = []
    for ph, lanes in PHASES:
        for ln, ids in lanes.items():
            for gid in ids:
                row, brain, kind, cells, arms, inits, cls = GRID[gid]
                if row in cut or (hold and gid in HELD):
                    continue
                args = (f"--brain {brain} --kinds {kind} --cells {','.join(cells)} --methods {','.join(arms)} "
                        f"--inits {inits[0]}-{inits[-1]}" + (" --classes 2" if cls == C56 else ""))
                out.append((ph, ln, gid, len(EXPECTED[gid]), args))
    return out


def precision(fp32_s, fp32_envs, bf16_s, bf16_fits, bitwise):
    """spec §4.1: bf16 one row per call if its tempo (full batches) is >= 1.3 x fp32's, its repeated call is bitwise
    identical and it fits (peak <= 23 GB at 10 envs); else fp32. Tempos as seconds per episode (None: no full batch)."""
    ok = (fp32_s is not None and bf16_s is not None and bool(bitwise) and bool(bf16_fits)
          and fp32_s >= 1.3 * bf16_s - TOL)
    return ("bf16-row", 10) if ok else ("fp32", fp32_envs)


def kbar30(recs):
    """spec §5.3: the median of the first N_CAL shadow samples in round-robin order over the tasks (init 40 of the 26
    tasks in orx.TASKS order, then 41, ...); a shift without a sample is skipped. (kbar_30, samples used, all samples)"""
    order = sorted(recs, key=lambda r: (r["init"], orx.TASKS.index((r["suite"], r["task"]))))
    xs = [r["shadow"]["sample"] for r in order if r.get("shadow") and r["shadow"]["sample"] is not None]
    if not xs:
        raise SystemExit("!!! no shadow sample in the calibration block")
    return float(np.median(xs[:N_CAL])), min(len(xs), N_CAL), xs


def trial_rows(p1, p2, p3):
    want = {"P1": (p1, P, "control", "none"), "P2": (p2, P, "step", "Gcal"), "P3": (p3, S, "step", "Gcal")}
    out = {}
    for k, (rs, brain, kind, arm) in want.items():
        rs = [r for r in rs if (r["brain"], r["kind"], r["cell"], r["arm"]) == (brain, kind, "A", arm)]
        if len({(r["suite"], r["task"], r["init"]) for r in rs}) != N_TRIAL:
            raise SystemExit(f"!!! trial row {k}: {len(rs)} records, expected {N_TRIAL} distinct episodes. The owner "
                             "decides, e.g. rerun the trial: it resumes and retries failed batches")
        out[k] = rs
    return out


def trial(p1, p2, p3, choice):
    """spec §7: tau_kappa (P95 jitter of pi0.5's plans after the pseudo-shift, C1-E3's choice 1, K = 20), kbar_30 of both
    brains, the precision and envs of the smoke; the report numbers; the journal line."""
    m = re.search(r"C1-E4 precision: precision=(fp32|bf16-row) envs=(\d+)", choice)
    if not m:
        raise SystemExit("!!! no precision choice: run the smoke first")
    rows = trial_rows(p1, p2, p3)
    tau, n_tau = pilot.tau_kappa(rows["P1"])
    kb = {b: kbar30(rows[k]) for b, k in ((P, "P2"), (S, "P3"))}
    jit = []  # spec §7 item 5: the shadow pair's jitter without a shift, along the episode's pseudo direction
    for r in rows["P1"]:
        if r.get("shadow"):
            u = np.array([np.cos(r["angle"]), np.sin(r["angle"]), 0.0])
            mag = hf.make_episode(r["suite"], r["task"], r["init"], "step", prefix="c1e4").mag
            jit.append((abs(float(np.array(r["shadow"]["diff"]) @ u)), mag))
    j = np.array(jit) if jit else np.zeros((0, 2))
    q = lambda v: {k: round(float(np.percentile(v, p)), 5) for k, p in  # noqa: E731
                   (("median", 50), ("p75", 75), ("p90", 90), ("p95", 95), ("max", 100))} if len(v) else None
    line = (f"C1-E4 trial: precision={m[1]} tau_k={tau[gauto.K]:.6f} kbar_pi05={kb[P][0]:.4f} "
            f"kbar_smolvla={kb[S][0]:.4f} N={N_CAL} envs={m[2]} pad=0")
    rep = {"tau_k_plans": n_tau, "tau_by_K": tau,
           "kbar": {b: {"kbar_30": v[0], "samples_used": v[1], "samples": len(v[2]), "kbar_all": float(np.median(v[2])),
                        "kappa0": gauto.KAPPA0[b]} for b, v in kb.items()},
           "shadow_jitter_m": q(j[:, 0]), "shadow_jitter_share": q(j[:, 0] / j[:, 1]), "shadow_pairs": len(j)}
    return line, rep


def forecast(pi_s, s_s, cut=(), hold=False):
    """spec §4.1, §12: the grid in hours and dollars on an RTX 4090 (pi0.5 one lane, SmolVLA two) and on a 48 GB card
    (pi0.5 two lanes, CARD48 guesses); rates in seconds per episode per lane from the trial."""
    n = defaultdict(int)
    for _, ln, _, k, _ in plan(cut, hold):
        n[LANE_BRAIN[ln]] += k
    h = {"4090": (pi_s * n[P] + s_s * n[S] / 2) / 3600,
         "48GB": (pi_s * n[P] / CARD48["pi_speed"] + s_s * n[S] / 2) / 3600}
    usd = {"4090": [round(h["4090"] * x, 1) for x in PRICE_4090],
           "48GB": [round(h["48GB"] * x * y, 1) for x, y in zip(PRICE_4090, CARD48["price"])]}
    cheaper = np.mean(usd["48GB"]) < np.mean(usd["4090"])
    return {"episodes": dict(n), "hours": {k: round(v, 1) for k, v in h.items()}, "usd": usd,
            "card48_cheaper": bool(cheaper), "fuse_32h": h["4090"] > 32}


def summarize(raw, baseline, override=None):
    succ = defaultdict(set)
    for r in raw:
        succ[KEY(r)].add(r["success"])
    if conflict := [k for k, v in succ.items() if len(v) > 1]:
        raise SystemExit(f"{len(conflict)} episodes recorded twice with different success, e.g. {conflict[0]}")
    want = set().union(*EXPECTED.values())
    rs = list({KEY(r): r for r in raw if KEY(r) in want}.values())  # the §6 grid only, one record per episode
    for b in (P, S):
        conf = sorted({json.dumps(r["conf"], sort_keys=True) for r in rs if r["brain"] == b})
        if len(conf) > 1:
            raise SystemExit(f"{b} records mix settings (conf): {conf}")
    idx, have = index(rs), {KEY(r) for r in rs}
    c_ok, c_det = control(rs)
    q0_ok, q0_det = q0(rs)  # the pod's own gate, so the ids held here are the ids the pod held
    ids = {g: e for g, e in EXPECTED.items() if q0_ok or g not in HELD}
    need = set().union(*ids.values())
    missing = {g: len(e - have) for g, e in ids.items() if e - have}
    s_rate, s_n = rate(idx, S, "control", "A", "none")
    base, n_miss = baseline.get("s10"), sum(missing.values())
    gate = {"records": len(raw), "ignored_outside_grid": sum(KEY(r) not in want for r in raw),
            "expected": len(need), "missing": n_miss, "missing_by_id": missing,
            "held_delay_ids": [] if q0_ok else list(HELD), "complete": n_miss <= 0.01 * len(need) + TOL,
            "control_pi": c_det | {"pass": c_ok}, "control_smolvla_A_none": s_rate, "control_smolvla_n": s_n,
            "baseline_s10": base,
            "baseline_ok": s_rate is not None and base is not None and abs(s_rate - base) <= 0.15 + TOL}
    gate["pass"] = gate["complete"] and gate["baseline_ok"] and c_ok
    res = {"read_gate": gate}
    if not gate["pass"]:
        if not override:
            return res  # spec §8: results are not read
        gate["override"] = override  # a deviation recorded in the spec journal before reading
    a40, n40 = rate(idx, P, "control", "A40", "none")
    dc = DELAY if a40 is None or a40 >= FLOOR - TOL else ("A20",)  # spec §8: the control floor of Q1
    rules = {n: rule(n, idx, dc) for n in RULES} | {"Q2a": q2a(rs)}
    rules["Q1"]["delay_cells"] = list(dc)
    rules["Q1"]["control_A40_none"] = a40
    if rules["Q0"]["pass"] and not q0_ok:  # passes on what ran, but id 1 is > 1% incomplete: the pod held
        rules["Q0"] |= {"pass": False, "note": f"строка 1 неполная (> 1%): не хватает {q0_det['id1_missing']}"}
    if not rules["Q0"]["pass"]:  # spec §8: the delay axis is not read; Q3a / Q3b on A go to the report
        for n in ("Q1", "Q3a", "Q3b"):
            rules[n] = {"test": RULES[n][0], "pass": None, "note": "не читалось: Q0 не прошло"}
        for n in ("Q3a", "Q3b"):  # their numbers on cell A alone, report only
            rules[n]["report_A"] = e2.rnd(e2.diff_pp(RULES[n][1](idx, ())))
    verdicts = {v: "ДА" if all(rules[n]["pass"] for n in need_rules) else "НЕТ" for v, need_rules in VERDICTS.items()}
    if "override" in gate:
        verdicts = {v: "ОТСТУПЛЕНИЕ (проверка перед чтением не пройдена, см. read_gate.override): " + x
                    for v, x in verdicts.items()}
    return res | {"verdicts": verdicts, "rules": rules, "report": report(rs, idx)}


# ---------------------------------------------------------------- §10 report (only reported, never a rule; the rest of
# §10 goes into the memo)

def report(rs, idx):
    out = {"success": {}, "eyes": {}, "delay_law": {}, "fresh_before_close": {}}
    for key in sorted(idx):
        g = list(idx[key].values())
        out["success"]["/".join(key)] = {
            "n": len(g), "success": round(float(np.mean([r["success"] for r in g])), 4),
            "calls": [round(float(np.mean([r[k] for r in g])), 2) for k in ("calls_sched", "calls_trig", "calls_shadow")],
            "engaged_before_close": round(float(np.mean([r["t_engage"] is not None and (r["t_close"] is None
                                                         or r["t_engage"] < r["t_close"]) for r in g])), 4)}
        if key[3].startswith("G@"):  # noisy eyes: false (before the shift), missed, delay
            ft = [(r["t_alarm"] if key[3] == "G@glr" else r["t_engage"], r["t_fire"], r["t_close"]) for r in g]
            on = [a - f for a, f, c in ft if a is not None and f is not None and a > f and (c is None or a < c)]
            out["eyes"]["/".join(key)] = {
                "false": round(float(np.mean([a is not None and (f is None or a <= f) for a, f, _ in ft])), 4),
                "missed": round(float(np.mean([a is None for a, _, _ in ft])), 4),
                "delay_median": float(np.median(on)) if on else None}
    bins = (("<=10", 0, 10), ("11-20", 11, 20), ("21-40", 21, 40), (">40", 41, 10 ** 9), ("no_close", None, None))
    for b in (P, S):
        for c in gauto.CELLS:
            x, y = idx.get((b, "step", c, "G"), {}), idx.get((b, "step", c, "none"), {})
            if not (x.keys() & y.keys()):
                continue
            cell = {}  # spec §10: G - none by tau = t_close - t_fire of "none" in the same episode and cell
            for name, lo, hi in bins:
                tau = {k: None if y[k]["t_close"] is None else y[k]["t_close"] - y[k]["t_fire"]
                       for k in x.keys() & y.keys() if y[k]["t_fire"] is not None}
                p = [(k[:2], x[k]["success"], y[k]["success"]) for k in sorted(tau)
                     if (tau[k] is None if lo is None else tau[k] is not None and lo <= tau[k] <= hi)]
                cell[name] = e2.rnd(e2.diff_pp(p))
            out["delay_law"][f"{b}/{c}"] = cell
            fresh = []  # the first plan observed after the shift arrives before the close (spec §10 mechanism)
            for r in y.values():
                if r["t_fire"] is None:
                    continue
                arr = [ta for to, ta in r["plans_in"] if to >= r["t_fire"] + 1]
                fresh.append(bool(arr) and (r["t_close"] is None or arr[0] < r["t_close"]))
            out["fresh_before_close"][f"{b}/{c}"] = round(float(np.mean(fresh)), 4) if fresh else None
    return out


def selftest():
    rng = np.random.default_rng(0)
    n_pi = sum(len(EXPECTED[g]) for g in GRID if GRID[g][1] == P)
    n_s = sum(len(EXPECTED[g]) for g in GRID if GRID[g][1] == S)
    rule_pi = sum(len(EXPECTED[g]) for g in GRID if GRID[g][1] == P and GRID[g][0] in RULE_ROWS)
    assert (n_pi, n_s, rule_pi) == (17074, 7972, 12568), (n_pi, n_s, rule_pi)  # spec §6: 17 074 / 7 973 / 12 567
    assert sum(k for *_, k, _ in plan()) == n_pi + n_s
    assert {g for _, _, g, _, _ in plan(hold=True)} == set(GRID) - set(HELD)
    assert {g for _, _, g, _, _ in plan(cut=("12", "14"))} == set(GRID) - {"12", "12d", "14a", "14b"}
    try:
        plan(cut=("7",))
        raise AssertionError("a rule row was cut")
    except SystemExit:
        pass
    assert plan()[0][:3] == ("control", "p", "6") and plan()[1][:3] == ("q0", "p", "1")
    assert precision(10.0, 6, 7.6, 1, 1) == ("bf16-row", 10) and precision(10.0, 6, 7.8, 1, 1) == ("fp32", 6)
    assert precision(10.0, 5, 5.0, 1, 0) == ("fp32", 5) and precision(10.0, 6, 5.0, 0, 1) == ("fp32", 6)

    conf = {P: {"kbar": 0.33}, S: {"kbar": 0.05}}
    succ_p = {"none": 0.70, "T0": 0.80, "G": 0.85, "Gkeep": 0.78, "Gk0": 0.86, "Gauto": 0.86, "GT": 0.70, "GautoT": 0.82,
              "G@glr": 0.80, "Gk0T": 0.82, "GR": 0.84, "PPC": 0.70, "G@ema": 0.78}

    def rec(key, p_succ):
        b, kind, cell, arm, suite, task, init = key
        ep = hf.make_episode(suite, task, init, kind, prefix="c1e4")
        return {"brain": b, "kind": kind, "cell": cell, "arm": arm, "suite": suite, "task": task, "init": init,
                "conf": conf[b], "mag_class": ep.mag_class, "success": bool(rng.random() < p_succ),
                "calls_sched": 14, "calls_trig": 0, "calls_shadow": 0, "t_engage": 31, "t_close": 40, "t_fire": 30,
                "t_alarm": 33, "plans_in": [[0, 0], [30, 30], [40, 40]]}

    def grid(fx):
        return [rec(k, fx(k)) for g in GRID for k in EXPECTED[g]]

    raw = grid(lambda k: 1.0 if k[1] == "control" else succ_p[k[3]] + (0.12 if k[2] in DELAY and k[3] == "G" else 0))
    base = {"s10": 0.95}
    out = summarize(raw, base)
    assert out["read_gate"]["pass"], out["read_gate"]
    r = out["rules"]
    assert r["Q0"]["pass"] and r["Q1"]["pass"] and r["Q2a"]["pass"] and r["Q3b"]["pass"], r
    assert not r["Q2c"]["pass"] or r["Q2c"]["diff_pp"] >= 3
    assert out["verdicts"]["МОЗГ САМ ОТРАБАТЫВАЕТ СДВИГ"] == "ДА"
    assert control(raw)[0] and q0(raw)[0]
    # Q0 fails: the delay ids are not expected, Q1 / Q3a / Q3b are not read
    raw0 = [x | {"success": x["success"] if x["arm"] != "T0" else x["success"] and rng.random() < 0.8}
            for x in raw if not (x["brain"] == P and x["cell"] in ("A10", *DELAY))]
    out0 = summarize(raw0, base)
    assert not q0(raw0)[0] and out0["read_gate"]["pass"] and out0["read_gate"]["held_delay_ids"] == list(HELD)
    assert out0["rules"]["Q1"]["pass"] is None and out0["verdicts"]["ОСЬ ЗАДЕРЖКИ НА НАСТОЯЩЕЙ VLA"] == "НЕТ"
    assert out0["rules"]["Q3a"]["report_A"]["pairs"] == 520  # G@glr - none on A, 0-19
    # the pi control check: 80% fails the gate, and the read gate with it
    rawc = [x | {"success": rng.random() < 0.8} if (x["brain"], x["kind"], x["cell"]) == (P, "control", "A") else x
            for x in raw]
    assert not control(rawc)[0] and not summarize(rawc, base)["read_gate"]["pass"]
    # the control floor: A40 control "none" below 50% reads Q1 on A20 only
    rawf = [x | {"success": False} if (x["brain"], x["kind"], x["cell"], x["arm"]) == (P, "control", "A40", "none")
            else x for x in raw]
    assert summarize(rawf, base)["rules"]["Q1"]["delay_cells"] == ["A20"]
    # Q2a off by more than 0.10
    rawk = [x | {"conf": {"kbar": 0.50}} if x["brain"] == P else x for x in raw]
    assert not summarize(rawk, base)["rules"]["Q2a"]["pass"]
    # an incomplete grid fails the gate
    assert not summarize(raw[: len(raw) // 2], base)["read_gate"]["complete"]

    # trial: kbar_30 in round-robin order, tau_kappa, the journal line
    def cal(b, smp):
        return [{"brain": b, "kind": "step", "cell": "A", "arm": "Gcal", "suite": s, "task": t, "init": i,
                 "shadow": {"sample": smp(i, ti)}} for i in range(40, 44) for ti, (s, t) in enumerate(orx.TASKS)]
    p2 = cal(P, lambda i, ti: 0.3 if i == 40 else 0.9)  # the first 26 (init 40) give 0.3, the next 4 give 0.9
    p3 = cal(S, lambda i, ti: None if ti % 2 else 0.04)
    walk = lambda n: [[0, 0, 0, -1]] * n + [[0, 0, 0, 1]] * (orx.H - n)  # noqa: E731  (still arm, closes at n)
    p1 = [{"brain": P, "kind": "control", "cell": "A", "arm": "none", "suite": s, "task": t, "init": i, "t_fire": 12,
           "t_close": 60, "angle": 0.0, "shadow": {"diff": [0.01, 0.0, 0.0]},
           "trace": {"plans": [[10, [0, 0, 0], walk(20)], [20, [0.001 * ti, 0, 0], walk(10)]]}}
          for i in range(44, 48) for ti, (s, t) in enumerate(orx.TASKS)]
    line, rep = trial(p1, p2, p3, "C1-E4 precision: precision=fp32 envs=6 (...)")
    assert rep["kbar"][P]["kbar_30"] == 0.3 and rep["kbar"][P]["samples_used"] == 30
    assert rep["kbar"][S]["samples"] == 52 and rep["kbar"][S]["kbar_30"] == 0.04
    assert line.startswith("C1-E4 trial: precision=fp32 tau_k=") and line.endswith("N=30 envs=6 pad=0"), line
    assert rep["shadow_pairs"] == 104 and rep["shadow_jitter_m"]["median"] == 0.01
    fc = forecast(7.0, 1.7)
    assert fc["episodes"] == {P: 17074, S: 7972} and fc["hours"]["4090"] > fc["hours"]["48GB"]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="?", help="JSON lines of scripts/c1e4_run.py, all brains")
    ap.add_argument("--baseline", help='JSON {"s10": rate}: C1-E2\'s, or this pod\'s stock SmolVLA lerobot-eval')
    ap.add_argument("--override-gate", help="the spec journal entry that records the owner's deviation")
    ap.add_argument("--control", nargs="+", metavar="JSONL", help="pi0.5 records: exit 0 if the control check passes, else 3")
    ap.add_argument("--q0", nargs="+", metavar="JSONL", help="pi0.5 records: exit 0 if Q0 passes, else 3")
    ap.add_argument("--plan", action="store_true", help='the pod\'s runner lines: "phase<TAB>lane<TAB>id<TAB>n<TAB>args"')
    ap.add_argument("--cut", default="", help='with --plan / --forecast: report rows cut by the §12 fuse, e.g. "12 16"')
    ap.add_argument("--hold", action="store_true", help="with --plan / --forecast: without the ids the Q0 gate holds")
    ap.add_argument("--precision", nargs=5, metavar=("FP32_S", "FP32_ENVS", "BF16_S", "BF16_FITS", "BITWISE"))
    ap.add_argument("--trial", nargs=3, metavar=("P1", "P2", "P3"))
    ap.add_argument("--choice", help="with --trial: the smoke's precision_choice.txt")
    ap.add_argument("--forecast", nargs=2, type=float, metavar=("PI_S", "S_S"), help="trial seconds per episode per lane")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.plan:
        for row in plan(a.cut.split(), a.hold):
            print("\t".join(map(str, row)))
    elif a.control or a.q0:
        ok, det = (control if a.control else q0)([r for f in (a.control or a.q0) for r in e2.load(f)])
        print(json.dumps(det, ensure_ascii=False))
        raise SystemExit(0 if ok else 3)
    elif a.precision:
        num = lambda x: None if x in ("", "-", "nan") else float(x)  # noqa: E731
        f_s, f_e, b_s, fits, bit = a.precision
        prec, envs = precision(num(f_s), int(f_e), num(b_s), int(fits), int(bit))
        print(f"C1-E4 precision: precision={prec} envs={envs} (fp32 {f_s} s/episode at {f_e} envs; bf16-row {b_s} "
              f"s/episode at 10 envs, fits {fits}, bitwise {bit}; rule: bf16-row if >= 1.3x faster, bitwise and fits)")
    elif a.trial:
        if not a.choice:
            ap.error("--trial needs --choice")
        line, rep = trial(*(e2.load(f) for f in a.trial), Path(a.choice).read_text())
        print(e2.dump(rep))
        print(line)
    elif a.forecast:
        print(json.dumps(forecast(*a.forecast, a.cut.split(), a.hold)))
    else:
        if not (a.grid and a.baseline):
            ap.error("grid and --baseline are required")
        src = Path(a.grid)
        text = e2.dump(summarize(e2.load(src), json.loads(Path(a.baseline).read_text()), a.override_gate))
        (src.parent / "summary.json").write_text(text + "\n")
        print(text)
