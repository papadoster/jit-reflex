"""C1-E4 VT-m (B): PBVS's machinery with VS-T's target (docs/superpowers/specs/2026-10-20-c1-e4-pbvs-design.md, its
journal "2026-10-22 — ветка VT-m: формула и правила (до кода и до данных B)", "VT-m: уточнения по ходу реализации" and
"VT-m: по ревью кода"), on pod C after a clean kgrid (scripts/gpu_c1e4m.sh POD=C mgrid): the runner plan per line and
the reading (the checks, VT-m - G-auto with its outcome, В1 VT-m - VS-T, В2 VT-m - PBVS, the noise floor's rule, the
counters). Reuses scripts/c1e4v_summary.py (the controls mode and k, groups, ids, unique, pairs, V2eps's key),
scripts/c1e4t_summary.py (the VT lines, the s* check) and through them scripts/c1e4c_summary.py; run from the
repository root. R = results/c1-e4/controls/pod_C:
    python scripts/c1e4b_summary.py --plan --line L
        L: Md0 Md10, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4b_summary.py [read] R/grid_all_M.jsonl [--controls R/C2_*.jsonl R/C3_*.jsonl]
            [--t R/grid_all_T.jsonl] [--v R/grid_all_V.jsonl] [--vst-k R/vst_k.json] [--pbvs-k R/pbvs_k.json]
            [--band-json R/band.json] [--out R/summary_M.json]
        the checks before reading, then the measurements, the noise floor, the journal's reading and the counters; a
        failed check prints only the checks, exit 3. Needs hf-libero (the task clusters)
    python scripts/c1e4b_summary.py --selftest
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4t_summary as ct  # noqa: E402
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import P, TOL, e2, maxwrap, run  # noqa: E402

ARM = "VTm"  # vservo.VTM: lambda 0.3 inherited
BAND = f" --band {cc.OUT}/band_eps.json"  # the journal: model A of band.json, delta 5 mm
# the journal's cells -> (source, d, the controls line it pairs with): d 0 PRO (VT2's, line 2's), d 10 Plus (VT3d10's,
# line 3's)
LINES = {"Md0": ("pro", 0, "2"), "Md10": ("plus", 10, "3")}
CELLS = tuple(f"d{d}" for _, d, _ in LINES.values())
# comparison -> the other side: main VT-m - G-auto, В1 VT-m - VS-T, В2 VT-m - PBVS (V2eps at d 0, V3d10 at d 10)
COMP = {"main": "Gauto", "B1": "VST", "B2": "PBVS"}
FLOOR_N, FLOOR_X = 30, 2  # the journal: >= 30 episodes with both aims (pooled), median signal >= 2 x median floor
# the journal's reading of the main comparison, word for word
MAIN = {"(i)": "«плоская кривая» снимается. Вывод: «информация полезна, беспамятная реализация неудачна». Пишем так, "
               "даже если это рушит тезис сводки.",
        "(ii)": "тезис тоже снимается, как неподдержанный: данные склоняются против него.",
        "(iii)": "тезис остаётся только в суженном виде: «цель из показа, использованная с памятью, не обгоняет наш "
                 "закон в пределах верхних границ X». Добавляются оговорки про пол шума и наследованный λ. Если "
                 "правило пола шума даёт «непоказательно», тезис этим исходом не усиливается."}
B1 = {True: "провал VS-T под задержкой характеризует конструкцию (беспамятность), а не информацию — как и сказал "
            "рецензент", False: "не показано"}
B2 = {"d0": "сравнение чистое: тот же порог 5 мм и та же модель (V2eps); измерено, только если пол шума показателен",
      "d10": "только отчёт: против V3d10, где порог band.json — смешаны цель и порог; В2 при d = 10 остаётся "
             "неизмеренным"}


def plan(line, split=None):
    """The journal's lines, phase line.lower(), in c1e4c_summary.mlines's format: VTm alone on eval300's cases of one
    source and d, the Base full (no --base-prefix: s* is its first close) in the same process, --band band_eps.json;
    the controls line's cases, order and Base groups of that source and d (asserted), as VT2 PRO and VT3d10 Plus."""
    split = split or maxwrap.load_split(cs.SPLIT)
    src, d, cl = LINES[line]
    one = lambda lns: [ln for ln in lns if ln["source"] == src and f"--d {d} " in ln["args"] + " "]  # noqa: E731
    lns = [ln | {"args": ln["args"] + BAND} for ln in one(cc.mlines(line.lower(), "M", P, "eval300", (ARM,), split, d,
                                                                       prefix=False))]
    ctl = one(cc.plan(cl, cv.CMODE, cv.CK, split))
    assert len(lns) == 1 and cv.groups(lns, split) == cv.groups(ctl, split), \
        f"{line}: not controls line {cl}'s {src} cases, order and groups at d {d}"
    return lns


def sides(vk, pk, band, split):
    """The runner lines the reading reads, {(side, cell): (records, arm, line)}: per cell (d0: PRO, d10: Plus) the M
    line's VTm, the controls line's Gauto (line 2 / 3), the VT line's VS-T (VT2 / VT3d10: vst_k.json's arm) and the V
    line's PBVS (V2eps / V3d10: pbvs_k.json's arm). records: M, C, T, V (grid_all_M, the controls, grid_all_T,
    grid_all_V)."""
    out = {}
    for L, (src, d, cl) in LINES.items():
        def one(lns, src=src, d=d):
            (ln,) = [ln for ln in lns if ln["source"] == src and f"--d {d} " in ln["args"] + " "]
            return ln
        c = f"d{d}"
        out["VTm", c] = ("M", ARM, plan(L, split)[0])
        out["Gauto", c] = ("C", "Gauto", one(cc.plan(cl, cv.CMODE, cv.CK, split)))
        out["VST", c] = ("T", cv.need(vk, "vst_k.json", "arm"), one(ct.plan("VT2" if d == 0 else "VT3d10", vk, split)))
        out["PBVS", c] = ("V", cv.need(pk, "pbvs_k.json", "arm"),
                          one(cv.plan("V2eps" if d == 0 else "V3d10", pk, band, split)))
    return out


def est(prs):
    """The VS-T reading's scheme (scripts/c1e4t_summary.py, e2.diff_pp): the paired difference VT-m - other in pp,
    every pair weighs one; the cluster bootstrap by source task, 2000 resamples (e2.N_BOOT), generator seed 0;
    percentiles 2.5 and 97.5 (two-sided 95 %), 5 (lo5_one_sided) and, of the same resamples (cv.upper95 at 2000),
    95 (hi95_one_sided). Raw floats but hi95."""
    p = [q[1:] for q in prs]
    return e2.diff_pp(p) | {"hi95_one_sided": cv.upper95(p, reps=e2.N_BOOT)}


def ref_ok(r, b):
    """A VTm record's s* is its own full Base's first-close offset: VS-T's check (ct.ref_ok) on its "vtm"."""
    return ct.ref_ok({"vst": r.get("vtm")}, b)


def floor(vtm):
    """The journal's noise floor: per episode with both a_floor (a_R) and a_first (a_N) defined, floor |a_R| and
    signal |a_N - a_R| on xy; pooled over the two cells; informative iff n >= FLOOR_N and median signal >= FLOOR_X x
    median floor (else not computable, or failed: a null not informative). vtm: the read VT-m records' "vtm"."""
    ab = [(np.array(v["a_floor"], float), np.array(v["a_first"], float)) for v in vtm
          if v["a_floor"] is not None and v["a_first"] is not None]
    f = [float(np.linalg.norm(a)) for a, _ in ab]
    s = [float(np.linalg.norm(b - a)) for a, b in ab]
    mf, ms = (float(np.median(f)), float(np.median(s))) if ab else (None, None)
    ok = len(ab) >= FLOOR_N and ms >= FLOOR_X * mf - TOL
    why = (f"не вычислимо: эпизодов с a_floor и a_first {len(ab)} < {FLOOR_N}" if len(ab) < FLOOR_N else
           None if ok else "знание утонуло в невязке модели")
    return {"n": len(ab), "median_floor_m": mf and round(mf, 6), "median_signal_m": ms and round(ms, 6),
            "ratio": round(ms / mf, 3) if ab and mf else None, "rule": f"n >= {FLOOR_N}, ratio >= {FLOOR_X}",
            "informative": ok, "verdict": "показательно" if ok else "непоказательно", "why": why}


def read(new, ctrl, trecs, vrecs, vk, pk, band, names, split):
    """The journal's checks before reading (completeness >= 99 % of every side's Dynamic arm and Base records, the M
    lines' VTm and full Bases among them; invalid pairs <= 2 % per comparison, a missing record not one; conf equal per
    d but cv.CONF_SKIP (arms, gpu, kbar, pbvs) across the read records; the M Bases' e equal to the VT Bases' on >= 98 %
    per cell; every VTm record's s* its own Base's first-close offset (ct.ref_ok); the M records on band_eps.json,
    band.json's model at 5 mm), then the measurements, the noise floor, the reading and the counters. new:
    grid_all_M.jsonl; ctrl: controls lines 2 and 3; trecs: grid_all_T.jsonl; vrecs: grid_all_V.jsonl (V2eps told from V2
    by cv.vk). Pairs: the journal's (by case_id and policy_seed, the Bases' e equal), cv.pairs as the K reading's: one
    Dynamic episode per side and case, the same seed, both valid, the two Bases' e equal. The noise floor over every
    read VTm record (an invalid one too: its aims are its own plans'). Records off the plan are counted, never read."""
    sd = sides(vk, pk, band, split)
    keyf = {"M": cc.lk, "C": cc.lk, "T": cc.lk, "V": lambda r: cv.vk(r, band)}
    kset = {"M": cc.lkeys, "C": cc.lkeys, "T": cc.lkeys, "V": cv.lkeys}
    by = {s: cv.unique(rs, keyf[s]) for s, rs in {"M": new, "C": ctrl, "T": trecs, "V": vrecs}.items()}
    want = {(s, x): n for n, (s, a, ln) in sd.items() for x in kset[s]([ln]) if x[3] == "base" or x[4] == a}
    used = {(s, x): r for s, b in by.items() for x, r in b.items() if (s, x) in want and not r.get("skipped_reason")}
    of = lambda n, kind: {r["case_id"]: r for k, r in used.items() if want[k] == n and r["kind"] == kind}  # noqa: E731
    comp = {}
    for n in sd:
        exp = [k for k, v in want.items() if v == n]
        miss = sum(k not in used for k in exp)
        comp["/".join(n)] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    one = {r["case_id"]: r for r in used.values()}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    pr = {(m, c): cv.pairs(of(("VTm", c), "dynamic"), of((o, c), "dynamic"), of(("VTm", c), "base"),
                           of((o, c), "base"), clu, cv.ids(sd["VTm", c][2], split))
          for m, o in COMP.items() for c in CELLS}
    bad = lambda drop: sum(v for w, v in drop.items() if w != "missing")  # noqa: E731  (missing: the completeness's)
    em, off = {}, {}
    for c in CELLS:  # the M Bases are the VT Bases (one deterministic Base); s* each record's own Base's
        mb, tb = of(("VTm", c), "base"), of(("VST", c), "base")
        a = sum(tb.get(k, {}).get("e", "-") == b["e"] for k, b in mb.items())
        em[c] = {"equal": a, "total": len(mb), "pass": a >= 0.98 * len(mb) - TOL}
        off[c] = sum(not ref_ok(r, mb[k]) for k, r in of(("VTm", c), "dynamic").items() if k in mb)
    conf = defaultdict(set)
    for r in used.values():
        conf[f"d{r['d']}"].add(json.dumps({f: v for f, v in r["conf"].items() if f not in cv.CONF_SKIP},
                                          sort_keys=True))
    mixed = {g: sorted(v) for g, v in conf.items() if len(v) > 1}
    epsb = band | {"delta": cv.EPS}  # band_eps.json
    off_band = Counter(want[k][1] for k, r in used.items() if k[0] == "M" and r["conf"].get("pbvs") != epsb)
    checks = {"records": len(new), "read": len(used),
              "skipped": sum(bool(r.get("skipped_reason")) for s, b in by.items() for x, r in b.items()
                             if (s, x) in want),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for r in new if ("M", cc.lk(r)) not in want)),
              "complete": comp, "e_match": em, "s_star": {"off": off, "pass": not any(off.values())},
              "band_in_conf": {"want": epsb, "off": dict(off_band), "pass": not off_band},
              "invalid_pairs": {f"{m}/{c}": {"pairs": len(p) + bad(dr), "invalid": bad(dr),
                                             "pass": bad(dr) <= 0.02 * (len(p) + bad(dr)) + TOL}
                                for (m, c), (p, dr) in pr.items()},
              "conf": {"except": cv.CONF_SKIP, "mixed": mixed, "pass": not mixed}}
    checks["pass"] = bool(all(v["pass"] for f in ("complete", "e_match", "invalid_pairs") for v in checks[f].values())
                          and all(checks[f]["pass"] for f in ("s_star", "band_in_conf", "conf")))
    if not checks["pass"]:
        return {"checks": checks}

    raw, meas = {}, {}
    for (m, c), (p, dr) in pr.items():
        _, a, ln = sd[COMP[m], c]
        raw[m, c] = est(p)
        meas[f"{m}/{c}"] = {"test": f"pi05, {ln['source']}, d = {c[1:]}: {ARM} - {a} ({ln['id']}), pp",
                            **e2.rnd(raw[m, c]), "invalid_pairs": bad(dr), "drops": dr}
    vtm = {c: [r["vtm"] for r in of(("VTm", c), "dynamic").values()] for c in CELLS}
    fl = floor(vtm["d0"] + vtm["d10"])
    for c in CELLS:  # В2: d 0 measured only on an informative floor; d 10 never (target and gate mixed)
        meas[f"B2/{c}"] |= {"measured": c == "d0" and fl["informative"], "note": B2[c]}
    mn = [raw["main", c] for c in CELLS]
    out = ("(i)" if any(v["lo5_one_sided"] > 0 for v in mn) else "(ii)" if all(v["diff_pp"] > 0 for v in mn)
           else "(iii)")
    reading = {"main": {"outcome": out, "sentence": MAIN[out]}
               | ({"X_pp": {c: meas[f"main/{c}"]["hi95_one_sided"] for c in CELLS}} if out == "(iii)" else {})
               | {"noise_floor": fl["verdict"], "label": "непоказательно" if out == "(iii)" and not fl["informative"]
                  else None},
               "B1": B1[raw["B1", "d10"]["lo5_one_sided"] > 0]}
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    report = {}
    for c, vs in vtm.items():  # the counters per cell, shares per episode over the episodes with a window
        w = [v for v in vs if v["steps_window"]]
        sh = lambda f: mean([v[f] / v["steps_window"] for v in w])  # noqa: E731
        report[c] = {"n": len(vs), "s_star_none": sum(v["s_star"] is None for v in vs), "window": len(w),
                     "acting_share": sh("steps_acting"), "steps_aim_share": sh("steps_aim"),
                     "no_aim_share": sh("no_aim"), "mean_e_m": mean([v["sum_e"] / v["steps_window"] for v in w])}
    return {"checks": checks, "measurements": meas, "noise_floor": fl, "reading": reading, "counters": report}


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 VT-m checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


# ------------------------------------------------------------------ selftest

def selftest():
    """The plan on the real split: one line per cell, VTm alone, no --base-prefix, no kbar, --band band_eps.json, the
    controls line's cases in its groups (every Base full: 113 PRO / 187 Plus cases, two episodes each); the
    bootstrap; the floor's rule; the reading on a synthetic pod."""
    split = maxwrap.load_split(cs.SPLIT)
    for line, ident, src, d, n in (("Md0", "M_pi05_eval300_pro_d0", "pro", 0, 226),
                                   ("Md10", "M_pi05_eval300_plus_d10", "plus", 10, 374)):
        (ln,) = plan(line, split)
        assert ln["id"] == ident and ln["phase"] == line.lower() and ln["arms"] == ["VTm"] and ln["source"] == src, ln
        assert ln["args"] == (f"--brain pi05 --source {src} --set eval300 --arms=VTm --d {d} "
                              "--band results/c1-e4/controls/pod_C/band_eps.json"), ln
        assert not ln["prefix"] and cc.n_of(ln) == n, ln
    # the estimate is the VS-T reading's (e2.diff_pp, 2000 resamples), hi95 of the same resamples
    rng = np.random.default_rng(5)
    prs = [(j, f"t{j % 13}", rng.random() < 0.6, rng.random() < 0.4) for j in range(150)]
    p = [x[1:] for x in prs]
    assert e2.N_BOOT == 2000 and e2.rnd(est(prs)) == e2.rnd(e2.diff_pp(p)) | {"hi95_one_sided": cv.upper95(p, 2000)}
    assert est(prs)["hi95_one_sided"] != cv.upper95(p), "2000 and 8000 resamples should differ here"
    selftest_floor()
    selftest_read(split)
    print("selftest ok")


F, S = [0.0006, -0.0008], [0.0006, 0.0022]  # |a_R| 1 mm, |a_N - a_R| 3 mm (|a_N| - |a_R| would be 1.28 mm)


def selftest_floor():
    """The rule: n >= 30 and median |a_N - a_R| >= 2 x median |a_R| on xy; one aim undefined: the episode out."""
    v = lambda a, b: {"a_floor": a, "a_first": b}  # noqa: E731
    f = floor([v(F, S)] * 30 + [v(None, S), v(F, None)] * 5)
    assert f == {"n": 30, "median_floor_m": 0.001, "median_signal_m": 0.003, "ratio": 3.0,
                 "rule": "n >= 30, ratio >= 2", "informative": True, "verdict": "показательно", "why": None}, f
    f = floor([v(F, S)] * 29 + [v(None, S)] * 9)
    assert not f["informative"] and f["n"] == 29 and f["verdict"] == "непоказательно" and "29 < 30" in f["why"], f
    f = floor([v([2 ** -10, 0], [3 * 2 ** -10, 0])] * 40)  # the ratio exactly 2: informative
    assert f["informative"] and f["ratio"] == 2.0, f
    f = floor([v(F, [0.0006, 0.0007])] * 40)  # 1.5 mm against 1 mm
    assert not f["informative"] and f["ratio"] == 1.5 and f["why"] == "знание утонуло в невязке модели", f
    assert floor([v(F, S)] * 25 + [v(F, [0.0006, -0.0003])] * 26)["ratio"] == 0.5  # the median, not the mean
    assert floor([])["n"] == 0 and not floor([])["informative"]


def selftest_read(split):
    """The reading on a synthetic pod: the M lines, controls lines 2 and 3, VT2 and VT3d10, V2eps, V3d10 and V2
    (band.json's PBVS at d 0, never В2's). Success by the case's place i in its cell in cycles of 4 (q: the places of
    the 4 that succeed, by side and d), so the differences are exact; task clusters from fake LIBERO names; one policy
    seed per case. A Base on a place 0 mod 50 never closed (its VT-m has no s*); both aims on places 0 mod 3 whose Base
    closed."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    vk, pk = {"arm": "VST0.3"}, {"arm": "PBVS0.3"}
    band = {"model": "A", "G": [0.0025, 0.0025], "B": [0.79, 0.79], "delta": 0.0262}
    sd = sides(vk, pk, band, split)
    cases = {c: cv.ids(sd["VTm", c][2], split) for c in CELLS}
    pos = {x: j for c in CELLS for j, x in enumerate(cases[c])}
    v2 = next(ln for ln in cv.plan("V2", pk, band, split) if ln["source"] == "pro")
    cf = {"brain": P, "revision": "r", "precision": "fp32", "s": 10, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01, "R": 360,
          "split": split["sha256"]}
    q = {("VTm", 0): 2, ("Gauto", 0): 2, ("VST", 0): 1, ("PBVS", 0): 3, ("V2", 0): 0,
         ("VTm", 10): 3, ("Gauto", 10): 1, ("VST", 10): 2, ("PBVS", 10): 2}
    cyc = lambda q: lambda lab, d, i: i % 4 < q.get((lab, d), 0)  # noqa: E731
    aims = lambda i, d: (F, S) if i % 3 == 0 else None  # noqa: E731

    def rec(x, lab, win, fl, pbvs=None):  # x: lk (+ eps)
        _, b, c, kind, a, d, _, arms = x[:8]
        h, i = sum(map(ord, c)), pos[c]
        s, closed = list(run.MAX_STEPS)[h % 4], i % 50 != 0
        r = {"brain": b, "case_id": c, "kind": kind, "arm": a, "d": d, "policy_seed": 195 + i,
             "conf": cf | {"d": d, "arms": list(arms), "kbar": 0.306 if "Gauto" in arms else None, "gpu": "RTX 4090"}
             | ({"pbvs": pbvs} if pbvs else {}),
             "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None, "e": 30,
             "prefix_ok": None if kind == "base" else True, "replay_miss": None, "skipped_reason": None,
             "success": None if kind == "base" else bool(win(lab, d, i)), "t_close": 40 if closed else None,
             "grasp_miss_m": 0.01 if closed else None}
        if a == ARM:
            ab = closed and fl(i, d) or (None, None)
            r["vtm"] = {"s_star": [0.006, -0.008] if closed else None, "steps_window": 20, "steps_acting": 5,
                        "sum_e": 0.2, "max_e": 0.03, "flips": 1, "steps_aim": 12 if closed else 0,
                        "no_aim": 0 if closed else 12, "a_floor": ab[0], "a_first": ab[1]}
        return r

    def world(win, fl=aims):
        out = defaultdict(list)
        epsb = band | {"delta": cv.EPS}
        for (lab, c), (s, a, ln) in sd.items():
            pb = {"M": epsb, "V": epsb if c == "d0" else band}.get(s)
            out[s] += [rec(x, lab if x[4] == a else x[4], win, fl, pb)
                       for x in sorted((cv.lkeys if s == "V" else cc.lkeys)([ln]))]
        out["V"] += [rec(x, "V2", win, fl, band) for x in sorted(cv.lkeys([v2]))]
        return out["M"], out["C"], out["T"], out["V"]

    def want(c, a, b, win, out=()):  # a difference by hand: the mean paired difference over the cell's cases
        d, cs_ = int(c[1:]), [x for x in cases[c] if x not in out]
        return round(100 * sum(win(a, d, pos[x]) - win(b, d, pos[x]) for x in cs_) / len(cs_), 3)

    win = cyc(q)
    new, ctrl, trecs, vrecs = world(win)
    rd = lambda n=new, c=ctrl, t=trecs, v=vrecs: read(n, c, t, v, vk, pk, band, names, split)  # noqa: E731
    probe = rec(("m", P, cases["d0"][0], "dynamic", ARM, 20, False, (ARM,)), "VTm", win, aims)  # off the plan
    res = rd(new + [probe] + new[:5])  # new[:5]: identical copies
    ch, ms, rg = res["checks"], res["measurements"], res["reading"]
    assert ch["pass"] and ch["off_plan"] == {f"{P}/VTm": 1} and ch["records"] == len(new) + 6, ch
    assert {n: (v["expected"], v["missing"]) for n, v in ch["complete"].items()} == {
        f"{s}/{c}": (226 if c == "d0" else 374, 0) for c in CELLS for s in ("VTm", "Gauto", "VST", "PBVS")}, ch
    assert ch["e_match"] == {"d0": {"equal": 113, "total": 113, "pass": True},
                             "d10": {"equal": 187, "total": 187, "pass": True}} and ch["s_star"]["pass"], ch
    assert ch["invalid_pairs"] == {f"{m}/{c}": {"pairs": len(cases[c]), "invalid": 0, "pass": True}
                                   for m in COMP for c in CELLS}, ch["invalid_pairs"]
    # exact differences; В2 at d 0 against V2eps (75 %), never V2 (0 %)
    got = {n: v["diff_pp"] for n, v in ms.items()}
    assert got == {f"{m}/{c}": want(c, "VTm", o, win) for m, o in COMP.items() for c in CELLS} and got["main/d0"] == 0 \
        and got["B2/d0"] == want("d0", "VTm", "PBVS", win) != want("d0", "VTm", "V2", win), got
    assert all(v["pairs"] == len(cases[n.split("/")[1]]) for n, v in ms.items()), ms
    assert all(v["lo2.5"] <= v["lo5_one_sided"] <= v["diff_pp"] <= v["hi95_one_sided"] <= v["hi97.5"]
               for v in ms.values()), ms
    m = ms["main/d10"]
    assert {n: v["test"] for n, v in ms.items()} == {
        f"{m_}/{c}": f"pi05, {'pro' if c == 'd0' else 'plus'}, d = {c[1:]}: VTm - {a} ({i}), pp"
        for m_, a, i0, i10 in (("main", "Gauto", "C2_nv", "C3"), ("B1", "VST0.3", "VT2", "VT3"),
                               ("B2", "PBVS0.3", "V2eps", "V3"))
        for c, i in (("d0", f"{i0}_pi05_eval300_pro_d0"), ("d10", f"{i10}_pi05_eval300_plus_d10"))}, ms
    # the reading: (i) on d 10's one-sided lower bound; В1 at d 10 shown; the floor informative (98 episodes)
    assert m["lo5_one_sided"] > 0 and rg["main"] == {"outcome": "(i)", "sentence": MAIN["(i)"],
                                                     "noise_floor": "показательно", "label": None}, rg
    assert rg["B1"] == B1[True] and ms["B1/d10"]["lo5_one_sided"] > 0, rg
    assert res["noise_floor"] == {"n": 37 + 61, "median_floor_m": 0.001, "median_signal_m": 0.003, "ratio": 3.0,
                                  "rule": "n >= 30, ratio >= 2", "informative": True, "verdict": "показательно",
                                  "why": None}, res["noise_floor"]
    assert {c: (ms[f"B2/{c}"]["measured"], ms[f"B2/{c}"]["note"]) for c in CELLS} == {
        "d0": (True, B2["d0"]), "d10": (False, B2["d10"])} and "смешаны цель и порог" in B2["d10"], ms
    assert res["counters"] == {c: {"n": len(cases[c]), "s_star_none": k, "window": len(cases[c]), "acting_share": 0.25,
                                   "steps_aim_share": round(0.6 * (len(cases[c]) - k) / len(cases[c]), 4),
                                   "no_aim_share": round(0.6 * k / len(cases[c]), 4), "mean_e_m": 0.01}
                               for c, k in (("d0", 3), ("d10", 4))}, res["counters"]

    # the outcomes: VT-m = G-auto (and VS-T at d 10) in both cells (qe) but flipped places, VT-m's success and the
    # others' failure there
    qe = q | {("VTm", 10): 1, ("VST", 10): 1}
    flip = lambda fl: lambda lab, d, i: (lab == "VTm") if i in fl.get(d, ()) else cyc(qe)(lab, d, i)  # noqa: E731
    tk = lambda c: (sum(map(ord, c)) % 4, sum(map(ord, c)) // 4 % 10)  # noqa: E731  (the case's fake task)
    three = [pos[c] for c in {tk(c): c for c in cases["d0"][::-1]}.values()][:3]
    rr = {}
    for name, fl, fl_aims in (("ii", {0: {5}, 10: {5}}, aims), ("iii", {0: {5}}, aims),
                              ("iii_floor", {0: {5}}, lambda i, d: (F, [0.0006, 0.0007]) if i % 3 == 0 else None),
                              ("iii_n29", {0: {5}}, lambda i, d: (F, S) if d == 0 and 1 <= i <= 29 else None),
                              ("i_one_sided", {0: set(three), 10: {5}}, aims),
                              ("i_floor", {0: set(three), 10: {5}}, lambda i, d: None)):
        rr[name] = r = rd(*world(flip(fl), fl_aims))
        assert r["checks"]["pass"], (name, r["checks"])
    mn = lambda n: [rr[n]["measurements"][f"main/{c}"] for c in CELLS]  # noqa: E731
    # (ii): both estimates above 0 (one case each), no lower bound above 0 (the task out of >= 5 % of the resamples)
    a, b = mn("ii")
    assert (a["diff_pp"], b["diff_pp"]) == (round(100 / 113, 3), round(100 / 187, 3)) and a["lo5_one_sided"] == 0 \
        == b["lo5_one_sided"], (a, b)
    assert rr["ii"]["reading"]["main"] == {"outcome": "(ii)", "sentence": MAIN["(ii)"], "noise_floor": "показательно",
                                           "label": None}, rr["ii"]["reading"]
    b = rr["ii"]["measurements"]["B1/d10"]  # В1 at d 10 above 0 by one case: its lower bound 0, not shown
    assert b["diff_pp"] == round(100 / 187, 3) and b["lo5_one_sided"] == 0 and rr["ii"]["reading"]["B1"] == B1[False]
    # (iii): d 10 at 0; X the upper one-sided bounds; the floor informative: no label, В2 at d 0 measured
    a, b = mn("iii")
    r = rr["iii"]["reading"]["main"]
    assert b["diff_pp"] == 0 < a["diff_pp"] and r == {"outcome": "(iii)", "sentence": MAIN["(iii)"],
                                                     "X_pp": {"d0": a["hi95_one_sided"], "d10": 0.0},
                                                     "noise_floor": "показательно", "label": None}, r
    assert rr["iii"]["measurements"]["B2/d0"]["measured"]
    # (iii) on a floor that fails (1.5 mm against 1 mm) or is not computable (29 episodes): "непоказательно", В2 not
    # measured at d 0
    for n, why in (("iii_floor", "знание утонуло в невязке модели"), ("iii_n29", "не вычислимо: эпизодов с a_floor и "
                                                                                 "a_first 29 < 30")):
        r = rr[n]
        assert r["reading"]["main"]["outcome"] == "(iii)" and r["reading"]["main"]["label"] == "непоказательно" and \
            r["noise_floor"]["why"] == why and not r["measurements"]["B2/d0"]["measured"], (n, r)
    # (i) on d 0's one-sided bound alone: three flipped tasks, no flipped task in 2.5-5 % of the resamples (lo2.5 = 0
    # < lo5); d 10 one case. With no floor at all (n = 0) still (i), no label: a positive outcome withdraws the claim
    a, b = mn("i_one_sided")
    assert a["lo2.5"] == 0 < a["lo5_one_sided"] and b["lo5_one_sided"] == 0 < b["diff_pp"], (a, b)
    assert rr["i_one_sided"]["reading"]["main"]["outcome"] == "(i)", rr["i_one_sided"]["reading"]
    r = rr["i_floor"]
    assert r["noise_floor"]["n"] == 0 and r["reading"]["main"] == {"outcome": "(i)", "sentence": MAIN["(i)"],
                                                                  "noise_floor": "непоказательно", "label": None}, r

    # invalid pairs: 2 VT-m episodes at d 0 with prefix_ok false (2 of 113 <= 2 %: dropped, counted; 3 stop the
    # reading), another seed on one VS-T episode at d 10; a missing record drops as missing, never invalid
    d0 = [x for x in new if x["d"] == 0 and x["kind"] == "dynamic"]
    c1, c2 = cases["d10"][7], cases["d10"][9]
    for n, ok in ((2, True), (3, False)):
        cut = {x["case_id"] for x in d0[:n]}
        r = rd([x | {"prefix_ok": False} if x in d0[:n] else x for x in new
                if x["case_id"] != c2 or x["kind"] == "base"],
               t=[x | {"policy_seed": 1} if x["case_id"] == c1 and x["kind"] == "dynamic" else x for x in trecs])
        ip = r["checks"]["invalid_pairs"]
        assert ip["main/d0"] == {"pairs": 113, "invalid": n, "pass": ok} and ip["B1/d10"] == {
            "pairs": 186, "invalid": 1, "pass": True} and r["checks"]["complete"]["VTm/d10"]["missing"] == 1, ip
        assert ("measurements" in r) == ok == r["checks"]["pass"], r["checks"]
        if ok:
            ms = r["measurements"]
            assert (ms["main/d0"]["drops"], ms["main/d0"]["pairs"], ms["B1/d10"]["drops"], ms["main/d10"]["drops"]) \
                == ({"invalid": 2}, 111, {"seed": 1, "missing": 1}, {"missing": 1}), ms
            assert ms["main/d0"]["diff_pp"] == want("d0", "VTm", "Gauto", win, cut) and ms["B1/d10"]["diff_pp"] == \
                want("d10", "VTm", "VST", win, (c1, c2)), ms
        else:
            cc.exits(lambda: finish(r), 3)

    # the checks before reading, each alone
    mb0 = [x for x in new if x["d"] == 0 and x["kind"] == "base"]
    cl = next(x for x in d0 if x["vtm"]["s_star"] is not None)
    nb = next(x for x in d0 if x["vtm"]["s_star"] is None)
    veps = [x for x in vrecs if x["d"] == 0 and x["conf"]["pbvs"]["delta"] == cv.EPS]  # V2eps's
    for why, kw, ok in (("complete", dict(n=[x for x in new if x not in d0[:3]]), False),  # 3 of Md0's 226
                        ("complete", dict(n=[x for x in new if x not in d0[:2]]), True),
                        ("complete", dict(c=[x for x in ctrl if not (x["d"] == 10 and x["arm"] == "Gauto"
                                                                     and pos[x["case_id"]] < 4)]), False),  # 4 of 374
                        ("complete", dict(v=[x for x in vrecs if x not in veps[:3]]), False),  # V2's never fill in
                        ("e_match", dict(n=[x | {"e": 31} if x in mb0[:3] else x for x in new]), False),  # 110 < 98 %
                        ("e_match", dict(n=[x | {"e": 31} if x in mb0[:2] else x for x in new]), True),
                        ("s_star", dict(n=[x | {"vtm": x["vtm"] | {"s_star": [0.005, 0.0]}} if x is cl else x
                                           for x in new]), False),
                        ("s_star", dict(n=[x | {"vtm": x["vtm"] | {"s_star": None}} if x is cl else x for x in new]),
                         False),
                        ("s_star", dict(n=[x | {"vtm": x["vtm"] | {"s_star": [0.0, 0.0]}} if x is nb else x
                                           for x in new]), False),
                        ("band_in_conf", dict(n=[x | {"conf": x["conf"] | {"pbvs": band}} if x is cl else x
                                                 for x in new]), False),
                        ("conf", dict(n=[x | {"conf": x["conf"] | {"t_ramp": 4}} if x["d"] == 10 else x
                                         for x in new]), False)):
        r = rd(**kw)
        c = r["checks"][why]
        assert r["checks"]["pass"] == ok == ("measurements" in r) and (
            c["pass"] if "pass" in c else all(v["pass"] for v in c.values())) == ok, (why, ok, c)
        if not ok:
            cc.exits(lambda: finish(r), 3)
    r = rd([x | {"e": 31} if x in mb0[:2] else x for x in new])  # 2 Bases off: their pairs drop on all three sides
    assert all(r["measurements"][f"{m}/d0"]["drops"] == {"e": 2} for m in COMP), r["measurements"]


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_M.jsonl: checks, measurements, reading")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--controls", nargs="+", default=[f"{cc.OUT}/grid_all_C.jsonl"], metavar="JSONL",
                    help="the reading: controls lines 2 and 3 (G-auto)")
    ap.add_argument("--t", nargs="+", default=[f"{cc.OUT}/grid_all_T.jsonl"], metavar="JSONL",
                    help="the reading: the VT lines (VS-T, В1)")
    ap.add_argument("--v", nargs="+", default=[f"{cc.OUT}/grid_all_V.jsonl"], metavar="JSONL",
                    help="the reading: the V lines (PBVS, В2)")
    ap.add_argument("--vst-k", default=f"{cc.OUT}/vst_k.json", help="the reading: VS-T's arm")
    ap.add_argument("--pbvs-k", default=f"{cc.OUT}/pbvs_k.json", help="the reading: PBVS's arm")
    ap.add_argument("--band-json", default=f"{cc.OUT}/band.json", help="the reading: V2eps apart from V2, band_eps")
    ap.add_argument("--out", help="the reading: JSON too")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f) if "kind" in x]  # noqa: E731
    jload = lambda f: json.loads(Path(f).read_text()) if Path(f).exists() else None  # noqa: E731
    a.read = a.read[a.read[:1] == ["read"]:]  # "read" before the files is optional
    if bool(a.read) + a.plan + a.selftest != 1:
        ap.error("one mode: --plan, --selftest, or [read] JSONL...")
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    if a.selftest:
        selftest()
    elif a.plan:
        print(cc.tsv(plan(a.line)))
    else:
        band = jload(a.band_json)
        cv.need(band, "band.json", "delta")
        finish(read(load(a.read), load(a.controls), load(a.t), load(a.v), jload(a.vst_k), jload(a.pbvs_k), band,
                    cs.libero_names(), maxwrap.load_split(cs.SPLIT)), a.out)
