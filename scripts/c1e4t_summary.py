"""C1-E4 VS-T: the teach-by-showing servo next to the reflex (docs/superpowers/specs/2026-10-20-c1-e4-pbvs-design.md,
its journal "2026-10-21, 11:00 UTC — ветка VS-T гоняется", with §4's lambda rule and §5's pairs and checks), on pod C
after a clean vgrid (scripts/gpu_c1e4m.sh POD=C vtgrid): the runner plan per line, the lambda selection from VT1
(vst_k.json) and the reading. Reuses scripts/c1e4v_summary.py (the controls lines, the V plan, unique, need, groups)
and through it scripts/c1e4c_summary.py and scripts/c1e4m_summary.py; pure numpy, the LeRobot env; run from the
repository root.
R = results/c1-e4/controls/pod_C:
    python scripts/c1e4t_summary.py --plan --line L [--vst-k R/vst_k.json]
        L: VT1 VT2 VT3d10 VT3d20, the runner lines as c1e4c_summary --plan (phase: L in lower case); exit 3 when VT2,
        VT3d10, VT3d20 need vst_k.json, not there yet
    python scripts/c1e4t_summary.py --select-vst R/VT1_*.jsonl [--out R/vst_k.json]   exit 2: VT1 incomplete
    python scripts/c1e4t_summary.py --base-gate R/V0_*.jsonl R/VT1_*.jsonl   VT1's full Bases against V0's: exit 0, or 3
        when a Base's e differs or a case has its Base in one line only (act_hash: counted)
    python scripts/c1e4t_summary.py [read] R/grid_all_T.jsonl [--controls R/C2_*.jsonl R/C3_*.jsonl]
            [--v R/grid_all_V.jsonl] [--p2 results/c1-e4/part2] [--out R/summary_T.json]
        the checks before reading, then Т1 with no threshold and the report; a failed check prints only the checks,
        exit 3. Needs hf-libero (the task clusters)
    python scripts/c1e4t_summary.py --selftest
"""

import argparse
import contextlib
import io
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4v_summary as cv  # noqa: E402
from c1e4v_summary import OUT, P, TOL, cc, cs, e2, maxwrap, run, vservo  # noqa: E402

VSTS = tuple(vservo.VST)  # VST0.1, VST0.3, VST1.0
TIE_T = ("VST0.3", "VST0.1", "VST1.0")  # §4: ties within 0.1 mm (c1e4c_summary.TIE_M) go in this order
LINES = ("VT1", "VT2", "VT3d10", "VT3d20")  # the vtgrid order; VT1 picks lambda (vst_k.json) for the rest
CELL = {"VT2": 0, "VT3d10": 10, "VT3d20": 20}  # the pair lines and their d: line 2's (M1), line 3's (D)
# conf equal across the read records except these (cv's): VS-T adds nothing to conf (its band is orx.EPS, s* per record)
CONF_SKIP = cv.CONF_SKIP
REF_TOL = 1e-5  # m: |s*| (rounded to 1e-6 per axis) against its Base's grasp_miss_m
# measurement -> (test, VT lines it needs); the journal's Т1
MEAS = {"Т1/d0": ("pi05, M1, d = 0, 300 pairs: VS-T - Gauto", ("VT2",)),
        "Т1/d10": ("pi05, D, d = 10, 300 pairs: VS-T - Gauto", ("VT3d10",)),
        "Т1/d20": ("pi05, D, d = 20, 300 pairs: VS-T - Gauto", ("VT3d20",)),
        "Т1/pool": ("pi05, D, the d of pooled_d (10 and 20: 600 pairs; clusters: the task, its d in it): VS-T - Gauto",
                    ("VT3d10", "VT3d20"))}


def plan(line, vk=None, split=None):
    """The journal's lines, phase line.lower(), in c1e4c_summary.mlines's format, every Base full (no --base-prefix: s*
    is its first close). VT1: VST0.1 / 0.3 / 1.0 on M0's 60 calibration cases, d 0; VT2: M1, eval300, d 0, the arm of
    vst_k.json, line 2's cases, order and Base groups (asserted); VT3d10, VT3d20: D, d 10 / 20, line 3's of that d."""
    split = split or maxwrap.load_split(cs.SPLIT)
    ph = line.lower()
    if line == "VT1":
        return cc.mlines(ph, "VT1", P, "calib", VSTS, split, prefix=False)
    arm = cv.need(vk, "vst_k.json", "arm")
    assert arm in VSTS, arm
    d = CELL[line]
    lns = cc.mlines(ph, "VT2" if d == 0 else "VT3", P, "eval300", (arm,), split, d, prefix=False)
    ctl = [ln for ln in cc.plan("2" if d == 0 else "3", cv.CMODE, cv.CK, split) if f"--d {d} " in ln["args"]]
    assert cv.groups(lns, split) == cv.groups(ctl, split), f"{line}: not the controls line's cases, order and groups"
    return lns


def select_vst(recs, split):
    """The journal: lambda by §4's rule on VT1 (c1e4c_summary.select's: mean grasp miss over the cases valid for all
    three, the lowest wins, ties within 0.1 mm in TIE_T's order; flagged under 40 valid). Exit 2: VT1 incomplete or no
    case valid."""
    with mock.patch.object(cc, "VS", VSTS), mock.patch.object(cc, "TIE", TIE_T):  # ponytail: its rule, our labels
        s = cc.select(recs, P, split)
    arm = s.pop("k")
    return {"arm": arm, "lambda": vservo.VST[arm]} | s


def base_gate(recs, split):
    """Before VT2 and VT3 (the review of 2d962f9):VT1's full Bases are V0's, the same calibration cases in the same
    groups of 6, both full: e bitwise equal on every case, the same cases in both; act_hash counted (reported, not a
    gate). V0: the Base-only line (conf arms empty); VT1: arms VSTS. (pass, details)."""
    cal = run.set_ids(split, "calib")
    b = lambda arms: {r["case_id"]: r for r in recs if r.get("kind") == "base" and r["d"] == 0  # noqa: E731
                      and tuple(r["conf"]["arms"]) == arms and r["case_id"] in cal and not r.get("skipped_reason")}
    v0, t1 = b(()), b(VSTS)
    both = sorted(v0.keys() & t1.keys(), key=cal.index)
    diff = [c for c in both if v0[c]["e"] != t1[c]["e"]]
    n = {"cases": len(cal), "v0": len(v0), "vt1": len(t1), "both": len(both), "e_equal": len(both) - len(diff),
         "e_differ": diff[:10], "one_line_only": sorted(v0.keys() ^ t1.keys(), key=cal.index)[:10],
         "act_hash_equal": sum(v0[c]["act_hash"] == t1[c]["act_hash"] for c in both)}
    return bool(both) and not diff and v0.keys() == t1.keys(), n


def ref_ok(r, b):
    """A VST record's s* is its Base's: None exactly when that Base never closed, else |s*| its grasp miss."""
    s = (r.get("vst") or {}).get("s_star", "-")
    if s is None or b.get("t_close") is None:
        return s is None and b.get("t_close") is None
    return s != "-" and abs(float(np.linalg.norm(s)) - b["grasp_miss_m"]) <= REF_TOL


def read(new, ctrl, vrecs, saved, mode, k, band, pk, vk, names, split):
    """The journal's checks before reading (§5: completeness <= 1 %, the new full Base's e is line 2's / 3's on >= 98 %
    per cell, invalid pairs <= 2 % per measurement; conf equal but CONF_SKIP; vst_k.json its rule on VT1; G-auto on the
    pod; each s* its Base's), then Т1 (estimate and 95 % CI, no threshold) and the report. new: grid_all_T.jsonl; ctrl:
    controls lines 2 and 3 (Gauto, PPCL; line 2's frozen VS1.0 never read here); vrecs: grid_all_V.jsonl (V2's and
    V3's PBVS records by band.json and pbvs_k.json, report only; none without them); saved: part 2's π records ("none",
    через под). Records off the plan are counted, never read."""
    cv.need(vk, "vst_k.json", "arm")
    arm, e300 = vk["arm"], set(run.set_ids(split, "eval300"))
    by = cv.unique(new, cc.lk)
    lines = {L: plan(L, vk, split) for L in LINES}
    want = {x: L for L, lns in lines.items() for x in cc.lkeys(lns)}
    used = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    of = lambda *Ls: [r for x, r in used.items() if want[x] in Ls]  # noqa: E731
    comp = {}
    for L, lns in lines.items():  # <= 1 % missing per line; a line not run passes
        exp = cc.lkeys(lns)
        miss = len(exp - used.keys())
        comp[L] = {"expected": len(exp), "missing": miss, "run": miss < len(exp),
                   "pass": miss == len(exp) or miss <= 0.01 * len(exp) + TOL}
    cused, _, ccomp = cv.controls(ctrl, mode, k, split)
    vused = []
    if band and pk:  # report only: the PBVS episodes of V2 and V3 on their plan (V2eps apart: vk's eps)
        vw = {x for L in ("V2", "V3") for x in cv.lkeys(cv.plan(L, pk, band, split))}
        vused = [r for x, r in cv.unique(vrecs, lambda r: cv.vk(r, band)).items()
                 if x in vw and not r.get("skipped_reason")]
    with contextlib.redirect_stderr(io.StringIO()):
        try:
            got = json.loads(json.dumps(select_vst([r for x, r in by.items() if want.get(x) == "VT1"], split)))
        except SystemExit:
            got = None
    cbase = {(r["case_id"], r["d"]): r for r in cused.values() if r["kind"] == "base"}
    em = defaultdict(lambda: [0, 0])
    for r in of(*CELL):  # the new full Base's e is line 2's / 3's, >= 98 % per cell
        if r["kind"] == "base":
            c = f"{want[cc.lk(r)]}/d{r['d']}"
            em[c][0] += cbase.get((r["case_id"], r["d"]), {}).get("e", "-") == r["e"]
            em[c][1] += 1
    e_match = {c: {"equal": a, "total": n, "pass": a >= 0.98 * n - TOL} for c, (a, n) in sorted(em.items())}
    vbase = {(want[x], r["case_id"]): r for x, r in used.items() if r["kind"] == "base"}
    refs = [(r, vbase.get((want[x], r["case_id"]))) for x, r in used.items() if r["kind"] == "dynamic"]
    off_ref = Counter(want[cc.lk(r)] for r, b in refs if b is not None and not ref_ok(r, b))
    p2 = [r for r in saved if r["case_id"] in e300 and r["kind"] == "dynamic" and r["arm"] == "none"]
    conf = defaultdict(set)
    for r in list(used.values()) + list(cused.values()) + p2 + vused:
        conf[f"d{r['d']}"].add(json.dumps({f: v for f, v in r["conf"].items() if f not in CONF_SKIP}, sort_keys=True))
    mixed = {g: sorted(v) for g, v in conf.items() if len(v) > 1}
    md = ((mode or {}).get("pi05_lmax") or {}).get("mode")
    checks = {"records": len(new), "read": len(used),
              "skipped": sum(bool(r.get("skipped_reason")) for r in by.values()),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for x, r in by.items() if x not in want)),
              "complete": comp, "controls_complete": ccomp, "vst_k": {"file": vk, "recomputed": got, "pass": got == vk},
              "e_match": e_match, "s_star": {"off": dict(off_ref), "pass": not off_ref},
              "gauto_on_pod": {"mode": md, "pass": md == "S"},
              "conf": {"except": CONF_SKIP, "mixed": mixed, "pass": not mixed}}
    checks["pass"] = bool(all(v["pass"] for d in (comp, ccomp, e_match) for v in d.values()) and all(
        checks[f]["pass"] for f in ("vst_k", "s_star", "gauto_on_pod", "conf")))
    if not checks["pass"]:
        return {"checks": checks}

    # the index: the pod's VS-T, the controls' Gauto and PPCL (line 2's VS1.0 never: G(k = 1.0) is the V reading's),
    # V2's / V3's PBVS, and part 2's "none" beside them
    pod = ([r for r in of(*CELL) if r["kind"] == "dynamic"] + [r for r in vused if r["kind"] == "dynamic"]
           + [r for r in cused.values() if r["kind"] == "dynamic" and r["arm"] != "VS1.0"])
    mi = cs.mindex(pod + p2)
    one = {r["case_id"]: r for r in saved + list(cused.values()) if r["case_id"] in e300}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    M = lambda a, b, d=0: cc.mp(mi, clu, P, a, b, d, e300)  # noqa: E731
    prs = {"Т1/d0": [M(arm, "Gauto")], "Т1/d10": [M(arm, "Gauto", 10)], "Т1/d20": [M(arm, "Gauto", 20)],
           "Т1/pool": [M(arm, "Gauto", 10), M(arm, "Gauto", 20)]}
    meas, inv = {}, {}
    for n, (test, Ls) in MEAS.items():
        ps, bad = [v for p, _ in prs[n] for v in p.values()], sum(b for _, b in prs[n])
        meas[n] = {"test": test, **(e2.rnd(e2.diff_pp(ps)) or dict.fromkeys(cc.EST)), "invalid_pairs": bad}
        miss = [L for L in Ls if not comp[L]["run"]]
        if n == "Т1/pool":  # the d that ran, named; a d not run is listed, never pooled in silence
            meas[n]["pooled_d"] = [CELL[L] for L in Ls if L not in miss]
        if miss:
            meas[n] |= {"lines_not_run": miss} | (dict.fromkeys(cc.EST) if n != "Т1/pool" or miss == list(Ls) else {})
        inv[n] = bad, len(ps) + bad
    checks["invalid_pairs"] = {n: {"pairs": t, "invalid": b, "pass": b <= 0.02 * t + TOL} for n, (b, t) in inv.items()}
    if not all(v["pass"] for v in checks["invalid_pairs"].values()):
        checks["pass"] = False
        return {"checks": checks}
    return {"checks": checks, "measurements": meas,
            "report": report(mi, M, {L: of(L) for L in CELL}, arm, (pk or {}).get("arm") if vused else None, md, vk)}


def report(mi, M, cells, arm, parm, md, vk):
    """The journal's report only, per cell (d): VS-T - PBVS (V2 / V3), VS-T - PPCL, VS-T - none (part 2, через под in
    mode С); VS-T's counters (acting share and mean |e| over the episodes with a reference; the episodes and window
    steps without one); grasp miss VS-T against Gauto on the cases with both; the reference's quality: the share of
    the cell's full Bases whose first close led to grasp_ok (and how many closed at all), and the same over the Bases
    whose first close is at or after their e (where VS-T could act); vst_k.json."""
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    dm = lambda a, b, d: e2.rnd(e2.diff_pp(list(M(a, b, d)[0].values())))  # noqa: E731
    via = {"через под": md == "S"}
    rep = defaultdict(dict)
    for L, rs in cells.items():
        d, c = CELL[L], f"d{CELL[L]}"
        rep["vst-pbvs"][c] = dm(arm, parm, d) if parm else {"pbvs": None, "why": "no V2 / V3 records read"}
        rep["vst-ppcl"][c] = dm(arm, "PPCL", d)
        rep["vst-none"][c] = (dm(arm, "none", d) or {}) | via
        dyn = [r for r in rs if r["kind"] == "dynamic"]
        st = [r["vst"] for r in dyn if r["vst"]["s_star"] is not None]
        win = [s for s in st if s["steps_window"]]
        me = [s["sum_e"] / s["steps_window"] for s in win]
        rep["vst_counters"][c] = {
            "n": len(dyn), "no_target_episodes": len(dyn) - len(st), "no_target_steps": sum(r["vst"]["no_target"]
                                                                                           for r in dyn),
            "window": len(win), "acting_share": mean([s["steps_acting"] / s["steps_window"] for s in win]),
            "mean_e_m": mean(me), "mean_e_p5_p95": [round(float(v), 4) for v in np.percentile(me, [5, 95])] if me
            else None, "max_e_m": mean([s["max_e"] for s in win]), "flips_per_episode": mean([s["flips"] for s in win]),
            "flipped_share": mean([s["flips"] > 0 for s in win])}
        x, y = mi.get((P, "dynamic", arm, d), {}), mi.get((P, "dynamic", "Gauto", d), {})
        both = [k for k in sorted(x.keys() & y.keys()) if None not in (x[k]["grasp_miss_m"], y[k]["grasp_miss_m"])]
        rep["grasp_miss"][c] = {"n": len(both), arm: mean([x[k]["grasp_miss_m"] for k in both]),
                                "Gauto": mean([y[k]["grasp_miss_m"] for k in both]),
                                f"{arm}-Gauto": mean([x[k]["grasp_miss_m"] - y[k]["grasp_miss_m"] for k in both])}
        bs = [r for r in rs if r["kind"] == "base"]
        closed = [b for b in bs if b.get("t_close") is not None]
        ok = sum(b.get("grasp_ok") is True for b in closed)
        rep["reference"][c] = {"bases": len(bs), "closed": len(closed), "closed_grasp_ok": ok,
                               "share": round(ok / len(closed), 4) if closed else None}
        # the review of 2d962f9, report only: the Bases whose first close is at or after their event step e (where
        # VS-T could act)
        late = [b for b in closed if b.get("e") is not None and b["t_close"] >= b["e"]]
        ok = sum(b.get("grasp_ok") is True for b in late)
        rep["reference_close_after_e"][c] = {"bases": len(late), "closed_grasp_ok": ok,
                                             "share": round(ok / len(late), 4) if late else None}
    return dict(rep) | {"vst_k": vk}


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 VS-T checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


# ------------------------------------------------------------------ selftest

def selftest():
    split = maxwrap.load_split(cs.SPLIT)
    e3 = cs.sources(run.set_ids(split, "eval300"))
    vk = {"arm": "VST0.3"}
    l1 = plan("VT1", split=split)
    assert [(ln["id"], ln["phase"], cc.n_of(ln)) for ln in l1] == [("VT1_pi05_calib_plus_d0", "vt1", 168),
                                                                   ("VT1_pi05_calib_pro_d0", "vt1", 72)], l1
    assert l1[0]["args"] == "--brain pi05 --source plus --set calib --arms=VST0.1,VST0.3,VST1.0 --d 0", l1[0]["args"]
    c2, c3 = cc.plan("2", cv.CMODE, cv.CK, split), cc.plan("3", cv.CMODE, cv.CK, split)
    for L, d in CELL.items():
        lns = plan(L, vk, split)
        tag = "VT2" if d == 0 else "VT3"
        assert [(ln["id"], ln["phase"], cc.n_of(ln), ln["args"]) for ln in lns] == [
            (f"{tag}_pi05_eval300_{s}_d{d}", L.lower(), 2 * len(e3[s]),
             f"--brain pi05 --source {s} --set eval300 --arms=VST0.3 --d {d}") for s in ("plus", "pro")], lns
        ctl = [ln for ln in (c2 if d == 0 else c3) if f"--d {d} " in ln["args"]]
        assert cv.groups(lns, split) == cv.groups(ctl, split) and len(cv.groups(lns, split)) == 51, L
        assert [x for ln in lns for x in cv.ids(ln, split)] == [x for s in ("plus", "pro") for x in e3[s]], L
    with mock.patch.object(cc, "plan", lambda *a: [ln | {"keys": {x for x in ln["keys"] if x[2] != e3["plus"][7]}}
                                                   for ln in c2]):
        try:  # VT2 off line 2's groups (one case short there): the plan refuses
            plan("VT2", vk, split)
            raise RuntimeError("VT2 off line 2's groups was planned")
        except AssertionError:
            pass
    for L in CELL:
        cc.exits(lambda: plan(L, None, split), 3)
    pk, band = {"arm": "PBVS0.3"}, {"delta": 0.026}
    other = [ln for L in cv.LINES for ln in cv.plan(L, pk, band, split)] + c2 + c3  # the controls' other ids: C*
    ours = [ln for L in LINES for ln in plan(L, vk, split)]
    assert len({ln["id"] for ln in ours + other}) == len(ours + other) and not cc.lkeys(ours) & cc.lkeys(other)
    assert all("--base-prefix" not in ln["args"] and "--band" not in ln["args"] and "--kbar" not in ln["args"]
               for ln in ours)
    selftest_select(split)
    selftest_gate(split)
    selftest_read(split)
    print("selftest ok")


def selftest_select(split):
    """§4's rule on VT1: mean grasp miss over the cases valid for all three, ties within 0.1 mm in the order 0.3, 0.1,
    1.0; flagged under 40 valid; exit 2 when a Base is missing."""
    cal = run.set_ids(split, "calib")

    def recs(miss, n_bad=0):
        out = [{"case_id": c, "brain": P, "kind": "base", "arm": "none", "d": 0, "conf": {"arms": list(VSTS)}}
               for c in cal]
        for j, c in enumerate(cal):
            for a in VSTS:
                out.append({"case_id": c, "brain": P, "kind": "dynamic", "arm": a, "d": 0, "e": 30,
                            "prefix_ok": j >= n_bad, "replay_miss": None, "skipped_reason": None,
                            "grasp_miss_m": miss[a] + 1e-4 * (j % 3)})
        return out

    s = select_vst(recs({"VST0.1": 0.0100, "VST0.3": 0.01009, "VST1.0": 0.02}), split)  # 0.09 mm: a tie, 0.3 first
    assert s == {"arm": "VST0.3", "lambda": 0.3, "scores": {"VST0.1": 0.0101, "VST0.3": 0.01019, "VST1.0": 0.0201},
                 "n_valid": 60, "n_cases": 60, "flag_lt40": False}, s
    s = select_vst(recs({"VST0.1": 0.0100, "VST0.3": 0.01011, "VST1.0": 0.02}, n_bad=21), split)  # 0.11 mm: 0.1 wins
    assert s["arm"] == "VST0.1" and s["lambda"] == 0.1 and s["n_valid"] == 39 and s["flag_lt40"], s
    s = select_vst(recs({"VST0.1": 0.012, "VST0.3": 0.011, "VST1.0": 0.0095}), split)
    assert s["arm"] == "VST1.0", s
    cc.exits(lambda: select_vst(recs({a: 0.01 for a in VSTS})[1:], split), 2)


def selftest_gate(split):
    """VT1's full Bases against V0's: equal e on every case passes (act_hash counted: 3 differ here); one e off, or a
    case in one line only, stops; Dynamic records and other lines' Bases are not read."""
    cal = run.set_ids(split, "calib")
    base = lambda c, arms, j, **kw: {"case_id": c, "brain": P, "kind": "base", "arm": "none", "d": 0,  # noqa: E731
                                     "e": 20 + j % 9, "act_hash": f"h{j}", "conf": {"arms": list(arms)},
                                     "skipped_reason": None} | kw
    v0 = [base(c, (), j) for j, c in enumerate(cal)]
    t1 = [base(c, VSTS, j, act_hash=f"h{j}" if j % 20 else "x") for j, c in enumerate(cal)]
    dyn = [{"case_id": c, "brain": P, "kind": "dynamic", "arm": "VST0.3", "d": 0, "e": 99, "act_hash": "y",
            "conf": {"arms": list(VSTS)}} for c in cal]
    other = [base(c, ("PBVS0.3",), j, e=99) for j, c in enumerate(cal)]  # V1's Bases (cut, other e): not read
    ok, n = base_gate(v0 + t1 + dyn + other, split)
    assert ok and n == {"cases": 60, "v0": 60, "vt1": 60, "both": 60, "e_equal": 60, "e_differ": [],
                        "one_line_only": [], "act_hash_equal": 57}, n
    ok, n = base_gate(v0 + [r | {"e": r["e"] + 1} if r["case_id"] == cal[7] else r for r in t1], split)
    assert not ok and n["e_differ"] == [cal[7]] and n["e_equal"] == 59, n
    ok, n = base_gate(v0 + t1[1:], split)
    assert not ok and n["one_line_only"] == [cal[0]] and n["e_equal"] == 59, n
    assert not base_gate(t1, split)[0] and not base_gate([], split)[0]


def selftest_read(split):
    """The reading on a synthetic pod: controls lines 2 and 3 (mode С, k_π = 1.0), VT1 -> vst_k.json, VT2, VT3d10,
    VT3d20, V2's and V3's PBVS, part 2's "none". Success by the case's place in cycles of 100 (pct: the arm's share in
    %, by (arm, d) or arm), so the differences on the 300 cases are exact; task clusters from fake LIBERO names; one
    policy seed per case. A Base whose place is 0 mod 50 never closed (its VS-T has no reference); a Base's grasp_ok
    false on places 0 mod 4."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    cal, e300 = run.set_ids(split, "calib"), run.set_ids(split, "eval300")
    pos = {c: j for j, c in enumerate(split["eval"])} | {c: j for j, c in enumerate(cal)}
    cf = {"brain": P, "revision": "r", "precision": "fp32", "s": 10, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01, "R": 360,
          "split": split["sha256"]}
    mode = {"pi05_lmax": {"mode": "S"}, "oftplus_lmax": {"mode": "S"}, "pi05_bench": {"mode": "P"}}
    k = {P: {"k": "VS1.0"}}
    band, pk = {"model": "A", "G": [0.0025, 0.0025], "B": [0.79, 0.79], "delta": 0.0262}, {"arm": "PBVS0.3"}
    miss = {"VST0.1": 0.012, "VST0.3": 0.0105, "VST1.0": 0.02, "Gauto": 0.011}

    def rec(x, pct, pbvs=None, gpu=True):  # x: lk
        _, b, c, kind, a, d, _, arms = x
        h, closed = sum(map(ord, c)), pos[c] % 50 != 0
        s = list(run.MAX_STEPS)[h % 4]
        r = {"brain": b, "case_id": c, "kind": kind, "arm": a, "method": cs.method(a), "d": d,
             "policy_seed": 195 + pos[c],
             "conf": cf | {"d": d, "arms": list(arms), "kbar": 0.306 if "Gauto" in arms else None}
             | ({"gpu": "NVIDIA GeForce RTX 4090"} if gpu else {}) | ({"pbvs": pbvs} if pbvs else {}),
             "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None,
             "distance_m": (0.06, 0.12)[h % 2], "e": 30, "prefix_ok": None if kind == "base" else True,
             "replay_miss": None, "skipped_reason": None, "calls_sched": 20, "calls_trig": 0,
             "success": None if kind == "base" else pos[c] % 100 < pct.get((a, d), pct.get(a, 50)),
             "t_close": ({0: 25, 1: 30}.get(pos[c] % 7, 40) if kind == "base" else 40) if kind == "dynamic" or closed
             else None,
             "grasp_ok": pos[c] % 4 != 0,
             "grasp_miss_m": miss.get(a, 0.015) if kind == "dynamic" or closed else None}
        if kind == "base" and set(arms) & set(VSTS):  # VS-T's full Base: s* its offset at the close
            r["grasp_miss_m"] = 0.008 if closed else None
        if a in VSTS:
            ref = closed and [0.0048, -0.0064]  # |s*| = 8 mm: its Base's grasp miss
            r["vst"] = {"s_star": ref or None, "steps_window": 20, "steps_acting": 5 if ref else 0,
                        "sum_e": 0.2 if ref else 0.0, "max_e": 0.03 if ref else 0.0,
                        "flips": int(pos[c] % 2) if ref else 0, "no_target": 0 if ref else 20}
        return r

    def world(pct):
        ctrl = [rec(x, pct) for L in "23" for x in sorted(cc.lkeys(cc.plan(L, mode, k, split)))]
        vt1 = [rec(x, pct) for x in sorted(cc.lkeys(plan("VT1", split=split)))]
        vk = json.loads(json.dumps(select_vst(vt1, split)))
        new = vt1 + [rec(x, pct) for L in CELL for x in sorted(cc.lkeys(plan(L, vk, split)))]
        vrecs = [rec(x[:8], pct, band) for L in ("V2", "V3")
                 for x in sorted(cv.lkeys(cv.plan(L, pk, band, split)))]
        vrecs += [r | {"success": r["success"] if r["kind"] == "base" else pos[r["case_id"]] % 100 < 70}
                  for x in sorted(cv.lkeys(cv.plan("V2eps", pk, band, split))) for r in [rec(x[:8], pct, band | {
                      "delta": cv.EPS})]]  # V2eps (band > 15 mm): its PBVS at 70 %, never read as V2's
        saved = [rec(("m", P, c, kd, "none", d, False, ()), pct, gpu=False) for c in e300 for d in (0, 10, 20)
                 for kd in ("base", "dynamic")]
        return new, ctrl, vrecs, saved, vk

    pct = {("VST0.3", 0): 58, ("VST0.3", 10): 52, ("VST0.3", 20): 45, "Gauto": 50, "PPCL": 45, "none": 30,
           ("PBVS0.3", 0): 55, ("PBVS0.3", 10): 60, ("PBVS0.3", 20): 40, "VS1.0": 20}
    new, ctrl, vrecs, saved, vk = world(pct)
    assert vk["arm"] == "VST0.3" and vk["n_valid"] == 60, vk
    rd = lambda n=new, c=ctrl, v=vrecs, m=mode, p=pk, kk=vk: read(  # noqa: E731
        n, c, v, saved, m, k, band, p, kk, names, split)
    probe = rec(("m", P, e300[0], "dynamic", "VST1.0", 0, False, ("VST1.0",)), pct)  # off the plan
    res = rd(new + [probe] + new[:5])  # new[:5]: an identical copy
    ch, ms, rep = res["checks"], res["measurements"], res["report"]
    assert ch["pass"] and ch["off_plan"] == {f"{P}/VST1.0": 1} and ch["read"] == len(new), ch
    assert [ch["complete"][L]["expected"] for L in LINES] == [240, 600, 600, 600] and ch["s_star"]["pass"], ch
    assert set(ch["e_match"]) == {"VT2/d0", "VT3d10/d10", "VT3d20/d20"} and ch["vst_k"]["pass"], ch
    # Т1: estimates and 95% CIs, no verdict, no threshold
    assert list(ms) == list(MEAS) and all(not {"pass", "outcome"} & set(v) and not re.search("[<>≤≥]", v["test"])
                                          for v in ms.values()), ms
    want = {"Т1/d0": 8, "Т1/d10": 2, "Т1/d20": -5, "Т1/pool": -1.5}
    assert {n: ms[n]["diff_pp"] for n in want} == want and ms["Т1/pool"]["pairs"] == 600, ms
    assert all(ms[n]["pairs"] == 300 and ms[n]["invalid_pairs"] == 0 for n in want if n != "Т1/pool"), ms
    assert ms["Т1/pool"]["pooled_d"] == [10, 20] and ms["Т1/d0"]["lo2.5"] <= 8 <= ms["Т1/d0"]["hi97.5"], ms
    # the report: VS-T - PBVS, - PPCL, - none (через под), counters, grasp miss, the reference's quality
    assert {c: v["diff_pp"] for c, v in rep["vst-pbvs"].items()} == {"d0": 3, "d10": -8, "d20": 5}, rep["vst-pbvs"]
    assert {c: v["diff_pp"] for c, v in rep["vst-ppcl"].items()} == {"d0": 13, "d10": 7, "d20": 0}, rep["vst-ppcl"]
    assert {c: (v["diff_pp"], v["через под"]) for c, v in rep["vst-none"].items()} == {
        "d0": (28, True), "d10": (22, True), "d20": (15, True)}, rep["vst-none"]
    closed = [j for j in range(300) if j % 50]  # the places whose Base closed: 294; flips on odd places
    fl = round(sum(j % 2 for j in closed) / len(closed), 4)
    assert rep["vst_counters"]["d10"] == {"n": 300, "no_target_episodes": 6, "no_target_steps": 120, "window": 294,
                                         "acting_share": 0.25, "mean_e_m": 0.01, "mean_e_p5_p95": [0.01, 0.01],
                                         "max_e_m": 0.03, "flips_per_episode": fl, "flipped_share": fl}, rep
    assert rep["grasp_miss"]["d0"] == {"n": 300, "VST0.3": 0.0105, "Gauto": 0.011, "VST0.3-Gauto": -0.0005}, rep
    ok = sum(j % 4 != 0 for j in closed)
    assert rep["reference"]["d20"] == {"bases": 300, "closed": 294, "closed_grasp_ok": ok,
                                       "share": round(ok / 294, 4)} and ok == 222, rep["reference"]
    late = [j for j in closed if j % 7]  # a Base on a place 0 mod 7 closed at 25, before its e = 30; 1 mod 7 at 30
    ok = sum(j % 4 != 0 for j in late)
    assert rep["reference_close_after_e"]["d0"] == {"bases": len(late), "closed_grasp_ok": ok,
                                                    "share": round(ok / len(late), 4)} and len(late) == 252, rep
    assert rep["vst_k"] == vk
    # line 2's frozen VS1.0 is never read here: its success leaves everything as it was
    r2 = rd(c=[r | {"success": not r["success"]} if r["arm"] == "VS1.0" else r for r in ctrl])
    assert r2["measurements"] == ms and r2["report"] == rep
    # no V records (band.json or pbvs_k.json missing): VS-T - PBVS not read, the rest as it was
    r2 = rd(v=[], p=None)
    assert r2["checks"]["pass"] and r2["measurements"] == ms and r2["report"]["vst-pbvs"]["d0"]["pbvs"] is None, r2
    assert {kk: v for kk, v in r2["report"].items() if kk != "vst-pbvs"} == {
        kk: v for kk, v in rep.items() if kk != "vst-pbvs"}

    # the checks before reading, each alone
    v2 = sorted((r for r in new if r["d"] == 0 and r["arm"] == "VST0.3" and r["case_id"] in set(e300)),
                key=lambda r: pos[r["case_id"]])  # VT2's VS-T episodes in the set's order
    cl = next(r for r in v2 if r["vst"]["s_star"] is not None)  # a case whose Base closed
    for why, kw in {"complete": dict(n=[r for r in new if r not in v2[:7]]),  # 7 of VT2's 600 records (> 1 %)
                    "vst_k": dict(kk=vk | {"arm": "VST0.1"}),
                    "s_star": dict(n=[r | {"vst": r["vst"] | {"s_star": [0.005, 0.0]}} if r is cl else r
                                      for r in new]),
                    "conf": dict(n=[r | {"conf": r["conf"] | {"t_ramp": 4}} if r["d"] == 20 else r for r in new]),
                    "gauto_on_pod": dict(m=mode | {"pi05_lmax": {"mode": "P"}}),
                    "controls_complete": dict(c=[r for r in ctrl if not (r["d"] == 10 and r["arm"] == "PPCL" and
                                                                         pos[r["case_id"]] < 20)])}.items():
        with contextlib.redirect_stderr(io.StringIO()):
            ch = rd(**kw)["checks"]
        c = ch[why]
        assert not ch["pass"] and (not c["pass"] if "pass" in c else not all(v["pass"] for v in c.values())), (why, c)
    # a reference without its Base's close, and one where the Base closed but s* is None: both off
    nb = next(r for r in v2 if r["vst"]["s_star"] is None)
    for old, bad in ((nb, [0.0, 0.0]), (cl, None)):
        ch = rd([r | {"vst": r["vst"] | {"s_star": bad}} if r is old else r for r in new])["checks"]
        assert ch["s_star"] == {"off": {"VT2": 1}, "pass": False} and not ch["pass"], ch["s_star"]
    # 6 of VT2's 300 cases with another Base e: 98 % pass and the pairs drop (2 % of Т1/d0); 7 fail. 6 VS-T episodes
    # with prefix_ok false: 2 % pass, 7 stop the reading
    for n, ok in ((6, True), (7, False)):
        cut = set(e300[:n])
        r = rd([x | {"e": 31} if x["case_id"] in cut and x["d"] == 0 and x["conf"]["arms"] == ["VST0.3"] else x
                for x in new])
        assert r["checks"]["e_match"]["VT2/d0"] == {"equal": 300 - n, "total": 300, "pass": ok} and (
            "measurements" in r) == ok, r["checks"]["e_match"]
        if ok:
            assert r["measurements"]["Т1/d0"]["invalid_pairs"] == 6 and r["measurements"]["Т1/d0"]["pairs"] == 294
        r = rd([x | {"prefix_ok": False} if x in v2 and x["case_id"] in cut else x for x in new])
        assert r["checks"]["invalid_pairs"]["Т1/d0"] == {"pairs": 300, "invalid": n, "pass": ok} and (
            "measurements" in r) == ok == r["checks"]["pass"], r["checks"].get("invalid_pairs")
        if not ok:
            cc.exits(lambda: finish(r), 3)
    # VT3d20 not run (the last line, cut): Т1 per d for the d that ran, the pool over it alone and named
    r = rd([x for x in new if not (x["d"] == 20)])
    ms2 = r["measurements"]
    assert r["checks"]["pass"] and not r["checks"]["complete"]["VT3d20"]["run"], r["checks"]
    assert ms2["Т1/d20"]["diff_pp"] is None and ms2["Т1/d20"]["lines_not_run"] == ["VT3d20"] and ms2["Т1/pool"] == ms2[
        "Т1/d10"] | {"test": MEAS["Т1/pool"][0], "pooled_d": [10], "lines_not_run": ["VT3d20"]}, ms2
    assert r["report"]["reference"]["d20"]["bases"] == 0 and r["report"]["vst_counters"]["d20"]["n"] == 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--select-vst", nargs="+", metavar="JSONL", help="VT1's records -> vst_k.json")
    g.add_argument("--base-gate", nargs="+", metavar="JSONL", help="V0's and VT1's records: exit 0, or 3")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_T.jsonl: checks, measurements, report")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--vst-k", default=f"{OUT}/vst_k.json", help="vst_k.json (--plan, the reading)")
    ap.add_argument("--band-json", default=f"{OUT}/band.json", help="the reading: V's band.json (VS-T - PBVS)")
    ap.add_argument("--pbvs-k", default=f"{OUT}/pbvs_k.json", help="the reading: V's pbvs_k.json (VS-T - PBVS)")
    ap.add_argument("--mode", default=f"{OUT}/mode.json", help="the controls' mode.json")
    ap.add_argument("--k", default=f"{OUT}/k_select.json", help="the controls' k_select.json")
    ap.add_argument("--controls", nargs="+", default=[f"{OUT}/grid_all_C.jsonl"], metavar="JSONL",
                    help="the reading: the controls records of lines 2 and 3")
    ap.add_argument("--v", nargs="+", default=[f"{OUT}/grid_all_V.jsonl"], metavar="JSONL",
                    help="the reading: the V lines' records (VS-T - PBVS; skipped when absent)")
    ap.add_argument("--p2", default=str(cc.P2), help="the reading: part 2's records (read only)")
    ap.add_argument("--out", help="--select-vst: default --vst-k; the reading: JSON too")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs if Path(f).exists() for x in e2.load(f) if "kind" in x]  # noqa: E731
    jload = lambda f: json.loads(Path(f).read_text()) if Path(f).exists() else None  # noqa: E731
    a.read = a.read[a.read[:1] == ["read"]:]  # "read" before the files is optional
    if bool(a.read) + sum(map(bool, (a.plan, a.select_vst, a.base_gate, a.selftest))) != 1:
        ap.error("one mode: an option of the group, or [read] JSONL...")
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    split = maxwrap.load_split(cs.SPLIT)
    if a.selftest:
        selftest()
    elif a.plan:
        print(cc.tsv(plan(a.line, jload(a.vst_k), split)))
    elif a.select_vst:
        s = select_vst(load(a.select_vst), split)
        Path(a.out or a.vst_k).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out or a.vst_k).write_text(json.dumps(s, indent=1) + "\n")
        print(e2.dump(s))
    elif a.base_gate:
        ok, n = base_gate(load(a.base_gate), split)
        print(e2.dump({"vt1_bases_as_v0": n, "pass": ok}))
        print(f"C1-E4 VS-T base gate: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    else:
        saved = load([Path(a.p2) / f for f in cc.SAVED["pi05_lmax"]])
        finish(read(load(a.read), load(a.controls), load(a.v), saved, jload(a.mode), jload(a.k), jload(a.band_json),
                    jload(a.pbvs_k), jload(a.vst_k), cs.libero_names(), split), a.out)
