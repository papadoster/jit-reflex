"""C1-E1 verdict: applies §5 of docs/superpowers/specs/2026-09-27-c1-e1-offline-design.md to the JSON lines of
`c1_scout_bench.py --mode e1`, with no manual decisions. Prints the read gate first and stops if it fails.
    python scripts/c1_e1_summary.py results/c1-e1/e1.jsonl        # writes summary.json next to the input
    python scripts/c1_e1_summary.py --selftest
"""

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

G, EPS, RHO_MIN, SIGN_MIN = 0.011, 1e-3, 0.3, 0.6  # m per action unit per step; spec §4-§5 thresholds
SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]


def headroom_cat(r):
    return "ЗАПАС ЕСТЬ" if r >= 0.2 else "СЛАБЫЙ" if r >= 0.05 else "НЕТ ЗАПАСА"


def rho(a0, d, c):  # E1b form on executed actions: share of the fresh-call change that the correction recovers
    true, base = np.clip(a0 + d, -1, 1), np.clip(a0, -1, 1)
    e_pred = np.linalg.norm(true - base)
    if e_pred < EPS:
        return None  # no executed response: spec §4 degenerate case
    return 1 - np.linalg.norm(true - np.clip(a0 + np.clip(c, -1, 1), -1, 1)) / e_pred


def rows(states):  # one row per (state, shift) with every statistic the rules and the report need
    out = []
    for st in states:
        a0, cl = np.array(st["a0"]), (st["suite"], st["task"])
        for kind, recs, key in (("arm", st["arm"], "dF0"), ("obj", st["obj"], "dO0")):
            by = {(r["ax"], r["sg"], r["mm"]): r for r in recs if not r.get("ik_fail")}
            for (ax, sg, mm), r in by.items():
                u = np.zeros(3)
                u[ax] = sg
                d, sgn = np.array(r[key]), (-1 if kind == "arm" else 1)  # arm: restoring = against u; obj: following
                row = {"kind": kind, "cl": cl, "suite": st["suite"], "stage": st["stage"], "k": st["k"], "mm": mm,
                       "resp": float(np.linalg.norm(d)), "true_sign": sgn * float(d[:3] @ u) > 0}
                for h in ("10", "25", "50"):
                    row[f"R{h}"] = sgn * G * float(np.array(r["S"][h]) @ u) / (mm / 1000)
                p2, m2 = by.get((ax, 1, 2)), by.get((ax, -1, 2))
                cands = {}
                if p2 and m2:  # tangent by central difference at +-2 mm, per mm
                    cands["vis"] = sg * mm * (np.array(p2[key]) - np.array(m2[key])) / 4
                if kind == "arm":
                    cands["s"] = np.array(r["Cs"])
                    if "nFS10" in r:
                        row["img_dom"] = r["nFI10"] < r["nFS10"]
                for name, c in cands.items():
                    row[f"rho_{name}"] = rho(a0, d, c)
                    row[f"L_{name}"] = float(np.linalg.norm(d - c) / np.linalg.norm(d)) if row["resp"] >= EPS else None
                    row[f"agree_{name}"] = float(c @ d) > 0
                    row[f"csign_{name}"] = sgn * float(c[:3] @ u) > 0
                out.append(row)
    return out


def med(rs, key, **where):
    x = [r[key] for r in rs if r.get(key) is not None and all(r[k] == v for k, v in where.items())]
    return (float(np.median(x)), len(x)) if x else (None, 0)


def share(rs, key, **where):
    x = [bool(r[key]) for r in rs if r.get(key) is not None and all(r[k] == v for k, v in where.items())]
    return (float(np.mean(x)), len(x)) if x else (None, 0)


def boot(rs, key, stat=np.median, n=2000, **where):  # 95% CI, clusters = tasks (whole rollouts resampled)
    by = defaultdict(list)
    for r in rs:
        if r.get(key) is not None and all(r[k] == v for k, v in where.items()):
            by[r["cl"]].append(r[key])
    cls, rng, vals = list(by), np.random.default_rng(0), []
    for _ in range(n):
        pick = rng.integers(len(cls), size=len(cls))
        vals.append(stat(np.concatenate([by[cls[i]] for i in pick])))
    return [round(float(np.percentile(vals, q)), 3) for q in (2.5, 97.5)]


def pooled_mm(rs, key, mms, **where):  # median over several |delta| at once (headroom pools 1 and 2 cm)
    return med([dict(r, mm=0) for r in rs if r["mm"] in mms], key, mm=0, **where)


def go(rs, name, kind, extra_ok=True):
    m, n = med(rs, f"rho_{name}", kind=kind, mm=10)
    per = {s: med(rs, f"rho_{name}", kind=kind, mm=10, suite=s)[0] for s in SUITES}
    n_ok = sum(v is not None and v >= RHO_MIN for v in per.values())
    ok = m is not None and m >= RHO_MIN and n_ok >= 3 and extra_ok
    return ok, {"median_rho_1cm": m, "n": n, "per_suite": per, "suites_passing": n_ok,
                "ci95": boot(rs, f"rho_{name}", kind=kind, mm=10) if n else None}


def summarize(lines):
    rolls = [l["e1_rollout"] for l in lines if "e1_rollout" in l]
    states = [l["e1_state"] for l in lines if "e1_state" in l]
    succ = sum(r["success"] for r in rolls)
    res = {"read_gate": {"rollouts": len(rolls), "states": len(states), "successes": succ,
                         "pass": len(rolls) == 32 and len(states) == 256 and succ >= 20}}
    if not res["read_gate"]["pass"]:
        return res  # spec §5: results are not read
    rs = rows(states)
    ik_fail = sum(r.get("ik_fail", False) for st in states for r in st["arm"])
    no_resp = {k: sum(r["resp"] < EPS for r in rs if r["kind"] == k) for k in ("arm", "obj")}
    hr = {}
    for kind in ("arm", "obj"):
        m, n = pooled_mm(rs, "R10", (10, 20), kind=kind)
        hr[kind] = {"median_R10_1-2cm": m, "n": n, "category": headroom_cat(m),
                    "ci95": boot([dict(r, mm=0) for r in rs if r["mm"] in (10, 20)], "R10", kind=kind, mm=0),
                    "report_R25_R50": {h: pooled_mm(rs, h, (10, 20), kind=kind)[0] for h in ("R25", "R50")},
                    "report_R10_5cm": med(rs, "R10", kind=kind, mm=50)[0]}
    agree_s = share(rs, "agree_s", kind="arm", mm=10)[0]
    arm_ok, obj_ok = hr["arm"]["category"] != "НЕТ ЗАПАСА", hr["obj"]["category"] != "НЕТ ЗАПАСА"
    g_s, d_s = go(rs, "s", "arm", arm_ok and agree_s is not None and agree_s >= SIGN_MIN)
    g_v, d_v = go(rs, "vis", "arm", arm_ok)
    g_o, d_o = go(rs, "vis", "obj", obj_ok)
    d_s["sign_agreement_1cm"] = agree_s
    passed = [n for n, g in (("GO-state", g_s), ("GO-vis-arm", g_v), ("GO-obj", g_o)) if g]
    verdict = "НЕТ ЗАПАСА" if not (arm_ok or obj_ok) else ("GO: " + ", ".join(passed) if passed else "KILL")
    phase = {"by_stage": {s: share(rs, "img_dom", kind="arm", stage=s) for s in ("far", "near", "closed")},
             "by_tertile": {i: share([dict(r, tert=min(2, 3 * r["k"] // 8)) for r in rs], "img_dom", kind="arm",
                                     tert=i) for i in range(3)}}
    lin = {f"{kind}_{name}": {mm: med(rs, f"L_{name}", kind=kind, mm=mm)[0] for mm in (1, 5, 10, 20)}
           for kind, name in (("arm", "s"), ("arm", "vis"), ("obj", "vis"))}
    signs = {f"{kind}_{name}": {mm: {"true": share(rs, "true_sign", kind=kind, mm=mm)[0],
                                     "corr": share(rs, f"csign_{name}", kind=kind, mm=mm)[0],
                                     "agree": share(rs, f"agree_{name}", kind=kind, mm=mm)[0]} for mm in (1, 5, 10, 20)}
             for kind, name in (("arm", "s"), ("arm", "vis"), ("obj", "vis"))}
    ms = {k: float(np.median([st["ms"][k] for st in states])) for k in ("call", "J", "render")}
    fd = (ms["render"] + ms["call"]) / ms["call"]  # one direction, one-sided
    cost = {"ms_median": ms, "calls_per_position": {"J_s (vmap VJP)": ms["J"] / ms["call"],
            "J_vis 2 directions one-sided": 2 * fd, "J_vis 2 directions central": 4 * fd},
            "aten_ops": [st["ops"] for st in states if st["ops"]][:1]}
    rho_s, rho_v = d_s["median_rho_1cm"], d_v["median_rho_1cm"]
    pred = {"P1 near>far image share": (phase["by_stage"]["near"][0] or 0) > (phase["by_stage"]["far"][0] or 0),
            "P2 rho_s<rho_vis": None if None in (rho_s, rho_v) else rho_s < rho_v,
            "P3 R_obj<0.2": hr["obj"]["median_R10_1-2cm"] < 0.2,
            "P4 L_vis 2cm>5mm": (lin["arm_vis"][20] or 0) > (lin["arm_vis"][5] or 0),
            "P5 J_s 2-3 calls": 2 <= cost["calls_per_position"]["J_s (vmap VJP)"] <= 3}
    res |= {"verdict": verdict, "headroom": hr, "GO-state": d_s, "GO-vis-arm": d_v, "GO-obj": d_o,
            "report": {"phase": phase, "linearity_L": lin, "signs": signs, "cost": cost, "predictions": pred,
                       "rho_2cm": {n: med(rs, f"rho_{n}", kind=k, mm=20)[0] for k, n in (("arm", "s"), ("arm", "vis"))}
                       | {"obj_vis": med(rs, "rho_vis", kind="obj", mm=20)[0]},
                       "ik_fail": ik_fail, "no_response": no_resp,
                       "stages": {s: sum(st["stage"] == s for st in states) for s in ("far", "near", "closed")}}}
    return res


def selftest():  # a policy that is exactly linear and fully restoring must give rho_vis = 1 and R10 = 1
    def rec(ax, sg, mm, gain):
        u = np.zeros(3)
        u[ax] = sg
        d = np.r_[-gain * mm * u, np.zeros(4)]  # restoring and linear in the signed shift
        s10 = -(mm / 1000) / G * u  # the ten actions sum to undoing the whole shift
        return {"ax": ax, "sg": sg, "mm": mm, "dF0": list(d), "S": {h: list(s10) for h in ("10", "25", "50")},
                "Cs": list(d), "nFS10": 1.0, "nFI10": 2.0}
    arm = [rec(ax, sg, mm, 0.01) for ax in (0, 1) for sg in (1, -1) for mm in (1, 2, 5, 10, 20, 50)]
    st = {"suite": "libero_spatial", "task": 0, "k": 0, "stage": "far", "a0": [0.0] * 7, "arm": arm, "obj": []}
    rs = rows([st])
    r10 = [r for r in rs if r["mm"] == 10]
    assert all(abs(r["R10"] - 1) < 1e-9 and abs(r["rho_vis"] - 1) < 1e-9 and abs(r["rho_s"] - 1) < 1e-9 for r in r10)
    assert all(r["true_sign"] and r["agree_vis"] and not r["img_dom"] for r in r10)
    assert headroom_cat(0.2) == "ЗАПАС ЕСТЬ" and headroom_cat(0.049) == "НЕТ ЗАПАСА"
    print("selftest ok")


if __name__ == "__main__":
    if sys.argv[1:] == ["--selftest"]:
        selftest()
        raise SystemExit
    src = Path(sys.argv[1])
    res = summarize([json.loads(l) for l in src.read_text().splitlines() if l.strip()])
    (src.parent / "summary.json").write_text(json.dumps(res, indent=1, ensure_ascii=False, default=str))
    print(json.dumps(res, indent=1, ensure_ascii=False, default=str))
