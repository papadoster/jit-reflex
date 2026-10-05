"""C1-E4 part 1 memo tables beyond summary.json (docs/c1/e4.md). Report only, read after the data; the rules (spec §8)
are not touched. Sections follow the owner's memo list:
  calib     - why pi0.5's kbar_30 sits far from the median of all its samples: samples by init / suite / class / near-far,
              init 40 against the other inits of its own task, kbar_n bands over 1000 round-robin orders, n90 with the
              online shadow pair's noise (bootstrap with replacement);
  suites    - the delay axis by suite (Spatial / Object / Goal): no-shift control, none and G under the step, G - none,
              time to the close, fresh plan before the close, grasp miss;
  curve     - G - none against d for both brains on inits 0-19 (rows 2 + 4 + 17 and 7 + 14);
  ppc       - row 11 (inits 0-9): PPC against G, T0 against none, noisy G against PPC, per cell;
  kappa     - G-auto + T against G + T on both brains (rows 3, 9, 15), the switch (G + T) - G, G-R (row 10), where kappa
              decides (s of the first plan after the engagement), the Z1 check, EMA against GLR (row 12);
  loss      - the share of the loss returned (spec §10) where a control exists;
  shots     - objects shot by the simulator at the shift (scripts/c1_shots.py's proxy) and the rules without them;
  brain     - pi0.5's brain time per call on the 4090 (grid_p.log batches).
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e4_memo_tables.py > results/c1-e4/memo_tables.json
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import c1e2_summary as e2  # noqa: E402
import c1e4_summary as s4  # noqa: E402
import gauto  # noqa: E402
import objreflex as orx  # noqa: E402

POD = Path(__file__).resolve().parents[1] / "results/c1-e4/pod"
P, S = s4.P, s4.S
I10, I20, I30, I40 = range(10), range(20), range(30), range(40)
C56 = (2,)
SUITE = {"libero_spatial": "Spatial", "libero_object": "Object", "libero_goal": "Goal"}
TAU_K = 0.039517  # the trial's tau_k (spec journal "trial")
rnd = e2.rnd


def r3(x):
    return None if x is None else round(float(x), 3)


def med(v):
    return r3(np.median(v)) if len(v) else None


def mean(v):
    return r3(np.mean(v)) if len(v) else None


# ---------------------------------------------------------------- calib
def calib(recs, brain, rng):
    order = sorted(recs, key=lambda r: (r["init"], orx.TASKS.index((r["suite"], r["task"]))))
    smp = [r for r in order if r.get("shadow") and r["shadow"]["sample"] is not None]
    x = np.array([r["shadow"]["sample"] for r in smp])
    k30, kall, k0 = float(np.median(x[:30])), float(np.median(x)), gauto.KAPPA0[brain]
    by = lambda f: {str(k): {"n": int(sum(f(r) == k for r in smp)),  # noqa: E731
                             "median": med([r["shadow"]["sample"] for r in smp if f(r) == k])}
                    for k in sorted({f(r) for r in smp}, key=str)}
    own = {}  # init 40 against the median of the same task's other inits
    for s, t in orx.TASKS:
        g = {r["init"]: r["shadow"]["sample"] for r in smp if (r["suite"], r["task"]) == (s, t)}
        if 40 in g and len(g) > 1:
            own[f"{SUITE[s]} {t}"] = (g[40], float(np.median([v for i, v in g.items() if i != 40])))
    tasks = sorted({(r["suite"], r["task"]) for r in recs}, key=orx.TASKS.index)
    cell = {(r["suite"], r["task"], r["init"]): r["shadow"]["sample"] for r in smp}
    ns = (10, 20, 30, 50, 75, len(x))
    curves = []  # 20000 round-robin orders: a random init order, a random task order within each round
    for _ in range(20000):
        seq = []
        for i in rng.permutation([40, 41, 42, 43]):
            seq += [cell[(s, t, int(i))] for j in rng.permutation(len(tasks)) for s, t in [tasks[j]]
                    if (s, t, int(i)) in cell]
        curves.append([np.median(seq[:n]) for n in ns])
    band = {str(n): [r3(np.percentile([c[j] for c in curves], q)) for q in (5, 50, 95)] for j, n in enumerate(ns)}
    k30s = np.array([c[2] for c in curves])
    n90 = None  # bootstrap with replacement: smallest n with 90% of medians within +-0.10 of the pool median
    for n in range(5, 401, 5):
        if np.mean(np.abs(np.median(rng.choice(x, size=(2000, n)), axis=1) - kall) <= 0.10) >= 0.90:
            n90 = n
            break
    near = [r["shadow"]["sample"] for r in smp if r["r"] <= 0.10]
    far = [r["shadow"]["sample"] for r in smp if r["r"] > 0.10]
    return {"samples": len(x), "shifts": len(recs), "kbar_30": r3(k30), "kbar_all": r3(kall), "kappa0": k0,
            "first30_inits": dict(Counter(str(r["init"]) for r in smp[:30])),
            "by_init": by(lambda r: r["init"]), "by_suite": by(lambda r: SUITE[r["suite"]]),
            "by_class": by(lambda r: ["1-2", "3-4", "5-6"][r["mag_class"]]),
            "near_le10cm": {"n": len(near), "median": med(near)}, "far": {"n": len(far), "median": med(far)},
            "iqr": [r3(np.percentile(x, 25)), r3(np.percentile(x, 75))], "sd": r3(np.std(x)),
            "init40_below_own_task": [sum(a < b for a, b in own.values()), len(own)],
            "init40_vs_own_task": {k: [r3(a), r3(b)] for k, (a, b) in own.items()},
            "round_robin_kbar_n_5_50_95": band,
            "round_robin_kbar30": {"share_within_0.10_of_kappa0": r3(np.mean(np.abs(k30s - k0) <= 0.10 + 1e-9)),
                                   "share_within_0.10_of_all": r3(np.mean(np.abs(k30s - kall) <= 0.10 + 1e-9)),
                                   "share_at_or_below_observed": r3(np.mean(k30s <= k30 + 1e-9))},
            "n90_bootstrap": n90}


# ---------------------------------------------------------------- shared helpers
def pairs(idx, brain, kind, cells, a, b, inits, cls=None, suite=None):
    p = s4.prs(idx, brain, kind, cells, a, b, inits, cls)
    return [x for x in p if suite is None or x[0][0] == suite]


def d(idx, *a, **k):
    return rnd(e2.diff_pp(pairs(idx, *a, **k)))


def recs(idx, brain, kind, cell, arm, inits, suite=None, cls=None):
    return [r for (s, _, i), r in idx.get((brain, kind, cell, arm), {}).items()
            if i in inits and (suite is None or s == suite) and (cls is None or r["mag_class"] in cls)]


def succ(rs):
    return mean([r["success"] for r in rs])


def fresh(rs):
    out = []
    for r in rs:
        if r["t_fire"] is None:
            continue
        arr = [ta for to, ta in r["plans_in"] if to >= r["t_fire"] + 1]
        out.append(bool(arr) and (r["t_close"] is None or arr[0] < r["t_close"]))
    return mean(out)


def tau(rs):
    return med([r["t_close"] - r["t_fire"] for r in rs if r["t_fire"] is not None and r["t_close"] is not None])


def miss(rs):
    """grasp at the close: share ok, median |miss|, median signed miss along the shift (> 0 overshoot), in cm"""
    m = [r for r in rs if r["t_close"] is not None and r["grasp_miss_m"] is not None]
    al = [100 * r["grasp_miss_along_m"] for r in m if r.get("grasp_miss_along_m") is not None]
    return {"grasp_ok": mean([bool(r["grasp_ok"]) for r in m]), "miss_cm": med([100 * r["grasp_miss_m"] for r in m]),
            "along_cm": med(al)}


# ---------------------------------------------------------------- suites
def suites(idx):
    out = {}
    for b, cells in ((P, ("A", "A10", "A20", "A40")), (S, ("A", "A20", "A40"))):
        for su, name in SUITE.items():
            row = {}
            for c in cells:
                no, g = recs(idx, b, "step", c, "none", I20, su), recs(idx, b, "step", c, "G", I20, su)
                ctrl = recs(idx, b, "control", c, "none", I10, su)
                row[c] = {"control_none": succ(ctrl), "none": succ(no), "G": succ(g),
                          "G-none": d(idx, b, "step", (c,), "G", "none", I20, suite=su),
                          "tau_close_none": tau(no), "fresh_before_close_none": fresh(no),
                          "grasp_none": miss(no), "grasp_G": miss(g),
                          "G_success_given_grasp_ok": mean([r["success"] for r in g if r["grasp_ok"]])}
            out[f"{b}/{name}"] = row
    return out


# ---------------------------------------------------------------- curve
def curve(idx):
    out = {}
    for b, cells in ((P, ("A", "A10", "A20", "A40")), (S, ("A", "A20", "A40"))):
        for c in cells:
            no, g = recs(idx, b, "step", c, "none", I20), recs(idx, b, "step", c, "G", I20)
            ctrl = succ(recs(idx, b, "control", c, "none", I10))
            gd = d(idx, b, "step", (c,), "G", "none", I20)
            out[f"{b}/{c}"] = {"control_none": ctrl, "none": succ(no), "G": succ(g), "G-none": gd,
                               "loss_share_G": r3((succ(g) - succ(no)) / (ctrl - succ(no))) if ctrl else None,
                               "fresh_before_close_none": fresh(no), "tau_close_none": tau(no)}
    for b, dc in ((P, ("A20", "A40")), (S, ("A20", "A40"))):  # the axis apart: (G - none)[d] - (G - none)[A], 0-19
        for c in dc:
            out[f"{b}/axis_{c}-A"] = rnd(s4.s3.did(pairs(idx, b, "step", (c,), "G", "none", I20),
                                               pairs(idx, b, "step", ("A",), "G", "none", I20)))
    for c in ("A20", "A40"):  # pi - S of (G - none)[d] - (G - none)[A]: report only, no bootstrap across brains
        a, z = out[f"{P}/axis_{c}-A"], out[f"{S}/axis_{c}-A"]
        out[f"pi-S_axis_{c}"] = r3(a["diff_pp"] - z["diff_pp"])
    return out


# ---------------------------------------------------------------- ppc
def ppc(idx):
    out = {}
    for cells, name in ((("A",), "A"), (("A20",), "A20"), (("A40",), "A40"), (("A20", "A40"), "A20+A40")):
        out[name] = {"PPC-G": d(idx, P, "step", cells, "PPC", "G", I10), "PPC-none": d(idx, P, "step", cells, "PPC", "none", I10),
                     "G-none": d(idx, P, "step", cells, "G", "none", I10), "T0-none": d(idx, P, "step", cells, "T0", "none", I10),
                     "G@glr-PPC": d(idx, P, "step", cells, "G@glr", "PPC", I10),
                     "G@glr-none": d(idx, P, "step", cells, "G@glr", "none", I10)}
        if len(cells) == 1:
            out[name]["success"] = {a: succ(recs(idx, P, "step", cells[0], a, I10))
                                    for a in ("none", "T0", "PPC", "G", "G@glr")}
            out[name]["trig_calls"] = {a: mean([r["calls_trig"] for r in recs(idx, P, "step", cells[0], a, I10)])
                                       for a in ("T0", "PPC")}
            out[name]["grasp"] = {a: miss(recs(idx, P, "step", cells[0], a, I10)) for a in ("none", "T0", "PPC", "G")}
    out["T0-none_C_0-39"] = d(idx, P, "step", ("C",), "T0", "none", I40)
    out["T0-none_A_0-9"] = out["A"]["T0-none"]
    out["smolvla_T0-none_C_0-19"] = d(idx, S, "step", ("C",), "T0", "none", I20)
    return out


# ---------------------------------------------------------------- kappa
def first_s(r):
    """s of the first plan observed at or after the engagement, before the close (kappa_log), or None"""
    kl = [e for e in (r.get("kappa_log") or []) if r["t_close"] is None or e["t"] < r["t_close"]]
    return kl[0]["s"] if kl else None


def s_bucket(r):
    s = first_s(r)
    return "no plan" if s is None else "s<=0.1" if s <= 0.1 else "0.1-0.5" if s <= 0.5 else "0.5-0.9" if s <= 0.9 else ">0.9"


def kappa(idx):
    out = {"T": {}, "switch": {}, "GR": {}, "where_kappa_decides": {}, "Z1_check": {}, "eyes": {}}
    for b, inits in ((P, I40), (P, I20), (S, I20)):
        for cells in (("A",), ("C",), ("A", "C")):
            k = f"{b}/{'+'.join(cells)}/0-{max(inits)}"
            out["T"][k] = {"GautoT-GT": d(idx, b, "step", cells, "GautoT", "GT", inits, C56),
                           "success": {a: succ([r for c in cells for r in recs(idx, b, "step", c, a, inits, cls=C56)])
                                       for a in ("GT", "GautoT", "Gk0T")}}
            if b == P and inits == I20:
                out["T"][k]["Gk0T-GautoT"] = d(idx, b, "step", cells, "Gk0T", "GautoT", I20, C56)
                out["T"][k]["Gk0T-GT"] = d(idx, b, "step", cells, "Gk0T", "GT", I20, C56)
            if "C" not in cells:
                out["switch"][k] = {"GT-G": d(idx, b, "step", cells, "GT", "G", inits, C56),
                                    "GautoT-Gauto": d(idx, b, "step", cells, "GautoT", "Gauto", inits, C56)}
    for b in (P, S):
        for c in ("A", "C"):
            gt = recs(idx, b, "step", c, "GT", I20, cls=C56)
            out["T"][f"{b}/{c}/grasp_class56_0-19"] = {a: miss(recs(idx, b, "step", c, a, I20, cls=C56))
                                                      for a in ("GT", "GautoT")}
            out["T"][f"{b}/{c}/T_fired_class56_0-19"] = mean([r["calls_trig"] > 0 for r in gt])
    out["GR"] = {"GR-G": d(idx, P, "step", ("A",), "GR", "G", I20), "GR-Gkeep": d(idx, P, "step", ("A",), "GR", "Gkeep", I20),
                 "GR-Gauto": d(idx, P, "step", ("A",), "GR", "Gauto", I20), "GR-none": d(idx, P, "step", ("A",), "GR", "none", I20),
                 "success_0-19": {a: succ(recs(idx, P, "step", "A", a, I20)) for a in ("none", "G", "Gkeep", "Gauto", "GR")}}
    kl = [e for r in recs(idx, P, "step", "A", "GR", I20) for e in (r.get("kappa_log") or [])
          if r["t_close"] is None or e["t"] < r["t_close"]]
    out["GR"]["plans"] = len(kl)
    out["GR"]["dead_zone_share"] = mean([abs(e["m"]["20"]) < TAU_K for e in kl])
    out["GR"]["k_hat_median"] = med([e["k_hat"] for e in kl])
    out["GR"]["k_hat_median_outside_dead_zone"] = med([e["k_hat"] for e in kl if abs(e["m"]["20"]) >= TAU_K])
    for b, arms in ((P, ("none", "G", "Gkeep", "Gk0", "Gauto", "GR")), (S, ("none", "G", "Gkeep", "Gk0", "Gauto"))):
        tab = {}
        for a in arms:
            inits = I20 if a == "GR" else I30
            rs = recs(idx, b, "step", "A", a, inits)
            ref = {k: v for k, v in idx.get((b, "step", "A", "G"), {}).items() if k[2] in inits}
            row = {}
            for bk in ("no plan", "s<=0.1", "0.1-0.5", "0.5-0.9", ">0.9"):
                # bucket by G's own first plan of the same episode, so every arm is read on the same episodes
                keys = {k for k, r in ref.items() if s_bucket(r) == bk}
                g = [r for (s, t, i), r in idx.get((b, "step", "A", a), {}).items() if (s, t, i) in keys]
                row[bk] = {"n": len(g), "success": succ(g), "along_cm": miss(g)["along_cm"]}
            tab[a] = row
        out["where_kappa_decides"][b] = tab
        z1 = {}
        for a, kap in (("Gauto", None), ("G", 1.0), ("Gkeep", 0.0)):
            err = [abs((e["k"] if kap is None else kap) - e["m"]["20"] / e["nd"])
                   for r in recs(idx, b, "step", "A", a, I30) for e in (r.get("kappa_log") or [])
                   if (r["t_close"] is None or e["t"] < r["t_close"]) and e["nd"] > 1e-3]
            z1[a] = {"plans": len(err), "median_abs_kappa_minus_raw": med(err)}
        out["Z1_check"][b] = z1
    for c in ("A", "A20", "A40"):  # row 12 against row 5 on inits 0-9
        row = {"G@glr-G@ema": d(idx, P, "step", (c,), "G@glr", "G@ema", I10),
               "G-G@glr": d(idx, P, "step", (c,), "G", "G@glr", I10)}
        for a, tk in (("G@glr", "t_alarm"), ("G@ema", "t_engage")):
            rs = [r for r in recs(idx, P, "step", c, a, I10) if r["t_fire"] is not None]
            on = [r[tk] - r["t_fire"] for r in rs if r[tk] is not None and r[tk] > r["t_fire"]
                  and (r["t_close"] is None or r[tk] < r["t_close"])]
            row[a] = {"success": succ(recs(idx, P, "step", c, a, I10)),
                      "false": mean([r[tk] is not None and r[tk] <= r["t_fire"] for r in rs]),
                      "missed": mean([r[tk] is None or (r["t_close"] is not None and r[tk] >= r["t_close"]) for r in rs]),
                      "delay_median": med(on), "grasp": miss(recs(idx, P, "step", c, a, I10))}
        for a in ("G@glr", "G@ema"):
            rs = [r for r in recs(idx, P, "step", c, a, I10) if r["t_fire"] is not None]
            row[a]["missed_by_class"] = {["1-2", "3-4", "5-6"][k]: mean(
                [r[("t_alarm" if a == "G@glr" else "t_engage")] is None or (r["t_close"] is not None and
                 r[("t_alarm" if a == "G@glr" else "t_engage")] >= r["t_close"]) for r in rs if r["mag_class"] == k])
                for k in range(3)}
        out["eyes"][c] = row
    calls = {}
    for b in (P, S):
        g = recs(idx, b, "step", "A", "Gauto", I30)
        eng = mean([r["t_engage"] is not None for r in g])
        calls[b] = {"engaged": eng, "shadow_calls_per_episode": r3(2 * eng),
                    "calls_per_episode_G": mean([r["calls_sched"] + r["calls_trig"] for r in recs(idx, b, "step", "A", "G", I30)])}
    out["gauto_calls"] = calls
    return out


# ---------------------------------------------------------------- loss
def loss(idx):
    out = {}
    for b, cells in ((P, ("A", "A20", "A40", "C")), (S, ("A",))):
        for c in cells:
            ctrl = succ(recs(idx, b, "control", c, "none", I10))
            if ctrl is None:
                continue
            row = {"control_none": ctrl}
            for a in ("G", "Gkeep", "Gk0", "Gauto", "GR", "T0", "PPC", "G@glr", "G@ema", "GT", "GautoT"):
                p = s4.prs(idx, b, "step", (c,), a, "none", I40)
                if not p:
                    continue
                m, n = np.mean([x[1] for x in p]), np.mean([x[2] for x in p])
                row[a] = {"pairs": len(p), "share": r3((m - n) / (ctrl - n)), "none_on_pairs": r3(n)}
            out[f"{b}/{c}"] = row
    return out


# ---------------------------------------------------------------- shots
def far(r):
    """scripts/c1_shots.py's proxy of an object shot by the simulator at the shift: unstable and grasp miss > 15 cm"""
    return bool(r.get("unstable")) and r.get("grasp_miss_m") is not None and r["grasp_miss_m"] > 0.15


def shots(idx):
    """shares of the proxy (in every arm of the episode) and the rules on pairs without it in both arms"""
    eps = {}
    for (b, kind, c, a), g in idx.items():
        if kind == "step":
            for k, r in g.items():
                if r["t_fire"] is not None:
                    eps.setdefault((b, c, k), []).append(far(r))
    share = {}
    for (b, c, _), v in eps.items():
        if len(v) > 1:
            n = share.setdefault(f"{b}/{c}", [0, 0])
            n[0], n[1] = n[0] + all(v), n[1] + 1
    out = {"share": {k: [v[0], v[1], r3(100 * v[0] / v[1])] for k, v in sorted(share.items())}}

    def keep(brain, kind, cells, a, b, inits, cls=None):
        p = []
        for c in cells:
            x, y = idx.get((brain, kind, c, a), {}), idx.get((brain, kind, c, b), {})
            for k in sorted(x.keys() & y.keys()):
                if k[2] in inits and (cls is None or y[k]["mag_class"] in cls) and not (far(x[k]) and far(y[k])):
                    p.append((k[:2], x[k]["success"], y[k]["success"]))
        return p
    for name, args in (("Q0", (P, "step", ("C",), "T0", "none", I40)), ("Q2b", (P, "step", ("A",), "Gauto", "Gk0", I30)),
                       ("Q2c", (P, "step", ("A",), "Gauto", "Gkeep", I30)),
                       ("Q2d", (P, "step", ("A", "C"), "GautoT", "GT", I40, C56)),
                       ("Q2e", (P, "step", ("A",), "Gauto", "G", I30)), ("Q2f", (S, "step", ("A",), "Gauto", "G", I30)),
                       ("Q3a", (P, "step", ("A", "A20", "A40"), "G@glr", "none", I20))):
        out[name] = rnd(e2.diff_pp(keep(*args)))
    out["Q1"] = rnd(s4.s3.did(keep(P, "step", ("A20", "A40"), "G", "none", I20), keep(P, "step", ("A",), "G", "none", I30)))
    return out


# ---------------------------------------------------------------- brain
def brain():
    sec_call = []
    for line in (POD / "logs/grid_p.log").read_text().splitlines():
        m = re.search(r"brain ([\d.]+) s (\d+) calls (\d+) rows", line)
        if m and int(m[2]):
            sec_call.append(float(m[1]) / int(m[2]))
    return {"batches": len(sec_call), "s_per_call_median": med(sec_call), "p10_p90": [r3(np.percentile(sec_call, 10)),
                                                                                    r3(np.percentile(sec_call, 90))]}


def main():
    rs = e2.load(POD / "grid_all.jsonl")
    idx = s4.index(rs)
    rng = np.random.default_rng(0)
    out = {"calib": {P: calib(e2.load(POD / "trial_p2.jsonl"), P, rng), S: calib(e2.load(POD / "trial_p3.jsonl"), S, rng)},
           "suites": suites(idx), "curve": curve(idx), "ppc": ppc(idx), "kappa": kappa(idx), "loss": loss(idx),
           "shots": shots(idx), "brain_pi05": brain()}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
