"""C1-E4o: offline choice of the self-calibrating reflex G-auto (docs/superpowers/specs/2026-10-03-c1-e4o-gauto-offline-design.md)
on the C1-E3 plans (D1), the C1-E4a shifts (D2) and the C1-E3 G / G-keep pairs (D3). Applies the selection rule of §5.
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e4o_gauto.py            # writes results/c1-e4/gauto_offline.json
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e4o_gauto.py --selftest
"""

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
E3 = ROOT / "results/c1-e3/grid_all.jsonl"
E4A = {"pi05": ROOT / "results/c1-e4/headroom_pi05.jsonl", "smolvla": ROOT / "results/c1-e4/headroom_smolvla.jsonl"}
OUT = ROOT / "results/c1-e4/gauto_offline.json"
KAPPA0 = {"pi05": 0.355, "smolvla": 0.038}  # C1-E4a medians, frozen
KAPPA_S = 0.038
ALPHAS = (0.0, 0.5, 1.0)
S_MAX, S_MAX_REPORT, S_FIT = 0.1, 0.2, 1.1
TOL, N_SEQ, LEN, BUDGET, NEAR = 0.10, 1000, 300, 60, 0.10
PRIOR_MU, PRIOR_SD, SD_FLOOR = 0.5, 0.3, 0.02
VALID = 0.25  # D3: |m_G - m_keep| >= VALID * |Delta|
ARMS = ("G", "GR", "Gkeep")


# --- data -------------------------------------------------------------------------------------------------------
def load_e3():
    """Fired step episodes without noise, not unstable, no push before the shift (spec §3, D1 filters)."""
    return [r for r in map(json.loads, E3.open()) if r["kind"] == "step" and not r["noise"] and r["t_fire"] is not None
            and not r["unstable"] and not r["pushed_before_fire"]]


def plans(recs):
    """D1: logged post-engagement plans of G, GR, Gkeep without the fallback and with the seen shift close to the true
    one; raw share r = m20/|Delta_N|, s = (S . u)/|Delta_N|."""
    out = []
    for e in recs:
        if e["arm"] not in ARMS:
            continue
        raw = e["kappa_log"]
        for j, k in enumerate(raw):
            if k["fb"] or not 0.8 <= k["nd"] / e["mag"] <= 1.25:
                continue
            u = np.array(k["d"]) / k["nd"]
            out.append({"alpha": e["alpha"], "cell": e["cell"], "arm": e["arm"], "task": (e["suite"], e["task"]),
                        "ep": (e["suite"], e["task"], e["init"], e["cell"], e["alpha"], e["arm"]),
                        "last": j == len(raw) - 1,  # the episode's true last plan (episodes where it is filtered drop out)
                        "r": k["m"]["20"] / k["nd"], "s": float(np.array(k["S"]) @ u) / k["nd"], "k_hat": k["k_hat"],
                        "near": e["r"] <= NEAR})
    return out


def yields(recs, pl, s_max):
    """I1: share of the (filtered) G-family episodes with >= 1 (filtered) plan s <= s_max, by cell."""
    good = {p["ep"] for p in pl if p["s"] <= s_max}
    by = defaultdict(lambda: [0, 0])
    for e in recs:
        if e["arm"] in ARMS:
            by[e["cell"]][0] += (e["suite"], e["task"], e["init"], e["cell"], e["alpha"], e["arm"]) in good
            by[e["cell"]][1] += 1
    return {c: v[0] / v[1] for c, v in sorted(by.items())}


def load_e4a():
    out = {}
    for b, f in E4A.items():
        rs = [json.loads(line) for line in f.open()]
        out[b] = [{"x": r["headroom"]["ratio"]["close"], "near": r["r"] <= NEAR, "task": (r["suite"], r["task"])}
                  for r in rs if r["kind"] == "step" and r.get("headroom")]
    return out


def pairs(recs):
    """D3: kappa* = m_keep / (m_keep - m_G) on the same episode under G and G-keep (same t_fire)."""
    idx = {(e["suite"], e["task"], e["init"], e["cell"], e["alpha"], e["arm"]): e for e in recs}
    out, seen = [], defaultdict(int)
    for k, g in idx.items():
        kp = idx.get(k[:5] + ("Gkeep",)) if k[5] == "G" else None
        if kp is None or kp["t_fire"] != g["t_fire"]:
            continue
        seen[(k[3], k[4])] += 1
        mg, mk = g["grasp_miss_along_m"], kp["grasp_miss_along_m"]
        if mg is None or mk is None or abs(mg - mk) < VALID * g["mag"]:
            continue
        out.append({"cell": k[3], "alpha": k[4], "x": mk / (mk - mg), "task": k[:2]})
    return out, dict(seen)


# --- sequences and estimators --------------------------------------------------------------------------------------
def seq_round_robin(tasks, rng):
    """(N_SEQ, LEN) indices: with replacement, in rounds over the tasks (random task order per round, one random
    sample of each task)."""
    by = defaultdict(list)
    for i, t in enumerate(tasks):
        by[t].append(i)
    groups = [np.array(v) for _, v in sorted(by.items())]
    rounds = -(-LEN // len(groups))
    out = np.empty((N_SEQ, rounds * len(groups)), int)
    for n in range(N_SEQ):
        row = []
        for _ in range(rounds):
            row += [g[rng.integers(len(g))] for g in (groups[j] for j in rng.permutation(len(groups)))]
        out[n] = row
    return out[:, :LEN]


def seq_task_blocks(tasks, rng):
    """(N_SEQ, n) indices: one pass over all samples, tasks in random order, each task's samples together (shuffled)."""
    by = defaultdict(list)
    for i, t in enumerate(tasks):
        by[t].append(i)
    groups = [np.array(v) for _, v in sorted(by.items())]
    return np.array([np.concatenate([rng.permutation(groups[j]) for j in rng.permutation(len(groups))])
                     for _ in range(N_SEQ)])


def running(xs, kind):
    """(seqs, n) running estimate over the prefixes of each sequence of values xs (seqs, n)."""
    n = xs.shape[1]
    if kind == "mean":
        return np.cumsum(xs, 1) / np.arange(1, n + 1)
    est = np.empty_like(xs)
    for i in range(n):
        pre = xs[:, : i + 1]
        med = np.median(pre, 1)
        est[:, i] = med if kind == "median" else bayes(pre, med)
    return est


def bayes(pre, med=None):
    """Posterior mean under N(PRIOR_MU, PRIOR_SD^2) with the noise sd from the MAD of the samples (floored)."""
    med = np.median(pre, 1) if med is None else med
    k = pre.shape[1]
    sd = (np.maximum(1.4826 * np.median(np.abs(pre - med[:, None]), 1), SD_FLOOR) if k >= 5
          else np.full(len(pre), PRIOR_SD))  # fewer than 5 samples: the prior sd
    w0, w = 1 / PRIOR_SD ** 2, k / sd ** 2
    return (w0 * PRIOR_MU + w * pre.mean(1)) / (w0 + w)


def task_median(x, tasks):
    """Median with every task weighted equally (weight 1 / its sample count): the lower weighted median."""
    cnt = defaultdict(int)
    for t in tasks:
        cnt[t] += 1
    w = np.array([1 / cnt[t] for t in tasks])
    o = np.argsort(x, kind="stable")
    cw = np.cumsum(w[o])
    return float(np.asarray(x, float)[o][np.searchsorted(cw, cw[-1] / 2)])


def full(x, kind):
    x = np.asarray(x, float)
    return float({"mean": np.mean, "median": np.median}[kind](x)) if kind != "bayes" else float(bayes(x[None])[0])


def n90(est, target):
    """Per sequence, n_j = the first n from which it stays within TOL of the target to the end (inf if outside at the
    end); n90 = the 90th percentile of n_j, None if that is inf."""
    out = np.abs(est - target) > TOL + 1e-12
    n = est.shape[1]
    last = np.where(out.any(1), n - 1 - np.argmax(out[:, ::-1], 1), -1)  # index of the last step outside, -1 if none
    nj = np.where(last == n - 1, np.inf, last + 2.0)
    v = float(np.percentile(nj, 90, method="inverted_cdf"))  # smallest v with >= 90% of n_j <= v
    return None if math.isinf(v) else int(v)


def converge(x, tasks, truth=None, kinds=("mean", "median", "bayes"), blocks=False):
    """Pool median as the target; n90 per estimator on round-robin sequences (and task blocks for the report);
    the full-pool value and its bias against the truth (default: the pool median)."""
    x = np.asarray(x, float)
    target = task_median(x, tasks)  # what the round-robin sequences converge to
    truth = float(np.median(x)) if truth is None else truth
    rr = x[seq_round_robin(tasks, np.random.default_rng(0))]
    res = {"n": len(x), "tasks": len(set(tasks)), "target_task_weighted_median": target, "pool_median": float(np.median(x)),
           "truth": truth}
    for kd in kinds:
        res[kd] = {"full": full(x, kd), "bias": full(x, kd) - truth, "n90": n90(running(rr, kd), target)}
        if blocks:
            res[kd]["n90_task_blocks"] = n90(running(x[seq_task_blocks(tasks, np.random.default_rng(0))], kd), target)
    return res


def sign_sgd(xs, start=0.5, eta=0.2):
    k = np.full(len(xs), start)
    est = np.empty_like(xs)
    for i in range(xs.shape[1]):
        k = k + eta / math.sqrt(i + 1) * np.sign(xs[:, i] - k)
        est[:, i] = k
    return est


# --- law accuracy -------------------------------------------------------------------------------------------------
def law(pl, alpha):
    """2-fold by task (sorted, even/odd): kappa_bar = median r on s <= S_MAX, c = LS through the origin on
    s in [0, S_FIT], both on the fit half; median |kappa_N - r| on the other half (all plans with s in [0, S_FIT] and the
    last plan per episode). G-R's k_hat reads its own plan: a reference, not a competitor."""
    p = [q for q in pl if q["alpha"] == alpha]
    tasks = sorted({q["task"] for q in p})
    half = {t: i % 2 for i, t in enumerate(tasks)}
    err, err_last, fits = defaultdict(list), defaultdict(list), []
    for f in (0, 1):
        fit = [q for q in p if half[q["task"]] == f]
        kb = float(np.median([q["r"] for q in fit if q["s"] <= S_MAX]))
        fq = [q for q in fit if 0 <= q["s"] <= S_FIT]
        s, r = np.array([q["s"] for q in fq]), np.array([q["r"] for q in fq])
        c = float(s @ (r - kb * (1 - s)) / (s @ s))
        fits.append({"fold_fit_tasks": [list(t) for t in tasks if half[t] == f], "kappa_bar": kb, "c": c})
        for store, last_only in ((err, False), (err_last, True)):
            ev = [q for q in p if half[q["task"]] != f and 0 <= q["s"] <= S_FIT and (q["last"] or not last_only)]
            s, r = np.array([q["s"] for q in ev]), np.array([q["r"] for q in ev])
            pred = {"Z0": np.full_like(r, kb), "Z1": s + kb * (1 - s), "Z2": c * s + kb * (1 - s),
                    "G": np.ones_like(r), "Gkeep": np.zeros_like(r), "GR_k_hat_reference": np.array([q["k_hat"] for q in ev])}
            for name, v in pred.items():
                store[name] += list(np.abs(v - r))
    bins = {}
    for lo, hi in ((-9, 0.1), (0.1, 0.5), (0.5, 0.9), (0.9, S_FIT), (S_FIT, 99)):
        qs = [q for q in p if lo < q["s"] <= hi]
        if qs:
            bins[f"s({lo},{hi}]"] = {"n": len(qs), "median_r": float(np.median([q["r"] for q in qs])),
                                     "median_s": float(np.median([q["s"] for q in qs]))}
    med = lambda d: {k: float(np.median(v)) for k, v in d.items()}  # noqa: E731
    return {"fits": fits, "median_abs_err": med(err), "median_abs_err_last_plan": med(err_last), "n_plans": len(p),
            "by_s": bins}


# --- the study ----------------------------------------------------------------------------------------------------
def study():
    recs = load_e3()
    pl = plans(recs)
    e4a = load_e4a()
    d3, d3_seen = pairs(recs)
    res = {"yield_I1": {str(S_MAX): yields(recs, pl, S_MAX), str(S_MAX_REPORT): yields(recs, pl, S_MAX_REPORT)}}
    res["D2"] = {b: converge([q["x"] for q in v], [q["task"] for q in v], KAPPA0[b], blocks=True) for b, v in e4a.items()}
    res["D2_buckets"] = {}
    for b, v in e4a.items():
        res["D2_buckets"][b] = {}
        for name, sel in (("near", True), ("far", False)):
            q = [w for w in v if w["near"] == sel]
            res["D2_buckets"][b][name] = converge([w["x"] for w in q], [w["task"] for w in q], kinds=("median", "bayes")) | {
                "share_of_shifts": len(q) / len(v)}
    res["D1"] = {}
    for a in ALPHAS:
        q = [w for w in pl if w["alpha"] == a and w["cell"] in "AB" and w["s"] <= S_MAX]
        res["D1"][str(a)] = (converge([w["r"] for w in q], [w["task"] for w in q], a + KAPPA_S) if len(q) >= 10
                             else {"n": len(q)})
    res["law"] = {str(a): law(pl, a) for a in ALPHAS}
    res["D3_report"] = {}
    for a in ALPHAS:
        for cell in "AB":
            q = [w for w in d3 if w["alpha"] == a and w["cell"] == cell]
            seen = d3_seen.get((cell, a), 0)
            if len(q) < 10:
                res["D3_report"][f"{cell} {a}"] = {"n": len(q), "paired": seen}
                continue
            x = np.array([w["x"] for w in q])
            med = float(np.median(x))
            est = sign_sgd(x[seq_round_robin([w["task"] for w in q], np.random.default_rng(0))])
            res["D3_report"][f"{cell} {a}"] = {"n": len(q), "paired": seen, "yield": len(q) / seen, "median_kappa_star": med,
                                              "iqr": [float(np.percentile(x, 25)), float(np.percentile(x, 75))],
                                              "n90_samples": n90(est, med)}
    res["choice"] = choose(res)
    return res


def choose(res):
    """Spec §5, mechanically."""
    ch = {}
    # 1. law, alpha = 0 only
    lw = res["law"]["0.0"]
    cs, err = [f["c"] for f in lw["fits"]], lw["median_abs_err"]
    if all(c < 0.9 for c in cs) and err["Z1"] - err["Z2"] >= 0.03:
        ch["law"] = "Z2"
    elif err["Z1"] <= err["Z0"] + 0.03:
        ch["law"] = "Z1"
    else:
        ch["law"] = "Z0"
    ch["law_inputs"] = {"c_folds": cs, "err": {z: err[z] for z in ("Z0", "Z1", "Z2")}}
    # 2. estimator, on D2
    d2 = res["D2"]
    o2 = {b: d2[b]["median"]["n90"] for b in d2}
    o3 = {b: d2[b]["bayes"]["n90"] for b in d2}
    le = lambda a, b: a is not None and (b is None or a <= b)  # noqa: E731  (None = never converges)
    if (o3["pi05"] is not None and (o2["pi05"] is None or o3["pi05"] <= 0.75 * o2["pi05"]) and le(o3["smolvla"], o2["smolvla"])
            and all(abs(d2[b]["bayes"]["bias"]) <= 0.05 for b in d2)):
        ch["estimator"] = "bayes"
    elif o2["pi05"] is not None or o3["pi05"] is not None:
        ch["estimator"] = "median"
    else:
        ch["estimator"] = "mean"
    est = ch["estimator"]
    # 3. source
    y = res["yield_I1"][str(S_MAX)]
    y_mix = (y.get("A", 0.0) + (y.get("C", 0.0) + y.get("F", 0.0)) / 2) / 2
    d1 = res["D1"]["0.5"]
    n_i1 = d1.get(est, {}).get("n90")
    shifts_i1 = math.inf if (n_i1 is None or y_mix == 0) else n_i1 / y_mix
    ch["source"] = "I1" if shifts_i1 <= BUDGET else "I2"
    ch["source_inputs"] = {"yield_mix_I1": y_mix, "n90_D1_alpha0.5": n_i1, "shifts_I1": shifts_i1}
    y_src = y_mix if ch["source"] == "I1" else 1.0
    n_src = n_i1 if ch["source"] == "I1" else d2["pi05"][est]["n90"]
    # 4. context
    bk = res["D2_buckets"]["pi05"]
    kb = est if est in bk["near"] else "median"
    diff = abs(bk["near"]["median"]["full"] - bk["far"]["median"]["full"])
    shifts = {nm: (math.inf if bk[nm][kb]["n90"] is None else bk[nm][kb]["n90"] / (bk[nm]["share_of_shifts"] * y_src))
              for nm in ("near", "far")}
    ch["buckets"] = bool(diff >= 0.10 and shifts["near"] <= BUDGET and not math.isinf(shifts["far"]))
    ch["bucket_inputs"] = {"median_diff": diff, "shifts": shifts}
    # 5. N
    n = max(shifts.values()) if ch["buckets"] else (math.inf if n_src is None else n_src / y_src)
    ch["N"] = LEN if math.isinf(n) else int(math.ceil(n / 10) * 10)  # never converges within LEN: N = LEN (spec §5.5)
    ch["N_converged"] = not math.isinf(n)
    return ch


def clean(x):
    """JSON-safe: infinities (never reaches the target) become None, tuple keys and numpy scalars plain."""
    if isinstance(x, dict):
        return {str(k): clean(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [clean(v) for v in x]
    if isinstance(x, (float, np.floating)):
        return None if math.isinf(x) else round(float(x), 4)
    return x


def selftest():
    rng = np.random.default_rng(1)
    tasks = [i % 26 for i in range(400)]
    idx = seq_round_robin(tasks, np.random.default_rng(0))
    assert idx.shape == (N_SEQ, LEN) and sorted({tasks[i] for i in idx[0, :26]}) == list(range(26))
    blk = seq_task_blocks(tasks, np.random.default_rng(0))
    assert blk.shape == (N_SEQ, 400) and sorted(blk[0]) == list(range(400))
    x = np.full(400, 0.3)
    assert converge(x, tasks)["median"]["n90"] == 1
    x = rng.normal(0.4, 0.2, 400)
    r = converge(x, tasks, 0.4)
    assert abs(r["median"]["bias"]) < 0.05 and 1 < r["median"]["n90"] < 60, r
    assert abs(r["bayes"]["full"] - x.mean()) < 0.01
    xs = np.vstack([rng.normal(0.4, 0.2, LEN) for _ in range(200)])
    assert abs(np.median(sign_sgd(xs)[:, -1]) - 0.4) < 0.05
    ok = np.zeros((10, 5)) + 0.3
    ok[:, 2] = 0.9  # leaves the band at n = 3, back from n = 4 on
    assert n90(ok, 0.3) == 4 and n90(ok + 1, 0.3) is None
    ok[0, 4] = 0.9  # one sequence of ten ends outside: the 90th percentile still 4
    assert n90(ok, 0.3) == 4
    ok[1, 4] = 0.9  # two of ten end outside: never
    assert n90(ok, 0.3) is None
    assert task_median(np.array([0.0, 0.0, 0.0, 1.0, 1.0]), ["a", "a", "a", "b", "c"]) == 1.0
    s = rng.uniform(0, 1.1, 500)
    rr = 0.8 * s + 0.3 * (1 - s) + rng.normal(0, 0.05, 500)
    assert abs(float(s @ (rr - 0.3 * (1 - s)) / (s @ s)) - 0.8) < 0.02
    assert clean({"a": math.inf, "b": (np.float64(0.123456), 2)}) == {"a": None, "b": [0.1235, 2]}
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--selftest", action="store_true")
    if p.parse_args().selftest:
        return selftest()
    text = json.dumps(clean(study()), ensure_ascii=False, indent=1)
    OUT.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
