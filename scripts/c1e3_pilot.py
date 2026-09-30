"""C1-E3 pod helpers for scripts/gpu_c1e3.sh: the padding test, the offline pilot choices and the s = 10 baseline (spec
docs/superpowers/specs/2026-09-29-c1-e3-handoff-reflex-design.md §7, §8). Pure numpy, no torch/LIBERO.
    python scripts/c1e3_pilot.py pad results/c1-e3/pad1.jsonl results/c1-e3/pad0.jsonl RATE1 RATE0
    python scripts/c1e3_pilot.py choose PAD results/c1-e3/pilot.jsonl results/c1-e3/pilot_b.jsonl
    python scripts/c1e3_pilot.py baseline results/c1-e3/base/s10_* > results/c1-e3/baseline.json
    python scripts/c1e3_pilot.py --selftest
choose prints the marker line "C1-E3 pilot: K=.. tau_k=.. rc=.. dir=.. beta=.. eps_noise=.. vmin_noise=.. pad=..",
which the spec journal carries before the grid (scripts/gpu_c1e3.sh grid checks it).
"""

import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402

N_PILOT, FALSE_MAX = 104, 0.05
ROWS = {"P1": ("A", "step", "GR", 0.0), "P2": ("A", "step", "GR", 1.0), "P3": ("A", "control", "none", 0.0),
        "P4": ("A", "close", "none", 0.0), "P5": ("C", "step", "none", 0.0), "P6": ("C", "control", "none", 0.0)}
RCS, BETAS, EPSS, VMINS = (0.03, 0.05, 0.08), (1.0, 0.5, 0.3, 0.2), (0.005, 0.010, 0.015, 0.020), (0.001, 0.002, 0.005, 0.010)


def load(files):
    return [json.loads(line) for f in files for line in Path(f).read_text().splitlines() if line.strip()]


def split(recs):
    """Pilot rows P1-P6 (spec §7); stops unless each has N_PILOT distinct episodes (the owner decides)."""
    out = {k: [r for r in recs if (r["cell"], r["kind"], r["arm"], r["alpha"]) == v] for k, v in ROWS.items()}
    for k, rs in out.items():
        if len(rs) != N_PILOT or len({(r["suite"], r["task"], r["init"]) for r in rs}) != N_PILOT:
            sys.exit(f"!!! pilot row {k} {ROWS[k]}: {len(rs)} records, expected {N_PILOT} distinct episodes. "
                     "The owner decides, e.g. rerun the pilot: it resumes and retries failed batches")
    return out


def seen(r, filt=None, tracker=None):
    """What the reflex would see at each step of a traced episode: tracker, then push filter (as HAgent.observe)."""
    tr = r["trace"]
    out = []
    for p, x, c in zip(np.array(tr["p"]), np.array(tr["eef"]), tr["c"]):
        q = tracker(p) if tracker else p
        out.append(filt(q, x, c) if filt else q)
    return np.array(out)


def engaged_steps(r, q, eps):
    """Steps before the first close command where G-R's engagement test holds against the plan executing at observe
    time (trace tobs): |q(t) - q(t_obs)| > eps. Exact up to the first one: G-R without T leaves the trajectory as is."""
    end = len(q) if r["t_close"] is None else r["t_close"]
    return [t for t in range(end) if r["trace"]["tobs"][t] >= 0 and np.linalg.norm(q[t] - q[r["trace"]["tobs"][t]]) > eps]


def plans(r):
    return [(t, np.array(x), np.array(a)) for t, x, a in r["trace"]["plans"]]


def tau_kappa(p3):
    """Choice 1: per K, the 95th percentile of |(g_N - g_R) . u| over plans observed after the pseudo-shift and before
    the close (u: the episode's random horizontal direction; R: the last plan observed at or before t_fire)."""
    vals = {k: [] for k in hf.KS}
    for r in p3:
        tf, tc = r["t_fire"], r["t_close"]
        ps = plans(r)
        ref = [pl for pl in ps if tf is not None and pl[0] <= tf]
        if not ref:
            continue
        u = np.array([math.cos(r["angle"]), math.sin(r["angle"]), 0.0])
        for new in ps:
            if new[0] >= tf + 1 and (tc is None or new[0] < tc):
                for k in hf.KS:
                    vals[k].append(abs(float(hf.target_diff(new, ref[-1], k)[0] @ u)))
    if not vals[hf.KS[0]]:
        sys.exit("!!! choice 1: no plan observed after a pseudo-shift in P3")
    return {k: float(np.percentile(v, 95)) for k, v in vals.items()}, len(vals[hf.KS[0]])


def choose_k(p12, tau):
    """Choice 2: K with the smallest mean |kappa_hat - kappa_model| on fallback plans of P1-P2; K = 20 with fewer
    than 20 of them. kappa_model = alpha + (1 - alpha) (S . Delta_hat) / |Delta|. Ties: K nearest 20, then smaller."""
    err = {k: [] for k in hf.KS}
    for r in p12:
        for e in r["kappa_log"]:
            if not e["fb"] or e["nd"] < 1e-9:
                continue
            d = np.array(e["d"]) / e["nd"]
            km = r["alpha"] + (1 - r["alpha"]) * float(np.array(e["S"]) @ d) / e["nd"]
            for k in hf.KS:
                err[k].append(abs(hf.kappa_hat(e["m"][str(k)], e["nd"], tau[k]) - km))
    n = len(err[20])
    mean = {k: float(np.mean(v)) if v else None for k, v in err.items()}
    return (20 if n < 20 else min(hf.KS, key=lambda k: (mean[k], abs(k - 20), k))), mean, n


def pushed(r):
    """The object moved > EPS from its start before the first close command (a push episode, spec §7)."""
    p = np.array(r["trace"]["p"])
    p = p if r["t_close"] is None else p[: r["t_close"]]
    return bool(np.linalg.norm(p - p[0], axis=1).max() > orx.EPS)


def shift_marked(r, mode, rc=0.05, direction=False):
    """Is the external shift's increment (observed at t_fire + 1) marked as a push?"""
    tr, t = r["trace"], r["t_fire"] + 1
    p, x, c = np.array(tr["p"]), np.array(tr["eef"]), tr["c"]
    return bool(c[t - 1] or c[t]) if mode == "contact" else hf.surr_push(p[t], p[t - 1], x[t], x[t - 1], rc, direction)


def filter_rates(p3, p4, mode, rc=0.05, direction=False):
    """(caught, passed, n caught, n passed): P3 push episodes where G-R with this filter would not engage before the
    close; P4 fired shifts whose increment the filter does not mark."""
    caught = [not engaged_steps(r, seen(r, hf.PushFilter(mode, rc, direction)), orx.EPS) for r in p3 if pushed(r)]
    ok = [r for r in p4 if r["t_fire"] is not None and r["t_fire"] + 1 < len(r["trace"]["p"])]
    passed = [not shift_marked(r, mode, rc, direction) for r in ok]
    if not caught or not passed:
        sys.exit(f"!!! choice 3: {len(caught)} push episodes in P3, {len(passed)} fired shifts in P4")
    return float(np.mean(caught)), float(np.mean(passed)), len(caught), len(passed)


def choose_surrogate(p3, p4):
    """Choice 3: max (caught + passed) / 2 over r_c x direction; ties: smaller r_c, then no direction."""
    res = {(rc, d): filter_rates(p3, p4, "surr", rc, bool(d)) for rc in RCS for d in (0, 1)}
    best = max(res, key=lambda c: ((res[c][0] + res[c][1]) / 2, -c[0], -c[1]))
    return best, res, filter_rates(p3, p4, "contact")


def tracker(r, beta):
    return hf.Tracker(r["seed"], *hf.NOISE["real"], beta)


def choose_noise(p5, p6):
    """Choice 4 at the realistic point: beta x eps' with false engagements in P6 <= 5% and the smallest median
    detection delay in P5 (steps from t_fire to the first engagement after it; none before the close: inf); ties:
    larger eps', then smaller beta. If no pair has <= 5%: the fewest false engagements. Then PPC's speed threshold at
    that beta: the smallest v_min with <= 5% false triggers in P6."""
    res = {}
    for beta in BETAS:
        for eps in EPSS:
            false = np.mean([bool(engaged_steps(r, seen(r, tracker=tracker(r, beta)), eps)) for r in p6])
            delay = [min([t - r["t_fire"] for t in engaged_steps(r, seen(r, tracker=tracker(r, beta)), eps)
                          if t > r["t_fire"]], default=math.inf) for r in p5 if r["t_fire"] is not None]
            res[beta, eps] = (float(false), float(np.median(delay)))
    best = pick_noise(res)
    ppc = {}
    for v in VMINS:
        trig = []
        for r in p6:
            q = seen(r, tracker=tracker(r, best[0]))
            end = len(q) if r["t_close"] is None else r["t_close"]
            trig.append(any(np.linalg.norm(q[t] - q[t - 1]) > v for t in range(1, end)))
        ppc[v] = float(np.mean(trig))
    ok = [v for v in VMINS if ppc[v] <= FALSE_MAX]
    vmin = min(ok) if ok else min(VMINS, key=lambda v: (ppc[v], -v))
    return best, res, vmin, ppc


def pick_noise(res):
    """{(beta, eps'): (false share, median delay)} -> the spec §7 choice 4 pair (see choose_noise)."""
    ok = [c for c in res if res[c][0] <= FALSE_MAX]
    return (min(ok, key=lambda c: (res[c][1], -c[1], c[0])) if ok
            else min(res, key=lambda c: (res[c][0], res[c][1], -c[1], c[0])))


def pad_test(pad, nopad, rate_pad, rate_nopad):
    """spec §7 item 6: the padded batch is adopted if every control pair none / GR where GR never engaged has bitwise
    equal actions in the padded run and the padded run is at most 15% slower. Returns (adopt, report)."""
    def pairs(recs):
        idx = {(r["suite"], r["task"], r["init"], r["arm"]): r for r in recs if r["kind"] == "control"}
        out = []
        for (s, t, i, arm), r in idx.items():
            g = idx.get((s, t, i, "GR"))
            if arm == "none" and g and g["t_engage"] is None and not g["g_on"]:
                out.append(r["act_hash"] == g["act_hash"])
        return out
    eq1, eq0 = pairs(pad), pairs(nopad)
    if not eq1:
        sys.exit("!!! padding test: no control pair where GR never engaged; the owner decides")
    slow = rate_pad / rate_nopad - 1
    adopt = all(eq1) and slow <= 0.15 + 1e-9
    return adopt, (f"padded: {sum(eq1)}/{len(eq1)} pairs bitwise equal, unpadded: {sum(eq0)}/{len(eq0)}; "
                   f"{rate_pad:.2f} vs {rate_nopad:.2f} s/episode ({100 * slow:+.0f}%)")


def baseline(dirs):
    """{"s10": rate, "n": episodes}: lerobot-eval s = 10 on the 26 tasks x inits 0-9 (as C1-E2's s10)."""
    k = n = 0
    for d in map(Path, dirs):
        for t in json.loads((d / "eval_info.json").read_text())["per_task"]:
            x = t["metrics"]["successes"]
            k, n = k + sum(map(bool, x)), n + len(x)
    if n != 260:
        sys.exit(f"!!! baseline: {n} episodes, expected 260")
    return {"s10": k / n, "n": n}


def choose(pad, files):
    rows = split(load(files))
    tau, n1 = tau_kappa(rows["P3"])
    k, kerr, n2 = choose_k(rows["P1"] + rows["P2"], tau)
    (rc, d), sres, oracle = choose_surrogate(rows["P3"], rows["P4"])
    (beta, eps), nres, vmin, ppc = choose_noise(rows["P5"], rows["P6"])
    print(f"1. tau_kappa (P95 over {n1} plans in P3): " + ", ".join(f"K={x}: {tau[x] * 100:.2f} cm" for x in hf.KS))
    print(f"2. K: {n2} fallback plans in P1-P2; mean |kappa_hat - kappa_model|: "
          + ", ".join(f"K={x}: {'-' if kerr[x] is None else f'{kerr[x]:.3f}'}" for x in hf.KS) + f" -> K={k}")
    for c, (a, b, na, nb) in sres.items():
        print(f"3. r_c={c[0]} dir={c[1]}: caught {a:.3f} (of {na}), passed {b:.3f} (of {nb}), score {(a + b) / 2:.3f}")
    print(f"   contact oracle (report): caught {oracle[0]:.3f}, passed {oracle[1]:.3f} -> r_c={rc} dir={d}")
    for c, (f, m) in nres.items():
        print(f"4. beta={c[0]} eps'={c[1] * 1000:.0f} mm: false {f:.3f}, median delay {m}")
    print("   PPC: " + ", ".join(f"v_min={v * 1000:.0f} mm/step false {ppc[v]:.3f}" for v in VMINS)
          + f" -> beta={beta} eps'={eps} v_min={vmin}")
    print(f"C1-E3 pilot: K={k} tau_k={tau[k]:.4f} rc={rc} dir={d} beta={beta} eps_noise={eps} vmin_noise={vmin} pad={pad}")


def selftest():
    def rec(**kw):  # a traced record with a walking arm; kw overrides
        n = kw.pop("n", 30)
        p = np.tile([0.1, 0.0, 0.9], (n, 1))
        x = np.linspace([0.0, 0.0, 1.0], [0.1, 0.0, 0.9], n)
        r = {"suite": "libero_spatial", "task": 0, "init": 44, "cell": "A", "kind": "control", "arm": "none",
             "alpha": 0.0, "seed": 5, "angle": 0.0, "t_fire": 10, "t_close": 25, "kappa_log": [],
             "trace": {"p": p.tolist(), "eef": x.tolist(), "c": [0] * n, "tobs": [-1] + [10 * (t // 10) for t in range(1, n)],
                       "plans": []}}
        for k, v in kw.items():
            (r["trace"] if k in ("p", "eef", "c", "tobs", "plans") else r)[k] = v
        return r

    # engagement replay and the push filter: a 9 mm push along the arm's way is caught by the surrogate at 5 cm
    p = np.tile([0.1, 0.0, 0.9], (30, 1))
    p[13:] += [0.009, 0, 0]
    x = np.tile([0.06, 0.0, 0.9], (30, 1))
    x[13:] += [0.004, 0, 0]  # the arm steps +x as the object moves: 3.6 cm from it
    r = rec(p=p.tolist(), eef=x.tolist())
    assert pushed(r) and engaged_steps(r, seen(r), orx.EPS) == [13, 14, 15, 16, 17, 18, 19]
    assert not engaged_steps(r, seen(r, hf.PushFilter("surr", 0.05, True)), orx.EPS)
    assert engaged_steps(r, seen(r, hf.PushFilter("surr", 0.03, True)), orx.EPS)  # beyond 3 cm: not marked
    # a sideways external jump at t_fire + 1 = 11 with the arm moving along x: passed with direction, marked without
    q = np.tile([0.1, 0.0, 0.9], (30, 1))
    q[11:] += [0, 0.03, 0]
    s = rec(p=q.tolist(), eef=x.tolist(), kind="close")
    assert shift_marked(s, "surr", 0.05, False) and not shift_marked(s, "surr", 0.05, True)
    assert not shift_marked(s, "contact") and shift_marked(rec(p=q.tolist(), c=[0] * 11 + [1] * 19), "contact")
    (best, res, oracle) = choose_surrogate([r], [s])
    assert best == (0.05, 1) and res[0.05, 1][:2] == (1.0, 1.0) and res[0.08, 0][:2] == (1.0, 0.0), res
    # ties: the arm far from both, every candidate scores 0.5 -> the smallest r_c, then no direction
    far = np.tile([0.5, 0, 0.9], (30, 1)).tolist()
    assert choose_surrogate([rec(p=p.tolist(), eef=far)], [rec(p=q.tolist(), eef=far)])[0] == (0.03, 0)

    # choice 1: R = the last plan at or before t_fire; N after it and before the close; P95 of |diff . u|
    ch = np.zeros((50, 4))
    ch[:, 0], ch[:, 3] = 0.5, -1.0
    ch[20:, 3] = 1.0
    pl = [[0, [0, 0, 0], ch.tolist()], [10, [0, 0, 0], ch.tolist()], [20, [0, 0.01, 0], ch.tolist()],
          [30, [0, 0.02, 0], ch.tolist()]]
    t1 = rec(plans=pl, angle=math.pi / 2, t_close=30)  # u = y: N at 20 differs by 1 cm; 30 is not before the close
    tau, n = tau_kappa([t1])
    assert n == 1 and all(abs(v - 0.01) < 1e-9 for v in tau.values()), tau

    # choice 2: fallback plans only; K = 20 with fewer than 20; else the smallest error
    log = lambda m: {"fb": True, "nd": 0.03, "d": [0, 0.03, 0], "S": [0, 0.024, 0], "m": m}  # noqa: E731
    good = {"10": 0.0, "20": 0.024, "30": 0.03}  # kappa_model = 0.8 at alpha 0; K = 20 hits it
    t2 = [dict(rec(), alpha=0.0, kappa_log=[log(good)] * 25)]
    assert choose_k(t2, {10: 0.01, 20: 0.01, 30: 0.01})[0] == 20
    t2b = [dict(rec(), alpha=0.0, kappa_log=[log({"10": 0.024, "20": 0.0, "30": 0.0})] * 25)]
    assert choose_k(t2b, {10: 0.01, 20: 0.01, 30: 0.01})[0] == 10
    assert choose_k([dict(rec(), kappa_log=[log({"10": 0.024, "20": 0.0, "30": 0.0})] * 19)], {10: 0.01, 20: 0.01, 30: 0.01})[0] == 20
    assert choose_k([dict(rec(), alpha=1.0, kappa_log=[log({"10": 0.0, "20": 0.03, "30": 0.0})] * 20)],
                    {10: 0.01, 20: 0.01, 30: 0.01})[0] == 20  # alpha 1: kappa_model = 1

    # choice 4: a still object; false engagements come from noise only, and fall as eps' and smoothing grow
    ctl = [dict(rec(n=60, t_close=None, tobs=[-1] + [50 * (t // 50) for t in range(1, 60)]), seed=sd) for sd in range(40)]
    stp = []
    for sd in range(40):
        q = np.tile([0.1, 0.0, 0.9], (60, 1))
        q[11:] += [0, 0.04, 0]
        stp.append(dict(rec(n=60, p=q.tolist(), t_close=None, tobs=[-1] + [50 * (t // 50) for t in range(1, 60)]),
                        seed=100 + sd, kind="step"))
    best, res, vmin, ppc = choose_noise(stp, ctl)
    assert res[1.0, 0.005][0] > res[0.2, 0.020][0] and res[best][0] <= FALSE_MAX, res
    assert 4 <= res[best][1] < math.inf, res  # the tracker lags 3 steps: seen at t_fire + 4 at the earliest
    assert ppc[0.010] <= ppc[0.001] and vmin in VMINS
    # the pick itself: the fastest with <= 5% false; ties larger eps', then smaller beta; none <= 5%: the fewest false
    res = {(1.0, 0.005): (0.5, 3.0), (0.5, 0.010): (0.05, 6.0), (0.3, 0.015): (0.02, 6.0), (0.2, 0.015): (0.0, 6.0)}
    assert pick_noise(res) == (0.2, 0.015)
    assert pick_noise({c: (f + 0.1, m) for c, (f, m) in res.items()}) == (0.2, 0.015)
    assert pick_noise({(1.0, 0.005): (0.5, 3.0), (0.5, 0.010): (0.3, 6.0), (0.3, 0.015): (0.3, 5.0)}) == (0.3, 0.015)

    # the padding test: all pairs equal and <= 15% slower adopts; one unequal pair or 16% slower does not
    def pr(arm, i, h, engaged=None):
        return {"suite": "libero_spatial", "task": 0, "init": i, "kind": "control", "arm": arm, "act_hash": h,
                "t_engage": engaged, "g_on": engaged is not None}
    same = [pr("none", i, str(i)) for i in range(4)] + [pr("GR", i, str(i)) for i in range(4)]
    assert pad_test(same, same, 1.1, 1.0)[0] and not pad_test(same, same, 1.16, 1.0)[0]
    odd = same[:-1] + [pr("GR", 3, "x")]
    assert not pad_test(odd, same, 1.0, 1.0)[0]
    assert pad_test(same[:-1] + [pr("GR", 3, "x", engaged=12)], same, 1.0, 1.0)[0]  # an engaged GR is not a pair
    print("selftest ok")


if __name__ == "__main__":
    cmd, a = (sys.argv[1:2] or ["--selftest"])[0], sys.argv[2:]
    if cmd == "--selftest":
        selftest()
    elif cmd == "pad":
        adopt, rep = pad_test(load(a[:1]), load(a[1:2]), float(a[2]), float(a[3]))
        print(f"{rep}\nC1-E3 padding: pad={int(adopt)}")
    elif cmd == "choose":
        choose(int(a[0]), a[1:])
    elif cmd == "baseline":
        print(json.dumps(baseline(a), indent=1))
    else:
        sys.exit(__doc__)
