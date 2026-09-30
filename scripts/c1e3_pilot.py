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
from fractions import Fraction
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402

N_PILOT, FALSE_MAX, N_PAD = 104, 0.05, 8  # N_PAD: control episodes run as none and as GR in the padding test
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
    """Is the external shift's increment (observed at t_fire + 1) marked as a push? The surrogate measures to the
    record's object box (journal item 14)."""
    tr, t = r["trace"], r["t_fire"] + 1
    p, x, c = np.array(tr["p"]), np.array(tr["eef"]), tr["c"]
    return (bool(c[t - 1] or c[t]) if mode == "contact"
            else hf.surr_push(p[t], p[t - 1], x[t], x[t - 1], rc, direction, r["box"]))


def filter_rates(p3, p4, mode, rc=0.05, direction=False):
    """(caught, passed, n caught, n passed), the shares as exact fractions (choice 3's ties): P3 push episodes where
    G-R with this filter (the surrogate with the record's object box) would not engage before the close; P4 fired
    shifts whose increment the filter does not mark."""
    caught = [not engaged_steps(r, seen(r, hf.PushFilter(mode, rc, direction, r["box"])), orx.EPS)
              for r in p3 if pushed(r)]
    ok = [r for r in p4 if r["t_fire"] is not None and r["t_fire"] + 1 < len(r["trace"]["p"])]
    passed = [not shift_marked(r, mode, rc, direction) for r in ok]
    if not caught or not passed:
        sys.exit(f"!!! choice 3: {len(caught)} push episodes in P3, {len(passed)} fired shifts in P4")
    return Fraction(sum(caught), len(caught)), Fraction(sum(passed), len(passed)), len(caught), len(passed)


def choose_surrogate(p3, p4):
    """Choice 3: max (caught + passed) / 2 over r_c x direction, exact; ties: smaller r_c, then no direction."""
    res = {(rc, d): filter_rates(p3, p4, "surr", rc, bool(d)) for rc in RCS for d in (0, 1)}
    best = max(res, key=lambda c: ((res[c][0] + res[c][1]) / 2, -c[0], -c[1]))
    return best, res, filter_rates(p3, p4, "contact")


def tracker(r, beta):
    return hf.Tracker(r["seed"], *hf.NOISE["real"], beta)


def ppc_fires(r, q, v):
    """Would our PPC measure a speed |q(t) - q(t - 1)| > v before the first close command? Only on steps where it
    measures: t - t_obs < H, the executing chunk not used up (objreflex.Agent._ppc_measure). Trace tobs is the plan
    it measures against at d = 0 (P5, P6: cell C). Exact up to the first: until then PPC leaves the trajectory."""
    tobs, end = r["trace"]["tobs"], len(q) if r["t_close"] is None else r["t_close"]
    return any(t - tobs[t] < orx.H and np.linalg.norm(q[t] - q[t - 1]) > v for t in range(1, end))


def choose_noise(p5, p6):
    """Choice 4 at the realistic point: beta x eps' with false engagements in P6 <= 5% and the smallest median
    detection delay in P5 (steps from t_fire to the first engagement after it; none before the close: inf); ties:
    larger eps', then smaller beta. If no pair has <= 5%: the fewest false engagements. Then PPC's speed threshold at
    that beta (pick_vmin)."""
    res = {}
    for beta in BETAS:
        for eps in EPSS:
            false = np.mean([bool(engaged_steps(r, seen(r, tracker=tracker(r, beta)), eps)) for r in p6])
            delay = [min([t - r["t_fire"] for t in engaged_steps(r, seen(r, tracker=tracker(r, beta)), eps)
                          if t > r["t_fire"]], default=math.inf) for r in p5 if r["t_fire"] is not None]
            res[beta, eps] = (float(false), float(np.median(delay)))
    best = pick_noise(res)
    qs = [(r, seen(r, tracker=tracker(r, best[0]))) for r in p6]
    ppc = {v: float(np.mean([ppc_fires(r, q, v) for r, q in qs])) for v in VMINS}
    return best, res, pick_vmin(ppc), ppc


def pick_vmin(ppc):
    """{v_min: false share in P6} -> the smallest v_min with <= 5% false triggers; if none has (journal item 15): the
    fewest false, ties the larger v_min."""
    ok = [v for v in VMINS if ppc[v] <= FALSE_MAX]
    return min(ok) if ok else min(VMINS, key=lambda v: (ppc[v], -v))


def pick_noise(res):
    """{(beta, eps'): (false share, median delay)} -> the spec §7 choice 4 pair (see choose_noise)."""
    ok = [c for c in res if res[c][0] <= FALSE_MAX]
    return (min(ok, key=lambda c: (res[c][1], -c[1], c[0])) if ok
            else min(res, key=lambda c: (res[c][0], res[c][1], -c[1], c[0])))


def pad_test(pad, nopad, rate_pad, rate_nopad):
    """spec §7 item 6: the padded batch is adopted if every control pair none / GR where GR never engaged has bitwise
    equal actions in the padded run and the padded run is at most 15% slower. Each run must hold all N_PAD control
    episodes as none and as GR (a failed batch is skipped with exit 0), whatever its --n-envs. Returns (adopt,
    report)."""
    def pairs(recs, run):
        recs = [r for r in recs if r["kind"] == "control" and r["arm"] in ("none", "GR")]
        idx = {(r["suite"], r["task"], r["init"], r["arm"]): r for r in recs}
        eps = {arm: {k[:3] for k in idx if k[3] == arm} for arm in ("none", "GR")}
        if len(recs) != 2 * N_PAD or len(eps["none"]) != N_PAD or eps["none"] != eps["GR"]:
            sys.exit(f"!!! padding test, {run} run: {len(recs)} control records, {len(eps['none'])} none and "
                     f"{len(eps['GR'])} GR episodes; expected the same {N_PAD} as each. The owner decides")
        return [idx[k + ("none",)]["act_hash"] == idx[k + ("GR",)]["act_hash"] for k in sorted(eps["none"])
                if idx[k + ("GR",)]["t_engage"] is None and not idx[k + ("GR",)]["g_on"]]
    eq1, eq0 = pairs(pad, "padded"), pairs(nopad, "unpadded")
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
        print(f"3. r_c={c[0]} dir={c[1]}: caught {float(a):.3f} (of {na}), passed {float(b):.3f} (of {nb}), "
              f"score {float((a + b) / 2):.3f}")
    print(f"   contact oracle (report): caught {float(oracle[0]):.3f}, passed {float(oracle[1]):.3f} "
          f"-> r_c={rc} dir={d}")
    for c, (f, m) in nres.items():
        print(f"4. beta={c[0]} eps'={c[1] * 1000:.0f} mm: false {f:.3f}, median delay {m}")
    print("   PPC: " + ", ".join(f"v_min={v * 1000:.0f} mm/step false {ppc[v]:.3f}" for v in VMINS)
          + f" -> beta={beta} eps'={eps} v_min={vmin}")
    if nres[beta, eps][0] > FALSE_MAX:  # spec §7 choice 4, journal item 15: the fallbacks are journalled
        print(f"fallback: no beta x eps' pair has <= {FALSE_MAX:.0%} false engagements in P6; the fewest false: "
              f"beta={beta} eps'={eps} ({nres[beta, eps][0]:.3f})")
    if ppc[vmin] > FALSE_MAX:
        print(f"fallback: no PPC speed threshold has <= {FALSE_MAX:.0%} false triggers in P6; the fewest false, ties "
              f"the larger: v_min={vmin} ({ppc[vmin]:.3f})")
    print(f"C1-E3 pilot: K={k} tau_k={tau[k]:.6f} rc={rc} dir={d} beta={beta} eps_noise={eps} vmin_noise={vmin} pad={pad}")


def selftest():
    def rec(**kw):  # a traced record with a walking arm and a point box (distances to p itself); kw overrides
        n, sp = kw.pop("n", 30), kw.pop("s", 10)
        p = np.tile([0.1, 0.0, 0.9], (n, 1))
        x = np.linspace([0.0, 0.0, 1.0], [0.1, 0.0, 0.9], n)
        # tobs as the runner traces it, before the step's call: the plan executing at observe time (d = 0)
        r = {"suite": "libero_spatial", "task": 0, "init": 44, "cell": "A", "kind": "control", "arm": "none",
             "alpha": 0.0, "seed": 5, "angle": 0.0, "t_fire": 10, "t_close": 25, "kappa_log": [], "box": [0.0] * 6,
             "trace": {"p": p.tolist(), "eef": x.tolist(), "c": [0] * n,
                       "tobs": [-1] + [sp * ((t - 1) // sp) for t in range(1, n)], "plans": []}}
        for k, v in kw.items():
            (r["trace"] if k in ("p", "eef", "c", "tobs", "plans") else r)[k] = v
        return r

    still = np.tile([0.1, 0.0, 0.9], (30, 1))
    far = np.tile([0.5, 0, 0.9], (30, 1)).tolist()
    # engagement replay and the push filter: a 9 mm push along the arm's way is caught by the surrogate at 5 cm
    p = still.copy()
    p[13:] += [0.009, 0, 0]
    x = np.tile([0.06, 0.0, 0.9], (30, 1))
    x[13:] += [0.004, 0, 0]  # the arm steps +x as the object moves: 3.6 cm from it
    r = rec(p=p.tolist(), eef=x.tolist())
    assert pushed(r) and engaged_steps(r, seen(r), orx.EPS) == list(range(13, 21))  # against plan 10 up to step 20
    assert not engaged_steps(r, seen(r, hf.PushFilter("surr", 0.05, True)), orx.EPS)
    assert engaged_steps(r, seen(r, hf.PushFilter("surr", 0.03, True)), orx.EPS)  # beyond 3 cm: not marked
    # the close step itself is not before the close: a 1 cm move seen first at t_close = 25 neither engages nor pushes
    late = still.copy()
    late[25:] += [0.01, 0, 0]
    lr = rec(p=late.tolist(), eef=far)
    assert not pushed(lr) and not engaged_steps(lr, seen(lr), orx.EPS)
    # a sideways external jump at t_fire + 1 = 11 with the arm still: passed with direction, marked without
    q = still.copy()
    q[11:] += [0, 0.03, 0]
    s = rec(p=q.tolist(), eef=x.tolist(), kind="close")
    assert shift_marked(s, "surr", 0.05, False) and not shift_marked(s, "surr", 0.05, True)
    assert not shift_marked(s, "contact") and shift_marked(rec(p=q.tolist(), c=[0] * 11 + [1] * 19), "contact")
    (best, res, oracle) = choose_surrogate([r], [s])
    assert best == (0.05, 1) and res[0.05, 1][:2] == (1, 1) and res[0.08, 0][:2] == (1, 0), res
    # ties: the arm far from both, every candidate scores 1/2 -> the smallest r_c, then no direction
    assert choose_surrogate([rec(p=p.tolist(), eef=far)], [rec(p=q.tolist(), eef=far)])[0] == (0.03, 0)
    # caught counts push episodes only: an object that wobbles +-4 mm (never 5 mm from its start) still engages
    wob = still.copy()
    wob[5:13] -= [0.004, 0, 0]
    wob[13:] += [0.004, 0, 0]
    nw = rec(p=wob.tolist(), eef=far)
    assert not pushed(nw) and engaged_steps(nw, seen(nw), orx.EPS) == list(range(13, 21))
    assert filter_rates([r, nw], [s], "surr", 0.05, True)[0::2] == (1, 1)
    # the replay measures to the record's box (journal item 14): the arm 8 cm above the object's origin, moving with
    # it, is 2.7 cm above a bowl's box (centre 2.6 cm above p, half sides 7.8 x 7.9 x 2.7 cm)
    bowl = [0.0, 0.0, 0.026, 0.078, 0.079, 0.027]
    up, sd = rec(p=p.tolist(), eef=(p + [0, 0, 0.08]).tolist()), rec(p=q.tolist(), eef=(q + [0, 0, 0.08]).tolist(),
                                                                      kind="close")
    assert filter_rates([up], [sd], "surr", 0.05, True)[:2] == (0, 1)
    assert filter_rates([dict(up, box=bowl)], [dict(sd, box=bowl)], "surr", 0.05, True)[:2] == (1, 0)
    # exact ties: caught 0/10 + passed 3/10 at 3 cm equals 1/10 + 2/10 at 5 cm, which floats would rank higher
    def push3(dist):  # a 9 mm push at 13 along +x, the arm dist sideways from p(12) and moving with the object
        return rec(p=p.tolist(), eef=(p + [-0.009, dist, 0]).tolist())

    def shift4(dist):  # a 3 cm external shift seen at 11 along +x, the arm dist sideways from p(10), moving with it
        z = still.copy()
        z[11:] += [0.03, 0, 0]
        return rec(p=z.tolist(), eef=(z + [-0.03, dist, 0]).tolist(), kind="close")

    p3, p4 = [push3(0.04)] + [push3(0.2)] * 9, [shift4(0.01)] * 7 + [shift4(0.04), shift4(0.06), shift4(0.2)]
    best, res, _ = choose_surrogate(p3, p4)
    assert [res[c, 0][:2] for c in RCS] == [(0, Fraction(3, 10)), (Fraction(1, 10), Fraction(1, 5)),
                                           (Fraction(1, 10), Fraction(1, 10))], res
    assert best == (0.03, 0), best

    # choice 1: R = the last plan at or before t_fire (plan 10, not plan 0); N after it and before the close (70);
    # P95 of |diff . u| over 1, 2, 3, 4, 10 cm, linear: 4 + 0.8 * 6 = 8.8 cm (median 3, max 10)
    ch = np.zeros((50, 4))
    ch[:, 0], ch[:, 3] = 0.5, -1.0
    ch[20:, 3] = 1.0
    pl = ([[0, [0, -0.2, 0], ch.tolist()], [10, [0, 0, 0], ch.tolist()]]
          + [[20 + 10 * i, [0, y, 0], ch.tolist()] for i, y in enumerate((0.01, 0.02, 0.03, 0.04, 0.10))]
          + [[70, [0, 1.0, 0], ch.tolist()]])
    tau, n = tau_kappa([rec(plans=pl, angle=math.pi / 2, t_close=70)])  # u = y
    assert n == 5 and all(abs(v - 0.088) < 1e-9 for v in tau.values()), tau

    # choice 2: fallback plans only; K = 20 with fewer than 20; else the smallest error
    def log(m, fb=True):  # Delta = 3 cm along y, S = 0.8 Delta
        return {"fb": fb, "nd": 0.03, "d": [0, 0.03, 0], "S": [0, 0.024, 0], "m": m}

    t_eq = {10: 0.01, 20: 0.01, 30: 0.01}
    good, off20 = {"10": 0.0, "20": 0.024, "30": 0.03}, {"10": 0.024, "20": 0.0, "30": 0.0}  # kappa_model 0.8
    assert choose_k([dict(rec(), kappa_log=[log(good)] * 25)], t_eq)[0] == 20
    assert choose_k([dict(rec(), kappa_log=[log(off20)] * 25)], t_eq)[0] == 10
    assert choose_k([dict(rec(), kappa_log=[log(off20)] * 19)], t_eq)[0] == 20  # fewer than 20
    # non-fallback plans count neither in the error (K = 20 errs by 0.8 on them) nor towards the 20
    k, _, n = choose_k([dict(rec(), kappa_log=[log(good)] * 25 + [log(off20, fb=False)] * 100)], t_eq)
    assert (k, n) == (20, 25)
    # alpha: kappa_model = alpha + (1 - alpha) S.Delta_hat / |Delta| = 1 at alpha 1, 0.9 at 0.5 (0.8 without alpha)
    assert choose_k([dict(rec(), alpha=1.0, kappa_log=[log({"10": 0.024, "20": 0.03, "30": 0.0})] * 20)], t_eq)[0] == 20
    assert choose_k([dict(rec(), alpha=0.5, kappa_log=[log({"10": 0.024, "20": 0.027, "30": 0.03})] * 20)],
                    t_eq)[0] == 20
    # tau per K: K = 10's 2.4 cm lies in its 3 cm dead band (kappa_hat 0, error 0.8)
    assert choose_k([dict(rec(), kappa_log=[log({"10": 0.024, "20": 0.02, "30": 0.0})] * 20)],
                    {10: 0.03, 20: 0.001, 30: 0.001})[0] == 20

    # choice 4: a still object; false engagements come from noise only, and fall as eps' and smoothing grow
    ctl = [dict(rec(n=60, s=50, t_close=None), seed=sd) for sd in range(40)]
    stp = []
    for sd in range(40):
        q = np.tile([0.1, 0.0, 0.9], (60, 1))
        q[11:] += [0, 0.04, 0]
        stp.append(dict(rec(n=60, s=50, p=q.tolist(), t_close=None), seed=100 + sd, kind="step"))
    best, res, vmin, ppc = choose_noise(stp, ctl)
    assert res[1.0, 0.005][0] > res[0.2, 0.020][0] and res[best][0] <= FALSE_MAX and best[0] < 1, res
    assert 4 <= res[best][1] < math.inf, res  # the tracker lags 3 steps: seen at t_fire + 4 at the earliest
    # PPC at the chosen beta: unsmoothed (beta 1) the 1 cm noise crosses every threshold
    assert vmin == 0.010 and ppc[0.010] <= FALSE_MAX < ppc[0.005], ppc
    # the pick itself: the fastest with <= 5% false; ties larger eps', then smaller beta; none <= 5%: the fewest false
    res = {(1.0, 0.005): (0.5, 3.0), (0.5, 0.010): (0.05, 6.0), (0.3, 0.015): (0.02, 6.0), (0.2, 0.015): (0.0, 6.0)}
    assert pick_noise(res) == (0.2, 0.015)
    assert pick_noise({c: (f + 0.1, m) for c, (f, m) in res.items()}) == (0.2, 0.015)
    assert pick_noise({(1.0, 0.005): (0.5, 3.0), (0.5, 0.010): (0.3, 6.0), (0.3, 0.015): (0.3, 5.0)}) == (0.3, 0.015)
    # PPC: the smallest v_min with <= 5% false; none (journal item 15): the fewest false, ties the larger
    assert pick_vmin({0.001: 0.5, 0.002: 0.05, 0.005: 0.0, 0.010: 0.0}) == 0.002
    assert pick_vmin({0.001: 0.5, 0.002: 0.3, 0.005: 0.2, 0.010: 0.2}) == 0.010
    assert pick_vmin({0.001: 0.5, 0.002: 0.3, 0.005: 0.1, 0.010: 0.2}) == 0.005
    # PPC measures only while the executing chunk lasts: a 2 cm jump seen at 50 (t - t_obs = H) is not measured
    def jump(t):
        z = np.tile([0.1, 0.0, 0.9], (60, 1))
        z[t:] += [0.02, 0, 0]
        return rec(n=60, s=50, t_close=None, p=z.tolist()), z

    assert not ppc_fires(*jump(50), 0.01) and ppc_fires(*jump(51), 0.01) and ppc_fires(*jump(49), 0.01)

    # the padding test: all 8 pairs equal and <= 15% slower adopts; one unequal pair or 16% slower does not
    def pr(arm, task, i, engaged=None, h=None):
        return {"suite": "libero_spatial", "task": task, "init": i, "kind": "control", "arm": arm,
                "act_hash": h or f"{task}.{i}", "t_engage": engaged, "g_on": engaged is not None}

    same = [pr(arm, tk, i) for arm in ("none", "GR") for tk in (0, 1) for i in range(48, 52)]
    assert pad_test(same, same, 1.1, 1.0)[0] and not pad_test(same, same, 1.16, 1.0)[0]
    assert not pad_test(same[:-1] + [pr("GR", 1, 51, h="x")], same, 1.0, 1.0)[0]
    assert pad_test(same[:-1] + [pr("GR", 1, 51, engaged=12, h="x")], same, 1.0, 1.0)[0]  # an engaged GR: no pair
    no151 = [r for r in same if (r["task"], r["init"]) != (1, 51)]
    for short in (same[:-1], [r for r in same if r["init"] != 51], same + same[:1],  # a skipped batch, a duplicate
                  no151 + same[:1] + same[8:9], same[:-1] + [pr("GR", 1, 52)]):  # 16 lines, 7 pairs; GR elsewhere
        for runs in ((short, same), (same, short)):
            try:
                pad_test(*runs, 1.0, 1.0)
            except SystemExit:
                continue
            raise AssertionError("the padding test ran without all 8 pairs")
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
