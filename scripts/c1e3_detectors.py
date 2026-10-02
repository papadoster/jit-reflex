"""C1-E3d: offline shift detectors and post-alarm following on the C1-E3 noise model (spec
docs/superpowers/specs/2026-10-03-c1-e3-detectors-offline-design.md). Mac only, numpy, no simulator.

Episodes are the grid's oracle G-R records (A, C, F): the object is still, a step shift is seen from t_fire + 1, the
noise stream of each seed is the grid's (Tracker generator [seed, 3]); calibration reuses the control windows with
streams [seed, 4..6]. Pilot traces (real object positions) are the robustness check. Detectors compare against the
estimate at the executing plan's observation, as the C1-E3 agent does.
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e3_detectors.py --check   # Tracker bit-match, noise-free steps, gate
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e3_detectors.py > results/c1-e3-det/detectors.json
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import c1e3_summary as s  # noqa: E402
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402

GRID = "results/c1-e3/grid_all.jsonl"
PILOTS = ("results/c1-e3/pilot.jsonl", "results/c1-e3/pilot_b.jsonl")
W = hf.Tracker.WARM
BETA, EPS = 0.2, 0.02  # the grid's noisy-eyes setting (C1-E3 spec journal 18)
DETS = ("ema0.2", "ema0.5", "cusum", "glr", "nis")
FOLS = ("ema0.2", "win3", "win5", "kfjump")
NU, GLR_W, NIS_W, REF_N = 0.01, 30, 5, 20  # CUSUM jump of interest, GLR window, NIS window, reference frames (§3)
Q, P0, SIG_FLOOR = 1e-3 ** 2, 0.05 ** 2, 1e-3  # Kalman process noise per step, jump prior, sigma floor (§3, §4)
LAG_N, FALSE_AT, H_FLOOR = 10, 0.01, 1e-9  # lag window, operating false rate, threshold floor for sigma = 0 points
CURVE = (0.005, 0.01, 0.02, 0.05, 0.10)
CALIB = (4, 5, 6)
CHUNK = np.zeros((orx.H, 7))


def ep(r, p=None):
    end = r["t_close"] if r["t_close"] is not None else r["steps"]  # alarms count before the first close (outcome())
    return {"seed": r["seed"], "cell": r["cell"], "tf": -1 if r["t_fire"] is None else r["t_fire"], "end": end,
            "tc": r["t_close"], "steps": r["steps"], "cls": r["mag_class"], "p": p,
            "delta": r["mag"] * np.array([np.cos(r["angle"]), np.sin(r["angle"]), 0.0])}


def truth(eps, T):
    """(N, T, 3) true object position: the pilot trace, else still with the step seen from t_fire + 1."""
    P = np.zeros((len(eps), T, 3))
    for e, x in enumerate(eps):
        if x["p"] is not None:
            n = min(T, len(x["p"]))
            P[e, :n], P[e, n:] = x["p"][:n], x["p"][n - 1]
        elif x["tf"] >= 0:
            P[e, x["tf"] + 1:] = x["delta"]
    return P


_NOISE = {}


def noise(seeds, T, stream):
    """The Tracker's draws, in its order: per update one normal(0, 1, 3), then one random()."""
    out_z, out_u = np.empty((len(seeds), W + T, 3)), np.empty((len(seeds), W + T))
    for e, sd in enumerate(seeds):
        if (sd, stream) not in _NOISE:
            rng, z, u = np.random.default_rng([sd, stream]), np.empty((W + T, 3)), np.empty(W + T)
            for i in range(W + T):
                z[i], u[i] = rng.normal(0.0, 1.0, 3), rng.random()
            _NOISE[sd, stream] = z, u
        out_z[e], out_u[e] = _NOISE[sd, stream]
    return out_z, out_u


def measure(P, z, u, sig, lag, q):
    """Stream positions 0..W-1 are the warm-up on p(0); position W + t sees p(t - lag). A frame is missing when
    u < q (never the first): nothing is updated then."""
    T = P.shape[1]
    src = np.concatenate([np.repeat(P[:, :1], W, 1), P[:, np.maximum(np.arange(T) - lag, 0)]], 1)
    deliv = u[:, :W + T] >= q
    deliv[:, 0] = True
    return src + sig * z[:, :W + T], deliv


def ema(meas, deliv, beta):
    est, e = np.empty_like(meas), meas[:, 0].copy()
    est[:, 0] = e
    for i in range(1, meas.shape[1]):
        e = np.where(deliv[:, i, None], beta * meas[:, i] + (1 - beta) * e, e)
        est[:, i] = e
    return est


def last_mean(meas, deliv, k):
    """Mean of the last k delivered frames up to each stream position (fewer at the start)."""
    N = meas.shape[0]
    cs = np.concatenate([np.zeros((N, 1, 3)), np.cumsum(meas * deliv[..., None], 1)], 1)
    cnt = np.concatenate([np.zeros((N, 1), int), np.cumsum(deliv, 1)], 1)
    out = np.empty_like(meas)
    for e in range(N):
        c = cnt[e, 1:]
        start = np.flatnonzero(deliv[e])[np.maximum(c - k, 0)]
        out[e] = (cs[e, 1:] - cs[e, start]) / (c - cnt[e, start])[:, None]
    return out


def tobs(cell, T):
    """Observation step of the executing plan at each observe step (-1 before the first plan): calls every s steps,
    answers after d, no triggered calls (the G family without T), same order as the C1-E3 runner."""
    sc, out = orx.Schedule(*orx.CELLS[cell]), np.full(T, -1)
    for t in range(T):
        out[t] = -1 if sc.t_obs is None else sc.t_obs
        sc.arrive(t)
        if sc.wants_call(t, False):
            sc.issue(t, CHUNK)
            sc.arrive(t)
    return out


def stats(meas, deliv, tob, sig):
    """Per detector: (statistic, change-step estimate k_hat) at every observe step; 0 before the first plan."""
    N, T = meas.shape[0], meas.shape[1] - W
    ok, pos, steps = tob >= 0, W + np.maximum(tob, 0), np.broadcast_to(np.arange(T), (N, T))
    out, ests = {}, {}
    for b in (0.2, 0.5):
        ests[b] = est = ema(meas, deliv, b)
        out[f"ema{b}"] = (np.where(ok, np.linalg.norm(est[:, W:] - est[:, pos], axis=2), 0.0), steps)
    ref = last_mean(meas, deliv, REF_N)
    cs = np.concatenate([np.zeros((N, 1, 3)), np.cumsum(meas * deliv[..., None], 1)], 1)
    cnt = np.concatenate([np.zeros((N, 1), int), np.cumsum(deliv, 1)], 1)
    cus, cus_k, glr, glr_k = np.zeros((N, T)), np.zeros((N, T), int), np.zeros((N, T)), np.zeros((N, T), int)
    rows = np.arange(N)
    for c in np.unique(tob[ok]):  # one segment per plan: data after its observation, reference at it
        seg = np.flatnonzero(tob == c)
        a, b, rc = seg[0], seg[-1], ref[:, W + c]
        sp, sn = np.zeros((N, 3)), np.zeros((N, 3))
        zp, zn = np.full((N, 3), c + 1), np.full((N, 3), c + 1)  # first step of each sum's current excursion
        for t in range(c + 1, b + 1):
            i, d = W + t, deliv[:, W + t, None]
            e = meas[:, i] - rc
            sp = np.where(d, np.maximum(0.0, sp + e - NU / 2), sp)
            sn = np.where(d, np.maximum(0.0, sn - e - NU / 2), sn)
            zp, zn = np.where(sp == 0, t + 1, zp), np.where(sn == 0, t + 1, zn)
            if t < a:
                continue
            both, zz = np.concatenate([sp, sn], 1), np.concatenate([zp, zn], 1)
            j = both.argmax(1)
            cus[:, t], cus_k[:, t] = both[rows, j], zz[rows, j]
            ks = np.arange(max(c + 1, t - GLR_W + 1), t + 1)
            n = cnt[:, i + 1, None] - cnt[:, W + ks]
            mean = (cs[:, i + 1, None] - cs[:, W + ks]) / np.maximum(n, 1)[..., None]
            g = np.where(n > 0, n * ((mean - rc[:, None]) ** 2).sum(2) / (2 * sig ** 2), 0.0)
            j = g.argmax(1)
            glr[:, t], glr_k[:, t] = g[rows, j], ks[j]
    out["cusum"], out["glr"] = (cus, cus_k), (glr, glr_k)
    x, P, nis = meas[:, 0].copy(), np.full(N, sig ** 2), np.zeros((N, W + T))  # static Kalman, never reset
    for i in range(1, W + T):
        P = P + Q
        d, S, v = deliv[:, i], P + sig ** 2, meas[:, i] - x
        nis[:, i] = np.where(d, (v ** 2).sum(1) / S, 0.0)
        K = np.where(d, P / S, 0.0)
        x, P = x + K[:, None] * v, (1 - K) * P
    csn = np.concatenate([np.zeros((N, 1)), np.cumsum(nis, 1)], 1)
    t = np.arange(T)
    win = csn[:, W + t + 1] - csn[:, W + t + 1 - NIS_W]
    out["nis"] = (np.where(ok, win, 0.0), np.broadcast_to(np.maximum(t - NIS_W + 1, 0), (N, T)))
    return out, ests[0.2], ref


def first(stat, h, lo, hi):
    """First step in [lo, hi] (per episode) with stat > h, else -1."""
    t = np.arange(stat.shape[1])
    m = (stat > h) & (t >= lo[:, None]) & (t <= hi[:, None])
    return np.where(m.any(1), m.argmax(1), -1)


def wmax(stat, lo, hi):
    t = np.arange(stat.shape[1])
    return np.where((t >= lo[:, None]) & (t <= hi[:, None]), stat, -np.inf).max(1)


def follow(fol, meas, deliv, est, ref, tob, P, tf, ta, kh, sig, n=LAG_N):
    """Applied-correction error over steps tf+1..tf+n, xy, m: (lag, trigger part, following part) per episode.
    Before the alarm the correction is 0; after it, estimate - reference at the alarm's plan observation."""
    N, rows = len(tf), np.arange(len(tf))[:, None]
    st = tf[:, None] + 1 + np.arange(n)
    on = (ta[:, None] >= 0) & (st >= ta[:, None])
    pos_a = W + tob[np.maximum(ta, 0)]
    D = P[rows, st] - P[np.arange(N), tf][:, None]
    if fol == "ema0.2":
        x, base = est[rows, W + st], est[np.arange(N), pos_a]
    elif fol.startswith("win"):
        x, base = last_mean(meas, deliv, int(fol[3:]))[rows, W + st], ref[np.arange(N), pos_a]
    else:  # Kalman with a jump model: restart at k_hat from the reference with a wide prior
        base, R = ref[np.arange(N), pos_a], sig ** 2
        x, xk, Pk = np.zeros((N, n, 3)), base.copy(), np.full(N, P0)
        k0 = np.where(ta >= 0, kh, 10 ** 9)
        for t in range(int(k0.min()) if (ta >= 0).any() else 0, int(st.max()) + 1):
            act = t >= k0
            Pk = np.where(act, Pk + Q, Pk)
            K = np.where(act & deliv[:, W + t], Pk / (Pk + R), 0.0)
            xk, Pk = xk + K[:, None] * (meas[:, W + t] - xk), (1 - K) * Pk
            rel = t - tf - 1
            sel = (rel >= 0) & (rel < n)
            x[sel, rel[sel]] = xk[sel]
    C = np.where(on[..., None], x - base[:, None], 0.0)
    e = np.linalg.norm((D - C)[..., :2], axis=2)
    trig = np.where(on, 0.0, np.linalg.norm(D[..., :2], axis=2))
    return e.mean(1), trig.mean(1), np.where(on, e, 0.0).mean(1)


def run_cell(eps, cell, T, tag, sig_model, stream):
    sig, lag, q = hf.NOISE[tag]
    P = truth(eps, T)
    z, u = noise([x["seed"] for x in eps], T, stream)
    meas, deliv = measure(P, z, u, sig, lag, q)
    tob = tobs(cell, T)
    st, est, ref = stats(meas, deliv, tob, sig_model)
    return {"st": st, "est": est, "ref": ref, "meas": meas, "deliv": deliv, "tob": tob, "P": P}


def arr(eps, key):
    return np.array([x[key] for x in eps])


def summarize(cells, h, sig_model):
    """Detection and lag metrics for every detector at thresholds h and for the grid's setting (EMA 0.2, EPS)."""
    rows = {}
    for det in (*DETS, "grid"):
        name, thr = ("ema0.2", EPS) if det == "grid" else (det, h[det])
        acc = {"a": [], "tf": [], "cls": [], "ctrl": [], "cell": [], **{f: [] for f in FOLS}}
        for cell, (S, Cn, eS, eC) in cells.items():
            fired = arr(eS, "tf") >= 0
            tf, end = arr(eS, "tf")[fired], arr(eS, "end")[fired]
            stat, kh = (x[fired] for x in S["st"][name])
            one = np.ones(len(tf), int)
            acc["a"].append(first(stat, thr, one, end - 1))
            acc["tf"].append(tf)
            acc["cls"].append(arr(eS, "cls")[fired])
            acc["cell"] += [cell] * len(tf)
            acc["ctrl"].append(first(Cn["st"][name][0], thr, np.ones(len(eC), int), arr(eC, "end") - 1))
            ta = first(stat, thr, one, tf + LAG_N)
            khat = kh[np.arange(len(tf)), np.maximum(ta, 0)]
            sub = {k: (S[k][fired] if k in ("meas", "deliv", "est", "ref", "P") else S[k]) for k in S}
            for f in FOLS:
                acc[f].append(follow(f, sub["meas"], sub["deliv"], sub["est"], sub["ref"], sub["tob"], sub["P"], tf,
                                     ta, khat, sig_model))
        a, tf, cls = (np.concatenate(acc[k]) for k in ("a", "tf", "cls"))
        cellv = np.array(acc["cell"])

        def det_stats(m, ctrl):
            on = a[m] > tf[m]
            return {"false_ctrl": None if ctrl is None else float((ctrl >= 0).mean()),
                    "false_step": float(((a[m] >= 0) & (a[m] <= tf[m])).mean()), "missed": float((a[m] < 0).mean()),
                    "delay_med": float(np.median(a[m][on] - tf[m][on])) if on.any() else None, "n": int(m.sum())}

        ctrl = np.concatenate(acc["ctrl"])
        row = {"thr": float(thr), **det_stats(np.ones(len(a), bool), ctrl), "n_ctrl": int(len(ctrl)),
               "missed_by_class": [float((a < 0)[cls == c].mean()) for c in (0, 1, 2)],
               "missed_3_6": float((a < 0)[cls > 0].mean()),
               "by_class": [det_stats(cls == c, None) for c in (0, 1, 2)],
               "by_cell": {c: det_stats(cellv == c, acc["ctrl"][i]) for i, c in enumerate(cells)}, "fol": {}}
        for f in FOLS:
            lag, trig, fol = (100 * np.concatenate([x[i] for x in acc[f]]) for i in range(3))
            ratio = lambda m: float(fol[m].sum() / lag[m].sum()) if lag[m].sum() > 0 else None  # noqa: E731
            row["fol"][f] = {"lag_cm": float(lag.mean()), "trig_cm": float(trig.mean()), "follow_cm": float(fol.mean()),
                             "S": ratio(np.ones(len(lag), bool)),
                             "lag_by_class": [float(lag[cls == c].mean()) for c in (0, 1, 2)],
                             "S_by_class": [ratio(cls == c) for c in (0, 1, 2)],
                             "lag_by_cell": {c: float(lag[cellv == c].mean()) for c in cells},
                             "S_by_cell": {c: ratio(cellv == c) for c in cells}}
        rows[det] = row
    return rows


def point(tag, sets, calib, T, sig_model, h=None):
    """All cells at one noise point; thresholds from the calibration controls unless given."""
    cells, cal = {}, {d: [] for d in DETS}
    for cell in "ACF":
        eS, eC = sets["step"][cell], sets["ctrl"][cell]
        if not eS or not eC:  # the pilot has no traces in F
            continue
        cells[cell] = (run_cell(eS, cell, T, tag, sig_model, 3), run_cell(eC, cell, T, tag, sig_model, 3), eS, eC)
        if h is None:
            for strm in calib:
                Cc = run_cell(eC, cell, T, tag, sig_model, strm)
                for d in DETS:
                    cal[d].append(wmax(Cc["st"][d][0], np.ones(len(eC), int), arr(eC, "end") - 1))
    if h is None:
        h = {d: max(float(np.quantile(np.concatenate(cal[d]), 1 - FALSE_AT)), H_FLOOR) for d in DETS}
    return cells, h, summarize(cells, h, sig_model)


def curves(cells):
    """Delay and misses against false alarms; thresholds from the main controls' window maxima."""
    out = {}
    for det in DETS:
        mx = np.concatenate([wmax(Cn["st"][det][0], np.ones(len(eC), int), arr(eC, "end") - 1)
                             for S, Cn, eS, eC in cells.values()])
        out[det] = []
        for fa in CURVE:
            thr = max(float(np.quantile(mx, 1 - fa)), H_FLOOR)
            a, tf = [], []
            for S, Cn, eS, eC in cells.values():
                fired = arr(eS, "tf") >= 0
                a.append(first(S["st"][det][0][fired], thr, np.ones(fired.sum(), int), arr(eS, "end")[fired] - 1))
                tf.append(arr(eS, "tf")[fired])
            a, tf = np.concatenate(a), np.concatenate(tf)
            on = a > tf
            out[det].append({"false_target": fa, "delay_med": float(np.median(a[on] - tf[on])) if on.any() else None,
                             "missed": float((a < 0).mean()), "false_step": float(((a >= 0) & (a <= tf)).mean())})
    return out


def grid_sets(idx):
    sets = {"step": {}, "ctrl": {}}
    for cell in "ACF":
        sets["step"][cell] = [ep(r) for k, r in sorted(idx[("step", cell, "GR", 0.0)].items()) if k[2] < 20]
        sets["ctrl"][cell] = [ep(r) for k, r in sorted(idx[("control", cell, "GR", 0.0)].items()) if k[2] < 10]
    return sets


def pilot_sets():
    sets = {"step": {"A": [], "C": []}, "ctrl": {"A": [], "C": []}}
    for f in PILOTS:
        for line in open(f):
            r = json.loads(line)
            if r.get("trace"):
                sets["ctrl" if r["kind"] == "control" else "step"][r["cell"]].append(ep(r, np.array(r["trace"]["p"])))
    return sets


def gate(raw, T):
    """Offline EMA 0.2 alarms at EPS against the recorded t_engage of every noisy G-R arm (window up to the close,
    inclusive, as in the runner)."""
    agree, by = 0, {}
    recs = [r for r in raw if r["arm"].startswith("GR@") and r["filter"] == ""]
    for cell in sorted({r["cell"] for r in recs}):
        for tag in hf.NOISE:
            rs = [r for r in recs if r["cell"] == cell and r["arm"] == f"GR@{tag}"]
            if not rs:
                continue
            eps = [ep(r) for r in rs]
            S = run_cell(eps, cell, T, tag, SIG_FLOOR, 3)
            hi = np.array([r["t_close"] if r["t_close"] is not None else r["steps"] - 1 for r in rs])
            a = first(S["st"]["ema0.2"][0], EPS, np.ones(len(rs), int), hi)
            rec = np.array([-1 if r["t_engage"] is None else r["t_engage"] for r in rs])
            ok = int((a == rec).sum())
            agree += ok
            by[f"{cell} {tag}"] = [ok, len(rs)]
    n = sum(v[1] for v in by.values())
    return {"agree": agree / n, "n": n, "by": by}


def check(raw, T):
    sd = 12345
    for tag in ("real", "s2", "L6"):
        sig, lag, q = hf.NOISE[tag]
        x = {"seed": sd, "cell": "C", "tf": 30, "end": 60, "tc": 60, "steps": 80, "cls": 1, "p": None,
             "delta": np.array([0.03, 0.01, 0.0])}
        P = truth([x], T)
        tr = hf.Tracker(sd, sig, lag, q, BETA)
        ref = np.array([tr(P[0, t]) for t in range(T)])
        meas, deliv = measure(P, *noise([sd], T, 3), sig, lag, q)
        assert np.array_equal(ema(meas, deliv, BETA)[0, W:], ref), tag
    for lag in (0, 3):  # noise-free step: every detector fires at t_fire + 1 + lag
        x = {"seed": sd, "cell": "C", "tf": 30, "end": 90, "tc": 90, "steps": 100, "cls": 1, "p": None,
             "delta": np.array([0.03, 0.0, 0.0])}
        P = truth([x], T)
        meas, deliv = measure(P, *noise([sd], T, 3), 0.0, lag, 0.0)
        st, _, _ = stats(meas, deliv, tobs("C", T), SIG_FLOOR)
        for d in DETS:
            a = first(st[d][0], H_FLOOR, np.array([1]), np.array([89]))[0]
            assert a == 31 + lag, (d, lag, a)
    g = gate(raw, T)
    print(json.dumps(g, indent=1))
    assert g["agree"] >= 0.90, "replication gate failed (spec §7)"
    print("check ok")


def main():
    raw = [json.loads(line) for line in open(GRID) if '"libero_10"' not in line]
    T = max(r["steps"] for r in raw) + 1
    if "--check" in sys.argv:
        return check(raw, T)
    g = gate(raw, T)
    assert g["agree"] >= 0.90, "replication gate failed (spec §7)"
    sets, out = grid_sets(s.index(raw)), {"gate": g, "points": {}, "one_setting": {}, "curves": {}}
    out["counts"] = {k: {c: [len(v), sum(x["tf"] >= 0 for x in v)] for c, v in sets[k].items()} for k in sets}
    h_real = None
    for tag in ("real", *hf.SWEEP):
        sig_model = max(hf.NOISE[tag][0], SIG_FLOOR)
        cells, h, rows = point(tag, sets, CALIB, T, sig_model)
        out["points"][tag], out["curves"][tag] = rows, curves(cells)
        if tag == "real":
            h_real = h
        else:  # one setting for all: the realistic point's thresholds and assumed sigma
            out["one_setting"][tag] = point(tag, sets, (), T, max(hf.NOISE["real"][0], SIG_FLOOR), h_real)[2]
    real, grid = out["points"]["real"], out["points"]["real"]["grid"]
    S = grid["fol"]["ema0.2"]["S"]
    out["R1"] = {"S": S, "need_2x2": 0.25 <= S <= 0.75}
    cand = [(row["fol"][f]["lag_cm"], d, f) for d, row in real.items() if d != "grid" and row["false_ctrl"] <= 0.02
            for f in FOLS]
    lag, d, f = min(cand)
    cut = 1 - lag / grid["fol"]["ema0.2"]["lag_cm"]
    out["R2"] = {"winner": [d, f], "lag_cm": lag, "grid_lag_cm": grid["fol"]["ema0.2"]["lag_cm"], "cut": cut,
                 "missed_3_6": real[d]["missed_3_6"], "grid_missed_3_6": grid["missed_3_6"],
                 "recommend_pod": bool(cut >= 0.30 and real[d]["missed_3_6"] <= grid["missed_3_6"])}
    pil = point("real", pilot_sets_by_cell(), (), T, max(hf.NOISE["real"][0], SIG_FLOOR), h_real)[2]
    pw, pg = pil[d], pil["grid"]
    pcut = 1 - pw["fol"][f]["lag_cm"] / pg["fol"]["ema0.2"]["lag_cm"]
    out["R3"] = {"false_ctrl": pw["false_ctrl"], "grid_false_ctrl": pg["false_ctrl"], "cut": pcut,
                 "holds": bool(pw["false_ctrl"] <= 0.02 and pcut >= 0.20)}
    out["pilot"] = pil
    print(json.dumps(out, indent=1))


def pilot_sets_by_cell():
    """The pilot traces in the point() layout; cell F has no traces, so its lists stay empty."""
    p = pilot_sets()
    return {k: {c: p[k].get(c, []) for c in "ACF"} for k in p}


if __name__ == "__main__":
    main()
