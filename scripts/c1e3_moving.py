"""C1-E3d add-ons, report only (docs/c1/e3-detectors.md, read after the data):
1. the lag of the realistic point split by shift class into the eyes' lag floor (point L3), sigma + dropouts and the
   part a better pipeline removes, also weighted by the oracle reflex's gain per class;
2. objects that keep moving after the shift (pilot traces, split into drift and ejections): following by EMA vs
   Kalman with a jump vs Kalman with a constant-velocity model (with and without predicting the L lagged steps forward).
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e3_moving.py
"""

import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import c1e2_summary as e2  # noqa: E402
import c1e3_detectors as dt  # noqa: E402
import c1e3_summary as s  # noqa: E402
import handoff as hf  # noqa: E402

MOVING, EJECT = 0.01, 0.20  # extra motion beyond the shift within 10 steps: drift above 1 cm, ejection above 20 cm
QA = (0.001, 0.005)  # constant-velocity Kalman: acceleration noise per step, m
PV0 = 0.01 ** 2  # velocity prior at the restart, (m/step)^2
WINDOWS = (10, 20)


def lag_split(d, idx):
    """Per class: grid lag = L floor + sigma/dropouts + removable; pooled and weighted by the oracle G-R gain."""
    real, best, floor = (d["points"][t][k]["fol"][f] for t, k, f in
                         (("real", "grid", "ema0.2"), ("real", "glr", "kfjump"), ("L3", "glr", "kfjump")))
    n = np.array([x["n"] for x in d["points"]["real"]["grid"]["by_class"]], float)
    g, b, fl = (np.array(x["lag_by_class"]) for x in (real, best, floor))
    parts = {"L": fl, "sigma": b - fl, "removable": g - b}
    gain = [e2.rnd(e2.diff_pp(s.prs(idx, "step", 0.0, "ACF", "GR", "none", range(20), (c,)))) for c in (0, 1, 2)]
    ok = lambda kind, c: [r["success"] for cell in "ACF" for k, r in idx[(kind, cell, "none", 0.0)].items()  # noqa: E731
                          if k[2] < 10 and (c is None or r["mag_class"] == c)]
    ctl = float(np.mean(ok("control", None)))  # 'none' loses vs control: all step episodes of the class, inits 0-9
    loss = [round(100 * (ctl - float(np.mean(ok("step", c)))), 2) for c in (0, 1, 2)]
    w = np.maximum([x["diff_pp"] for x in gain], 0.0)

    def share(weight):
        tot = (weight * g).sum()
        return {k: float((weight * v).sum() / tot) for k, v in parts.items()}
    big = np.array([0.0, 1.0, 1.0])
    return {"n": n.tolist(), "lag_by_class": g.tolist(), "parts_by_class": {k: v.tolist() for k, v in parts.items()},
            "class_share_of_lag": (n * g / (n * g).sum()).tolist(), "all": share(n), "shift_3_6": share(n * big),
            "gain_by_class": gain, "gain_weighted": share(n * w), "control_none_pct": round(100 * ctl, 2),
            "none_loss_vs_control_by_class": loss}


def cv_follow(meas, deliv, ref, tob, P, tf, ta, kh, sig, lag, qa, predict, n_steps):
    """Constant-velocity Kalman per axis restarted at k_hat (x = reference, v = 0); estimate = x (+ lag * v)."""
    N = len(tf)
    st = tf[:, None] + 1 + np.arange(n_steps)
    on = (ta[:, None] >= 0) & (st >= ta[:, None])
    base = ref[np.arange(N), dt.W + tob[np.maximum(ta, 0)]]
    D = P[np.arange(N)[:, None], st] - P[np.arange(N), tf][:, None]
    F, Qm, R = np.array([[1.0, 1.0], [0.0, 1.0]]), qa ** 2 * np.array([[0.25, 0.5], [0.5, 1.0]]), sig ** 2
    x = np.zeros((N, n_steps, 3))
    for e in range(N):
        if ta[e] < 0:
            continue
        X = np.stack([base[e], np.zeros(3)])  # rows: position, velocity; columns: axes
        Pm = np.diag([dt.P0, PV0])
        for t in range(kh[e], st[e, -1] + 1):
            if t > kh[e]:
                X, Pm = F @ X, F @ Pm @ F.T + Qm
            if deliv[e, dt.W + t]:
                K = Pm[:, 0] / (Pm[0, 0] + R)
                X = X + np.outer(K, meas[e, dt.W + t] - X[0])
                Pm = Pm - np.outer(K, Pm[0])
            rel = t - tf[e] - 1
            if 0 <= rel < n_steps:
                x[e, rel] = X[0] + (lag * X[1] if predict else 0.0)
    C = np.where(on[..., None], x - base[:, None], 0.0)
    return np.linalg.norm((D - C)[..., :2], axis=2).mean(1)


def moving(d):
    """Pilot shift episodes split by the object's extra motion within 10 steps (after the data, report only):
    still (<= 1 cm), drift (1-20 cm), eject (> 20 cm: the object is thrown at the shift itself, no contact)."""
    T = 301
    sig, lag, q = hf.NOISE["real"]
    sig_m = max(sig, dt.SIG_FLOOR)
    lags, groups = {}, []
    for cell, eps in dt.pilot_sets()["step"].items():
        eps = [x for x in eps if x["tf"] >= 0]
        tf, P = dt.arr(eps, "tf"), dt.truth(eps, T)
        rows = np.arange(len(eps))
        D10 = np.linalg.norm((P[rows[:, None], tf[:, None] + 1 + np.arange(10)] - P[rows, tf][:, None])[..., :2], axis=2)
        extra = D10.max(1) - np.linalg.norm(dt.arr(eps, "delta")[:, :2], axis=1)
        groups.append(np.where(extra <= MOVING, "still", np.where(extra <= EJECT, "drift", "eject")))
        z, u = dt.noise([x["seed"] for x in eps], T, 3)
        meas, deliv = dt.measure(P, z, u, sig, lag, q)
        tob = dt.tobs(cell, T)
        st, est, ref = dt.stats(meas, deliv, tob, sig_m)
        for det, name, thr in (("grid", "ema0.2", dt.EPS), ("glr", "glr", d["points"]["real"]["glr"]["thr"])):
            stat, kh = st[name]
            for nw in WINDOWS:
                ta = dt.first(stat, thr, np.ones(len(eps), int), tf + nw)
                khat = kh[rows, np.maximum(ta, 0)]
                for f in ("ema0.2", "kfjump"):
                    lags.setdefault((det, nw, f), []).append(dt.follow(f, meas, deliv, est, ref, tob, P, tf, ta, khat, sig_m, nw)[0])
                for qa in QA:
                    for pred in (False, True):
                        lags.setdefault((det, nw, f"cv{qa * 1000:g}mm" + ("+pred" if pred else "")), []).append(
                            cv_follow(meas, deliv, ref, tob, P, tf, ta, khat, sig_m, lag, qa, pred, nw))
    g = np.concatenate(groups)
    out = {sub: {"n": int((g == sub).sum())} for sub in ("still", "drift", "eject")}
    for (det, nw, f), v in lags.items():
        v = np.concatenate(v)
        for sub in out:
            out[sub].setdefault(f"{det} {nw}", {})[f] = round(100 * float(v[g == sub].mean()), 2)
    return out


def pilot_without(d, drop):
    """The pilot check (rule R3 layout) with the shift episodes of the given groups left out."""
    T, keep = 301, []
    sets = dt.pilot_sets_by_cell()
    for cell, eps in sets["step"].items():
        eps = [x for x in eps if x["tf"] >= 0]
        if eps:
            P, tf = dt.truth(eps, T), dt.arr(eps, "tf")
            rows = np.arange(len(eps))
            D10 = np.linalg.norm((P[rows[:, None], tf[:, None] + 1 + np.arange(10)] - P[rows, tf][:, None])[..., :2], axis=2)
            extra = D10.max(1) - np.linalg.norm(dt.arr(eps, "delta")[:, :2], axis=1)
            grp = np.where(extra <= MOVING, "still", np.where(extra <= EJECT, "drift", "eject"))
            eps = [x for x, gr in zip(eps, grp) if gr not in drop]
        sets["step"][cell] = eps
        keep.append(len(eps))
    h = {k: d["points"]["real"][k]["thr"] for k in dt.DETS}
    rows = dt.point("real", sets, (), T, max(hf.NOISE["real"][0], dt.SIG_FLOOR), h)[2]
    g, w = rows["grid"]["fol"]["ema0.2"], rows["glr"]["fol"]["kfjump"]
    return {"n_step": sum(keep), "grid_lag_cm": round(g["lag_cm"], 3), "grid_S": round(g["S"], 3),
            "grid_S_by_cell": {k: round(v, 3) for k, v in g["S_by_cell"].items()},
            "glr_kf_lag_cm": round(w["lag_cm"], 3), "cut": round(1 - w["lag_cm"] / g["lag_cm"], 3)}


def main():
    d = json.load(open("results/c1-e3-det/detectors.json"))
    raw = [json.loads(line) for line in open(dt.GRID) if '"libero_10"' not in line]
    print(json.dumps({"lag_split": lag_split(d, s.index(raw)), "moving": moving(d),
                      "pilot_all": pilot_without(d, ()), "pilot_no_eject": pilot_without(d, ("eject",)),
                      "pilot_still_only": pilot_without(d, ("eject", "drift"))}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
