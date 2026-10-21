"""C1-E4 PBVS: a position-based visual servo next to the reflex (docs/superpowers/specs/2026-10-20-c1-e4-pbvs-design.md,
"PBVS spec"), and the noisy cell PPCL@glr + PPC@glr of the controls spec's journal "сводное решение владельца" (the
same journal of both specs and their later entries: the frozen 108 VS1.0 cases read as G(k = 1.0), V2eps the servo at
our own 5 mm gate, the noisy cell's table), on pod C after the controls grid (scripts/gpu_c1e4m.sh POD=C vgrid): the
runner plan per line, the arm model and its dead band from V0 (§3, band.json), the lambda selection from V1 (§4,
pbvs_k.json), the forecast (§8), the smoke gate, the controls lines the V lines pair with, and the reading (§5, §6).
Reuses scripts/c1e4c_summary.py (mlines, lk, select, mp, read_tempo) and scripts/c1e4m_summary.py; pure numpy, the
LeRobot env; run from the repository root. R = results/c1-e4/controls/pod_C:
    python scripts/c1e4v_summary.py --plan --line L [--band-json R/band.json] [--pbvs-k R/pbvs_k.json]
        L: vsmoke V0 V1 V2 V3d10 V3d20 (V3: both) Nglr V2eps, the runner lines as c1e4c_summary --plan (phase: L in
        lower case); exit 3 when V2, V3*, V2eps need pbvs_k.json, or V2eps band.json, not there yet; V2eps is empty
        unless band.json's delta > 15 mm
    python scripts/c1e4v_summary.py --band R/V0_*.jsonl [--out R/band.json]   and band_eps.json beside it when delta >
        15 mm (delta orx.EPS, the rest band.json's); exit 2: V0 incomplete
    python scripts/c1e4v_summary.py --select R/V1_*.jsonl [--out R/pbvs_k.json]   exit 2: V1 incomplete
    python scripts/c1e4v_summary.py --forecast R/tempo_grid.txt [--done R/V*.jsonl R/Nglr*.jsonl]   per line hours,
        $ and what is left; exit 3 when V1 + V2 + V3 cost above $4 (V3, then Nglr and V2eps, wait for the owner)
    python scripts/c1e4v_summary.py --smoke-check R/Vsmoke_pi05*.jsonl --controls R/C2_pi05_eval300_plus_d0.jsonl
        exit 0, or 3 (also when a smoke Base's e is not that file's)
    python scripts/c1e4v_summary.py --controls-check R/C2_*.jsonl R/C3_*.jsonl [--mode R/mode.json] [--k R/k_select.json]
        controls lines 2 and 3 complete (c1e4c_summary's <= 1 % missing) and FROZEN's Base cases V2's first 108 Plus,
        in order: exit 0, or 3
    python scripts/c1e4v_summary.py [read] R/grid_all_V.jsonl [--controls R/grid_all_C.jsonl] [--p2 results/c1-e4/part2]
            [--out R/summary_V.json]
        the checks before reading (§5), then the measurements Н1-Н3, И5 (and at the 5 mm gate) with no threshold and the
        report (§6, the journals); a failed check prints only the checks, exit 3. Needs hf-libero (the task clusters)
    python scripts/c1e4v_summary.py --selftest
"""

import argparse
import contextlib
import io
import json
import math
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
from c1e4m_summary import P, TOL, e2, maxwrap, orx, run  # noqa: E402
import vservo  # noqa: E402

OUT = cc.OUT
LAMS = tuple(vservo.LAMBDAS)  # spec §4: PBVS0.1, PBVS0.3, PBVS1.0
TIE_L = ("PBVS0.3", "PBVS0.1", "PBVS1.0")  # spec §4: ties within 0.1 mm (c1e4c_summary.TIE_M) go in this order
NGLR = ("PPCL@glr", "PPC@glr")  # controls journal "сводное решение владельца": the noisy cell, on the same frames
# the vgrid order (V0 and its band before the smoke); V3 runs as one line per d (V3d10, V3d20: owner 2026-10-21, the
# last line a single cuttable delay); --plan --line V3 gives both
LINES = ("vsmoke", "V0", "V1", "V2", "V3d10", "V3d20", "Nglr", "V2eps")
V3D = {"V3d10": 10, "V3d20": 20}
GATE, CAP = ("V1", "V2", "V3d10", "V3d20"), 4.0  # spec §8: V3 waits for the owner above $4 (the pod: V3_OK=1)
# spec §3: the scouting models (docs/c1/scouting.md §3: the integrator per axis; ARX one G, B for x and y)
SCOUT = {"I": {"G": [0.0105, 0.0121], "B": [0.0, 0.0]}, "A": {"G": [0.0025, 0.0025], "B": [0.79, 0.79]}}
ORDER = ("I", "A", "I*", "A*")  # spec §3 rule 3: the tie order, fewer parameters and no fit on our own data first
TIE_D, FLOOR, FLAG = 5e-4, 0.005, 0.015  # m: tie 0.5 mm; delta >= 5 mm; flagged above 15 mm (and V2eps runs)
EPS = orx.EPS  # PBVS journal: V2eps's delta, the 5 mm gate of G and G-auto (the competitor aligned to our own gate)
# controls journal (the latch's call rate): the forecast weighs a PPCL@glr record by the worst case, a call at t_obs + 4
# at the latest (PPCL_K_CAP) against the schedule's s = 10
WEIGHT = {"PPCL@glr": 2.5}
STATS = ("steps_window", "steps_acting", "sum_e", "max_e", "flips")  # vservo.PBVSAgent.pbvs_stats, a record's "pbvs"
CONF_SKIP = ("arms", "gpu", "kbar", "pbvs")  # spec §5: every other conf field equal across the read records of a d
G1 = "G(k=1.0)"  # controls journal: line 2's frozen VS1.0 records under their right name (report only)
FROZEN = f"C2_{P}_eval300_plus_d0.jsonl"  # line 2's file with them: its first VS_STOP Plus cases (controls journal)
# the pod's controls mode and k (controls journal: pi0.5 LIBERO-MAX mode С, k_π = 1.0): only line 2's and 3's cases,
# sources and groups are read off c1e4c_summary.plan here, which do not depend on them
CMODE, CK = {"pi05_lmax": {"mode": "S"}}, {P: {"k": "VS1.0"}}
# measurement -> (test, V lines it needs); spec §6, the journals' И5 and V2eps
MEAS = {"Н1": ("pi05, M1, d = 0, 300 pairs: PBVS - Gauto", ("V2",)),
        "Н2/d10": ("pi05, D, d = 10, 300 pairs: PBVS - Gauto", ("V3d10",)),
        "Н2/d20": ("pi05, D, d = 20, 300 pairs: PBVS - Gauto", ("V3d20",)),
        "Н2/pool": ("pi05, D, the d of pooled_d (10 and 20: 600 pairs; clusters: the task, its d in it): PBVS - Gauto",
                    ("V3d10", "V3d20")),
        "Н3/d0": ("pi05, M1, d = 0: PBVS - PPCL", ("V2",)),
        "Н3/d10": ("pi05, D, d = 10: PBVS - PPCL", ("V3d10",)),
        "Н3/d20": ("pi05, D, d = 20: PBVS - PPCL", ("V3d20",)),
        "И5": ("pi05, M1, d = 0, glr eyes, 300 pairs: PPCL@glr - PPC@glr", ("Nglr",)),
        "Н1/eps": ("pi05, M1, d = 0, the servo at our 5 mm gate (V2eps): PBVS - Gauto", ("V2eps",)),
        "PBVSeps-PBVS": ("pi05, M1, d = 0: PBVS at the 5 mm gate - PBVS at band.json's gate", ("V2", "V2eps"))}


def need(d, what, field):
    """pbvs_k.json's arm or band.json's delta, or exit 3: the line needs it and it is not there yet."""
    if (d or {}).get(field) is None:
        print(f"!!! no {what} yet", file=sys.stderr)
        raise SystemExit(3)
    return d[field]


def ids(ln, split):
    """A runner line's cases in the set's order (the runner's)."""
    have = {x[2] for x in ln["keys"]}
    return [c for c in run.set_ids(split, re.search(r"--set (\S+)", ln["args"])[1]) if c in have]


def groups(lns, split):
    """The runner's groups of a fresh run, its Base batches: per line (d) its cases in order, cc.NE at a time."""
    return {(int(re.search(r"--d (\d+)", ln["args"])[1]), tuple(c[i:i + cc.NE]))
            for ln in lns for c in [ids(ln, split)] for i in range(0, len(c), cc.NE)}


def plan(line, pk=None, band=None, split=None):
    """spec §3-§5 and the journal: the runner lines of one line (LINES), phase line.lower(), in c1e4c_summary.mlines's
    format plus "eps" (V2eps's records differ from V2's only in conf["pbvs"]["delta"]: vk tells them apart). V0: Base
    only on the 60 calibration cases, full episodes (no --base-prefix: the Base never moved), --log-traj; V1: the three
    PBVS arms there; vsmoke: PBVS0.3 on the first 6 Plus cases of eval300; V2, Nglr, V2eps (only when band.json's delta
    > 15 mm): M1, eval300, d 0; V3d10, V3d20: d 10, d 20 (V3: both). V2, Nglr, V2eps run line 2's cases, sources,
    order and so its groups and Base batches (line 2's first VS_STOP Plus cases, a multiple of cc.NE, then the rest of
    Plus and PRO); V3d10, V3d20 line 3's of their d.
    V2's first VS_STOP Plus cases are the frozen VS1.0 file's (asserted): the G(k = 1.0) pair."""
    split = split or maxwrap.load_split(cs.SPLIT)
    ph, bj = line.lower(), f" --band {OUT}/band.json"
    ext = lambda lns, s, eps=False: [ln | {"args": ln["args"] + s, "eps": eps} for ln in lns]  # noqa: E731
    if line == "V0":  # a Base-only record's cs.key ends True (no arms)
        return [ln | {"args": ln["args"] + " --log-traj", "eps": False, "keys": {x[:6] + (True,) for x in ln["keys"]}}
                for ln in cc.mlines(ph, "V0", P, "calib", (), split, prefix=False)]
    if line == "V1":
        return ext(cc.mlines(ph, "V1", P, "calib", LAMS, split), bj)
    if line == "vsmoke":
        ln = cc.mlines(ph, "Vsmoke", P, "eval300", ("PBVS0.3",), split,
                       cases=cs.sources(run.set_ids(split, "eval300"))["plus"][:6])[0]
        return ext([ln | {"id": "Vsmoke_pi05"}], bj)
    if line == "Nglr":
        return ext(cc.mlines(ph, "Nglr", P, "eval300", NGLR, split), "")
    arm = need(pk, "pbvs_k.json", "arm")
    assert arm in LAMS, arm
    if line in ("V3", *V3D):  # V3d10 / V3d20: one d each (tag V3: the same --out files), V3 both
        return ext([x for d in (10, 20) if line in ("V3", f"V3d{d}")
                    for x in cc.mlines(ph, "V3", P, "eval300", (arm,), split, d)], bj)
    assert line in ("V2", "V2eps"), line
    if line == "V2eps" and not need(band, "band.json", "delta") > FLAG:
        return []
    lns = ext(cc.mlines(ph, line, P, "eval300", (arm,), split),
              bj if line == "V2" else f" --band {OUT}/band_eps.json", line == "V2eps")
    c2 = {ln["id"]: ln for ln in cc.plan("2", CMODE, CK, split)}[f"C2_{P}_eval300_plus_d0"]
    v2 = ids(next(ln for ln in lns if ln["source"] == "plus"), split)
    assert v2[:len(ids(c2, split))] == ids(c2, split) and len(ids(c2, split)) == cc.VS_STOP["plus"], \
        "V2's first Plus cases are not the frozen VS1.0 file's"
    return lns


def lkeys(lns):
    """cc.lkeys plus each line's eps."""
    return {x + (ln["eps"],) for ln in lns for x in cc.lkeys([ln])}


def vk(r, band):
    """A V record's key in the plan: cc.lk and eps, its conf["pbvs"]["delta"] not band.json's (V2eps)."""
    return cc.lk(r) + (band is not None and "pbvs" in r["conf"] and r["conf"]["pbvs"]["delta"] != band["delta"],)


def unique(recs, key):
    """{key: record}; one key twice with other content: exit 2 (a grid_all file and its line files both)."""
    by = {}
    for r in recs:
        if by.setdefault(key(r), r) != r:
            print(f"!!! {key(r)} recorded twice with other content: pass grid_all_*.jsonl or its line files, not both",
                  file=sys.stderr)
            raise SystemExit(2)
    return by


# ------------------------------------------------------------------ the arm model and its dead band (spec §3)

def window(r):
    """A V0 Base record's xy x(t) and clipped executed action a(t) (the action at step t moves the arm between t and
    t + 1), its plans' arrival steps and the window's last step: the first close (t_close), else the last step."""
    x = np.asarray(r["traj"]["x"], float)[:, :2]
    a = np.clip(np.asarray(r["traj"]["a"], float)[:, :2], -1, 1)
    return x, a, sorted(ta for _, ta in r["plans_in"]), len(x) - 1 if r.get("t_close") is None else r["t_close"]


def errors(r, G, B):
    """spec §3 rule 1: |e(t)| on xy from the first plan arrival to the first close (both included: the regulator still
    acts at the close step). Plan N, the latest arrived (at t_a), executes its rows from t_a (in Base the executed
    actions are its clipped rows); its path w(t) = x(t_a) + sum_{j=t_a}^{t-1} D_j, D_j = G a_j + B D_{j-1}, D_{t_a-1} =
    x(t_a) - x(t_a-1) (0 at t_a = 0), as vservo.PBVSAgent; the object is still in Base: e = x(t) - w(t)."""
    x, a, arr, end = window(r)
    G, B, out = np.asarray(G, float), np.asarray(B, float), []
    for ta, tb in zip(arr, arr[1:] + [end + 1]):
        if ta > end:
            break
        w, dx = x[ta], x[ta] - x[ta - 1] if ta > 0 else np.zeros(2)
        out.append(0.0)
        for t in range(ta + 1, min(tb, end + 1)):
            dx = G * a[t - 1] + B * dx
            w = w + dx
            out.append(float(np.linalg.norm(x[t] - w)))
    return out


def fit(recs, arx):
    """spec §3, I* and A*: least squares per axis on the steps j from the first arrival to before the close, dx_j =
    x(j + 1) - x(j) = G a_j (+ B dx_{j-1}, dx_{-1} = 0)."""
    a, dx, prev = [], [], []
    for r in recs:
        x, aa, arr, end = window(r)
        for j in range(arr[0], min(end, len(x) - 1)):
            a.append(aa[j])
            dx.append(x[j + 1] - x[j])
            prev.append(x[j] - x[j - 1] if j > 0 else np.zeros(2))
    a, dx, prev = map(np.array, (a, dx, prev))
    if not arx:
        return {"G": ((a * dx).sum(0) / (a * a).sum(0)).tolist(), "B": [0.0, 0.0]}
    gb = [np.linalg.lstsq(np.column_stack([a[:, k], prev[:, k]]), dx[:, k], rcond=None)[0] for k in (0, 1)]
    return {"G": [float(g) for g, _ in gb], "B": [float(b) for _, b in gb]}


def order_stat(v):
    """spec §3 rule 2: the ceil(0.95 n)-th smallest, the least delta with at most 5 % of the episodes above it (the
    4th largest of 60)."""
    return sorted(v)[math.ceil(0.95 * len(v)) - 1]


def band(recs, split):
    """spec §3, the band of V0: per model the worst |e| of each episode (errors), delta_M = order_stat; I, A with the
    scouting coefficients, I*, A* cross-fit in 2 folds (the 60 cases halved in the set's order, each half's errors with
    the other half's fit; the final coefficients on all). The least delta_M wins, within TIE_D in ORDER; delta =
    max(delta_M, FLOOR), flagged above FLAG. Exit 2 when a calibration case has no Base record with a trajectory (V0
    incomplete or failed)."""
    cal = run.set_ids(split, "calib")
    by = {r["case_id"]: r for r in recs if r.get("kind") == "base" and r.get("brain") == P and r.get("traj")
          and not r.get("skipped_reason")}
    if short := [c for c in cal if c not in by]:
        print(f"!!! V0 incomplete: {len(short)} of {len(cal)} calibration cases without a Base record with traj, e.g. "
              f"{short[:3]}", file=sys.stderr)
        raise SystemExit(2)
    eps = [by[c] for c in cal]
    halves = (eps[:len(eps) // 2], eps[len(eps) // 2:])
    worst = lambda rs, c: [max(errors(r, c["G"], c["B"])) for r in rs]  # noqa: E731
    coef = {m: dict(c) for m, c in SCOUT.items()}
    dm = {m: order_stat(worst(eps, c)) for m, c in coef.items()}
    folds = {}
    for m, arx in (("I*", False), ("A*", True)):
        folds[m] = [fit(h, arx) for h in halves]  # fitted on the first half, on the second
        dm[m] = order_stat(worst(halves[0], folds[m][1]) + worst(halves[1], folds[m][0]))
        coef[m] = fit(eps, arx)
    m = next(m for m in ORDER if dm[m] <= min(dm.values()) + TIE_D)
    delta = max(dm[m], FLOOR)
    r9 = lambda c: {k: [round(float(v), 9) for v in c[k]] for k in ("G", "B")}  # noqa: E731
    return {"model": m, **r9(coef[m]), "delta": round(delta, 9), "delta_M": {k: round(v, 9) for k, v in dm.items()},
            "coef": {k: r9(c) for k, c in coef.items()}, "folds": {k: [r9(c) for c in v] for k, v in folds.items()},
            "n": len(eps), "flag_gt_15mm": delta > FLAG}


def select(recs, split):
    """spec §4: lambda from V1, c1e4c_summary.select's rule (controls spec §5: mean grasp miss over the cases valid for
    all three, the lowest wins, ties within 0.1 mm) with the PBVS candidates and their tie order. Exit 2: V1 incomplete
    or no case valid."""
    with mock.patch.object(cc, "VS", LAMS), mock.patch.object(cc, "TIE", TIE_L):  # ponytail: its rule, our labels
        s = cc.select(recs, P, split)
    arm = s.pop("k")
    return {"arm": arm, "lambda": vservo.LAMBDAS[arm]} | s


# ------------------------------------------------------------------ the forecast, the smoke, the controls lines

def forecast(tempo, pk=None, band=None, done=frozenset(), split=None):
    """spec §8 and the journal: per line the episodes, records (cut Bases too), hours at the pod's seconds per record
    (lmax_pi05: c1e4c_summary.forecast's convention), $ at cc.RATE and what is left (vk keys not in done). Without
    pbvs_k.json a placeholder arm (the records do not depend on lambda); without band.json V2eps counts (the costlier
    case). A PPCL@glr record weighs WEIGHT (the latch's worst-case call rate). The gate: GATE's cost above CAP -> V3,
    then Nglr and V2eps, wait."""
    split = split or maxwrap.load_split(cs.SPLIT)
    t, h = tempo["lmax_pi05"] / 3600, {}
    w = lambda ks: t * sum(WEIGHT.get(x[4], 1.0) for x in ks)  # noqa: E731  (x[4]: the arm)
    rows = {}
    for L in LINES:
        lns = plan(L, pk or {"arm": "PBVS0.3"}, band or {"delta": 1.0}, split)
        keys = lkeys(lns)
        h[L] = w(keys), w(keys - done)
        rows[L] = {"episodes": sum(map(cc.n_of, lns)), "records": len(keys)}
    hs = lambda Ls, j: sum(h[L][j] for L in Ls)  # noqa: E731  (j: 0 all, 1 left)
    out = lambda Ls: {"hours": round(hs(Ls, 0), 2), "usd": round(cc.RATE * hs(Ls, 0), 2),  # noqa: E731
                      "left_hours": round(hs(Ls, 1), 2), "left_usd": round(cc.RATE * hs(Ls, 1), 2)}
    gate = out(GATE)
    return {"usd_per_h": cc.RATE, "tempo_s": tempo["lmax_pi05"], "excludes": "the pod's tests and selftests",
            "weights": WEIGHT | {"why": "Nglr: PPCL@glr's worst case, a call at t_obs + 4 at the latest while latched "
                                        "(PPCL_K_CAP) against s = 10, about 2.5x PPC@glr's calls"},
            "using": {"arm": (pk or {}).get("arm"), "band_delta": (band or {}).get("delta")},
            "lines": {L: v | out((L,)) for L, v in rows.items()}, "total": out(LINES),
            "gate": {"lines": GATE, "cap_usd": CAP, **gate, "v3_waits": cc.RATE * hs(GATE, 0) > CAP}}


def smoke_check(recs, frozen):
    """The smoke's gate: PBVS0.3 with prefix_ok in >= 5 of the 6 cases, the 6 cases complete (a Base and a PBVS0.3
    record each, none skipped), every PBVS0.3 record with its counters, and the pairing's precondition: each smoke Base's
    e is the one of the same case's Base in the frozen file (FROZEN, line 2's first group: the same 6 cases in one
    batch), before the V lines pay; their act_hash reported. (pass, details)."""
    rs = [r for r in recs if "kind" in r]
    dyn = [r for r in rs if r["kind"] == "dynamic" and r["arm"] == "PBVS0.3"]
    fb = {r["case_id"]: r for r in frozen if r.get("kind") == "base" and r["d"] == 0}
    bs = [r for r in rs if r["kind"] == "base"]
    n = {"prefix_ok": len({r["case_id"] for r in dyn if r.get("prefix_ok") is True}),
         "cases": len({r["case_id"] for r in rs}), "pbvs": len(dyn), "base": len(bs),
         "skipped": sum(bool(r.get("skipped_reason")) for r in rs),
         "counters": all(set(STATS) <= set(r.get("pbvs") or {}) for r in dyn),
         "base_e_as_line_2": sum(r["e"] == fb.get(r["case_id"], {}).get("e", "-") for r in bs),
         "base_act_hash_as_line_2": sum(r.get("act_hash") == fb.get(r["case_id"], {}).get("act_hash", "-") for r in bs)}
    return (n["prefix_ok"] >= 5 and n["cases"] == n["pbvs"] == n["base"] == n["base_e_as_line_2"] == 6
            and not n["skipped"] and n["counters"]), n


def frozen_check(frozen, split):
    """The G(k = 1.0) pair's precondition on disk: the frozen file's Base case ids in its order (a case's lines are
    written together, Base last) are V2's planned first VS_STOP Plus cases, in order."""
    got = [r["case_id"] for r in frozen if r.get("kind") == "base"]
    want = ids(plan("V2", {"arm": LAMS[0]}, split=split)[0], split)[:cc.VS_STOP["plus"]]
    return {"bases": len(got), "first_equal": sum(a == b for a, b in zip(got, want)), "want": len(want),
            "pass": got[:len(want)] == want}


def controls(recs, mode, k, split):
    """The controls lines 2 and 3 (c1e4c_summary.plan from mode.json and k_select.json) the V lines pair with: their
    records on the plan (cc.lk, skipped ones missing), per line its completeness (c1e4c_summary.grid's <= 1 % missing,
    run). (used {lk: record}, {lk: line}, completeness)."""
    want = {x: L for L in "23" for x in cc.lkeys(cc.plan(L, mode, k, split))}
    used = {x: r for x, r in unique(recs, cc.lk).items() if x in want and not r.get("skipped_reason")}
    comp = {}
    for L in "23":
        exp = [x for x, v in want.items() if v == L]
        miss = sum(x not in used for x in exp)
        comp[L] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    return used, want, comp


# ------------------------------------------------------------------ the reading (spec §5, §6)

def read(new, ctrl, saved, mode, k, band, pk, names, split):
    """spec §5 checks before reading, then the §6 measurements (estimate and 95 % CI, no threshold) and the report.
    new: grid_all_V.jsonl (V0-V3, Nglr, V2eps); ctrl: controls lines 2 and 3 (Gauto, PPCL; line 2's frozen VS1.0 =
    G(k = 1.0), read for its one report line only); saved: part 2's π LIBERO-MAX records ("none", "через под", and
    its G, PPC with glr for the retention). Records off the plan are counted, never read."""
    need(pk, "pbvs_k.json", "arm")
    need(band, "band.json", "delta")
    arm, e300 = pk["arm"], set(run.set_ids(split, "eval300"))
    by = unique(new, lambda r: vk(r, band))
    lines = {L: plan(L, pk, band, split) for L in LINES[1:]}  # the smoke is never read
    want = {x: L for L, lns in lines.items() for x in lkeys(lns)}
    used = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    of = lambda *Ls: [r for x, r in used.items() if want[x] in Ls]  # noqa: E731
    comp = {}
    for L, lns in lines.items():  # spec §5: <= 1 % missing per line; a line not run (waits for the owner) passes
        exp = lkeys(lns)
        miss = len(exp - used.keys())
        comp[L] = {"expected": len(exp), "missing": miss, "run": miss < len(exp),
                   "pass": miss == len(exp) or miss <= 0.01 * len(exp) + TOL}
    cused, cwant, ccomp = controls(ctrl, mode, k, split)
    rec_of = lambda L: [r for x, r in by.items() if want.get(x) == L]  # noqa: E731  (the pod's --band / --select read)

    def again(f, L, file):  # band.json and pbvs_k.json are their rules on V0, V1 (spec §3 rule 6, §4)
        try:
            got = json.loads(json.dumps(f(rec_of(L), split)))
        except SystemExit:
            got = None
        return {"file": file, "recomputed": got, "pass": got == file}
    with contextlib.redirect_stderr(io.StringIO()):
        recomputed = {"band": again(band_of, "V0", band), "pbvs_k": again(select, "V1", pk)}
    epsb = band | {"delta": EPS}  # band_eps.json
    off_band = Counter(want[x] for x, r in used.items() if r["conf"].get("pbvs") != (
        epsb if want[x] == "V2eps" else band if want[x] in ("V1", "V2", *V3D) else None))
    cbase = {(r["case_id"], r["d"]): r for x, r in cused.items() if r["kind"] == "base"}
    em = defaultdict(lambda: [0, 0])
    for r in of("V2", *V3D, "Nglr", "V2eps"):  # spec §5: the new Base's e is line 2's / 3's, >= 98 % per cell
        if r["kind"] == "base":
            c = f"{want[vk(r, band)]}/d{r['d']}"
            em[c][0] += cbase.get((r["case_id"], r["d"]), {}).get("e", "-") == r["e"]
            em[c][1] += 1
    e_match = {c: {"equal": a, "total": n, "pass": a >= 0.98 * n - TOL} for c, (a, n) in sorted(em.items())}
    p2 = [r for r in saved if r["case_id"] in e300 and r["kind"] == "dynamic"
          and (r["arm"] == "none" or r["d"] == 0 and r["arm"] in ("G", "PPC", "G@glr", "PPC@glr"))]
    conf = defaultdict(set)
    for r in list(used.values()) + list(cused.values()) + p2:  # spec §5: conf equal except CONF_SKIP
        conf[f"d{r['d']}"].add(json.dumps({f: v for f, v in r["conf"].items() if f not in CONF_SKIP}, sort_keys=True))
    mixed = {g: sorted(v) for g, v in conf.items() if len(v) > 1}
    checks = {"records": len(new), "read": len(used), "skipped": sum(bool(r.get("skipped_reason")) for r in by.values()),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for x, r in by.items() if x not in want)),
              "complete": comp, "controls_complete": ccomp, **recomputed,
              "band_in_conf": {"off": dict(off_band), "pass": not off_band}, "e_match": e_match,
              "gauto_on_pod": {"mode": ((mode or {}).get("pi05_lmax") or {}).get("mode"),
                               "pass": ((mode or {}).get("pi05_lmax") or {}).get("mode") == "S"},
              "conf": {"except": CONF_SKIP, "mixed": mixed, "pass": not mixed}}
    checks["pass"] = bool(all(v["pass"] for d in (comp, ccomp, e_match) for v in d.values()) and all(
        checks[f]["pass"] for f in ("band", "pbvs_k", "band_in_conf", "gauto_on_pod", "conf")))
    if not checks["pass"]:
        return {"checks": checks}

    # the index: V2eps's arm as arm/eps; part 2's "none" beside the pod's arms, its other arms apart (part 2's PPC@glr
    # is not the pod's); line 2's VS1.0 = G(k = 1.0) only in its own report line (pairs), never in the index
    lab = lambda r: r | {"arm": f"{r['arm']}/eps"} if want[vk(r, band)] == "V2eps" else r  # noqa: E731
    pod = ([lab(r) for r in of("V2", *V3D, "Nglr", "V2eps") if r["kind"] == "dynamic"]
           + [r for r in cused.values() if r["kind"] == "dynamic" and r["arm"] != "VS1.0"])
    mi, m2 = cs.mindex(pod + [r for r in p2 if r["arm"] == "none"]), cs.mindex(p2)
    one = {r["case_id"]: r for r in saved + list(cused.values()) if r["case_id"] in e300}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    M = lambda a, b, d=0, cases=e300, idx=mi: cc.mp(idx, clu, P, a, b, d, cases)  # noqa: E731
    ae = f"{arm}/eps"
    prs = {"Н1": [M(arm, "Gauto")], "Н2/d10": [M(arm, "Gauto", 10)], "Н2/d20": [M(arm, "Gauto", 20)],
           "Н2/pool": [M(arm, "Gauto", 10), M(arm, "Gauto", 20)], "Н3/d0": [M(arm, "PPCL")],
           "Н3/d10": [M(arm, "PPCL", 10)], "Н3/d20": [M(arm, "PPCL", 20)], "И5": [M(*NGLR)],
           "Н1/eps": [M(ae, "Gauto")], "PBVSeps-PBVS": [M(ae, arm)]}
    meas, inv = {}, {}
    for n, (test, Ls) in MEAS.items():
        if n in ("Н1/eps", "PBVSeps-PBVS") and not lines["V2eps"]:
            continue  # the band is within 15 mm: no V2eps (journal)
        ps, bad = [v for p, _ in prs[n] for v in p.values()], sum(b for _, b in prs[n])
        meas[n] = {"test": test, **(e2.rnd(e2.diff_pp(ps)) or dict.fromkeys(cc.EST)), "invalid_pairs": bad}
        miss = [L for L in Ls if not comp[L]["run"]]
        if n == "Н2/pool":  # the d that ran, named; a d not run is listed, never pooled in silence
            meas[n]["pooled_d"] = [V3D[L] for L in Ls if L not in miss]
        if miss:
            meas[n] |= {"lines_not_run": miss} | (dict.fromkeys(cc.EST) if n != "Н2/pool" or miss == list(Ls) else {})
        inv[n] = bad, len(ps) + bad
    checks["invalid_pairs"] = {n: {"pairs": t, "invalid": b, "pass": b <= 0.02 * t + TOL} for n, (b, t) in inv.items()}
    if not all(v["pass"] for v in checks["invalid_pairs"].values()):
        checks["pass"] = False
        return {"checks": checks}
    d0 = lambda rs, a=None: {r["case_id"]: r for r in rs if r["d"] == 0 and (  # noqa: E731
        r["kind"] == "base" if a is None else r["kind"] == "dynamic" and r["arm"] == a)}
    ep = {"V2": (d0(of("V2"), arm), d0(of("V2"))),
          "frozen": (d0(cused.values(), "VS1.0"), d0(cused.values())),
          "Nglr": ({a: d0(of("Nglr"), a) for a in NGLR}, d0(of("Nglr"))),
          "p2": ({a: d0(p2, a) for a in ("none", "G@glr")}, d0([r for r in saved if r["case_id"] in e300]))}
    gpu = sorted({str(r["conf"].get("gpu")) for r in list(used.values()) + list(cused.values())})
    return {"checks": checks, "measurements": meas,
            "report": report(mi, m2, M, clu, one, e300, arm, ae if lines["V2eps"] else None, mode, band, pk, ep, gpu,
                             checks)}


def pairs(x, y, xb, yb, clu, cases):
    """The journals' one-to-one pairs across runs (G(k = 1.0), G@glr through the pod): per case one Dynamic episode of
    each side, x[c] and y[c], matched on case_id and policy_seed (one seed per case, shared by its arms and its Base: a
    mismatch drops the pair); each valid against its own Base (prefix_ok, no replay_miss); the two Bases xb[c], yb[c]
    with the same e, the same pre-event prefix. The Bases enter only these checks, never the difference. ([(case,
    cluster, success x, success y)], drops by reason)."""
    out, drop = [], Counter()
    for c in cases:
        q = (x.get(c), y.get(c), xb.get(c), yb.get(c))
        if None in q:
            drop["missing"] += 1
        elif len({r["policy_seed"] for r in q}) > 1:
            drop["seed"] += 1
        elif any(r.get("prefix_ok") is not True or r.get("replay_miss") is not None for r in q[:2]):
            drop["invalid"] += 1
        elif q[2]["e"] != q[3]["e"]:
            drop["e"] += 1
        else:
            out.append((c, clu[c], q[0]["success"], q[1]["success"]))
    return out, dict(drop)


def est(prs):
    """pairs' estimate: the mean of success x - success y in {-1, 0, 1} x 100 over the pairs (never a per-case mean),
    the cluster bootstrap by source task (e2.diff_pp: 2000 reps, percentile 95 %)."""
    return e2.rnd(e2.diff_pp([p[1:] for p in prs])) or dict.fromkeys(cc.EST)


def report(mi, m2, M, clu, one, e300, arm, ae, mode, band, pk, ep, gpu, checks):
    """spec §6 "только отчёт" and the journals: PBVS - none per cell ("через под" in mode С); the two PBVS lines side
    by side with their gates (band.json's delta; V2eps at our 5 mm); the 6 / 12 cm slices on M1; PBVS - G(k = 1.0) on
    line 2's frozen Plus cases (pairs); brain calls, the PBVS counters, the latch shares; grasp miss PBVS against
    Gauto; the noisy cell's one table (success, calls per episode = calls_sched + calls_trig, latched share; G@glr and
    "none" of part 2 through the pod) with PPCL@glr - G@glr and its calls difference under it, and the retention
    (PPCL@glr - none against PPCL - none; part 2's G and PPC beside it); band.json and pbvs_k.json."""
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    med = lambda v: round(float(np.median(v)), 4) if len(v) else None  # noqa: E731
    pct = lambda rs: mean([100 * r["success"] for r in rs if r["success"] is not None])  # noqa: E731  (a cut Base: none)
    calls = lambda r: r["calls_sched"] + r["calls_trig"]  # noqa: E731
    dm = lambda a, b, d=0, cases=e300, idx=mi: e2.rnd(e2.diff_pp(list(M(a, b, d, cases, idx)[0].values())))  # noqa
    via = {"через под": ((mode or {}).get("pi05_lmax") or {}).get("mode") == "S"}
    rep = defaultdict(dict)
    for d in (0, 10, 20):
        rep["pbvs_none"][f"{arm}/d{d}"] = (dm(arm, "none", d) or {}) | via
    for L, a, gate in (("V2", arm, band["delta"]), ("V2eps", ae, EPS)):  # PBVS journal: side by side, with the gate
        if a:
            g = list(mi.get((P, "dynamic", a, 0), {}).values())
            rep["pbvs_gates"][L] = {"arm": arm, "gate_m": gate, "n": len(g), "success_pct": pct(g),
                                    "pbvs-Gauto": dm(a, "Gauto"), "pbvs-none": (dm(a, "none") or {}) | via}
    dist = {c: one[c]["distance_m"] for c in e300 if c in one and one[c]["distance_m"] is not None}
    near, far = {c for c, v in dist.items() if v < 0.09}, {c for c, v in dist.items() if v >= 0.09}
    for x, y in ((arm, "Gauto"), (arm, "PPCL"), (arm, "none")):
        rep["slices_M1"][f"{x}-{y}"] = {"6cm": dm(x, y, 0, near), "12cm": dm(x, y, 0, far)}
    (vd, vb), (fd, fb) = ep["V2"], ep["frozen"]
    prs, drop = pairs(vd, fd, vb, fb, clu, sorted(fd))
    rep["pbvs_g1"] = {"test": f"pi05, M1, d = 0, line 2's frozen Plus cases, one pair per case and seed: {arm} - {G1}",
                      "cases": len(fd), "drops": drop, **est(prs)}
    for key, g in sorted(mi.items(), key=lambda kv: str(kv[0])):  # per arm and cell
        g, name = list(g.values()), "/".join(map(str, key))
        post = [sum(t >= r["e"] for t, _ in r["plans_in"]) for r in g if r.get("e") is not None]
        rep["calls"][name] = {"n": len(g), "sched": mean([r["calls_sched"] for r in g]),
                              "trig": mean([r["calls_trig"] for r in g]), "plans_after_event": mean(post)}
        st = [r["pbvs"] for r in g if r.get("pbvs") and r["pbvs"]["steps_window"]]
        if key[2] in (arm, ae):  # spec §6: the share of acting steps after t_e, mean |e| and its spread, flips
            me = [s["sum_e"] / s["steps_window"] for s in st]
            rep["pbvs_counters"][name] = {
                "n": len(g), "window": len(st), "acting_share": mean([s["steps_acting"] / s["steps_window"] for s in st]),
                "mean_e_m": mean(me), "mean_e_p5_p95": [round(float(v), 4) for v in np.percentile(me, [5, 95])]
                if me else None, "max_e_m": mean([s["max_e"] for s in st]),
                "flips_per_episode": mean([s["flips"] for s in st]), "flipped_share": mean([s["flips"] > 0 for s in st])}
        if key[2].startswith("PPCL"):  # controls journal: the latch shares, held a chunk / triggered once
            lt = [r["latch"] for r in g]
            rep["latch"][name] = {"n": len(lt), "chunks_share": mean([x["chunks"] > 0 for x in lt]),
                                  "trig_share": mean([x["trig"] > 0 for x in lt])}
    for d in (0, 10, 20):  # spec §6: grasp miss PBVS against Gauto, on the cases with both
        x, y = mi.get((P, "dynamic", arm, d), {}), mi.get((P, "dynamic", "Gauto", d), {})
        both = [c for c in sorted(x.keys() & y.keys()) if None not in (x[c]["grasp_miss_m"], y[c]["grasp_miss_m"])]
        rep["grasp_miss"][f"d{d}"] = {"n": len(both), arm: mean([x[c]["grasp_miss_m"] for c in both]),
                                      "Gauto": mean([y[c]["grasp_miss_m"] for c in both]),
                                      f"{arm}-Gauto": mean([x[c]["grasp_miss_m"] - y[c]["grasp_miss_m"] for c in both])}
    (nd, nb), (pd, pb) = ep["Nglr"], ep["p2"]  # controls journal: the noisy cell's one table, calls in it
    row = lambda rs, latch=False: {  # noqa: E731
        "n": len(rs), "success_pct": pct(rs), "calls_mean": mean([calls(r) for r in rs]),
        "calls_median": med([calls(r) for r in rs])} | (
        {"latched_share": mean([r["latch"]["chunks"] > 0 for r in rs])} if latch else {})
    tag = " (part 2, через под)" if via["через под"] else " (part 2)"
    rep["nglr"]["table"] = {"PPCL@glr": row(list(nd["PPCL@glr"].values()), True),
                            "PPC@glr (this pod)": row(list(nd["PPC@glr"].values())),
                            f"G@glr{tag}": row(list(pd["G@glr"].values())), f"none{tag}": row(list(pd["none"].values())),
                            "Base (this pod, cut at the event)": {
                                k: v for k, v in row(list(nb.values())).items() if k != "success_pct"}}
    prs, drop = pairs(nd["PPCL@glr"], pd["G@glr"], nb, pb, clu, sorted(e300))
    rep["nglr"]["PPCL@glr-G@glr"] = {"test": "pi05, M1, d = 0, glr eyes, one pair per case and seed (report only): "
                                             "PPCL@glr - G@glr of part 2", "drops": drop, **est(prs)} | via
    dc = [calls(nd["PPCL@glr"][c]) - calls(pd["G@glr"][c]) for c, *_ in prs]
    rep["nglr"]["calls_PPCL@glr-G@glr"] = {"pairs": len(dc), "mean": mean(dc), "median": med(dc)} | via
    ret = lambda n, c: (round(n["diff_pp"] / c["diff_pp"], 3)  # noqa: E731
                        if n and c and n.get("diff_pp") is not None and c.get("diff_pp") else None)
    a, b = dm("PPCL@glr", "none"), dm("PPCL", "none")
    rep["retention"]["pod"] = {"PPCL@glr-none": (a or {}) | via, "PPCL-none": (b or {}) | via, "ratio": ret(a, b)}
    for x in ("G", "PPC"):  # part 2's own records, one run (not through the pod)
        a, b = dm(f"{x}@glr", "none", idx=m2), dm(x, "none", idx=m2)
        rep["retention"][f"part2/{x}"] = {f"{x}@glr-none": a, f"{x}-none": b, "ratio": ret(a, b)}
    return dict(rep) | {"gpu": gpu, "band": band, "pbvs_k": pk, "e_match": checks["e_match"],
                        "invalid_pairs": checks["invalid_pairs"]}


band_of = band  # read()'s name for the rule (its argument band is band.json)


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 PBVS checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


def write_band(b, out):
    """band.json, and band_eps.json beside it when delta > 15 mm (journal: delta 5 mm, the rest band.json's)."""
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(b, indent=1) + "\n")
    if b["delta"] > FLAG:
        Path(out).with_name("band_eps.json").write_text(json.dumps(b | {"delta": EPS}, indent=1) + "\n")


# ------------------------------------------------------------------ selftest (spec §9 item 9)

def toy(n=60, G=(0.009, 0.013), B=(0.0, 0.0), seed=0, close=45, bumps=(), s=10):
    """A V0 Base record of a synthetic arm dx_j = G a_j + B dx_{j-1} (dx_{-1} = 0), a row per step beyond +-1 now and
    then (the runner records the clipped one), plans every s steps (d 0), the close at close (None: no close); bumps:
    (step, metres on x) added to x(step) alone; traj rounded to 1e-6 as the runner's."""
    rng = np.random.default_rng(seed)
    a, x, dx = np.clip(rng.uniform(-1.3, 1.3, (n, 2)), -1, 1), np.zeros((n, 2)), np.zeros(2)
    for j in range(n - 1):
        dx = np.asarray(G) * a[j] + np.asarray(B) * dx
        x[j + 1] = x[j] + dx
    for t, b in bumps:
        x[t, 0] += b
    r3 = lambda v: np.round(np.column_stack([v, np.zeros(n)]), 6).tolist()  # noqa: E731
    return {"kind": "base", "brain": P, "arm": "none", "skipped_reason": None, "t_close": close,
            "plans_in": [[t, t] for t in range(0, n, s)], "traj": {"x": r3(x), "p": r3(np.ones((n, 2))), "a": r3(a)}}


def selftest():
    split = maxwrap.load_split(cs.SPLIT)
    cal, e3 = run.set_ids(split, "calib"), cs.sources(run.set_ids(split, "eval300"))
    v0 = lambda f: [f(j) | {"case_id": c} for j, c in enumerate(cal)]  # noqa: E731

    # the band (spec §3). An exact integrator off the scouting G: I* recovers it in both folds and on all, delta_M ~ 0
    # (the traj's 1e-6 rounding); A* (B ~ 0) ties with it within 0.5 mm and I* goes first; I is worse; the floor
    b = band(v0(lambda j: toy(seed=j)), split)
    assert b["model"] == "I*" and np.allclose(b["G"], [0.009, 0.013], atol=1e-6) and b["B"] == [0.0, 0.0], b
    assert all(np.allclose(f["G"], [0.009, 0.013], atol=1e-6) for f in b["folds"]["I*"]) and b["n"] == 60, b
    assert b["delta_M"]["I*"] < 1e-5 and b["delta_M"]["A*"] < 1e-5 and b["delta_M"]["I"] > TIE_D + 1e-5, b
    assert b["delta"] == FLOOR and not b["flag_gt_15mm"] and b["coef"]["I"] == SCOUT["I"], b
    # an exact ARX arm (its own first step): A* recovers it, the integrators miss by > 0.5 mm; the path's initial step
    # is the arm's actual one (with 0 there the plans' first steps would err)
    b = band(v0(lambda j: toy(seed=j, G=(0.003, 0.002), B=(0.6, 0.7))), split)
    assert b["model"] == "A*" and np.allclose(b["G"], [0.003, 0.002], atol=1e-5), b
    assert np.allclose(b["B"], [0.6, 0.7], atol=1e-3) and b["delta_M"]["A*"] < 1e-5 < b["delta_M"]["I*"] - TIE_D, b
    # the order statistic, the window and the flag on the scouting integrator, exact but for bumps at step 25 (not an
    # anchor: e(25) = the bump): 30, 25, 20, 10, 8 mm in 5 episodes -> delta_I = 10 mm (the 4th largest), I first in
    # the tie; bumps after the close (50 mm, every episode) are outside the window
    bm = {0: 0.030, 7: 0.025, 20: 0.020, 33: 0.010, 41: 0.008}
    b = band(v0(lambda j: toy(seed=j, G=SCOUT["I"]["G"], bumps=[(25, bm.get(j, 0.0)), (50, 0.05)])), split)
    assert b["model"] == "I" and abs(b["delta_M"]["I"] - 0.010) < 2e-6 and abs(b["delta"] - 0.010) < 2e-6, b
    assert not b["flag_gt_15mm"] and all(abs(b["delta_M"][m] - 0.010) < TIE_D for m in ("I*", "A*")), b
    assert b["delta_M"]["A"] > 0.03, b  # the scouting ARX is far off an integrator arm
    b = band(v0(lambda j: toy(seed=j, G=SCOUT["I"]["G"], close=None, bumps=[(50, 0.05 * (j < 4))])), split)
    assert abs(b["delta"] - 0.05) < 2e-3 and b["flag_gt_15mm"], b  # no close: the window runs to the end
    # the cross-fit: the halves are two arms (G1, G2); each fold's fit is its half's, delta_M of I* is out of fold
    # (in fold it would be ~ 0), the final G between the two
    g1, g2 = (0.009, 0.013), (0.012, 0.010)
    recs = v0(lambda j: toy(seed=j, G=g1 if j < 30 else g2))
    b = band(recs, split)
    f = b["folds"]["I*"]
    assert np.allclose(f[0]["G"], g1, atol=1e-6) and np.allclose(f[1]["G"], g2, atol=1e-6), f
    oof = [max(errors(r, g2, (0, 0))) for r in recs[:30]] + [max(errors(r, g1, (0, 0))) for r in recs[30:]]
    assert abs(b["delta_M"]["I*"] - order_stat(oof)) < 1e-6 and b["delta_M"]["I*"] > 1e-3, b
    assert all(min(u, v) < w < max(u, v) for u, v, w in zip(g1, g2, b["coef"]["I*"]["G"])), b
    # incomplete V0: a case missing, skipped, or without a trajectory
    full = v0(lambda j: toy(seed=j))
    for bad in (full[1:], [r | {"skipped_reason": "x"} if r is full[3] else r for r in full],
                [r | {"traj": None} if r is full[3] else r for r in full]):
        cc.exits(lambda: band(bad, split), 2)
    with tempfile.TemporaryDirectory() as tmp:  # band_eps.json only above 15 mm: delta 5 mm, the rest band.json's
        write_band(b | {"delta": 0.02}, Path(tmp) / "band.json")
        z = json.loads((Path(tmp) / "band_eps.json").read_text())
        assert z == json.loads(json.dumps(b | {"delta": EPS})), z
        write_band(b, Path(tmp) / "b" / "band.json")
        assert not (Path(tmp) / "b" / "band_eps.json").exists()
    selftest_agent()

    # the selection (spec §4): c1e4c_summary.select's rule with PBVS labels, ties in the order 0.3, 0.1, 1.0
    def v1(miss, edit=lambda j, v: {}):
        mk = lambda c, kd, a, **kw: {"brain": P, "case_id": c, "kind": kd, "arm": a, "e": 30, "prefix_ok": True,  # noqa
                                     "replay_miss": None, "skipped_reason": None} | kw
        return [mk(c, "dynamic", v, grasp_miss_m=miss[v] + 0.001 * (j % 3)) | edit(j, v) for j, c in enumerate(cal)
                for v in LAMS] + [mk(c, "base", "none", prefix_ok=None) for c in cal]
    s = select(v1({"PBVS0.1": 0.010, "PBVS0.3": 0.012, "PBVS1.0": 0.020}), split)
    assert s == {"arm": "PBVS0.1", "lambda": 0.1, "scores": {"PBVS0.1": 0.011, "PBVS0.3": 0.013, "PBVS1.0": 0.021},
                 "n_valid": 60, "n_cases": 60, "flag_lt40": False}, s
    for miss, arm in (({"PBVS0.1": 0.0100, "PBVS0.3": 0.01005, "PBVS1.0": 0.0103}, "PBVS0.3"),
                      ({"PBVS0.1": 0.01005, "PBVS0.3": 0.0102, "PBVS1.0": 0.0100}, "PBVS0.1"),
                      ({"PBVS0.1": 0.0102, "PBVS0.3": 0.0102, "PBVS1.0": 0.0100}, "PBVS1.0")):
        assert select(v1(miss), split)["arm"] == arm, (miss, arm)
    s = select(v1({"PBVS0.1": 0.010, "PBVS0.3": 0.012, "PBVS1.0": 0.020},
                  lambda j, v: {"prefix_ok": False} if j < 21 and v == "PBVS1.0" else {}), split)
    assert s["n_valid"] == 39 and s["flag_lt40"] and s["arm"] == "PBVS0.1", s
    cc.exits(lambda: select([r for r in v1({v: 0.01 for v in LAMS}) if r["kind"] != "base"], split), 2)
    assert cc.VS == ("VS0.1", "VS0.3", "VS1.0") and cc.TIE[0] == "VS0.3"  # the controls' own labels untouched

    # the plan
    pk, bd = {"arm": "PBVS0.3"}, {"delta": 0.02}
    l0 = plan("V0", split=split)
    assert [(ln["id"], ln["phase"], cc.n_of(ln)) for ln in l0] == [("V0_pi05_calib_plus_d0", "v0", 42),
                                                                   ("V0_pi05_calib_pro_d0", "v0", 18)], l0
    assert l0[0]["args"] == "--brain pi05 --source plus --set calib --arms= --d 0 --log-traj", l0[0]["args"]
    assert all(x[3] == "base" and x[6] is True for ln in l0 for x in ln["keys"])  # cs.key of a Base-only record
    l1 = plan("V1", split=split)
    assert [(ln["id"], cc.n_of(ln)) for ln in l1] == [("V1_pi05_calib_plus_d0", 126), ("V1_pi05_calib_pro_d0", 54)]
    assert l1[0]["args"] == ("--brain pi05 --source plus --set calib --arms=PBVS0.1,PBVS0.3,PBVS1.0 --d 0 "
                             f"--base-prefix --band {OUT}/band.json"), l1[0]["args"]
    sm = plan("vsmoke", split=split)
    assert [(ln["id"], ln["phase"], cc.n_of(ln)) for ln in sm] == [("Vsmoke_pi05", "vsmoke", 6)] and sm[0]["args"] == (
        f"--brain pi05 --source plus --set eval300 --arms=PBVS0.3 --d 0 --cases {','.join(e3['plus'][:6])} "
        f"--base-prefix --band {OUT}/band.json"), sm
    l2, l3, ng = plan("V2", pk, split=split), plan("V3", pk, split=split), plan("Nglr", split=split)
    assert [(ln["id"], cc.n_of(ln)) for ln in l2] == [("V2_pi05_eval300_plus_d0", 187), ("V2_pi05_eval300_pro_d0", 113)]
    assert l2[1]["args"] == ("--brain pi05 --source pro --set eval300 --arms=PBVS0.3 --d 0 --base-prefix "
                             f"--band {OUT}/band.json"), l2[1]["args"]
    assert [ln["id"] for ln in l3] == [f"V3_pi05_eval300_{s}_d{d}" for d in (10, 20) for s in ("plus", "pro")]
    assert [ln["args"] for ln in l3] == [f"--brain pi05 --source {s} --set eval300 --arms=PBVS0.3 --d {d} --base-prefix "
                                         f"--band {OUT}/band.json" for d in (10, 20) for s in ("plus", "pro")], l3
    assert [(ln["id"], cc.n_of(ln)) for ln in ng] == [("Nglr_pi05_eval300_plus_d0", 374),
                                                      ("Nglr_pi05_eval300_pro_d0", 226)], ng
    assert ng[0]["args"] == "--brain pi05 --source plus --set eval300 --arms=PPCL@glr,PPC@glr --d 0 --base-prefix"
    # line 2's Base batches (its first 108 Plus cases with VS, then the rest, then PRO) are V2's, Nglr's and V2eps's;
    # line 3's are V3's
    c2, c3 = cc.plan("2", CMODE, CK, split), cc.plan("3", CMODE, CK, split)
    assert [len(ids(ln, split)) for ln in c2] == [cc.VS_STOP["plus"], 187 - cc.VS_STOP["plus"], 113], c2
    assert groups(l2, split) == groups(c2, split) == groups(ng, split) and len(groups(c2, split)) == 32 + 19
    assert groups(l3, split) == groups(c3, split) and len(groups(l3, split)) == 2 * 51
    for L, d in V3D.items():  # one d each (owner 2026-10-21): line 3's cases, order and Base groups of that d
        ld = plan(L, pk, split=split)
        assert [(ln["id"], ln["args"], ln["keys"]) for ln in ld] == [
            (ln["id"], ln["args"], ln["keys"]) for ln in l3 if f"--d {d} " in ln["args"]] and len(ld) == 2, ld
        assert {ln["phase"] for ln in ld} == {L.lower()} and groups(ld, split) == {
            g for g in groups(c3, split) if g[0] == d} and len(groups(ld, split)) == 51
    le = plan("V2eps", pk, bd, split)
    assert [ln["id"] for ln in le] == ["V2eps_pi05_eval300_plus_d0", "V2eps_pi05_eval300_pro_d0"] and all(
        ln["eps"] and ln["args"].endswith(f"--band {OUT}/band_eps.json") for ln in le) and groups(le, split) == groups(
        l2, split), le
    assert plan("V2eps", pk, {"delta": FLAG}, split) == [] and not lkeys(l2) & lkeys(le)
    assert ids(l2[0], split)[:108] == e3["plus"][:108] == ids(c2[0], split)  # the frozen VS1.0 file's cases
    with mock.patch.object(cc, "plan", lambda *a: [c2[0] | {"keys": {x for x in c2[0]["keys"]
                                                                       if x[2] != e3["plus"][5]}}] + c2[1:]):
        try:  # V2 not on the frozen file's cases (here: one missing): the plan refuses
            plan("V2", pk, split=split)
            raise RuntimeError("V2 off the frozen file's cases was planned")
        except AssertionError:
            pass
    for L, a in (("V2", (None, bd)), ("V3", (None, None)), ("V3d10", (None, bd)),
                 ("V3d20", (None, bd)), ("V2eps", (None, bd)), ("V2eps", (pk, None))):
        cc.exits(lambda: plan(L, *a, split), 3)
    every = [ln for L in LINES for ln in plan(L, pk, bd, split)]
    assert len({ln["id"] for ln in every}) == len(every) and all(
        ln["runner"] == "c1e4m" and f"--source {ln['source']} " in ln["args"] and ln["phase"] in (
            "vsmoke", "v0", "v1", "v2", "v3d10", "v3d20", "nglr", "v2eps") for ln in every)
    assert all(("--band" in ln["args"]) == any(a in LAMS for a in ln["arms"]) and "--kbar" not in ln["args"]
               for ln in every)

    with tempfile.TemporaryDirectory() as tmp:  # the forecast reads c1e4c_summary's tempo lines
        tf = Path(tmp) / "tempo.txt"
        tf.write_text("tempo (s/episode): lmax_pi05=36 lmax_oftplus=1 lmax_oft=1 bench_pi05=1\n")
        t36 = cc.read_tempo(tf)
    # 36 s per record: hours = records / 100. V1: 3 arms + Base on 60; V2, V3d10, V3d20: PBVS + Base on 300; Nglr:
    # its PPCL@glr records weigh 2.5 (300 x 2.5 + 600 = 1350 -> 13.5 h); V2eps: none (the band within 15 mm)
    f = forecast(t36, pk, {"delta": 0.01}, split=split)
    assert [f["lines"][L]["records"] for L in LINES] == [12, 60, 240, 600, 600, 600, 900, 0], f["lines"]
    assert [f["lines"][L]["hours"] for L in ("V3d10", "V3d20", "Nglr", "V2eps")] == [6.0, 6.0, 13.5, 0], f["lines"]
    assert f["gate"] == {"lines": GATE, "cap_usd": CAP, "hours": 20.4, "usd": round(20.4 * cc.RATE, 2),
                         "left_hours": 20.4, "left_usd": f["gate"]["usd"], "v3_waits": True}, f["gate"]
    assert f["lines"]["Nglr"]["hours"] == 13.5 and f["lines"]["Nglr"]["usd"] == round(13.5 * cc.RATE, 2), f["lines"]
    assert f["total"]["hours"] == 34.62 and f["weights"]["PPCL@glr"] == 2.5 and "PPCL_K_CAP" in f["weights"]["why"]
    for usd, waits in ((3.99, False), (4.01, True)):  # the gate's $ on either side of the cap
        f = forecast(dict(t36, lmax_pi05=36 * usd / (20.4 * cc.RATE)), pk, {"delta": 0.01}, split=split)
        assert f["gate"]["v3_waits"] == waits and f["gate"]["usd"] == usd, (usd, f["gate"])
    assert forecast(t36, split=split)["lines"]["V2eps"]["records"] == 600  # no band.json: V2eps counts
    done = lkeys(plan("V1", split=split)) | {next(iter(lkeys(l2)))}
    f = forecast(t36, pk, {"delta": 0.01}, done, split)
    assert f["lines"]["V1"]["left_hours"] == 0 and f["lines"]["V2"]["left_hours"] == 5.99 and f["gate"]["hours"] == 20.4

    # the smoke's gate: PBVS0.3 prefix_ok in >= 5 of 6, the 6 cases complete, the counters there, each Base's e the
    # frozen file's (its act_hash reported, not a gate)
    sr = ([{"case_id": c, "kind": "dynamic", "arm": "PBVS0.3", "prefix_ok": True, "pbvs": dict.fromkeys(STATS, 0)}
           for c in e3["plus"][:6]] + [{"case_id": c, "kind": "base", "arm": "none", "d": 0, "e": 30 + j,
                                        "act_hash": c} for j, c in enumerate(e3["plus"][:6])])
    fz = [r | {"kind": "dynamic", "arm": "VS1.0"} for r in sr[6:]] + sr[6:]
    one = lambda n, f, k="dynamic": [r | f if r["kind"] == k and r["case_id"] in e3["plus"][:n] else r  # noqa: E731
                                     for r in sr]
    ok, n = smoke_check(sr, fz)
    assert ok and n["base_e_as_line_2"] == n["base_act_hash_as_line_2"] == 6 and smoke_check(
        one(1, {"prefix_ok": False}), fz)[0], n
    assert not smoke_check(one(2, {"prefix_ok": False}), fz)[0] and not smoke_check(sr[1:], fz)[0]
    assert not smoke_check(one(1, {"pbvs": {"steps_window": 3}}), fz)[0]
    assert not smoke_check(one(1, {"skipped_reason": "x"}), fz)[0]
    ok, n = smoke_check(one(1, {"e": 99}, "base"), fz)  # one Base's e not line 2's: stop before the V lines pay
    assert not ok and n["base_e_as_line_2"] == 5, n
    assert not smoke_check(sr, fz[:-1])[0]  # a case missing in the frozen file
    ok, n = smoke_check(one(1, {"act_hash": "x"}, "base"), fz)
    assert ok and n["base_act_hash_as_line_2"] == 5, n  # act_hash: reported only

    # the frozen file on disk: its Base case ids, in its order, are V2's first 108 Plus cases
    c2l = cc.plan("2", CMODE, CK, split)[0]
    f108 = [{"case_id": c, "kind": kd, "arm": a} for c in ids(c2l, split)
            for kd, a in [("dynamic", "PPCL"), ("dynamic", "VS1.0"), ("dynamic", "Gauto"), ("base", "none")]]
    assert frozen_check(f108, split) == {"bases": 108, "first_equal": 108, "want": 108, "pass": True}
    sw = f108[4:8] + f108[:4] + f108[8:]  # two cases swapped
    for bad in (sw, f108[4:], f108[:-1], f108[:-4] + f108[-4:-1] + [f108[-1] | {"case_id": e3["pro"][0]}]):
        assert not frozen_check(bad, split)["pass"]
    selftest_read(split)
    print("selftest ok")


def selftest_agent():
    """The band's conventions are the regulator's: vservo.PBVSAgent on a still object, its window forced open from step
    0 and its band too wide to act (it acts as Base), on a toy ARX arm off its model; the plans' xy rows partly beyond
    +-1. Its |e| per window step (sum and max) equal errors() on the record the runner would write (x per step before
    its action, the executed clipped action, plans_in, t_close)."""
    for G, B in (((0.004, 0.006), (0.5, 0.3)), ((0.011, 0.011), (0.0, 0.0))):
        ag = vservo.PBVSAgent("PBVS1.0", 10, 0, {"model": "A", "G": list(G), "B": list(B), "delta": 1e9})
        x, dx, xs, acts, pin = np.zeros(3), np.zeros(2), [], [], []
        for t in range(57):
            ag.observe(t, x, np.ones(3))
            ag.t_e = 0
            if ag.sched.wants_call(t, ag.trigger(t)):
                k = np.arange(50)[:, None]
                ch = np.column_stack([1.4 * np.sin(0.3 * k + t), np.cos(0.2 * k - t), -k / 50, np.zeros((50, 3)),
                                      np.where(t + k >= 44, 1.0, -1.0)])
                ag.sched.issue(t, ch)
            if ag.sched.arrive(t):
                ag.on_new_chunk()
                pin.append([ag.sched.t_obs, t])
            a = np.clip(ag.act(t, ag.sched.action(t)), -1, 1)
            xs.append(x.copy())
            acts.append(a)
            dx = np.array([0.006, 0.009]) * a[:2] + np.array([0.7, 0.4]) * dx  # the true arm, not the model
            x = x + np.append(dx, 0.0)
        r = {"traj": {"x": xs, "a": [v[:3] for v in acts]}, "plans_in": pin,
             "t_close": int(np.argmax(np.asarray(acts)[:, 6] > 0))}
        e, st = errors(r, G, B), ag.pbvs_stats
        assert r["t_close"] == 44 and st["steps_window"] == len(e) == 45 and st["steps_acting"] == 0, (st, len(e))
        assert abs(st["max_e"] - max(e)) < 1e-12 and abs(st["sum_e"] - sum(e)) < 1e-12 and max(e) > 1e-3, (st, e)


def selftest_read(split):
    """The reading on a synthetic pod: line 2 and 3's records (mode С, k_π = 1.0: line 2's first 108 Plus cases with
    VS1.0), V0's toy arm, V1 -> pbvs_k.json, V2, V3, Nglr, V2eps and part 2's records. Success by the case's place in
    cycles of 100 (pct: the arm's share in %, by (arm, d) or arm; ("eps", d) for V2eps; else 50), so the differences on
    the 300 cases are exact; task clusters from fake LIBERO names as c1e4c_summary's selftest; one policy seed per case."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    cal, e300 = run.set_ids(split, "calib"), run.set_ids(split, "eval300")
    f108 = cs.sources(e300)["plus"][:cc.VS_STOP["plus"]]
    pos = {c: j for j, c in enumerate(split["eval"])} | {c: j for j, c in enumerate(cal)}
    cf = {"brain": P, "revision": "r", "precision": "fp32", "s": 10, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01, "R": 360,
          "split": split["sha256"]}
    mode, k = {"pi05_lmax": {"mode": "S"}, "oftplus_lmax": {"mode": "S"}, "pi05_bench": {"mode": "P"}}, {P: {"k": "VS1.0"}}

    def rec(x, pct, pbvs=None, gpu=True):  # x: lk (+ eps)
        _, b, c, kind, a, d, _, arms = x[:8]
        eps, h = len(x) > 8 and x[8], sum(map(ord, c))
        s = list(run.MAX_STEPS)[h % 4]
        r = {"brain": b, "case_id": c, "kind": kind, "arm": a, "method": cs.method(a), "d": d, "policy_seed": 195 + pos[c],
             "conf": cf | {"d": d, "arms": list(arms), "kbar": 0.306 if "Gauto" in arms else None}
             | ({"gpu": "NVIDIA GeForce RTX 4090"} if gpu else {}) | ({"pbvs": pbvs} if pbvs else {}),
             "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None,
             "distance_m": (0.06, 0.12)[h % 2], "e": 30, "prefix_ok": None if kind == "base" else True,
             "replay_miss": None, "skipped_reason": None, "plans_in": [[0, 0], [10, 10], [30, 30 + d]], "t_close": 40,
             "calls_sched": 20, "calls_trig": 0 if kind == "base" else {
                 "PPCL@glr": 30 + 10 * (pos[c] % 10 == 0), "PPC@glr": 5, "PPCL": 2, "PPC": 1}.get(a, 0),
             "success": None if kind == "base" else pos[c] % 100 < pct.get(("eps", d) if eps else (a, d), pct.get(a, 50)),
             "grasp_miss_m": {"PBVS0.1": 0.012, "PBVS0.3": 0.010, "PBVS1.0": 0.02, "Gauto": 0.011}.get(a, 0.015)}
        if a.startswith("PPCL"):
            r["latch"] = {"trig": int(pos[c] % 4 == 0), "chunks": int(pos[c] % 10 == 0), "resets": 0}
        if a in LAMS:
            r["pbvs"] = {"steps_window": 20, "steps_acting": 5, "sum_e": 0.2, "max_e": 0.03, "flips": int(pos[c] % 2)}
        return r

    def world(pct, bump=0.0):
        """The pod's V records, controls lines 2 and 3, band.json, pbvs_k.json and part 2's records."""
        ctrl = [rec(x, pct) for L in "23" for x in cc.lkeys(cc.plan(L, mode, k, split))]
        v0 = [rec(x, pct) | toy(seed=pos[x[2]], bumps=[(25, bump)]) | {"case_id": x[2]}
              for x in lkeys(plan("V0", split=split))]
        bd = json.loads(json.dumps(band(v0, split)))
        v1 = [rec(x, pct, bd) for x in lkeys(plan("V1", split=split))]
        pk = json.loads(json.dumps(select(v1, split)))
        new = v0 + v1 + [rec(x, pct, (bd | {"delta": EPS} if x[8] else bd) if L != "Nglr" else None)
                         for L in ("V2", "V3", "Nglr", "V2eps") for x in lkeys(plan(L, pk, bd, split))]
        saved = [rec(("m", P, c, kd, a, d, False, ()), pct, gpu=False) for c in e300 for d in (0, 10, 20)
                 for kd, a in [("base", "none"), ("dynamic", "none")] + [("dynamic", a) for a in (
                     ("G", "PPC", "G@glr", "PPC@glr") if d == 0 else ())]]
        return new, ctrl, saved, bd, pk

    pct = {("PBVS0.3", 0): 55, ("PBVS0.3", 10): 60, ("PBVS0.3", 20): 40, "Gauto": 50, "PPCL": 45, "none": 30,
           "PPCL@glr": 41, "PPC@glr": 38, ("eps", 0): 57, "VS1.0": 20, "G": 44, "PPC": 40, "G@glr": 37}
    new, ctrl, saved, bd, pk = world(pct)
    assert bd["model"] == "I*" and bd["delta"] == FLOOR and pk["arm"] == "PBVS0.3", (bd, pk)
    rd = lambda n=new, c=ctrl, m=mode, b=bd, p=pk: read(n, c, saved, m, k, b, p, names, split)  # noqa: E731
    probe = rec(("m", P, e300[0], "dynamic", "PBVS1.0", 0, False, ("PBVS1.0",), False), pct, bd)  # off the plan
    res = rd(new + [probe] + new[:5])  # new[:5]: an identical copy
    ch, ms, rep = res["checks"], res["measurements"], res["report"]
    assert ch["pass"] and ch["off_plan"] == {f"{P}/PBVS1.0": 1} and ch["read"] == len(new), ch
    assert ch["complete"]["V2eps"] == {"expected": 0, "missing": 0, "run": False, "pass": True}, ch["complete"]
    assert ch["complete"]["Nglr"]["expected"] == 900 and ch["complete"]["V0"]["expected"] == 60, ch["complete"]
    assert ch["controls_complete"]["2"] == {"expected": 108 * 4 + 192 * 3, "missing": 0, "pass": True}, ch
    assert set(ch["e_match"]) == {"V2/d0", "V3d10/d10", "V3d20/d20", "Nglr/d0"} and ch["e_match"]["V3d20/d20"] == {
        "equal": 300, "total": 300, "pass": True}, ch["e_match"]
    # the measurements: estimates and 95% CIs, no verdict, no threshold
    assert list(ms) == list(MEAS)[:8] and all(not {"pass", "outcome"} & set(v) and not re.search("[<>≤≥]", v["test"])
                                              for v in ms.values()), ms
    want = {"Н1": 5, "Н2/d10": 10, "Н2/d20": -10, "Н2/pool": 0, "Н3/d0": 10, "Н3/d10": 15, "Н3/d20": -5, "И5": 3}
    assert {n: ms[n]["diff_pp"] for n in want} == want and ms["Н2/pool"]["pairs"] == 600, ms
    assert ms["Н2/pool"]["pooled_d"] == [10, 20] and not [n for n in ms if "lines_not_run" in ms[n]], ms
    assert all(ms[n]["pairs"] == 300 and ms[n]["invalid_pairs"] == 0 for n in want if n != "Н2/pool"), ms
    assert ms["Н2/pool"]["clusters"] == ms["Н2/d10"]["clusters"] and ms["Н1"]["lo2.5"] <= 5 <= ms["Н1"]["hi97.5"]
    # the report: PBVS - none through the pod, the gate, the slices, counters, grasp miss, retention
    assert rep["pbvs_none"]["PBVS0.3/d0"]["diff_pp"] == 25 and rep["pbvs_none"]["PBVS0.3/d0"]["через под"], rep
    assert list(rep["pbvs_gates"]) == ["V2"] and rep["pbvs_gates"]["V2"]["gate_m"] == FLOOR, rep["pbvs_gates"]
    assert all(sum(v[s]["pairs"] for s in ("6cm", "12cm")) == 300 for v in rep["slices_M1"].values()), rep
    pc = rep["pbvs_counters"][f"{P}/dynamic/PBVS0.3/0"]
    assert pc == {"n": 300, "window": 300, "acting_share": 0.25, "mean_e_m": 0.01, "mean_e_p5_p95": [0.01, 0.01],
                  "max_e_m": 0.03, "flips_per_episode": 0.5, "flipped_share": 0.5}, pc
    assert rep["latch"][f"{P}/dynamic/PPCL@glr/0"] == {"n": 300, "chunks_share": 0.1, "trig_share": 0.25}, rep["latch"]
    assert rep["grasp_miss"]["d10"] == {"n": 300, "PBVS0.3": 0.01, "Gauto": 0.011, "PBVS0.3-Gauto": -0.001}, rep
    rt = rep["retention"]
    assert rt["pod"]["PPCL@glr-none"]["diff_pp"] == 11 and rt["pod"]["PPCL-none"]["diff_pp"] == 15, rt
    assert rt["pod"]["ratio"] == round(11 / 15, 3) and rt["pod"]["PPCL-none"]["через под"], rt
    assert rt["part2/G"]["G@glr-none"]["diff_pp"] == 7 and rt["part2/PPC"]["PPC-none"]["diff_pp"] == 10, rt
    assert rt["part2/PPC"]["PPC@glr-none"]["diff_pp"] == 8 and "через под" not in rt["part2/G"]["G-none"], rt
    assert rep["calls"][f"{P}/dynamic/none/10"]["n"] == 300 and rep["band"] == bd and rep["pbvs_k"] == pk
    assert not [x for x in rep["calls"] if "VS1.0" in x or G1 in x], rep["calls"]  # VS1.0 is read in its line only
    # the noisy cell's one table: success, calls per episode (sched + trig: mean, median), the latched share; G@glr and
    # "none" of part 2 through the pod; under it PPCL@glr - G@glr and the calls difference, one pair per case and seed
    nt = rep["nglr"]
    assert nt["table"] == {
        "PPCL@glr": {"n": 300, "success_pct": 41.0, "calls_mean": 51.0, "calls_median": 50.0, "latched_share": 0.1},
        "PPC@glr (this pod)": {"n": 300, "success_pct": 38.0, "calls_mean": 25.0, "calls_median": 25.0},
        "G@glr (part 2, через под)": {"n": 300, "success_pct": 37.0, "calls_mean": 20.0, "calls_median": 20.0},
        "none (part 2, через под)": {"n": 300, "success_pct": 30.0, "calls_mean": 20.0, "calls_median": 20.0},
        "Base (this pod, cut at the event)": {"n": 300, "calls_mean": 20.0, "calls_median": 20.0}}, nt["table"]
    assert nt["PPCL@glr-G@glr"]["diff_pp"] == 4 and nt["PPCL@glr-G@glr"]["pairs"] == 300 and nt["PPCL@glr-G@glr"][
        "drops"] == {} and nt["PPCL@glr-G@glr"]["через под"], nt
    assert nt["calls_PPCL@glr-G@glr"] == {"pairs": 300, "mean": 31.0, "median": 30.0, "через под": True}, nt
    # G(k = 1.0): line 2's frozen 108 Plus cases, one pair per case (the V2 PBVS episode, the frozen VS1.0 one), the
    # difference per pair in {-1, 0, 1}; a frozen case with another policy seed, one with another Base e: both drop
    g1 = lambda cs_: round(100 * (sum(pos[c] % 100 < 55 for c in cs_) - sum(pos[c] % 100 < 20 for c in cs_)) / len(  # noqa
        cs_), 3)
    assert rep["pbvs_g1"]["cases"] == rep["pbvs_g1"]["pairs"] == 108 and rep["pbvs_g1"]["drops"] == {} and rep[
        "pbvs_g1"]["diff_pp"] == g1(f108), rep["pbvs_g1"]
    off = [r | {"policy_seed": 1} if r["arm"] == "VS1.0" and r["case_id"] == f108[3] else
           r | {"e": 29} if r["kind"] == "base" and r["d"] == 0 and r["case_id"] == f108[7] else r for r in ctrl]
    r2 = rd(c=off)
    g = r2["report"]["pbvs_g1"]
    assert r2["checks"]["pass"] and g["cases"] == 108 and g["pairs"] == 106 and g["drops"] == {"seed": 1, "e": 1}, g
    assert g["diff_pp"] == g1([c for c in f108 if c not in (f108[3], f108[7])]) and r2["measurements"] == ms, g
    # line 2's frozen VS1.0 is read for G(k = 1.0) only: its success leaves every other number as it was
    r2 = rd(c=[r | {"success": not r["success"]} if r["arm"] == "VS1.0" else r for r in ctrl])
    assert r2["measurements"] == ms and r2["report"]["pbvs_g1"]["diff_pp"] != rep["pbvs_g1"]["diff_pp"], r2["report"]
    assert {kk: v for kk, v in r2["report"].items() if kk != "pbvs_g1"} == {
        kk: v for kk, v in rep.items() if kk != "pbvs_g1"}
    cc.exits(lambda: rd(new + [new[700] | {"success": not new[700]["success"]}]), 2)

    # the checks before reading, each alone
    v2d = lambda r: r["d"] == 0 and r["arm"] == "PBVS0.3" and r["case_id"] in e300 and not vk(r, bd)[-1]  # noqa: E731
    v2 = [r for r in new if v2d(r)]
    for why, kw in {"complete": dict(n=[r for r in new if r not in v2[:7]]),  # 7 of V2's 600 records (> 1 %)
                    "band": dict(b=bd | {"delta": 0.006}), "pbvs_k": dict(p=pk | {"arm": "PBVS0.1"}),
                    "band_in_conf": dict(n=[r | {"conf": r["conf"] | {"pbvs": bd | {"G": [0.01, 0.01]}}}
                                            if r is v2[3] else r for r in new]),
                    "conf": dict(n=[r | {"conf": r["conf"] | {"t_ramp": 4}} if r["d"] == 20 else r for r in new]),
                    "gauto_on_pod": dict(m=mode | {"pi05_lmax": {"mode": "P"}}),
                    "controls_complete": dict(c=[r for r in ctrl if not (r["d"] == 10 and r["arm"] == "PPCL" and
                                                                         pos[r["case_id"]] < 20)])}.items():
        with contextlib.redirect_stderr(io.StringIO()):
            ch = rd(**kw)["checks"]
        c = ch[why]
        assert not ch["pass"] and (not c["pass"] if "pass" in c else not all(v["pass"] for v in c.values())), (why, c)
    # 6 of V2's 300 cases with another Base e (theirs and their PBVS episode's): 98 % pass and the pairs drop (2 %
    # of Н1); 7 fail. 6 PBVS episodes with prefix_ok false: 2 % pass, 7 stop the reading
    for n, ok in ((6, True), (7, False)):
        cut = set(e300[:n])
        r = rd([x | {"e": 31} if x["case_id"] in cut and x["d"] == 0 and x["conf"].get("pbvs") == bd else x
                for x in new])
        assert r["checks"]["e_match"]["V2/d0"] == {"equal": 300 - n, "total": 300, "pass": ok} and (
            "measurements" in r) == ok, r["checks"]["e_match"]
        if ok:
            assert r["measurements"]["Н1"]["invalid_pairs"] == 6 and r["measurements"]["Н1"]["pairs"] == 294
        r = rd([x | {"prefix_ok": False} if v2d(x) and x["kind"] == "dynamic" and x["case_id"] in cut else x
                for x in new])
        assert r["checks"]["invalid_pairs"]["Н1"] == {"pairs": 300, "invalid": n, "pass": ok} and (
            "measurements" in r) == ok == r["checks"]["pass"], r["checks"].get("invalid_pairs")
        if not ok:
            cc.exits(lambda: finish(r), 3)
    # V3 held for the owner (spec §8), so Nglr too: their measurements are not read, the rest are
    r = rd([x for x in new if not (x["case_id"] in e300 and (x["d"] > 0 or "PPCL@glr" in x["conf"]["arms"]))])
    ms = r["measurements"]
    assert r["checks"]["pass"] and not r["checks"]["complete"]["V3d10"]["run"] and ms["Н1"]["diff_pp"] == 5, r["checks"]
    assert all(ms[n]["diff_pp"] is None and ms[n]["lines_not_run"] for n in ("Н2/pool", "Н3/d20", "И5")), ms
    assert ms["Н2/pool"]["pooled_d"] == [] and ms["Н2/pool"]["lines_not_run"] == ["V3d10", "V3d20"], ms["Н2/pool"]
    # one d of V3 not run (V3d20, the last line, cut): Н2 per d for the d that ran, the pool over it alone and named
    # (pooled_d), the missing line listed; nothing pooled in silence, no crash
    r = rd([x for x in new if not (x["case_id"] in e300 and x["d"] == 20)])
    ms = r["measurements"]
    assert r["checks"]["pass"] and r["checks"]["complete"]["V3d10"]["run"] and not r["checks"]["complete"]["V3d20"][
        "run"] and set(r["checks"]["e_match"]) == {"V2/d0", "V3d10/d10", "Nglr/d0"}, r["checks"]
    assert ms["Н2/d10"]["diff_pp"] == 10 and ms["Н3/d10"]["diff_pp"] == 15 and ms["Н2/pool"] == ms["Н2/d10"] | {
        "test": MEAS["Н2/pool"][0], "pooled_d": [10], "lines_not_run": ["V3d20"]}, ms["Н2/pool"]
    assert all(ms[n]["diff_pp"] is None and ms[n]["lines_not_run"] == ["V3d20"] for n in ("Н2/d20", "Н3/d20")), ms
    assert r["report"]["grasp_miss"]["d20"]["n"] == 0 and r["report"]["grasp_miss"]["d10"]["n"] == 300, r["report"]
    # a band above 15 mm: V2eps (the servo at our 5 mm gate) runs, Н1 at that gate and PBVSeps - PBVS are measured,
    # both PBVS lines side by side with their gates; its records differ from V2's only in conf["pbvs"]["delta"]
    new, ctrl, saved, bd, pk = world(pct, bump=0.02)
    assert bd["flag_gt_15mm"] and bd["delta"] > FLAG, bd
    res = rd(new, ctrl, mode, bd, pk)
    ms, ch, rep = res["measurements"], res["checks"], res["report"]
    assert ch["pass"] and ch["complete"]["V2eps"]["expected"] == 600 and ch["e_match"]["V2eps/d0"]["equal"] == 300, ch
    assert ms["Н1/eps"]["diff_pp"] == 7 and ms["PBVSeps-PBVS"]["diff_pp"] == 2 and ms["Н1"]["diff_pp"] == 5, ms
    pg = rep["pbvs_gates"]
    assert [(L, v["gate_m"], v["success_pct"], v["pbvs-Gauto"]["diff_pp"], v["pbvs-none"]["diff_pp"])
            for L, v in pg.items()] == [("V2", bd["delta"], 55.0, 5, 25), ("V2eps", EPS, 57.0, 7, 27)], pg
    assert f"{P}/dynamic/PBVS0.3/eps/0" in rep["pbvs_counters"], rep["pbvs_counters"]
    z = [r for r in new if r["conf"].get("pbvs", {}).get("delta") == EPS and r["kind"] == "dynamic"]
    v = {r["case_id"]: r for r in new if v2d(r) and r["kind"] == "dynamic"}
    assert len(z) == 300 and all(x["conf"] | {"pbvs": None} == v[x["case_id"]]["conf"] | {"pbvs": None} and
                                 x["conf"]["pbvs"] == v[x["case_id"]]["conf"]["pbvs"] | {"delta": EPS} for x in z)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--band", nargs="+", metavar="JSONL", help="V0's records -> band.json (+ band_eps.json)")
    g.add_argument("--select", nargs="+", metavar="JSONL", help="V1's records -> pbvs_k.json")
    g.add_argument("--forecast", metavar="TEMPO", help="the tempo file: hours and $ per line, the V3 gate (exit 3)")
    g.add_argument("--smoke-check", nargs="+", metavar="JSONL", help="the smoke's records: exit 0, or 3")
    g.add_argument("--controls-check", nargs="+", metavar="JSONL", help="controls lines 2, 3 complete: exit 0, or 3")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_V.jsonl: checks, measurements, report")
    ap.add_argument("--line", choices=LINES + ("V3",))
    ap.add_argument("--band-json", default=f"{OUT}/band.json", help="band.json (--plan, --forecast, the reading)")
    ap.add_argument("--pbvs-k", default=f"{OUT}/pbvs_k.json", help="pbvs_k.json (--plan, --forecast, the reading)")
    ap.add_argument("--mode", default=f"{OUT}/mode.json", help="the controls' mode.json")
    ap.add_argument("--k", default=f"{OUT}/k_select.json", help="the controls' k_select.json")
    ap.add_argument("--controls", nargs="+", default=[f"{OUT}/grid_all_C.jsonl"], metavar="JSONL",
                    help="the reading: the controls records of lines 2 and 3")
    ap.add_argument("--p2", default=str(cc.P2), help="the reading: part 2's records (read only)")
    ap.add_argument("--done", nargs="+", default=[], metavar="JSONL", help="--forecast: the V lines' records so far")
    ap.add_argument("--out", help="--band: default --band-json; --select: default --pbvs-k; the reading: JSON too")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f) if "kind" in x]  # noqa: E731
    jload = lambda f: json.loads(Path(f).read_text()) if Path(f).exists() else None  # noqa: E731
    a.read = a.read[a.read[:1] == ["read"]:]  # "read" before the files is optional
    if bool(a.read) + sum(map(bool, (a.plan, a.band, a.select, a.forecast, a.smoke_check, a.controls_check,
                                     a.selftest))) != 1:
        ap.error("one mode: an option of the group, or [read] JSONL...")
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    split = maxwrap.load_split(cs.SPLIT)
    if a.selftest:
        selftest()
    elif a.plan:
        if lns := plan(a.line, jload(a.pbvs_k), jload(a.band_json), split):
            print(cc.tsv(lns))
    elif a.band:
        b = band(load(a.band), split)
        write_band(b, a.out or a.band_json)
        print(e2.dump(b))
    elif a.select:
        s = select(load(a.select), split)
        Path(a.out or a.pbvs_k).parent.mkdir(parents=True, exist_ok=True)
        Path(a.out or a.pbvs_k).write_text(json.dumps(s, indent=1) + "\n")
        print(e2.dump(s))
    elif a.forecast:
        bd = jload(a.band_json)
        f = forecast(cc.read_tempo(a.forecast), jload(a.pbvs_k), bd, {vk(r, bd) for r in load(a.done)}, split)
        print(e2.dump(f))
        raise SystemExit(3 if f["gate"]["v3_waits"] else 0)  # the stage says whether the hold is in force
    elif a.smoke_check:
        ok, n = smoke_check(load(a.smoke_check), load(a.controls))
        print(e2.dump({"smoke": n, "pass": ok}))
        print(f"C1-E4 PBVS smoke: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    elif a.controls_check:
        _, _, comp = controls(load(a.controls_check), jload(a.mode), jload(a.k), split)
        fz = [f for f in a.controls_check if Path(f).name == FROZEN]  # on disk, in its own order
        comp["frozen"] = frozen_check(load(fz), split) | {"file": fz}
        print(e2.dump(comp))
        raise SystemExit(0 if all(v["pass"] for v in comp.values()) else 3)
    else:
        saved = load([Path(a.p2) / f for f in cc.SAVED["pi05_lmax"]])
        finish(read(load(a.read), load(a.controls), saved, jload(a.mode), jload(a.k), jload(a.band_json),
                    jload(a.pbvs_k), cs.libero_names(), split), a.out)
