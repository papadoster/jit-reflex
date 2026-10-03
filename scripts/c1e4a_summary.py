"""C1-E4a verdict: applies section 4 of docs/superpowers/specs/2026-10-03-c1-e4-pi05-headroom-design.md to the JSON
lines of scripts/c1e2_run.py (no manual decisions). Writes summary.json next to the pi0.5 file.
    python scripts/c1e4a_summary.py [results/c1-e4/headroom_pi05.jsonl results/c1-e4/headroom_smolvla.jsonl]
    python scripts/c1e4a_summary.py --selftest
If the pipeline check fails (incomplete runs, control below 47/52, an all-zero headroom record), only the check is
returned and the headroom is not read.
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import objreflex as orx  # noqa: E402

N_BOOT, TOL = 10_000, 1e-9  # TOL: equality to a threshold passes despite float rounding
STEP_INITS, CONTROL_INITS, CONTROL_MIN = range(10), (40, 41), 47
SEEING, BLIND = 0.30, 0.15
HS = ("10", "25", "50", "close")
C1E2 = Path(__file__).resolve().parents[1] / "results/c1-e2/grid_all.jsonl"


def load(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def pick(rs, kind, inits):
    """One record per (suite, task, init) of cell A, method none; the last one wins (a resumed rerun)."""
    return {(r["suite"], r["task"], r["init"]): r for r in rs
            if r["kind"] == kind and r["cell"] == "A" and r["method"] == "none" and r["init"] in inits}


def check(pi_ctl, pi_step, sm_step):
    """Spec §4.1: complete runs of the right brains, control success >= 47/52, no step record with every headroom
    ratio exactly 0, at least one fired shift per brain."""
    want = {(s, t, i) for s, t in orx.TASKS for i in STEP_INITS}
    want_c = {(s, t, i) for s, t in orx.TASKS for i in CONTROL_INITS}
    zeros = [k for d in (pi_step, sm_step) for k, r in d.items()
             if r.get("headroom") and all(r["headroom"]["ratio"][h] == 0.0 for h in HS)]
    ok_c = sum(r["success"] for r in pi_ctl.values())
    out = {"missing_control": len(want_c - pi_ctl.keys()), "missing_step_pi05": len(want - pi_step.keys()),
           "missing_step_smolvla": len(want - sm_step.keys()), "control_success": [ok_c, len(pi_ctl)],
           "all_zero_records": [list(k) for k in zeros]}
    out["wrong_policy"] = sum(n not in r.get("policy", "") for n, d in (("pi05", pi_ctl), ("pi05", pi_step),
                                                                         ("smolvla", sm_step)) for r in d.values())
    out["fired"] = [sum(bool(r.get("headroom")) for r in d.values()) for d in (pi_step, sm_step)]
    out["pass"] = (not any(out[k] for k in ("missing_control", "missing_step_pi05", "missing_step_smolvla", "wrong_policy"))
                   and ok_c >= CONTROL_MIN and not zeros and min(out["fired"]) > 0)
    return out


def by_task(d, h, where=lambda r: True):
    """{task cluster: [ratio at h]} over fired step records."""
    cl = defaultdict(list)
    for k, r in d.items():
        if r.get("headroom") and where(r):
            cl[k[:2]].append(r["headroom"]["ratio"][h])
    return cl


def boot(cl_a, cl_b=None):
    """Median (and median a - median b on the same resampled tasks), 90% percentile bootstrap over the 26 tasks
    (a task with no fired shift adds nothing to a resample)."""
    keys = sorted(orx.TASKS)
    if not cl_a:
        return None
    idx = np.random.default_rng(0).integers(len(keys), size=(N_BOOT, len(keys)))

    def med(cl, ks):
        x = [v for k in ks for v in cl.get(k, [])]
        return float(np.median(x)) if x else np.nan
    a = np.array([med(cl_a, [keys[j] for j in row]) for row in idx])
    out = {"median": med(cl_a, keys), "lo5": float(np.nanpercentile(a, 5)), "hi95": float(np.nanpercentile(a, 95)),
           "n": sum(map(len, cl_a.values()))}
    if cl_b is not None:
        b = np.array([med(cl_b, [keys[j] for j in row]) for row in idx])
        out |= {"diff": out["median"] - med(cl_b, keys), "diff_lo5": float(np.nanpercentile(a - b, 5)),
                "diff_hi95": float(np.nanpercentile(a - b, 95))}
    return out


def verdict(m_pi, lo_pi, m_s):
    """Spec §4.2."""
    if m_pi >= SEEING - TOL and lo_pi > m_s:
        return "ЗРЯЧИЙ"
    if m_pi <= max(BLIND, m_s) + TOL:
        return "СЛЕПОЙ"
    return "СЕРАЯ ЗОНА"


def rnd(x):
    return {k: rnd(v) for k, v in x.items()} if isinstance(x, dict) else round(x, 4) if isinstance(x, float) else x


def summarize(pi_rs, sm_rs, c1e2_rs=()):
    pi_ctl, pi_step, sm_step = pick(pi_rs, "control", CONTROL_INITS), pick(pi_rs, "step", STEP_INITS), pick(sm_rs, "step", STEP_INITS)
    res = {"check": check(pi_ctl, pi_step, sm_step)}
    if not res["check"]["pass"]:
        return res
    main = boot(by_task(pi_step, "close"), by_task(sm_step, "close"))
    m_s = boot(by_task(sm_step, "close"))
    res["main"] = {"pi05": main, "smolvla": m_s, "verdict": verdict(main["median"], main["lo5"], m_s["median"])}
    near = lambda r: r["headroom"]["dist_m"] <= 0.10  # noqa: E731  (C1-E2 near/far stage)
    rep = {}
    for name, d in (("pi05", pi_step), ("smolvla", sm_step)):
        fired = [r for r in d.values() if r.get("headroom")]
        rep[name] = {
            "median": {h: float(np.median([r["headroom"]["ratio"][h] for r in fired])) for h in HS},
            "by_class": {c: {h: float(np.median(v)) for h in HS if (v := [r["headroom"]["ratio"][h] for r in fired
                                                                          if r["mag_class"] == c])} for c in (0, 1, 2)},
            "near": boot(by_task(d, "close", near)), "far": boot(by_task(d, "close", lambda r: not near(r))),
            "share_close_ge_0.5": float(np.mean([r["headroom"]["ratio"]["close"] >= 0.5 for r in fired])),
            "k_close_median": [float(np.median([r["headroom"]["ratio"][k] for r in fired])) for k in ("k_close_before", "k_close")],
            "k_close_0": sum(r["headroom"]["ratio"]["k_close"] == 0 for r in fired),
            "fired": [len(fired), len(d)], "success_none_step": float(np.mean([r["success"] for r in d.values()]))}
    e2 = pick(c1e2_rs, "step", STEP_INITS)  # SmolVLA on the pod (C1-E2), same episodes: a check only
    both = [k for k in sm_step.keys() & e2.keys() if sm_step[k].get("headroom") and e2[k].get("headroom")]
    rep["smolvla_mac_vs_c1e2"] = {"n": len(both), **{h: [float(np.median([sm_step[k]["headroom"]["ratio"][h] for k in both])),
                                                         float(np.median([e2[k]["headroom"]["ratio"][h] for k in both]))]
                                                     for h in ("10", "25", "50")}} if both else None
    res["report"] = rep
    return rnd(res)


def selftest():
    def rec(s, t, i, kind, ratio=None, success=True, dist=0.15, cls=0, pol="x/pi05_libero"):
        hr = None if ratio is None else {"ratio": {"10": ratio, "25": ratio, "50": ratio, "close": ratio, "k_close": 20,
                                                   "k_close_before": 22}, "dist_m": dist}
        return {"policy": pol, "suite": s, "task": t, "init": i, "kind": kind, "cell": "A", "method": "none",
                "success": success, "headroom": hr, "mag_class": cls}

    def runs(r_pi, r_sm, ctl_fail=0):
        pi = [rec(s, t, i, "control", success=j >= ctl_fail) for j, (s, t, i) in
              enumerate((s, t, i) for s, t in orx.TASKS for i in CONTROL_INITS)]
        pi += [rec(s, t, i, "step", r_pi + 0.01 * (i % 3)) for s, t in orx.TASKS for i in STEP_INITS]
        sm = [rec(s, t, i, "step", r_sm + 0.01 * (i % 3), pol="HuggingFaceVLA/smolvla_libero") for s, t in orx.TASKS
              for i in STEP_INITS]
        return pi, sm

    assert summarize(*runs(0.5, 0.06))["main"]["verdict"] == "ЗРЯЧИЙ"
    assert summarize(*runs(0.12, 0.06))["main"]["verdict"] == "СЛЕПОЙ"
    assert summarize(*runs(0.18, 0.20))["main"]["verdict"] == "СЛЕПОЙ"  # not above SmolVLA on the same episodes
    assert summarize(*runs(0.22, 0.06))["main"]["verdict"] == "СЕРАЯ ЗОНА"
    res = summarize(*runs(0.5, 0.06, ctl_fail=6))  # 46/52
    assert not res["check"]["pass"] and "main" not in res and res["check"]["control_success"] == [46, 52]
    assert summarize(*runs(0.5, 0.06, ctl_fail=5))["check"]["pass"]  # 47/52 passes
    pi, sm = runs(0.5, 0.06)
    pi[-1]["headroom"]["ratio"] = {h: 0.0 for h in HS} | {"k_close": 3}
    assert summarize(pi, sm)["check"]["all_zero_records"] == [[pi[-1]["suite"], pi[-1]["task"], pi[-1]["init"]]]
    pi, sm = runs(0.5, 0.06)
    assert summarize(pi, sm[:-1])["check"]["missing_step_smolvla"] == 1
    assert summarize(sm, pi)["check"]["wrong_policy"] > 0  # swapped files
    pi, sm = runs(0.5, 0.06)
    for r in sm:
        r["headroom"] = None
    assert summarize(pi, sm)["check"]["fired"][1] == 0 and not summarize(pi, sm)["check"]["pass"]
    assert pick([rec("libero_spatial", 0, 0, "step", 0.1), rec("libero_spatial", 0, 0, "step", 0.2)], "step", [0])[("libero_spatial", 0, 0)]["headroom"]["ratio"]["10"] == 0.2
    print("selftest ok")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("pi05", nargs="?", default="results/c1-e4/headroom_pi05.jsonl")
    p.add_argument("smolvla", nargs="?", default="results/c1-e4/headroom_smolvla.jsonl")
    p.add_argument("--selftest", action="store_true")
    a = p.parse_args()
    if a.selftest:
        return selftest()
    res = summarize(load(a.pi05), load(a.smolvla), load(C1E2) if C1E2.exists() else ())
    text = json.dumps(res, ensure_ascii=False, indent=1)
    Path(a.pi05).with_name("summary.json").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
