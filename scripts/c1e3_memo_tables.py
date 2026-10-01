"""C1-E3 memo tables beyond summary.json (docs/c1/e3.md): the hand-off breakdown by shift class and the noise section
(per-point false / missed detections of G-R and PPC, and noisy G-R's loss vs oracle split by its detection outcome).

Report only, read after the data. The noise section needs each detector's first detection: G-R's engagement is in the
records (t_engage); PPC's own speed trigger is not, so it is replayed offline: the same seed's noise stream through the
real HAgent and Schedule, a dummy chunk without a close command, the object still except the step shift, no pushes
(the replay reproduces the recorded G-R engagement in ~97% of noisy episodes, PPC arms' engagement in ~88%).
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e3_memo_tables.py results/c1-e3/grid_all.jsonl
"""

import collections
import json
import sys
from pathlib import Path

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parent), str(Path(__file__).resolve().parents[1] / "src")]
import c1e2_summary as e2  # noqa: E402
import c1e3_summary as s  # noqa: E402
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402

BETA, EPS, VMIN = 0.2, 0.02, 0.01  # the pilot's noisy-eyes choice (spec journal 18), in every noisy row
CHUNK = np.zeros((orx.H, 7))
CHUNK[:, 0], CHUNK[:, 6] = 0.5, -1.0


def replay(r, method, tag):
    """(engagement t_e, first PPC detection) of record r with the noisy eyes `tag`, replayed offline."""
    ep = hf.make_episode(r["suite"], r["task"], r["init"], r["kind"])
    ag = hf.HAgent(method, *orx.CELLS[r["cell"]], eps=EPS, ppc_v_min=VMIN,
                   tracker=hf.Tracker(ep.seed, *hf.NOISE[tag], BETA))
    det = []
    if method == "PPC":
        measure = ag._ppc_measure

        def logged(t):
            before = ag.ppc_off
            measure(t)
            if ag.ppc_off is not before and ag.ppc_off:
                det.append(t)
        ag._ppc_measure = logged
    tf, tc = r["t_fire"], r["t_close"]
    shift = ep.delta if r["kind"] != "control" and tf is not None else np.zeros(3)
    for t in range((tc if tc is not None else r["steps"] - 1) + 1):
        ag.observe(t, np.zeros(3), shift if tf is not None and t > tf else np.zeros(3))
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ag.sched.issue(t, CHUNK)
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        ag.act(t, ag.sched.action(t))
        if t == tc:
            ag.grasp_started = True
    return ag.t_e, (det[0] if det else None)


def outcome(t_det, r):
    """false: before the shift is observable (any detection in control); on time / missed: before the first close."""
    tf, tc = r["t_fire"], r["t_close"]
    seen = t_det is not None and (tc is None or t_det < tc)
    if r["kind"] == "control":
        return "false" if seen else "quiet"
    if tf is None:
        return "false" if seen else "no shift"
    if seen and t_det <= tf:
        return "false"
    return "on time" if seen else "missed"


def detections(recs, det):
    oc = collections.Counter(outcome(det(r), r) for r in recs)
    fired = sum(r["t_fire"] is not None for r in recs) or 1
    delay = [det(r) - r["t_fire"] for r in recs if outcome(det(r), r) == "on time"]
    return {"false": oc["false"] / len(recs), "missed": oc["missed"] / fired,
            "delay": float(np.median(delay)) if delay else None, "success": float(np.mean([r["success"] for r in recs]))}


def main(path):
    raw = [json.loads(line) for line in open(path) if '"libero_10"' not in line]  # the 26 rule tasks
    idx = s.index(raw)
    d = lambda *a: e2.rnd(e2.diff_pp(s.prs(idx, *a)))  # noqa: E731
    cell = lambda kind, c, arm, inits=range(10): [r for k, r in idx.get((kind, c, arm, 0.0), {}).items()  # noqa: E731
                                                  if k[2] in inits]
    out = {"hand_off": {}, "noise_C": {}, "noise_real": {}, "noise_link": {}}

    for al, inits in ((0.0, range(30)), (1.0, range(20))):  # rows 2 and 11 (Q1, Q3b)
        for c in (0, 1, 2):
            row = {"GR-G": d("step", al, "AB", "GR", "G", inits, (c,))}
            for arm in ("G", "GR", "Gkeep"):
                rs = [r for (k, ce, a, alp), dd in idx.items() if k == "step" and ce in "AB" and a == arm and alp == al
                      for key, r in dd.items() if key[2] in inits and r["mag_class"] == c]
                kl = [e for r in rs for e in r["kappa_log"]]
                row[arm] = {"n": len(rs), "miss_along_cm": round(100 * float(np.median(
                    [r["grasp_miss_along_m"] for r in rs if r["grasp_miss_along_m"] is not None])), 2),
                    "kappa_dead": round(float(np.mean([e["k_hat"] == 0 and abs(e["m"]["20"]) < 0.037044 for e in kl])), 3),
                    "kappa_mean": round(float(np.mean([e["k"] for e in kl])), 3)}
            out["hand_off"][f"alpha {al} class {c}"] = row
    out["hand_off"]["Gkeep-G a0 AB 0-9"] = d("step", 0.0, "AB", "Gkeep", "G", range(10))
    out["hand_off"]["GR-Gkeep a0 AB 0-9"] = d("step", 0.0, "AB", "GR", "Gkeep", range(10))
    out["hand_off"]["GR-Gkeep a1 A 0-19"] = d("step", 1.0, "A", "GR", "Gkeep", range(20))
    out["hand_off"]["GR-none ACF 0-19"] = d("step", 0.0, "ACF", "GR", "none", range(20))
    out["hand_off"]["G-none A 0-19"] = d("step", 0.0, "A", "G", "none", range(20))

    for tag in ("oracle", "s0.5", "s1", "s2", "L3", "L6", "d10", "real"):  # C, step, inits 0-9 (rows 1, 3, 9, 22, 23, 26)
        gr = "GR" if tag == "oracle" else f"GR@{tag}"
        row = {"GR": detections(cell("step", "C", gr), lambda r: r["t_engage"]), "GR-none": d("step", 0.0, "C", gr, "none", range(10))}
        if tag != "oracle":
            row["GR-oracle"] = d("step", 0.0, "C", gr, "GR", range(10))
        row["GR missed by class"] = [round(float(np.mean([outcome(r["t_engage"], r) == "missed" for r in cell("step", "C", gr)
                                                          if r["mag_class"] == c and r["t_fire"] is not None])), 3) for c in (0, 1, 2)]
        if tag in ("s0.5", "s1", "s2", "real"):
            ppc = cell("step", "C", f"PPC@{tag}")
            row["PPC"] = detections(ppc, lambda r, tag=tag: replay(r, "PPC", tag)[1])
            row["PPC-none"] = d("step", 0.0, "C", f"PPC@{tag}", "none", range(10))
            row["GR-PPC"] = d("step", 0.0, "C", gr, f"PPC@{tag}", range(10))
            row["PPC calls_trig"] = round(float(np.mean([r["calls_trig"] for r in ppc])), 2)
        if tag == "oracle":
            row["PPC-none"] = d("step", 0.0, "C", "PPC", "none", range(10))
            row["GR-PPC"] = d("step", 0.0, "C", "GR", "PPC", range(10))
            row["PPC calls_trig"] = round(float(np.mean([r["calls_trig"] for r in cell("step", "C", "PPC")])), 2)
        out["noise_C"][tag] = row
    for c in "ACF":  # the realistic point: shift rows 20, 22; control row 21 (PPC replayed on the same control episodes)
        ctl = cell("control", c, "GR@real")
        out["noise_real"][c] = {"GR step": detections(cell("step", c, "GR@real"), lambda r: r["t_engage"]),
                                "PPC step": detections(cell("step", c, "PPC@real"), lambda r: replay(r, "PPC", "real")[1]),
                                "GR control false": detections(ctl, lambda r: r["t_engage"])["false"],
                                "PPC control false": detections(ctl, lambda r: replay(r, "PPC", "real")[1])["false"]}
    for arm, cells, inits in (("GR@real", "ACF", range(20)), ("GR@s2", "C", range(10)), ("GR@L6", "C", range(10))):
        for cat in ("on time", "missed", "false"):
            trip = [(r, idx[("step", c, "none", 0.0)][k], idx[("step", c, "GR", 0.0)][k]) for c in cells
                    for k, r in idx[("step", c, arm, 0.0)].items() if k[2] in inits and outcome(r["t_engage"], r) == cat]
            if trip:
                out["noise_link"][f"{arm} {cat}"] = {"n": len(trip), **{lab: round(float(np.mean([t[i]["success"] for t in trip])), 3)
                                                                       for i, lab in enumerate(("arm", "none", "oracle GR"))}}
    out["loss_split"] = {"ACF real 0-19 GR@real-GR": d("step", 0.0, "ACF", "GR@real", "GR", range(20))}
    for tag, cells, inits in [(t, "C", range(10)) for t in ("s0.5", "s1", "s2", "L3", "L6", "d10", "real")] + [("real", "ACF", range(20))]:
        loss, tot = collections.Counter(), 0  # noisy G-R's loss vs oracle G-R, by its own detection outcome, pp of the point
        for c in cells:
            orc = idx[("step", c, "GR", 0.0)]
            for k, r in idx[("step", c, f"GR@{tag}", 0.0)].items():
                if k[2] in inits:
                    loss[outcome(r["t_engage"], r)] += orc[k]["success"] - r["success"]
                    tot += 1
        out["loss_split"][f"{cells} {tag}"] = {cat: round(100 * v / tot, 2) for cat, v in loss.items()}
    ppc2, none = cell("step", "C", "PPC@s2"), idx[("step", "C", "none", 0.0)]
    for lo, hi in ((0, 0), (1, 2)):
        rs = [r for r in ppc2 if lo <= r["calls_trig"] <= hi]
        out["noise_link"][f"PPC@s2 calls_trig {lo}-{hi}"] = {"n": len(rs), "arm": round(float(np.mean([r["success"] for r in rs])), 3),
                                                              "none": round(float(np.mean([none[(r["suite"], r["task"], r["init"])]["success"]
                                                                                           for r in rs])), 3)}
    agree = collections.Counter()
    for r in raw:
        if "@" in r["arm"]:
            method, _, tag = hf.parse_arm(r["arm"])
            agree[method, replay(r, method, tag)[0] == r["t_engage"]] += 1
    out["replay_agreement"] = {m: round(agree[m, True] / (agree[m, True] + agree[m, False]), 3) for m in ("GR", "PPC")}
    print(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/c1-e3/grid_all.jsonl")
