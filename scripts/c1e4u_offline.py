"""C1-E4 part 2, task 7 offline (Mac, no brain): the carry unwind against the 6 part-1 memo traces (pi0.5, cell A40,
s = 10, d = 40; libero_spatial 0 init 7, libero_spatial 2 init 1, libero_goal 6 init 4; step G and control none).

The release is the first open command at a step >= t_c + tau_close (t_c: the first close). N_rel = t_obs of the plan
executing at it. GautoU (src/gauto.py GAgent._unwind) acts only when N_rel < t_c; it then withdraws
E = C(t_c) - C(N_rel), the extra the plan does not know, and the object should land offset by E. Placement error = the
object's xy at t_rel + SETTLE minus the control's final object xy (same triple). Pre-registered (plan, task 7): the
relation holds, and the Mac pilot runs, iff angle(E, error) < 45 deg and |error| / |E| in [0.5, 2] in all three.

The trace's "c" is the robot-object contact flag, not C (scripts/c1e4_run.py --trace), and its "tobs" is read before
the step's arrival (one step stale at arrival steps). So C(t), the executing plan and the gripper command are rebuilt by
replaying the episode's G law (src/gauto.py, oracle eyes = the traced object, the traced chunks on the conveyor); the
replay must reproduce the record's call steps, tobs, t_engage, t_close and final c_extra (asserted).
    ~/Desktop/M2R-c1-env/bin/python scripts/c1e4u_offline.py      # writes results/c1-e4/part2_unwind_offline.json
"""

import json
import platform
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gauto  # noqa: E402

TRACES = ROOT / "results/c1-e4/mac_traces_a40.jsonl"
OFFLINE = ROOT / "results/c1-e4/part2_offline.json"
OUT = ROOT / "results/c1-e4/part2_unwind_offline.json"
TAU_DEFAULT = 16
SETTLE = 5  # steps after the release at which the placement is read
ANGLE_MAX, RATIO = 45.0, (0.5, 2.0)  # pre-registered criterion
LIFT = 0.03  # part 1 grasp_ok: lifted >= 3 cm while closed
DOWN = 0.01  # report only: touch-down = first step after the carry peak within 1 cm of the height at the release


def tau_close():
    """tau_close from task 5's JSON (top level, close_obj, or the older free-closing close block), else 16."""
    d = json.loads(OFFLINE.read_text()) if OFFLINE.exists() else {}
    for key, v in (("tau_close", d.get("tau_close")), ("close_obj.tau_close", d.get("close_obj", {}).get("tau_close")),
                   ("close.tau_close", d.get("close", {}).get("tau_close"))):
        if v is not None:
            return int(v), f"{OFFLINE.relative_to(ROOT)}:{key}"
    return TAU_DEFAULT, f"missing in {OFFLINE.relative_to(ROOT)}: default {TAU_DEFAULT}"


def replay(r):
    """Per step: C at observe (c_hist), C after act, the executing plan's t_obs, the executed gripper command."""
    tr, cf = r["trace"], r["conf"]
    ag = gauto.GAgent(r["method"], *gauto.CELLS[r["cell"]], t_ramp=cf["t_ramp"], k_p=cf["k_p"], tau_k=cf["tau_k"],
                      K=gauto.K)
    plans = {p[0]: p[2] for p in tr["plans"]}
    p, x, con = np.array(tr["p"]), np.array(tr["eef"]), tr["c"]
    c_obs, c_act, tobs, grip, calls = [], [], [], [], []
    for t in range(r["steps"]):
        assert tr["tobs"][t] == (-1 if ag.sched.t_obs is None else ag.sched.t_obs), (t, "trace tobs")
        ag.observe(t, x[t], p[t], bool(con[t]))
        if ag.sched.arrive(t):
            ag.on_new_chunk()
        if ag.sched.wants_call(t, ag.trigger(t)):
            ch = np.zeros((len(plans[t]), 7))
            ch[:, [0, 1, 2, 6]] = plans[t]  # traced columns: xyz and the gripper (rotation unused by G's law)
            ag.sched.issue(t, ch)
            calls.append(t)
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        a = np.clip(ag.act(t, ag.sched.action(t)), -1, 1)
        c_obs.append(ag.c_hist[t])
        c_act.append(ag.c.copy())
        tobs.append(ag.sched.t_obs)
        grip.append(a[6])
    grip = np.array(grip)
    assert calls == sorted(plans), "call steps"
    assert np.abs(ag.c - r["c_extra"]).max() < 1e-4, (ag.c, r["c_extra"])
    assert ag.t_e == r["t_engage"] and int(np.argmax(grip > 0)) == r["t_close"], (ag.t_e, r["t_engage"])
    return np.array(c_obs), np.array(c_act), np.array(tobs), grip, p, np.array(con)


def angle_ratio(e, err):
    ne, nr = float(np.linalg.norm(e)), float(np.linalg.norm(err))
    if ne < 1e-9:
        return None, None
    return float(np.degrees(np.arccos(np.clip(e @ err / (ne * nr), -1, 1)))), nr / ne


def holds(ang, ratio):
    return ang is not None and ang < ANGLE_MAX and RATIO[0] <= ratio <= RATIO[1]


def segments(grip, t0):
    """Gripper command changes from t0 on: [[step, "close" | "open"], ...]."""
    out = []
    for t in range(t0, len(grip)):
        s = "close" if grip[t] > 0 else "open"
        if not out or out[-1][1] != s:
            out.append([t, s])
    return out


def episode(r, ref_xy, tau):
    c_obs, c_act, tobs, grip, p, con = replay(r)
    tc = r["t_close"]
    t_open = next(t for t in range(tc, len(grip)) if grip[t] <= 0)
    t_rel = next(t for t in range(tc + tau, len(grip)) if grip[t] <= 0)
    n_rel = int(tobs[t_rel])
    c_tc = c_act[tc]  # C after the close step's act; the reflex is silent after it
    assert np.allclose(c_act[t_rel], c_tc)
    e = (c_tc - c_obs[n_rel])[:2]
    err0, err5 = (p[t_rel, :2] - ref_xy), (p[t_rel + SETTLE, :2] - ref_xy)
    ang, ratio = angle_ratio(e, err5)
    z0 = p[0, 2]
    peak = tc + int(np.argmax(p[tc:t_rel + 1, 2]))
    t_down = next(t for t in range(peak, t_rel + 1) if p[t, 2] - p[t_rel, 2] <= DOWN)
    n_down = int(tobs[t_down])
    e_down = (c_tc - c_obs[n_down])[:2]
    ang_d, ratio_d = angle_ratio(e_down, err5)
    ang_c, ratio_c = angle_ratio(c_tc[:2], err5)  # report only: the whole carried offset, as if every plan were stale
    carry = sorted({int(n) for n in tobs[tc:t_rel + 1]})
    arrive = {a[0]: a[1] for a in r["plans_in"]}
    after = p[t_rel + SETTLE:]
    rnd = lambda v: np.round(v, 4).tolist()  # noqa: E731
    return {
        "suite": r["suite"], "task": r["task"], "init": r["init"], "mag_m": round(r["mag"], 4),
        "grasp_miss_m": round(r["grasp_miss_m"], 4), "steps": r["steps"], "success": r["success"],
        "t_fire": r["t_fire"], "t_engage": r["t_engage"], "t_c": tc, "t_open_first": t_open, "t_rel": t_rel,
        "carry_steps": t_rel - tc, "N_rel": n_rel, "law_acts": n_rel < tc,
        "C_tc": rnd(c_tc[:2]), "C_N_rel": rnd(c_obs[n_rel][:2]), "E": rnd(e),
        "E_norm": round(float(np.linalg.norm(e)), 4),
        "err_t_rel": rnd(err0), "err": rnd(err5), "err_norm": round(float(np.linalg.norm(err5)), 4),
        "angle_deg": None if ang is None else round(ang, 1), "ratio": None if ratio is None else round(ratio, 3),
        "holds": holds(ang, ratio),
        "vs_C_tc": {"angle_deg": None if ang_c is None else round(ang_c, 1),
                    "ratio": None if ratio_c is None else round(ratio_c, 3)},
        "z_rel_t_rel": round(float(p[t_rel, 2] - z0), 4), "contact_t_rel": bool(con[t_rel]),
        "closed_t_c_to_t_rel": bool((grip[tc:t_rel] > 0).all()),
        "lifted_in_carry": bool((p[tc:t_rel, 2] - z0 >= LIFT).any()), "z_peak_rel": round(float(p[peak, 2] - z0), 4),
        "carry_plans": [[n, arrive[n], n < tc] for n in carry],  # [t_obs, arrival, observed before t_c]
        "touchdown": {"t_down": t_down, "N_down": n_down, "stale": n_down < tc, "E": rnd(e_down),
                      "angle_deg": None if ang_d is None else round(ang_d, 1),
                      "ratio": None if ratio_d is None else round(ratio_d, 3)},
        "after_release": {"grip": segments(grip, t_rel), "err_end": rnd(p[-1, :2] - ref_xy),
                          "z_rel_end": round(float(p[-1, 2] - z0), 4),
                          "relifted": bool((after[:, 2] - after[0, 2] >= LIFT).any()),  # 3 cm over the settled height
                          "moved_after_settle_m": round(float(np.linalg.norm(after[-1, :2] - after[0, :2])), 4)},
    }


def main():
    tau, tau_src = tau_close()
    recs = [json.loads(line) for line in TRACES.open()]
    ref = {}
    for r in recs:
        if r["kind"] == "control":
            assert r["success"] and r["method"] == "none"
            replay(r)  # same reconstruction checks on the control arm
            ref[(r["suite"], r["task"], r["init"])] = (np.array(r["trace"]["p"][-1][:2]), r["steps_to_success"])
    eps = []
    for r in recs:
        if r["kind"] == "step":
            assert r["method"] == "G" and r["grasp_ok"]
            xy, n_ok = ref[(r["suite"], r["task"], r["init"])]
            eps.append(episode(r, xy, tau) | {"control_steps_to_success": n_ok,
                                              "control_final_xy": np.round(xy, 4).tolist()})
    verdict = "pilot" if len(eps) == 3 and all(e["holds"] for e in eps) else "no pilot"
    out = {"meta": {"script": "scripts/c1e4u_offline.py", "traces": str(TRACES.relative_to(ROOT)),
                    "python": platform.python_version(), "numpy": np.__version__, "cell": "A40", "s_d": [10, 40],
                    "tau_close": tau, "tau_close_source": tau_src, "settle_steps": SETTLE,
                    "criterion": f"angle(E, err) < {ANGLE_MAX:g} deg and |err|/|E| in [{RATIO[0]}, {RATIO[1]}] in all "
                                 "three episodes -> pilot",
                    "E": "C(t_c) - C(t_obs of the plan executing at t_rel), xy; C rebuilt by replaying G on the trace",
                    "err": f"object xy at t_rel + {SETTLE} minus the control's final object xy (same triple); the "
                           "controls succeed with the gripper still closed, so that is the object set on the target",
                    "touchdown": f"report only: first step after the carry peak within {DOWN * 100:g} cm of the height "
                                 "at t_rel; no rule reads it"},
           "episodes": eps, "n_law_acts": sum(e["law_acts"] for e in eps), "verdict": verdict}
    OUT.write_text(json.dumps(out, indent=1) + "\n")
    print(f"tau_close {tau} ({tau_src})")
    print("triple         t_fire t_c t_rel N_rel acts  E(cm)          err(cm)        ang   ratio | down N  stale "
          "ang   ratio | vs C(t_c)   | z@rel  contact")
    for e in eps:
        td = e["touchdown"]
        name = f"{e['suite'][7:]}:{e['task']} i{e['init']}"
        print(f"{name:<14} {e['t_fire']:>6} {e['t_c']:>3} {e['t_rel']:>5} "
              f"{e['N_rel']:>5} {str(e['law_acts']):5} {str([round(100 * v, 1) for v in e['E']]):14} "
              f"{str([round(100 * v, 1) for v in e['err']]):14} {str(e['angle_deg']):5} {str(e['ratio']):5} | "
              f"{td['t_down']:>4} {td['N_down']:>2} {str(td['stale']):5} {str(td['angle_deg']):5} "
              f"{str(td['ratio']):5} | "
              f"{e['vs_C_tc']['angle_deg']:5} {e['vs_C_tc']['ratio']:5} | "
              f"{100 * e['z_rel_t_rel']:4.1f}cm {e['contact_t_rel']}")
    print(f"law acts in {out['n_law_acts']}/3; verdict: {verdict}")


if __name__ == "__main__":
    main()
