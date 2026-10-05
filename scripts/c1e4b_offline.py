"""C1-E4 part 2, task 5 (Mac, no brain): the arm's lag behind a constant command, the free finger-closing time, the
noisy CV-Kalman eyes' (q_acc, gate window m) by the rule fixed in the plan before the data, and tau_lead.

(a) LiberoEnv libero_spatial task 0 init 0: a_x = v / G_POS for 120 steps (other axes 0, gripper -1); on steps 40-120
    fit x(t) ~ x0 + v_act (t - tau_arm) to the EEF of the observation (t = observation after t actions). The window
    ends where the commanded path passes the 4 cm/s run's 24 cm: 6 and 8 cm/s reach the arm's limit (x ~ 0.15 m)
    before step 120, so their window is 40-80 / 40-60 (the 40-120 fit is kept beside it).
(b) Same env, arm still: gripper -1 up to step 10, +1 from it; tau_close = close actions until the fingers stop
    (|d width| < 0.1 mm per step three steps in a row; width = |q0| + |q1| of robot0_gripper_qpos).
(c) numpy only: lead.CVKalman at gauto.NOISE, grid q_acc x m; uniform motion 2 / 4 / 6 cm/s from step 12 and a step
    of (4.5, 2) cm at step 12, seeds 0..N-1, 100 steps; selection rule of the plan (task 5).
tau_lead (oracle) = 1 + tau_arm(4 cm/s) + tau_close; tau_lead (noisy) = that + the estimate's lag at the chosen pair.
    MUJOCO_GL=cgl HF_HUB_OFFLINE=1 nice -n 10 ~/Desktop/M2R-c1-env/bin/python scripts/c1e4b_offline.py
"""

import argparse
import json
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import gauto  # noqa: E402
import lead  # noqa: E402
import objreflex as orx  # noqa: E402

HZ = lead.HZ
ARM_SPEEDS = (0.01, 0.02, 0.04, 0.06, 0.08)  # m/s
ARM_STEPS, FIT_FROM = 120, 40
TRAVEL = 0.04 / HZ * ARM_STEPS  # m, the 4 cm/s run's commanded path; past it 6 and 8 cm/s hit the reach limit (x ~ 0.15)
CLOSE_AT, CLOSE_STEPS, STILL, N_STILL = 10, 60, 1e-4, 3
GRID_Q, GRID_M = (1e-8, 1e-7, 1e-6), (6, 10, 16, 20)
CV_T, CV_T0, OPEN_FROM = 100, 12, 40
P0, STEP = np.array([0.1, 0.0, 0.0]), np.array([0.045, 0.02, 0.0])
FALSE_MAX, TIE = 0.01, 0.02
RULE = ("among pairs with false-open share on the step <= 1%: max open share at 4 cm/s; ties (+-0.02) -> smaller m "
        "(then the larger open share); oracle fixed q_acc 1e-8, m 6")
REF = {"q_acc": 1e-8, "m": 20, "open": {"2": 0.18, "4": 0.59, "6": 0.79}, "false_open": 0.002,
       "source": "plan, cvsim.py 2026-10-05: 20 motion seeds, 40 step seeds"}
ENV = {"suite": "libero_spatial", "task": 0, "init": 0, "reset_seed": 0, "render": 64}


def cms(v):
    return str(round(v * 100))


# --- (c) noisy eyes, numpy ------------------------------------------------------------------------------------------
def track(kf, path, ms, oracle=False):
    """One run of the filter over CV_T steps; gates for every window m. Seen = the eyes' output (oracle: the truth)."""
    seen, gates, est, vel = [], {m: [] for m in ms}, [], []
    for t in range(CV_T):
        p = path(t)
        e = kf(p)
        seen.append(p if oracle else e)
        for m in ms:
            gates[m].append(lead.lead_gate(kf, seen, m))
        est.append(e)
        vel.append(kf.v)
    return {m: np.array(g) for m, g in gates.items()}, np.array(est), np.array(vel)


def move(vs):
    return lambda t: P0 + np.array([vs * max(t - CV_T0, 0), 0.0, 0.0])


def jump(t):
    return P0 + (STEP if t >= CV_T0 else 0.0)


def sweep(q, ms, seeds, oracle=False):
    """Rows per m: open share on the motion (steps >= OPEN_FROM), false-open share on the step (all steps), lag of the
    position estimate along the motion (steps) and the velocity error |v_hat_xy - v| / |v|, v_hat_x / v."""
    make = (lambda s: lead.CVKalman.oracle(q)) if oracle else (lambda s: lead.CVKalman(s, *gauto.NOISE, q))
    rows = {m: {"q_acc": q, "m": m, "open": {}, "lag_steps": {}, "v_rel_err": {}, "v_ratio": {}} for m in ms}
    for v in lead.SPEEDS:
        vs = v / HZ
        g, lag, err, ratio = {m: [] for m in ms}, [], [], []
        for s in seeds:
            gates, est, vel = track(make(s), move(vs), ms, oracle)
            for m in ms:
                g[m].append(gates[m][OPEN_FROM:].mean())
            t = np.arange(OPEN_FROM, CV_T)
            lag.append(((P0[0] + vs * (t - CV_T0) - est[t, 0]) / vs).mean())
            err.append((np.linalg.norm(vel[t, :2] - [vs, 0.0], axis=1) / vs).mean())
            ratio.append((vel[t, 0] / vs).mean())
        for m in ms:
            r = rows[m]
            r["open"][cms(v)] = float(np.mean(g[m]))
            r["lag_steps"][cms(v)], r["v_rel_err"][cms(v)] = float(np.mean(lag)), float(np.mean(err))
            r["v_ratio"][cms(v)] = float(np.mean(ratio))
    fo = {m: [] for m in ms}
    for s in seeds:
        gates, _, _ = track(make(s), jump, ms, oracle)
        for m in ms:
            fo[m].append(gates[m].mean())
    for m in ms:
        rows[m]["false_open"] = float(np.mean(fo[m]))
    return [rows[m] for m in ms]


def select(rows):
    ok = [r for r in rows if r["false_open"] <= FALSE_MAX]
    assert ok, "no (q_acc, m) pair passes the false-open rule"
    best = max(r["open"]["4"] for r in ok)
    return min((r for r in ok if r["open"]["4"] >= best - TIE), key=lambda r: (r["m"], -r["open"]["4"]))


# --- (a), (b) LiberoEnv ---------------------------------------------------------------------------------------------
def make_env():
    from lerobot.envs.libero import LiberoEnv, _get_suite
    return LiberoEnv(task_suite=_get_suite(ENV["suite"]), task_id=ENV["task"], task_suite_name=ENV["suite"],
                     obs_type="pixels_agent_pos", observation_width=ENV["render"], observation_height=ENV["render"])


def rollout(env, actions):
    """Fresh reset (init 0, seed 0), then the actions; the robot state of every observation (len(actions) + 1)."""
    env.init_state_id = ENV["init"]
    obs, _ = env.reset(seed=ENV["reset_seed"])
    out = []
    for a in [None, *actions]:
        if a is not None:
            obs, *_ = env.step(np.asarray(a, float))
        rs = obs["robot_state"]
        out.append((np.array(rs["eef"]["pos"], float), np.array(rs["gripper"]["qpos"], float)))
    return out


def fit(x, t, v):
    """x(t) ~ x(0) + v_act (t - tau_arm) on the steps t."""
    b, c = np.polyfit(t, x[t], 1)
    return {"v_act_over_v": float(b / (v / HZ)), "tau_arm": float((x[0] - c) / b),
            "fit_max_resid_mm": float(1e3 * np.abs(x[t] - (b * t + c)).max())}


def arm_lag(env, v):
    """Fit on steps 40-120, cut to the commanded path TRAVEL (only 6 and 8 cm/s are cut; the spec window kept too)."""
    a = np.zeros(7)
    a[0], a[6] = v / HZ / orx.G_POS, -1.0
    eef = np.array([e for e, _ in rollout(env, [a] * ARM_STEPS)])
    end = min(ARM_STEPS, round(TRAVEL / (v / HZ)))
    r = {"a_x": float(a[0]), "fit_steps": [FIT_FROM, end], **fit(eef[:, 0], np.arange(FIT_FROM, end + 1), v),
         "yz_drift_max_mm": float(1e3 * np.linalg.norm(eef[:, 1:] - eef[0, 1:], axis=1).max()),
         "x0": float(eef[0, 0]), "x_end": float(eef[-1, 0])}
    if end < ARM_STEPS:
        r["spec_window_40_120"] = fit(eef[:, 0], np.arange(FIT_FROM, ARM_STEPS + 1), v)
    return r


def close_time(env):
    acts = [np.r_[np.zeros(6), -1.0 if t < CLOSE_AT else 1.0] for t in range(CLOSE_STEPS)]
    st = rollout(env, acts)
    w = np.array([np.abs(g).sum() for _, g in st])
    d = np.abs(np.diff(w[CLOSE_AT:]))  # d[i]: the change made by close action i + 1
    moving = d >= STILL
    assert moving.any(), "the fingers never moved"
    k0 = int(np.argmax(moving))
    tau = next(i for i in range(k0, len(d) - N_STILL + 1) if not moving[i: i + N_STILL].any())
    frac = (w[CLOSE_AT] - w[CLOSE_AT:]) / (w[CLOSE_AT] - w[-1])  # share of the closing done after j close actions
    return {"tau_close": tau, "width_open_mm": float(1e3 * w[CLOSE_AT]), "width_closed_mm": float(1e3 * w[-1]),
            "width_at_tau_close_mm": float(1e3 * w[CLOSE_AT + tau]), "eef_z": float(st[CLOSE_AT][0][2]),
            "first_moving_action": k0 + 1, "actions_to_90pct": int(np.argmax(frac >= 0.9)),
            "actions_to_95pct": int(np.argmax(frac >= 0.95)), "note_pct": "report only, no rule reads them"}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=40, help="seeds 0..N-1 for the motion and for the step")
    p.add_argument("--no-env", action="store_true", help="skip (a), (b) and tau_lead (check of (c) only)")
    p.add_argument("--out", default="results/c1-e4/part2_offline.json", help="relative to the repo root")
    args = p.parse_args()
    seeds = list(range(args.seeds))
    tb = time.time()
    rows = [r for q in GRID_Q for r in sweep(q, GRID_M, seeds)]
    pick = select(rows)
    oracle = sweep(1e-8, (lead.GATE_M,), [0], oracle=True)[0]
    t_cv = time.time() - tb
    ref = next(r for r in rows if r["q_acc"] == REF["q_acc"] and r["m"] == REF["m"])
    print("q_acc    m   open 2/4/6       false   lag(4)  v_err(4)")
    for r in rows:
        print(f"{r['q_acc']:.0e} {r['m']:3d}   {r['open']['2']:.2f}/{r['open']['4']:.2f}/{r['open']['6']:.2f}   "
              f"{r['false_open']:.4f}  {r['lag_steps']['4']:6.2f}  {r['v_rel_err']['4']:.3f}"
              + ("   <- chosen" if r is pick else ""))
    assert pick["false_open"] <= FALSE_MAX
    out = {"meta": {"script": "scripts/c1e4b_offline.py", "args": vars(args), "python": platform.python_version(),
                    "numpy": np.__version__, "env": ENV, "cv_seeds": f"0..{args.seeds - 1} (motion and step)",
                    "noise": dict(zip(("sigma", "lag", "q_drop"), gauto.NOISE)), "G_POS": orx.G_POS, "hz": HZ},
           "cv": {"grid": rows, "rule": RULE, "chosen": {"q_acc": pick["q_acc"], "m": pick["m"]},
                  "oracle": oracle, "reference_check": {"plan": REF, "ours": ref}, "seconds": round(t_cv, 1)}}
    if not args.no_env:
        import mujoco
        out["meta"]["mujoco"] = mujoco.__version__
        tb = time.time()
        env = make_env()
        arm = {cms(v): arm_lag(env, v) for v in ARM_SPEEDS}
        close = close_time(env)
        env.close()
        a4 = arm["4"]
        assert 0.5 <= a4["v_act_over_v"] <= 1.5 and math.isfinite(a4["tau_arm"]) and a4["tau_arm"] >= 0, a4
        t_arm, lag = round(a4["tau_arm"]), round(pick["lag_steps"]["4"])
        t_or = 1 + t_arm + close["tau_close"]
        out["arm"] = arm
        out["close"] = close
        out["tau_lead"] = {"oracle": t_or, "noisy": t_or + lag, "parts": {"G_step": 1, "tau_arm_4cms": t_arm,
                           "tau_close": close["tau_close"], "estimate_lag_4cms": lag},
                           "note": "each part rounded to a step, then summed"}
        out["meta"]["env_seconds"] = round(time.time() - tb, 1)
        for v, r in arm.items():
            print(f"arm {v} cm/s: tau_arm {r['tau_arm']:.2f}  v_act/v {r['v_act_over_v']:.3f}  "
                  f"resid {r['fit_max_resid_mm']:.2f} mm  fit {r['fit_steps']}  x {r['x0']:.3f}->{r['x_end']:.3f}")
        print(f"tau_close {close['tau_close']} (90% {close['actions_to_90pct']}, 95% {close['actions_to_95pct']})  tau_lead oracle {t_or}  noisy {t_or + lag}")
    out = json.loads(json.dumps(out), parse_float=lambda s: float(f"{float(s):.4g}"))  # 4 significant digits
    (ROOT / args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
