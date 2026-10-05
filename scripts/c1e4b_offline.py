"""C1-E4 part 2, task 5 (Mac, no brain): the arm's lag behind a constant command, the free finger-closing time, the
noisy CV-Kalman eyes' (q_acc, gate window m) by the rule fixed in the plan before the data, and tau_lead.

(a) LiberoEnv libero_spatial task 0 init 0: a_x = v / G_POS for 120 steps (other axes 0, gripper -1); on steps 40-120
    fit x(t) ~ x0 + v_act (t - tau_arm) to the EEF of the observation (t = observation after t actions). The window
    ends where the commanded path passes the 4 cm/s run's 24 cm: 6 and 8 cm/s reach the arm's limit (x ~ 0.15 m)
    before step 120, so their window is 40-80 / 40-60 (the 40-120 fit is kept beside it).
(b) Same env, arm still: gripper -1 up to step 10, +1 from it; tau_close_free = close actions until the fingers stop
    (|d width| < 0.1 mm per step three steps in a row; width = |q0| + |q1| of robot0_gripper_qpos). Report only: the
    count is dominated by the creep tail near full closure.
(b2) With the object: scripted top-down grasps (oracle positions, raw delta-EEF clipped to +-1 via G_POS) on the bench
    tasks libero_spatial 0, libero_object 0, libero_goal 1 (goal 0 opens a drawer: no free object), Mac inits 48-49;
    open, 8 cm above the grasp point, down to it, 3 still steps, close from t0 with the arm still (same stop rule) until
    the stop is confirmed, then a 10-step +3 cm test lift. Held: width at the stop > 5 mm and the object rose >= 1.5 cm.
    tau_close = median over the held grasps (>= 4 of 6, rule fixed in the follow-up spec before the data).
(c) numpy only: lead.CVKalman at gauto.NOISE, grid q_acc x m; uniform motion 2 / 4 / 6 cm/s from step 12 and a step
    of (4.5, 2) cm at step 12, seeds 0..N-1, 100 steps; selection rule of the plan (task 5).
(c2) numpy only, after the Task 6 Mac check (a still object's CV estimate left G's 5 mm dead band on ~34% of the steps
    and PPC's 1 mm/step on ~83%): the dead bands of the @cv arms at gauto.NOISE and the chosen q_acc, 200 still-object
    seeds of 300 steps after the warm-up (BAND_RULE); for information the engagement delay they cost at 2 / 4 / 6 cm/s.
tau_lead (oracle) = 1 + tau_arm(4 cm/s) + tau_close (b2); tau_lead (noisy) = that + the estimate's lag at the chosen
    pair.
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
BAND_SEEDS, BAND_T, BAND_K, BAND_P = 200, 300, 30, 95  # seeds, steps, max lag k = s + d of A20, percentile
BAND_RULE = ("eps_cv = P95 over 200 still-object seeds of max_t max_{1<=k<=30} |p_hat(t) - p_hat(t-k)| (30 = s + d "
             "of A20, the noisy cell: <= 5% of the episodes with a false engagement); ppc_v_min_cv = P95 over the "
             "seeds of max_t |p_hat(t) - p_hat(t-1)|; gauto.NOISE, the chosen q_acc, 300 steps after the warm-up; the "
             "runner sets them on every @cv arm (eps, ppc_v_min), as C1-E3 set them per row")
DELAY_RULE = ("info only: steps from the motion start t0 = 5 + seed % 11 (the t_fire range) to the engagement "
              "|p_hat(t) - p_hat(t_obs)| > eps_cv, t_obs the plan executing at observe time on the A20 conveyor (call "
              "0 at once, call c >= 10 from c + 20 on); median over the seeds without a false engagement before t0, "
              "null when no engagement in 300 steps")
ENV = {"suite": "libero_spatial", "task": 0, "init": 0, "reset_seed": 0, "render": 64}
GRASP_INITS = (48, 49)  # Mac debug inits, never read for rules
# grasp point = object + EEF-object offset at the first close command, median over part 1's successful pi0.5 control
# episodes (results/c1-e4/pod/trial_p1.jsonl, inits 44-47): the bowls are taken by the rim (the 77 mm open gripper
# cannot span a bowl at its centre), the can near its centre
GRASP_AT = {("libero_spatial", 0): (0.0137, -0.0446, 0.0560), ("libero_object", 0): (-0.0100, -0.0073, 0.0054),
            ("libero_goal", 1): (-0.0171, 0.0327, 0.0252)}
ABOVE, TOL, CAP, GAIN, HOLD = 0.08, 0.002, 80, 0.5, 3  # m, m, steps per move, P gain on err / G_POS, still steps
CLOSE_OBJ_STEPS, LIFT, LIFT_STEPS = 100, 0.03, 10  # close-step cap (a bowl slides on: long creep after contact)
HELD_W, HELD_LIFT = 0.005, 0.5  # held: width at the stop > 5 mm and the object rose >= HELD_LIFT * LIFT
CLOSE_RULE = "median with object (free closing: creep tail; report only)"


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


# --- (c2) dead bands of the cv eyes on a still object, numpy --------------------------------------------------------
def still_maxima(q, seed):
    """One still-object run of the cv eyes: (max_t max_{1<=k<=BAND_K} |p_hat(t) - p_hat(t-k)|, max_t |p_hat(t) -
    p_hat(t-1)|), t over BAND_T steps after the warm-up."""
    kf = lead.CVKalman(seed, *gauto.NOISE, q)
    est = np.array([kf(P0) for _ in range(BAND_T)])
    win = max(float(np.linalg.norm(est[k:] - est[:-k], axis=1).max()) for k in range(1, BAND_K + 1))
    return win, float(np.linalg.norm(np.diff(est, axis=0), axis=1).max())


def engage_delay(q, eps, v, seed, s=10, d=20):
    """DELAY_RULE for one seed at v (m/s): the delay in steps, "false" (engaged before t0) or None (never)."""
    kf, t0 = lead.CVKalman(seed, *gauto.NOISE, q), 5 + seed % 11
    est = [kf(P0 + np.array([v / HZ * max(t - t0, 0), 0.0, 0.0])) for t in range(BAND_T)]
    for t in range(1, BAND_T):
        c = 0 if t <= s + d else (t - 1 - d) // s * s  # the latest call that arrived by t - 1
        if np.linalg.norm(est[t] - est[c]) > eps:
            return "false" if t < t0 else t - t0
    return None


def still_bands(q, seeds):
    win, vel = np.array([still_maxima(q, sd) for sd in seeds]).T
    eps, v_min = float(np.percentile(win, BAND_P)), float(np.percentile(vel, BAND_P))
    delay = {}
    for v in lead.SPEEDS:
        r = [engage_delay(q, eps, v, sd) for sd in seeds]
        ok = [np.inf if x is None else x for x in r if x != "false"]
        med = float(np.median(ok)) if ok else None
        delay[cms(v)] = {"median_steps": med if med is not None and math.isfinite(med) else None,
                         "never": r.count(None), "false_before_t0": r.count("false")}
    return {"eps_cv": eps, "ppc_v_min_cv": v_min, "rule": BAND_RULE, "q_acc": q, "seeds": len(seeds),
            "still_window_max": {"median": float(np.median(win)), "max": float(win.max())},
            "still_step_max": {"median": float(np.median(vel)), "max": float(vel.max())},
            "false_engage_share_at_eps": float((win > eps).mean()),
            "engage_delay_a20": delay, "delay_rule": DELAY_RULE}


# --- (a), (b) LiberoEnv ---------------------------------------------------------------------------------------------
def make_env(suite=ENV["suite"], task=ENV["task"]):
    from lerobot.envs.libero import LiberoEnv, _get_suite
    return LiberoEnv(task_suite=_get_suite(suite), task_id=task, task_suite_name=suite,
                     obs_type="pixels_agent_pos", observation_width=ENV["render"], observation_height=ENV["render"])


def width(obs):
    return float(np.abs(obs["robot_state"]["gripper"]["qpos"]).sum())


def stop(w):
    """w[j] = width after j close actions -> (first moving action index k0, tau = close actions until the fingers stop:
    |d width| < 0.1 mm per step three steps in a row, from k0 on)."""
    moving = np.abs(np.diff(w)) >= STILL
    assert moving.any(), "the fingers never moved"
    k0 = int(np.argmax(moving))
    return k0, next(i for i in range(k0, len(moving) - N_STILL + 1) if not moving[i: i + N_STILL].any())


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
    k0, tau = stop(w[CLOSE_AT:])
    frac = (w[CLOSE_AT] - w[CLOSE_AT:]) / (w[CLOSE_AT] - w[-1])  # share of the closing done after j close actions
    return {"tau_close_free": tau, "width_open_mm": float(1e3 * w[CLOSE_AT]), "width_closed_mm": float(1e3 * w[-1]),
            "width_at_tau_close_mm": float(1e3 * w[CLOSE_AT + tau]), "eef_z": float(st[CLOSE_AT][0][2]),
            "first_moving_action": k0 + 1, "actions_to_90pct": int(np.argmax(frac >= 0.9)),
            "actions_to_95pct": int(np.argmax(frac >= 0.95)),
            "note": "report only: free closing ends in a creep tail just under the 0.1 mm/step threshold; tau_close "
                    "is measured with the object (close_obj)"}


def grasp(env, suite, task, init):
    """Scripted top-down grasp on a fresh reset (init, seed 0); positions from the sim as the runner's ShiftEnv reads
    them (target = the first object of interest with a free joint, EEF site)."""
    env.init_state_id = init
    obs, _ = env.reset(seed=ENV["reset_seed"])
    e = env._env.env
    name = next(n for n in e.obj_of_interest if n in e.objects_dict)
    obj = lambda: np.array(e.sim.data.body_xpos[e.obj_body_id[name]])  # noqa: E731
    eef = lambda: np.array(e.sim.data.site_xpos[e.robots[0].eef_site_id])  # noqa: E731
    act = lambda xyz, g: env.step(np.r_[np.clip(xyz, -1, 1), 0.0, 0.0, 0.0, g])[0]  # noqa: E731
    p0 = obj()
    goal = p0 + GRASP_AT[(suite, task)]
    moves = []
    for target in (goal + [0.0, 0.0, ABOVE], goal):  # open gripper; above the grasp point, then straight down
        n = 0
        while n < CAP and np.linalg.norm(target - eef()) >= TOL:
            act(GAIN * (target - eef()) / orx.G_POS, -1.0)
            n += 1
        moves.append(n)
    err = 1e3 * np.linalg.norm(goal - eef())
    for _ in range(HOLD):
        obs = act(np.zeros(3), -1.0)
    p_t0 = obj()
    w = [width(obs)]  # w[j]: width after j close actions, the arm still
    while len(w) <= CLOSE_OBJ_STEPS and not (len(w) > N_STILL + 1
                                            and (np.abs(np.diff(w[-N_STILL - 1:])) < STILL).all()):
        w.append(width(act(np.zeros(3), 1.0)))  # until the stop is confirmed: a long squeeze lets the rim slip out
    k0, tau = stop(np.array(w))
    z_eef, z_obj, p_closed = eef()[2], obj()[2], obj()
    for _ in range(LIFT_STEPS):
        act([0.0, 0.0, LIFT / LIFT_STEPS / orx.G_POS], 1.0)
    lift, lift_eef = obj()[2] - z_obj, eef()[2] - z_eef
    return {"suite": suite, "task": task, "init": init, "object": name, "tau_close_obj": tau,
            "width_at_stop_mm": 1e3 * w[tau], "held": bool(w[tau] > HELD_W and lift >= HELD_LIFT * LIFT),
            "lift_obj_mm": 1e3 * lift, "lift_eef_mm": 1e3 * lift_eef, "first_moving_action": k0 + 1,
            "width_open_mm": 1e3 * w[0], "grasp_point_err_mm": err, "move_steps": moves,
            "obj_moved_before_close_mm": 1e3 * np.linalg.norm(p_t0 - p0),
            "obj_moved_while_closing_mm": 1e3 * np.linalg.norm(p_closed - p_t0)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seeds", type=int, default=40, help="seeds 0..N-1 for the motion and for the step")
    p.add_argument("--no-env", action="store_true", help="skip (a), (b), (b2) and tau_lead (check of (c) only)")
    p.add_argument("--out", default="results/c1-e4/part2_offline.json", help="relative to the repo root")
    args = p.parse_args()
    seeds = list(range(args.seeds))
    tb = time.time()
    rows = [r for q in GRID_Q for r in sweep(q, GRID_M, seeds)]
    pick = select(rows)
    oracle = sweep(1e-8, (lead.GATE_M,), [0], oracle=True)[0]
    bands = still_bands(pick["q_acc"], range(BAND_SEEDS))
    t_cv = time.time() - tb
    ref = next(r for r in rows if r["q_acc"] == REF["q_acc"] and r["m"] == REF["m"])
    print("q_acc    m   open 2/4/6       false   lag(4)  v_err(4)")
    for r in rows:
        print(f"{r['q_acc']:.0e} {r['m']:3d}   {r['open']['2']:.2f}/{r['open']['4']:.2f}/{r['open']['6']:.2f}   "
              f"{r['false_open']:.4f}  {r['lag_steps']['4']:6.2f}  {r['v_rel_err']['4']:.3f}"
              + ("   <- chosen" if r is pick else ""))
    assert pick["false_open"] <= FALSE_MAX
    print(f"cv bands: eps_cv {1e3 * bands['eps_cv']:.2f} mm, ppc_v_min_cv {1e3 * bands['ppc_v_min_cv']:.3f} mm/step; "
          "engagement delay A20 (median steps) " + ", ".join(
              f"{v} cm/s {r['median_steps']} (never {r['never']}, false {r['false_before_t0']})"
              for v, r in bands["engage_delay_a20"].items()))
    out = {"meta": {"script": "scripts/c1e4b_offline.py", "args": vars(args), "python": platform.python_version(),
                    "numpy": np.__version__, "env": ENV, "cv_seeds": f"0..{args.seeds - 1} (motion and step)",
                    "noise": dict(zip(("sigma", "lag", "q_drop"), gauto.NOISE)), "G_POS": orx.G_POS, "hz": HZ},
           "cv": {"grid": rows, "rule": RULE, "chosen": {"q_acc": pick["q_acc"], "m": pick["m"]},
                  "oracle": oracle, "reference_check": {"plan": REF, "ours": ref}, "bands": bands,
                  "seconds": round(t_cv, 1)}}
    if not args.no_env:
        import mujoco
        out["meta"]["mujoco"] = mujoco.__version__
        tb = time.time()
        env = make_env()
        arm = {cms(v): arm_lag(env, v) for v in ARM_SPEEDS}
        close = close_time(env)
        env.close()
        tg = time.time()
        grasps = []
        for st, tk in GRASP_AT:
            env = make_env(st, tk)
            grasps += [grasp(env, st, tk, i) for i in GRASP_INITS]
            env.close()
        t_grasp = time.time() - tg
        for v, r in arm.items():
            print(f"arm {v} cm/s: tau_arm {r['tau_arm']:.2f}  v_act/v {r['v_act_over_v']:.3f}  "
                  f"resid {r['fit_max_resid_mm']:.2f} mm  fit {r['fit_steps']}  x {r['x0']:.3f}->{r['x_end']:.3f}")
        print(f"tau_close_free {close['tau_close_free']} (90% {close['actions_to_90pct']}, "
              f"95% {close['actions_to_95pct']}), report only")
        for g in grasps:
            print(f"{g['suite']}:{g['task']} init {g['init']} {g['object']}: tau_close_obj {g['tau_close_obj']}  "
                  f"width {g['width_at_stop_mm']:.1f} mm  held {g['held']}  lift {g['lift_obj_mm']:.1f} / eef "
                  f"{g['lift_eef_mm']:.1f} mm  err {g['grasp_point_err_mm']:.1f} mm  "
                  f"moved {g['obj_moved_before_close_mm']:.1f} / {g['obj_moved_while_closing_mm']:.1f} mm")
        held = [g["tau_close_obj"] for g in grasps if g["held"]]
        assert len(held) >= 4, f"only {len(held)} of {len(grasps)} grasps held: report and stop"
        med = float(np.median(held))
        t_close = math.floor(med + 0.5)  # half up
        assert 1 <= t_close <= 16, t_close
        a4 = arm["4"]
        assert 0.5 <= a4["v_act_over_v"] <= 1.5 and math.isfinite(a4["tau_arm"]) and a4["tau_arm"] >= 0, a4
        t_arm, lag = round(a4["tau_arm"]), round(pick["lag_steps"]["4"])
        t_or = 1 + t_arm + t_close
        out["arm"] = arm
        out["close"] = close
        out["close_obj"] = {"grasps": grasps, "n_held": len(held), "tau_close_median": med, "tau_close": t_close,
                            "tau_close_rule": CLOSE_RULE, "grasp_at": {f"{s}:{t}": v for (s, t), v in GRASP_AT.items()},
                            "held_rule": f"width at the stop > {1e3 * HELD_W:g} mm and the object rose >= "
                                         f"{1e3 * HELD_LIFT * LIFT:g} mm on the {LIFT_STEPS}-step +{1e2 * LIFT:g} cm "
                                         "lift that starts once the stop is confirmed",
                            "inits": list(GRASP_INITS), "seconds": round(t_grasp, 1)}
        out["tau_lead"] = {"oracle": t_or, "noisy": t_or + lag, "parts": {"G_step": 1, "tau_arm_4cms": t_arm,
                           "tau_close": t_close, "estimate_lag_4cms": lag}, "tau_close_rule": CLOSE_RULE,
                           "note": "each part rounded to a step (tau_close half up), then summed"}
        out["meta"]["env_seconds"] = round(time.time() - tb, 1)
        print(f"tau_close {t_close} (median {med:g} over {len(held)} held)  tau_lead oracle {t_or}  noisy {t_or + lag}")
    out = json.loads(json.dumps(out), parse_float=lambda s: float(f"{float(s):.4g}"))  # 4 significant digits
    (ROOT / args.out).write_text(json.dumps(out, indent=1) + "\n")
    print("wrote", args.out)


if __name__ == "__main__":
    main()
