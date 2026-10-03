"""C1-E4 runner: LIBERO + a frozen VLA (pi0.5 or SmolVLA) with the reflexes of src/gauto.py on the call conveyor (spec
docs/superpowers/specs/2026-10-03-c1-e4-design.md). Part 1 only (our bench); LIBERO-MAX is part 2.

Runs in the separate PyTorch/LeRobot env. One JSON line per episode. --methods takes arm labels (gauto.parse_arm):
none, T0, PPC, G, GT, Gkeep, GR, Gk0, Gk0T, Gauto, GautoT, Gcal (G + the shadow paired call at the engagement), and
G@glr / G@ema (noisy eyes). Example (Mac, debug only, never read):
    MUJOCO_GL=cgl ~/Desktop/M2R-c1-env/bin/python scripts/c1e4_run.py --brain smolvla --tasks libero_spatial:0 \
        --cells A,A40 --methods none,Gauto,Gcal --kbar 0.04 --inits 48-49 --n-envs 2 --vector sync --device mps --out x.jsonl
Pod: MUJOCO_GL=egl ... --vector async --device cuda (scripts/gpu_c1e4.sh)
"""

import argparse
import hashlib
import json
import os
import sys
import time
import traceback
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gauto  # noqa: E402
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoCfg  # noqa: E402
from lerobot.envs.libero import LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--brain", choices=sorted(gauto.BRAINS), required=True)
p.add_argument("--precision", choices=["fp32", "bf16-row"], default="fp32",
               help="spec §4.1: fp32 batched, or (pi0.5 only) bf16 with one row per brain call")
p.add_argument("--tasks", default="all", help="all (the 26 of orx.TASKS) or suite:task[,suite:task]")
p.add_argument("--cells", default="A", help="of gauto.CELLS")
p.add_argument("--methods", default="none", help="arm labels, e.g. none,G,Gauto,G@glr")
p.add_argument("--kinds", default="step", help="step, control")
p.add_argument("--inits", default="0-9")
p.add_argument("--classes", default="0,1,2", help="shift classes kept (step); class 5-6 cm only: 2")
p.add_argument("--t-ramp", type=int, default=5)
p.add_argument("--k-p", type=float, default=1.0)
p.add_argument("--tau-k", type=float, default=0.01, help="G-R dead band, m (the trial's P95 jitter, spec §7)")
p.add_argument("--kbar", type=float, help="G-auto's frozen kbar_30 for this brain (the trial, spec §5.3)")
p.add_argument("--shadow-pseudo", action="store_true",
               help="trial row P1: the shadow pair at a control episode's pseudo-shift (spec §7 item 5)")
p.add_argument("--trace", action="store_true", help="per-step positions, contacts and plans (trial row P1)")
p.add_argument("--repeat-check", action="store_true", help="repeat the first brain call and print whether it matches")
p.add_argument("--n-envs", type=int, default=6)
p.add_argument("--vector", choices=["async", "sync"], default="async")
p.add_argument("--device", default="cuda")
p.add_argument("--out", required=True, help="JSON lines")
args = p.parse_args()
REPO, REV = gauto.BRAINS[args.brain]
KAPPA0 = gauto.KAPPA0[args.brain]
ARMS = args.methods.split(",")
METHODS = {gauto.parse_arm(a)[0] for a in ARMS}
assert args.precision == "fp32" or args.brain == "pi05", "bf16-row is pi0.5's fallback only (spec §4.1)"
assert args.kbar is not None or not METHODS & {"Gauto", "GautoT"}, "G-auto needs --kbar"
assert not args.shadow_pseudo or args.kinds == "control", "--shadow-pseudo runs on control episodes"
CONF = {"brain": args.brain, "revision": REV, "precision": args.precision, "t_ramp": args.t_ramp, "k_p": args.k_p,
        "tau_k": args.tau_k, "kbar": args.kbar, "kappa0": KAPPA0, "shadow_pseudo": args.shadow_pseudo}
dev = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = False  # literal fp32 on CUDA (spec §4.1)
torch.backends.cudnn.allow_tf32 = False


def span(txt):
    a, _, b = txt.partition("-")
    return list(range(int(a), int(b or a) + 1))


tasks = orx.TASKS if args.tasks == "all" else [(t.split(":")[0], int(t.split(":")[1])) for t in args.tasks.split(",")]
done = set()  # resume: skip episodes already written
if Path(args.out).exists():
    text = Path(args.out).read_text()
    if text and not text.endswith("\n"):  # a half-written last line: cut it, or the next record is glued onto it
        print(f"warning: dropping a half-written last line of {args.out}: {text[text.rfind(chr(10)) + 1:][:80]!r}", flush=True)
        text = text[: text.rfind("\n") + 1]
        os.truncate(args.out, len(text.encode()))
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: skipping a malformed line in {args.out}: {line[:80]!r}", flush=True)
            continue
        assert r["conf"] == CONF, f"{args.out} was written with another configuration: {r['conf']} vs {CONF}"
        done.add((r["suite"], r["task"], r["init"], r["kind"], r["cell"], r["arm"]))


class ShiftEnv(LiberoEnv):
    """LiberoEnv plus a pending shift of the target object applied before the next physics step, and oracle data in
    info: object of interest and EEF positions, robot-object contact (as scripts/c1e3_run.py, without the box and the
    placement target). The target is the first object of interest with a free joint. The MjSim is re-read on every call
    and the contact masks on every reset: a hard reset rebuilds the sim."""

    pending_dq = None

    def _target(self):
        e = self._env.env
        name = next(n for n in e.obj_of_interest if n in e.objects_dict)
        jnt = e.objects_dict[name].joints[0]
        return name, e.sim.model.get_joint_qpos_addr(jnt)[0], e.sim.model.get_joint_qvel_addr(jnt)[0]

    def _setup(self):
        e = self._env.env
        m, root = e.sim.model._model, e.obj_body_id[self._target()[0]]
        bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) or "" for b in range(m.nbody)]

        def under(b):  # body b is the object's root or one of its descendants
            while b > 0:
                if b == root:
                    return True
                b = m.body_parentid[b]
            return False

        self._robot = np.array([bodies[b].startswith(("robot0_", "gripper0_")) for b in m.geom_bodyid])
        self._obj = np.array([under(b) for b in m.geom_bodyid])
        assert self._robot.any() and self._obj.any(), "no robot or object geoms"

    def _oracle(self, info):
        e = self._env.env
        d = e.sim.data
        info["obj_pos"] = np.array(d.body_xpos[e.obj_body_id[self._target()[0]]])
        info["eef_pos"] = np.array(d.site_xpos[e.robots[0].eef_site_id])
        # MuJoCo resets the state silently on a bad acceleration; the counter lives since the (hard) reset
        info["bad_qacc"] = int(d._data.warning[mujoco.mjtWarning.mjWARN_BADQACC].number)
        c = d._data.contact
        g1, g2 = np.asarray(c.geom1, int), np.asarray(c.geom2, int)
        info["contact"] = bool(((self._robot[g1] & self._obj[g2]) | (self._robot[g2] & self._obj[g1])).any())
        return info

    def reset(self, seed=None, **kw):
        obs, info = super().reset(seed=seed, **kw)
        self._setup()
        return obs, self._oracle(info)

    def step(self, action):
        if self.pending_dq is not None:
            e = self._env.env
            _, qa, va = self._target()
            e.sim.data.qpos[qa: qa + 3] += self.pending_dq
            e.sim.data.qvel[va: va + 6] = 0.0
            e.sim.forward()
            self.pending_dq = None
        obs, r, _, trunc, info = super().step(action)
        # never report termination: the runner ends episodes itself (is_success, step limit), and a terminated env
        # trips the DISABLED-autoreset assert of SyncVectorEnv on the next step of the batch
        return obs, r, False, trunc, self._oracle(info)


def make_vec(suite, task):
    fns = [lambda: ShiftEnv(task_suite=_get_suite(suite), task_id=task, task_suite_name=suite,
                            obs_type="pixels_agent_pos", observation_width=360, observation_height=360)
           for _ in range(args.n_envs)]
    kw = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
    # fork: the env factories are closures in __main__, and spawn/forkserver would re-run this script per worker
    return gym.vector.AsyncVectorEnv(fns, context="fork", **kw) if args.vector == "async" else gym.vector.SyncVectorEnv(fns, **kw)


def rows(x, idx):
    """The rows idx of a (nested) batched observation, as copies."""
    return {k: rows(v, idx) for k, v in x.items()} if isinstance(x, dict) else x[idx]


def cat(xs):
    """(Nested) observation rows stacked along the batch axis."""
    return {k: cat([x[k] for x in xs]) for k in xs[0]} if isinstance(xs[0], dict) else np.concatenate(xs)


def brain(obs, desc, noises):
    """Raw-unit chunks (B, H, 7) for B observations with their task descriptions, each with its own noise. bf16-row:
    one row per call, so a chunk never depends on the batch (spec §4.1)."""
    if args.precision == "bf16-row" and len(desc) > 1:
        return np.concatenate([brain(rows(obs, [j]), [desc[j]], [noises[j]]) for j in range(len(desc))])
    b = preprocess_observation(obs)
    b["task"] = list(desc)
    with torch.no_grad():
        ch = policy.predict_action_chunk(pre(env_pre(b)), noise=torch.cat(noises).to(dev))
    return post(ch).float().cpu().numpy()


def make_agent(arm, cell, ep):
    method, noise = gauto.parse_arm(arm)
    kw = {"t_ramp": args.t_ramp, "k_p": args.k_p, "tau_k": args.tau_k, "K": gauto.K}
    if noise == "glr":
        kw["tracker"] = gauto.GLRKalman(ep.seed, *gauto.NOISE)
    elif noise == "ema":
        kw |= {"tracker": hf.Tracker(ep.seed, *gauto.NOISE, gauto.EMA_BETA), "eps": gauto.EMA_EPS}
    kbar = KAPPA0 if method in ("Gk0", "Gk0T") else args.kbar if method in ("Gauto", "GautoT") else None
    return gauto.GAgent(method, *gauto.CELLS[cell], kbar=kbar, **kw)


repeat_done = False


def run_batch(vec, eps):
    """Run up to n_envs episodes (same task, same cell and kind) to the end; one record per real episode."""
    global repeat_done
    n = args.n_envs
    pad = eps + [eps[-1]] * (n - len(eps))  # padding envs are run and ignored
    vec.set_attr("init_state_id", [e["ep"].init for e in pad])
    # one reset seed for every slot: LIBERO seeds numpy with it and places the fixtures (cabinet, stove, rack) from it
    obs, info = vec.reset(seed=[0] * n)
    desc = vec.call("task_description")
    agents = [make_agent(e["arm"], e["cell"], e["ep"]) for e in pad]
    perts = [hf.Perturbation(e["ep"]) for e in pad]
    gens = [torch.Generator().manual_seed(e["ep"].seed) for e in pad]
    # spec §5.3, §7: the shadow pair at G-auto's calibration engagement, or at a control episode's pseudo-shift (P1)
    shadow_on = [args.shadow_pseudo or agents[i].method == "Gcal" for i in range(n)]
    t_max = vec.get_attr("_max_episode_steps")[0]
    live, succ = np.ones(n, bool), np.zeros(n, bool)
    live[len(eps):] = False  # padding envs only step no-ops: no brain calls
    steps_ok, bad = [None] * n, info["bad_qacc"].copy()
    xs, acts, objs, cons, tobs, plans, got = ([[] for _ in range(n)] for _ in range(7))
    cache, shadow = [None] * n, [None] * n  # the observation rows at t_fire; the shadow pair's log entry

    def installed(i, t):
        agents[i].on_new_chunk()
        got[i].append([agents[i].sched.t_obs, t])

    for t in range(t_max):
        eef, obj, con = info["eef_pos"], info["obj_pos"], info["contact"]
        dqs = [None] * n
        for i in range(n):
            if not live[i]:
                continue
            ag, pt = agents[i], perts[i]
            dqs[i] = pt.dq(t, eef[i], obj[i], ag.grasp_started)
            if shadow_on[i] and pt.t_fire == t:
                cache[i] = rows(obs, [i])  # pre-shift: the shift moves the object before this step's physics
            tobs[i].append(-1 if ag.sched.t_obs is None else ag.sched.t_obs)
            ag.observe(t, eef[i], obj[i], bool(con[i]))
            if ag.sched.arrive(t):
                installed(i, t)
        due = [i for i in range(n) if live[i] and cache[i] is not None and shadow[i] is None
               and (t == perts[i].t_fire + 1 if args.shadow_pseudo else agents[i].t_e == t)]
        if due:  # two calls with one noise: the cached pre-shift observation and the current one; both plans dropped
            z = [torch.randn(1, orx.H, cfg.max_action_dim, generator=torch.Generator().manual_seed(pad[i]["ep"].seed + 1))
                 for i in due]
            two = brain(cat([cache[i] for i in due] + [rows(obs, [i]) for i in due]), [desc[i] for i in due] * 2, z + z)
            for j, i in enumerate(due):
                ag, tf = agents[i], perts[i].t_fire
                shadow[i] = gauto.shadow_sample((tf, ag.x_hist[tf], two[j]), (t, ag.x_hist[t], two[len(due) + j]),
                                                ag.p_hist[tf], ag.p_hist[t])
        want = [i for i in range(n) if live[i] and agents[i].sched.wants_call(t, agents[i].trigger(t))]
        if want:
            nz = [torch.randn(1, orx.H, cfg.max_action_dim, generator=gens[i]) for i in want]
            chunks = brain(rows(obs, want), [desc[i] for i in want], nz)
            if args.repeat_check and not repeat_done:  # spec §4.1: the same call again must give the same chunk
                again = brain(rows(obs, want), [desc[i] for i in want], nz)
                print(f"C1-E4 repeat: rows={len(want)} max_abs={np.abs(again - chunks).max():.3e} "
                      f"bitwise={int(np.array_equal(again, chunks))}", flush=True)
                repeat_done = True
            for j, i in enumerate(want):
                ag = agents[i]
                if args.trace:
                    plans[i].append([t, np.round(eef[i], 5).tolist(), np.round(chunks[j][:, [0, 1, 2, 6]], 4).tolist()])
                ag.sched.issue(t, chunks[j])
                if ag.sched.arrive(t):
                    installed(i, t)
        a = np.zeros((n, 7))
        a[:, 6] = -1.0
        for i in range(n):
            if live[i]:
                a[i] = np.clip(agents[i].act(t, agents[i].sched.action(t)), -1, 1)
                xs[i].append(eef[i])
                acts[i].append(a[i])
                objs[i].append(obj[i])
                cons[i].append(bool(con[i]))
        if any(dq is not None for dq in dqs):  # ShiftEnv clears it after applying
            vec.set_attr("pending_dq", dqs)
        obs, _, _, _, info = vec.step(a)
        bad[live] = np.maximum(bad[live], info["bad_qacc"][live])  # live steps only
        for i in range(n):
            if live[i] and bool(info["is_success"][i]):
                succ[i], steps_ok[i], live[i] = True, t + 1, False
        if not live.any():
            break
    out = []
    for i, e in enumerate(eps):
        ag, pt, ep = agents[i], perts[i], e["ep"]
        method, noise = gauto.parse_arm(e["arm"])
        rec = {"brain": args.brain, "policy": REPO, "suite": ep.suite, "task": ep.task, "init": ep.init, "kind": ep.kind,
               "cell": e["cell"], "arm": e["arm"], "method": method, "noise": noise, "conf": CONF, "seed": ep.seed,
               "mag_class": ep.mag_class, "mag": ep.mag, "angle": ep.angle, "r": ep.r,
               "success": bool(succ[i]), "steps_to_success": steps_ok[i], "steps": len(acts[i]),
               "calls_sched": ag.sched.n_sched, "calls_trig": ag.sched.n_trig, "calls_shadow": 2 * (shadow[i] is not None),
               "t_fire": pt.t_fire, "g_on": ag.g_on, "t_engage": ag.t_e, "bad_qacc": int(bad[i]),
               "t_alarm": getattr(ag.tracker, "t_alarm", None), "c_extra": np.round(ag.c, 6).tolist(),
               "kappa_log": ag.kappa_log, "shadow": shadow[i], "plans_in": got[i],
               "act_hash": hashlib.sha256(np.asarray(acts[i]).tobytes()).hexdigest()[:16]}
        rec |= orx.episode_metrics(xs[i], acts[i], objs[i])
        x, o, c, tc = np.asarray(xs[i]), np.asarray(objs[i]), np.asarray(cons[i]), rec["t_close"]
        # signed grasp miss along the shift (as C1-E2/E3): > 0 when the gripper closed beyond the object
        rec["grasp_miss_along_m"] = (float((x[tc] - o[tc])[:2] @ ep.delta[:2]) / ep.mag
                                     if pt.t_fire is not None and ep.mag > 0 and tc is not None else None)
        # pushes before the shift, up to its last pre-shift observation; with no shift, up to the first close command
        end = pt.t_fire if pt.t_fire is not None and ep.kind != "control" else tc
        w = o if end is None else o[: end + 1]
        rec["pushed_before_fire"] = bool(len(w) > 1 and np.linalg.norm(w - w[0], axis=1).max() > orx.EPS)
        # unstable shift: the object moved another > 1 cm, seen 20 steps after the shift or at the first close command
        k = None if pt.t_fire is None else min(pt.t_fire + 21, len(o) if tc is None else tc)
        rec["unstable"] = bool(ep.kind != "control" and k is not None and pt.t_fire < k < len(o)
                               and np.linalg.norm(o[k] - (pt.p_pre + ep.delta)) > 0.01)
        rec["contact_at_shift"] = (bool(c[pt.t_fire: pt.t_fire + 2].any())
                                   if pt.t_fire is not None and ep.kind != "control" else None)
        if args.trace:
            rec["trace"] = {"p": np.round(o, 5).tolist(), "eef": np.round(x, 5).tolist(), "c": c.astype(int).tolist(),
                            "tobs": tobs[i], "plans": plans[i]}
        out.append(rec)
    return out


path = snapshot_download(REPO, revision=REV)  # pi0.5's pinned revision (spec §4.1); the cache serves it offline
cfg = PreTrainedConfig.from_pretrained(path)
cfg.pretrained_path, cfg.device, cfg.compile_model = path, args.device, False  # pi0.5 v044: true + max-autotune
if hasattr(cfg, "load_vlm_weights"):  # SmolVLA only
    cfg.load_vlm_weights = False
if args.precision == "bf16-row":
    cfg.dtype = "bfloat16"
assert cfg.chunk_size == orx.H, cfg.chunk_size
env_cfg = LiberoCfg(task="libero_spatial", task_ids=[0])
policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval().requires_grad_(False)
if args.precision == "fp32":  # spec §4.1: in bf16 a chunk depends on the batch it was computed in (C1-E2 spec journal)
    policy = policy.float()
pre, post = make_pre_post_processors(cfg, path, preprocessor_overrides={"device_processor": {"device": args.device}})
env_pre, _ = env_cfg.get_env_processors()  # the LIBERO env postprocessor is the identity

classes = {int(c) for c in args.classes.split(",")}
episodes = [{"ep": ep, "cell": c, "arm": m}
            for st, tk in tasks for k in args.kinds.split(",") for c in args.cells.split(",") for m in ARMS
            for i in span(args.inits)
            if (ep := hf.make_episode(st, tk, i, k, prefix="c1e4")).kind == "control" or ep.mag_class in classes
            if (st, tk, i, k, c, m) not in done]
print(f"{len(episodes)} episodes to run", flush=True)
t0, n_done = time.time(), 0
n_try = n_fail = 0  # batches over the whole invocation: abort on a systematic failure, never on one bad task
failed_tasks = set()
for st, tk in tasks:
    todo = [e for e in episodes if (e["ep"].suite, e["ep"].task) == (st, tk)]
    if not todo:
        continue
    vec = make_vec(st, tk)
    for key in sorted({(e["cell"], e["ep"].kind) for e in todo}):
        group = [e for e in todo if (e["cell"], e["ep"].kind) == key]
        for j in range(0, len(group), args.n_envs):
            n_try += 1
            tb = time.time()
            try:
                recs = run_batch(vec, group[j: j + args.n_envs])
            except Exception as exc:  # spec §8: a fallen batch is skipped here; a rerun with resume retries it
                n_fail += 1
                failed_tasks.add((st, tk))
                print(f"FAILED {st}:{tk} {key[0]},{key[1]} [{j}:{j + args.n_envs}]: {exc!r}", flush=True)
                traceback.print_exc()
                try:
                    vec.close(terminate=True)
                except Exception:  # an async worker died natively: kill the rest directly
                    for pr in getattr(vec, "processes", []):
                        pr.terminate()
                if n_fail >= 5 and n_fail > 0.2 * n_try and len(failed_tasks) >= 3:
                    print(f"ABORT: systematic failure, {n_fail} of {n_try} batches failed in {len(failed_tasks)} tasks",
                          flush=True)
                    sys.exit(1)
                vec = make_vec(st, tk)
                continue
            with open(args.out, "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
            n_done += len(recs)
            rate = (time.time() - t0) / n_done
            # the pod's precision rule reads full batches only (scripts/gpu_c1e4.sh smoke)
            print(f"batch {st}:{tk} {key[0]},{key[1]} real={len(recs)}/{args.n_envs} {time.time() - tb:.1f} s", flush=True)
            print(f"{n_done}/{len(episodes)} episodes, {rate:.2f} s/episode, eta {rate * (len(episodes) - n_done) / 3600:.2f} h",
                  flush=True)
    vec.close()
