"""C1-E3 runner: LIBERO + frozen SmolVLA with the reflexes of src/handoff.py (spec
docs/superpowers/specs/2026-09-29-c1-e3-handoff-reflex-design.md).

Runs in the separate PyTorch/LeRobot env. One JSON line per episode. --methods takes arm labels (handoff.parse_arm):
a method (none, T0, PPC, G, GT, Gkeep, GkeepT, GR, GRT, GRret), optionally +surr / +contact (push filter) and
@<noise tag> (noisy tracker, handoff.NOISE). Example (Mac, debug only):
    MUJOCO_GL=cgl ~/Desktop/M2R-c1-env/bin/python scripts/c1e3_run.py --tasks libero_spatial:0 --cells A \
        --methods none,GR --kinds step --inits 48-49 --n-envs 2 --vector sync --device mps --out x.jsonl
Pod: MUJOCO_GL=egl ... --vector async --n-envs 10 --device cuda
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import handoff as hf  # noqa: E402
import objreflex as orx  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoCfg  # noqa: E402
from lerobot.envs.libero import LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tasks", default="all", help="all (the 26), libero_10 (handoff.TASKS10), or suite:task[,suite:task]")
p.add_argument("--cells", default="A")
p.add_argument("--methods", default="none", help="arm labels, e.g. none,GR,GR+surr,GR@real")
p.add_argument("--kinds", default="step", help="step, close, control")
p.add_argument("--inits", default="0-9")
p.add_argument("--classes", default="0,1,2", help="shift classes kept (step and close); spec §6 rows 5-8: 2")
p.add_argument("--alpha", type=float, default=0.0, help="synthetic seeing brain SmolVLA-alpha (spec §4)")
p.add_argument("--t-ramp", type=int, default=5)
p.add_argument("--k-p", type=float, default=1.0)
# the pilot's choices (spec §7); the defaults are the provisional values of pilot rows P1-P2
p.add_argument("--tau-k", type=float, default=0.01)
p.add_argument("--K", type=int, default=20)
p.add_argument("--rc", type=float, default=0.05, help="surrogate push radius, m")
p.add_argument("--dir", type=int, default=0, help="surrogate direction condition (0/1)")
p.add_argument("--beta", type=float, default=1.0, help="noisy tracker EMA")
p.add_argument("--eps-noise", type=float, default=orx.EPS, help="G-R dead band and engagement with noisy eyes, m")
p.add_argument("--vmin-noise", type=float, default=orx.PPC_V_MIN, help="PPC speed threshold with noisy eyes, m/step")
p.add_argument("--pad-brain", type=int, default=0, help="constant brain batch of n_envs rows + deterministic torch")
p.add_argument("--trace", action="store_true", help="per-step positions, contacts and plans (pilot rows P3-P6)")
p.add_argument("--save-images", type=int, default=0, help="brain input images at the shift for the first N episodes")
p.add_argument("--n-envs", type=int, default=10)
p.add_argument("--vector", choices=["async", "sync"], default="async")
p.add_argument("--device", default="cuda")
p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
p.add_argument("--out", help="JSON lines (required unless --check-targets)")
p.add_argument("--check-targets", action="store_true", help="print each task's target and placement target, exit")
args = p.parse_args()
if not (args.out or args.check_targets):
    p.error("--out is required")
CONF = {"t_ramp": args.t_ramp, "k_p": args.k_p, "tau_k": args.tau_k, "K": args.K, "rc": args.rc, "dir": args.dir,
        "beta": args.beta, "eps_noise": args.eps_noise, "vmin_noise": args.vmin_noise, "pad": args.pad_brain}
ARMS = args.methods.split(",")
for arm in ARMS:
    # spec §5: the alpha brain does not combine with the return
    assert hf.parse_arm(arm)[0] != "GRret" or args.alpha == 0, f"{arm} runs with --alpha 0 only"
dev = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = False  # literal fp32 on CUDA (spec §4)
torch.backends.cudnn.allow_tf32 = False
if args.pad_brain:  # spec §4 pair noise: constant batch plus deterministic kernels
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # read at the first cuBLAS call, below
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cudnn.benchmark = False
IMG_DIR = Path("eval_output/c1e3_images")  # never committed (the owner's check only, spec §7)


def span(txt):
    a, _, b = txt.partition("-")
    return list(range(int(a), int(b or a) + 1))


if args.tasks == "all":
    tasks = orx.TASKS
elif args.tasks == "libero_10":
    tasks = hf.TASKS10
else:
    tasks = [(t.split(":")[0], int(t.split(":")[1])) for t in args.tasks.split(",")]
done = set()  # resume: skip episodes already written
if args.out and Path(args.out).exists():
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
        done.add((r["suite"], r["task"], r["init"], r["kind"], r["cell"], r["arm"], r["alpha"]))

class ShiftEnv(LiberoEnv):
    """LiberoEnv plus a pending shift of the target object applied before the next physics step, and oracle data in
    info: object of interest and EEF positions, robot-object contact, the placement target's position (spec §4); at
    reset also the object's box (journal item 14). The target is the first object of interest with a free joint. The
    MjSim is re-read on every call and the contact masks, box and placement target on every reset: a hard reset
    rebuilds the sim."""

    pending_dq = None

    def _target(self):
        e = self._env.env
        name = next(n for n in e.obj_of_interest if n in e.objects_dict)
        jnt = e.objects_dict[name].joints[0]
        return name, e.sim.model.get_joint_qpos_addr(jnt)[0], e.sim.model.get_joint_qvel_addr(jnt)[0]

    def _setup(self):
        e = self._env.env
        m, name = e.sim.model._model, self._target()[0]
        root = e.obj_body_id[name]
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
        # journal item 14: the world-axis-aligned box of the object's geoms (their local AABB corners in the world),
        # attached to the body origin p as [centre offset c from p, half sides h]
        d, g = e.sim.data._data, np.flatnonzero(self._obj)
        signs = np.array(np.meshgrid(*[[-1.0, 1.0]] * 3)).T.reshape(-1, 3)
        local = m.geom_aabb[g, None, :3] + signs * m.geom_aabb[g, None, 3:]  # (geoms, 8, 3)
        pts = (d.geom_xpos[g, None] + np.einsum("gij,gkj->gki", d.geom_xmat[g].reshape(-1, 3, 3), local)).reshape(-1, 3)
        lo, hi = pts.min(0) - d.xpos[root], pts.max(0) - d.xpos[root]
        self._box = np.r_[(lo + hi) / 2, (hi - lo) / 2]
        self._goal = None  # spec §4: the last argument of the goal's On/In predicate whose first is the target object
        for pred in e.parsed_problem["goal_state"]:
            if len(pred) == 3 and pred[0].lower() in ("on", "in") and pred[1] == name:
                if pred[2] in e.obj_body_id:
                    self._goal = ("body", e.obj_body_id[pred[2]])
                else:
                    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, pred[2])
                    assert sid >= 0, f"placement target {pred[2]} is neither a body nor a site"
                    self._goal = ("site", sid)
                break

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
        kind, i = self._goal or (None, None)
        info["goal_pos"] = (np.array(d.body_xpos[i]) if kind == "body" else np.array(d.site_xpos[i]) if kind == "site"
                            else np.full(3, np.nan))
        return info

    def reset(self, seed=None, **kw):
        obs, info = super().reset(seed=seed, **kw)
        self._setup()
        info["box"] = self._box.copy()  # through info: an AsyncVectorEnv worker's attributes are out of reach
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


def brain(obs, desc, noises):
    """Raw-unit chunks (B, H, 7) for B env observations with their task descriptions, each with its own noise."""
    b = preprocess_observation(obs)
    b["task"] = list(desc)
    with torch.no_grad():
        ch = policy.predict_action_chunk(pre(env_pre(b)), noise=torch.cat(noises).to(dev))
    return post(ch).cpu().numpy()


def make_agent(arm, cell, ep, box):
    method, filt, noise = hf.parse_arm(arm)
    kw = {"t_ramp": args.t_ramp, "k_p": args.k_p, "tau_k": args.tau_k, "K": args.K}
    if noise:
        sigma, lag, q = hf.NOISE[noise]
        kw |= {"tracker": hf.Tracker(ep.seed, sigma, lag, q, args.beta), "eps": args.eps_noise,
               "ppc_v_min": args.vmin_noise}
    if filt:
        kw["push"] = hf.PushFilter(filt, args.rc, bool(args.dir), box)
    return hf.HAgent(method, *orx.CELLS[cell], **kw)


def release(ep, t_close, acts, objs, goals, success):
    """spec §4: the object's offset from the placement target at the first open command after the first close, along
    the shift direction, across it (table plane) and in height; lifted: >= LIFT above its height at the close. LIBERO
    often reports success while the object is still held: then the success step stands for the release (at "success").
    None without a close command, a placement target, or a release."""
    if t_close is None or np.isnan(goals[0]).any():
        return None
    opened = [t for t in range(t_close + 1, len(acts)) if acts[t][6] <= 0]
    if not (opened or success):
        return None
    t, at = (opened[0], "open") if opened else (len(acts) - 1, "success")
    rel, u = objs[t] - goals[t], np.array([np.cos(ep.angle), np.sin(ep.angle)])
    lifted = bool((objs[t_close: t + 1, 2] - objs[t_close, 2] >= hf.LIFT).any())
    return {"t": t, "at": at, "along": round(float(rel[:2] @ u), 5),
            "across": round(float(rel[0] * -u[1] + rel[1] * u[0]), 5), "dz": round(float(rel[2]), 5), "lifted": lifted}


def save_images(obs, i, tag):
    IMG_DIR.mkdir(parents=True, exist_ok=True)
    from PIL import Image  # noqa: PLC0415  (only the pilot's sanity images)

    for cam, px in obs["pixels"].items():
        Image.fromarray(np.asarray(px[i], np.uint8)).save(IMG_DIR / f"{tag}_{cam}.png")


n_img = 0


def run_batch(vec, eps):
    """Run up to n_envs episodes (same task, same cell and kind) to the end; one record per real episode."""
    global n_img
    n = args.n_envs
    pad = eps + [eps[-1]] * (n - len(eps))  # padding envs are run and ignored
    vec.set_attr("init_state_id", [e["ep"].init for e in pad])
    # one reset seed for every slot: LIBERO seeds numpy with it and places the fixtures (cabinet, stove, rack) from it
    obs, info = vec.reset(seed=[0] * n)
    desc, boxes = vec.call("task_description"), info["box"].copy()
    agents = [make_agent(e["arm"], e["cell"], e["ep"], boxes[i]) for i, e in enumerate(pad)]
    perts = [hf.Perturbation(e["ep"], boxes[i]) for i, e in enumerate(pad)]
    sbs = [hf.SeeingBrain(args.alpha, args.t_ramp) for _ in pad]
    gens = [torch.Generator().manual_seed(e["ep"].seed) for e in pad]
    t_max = vec.get_attr("_max_episode_steps")[0]
    live, succ = np.ones(n, bool), np.zeros(n, bool)
    live[len(eps):] = False  # padding envs only step no-ops: no brain calls
    steps_ok, bad = [None] * n, info["bad_qacc"].copy()
    xs, acts, objs, cons, goals, tobs, plans = ([[] for _ in range(n)] for _ in range(7))
    shot = [False] * n

    for t in range(t_max):
        eef, obj, con, goal = info["eef_pos"], info["obj_pos"], info["contact"], info["goal_pos"]
        dqs = [None] * n
        for i in range(n):
            if not live[i]:
                continue
            ag, pt = agents[i], perts[i]
            dqs[i] = pt.dq(t, eef[i], obj[i], ag.grasp_started)
            if (args.save_images and pt.t_fire is not None and pad[i]["ep"].kind != "control"
                    and t in (pt.t_fire, pt.t_fire + 1) and (shot[i] or n_img < args.save_images)):
                e = pad[i]
                save_images(obs, i, f"{e['ep'].suite}_{e['ep'].task}_{e['ep'].init}_{e['ep'].kind}_{e['cell']}_"
                                    f"{e['arm']}_a{args.alpha:g}_{'pre' if t == pt.t_fire else 'post'}")
                n_img += not shot[i]
                shot[i] = True
            tobs[i].append(-1 if ag.sched.t_obs is None else ag.sched.t_obs)
            ag.observe(t, eef[i], obj[i], bool(con[i]))
            if ag.sched.arrive(t):
                ag.on_new_chunk()
        want = [i for i in range(n) if live[i] and agents[i].sched.wants_call(t, agents[i].trigger(t))]
        if want:
            calls = list(range(n)) if args.pad_brain else want  # padded: every slot, the others as dummies
            nz = [torch.randn(1, orx.H, cfg.max_action_dim, generator=gens[i]) if i in want
                  else torch.zeros(1, orx.H, cfg.max_action_dim) for i in calls]
            chunks = brain(rows(obs, calls), [desc[i] for i in calls], nz)
            for i in want:
                ag = agents[i]
                start = 0 if ag.sched.chunk is None else ag.sched.d  # the answer's first executed index
                ch = sbs[i].chunk(t, chunks[calls.index(i)], perts[i], obj[i], start, ag.grasp_started)
                if args.trace:
                    plans[i].append([t, np.round(eef[i], 5).tolist(), np.round(ch[:, [0, 1, 2, 6]], 4).tolist()])
                ag.sched.issue(t, ch)
                if ag.sched.arrive(t):
                    ag.on_new_chunk()
        a = np.zeros((n, 7))
        a[:, 6] = -1.0
        for i in range(n):
            if live[i]:
                a[i] = np.clip(agents[i].act(t, agents[i].sched.action(t)), -1, 1)
                sbs[i].executed(t, agents[i].sched.t_obs, a[i], perts[i])
                xs[i].append(eef[i])
                acts[i].append(a[i])
                objs[i].append(obj[i])
                cons[i].append(bool(con[i]))
                goals[i].append(goal[i])
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
        ag, pt, ep, sb = agents[i], perts[i], e["ep"], sbs[i]
        method, filt, noise = hf.parse_arm(e["arm"])
        rec = {"suite": ep.suite, "task": ep.task, "init": ep.init, "kind": ep.kind, "cell": e["cell"],
               "arm": e["arm"], "method": method, "filter": filt, "noise": noise, "alpha": args.alpha, "conf": CONF,
               "seed": ep.seed, "mag_class": ep.mag_class, "mag": ep.mag, "angle": ep.angle, "r": ep.r,
               "box": np.round(boxes[i], 5).tolist(),
               "success": bool(succ[i]), "steps_to_success": steps_ok[i], "steps": len(acts[i]),
               "calls_sched": ag.sched.n_sched, "calls_trig": ag.sched.n_trig, "t_fire": pt.t_fire,
               "g_on": ag.g_on, "t_engage": ag.t_e, "t_lift": ag.t_lift, "bad_qacc": int(bad[i]),
               # the return still unsent at the end (None: it never started): finished vs cut by the open command
               "ret_left": None if ag.ret_left is None else np.round(ag.ret_left, 6).tolist(),
               "c_extra": np.round(ag.c, 6).tolist(),
               "kappa_log": [k | {"S": np.round(sb.s_at[k["t"]], 6).tolist()} for k in ag.kappa_log],
               "act_hash": hashlib.sha256(np.asarray(acts[i]).tobytes()).hexdigest()[:16]}
        rec |= orx.episode_metrics(xs[i], acts[i], objs[i])
        x, o, c, tc = np.asarray(xs[i]), np.asarray(objs[i]), np.asarray(cons[i]), rec["t_close"]
        rec["release"] = release(ep, tc, np.asarray(acts[i]), o, np.asarray(goals[i]), rec["success"])
        # signed grasp miss along the shift (spec §9, as C1-E2): > 0 when the gripper closed beyond the object
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
        # push marks per step before the first close (spec §9 filter report): surrogate at the grid's r_c and
        # direction with the object's box, contact oracle at t - 1 or t, both
        last = len(o) if tc is None else tc
        sf = [hf.surr_push(o[t], o[t - 1], x[t], x[t - 1], args.rc, bool(args.dir), boxes[i]) for t in range(1, last)]
        cf = [bool(c[t - 1] or c[t]) for t in range(1, last)]
        rec["push_steps"] = [sum(sf), sum(cf), sum(a and b for a, b in zip(sf, cf))]
        rec["contact_at_shift"] = (bool(c[pt.t_fire: pt.t_fire + 2].any())
                                   if pt.t_fire is not None and ep.kind != "control" else None)
        if args.trace:
            rec["trace"] = {"p": np.round(o, 5).tolist(), "eef": np.round(x, 5).tolist(), "c": c.astype(int).tolist(),
                            "tobs": tobs[i], "plans": plans[i]}
        out.append(rec)
    return out


if args.check_targets:  # the target object and placement target of every task (Mac check, no brain)
    for st, tk in tasks:
        env = ShiftEnv(task_suite=_get_suite(st), task_id=tk, task_suite_name=st, obs_type="pixels_agent_pos",
                       observation_width=64, observation_height=64)
        _, info = env.reset(seed=0)
        m, (name, _, _) = env._env.env.sim.model._model, env._target()
        free = m.jnt_type[m.body_jntadr[env._env.env.obj_body_id[name]]] == mujoco.mjtJoint.mjJNT_FREE
        print(f"{st}:{tk} target {name} ({'free' if free else 'NOT FREE'}), goal {env._goal}, "
              f"obj {np.round(info['obj_pos'], 3)}, goal pos {np.round(info['goal_pos'], 3)}, "
              f"geoms robot {env._robot.sum()} object {env._obj.sum()}, max steps {env._max_episode_steps}", flush=True)
        env.close()
    raise SystemExit

cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.pretrained_path, cfg.device, cfg.load_vlm_weights = args.policy, args.device, False
assert cfg.chunk_size == orx.H, cfg.chunk_size
env_cfg = LiberoCfg(task="libero_spatial", task_ids=[0])
# spec §4: fp32 (in bf16 a chunk depends on the batch it was computed in, C1-E2 spec journal)
policy = make_policy(cfg=cfg, env_cfg=env_cfg).float().eval().requires_grad_(False)
pre, post = make_pre_post_processors(cfg, args.policy, preprocessor_overrides={"device_processor": {"device": args.device}})
env_pre, _ = env_cfg.get_env_processors()  # the LIBERO env postprocessor is the identity


classes = {int(c) for c in args.classes.split(",")}
episodes = [{"ep": ep, "cell": c, "arm": m}
            for st, tk in tasks for k in args.kinds.split(",") for c in args.cells.split(",") for m in ARMS
            for i in span(args.inits)
            if (ep := hf.make_episode(st, tk, i, k)).kind == "control" or ep.mag_class in classes
            if (st, tk, i, k, c, m, args.alpha) not in done]
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
            print(f"{n_done}/{len(episodes)} episodes, {rate:.2f} s/episode, eta {rate * (len(episodes) - n_done) / 3600:.2f} h",
                  flush=True)
    vec.close()
