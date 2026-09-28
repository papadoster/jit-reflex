"""C1-E2 runner: LIBERO + frozen SmolVLA with the object-anchored reflexes of src/objreflex.py.

Runs in the separate PyTorch/LeRobot env. One JSON line per episode. Example (Mac smoke, debug only):
    MUJOCO_GL=cgl ~/Desktop/M2R-c1-env/bin/python scripts/c1e2_run.py --tasks libero_spatial:0 \
        --cells C --methods none,G --kinds step --inits 48-49 --n-envs 2 --vector sync --device mps --out x.jsonl
Pod: MUJOCO_GL=egl ... --vector async --n-envs 16 --device cuda
"""

import argparse
import json
import sys
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import objreflex as orx  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoCfg  # noqa: E402
from lerobot.envs.libero import LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402
from lerobot.processor.env_processor import LiberoProcessorStep  # noqa: E402
from lerobot.utils.constants import ACTION, OBS_STATE  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--tasks", default="all", help="all, or suite:task[,suite:task]")
p.add_argument("--cells", default="A")
p.add_argument("--methods", default="none")
p.add_argument("--kinds", default="step", help="step, smooth, control")
p.add_argument("--inits", default="0-9")
p.add_argument("--t-ramp", type=int, default=5)
p.add_argument("--k-p", type=float, default=1.0)
p.add_argument("--headroom", action="store_true", help="C1-E1 headroom: two extra brain calls at the step shift, method none")
p.add_argument("--n-envs", type=int, default=16)
p.add_argument("--vector", choices=["async", "sync"], default="async")
p.add_argument("--device", default="cuda")
p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
p.add_argument("--out", required=True)
args = p.parse_args()
dev = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = False  # literal fp32 on CUDA (spec §4)
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
        Path(args.out).write_text(text)
    for line in text.splitlines():
        try:
            r = json.loads(line)
        except json.JSONDecodeError:
            print(f"warning: skipping a malformed line in {args.out}: {line[:80]!r}", flush=True)
            continue
        done.add((r["suite"], r["task"], r["init"], r["kind"], r["cell"], r["method"]))

cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.pretrained_path, cfg.device, cfg.load_vlm_weights = args.policy, args.device, False
assert cfg.chunk_size == orx.H, cfg.chunk_size
env_cfg = LiberoCfg(task="libero_spatial", task_ids=[0])
# spec §4: fp32. make_policy keeps the checkpoint's bf16 body, and in bf16 a chunk depends on the batch it was
# computed in (1.8e-2 vs 2e-6 in fp32), which would break the pairing of episodes across env slots
policy = make_policy(cfg=cfg, env_cfg=env_cfg).float().eval().requires_grad_(False)
pre, post = make_pre_post_processors(cfg, args.policy, preprocessor_overrides={"device_processor": {"device": args.device}})
env_pre, _ = env_cfg.get_env_processors()  # the LIBERO env postprocessor is the identity


def stat(pipe, key, name):
    step = next(st for st in pipe.steps if hasattr(st, "_tensor_stats") and key in st._tensor_stats)
    return step._tensor_stats[key][name].float().cpu().numpy()


STD_S, STD_A = stat(pre, OBS_STATE, "std"), stat(post, ACTION, "std")


class ShiftEnv(LiberoEnv):
    """LiberoEnv plus a pending shift of the target object applied before the next physics step, and oracle
    positions (object of interest, EEF) in info. The target is the first object of interest with a free joint.
    The MjSim is re-read on every call: a hard reset rebuilds it."""

    pending_dq = None
    keep_px = False  # headroom: keep the camera images of the shifted state, rendered before the physics step
    shift_px = None

    def _target(self):
        e = self._env.env
        name = next(n for n in e.obj_of_interest if n in e.objects_dict)
        jnt = e.objects_dict[name].joints[0]
        return name, e.sim.model.get_joint_qpos_addr(jnt)[0], e.sim.model.get_joint_qvel_addr(jnt)[0]

    def _oracle(self, info):
        e = self._env.env
        name, _, _ = self._target()
        info["obj_pos"] = np.array(e.sim.data.body_xpos[e.obj_body_id[name]])
        info["eef_pos"] = np.array(e.sim.data.site_xpos[e.robots[0].eef_site_id])
        return info

    def reset(self, seed=None, **kw):
        obs, info = super().reset(seed=seed, **kw)
        self.shift_px = None
        return obs, self._oracle(info)

    def step(self, action):
        if self.pending_dq is not None:
            e = self._env.env
            _, qa, va = self._target()
            e.sim.data.qpos[qa: qa + 3] += self.pending_dq
            e.sim.data.qvel[va: va + 6] = 0.0
            e.sim.forward()
            self.pending_dq = None
            if self.keep_px:  # the camera sensors directly: a forced observable update would shift robosuite's
                # sampling phase and change every later observation of this episode
                self.shift_px = {self.camera_name_mapping[c]: np.array(e._observables[c]._sensor({}))
                                 for c in self.camera_name}
        obs, r, _, trunc, info = super().step(action)
        # never report termination: the runner ends episodes itself (is_success, step limit), and a terminated env
        # trips the DISABLED-autoreset assert of SyncVectorEnv on the next step of the batch
        return obs, r, False, trunc, self._oracle(info)


def make_vec(suite, task):
    fns = [lambda: ShiftEnv(task_suite=_get_suite(suite), task_id=task, task_suite_name=suite,
                            obs_type="pixels_agent_pos", observation_width=360, observation_height=360)
           for _ in range(args.n_envs)]
    kw = {"autoreset_mode": gym.vector.AutoresetMode.DISABLED}
    return gym.vector.AsyncVectorEnv(fns, **kw) if args.vector == "async" else gym.vector.SyncVectorEnv(fns, **kw)


def raw_state(obs, i):
    """The 8-dim LIBERO state (EEF pos, axis-angle, finger qpos) of env i before normalisation."""
    rs = obs["robot_state"]
    q = torch.as_tensor(np.asarray(rs["eef"]["quat"][i: i + 1]), dtype=torch.float32)
    aa = LiberoProcessorStep()._quat2axisangle(q)[0].numpy()
    return np.concatenate([rs["eef"]["pos"][i], aa, rs["gripper"]["qpos"][i]])


def brain(vec, obs, idx, noises, j_idx=()):
    """Raw-unit chunks (len(idx), H, 7) for the envs in idx, each with its own fixed noise; for envs in j_idx also
    the raw J by state at every chunk position, (H, 7, 8) (GJ row, CUDA/CPU only)."""
    b = env_pre(preprocess_observation(obs))
    task = vec.call("task_description")
    sub = pre({k: v[idx] for k, v in b.items() if torch.is_tensor(v)} | {"task": [task[i] for i in idx]})
    nz = torch.cat(noises).to(dev)
    with torch.no_grad():
        ch = policy.predict_action_chunk(sub, noise=nz)
    js = {}
    if j_idx:
        import c1e2_j  # noqa: PLC0415  (only the GJ row needs it)

        for pos, i in enumerate(idx):
            if i in j_idx:
                one = {k: (v[[pos]] if torch.is_tensor(v) else [v[pos]] if isinstance(v, list) else v) for k, v in sub.items()}
                J = c1e2_j.jacobian_all_positions(policy, one, nz[pos: pos + 1]).float().cpu().numpy()
                js[i] = STD_A[None, :, None] * J / STD_S[None, None, :]
    return post(ch).cpu().numpy(), js


class JCorr:
    """GJ (spec §5): J_k (o_t - o_hat_k) for the chunk observed at t_obs. o_hat_k: the state observed at t_obs with the
    position advanced by the integrator over the chunk's first k actions and by G's extra since t_obs."""

    def __init__(self, ag, J, o_obs, chunk, t_obs):
        self.ag, self.J, self.o_obs, self.chunk, self.t_obs = ag, J, o_obs, chunk, t_obs

    def __call__(self, t, k):
        o_hat = self.o_obs.copy()
        o_hat[:3] += orx.G_POS * np.clip(self.chunk[:k, :3], -1, 1).sum(0) + (self.ag.c - self.ag.c_hist[self.t_obs])
        return self.J[k] @ (self.ag.o_raw[t] - o_hat)


def run_batch(vec, eps):
    """Run up to n_envs episodes (same task, same cell and kind) to the end; one record per real episode."""
    n = args.n_envs
    pad = eps + [eps[-1]] * (n - len(eps))  # padding envs are run and ignored
    keep = [args.headroom and j < len(eps) and e["method"] == "none" and e["ep"].kind == "step" for j, e in enumerate(pad)]
    vec.set_attr("init_state_id", [e["ep"].init for e in pad])
    vec.set_attr("keep_px", keep)
    # one reset seed for every slot: LIBERO seeds numpy with it and places the fixtures (cabinet, stove, rack) from it
    obs, info = vec.reset(seed=[0] * n)
    agents = [orx.Agent(e["method"], *orx.CELLS[e["cell"]], t_ramp=args.t_ramp, k_p=args.k_p) for e in pad]
    perts = [orx.Perturbation(e["ep"]) for e in pad]
    gens = [torch.Generator().manual_seed(e["ep"].seed) for e in pad]
    t_max = vec.get_attr("_max_episode_steps")[0]
    live, succ = np.ones(n, bool), np.zeros(n, bool)
    live[len(eps):] = False  # padding envs only step no-ops: no brain calls, no J
    steps_ok, hr = [None] * n, [None] * n
    xs, acts, objs = [[] for _ in range(n)], [[] for _ in range(n)], [[] for _ in range(n)]
    pend_j = {}

    def installed(i):  # a new chunk starts being used by env i
        ag = agents[i]
        ag.on_new_chunk()
        if pad[i]["method"] == "GJ":
            t_obs = ag.sched.t_obs
            ag.j_corr = JCorr(ag, pend_j.pop((i, t_obs)), ag.o_raw[t_obs], ag.sched.chunk, t_obs)

    for t in range(t_max):
        eef, obj = info["eef_pos"], info["obj_pos"]
        dqs = [None] * n
        for i in range(n):
            if not live[i]:
                continue
            dqs[i] = perts[i].dq(t, eef[i], obj[i], agents[i].grasp_started)
            if dqs[i] is not None and perts[i].t_fire == t:
                agents[i].p_pre = obj[i].copy()
            agents[i].observe(t, eef[i], obj[i])
            if pad[i]["method"] == "GJ":
                agents[i].o_raw[t] = raw_state(obs, i)
            if agents[i].sched.arrive(t):
                installed(i)
        want = [i for i in range(n) if live[i] and agents[i].sched.wants_call(t, agents[i].trigger(t))]
        if want:
            j_idx = [i for i in want if pad[i]["method"] == "GJ"]
            nz = [torch.randn(1, orx.H, cfg.max_action_dim, generator=gens[i]) for i in want]
            chunks, js = brain(vec, obs, want, nz, j_idx)
            for i, ch in zip(want, chunks):
                if i in js:
                    pend_j[(i, t)] = js[i]
                agents[i].sched.issue(t, ch)
                if agents[i].sched.arrive(t):
                    installed(i)
        a = np.zeros((n, 7))
        a[:, 6] = -1.0
        for i in range(n):
            if live[i]:
                a[i] = np.clip(agents[i].act(t, agents[i].sched.action(t)), -1, 1)
                xs[i].append(eef[i])
                acts[i].append(a[i])
                objs[i].append(obj[i])
        vec.set_attr("pending_dq", dqs)
        prev_obs = obs
        obs, _, _, _, info = vec.step(a)
        for i in range(n):  # C1-E1 headroom on the same state: the brain's fresh plan before vs after the step shift
            if keep[i] and perts[i].t_fire == t:
                z = torch.randn(1, orx.H, cfg.max_action_dim, generator=torch.Generator().manual_seed(pad[i]["ep"].seed + 1))
                shifted = {**prev_obs, "pixels": {k: v.copy() for k, v in prev_obs["pixels"].items()}}
                for k, px in vec.get_attr("shift_px")[i].items():
                    shifted["pixels"][k][i] = px
                c0, c1 = brain(vec, prev_obs, [i], [z])[0][0], brain(vec, shifted, [i], [z])[0][0]
                hr[i] = {"ratio": orx.headroom(c0, c1, pad[i]["ep"].delta),
                         "dist_m": float(np.linalg.norm(eef[i] - obj[i]))}
        for i in range(n):
            if live[i] and bool(info["is_success"][i]):
                succ[i], steps_ok[i], live[i] = True, t + 1, False
        if not live.any():
            break
    out = []
    for i, e in enumerate(eps):
        ag, pt, ep = agents[i], perts[i], e["ep"]
        rec = {"suite": ep.suite, "task": ep.task, "init": ep.init, "kind": ep.kind, "cell": e["cell"], "method": e["method"],
               "seed": ep.seed, "mag_class": ep.mag_class, "mag": ep.mag, "angle": ep.angle, "r": ep.r,
               "t_ramp": args.t_ramp, "k_p": args.k_p,
               "success": bool(succ[i]), "steps_to_success": steps_ok[i], "steps": len(acts[i]),
               "calls_sched": ag.sched.n_sched, "calls_trig": ag.sched.n_trig, "t_fire": pt.t_fire,
               "g_on": ag.g_on, "headroom": hr[i]}
        rec |= orx.episode_metrics(xs[i], acts[i], objs[i])
        o, tc = np.asarray(objs[i]), rec["t_close"]
        # signed grasp miss along the shift (spec §9 overshoot): > 0 when the gripper closed beyond the object
        rec["grasp_miss_along_m"] = (float((xs[i][tc] - o[tc])[:2] @ ep.delta[:2]) / ep.mag
                                     if ep.kind != "control" and ep.mag > 0 and tc is not None else None)
        # pushes before the shift, up to its last pre-shift observation; with no shift, up to the first close command
        end = pt.t_fire if pt.t_fire is not None else tc
        w = o if end is None else o[: end + 1]
        rec["pushed_before_fire"] = bool(len(w) > 1 and np.linalg.norm(w - w[0], axis=1).max() > orx.EPS)
        # unstable shift: the object moved another > 1 cm, seen 20 steps after the shift or at the first close command
        # if that comes earlier (a grasp moves the object too)
        k = None if pt.t_fire is None else min(pt.t_fire + 21, len(o) if tc is None else tc)
        rec["unstable"] = bool(ep.kind == "step" and k is not None and pt.t_fire < k < len(o)
                               and np.linalg.norm(o[k] - (pt.p_pre + ep.delta)) > 0.01)
        out.append(rec)
    return out


episodes = [{"ep": orx.make_episode(st, tk, i, k), "cell": c, "method": m}
            for st, tk in tasks for k in args.kinds.split(",") for c in args.cells.split(",")
            for m in args.methods.split(",") for i in span(args.inits)
            if (st, tk, i, k, c, m) not in done]
print(f"{len(episodes)} episodes to run", flush=True)
t0, n_done = time.time(), 0
for st, tk in tasks:
    todo = [e for e in episodes if (e["ep"].suite, e["ep"].task) == (st, tk)]
    if not todo:
        continue
    vec = make_vec(st, tk)
    for key in sorted({(e["cell"], e["ep"].kind) for e in todo}):
        group = [e for e in todo if (e["cell"], e["ep"].kind) == key]
        for j in range(0, len(group), args.n_envs):
            recs = run_batch(vec, group[j: j + args.n_envs])
            with open(args.out, "a") as f:
                for rec in recs:
                    f.write(json.dumps(rec) + "\n")
            n_done += len(recs)
            rate = (time.time() - t0) / n_done
            print(f"{n_done}/{len(episodes)} episodes, {rate:.2f} s/episode, eta {rate * (len(episodes) - n_done) / 3600:.2f} h",
                  flush=True)
    vec.close()
