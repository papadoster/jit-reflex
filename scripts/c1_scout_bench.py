"""C1 scouting: SmolVLA call time and the cost and quality of J = d a / d state on a real LIBERO observation.

Runs in the separate PyTorch/LeRobot env, not in the JAX uv env. One state per process, batch 1:
    MUJOCO_GL=cgl ~/Desktop/M2R-c1-env/bin/python scripts/c1_scout_bench.py --mode mem            # peak memory first
    MUJOCO_GL=cgl ~/Desktop/M2R-c1-env/bin/python scripts/c1_scout_bench.py --state 40 --out x.jsonl

The state enters SmolVLA as one token at the end of the VLM prefix; image and language tokens cannot
attend to it. So the image+language K/V cache is computed once without grad, and only the state token's
pass through the VLM layers plus the expert's flow steps are differentiated ("split" path). The split
chunk is checked against the stock `sample_actions` chunk, and J against central finite differences.
"""

import argparse
import gc
import json
import time

import numpy as np
import torch
import torch.autograd.forward_ad as fwAD
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.flop_counter import FlopCounterMode
from transformers.cache_utils import DynamicCache

from lerobot.configs.policies import PreTrainedConfig
from lerobot.envs.configs import LiberoEnv
from lerobot.envs.factory import make_env
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.smolvla.modeling_smolvla import make_att_2d_masks
from lerobot.policies.smolvla.smolvlm_with_expert import apply_rope
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE

p = argparse.ArgumentParser()
p.add_argument("--mode", choices=["mem", "bench", "check", "visual", "arm", "e1"], default="bench")
p.add_argument("--policy", default="HuggingFaceVLA/smolvla_libero")
p.add_argument("--rename", default="", help="e.g. image:camera1,image2:camera2 for lerobot/smolvla_libero")
p.add_argument("--device", default="mps")
p.add_argument("--dtype", default="bfloat16", help="VLM + expert weights; the small projections stay float32")
p.add_argument("--suite", default="libero_spatial")
p.add_argument("--task", type=int, default=0)
p.add_argument("--state", type=int, default=0, help="env step of the measured state (rollout with 10-step chunks)")
p.add_argument("--reps", type=int, default=5)
p.add_argument("--init-state", type=int, default=1, help="e1: LIBERO initial state of the rollout")
p.add_argument("--ks", default="0,1,2,3,4,5,6,7", help="e1: which of the 8 states t_k = k*T//8 to measure")
p.add_argument("--out", default="", help="JSON lines, one per measurement, written as they finish")
args = p.parse_args()
dev = torch.device(args.device)
sync = {"mps": torch.mps.synchronize, "cuda": torch.cuda.synchronize}.get(dev.type, lambda: None)
rename = {f"observation.images.{a}": f"observation.images.{b}" for a, b in (kv.split(":") for kv in args.rename.split(",") if kv)}


def emit(key, val):
    line = json.dumps({"policy": args.policy, "device": args.device, "dtype": args.dtype, "state": args.state, key: val})
    print(line, flush=True)
    if args.out:
        with open(args.out, "a") as f:
            f.write(line + "\n")


def free():
    gc.collect()
    if dev.type == "mps":
        torch.mps.empty_cache()
    if dev.type == "cuda":
        torch.cuda.reset_peak_memory_stats()


def gib():  # memory held by the device allocator now (MPS has no peak counter)
    if dev.type == "mps":
        return round(torch.mps.driver_allocated_memory() / 2**30, 2)
    if dev.type == "cuda":
        return round(torch.cuda.max_memory_allocated() / 2**30, 2)
    return None


# --- policy, env, processors: the same chain as lerobot-eval ---------------------------------------------
env_cfg = LiberoEnv(task=args.suite, task_ids=[args.task])
env = make_env(env_cfg, n_envs=1)[args.suite][args.task]
cfg = PreTrainedConfig.from_pretrained(args.policy)
cfg.pretrained_path, cfg.device, cfg.load_vlm_weights, cfg.n_action_steps = args.policy, args.device, False, 10
policy = make_policy(cfg=cfg, env_cfg=env_cfg, rename_map=rename or None).eval().requires_grad_(False)
policy.model.vlm_with_expert.to(getattr(torch, args.dtype))  # as stored in the checkpoint: bf16 body, fp32 projections
pre, post = make_pre_post_processors(cfg, args.policy, preprocessor_overrides={
    "device_processor": {"device": args.device}, "rename_observations_processor": {"rename_map": rename}})
env_pre, env_post = env_cfg.get_env_processors()
m, vwe = policy.model, policy.model.vlm_with_expert
H, NS, A = cfg.chunk_size, cfg.num_steps, cfg.action_feature.shape[0]
free()
emit("mem_after_load_gib", gib())


def stats(pipe, key, name="std"):
    step = next(s for s in pipe.steps if hasattr(s, "_tensor_stats") and key in s._tensor_stats)
    return step._tensor_stats[key][name].to(dev).float()


std_s, std_a, mean_s = stats(pre, OBS_STATE), stats(post, ACTION), stats(pre, OBS_STATE, "mean")
SD = std_s.numel()


def to_batch(obs):
    b = preprocess_observation(obs)
    b["task"] = list(env.call("task_description"))
    return pre(env_pre(b))


def timeit(fn, reps=args.reps):
    fn(), sync()
    ts = []
    for _ in range(reps):
        t = time.perf_counter()
        fn(), sync()
        ts.append(time.perf_counter() - t)
    free()
    return round(1e3 * float(np.median(ts)), 1)


class OpCount(TorchDispatchMode):  # aten ops ~ kernel launches in eager mode
    n = 0

    def __torch_dispatch__(self, func, types, a=(), kw=None):
        OpCount.n += 1
        return func(*a, **(kw or {}))


def cost(fn):
    OpCount.n = 0
    with FlopCounterMode(display=False) as fc, OpCount():
        fn()
    free()
    return {"gflop": round(fc.get_total_flops() / 1e9, 2), "aten_ops": OpCount.n}


# --- model pieces ------------------------------------------------------------------------------------------
class Obs:
    def __init__(self, batch):
        self.batch = batch
        self.images, self.img_masks = policy.prepare_images(batch)
        self.lang, self.lmask = batch[OBS_LANGUAGE_TOKENS], batch[OBS_LANGUAGE_ATTENTION_MASK]
        self.state = policy.prepare_state(batch)  # normalized, padded to max_state_dim
        g = torch.Generator(device="cpu").manual_seed(0)
        self.noise = torch.randn(1, H, cfg.max_action_dim, generator=g).to(dev)
        self.imglang_cache()

    def prefix(self, state):
        embs, pad, att = m.embed_prefix(self.images, self.img_masks, self.lang, self.lmask, state=state)
        return embs, pad, make_att_2d_masks(pad, att), torch.cumsum(pad, dim=1) - 1

    def stock_cache(self, state):
        embs, pad, att2d, pos = self.prefix(state)
        _, cache = vwe.forward(attention_mask=att2d, position_ids=pos, inputs_embeds=[embs, None], use_cache=True)
        return cache, pad

    def imglang_cache(self):  # everything but the state token; no grad, computed once per call
        with torch.no_grad():
            embs, pad, att2d, pos = self.prefix(self.state)
            _, cache = vwe.forward(attention_mask=att2d[:, :-1, :-1], position_ids=pos[:, :-1],
                                   inputs_embeds=[embs[:, :-1], None], use_cache=True)
        self.il, self.pad, self.pos_s, self.L = cache, pad, pos[:, -1:], pos.shape[1] - 1

    def split_cache(self, state):  # state token through the VLM layers against the cached image+language K/V
        B = state.shape[0]
        h = m.state_proj(state)[:, None, :]
        pos = self.pos_s.expand(B, -1)  # stock position id of the state token: cumsum(pad) - 1
        mask = self.pad[:, None, :].expand(B, -1, -1)  # padded prefix tokens stay masked, as in stock
        cache = DynamicCache()
        for i, layer in enumerate(vwe.get_vlm_model().text_model.layers[: vwe.num_vlm_layers]):
            x = layer.input_layernorm(h).to(layer.self_attn.q_proj.weight.dtype)
            shp = (B, 1, -1, layer.self_attn.head_dim)
            q = apply_rope(layer.self_attn.q_proj(x).view(shp), pos)
            k = apply_rope(layer.self_attn.k_proj(x).view(shp), pos)
            v = layer.self_attn.v_proj(x).view(shp)
            K = torch.cat([self.il.layers[i].keys.transpose(1, 2).expand(B, -1, -1, -1), k], 1)
            V = torch.cat([self.il.layers[i].values.transpose(1, 2).expand(B, -1, -1, -1), v], 1)
            att = vwe.eager_attention_forward(mask, B, layer.self_attn.head_dim, q, K, V)
            dt = layer.self_attn.o_proj.weight.dtype  # the stock forward adds the residual in place, in dt
            h = (layer.self_attn.o_proj(att.to(dt)) + h).to(dt)
            h = layer.mlp(layer.post_attention_layernorm(h)) + h
            cache.update(K.transpose(1, 2), V.transpose(1, 2), i)
        return cache, self.pad.expand(B, -1)

    def flow(self, cache, pad, grad_steps=NS):  # T_k: gradient through the last k flow steps only
        x, dt = self.noise.expand(pad.shape[0], -1, -1), -1.0 / NS
        for step in range(NS):
            t = torch.full((pad.shape[0],), 1.0 + step * dt, device=dev)
            with torch.set_grad_enabled(torch.is_grad_enabled() and step >= NS - grad_steps):
                x = x + dt * m.denoise_step(prefix_pad_masks=pad, past_key_values=cache, x_t=x, timestep=t)
        return x[..., :A]  # normalized actions, (B, H, A)

    def a(self, s8, split=True, k=NS):  # s8: normalized state, (B, SD)
        s = torch.nn.functional.pad(s8, (0, cfg.max_state_dim - SD))
        return self.flow(*(self.split_cache(s) if split else self.stock_cache(s)), grad_steps=k)


def jac_rev(o, split=True, k=NS, npos=1, mem_key=None):  # 7*npos VJPs from one batch-1 graph, (npos, A, SD)
    s8 = o.state[:, :SD].clone().requires_grad_(True)
    out = o.a(s8, split, k)[0, :npos].reshape(-1)
    if mem_key:
        emit(mem_key, gib())  # the graph is held here: this is the peak for reverse mode
    rows = [torch.autograd.grad(out[i], s8, retain_graph=i < len(out) - 1)[0][0] for i in range(len(out))]
    if mem_key:
        emit(mem_key + "_after_vjps", gib())  # allocator high-water mark since the last free()
    return torch.stack(rows).view(npos, A, SD)


def jac_vmap(o, k=NS):  # the 7 VJPs of a[0] as one vmapped backward: ~7x fewer kernel launches than the loop
    s8 = o.state[:, :SD].clone().requires_grad_(True)
    out = o.a(s8, True, k)[0, 0]
    return torch.autograd.grad(out, s8, torch.eye(A, device=dev), is_grads_batched=True)[0][:, 0]


def jac_fwd(o):  # all H chunk positions from one batch-8 forward-mode pass; forward AD is missing on MPS
    with fwAD.dual_level():
        s8 = fwAD.make_dual(o.state[:, :SD].expand(SD, -1).contiguous(), torch.eye(SD, device=dev))
        return fwAD.unpack_dual(o.a(s8)).tangent.permute(1, 2, 0)  # (H, A, SD)


def jac_fd(o, eps):
    s8 = o.state[:, :SD]
    with torch.no_grad():
        cols = [(o.a(s8 + eps * e) - o.a(s8 - eps * e))[0, 0] / (2 * eps) for e in torch.eye(SD, device=dev)[:, None]]
    return torch.stack(cols, 1)


raw = lambda J: std_a[:, None] * J / std_s[None, :]  # d a_raw / d s_raw
rel = lambda X, Y: float((X - Y).norm() / Y.norm())
nograd = lambda f: torch.no_grad()(f)

# --- reach the measured state: rollout with 10-step chunks, as the maintainers advise for LIBERO -------------
torch.manual_seed(0)  # the rollout samples flow noise on every call; same seed -> same measured state
obs, _ = env.reset(seed=0)
policy.reset()
for t in range(args.state):
    with torch.no_grad():
        act = post(policy.select_action(to_batch(obs)))
    obs, _, term, _, _ = env.step(env_post({ACTION: act})[ACTION].cpu().numpy())
    if term[0]:
        raise SystemExit(f"episode ended at step {t}")
o = Obs(to_batch(obs))
free()
emit("meta", {"suite": args.suite, "task": args.task, "chunk": H, "flow_steps": NS, "vlm_layers": vwe.num_vlm_layers,
              "prefix_tokens": o.L + 1, "params_M": round(sum(q.numel() for q in policy.parameters()) / 1e6, 1),
              "state_raw": [round(float(x), 4) for x in o.state[0, :SD] * std_s + mean_s]})

if args.mode == "mem":
    with torch.no_grad():
        policy.predict_action_chunk(o.batch)
    emit("mem_after_call_gib", gib())
    free()
    jac_rev(o, mem_key="mem_J_a0_split_graph_gib")
    free()
    jac_rev(o, npos=10, mem_key="mem_J_10pos_split_graph_gib")
    free()
    jac_rev(o, split=False, mem_key="mem_J_a0_stock_graph_gib")
    free()
    raise SystemExit

if args.mode in ("visual", "arm", "e1"):  # re-render the scene from an edited MuJoCo state
    import mujoco

    le = env.envs[0].unwrapped
    ctrl, sim, G_POS = le._env, le._env.env.sim, 0.011  # LIBERO OffScreenRenderEnv; m per action unit per step
    rob, name = ctrl.env.robots[0], ctrl.env.obj_of_interest[0]
    qa, va = np.array(rob._ref_joint_pos_indexes), np.array(rob._ref_joint_vel_indexes)
    jnt = ctrl.env.objects_dict[name].joints[0]  # free joint of the object of interest: qpos pos(3)+quat(4), qvel 6
    adr, vadr = sim.model.get_joint_qpos_addr(jnt)[0], sim.model.get_joint_qvel_addr(jnt)[0]
    batch_axis = lambda d: {k: batch_axis(v) for k, v in d.items()} if isinstance(d, dict) else np.asarray(d)[None]

    def render(st):  # -> (policy batch, raw LIBERO obs, ms); st is a flattened state [time, qpos, qvel]
        t0 = time.perf_counter()
        raw_obs = ctrl.regenerate_obs_from_state(st)
        t_r = 1e3 * (time.perf_counter() - t0)
        return to_batch(batch_axis(le._format_raw_obs(raw_obs))), raw_obs, t_r

    def chunk(img_b, state_b):  # raw-unit chunk from the images of one batch and the state of another; o's noise
        imgs, masks = policy.prepare_images(img_b)
        with torch.no_grad():
            ch = m.sample_actions(imgs, masks, o.lang, o.lmask, policy.prepare_state(state_b), noise=o.noise)
        return ch[0, :, :A] * std_a

    def move_arm(s_from, dxy):  # arm qpos that shift the EEF site by dxy metres, orientation held (Gauss-Newton IK)
        st = s_from.copy()
        ctrl.set_state(st), sim.forward()
        target = sim.data.site_xpos[rob.eef_site_id] + np.r_[dxy, 0.0]
        for _ in range(6):
            jp, jr = np.zeros((3, sim.model.nv)), np.zeros((3, sim.model.nv))
            mujoco.mj_jacSite(sim.model._model, sim.data._data, jp, jr, rob.eef_site_id)
            err = np.r_[target - sim.data.site_xpos[rob.eef_site_id], 0.0, 0.0, 0.0]
            st[1 + qa] += np.linalg.pinv(np.vstack([jp, jr])[:, va]) @ err
            ctrl.set_state(st), sim.forward()
        return st, float(np.linalg.norm(target - sim.data.site_xpos[rob.eef_site_id]))

    def move_obj(s_from, dxyz):  # object of interest shifted by dxyz metres, its velocity zeroed
        st = s_from.copy()
        st[1 + adr: 4 + adr] += dxyz
        st[1 + sim.model.nq + vadr: 1 + sim.model.nq + vadr + 6] = 0.0
        return st

    unit = lambda v: v / (v.norm() + 1e-12)
    cos = lambda x, y: round(float(unit(x.flatten()) @ unit(y.flatten())), 3)
    if args.mode != "e1":
        s0 = ctrl.get_sim_state()
        b0, raw0, _ = render(s0)
        base = chunk(b0, b0)

if args.mode == "visual":  # Q7: d action along "object of interest moved by delta"
    p0 = raw0[f"{name}_pos"].copy()
    emit("visual_meta", {"object": name, "object_pos": [round(float(x), 4) for x in p0],
                         "segmentation_keys": [k for k in raw0 if "segment" in k]})
    for axis in (0, 1):
        for d_m in (0.01, 0.02, 0.05):
            for sign in (1, -1):
                u = np.zeros(3)
                u[axis] = sign * d_m
                b, raw_u, t_r = render(move_obj(s0, u))
                t0 = time.perf_counter()
                da = chunk(b, b0) - base  # images change, the arm state does not
                t_c = 1e3 * (time.perf_counter() - t0)
                disp = G_POS * da[:10, :3].sum(0).cpu().numpy()  # extra commanded arm displacement over 10 steps
                emit("visual_fd", {"axis": "xy"[axis], "delta_cm": sign * d_m * 100,
                                   "obj_moved_cm": round(float(np.linalg.norm(raw_u[f"{name}_pos"] - p0)) * 100, 2),
                                   "da0_per_cm": [round(float(x) / (d_m * 100), 4) for x in da[0]],
                                   "da0_norm": round(float(da[0].norm()), 4),
                                   "arm_follow_10steps_cm": [round(float(x) * 100, 2) for x in disp],
                                   "cos_follow": round(float(disp @ u / (np.linalg.norm(disp) * d_m + 1e-12)), 3),
                                   "ms_render": round(t_r, 1), "ms_call": round(t_c, 1)})
    raise SystemExit

if args.mode == "arm":  # does SmolVLA read the arm from the state vector or from the pixels?
    s_state0 = b0[OBS_STATE]
    for dxy in ((0.01, 0), (-0.01, 0), (0, 0.01), (0, -0.01)):
        b1, raw1, _ = render(move_arm(s0, np.array(dxy, dtype=float))[0])
        d_full, d_state, d_img = chunk(b1, b1) - base, chunk(b0, b1) - base, chunk(b1, b0) - base
        ds_raw = (b1[OBS_STATE] - s_state0)[0] * std_s  # normalized -> raw
        emit("arm_shift", {
            "dxy_cm": [100 * x for x in dxy],
            "eef_moved_cm": [round(float(x) * 100, 2) for x in raw1["robot0_eef_pos"] - raw0["robot0_eef_pos"]],
            "state_delta_raw": [round(float(x), 4) for x in ds_raw],
            "a0_norm": {"full": round(float(d_full[0].norm()), 4), "state_only": round(float(d_state[0].norm()), 4),
                        "image_only": round(float(d_img[0].norm()), 4)},
            "a10_norm": {"full": round(float(d_full[:10].norm()), 4), "state_only": round(float(d_state[:10].norm()), 4),
                         "image_only": round(float(d_img[:10].norm()), 4)},
            "cos_a10": {"full_state": cos(d_full[:10], d_state[:10]), "full_image": cos(d_full[:10], d_img[:10])},
            "additivity_resid": round(float((d_full[:10] - d_state[:10] - d_img[:10]).norm() / (d_full[:10].norm() + 1e-12)), 3),
        })
    raise SystemExit

if args.mode == "e1":  # C1-E1, docs/superpowers/specs/2026-09-27-c1-e1-offline-design.md: one rollout, its 8 states
    SUITES = ["libero_spatial", "libero_object", "libero_goal", "libero_10"]
    MAGS_MM = (1, 2, 5, 10, 20, 50)
    v = lambda x: [round(float(y), 6) for y in x]
    torch.manual_seed(1000 + 10 * SUITES.index(args.suite) + args.task)
    le.init_state_id = args.init_state  # LiberoEnv resets to init_states[init_state_id]
    obs, _ = env.reset(seed=0)
    policy.reset()
    sim, rob = ctrl.env.sim, ctrl.env.robots[0]  # a hard reset rebuilds the MjSim: rebind everything taken from it
    qa, va = np.array(rob._ref_joint_pos_indexes), np.array(rob._ref_joint_vel_indexes)
    name = ctrl.env.obj_of_interest[0]
    jnt = ctrl.env.objects_dict[name].joints[0]
    adr, vadr = sim.model.get_joint_qpos_addr(jnt)[0], sim.model.get_joint_qvel_addr(jnt)[0]
    sims, grips, last_g = [], [], -1.0  # state before each step; gripper command executed before it (-1 = open)
    for t in range(le._max_episode_steps):
        sims.append(ctrl.get_sim_state())
        grips.append(last_g)
        with torch.no_grad():
            act = env_post({ACTION: post(policy.select_action(to_batch(obs)))})[ACTION].cpu().numpy()
        last_g = float(act[0, 6])
        obs, _, term, _, _ = env.step(act)
        if term[0]:
            break
    T, success = t + 1, bool(ctrl.check_success())
    emit("e1_rollout", {"suite": args.suite, "task": args.task, "init_state": args.init_state, "T": T,
                        "max_steps": le._max_episode_steps, "success": success, "object": name})
    for k in (int(x) for x in args.ks.split(",")):
        tk = k * T // 8
        s_k = sims[tk]
        b0, raw0, t_r0 = render(s_k)
        o = Obs(b0)  # chunk() and jac_vmap() read the current o
        t0 = time.perf_counter()
        a = chunk(b0, b0)
        ms_call = [1e3 * (time.perf_counter() - t0)]
        jac_vmap(o)  # warm-up, then the timed one
        t0 = time.perf_counter()
        J_raw = raw(jac_vmap(o))  # (A, SD): d a0_raw / d s_raw, exact, fp32
        sync()
        ms_J = 1e3 * (time.perf_counter() - t0)
        ops = {"call": cost(lambda: chunk(b0, b0)), "J": cost(lambda: jac_vmap(o))} if k == int(args.ks.split(",")[0]) else None
        free()
        closed = grips[tk] > 0
        dist = float(np.linalg.norm(raw0[f"{name}_to_robot0_eef_pos"]))
        stage = "closed" if closed else ("near" if dist <= 0.10 else "far")
        carry = closed and float(np.linalg.norm(raw0[f"{name}_pos"] - raw0["robot0_eef_pos"])) <= 0.06
        arm, objs, ms_render = [], [], [t_r0]
        for ax in (0, 1):
            for sg in (1, -1):
                for mm in MAGS_MM:
                    dxy = np.zeros(2)
                    dxy[ax] = sg * mm / 1000
                    st, ik_err = move_arm(s_k, dxy)
                    rec = {"ax": ax, "sg": sg, "mm": mm, "ik_err_mm": round(1e3 * ik_err, 3)}
                    if ik_err > 5e-4:  # spec §4: shift rejected
                        arm.append(rec | {"ik_fail": True})
                        continue
                    if carry:  # spec §4: a held object moves with the hand
                        st[1 + adr: 3 + adr] += dxy
                    b1, raw1, t_r = render(st)
                    t0 = time.perf_counter()
                    dF = chunk(b1, b1) - a
                    ms_call.append(1e3 * (time.perf_counter() - t0))
                    ms_render.append(t_r)
                    ds = (b1[OBS_STATE] - b0[OBS_STATE])[0] * std_s
                    rec |= {"dF0": v(dF[0]), "S": {h: v(dF[:h, :3].sum(0)) for h in (10, 25, 50)}, "ds": v(ds),
                            "Cs": v(J_raw @ ds)}
                    if mm in (10, 20):
                        dS, dI = chunk(b0, b1) - a, chunk(b1, b0) - a
                        rec |= {"dS0": v(dS[0]), "dI0": v(dI[0]), "nFS10": round(float((dF[:10] - dS[:10]).norm()), 6),
                                "nFI10": round(float((dF[:10] - dI[:10]).norm()), 6)}
                    arm.append(rec)
                    if closed:  # spec §4: no object shifts while it is held
                        continue
                    b1, raw1, t_r = render(move_obj(s_k, np.r_[dxy, 0.0]))
                    dO = chunk(b1, b1) - a
                    objs.append({"ax": ax, "sg": sg, "mm": mm, "dO0": v(dO[0]),
                                 "S": {h: v(dO[:h, :3].sum(0)) for h in (10, 25, 50)},
                                 "moved_mm": round(1e3 * float(np.linalg.norm(raw1[f"{name}_pos"] - raw0[f"{name}_pos"])), 3)})
        emit("e1_state", {"suite": args.suite, "task": args.task, "k": k, "t": tk, "T": T, "stage": stage,
                          "dist_m": round(dist, 4), "carry": carry, "a0": v(a[0]), "J_raw": [v(r) for r in J_raw],
                          "arm": arm, "obj": objs, "ops": ops,
                          "ms": {"call": round(float(np.median(ms_call)), 1), "render": round(float(np.median(ms_render)), 1),
                                 "J": round(ms_J, 1)}})
        free()
    raise SystemExit

# --- timing and cost of one call and its parts ---------------------------------------------------------------
if args.mode == "bench":  # "check" skips timing: split path and J correctness only, meant for --dtype float32
    with torch.no_grad():
        cache_pad = o.split_cache(o.state)
    ms = {
        "call_stock": nograd(lambda: policy.predict_action_chunk(o.batch)),
        "prefix_stock": nograd(lambda: o.stock_cache(o.state)),
        "imglang_cache": o.imglang_cache,
        "state_token_pass": nograd(lambda: o.split_cache(o.state)),
        "flow_10_steps": nograd(lambda: o.flow(*cache_pad)),
        "J_a0_exact_stock_path": lambda: jac_rev(o, split=False),
        "J_a0_exact_split": lambda: jac_rev(o),
        **{f"J_a0_T{k}_split": (lambda k=k: jac_rev(o, k=k)) for k in (1, 3, 5)},
        "J_10pos_T3_split": lambda: jac_rev(o, k=3, npos=10),
    }
    if dev.type != "mps":
        ms["J_all_pos_fwd"] = lambda: jac_fwd(o)
    ms["J_a0_exact_vmap_vjp"] = lambda: jac_vmap(o)
    ms["J_a0_T3_vmap_vjp"] = lambda: jac_vmap(o, k=3)
    for name, fn in ms.items():
        try:
            emit(f"ms_{name}", timeit(fn))
        except Exception as e:  # vmap over backward is not supported by every MPS kernel
            emit(f"ms_{name}", f"failed: {type(e).__name__}: {str(e)[:120]}")
    prev = policy.predict_action_chunk(o.batch)[:, 10:]  # leftover of a previous chunk after 10 executed steps
    policy.config.rtc_config = RTCConfig(execution_horizon=10)
    policy.init_rtc_processor()
    # LeRobot's RTC takes v_t before x_t requires grad, so its guidance has no VJP through the denoiser
    emit("ms_call_rtc_lerobot_d3", timeit(nograd(lambda: policy.predict_action_chunk(o.batch, inference_delay=3, prev_chunk_left_over=prev))))
    policy.config.rtc_config = None
    policy.init_rtc_processor()
    for name in ("call_stock", "J_a0_exact_split", "J_a0_T3_split"):
        emit(f"cost_{name}", cost(ms[name]))

# --- J quality at this state ---------------------------------------------------------------------------------
with torch.no_grad():
    stock = m.sample_actions(o.images, o.img_masks, o.lang, o.lmask, o.state, noise=o.noise)[..., :A]
    emit("split_vs_stock_max_abs", float((o.a(o.state[:, :SD]) - stock).abs().max()))
J = jac_rev(o)[0]
emit("J_rev_vs_fd", round(rel(J, jac_fd(o, 1e-2 if args.dtype != "float32" else 1e-3)), 4))
emit("Tk_vs_exact", {f"T{k}": round(rel(jac_rev(o, k=k)[0], J), 3) for k in (1, 3, 5)})
try:  # one vmapped backward instead of 7: same J, fewer kernel launches?
    emit("J_vmap_vs_loop", round(rel(jac_vmap(o), J), 5))
    emit("ms_J_a0_loop_vs_vmap", {"loop": timeit(lambda: jac_rev(o), 3), "vmap": timeit(lambda: jac_vmap(o), 3)})
except Exception as e:
    emit("J_vmap_vs_loop", f"failed: {type(e).__name__}: {str(e)[:120]}")
if dev.type != "mps":
    emit("J_fwd_vs_rev", round(rel(jac_fwd(o)[0], J), 5))
Jr = raw(J)
emit("J_raw_pos_block_per_cm", [[round(float(x) * 0.01, 3) for x in row] for row in Jr[:3, :3]])
emit("J_raw_full", [[round(float(x), 3) for x in row] for row in Jr])
emit("J_norm_fro", round(float(J.norm()), 3))
J10 = jac_rev(o, npos=10)
emit("J_fro_by_position", {j: round(float(J10[j].norm()), 3) for j in (0, 1, 3, 6, 9)})
rng = np.random.default_rng(0)
lin = {}
for d_m in (0.002, 0.005, 0.01, 0.02):  # position offsets in metres, 4 random directions each
    r, n_da, n_jds = [], [], []
    for _ in range(4):
        u = torch.tensor(rng.standard_normal(3), dtype=torch.float32, device=dev)
        ds = torch.zeros(SD, device=dev)
        ds[:3] = d_m * u / u.norm() / std_s[:3]
        with torch.no_grad():
            da = (o.a(o.state[:, :SD] + ds) - o.a(o.state[:, :SD]))[0, 0]
        r.append(float((da - J @ ds).norm() / da.norm().clamp_min(1e-12)))
        n_da.append(float(da.norm()))
        n_jds.append(float((J @ ds).norm()))
    lin[f"{d_m * 100:g}cm"] = {"resid": round(float(np.median(r)), 3), "da_norm": round(float(np.median(n_da)), 4),
                               "Jds_norm": round(float(np.median(n_jds)), 4)}
emit("lin_residual_pos", lin)
