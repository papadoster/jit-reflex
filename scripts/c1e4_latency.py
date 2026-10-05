"""C1-E4 part 2, item 5: the brain's answer time for ONE row per call after warm-up, and the reflex's own time per step,
so the delay axis d can be tied to hardware. Report only (no rule reads it).

One LIBERO observation (libero_spatial task 0, init 0, after reset), the brain as scripts/c1e4_run.py builds it (pinned
revision, compile off, fp32 or bf16), WARM untimed calls, then N timed calls of one row each (preprocessing, the
chunk, postprocessing and the copy back to the CPU, as the runner's brain time). The reflex: a G-auto GAgent on the
call conveyor fed a moving object for STEPS steps (observe + act per step, numpy only). One JSON line to stdout:
median, P10, P90 in ms and in 20 Hz steps.
    MUJOCO_GL=cgl ~/Desktop/M2R-c1-env/bin/python scripts/c1e4_latency.py --brain pi05 --device mps
    Pod: MUJOCO_GL=egl ... --device cuda --precision fp32 | bf16
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gauto  # noqa: E402
import objreflex as orx  # noqa: E402
from lerobot.configs.policies import PreTrainedConfig  # noqa: E402
from lerobot.envs.configs import LiberoEnv as LiberoCfg  # noqa: E402
from lerobot.envs.libero import LiberoEnv, _get_suite  # noqa: E402
from lerobot.envs.utils import preprocess_observation  # noqa: E402
from lerobot.policies.factory import make_policy, make_pre_post_processors  # noqa: E402

p = argparse.ArgumentParser()
p.add_argument("--brain", choices=sorted(gauto.BRAINS), required=True)
p.add_argument("--precision", choices=["fp32", "bf16"], default="fp32")
p.add_argument("--device", default="cuda")
p.add_argument("--warm", type=int, default=3)
p.add_argument("--n", type=int, default=20)
p.add_argument("--steps", type=int, default=20000, help="reflex steps timed")
args = p.parse_args()
dev = torch.device(args.device)
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

repo, rev = gauto.BRAINS[args.brain]
path = snapshot_download(repo, revision=rev)
cfg = PreTrainedConfig.from_pretrained(path)
cfg.pretrained_path, cfg.device, cfg.compile_model = path, args.device, False
if hasattr(cfg, "load_vlm_weights"):
    cfg.load_vlm_weights = False
if args.precision == "bf16":
    cfg.dtype = "bfloat16"
env_cfg = LiberoCfg(task="libero_spatial", task_ids=[0])
policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval().requires_grad_(False)
if args.precision == "fp32":
    policy = policy.float()
pre, post = make_pre_post_processors(cfg, path, preprocessor_overrides={"device_processor": {"device": args.device}})
env_pre, _ = env_cfg.get_env_processors()

env = LiberoEnv(task_suite=_get_suite("libero_spatial"), task_id=0, task_suite_name="libero_spatial",
                obs_type="pixels_agent_pos", observation_width=360, observation_height=360)
env.init_state_id = 0
obs, _ = env.reset(seed=0)
desc = env.task_description
batched = lambda x: {k: batched(v) for k, v in x.items()} if isinstance(x, dict) else np.asarray(x)[None]  # noqa: E731
batch = batched(obs)
gen = torch.Generator().manual_seed(0)


def call():
    b = preprocess_observation(batch)
    b["task"] = [desc]
    z = torch.randn(1, orx.H, cfg.max_action_dim, generator=gen).to(dev)
    tb = time.perf_counter()
    with torch.no_grad():
        ch = policy.predict_action_chunk(pre(env_pre(b)), noise=z)
    post(ch).float().cpu().numpy()
    return time.perf_counter() - tb


for _ in range(args.warm):
    call()
ms = np.array([1000 * call() for _ in range(args.n)])

ag = gauto.GAgent("Gauto", 10, 0, kbar=0.157)
chunk = np.zeros((orx.H, 7))
rng = np.random.default_rng(0)
tr = time.perf_counter()
for t in range(args.steps):
    obj = np.array([0.1, 0.0, 0.9]) + (0.03 if t % 280 > 30 else 0.0)  # a step shift every episode-length
    if t % 280 == 0:
        ag = gauto.GAgent("Gauto", 10, 0, kbar=0.157)
    k = t % 280
    ag.observe(k, np.array([0.1, 0.05, 1.0]) + 1e-3 * rng.normal(size=3), obj)
    if ag.sched.arrive(k):
        ag.on_new_chunk()
    if ag.sched.wants_call(k, ag.trigger(k)):
        ag.sched.issue(k, chunk)
        if ag.sched.arrive(k):
            ag.on_new_chunk()
    ag.act(k, ag.sched.action(k))
us = 1e6 * (time.perf_counter() - tr) / args.steps

q = lambda v: [round(float(np.percentile(v, x)), 1) for x in (50, 10, 90)]  # noqa: E731
print(json.dumps({"brain": args.brain, "precision": args.precision, "device": args.device,
                  "torch": torch.__version__, "calls": args.n, "ms_med_p10_p90": q(ms),
                  "steps20hz_med_p10_p90": [round(x / 50, 2) for x in q(ms)], "reflex_us_per_step": round(us, 1)}))
