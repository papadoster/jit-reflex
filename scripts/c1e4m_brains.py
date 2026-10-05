"""C1-E4 part 2: the VLA brains on RAW LIBERO observations (the robosuite dict LIBERO-MAX's env returns:
agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, ...), answering with chunks (B, H, 7) in LIBERO env units
(raw delta EEF, gripper > 0 closes), as scripts/c1e4_run.py's brain() (plan
docs/superpowers/plans/2026-10-05-c1-e4-part2.md, Task 8).

    brain = make_brain(name, device, precision)       # pi05, smolvla (LeRobot), oft, oftplus (openvla-oft, pod only)
    chunks = brain(raw_obs_list, robosuite_envs, instructions, seeds, suites)

- robosuite_envs: per row an env with .robots, or that robot's controller.ee_ori_mat (3x3, robot_state_of(env)), so
  workers can send the matrix instead of a live env. Render size (RES): 360 px for LeRobot (part 1), 256 for OFT
  (LIBERO-MAX); every call asserts it.

- pi05 / smolvla: the raw observation goes through LeRobot's own LiberoEnv._format_raw_obs (format_raw_obs), so the
  rest is part 1's pipeline unchanged: preprocess_observation, the LIBERO env processor (180-degree image flip, state =
  eef pos + axis-angle + gripper qpos), the policy's pre/post processors, the pinned revisions of gauto.BRAINS. Flow
  noise per row from torch.Generator().manual_seed(seed); LIBERO-MAX's runners use seed = policy_seed + policy_step.
  fp32 batched, or (pi0.5 only) bf16 one row per call (part 1, spec 2026-10-03 §4.1).
- oft / oftplus: openvla-oft (MIT, github.com/moojink/openvla-oft) from its own venv, imported inside the builder;
  the steps of its README and of experiments/robot/libero/run_libero_eval.py's prepare_observation / process_action,
  built from its utils (not its run loop): 180-degree flip of both images, the RLDS resize to 224 (JPEG round trip,
  Lanczos), center crop 0.9, proprio = eef pos + axis-angle + gripper qpos, the suite's unnorm key, then the gripper
  [0, 1] (1 = open) -> -1 open / +1 close, binarized. bf16, one row per call, H = 8. The L1 action head is
  deterministic: seeds are not used. The suite (case["task_suite_name"]) picks the unnorm key: OFT names them
  libero_*_no_noops, OFT+ libero_* (the same statistics under all four). LIBERO-MAX renders OFT's env at 256 px.
  Env: OPENVLA_OFT_ROOT (the openvla-oft checkout, put on sys.path if set), C1E4_OFT_CKPT (checkpoint parent dir,
  default ~/oft-ckpt; the snapshot is a local_dir copy without lora_adapter/, because openvla-oft's local-checkpoint
  loader get_vla rewrites config.json in place on load, keeping config.json.back.*: a SHA check of the snapshot covers
  the weight files only, or runs before the first load). cuDNN deterministic, no benchmark (as LIBERO-MAX).

Pod self-test (one call on one LIBERO observation, prints shape and ranges):
    MUJOCO_GL=egl python scripts/c1e4m_brains.py --selftest-oft oft --device cuda
"""

import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import gauto  # noqa: E402
import objreflex as orx  # noqa: E402

OFT = {"oft": ("moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10",
               "638918f3d1c2e43a39a8a20772bdb8b91835e4b7"),
       "oftplus": ("Sylvest/openvla-7b-oft-finetuned-libero-plus-mixdata", "a85655ec941bae6644c9fbdf62db02b9726d7cf5")}
NAMES = sorted(gauto.BRAINS) + sorted(OFT)
RES = {"pi05": 360, "smolvla": 360, "oft": 256, "oftplus": 256}  # render size: part 1's, LIBERO-MAX's OFT
CAMS = ("agentview_image", "robot0_eye_in_hand_image")  # LiberoEnv's default camera_name
CAM_MAP = {"agentview_image": "image", "robot0_eye_in_hand_image": "image2"}  # LiberoEnv's default mapping


def robot_state_of(env):
    """What format_raw_obs reads of an env (its controller's EEF rotation matrix), to send instead of the env."""
    return env.robots[0].controller.ee_ori_mat.copy()


def _check_res(name, raw_obs_list):
    assert all(o["agentview_image"].shape[:2] == (RES[name],) * 2 for o in raw_obs_list), \
        f"{name} takes {RES[name]} px renders, got {[o['agentview_image'].shape for o in raw_obs_list]}"


def format_raw_obs(raw_obs, robosuite_env, camera_names=CAMS):
    """LeRobot LiberoEnv(obs_type="pixels_agent_pos")'s observation of a raw LIBERO observation: LeRobot's own
    _format_raw_obs on a stand-in for the LiberoEnv (it reads these four attributes). robosuite_env: anything with
    .robots (OffScreenRenderEnv, its inner env, LIBERO-MAX's wrapper; its controller gives eef mat), or that matrix
    (robot_state_of(env))."""
    from lerobot.envs.libero import LiberoEnv
    if not hasattr(robosuite_env, "robots"):  # the 3x3 ee_ori_mat
        robosuite_env = SimpleNamespace(robots=[SimpleNamespace(controller=SimpleNamespace(ee_ori_mat=robosuite_env))])
    shim = SimpleNamespace(camera_name=list(camera_names), camera_name_mapping=CAM_MAP, obs_type="pixels_agent_pos",
                           _env=robosuite_env)
    return LiberoEnv._format_raw_obs(shim, raw_obs)


def _stack(xs):
    """(Nested) observations stacked along a new batch axis, as a vector env batches them."""
    return {k: _stack([x[k] for x in xs]) for k in xs[0]} if isinstance(xs[0], dict) else np.stack(xs)


class LeRobotBrain:
    """pi0.5 / SmolVLA as scripts/c1e4_run.py builds and calls them."""

    def __init__(self, name, device, precision):
        import torch
        from huggingface_hub import snapshot_download
        from lerobot.configs.policies import PreTrainedConfig
        from lerobot.envs.configs import LiberoEnv as LiberoCfg
        from lerobot.envs.utils import preprocess_observation
        from lerobot.policies.factory import make_policy, make_pre_post_processors
        assert precision in ("fp32", "bf16"), precision
        assert precision == "fp32" or name == "pi05", "bf16 is pi0.5's fallback only (spec §4.1)"
        torch.backends.cuda.matmul.allow_tf32 = False  # literal fp32 on CUDA (spec §4.1)
        torch.backends.cudnn.allow_tf32 = False
        self.name, self.precision, self.dev = name, precision, torch.device(device)
        self.torch, self.preprocess = torch, preprocess_observation
        repo, rev = gauto.BRAINS[name]
        path = snapshot_download(repo, revision=rev)
        cfg = PreTrainedConfig.from_pretrained(path)
        cfg.pretrained_path, cfg.device, cfg.compile_model = path, device, False
        if hasattr(cfg, "load_vlm_weights"):  # SmolVLA only
            cfg.load_vlm_weights = False
        if precision == "bf16":
            cfg.dtype = "bfloat16"
        assert cfg.chunk_size == orx.H, cfg.chunk_size
        env_cfg = LiberoCfg(task="libero_spatial", task_ids=[0])
        policy = make_policy(cfg=cfg, env_cfg=env_cfg).eval().requires_grad_(False)
        if precision == "fp32":
            policy = policy.float()
            if self.dev.type == "cuda":
                torch.cuda.empty_cache()
        self.pre, self.post = make_pre_post_processors(
            cfg, path, preprocessor_overrides={"device_processor": {"device": device}})
        self.env_pre, _ = env_cfg.get_env_processors()
        self.policy, self.cfg, self.H = policy, cfg, cfg.chunk_size

    def __call__(self, raw_obs_list, robosuite_envs, instructions, seeds, suites=None):
        _check_res(self.name, raw_obs_list)
        if self.precision == "bf16" and len(seeds) > 1:  # a bf16 chunk depends on its batch: one row per call
            return np.concatenate([self([o], [e], [i], [s]) for o, e, i, s in
                                   zip(raw_obs_list, robosuite_envs, instructions, seeds)])
        torch = self.torch
        b = self.preprocess(_stack([format_raw_obs(o, e) for o, e in zip(raw_obs_list, robosuite_envs)]))
        b["task"] = list(instructions)
        noise = torch.cat([torch.randn(1, self.H, self.cfg.max_action_dim,
                                       generator=torch.Generator().manual_seed(int(s))) for s in seeds])
        with torch.no_grad():
            ch = self.policy.predict_action_chunk(self.pre(self.env_pre(b)), noise=noise.to(self.dev))
        return self.post(ch).float().cpu().numpy()


class OFTBrain:
    """OpenVLA-OFT / OFT+ per openvla-oft's README quick start and LIBERO eval (pod, openvla-oft's venv)."""

    H = 8

    def __init__(self, name, device, precision):
        import torch
        assert precision == "bf16", "openvla-oft runs in bf16"
        if os.environ.get("OPENVLA_OFT_ROOT"):
            sys.path.insert(0, os.environ["OPENVLA_OFT_ROOT"])
        torch.backends.cudnn.deterministic = True  # as LIBERO-MAX's runners
        torch.backends.cudnn.benchmark = False
        from experiments.robot.libero.libero_utils import get_libero_image, get_libero_wrist_image, quat2axisangle
        # openvla_utils runs json_numpy.patch() on import: in this process json.dumps of numpy scalars and arrays stops
        # raising, so a slip passes silently; the runner must cast record fields to Python types itself
        from experiments.robot.openvla_utils import (DEVICE, OPENVLA_IMAGE_SIZE, get_action_head, get_processor,
                                                     get_proprio_projector, get_vla, get_vla_action,
                                                     resize_image_for_policy)
        from experiments.robot.robot_utils import invert_gripper_action, normalize_gripper_action
        from huggingface_hub import snapshot_download
        from prismatic.vla.constants import NUM_ACTIONS_CHUNK, PROPRIO_DIM
        assert NUM_ACTIONS_CHUNK == self.H and PROPRIO_DIM == 8, (NUM_ACTIONS_CHUNK, PROPRIO_DIM)
        assert DEVICE.type == torch.device(device).type, f"openvla-oft runs on {DEVICE}, asked {device}"
        self.name, self.torch = name, torch
        repo, sha = OFT[name]
        root = Path(os.environ.get("C1E4_OFT_CKPT", "~/oft-ckpt")).expanduser().resolve()
        # absolute path, no dots: openvla-oft takes it as local (not a hub id), transformers imports its modeling code
        path = snapshot_download(repo, revision=sha, local_dir=str(root / f"{repo.split('/')[1]}-{sha[:10]}"),
                                 ignore_patterns=["lora_adapter/*"])
        self.cfg = SimpleNamespace(pretrained_checkpoint=path, use_l1_regression=True, use_diffusion=False,
                                   use_film=False, num_images_in_input=2, use_proprio=True, center_crop=True,
                                   load_in_8bit=False, load_in_4bit=False, lora_rank=32, unnorm_key=None,
                                   num_open_loop_steps=self.H)  # the attributes of GenerateConfig its utils read
        self.vla = get_vla(self.cfg)  # bf16, on DEVICE, norm_stats from dataset_statistics.json
        self.processor = get_processor(self.cfg)
        self.action_head = get_action_head(self.cfg, self.vla.llm_dim)
        self.proprio_projector = get_proprio_projector(self.cfg, self.vla.llm_dim, proprio_dim=PROPRIO_DIM)
        self.f = SimpleNamespace(img=get_libero_image, wrist=get_libero_wrist_image, aa=quat2axisangle,
                                 resize=lambda im: resize_image_for_policy(im, OPENVLA_IMAGE_SIZE),
                                 act=get_vla_action, norm_g=normalize_gripper_action, inv_g=invert_gripper_action)

    def unnorm_key(self, suite):
        """openvla-oft's check_unnorm_key (and LIBERO-MAX's _set_unnorm_key): the suite, else suite + _no_noops."""
        keys = [k for k in (suite, suite + "_no_noops") if k in self.vla.norm_stats]
        if not keys:
            raise KeyError(f"{self.name}: no unnorm key for {suite}; has {sorted(self.vla.norm_stats)}")
        return keys[0]

    def __call__(self, raw_obs_list, robosuite_envs, instructions, seeds, suites=None):
        _check_res(self.name, raw_obs_list)
        assert suites is not None and len(suites) == len(raw_obs_list), "OFT needs each row's suite (unnorm key)"
        f, out = self.f, []
        for o, text, suite in zip(raw_obs_list, instructions, suites):
            self.cfg.unnorm_key = self.unnorm_key(suite)
            obs = {"full_image": f.resize(f.img(o)), "wrist_image": f.resize(f.wrist(o)),  # prepare_observation
                   "state": np.concatenate((o["robot0_eef_pos"], f.aa(np.array(o["robot0_eef_quat"])),
                                            o["robot0_gripper_qpos"]))}
            with self.torch.no_grad():  # robot_utils.get_action
                a = np.asarray(f.act(self.cfg, self.vla, self.processor, obs, text, action_head=self.action_head,
                                     proprio_projector=self.proprio_projector, use_film=False), dtype=np.float32)
            # process_action: [0, 1] -> sign(2g - 1), then flipped: -1 open, +1 close (RLDS has 1 = open)
            out.append(f.inv_g(f.norm_g(a, binarize=True)))
        return np.stack(out)


def make_brain(name, device, precision):
    """A brain by name: pi05, smolvla (LeRobot; fp32 or bf16) or oft, oftplus (openvla-oft; bf16, pod)."""
    assert name in NAMES, name
    return (OFTBrain if name in OFT else LeRobotBrain)(name, device, precision)


def raw_libero_obs(suite="libero_spatial", task=0, init=0, res=256):
    """(raw observation, env, instruction) of standard LIBERO as openvla-oft's eval starts an episode: env seed 0,
    the init state, 10 no-op steps."""
    import torch
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    t = benchmark.get_benchmark_dict()[suite]().get_task(task)
    env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), t.problem_folder, t.bddl_file),
                             camera_heights=res, camera_widths=res)
    env.seed(0)
    env.reset()
    states = torch.load(os.path.join(get_libero_path("init_states"), t.problem_folder, t.init_states_file),
                        weights_only=False)
    obs = env.set_init_state(states[init])
    for _ in range(10):
        obs, _, _, _ = env.step([0, 0, 0, 0, 0, 0, -1])
    return obs, env, t.language


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--selftest-oft", choices=sorted(OFT), required=True)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()
    brain = make_brain(args.selftest_oft, args.device, "bf16")
    raw, env, text = raw_libero_obs()
    tb = time.perf_counter()
    ch = brain([raw], [env], [text], [0], ["libero_spatial"])
    ms = 1000 * (time.perf_counter() - tb)
    again = brain([raw], [env], [text], [1], ["libero_spatial"])
    print(f"{args.selftest_oft}: {text!r} shape={ch.shape} dtype={ch.dtype} first call {ms:.0f} ms "
          f"repeat_bitwise={int(np.array_equal(ch, again))}")
    print("min", np.round(ch.min((0, 1)), 4).tolist(), "max", np.round(ch.max((0, 1)), 4).tolist())
    print("gripper", ch[0, :, 6].tolist(), "(+1 close, -1 open; at the start: -1)")
    assert ch.shape == (1, brain.H, 7) and set(np.unique(ch[..., 6])) <= {-1.0, 0.0, 1.0}
