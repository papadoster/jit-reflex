"""scripts/c1e4m_brains.py: the raw-observation formatting is bitwise LeRobot's LiberoEnv's (from the env or from its
EEF matrix), and the module imports without LeRobot / torch / openvla-oft (the pod's openvla-oft venv has none of
LeRobot). The brain calls run on stub brains (__new__ plus fake parts): no model is loaded."""

import contextlib
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPTS))
import c1e4m_brains as cb  # noqa: E402


def test_import_is_light():
    code = ("import sys; sys.path.insert(0, %r); import c1e4m_brains; "
            "assert not {'lerobot', 'torch', 'experiments', 'prismatic', 'libero'} & set(sys.modules)" % str(SCRIPTS))
    subprocess.run([sys.executable, "-c", code], check=True)


def _leaves(x, key=()):
    if isinstance(x, dict):
        for k in x:
            yield from _leaves(x[k], key + (k,))
    else:
        yield key, x


def test_format_raw_obs_bitwise_liberoenv():
    os.environ.setdefault("MUJOCO_GL", "cgl" if sys.platform == "darwin" else "egl")
    libero = pytest.importorskip("lerobot.envs.libero")
    try:
        env = libero.LiberoEnv(task_suite=libero._get_suite("libero_spatial"), task_id=0,
                               task_suite_name="libero_spatial", obs_type="pixels_agent_pos",
                               observation_width=360, observation_height=360)
        env.init_state_id = 0
        obs, _ = env.reset(seed=0)
    except Exception as exc:  # no GL / no LIBERO assets here
        pytest.skip(f"LIBERO cannot render: {exc!r}")
    try:
        a = np.array([0.3, -0.2, 0.1, 0.05, 0.0, -0.05, 1.0])
        for step in (False, True):  # after the reset, then after a step that moves the arm and closes
            ref = env.step(a)[0] if step else obs
            inner = env._env.env  # the robosuite env; LIBERO-MAX's wrappers proxy .robots to it
            raw = inner._get_observations()
            for robot in (inner, cb.robot_state_of(inner)):  # the env, or the matrix a worker sends
                got, want = dict(_leaves(cb.format_raw_obs(raw, robot))), dict(_leaves(ref))
                assert got.keys() == want.keys()
                for k in want:
                    assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), k
    finally:
        env.close()


def _raw(res, v=0.0):
    """A raw LIBERO observation's keys (LiberoEnv._format_raw_obs reads all), filled with v."""
    return {"agentview_image": np.full((res, res, 3), v, np.uint8), "robot0_eye_in_hand_image": np.full((res, res, 3), v,
            np.uint8), "robot0_eef_pos": np.full(3, v), "robot0_eef_quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "robot0_gripper_qpos": np.full(2, v), "robot0_gripper_qvel": np.zeros(2),
            "robot0_joint_pos": np.zeros(7), "robot0_joint_vel": np.zeros(7)}


def _lerobot_stub(name="smolvla"):
    """LeRobotBrain without a model: identity processors, a policy that records its batch and noise."""
    torch = pytest.importorskip("torch")
    b, seen = cb.LeRobotBrain.__new__(cb.LeRobotBrain), {}
    b.name, b.precision, b.dev, b.torch, b.H = name, "fp32", torch.device("cpu"), torch, 4
    b.cfg = SimpleNamespace(max_action_dim=5)
    b.preprocess = b.pre = b.env_pre = b.post = lambda x: x

    def predict_action_chunk(batch, noise):
        seen["batch"], seen["noise"] = batch, noise
        return torch.arange(len(noise), dtype=torch.float32)[:, None, None].expand(-1, b.H, 7)
    b.policy = SimpleNamespace(predict_action_chunk=predict_action_chunk)
    return b, seen


def test_lerobot_batched_two_rows_per_row_noise():
    pytest.importorskip("lerobot.envs.libero")
    torch = pytest.importorskip("torch")
    b, seen = _lerobot_stub()
    mats = [np.eye(3), np.diag([1.0, -1.0, -1.0])]
    out = b([_raw(360, 1), _raw(360, 2)], mats, ["pick a", "pick b"], [7, 8])
    assert out.shape == (2, b.H, 7) and out[:, 0, 0].tolist() == [0.0, 1.0]  # one call, rows in order
    batch = seen["batch"]  # _stack: nested dicts, a new batch axis on every leaf
    assert batch["pixels"]["image"].shape == (2, 360, 360, 3) and batch["pixels"]["image2"].shape == (2, 360, 360, 3)
    assert batch["pixels"]["image"][:, 0, 0, 0].tolist() == [1, 2] and batch["robot_state"]["eef"]["pos"].shape == (2, 3)
    assert np.array_equal(batch["robot_state"]["eef"]["mat"], np.stack(mats)) and batch["task"] == ["pick a", "pick b"]
    noise = seen["noise"]
    assert noise.shape == (2, b.H, 5) and not torch.equal(noise[0], noise[1])
    for row, s in zip(noise, (7, 8)):  # each row its own seed's draw, whatever the batch
        assert torch.equal(row, torch.randn(1, b.H, 5, generator=torch.Generator().manual_seed(s))[0])


def test_render_size_guard():
    b, _ = _lerobot_stub("pi05")
    with pytest.raises(AssertionError, match="360"):
        b([_raw(256)], [np.eye(3)], ["x"], [0])
    o = cb.OFTBrain.__new__(cb.OFTBrain)
    o.name = "oftplus"
    with pytest.raises(AssertionError, match="256"):
        o([_raw(360)], [None], ["x"], [0], ["libero_spatial"])


def _norm_g(a, binarize=True):  # openvla-oft experiments/robot/robot_utils.py normalize_gripper_action
    a = a.copy()
    a[..., -1] = 2 * a[..., -1] - 1
    if binarize:
        a[..., -1] = np.sign(a[..., -1])
    return a


def _inv_g(a):  # openvla-oft invert_gripper_action
    a = a.copy()
    a[..., -1] *= -1.0
    return a


def test_oft_postprocess_and_unnorm_key():
    b, seen = cb.OFTBrain.__new__(cb.OFTBrain), []
    b.name, b.torch, b.cfg = "oft", SimpleNamespace(no_grad=contextlib.nullcontext), SimpleNamespace(unnorm_key=None)
    b.processor = b.action_head = b.proprio_projector = None
    b.vla = SimpleNamespace(norm_stats=dict.fromkeys(["libero_spatial_no_noops", "libero_goal", "libero_goal_no_noops"]))
    g = [1.0, 0.0, 0.9, 0.1, 1.0, 0.0, 1.0, 0.0]  # the model's gripper: RLDS [0, 1], 1 = open

    def act(cfg, vla, processor, obs, text, **kw):
        seen.append((cfg.unnorm_key, obs["state"].shape, obs["full_image"][0, 0, 0]))
        return [np.r_[np.full(6, 0.25), gi] for gi in g]
    b.f = SimpleNamespace(img=lambda o: o["agentview_image"][::-1, ::-1], wrist=lambda o: o["robot0_eye_in_hand_image"],
                          aa=lambda q: q[:3], resize=lambda im: im, act=act, norm_g=_norm_g, inv_g=_inv_g)
    raw = _raw(256)
    raw["agentview_image"][-1, -1] = 9  # the flip puts it at [0, 0]
    out = b([raw, raw], [None, None], ["x", "y"], [0, 1], ["libero_spatial", "libero_goal"])
    assert out.shape == (2, 8, 7) and out.dtype == np.float32 and np.all(out[..., :6] == 0.25)
    assert out[0, :, 6].tolist() == [-1, 1, -1, 1, -1, 1, -1, 1]  # g = 1 -> -1 open, g = 0 -> +1 close
    # the suite's key if present, else suite_no_noops; state = eef pos + axis-angle + gripper qpos
    assert seen == [("libero_spatial_no_noops", (8,), 9), ("libero_goal", (8,), 9)]
    with pytest.raises(KeyError, match="libero_10"):
        b([raw], [None], ["z"], [0], ["libero_10"])
