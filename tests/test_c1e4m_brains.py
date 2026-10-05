"""scripts/c1e4m_brains.py: the raw-observation formatting is bitwise LeRobot's LiberoEnv's, and the module imports
without LeRobot / torch / openvla-oft (the pod's openvla-oft venv has none of LeRobot)."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPTS = Path(__file__).resolve().parents[1] / "scripts"


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
    sys.path.insert(0, str(SCRIPTS))
    import c1e4m_brains as cb
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
            ours = cb.format_raw_obs(inner._get_observations(), inner)
            got, want = dict(_leaves(ours)), dict(_leaves(ref))
            assert got.keys() == want.keys()
            for k in want:
                assert got[k].dtype == want[k].dtype and np.array_equal(got[k], want[k]), k
    finally:
        env.close()
