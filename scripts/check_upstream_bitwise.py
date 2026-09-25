"""Bitwise check that upstream methods are untouched: current src/ vs pristine upstream (UPSTREAM_REF).

Runs the same computation in two subprocesses, one importing model / eval_flow / train_expert from
`git show UPSTREAM_REF:src/...`, one from src/, and compares every output exactly (np.array_equal).
Rerun after any change to src/model.py or src/eval_flow.py. From the repo root:

    uv run --offline python scripts/check_upstream_bitwise.py
"""

import os
import pathlib
import pickle
import subprocess
import sys
import tempfile

import numpy as np

LEVEL = "worlds/l/grasp_easy.json"
CHECKPOINT = "checkpoints/bc/31/policies/worlds_l_grasp_easy.pkl"
UPSTREAM_FILES = ("model", "eval_flow", "train_expert")
# the last Physical Intelligence commit in this repository's history (the fork point): works without any remote
UPSTREAM_REF = "23e8e2f3da35571ea1385f687d38b711a4d7ad96"
NUM_STEPS, DELAY, HORIZON, BATCH = 5, 2, 4, 16


def compute(src_dir: str, out_path: str):
    """Upstream-method outputs on real grasp_easy reset observations with fixed keys -> out_path (.npz)."""
    import flax.nnx as nnx
    import jax
    import kinetix.environment.env as kenv
    import kinetix.environment.env_state as kenv_state

    import eval_flow
    import model as _model
    import train_expert

    for m in (eval_flow, _model, train_expert):  # otherwise the comparison would be vacuous
        assert pathlib.Path(m.__file__).resolve().parent == pathlib.Path(src_dir).resolve(), m.__file__

    # env, levels and static params exactly as eval_flow.main
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels([LEVEL], static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    level = jax.tree.map(lambda x: x[0], levels)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    # policy loaded as in eval_flow.main: split / replace_by_pure_dict / merge
    with open(CHECKPOINT, "rb") as f:
        state_dict = pickle.load(f)
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=_model.ModelConfig(), rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    policy = nnx.merge(graphdef, state)

    k_reset, k_prev, k_act, k_eval = jax.random.split(jax.random.key(0), 4)
    obs, _ = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(jax.random.split(k_reset, BATCH), level, env_params)
    prev = policy.action(k_prev, obs, NUM_STEPS)
    prefix = policy.action_chunk_size - HORIZON
    res = {
        "action": policy.action(k_act, obs, NUM_STEPS),
        "realtime_action": policy.realtime_action(k_act, obs, NUM_STEPS, prev, DELAY, prefix, "exp", 5.0),
        "hard_masking_action": policy.realtime_action(k_act, obs, NUM_STEPS, prev, DELAY, prefix, "zeros", 5.0),
        "bid_action": policy.bid_action(k_act, obs, NUM_STEPS, prev, DELAY, prefix, 16),
    }
    for name, method in (("naive", eval_flow.NaiveMethodConfig()), ("realtime", eval_flow.RealtimeMethodConfig())):
        config = eval_flow.EvalConfig(num_evals=16, inference_delay=DELAY, execute_horizon=HORIZON, method=method)
        # [0] only: XLA then drops the video
        info = jax.jit(lambda rng: eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params)[0])(
            k_eval
        )
        res.update({f"eval_{name}.{k}": v for k, v in info.items()})
    np.savez(out_path, **jax.device_get(res))


def main():
    root = pathlib.Path(__file__).resolve().parent.parent
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        (tmp / "upstream").mkdir()
        for name in UPSTREAM_FILES:
            code = subprocess.run(
                ["git", "show", f"{UPSTREAM_REF}:src/{name}.py"], cwd=root, check=True, capture_output=True, text=True
            ).stdout
            (tmp / "upstream" / f"{name}.py").write_text(code)
        results = {}
        for label, src_dir in (("upstream", tmp / "upstream"), ("current", root / "src")):
            out = tmp / f"{label}.npz"
            p = subprocess.run(
                [sys.executable, __file__, str(src_dir), str(out)],
                cwd=root,
                env={**os.environ, "PYTHONPATH": str(src_dir)},
                capture_output=True,
                text=True,
            )
            if p.returncode:
                sys.exit(f"{label} run failed:\n{p.stdout}{p.stderr}")
            results[label] = dict(np.load(out))

    up, cur = results["upstream"], results["current"]
    ok = True
    for k in up:
        same = np.array_equal(up[k], cur[k])
        ok &= same
        print(f"{k} {list(up[k].shape)}: {'IDENTICAL' if same else 'DIFFERENT'}")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    if len(sys.argv) == 3:
        compute(*sys.argv[1:])
    else:
        main()
