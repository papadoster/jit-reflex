"""Deterministic code check (spec changelog, 512-state run): the exact reflex in b2.probe_state and diag.probe_state
on the same states, z and action noise must agree to float precision.

jax.random.normal is patched to depend only on the array size, so z ([1, H, A] in diag, [H, A] in b2) and the noise
([M, 8, A] in both, one chunk) get identical values in both codes. Run from the repo root:
  UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline python <this file>
"""

import pickle
import sys

import jax
import numpy as np

sys.path.insert(0, "src")
import b2  # noqa: E402
import diag  # noqa: E402
import predictors  # noqa: E402
import probe  # noqa: E402
import train_expert  # noqa: E402

FIX = jax.random.key(123)
real_normal = jax.random.normal


def normal(key, shape=(), dtype=float):
    return real_normal(jax.random.fold_in(FIX, int(np.prod(shape))), shape, dtype)


worst = 0.0
for level in ("worlds/l/trampoline.json", "worlds/l/mjc_walker.json", "worlds/l/grasp_easy.json"):
    env, params, levels, O, A = probe.setup([level])
    lv = jax.tree.map(lambda x: x[0], levels)
    policy = probe.make_policy(probe.load_state_dict("checkpoints/bc", level), O, A)
    bnd = probe.collect(env, params, policy, lv, jax.random.key(0), 16, 4, train_expert.ACTION_NOISE_STD, b2.S)
    std = predictors.alive_std(bnd)
    obs, state = probe.sample(bnd, jax.random.key(1), 6)
    with open(f"{predictors.WM_DIR}/{predictors.level_name(level)}.pkl", "rb") as f:
        wm = pickle.load(f)
    jax.random.normal = normal  # patched only while tracing the two probes
    try:
        d_fn = jax.jit(lambda o, r, k: diag.probe_state(policy, env._env, params, wm, (), r, o, k, std, 4, 1, b2.S))
        b_fn = jax.jit(lambda o, r, k: b2.probe_state(policy, env._env, params, wm, r, o, k, std, 4, b2.S))
        for i in range(obs.shape[0]):
            raw = jax.tree.map(lambda x: x[i], state).env_state
            key = jax.random.key(10 + i)
            d, b = d_fn(obs[i], raw, key), b_fn(obs[i], raw, key)
            pairs = {  # (diag name, diag predictor index) -> (b2 name, b2 predictor index); diag here: oracle, learned
                "oracle pred": (d["pred"][0], b["pred"][0]), "oracle reflex": (d["lin_clip"][0], b["reflex"][0]),
                "oracle chunk_j": (d["chunk_lin_clip"][0], b["chunk_j"][0]), "oracle e": (d["e"][0], b["e"][0]),
                "learned pred": (d["pred"][-1], b["pred"][1]), "learned reflex": (d["lin_clip"][-1], b["reflex"][1]),
                "valid": (d["valid"], b["valid"]),
            }
            for name, (x, y) in pairs.items():
                err = float(np.max(np.abs(np.asarray(x, float) - np.asarray(y, float))))
                scale = float(np.max(np.abs(np.asarray(x, float)))) or 1.0
                worst = max(worst, err / scale)
                print(f"{level} state {i} {name}: max |diff| {err:.2e} (scale {scale:.2e})")
    finally:
        jax.random.normal = real_normal
print(f"worst relative difference: {worst:.2e}", "MATCH" if worst < 1e-5 else "MISMATCH")
