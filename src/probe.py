"""E1 kill-test: is a frozen flow policy locally linear along its predicted trajectory?

For sampled states x_t: chunk A = pi(z, o_t); predicted obs o^_{t+k} (noise-free fork of the simulator);
noisy obs o_{t+k} (action noise sigma); oracle a* = pi(roll(z, -k), o_{t+k})[0] = what a fresh call would do now.
The noise is rolled so that row 0 is z[k], the noise chunk[k] came from (reflex.shifted_noise).
See docs/superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md (section 5, E1).
"""

import json
import math
import pathlib
import pickle
import time
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import numpy as np
import pandas as pd
import tyro

import model as _model
import reflex
import train_expert

LEVELS = (
    "worlds/l/grasp_easy.json",
    "worlds/l/catapult.json",
    "worlds/l/cartpole_thrust.json",
    "worlds/l/hard_lunar_lander.json",
    "worlds/l/mjc_half_cheetah.json",
    "worlds/l/mjc_swimmer.json",
    "worlds/l/mjc_walker.json",
    "worlds/l/h17_unicycle.json",
    "worlds/l/chain_lander.json",
    "worlds/l/catcher_v3.json",
    "worlds/l/trampoline.json",
    "worlds/l/car_launch.json",
)

ERRORS = ("chunk", "pred", "lin", "lin_clip", "chunk_lin", "floor", "dev")


def executed(action, env_state):
    """The action Kinetix actually applies (env.convert_continuous_actions), for E1b.

    Motor dims go to their joints via motor_bindings and are clipped to [-1, 1], thruster dims to [0, 1]; joints
    without a running motor and inactive thrusters are zeroed. So "-0.5" and "-0.1" on a thruster are the same action,
    and action dims that drive nothing don't count. action [..., A] -> [..., num_joints + num_thrusters].
    """
    static = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    mask = jnp.concatenate([env_state.joint.active & env_state.joint.motor_on, env_state.thruster.active])

    def one(a):
        return kenv.convert_continuous_actions(a, env_state, static, None) * mask

    return jnp.vectorize(one, signature="(a)->(e)")(action)


def errors(chunk, a_ref, jac, nom_obs, obs, a_star, max_correction: float = 1.0, act=lambda a: a):
    """Squared action errors against the oracle a* (spec E1). lin_clip clips J·delta like the E2 method (E1b).

    chunk, a_ref [K, A]; jac [K, A, O]; nom_obs [K, O]; obs [..., K, O]; a_star [..., K, A] -> dict of [..., K].
    act maps an action to the space it is compared in: identity (E1) or `executed` (E1b).
    """
    delta = obs - nom_obs
    lin = jnp.einsum("kao,...ko->...ka", jac, delta)

    def sq(x):
        return jnp.sum(jnp.square(x), axis=-1)

    a_star = act(a_star)
    return {
        "chunk": sq(act(chunk) - a_star),
        "pred": sq(act(a_ref) - a_star),
        "lin": sq(act(a_ref + lin) - a_star),
        "lin_clip": sq(act(a_ref + jnp.clip(lin, -max_correction, max_correction)) - a_star),
        "chunk_lin": sq(act(chunk + lin) - a_star),
        "floor": jnp.broadcast_to(sq(act(chunk) - act(a_ref)), a_star.shape[:-1]),
        "dev": jnp.sqrt(sq(delta)),
    }


def verdict(summary: pd.DataFrame, on: str = "lin") -> dict:
    """Pre-registered E1 decision rule (spec section 5). summary needs columns level, sigma, k, `on`, pred, rel.

    Level rho is pooled over k = 1..4: 1 - sum_k mean e_on / sum_k mean e_pred. E1: on="lin"; E1b: on="lin_clip".
    """
    near = summary[summary["k"].between(1, 4)]
    rel = float(near[near["sigma"] == 0.1]["rel"].mean())
    sigma = 0.1 if rel >= 0.1 else 0.2  # relevance check: deviations must actually change the policy's decisions
    g = near[near["sigma"] == sigma].groupby("level")[[on, "pred"]].sum()
    per_level = 1 - g[on] / g["pred"]  # pooled over k = 1..4: one k with a tiny e_pred can't decide a level
    r, n_ok, n = float(per_level.median()), int((per_level >= 0.3).sum()), len(per_level)
    if r >= 0.5 and n_ok >= math.ceil(2 * n / 3):
        v = "GO"
    elif r < 0.2:
        v = "KILL"
    else:
        v = "GRAY"
    return {"verdict": v, "on": on, "sigma": sigma, "R": r, "levels_ok": n_ok, "levels": n, "rel": rel}


def setup(level_paths: Sequence[str]):
    """AutoReplay env without noise, env params, stacked levels, obs_dim, action_dim; same config as eval_flow.main."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    env = wrappers.AutoReplayWrapper(
        kenv.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1",
            static_env_params=static_env_params.replace(screen_dim=train_expert.SCREEN_DIM),
        )
    )
    level0 = jax.tree.map(lambda x: x[0], levels)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level0, env_params)[0].shape[-1]
    return env, env_params, levels, obs_dim, env.action_space(env_params).shape[0]


def load_state_dict(run_path: str, level_path: str, step: int = -1):
    level_name = level_path.replace("/", "_").replace(".json", "")
    dirs = sorted(
        (p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)
    )
    with (dirs[step] / "policies" / f"{level_name}.pkl").open("rb") as f:
        return pickle.load(f)


def make_policy(state_dict, obs_dim: int, action_dim: int):
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=_model.ModelConfig(), rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def collect(env, env_params, policy, level, key, num_envs: int, horizon: int, sigma: float, num_steps: int):
    """Naive chunked rollouts with Gaussian action noise. Returns (obs, state, alive) at every chunk boundary, [C, E, ...].

    alive = the first episode has not ended yet (later episodes are AutoReplay repeats and are ignored).
    """
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(
        jax.random.split(k_reset, num_envs), level, env_params
    )
    env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def run_chunk(carry, key):
        obs, state, alive = carry
        k_act, k_noise, k_env = jax.random.split(key, 3)
        actions = policy.action(k_act, obs, num_steps)[:, :horizon]
        actions = actions + sigma * jax.random.normal(k_noise, actions.shape)

        def one_step(c, xs):
            obs, state, alive = c
            action, k = xs
            obs, state, _, done, _ = env_step(jax.random.split(k, num_envs), state, action, env_params)
            return (obs, state, alive & ~done), None

        carry, _ = jax.lax.scan(
            one_step, (obs, state, alive), (actions.swapaxes(0, 1), jax.random.split(k_env, horizon))
        )
        return carry, (obs, state, alive)

    n_chunks = env_params.max_timesteps // horizon
    return jax.lax.scan(run_chunk, (obs, state, jnp.ones(num_envs, bool)), jax.random.split(k_run, n_chunks))[1]


def sample(boundaries, key, num_states: int):
    """num_states random alive (obs, state) pairs from collect() output."""
    obs, state, alive = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), boundaries)
    idx = jax.random.choice(key, alive.shape[0], (num_states,), replace=False, p=alive / alive.sum())
    return obs[idx], jax.tree.map(lambda x: x[idx], state)


def probe_one(
    env, env_params, policy, state, obs, key, sigmas: Sequence[float], num_draws: int, num_steps: int,
    executed_actions: bool = False,
):
    """E1 errors for one state: ERRORS + 'valid', each [S, M, K] (sigmas, draws, offsets k = 1..H-1)."""
    H, A = policy.action_chunk_size, policy.action_dim
    K = H - 1
    k_z, k_nom, k_eps, k_roll = jax.random.split(key, 4)
    z = jax.random.normal(k_z, (H, A))
    chunk = policy.action_from_noise(z[None], obs[None], num_steps)[0]  # [H, A]

    def rollout(actions, key):  # obs after each of the K actions, and "the episode ended by then"
        def one_step(c, xs):
            st, ended = c
            action, k = xs
            o, st, _, done, _ = env.step(k, st, action, env_params)
            return (st, ended | done), (o, ended | done)

        return jax.lax.scan(one_step, (state, jnp.bool_(False)), (actions, jax.random.split(key, K)))[1]

    nom_obs, nom_ended = rollout(chunk[:K], k_nom)  # o^_{t+1..t+K}
    shifted = reflex.shifted_noise(z)[1:]  # [K, H, A]: offset k starts from row z[k]
    a_ref, jac = jax.vmap(lambda n, o: reflex.first_action_and_jacobian(policy, n, o, num_steps))(shifted, nom_obs)

    S = len(sigmas)
    noise = jax.random.normal(k_eps, (S, num_draws, K, A))
    noisy_actions = chunk[:K] + jnp.asarray(sigmas)[:, None, None, None] * noise
    roll_keys = jax.random.split(k_roll, S * num_draws).reshape(S, num_draws)
    obs_t, ended = jax.vmap(jax.vmap(rollout))(noisy_actions, roll_keys)  # [S, M, K, O], [S, M, K]

    flat = obs_t.reshape(-1, obs_t.shape[-1])
    noises = jnp.broadcast_to(shifted, (*obs_t.shape[:-1], H, A)).reshape(-1, H, A)  # K axes aligned, like flat
    a_star = policy.action_from_noise(noises, flat, num_steps)[:, 0]
    act = (lambda a: executed(a, state.env_state)) if executed_actions else (lambda a: a)
    out = errors(chunk[1:], a_ref, jac, nom_obs, obs_t, a_star.reshape(*obs_t.shape[:-1], A), act=act)
    out["valid"] = ~(ended | nom_ended)
    return out


def summarize(level: str, res: dict, sigmas: Sequence[float]) -> pd.DataFrame:
    """Mean errors over valid (state, draw) pairs per sigma and offset k, plus rho, rho_total, rel."""
    valid = res["valid"]  # [N, S, M, K]
    rows = []
    for si, sigma in enumerate(sigmas):
        for k in range(valid.shape[-1]):
            v = valid[:, si, :, k]
            means = {name: float(res[name][:, si, :, k][v].mean()) for name in ERRORS}
            rows.append({"level": level, "sigma": sigma, "k": k + 1, "n": int(v.sum()), **means})
    df = pd.DataFrame(rows)
    df["rho"] = 1 - df["lin"] / df["pred"]
    df["rho_clip"] = 1 - df["lin_clip"] / df["pred"]
    df["rho_total"] = 1 - df["lin"] / df["chunk"]
    df["rel"] = df["pred"] / df["chunk"]
    return df


def run(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = LEVELS,
    num_envs: int = 64,
    collect_horizon: int = 4,
    num_states: int = 256,
    num_draws: int = 4,
    sigmas: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    num_flow_steps: int = 5,
    batch_size: int = 16,  # states per vmapped batch; lower it if RAM runs out
    seed: int = 0,
    output_dir: str = "results/probe",
    verdict_on: str = "lin",  # E1: "lin"; E1b: "lin_clip" (spec section 5, E1b)
    action_space: str = "raw",  # E1: "raw"; E1b: "executed" (what Kinetix applies, see `executed`)
):
    """E1 kill-test (spec section 5): writes summary.csv and verdict.json to output_dir."""
    env, env_params, levels, obs_dim, action_dim = setup(level_paths)
    sigmas = tuple(sigmas)

    @jax.jit
    def probe_level(state_dict, level, key):
        policy = make_policy(state_dict, obs_dim, action_dim)
        k_collect, k_sample, k_probe = jax.random.split(key, 3)
        boundaries = collect(
            env, env_params, policy, level, k_collect, num_envs, collect_horizon,
            train_expert.ACTION_NOISE_STD, num_flow_steps,
        )
        obs, state = sample(boundaries, k_sample, num_states)

        def one(x):
            return probe_one(
                env, env_params, policy, x[0], x[1], x[2], sigmas, num_draws, num_flow_steps,
                executed_actions=action_space == "executed",
            )

        res = jax.lax.map(one, (state, obs, jax.random.split(k_probe, num_states)), batch_size=batch_size)
        return res, boundaries[2].sum()

    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    frames = []
    for i, level_path in enumerate(level_paths):
        level = jax.tree.map(lambda x: x[i], levels)
        res, n_alive = jax.device_get(
            probe_level(load_state_dict(run_path, level_path), level, jax.random.key(seed + i))
        )
        print(f"{level_path}: {int(n_alive)} alive boundary states")
        assert n_alive >= num_states, "too few alive states: sample() would pick finished episodes; raise --num-envs"
        frames.append(summarize(level_path, res, sigmas))
        summary = pd.concat(frames, ignore_index=True)
        summary.to_csv(out / "summary.csv", index=False)  # after every level: a crash keeps the finished ones
    v = verdict(summary, verdict_on)
    (out / "verdict.json").write_text(json.dumps(v, indent=2))
    near = summary[(summary["sigma"] == v["sigma"]) & summary["k"].between(1, 4)]
    g = near.groupby("level")
    table = g[["rel", "floor"]].mean()
    table.insert(0, "rho", 1 - g["lin"].sum() / g["pred"].sum())  # pooled over k, as in verdict()
    table.insert(1, "rho_clip", 1 - g["lin_clip"].sum() / g["pred"].sum())
    table.insert(2, "rho_total", 1 - g["lin"].sum() / g["chunk"].sum())
    print(table.round(3).to_string())
    print(json.dumps(v))


def cost(
    run_path: str = "checkpoints/bc",
    level_path: str = LEVELS[0],
    batch: int = 1,
    num_flow_steps: int = 5,
    delay: int = 2,
    horizon: int = 6,
    repeats: int = 20,
    out: str | None = None,
):
    """Per-call cost of every E2 method: network evaluations (analytic), GFLOP and measured latency (spec section 4).

    GFLOP of one network evaluation comes from cost_analysis on a loop-free call: whole calls contain lax.scan loops,
    whose bodies XLA may count only once. The predictor (a simulator fork here) is not included.
    """
    _, _, _, obs_dim, action_dim = setup([level_path])
    policy = make_policy(load_state_dict(run_path, level_path), obs_dim, action_dim)
    H = policy.action_chunk_size
    key = jax.random.key(0)
    noise = jax.random.normal(key, (batch, H, action_dim))
    obs, ref, prev = jnp.zeros((batch, obs_dim)), jnp.zeros((batch, H, obs_dim)), jnp.zeros((batch, H, action_dim))
    one_eval = jax.jit(lambda o, x: policy(o, x, jnp.zeros(())))
    analysis = one_eval.lower(obs[:1], noise[:1]).compile().cost_analysis()
    gflop_per_eval = (analysis[0] if isinstance(analysis, list) else analysis)["flops"] / 1e9

    def with_package(requery, feedback):
        def call(noise, obs, ref, prev):
            chunk = policy.action_from_noise(noise, obs, num_flow_steps)
            return chunk, reflex.package(policy, noise, ref, chunk, num_flow_steps, requery, feedback)

        return call

    calls = {
        "naive": lambda noise, obs, ref, prev: policy.action_from_noise(noise, obs, num_flow_steps),
        "realtime": lambda noise, obs, ref, prev: policy.realtime_action(
            key, obs, num_flow_steps, prev, delay, H - horizon, "exp", 5.0
        ),
        "pred": with_package(True, False),
        "reflex": with_package(True, True),
        "reflex_chunk": with_package(False, True),
    }
    rows = []
    for name, fn in calls.items():
        f = jax.jit(fn)
        jax.block_until_ready(f(noise, obs, ref, prev))
        start = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(f(noise, obs, ref, prev))
        evals = reflex.forward_equivalents(name, num_flow_steps, H, action_dim)
        rows.append({
            "method": name,
            "batch": batch,
            "evals_per_call": evals,
            "gflop_per_call": evals * gflop_per_eval * batch,
            "ms_per_call": (time.perf_counter() - start) / repeats * 1e3,
        })
    df = pd.DataFrame(rows)
    df["latency_vs_realtime"] = df["ms_per_call"] / df.loc[df["method"] == "realtime", "ms_per_call"].item()
    print(f"one network evaluation (batch 1): {gflop_per_eval:.4f} GFLOP")
    print(df.round(3).to_string(index=False))
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"run": run, "cost": cost})
