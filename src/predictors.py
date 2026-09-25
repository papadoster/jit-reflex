"""B1 predictors for the reflex: wrong-physics simulator copies and a learned world model, plus their offline errors.

See docs/superpowers/specs/2026-09-25-b1-staleness-predictors-design.md (sections 3 and 4.2).
"""

import pathlib
import pickle
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tyro

import probe
import train_expert

WM_DIR = "results/b1/world_models"
BODY_PARAMS = ("inverse_mass", "inverse_inertia", "friction")


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def _params(state):
    """The physical parameters a phys predictor gets wrong: (polygon, circle, motor power, thruster power)."""

    def body(b):
        return {k: getattr(b, k) for k in BODY_PARAMS}

    return body(state.polygon), body(state.circle), state.joint.motor_power, state.thruster.power


def _with(state, polygon, circle, motor_power, thruster_power):
    return state.replace(
        polygon=state.polygon.replace(**polygon),
        circle=state.circle.replace(**circle),
        joint=state.joint.replace(motor_power=motor_power),
        thruster=state.thruster.replace(power=thruster_power),
    )


def phys_factors(key, state, error: float):
    """1 + error * (+-1) for every entry of _params(state), with a random sign per entry (body, joint, thruster).

    A fixed key gives every method and every error size the same signs: a paired comparison and a clean dose curve.
    """
    leaves, treedef = jax.tree.flatten(_params(state))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef, [1 + error * jax.random.rademacher(k, x.shape).astype(x.dtype) for k, x in zip(keys, leaves)]
    )


def phys_step(env, key, state, action, params, factors):
    """One predictor step with wrong physics -> (obs, state), like env.step_env(...)[:2].

    Steps with the parameters x factors, then puts the true parameters back: the symbolic observation contains the
    parameters themselves, and a prediction must differ from the oracle's only through the motion.
    """
    wrong = _with(state, *jax.tree.map(jnp.multiply, _params(state), factors))
    nxt = env.step_env(key, wrong, action, params)[1]
    nxt = _with(nxt, *_params(state))
    return env.get_obs(nxt), nxt


def init(key, obs_dim: int, action_dim: int, hidden: int):
    sizes = (obs_dim + action_dim, hidden, hidden, obs_dim)
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (m, n)) / jnp.sqrt(m), jnp.zeros(n)) for k, m, n in zip(keys, sizes[:-1], sizes[1:])]


def _net(wm, obs, action):
    """Normalized one-step change of obs predicted from (obs, action): obs [O], action [A] -> [O]."""
    x = jnp.concatenate([(obs - wm["x_mean"]) / wm["x_std"], action])
    for w, b in wm["layers"][:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = wm["layers"][-1]
    return x @ w + b


def apply(wm, obs, action):
    """Next obs. Dims that never move in the training data (mask 0) are copied exactly."""
    return obs + wm["mask"] * (_net(wm, obs, action) * wm["d_std"] + wm["d_mean"])


def wm_rollout(wm, obs, actions):
    """Predicted obs after each action: obs [O], actions [T, A] -> [T, O]."""

    def body(o, a):
        o = apply(wm, o, a)
        return o, o

    return jax.lax.scan(body, obs, actions)[1]


def fit(obs, act, nxt, key, hidden: int = 256, steps: int = 10_000, batch: int = 512, lr: float = 1e-3):
    """World model (spec B1 3.3) from transitions obs [N, O], act [N, A], nxt [N, O]."""
    obs, act, nxt = jnp.asarray(obs), jnp.asarray(act), jnp.asarray(nxt)
    delta = nxt - obs
    x_std, d_std = obs.std(0), delta.std(0)
    mask = (d_std > 1e-6).astype(obs.dtype)
    k_init, k_steps = jax.random.split(key)
    wm = {
        "layers": init(k_init, obs.shape[1], act.shape[1], hidden),
        "x_mean": obs.mean(0),
        "x_std": jnp.where(x_std > 1e-6, x_std, 1.0),
        "d_mean": delta.mean(0) * mask,
        "d_std": jnp.where(mask > 0, d_std, 1.0),
        "mask": mask,
    }
    target = (delta - wm["d_mean"]) / wm["d_std"]
    opt = optax.adam(lr)

    def loss(layers, o, a, t):
        pred = jax.vmap(lambda o_, a_: _net(wm | {"layers": layers}, o_, a_))(o, a)
        return jnp.sum(mask * (pred - t) ** 2) / (mask.sum() * o.shape[0])

    @jax.jit
    def run(layers, keys, obs, act, target):
        def body(carry, k):
            layers, opt_state = carry
            idx = jax.random.randint(k, (batch,), 0, obs.shape[0])
            grads = jax.grad(loss)(layers, obs[idx], act[idx], target[idx])
            updates, opt_state = opt.update(grads, opt_state)
            return (optax.apply_updates(layers, updates), opt_state), None

        return jax.lax.scan(body, (layers, opt.init(layers)), keys)[0][0]

    return wm | {"layers": run(wm["layers"], jax.random.split(k_steps, steps), obs, act, target)}


def one_step_nmse(wm, obs, act, nxt):
    """Mean squared one-step error on the moving dims, in units of their change's std (1.0 = predicting the mean)."""
    pred = jax.vmap(lambda o, a: apply(wm, o, a))(jnp.asarray(obs), jnp.asarray(act))
    z = wm["mask"] * (pred - jnp.asarray(nxt)) / wm["d_std"]
    return jnp.sum(z**2) / (wm["mask"].sum() * pred.shape[0])


def collect_transitions(env, env_params, policy, level, key, num_envs, horizon, sigma, num_steps):
    """(obs, executed action, next obs, valid) at every step of naive chunked rollouts with action noise.

    Like probe.collect, but per step. valid = the first episode is still running and this step did not end it
    (AutoReplay would put the reset observation into next obs). Each output is [C, horizon, E, ...].
    """
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(
        jax.random.split(k_reset, num_envs), level, env_params
    )
    env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def run_chunk(carry, key):
        k_act, k_noise, k_env = jax.random.split(key, 3)
        actions = policy.action(k_act, carry[0], num_steps)[:, :horizon]
        actions = actions + sigma * jax.random.normal(k_noise, actions.shape)

        def one_step(c, xs):
            obs, state, alive = c
            action, k = xs
            nobs, state, _, done, _ = env_step(jax.random.split(k, num_envs), state, action, env_params)
            return (nobs, state, alive & ~done), (obs, action, nobs, alive & ~done)

        return jax.lax.scan(one_step, carry, (actions.swapaxes(0, 1), jax.random.split(k_env, horizon)))

    n_chunks = env_params.max_timesteps // horizon
    return jax.lax.scan(run_chunk, (obs, state, jnp.ones(num_envs, bool)), jax.random.split(k_run, n_chunks))[1]


def train(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = probe.LEVELS,
    num_envs: int = 128,
    horizon: int = 4,
    hidden: int = 256,
    steps: int = 10_000,
    batch: int = 512,
    lr: float = 1e-3,
    num_flow_steps: int = 5,
    seed: int = 100,  # level i uses seed + i: disjoint from every eval seed
    out_dir: str = WM_DIR,
):
    """One world model per level from naive rollouts with action noise (spec B1 3.3); the last 10% of envs validate."""
    env, env_params, levels, obs_dim, action_dim = probe.setup(level_paths)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def data(state_dict, level, key):
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        return collect_transitions(
            env, env_params, policy, level, key, num_envs, horizon, train_expert.ACTION_NOISE_STD, num_flow_steps
        )

    n_val = max(1, num_envs // 10)
    tr, va = slice(0, num_envs - n_val), slice(num_envs - n_val, num_envs)
    rows = []
    for i, level_path in enumerate(level_paths):
        k_data, k_fit = jax.random.split(jax.random.key(seed + i))
        level = jax.tree.map(lambda x: x[i], levels)
        obs, act, nxt, valid = jax.device_get(data(probe.load_state_dict(run_path, level_path), level, k_data))

        def pick(x, envs):  # valid transitions of these envs, [N, ...]
            return x[:, :, envs].reshape(-1, *x.shape[3:])[valid[:, :, envs].reshape(-1)]

        wm = fit(pick(obs, tr), pick(act, tr), pick(nxt, tr), k_fit, hidden, steps, batch, lr)
        val = float(one_step_nmse(wm, pick(obs, va), pick(act, va), pick(nxt, va)))
        with (out / f"{level_name(level_path)}.pkl").open("wb") as f:
            pickle.dump(jax.device_get(wm), f)
        rows.append({
            "level": level_path, "train": int(valid[:, :, tr].sum()), "val": int(valid[:, :, va].sum()),
            "moving_dims": int(wm["mask"].sum()), "val_nmse": val,
        })
        print(rows[-1])
        pd.DataFrame(rows).to_csv(out / "train_log.csv", index=False)  # after every level


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"train": train})
