"""A2C2 baseline, bt-kinetix variant (spec B2+B5 §9): a residual MLP head on a frozen flow policy, applied at every
executed step to the chunk's action at chunk index k.

Head, loss and training recipe adapted from TheAyos/bt-kinetix, commit fe7e503bb99cf98cb02e73b935b8cb223fd84164
(src-bt/model.py ResidualPolicy, src-bt/train_residual.py), under the MIT License:

Copyright (c) 2025 Physical Intelligence

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the
Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR
OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""

import json
import os
import pathlib
import pickle
import time
from typing import Sequence

import einops
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.wrappers as wrappers
import numpy as np
import optax
import tyro

import predictors
import probe
import reflex
import train_expert

LR, WD, CLIP, WARMUP, BATCH, EPOCHS = 1e-4, 1e-3, 5.0, 500, 512, 16  # bt-kinetix train_residual.Config
H = 8  # action chunk size of these policies


class Head(nnx.Module):
    """bt-kinetix ResidualPolicy: [obs, base action, (cos, sin) of the chunk index] -> 256 -> 512 -> residual."""

    def __init__(self, obs_dim: int, action_dim: int, *, rngs: nnx.Rngs):
        self.residual_policy = nnx.Sequential(
            nnx.Linear(obs_dim + action_dim + 2, 256, rngs=rngs),
            nnx.relu,
            nnx.Linear(256, 512, rngs=rngs),
            nnx.relu,
            nnx.Linear(512, action_dim, rngs=rngs),
        )

    def apply_residual(self, obs, base_action, time_feature):
        return base_action + self.residual_policy(jnp.concatenate([obs, base_action, time_feature], axis=-1))

    def loss(self, obs, base_action, time_feature, target_action):
        pred = self.residual_policy(jnp.concatenate([obs, base_action, time_feature], axis=-1))
        return jnp.mean(jnp.square(pred - (target_action - base_action)))


def time_feature(k, chunk_size: int):
    """(cos, sin) of 2 pi k / H, the order bt-kinetix uses: k [...] -> [..., 2]."""
    x = k * (2 * jnp.pi / chunk_size)
    return jnp.stack([jnp.cos(x), jnp.sin(x)], axis=-1)


def make_head(state_dict, obs_dim: int, action_dim: int) -> Head:
    head = Head(obs_dim, action_dim, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(head)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def load_heads(root: str, name: str, level_paths: Sequence[str]):
    """Stacked state dicts of <root>/<name>/<level>.pkl, in level_paths order (as eval_flow stacks policies)."""
    sds = []
    for level_path in level_paths:
        with (pathlib.Path(root) / name / f"{predictors.level_name(level_path)}.pkl").open("rb") as f:
            sds.append(pickle.load(f))
    return jax.tree.map(lambda *x: jnp.array(x), *sds)


def fit(rng, obs_dim: int, action_dim: int, batch_fn, data, n_items: int, num_epochs: int = EPOCHS,
        batch_size: int = BATCH):
    """bt-kinetix train_residual for one level; rng = the level's key (bt-kinetix: split(key(seed), 12)[level]).

    batch_fn(data, key, idx [B]) -> (obs, base, time_feature, target). data goes into jit as an argument, as in
    bt-kinetix: closed over, a 1 M x 679 set would be baked into the program (measured: +24 s compile, +5 GB RAM).
    Every epoch is a permutation of the n_items items in minibatches of batch_size (a whole number of them).
    Returns (Head, mean loss per epoch).
    """
    rng, key = jax.random.split(rng)  # bt-kinetix init
    head = Head(obs_dim, action_dim, rngs=nnx.Rngs(key))
    opt = nnx.Optimizer(head, optax.chain(
        optax.clip_by_global_norm(CLIP),
        optax.adamw(optax.warmup_constant_schedule(0, LR, WARMUP), weight_decay=WD),
    ))
    graphdef, state = nnx.split((head, opt))
    steps = n_items // batch_size

    @jax.jit
    def epoch(rng, state, data):
        def minibatch(carry, idx):
            rng, state = carry
            head, opt = nnx.merge(graphdef, state)
            rng, key = jax.random.split(rng)
            batch = batch_fn(data, key, idx)
            loss, grads = nnx.value_and_grad(lambda h: h.loss(*batch))(head)
            opt.update(grads)
            return (rng, nnx.split((head, opt))[1]), loss

        _, key = jax.random.split(rng)  # bt-kinetix: the permutation key; the minibatches continue from rng itself
        perm = jax.random.permutation(key, n_items)[: steps * batch_size].reshape(steps, batch_size)
        (rng, state), losses = jax.lax.scan(minibatch, (rng, state), perm)
        return rng, state, losses.mean()

    losses = []
    for _ in range(num_epochs):
        rng, state, loss = epoch(rng, state, data)
        losses.append(float(loss))
    return nnx.merge(graphdef, state)[0], losses


def expert_data(path, H: int = H, batch_size: int = BATCH):
    """bt-kinetix's data layout: npz [steps, envs, ...] -> per-env contiguous [(envs steps), ...], truncated so the
    window starts are a whole number of batches. Returns ({obs, action, done}, number of window starts)."""
    with np.load(path) as z:
        data = {k: einops.rearrange(z[k], "s e ... -> (e s) ...") for k in ("obs", "action", "done")}
    n = (data["obs"].shape[0] - H + 1) // batch_size * batch_size
    return {k: jnp.asarray(v[: n + H - 1]) for k, v in data.items()}, n


def expert_batch(policy, H: int = H, num_steps: int = 5):
    """bt-kinetix create_residual_batch: the window at idx, a base chunk from its first obs (own noise per item), a
    random position t before the window's first done; the target is the expert's action at t."""

    def fn(data, key, idx):
        win = idx[:, None] + jnp.arange(H)
        obs, act, done = data["obs"][win], data["action"][win], data["done"][win]
        done_idx = jnp.where(done.any(-1), done.argmax(-1), H)
        B = idx.shape[0]
        keys = jax.random.split(key, B + 1)[1:]
        base = jax.vmap(lambda o, k: policy.action(k, o[None], num_steps)[0])(obs[:, 0], keys)  # [B, H, A]
        t = jax.random.randint(key, (B,), 0, jnp.maximum(done_idx, 1))
        r = jnp.arange(B)
        return obs[r, t], jax.lax.stop_gradient(base[r, t]), time_feature(t, H), act[r, t]

    return fn


def distill_data(env, env_params, policy, level, key, num_envs: int, num_chunks: int, num_steps: int = 5):
    """Spec §9: naive rollouts at d = 0, s = H with the eval env's action noise. Per chunk (a window): the obs before
    each of its H steps, the chunk, its noise z and done after each step: obs [W, H, O], chunk, z [W, H, A], done
    [W, H], W = num_chunks * num_envs (chunk-major). env: AutoReplay(NoisyAction(raw env))."""
    Hh, A = policy.action_chunk_size, policy.action_dim
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(
        jax.random.split(k_reset, num_envs), level, env_params
    )
    env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def run_chunk(carry, key):
        obs, state = carry
        k_z, k_env = jax.random.split(key)
        z = jax.random.normal(k_z, (num_envs, Hh, A))
        chunk = policy.action_from_noise(z, obs, num_steps)

        def one_step(c, xs):
            obs, state = c
            a, k = xs
            nobs, state, _, done, _ = env_step(jax.random.split(k, num_envs), state, a, env_params)
            return (nobs, state), (obs, done)

        (obs, state), (o, done) = jax.lax.scan(
            one_step, (obs, state), (chunk.swapaxes(0, 1), jax.random.split(k_env, Hh))
        )
        return (obs, state), (o.swapaxes(0, 1), chunk, z, done.swapaxes(0, 1))

    _, out = jax.lax.scan(run_chunk, (obs, state), jax.random.split(k_run, num_chunks))
    return jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), out)


def fresh_targets(policy, obs, z, num_steps: int = 5, batch_size: int = 1024):
    """a*[w, k] = pi(roll(z_w, -k), obs[w, k])[0]: the fresh call with the chunk's shifted noise, the B2 yardstick."""

    def one(x):
        o, zz = x
        return policy.action_from_noise(reflex.shifted_noise(zz), o, num_steps)[:, 0]

    return jax.lax.map(one, (obs, z), batch_size=batch_size)


def distill_items(obs, chunk, done, targets):
    """Valid (window, k) pairs as flat arrays: k before the window's first done, as bt-kinetix samples positions."""
    Hh = obs.shape[1]
    done_idx = jnp.where(done.any(-1), done.argmax(-1), Hh)
    w, k = np.nonzero(np.asarray(jnp.arange(Hh)[None] < jnp.maximum(done_idx, 1)[:, None]))
    return {"obs": obs[w, k], "base": chunk[w, k], "k": jnp.asarray(k), "target": targets[w, k]}


def _mse(head, obs, base, tf, target):
    """(head's MSE, the base action's MSE) against the target."""
    return (float(jnp.mean(jnp.square(head.apply_residual(obs, base, tf) - target))),
            float(jnp.mean(jnp.square(base - target))))


def _save(head, out_dir: str, level_path: str, row: dict):
    """<out_dir>/<level>.pkl (atomic) and a row of <out_dir>/train_log.csv."""
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    f = out / f"{predictors.level_name(level_path)}.pkl"
    part = f.with_suffix(".part")
    with part.open("wb") as fh:
        pickle.dump(nnx.state(head).to_pure_dict(), fh)
    os.replace(part, f)
    log = out / "train_log.csv"
    new = not log.exists()
    with log.open("a") as fh:
        if new:
            fh.write(",".join(row) + "\n")
        fh.write(",".join(str(v) for v in row.values()) + "\n")
    print(json.dumps(row), flush=True)


def _env_and_dims(run_path: str, level_path: str):
    env, env_params, levels, O, A = probe.setup(list(probe.LEVELS))
    i = probe.LEVELS.index(level_path)
    policy = probe.make_policy(probe.load_state_dict(run_path, level_path), O, A)
    return env, env_params, jax.tree.map(lambda x: x[i], levels), policy, O, A, i


def expert(data_dir: str, level_paths: Sequence[str] = probe.LEVELS, out_dir: str = "results/b2b5/a2c2",
           run_path: str = "checkpoints/bc", seed: int = 0, num_epochs: int = EPOCHS, check_items: int = 20_000):
    """Spec §9 A2C2: one head per level from <data_dir>/<level>.npz (the pod downloads, trains and deletes one level
    at a time). The MSE check is in-sample (bt-kinetix trains on all of the data)."""
    keys = jax.random.split(jax.random.key(seed), len(probe.LEVELS))
    for level_path in level_paths:
        start = time.time()
        _, _, _, policy, O, A, i = _env_and_dims(run_path, level_path)
        data, n = expert_data(pathlib.Path(data_dir) / f"{predictors.level_name(level_path)}.npz")
        head, losses = fit(keys[i], O, A, expert_batch(policy), data, n, num_epochs)
        idx = jax.random.choice(jax.random.key(1), n, (min(check_items, n),), replace=False)
        mse, mse_base = _mse(head, *expert_batch(policy)(data, jax.random.key(2), idx))
        _save(head, out_dir, level_path, {
            "level": level_path, "transitions": int(data["obs"].shape[0]), "items": n,
            "steps": num_epochs * (n // BATCH), "loss_first": losses[0], "loss_last": losses[-1], "mse": mse,
            "mse_base": mse_base, "check": "in-sample", "seconds": round(time.time() - start),
        })


def distill(level_paths: Sequence[str] = probe.LEVELS, out_dir: str = "results/b2b5/a2c2_distill",
            run_path: str = "checkpoints/bc", num_envs: int = 128, num_chunks: int = 992, seed: int = 5000,
            num_epochs: int = EPOCHS, num_steps: int = 5):
    """Spec §9 A2C2-distill: per level, num_envs x num_chunks x H transitions (default ~1 M, the expert set's size),
    targets = fresh calls with the chunk's shifted noise, the A2C2 recipe; MSE checked on 10% extra held-out windows."""
    for level_path in level_paths:
        start = time.time()
        env, env_params, level, policy, O, A, i = _env_and_dims(run_path, level_path)
        noisy = wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env._env))  # the eval env, per env
        k_data, k_val, k_fit = jax.random.split(jax.random.key(seed + i), 3)
        targets = jax.jit(lambda o, z: fresh_targets(policy, o, z, num_steps))  # nnx modules go in closures, not args

        def items(k, chunks):
            obs, chunk, z, done = jax.jit(
                lambda kk: distill_data(noisy, env_params, policy, level, kk, num_envs, chunks, num_steps)
            )(k)
            return distill_items(obs, chunk, done, targets(obs, z)), obs.shape[0] * H

        train, n_trans = items(k_data, num_chunks)
        val, _ = items(k_val, max(1, num_chunks // 10))
        n = len(train["k"])

        def batch(d, key, idx):
            return d["obs"][idx], d["base"][idx], time_feature(d["k"][idx], H), d["target"][idx]

        head, losses = fit(k_fit, O, A, batch, train, n, num_epochs)
        mse, mse_base = _mse(head, val["obs"], val["base"], time_feature(val["k"], H), val["target"])
        _save(head, out_dir, level_path, {
            "level": level_path, "transitions": n_trans, "items": n, "steps": num_epochs * (n // BATCH),
            "loss_first": losses[0], "loss_last": losses[-1], "mse": mse, "mse_base": mse_base, "check": "held-out",
            "seconds": round(time.time() - start),
        })


def synth(level_path: str, out_dir: str, run_path: str = "checkpoints/bc", num_envs: int = 16, num_chunks: int = 80,
          seed: int = 99):
    """Rehearsal only: a toy 'expert' npz in bt-kinetix's layout ([steps, envs, ...]: obs, action, done) from the BC
    policy's own naive rollouts, so `expert` runs end to end on the Mac without the 33 GB dataset."""
    env, env_params, level, policy, O, A, _ = _env_and_dims(run_path, level_path)
    noisy = wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env._env))
    obs, chunk, _, done = jax.jit(lambda k: distill_data(noisy, env_params, policy, level, k, num_envs, num_chunks))(
        jax.random.key(seed)
    )

    def steps_envs(x):  # [C*E, H, ...] chunk-major -> [C*H, E, ...]
        x = np.asarray(x).reshape(num_chunks, num_envs, H, *x.shape[2:])
        return np.swapaxes(x, 1, 2).reshape(num_chunks * H, num_envs, *x.shape[3:])

    pathlib.Path(out_dir).mkdir(parents=True, exist_ok=True)
    np.savez(pathlib.Path(out_dir) / f"{predictors.level_name(level_path)}.npz",
             obs=steps_envs(obs), action=steps_envs(chunk), done=steps_envs(done))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"expert": expert, "distill": distill, "synth": synth})
