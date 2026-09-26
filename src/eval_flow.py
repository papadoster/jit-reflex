import collections
import dataclasses
import functools
import math
import os
import pathlib
import pickle
import time
from typing import Sequence

import flax.nnx as nnx
import jax
from jax.experimental import shard_map
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import kinetix.render.renderer_pixels as renderer_pixels
import numpy as np
import pandas as pd
import tyro

import a2c2
import model as _model
import predictors as _predictors
import reflex
import train_expert


@dataclasses.dataclass(frozen=True)
class NaiveMethodConfig:
    pass


@dataclasses.dataclass(frozen=True)
class RealtimeMethodConfig:
    prefix_attention_schedule: _model.PrefixAttentionSchedule = "exp"
    max_guidance_weight: float = 5.0


@dataclasses.dataclass(frozen=True)
class BIDMethodConfig:
    n_samples: int = 16
    bid_k: int | None = None


@dataclasses.dataclass(frozen=True)
class ReflexMethodConfig:
    requery: bool = True  # nominal action = pi(z, predicted obs)[0] instead of the chunk's action
    feedback: bool = True  # add clip(J @ (obs - predicted obs)) at every step
    max_correction: float = 1.0
    package_batch: int = 16  # envs per Jacobian batch (memory knob)
    rtc: bool = False  # E2b: the chunk comes from RTC (realtime_action) instead of plain sampling
    predictor: str = "oracle"  # B1: "oracle" | "phys" (wrong physics) | "learned" (world model), spec B1 3.1
    phys_error: float = 0.0  # B1: relative parameter error of the "phys" predictor
    cand: str | None = None  # B2+B5: "T3" / "M3" package (reflex.first_action_fn); None = the exact reflex
    j_delay: int = 0  # B2+B5 late J: no J on the first j_delay executed positions of the new chunk


@dataclasses.dataclass(frozen=True)
class HeadMethodConfig:
    head: str = "a2c2"  # B2+B5 §9: naive chunk + a per-step residual head (a2c2.Head) from <heads_root>/<head>/


@dataclasses.dataclass(frozen=True)
class EvalConfig:
    step: int = -1
    weak_step: int | None = None
    num_evals: int = 2048
    num_flow_steps: int = 5

    inference_delay: int = 0
    execute_horizon: int = 1
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig | ReflexMethodConfig | HeadMethodConfig = (
        NaiveMethodConfig()
    )
    kick_prob: float = 0.0  # E2 perturbations: per-step probability of a velocity kick to all dynamic bodies
    kick_std: float = 0.0

    model: _model.ModelConfig = _model.ModelConfig()


def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicy | None = None,
    world_model=None,
    head=None,
    e_std=None,
    e_edges=None,
):
    base_env = env
    if config.kick_prob > 0:
        env = train_expert.KickWrapper(env, config.kick_prob, config.kick_std)
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    # the reflex's (oracle) predictor: raw physics, no noise, no kicks, and no auto-reset, so a predicted solve
    # doesn't turn the rest of the prediction into the level's initial observation
    nominal_step = jax.vmap(base_env.step_env, in_axes=(None, 0, 0, None))
    is_reflex = isinstance(config.method, ReflexMethodConfig)
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"
    d, s = config.inference_delay, config.execute_horizon
    assert s + d <= policy.action_chunk_size, f"{s=} + {d=} > H: padded zero actions would be executed"
    H = policy.action_chunk_size
    # B2+B5: chunk index of each executed step (the first d come from the previous chunk), for the head and |e|
    ks = jnp.concatenate([jnp.arange(d) + s, jnp.arange(d, s)]) if (head is not None or e_std is not None) else None
    phys_key = jax.random.fold_in(rng, 1)  # B1: the same parameter-error signs for every method (paired comparison)

    def predict(key, raw_state, obs, actions):
        """Predicted obs after each planned action [B, T, A] -> [B, T, O] (spec B1 3.1)."""
        m = config.method
        if m.predictor == "learned":
            assert world_model is not None, "predictor 'learned' needs world models: src/predictors.py train"
            return jax.vmap(_predictors.wm_rollout, in_axes=(None, 0, 0))(world_model, obs, actions)
        if m.predictor == "phys":
            factors = _predictors.phys_factors(phys_key, raw_state, m.phys_error)
            step = jax.vmap(functools.partial(_predictors.phys_step, base_env), in_axes=(None, 0, 0, None, 0))

            def fn(st, a):
                return step(key, st, a, env_params, factors)
        else:
            assert m.predictor == "oracle", m.predictor

            def fn(st, a):
                return nominal_step(key, st, a, env_params)[:2]

        return reflex.nominal_obs(fn, raw_state, actions.swapaxes(0, 1)).swapaxes(0, 1)

    def execute_chunk(carry, _):
        def step(carry, xs):
            rng, obs, env_state, alive, hist = carry
            action, pkg_t, k, w_t = xs
            if pkg_t is not None and "gain" in pkg_t:
                action = reflex.correct(action, pkg_t["gain"], pkg_t["ref"], obs, config.method.max_correction)
            if head is not None:  # B2+B5 §9: the residual on the chunk's action at chunk index k
                action = head.apply_residual(obs, action, jnp.broadcast_to(a2c2.time_feature(k, H), (obs.shape[0], 2)))
            if hist is not None:  # B2+B5 §8: |o - o^| of this step in the diagnostic's units, first episode only
                e = _predictors.normalized_error(pkg_t["ref"], obs, e_std)
                b = jnp.searchsorted(e_edges, e, side="right")
                hist = hist.at[k].add(jnp.zeros(hist.shape[1], hist.dtype).at[b].add((alive & w_t).astype(hist.dtype)))
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            if alive is not None:
                alive = alive & ~done
            return (rng, next_obs, next_env_state, alive, hist), (done, env_state, info)

        rng, obs, env_state, action_chunk, n, pkg, alive, c, hist = carry
        # B2+B5 §8: the first chunk's first d steps run the initial package, whose ref is the reset obs (no prediction)
        w = None if c is None else (c > 0) | (jnp.arange(s) >= d)
        rng, key = jax.random.split(rng)
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"{config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
            )
        elif is_reflex:
            noise = jax.random.normal(key, (obs.shape[0], policy.action_chunk_size, policy.action_dim))
            if config.method.rtc:  # E2b: RTC chunk (its initial noise is this same `noise`), reflex feedback on top
                next_action_chunk = policy.realtime_action(
                    key, obs, config.num_flow_steps, action_chunk, d, policy.action_chunk_size - s, "exp", 5.0
                )
            else:
                next_action_chunk = policy.action_from_noise(noise, obs, config.num_flow_steps)  # == policy.action(key)
            # steps t..t+H-1 run the previous package's actions for d steps, then this chunk
            planned = jnp.concatenate([pkg["nom"][:, :d], next_action_chunk[:, d:]], axis=1)
            n_pred = max(d + s - 1, 1)  # the package reads only chunk indices d..d+s-1 (spec B1 3.4)
            # BatchEnv/LogWrapper -> AutoReplay -> raw EnvState
            pred = predict(key, env_state.env_state.env_state, obs, planned[:, :n_pred])  # [B, n_pred, O]
            ref = jnp.concatenate([obs[:, None], pred], axis=1)  # predicted obs per chunk index
            ref = jnp.pad(ref, ((0, 0), (0, policy.action_chunk_size - ref.shape[1]), (0, 0)))  # never read past d+s-1
            nom, gain = reflex.package(
                policy,
                noise,
                ref,
                next_action_chunk,
                config.num_flow_steps,
                config.method.requery,
                config.method.feedback,
                config.method.package_batch,
                used=(d, d + s),
                cand=config.method.cand,
                j_from=d + config.method.j_delay if config.method.j_delay else 0,
            )
            new_pkg = {"nom": nom, "ref": ref} | ({} if gain is None else {"gain": gain})
        elif isinstance(config.method, HeadMethodConfig):  # the base chunk is naive's (same key, same chunk)
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # we execute `inference_delay` actions from the *previously generated* action chunk, and then the remaining
        # `execute_horizon - inference_delay` actions from the newly generated action chunk
        action_chunk_to_execute = jnp.concatenate([action_chunk[:, :d], next_action_chunk[:, d:s]], axis=1)
        xs_pkg, next_pkg = None, None
        if is_reflex:
            # the package lives in chunk frame and is merged and shifted exactly like the action chunk
            exec_pkg = jax.tree.map(lambda old, new: jnp.concatenate([old[:, :d], new[:, d:s]], axis=1), pkg, new_pkg)
            action_chunk_to_execute = exec_pkg.pop("nom")
            xs_pkg = jax.tree.map(lambda x: x.swapaxes(0, 1), exec_pkg)
            next_pkg = jax.tree.map(lambda x: jnp.concatenate([x[:, s:], jnp.zeros_like(x[:, :s])], axis=1), new_pkg)
        # throw away the first `execute_horizon` actions from the newly generated action chunk, to align it with the
        # correct frame of reference for the next scan iteration
        next_action_chunk = jnp.concatenate(
            [next_action_chunk[:, s:], jnp.zeros((obs.shape[0], s, policy.action_dim))], axis=1
        )
        next_n = jnp.concatenate([n[s:], jnp.zeros(s, dtype=jnp.int32)])
        (rng, next_obs, next_env_state, alive, hist), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state, alive, hist), (action_chunk_to_execute.transpose(1, 0, 2), xs_pkg, ks, w)
        )
        next_c = None if c is None else c + 1
        return (rng, next_obs, next_env_state, next_action_chunk, next_n, next_pkg, alive, next_c, hist), (
            dones, env_states, infos,
        )

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    pkg = None
    if is_reflex:  # the first d steps of the first chunk run open-loop
        pkg = {"nom": action_chunk, "ref": jnp.repeat(obs[:, None], action_chunk.shape[1], axis=1)}
        if config.method.feedback:
            pkg["gain"] = jnp.zeros((*action_chunk.shape, obs.shape[-1]))
    alive = jnp.ones(config.num_evals, bool) if e_std is not None else None
    c = jnp.zeros((), jnp.int32) if e_std is not None else None  # executed chunks so far
    hist = jnp.zeros((H, len(e_edges) + 1), jnp.float32) if e_std is not None else None
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    (*_, hist), (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n, pkg, alive, c, hist),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    first_done_idx = jnp.argmax(dones, axis=0)  # only consider the first episode of each rollout
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    solved = infos["returned_episode_solved"][first_done_idx, jnp.arange(config.num_evals)]
    lengths = infos["returned_episode_lengths"][first_done_idx, jnp.arange(config.num_evals)]
    return_info["solved_length"] = jnp.where(  # nan, not 0, when nothing is solved: plot means skip it
        jnp.sum(solved) > 0, jnp.sum(lengths * solved) / jnp.maximum(jnp.sum(solved), 1), jnp.nan
    )
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    if hist is not None:
        return_info["e_hist"] = hist
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video


METHODS = {
    "naive": NaiveMethodConfig(),
    "realtime": RealtimeMethodConfig(),
    "bid": BIDMethodConfig(),
    "hard_masking": RealtimeMethodConfig(prefix_attention_schedule="zeros"),
    "pred": ReflexMethodConfig(feedback=False),
    "reflex": ReflexMethodConfig(),
    "reflex_chunk": ReflexMethodConfig(requery=False),
    "reflex_off": ReflexMethodConfig(requery=False, feedback=False),  # sanity check: must reproduce naive
    "rtc_reflex": ReflexMethodConfig(requery=False, rtc=True),  # E2b: RTC chunk + J feedback
    "rtc_reflex_off": ReflexMethodConfig(requery=False, feedback=False, rtc=True),  # sanity: must reproduce realtime
    "t3": ReflexMethodConfig(cand="T3"),  # B2+B5 §3
    "m3": ReflexMethodConfig(cand="M3"),
    **{f"late{k}": ReflexMethodConfig(cand="T3", j_delay=k) for k in range(1, 5)},
    "realtime10": RealtimeMethodConfig(),  # RTC with 10 flow steps, see FLOW_STEPS
    "a2c2": HeadMethodConfig("a2c2"),
    "a2c2_distill": HeadMethodConfig("a2c2_distill"),
}

FLOW_STEPS = {"realtime10": 10}  # B2+B5: methods whose chunk (and first chunk) use more flow steps
HIST = ("pred", "reflex", "t3", "m3", *(f"late{k}" for k in range(1, 5)))  # B2+B5 §8: |e| histograms, learned only
KEYS = ["seed", "delay", "execute_horizon", "method", "predictor"]  # one config of a run (resume, hist.csv)


def e_edges(bins_csv: str) -> np.ndarray:
    """Inner edges of the diagnostic's 10 |e| bins (predictor "all"): searchsorted(side="right") gives the bin."""
    b = pd.read_csv(bins_csv).query("predictor == 'all'").sort_values("bin")
    return b["e_lo"].to_numpy(np.float32)[1:]


def parse_cells(cells: Sequence[str]) -> list[tuple[int, int]]:
    """B2+B5 grid cells "d,s" -> [(d, s)], in the given order."""
    return [tuple(int(x) for x in c.split(",")) for c in cells]


def stale(df: pd.DataFrame, level_paths: Sequence[str], done: set) -> np.ndarray:
    """Rows of this call's levels whose config is not done: it reruns, so they would be written twice."""
    keys = df[KEYS].itertuples(index=False, name=None)
    return df["level"].isin(level_paths).to_numpy() & np.array([k not in done for k in keys], bool)


def load_done(out_dir: pathlib.Path, level_paths: Sequence[str]) -> tuple[pd.DataFrame, set]:
    """Resume (B2+B5 §10): (old rows to keep, keys of the configs with a row for every level of this call). A config
    with only some of them reruns, so its rows of these levels go; every other row (other levels, other configs) stays,
    so a call with another level set in the same folder deletes nothing of the others."""
    if not (out_dir / "results.csv").exists():
        return pd.DataFrame(), set()
    old = pd.read_csv(out_dir / "results.csv")
    n = old[old["level"].isin(level_paths)].groupby(KEYS)["level"].nunique()
    done = set(n.index[n == len(set(level_paths))])
    return old[~stale(old, level_paths, done)], done


def horizons_for(delay: int, chunk_size: int, horizons: Sequence[int], minmax: bool) -> list[int]:
    """Execute horizons to evaluate at `delay`: explicit list, {min, max}, or upstream's full sweep.

    Keeps max(1, delay) <= s <= chunk_size - delay: upstream asserts s >= d, and s + d > H would execute padding zeros.
    """
    lo, hi = max(1, delay), chunk_size - delay
    if horizons:
        return [s for s in horizons if lo <= s <= hi]
    return sorted({lo, hi}) if minmax else list(range(lo, hi + 1))


def parse_predictor(name: str) -> dict:
    """CLI predictor name -> ReflexMethodConfig fields: 'oracle', 'learned' or 'phys<p>' (e.g. 'phys0.2')."""
    if name.startswith("phys"):
        return {"predictor": "phys", "phys_error": float(name[4:])}
    assert name in ("oracle", "learned"), f"unknown predictor {name!r}"
    return {"predictor": name}


def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
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
    ),
    seeds: Sequence[int] = (0,),
    output_dir: str | None = "eval_output",
    methods: Sequence[str] = ("naive", "realtime", "bid", "hard_masking"),
    delays: Sequence[int] = (0, 1, 2, 3, 4),
    horizons: Sequence[int] = (),
    minmax: bool = False,
    max_correction: float = 1.0,
    package_batch: int = 16,
    predictors: Sequence[str] = ("oracle",),  # B1: predictors for the reflex methods, see parse_predictor
    world_model_dir: str = _predictors.WM_DIR,
    cells: Sequence[str] = (),  # B2+B5: explicit "d,s" cells instead of delays x horizons
    heads_root: str = "results/b2b5",  # B2+B5 §9: <heads_root>/<a2c2|a2c2_distill>/<level>.pkl
    e_std: str | None = None,  # B2+B5 §8: results/b2b5/e_std.npz switches the |e| histograms on (HIST, learned)
    e_bins: str = "results/b1/diag/bins.csv",
):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)

    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)

    # load policies from best checkpoints by solve rate
    state_dicts = []
    weak_state_dicts = []
    for level_path in level_paths:
        level_name = level_path.replace("/", "_").replace(".json", "")
        log_dirs = list(filter(lambda p: p.is_dir() and p.name.isdigit(), pathlib.Path(run_path).iterdir()))
        log_dirs = sorted(log_dirs, key=lambda p: int(p.name))
        # load policy
        with (log_dirs[config.step] / "policies" / f"{level_name}.pkl").open("rb") as f:
            state_dicts.append(pickle.load(f))
        if config.weak_step is not None:
            with (log_dirs[config.weak_step] / "policies" / f"{level_name}.pkl").open("rb") as f:
                weak_state_dicts.append(pickle.load(f))
    state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *state_dicts))
    if config.weak_step is not None:
        weak_state_dicts = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *weak_state_dicts))
    else:
        weak_state_dicts = None
    world_models = None
    if "learned" in predictors:
        wms = []
        for level_path in level_paths:
            with (pathlib.Path(world_model_dir) / f"{_predictors.level_name(level_path)}.pkl").open("rb") as f:
                wms.append(pickle.load(f))
        world_models = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *wms))
    heads = {
        m: jax.device_put(a2c2.load_heads(heads_root, METHODS[m].head, level_paths))
        for m in methods
        if isinstance(METHODS[m], HeadMethodConfig)
    }
    e_stds, edges = None, e_edges(e_bins)
    if e_std is not None:
        with np.load(e_std) as z:
            e_stds = jax.device_put(jnp.stack([jnp.asarray(z[_predictors.level_name(p)]) for p in level_paths]))

    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)[
        0
    ].shape[-1]
    action_dim = env.action_space(env_params).shape[0]

    mesh = jax.make_mesh((jax.local_device_count(),), ("x",))
    pspec = jax.sharding.PartitionSpec("x")
    sharding = jax.sharding.NamedSharding(mesh, pspec)

    @functools.partial(jax.jit, static_argnums=(0,), in_shardings=sharding, out_shardings=sharding)
    @functools.partial(
        shard_map.shard_map,
        mesh=mesh,
        in_specs=(None, pspec, pspec, pspec, pspec, pspec, pspec, pspec),
        out_specs=pspec,
    )
    @functools.partial(jax.vmap, in_axes=(None, 0, 0, 0, 0, 0, 0, 0))
    def _eval(config, rng, level, state_dict, weak_state_dict, world_model, head_state, e_std):
        policy = _model.FlowPolicy(
            obs_dim=obs_dim,
            action_dim=action_dim,
            config=config.model,
            rngs=nnx.Rngs(rng),
        )
        graphdef, state = nnx.split(policy)
        state.replace_by_pure_dict(state_dict)
        policy = nnx.merge(graphdef, state)
        if weak_state_dict is not None:
            graphdef, state = nnx.split(policy)
            state.replace_by_pure_dict(weak_state_dict)
            weak_policy = nnx.merge(graphdef, state)
        else:
            weak_policy = None
        head = None if head_state is None else a2c2.make_head(head_state, obs_dim, action_dim)
        eval_info, _ = eval(
            config, env, rng, level, policy, env_params, static_env_params, weak_policy, world_model, head, e_std, edges
        )
        return eval_info

    out_dir = pathlib.Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    old, done = load_done(out_dir, level_paths)  # old rows are kept as they are, even with other columns
    if (out_dir / "results.csv").exists():
        print(f"resume: {len(done)} configs already done in {out_dir}")
    results = collections.defaultdict(list)  # this run's rows
    hist_old, hist_rows = pd.DataFrame(), []  # |e| histograms (B2+B5 §8): old rows as in load_done, this run's rows
    if (out_dir / "hist.csv").exists():
        hist_old = pd.read_csv(out_dir / "hist.csv")
        hist_old = hist_old[~stale(hist_old, level_paths, done)]
    grid = parse_cells(cells) or [
        (d, s) for d in delays for s in horizons_for(d, config.model.action_chunk_size, horizons, minmax)
    ]
    for seed in seeds:
        rngs = jax.random.split(jax.random.key(seed), len(level_paths))
        for inference_delay, execute_horizon in grid:
            for name in methods:
                method = METHODS[name]
                is_reflex = isinstance(method, ReflexMethodConfig)
                for predictor in predictors if is_reflex else ("-",):
                    if (seed, inference_delay, execute_horizon, name, predictor) in done:
                        continue
                    print(f"{time.strftime('%H:%M:%S')} {seed=} {name=} {predictor=} {inference_delay=} "
                          f"{execute_horizon=}")
                    m = method
                    if is_reflex:
                        m = dataclasses.replace(
                            method, max_correction=max_correction, package_batch=package_batch,
                            **parse_predictor(predictor),
                        )
                    c = dataclasses.replace(
                        config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=m,
                        num_flow_steps=FLOW_STEPS.get(name, config.num_flow_steps),
                    )
                    start = time.time()
                    hist_on = e_stds is not None and name in HIST and predictor == "learned"
                    out = jax.device_get(_eval(
                        c, rngs, levels, state_dicts, weak_state_dicts, world_models, heads.get(name),
                        e_stds if hist_on else None,
                    ))
                    hist = out.pop("e_hist", None)
                    if hist is not None:
                        for i, level_path in enumerate(level_paths):
                            for k, b in zip(*np.nonzero(hist[i])):
                                hist_rows.append({
                                    "seed": seed, "delay": inference_delay, "execute_horizon": execute_horizon,
                                    "method": name, "predictor": predictor, "level": level_path, "k": int(k),
                                    "bin": int(b), "count": int(hist[i][k, b]),
                                })
                    for i in range(len(level_paths)):
                        for k, v in out.items():
                            results[k].append(v[i])
                        results["seed"].append(seed)
                        results["delay"].append(inference_delay)
                        results["method"].append(name)
                        results["predictor"].append(predictor)
                        results["level"].append(level_paths[i])
                        results["execute_horizon"].append(execute_horizon)
                        results["max_correction"].append(max_correction if is_reflex else float("nan"))
                        results["kick_std"].append(config.kick_std if config.kick_prob > 0 else 0.0)
                        results["seconds"].append(time.time() - start)
                    # after every config: a crash (or Ctrl-C) keeps the finished ones; written aside and renamed,
                    # so a kill mid-write never truncates results.csv; hist.csv first, so a config done in results.csv
                    # always has its |e| rows
                    if len(hist_old) or hist_rows:
                        part = out_dir / "hist.csv.part"
                        pd.concat([hist_old, pd.DataFrame(hist_rows)], ignore_index=True).to_csv(part, index=False)
                        os.replace(part, out_dir / "hist.csv")
                    part = out_dir / "results.csv.part"
                    pd.concat([old, pd.DataFrame(results)], ignore_index=True).to_csv(part, index=False)
                    os.replace(part, out_dir / "results.csv")


if __name__ == "__main__":
    tyro.cli(main)
