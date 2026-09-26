"""B1 diagnostic (exploratory): which prediction error J sees, and at which deviation and horizon the tangent breaks.

For a sampled state: a plan of a few policy chunks, each queried where the noise-free truth arrives; real rollouts add
action noise; every predictor rolls the same plan. At step k the reflex's pi(ô_k) + J·(o_k − ô_k) is compared with a
fresh call pi(o_k), as in E1b. See docs/superpowers/specs/2026-09-26-b1-diagnostic-design.md.
"""

from typing import Sequence

import jax
import jax.numpy as jnp

import predictors
import probe
import reflex
import train_expert

# per-pair quantities summed within a level (pooled ratios) and averaged (means); spec section 4
SUMS = ("pred", "lin", "lin_clip", "chunk", "chunk_lin_clip",
        "rs_pred", "top_pred", "d_pred", "rs_noise", "top_noise", "d_noise")
MEANS = ("e", "e_pred", "e_noise", "je", "je_pred", "je_noise", "cos_fix", "cos_spur", "clip")


def row_space_share(jac, e):
    """|e|^2 inside J's row space and along its top right singular vector (divide by |e|^2 for the shares).

    jac [..., A, O], e [..., O] (broadcast) -> ([...], [...]). Singular values below 1e-6 of the largest are rank loss.
    """
    _, s, vt = jnp.linalg.svd(jac, full_matrices=False)
    proj = jnp.einsum("...ao,...o->...a", vt, e)
    return jnp.sum(jnp.where(s > 1e-6 * s[..., :1], proj**2, 0.0), -1), proj[..., 0] ** 2


def cosine(x, y):
    """Cosine over the last axis; 0 where either vector is zero."""
    den = jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1)
    return jnp.where(den > 0, jnp.sum(x * y, -1) / jnp.where(den > 0, den, 1.0), 0.0)


def roll(step, state, actions):
    """step(state, action) -> (obs, state, done). actions [T, A] -> obs [T, O], ended by then [T], final state."""

    def body(c, a):
        s, ended = c
        o, s, done = step(s, a)
        return (s, ended | done), (o, ended | done)

    (state, _), (obs, ended) = jax.lax.scan(body, (state, jnp.bool_(False)), actions)
    return obs, ended, state


def chain(policy, step, state, obs, zs, num_steps: int):
    """Plan of len(zs) chunks; chunk c = pi(zs[c], o) at the obs where the noise-free truth arrived.

    zs [C, H, A] -> plan [C*H, A], truth obs after each action [C*H, O], ended by then [C*H].
    """
    plan, truth, ends, ended = [], [], [], jnp.bool_(False)
    for z in zs:  # C is small and static
        chunk = policy.action_from_noise(z[None], obs[None], num_steps)[0]
        o, e, state = roll(step, state, chunk)
        plan.append(chunk)
        truth.append(o)
        ends.append(e | ended)
        obs, ended = o[-1], ends[-1][-1]
    return jnp.concatenate(plan), jnp.concatenate(truth), jnp.concatenate(ends)


def probe_state(
    policy, base, params, wm, phys: Sequence[float], raw, obs, key, std, num_draws: int, num_chunks: int,
    num_steps: int,
):
    """Diagnostic of one state (spec sections 3-4).

    Returns SUMS + MEANS, each [P, M, K] (predictors oracle, phys..., learned; draws; k = 1..T-1), plus valid [M, K]
    (common to all predictors), a_ref [P, K, A] and plan [T, A] for the invariant checks. base is the raw Kinetix env
    (no auto-reset), raw its state. Actions are compared as executed (probe.executed), as in E1b.
    """
    H, A = policy.action_chunk_size, policy.action_dim
    k_z, k_f, k_n = jax.random.split(key, 3)
    zs = jax.random.normal(k_z, (num_chunks, H, A))

    def true_step(s, a):
        o, s, _, done, _ = base.step_env(key, s, a, params)
        return o, s, done

    plan, truth, truth_ended = chain(policy, true_step, raw, obs, zs, num_steps)
    T = plan.shape[0]
    K = T - 1
    noisy = plan + train_expert.ACTION_NOISE_STD * jax.random.normal(k_n, (num_draws, T, A))
    real, real_ended, _ = jax.vmap(lambda a: roll(true_step, raw, a))(noisy)  # [M, T, O], [M, T]

    preds = [truth]
    for p in phys:
        f = predictors.phys_factors(k_f, raw, p)  # same key: same signs for every p, as in B1

        def phys_step(s, a, f=f):
            o, s = predictors.phys_step(base, key, s, a, params, f)
            return o, s, jnp.bool_(False)  # a prediction is not cut by the episode's end

        preds.append(roll(phys_step, raw, plan)[0])
    preds.append(predictors.wm_rollout(wm, obs, plan))
    nom = jnp.stack(preds)[:, :K]  # [P, K, O]: ô_k after k actions, k = 1..K
    o, o_star = real[:, :K], truth[:K]  # [M, K, O], [K, O]
    z = jax.vmap(reflex.shifted_noise)(zs).reshape(T, H, A)[1:]  # z[k-1]: row 0 is the noise plan[k] came from
    plan_k = plan[1:]  # the plan's own action at step k: the nominal of rtc_reflex-like corrections (measure 8)
    a_star = policy.action_from_noise(
        jnp.broadcast_to(z, (num_draws, K, H, A)).reshape(-1, H, A), o.reshape(num_draws * K, -1), num_steps
    )[:, 0].reshape(num_draws, K, A)
    valid = ~(real_ended[:, :K] | truth_ended[:K])

    def act(a):
        return probe.executed(a, raw)

    need = act(a_star)

    def per_predictor(n):
        a_ref, jac = jax.vmap(lambda zk, ok: reflex.first_action_and_jacobian(policy, zk, ok, num_steps))(z, n)
        err = probe.errors(plan_k, a_ref, jac, n, o, a_star, act=act)  # [M, K]
        e_pred, e_noise = o_star - n, o - o_star  # [K, O], [M, K, O]
        je = jnp.einsum("kao,mko->mka", jac, o - n)
        je_pred = jnp.einsum("kao,ko->ka", jac, e_pred)
        rs_pred, top_pred = row_space_share(jac, e_pred)
        rs_noise, top_noise = row_space_share(jac, e_noise)
        out = {name: err[name] for name in ("pred", "lin", "lin_clip", "chunk", "chunk_lin_clip")} | {
            "rs_pred": rs_pred, "top_pred": top_pred, "d_pred": jnp.sum(e_pred**2, -1),
            "rs_noise": rs_noise, "top_noise": top_noise, "d_noise": jnp.sum(e_noise**2, -1),
            "e": predictors.normalized_error(n, o, std),
            "e_pred": predictors.normalized_error(n, o_star, std),
            "e_noise": predictors.normalized_error(o_star, o, std),
            "je": jnp.linalg.norm(je, axis=-1),
            "je_pred": jnp.linalg.norm(je_pred, axis=-1),
            "je_noise": jnp.linalg.norm(jnp.einsum("kao,mko->mka", jac, e_noise), axis=-1),
            "cos_fix": cosine(act(a_ref + jnp.clip(je, -1.0, 1.0)) - act(a_ref), need - act(a_ref)),
            "cos_spur": cosine(act(plan_k + jnp.clip(je_pred, -1.0, 1.0)) - act(plan_k), need - act(plan_k)),
            "clip": jnp.mean(jnp.abs(je) > 1.0, -1),
        }
        return jax.tree.map(lambda x: jnp.broadcast_to(x, valid.shape), out), a_ref

    stats, a_ref = jax.vmap(per_predictor)(nom)
    return stats | {"valid": valid, "a_ref": a_ref, "plan": plan}
