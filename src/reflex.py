"""JIT Reflex: linearize a frozen chunking flow policy along its predicted trajectory.

Between policy calls the robot acts with  a = nom + clip(gain @ (obs - ref)),  where ref_j is the predicted
observation at chunk index j, nom_j = pi(roll(z, -j), ref_j)[0] (or the chunk's action) and
gain_j = d pi(roll(z, -j), o)[0] / d o at o = ref_j. The noise is rolled so that row 0 is z[j]: see shifted_noise.
See docs/superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md (section 4).
"""

import jax
import jax.numpy as jnp


def first_action_and_jacobian(policy, noise, obs, num_steps):
    """noise [H, A], obs [O] -> (a0 [A], jac [A, O]): first action of pi(noise, obs) and its obs-Jacobian.

    Reverse mode: A (= 6 in Kinetix) VJPs instead of O (hundreds) JVPs.
    """

    def first_action(o):
        return policy.action_from_noise(noise[None], o[None], num_steps)[0, 0]

    a0, vjp = jax.vjp(first_action, obs)
    (jac,) = jax.vmap(vjp)(jnp.eye(a0.shape[0], dtype=a0.dtype))
    return a0, jac


def nominal_obs(step_fn, state, actions):
    """Noise-free rollout. step_fn(state, action) -> (obs, state); actions [T, ...] -> obs after each action [T, ...]."""

    def body(s, a):
        o, s = step_fn(s, a)
        return s, o

    return jax.lax.scan(body, state, actions)[1]


def shifted_noise(noise):
    """noise [H, A] -> [H, H, A]; entry j = roll(noise, -j), so row 0 is noise[j], the noise of chunk index j.

    A fresh call's first action comes from noise row 0, so re-asking pi about chunk index j must start from row j:
    with the unshifted noise, pi(z, o^_j)[0] is effectively a different sample than chunk[j].
    """
    return jax.vmap(lambda j: jnp.roll(noise, -j, axis=0))(jnp.arange(noise.shape[0]))


def package(policy, noise, ref, chunk, num_steps, requery: bool, feedback: bool, batch_size: int = 16, used=None):
    """Reflex package for one policy call, in chunk frame.

    noise [B, H, A] (the call's sampling noise z), ref [B, H, O] (predicted obs per chunk index), chunk [B, H, A].
    Returns nom [B, H, A] (nom_j = pi(roll(z, -j), ref_j)[0] if requery else chunk) and gain [B, H, A, O]
    (gain_j = d pi(roll(z, -j), o)[0] / d o at ref_j; None without feedback).
    used = (lo, hi): compute only chunk indices lo..hi-1, the ones eval executes (d..d+s-1, spec B1 3.4); the other
    entries are zeros and are never read. Envs are processed batch_size at a time: Jacobian activations are large.
    """
    if not (requery or feedback):
        return chunk, None
    H = noise.shape[1]
    lo, hi = used or (0, H)

    def per_env(x):
        n, r = shifted_noise(x[0])[lo:hi], x[1][lo:hi]
        if feedback:
            return jax.vmap(lambda nj, o: first_action_and_jacobian(policy, nj, o, num_steps))(n, r)
        return policy.action_from_noise(n, r, num_steps)[:, 0], None

    def full(x):  # back to chunk frame [B, H, ...]
        return jnp.zeros((x.shape[0], H, *x.shape[2:]), x.dtype).at[:, lo:hi].set(x)

    a0, jac = jax.lax.map(per_env, (noise, ref), batch_size=min(batch_size, noise.shape[0]))
    return (full(a0) if requery else chunk), (None if jac is None else full(jac))


def correct(nom, gain, ref, obs, max_correction):
    """a = nom + clip(gain @ (obs - ref), +-max_correction); broadcasts over leading dims."""
    delta = jnp.einsum("...ao,...o->...a", gain, obs - ref)
    return nom + jnp.clip(delta, -max_correction, max_correction)


def forward_equivalents(
    method: str, num_steps: int = 5, chunk_size: int = 8, action_dim: int = 6, positions: int | None = None
) -> int:
    """Network evaluations per policy call, one VJP counted as 2 (spec section 4, budget).

    positions: package entries computed per call. None = all H (phase A); eval computes only the s executed ones
    (spec B1 3.4). Not counted: the reflex's per-step correction (action_dim x obs_dim MACs, negligible) and the
    predictor (a simulator fork or a small world model).
    """
    S, H, A = num_steps, chunk_size, action_dim
    P = H if positions is None else positions
    return {
        "naive": S,
        "reflex_off": S,
        "realtime": 3 * S,  # one guidance VJP per flow step
        "hard_masking": 3 * S,
        "bid": 16 * S,  # default n_samples=16, no weak policy
        "pred": S + P * S,  # chunk + pi at P predicted states
        "reflex": S + P * (S + 2 * A * S),  # + A VJPs through the whole flow at each predicted state
        "reflex_chunk": S + P * (S + 2 * A * S),
        "rtc_reflex": 3 * S + P * (S + 2 * A * S),  # E2b: RTC chunk + the same package
        "rtc_reflex_off": 3 * S,
    }[method]
