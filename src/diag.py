"""B1 diagnostic (exploratory): which prediction error J sees, and at which deviation and horizon the tangent breaks.

For a sampled state: a plan of a few policy chunks, each queried where the noise-free truth arrives; real rollouts add
action noise; every predictor rolls the same plan. At step k the reflex's pi(ô_k) + J·(o_k − ô_k) is compared with a
fresh call pi(o_k), as in E1b. See docs/superpowers/specs/2026-09-26-b1-diagnostic-design.md.
"""

import jax.numpy as jnp


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
