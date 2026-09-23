"""E1 kill-test: is a frozen flow policy locally linear along its predicted trajectory?

For sampled states x_t: chunk A = pi(z, o_t); predicted obs o^_{t+k} (noise-free fork of the simulator);
noisy obs o_{t+k} (action noise sigma); oracle a* = pi(z, o_{t+k})[0] = what a fresh call would do now.
See docs/superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md (section 5, E1).
"""

import math

import jax.numpy as jnp
import pandas as pd

ERRORS = ("chunk", "pred", "lin", "chunk_lin", "floor", "dev")


def errors(chunk, a_ref, jac, nom_obs, obs, a_star):
    """Squared action errors against the oracle a* (spec E1).

    chunk, a_ref [K, A]; jac [K, A, O]; nom_obs [K, O]; obs [..., K, O]; a_star [..., K, A] -> dict of [..., K].
    """
    delta = obs - nom_obs
    lin = jnp.einsum("kao,...ko->...ka", jac, delta)

    def sq(x):
        return jnp.sum(jnp.square(x), axis=-1)

    return {
        "chunk": sq(chunk - a_star),
        "pred": sq(a_ref - a_star),
        "lin": sq(a_ref + lin - a_star),
        "chunk_lin": sq(chunk + lin - a_star),
        "floor": jnp.broadcast_to(sq(chunk - a_ref), a_star.shape[:-1]),
        "dev": jnp.sqrt(sq(delta)),
    }


def verdict(summary: pd.DataFrame) -> dict:
    """Pre-registered E1 decision rule (spec section 5). summary needs columns level, sigma, k, rho, rel."""
    near = summary[summary["k"].between(1, 4)]
    rel = float(near[near["sigma"] == 0.1]["rel"].mean())
    sigma = 0.1 if rel >= 0.1 else 0.2  # relevance check: deviations must actually change the policy's decisions
    per_level = near[near["sigma"] == sigma].groupby("level")["rho"].mean()
    r, n_ok, n = float(per_level.median()), int((per_level >= 0.3).sum()), len(per_level)
    if r >= 0.5 and n_ok >= math.ceil(2 * n / 3):
        v = "GO"
    elif r < 0.2:
        v = "KILL"
    else:
        v = "GRAY"
    return {"verdict": v, "sigma": sigma, "R": r, "levels_ok": n_ok, "levels": n, "rel": rel}
