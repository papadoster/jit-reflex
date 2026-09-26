"""B2 offline: cheaper reflex packages, scored against a fresh call as in the B1 diagnostic.

For a sampled state: one policy chunk, real rollouts with action noise, predictors oracle and learned. At chunk
index k = 1..7 each package's nom + clip(J·(o − ô)) is compared with a fresh call pi(o) as executed actions (E1b).
The candidates cut the depth of the reflex call: T_k (J through the last k flow steps), W_k (warm start: the last k
steps from the chunk's own flow state), M_m (an m-step flow). See docs/superpowers/specs/2026-09-26-b2-offline-design.md.
"""

import hashlib
import json
import os
import pathlib
import pickle
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import tyro

import diag
import predictors
import probe
import reflex
import train_expert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S, A, P = 5, 6, 5  # flow steps, action dims, package positions (s = 5): the prices of spec section 3
CANDIDATES = ("T1", "T2", "T3", "W1", "W2", "W3", "M1", "M2", "M3")
ROWS = ("pred", "reflex", "chunk_j", "shared", *CANDIDATES)  # pred = no J; chunk_j = chunk + exact J (rtc_reflex-like)
SUMS = (*ROWS, "nom_chunk_j", *(f"nom_{c}" for c in CANDIDATES))  # squared executed-action errors, summed per level
NOM = {"reflex": "pred", "shared": "pred", "chunk_j": "nom_chunk_j"} | {c: f"nom_{c}" for c in CANDIDATES}
SHARED_K = 5  # the one-J row: J at chunk index 5, the middle of the executed 3..7 at d = 3, s = 5
E_MAX = 2.3  # spec section 6: the rule pools pairs with |e| <= 2.3
MIN_GAIN = 0.1  # levels where the exact reflex improves res over pred by < 10% leave the median of R
PASS_R, STRICT_R, MIN_LEVELS = 0.8, 0.9, 10
OUT = "results/b2/offline"
DIAG_BINS = "results/b1/diag/bins.csv"  # the |e| bin edges: deciles of the diagnostic's pairs (predictor "all")


def depth(row: str) -> int:
    """Sequential flow steps after the chunk and the predictor, forward = backward = 1 (spec section 6)."""
    if row == "pred":
        return S
    if row in ("reflex", "chunk_j", "shared"):
        return 2 * S
    n = int(row[1:])
    return S + n if row[0] == "T" else 2 * n


def fe(row: str) -> int:
    """Forward equivalents per call, a VJP counted as 2 (reflex.forward_equivalents; spec section 3)."""
    if row == "pred":
        return S + P * S
    if row in ("reflex", "chunk_j"):
        return S + P * (S + 2 * A * S)
    if row == "shared":
        return S + P * S + 2 * A * S
    n = int(row[1:])
    return S + P * (S + 2 * A * n) if row[0] == "T" else S + P * (n + 2 * A * n)


def run_flow(policy, x, obs, t, n: int, dt: float):
    """n Euler steps of pi's flow, as in model.action_from_noise, from x [H, A] at time t, conditioned on obs [O].

    Returns ((x, t) after the steps, the states before each step [n, H, A]).
    """

    def step(c, _):
        x, t = c
        return (x + dt * policy(obs[None], x[None], t)[0], t + dt), x

    return jax.lax.scan(step, (x, jnp.asarray(t, x.dtype)), None, length=n)


def flow_states(policy, noise, obs, num_steps: int):
    """The chunk's flow states x_0 = noise, ..., x_S = the chunk: [S + 1, H, A]."""
    (x, _), xs = run_flow(policy, noise, obs, 0.0, num_steps, 1 / num_steps)
    return jnp.concatenate([xs, x[None]])


def first_action_fn(policy, name: str, z, warm, num_steps: int):
    """o [O] -> the first action of package `name` ("T2", "W1", "M3", ...) at one chunk index (spec section 3).

    z [H, A] is the call's noise rolled to this index (reflex.shifted_noise); warm [S + 1, H, A] is the chunk's flow
    states rolled the same way (W only). T_n, W_n and M_n with n = num_steps are the exact reflex.
    """
    kind, n = name[0], int(name[1:])
    if kind == "T":  # the exact forward pass; J only through the last n steps

        def f(o):
            (x, t), _ = run_flow(policy, z, o, 0.0, num_steps - n, 1 / num_steps)
            (x, _), _ = run_flow(policy, jax.lax.stop_gradient(x), o, t, n, 1 / num_steps)
            return x[0]

    elif kind == "W":  # the last n steps from the chunk's state at step S - n (queried at o_0, not at ô)

        def f(o):
            (x, _), _ = run_flow(policy, warm[num_steps - n], o, (num_steps - n) / num_steps, n, 1 / num_steps)
            return x[0]

    else:
        assert kind == "M", name

        def f(o):  # an n-step flow from the same noise
            (x, _), _ = run_flow(policy, z, o, 0.0, n, 1 / n)
            return x[0]

    return f


def action_and_jacobian(f, o):
    """f: o [O] -> a [A]. Returns (a, da/do [A, O]) from A reverse-mode VJPs, as reflex.first_action_and_jacobian."""
    a0, vjp = jax.vjp(f, o)
    (jac,) = jax.vmap(vjp)(jnp.eye(a0.shape[0], dtype=a0.dtype))
    return a0, jac
