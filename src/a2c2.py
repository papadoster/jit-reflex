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

import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp

import predictors


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
