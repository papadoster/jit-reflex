import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import a2c2


def test_zero_head_returns_the_base_action():
    head = a2c2.Head(5, 3, rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(head)
    zero = nnx.merge(graphdef, jax.tree.map(jnp.zeros_like, state))
    base = jnp.arange(6.0).reshape(2, 3)
    out = zero.apply_residual(jnp.ones((2, 5)), base, a2c2.time_feature(jnp.array([0, 3]), 8))
    assert (out == base).all()


def test_time_feature_is_cos_sin_of_the_chunk_index():
    tf = a2c2.time_feature(jnp.array([0, 2]), 8)
    np.testing.assert_allclose(tf, [[1.0, 0.0], [0.0, 1.0]], atol=1e-6)


def test_head_state_round_trips():
    head = a2c2.Head(5, 3, rngs=nnx.Rngs(1))
    sd = nnx.state(head).to_pure_dict()
    back = a2c2.make_head(sd, 5, 3)
    x = (jnp.ones((1, 5)), jnp.ones((1, 3)), a2c2.time_feature(jnp.array([1]), 8))
    assert (back.apply_residual(*x) == head.apply_residual(*x)).all()
