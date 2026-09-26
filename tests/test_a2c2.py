import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import a2c2
import reflex
from test_reflex import small_policy


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


def test_fit_learns_a_linear_residual():
    key = jax.random.key(0)
    obs = jax.random.normal(key, (2048, 5))
    data = {"obs": obs, "base": jnp.zeros((2048, 3)), "t": jnp.zeros(2048, jnp.int32), "target": obs[:, :3] * 0.5}

    def batch(d, k, idx):
        return d["obs"][idx], d["base"][idx], a2c2.time_feature(d["t"][idx], 8), d["target"][idx]

    head, losses = a2c2.fit(jax.random.key(1), 5, 3, batch, data, 2048, num_epochs=30, batch_size=32)
    assert losses[-1] < 0.3 * losses[0]


def test_expert_batch_samples_before_the_first_done():
    policy = small_policy(5, 3)
    N = 64
    data = {
        "obs": jax.random.normal(jax.random.key(2), (N, 5)),
        "action": jnp.ones((N, 3)),
        "done": jnp.zeros(N, bool).at[jnp.arange(2, N, 8)].set(True),  # done at window offset 2
    }
    fn = a2c2.expert_batch(policy)
    obs, base, tf, target = fn(data, jax.random.key(3), jnp.arange(0, N - 8, 8))
    t = jnp.round(jnp.arctan2(tf[:, 1], tf[:, 0]) / (2 * jnp.pi / 8)).astype(int) % 8
    assert (t < 2).all()  # t in [0, done_idx)
    idx = jnp.arange(0, N - 8, 8) + t
    assert (obs == data["obs"][idx]).all() and (target == 1).all()


def test_expert_data_layout_and_truncation(tmp_path):
    s, e, O, A = 100, 3, 4, 2
    np.savez(
        tmp_path / "x.npz", obs=np.arange(s * e * O, dtype=np.float32).reshape(s, e, O),
        action=np.zeros((s, e, A), np.float32), done=np.zeros((s, e), bool),
    )
    data, n = a2c2.expert_data(tmp_path / "x.npz", H=8, batch_size=16)
    assert n == ((s * e - 7) // 16) * 16
    assert data["obs"].shape == (n + 7, O)
    np.testing.assert_array_equal(data["obs"][1], np.arange(s * e * O).reshape(s, e, O)[1, 0])  # env 0, step 1


def test_fresh_targets_are_fresh_calls_with_the_shifted_noise():
    policy = small_policy(5, 3)
    obs = jax.random.normal(jax.random.key(4), (2, 8, 5))
    z = jax.random.normal(jax.random.key(5), (2, 8, 3))
    t = a2c2.fresh_targets(policy, obs, z, 5, batch_size=1)
    for w in range(2):
        for k in range(8):
            want = policy.action_from_noise(reflex.shifted_noise(z[w])[k][None], obs[w, k][None], 5)[0, 0]
            np.testing.assert_allclose(t[w, k], want, atol=1e-6)


def test_distill_items_drop_positions_after_done():
    obs = jnp.zeros((2, 8, 5))
    chunk = jnp.arange(2 * 8 * 3, dtype=jnp.float32).reshape(2, 8, 3)
    done = jnp.zeros((2, 8), bool).at[0, 3].set(True)
    items = a2c2.distill_items(obs, chunk, done, chunk + 1)
    assert len(items["k"]) == 3 + 8  # window 0 keeps k = 0..2, window 1 all 8
    assert (items["target"] == items["base"] + 1).all()
