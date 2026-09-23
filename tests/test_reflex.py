import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import eval_flow
import model as _model
import reflex


def test_horizons_for():
    assert eval_flow.horizons_for(0, 8, (), False) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert eval_flow.horizons_for(2, 8, (), False) == [2, 3, 4, 5, 6]
    assert eval_flow.horizons_for(2, 8, (), True) == [2, 6]
    assert eval_flow.horizons_for(4, 8, (), True) == [4]
    assert eval_flow.horizons_for(3, 8, (1, 3, 5, 7), False) == [3, 5]


def small_policy(obs_dim=5, action_dim=2, seed=0):
    """Tiny FlowPolicy with every weight randomized (zero-init AdaLN would make the blocks identity)."""
    cfg = _model.ModelConfig(channel_dim=16, channel_hidden_dim=32, token_hidden_dim=8, num_layers=2)
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=cfg, rngs=nnx.Rngs(seed))
    graphdef, state = nnx.split(policy)
    leaves, treedef = jax.tree.flatten(state.to_pure_dict())
    keys = jax.random.split(jax.random.key(seed + 1), len(leaves))
    state.replace_by_pure_dict(
        jax.tree.unflatten(treedef, [x + 0.3 * jax.random.normal(k, x.shape) for x, k in zip(leaves, keys)])
    )
    return nnx.merge(graphdef, state)


def test_action_from_noise_matches_action():
    policy = small_policy()
    key = jax.random.key(0)
    obs = jax.random.normal(jax.random.key(1), (3, 5))
    noise = jax.random.normal(key, (3, policy.action_chunk_size, policy.action_dim))
    np.testing.assert_allclose(policy.action(key, obs, 5), policy.action_from_noise(noise, obs, 5), atol=1e-6)


def test_first_action_and_jacobian_matches_finite_differences():
    policy = small_policy()
    noise = jax.random.normal(jax.random.key(2), (policy.action_chunk_size, policy.action_dim))
    obs = jax.random.normal(jax.random.key(3), (5,))
    a0, jac = reflex.first_action_and_jacobian(policy, noise, obs, 5)
    assert a0.shape == (2,) and jac.shape == (2, 5)

    def f(o):
        return policy.action_from_noise(noise[None], o[None], 5)[0, 0]

    np.testing.assert_allclose(a0, f(obs), atol=1e-6)
    v, eps = jax.random.normal(jax.random.key(4), (5,)), 1e-2
    fd = (f(obs + eps * v) - f(obs - eps * v)) / (2 * eps)
    np.testing.assert_allclose(jac @ v, fd, rtol=1e-2, atol=1e-3)


def test_nominal_obs_rolls_forward():
    def step(state, action):
        return state + action, state + action  # (obs, state)

    obs = reflex.nominal_obs(step, jnp.zeros(2), jnp.ones((3, 2)))
    np.testing.assert_allclose(obs, [[1, 1], [2, 2], [3, 3]])


def test_package_flags():
    policy = small_policy()
    B, H, A, O = 2, policy.action_chunk_size, policy.action_dim, 5
    noise = jax.random.normal(jax.random.key(5), (B, H, A))
    ref = jax.random.normal(jax.random.key(6), (B, H, O))
    chunk = jnp.ones((B, H, A))
    a0, jac = reflex.first_action_and_jacobian(policy, noise[1], ref[1, 3], 5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True)
    assert nom.shape == (B, H, A) and gain.shape == (B, H, A, O)
    np.testing.assert_allclose(nom[1, 3], a0, atol=1e-5)
    np.testing.assert_allclose(gain[1, 3], jac, atol=1e-5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=False)
    assert gain is None
    np.testing.assert_allclose(nom[1, 3], a0, atol=1e-5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=False, feedback=False)
    assert gain is None
    np.testing.assert_allclose(nom, chunk)


def test_correct_is_identity_at_zero_deviation_and_clips():
    nom = jnp.array([0.1, -0.2])
    gain = jnp.array([[10.0, 0.0], [0.0, 1.0]])
    ref = jnp.array([1.0, 2.0])
    np.testing.assert_allclose(reflex.correct(nom, gain, ref, ref, 0.5), nom)
    out = reflex.correct(nom, gain, ref, ref + jnp.array([1.0, 0.1]), 0.5)
    np.testing.assert_allclose(out, [0.1 + 0.5, -0.2 + 0.1], atol=1e-6)


def test_forward_equivalents():
    assert reflex.forward_equivalents("naive") == 5
    assert reflex.forward_equivalents("realtime") == 15
    assert reflex.forward_equivalents("pred") == 45
    assert reflex.forward_equivalents("reflex") == reflex.forward_equivalents("reflex_chunk") == 525
