import jax
import jax.numpy as jnp
import numpy as np

import predictors
import probe


def _level_state(level="worlds/l/grasp_easy.json"):
    env, env_params, levels, _, action_dim = probe.setup([level])
    _, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    return env._env, env_params, state.env_state, action_dim  # raw Kinetix env and raw EnvState, as eval uses


def test_phys_factors_are_one_plus_minus_error_and_paired():
    _, _, raw, _ = _level_state()
    f2 = predictors.phys_factors(jax.random.key(1), raw, 0.2)
    f3 = predictors.phys_factors(jax.random.key(1), raw, 0.3)
    for a, b in zip(jax.tree.leaves(f2), jax.tree.leaves(f3)):
        np.testing.assert_allclose(np.abs(np.asarray(a) - 1), 0.2, atol=1e-6)
        np.testing.assert_array_equal(np.sign(np.asarray(a) - 1), np.sign(np.asarray(b) - 1))  # same key: same signs


def test_phys_zero_error_is_the_oracle():
    env, params, raw, A = _level_state()
    f = predictors.phys_factors(jax.random.key(1), raw, 0.0)
    a = jnp.full(A, 0.5)
    o1, s1 = predictors.phys_step(env, jax.random.key(2), raw, a, params, f)
    o0, s0 = env.step_env(jax.random.key(2), raw, a, params)[:2]
    np.testing.assert_array_equal(o1, o0)
    jax.tree.map(np.testing.assert_array_equal, s1, s0)


def test_phys_error_changes_the_motion_but_not_the_parameter_features():
    env, params, raw, A = _level_state()
    f = predictors.phys_factors(jax.random.key(1), raw, 0.3)
    a = jnp.full(A, 0.5)
    s_true = s_wrong = raw
    for _ in range(10):
        o_true, s_true = env.step_env(jax.random.key(2), s_true, a, params)[:2]
        o_wrong, s_wrong = predictors.phys_step(env, jax.random.key(2), s_wrong, a, params, f)
    for x, y in zip(jax.tree.leaves(predictors._params(s_wrong)), jax.tree.leaves(predictors._params(raw))):
        np.testing.assert_array_equal(x, y)  # the true parameters are back in the predicted state
    leaked = predictors._with(s_wrong, *jax.tree.map(jnp.multiply, predictors._params(s_wrong), f))
    assert not np.allclose(env.get_obs(leaked), o_wrong)  # the observation shows the parameters; ours shows the true ones
    assert not np.allclose(o_wrong, o_true)  # but the bodies moved differently


def test_world_model_learns_linear_dynamics_and_copies_static_dims():
    N, O, A = 2048, 4, 2
    obs = jax.random.normal(jax.random.key(0), (N, O)).at[:, 3].set(5.0)  # dim 3 never moves
    act = jax.random.normal(jax.random.key(1), (N, A))
    B = jnp.array([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [0.0, 0.0]])
    nxt = obs + 0.1 * act @ B.T
    wm = predictors.fit(obs, act, nxt, jax.random.key(2), hidden=32, steps=3000, batch=256, lr=3e-3)
    np.testing.assert_array_equal(np.asarray(wm["mask"]), [1, 1, 1, 0])
    assert float(predictors.one_step_nmse(wm, obs, act, nxt)) < 0.05
    roll = predictors.wm_rollout(wm, obs[0], act[:5])
    assert roll.shape == (5, O)
    np.testing.assert_array_equal(np.asarray(roll[:, 3]), 5.0)  # static dims are copied exactly


def test_normalized_error_counts_only_moving_dims():
    truth = jnp.zeros((2, 3))
    std = jnp.array([2.0, 1.0, 0.0])  # dim 2 never moves
    pred = truth.at[:, 0].set(2.0).at[:, 2].set(100.0)
    np.testing.assert_allclose(predictors.normalized_error(pred, truth, std), [1.0, 1.0])
