import jax
import jax.numpy as jnp
import numpy as np

import diag
import predictors
import probe
import train_expert
from test_reflex import small_policy


def test_row_space_share_and_cosine():
    jac = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # row space = axes 0 and 1, top direction = axis 0
    row, top = diag.row_space_share(jac, jnp.array([3.0, 4.0, 12.0]))
    np.testing.assert_allclose([row, top], [25.0, 9.0], rtol=1e-5)
    row, _ = diag.row_space_share(jnp.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]), jnp.array([1.0, -1.0, 5.0]))
    np.testing.assert_allclose(row, 0.0, atol=1e-5)  # rank 1: (1, -1, 5) is orthogonal to the row space
    rows, _ = diag.row_space_share(jnp.stack([jac, 2 * jac]), jnp.ones((4, 2, 3)))  # [K, A, O] with [M, K, O]
    assert rows.shape == (4, 2)
    np.testing.assert_allclose(rows, 2.0, rtol=1e-5)  # e = ones: the row space {axis 0, axis 1} holds 1 + 1
    np.testing.assert_allclose(diag.cosine(jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0])), 2**-0.5, rtol=1e-6)
    assert float(diag.cosine(jnp.zeros(2), jnp.ones(2))) == 0.0


def _probe_state(seed: int = 2):
    """probe_state at grasp_easy's start: small policy, phys 0.2, a holding model -> (env, params, state, out)."""
    env, params, levels, O, A = probe.setup(["worlds/l/grasp_easy.json"])
    obs, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), params)
    policy = small_policy(O, A)
    wm = {  # mask 0: this "model" holds the observation
        "layers": predictors.init(jax.random.key(1), O, A, 8), "x_mean": jnp.zeros(O), "x_std": jnp.ones(O),
        "d_mean": jnp.zeros(O), "d_std": jnp.ones(O), "mask": jnp.zeros(O),
    }
    out = diag.probe_state(
        policy, env._env, params, wm, (0.2,), state.env_state, obs, jax.random.key(seed), jnp.ones(O),
        num_draws=2, num_chunks=2, num_steps=2,
    )
    return env, params, state, out


def test_probe_state_oracle_invariants():
    _, _, _, out = _probe_state()
    A = out["plan"].shape[-1]
    P, M, K = 3, 2, 15  # oracle, phys0.2, learned; draws; k = 1..2*8-1
    assert out["e"].shape == (P, M, K) and out["valid"].shape == (M, K) and out["a_ref"].shape == (P, K, A)
    assert set(diag.SUMS + diag.MEANS) <= set(out)
    np.testing.assert_array_equal(out["e_pred"][0], 0)  # the oracle's prediction is the noise-free truth
    np.testing.assert_allclose(out["e"][0], out["e_noise"][0], rtol=1e-6)
    # chunk 1 was queried at o*_8 with zs[1]: at k = 8 the oracle's re-query must be the plan's own action.
    # A one-step shift of the noise index breaks this equality.
    np.testing.assert_allclose(out["a_ref"][0, 7], out["plan"][8], atol=1e-5)
    assert float(out["e_pred"][1].max()) > 0 and float(out["e_pred"][2].max()) > 0  # phys and the holding model err
    for name in diag.SUMS + diag.MEANS:
        assert np.isfinite(np.asarray(out[name])).all(), name


def test_probe_state_without_noise(monkeypatch):
    monkeypatch.setattr(train_expert, "ACTION_NOISE_STD", 0.0)
    env, params, state, out = _probe_state(seed=3)  # seed 3: the noise-free plan ends the episode at action 11
    np.testing.assert_allclose(out["e_noise"], 0, atol=1e-4)  # vmapped real rollout and chunked truth compile apart
    np.testing.assert_allclose(out["pred"][0], 0, atol=1e-8)
    np.testing.assert_allclose(out["chunk"][0][:, 7], 0, atol=1e-8)  # k = 8: chunk 1 was queried here
    s, d = state.env_state, None  # d: the first action (1-indexed) whose done fires
    for i, a in enumerate(out["plan"]):
        _, s, _, done, _ = env._env.step_env(jax.random.key(0), s, a, params)
        if bool(done):
            d = i + 1
            break
    K = out["valid"].shape[1]
    assert d is not None and 1 < d <= K  # the end falls inside the horizon, so a shifted valid shows
    expected = np.arange(1, K + 1) < d
    np.testing.assert_array_equal(out["valid"], np.broadcast_to(expected, out["valid"].shape))


def test_chain_carries_the_end_of_episode():
    policy = small_policy(3, 2)
    zs = jax.random.normal(jax.random.key(0), (2, 8, 2))

    def step(s, a):  # obs = [s+1]*3; done fires only at step 3
        return jnp.full(3, s + 1.0), s + 1, s + 1 == 3

    plan, truth, ended = diag.chain(policy, step, jnp.int32(0), jnp.zeros(3), zs, 2)
    assert ended.tolist() == [False, False] + [True] * 14  # once ended, ended across the chunk boundary too
    np.testing.assert_allclose(plan[8:], policy.action_from_noise(zs[1][None], truth[7][None], 2)[0], atol=1e-6)
