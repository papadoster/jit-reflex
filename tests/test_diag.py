import jax
import jax.numpy as jnp
import numpy as np

import diag
import predictors
import probe
from test_reflex import small_policy


def test_row_space_share_and_cosine():
    jac = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # row space = axes 0 and 1, top direction = axis 0
    row, top = diag.row_space_share(jac, jnp.array([3.0, 4.0, 12.0]))
    np.testing.assert_allclose([row, top], [25.0, 9.0], rtol=1e-5)
    row, _ = diag.row_space_share(jnp.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]), jnp.array([1.0, -1.0, 5.0]))
    np.testing.assert_allclose(row, 0.0, atol=1e-5)  # rank 1: (1, -1, 5) is orthogonal to the row space
    rows, _ = diag.row_space_share(jnp.stack([jac, 2 * jac]), jnp.ones((4, 2, 3)))  # [K, A, O] with [M, K, O]
    assert rows.shape == (4, 2)
    np.testing.assert_allclose(diag.cosine(jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0])), 2**-0.5, rtol=1e-6)
    assert float(diag.cosine(jnp.zeros(2), jnp.ones(2))) == 0.0


def test_probe_state_oracle_invariants():
    env, params, levels, O, A = probe.setup(["worlds/l/grasp_easy.json"])
    obs, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), params)
    policy = small_policy(O, A)
    wm = {  # mask 0: this "model" holds the observation
        "layers": predictors.init(jax.random.key(1), O, A, 8), "x_mean": jnp.zeros(O), "x_std": jnp.ones(O),
        "d_mean": jnp.zeros(O), "d_std": jnp.ones(O), "mask": jnp.zeros(O),
    }
    out = diag.probe_state(
        policy, env._env, params, wm, (0.2,), state.env_state, obs, jax.random.key(2), jnp.ones(O),
        num_draws=2, num_chunks=2, num_steps=2,
    )
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
