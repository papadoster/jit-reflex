import jax.numpy as jnp
import numpy as np

import diag


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
