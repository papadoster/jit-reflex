import jax
import jax.numpy as jnp
import numpy as np

import b2
import predictors
import probe
import reflex
import train_expert
from test_reflex import small_policy

S = b2.S


def _flow_setup():
    """Small policy, the call's noise z, the call-time obs o0 and a later obs o."""
    policy = small_policy(5, 3)
    z = jax.random.normal(jax.random.key(0), (policy.action_chunk_size, 3))
    o0 = jnp.linspace(-1.0, 1.0, 5)
    return policy, z, o0, o0 + 0.3 * jnp.linspace(1.0, -1.0, 5)


def test_prices_match_the_spec_and_forward_equivalents():
    assert [b2.depth(r) for r in ("pred", "reflex", "chunk_j", "shared", "T1", "T3", "W1", "W3", "M2")] == [
        5, 10, 10, 10, 6, 8, 2, 6, 4
    ]
    assert [b2.fe(r) for r in ("T1", "T2", "T3", "W1", "W2", "W3", "M1", "M3", "shared")] == [
        90, 150, 210, 70, 135, 200, 70, 200, 90
    ]
    assert b2.fe("reflex") == reflex.forward_equivalents("reflex", positions=b2.P) == 330
    assert b2.fe("pred") == reflex.forward_equivalents("pred", positions=b2.P) == 30


def test_flow_states_end_at_the_chunk():
    policy, z, o0, _ = _flow_setup()
    xs = b2.flow_states(policy, z, o0, S)
    assert xs.shape == (S + 1, *z.shape)
    np.testing.assert_array_equal(xs[0], z)
    np.testing.assert_allclose(xs[-1], policy.action_from_noise(z[None], o0[None], S)[0], atol=1e-6)


def test_full_depth_packages_are_the_exact_reflex():
    policy, z, o0, o = _flow_setup()
    j = 3
    zj = reflex.shifted_noise(z)[j]
    warm = jnp.roll(b2.flow_states(policy, z, o0, S), -j, axis=1)  # the chunk's flow states rolled like zj
    a, jac = reflex.first_action_and_jacobian(policy, zj, o, S)
    for name in (f"T{S}", f"W{S}", f"M{S}"):
        a_c, jac_c = b2.action_and_jacobian(b2.first_action_fn(policy, name, zj, warm, S), o)
        np.testing.assert_allclose(a_c, a, atol=1e-5, err_msg=name)
        np.testing.assert_allclose(jac_c, jac, atol=1e-5, err_msg=name)
    for m in (1, 2, 3):  # M_m is the exact reflex of an m-step flow
        a_m, jac_m = reflex.first_action_and_jacobian(policy, zj, o, m)
        a_c, jac_c = b2.action_and_jacobian(b2.first_action_fn(policy, f"M{m}", zj, warm, S), o)
        np.testing.assert_allclose(a_c, a_m, atol=1e-6, err_msg=f"M{m}")
        np.testing.assert_allclose(jac_c, jac_m, atol=1e-6, err_msg=f"M{m}")


def test_truncated_jacobian_is_the_last_steps_chain():
    policy, z, _, o = _flow_setup()
    a, jac = reflex.first_action_and_jacobian(policy, z, o, S)
    for k in (1, 2):
        a_k, jac_k = b2.action_and_jacobian(b2.first_action_fn(policy, f"T{k}", z, None, S), o)
        np.testing.assert_allclose(a_k, a, atol=1e-6)  # T_k's forward pass is the exact one
        (x, t), _ = b2.run_flow(policy, z, o, 0.0, S - k, 1 / S)  # the state entering the last k steps, held fixed

        def last(obs, x=x, t=t, k=k):
            return b2.run_flow(policy, x, obs, t, k, 1 / S)[0][0][0]

        np.testing.assert_allclose(jac_k, jax.jacrev(last)(o), atol=1e-6)
        assert not np.allclose(jac_k, jac, atol=1e-4)  # the truncation must actually change J


def test_warm_start_starts_from_the_chunks_state():
    policy, z, o0, o = _flow_setup()
    xs = b2.flow_states(policy, z, o0, S)
    for n in (1, 2, 3):  # at j = 0 and o = o0 the last n steps rebuild the chunk
        a_w, _ = b2.action_and_jacobian(b2.first_action_fn(policy, f"W{n}", z, xs, S), o0)
        np.testing.assert_allclose(a_w, xs[-1][0], atol=1e-6, err_msg=f"W{n}")
    a_w, _ = b2.action_and_jacobian(b2.first_action_fn(policy, "W1", z, xs, S), o)
    assert not np.allclose(a_w, xs[-1][0], atol=1e-4)  # at another obs it moves


def _probe(seed: int = 2):
    """probe_state at grasp_easy's start with the small policy and a world model that holds the observation."""
    env, params, levels, O, A_ = probe.setup(["worlds/l/grasp_easy.json"])
    obs, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), params)
    wm = {  # mask 0: this "model" holds the observation
        "layers": predictors.init(jax.random.key(1), O, A_, 8), "x_mean": jnp.zeros(O), "x_std": jnp.ones(O),
        "d_mean": jnp.zeros(O), "d_std": jnp.ones(O), "mask": jnp.zeros(O),
    }
    return b2.probe_state(
        small_policy(O, A_), env._env, params, wm, state.env_state, obs, jax.random.key(seed), jnp.ones(O),
        num_draws=2, num_steps=S,
    )


def test_probe_state_shapes_and_exact_nominals():
    out = _probe()
    P_, M, K = 2, 2, 7  # oracle, learned; draws; k = 1..H-1
    assert out["valid"].shape == (M, K)
    for name in (*b2.SUMS, "e"):
        assert out[name].shape == (P_, M, K), name
        assert np.isfinite(np.asarray(out[name])).all(), name
    for k in (1, 2, 3):  # T_k's forward pass is exact: its nominal error is pred's
        np.testing.assert_allclose(out[f"nom_T{k}"], out["pred"], rtol=1e-4, atol=1e-7)
    assert float(out["e"][1].max()) > 0  # the holding model errs


def test_probe_state_without_noise(monkeypatch):
    monkeypatch.setattr(train_expert, "ACTION_NOISE_STD", 0.0)
    out = _probe(seed=3)
    np.testing.assert_allclose(out["e"][0], 0, atol=1e-3)  # the oracle predicts the noise-free truth
    for name in ("pred", "reflex", "shared", "T1", "T2", "T3"):  # a* is the exact nominal, the correction ~0
        np.testing.assert_allclose(out[name][0], 0, atol=1e-6, err_msg=name)
    assert float(out["nom_W1"][0].max()) > 0 and float(out["nom_M1"][0].max()) > 0  # approximate nominals


def test_probe_state_shared_and_full_depth(monkeypatch):
    monkeypatch.setattr(b2, "CANDIDATES", ("T5", "W5", "M5"))
    out = _probe()
    i = b2.SHARED_K - 1  # at chunk index 5 the shared J is the exact J
    np.testing.assert_allclose(out["shared"][..., i], out["reflex"][..., i], rtol=1e-5, atol=1e-7)
    for c in b2.CANDIDATES:  # full depth = the exact reflex; W5 starts from warm[0] = roll(z, -k) = zk
        np.testing.assert_allclose(out[f"nom_{c}"], out["pred"], rtol=1e-4, atol=1e-6, err_msg=c)
        np.testing.assert_allclose(out[c], out["reflex"], rtol=1e-4, atol=1e-6, err_msg=c)
