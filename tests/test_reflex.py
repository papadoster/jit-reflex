import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import eval_flow
import model as _model
import probe
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
        return 10 * (state + action), state + action  # (obs, state); obs != state pins the unpack order

    obs = reflex.nominal_obs(step, jnp.zeros(2), jnp.ones((3, 2)))
    np.testing.assert_allclose(obs, [[10, 10], [20, 20], [30, 30]])


def test_package_flags():
    policy = small_policy()
    B, H, A, O = 2, policy.action_chunk_size, policy.action_dim, 5
    noise = jax.random.normal(jax.random.key(5), (B, H, A))
    ref = jax.random.normal(jax.random.key(6), (B, H, O))
    chunk = jnp.ones((B, H, A))
    a0, jac = reflex.first_action_and_jacobian(policy, jnp.roll(noise[1], -3, axis=0), ref[1, 3], 5)
    a0_first, jac_first = reflex.first_action_and_jacobian(policy, noise[1], ref[1, 0], 5)  # index 0: roll by 0

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True)
    assert nom.shape == (B, H, A) and gain.shape == (B, H, A, O)
    np.testing.assert_allclose(nom[1, 3], a0, atol=1e-5)
    np.testing.assert_allclose(gain[1, 3], jac, atol=1e-5)
    np.testing.assert_allclose(nom[1, 0], a0_first, atol=1e-5)
    np.testing.assert_allclose(gain[1, 0], jac_first, atol=1e-5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=False)
    assert gain is None
    np.testing.assert_allclose(nom[1, 3], a0, atol=1e-5)
    np.testing.assert_allclose(nom[1, 0], a0_first, atol=1e-5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=False, feedback=False)
    assert gain is None
    np.testing.assert_allclose(nom, chunk)


def test_shifted_noise():
    z = jax.random.normal(jax.random.key(8), (8, 3))
    s = reflex.shifted_noise(z)
    assert s.shape == (8, 8, 3)
    np.testing.assert_array_equal(s[:, 0], z)  # row 0 of entry j is z[j]
    np.testing.assert_array_equal(s[0], z)
    np.testing.assert_array_equal(s[3], jnp.roll(z, -3, axis=0))


def test_correct_is_identity_at_zero_deviation_and_clips():
    nom = jnp.array([0.1, -0.2])
    gain = jnp.array([[0.0, 10.0, 0.0], [1.0, 0.0, 0.0]])  # [A=2, O=3], off-diagonal: pins the einsum orientation
    ref = jnp.array([1.0, 2.0, 3.0])
    np.testing.assert_allclose(reflex.correct(nom, gain, ref, ref, 0.5), nom)
    out = reflex.correct(nom, gain, ref, ref + jnp.array([0.1, 1.0, 0.0]), 0.5)
    np.testing.assert_allclose(out, [0.1 + 0.5, -0.2 + 0.1], atol=1e-6)


def test_forward_equivalents():
    assert reflex.forward_equivalents("naive") == 5
    assert reflex.forward_equivalents("realtime") == 15
    assert reflex.forward_equivalents("pred") == 45
    assert reflex.forward_equivalents("reflex") == reflex.forward_equivalents("reflex_chunk") == 525


def test_errors_exact_for_linear_oracle():
    K, A, O = 3, 2, 4
    k = jax.random.split(jax.random.key(7), 4)
    jac = jax.random.normal(k[0], (K, A, O))
    a_ref = jax.random.normal(k[1], (K, A))
    nom_obs = jax.random.normal(k[2], (K, O))
    obs = nom_obs + 0.1 * jax.random.normal(k[3], (5, K, O))
    a_star = a_ref + jnp.einsum("kao,nko->nka", jac, obs - nom_obs)  # an oracle that IS linear
    e = probe.errors(a_ref, a_ref, jac, nom_obs, obs, a_star)
    np.testing.assert_allclose(e["lin"], 0, atol=1e-8)
    np.testing.assert_allclose(e["floor"], 0)
    assert e["pred"].shape == (5, K) and float(e["pred"].mean()) > 0


def _summary(rho_by_level, rel=0.5):
    rows = [
        {"level": lvl, "sigma": s, "k": k, "rho": r, "rel": rel, "pred": 1.0, "lin": 1 - r}
        for lvl, r in rho_by_level.items()
        for s in (0.1, 0.2)
        for k in range(1, 8)
    ]
    return pd.DataFrame(rows)


def test_verdict_rules():
    assert probe.verdict(_summary({f"l{i}": 0.8 for i in range(12)}))["verdict"] == "GO"
    assert probe.verdict(_summary({f"l{i}": 0.1 for i in range(12)}))["verdict"] == "KILL"
    assert probe.verdict(_summary({f"l{i}": 0.3 for i in range(12)}))["verdict"] == "GRAY"
    seven_good = {f"l{i}": (0.9 if i < 7 else 0.1) for i in range(12)}  # median 0.9 but only 7/12 levels ok
    assert probe.verdict(_summary(seven_good))["verdict"] == "GRAY"
    weak_noise = _summary({f"l{i}": 0.1 for i in range(12)}, rel=0.05)  # deviations barely matter at sigma=0.1
    weak_noise.loc[weak_noise["sigma"] == 0.2, ["rho", "lin"]] = [0.8, 0.2]
    v = probe.verdict(weak_noise)
    assert v["sigma"] == 0.2 and v["verdict"] == "GO"


def test_verdict_pools_over_k():
    df = _summary({f"l{i}": 0.8 for i in range(12)})
    df.loc[df["k"] == 1, ["pred", "lin"]] = [1e-3, 0.01]  # per-k rho = -9 at k = 1
    df.loc[df["k"].between(2, 4), ["pred", "lin"]] = [1.0, 0.2]
    # pooled: 1 - 0.61 / 3.001 = 0.80 -> GO; the mean of per-k rho would be (-9 + 3 * 0.8) / 4 = -1.65 -> KILL
    assert probe.verdict(df)["verdict"] == "GO"


def test_lin_clip_uses_the_e2_bound():
    jac = jnp.array([[[10.0], [0.1]]])  # [K=1, A=2, O=1]
    zeros_a, zeros_o = jnp.zeros((1, 2)), jnp.zeros((1, 1))
    e = probe.errors(zeros_a, zeros_a, jac, zeros_o, jnp.ones((1, 1)), zeros_a)
    np.testing.assert_allclose(e["lin"], [100.0 + 0.01], rtol=1e-6)
    np.testing.assert_allclose(e["lin_clip"], [1.0 + 0.01], rtol=1e-6)  # 10 is clipped to max_correction = 1


def test_verdict_on_clipped_correction():
    s = _summary({f"l{i}": 0.1 for i in range(12)})
    s["lin_clip"] = 0.2  # rho_clip = 0.8 with pred = 1
    assert probe.verdict(s)["verdict"] == "KILL"
    assert probe.verdict(s, "lin_clip")["verdict"] == "GO"
