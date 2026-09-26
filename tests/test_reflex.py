import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import pytest

import eval_flow
import model as _model
import plot
import probe
import reflex
import train_expert


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
    assert reflex.forward_equivalents("rtc_reflex") == 535


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


def test_executed_action_is_what_kinetix_applies():
    env, env_params, levels, _, action_dim = probe.setup(["worlds/l/grasp_easy.json"])
    _, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    s = state.env_state
    a = jnp.array([3.0, -3.0, 0.5, 0.2, -0.5, 2.0])
    b = a.at[0].set(1.5).at[4].set(-0.1)  # motor 3.0 vs 1.5 -> both 1; thruster -0.5 vs -0.1 -> both 0 (off)
    np.testing.assert_array_equal(probe.executed(a, s), probe.executed(b, s))
    assert probe.executed(jnp.ones((3, 7, action_dim)), s).shape[:2] == (3, 7)


def test_wilson_interval():
    lo, hi = plot.wilson(np.array([0.5]), np.array([100]))
    assert 0.40 < lo[0] < 0.41 and 0.59 < hi[0] < 0.60


def test_gate2_detects_go(tmp_path):
    rows = []
    for seed in (0, 1, 2):
        for d in (1, 2, 3, 4):
            for s in sorted({max(1, d), 8 - d}):
                for method, rate in (("naive", 0.5), ("realtime", 0.6), ("pred", 0.55), ("reflex", 0.75)):
                    rows.append({"seed": seed, "delay": d, "execute_horizon": s, "method": method,
                                 "level": "l", "returned_episode_solved": rate})
    pd.DataFrame(rows).to_csv(tmp_path / "main.csv", index=False)
    out = plot.gate2(str(tmp_path / "main.csv"), str(tmp_path / "no-kicks*/results.csv"))
    assert out["a"] and out["b"] and out["pred_check"] and not out["c"]
    assert out["decision"] == "GO"
    assert out["b_work_ratio"] > 1 and "fewer policy calls" in out["b_claim"]  # fewer calls != less compute


def test_kick_moves_only_active_dynamic_bodies():
    env, env_params, levels, _, _ = probe.setup(["worlds/l/grasp_easy.json"])
    _, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    raw = state.env_state
    kicked = train_expert.KickWrapper(env, prob=1.0, std=1.0).kick(jax.random.key(1), raw)
    for before, after in ((raw.polygon, kicked.polygon), (raw.circle, kicked.circle)):
        moved = np.any(np.asarray(after.velocity != before.velocity), axis=-1)
        dynamic = np.asarray((before.inverse_mass > 0) & before.active)
        np.testing.assert_array_equal(moved, dynamic)
        assert dynamic.any() or before is raw.circle  # the level has at least one dynamic polygon


def test_package_computes_only_used_positions():
    policy = small_policy()
    B, H, A, O = 2, policy.action_chunk_size, policy.action_dim, 5
    noise = jax.random.normal(jax.random.key(10), (B, H, A))
    ref = jax.random.normal(jax.random.key(11), (B, H, O))
    chunk = jnp.ones((B, H, A))
    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True)
    nom_u, gain_u = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True, used=(2, 5))
    assert nom_u.shape == nom.shape and gain_u.shape == gain.shape
    np.testing.assert_allclose(nom_u[:, 2:5], nom[:, 2:5], atol=1e-5)
    np.testing.assert_allclose(gain_u[:, 2:5], gain[:, 2:5], atol=1e-5)
    assert not np.any(nom_u[:, :2]) and not np.any(nom_u[:, 5:]) and not np.any(gain_u[:, 5:])
    nom_c, gain_c = reflex.package(policy, noise, ref, chunk, 5, requery=False, feedback=True, used=(2, 5))
    np.testing.assert_array_equal(nom_c, chunk)  # reflex_chunk keeps the chunk
    np.testing.assert_allclose(gain_c[:, 2:5], gain[:, 2:5], atol=1e-5)


def test_forward_equivalents_counts_only_computed_positions():
    assert reflex.forward_equivalents("reflex", positions=1) == 5 + 1 * (5 + 2 * 6 * 5)  # 70
    assert reflex.forward_equivalents("pred", positions=7) == 5 + 7 * 5  # 40
    assert reflex.forward_equivalents("reflex", positions=None) == 525  # phase A: all H positions
    assert reflex.forward_equivalents("naive", positions=3) == 5  # no package: positions don't matter


def test_parse_predictor():
    assert eval_flow.parse_predictor("oracle") == {"predictor": "oracle"}
    assert eval_flow.parse_predictor("learned") == {"predictor": "learned"}
    assert eval_flow.parse_predictor("phys0.2") == {"predictor": "phys", "phys_error": 0.2}


def _b1_rows(mid_pred=0.62, mid_extra=0.02, mid_rtc=0.62, mid_d3=0.6, base=(0.5, 0.6)):
    """Synthetic B1 results: d = 1, s = 1..7 (5 predictors for pred / reflex, 3 for rtc_reflex) and d = 3, s = 5."""
    rows = []

    def add(d, seed, s, method, predictor, rate):
        rows.append({"delay": d, "seed": seed, "execute_horizon": s, "method": method, "predictor": predictor,
                     "level": "l", "returned_episode_solved": rate})

    for seed in (10, 11, 12):
        for s in range(1, 8):
            add(1, seed, s, "naive", "-", base[0])
            add(1, seed, s, "realtime", "-", base[1])
            for pr, pred, j in (
                ("oracle", 0.62, 0.01 * s),
                ("phys0.1", 0.62, 0.01 * s + 0.01),
                ("phys0.2", mid_pred, 0.01 * s + mid_extra),
                ("phys0.3", 0.62, 0.01 * s),
                ("learned", 0.4, 0.0),
            ):
                add(1, seed, s, "pred", pr, pred)
                add(1, seed, s, "reflex", pr, pred + j)
            for pr, rate in (("oracle", 0.62), ("phys0.2", mid_rtc), ("learned", 0.55)):
                add(1, seed, s, "rtc_reflex", pr, rate)
        add(3, seed, 5, "naive", "-", base[0] - 0.1)
        add(3, seed, 5, "realtime", "-", base[1] - 0.1)
        for pr, reflex_rate in (("oracle", 0.6), ("phys0.2", mid_d3), ("learned", 0.45)):
            add(3, seed, 5, "pred", pr, 0.5)
            add(3, seed, 5, "reflex", pr, reflex_rate)
            add(3, seed, 5, "rtc_reflex", pr, 0.45)
    return pd.DataFrame(rows)


def test_b1_rules(tmp_path):
    def run(**kw):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        _b1_rows(**kw).to_csv(d / "results.csv", index=False)
        return plot.b1(str(d / "*.csv"), errors_csv=str(tmp_path / "missing.csv"), out_dir=str(d))

    out = run()  # the oracle passes both slices; phys0.2 passes D1 through reflex; learned fails everywhere
    assert out["p_mid"] == "phys0.2" and out["informative_slices"] == ["d1", "d3"]
    assert out["slices"]["d1"]["phys0.2"]["status"] == "PASS"
    assert out["verdict"] == "SURVIVES" and out["R4_learned"] == "TOO-WEAK"
    assert out["R2_staleness"] and abs(out["R2_gap"] - 0.04) < 1e-9  # J(oracle): s>=5 -> 0.06, s<=3 -> 0.02
    assert out["R3_j_grows_with_error"] and abs(out["R3_gap"] - 0.02) < 1e-9

    out = run(mid_pred=0.5, mid_extra=-0.2, mid_rtc=0.55, mid_d3=0.45)  # phys0.2 below the baseline everywhere
    assert out["verdict"] == "ORACLE-BOUND"

    out = run(mid_rtc=0.605, mid_pred=0.5, mid_extra=0.0, mid_d3=0.45)  # D1: rtc +0.5 pp (<1 pp), reflex < 0
    assert out["slices"]["d1"]["phys0.2"]["status"] == "GRAY" and out["verdict"] == "GRAY"

    out = run(base=(0.9, 0.95))  # nothing beats the baseline even with the oracle
    assert out["informative_slices"] == [] and out["verdict"] == "NO-EDGE" and out["R4_learned"] == "n/a"


def _b1_run(tmp_path, df, **kw):
    d = tmp_path / str(len(list(tmp_path.iterdir())))
    d.mkdir()
    df.to_csv(d / "results.csv", index=False)
    return plot.b1(str(d / "*.csv"), errors_csv=str(tmp_path / "missing.csv"), out_dir=str(d), **kw)


def test_b1_refuses_incomplete_grid(tmp_path):
    df = _b1_rows()
    hole = (df["method"] == "realtime") & (df["delay"] == 1) & (df["seed"] == 11) & (df["execute_horizon"] == 6)
    with pytest.raises(AssertionError, match="incomplete baseline"):
        _b1_run(tmp_path, df[~hole])
    lost_rtc = (df["method"] == "rtc_reflex") & (df["delay"] == 1)  # eval_d1_rtc lost
    lost_d3 = df["method"].isin(["pred", "reflex", "rtc_reflex"]) & (df["delay"] == 3) & (df["seed"] == 12)
    for lost in (lost_rtc, lost_d3):
        with pytest.raises(AssertionError, match="incomplete B1 grid"):
            _b1_run(tmp_path, df[~lost])


def test_b1_missing_data_is_never_a_verdict(tmp_path):
    df = _b1_rows(mid_pred=0.5, mid_extra=-0.2, mid_rtc=0.55, mid_d3=0.45)  # ORACLE-BOUND on the full grid
    out = _b1_run(tmp_path, df[~((df["method"] == "rtc_reflex") & (df["delay"] == 1))], strict=False)
    assert out["slices"]["d1"]["oracle"]["status"] == "PASS"  # reflex alone passes
    assert out["slices"]["d1"]["phys0.2"]["status"] == "MISSING" and out["verdict"] == "INCOMPLETE"

    df = _b1_rows()
    lost = df["method"].isin(["pred", "reflex", "rtc_reflex"]) & (df["delay"] == 3) & (df["seed"] == 12)
    out = _b1_run(tmp_path, df[~lost], strict=False)  # d3 has 2 of 3 seeds: MISSING, never PASS
    assert out["slices"]["d3"]["oracle"]["status"] == "MISSING"
    assert out["verdict"] == "SURVIVES"  # D1 passes on its own
    assert out["R4_learned"] == "INCOMPLETE"  # learned FAILs D1, but D3 might be informative

    df = _b1_rows(base=(0.9, 0.95))  # NO-EDGE on the full grid
    out = _b1_run(tmp_path, df[df["delay"] == 1], strict=False)
    assert out["slices"]["d3"]["oracle"]["status"] == "MISSING" and out["verdict"] == "INCOMPLETE"


def test_b1_explicit_p_mid(tmp_path):
    out = _b1_run(tmp_path, _b1_rows(), p_mid="phys0.3", strict=False)  # phys0.3 has no rtc_reflex rows
    assert out["p_mid"] == "phys0.3" and not out["R3_j_grows_with_error"]  # J(phys0.3) == J(oracle)
    assert out["slices"]["d3"]["phys0.3"]["status"] == "MISSING" and out["verdict"] == "SURVIVES"  # d1 via reflex
    with pytest.raises(AssertionError):
        _b1_run(tmp_path, _b1_rows(), p_mid="phys0.5")


def test_b1_ci_resamples_level_seed_cells(tmp_path):
    _b1_rows().to_csv(tmp_path / "results.csv", index=False)
    out = plot.b1_ci(str(tmp_path / "*.csv"), out_dir=str(tmp_path), n_boot=200)
    r = out[(out["what"] == "J oracle") & (out["where"] == "d1 s7")].iloc[0]
    assert r["cells"] == 3  # one level x three seeds
    assert r["mean"] == pytest.approx(7.0) and r["lo"] == pytest.approx(7.0) and r["hi"] == pytest.approx(7.0)
    assert (tmp_path / "b1_ci.png").exists()
