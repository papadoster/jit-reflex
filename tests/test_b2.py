import json

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

import b2
import diag
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


def _ratios(rows: dict) -> pd.DataFrame:
    """A ratios() table of 12 levels for 'learned': rows = {row: [(R, gain, beats_pred), ...12]}."""
    out = []
    for row, vals in rows.items():
        for i, (R, gain, beats) in enumerate(vals):
            out.append({"level": f"l{i}", "predictor": "learned", "row": row, "res": 0.0, "R": R, "gain": gain,
                        "beats_pred": beats})
    return pd.DataFrame(out)


def _rows(**over):
    base = {c: [(0.5, 0.5, True)] * 12 for c in b2.CANDIDATES}  # everything fails at R = 0.5
    base |= {"reflex": [(1.0, 0.5, True)] * 12, "chunk_j": [(0.5, 0.5, True)] * 12, "shared": [(0.5, 0.5, True)] * 12}
    base["T1"] = [(0.85, 0.5, True)] * 12  # passes, not strict; depth 6, FE 90
    base["T2"] = [(0.9, 0.5, True)] * 12  # strict; depth 7, FE 150
    base["W3"] = [(0.99, 0.5, True)] * 12  # strict; depth 6, FE 200
    # W1: 6 levels with R 0.95, 6 with a tiny gain and R -1. The guard leaves the 6 out: median 0.95, not -0.025
    base["W1"] = [(0.95, 0.5, True)] * 6 + [(-1.0, 0.05, True)] * 6
    base["M1"] = [(0.97, 0.5, True)] * 9 + [(0.97, 0.5, False)] * 3  # R 0.97 but beats pred on 9 < 10 levels
    return base | over


def test_rule_guard_and_selection():
    rt = b2.rule(_ratios(_rows()))
    assert rt.loc["W1", "levels_in_median"] == 6 and rt.loc["W1", "R"] == 0.95
    assert rt.loc["W1", "left_out"] == " ".join(f"l{i}" for i in range(6, 12))
    assert not rt.loc["M1", "pass"] and rt.loc["T1", "pass"] and not rt.loc["T1", "strict"]
    assert rt.loc["T1", "depth"] == 6 and rt.loc["W3", "fe"] == 200
    # 1: the shallowest pass is W1 (depth 2); 2: the best passing T_k is T2; 3: the shallowest strict is W1 again
    assert b2.select(rt) == {"passed": ["W1", "T1", "W3", "T2"], "gpu": ["W1", "T2"], "fallback": False}
    # a full tie of depth and FE (W1 and M1) goes to the larger R
    rt = b2.rule(_ratios(_rows(M1=[(0.97, 0.5, True)] * 12)))
    assert b2.select(rt)["gpu"] == ["M1", "T2"]
    # the shallowest pass is not strict: point 3 adds the shallowest strict one (W3: depth 6 < T2's 7)
    rt = b2.rule(_ratios(_rows(W1=[(0.85, 0.5, True)] * 12)))
    assert b2.select(rt)["gpu"] == ["W1", "T2", "W3"]
    # nothing passes: the best R goes as an exploratory row
    rt = b2.rule(_ratios(_rows(T1=[(0.6, 0.5, True)] * 12, T2=[(0.7, 0.5, True)] * 12, W3=[(0.5, 0.5, True)] * 12,
                               W1=[(0.5, 0.5, True)] * 12)))
    assert b2.select(rt) == {"passed": [], "gpu": ["M1"], "fallback": True}


def _raw(N=40, M=2, K=7):
    """Synthetic probe_state output over N states, alike for both predictors. Pairs with |e| <= E_MAX: pred 1,
    reflex 0.5, every candidate 0.55 (R = 0.9). Pairs above it: candidates 5 (R would be negative if they counted)."""
    r = {x: np.ones((N, 2, M, K)) for x in b2.SUMS}
    r["reflex"] *= 0.5
    for c in b2.CANDIDATES:
        r[c] = np.full((N, 2, M, K), 0.55)
        r[c][N // 2:] = 5.0
    e = np.random.default_rng(0).uniform(0.1, 2.0, (N, 2, M, K))
    e[N // 2:] += 10.0
    return r | {"e": e, "valid": np.ones((N, M, K), bool), "predictors": np.array(["oracle", "learned"])}


def test_summarize_cuts_at_e_max_and_writes_everything(tmp_path, monkeypatch):
    raws = {"a": _raw(), "b": _raw()}
    t = pd.concat([b2.level_table(r, lv) for lv, r in raws.items()], ignore_index=True)
    assert (t["n"] == 20 * 2 * 7).all()  # only the half with |e| <= E_MAX
    r = b2.ratios(t)
    np.testing.assert_allclose(r.loc[r["row"] == "T1", "R"], 0.9)
    np.testing.assert_allclose(r.loc[r["row"] == "T1", "res"], 0.55)  # per pair
    monkeypatch.setattr(diag, "MIN_BIN", 1)
    (tmp_path / "raw").mkdir()
    for lv, x in raws.items():
        np.savez_compressed(tmp_path / "raw" / f"{lv}.npz", **x)
    b2.summarize(str(tmp_path))
    for f in ("summary.csv", "ratios.csv", "rule.csv", "selection.json", "bins.csv", "check.json", "b2.png"):
        assert (tmp_path / f).exists(), f
    assert json.loads((tmp_path / "check.json").read_text())["oracle_reflex_rho_clip_k1_4"] == 0.5
    assert json.loads((tmp_path / "selection.json").read_text())["fallback"]  # 2 levels < MIN_LEVELS
    bn = pd.read_csv(tmp_path / "bins.csv")
    assert set(bn["predictor"]) == {"oracle", "learned"} and set(bn["row"]) == set(b2.ROWS)
    top = bn[(bn["bin"] == bn["bin"].max()) & (bn["row"] == "T1")]
    np.testing.assert_allclose(top["res"], 5.0)  # the open last bin holds the |e| > 10 pairs


def test_a_level_without_pairs_does_not_beat_pred():
    row = {x: 0.0 for x in b2.SUMS}
    t = pd.DataFrame([{"level": "a", "predictor": "learned", "n": 10} | row | {"pred": 1.0, "reflex": 0.5, "T1": 0.6},
                      {"level": "b", "predictor": "learned", "n": 0} | row])
    r = b2.ratios(t)
    assert r.loc[r["row"] == "T1", "beats_pred"].tolist() == [True, False]
