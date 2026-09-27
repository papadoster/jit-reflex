"""Exploratory check (d) of docs/results/b2b5-checks.md: does the |e| normalization inflate |e| through
near-constant features (std just above predictors.normalized_error's 1e-6 cut, i.e. float32 rounding)?

The diagnostic's pairs (src/diag.py: seeds 3000 + i, 128 states x 4 draws, 4 chunks) are recomputed by a reduced copy
of diag.probe_state (same keys and operations, only the outputs rho_clip needs), and each pair's |e| is taken on the
same pair with the original std and with a relative cut: a feature moves if std > max(1e-6, tau * rms), tau in TAUS.
Then rho_clip by |e| bin (diag.bins) and its 0.3 crossing as in diag.summarize: published (stored raw), this run,
without cartpole_thrust, and with each tau; and the share of |e| > 2.3 for the learned predictor at k <= 7.
    uv run --offline python scripts/check_e_norm.py      (~35 min on the Mac CPU)
Writes results/b2b5/checks/e_norm.txt, e_norm_bins.csv, e_norm_dims.csv, e_norm_share.csv."""
import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import diag  # noqa: E402
import predictors  # noqa: E402
import probe  # noqa: E402
import reflex  # noqa: E402
import train_expert  # noqa: E402

OUT = pathlib.Path("results/b2b5/checks")
RAW = pathlib.Path(diag.OUT) / "raw"
TAUS = (1e-5, 1e-4, 1e-3)
VARIANTS = ("orig", *(f"tau{t:g}" for t in TAUS))
PHYS, NUM_ENVS, NUM_STATES, NUM_DRAWS, NUM_CHUNKS, S, SEED = (0.1, 0.2, 0.3), 64, 128, 4, 4, 5, 3000
NAMES = ["oracle", *(f"phys{p}" for p in PHYS), "learned"]
CARTPOLE, E_FAR, K_NEAR = "worlds_l_cartpole_thrust", 2.3, 7


def rms(boundaries):
    """Per-feature root mean square over the alive chunk-boundary states (the states alive_std uses): [O]."""
    obs = boundaries[0].reshape(-1, boundaries[0].shape[-1])
    alive = boundaries[2].reshape(-1, 1)
    return jnp.sqrt((jnp.square(obs) * alive).sum(0) / alive.sum())


def probe_pairs(policy, base, params, wm, raw, obs, key, stds):
    """diag.probe_state's pairs (same key splits and ops), reduced to pred, lin_clip, valid and |e| under each std."""
    H, A = policy.action_chunk_size, policy.action_dim
    k_z, k_f, k_n, k_env = jax.random.split(key, 4)
    zs = jax.random.normal(k_z, (NUM_CHUNKS, H, A))

    def true_step(s, a):
        o, s, _, done, _ = base.step_env(k_env, s, a, params)
        return o, s, done

    plan, truth, truth_ended = diag.chain(policy, true_step, raw, obs, zs, S)
    T = plan.shape[0]
    K = T - 1
    noisy = plan + train_expert.ACTION_NOISE_STD * jax.random.normal(k_n, (NUM_DRAWS, T, A))
    real, real_ended, _ = jax.vmap(lambda a: diag.roll(true_step, raw, a))(noisy)
    preds = [truth]
    for p in PHYS:
        f = predictors.phys_factors(k_f, raw, p)

        def phys_step(s, a, f=f):
            o, s = predictors.phys_step(base, k_env, s, a, params, f)
            return o, s, jnp.bool_(False)

        preds.append(diag.roll(phys_step, raw, plan)[0])
    preds.append(predictors.wm_rollout(wm, obs, plan))
    nom = jnp.stack(preds)[:, :K]
    o = real[:, :K]
    z = jax.vmap(reflex.shifted_noise)(zs).reshape(T, H, A)[1:]
    a_star = policy.action_from_noise(
        jnp.broadcast_to(z, (NUM_DRAWS, K, H, A)).reshape(-1, H, A), o.reshape(NUM_DRAWS * K, -1), S
    )[:, 0].reshape(NUM_DRAWS, K, A)
    valid = ~(real_ended[:, :K] | truth_ended[:K])

    def per_predictor(n):
        a_ref, jac = jax.vmap(lambda zk, ok: reflex.first_action_and_jacobian(policy, zk, ok, S))(z, n)
        err = probe.errors(plan[1:], a_ref, jac, n, o, a_star, act=lambda a: probe.executed(a, raw))
        e = jnp.stack([predictors.normalized_error(n, o, sd) for sd in stds], -1)  # [M, K, V]
        return {"pred": err["pred"], "lin_clip": err["lin_clip"], "e": e}

    return jax.vmap(per_predictor)(nom) | {"valid": valid}


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    env, env_params, levels, obs_dim, action_dim = probe.setup(probe.LEVELS)
    base = env._env

    @jax.jit
    def level_run(state_dict, level, key, wm):  # diag.run's level_diag with probe_pairs
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        k_c, k_s, k_p = jax.random.split(key, 3)
        boundaries, std = diag.level_std(env, env_params, policy, level, k_c, NUM_ENVS, S)
        obs, state = probe.sample(boundaries, k_s, NUM_STATES)
        r = rms(boundaries)
        stds = [std] + [jnp.where(std > jnp.maximum(1e-6, t * r), std, 0.0) for t in TAUS]

        def one(x):
            return probe_pairs(policy, base, env_params, wm, x[1].env_state, x[0], x[2], stds)

        res = jax.lax.map(one, (obs, state, jax.random.split(k_p, NUM_STATES)), batch_size=8)
        return res, std, r

    new, stored, dims = {}, {}, []
    for i, level_path in enumerate(probe.LEVELS):
        name = predictors.level_name(level_path)
        with open(pathlib.Path(predictors.WM_DIR) / f"{name}.pkl", "rb") as fh:
            wm = pickle.load(fh)
        res, std, r = jax.device_get(level_run(probe.load_state_dict("checkpoints/bc", level_path),
                                               jax.tree.map(lambda x: x[i], levels), jax.random.key(SEED + i), wm))
        new[name] = {k: np.asarray(v) for k, v in res.items()}
        with np.load(RAW / f"{name}.npz") as zf:
            stored[name] = {k: zf[k] for k in ("e", "pred", "lin_clip", "valid")}
        moving = std > 1e-6
        row = {"level": name, "moving": int(moving.sum()), "median_std": float(np.median(std[moving])),
               "same_valid": float((new[name]["valid"] == stored[name]["valid"]).mean()),
               "same_e_1e-5": float((np.abs(new[name]["e"][..., 0] - stored[name]["e"]) < 1e-5).mean())}
        for t in TAUS:
            cut = moving & ~(std > np.maximum(1e-6, t * r))
            row[f"dropped_tau{t:g}"] = int(cut.sum())
            row[f"dropped_std_tau{t:g}"] = " ".join(f"{x:.1e}" for x in np.sort(std[cut]))
        dims.append(row)
        print(row, flush=True)
    summarize(new, stored, pd.DataFrame(dims))


def kept_table(raws):
    """diag.bins needs only which k a level keeps: the oracle's frac of valid pairs >= MIN_FRAC."""
    return pd.DataFrame([{"level": lv, "predictor": "oracle", "k": k + 1, "frac": float(r["valid"][:, :, k].mean())}
                         for lv, r in raws.items() for k in range(r["valid"].shape[-1])])


def crossing(b):
    below = b[b["rho_clip"] < 0.3]
    return (float(below["e_lo"].iloc[0]), float(below["e_hi"].iloc[0])) if len(below) else (np.nan, np.nan)


def summarize(new, stored, dims):
    t_new, t_old = kept_table(new), kept_table(stored)
    published = pd.read_csv(pathlib.Path(diag.OUT) / "bins.csv").query("predictor == 'all'")
    edges = [*published["e_lo"], published["e_hi"].iloc[-1]]
    cases = {"published (stored raw)": (stored, t_old, None), "stored, no cartpole": (stored, t_old, CARTPOLE)}
    for v, name in enumerate(VARIANTS):
        cases[f"new run, {name}"] = ({lv: r | {"e": r["e"][..., v]} for lv, r in new.items()}, t_new, None)
        cases[f"new run, {name}, no cartpole"] = ({lv: r | {"e": r["e"][..., v]} for lv, r in new.items()}, t_new,
                                                  CARTPOLE)
    rows, lines = [], []
    for case, (raws, t, drop) in cases.items():
        raws = {lv: r for lv, r in raws.items() if lv != drop}
        for kind, ed in (("deciles", None), ("published edges", edges)):
            b = diag.bins(raws, t, edges=ed)
            rows.append(b.assign(case=case, edges=kind))
            lo, hi = crossing(b)
            lines.append(f"{case:34s} {kind:16s} rho_clip < 0.3 from |e| {lo:.2f}-{hi:.2f}; rho by bin "
                         + " ".join(f"{x:.2f}" for x in b["rho_clip"]))
    check = diag.bins(stored, t_old)  # the published table must come back from the stored raw
    ok = np.allclose(check["rho_clip"], published["rho_clip"]) and np.allclose(check["e_lo"], published["e_lo"])
    share = []
    for lv, r in new.items():
        p = NAMES.index("learned")
        v = r["valid"][:, :, :K_NEAR]
        for j, name in enumerate(VARIANTS):
            e = r["e"][:, p, :, :K_NEAR, j][v]
            share.append({"level": lv, "variant": name, "pairs": int(v.sum()), "share_far": float((e > E_FAR).mean()),
                          "median_e": float(np.median(e))})
    sh = pd.DataFrame(share)
    pooled = sh.assign(n_far=sh.share_far * sh.pairs).groupby("variant")[["n_far", "pairs"]].sum()
    pooled_nc = sh[sh.level != CARTPOLE].assign(n_far=lambda d: d.share_far * d.pairs).groupby("variant")[
        ["n_far", "pairs"]].sum()
    hist = pd.read_csv("results/b2b5/eval_B/hist.csv")
    far_from = int(np.argmax(np.asarray(edges) >= E_FAR - 0.05))  # hist bins are the published edges: 2.32 and up

    def far_share(h):
        return h.groupby(["method", "delay", "execute_horizon"]).apply(
            lambda x: x.loc[x["bin"] >= far_from, "count"].sum() / x["count"].sum())

    b5 = pd.DataFrame({"all levels": far_share(hist),
                       "no cartpole": far_share(hist[~hist.level.str.contains("cartpole_thrust")])})
    b5["diff_pp"] = 100 * (b5["no cartpole"] - b5["all levels"])
    pd.concat(rows).to_csv(OUT / "e_norm_bins.csv", index=False)
    dims.to_csv(OUT / "e_norm_dims.csv", index=False)
    sh.to_csv(OUT / "e_norm_share.csv", index=False)
    txt = (f"published bins reproduced from the stored raw: {ok}\n\nfeatures per level:\n" + dims.to_string()
           + "\n\nrho_clip by |e| bin and the 0.3 crossing:\n" + "\n".join(lines)
           + f"\n\nlearned, k <= {K_NEAR}: share of pairs with |e| > {E_FAR}, pooled over levels:\n"
           + (pooled.n_far / pooled.pairs).round(3).to_string()
           + "\nwithout cartpole_thrust:\n" + (pooled_nc.n_far / pooled_nc.pairs).round(3).to_string()
           + "\nper level (median |e| and share far):\n"
           + sh.pivot(index="level", columns="variant", values=["median_e", "share_far"]).round(2).to_string()
           + f"\n\nB2+B5 closed loop (eval_B/hist.csv), share of steps with |e| > {E_FAR} (bins from {edges[far_from]:.3f}):\n"
           + b5.round(3).to_string() + "\n")
    (OUT / "e_norm.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    run()
