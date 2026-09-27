"""Exploratory check (c) of docs/results/b2b5-checks.md: is T3's correction softer than the exact reflex's?

On the offline-B2 states and keys (src/b2.py: seeds 4000 + i, 512 states x 4 draws, k = 1..7), learned predictor
only: the executed correction c = exec(nom + clip(J·δ)) − exec(nom) of the exact J and of T3, against the needed
n = exec(a*) − exec(nom). The states are a fresh draw with b2's seeds (a new XLA program does not reproduce b2's states
bit for bit: chaotic contacts amplify rounding). Code check: on the same inputs, the residuals here equal b2's
definition (probe.errors lin_clip); distribution check: per-level mean residuals against b2's stored raw. |e| bins use
the original std and the relative cut of check (d) (tau 1e-4).
    uv run --offline python scripts/check_t3_soft.py      (~15 min on the Mac CPU)
Writes results/b2b5/checks/t3_soft.csv (per |e| bin), t3_soft_levels.csv and t3_soft.txt."""
import pathlib
import pickle
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
sys.path.insert(0, "scripts")
import b2  # noqa: E402
import diag  # noqa: E402
import predictors  # noqa: E402
import probe  # noqa: E402
import reflex  # noqa: E402
import train_expert  # noqa: E402
from check_e_norm import rms  # noqa: E402

OUT = pathlib.Path("results/b2b5/checks")
RAW = pathlib.Path(b2.OUT) / "raw"
S, NUM_ENVS, NUM_STATES, NUM_DRAWS, BATCH, SEED = b2.S, 256, 512, 4, 8, 4000  # as b2.run's defaults


def probe_soft(policy, base, params, wm, raw, obs, key, stds):
    """b2.probe_state's state, draws and fresh calls (same key splits), learned predictor, rows reflex and T3."""
    H, A_ = policy.action_chunk_size, policy.action_dim
    k_z, k_n, k_env = jax.random.split(key, 3)
    z = jax.random.normal(k_z, (H, A_))

    def true_step(s, a):
        o, s, _, done, _ = base.step_env(k_env, s, a, params)
        return o, s, done

    xs = b2.flow_states(policy, z, obs, S)
    chunk = xs[-1]
    _, truth_ended, _ = diag.roll(true_step, raw, chunk)
    K = H - 1
    noisy = chunk + train_expert.ACTION_NOISE_STD * jax.random.normal(k_n, (NUM_DRAWS, H, A_))
    real, real_ended, _ = jax.vmap(lambda a: diag.roll(true_step, raw, a))(noisy)
    nom_obs = predictors.wm_rollout(wm, obs, chunk)[:K]  # [K, O]: learned ô_k
    o = real[:, :K]
    zk = reflex.shifted_noise(z)[1:]
    warm = jax.vmap(lambda k: jnp.roll(xs, -k, axis=1))(jnp.arange(1, H))
    a_star = policy.action_from_noise(
        jnp.broadcast_to(zk, (NUM_DRAWS, K, H, A_)).reshape(-1, H, A_), o.reshape(NUM_DRAWS * K, -1), S
    )[:, 0].reshape(NUM_DRAWS, K, A_)
    valid = ~(real_ended[:, :K] | truth_ended[:K])

    def ex(a):
        return probe.executed(a, raw)

    a_ref, jac = jax.vmap(lambda zj, oj: reflex.first_action_and_jacobian(policy, zj, oj, S))(zk, nom_obs)
    a_3, jac_3 = jax.vmap(
        lambda zj, wj, oj: b2.action_and_jacobian(b2.first_action_fn(policy, "T3", zj, wj, S), oj)
    )(zk, warm, nom_obs)
    delta = o - nom_obs  # [M, K, O]
    need = ex(a_star) - ex(a_ref)  # [M, K, E]
    out = {"e": predictors.normalized_error(nom_obs, o, stds[0]), "e_rel": predictors.normalized_error(nom_obs, o, stds[1]),
           "valid": valid,
           "need": jnp.linalg.norm(need, axis=-1), "nom_gap": jnp.abs(a_3 - a_ref).max(-1)[None].repeat(NUM_DRAWS, 0)}
    for name, a0, j in (("reflex", a_ref, jac), ("T3", a_3, jac_3)):
        lin = jnp.einsum("kao,mko->mka", j, delta)
        c = ex(a0 + jnp.clip(lin, -1.0, 1.0)) - ex(a0)
        nn = jnp.sum(need * need, -1)
        out |= {
            f"size_{name}": jnp.linalg.norm(c, axis=-1),
            f"lin_{name}": jnp.linalg.norm(lin, axis=-1),
            f"proj_{name}": jnp.sum(c * need, -1) / jnp.where(nn > 0, nn, 1.0),
            f"res_{name}": jnp.sum(jnp.square(ex(a0 + jnp.clip(lin, -1.0, 1.0)) - ex(a_star)), -1),
            f"b2res_{name}": probe.errors(chunk[1:], a0, j, nom_obs, o, a_star, act=ex)["lin_clip"],  # b2's definition
        }
    return out


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    env, env_params, levels, obs_dim, action_dim = probe.setup(probe.LEVELS)
    base = env._env

    @jax.jit
    def level_run(state_dict, level, key, wm):  # b2.run's level_run with probe_soft
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        k_c, k_s, k_p = jax.random.split(key, 3)
        boundaries = probe.collect(env, env_params, policy, level, k_c, NUM_ENVS, 4, train_expert.ACTION_NOISE_STD, S)
        std = predictors.alive_std(boundaries)
        stds = [std, jnp.where(std > jnp.maximum(1e-6, 1e-4 * rms(boundaries)), std, 0.0)]  # check (d)'s tau 1e-4
        obs, state = probe.sample(boundaries, k_s, NUM_STATES)

        def one(x):
            return probe_soft(policy, base, env_params, wm, x[1].env_state, x[0], x[2], stds)

        return jax.lax.map(one, (obs, state, jax.random.split(k_p, NUM_STATES)), batch_size=BATCH)

    rows, checks = [], []
    for i, level_path in enumerate(probe.LEVELS):
        name = predictors.level_name(level_path)
        with open(pathlib.Path(predictors.WM_DIR) / f"{name}.pkl", "rb") as fh:
            wm = pickle.load(fh)
        r = jax.device_get(level_run(probe.load_state_dict("checkpoints/bc", level_path),
                                     jax.tree.map(lambda x: x[i], levels), jax.random.key(SEED + i), wm))
        code = max(float(np.abs(r[f"res_{c}"] - r[f"b2res_{c}"]).max()) for c in ("reflex", "T3"))
        with np.load(RAW / f"{name}.npz") as zf:
            ms = zf["valid"] & (zf["e"][:, 1] <= b2.E_MAX)
            old = {c: float(zf[c][:, 1][ms].mean()) for c in ("reflex", "T3")}
        mn = r["valid"] & (r["e"] <= b2.E_MAX)
        checks.append(f"{name}: code max |res - b2 def| = {code:.1e}; nom_T3 gap {float(r['nom_gap'].max()):.1e}; "
                      f"mean res |e|<=2.3 reflex {float(r['res_reflex'][mn].mean()):.4f} (b2 {old['reflex']:.4f}), "
                      f"T3 {float(r['res_T3'][mn].mean()):.4f} (b2 {old['T3']:.4f})")
        print(checks[-1], flush=True)
        m = r["valid"]
        rows.append(pd.DataFrame({"level": name, **{k: np.asarray(v)[m] for k, v in r.items() if k not in ("valid",)}}))
    df = pd.concat(rows)
    edges = pd.read_csv(b2.DIAG_BINS).query("predictor == 'all'").sort_values("bin")["e_lo"].to_numpy()

    def summary(g):
        o = g["need"] > 1e-6  # the projection needs a nonzero needed correction
        return pd.Series({
            "pairs": len(g), "e_lo": edges[g.name],
            "size_ratio": g["size_T3"].sum() / g["size_reflex"].sum(),
            "lin_ratio": g["lin_T3"].sum() / g["lin_reflex"].sum(),
            "overshoot_reflex": float((g.loc[o, "proj_reflex"] > 1).mean()),
            "overshoot_T3": float((g.loc[o, "proj_T3"] > 1).mean()),
            "proj_reflex": float(g.loc[o, "proj_reflex"].median()), "proj_T3": float(g.loc[o, "proj_T3"].median()),
            "res_reflex": g["res_reflex"].mean(), "res_T3": g["res_T3"].mean(),
        })

    parts, lvs = [], []
    for col in ("e", "e_rel"):
        d = df.assign(bin=np.searchsorted(edges, df[col].to_numpy(), side="right") - 1)
        parts.append(d.groupby("bin").apply(summary).assign(e_col=col))
        far = d[d[col] > b2.E_MAX]
        lvs.append(far.groupby("level").apply(lambda g: pd.Series({
            "pairs": len(g), "size_ratio": g["size_T3"].sum() / g["size_reflex"].sum(),
            "overshoot_reflex": float((g.loc[g.need > 1e-6, "proj_reflex"] > 1).mean()),
            "overshoot_T3": float((g.loc[g.need > 1e-6, "proj_T3"] > 1).mean())})).assign(e_col=col))
    by_bin, lv = pd.concat(parts), pd.concat(lvs)
    by_bin.to_csv(OUT / "t3_soft.csv")
    lv.to_csv(OUT / "t3_soft_levels.csv")
    txt = "\n".join(checks)
    for col in ("e", "e_rel"):
        b, l_ = by_bin[by_bin.e_col == col], lv[lv.e_col == col]
        txt += (f"\n\n=== |e| = {col} ===\nper |e| bin (all levels pooled):\n" + b.round(3).to_string()
                + f"\n\n|e| > {b2.E_MAX}, per level:\n" + l_.round(3).to_string()
                + f"\nlevels with size_ratio < 0.9: {(l_.size_ratio < 0.9).sum()}/{len(l_)}, "
                  f"overshoot T3 < reflex: {(l_.overshoot_T3 < l_.overshoot_reflex).sum()}/{len(l_)}")
    (OUT / "t3_soft.txt").write_text(txt + "\n")
    print(txt)


if __name__ == "__main__":
    run()
