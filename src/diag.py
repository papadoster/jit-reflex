"""B1 diagnostic (exploratory): which prediction error J sees, and at which deviation and horizon the tangent breaks.

For a sampled state: a plan of a few policy chunks, each queried where the noise-free truth arrives; real rollouts add
action noise; every predictor rolls the same plan. At step k the reflex's pi(ô_k) + J·(o_k − ô_k) is compared with a
fresh call pi(o_k), as in E1b. See docs/superpowers/specs/2026-09-26-b1-diagnostic-design.md.
"""

import json
import os
import pathlib
import pickle
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import tyro

import predictors
import probe
import reflex
import train_expert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# per-pair quantities summed within a level (pooled ratios) and averaged (means); spec section 4
SUMS = ("pred", "lin", "lin_clip", "chunk", "chunk_lin_clip",
        "rs_pred", "top_pred", "d_pred", "rs_noise", "top_noise", "d_noise")
MEANS = ("e", "e_pred", "e_noise", "je", "je_pred", "je_noise", "cos_fix", "cos_spur", "clip")
OUT = "results/b1/diag"
MIN_FRAC = 0.3  # spec section 8: a level's point at k counts if >= 30% of its (state, draw) pairs are still valid
MIN_BIN = 30  # spec measure 9: pairs a level needs in an |e| bin
CHUNK = 8  # action chunk size of these policies: chunk boundaries on the figure's k axis
MEASURES = ("rho_clip", "res", "rho_chunk", "share_pred", "top1_pred", "share_noise", "top1_noise", *MEANS)


def row_space_share(jac, e):
    """|e|^2 inside J's row space and along its top right singular vector (divide by |e|^2 for the shares).

    jac [..., A, O], e [..., O] (broadcast) -> ([...], [...]). Singular values below eps·max(A, O) of the largest are
    rank loss, as numpy.linalg.matrix_rank.
    """
    _, s, vt = jnp.linalg.svd(jac, full_matrices=False)
    proj = jnp.einsum("...ao,...o->...a", vt, e)
    sq = jnp.where(s > jnp.finfo(s.dtype).eps * max(jac.shape[-2:]) * s[..., :1], proj**2, 0.0)
    return sq.sum(-1), sq[..., 0]


def cosine(x, y):
    """Cosine over the last axis; 0 where either vector is zero."""
    den = jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1)
    return jnp.where(den > 0, jnp.sum(x * y, -1) / jnp.where(den > 0, den, 1.0), 0.0)


def roll(step, state, actions):
    """step(state, action) -> (obs, state, done). actions [T, A] -> obs [T, O], ended by then [T], final state."""

    def body(c, a):
        s, ended = c
        o, s, done = step(s, a)
        return (s, ended | done), (o, ended | done)

    (state, _), (obs, ended) = jax.lax.scan(body, (state, jnp.bool_(False)), actions)
    return obs, ended, state


def chain(policy, step, state, obs, zs, num_steps: int):
    """Plan of len(zs) chunks; chunk c = pi(zs[c], o) at the obs where the noise-free truth arrived.

    zs [C, H, A] -> plan [C*H, A], truth obs after each action [C*H, O], ended by then [C*H].
    """
    plan, truth, ends, ended = [], [], [], jnp.bool_(False)
    for z in zs:  # C is small and static
        chunk = policy.action_from_noise(z[None], obs[None], num_steps)[0]
        o, e, state = roll(step, state, chunk)
        plan.append(chunk)
        truth.append(o)
        ends.append(e | ended)
        obs, ended = o[-1], ends[-1][-1]
    return jnp.concatenate(plan), jnp.concatenate(truth), jnp.concatenate(ends)


def probe_state(
    policy, base, params, wm, phys: Sequence[float], raw, obs, key, std, num_draws: int, num_chunks: int,
    num_steps: int,
):
    """Diagnostic of one state (spec sections 3-4).

    Returns SUMS + MEANS, each [P, M, K] (predictors oracle, phys..., learned; draws; k = 1..T-1), plus valid [M, K]
    (common to all predictors), used [A] (bound action dims), a_ref [P, K, A] and plan [T, A] for the invariant
    checks. base is the raw Kinetix env (no auto-reset), raw its state. Actions are compared as executed
    (probe.executed), as in E1b.
    """
    H, A = policy.action_chunk_size, policy.action_dim
    k_z, k_f, k_n, k_env = jax.random.split(key, 4)
    zs = jax.random.normal(k_z, (num_chunks, H, A))

    def true_step(s, a):
        o, s, _, done, _ = base.step_env(k_env, s, a, params)
        return o, s, done

    plan, truth, truth_ended = chain(policy, true_step, raw, obs, zs, num_steps)
    T = plan.shape[0]
    K = T - 1
    noisy = plan + train_expert.ACTION_NOISE_STD * jax.random.normal(k_n, (num_draws, T, A))
    real, real_ended, _ = jax.vmap(lambda a: roll(true_step, raw, a))(noisy)  # [M, T, O], [M, T]

    preds = [truth]
    for p in phys:
        f = predictors.phys_factors(k_f, raw, p)  # same key: same signs for every p, as in B1

        def phys_step(s, a, f=f):
            o, s = predictors.phys_step(base, k_env, s, a, params, f)
            return o, s, jnp.bool_(False)  # a prediction is not cut by the episode's end

        preds.append(roll(phys_step, raw, plan)[0])
    preds.append(predictors.wm_rollout(wm, obs, plan))
    nom = jnp.stack(preds)[:, :K]  # [P, K, O]: ô_k after k actions, k = 1..K
    o, o_star = real[:, :K], truth[:K]  # [M, K, O], [K, O]
    z = jax.vmap(reflex.shifted_noise)(zs).reshape(T, H, A)[1:]  # z[k-1]: row 0 is the noise plan[k] came from
    plan_k = plan[1:]  # the plan's own action at step k: the nominal of rtc_reflex-like corrections (measure 8)
    a_star = policy.action_from_noise(
        jnp.broadcast_to(z, (num_draws, K, H, A)).reshape(-1, H, A), o.reshape(num_draws * K, -1), num_steps
    )[:, 0].reshape(num_draws, K, A)
    valid = ~(real_ended[:, :K] | truth_ended[:K])

    def act(a):
        return probe.executed(a, raw)

    need = act(a_star)
    # E1b: J rows of dims that drive nothing are as large as bound ones; measures 2, 4, 7 use bound dims only
    used = jnp.abs(jax.jacfwd(act)(jnp.full(A, 0.5))).sum(0) > 0  # action dims that change the executed action

    def per_predictor(n):
        a_ref, jac = jax.vmap(lambda zk, ok: reflex.first_action_and_jacobian(policy, zk, ok, num_steps))(z, n)
        err = probe.errors(plan_k, a_ref, jac, n, o, a_star, act=act)  # [M, K]
        ju = jac * used[:, None]
        e_pred, e_noise = o_star - n, o - o_star  # [K, O], [M, K, O]
        je = jnp.einsum("kao,mko->mka", ju, o - n)
        je_pred = jnp.einsum("kao,ko->ka", ju, e_pred)
        rs_pred, top_pred = row_space_share(ju, e_pred)
        rs_noise, top_noise = row_space_share(ju, e_noise)
        out = {name: err[name] for name in ("pred", "lin", "lin_clip", "chunk", "chunk_lin_clip")} | {
            "rs_pred": rs_pred, "top_pred": top_pred, "d_pred": jnp.sum(e_pred**2, -1),
            "rs_noise": rs_noise, "top_noise": top_noise, "d_noise": jnp.sum(e_noise**2, -1),
            "e": predictors.normalized_error(n, o, std),
            "e_pred": predictors.normalized_error(n, o_star, std),
            "e_noise": predictors.normalized_error(o_star, o, std),
            "je": jnp.linalg.norm(je, axis=-1),
            "je_pred": jnp.linalg.norm(je_pred, axis=-1),
            "je_noise": jnp.linalg.norm(jnp.einsum("kao,mko->mka", ju, e_noise), axis=-1),
            "cos_fix": cosine(act(a_ref + jnp.clip(je, -1.0, 1.0)) - act(a_ref), need - act(a_ref)),
            "cos_spur": cosine(act(plan_k + jnp.clip(je_pred, -1.0, 1.0)) - act(plan_k), need - act(plan_k)),
            "clip": jnp.sum((jnp.abs(je) > 1.0) & used, -1) / jnp.maximum(used.sum(), 1),
        }
        return jax.tree.map(lambda x: jnp.broadcast_to(x, valid.shape), out), a_ref

    stats, a_ref = jax.vmap(per_predictor)(nom)
    return stats | {"valid": valid, "used": used, "a_ref": a_ref, "plan": plan}


def stored_config(res) -> dict | None:
    """The run settings saved with a level's raw arrays, without the level's own seed; None if none were saved."""
    if "config" not in res:
        return None
    return {k: v for k, v in json.loads(str(res["config"])).items() if k != "seed"}


def run(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = probe.LEVELS,
    phys: Sequence[float] = (0.1, 0.2, 0.3),
    world_model_dir: str = predictors.WM_DIR,
    num_envs: int = 64,
    num_states: int = 128,
    num_draws: int = 4,
    num_chunks: int = 4,
    num_flow_steps: int = 5,
    batch_size: int = 8,  # states per vmapped batch; lower it if RAM runs out
    seed: int = 3000,  # level probe.LEVELS[i] uses seed + i: disjoint from phase A and B1
    out_dir: str = OUT,
):
    """Spec section 3: raw arrays per level to out_dir/raw/<level>.npz (a finished level is skipped), then summarize."""
    env, env_params, levels, obs_dim, action_dim = probe.setup(level_paths)
    base = env._env  # raw Kinetix env: no auto-reset, as the B1 predictors
    phys = tuple(phys)
    names = np.array(["oracle", *(f"phys{p}" for p in phys), "learned"])
    raw_dir = pathlib.Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    config = {  # saved with every level: a resumed run must not mix files of other settings
        "run_path": run_path, "num_envs": num_envs, "num_states": num_states, "num_draws": num_draws,
        "num_chunks": num_chunks, "num_flow_steps": num_flow_steps, "phys": list(phys),
        "world_model_dir": world_model_dir,
    }

    @jax.jit
    def level_diag(state_dict, level, key, wm):
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        k_c, k_s, k_p = jax.random.split(key, 3)
        boundaries = probe.collect(
            env, env_params, policy, level, k_c, num_envs, 4, train_expert.ACTION_NOISE_STD, num_flow_steps
        )
        std = predictors.alive_std(boundaries)
        obs, state = probe.sample(boundaries, k_s, num_states)

        def one(x):
            return probe_state(
                policy, base, env_params, wm, phys, x[1].env_state, x[0], x[2], std,
                num_draws, num_chunks, num_flow_steps,
            )

        res = jax.lax.map(one, (obs, state, jax.random.split(k_p, num_states)), batch_size=batch_size)
        return res, boundaries[2].sum()

    for i, level_path in enumerate(level_paths):
        name = predictors.level_name(level_path)
        f = raw_dir / f"{name}.npz"
        level_seed = seed + probe.LEVELS.index(level_path)  # position in the full list: a subset keeps its seeds
        if f.exists():
            with np.load(f) as z:
                old = stored_config(z)
            if old != config:
                raise ValueError(f"{f} was made with {old}, this run has {config}: use another --out-dir")
            print(f"{level_path}: {f} exists, skipped", flush=True)
            continue
        with (pathlib.Path(world_model_dir) / f"{name}.pkl").open("rb") as fh:
            wm = pickle.load(fh)
        start = time.time()
        res, n_alive = jax.device_get(level_diag(
            probe.load_state_dict(run_path, level_path), jax.tree.map(lambda x: x[i], levels),
            jax.random.key(level_seed), wm,
        ))
        assert n_alive >= num_states, "too few alive states: raise --num-envs"
        part = f.with_suffix(".part")
        with part.open("wb") as fh:  # np.savez adds .npz to a path, not to a file handle
            np.savez_compressed(fh, predictors=names, config=np.array(json.dumps(config | {"seed": level_seed})), **res)
        os.replace(part, f)  # a killed run leaves no half-written level that a resume would skip
        print(f"{level_path}: done in {time.time() - start:.0f} s", flush=True)
    summarize(out_dir)


def add_ratios(t: pd.DataFrame) -> pd.DataFrame:
    """Pooled ratios of spec section 4 from the SUMS columns (a row may pool one k or several)."""

    def r(a, b):
        return t[a] / t[b].where(t[b] > 0)

    return t.assign(
        rho_clip=1 - r("lin_clip", "pred"), res=r("lin", "pred"), rho_chunk=1 - r("chunk_lin_clip", "chunk"),
        share_pred=r("rs_pred", "d_pred"), top1_pred=r("top_pred", "d_pred"),
        share_noise=r("rs_noise", "d_noise"), top1_noise=r("top_noise", "d_noise"),
    )


def level_table(res: dict, level: str) -> pd.DataFrame:
    """One level per (predictor, k): n and frac of valid pairs, SUMS pooled over them, MEANS averaged, ratios."""
    valid = res["valid"]  # [N, M, K]
    rows = []
    for p, name in enumerate(res["predictors"]):
        for k in range(valid.shape[-1]):
            v = valid[:, :, k]
            row = {"level": level, "predictor": str(name), "k": k + 1, "n": int(v.sum()), "frac": float(v.mean())}
            row |= {x: float(res[x][:, p, :, k][v].sum()) for x in SUMS}
            row |= {x: float(res[x][:, p, :, k][v].mean()) if v.any() else float("nan") for x in MEANS}
            rows.append(row)
    t = add_ratios(pd.DataFrame(rows))
    if "used" in res:  # rank of the masked J; the row-space share of a random error is about rank / O
        t["bound_dims"] = int(np.asarray(res["used"]).any(0).sum())
    return t


def curves(t: pd.DataFrame) -> pd.DataFrame:
    """Median over levels per (predictor, k) in two variants (spec section 4): 'all' = levels kept at this k,
    'survivors' = levels kept at every k. 'levels' = how many levels a median is over."""
    K = t["k"].max()
    kept = t[t["frac"] >= MIN_FRAC]
    n_kept = kept[kept["predictor"] == "oracle"].groupby("level")["k"].nunique()  # the mask is common to predictors
    survivors = n_kept[n_kept == K].index
    out = []
    for variant, d in (("all", kept), ("survivors", kept[kept["level"].isin(survivors)])):
        g = d.groupby(["predictor", "k"])
        c = g[list(MEASURES)].median()
        c["levels"] = g["level"].nunique()
        out.append(c.reset_index().assign(variant=variant))
    return pd.concat(out, ignore_index=True)


def near(t: pd.DataFrame, k_max: int = 7) -> pd.DataFrame:
    """Q1/Q2 table (spec section 5): per predictor, median over levels of values pooled over k = 1..k_max."""
    d = t[(t["k"] <= k_max) & (t["frac"] >= MIN_FRAC)]
    by = d.groupby(["level", "predictor"])
    lv = add_ratios(by[list(SUMS)].sum().join(by[list(MEANS)].mean()).reset_index())
    return lv.groupby("predictor")[list(MEASURES)].median()


def bins(raws: dict, t: pd.DataFrame, n_bins: int = 10, p: int | None = None, edges=None) -> pd.DataFrame:
    """Spec measure 9: rho_clip against |e| over the pairs of all levels and kept k, of predictor index p (None: all).

    Bins are deciles of |e| over these pairs unless edges are given. levels = levels with >= MIN_BIN pairs in a bin
    (rho_clip is the median over them), pairs = the bin's pairs over all levels, k_med = their median k.
    """
    kept = t[(t["predictor"] == "oracle") & (t["frac"] >= MIN_FRAC)].groupby("level")["k"].apply(list)
    sel = slice(None) if p is None else slice(p, p + 1)
    pairs = []
    for level, res in raws.items():
        ks = np.asarray(kept.get(level, []), int) - 1
        e, lin_clip, pred = (res[x][:, sel][..., ks] for x in ("e", "lin_clip", "pred"))
        v = np.broadcast_to(res["valid"][..., ks][:, None], e.shape)
        pairs.append((e[v], lin_clip[v], pred[v], np.broadcast_to(ks + 1, e.shape)[v]))
    if edges is None:
        edges = np.quantile(np.concatenate([x[0] for x in pairs]), np.linspace(0, 1, n_bins + 1))
    rows = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        rhos, ks_in = [], []
        for e, lin_clip, pred, k in pairs:
            m = (e >= lo) & ((e < hi) if b < len(edges) - 2 else (e <= hi))
            ks_in.append(k[m])
            if m.sum() >= MIN_BIN and pred[m].sum() > 0:
                rhos.append(1 - lin_clip[m].sum() / pred[m].sum())
        ks_in = np.concatenate(ks_in)
        rows.append({"bin": b, "e_lo": lo, "e_hi": hi, "rho_clip": float(np.median(rhos)) if rhos else np.nan,
                     "levels": len(rhos), "pairs": len(ks_in),
                     "k_med": float(np.median(ks_in)) if len(ks_in) else np.nan})
    return pd.DataFrame(rows)


def first_below(x, y, thr: float):
    """The first x where y < thr (NaN never counts), or None."""
    below = np.asarray(y, float) < thr
    return float(np.asarray(x)[below.argmax()]) if below.any() else None


def figure(cv: pd.DataFrame, bn: pd.DataFrame, path: pathlib.Path):
    K = int(cv["k"].max())
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.5))
    colors = dict(zip(sorted(cv["predictor"].unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for (pred, variant), d in cv.groupby(["predictor", "variant"]):
        d = d.sort_values("k")
        a.plot(d["k"], d["rho_clip"], ls="-" if variant == "all" else "--", color=colors[pred],
               label=pred if variant == "all" else None)
        if variant == "all" and pred != "oracle":
            c.plot(d["e_pred"], d["je_pred"], marker=".", color=colors[pred], label=pred)
    lv = cv[(cv["predictor"] == "oracle") & (cv["variant"] == "all")].sort_values("k")
    a2 = a.twinx()
    a2.step(lv["k"], lv["levels"], where="mid", color="gray", lw=0.8)
    a2.set_ylabel("levels at k (solid curves)")
    surv = cv.loc[cv["variant"] == "survivors", "levels"]
    for y in (0.3, 0.0):
        a.axhline(y, c="gray", lw=0.6, ls=":")
        b.axhline(y, c="gray", lw=0.6, ls=":")
    a.set_title(f"ρ_clip vs k (dashed: the {int(surv.max()) if len(surv) else 0} levels alive to k = {K})")
    bounds = ", ".join(map(str, range(CHUNK, K + 1, CHUNK)))
    a.set_xlabel("k, steps after the call" + (f" (chunk boundaries at {bounds})" if bounds else ""))
    a.legend(fontsize=7)
    for pred, d in bn.groupby("predictor"):
        mid = (d["e_lo"] + d["e_hi"]) / 2
        if pred == "all":
            b.plot(mid, d["rho_clip"], marker="o", color="black", lw=2, label="all predictors")
            for x, y, n in zip(mid, d["rho_clip"], d["levels"]):
                b.annotate(str(n), (x, y), fontsize=7)
        else:
            b.plot(mid, d["rho_clip"], marker=".", lw=1, color=colors[pred], label=pred)
    b.set_xscale("log")
    b.set_xlabel("|o − ô| (normalized), decile bins of all pairs; labels = levels (black)")
    b.set_title("ρ_clip vs deviation (all k)")
    b.legend(fontsize=7)
    c.set_xlabel("|o* − ô| (normalized)")
    c.set_ylabel("|J·(o* − ô)|")
    c.set_title(f"prediction error J sees, k = 1…{K}")
    c.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(out_dir: str = OUT):
    """Tables, thresholds and figure from out_dir/raw/*.npz (spec sections 4-5)."""
    out = pathlib.Path(out_dir)
    raws = {f.stem: dict(np.load(f)) for f in sorted((out / "raw").glob("*.npz"))}
    configs = {json.dumps(c, sort_keys=True) for r in raws.values() if (c := stored_config(r)) is not None}
    assert len(configs) <= 1, f"raw files of different runs in {out / 'raw'}: {configs}"
    t = pd.concat([level_table(r, lv) for lv, r in raws.items()], ignore_index=True)
    cv, nr, pooled = curves(t), near(t), bins(raws, t)
    edges = [*pooled["e_lo"], pooled["e_hi"].iloc[-1]]  # every predictor on the pooled bins (spec measure 9)
    names = [str(x) for x in next(iter(raws.values()))["predictors"]]  # one config: the same predictors everywhere
    bn = pd.concat([pooled.assign(predictor="all")]
                   + [bins(raws, t, p=p, edges=edges).assign(predictor=n) for p, n in enumerate(names)],
                   ignore_index=True)
    o = t[(t["predictor"] == "oracle") & t["k"].between(1, 4) & (t["frac"] >= MIN_FRAC)].groupby("level")
    rho = float((1 - o["lin_clip"].sum() / o["pred"].sum()).median())
    e_pred = float(t.loc[t["predictor"] == "oracle", "e_pred"].abs().max())
    check = {"oracle_rho_clip_k1_4": rho, "e1b": 0.54, "oracle_e_pred_max": e_pred,
             "ok": abs(rho - 0.54) <= 0.1 and e_pred == 0}
    th = {"k": {}, "e": {}}
    for pred, d in bn.groupby("predictor", sort=False):  # spec section 5 Q3: per predictor; "all" = pooled
        below = {thr: d[d["rho_clip"] < thr] for thr in (0.3, 0.0)}  # NaN is never below
        th["e"][pred] = {f"<{thr}": {"e": float(b["e_lo"].iloc[0]), "levels": int(b["levels"].iloc[0])}
                         if len(b) else None for thr, b in below.items()}
    for (variant, pred), d in cv.groupby(["variant", "predictor"]):
        d = d.sort_values("k")
        th["k"][f"{variant}/{pred}"] = {f"<{thr}": first_below(d["k"], d["rho_clip"], thr) for thr in (0.3, 0.0)}
    t.to_csv(out / "summary.csv", index=False)
    cv.to_csv(out / "curves.csv", index=False)
    nr.to_csv(out / "near.csv")
    bn.to_csv(out / "bins.csv", index=False)
    (out / "check.json").write_text(json.dumps(check, indent=2))
    (out / "thresholds.json").write_text(json.dumps(th, indent=2))
    figure(cv, bn, out / "diag.png")
    print(nr.round(3).to_string())
    print(bn.round(3).to_string(index=False))
    print(json.dumps(check))
    print(json.dumps(th))
    if not check["ok"]:
        print("!!! sanity check failed (spec section 5): look for a bug before reading anything")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"run": run, "summarize": summarize})
