"""B2 offline: cheaper reflex packages, scored against a fresh call as in the B1 diagnostic.

For a sampled state: one policy chunk, real rollouts with action noise, predictors oracle and learned. At chunk
index k = 1..7 each package's nom + clip(J·(o − ô)) is compared with a fresh call pi(o) as executed actions (E1b).
The candidates cut the depth of the reflex call: T_k (J through the last k flow steps), W_k (warm start: the last k
steps from the chunk's own flow state), M_m (an m-step flow). See docs/superpowers/specs/2026-09-26-b2-offline-design.md.
"""

import hashlib
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

import diag
import predictors
import probe
import reflex
import train_expert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

S, A, P = 5, 6, 5  # flow steps, action dims, package positions (s = 5): the prices of spec section 3
CANDIDATES = ("T1", "T2", "T3", "W1", "W2", "W3", "M1", "M2", "M3")
ROWS = ("pred", "reflex", "chunk_j", "shared", *CANDIDATES)  # pred = no J; chunk_j = chunk + exact J (rtc_reflex-like)
SUMS = (*ROWS, "nom_chunk_j", *(f"nom_{c}" for c in CANDIDATES))  # squared executed-action errors, summed per level
NOM = {"reflex": "pred", "shared": "pred", "chunk_j": "nom_chunk_j"} | {c: f"nom_{c}" for c in CANDIDATES}
SHARED_K = 5  # the one-J row: J at chunk index 5, the middle of the executed 3..7 at d = 3, s = 5
E_MAX = 2.3  # spec section 6: the rule pools pairs with |e| <= 2.3
MIN_GAIN = 0.1  # levels where the exact reflex improves res over pred by < 10% leave the median of R
PASS_R, STRICT_R, MIN_LEVELS = 0.8, 0.9, 10
OUT = "results/b2/offline"
DIAG_BINS = "results/b1/diag/bins.csv"  # the |e| bin edges: deciles of the diagnostic's pairs (predictor "all")


def depth(row: str) -> int:
    """Sequential flow steps after the chunk and the predictor, forward = backward = 1 (spec section 6)."""
    if row == "pred":
        return S
    if row in ("reflex", "chunk_j", "shared"):
        return 2 * S
    n = int(row[1:])
    return S + n if row[0] == "T" else 2 * n


def fe(row: str) -> int:
    """Forward equivalents per call, a VJP counted as 2 (reflex.forward_equivalents; spec section 3)."""
    if row == "pred":
        return S + P * S
    if row in ("reflex", "chunk_j"):
        return S + P * (S + 2 * A * S)
    if row == "shared":
        return S + P * S + 2 * A * S
    n = int(row[1:])
    return S + P * (S + 2 * A * n) if row[0] == "T" else S + P * (n + 2 * A * n)


def run_flow(policy, x, obs, t, n: int, dt: float):
    """n Euler steps of pi's flow, as in model.action_from_noise, from x [H, A] at time t, conditioned on obs [O].

    Returns ((x, t) after the steps, the states before each step [n, H, A]).
    """

    def step(c, _):
        x, t = c
        return (x + dt * policy(obs[None], x[None], t)[0], t + dt), x

    return jax.lax.scan(step, (x, jnp.asarray(t, x.dtype)), None, length=n)


def flow_states(policy, noise, obs, num_steps: int):
    """The chunk's flow states x_0 = noise, ..., x_S = the chunk: [S + 1, H, A]."""
    (x, _), xs = run_flow(policy, noise, obs, 0.0, num_steps, 1 / num_steps)
    return jnp.concatenate([xs, x[None]])


def first_action_fn(policy, name: str, z, warm, num_steps: int):
    """o [O] -> the first action of package `name` ("T2", "W1", "M3", ...) at one chunk index (spec section 3).

    z [H, A] is the call's noise rolled to this index (reflex.shifted_noise); warm [S + 1, H, A] is the chunk's flow
    states rolled the same way (W only). T_n, W_n and M_n with n = num_steps are the exact reflex.
    """
    kind, n = name[0], int(name[1:])
    if kind == "T":  # the exact forward pass; J only through the last n steps

        def f(o):
            (x, t), _ = run_flow(policy, z, o, 0.0, num_steps - n, 1 / num_steps)
            (x, _), _ = run_flow(policy, jax.lax.stop_gradient(x), o, t, n, 1 / num_steps)
            return x[0]

    elif kind == "W":  # the last n steps from the chunk's state at step S - n (queried at o_0, not at ô)

        def f(o):
            (x, _), _ = run_flow(policy, warm[num_steps - n], o, (num_steps - n) / num_steps, n, 1 / num_steps)
            return x[0]

    else:
        assert kind == "M", name

        def f(o):  # an n-step flow from the same noise
            (x, _), _ = run_flow(policy, z, o, 0.0, n, 1 / n)
            return x[0]

    return f


def action_and_jacobian(f, o):
    """f: o [O] -> a [A]. Returns (a, da/do [A, O]) from A reverse-mode VJPs, as reflex.first_action_and_jacobian."""
    a0, vjp = jax.vjp(f, o)
    (jac,) = jax.vmap(vjp)(jnp.eye(a0.shape[0], dtype=a0.dtype))
    return a0, jac


def probe_state(policy, base, params, wm, raw, obs, key, std, num_draws: int, num_steps: int):
    """One state (spec sections 4-5). Returns SUMS and e, each [P, M, K] (predictors oracle, learned; draws; chunk
    index k = 1..H-1), and valid [M, K]. SUMS are squared executed-action errors against the fresh call: res of every
    row of ROWS and nom_<row> (the row's nominal alone). base is the raw Kinetix env (no auto-reset), raw its state.
    """
    H, A_ = policy.action_chunk_size, policy.action_dim
    k_z, k_n, k_env = jax.random.split(key, 3)
    z = jax.random.normal(k_z, (H, A_))

    def true_step(s, a):
        o, s, _, done, _ = base.step_env(k_env, s, a, params)
        return o, s, done

    xs = flow_states(policy, z, obs, num_steps)
    chunk = xs[-1]
    truth, truth_ended, _ = diag.roll(true_step, raw, chunk)
    K = H - 1
    noisy = chunk + train_expert.ACTION_NOISE_STD * jax.random.normal(k_n, (num_draws, H, A_))
    real, real_ended, _ = jax.vmap(lambda a: diag.roll(true_step, raw, a))(noisy)
    nom = jnp.stack([truth, predictors.wm_rollout(wm, obs, chunk)])[:, :K]  # [P, K, O]: ô_k after k actions
    o = real[:, :K]
    zk = reflex.shifted_noise(z)[1:]  # [K, H, A]: row 0 is the noise chunk[k] came from
    warm = jax.vmap(lambda k: jnp.roll(xs, -k, axis=1))(jnp.arange(1, H))  # [K, S + 1, H, A], rolled like zk
    plan_k = chunk[1:]
    a_star = policy.action_from_noise(
        jnp.broadcast_to(zk, (num_draws, K, H, A_)).reshape(-1, H, A_), o.reshape(num_draws * K, -1), num_steps
    )[:, 0].reshape(num_draws, K, A_)
    valid = ~(real_ended[:, :K] | truth_ended[:K])

    def act(a):
        return probe.executed(a, raw)

    def per_predictor(n):
        a_ref, jac = jax.vmap(lambda zj, oj: reflex.first_action_and_jacobian(policy, zj, oj, num_steps))(zk, n)
        err = probe.errors(plan_k, a_ref, jac, n, o, a_star, act=act)
        shared = jnp.broadcast_to(jac[SHARED_K - 1], jac.shape)
        out = {
            "pred": err["pred"], "reflex": err["lin_clip"], "nom_chunk_j": err["chunk"],
            "chunk_j": err["chunk_lin_clip"],
            "shared": probe.errors(plan_k, a_ref, shared, n, o, a_star, act=act)["lin_clip"],
            "e": predictors.normalized_error(n, o, std),
        }
        for c in CANDIDATES:  # static loop: each vmap traces at once, so c is bound correctly
            a_c, jac_c = jax.vmap(
                lambda zj, wj, oj, c=c: action_and_jacobian(first_action_fn(policy, c, zj, wj, num_steps), oj)
            )(zk, warm, n)
            err_c = probe.errors(plan_k, a_c, jac_c, n, o, a_star, act=act)
            out[c], out[f"nom_{c}"] = err_c["lin_clip"], err_c["pred"]
        return jax.tree.map(lambda x: jnp.broadcast_to(x, valid.shape), out)

    return jax.vmap(per_predictor)(nom) | {"valid": valid}


def sha256(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def run(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = probe.LEVELS,
    world_model_dir: str = predictors.WM_DIR,
    num_envs: int = 64,
    num_states: int = 128,
    num_draws: int = 4,
    batch_size: int = 8,  # states per vmapped batch; lower it if RAM runs out
    seed: int = 4000,  # level probe.LEVELS[i] uses seed + i: disjoint from phase A, B1 and the diagnostic
    out_dir: str = OUT,
):
    """Spec section 4: raw arrays per level to out_dir/raw/<level>.npz (a finished level is skipped), then summarize."""
    raw_dir = pathlib.Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    wm_dir = pathlib.Path(world_model_dir)
    config = {  # saved with every level: a resumed run must not mix files of other settings or world models
        "run_path": run_path, "num_envs": num_envs, "num_states": num_states, "num_draws": num_draws,
        "num_flow_steps": S, "candidates": list(CANDIDATES), "shared_k": SHARED_K, "base_seed": seed,
        "world_models": {predictors.level_name(p): sha256(wm_dir / f"{predictors.level_name(p)}.pkl")
                         for p in probe.LEVELS},
    }
    for level_path in level_paths:  # all files before any level is computed under this config
        f = raw_dir / f"{predictors.level_name(level_path)}.npz"
        if f.exists():
            with np.load(f) as zf:
                old = diag.stored_config(zf)
            if old != config:
                raise ValueError(f"{f} was made with {old}, this run has {config}: use another --out-dir")
    env, env_params, levels, obs_dim, action_dim = probe.setup(level_paths)
    base = env._env  # raw Kinetix env: no auto-reset, as the B1 predictors
    names = np.array(["oracle", "learned"])

    @jax.jit
    def level_run(state_dict, level, key, wm):
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        k_c, k_s, k_p = jax.random.split(key, 3)
        boundaries = probe.collect(env, env_params, policy, level, k_c, num_envs, 4, train_expert.ACTION_NOISE_STD, S)
        std = predictors.alive_std(boundaries)
        obs, state = probe.sample(boundaries, k_s, num_states)

        def one(x):
            return probe_state(policy, base, env_params, wm, x[1].env_state, x[0], x[2], std, num_draws, S)

        res = jax.lax.map(one, (obs, state, jax.random.split(k_p, num_states)), batch_size=batch_size)
        return res, boundaries[2].sum()

    for i, level_path in enumerate(level_paths):
        name = predictors.level_name(level_path)
        f = raw_dir / f"{name}.npz"
        level_seed = seed + probe.LEVELS.index(level_path)  # position in the full list: a subset keeps its seeds
        if f.exists():
            print(f"{level_path}: {f} exists, skipped", flush=True)
            continue
        with (wm_dir / f"{name}.pkl").open("rb") as fh:
            wm = pickle.load(fh)
        start = time.time()
        res, n_alive = jax.device_get(level_run(
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


def level_table(res: dict, level: str) -> pd.DataFrame:
    """One row per predictor: SUMS pooled over the valid pairs with |e| <= E_MAX at k = 1..7 and their number n."""
    rows = []
    for p, name in enumerate(res["predictors"]):
        m = res["valid"] & (res["e"][:, p] <= E_MAX)  # [N, M, K]
        rows.append({"level": level, "predictor": str(name), "n": int(m.sum())}
                    | {x: float(res[x][:, p][m].sum()) for x in SUMS})
    return pd.DataFrame(rows)


def ratios(t: pd.DataFrame) -> pd.DataFrame:
    """Per level, predictor and row (spec section 5): res per pair, R = (pred - row) / (pred - reflex), the exact
    reflex's gain over pred (pred - reflex) / pred, and whether the row beats pred."""
    gain = (t["pred"] - t["reflex"]) / t["pred"]
    return pd.concat([
        t[["level", "predictor"]].assign(
            row=row, res=t[row] / t["n"], R=(t["pred"] - t[row]) / (t["pred"] - t["reflex"]), gain=gain,
            beats_pred=(t[row] <= t["pred"]) & (t["n"] > 0),  # a level without pairs beats nothing
        ) for row in ROWS[1:]
    ], ignore_index=True)


def rule(r: pd.DataFrame, predictor: str = "learned") -> pd.DataFrame:
    """Spec section 6 per row: the median R over levels with gain >= MIN_GAIN (the others listed in left_out),
    levels where the row beats pred, pass at PASS_R and strict at STRICT_R (both need MIN_LEVELS), depth and FE."""
    out = []
    for row, g in r[r["predictor"] == predictor].groupby("row", sort=False):
        kept = g[g["gain"] >= MIN_GAIN]
        med, beats = float(kept["R"].median()), int(g["beats_pred"].sum())
        out.append({"row": row, "R": med, "levels_in_median": len(kept), "beats_pred": beats,
                    "left_out": " ".join(g.loc[~(g["gain"] >= MIN_GAIN), "level"]),  # NaN gain (no pairs) too
                    "pass": bool(med >= PASS_R and beats >= MIN_LEVELS),
                    "strict": bool(med >= STRICT_R and beats >= MIN_LEVELS), "depth": depth(row), "fe": fe(row)})
    return pd.DataFrame(out).set_index("row")


def select(rt: pd.DataFrame) -> dict:
    """Spec section 6: the candidates for the B2+B5 GPU run (the reference rows go anyway). Order: depth, then FE,
    then the larger R (W_k and M_k tie on both)."""
    c = rt.loc[list(CANDIDATES)].sort_values(["depth", "fe", "R"], ascending=[True, True, False])
    passed = c[c["pass"]]
    if passed.empty:
        R = c["R"].dropna()
        return {"passed": [], "gpu": [str(R.idxmax())] if len(R) else [], "fallback": True}
    gpu = [passed.index[0]]  # 1: the shallowest pass
    t = passed[passed.index.str.startswith("T")]
    if len(t):
        gpu.append(t["R"].idxmax())  # 2: the best passing T_k
    strict = c[c["strict"]]
    if len(strict):
        gpu.append(strict.index[0])  # 3: the shallowest at R >= 0.9
    return {"passed": [str(x) for x in passed.index], "gpu": [str(x) for x in dict.fromkeys(gpu)], "fallback": False}


def by_k(raws: dict) -> pd.DataFrame:
    """Descriptive, for the memo: per chunk index k, the median over levels of R (levels with gain >= MIN_GAIN, as in
    rule) and of res per pair, with the number of levels in each median."""
    r = pd.concat([ratios(pd.concat([
        level_table({x: res[x][..., k - 1:k] for x in (*SUMS, "e", "valid")} | {"predictors": res["predictors"]}, lv)
        for lv, res in raws.items()], ignore_index=True)).assign(k=k)
        for k in range(1, next(iter(raws.values()))["valid"].shape[-1] + 1)], ignore_index=True)
    keys = ["predictor", "row", "k"]
    R = r[r["gain"] >= MIN_GAIN].groupby(keys, sort=False)["R"].agg(R="median", levels_in_median="count")
    res = r.groupby(keys, sort=False)["res"].agg(res="median", levels="count")
    return res.join(R).fillna({"levels_in_median": 0}).astype({"levels_in_median": int}).reset_index()


def _rho(res, nom) -> float:
    return float(1 - res / nom) if nom > 0 else float("nan")


def bins(raws: dict, edges, p: int) -> pd.DataFrame:
    """Spec section 5 for predictor index p: res per pair of every row in |e| bins (median over levels with
    >= diag.MIN_BIN pairs in the bin) and its rho_clip against its own nominal. The last bin is open."""
    out = []
    for b in range(len(edges) - 1):
        lo, hi = edges[b], edges[b + 1]
        per = []
        for res in raws.values():
            e = res["e"][:, p]
            m = res["valid"] & (e >= lo) & ((e < hi) if b < len(edges) - 2 else (e <= hi))
            if m.sum() >= diag.MIN_BIN:
                per.append({row: (res[row][:, p][m].mean(),
                                  _rho(res[row][:, p][m].sum(), res[NOM[row]][:, p][m].sum()) if row in NOM else np.nan)
                            for row in ROWS})
        for row in ROWS:
            rho = [x[row][1] for x in per]  # NaN where a level's nominal sum in the bin is 0: skipped
            out.append({"bin": b, "e_lo": lo, "e_hi": hi, "row": row, "levels": len(per),
                        "res": float(np.median([x[row][0] for x in per])) if per else np.nan,
                        "rho_clip": float(np.nanmedian(rho)) if np.isfinite(rho).any() else np.nan})
    return pd.DataFrame(out)


def _style(row: str) -> dict:
    """pred black dashed, reflex black bold, chunk_j and shared grey; T solid, W dash-dot, M dotted, colour per k."""
    if row in ("pred", "reflex"):
        return {"color": "k", "ls": "--" if row == "pred" else "-", "lw": 2}
    if row in ("chunk_j", "shared"):
        return {"color": "0.6", "ls": "-" if row == "chunk_j" else "--", "lw": 1.5}
    return {"color": f"C{int(row[1:]) - 1}", "ls": {"T": "-", "W": "-.", "M": ":"}[row[0]], "lw": 1.5}


def figure(bn: pd.DataFrame, path: pathlib.Path):
    """Residual against the fresh call per |e| bin, one panel per predictor; pred dashed, the exact reflex bold."""
    preds = list(dict.fromkeys(bn["predictor"]))
    fig, axes = plt.subplots(1, len(preds), figsize=(6 * len(preds), 4.5), sharey=True, squeeze=False)
    for ax, pred in zip(axes[0], preds):
        d = bn[bn["predictor"] == pred]
        for row, g in d.groupby("row", sort=False):
            ax.plot(g["bin"], g["res"], label=row, **_style(row))
        g = d[d["row"] == "pred"]
        ax.set_xticks(g["bin"], [f"{x:.2g}" for x in g["e_lo"]], rotation=45)
        ax.set(title=pred, xlabel="|e| bin, lower edge", yscale="log")
    axes[0][0].set_ylabel("residual vs fresh call, per pair")
    axes[0][-1].legend(ncol=2, fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def summarize(out_dir: str = OUT):
    """Tables, the rule, the selection and the figure from out_dir/raw/*.npz (spec sections 5-6)."""
    out = pathlib.Path(out_dir)
    raws = {f.stem: dict(np.load(f)) for f in sorted((out / "raw").glob("*.npz"))}
    configs = {json.dumps(diag.stored_config(r), sort_keys=True) for r in raws.values()}  # no config ("null") counts
    assert len(configs) == 1, f"raw files of different runs in {out / 'raw'}: {configs}"
    names = [str(x) for x in next(iter(raws.values()))["predictors"]]
    t = pd.concat([level_table(r, lv) for lv, r in raws.items()], ignore_index=True)
    r = ratios(t)
    rt = pd.concat({p: rule(r, p) for p in names}, names=["predictor"])
    sel = select(rule(r, "learned")) | {"levels": len(raws)}
    d = pd.read_csv(DIAG_BINS)
    edges = [*d.loc[d["predictor"] == "all", "e_lo"], np.inf]
    bn = pd.concat([bins(raws, edges, p).assign(predictor=n) for p, n in enumerate(names)], ignore_index=True)
    rho = []  # sanity (spec section 8): the oracle's exact reflex at k = 1..4, as in E1b and the diagnostic
    for res in raws.values():
        m = res["valid"][..., :4]
        rho.append(_rho(res["reflex"][:, 0][..., :4][m].sum(), res["pred"][:, 0][..., :4][m].sum()))
    check = {"oracle_reflex_rho_clip_k1_4": float(np.median(rho)), "diag": 0.578}
    check["ok"] = abs(check["oracle_reflex_rho_clip_k1_4"] - check["diag"]) <= 0.1
    t.to_csv(out / "summary.csv", index=False)
    r.to_csv(out / "ratios.csv", index=False)
    rt.to_csv(out / "rule.csv")
    bn.to_csv(out / "bins.csv", index=False)
    by_k(raws).to_csv(out / "by_k.csv", index=False)
    (out / "selection.json").write_text(json.dumps(sel, indent=2))
    (out / "check.json").write_text(json.dumps(check, indent=2))
    figure(bn, out / "b2.png")
    print(rt.round(3).to_string())
    print(json.dumps(sel))
    if len(raws) != len(probe.LEVELS):
        print(f"!!! rule applied to {len(raws)} levels, not {len(probe.LEVELS)}")
    print(json.dumps(check))
    if not check["ok"]:
        print("!!! sanity check failed: look for a bug before reading anything")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"run": run, "summarize": summarize})
