"""Figures and tables for E1 (probe), E2 (closed loop) and B1, and the Gate 2 / B1 decision rules."""

import glob
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tyro  # noqa: E402

import reflex  # noqa: E402


def probe(summary_csv: str = "results/probe/summary.csv", out: str = "results/probe/rho.png"):
    """rho(k) per level, one line per sigma (the verdict's sigma bold), verdict in the title."""
    df = pd.read_csv(summary_csv)
    v = json.loads((pathlib.Path(summary_csv).parent / "verdict.json").read_text())
    on = v.get("on", "lin")  # the oldest verdict.json has no "on": it was computed on "lin"
    col = "rho_clip" if on == "lin_clip" else "rho"
    levels = sorted(df["level"].unique())
    cols = 4
    rows = math.ceil(len(levels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, level in zip(axes.flat, levels):
        for sigma, g in df[df["level"] == level].groupby("sigma"):
            ax.plot(g["k"], g[col], marker="o", label=f"σ={sigma}", lw=2.5 if np.isclose(sigma, v["sigma"]) else 1)
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.axhline(0.2, ls=":", c="gray", lw=0.8)
        ax.set_title(pathlib.Path(level).stem)
        ax.set_ylim(-0.5, 1.05)
    axes.flat[0].legend(fontsize=7)
    fig.suptitle(
        f"E1: ρ(k) = 1 − e_lin/e_pred · {v['verdict']} (on={on}, R={v['R']:.2f}, σ={v['sigma']}, "
        f"ok {v['levels_ok']}/{v['levels']})"
    )
    fig.supxlabel("k (steps after the policy call)")
    fig.supylabel(col)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def wilson(p, n, z=1.96):
    """95% Wilson interval for proportions p over n trials (arrays)."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def success(
    results_glob: str = "results/eval/main/results.csv",
    oracle_glob: str = "results/eval/oracle/results.csv",
    num_evals: int = 256,
    out: str = "results/eval/main/success.png",
):
    """Solve rate (mean over levels and seeds) vs delay, for s = d and s = 8 - d; oracle d=0, s=1 dashed."""
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(results_glob))])
    df["horizon"] = np.where(df["execute_horizon"] == df["delay"].clip(lower=1), "s = d", "s = 8 − d")
    g = df.groupby(["horizon", "method", "delay"])["returned_episode_solved"].agg(["mean", "count"]).reset_index()
    oracle_files = sorted(glob.glob(oracle_glob))
    oracle = pd.concat([pd.read_csv(f) for f in oracle_files])["returned_episode_solved"].mean() if oracle_files else None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (horizon, h) in zip(axes, g.groupby("horizon")):
        for method, m in h.groupby("method"):
            lo, hi = wilson(m["mean"].to_numpy(), m["count"].to_numpy() * num_evals)
            ax.errorbar(m["delay"], m["mean"], yerr=[m["mean"] - lo, hi - m["mean"]], marker="o", capsize=3, label=method)
        if oracle is not None:
            ax.axhline(oracle, ls="--", c="black", lw=0.8, label="oracle d=0 s=1")
        ax.set_title(horizon)
        ax.set_xlabel("inference delay d")
    axes[0].set_ylabel("solve rate (mean over levels)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(g.pivot_table(index=["horizon", "delay"], columns="method", values="mean").round(3).to_string())
    calls = df.drop_duplicates(["delay", "execute_horizon"])[["delay", "execute_horizon"]].copy()
    calls["calls_per_episode"] = np.ceil(256 / calls["execute_horizon"]).astype(int)
    print(calls.sort_values(["delay", "execute_horizon"]).to_string(index=False))
    # work = calls x network evaluations per call: fewer calls must not hide more compute
    df["work_per_episode"] = [
        reflex.forward_equivalents(m) * math.ceil(256 / s) for m, s in zip(df["method"], df["execute_horizon"])
    ]
    print("network evaluations per episode:")
    print(df.pivot_table(index=["delay", "execute_horizon"], columns="method", values="work_per_episode").to_string())
    print(f"saved {out}")


def table(results_glob: str):
    """Mean solve rate and solved-episode length over levels and seeds per (run dir, method, delay, horizon)."""
    df = pd.concat([pd.read_csv(f).assign(run=pathlib.Path(f).parent.name) for f in sorted(glob.glob(results_glob))])
    cols = [c for c in ("returned_episode_solved", "solved_length") if c in df.columns]
    print(df.groupby(["run", "method", "delay", "execute_horizon"])[cols].mean().round(3).to_string())


def gate2(
    main_csv: str = "results/eval/main/results.csv",
    kick_glob: str = "results/eval/kick*[0-9]/results.csv",
) -> dict:
    """Spec Gate 2: conditions (a), (b), (c) and the pred check. Solve rates are means over levels."""
    df = pd.read_csv(main_csv)
    t = df.pivot_table(
        index=["seed", "delay", "execute_horizon"], columns="method", values="returned_episode_solved"
    ).reset_index()
    long = t[(t["delay"] >= 2) & (t["execute_horizon"] == 8 - t["delay"])]
    a_by_seed = (long["reflex"] - long["realtime"]).groupby(long["seed"]).mean()
    a = bool(a_by_seed.mean() >= 0.05 and (a_by_seed > 0).all())
    short = t[t["execute_horizon"] == t["delay"].clip(lower=1)].set_index(["seed", "delay"])["realtime"]
    far = t[t["delay"].between(1, 3) & (t["execute_horizon"] == 8 - t["delay"])].set_index(["seed", "delay"])["reflex"]
    b_gap = float((far - short.reindex(far.index)).mean())
    # (b) counts calls; report the work next to it (spec: fewer calls must not hide more compute)
    calls_saved = float(np.mean([math.ceil(256 / max(1, d)) / math.ceil(256 / (8 - d)) for d in (1, 2, 3)]))
    work_ratio = float(np.mean([
        reflex.forward_equivalents("reflex") * math.ceil(256 / (8 - d))
        / (reflex.forward_equivalents("realtime") * math.ceil(256 / max(1, d)))
        for d in (1, 2, 3)
    ]))
    total_gain = float((t["reflex"] - t["naive"]).mean())
    j_gain = float((t["reflex"] - t["pred"]).mean())
    pred_check = bool(total_gain > 0 and j_gain >= 0.5 * total_gain)
    c_gap = None  # no kick runs; None keeps gate2.json valid JSON
    kick_files = sorted(glob.glob(kick_glob))
    if kick_files:
        k = pd.concat([pd.read_csv(f) for f in kick_files]).pivot_table(
            index=["kick_std", "seed", "execute_horizon"], columns="method", values="returned_episode_solved"
        )
        c_gap = float((k["reflex"] - k[["naive", "realtime"]].max(axis=1)).mean())
    out = {
        "a": a, "a_gap": float(a_by_seed.mean()),
        "b": bool(b_gap >= -0.02), "b_gap": b_gap, "b_calls_saved": calls_saved, "b_work_ratio": work_ratio,
        "c": bool(c_gap is not None and c_gap >= 0.05), "c_gap": c_gap,
        "pred_check": pred_check, "total_gain": total_gain, "j_gain": j_gain,
    }
    if out["b"]:
        out["b_claim"] = f"{calls_saved:.1f}x fewer policy calls at {work_ratio:.1f}x the network evaluations of realtime"
    out["decision"] = "GO" if (out["a"] or out["b"] or out["c"]) and pred_check else "NEGATIVE"
    print(json.dumps(out, indent=2))
    return out


B1_SLICES = {"d1": (1, (5, 6, 7)), "d3": (3, (5,))}  # spec B1 section 5: the rare-call slices behind the verdict
B1_SEEDS = (10, 11, 12)  # spec B1 section 4.4


def _slice_status(G: pd.DataFrame, delay: int, horizons, predictor: str, seeds):
    """PASS / FAIL / GRAY / MISSING of one slice for one predictor (spec B1 section 5), with per-method details.

    G: reflex-type methods' G, index (delay, seed, predictor, execute_horizon). A method counts only with every
    (seed, s) cell of the slice. PASS: some complete method has pooled >= +1 pp and > 0 in every seed. FAIL: reflex
    and rtc_reflex both complete with pooled <= 0. Otherwise an incomplete method makes the slice MISSING, never FAIL.
    """
    try:
        g = G.xs((delay, predictor), level=("delay", "predictor"))
    except KeyError:
        return "MISSING", {}
    full = pd.MultiIndex.from_product([seeds, horizons], names=["seed", "execute_horizon"])
    g = g.reindex(index=full, columns=["reflex", "rtc_reflex"])
    detail = {}
    for m in g:
        if g[m].notna().all():
            per_seed = g[m].groupby(level="seed").mean()
            detail[m] = {"pooled": float(per_seed.mean()), "min_seed": float(per_seed.min())}
    if any(v["pooled"] >= 0.01 - 1e-9 and v["min_seed"] > 0 for v in detail.values()):
        return "PASS", detail
    if len(detail) < 2:
        return "MISSING", detail
    if all(v["pooled"] <= 0 for v in detail.values()):
        return "FAIL", detail
    return "GRAY", detail


def _missing_cells(lv: pd.Series, phys, p_mid: str):
    """Cells of the spec B1 4.4 GPU grid with no solve rate, as (delay, seed, method, predictor, s)."""
    d1, mid = ["oracle", *phys, "learned"], ["oracle", p_mid, "learned"]
    cells = [(1, s, m, "-") for s in range(1, 8) for m in ("naive", "realtime")]
    cells += [(1, s, m, pr) for s in range(1, 8) for m in ("pred", "reflex") for pr in d1]
    cells += [(1, s, "rtc_reflex", pr) for s in range(1, 8) for pr in mid]
    cells += [(3, 5, m, "-") for m in ("naive", "realtime")]
    cells += [(3, 5, m, pr) for m in ("pred", "reflex", "rtc_reflex") for pr in mid]
    idx = pd.MultiIndex.from_tuples(
        [(d, seed, m, pr, s) for d, s, m, pr in cells for seed in B1_SEEDS], names=lv.index.names
    )
    got = lv.reindex(idx)
    return got[got.isna()].index.tolist()


def b1(
    results_glob: str = "results/b1/gpu/eval*/results.csv",
    errors_csv: str = "results/b1/gpu/errors.csv",
    out_dir: str = "results/b1/gpu",
    p_mid: str | None = None,
    strict: bool = True,
) -> dict:
    """B1 verdict and rules R1-R4 (spec B1 section 5), b1.json and b1.png. Solve rates are means over levels.

    G = method - max(naive, realtime) at the same (delay, seed, s); J = reflex - pred at d = 1. p_mid: the middle phys
    level, pass it explicitly (e.g. "phys0.4" after calibration); None = the middle of an odd number of phys levels.
    Refuses a grid where naive / realtime are missing next to a reflex-type row. strict: also refuse any hole in the
    spec B1 4.4 GPU grid (seeds 10-12); --no-strict is for the rehearsal's tiny grid, where holes give MISSING slices.
    A MISSING slice never turns into a verdict: SURVIVES / ENOUGH still stand on a PASS, anything else is INCOMPLETE.
    """
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(results_glob))])
    lv = df.groupby(["delay", "seed", "method", "predictor", "execute_horizon"])["returned_episode_solved"].mean()
    base = lv.xs("-", level="predictor").unstack("method")  # (delay, seed, s) x {naive, realtime}
    t = lv.drop("-", level="predictor").unstack("method")  # (delay, seed, predictor, s) x {pred, reflex, rtc_reflex}
    need = base[["naive", "realtime"]].reindex(t.index.droplevel("predictor").unique())
    assert need.notna().all().all(), "incomplete baseline: naive/realtime missing for some (delay, seed, s)"
    phys = sorted(
        (p for p in lv.index.get_level_values("predictor").unique() if p.startswith("phys")), key=lambda p: float(p[4:])
    )
    if p_mid is None:
        assert len(phys) % 2 == 1, f"p_mid is ambiguous for phys levels {phys}: pass --p-mid"
        p_mid = phys[len(phys) // 2]
    assert p_mid in phys, f"p_mid {p_mid} not among phys levels {phys}"
    if strict:  # a lost run must stop here, not become a verdict
        missing = _missing_cells(lv, phys, p_mid)
        assert not missing, (
            f"incomplete B1 grid: {len(missing)} missing (delay, seed, method, predictor, s) cells, "
            f"e.g. {missing[:10]}; rerun them or pass --no-strict"
        )
    seeds = sorted(df["seed"].unique())
    best = base[["naive", "realtime"]].max(axis=1).reindex(t.index.droplevel("predictor")).to_numpy()
    G = t.sub(pd.Series(best, index=t.index), axis=0)
    t1 = t.xs(1, level="delay")
    J = t1["reflex"] - t1["pred"]  # (seed, predictor, s)
    jo = J.xs("oracle", level="predictor").unstack("execute_horizon")  # seed x s
    stale = jo.loc[:, jo.columns >= 5].mean(axis=1) - jo.loc[:, jo.columns <= 3].mean(axis=1)
    jbar = J.groupby(level=["predictor", "seed"]).mean()
    grow = jbar[p_mid] - jbar["oracle"]

    out = {"p_mid": p_mid, "slices": {}}
    for name, (d, horizons) in B1_SLICES.items():
        out["slices"][name] = {}
        for pr in ("oracle", p_mid, "learned"):
            status, detail = _slice_status(G, d, horizons, pr, seeds)
            out["slices"][name][pr] = {"status": status, **detail}
    informative = [n for n in B1_SLICES if out["slices"][n]["oracle"]["status"] == "PASS"]
    oracle_missing = any(out["slices"][n]["oracle"]["status"] == "MISSING" for n in B1_SLICES)

    def rule(pr, passed, failed, no_edge):  # R1 / R4; missing data gives INCOMPLETE, never a negative outcome
        st = [out["slices"][n][pr]["status"] for n in informative]
        if "PASS" in st:
            return passed
        if oracle_missing or "MISSING" in st:
            return "INCOMPLETE"
        if not informative:
            return no_edge
        return failed if all(x == "FAIL" for x in st) else "GRAY"

    out["informative_slices"] = informative
    out["verdict"] = rule(p_mid, "SURVIVES", "ORACLE-BOUND", "NO-EDGE")
    out["R4_learned"] = rule("learned", "ENOUGH", "TOO-WEAK", "n/a")
    out |= {
        "R2_staleness": bool(stale.mean() >= 0.03 - 1e-9 and (stale > 0).all()),
        "R2_gap": float(stale.mean()),
        "R3_j_grows_with_error": bool(grow.mean() >= 0.01 - 1e-9 and (grow > 0).all()),
        "R3_gap": float(grow.mean()),
    }
    out["drop_oracle_to_p_mid"] = {  # report only: the spec's prediction is that rtc_reflex drops more than reflex
        n: {
            m: out["slices"][n]["oracle"][m]["pooled"] - out["slices"][n][p_mid][m]["pooled"]
            for m in ("reflex", "rtc_reflex")
            if m in out["slices"][n]["oracle"] and m in out["slices"][n][p_mid]
        }
        for n in B1_SLICES
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for pr, g in J.groupby(level="predictor"):
        g = g.groupby(level="execute_horizon").mean()
        axes[0].plot(g.index, g * 100, marker="o", label=pr)
    g1 = G.xs(1, level="delay").groupby(level=["predictor", "execute_horizon"]).mean()
    for pr, g in g1.groupby(level="predictor"):
        s = g.index.get_level_values("execute_horizon")
        line = axes[1].plot(s, g["reflex"] * 100, marker="o", label=pr)[0]
        if "rtc_reflex" in g and g["rtc_reflex"].notna().any():
            axes[1].plot(s, g["rtc_reflex"] * 100, ls="--", c=line.get_color())
    axes[0].set_title("d = 1: J = reflex − pred (pp)")
    axes[1].set_title("d = 1: G = method − max(naive, RTC) (pp); dashed: rtc_reflex")
    for ax in axes[:2]:
        ax.axhline(0, c="gray", lw=0.8)
        ax.set_xlabel("execute horizon s")
    axes[0].legend(fontsize=8)
    if pathlib.Path(errors_csv).exists():
        e = pd.read_csv(errors_csv)
        ratio = e[e["k"] == 4].groupby("predictor")["ratio"].median()
        jb = J.groupby(level="predictor").mean()
        common = [p for p in jb.index if p in ratio.index]
        axes[2].scatter(ratio[common], jb[common] * 100)
        for p in common:
            axes[2].annotate(p, (ratio[p], jb[p] * 100), fontsize=8)
        axes[2].set_xlabel("prediction error at k = 4 (units of action-noise deviation)")
    axes[2].set_title("d = 1: mean J over s vs predictor error (pp)")
    fig.suptitle(f"B1: {out['verdict']}")
    fig.tight_layout()
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "b1.png", dpi=150)
    plt.close(fig)
    print((G.groupby(level=["delay", "predictor", "execute_horizon"]).mean() * 100).round(1).to_string())
    print("network evaluations per step:", {
        m: [round(reflex.forward_equivalents(m, positions=s) / s, 1) for s in range(1, 8)]
        for m in ("naive", "realtime", "pred", "reflex", "rtc_reflex")
    })
    (out_path / "b1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out


def _boot(x, n_boot: int, rng) -> tuple[float, float, float]:
    """Mean and 95% percentile bootstrap interval of per-cell values x [cells]."""
    x = np.asarray(x, float)
    means = x[rng.integers(0, len(x), (n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def b1_ci(
    results_glob: str = "results/b1/gpu/eval*/results.csv", out_dir: str = "results/b1/gpu", n_boot: int = 10_000
) -> pd.DataFrame:
    """Report only, not a decision rule: B1 means (pp) with 95% bootstrap intervals over level x seed cells.

    Episodes of one (level, seed) cell share the level and the rng, so cells, not episodes, are resampled.
    G is taken against the baseline (naive or realtime) with the higher pooled mean, as in b1. Writes b1_ci.csv/png.
    """
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(results_glob))])
    df["arm"] = df["method"] + ":" + df["predictor"]
    w = df.pivot_table(index=["delay", "execute_horizon", "level", "seed"], columns="arm", values="returned_episode_solved")
    rng = np.random.default_rng(0)
    rows = []

    def add(what, where, x):
        if x.notna().all():
            mean, lo, hi = _boot(x, n_boot, rng)
            rows.append({"what": what, "where": where, "mean": 100 * mean, "lo": 100 * lo, "hi": 100 * hi, "cells": len(x)})

    def gains(g, where):  # g: cells x arms of one slice or one (delay, s)
        base = g[g[["naive:-", "realtime:-"]].mean().idxmax()]
        for arm in g.columns:
            if not arm.endswith(":-"):
                add(f"G {arm}", where, g[arm] - base)
        for pr in sorted({a.split(":")[1] for a in g.columns if a.startswith("reflex:")}):
            add(f"J {pr}", where, g[f"reflex:{pr}"] - g[f"pred:{pr}"])
        for m in ("pred", "reflex", "rtc_reflex"):  # learned world model vs oracle
            if f"{m}:learned" in g and f"{m}:oracle" in g:
                add(f"{m}: learned - oracle", where, g[f"{m}:learned"] - g[f"{m}:oracle"])

    cells = ["level", "seed"]
    d1 = w.xs(1, level="delay")
    for s, g in d1.groupby(level="execute_horizon"):
        gains(g.droplevel("execute_horizon"), f"d1 s{s}")
    for name, (d, horizons) in B1_SLICES.items():
        g = w.xs(d, level="delay")
        g = g[g.index.get_level_values("execute_horizon").isin(horizons)].groupby(level=cells).mean()
        gains(g, f"slice {name}")
    out = pd.DataFrame(rows)
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_path / "b1_ci.csv", index=False)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    per_s = out[out["where"].str.startswith("d1 s")].assign(s=lambda x: x["where"].str[4:].astype(int))
    for ax, prefix, preds in (
        (axes[0], "J ", ("oracle", "phys0.1", "phys0.2", "phys0.3", "learned")),
        (axes[1], "G ", ("reflex:oracle", "reflex:phys0.2", "reflex:learned", "rtc_reflex:oracle", "rtc_reflex:learned")),
    ):
        for p in preds:
            r = per_s[per_s["what"] == prefix + p].sort_values("s")
            if len(r):
                ax.errorbar(r["s"], r["mean"], yerr=[r["mean"] - r["lo"], r["hi"] - r["mean"]], marker="o", capsize=3,
                            ls="--" if p.startswith("rtc") else "-", label=p)
        ax.axhline(0, c="gray", lw=0.8)
        ax.set_xlabel("execute horizon s (d = 1)")
        ax.legend(fontsize=7)
    axes[0].set_title("J = reflex − pred (pp), 95% bootstrap over level × seed")
    axes[1].set_title("G = method − best of naive / RTC (pp)")
    fig.tight_layout()
    fig.savefig(out_path / "b1_ci.png", dpi=150)
    plt.close(fig)
    print(out.round(1).to_string(index=False))
    return out


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict(
        {"probe": probe, "success": success, "table": table, "gate2": gate2, "b1": b1, "b1-ci": b1_ci}
    )
