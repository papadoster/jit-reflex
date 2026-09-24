"""Figures and tables for E1 (probe) and E2 (closed loop), and the Gate 2 check (spec section 5)."""

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


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"probe": probe, "success": success, "table": table, "gate2": gate2})
