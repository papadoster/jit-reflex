"""A2C2 fix rerun (spec docs/superpowers/specs/2026-09-28-a2c2-fix-design.md): the experts fixed before data, the
head's latency, the zero-head check and the report-only summary next to the B2+B5 results."""

import glob
import json
import pathlib
import pickle
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
import tyro

import a2c2
import b2b5
import eval_flow
import plot
import predictors
import probe

OUT = "results/b2b5_fix"
METHOD = "a2c2_paper"
# spec §2: per level the best checkpoint of all seeds and steps by returned_episode_solved in rtc-assets stats/
# (ties: lower seed, then lower step), read before any data of this run
EXPERTS = {
    "worlds/l/grasp_easy.json": (6, 700),
    "worlds/l/catapult.json": (3, 980),
    "worlds/l/cartpole_thrust.json": (1, 700),
    "worlds/l/hard_lunar_lander.json": (6, 540),
    "worlds/l/mjc_half_cheetah.json": (6, 940),
    "worlds/l/mjc_swimmer.json": (7, 900),
    "worlds/l/mjc_walker.json": (7, 580),
    "worlds/l/h17_unicycle.json": (0, 700),
    "worlds/l/chain_lander.json": (2, 840),
    "worlds/l/catcher_v3.json": (4, 380),
    "worlds/l/trampoline.json": (2, 840),
    "worlds/l/car_launch.json": (0, 920),
}
DEV = ("worlds/l/trampoline.json", "worlds/l/mjc_walker.json", "worlds/l/car_launch.json")  # spec §4: chosen on these


def expert_path(level_path: str) -> str:
    s, step = EXPERTS[level_path]
    return f"checkpoints/expert/seed_{s}_step_{step}_{predictors.level_name(level_path)}.pkl"


def expert_url(level_path: str) -> str:
    s, step = EXPERTS[level_path]
    return f"https://storage.googleapis.com/rtc-assets/expert/seed_{s}/{step}/policies/{predictors.level_name(level_path)}.pkl"


def experts():
    """One line per level for the pod script: <level short name> <local path> <url>."""
    for lp in probe.LEVELS:
        print(lp.split("/")[-1][:-5], expert_path(lp), expert_url(lp))


def _paper_head(obs_dim: int, action_dim: int):
    return a2c2.Head(a2c2.HISTORY * obs_dim + action_dim, action_dim, **a2c2.PAPER_HEAD, rngs=nnx.Rngs(0))


def latency(out_dir: str = OUT, level_path: str = probe.LEVELS[0], repeats: int = 200, warmup: int = 20):
    """Spec §3: the a2c2_paper head's ms at batch 1 on the default device and on the CPU (B2+B5 §5.1: the head runs on
    the CPU next to the GPU; if it misses the tact, d' grows by 1), and its GFLOP from a CPU copy (b2b5.cpu_gflop:
    the GPU count misses matmuls). Writes head_latency.json."""
    _, _, _, O, A = probe.setup([level_path])
    OH, cpu = a2c2.HISTORY * O + A, jax.devices("cpu")[0]
    args = (jnp.zeros((1, OH)), jnp.zeros((1, A)), a2c2.time_feature(jnp.zeros(1, jnp.int32), a2c2.H))
    head = _paper_head(O, A)
    h = jax.jit(lambda o, a, t: head.apply_residual(o, a, t))
    with jax.default_device(cpu):
        head_cpu = _paper_head(O, A)
        h_cpu = jax.jit(lambda o, a, t: head_cpu.apply_residual(o, a, t))
        cargs = jax.device_put(args, cpu)
        gflop = b2b5._flops(h_cpu, *cargs) / 1e9
    info = {"device": str(jax.devices()[0]), "head": METHOD, "in_dim": OH + A + 2,
            "head_ms_gpu": b2b5._time(h, args, repeats, warmup), "head_ms_cpu": b2b5._time(h_cpu, cargs, repeats, warmup),
            "head_gflop": gflop}
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "head_latency.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def zero_head_check(out_dir: str = OUT, level_path: str = "worlds/l/trampoline.json", num_evals: int = 8,
                    seed: int = 20):
    """Spec §6: a2c2_paper with a zero last layer must give naive bit for bit. The history wrappers pass keys and
    action noise through unchanged and the policy sees the current frame, so any difference is a bug."""
    _, _, _, O, A = probe.setup([level_path])
    head = _paper_head(O, A)
    last = head.residual_policy.layers[-1]
    last.kernel.value = jnp.zeros_like(last.kernel.value)
    last.bias.value = jnp.zeros_like(last.bias.value)
    root = pathlib.Path(out_dir) / "zero_head"
    (root / METHOD).mkdir(parents=True, exist_ok=True)
    with (root / METHOD / f"{predictors.level_name(level_path)}.pkl").open("wb") as f:
        pickle.dump(nnx.state(head).to_pure_dict(), f)
    eval_flow.main(config=eval_flow.EvalConfig(num_evals=num_evals), level_paths=[level_path], seeds=[seed],
                   methods=["naive", METHOD], cells=["1,5"], output_dir=str(root / "eval"),
                   run_path="checkpoints/bc", heads_root=str(root))
    d = pd.read_csv(root / "eval" / "results.csv").set_index("method")
    cols = ["returned_episode_solved", "returned_episode_returns", "returned_episode_lengths"]
    same = bool((d.loc["naive", cols] == d.loc[METHOD, cols]).all())
    line = f"zero-head {METHOD} == naive: {same}\n{d[cols].to_string()}"
    (pathlib.Path(out_dir) / "zero_head.txt").write_text(line + "\n")
    print(line)
    assert same, "a2c2_paper with a zero head differs from naive: look for a bug before any run"


ROWS = {  # spec §4: rival -> the flag its row carries
    ("t3", "learned"): "", ("realtime", "-"): "", ("a2c2_distill", "-"): "неравное по голове",
    ("a2c2", "-"): "старый вариант", ("naive", "-"): "",
}
SLICES = {"D1": b2b5.D1, "(2,2)": [(2, 2)], "(2,6)": [(2, 6)], "(3,3)": [(3, 3)], "(3,4)": [(3, 4)],
          "D3": [b2b5.D3], "D4": [b2b5.D4]}


def level_sets() -> dict:
    return {"all 12": list(probe.LEVELS), "held-out 9": [lv for lv in probe.LEVELS if lv not in DEV],
            "development 3": list(DEV)}


def cells_of(df: pd.DataFrame) -> pd.Series:
    """returned_episode_solved per (method, predictor, delay, execute_horizon, seed, level)."""
    return df.set_index(["method", "predictor", "delay", "execute_horizon", "seed", "level"])[
        "returned_episode_solved"].sort_index()


def diff(cell: pd.Series, a, b, cells, levels) -> dict | None:
    """a - b, averaged over `cells`, on `levels`: pooled pp, per-seed pp, 95% bootstrap over level x seed (as B2+B5).
    None if a and b share no (seed, level)."""
    try:
        x = pd.concat([cell.xs((*a, *c)) - cell.xs((*b, *c)) for c in cells], axis=1).mean(axis=1, skipna=False)
    except KeyError:
        return None
    x = x[x.index.get_level_values("level").isin(levels)].dropna()
    if x.empty:
        return None
    m, lo, hi = plot._boot(x.to_numpy(), 10_000, np.random.default_rng(0))
    return {"pooled_pp": 100 * m, "per_seed_pp": [round(100 * v, 6) for v in x.groupby(level="seed").mean()],
            "ci95_pp": [100 * lo, 100 * hi], "levels": int(x.index.get_level_values("level").nunique())}


def head_cells(lat: pd.DataFrame, info: dict, head: dict) -> dict:
    """Spec §3 (B2+B5 §5.1): a2c2_paper's latency-fair cells per base d: naive's d' = ceil(r_naive d), plus 1 when the
    head's CPU time exceeds the tact realtime_ms / d; none when d' > 4."""
    r = float((lat[lat.method == "naive"]["ms"] / info["realtime_ms"]).median())
    out = {}
    for base in (1, 2, 3, 4):
        n = b2b5._ceil(r * base) + int(head["head_ms_cpu"] > info["realtime_ms"] / base)
        out[base] = [(n, s) for s in range(n, 9 - n)] if n <= 4 else []
    return out


def summarize(new_dir: str = OUT, old_dir: str = b2b5.OUT, strict: bool = True):
    """Spec §4, report only: fix.json and fix.txt in new_dir, levels.csv (per-level P at D1, D3, D4)."""
    new, old = pathlib.Path(new_dir), pathlib.Path(old_dir)
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(f"{old}/eval_*/results.csv"))]
                   + [pd.read_csv(new / "eval" / "results.csv")])
    df = df[df.seed.isin(b2b5.SEEDS)].drop_duplicates(["seed", "delay", "execute_horizon", "method", "predictor",
                                                        "level"])
    cell = cells_of(df)
    have = set(zip(df.method, df.delay, df.execute_horizon, df.seed, df.level))
    missing = [(d, s, sd, lv) for d, s in b2b5.ALL16 for sd in b2b5.SEEDS for lv in probe.LEVELS
               if (METHOD, d, s, sd, lv) not in have]
    if missing and strict:
        raise SystemExit(f"{len(missing)} (cell, seed, level) of {METHOD} missing, rerun the eval: {missing[:5]}")
    a = (METHOD, "-")
    rows = {f"{METHOD} − {b[0]}" + (f" ({flag})" if flag else ""): {
        ls: {sl: diff(cell, a, b, cs, lvs) for sl, cs in SLICES.items()} for ls, lvs in level_sets().items()}
        for b, flag in ROWS.items()}
    P = df.groupby(["method", "predictor", "delay", "execute_horizon", "level"])["returned_episode_solved"].mean()

    def per_level(m, p, cs):  # P per level averaged over cells cs; empty if any cell is missing (non-strict runs)
        try:
            return sum(P.xs((m, p, *c)) for c in cs) / len(cs)
        except KeyError:
            return pd.Series(dtype=float)

    lv_rows = []
    for sl in ("D1", "D3", "D4"):
        t = pd.DataFrame({m: per_level(m, p, SLICES[sl]) for m, p in [a, *ROWS]})
        lv_rows.append(t.assign(slice=sl, development=t.index.isin(DEV)))
    levels = pd.concat(lv_rows)
    pooled = {(m, sl): float(per_level(m, p, SLICES[sl]).mean()) for m, p in [a, ("realtime", "-"), ("a2c2", "-")]
              for sl in ("D3", "D4")}
    held = level_sets()["held-out 9"]
    old_d = {ls: rows[f"{METHOD} − a2c2 (старый вариант)"][ls] for ls in ("all 12", "held-out 9")}
    pa2 = {lv: {"a2c2_paper": float(per_level(*a, b2b5.D1).get(lv, np.nan)),
                "naive": float(per_level("naive", "-", b2b5.D1).get(lv, np.nan))}
           for lv in ("worlds/l/trampoline.json", "worlds/l/mjc_walker.json")}
    lat, info = pd.read_csv(old / "latency.csv"), json.loads((old / "latency.json").read_text())
    head = json.loads((new / "head_latency.json").read_text())
    fair = {}
    for base, cs in head_cells(lat, info, head).items():
        fair[base] = [{"cell": c, "P_all12": float(per_level(*a, [c]).mean()),
                       "P_held9": float(per_level(*a, [c]).reindex(held).mean()),
                       "gpu_ms_step": b2b5.gpu_ms_step("a2c2", *c, lat, info) - info["head_ms_gpu"]
                       + head["head_ms_gpu"]} for c in cs]
    log = pd.read_csv(new / METHOD / "train_log.csv").drop_duplicates("level", keep="last")
    mse_ok = {r.level: bool(r.mse < r.mse_base) for r in log.itertuples()}
    expert_gflop = 2 * (2722 * 256 + 256 * 256 + 256 * 6) / 1e9  # train_expert.Agent actor: 2722 -> 256 -> 256 -> 6
    compute = [{"level": r.level, "transitions": int(r.transitions * 1.1), "train_gflop": r.steps * 512 * 3
                * head["head_gflop"] + r.transitions * 1.1 * (5 * info["gflop_per_eval"] / 8 + expert_gflop)}
               for r in log.itertuples()]
    out = {
        "rows": rows,
        "strawman_flag": pooled[(METHOD, "D3")] < pooled[("realtime", "D3")],  # §9 guard: worse than RTC at D3
        "P-A1 beats old A2C2 at D3 and D4": {ls: {sl: (old_d[ls][sl] or {}).get("pooled_pp", float("nan")) > 0
                                                  for sl in ("D3", "D4")} for ls in old_d},
        "P-A2 not below naive at D1": {lv: v["a2c2_paper"] >= v["naive"] for lv, v in pa2.items()} | {"values": pa2},
        "latency_fair": fair, "head": head, "mse < mse_base (held-out)": mse_ok, "training_compute": compute,
        "missing": len(missing),
    }
    (new / "fix.json").write_text(json.dumps(out, indent=1, default=str))
    levels.to_csv(new / "levels.csv")
    txt = [f"{name} [{ls}] {sl}: {r['pooled_pp']:+.2f} seeds {r['per_seed_pp']} ci {np.round(r['ci95_pp'], 2).tolist()}"
           for name, by in rows.items() for ls, sls in by.items() for sl, r in sls.items() if r]
    txt += [f"strawman flag: {out['strawman_flag']}", f"P-A1: {out['P-A1 beats old A2C2 at D3 and D4']}",
            f"P-A2: {out['P-A2 not below naive at D1']}", f"missing: {len(missing)}",
            "per level (development levels flagged):", levels.round(3).to_string()]
    (new / "fix.txt").write_text("\n".join(txt) + "\n")
    print("\n".join(txt))


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"experts": experts, "latency": latency, "zero-head-check": zero_head_check,
                                          "summarize": summarize})
