"""Report-only tables for the B2+B5 memo (docs/results/b2b5.md), from the committed pod results. Not decision rules:
the rules and their verdicts are in results/b2b5/b2b5.json. Writes results/b2b5/memo_tables.txt.
    uv run --offline python scripts/b2b5_tables.py"""
import io
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, "src")
from plot import _boot  # noqa: E402

O = "results/b2b5"
df = pd.concat([pd.read_csv(f"{O}/eval_{w}/results.csv") for w in "AB"])
df["level"] = df["level"].str.replace("worlds/l/", "").str.replace(".json", "")
K = ["method", "predictor", "delay", "execute_horizon"]
cell = df.set_index([*K, "seed", "level"])["returned_episode_solved"].sort_index()
P = 100 * df.groupby(K)["returned_episode_solved"].mean()  # pooled over seeds and levels, %
CELLS = [(1, 5), (1, 6), (1, 7), (2, 2), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]
D1 = [(1, 5), (1, 6), (1, 7)]
buf = io.StringIO()


def out(*a):
    print(*a, file=buf)


def key(m):
    return (m, "learned" if m in ("pred", "reflex", "t3", "m3", "rtc_reflex", "late1", "late2") else "-")


def diff_ci(a, b, cells):
    """a - b in pp over the given cells: pooled, per seed, 95% bootstrap over level x seed (x cell) values."""
    x = pd.concat([cell.xs((*key(a), *c)) - cell.xs((*key(b), *c)) for c in cells], axis=1).mean(axis=1)
    m, lo, hi = _boot(x.to_numpy(), 10_000, np.random.default_rng(0))
    seeds = x.groupby(level="seed").mean()
    return 100 * m, [round(100 * v, 2) for v in seeds], (round(100 * lo, 2), round(100 * hi, 2))


out("## A. Pooled P (%), 12 levels x seeds 20-22")
meths = ["naive", "realtime", "realtime10", "pred", "reflex", "t3", "m3", "rtc_reflex", "late1", "late2", "a2c2",
         "a2c2_distill"]
tab = pd.DataFrame({m: [P.get((*key(m), *c), np.nan) for c in CELLS] for m in meths},
                   index=[f"({d},{s})" for d, s in CELLS])
out(tab.round(1).to_string())
nav = df[df.method == "naive"].groupby(["delay", "execute_horizon"])["returned_episode_solved"].mean().mul(100)
out("\nnaive, realtime, realtime10, a2c2, a2c2_distill over all their (d, s):")
for m in ("naive", "realtime", "realtime10", "a2c2", "a2c2_distill"):
    s = P.xs(key(m)[0]).xs("-")
    out(f"  {m}: min {s.min():.1f} max {s.max():.1f}; by d (mean over s): "
        + ", ".join(f"d{d} {s.xs(d).mean():.1f}" for d in sorted(s.index.get_level_values(0).unique())))

out("\n## B. Differences in pp: pooled, per seed, 95% bootstrap (level x seed)")
for a, b in [("a2c2_distill", "t3"), ("a2c2_distill", "naive"), ("a2c2_distill", "realtime"), ("a2c2_distill", "reflex"),
             ("a2c2", "naive"), ("a2c2", "realtime"), ("t3", "reflex"), ("m3", "reflex"), ("m3", "t3")]:
    for name, cs in [("D1", D1), ("(2,2)", [(2, 2)]), ("(2,6)", [(2, 6)]), ("(3,3)", [(3, 3)]), ("(3,4)", [(3, 4)]),
                     ("D3", [(3, 5)]), ("D4", [(4, 4)])]:
        m, sd, ci = diff_ci(a, b, cs)
        out(f"  {a} - {b} {name}: {m:+.2f} seeds {sd} ci {ci}")

out("\n## C. Per level (%), mean over seeds; D1 = mean over s 5-7")
lv = 100 * df.groupby([*K, "level"])["returned_episode_solved"].mean()


def per_level(m, cs):
    return sum(lv.xs((*key(m), *c)) for c in cs) / len(cs)


for name, cs in [("D1", D1), ("D3 (3,5)", [(3, 5)]), ("D4 (4,4)", [(4, 4)])]:
    t = pd.DataFrame({m: per_level(m, cs) for m in ["naive", "realtime", "pred", "reflex", "t3", "a2c2", "a2c2_distill"]})
    t["distill-t3"] = t["a2c2_distill"] - t["t3"]
    t["distill-naive"] = t["a2c2_distill"] - t["naive"]
    t["t3-reflex"] = t["t3"] - t["reflex"]
    out(f"\n{name}:\n" + t.round(1).to_string())
    out(f"  levels distill > t3: {(t['distill-t3'] > 0).sum()}/12, t3 > reflex: {(t['t3-reflex'] > 0).sum()}/12")

out("\n## D. distill - t3 per level and seed at D3: sign counts")
x = cell.xs((*key("a2c2_distill"), 3, 5)) - cell.xs((*key("t3"), 3, 5))
out((100 * x.unstack("seed")).round(1).to_string())

out("\n## E. Latency-fair frontier (measured): best config per method, not dominated on GPU-ms")
fr = pd.read_csv(f"{O}/frontier.csv")
for b in (1, 2, 3, 4):
    g = fr[(fr.view == "latency-fair measured") & (fr.base == b)]
    best = g.loc[g.groupby("label")["P"].idxmax()].sort_values("P", ascending=False)
    out(f"base d = {b}:")
    for _, r in best.iterrows():
        out(f"  {r.label:28s} ({r.delay},{r.execute_horizon}) P {100 * r.P:.1f} gpu-ms/step {r.gpu_ms_step:.3f} "
            f"fe/step {r.fe_step:.2f} dominated {r.dominated_gpu_ms_step}")
    nd = g[g.dominated_gpu_ms_step == False]  # noqa: E712
    out("  not dominated: " + ", ".join(f"{r.label} ({r.delay},{r.execute_horizon}) {100 * r.P:.1f}"
                                         for _, r in nd.sort_values("gpu_ms_step").iterrows()))

out("\n## F. B3: share of executed steps with |e| > 2.3 (hist_summary.csv, all k)")
h = pd.read_csv(f"{O}/hist_summary.csv")
h = h.groupby(["method", "delay", "execute_horizon"])[["count", "far_count"]].sum()
h["share_far"] = h.far_count / h["count"]
out(h["share_far"].unstack("method").round(3).to_string())

out("\n## G. Training compute (GFLOP per level)")
tc = pd.read_csv(f"{O}/training_compute.csv")
out(tc.groupby("method")["train_gflop"].agg(["mean", "min", "max", "count"]).to_string())

open(f"{O}/memo_tables.txt", "w").write(buf.getvalue())
print(buf.getvalue())
