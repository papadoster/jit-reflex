"""B2+B5 GPU run (spec docs/superpowers/specs/2026-09-26-b2b5-gpu-design.md): the grid, the config lock, the |e| unit,
latency and the placement of methods by it, the forecast, and the pre-registered rules."""

import glob
import hashlib
import json
import math
import pathlib
import pickle
import threading
import time
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import tyro

import a2c2
import diag
import eval_flow
import plot
import predictors
import probe
import reflex

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

OUT = "results/b2b5"
SEEDS, EXT_SEEDS = (20, 21, 22), (23, 24, 25)
CHEAP = ("naive", "realtime", "realtime10", "a2c2", "a2c2_distill")
PRED = ("pred", "reflex", "t3", "m3")
REFLEX = (*PRED, "rtc_reflex", *(f"late{k}" for k in range(1, 5)))  # methods with a predictor
ALL16 = [(d, s) for d in range(1, 5) for s in range(d, 9 - d)]  # s >= d, d + s <= 8
R9 = [(1, 5), (1, 6), (1, 7), (2, 2), (2, 6), (3, 3), (3, 4), (3, 5), (4, 4)]
D1, D3, D4 = [(1, 5), (1, 6), (1, 7)], (3, 5), (4, 4)
BLOCKS = [  # (worker, methods, predictors, cells): spec §4
    ("A", CHEAP, ("-",), ALL16),
    ("A", ("rtc_reflex",), ("learned", "oracle"), [*D1, D3]),
    ("B", PRED, ("learned",), R9),
    ("B", ("late1",), ("learned",), [D4, (3, 3), (3, 4), (3, 5)]),
    ("B", ("late2",), ("learned",), [D4]),
    ("B", PRED, ("oracle",), [D3]),
    ("B", ("pred", "reflex"), ("phys0.2",), [D3]),
]


def worker_of(method: str) -> str:
    return "A" if method in CHEAP or method == "rtc_reflex" else "B"


def extra_blocks(out_dir: str = OUT) -> list:
    """Late-J cells added from the measured latency (spec §5.3), written by `place`."""
    f = pathlib.Path(out_dir) / "extra_blocks.json"
    return [tuple(b) for b in json.loads(f.read_text())] if f.exists() else []


def configs(blocks=None) -> set:
    """{(method, predictor, d, s)}; predictor '-' for methods without one, as eval_flow writes it."""
    return {
        (m, p if m in REFLEX else "-", d, s)
        for _, ms, ps, cs in (BLOCKS if blocks is None else blocks) for m in ms for p in ps for d, s in cs
    }


def _args(methods, preds, cells, seeds, out: str) -> str:
    preds = [p for p in preds if p != "-"] or ["oracle"]  # eval_flow ignores predictors for methods without one
    return (f"--methods {' '.join(methods)} --predictors {' '.join(preds)} "
            f"--cells {' '.join(f'{d},{s}' for d, s in cells)} --seeds {' '.join(map(str, seeds))} --output-dir {out}")


def command_lines(worker: str, out_dir: str = OUT, seeds: Sequence[int] = SEEDS) -> list[str]:
    return [
        _args(ms, ps, [tuple(c) for c in cs], seeds, f"{out_dir}/eval_{w}")
        for w, ms, ps, cs in BLOCKS + extra_blocks(out_dir) if w == worker
    ]


def commands(worker: str, out_dir: str = OUT):
    """eval_flow.py arguments of a worker's blocks, one line each; the pod script adds the shared ones."""
    print("\n".join(command_lines(worker, out_dir)))


def _sha(path) -> str:
    return hashlib.sha256(pathlib.Path(path).read_bytes()).hexdigest()


def inputs() -> list[str]:
    """Committed inputs of the pod run (spec §4)."""
    return sorted(glob.glob("results/b1/world_models/*.pkl")) + [f"{OUT}/e_std.npz"]


def grid_sha() -> str:
    return hashlib.sha256(json.dumps(BLOCKS).encode()).hexdigest()


def lock(write: bool = False, check: bool = False, add: Sequence[str] = (), path: str = f"{OUT}/lock.json"):
    """Spec §4, §10. --write (Mac, before the pod): SHA-256 of the committed inputs and of the grid. --check (pod
    start and resume): stop on any difference, also in created files already recorded. --add <dir>...: record the
    weights created on the pod."""
    p = pathlib.Path(path)
    if write:
        p.parent.mkdir(parents=True, exist_ok=True)
        lk = {"inputs": {f: _sha(f) for f in inputs()}, "grid": grid_sha(), "created": {}}
        p.write_text(json.dumps(lk, indent=1))
        return
    lk = json.loads(p.read_text())
    if check:
        bad = [f for f, h in {**lk["inputs"], **lk["created"]}.items() if not pathlib.Path(f).exists() or _sha(f) != h]
        bad += ["grid"] * (lk["grid"] != grid_sha()) + ["input list"] * (sorted(lk["inputs"]) != sorted(inputs()))
        if bad:
            raise SystemExit(f"lock mismatch: {bad}")
        print("lock OK")
    if add:
        new = {f: _sha(f) for d in add for f in sorted(glob.glob(f"{d}/*.pkl"))}
        bad = [f for f, h in new.items() if lk["created"].get(f, h) != h]
        if bad:
            raise SystemExit(f"created file changed since it was recorded: {bad}")
        lk["created"].update(new)
        p.write_text(json.dumps(lk, indent=1))


def e_std(out: str = f"{OUT}/e_std.npz", run_path: str = "checkpoints/bc", num_envs: int = 64, seed: int = 3000,
          num_flow_steps: int = 5):
    """Spec §8: each level's |e| unit as the diagnostic computed it: diag.level_std with the key seed + level index,
    the first of its three splits, 64 envs."""
    env, env_params, levels, O, A = probe.setup(list(probe.LEVELS))
    fn = jax.jit(lambda sd, lv, k: diag.level_std(
        env, env_params, probe.make_policy(sd, O, A), lv, k, num_envs, num_flow_steps
    )[1])
    stds = {}
    for i, lp in enumerate(probe.LEVELS):
        k_c = jax.random.split(jax.random.key(seed + i), 3)[0]
        stds[predictors.level_name(lp)] = np.asarray(
            fn(probe.load_state_dict(run_path, lp), jax.tree.map(lambda x: x[i], levels), k_c)
        )
        print(lp, flush=True)
    np.savez(out, **stds)


B1_CELLS = [(1, 1), (1, 4), (1, 7)]  # B1's cost cells: kept for the comparison with B1 (spec §5.1)


def latency_cells() -> dict:
    """Cells timed per method (spec §5.1): methods without a predictor at B1's cells; with one, every cell they run in
    (their latency grows with the model's d + s - 1 step rollout) plus B1's."""
    cells = {m: B1_CELLS for m in ("naive", "realtime", "realtime10")}
    for _, ms, _, cs in BLOCKS:
        for m in ms:
            if m in ("pred", "reflex", "t3", "m3", "rtc_reflex"):
                cells[m] = sorted(set(cells.get(m, B1_CELLS)) | set(cs))
    return cells


def method_call(policy, wm, name: str, d: int, s: int, num_steps: int = 5):
    """One call of `name` at batch 1 as eval_flow makes it: f(key, noise, obs, prev) -> outputs. Predictor methods
    include the learned model's rollout over the planned actions, d + s - 1 steps."""
    H = policy.action_chunk_size

    def chunk(key, noise, obs, prev):
        if name in ("realtime", "realtime10", "rtc_reflex"):
            n = eval_flow.FLOW_STEPS.get(name, num_steps)
            return policy.realtime_action(key, obs, n, prev, d, H - s, "exp", 5.0)
        return policy.action_from_noise(noise, obs, num_steps)

    if name in ("naive", "realtime", "realtime10"):
        return chunk
    m = eval_flow.METHODS[name]

    def call(key, noise, obs, prev):
        c = chunk(key, noise, obs, prev)
        planned = jnp.concatenate([prev[:, :d], c[:, d:]], axis=1)
        pred = jax.vmap(predictors.wm_rollout, in_axes=(None, 0, 0))(wm, obs, planned[:, : max(d + s - 1, 1)])
        ref = jnp.concatenate([obs[:, None], pred], axis=1)
        ref = jnp.pad(ref, ((0, 0), (0, H - ref.shape[1]), (0, 0)))
        return reflex.package(policy, noise, ref, c, num_steps, m.requery, m.feedback, used=(d, d + s), cand=m.cand)

    return call


def _time(f, args, repeats: int, warmup: int) -> float:
    """Median ms of f(*args) until ready."""
    for _ in range(warmup):
        jax.block_until_ready(f(*args))
    t = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        jax.block_until_ready(f(*args))
        t.append(time.perf_counter() - t0)
    return 1e3 * float(np.median(t))


def _concurrent(fa, fb, args, repeats: int, warmup: int):
    """Spec §5.2: (ta, tb, ta while another thread runs fb back to back, tb while one runs fa)."""
    ta, tb = _time(fa, args, repeats, warmup), _time(fb, args, repeats, warmup)

    def under(f, g):
        stop, err = threading.Event(), []

        def loop():
            try:
                while not stop.is_set():
                    jax.block_until_ready(g(*args))
            except Exception as e:  # e.g. OOM: kappa must not be measured without the load
                err.append(e)

        th = threading.Thread(target=loop)
        th.start()
        try:
            t = _time(f, args, repeats, warmup)
            alive = th.is_alive()
        finally:
            stop.set()
            th.join()
        if err or not alive:
            raise RuntimeError(f"background load died: {err[0] if err else 'thread exited'!r}")
        return t

    return ta, tb, under(fa, fb), under(fb, fa)


def _flops(f, *args) -> float:
    a = f.lower(*args).compile().cost_analysis()
    return (a[0] if isinstance(a, list) else a)["flops"]


def latency(out_dir: str = OUT, level_path: str = probe.LEVELS[0], run_path: str = "checkpoints/bc",
            repeats: int = 200, warmup: int = 20):
    """Spec §5.1-5.2 at batch 1: latency.csv (median ms per method and cell) and latency.json (realtime's ms, kappa
    and kappa_j at (4, 4), the A2C2 head's ms on GPU and CPU, GFLOP of one network evaluation and of one head step)."""
    env, env_params, levels, O, A = probe.setup([level_path])
    policy = probe.make_policy(probe.load_state_dict(run_path, level_path), O, A)
    with open(f"{predictors.WM_DIR}/{predictors.level_name(level_path)}.pkl", "rb") as f:
        wm = pickle.load(f)
    H, key = policy.action_chunk_size, jax.random.key(0)
    obs = env.reset_to_level(key, jax.tree.map(lambda x: x[0], levels), env_params)[0][None]
    noise = jax.random.normal(key, (1, H, A))
    prev = policy.action_from_noise(noise, obs, 5)
    args = (key, noise, obs, prev)
    rows, fns = [], {}
    for m, cells in latency_cells().items():
        for d, s in cells:
            fns[(m, d, s)] = f = jax.jit(method_call(policy, wm, m, d, s))
            rows.append({"method": m, "delay": d, "execute_horizon": s, "ms": _time(f, args, repeats, warmup)})
            print(rows[-1], flush=True)
    lat = pd.DataFrame(rows)
    ta, tb, ta_b, tb_a = _concurrent(fns[("pred", 4, 4)], fns[("t3", 4, 4)], args, repeats, warmup)
    head = a2c2.Head(O, A, rngs=nnx.Rngs(0))
    h = jax.jit(lambda o, a, t: head.apply_residual(o, a, t))
    hargs = (obs, prev[:, 0], a2c2.time_feature(jnp.zeros(1, jnp.int32), H))
    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        head_cpu = a2c2.Head(O, A, rngs=nnx.Rngs(0))
        h_cpu = jax.jit(lambda o, a, t: head_cpu.apply_residual(o, a, t))
    one = jax.jit(lambda o, x: policy(o, x, jnp.zeros(())))
    info = {
        "device": str(jax.devices()[0]), "level": level_path,
        "realtime_ms": float(lat[lat.method == "realtime"]["ms"].median()),
        "kappa": ta_b / ta, "kappa_j": tb_a / tb, "t_nom": ta, "t_j": tb, "t_nom_with_j": ta_b, "t_j_with_nom": tb_a,
        "head_ms_gpu": _time(h, hargs, repeats, warmup),
        "head_ms_cpu": _time(h_cpu, jax.device_put(hargs, cpu), repeats, warmup),
        "gflop_per_eval": _flops(one, obs, noise) / 1e9, "head_gflop": _flops(h, *hargs) / 1e9,
    }
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    lat.to_csv(out / "latency.csv", index=False)
    (out / "latency.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))


def _ceil(x: float) -> int:
    return math.ceil(round(x, 9))


def other_side(x: float) -> int | None:
    """Spec §5.1: the d' a ±3% latency error could give instead of ceil(x); None if x is not near an integer."""
    m = round(x)
    if abs(x - m) > 0.03 * x:
        return None
    return m + 1 if _ceil(x) == m else m


def placement(lat: pd.DataFrame, info: dict, base: int, kappa: float | None = None, kappa_j: float | None = None,
              grid: set | None = None) -> dict:
    """Spec §5.1-5.3 at base delay `base`: {method: [(d, s)]} and "late": [{d, s, delta, method, x_nom, x_j}], where
    method is the eval method whose data the cell uses (late<delta>, t3 at delta <= 0, pred at delta >= s).
    kappa, kappa_j: None = measured; 1.0 = the second-GPU scenario. grid: keep only cells that were run."""
    kappa = info["kappa"] if kappa is None else kappa
    kappa_j = info["kappa_j"] if kappa_j is None else kappa_j
    if kappa <= 1.10:  # spec §5.2: the concurrent J counts as free
        kappa = kappa_j = 1.0
    rt = info["realtime_ms"]
    r = lat.set_index(["method", "delay", "execute_horizon"])["ms"] / rt

    def feasible(n):
        return [(n, s) for s in range(n, 9 - n)]

    def ran(m, cells):
        pr = "learned" if m in REFLEX else "-"
        return [c for c in cells if grid is None or (m, pr, *c) in grid]

    out = {m: ran(m, feasible(_ceil(float(r.xs(m).median()) * base))) for m in ("naive", "realtime", "realtime10")}
    n_head = _ceil(float(r.xs("naive").median()) * base) + int(info["head_ms_cpu"] > rt / base)
    out["a2c2"] = ran("a2c2", feasible(n_head))
    out["a2c2_distill"] = ran("a2c2_distill", feasible(n_head))
    for m in ("pred", "reflex", "t3", "m3", "rtc_reflex"):
        out[m] = ran(m, [(d, s) for (d, s), x in r.xs(m).items() if _ceil(x * base) == d])
    out["late"] = []
    for (d, s), x in r.xs("pred").items():
        x_nom = kappa * x * base
        if _ceil(x_nom) != d or ("t3", d, s) not in r.index:
            continue
        x_j = kappa_j * r[("t3", d, s)] * base
        delta = _ceil(x_j) - d
        method = "t3" if delta <= 0 else "pred" if delta >= s else f"late{delta}"
        if method not in eval_flow.METHODS:
            raise ValueError(f"late J at (d={d}, s={s}) needs delta={delta}: eval_flow has no {method}")
        if ran(method, [(d, s)]) or grid is None or method.startswith("late"):
            out["late"].append({"d": d, "s": s, "delta": delta, "method": method, "x_nom": x_nom, "x_j": x_j})
    return out


def place(out_dir: str = OUT):
    """Spec §5.3: extra_blocks.json with the late-J cells bases 2 and 3 need but the grid lacks (run by worker B),
    under the measured kappa and the second-GPU scenario; then placement.json for bases 1-4 on the grid with them."""
    out = pathlib.Path(out_dir)
    lat, info = pd.read_csv(out / "latency.csv"), json.loads((out / "latency.json").read_text())
    scens, grid, extra = (("measured", None, None), ("second_gpu", 1.0, 1.0)), configs(), []
    for _, k, kj in scens:
        for base in (2, 3):
            for c in placement(lat, info, base, k, kj)["late"]:
                b = ("B", [c["method"]], ["learned"], [[c["d"], c["s"]]])
                if c["method"].startswith("late") and (c["method"], "learned", c["d"], c["s"]) not in grid:
                    extra += [b] * (b not in extra)
    ran = configs(BLOCKS + extra)
    res = {f"{sc} base {base}": placement(lat, info, base, k, kj, ran) for sc, k, kj in scens for base in (1, 2, 3, 4)}
    (out / "placement.json").write_text(json.dumps(res, indent=1))
    (out / "extra_blocks.json").write_text(json.dumps(extra))
    print(json.dumps({"extra_blocks": extra, "measured base 3": res["measured base 3"]}, indent=1))


CFG = ["method", "predictor", "delay", "execute_horizon"]


def hours_left(df: pd.DataFrame, todo: set) -> tuple[int, float]:
    """(configs done, hours left) counted per (config, seed) run; eval_flow runs every config's first seed before
    any second. A config's first run includes its compile: a config not started costs its method's mean first-run
    seconds (else the worker's, else 300 s), each later seed the method's mean later-run seconds (else first-run)."""
    runs = df.groupby([*CFG, "seed"])["seconds"].first().reset_index()
    first = runs["seed"] == runs.groupby(CFG)["seed"].transform("min")
    n = runs.groupby(CFG).size()
    f_m, l_m = runs[first].groupby("method")["seconds"].mean(), runs[~first].groupby("method")["seconds"].mean()
    f_all = float(runs.loc[first, "seconds"].mean()) if first.any() else 300.0
    done, sec = 0, 0.0
    for c in todo:
        k, f1 = n.get(c, 0), f_m.get(c[0], f_all)
        done += k >= len(SEEDS)
        sec += (k == 0) * f1 + max(len(SEEDS) - max(k, 1), 0) * l_m.get(c[0], f1)
    return done, sec / 3600


def forecast(out_dir: str = OUT):
    """Spec §10 safety: hours left per worker from the seconds of the runs so far (see hours_left)."""
    for w in ("A", "B"):
        todo = configs([b for b in BLOCKS + extra_blocks(out_dir) if b[0] == w])
        f = pathlib.Path(out_dir) / f"eval_{w}" / "results.csv"
        df = pd.read_csv(f) if f.exists() else pd.DataFrame(columns=[*CFG, "seed", "seconds"])
        done, hours = hours_left(df, todo)
        print(f"worker {w}: {done} of {len(todo)} configs done, ~{hours:.1f} h left")


E_BINS, E_FAR = "results/b1/diag/bins.csv", 2.3  # spec §8: rho_clip < 0.3 beyond |e| ~ 2.3 (diagnostic)
H1_LEVELS = ("worlds/l/mjc_walker.json", "worlds/l/mjc_swimmer.json", "worlds/l/catapult.json",
             "worlds/l/catcher_v3.json")


def decide(kind: str, g: pd.Series) -> str:
    """Spec §6 statuses from per-seed differences g (fractions)."""
    if g.isna().any():
        return "MISSING"
    m = g.mean()
    if kind == "keep":
        return "KEEPS" if m >= -0.01 - 1e-9 else "LOSES" if (g < 0).all() else "GRAY"
    hi, lo = {"win": ("WIN", "LOSE"), "pass": ("PASS", "FAIL")}[kind]
    return hi if m >= 0.01 - 1e-9 and (g > 0).all() else lo if m <= 1e-9 else "GRAY"


def fe_step(method: str, s: int, info: dict) -> float:
    """Spec §5.4: network evaluations per env step (VJP = 2); A2C2 adds its head at every step."""
    base = {"a2c2": "naive", "a2c2_distill": "naive", "realtime10": "realtime"}.get(method, method)
    base = "late" if method.startswith("late") else base
    fe = reflex.forward_equivalents(base, num_steps=10 if method == "realtime10" else 5, positions=s) / s
    return fe + (info["head_gflop"] / info["gflop_per_eval"] if method.startswith("a2c2") else 0.0)


def gpu_ms_step(method: str, d: int, s: int, lat: pd.DataFrame, info: dict, second_gpu: bool = False) -> float:
    """Spec §5.4: GPU-ms per env step at batch 1; late J pays t3's call (doubled in the second-GPU scenario)."""
    by = lat.set_index(["method", "delay", "execute_horizon"])["ms"]
    m = "t3" if method.startswith("late") else "naive" if method.startswith("a2c2") else method
    ms = float(by.xs(m).median()) if m in ("naive", "realtime", "realtime10") else float(by[(m, d, s)])
    ms = ms * (2 if second_gpu and method.startswith("late") else 1) / s
    return ms + (info["head_ms_gpu"] if method.startswith("a2c2") else 0.0)


def summarize(out_dir: str = OUT, strict: bool = True, seeds: Sequence[int] = SEEDS,
              diag_summary: str = "results/b1/diag/summary.csv") -> dict:
    """Spec §6-7: rules (with the GRAY extension), predictions and hypotheses, prices, |e| shares, training compute.
    Writes b2b5.json, frontier.csv, hist_summary.csv, training_compute.csv, b2b5.png and extension.txt (empty unless a
    rule is GRAY). strict: stop if a grid config misses a seed; without it a missing config gives MISSING."""
    out = pathlib.Path(out_dir)
    df = pd.concat([pd.read_csv(f) for f in sorted(out.glob("eval_*/results.csv"))]).drop_duplicates(
        ["seed", "delay", "execute_horizon", "method", "predictor", "level"]
    )
    P = df.groupby(["method", "predictor", "delay", "execute_horizon", "seed"])["returned_episode_solved"].mean()
    cell = df.set_index(["method", "predictor", "delay", "execute_horizon", "seed", "level"])[
        "returned_episode_solved"].sort_index()
    lat, info = pd.read_csv(out / "latency.csv"), json.loads((out / "latency.json").read_text())
    ratio = lat.set_index(["method", "delay", "execute_horizon"])["ms"] / info["realtime_ms"]
    grid = configs(BLOCKS + extra_blocks(out_dir))
    missing = [(*c, sd) for c in sorted(grid) for sd in seeds if (*c, sd) not in P.index]
    if missing and strict:
        raise SystemExit(f"{len(missing)} (config, seed) cells missing, rerun the workers: {missing[:10]}")

    def ps(cfg, sd=seeds):
        return pd.Series([P.get((*cfg, x), np.nan) for x in sd], index=list(sd))

    def best(rivals, sd=seeds):
        return pd.concat([ps(r, sd) for r in rivals], axis=1).max(axis=1, skipna=False)

    def rivals_at(pl):
        return ([(m, "-", d, s) for m in ("naive", "realtime", "realtime10") for d, s in pl[m]]
                + [("pred", "learned", d, s) for d, s in pl["pred"]])

    def ci(cand, rivals):  # report only: 95% bootstrap over level x seed cells, the best rival chosen per seed
        diffs = []
        for x in seeds:
            r = max(rivals, key=lambda c: P.get((*c, x), -1.0))
            diffs.append((cell.xs((*cand, x)) - cell.xs((*r, x))).to_numpy())
        m, lo, hi = plot._boot(np.concatenate(diffs), 10_000, np.random.default_rng(0))
        return [100 * lo, 100 * hi]

    pls = {(b, sc): placement(lat, info, b, *((None, None) if sc == "measured" else (1.0, 1.0)), grid=grid)
           for b in (1, 2, 3, 4) for sc in ("measured", "second_gpu")}
    pl3, pl3_2 = pls[(3, "measured")], pls[(3, "second_gpu")]
    rules = {f"B2-R1 {c}{sfx}": ("keep", (c, "learned", *cl), [("reflex", "learned", *cl)])  # §6.1: D4 shown only
             for c in ("t3", "m3") for cl, sfx in ((D3, ""), (D4, " D4 (report)"))}
    for name, pl in (("B5-R1 late", pl3), ("B5-R1 late, second GPU (report)", pl3_2)):
        cand = [c for c in pl["late"] if c["s"] == 8 - c["d"]]
        rules[name] = ("win", (cand[0]["method"], "learned", cand[0]["d"], cand[0]["s"]) if cand else None,
                       rivals_at(pl))
    x_m3 = float(ratio[("m3", *D4)] * 3)
    rules["B5-R2 m3"] = ("win", ("m3", "learned", *D4) if _ceil(x_m3) <= 4 else None, rivals_at(pl3))  # §6.3
    for d, s in (D3, D4):
        rules[f"B5-R3 t3 ({d},{s})"] = ("pass", ("t3", "learned", d, s),
                                         [(m, "-", d, s) for m in ("naive", "realtime", "realtime10")]
                                         + [("pred", "learned", d, s)])
    verdicts, ext = {}, set()
    for name, (kind, cand, rivals) in rules.items():
        if cand is None:
            verdicts[name] = {"verdict": "NOT-FEASIBLE"}
            continue
        g = ps(cand) - best(rivals)
        v = {"verdict": decide(kind, g), "pooled_pp": 100 * float(g.mean()), "per_seed_pp": (100 * g).tolist(),
             "cand": list(cand), "rivals": [list(r) for r in rivals]}
        if v["verdict"] != "MISSING":
            v["ci95_pp"] = ci(cand, rivals)
        if v["verdict"] == "GRAY" and "report" not in name:
            all_sd = (*seeds, *EXT_SEEDS)
            g6 = ps(cand, all_sd) - best(rivals, all_sd)
            if g6.notna().all():
                v |= {"verdict_6_seeds": decide(kind, g6), "pooled_6_pp": 100 * float(g6.mean())}
            else:
                ext |= {cand, *rivals}
        verdicts[name] = v
    for c in ("t3", "m3"):  # spec §6.1: B2's goal is met if c KEEPS and r_c(3,5) <= 1.2
        v = verdicts[f"B2-R1 {c}"]
        v |= {"r(3,5)": float(ratio[(c, *D3)]),
              "B2 goal": v.get("verdict_6_seeds", v["verdict"]) == "KEEPS" and bool(ratio[(c, *D3)] <= 1.2)}
    late = [c for c in pl3["late"] if c["s"] == 8 - c["d"]]
    k, kj = (info["kappa"], info["kappa_j"]) if info["kappa"] > 1.10 else (1.0, 1.0)  # spec §5.2
    x_nom, x_j = ((late[0]["x_nom"], late[0]["x_j"]) if late  # none (NOT-FEASIBLE): §6.2's (4,4) measurement
                  else (k * ratio[("pred", *D4)] * 3, kj * ratio[("t3", *D4)] * 3))
    for name, x in (("B5-R1 late", float(x_nom)), ("B5-R2 m3", x_m3)):
        n = other_side(x)  # spec §5.1: within ±3% of a d' edge, the verdict at the other d' goes next to it
        e = {"x": x, "other_d": n}
        if n is not None:
            dl = _ceil(x_j) - n  # late: the same J latency, delta counted from the other d'
            m = "m3" if name.endswith("m3") else "t3" if dl <= 0 else "pred" if dl >= 8 - n else f"late{dl}"
            ve = decide("win", ps((m, "learned", n, 8 - n)) - best(rivals_at(pl3))) if n <= 4 else "NOT-FEASIBLE"
            e["verdict_other_d"] = "no data" if ve == "MISSING" else ve
        verdicts[name]["edge"] = e
    lines = [
        f"{worker_of(m)} " + _args((m,), (p,), [(d, s)], EXT_SEEDS, f"{out_dir}/eval_ext_{worker_of(m)}")
        for m, p, d, s in sorted(ext) if any((m, p, d, s, x) not in P.index for x in EXT_SEEDS)
    ]
    (out / "extension.txt").write_text("".join(ln + "\n" for ln in lines))

    def pooled(cfg):
        return float(ps(cfg).mean())

    hyp = {
        "P1 m3-t3 D3 pp": 100 * (pooled(("m3", "learned", *D3)) - pooled(("t3", "learned", *D3))),
        "P1 m3-t3 D1 pp": 100 * float(np.mean([pooled(("m3", "learned", *c)) - pooled(("t3", "learned", *c))
                                               for c in D1])),
        "H1 D1 pp": 100 * float(np.mean([pooled(("rtc_reflex", "learned", *c)) - pooled(("rtc_reflex", "oracle", *c))
                                         for c in D1])),
        "H1 D3 pp": 100 * (pooled(("rtc_reflex", "learned", *D3)) - pooled(("rtc_reflex", "oracle", *D3))),
        "E2 late1-pred (4,4) pp": 100 * (pooled(("late1", "learned", *D4)) - pooled(("pred", "learned", *D4))),
    }
    hyp["P1 held"] = abs(hyp["P1 m3-t3 D3 pp"]) <= 1 and hyp["P1 m3-t3 D1 pp"] <= -1
    lv = df[df.seed.isin(seeds)].groupby(["method", "predictor", "delay", "execute_horizon", "level"])[
        "returned_episode_solved"].mean()
    for sl, cells in (("D1", D1), ("D3", [D3])):  # spec §7 H1: on each slice, D1's per-level difference over its s
        try:  # per-level parts; a partial grid (the rehearsal) lacks some rows: None instead of a crash
            pos = (sum(lv.xs(("rtc_reflex", "learned", *c)) - lv.xs(("rtc_reflex", "oracle", *c)) for c in cells)
                   / len(cells)).clip(lower=0)
            share = float(pos[list(H1_LEVELS)].sum() / pos.sum()) if pos.sum() > 0 else None
        except KeyError:
            share = None
        hyp[f"H1 share on the 4 levels {sl}"] = share
    hyp["H1 held"] = hyp["H1 D1 pp"] >= 0 and hyp["H1 D3 pp"] >= 0 and all(
        (hyp[f"H1 share on the 4 levels {sl}"] or 0) >= 0.5 for sl in ("D1", "D3"))
    try:
        t = pd.read_csv(diag_summary)
        t = t[(t.k <= 7) & (t.frac >= diag.MIN_FRAC)].groupby(["level", "predictor"])[["je_pred", "e_pred"]].mean()
        vis = t["je_pred"] / t["e_pred"]
        gap = vis.xs("phys0.2", level="predictor") - vis.xs("learned", level="predictor")
        drop = {p: lv.xs(("reflex", "oracle", *D3)) - lv.xs(("reflex", p, *D3)) for p in ("learned", "phys0.2")}
        h2 = pd.DataFrame({"drop_diff": drop["phys0.2"] - drop["learned"],
                           "vis_gap": gap.rename(index=lambda n: n.replace("worlds_l_", "worlds/l/") + ".json")})
        hyp["H2 spearman"] = float(h2.corr(method="spearman").iloc[0, 1])
        hyp["H2 levels with both > 0"] = int(((h2.drop_diff > 0) & (h2.vis_gap > 0)).sum())
    except KeyError:
        hyp["H2 spearman"] = hyp["H2 levels with both > 0"] = None

    rows = []
    for (b, sc), pl in pls.items():
        for m, cells in pl.items():
            items = ([(c["method"], c["d"], c["s"], "late") for c in cells] if m == "late"
                     else [(m, d, s, m) for d, s in cells])
            for em, d, s, label in items:
                pr = "learned" if em in REFLEX else "-"
                if (em, pr, d, s) in grid:
                    rows.append({"view": f"latency-fair {sc}", "base": b, "label": label, "method": em, "delay": d,
                                 "execute_horizon": s, "P": pooled((em, pr, d, s)),
                                 "gpu_ms_step": gpu_ms_step(em, d, s, lat, info, sc == "second_gpu"),
                                 "fe_step": fe_step(em, s, info)})
    for d, s in [*D1, (3, 3), (3, 4), D3, D4]:
        for m, pr, dd, ss in sorted(grid):
            if (dd, ss) == (d, s) and pr in ("-", "learned"):
                rows.append({"view": "same-delay", "base": d, "label": m, "method": m, "delay": d,
                             "execute_horizon": s, "P": pooled((m, pr, d, s)),
                             "gpu_ms_step": gpu_ms_step(m, d, s, lat, info), "fe_step": fe_step(m, s, info)})
    fr = pd.DataFrame(rows)
    for price in ("gpu_ms_step", "fe_step"):
        fr[f"dominated_{price}"] = [
            bool(((g[price] <= r[price]) & (g["P"] >= r["P"]) & (g.index != i)).any())
            for i, r in fr.iterrows() for g in [fr[(fr.view == r.view) & (fr.base == r.base)]]
        ]
    fr.to_csv(out / "frontier.csv", index=False)
    _figure(fr, out / "b2b5.png")

    hs = [pd.read_csv(f) for f in sorted(out.glob("eval_*/hist.csv"))]
    if hs:
        h = pd.concat(hs)
        edges = pd.read_csv(E_BINS).query("predictor == 'all'").sort_values("bin")["e_lo"].to_numpy()
        h["far"] = h["bin"] >= int(np.argmin(np.abs(edges - E_FAR)))
        g = h.assign(far_count=h["count"] * h["far"]).groupby(["method", "delay", "execute_horizon", "k"])[
            ["count", "far_count"]].sum()
        g.assign(share_far=g.far_count / g["count"]).to_csv(out / "hist_summary.csv")
    _training_compute(out, info).to_csv(out / "training_compute.csv", index=False)
    res = {"verdicts": verdicts, "hypotheses": hyp, "latency": info, "missing": len(missing)}
    (out / "b2b5.json").write_text(json.dumps(res, indent=1, default=str))
    print(json.dumps(res, indent=1, default=str))
    return verdicts


def _figure(fr: pd.DataFrame, path):
    """Row 1: latency-fair (measured kappa), base 1-4, P against GPU-ms per step. Row 2: same delay, P against FE per
    step, d = 1 (s 5-7), 3 (s 3-5), 4. Configs without data (P NaN) are not drawn."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    colors = {lb: plt.cm.tab20(i) for i, lb in enumerate(sorted(fr.label.unique()))}  # one color per label everywhere
    for ax, b in zip(axes[0], (1, 2, 3, 4)):
        g = fr[(fr.view == "latency-fair measured") & (fr.base == b)]
        for label, gg in g.groupby("label"):
            ax.scatter(gg.gpu_ms_step, 100 * gg.P, color=colors[label], s=18)
        ax.set_title(f"latency-fair, base d = {b}")
        ax.set_xlabel("GPU-ms per step (batch 1)")
    for ax, b in zip(axes[1], (1, 3, 4)):
        g = fr[(fr.view == "same-delay") & (fr.base == b)]
        for label, gg in g.groupby("label"):
            ax.scatter(gg.fe_step, 100 * gg.P, color=colors[label], s=18)
        ax.set_xscale("log")
        ax.set_title(f"same delay d = {b}")
        ax.set_xlabel("network evaluations per step")
    for ax in axes.flat:
        ax.set_ylabel("solved (%)")
    axes[1][3].axis("off")
    axes[1][3].legend(handles=[plt.Line2D([], [], marker="o", ls="", color=c, label=lb) for lb, c in colors.items()],
                      loc="center")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _training_compute(out: pathlib.Path, info: dict) -> pd.DataFrame:
    """Spec §5.4 report column, per level: env transitions used and training FLOPs (analytic, 6 x params per item)."""
    wm_params = (679 + 6) * 256 + 256 * 256 + 256 * 679 + 2 * 256 + 679  # (O + A) -> 256 -> 256 -> O, O = 679, A = 6
    rows = [{"method": "world model", "transitions": 128 * 256, "train_gflop": 6 * wm_params * 10_000 * 512 / 1e9}]
    head_gflop = info["head_gflop"]
    for name in ("a2c2", "a2c2_distill"):
        f = out / name / "train_log.csv"
        if not f.exists():
            continue
        for r in pd.read_csv(f).to_dict("records"):
            per_item = 3 * head_gflop + (5 * info["gflop_per_eval"] if name == "a2c2" else 0.0)
            collect = 5 * info["gflop_per_eval"] * r["transitions"] * (1 + 1 / 8) if name == "a2c2_distill" else 0.0
            rows.append({"method": name, "level": r["level"], "transitions": r["transitions"],
                         "train_gflop": r["steps"] * 512 * per_item + collect})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "commands": commands, "lock": lock, "e-std": e_std, "latency": latency, "place": place, "forecast": forecast,
        "summarize": summarize,
    })
