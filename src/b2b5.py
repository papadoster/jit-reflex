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


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({
        "commands": commands, "lock": lock, "e-std": e_std, "latency": latency, "place": place, "forecast": forecast,
    })
