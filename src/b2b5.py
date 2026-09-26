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
    ("B", ("late1",), ("learned",), [(4, 4), (3, 3), (3, 4), (3, 5)]),
    ("B", ("late2",), ("learned",), [(4, 4)]),
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
        bad += ["grid"] * (lk["grid"] != grid_sha()) + ["input list"] * (sorted(lk["inputs"]) != inputs())
        if bad:
            raise SystemExit(f"lock mismatch: {bad}")
        print("lock OK")
    if add:
        for d in add:
            lk["created"].update({f: _sha(f) for f in sorted(glob.glob(f"{d}/*.pkl"))})
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


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"commands": commands, "lock": lock, "e-std": e_std})
