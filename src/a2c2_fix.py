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


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"experts": experts, "latency": latency})
