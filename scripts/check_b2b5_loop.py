"""Closed-loop identities of spec B2+B5 §11 on one level (Mac CPU). Run from the repo root:
  UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline python scripts/check_b2b5_loop.py
1. a2c2 with an all-zero head reproduces naive episode for episode;
2. the |e| histogram of t3 (learned) counts each executed step of the first episode once, except the first D
   (the initial package, no prediction);
3. a second eval_flow run over the same output directory runs nothing (resume);
4. results.csv records the Jacobian batch used by reflex methods (NaN for the others).
"""

import pathlib
import pickle
import sys
import tempfile

import flax.nnx as nnx
import jax
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import a2c2  # noqa: E402
import eval_flow  # noqa: E402
import predictors  # noqa: E402
import probe  # noqa: E402

LEVEL, N, D, S = "worlds/l/catapult.json", 4, 2, 4
tmp = pathlib.Path(tempfile.mkdtemp())
_, _, _, O, A = probe.setup([LEVEL])
name = predictors.level_name(LEVEL)
(tmp / "heads" / "a2c2").mkdir(parents=True)
with (tmp / "heads" / "a2c2" / f"{name}.pkl").open("wb") as f:
    pickle.dump(jax.tree.map(np.zeros_like, nnx.state(a2c2.Head(O, A, rngs=nnx.Rngs(0))).to_pure_dict()), f)
np.savez(tmp / "e_std.npz", **{name: np.ones(O, np.float32)})
common = dict(
    run_path="checkpoints/bc", config=eval_flow.EvalConfig(num_evals=N), level_paths=[LEVEL], seeds=[99],
    cells=[f"{D},{S}"], heads_root=str(tmp / "heads"), e_std=str(tmp / "e_std.npz"),
)
cols = ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]
eval_flow.main(**common, methods=["naive", "a2c2"], output_dir=str(tmp / "a"))
r = pd.read_csv(tmp / "a" / "results.csv").set_index("method")
ok1 = bool((r.loc["naive", cols] == r.loc["a2c2", cols]).all())
eval_flow.main(**common, methods=["t3"], predictors=["learned"], output_dir=str(tmp / "b"))
res, h = pd.read_csv(tmp / "b" / "results.csv"), pd.read_csv(tmp / "b" / "hist.csv")
ok2 = bool(h["count"].sum() == (res["returned_episode_lengths"].iloc[0] - D) * N)
eval_flow.main(**common, methods=["t3"], predictors=["learned"], output_dir=str(tmp / "b"))
ok3 = len(pd.read_csv(tmp / "b" / "results.csv")) == len(res) and len(pd.read_csv(tmp / "b" / "hist.csv")) == len(h)
ok4 = bool((res["package_batch"] == 16).all() and r["package_batch"].isna().all())
print(f"zero head == naive: {ok1}\n|e| counts == first-episode steps - d: {ok2}\nresume adds nothing: {ok3}")
print(f"package_batch column: {ok4}")
ok = ok1 and ok2 and ok3 and ok4
print("ALL OK" if ok else "FAILED")
sys.exit(0 if ok else 1)
