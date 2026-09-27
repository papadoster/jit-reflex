"""Exploratory check (a), step 2 (docs/results/b2b5-checks.md): what the A2C2 and distill heads do in the closed loop.

eval_flow itself runs d = 1, s = 5 with 64 envs and seed 20 on trampoline, mjc_walker and car_launch (control); a
patched a2c2.Head.apply_residual records (obs, chunk action, corrected action) at every step by a host callback. Then per
level and head: the residual per action dim, the share of saturated actions (|a| > 1) with and without the head, and
the distance of the executed action to a fresh policy call pi(o_t) (new flow noise; actions clipped to [-1, 1]).
    uv run --offline python scripts/check_a2c2_loop.py     (~10 min on the Mac CPU)
Writes results/b2b5/checks/a2c2_loop.txt."""
import pathlib
import sys
import tempfile

import jax
import numpy as np
import pandas as pd

sys.path.insert(0, "src")
import a2c2  # noqa: E402
import eval_flow  # noqa: E402
import probe  # noqa: E402

LEVELS = ("trampoline", "mjc_walker", "car_launch")
HEADS = ("a2c2", "a2c2_distill")
N_ENVS, N_FRESH = 64, 2048
REC = []
_apply = a2c2.Head.apply_residual


def recording(self, obs, base_action, time_feature):
    out = _apply(self, obs, base_action, time_feature)
    jax.debug.callback(lambda o, b, a: REC.append((np.asarray(o), np.asarray(b), np.asarray(a))), obs, base_action, out)
    return out


def main():
    a2c2.Head.apply_residual = recording
    out, rows = pathlib.Path("results/b2b5/checks"), []
    env, env_params, levels, obs_dim, action_dim = probe.setup([f"worlds/l/{lv}.json" for lv in LEVELS])
    for i, lv in enumerate(LEVELS):
        policy = probe.make_policy(probe.load_state_dict("checkpoints/bc", f"worlds/l/{lv}.json"), obs_dim, action_dim)
        fresh = jax.jit(lambda k, o: policy.action(k, o, 5)[:, 0])
        for m in HEADS:
            REC.clear()
            with tempfile.TemporaryDirectory() as tmp:
                eval_flow.main(config=eval_flow.EvalConfig(num_evals=N_ENVS), level_paths=[f"worlds/l/{lv}.json"],
                               seeds=[20], methods=[m], cells=["1,5"], output_dir=tmp, run_path="checkpoints/bc")
                solved = float(pd.read_csv(pathlib.Path(tmp) / "results.csv")["returned_episode_solved"].mean())
            o, b, a = (np.concatenate([np.reshape(r[j], (-1, r[j].shape[-1])) for r in REC]) for j in range(3))
            idx = np.random.default_rng(0).choice(len(o), min(N_FRESH, len(o)), replace=False)
            f = np.asarray(fresh(jax.random.key(1), o[idx]))
            d_head = np.linalg.norm(np.clip(a[idx], -1, 1) - f, axis=-1)
            d_base = np.linalg.norm(np.clip(b[idx], -1, 1) - f, axis=-1)
            r = a - b
            rows.append({
                "level": lv, "head": m, "solved": solved, "steps": len(o),
                "resid_mean": np.round(r.mean(0), 2).tolist(), "resid_abs": np.round(np.abs(r).mean(0), 2).tolist(),
                "sat_base": np.round((np.abs(b) > 1).mean(0), 3).tolist(),
                "sat_head": np.round((np.abs(a) > 1).mean(0), 3).tolist(),
                "dist_to_fresh_base": float(d_base.mean()), "dist_to_fresh_head": float(d_head.mean()),
                "head_closer_share": float((d_head < d_base).mean()),
            })
            print(rows[-1], flush=True)
    t = pd.DataFrame(rows)
    (out / "a2c2_loop.txt").write_text(t.to_string() + "\n")
    print(t.to_string())


if __name__ == "__main__":
    main()
