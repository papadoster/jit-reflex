"""C1 scouting: how well can the arm state be predicted from the executed plan alone (the world model for o_hat)?

Fits per-dimension least-squares models on lerobot/libero (state: eef pos 3, axis-angle 3, gripper qpos 2;
action: 7-dim delta EEF + gripper, OSC_POSE) and reports open-loop k-step errors on held-out episodes.
    hold:  s_hat[t+k] = s[t]                                   (the stale state)
    int:   ds[t+1] = G a[t]                                    (integrator of the commanded delta)
    arx:   ds[t+1] = G a[t] + B ds[t] + c                      (adds the controller's lag)
Runs in the separate torch env:  ~/Desktop/M2R-c1-env/bin/python scripts/c1_scout_wm.py
"""

import glob
import json

import numpy as np
import pandas as pd
from huggingface_hub import snapshot_download

root = snapshot_download("lerobot/libero", repo_type="dataset", allow_patterns=["data/*", "meta/*.json"])
df = pd.concat(pd.read_parquet(f, columns=["episode_index", "frame_index", "observation.state", "action"])
               for f in sorted(glob.glob(f"{root}/data/*/*.parquet")))
df = df.sort_values(["episode_index", "frame_index"])
eps = [(np.stack(g["observation.state"]), np.stack(g["action"])) for _, g in df.groupby("episode_index")]
rng = np.random.default_rng(0)
test = set(rng.choice(len(eps), size=len(eps) // 10, replace=False).tolist())
train = [e for i, e in enumerate(eps) if i not in test]
held = [e for i, e in enumerate(eps) if i in test]
A_OF = {0: 0, 1: 1, 2: 2, 6: 6, 7: 6}  # modelled state dim -> the action dim that drives it (x, y, z, both fingers)


def fit(kind):  # least squares per state dim on one-step deltas
    W = {}
    for d, ad in A_OF.items():
        X, y = [], []
        for s, a in train:
            ds = np.diff(s[:, d])
            feats = [a[1:-1, ad]] + ([ds[:-1], np.ones(len(ds) - 1)] if kind == "arx" else [])
            X.append(np.stack(feats, 1))
            y.append(ds[1:])
        W[d] = np.linalg.lstsq(np.concatenate(X), np.concatenate(y), rcond=None)[0]
    return W


def rollout(W, kind, s, a, t, k):  # open-loop prediction of s[t+k] from s[t], s[t-1] and a[t..t+k-1]
    x, v = s[t].copy(), s[t] - s[t - 1]
    for i in range(k):
        nv = v.copy()
        for d, ad in A_OF.items():
            nv[d] = np.dot(W[d], [a[t + i, ad]] + ([v[d], 1.0] if kind == "arx" else []))
        x, v = x + nv, nv
    return x


models = {"int": fit("int"), "arx": fit("arx")}
idle = np.mean([np.mean(np.abs(a[:, :6]).max(1) < 1e-3) for _, a in eps])  # no-op frames kept in the data?
res = {"episodes": len(eps), "frames": len(df), "held_out_episodes": len(held), "share_noop_frames": round(float(idle), 4),
       "G_pos_m_per_unit": [round(float(models["int"][d][0]), 4) for d in range(3)],
       "arx_pos_(G,B,c)": [[round(float(w), 4) for w in models["arx"][d]] for d in range(3)], "err_mm": {}}
for k in (1, 5, 10, 25):
    errs = {m: [] for m in ("hold", "int", "arx")}
    for s, a in held:
        for t in range(1, len(s) - k, 5):
            errs["hold"].append(np.linalg.norm(s[t + k, :3] - s[t, :3]))
            for m in ("int", "arx"):
                errs[m].append(np.linalg.norm(s[t + k, :3] - rollout(models[m], m, s, a, t, k)[:3]))
    res["err_mm"][k] = {m: {"median": round(1e3 * float(np.median(e)), 2), "p90": round(1e3 * float(np.percentile(e, 90)), 2)}
                        for m, e in errs.items()}
print(json.dumps(res, indent=1))
