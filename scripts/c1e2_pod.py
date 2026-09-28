"""C1-E2 pod helpers for scripts/gpu_c1e2.sh (stdlib only).
    python scripts/c1e2_pod.py baseline results/c1-e2/base/s1_libero_spatial ... > results/c1-e2/baseline.json
    python scripts/c1e2_pod.py pilot results/c1-e2/pilot_tr*_kp*.jsonl
    python scripts/c1e2_pod.py --selftest
baseline: lerobot-eval 0.6.1 output dirs named s<n_action_steps>_<suite>, each with its eval_info.json, where
per_task[i]["metrics"]["successes"] holds one bool per episode.
"""

import json
import re
import sys
import tempfile
from fractions import Fraction
from pathlib import Path

N_TASKS, N_EP, N_PILOT = 26, 10, 104
PILOT = {(tr, kp) for tr in (1, 5, 10) for kp in (0.3, 1.0)}  # spec §7


def baseline(dirs):
    """{"s1": rate, "s10": rate, "n": {s: episodes}, "suites": {s: {suite: [rate, episodes]}}}, rates in [0, 1]."""
    cnt = {}  # s -> suite -> [successes, episodes]
    for d in map(Path, dirs):
        s = "s" + re.match(r"s(\d+)_", d.name).group(1)
        for t in json.loads((d / "eval_info.json").read_text())["per_task"]:
            x = t["metrics"]["successes"]
            if len(x) != N_EP:
                sys.exit(f"!!! {d}: {t['task_group']}:{t['task_id']} has {len(x)} episodes, expected {N_EP}")
            c = cnt.setdefault(s, {}).setdefault(t["task_group"], [0, 0])
            c[0], c[1] = c[0] + sum(map(bool, x)), c[1] + len(x)
    n = {s: sum(c[1] for c in v.values()) for s, v in cnt.items()}
    if set(n) != {"s1", "s10"} or set(n.values()) != {N_TASKS * N_EP}:
        sys.exit(f"!!! baseline episodes per setting {n}, expected s1 and s10 with {N_TASKS * N_EP} each")
    return {s: sum(c[0] for c in v.values()) / n[s] for s, v in cnt.items()} | {
        "n": n, "suites": {s: {g: [k / m, m] for g, (k, m) in v.items()} for s, v in cnt.items()}}


def pilot(files):
    """The spec §7 choice (T_ramp, K_p): highest success; ties -> larger T_ramp, then smaller K_p."""
    res = {}
    for f in files:
        rs = [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
        pair = {(r["t_ramp"], r["k_p"]) for r in rs}
        if len(pair) != 1:
            sys.exit(f"!!! {f}: coefficients {pair}")
        res[pair.pop()] = (sum(r["success"] for r in rs), len(rs))
    if set(res) != PILOT:
        sys.exit(f"!!! pilot combos {sorted(res)}, expected {sorted(PILOT)}")
    for (tr, kp), (k, n) in sorted(res.items()):
        warn = "" if n == N_PILOT else f"   !!! only {n} of {N_PILOT} episodes"
        print(f"T_ramp={tr:<3} K_p={kp:<4} success {k}/{n} = {k / n:.3f}{warn}")
    return max(res, key=lambda c: (Fraction(*res[c]), c[0], -c[1]))


def stops(f, *a):
    try:
        f(*a)
    except SystemExit:
        return True
    return False


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def run_pilot(scores):  # {(tr, kp): (successes, episodes)}
            for f in tmp.glob("*.jsonl"):
                f.unlink()
            for (tr, kp), (k, n) in scores.items():
                (tmp / f"pilot_tr{tr}_kp{kp}.jsonl").write_text(
                    "".join(json.dumps({"t_ramp": tr, "k_p": kp, "success": i < k}) + "\n" for i in range(n)))
            return pilot(sorted(tmp.glob("*.jsonl")))

        tie = {c: (60, N_PILOT) for c in PILOT}
        assert run_pilot(tie) == (10, 0.3)  # all equal: larger T_ramp, then smaller K_p
        assert run_pilot(tie | {(10, 0.3): (59, N_PILOT)}) == (10, 1.0)  # T_ramp decides before K_p
        assert run_pilot(tie | {(10, 0.3): (59, N_PILOT), (10, 1.0): (59, N_PILOT)}) == (5, 0.3)
        assert run_pilot(tie | {(1, 1.0): (61, N_PILOT)}) == (1, 1.0)  # success first
        assert run_pilot(tie | {(1, 1.0): (58, 96)}) == (1, 1.0)  # a rate, not a count: 58/96 > 60/104
        assert stops(run_pilot, {(1, 0.3): (1, N_PILOT)})  # a missing combo stops the choice
        # eval_info.json as lerobot_eval.eval_main writes it (per_group and overall are not read)
        for s, ks in (("s1", (7, 7, 7)), ("s10", (9, 8, 5))):
            for (suite, ids), k in zip((("libero_spatial", range(10)), ("libero_object", range(10)),
                                        ("libero_goal", (1, 2, 4, 6, 8, 9))), ks):
                m = {"sum_rewards": [0.0] * N_EP, "max_rewards": [1.0] * N_EP,
                     "successes": [j < k for j in range(N_EP)], "video_paths": [], "predicted_video_paths": []}
                agg = {"pc_success": 10.0 * k, "n_episodes": N_EP * len(ids)}
                (tmp / f"{s}_{suite}").mkdir()
                (tmp / f"{s}_{suite}" / "eval_info.json").write_text(json.dumps(
                    {"per_task": [{"task_group": suite, "task_id": i, "metrics": m} for i in ids],
                     "per_group": {suite: agg}, "overall": agg}))
        b = baseline(sorted(tmp.glob("s*_*")))
        assert (b["s1"], b["s10"], b["n"]) == (0.7, 200 / 260, {"s1": 260, "s10": 260}), b
        assert b["suites"]["s10"]["libero_goal"] == [0.5, 60], b
        assert stops(baseline, sorted(tmp.glob("s*_libero_[so]*")))  # a missing suite stops the baseline
    print("selftest ok")


if __name__ == "__main__":
    cmd, args = (sys.argv[1:2] or ["--selftest"])[0], sys.argv[2:]
    if cmd == "--selftest":
        selftest()
    elif cmd == "baseline":
        print(json.dumps(baseline(args), indent=1))
    elif cmd == "pilot":
        tr, kp = pilot(args)
        print(f"chosen (spec §7: highest success; ties -> larger T_ramp, then smaller K_p): T_ramp={tr} K_p={kp}\n"
              f"Next: 1) on the Mac, add this exact line to the spec journal, commit and push:\n"
              f"     C1-E2 coefficients: T_ramp={tr} K_p={kp}\n"
              f"  2) on the pod: git pull && ./scripts/gpu_c1e2.sh grid {tr} {kp}")
    else:
        sys.exit(__doc__)
