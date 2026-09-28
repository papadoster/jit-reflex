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
    """The spec §7 choice (T_ramp, K_p): highest success; ties -> larger T_ramp, then smaller K_p. §7 compares the six
    combos on the same episodes: stops unless each has the same N_PILOT distinct episodes (the owner decides)."""
    res, eps = {}, {}
    for f in files:
        rs = [json.loads(line) for line in Path(f).read_text().splitlines() if line.strip()]
        pair = {(r["t_ramp"], r["k_p"]) for r in rs}
        if len(pair) != 1:
            sys.exit(f"!!! {f}: coefficients {pair}")
        c = pair.pop()
        if c in res:
            sys.exit(f"!!! {f}: a second file with T_ramp={c[0]} K_p={c[1]}")
        res[c] = (sum(r["success"] for r in rs), len(rs))
        eps[c] = {(r["suite"], r["task"], r["init"], r["kind"], r["cell"], r["method"]) for r in rs}
    if set(res) != PILOT:
        sys.exit(f"!!! pilot combos {sorted(res)}, expected {sorted(PILOT)}")
    for (tr, kp), (k, n) in sorted(res.items()):
        print(f"T_ramp={tr:<3} K_p={kp:<4} success {k}/{n} = {k / n:.3f} ({len(eps[tr, kp])} distinct episodes)")
    if any(n != N_PILOT or len(eps[c]) != N_PILOT or eps[c] != eps[1, 0.3] for c, (_, n) in res.items()):
        sys.exit(f"!!! spec §7 compares the six combos on the same {N_PILOT} episodes, and these are not (above). "
                 "The owner decides, e.g. rerun the pilot: it resumes and retries failed batches")
    return max(res, key=lambda c: (res[c][0], c[0], -c[1]))


def stops(f, *a):
    try:
        f(*a)
    except SystemExit:
        return True
    return False


def selftest():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        def run_pilot(scores):  # {(tr, kp): (successes, episodes)}; episode i: task i // 4, init 44 + i % 4
            for f in tmp.glob("*.jsonl"):
                f.unlink()
            for (tr, kp), (k, n) in scores.items():
                (tmp / f"pilot_tr{tr}_kp{kp}.jsonl").write_text("".join(json.dumps(
                    {"suite": "libero_spatial", "task": i // 4, "init": 44 + i % 4, "kind": "step", "cell": "C",
                     "method": "G", "t_ramp": tr, "k_p": kp, "success": i < k}) + "\n" for i in range(n)))
            return pilot(sorted(tmp.glob("*.jsonl")))

        def edited(old, new, files="pilot_tr5_kp0.3.jsonl"):  # the six files of a tie, one line edited in files
            run_pilot(tie)
            for f in tmp.glob(files):
                f.write_text(f.read_text().replace(old, new, 1))
            return sorted(tmp.glob("*.jsonl"))

        tie = {c: (60, N_PILOT) for c in PILOT}
        assert run_pilot(tie) == (10, 0.3)  # all equal: larger T_ramp, then smaller K_p
        assert run_pilot(tie | {(10, 0.3): (59, N_PILOT)}) == (10, 1.0)  # T_ramp decides before K_p
        assert run_pilot(tie | {(10, 0.3): (59, N_PILOT), (10, 1.0): (59, N_PILOT)}) == (5, 0.3)
        assert run_pilot(tie | {(1, 1.0): (61, N_PILOT)}) == (1, 1.0)  # success first
        assert stops(run_pilot, {(1, 0.3): (1, N_PILOT)})  # a missing combo stops the choice
        # not the same N_PILOT episodes in every combo (§7) stops: fewer, more, another episode, a duplicate
        assert stops(run_pilot, tie | {(1, 1.0): (58, 96)}) and stops(run_pilot, tie | {(5, 1.0): (60, N_PILOT + 1)})
        assert stops(pilot, edited('"init": 47', '"init": 48')) and stops(pilot, edited('"init": 47', '"init": 46'))
        assert stops(pilot, edited('"init": 47', '"init": 46', "*.jsonl"))  # the same duplicate in all six
        assert pilot(fs := edited("", "")) == (10, 0.3)  # the unedited files pass
        fs[0].write_text(fs[0].read_text() + fs[0].read_text().splitlines()[0] + "\n")  # one record written twice
        assert stops(pilot, fs)
        # two files with the same (T_ramp, K_p) stop
        run_pilot(tie)
        (tmp / "pilot_tr5_kp0.3_copy.jsonl").write_text((tmp / "pilot_tr5_kp0.3.jsonl").read_text())
        assert stops(pilot, sorted(tmp.glob("*.jsonl")))
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
