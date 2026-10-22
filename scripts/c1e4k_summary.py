"""C1-E4 kappa-law ablation (C): G-auto - G(k = 1.0) along the delay axis (docs/superpowers/specs/
2026-10-20-c1-e4-pbvs-design.md, its journal "2026-10-22 — рецензия: задачи A, B, C", C), on pod C after a clean vtgrid
(scripts/gpu_c1e4m.sh POD=C kgrid): the runner plan per line and the reading. G(k = 1.0) is VS1.0, the controls spec
§5's k_π (k_select.json), G by code (tests/test_gauto.py test_vs1_is_g_by_executed_code). Reuses
scripts/c1e4v_summary.py (the controls mode and k, groups, ids, unique, pairs) and through it scripts/c1e4c_summary.py;
run from the repository root. R = results/c1-e4/controls/pod_C:
    python scripts/c1e4k_summary.py --plan --line L
        L: Kd10 Kd20, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4k_summary.py [read] R/grid_all_K.jsonl [--controls R/C2_pi05_eval300_plus_d0.jsonl R/C3_*.jsonl]
            [--out R/summary_K.json]
        the checks before reading, then D(d), the trend T = D(20) - D(0) and the journal's reading of T; a failed check
        prints only the checks, exit 3. Needs hf-libero (the task clusters)
    python scripts/c1e4k_summary.py --selftest
"""

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import P, TOL, e2, maxwrap, run  # noqa: E402

ARM = "VS1.0"  # the controls spec §5's k_π (k_select.json on pod C): G with kappa fixed at 1.0
LINES = {"Kd10": 10, "Kd20": 20}  # the journal: Plus only; d = 0 is line 2's frozen file (G-auto and VS1.0 together)
REPS = 8000  # the journal: cluster resamples (e2.diff_pp's scheme, which has 2000)
# the journal's reading of T, word for word: its one-sided lower 95 % bound above zero, or not
READING = {True: "преимущество закона З1 над лучшим фиксированным k растёт с задержкой",
           False: "рост с задержкой не показан"}


def plan(line, split=None):
    """The journal's lines, phase line.lower(), in c1e4c_summary.mlines's format: VS1.0 alone on eval300's Plus cases
    at d 10 / 20 with --base-prefix, controls line 3's Plus cases, order and Base groups of that d (asserted): the pairs
    with its G-auto, as V3's PBVS."""
    split = split or maxwrap.load_split(cs.SPLIT)
    d = LINES[line]
    plus = lambda lns: [ln for ln in lns if ln["source"] == "plus" and f"--d {d} " in ln["args"]]  # noqa: E731
    lns = plus(cc.mlines(line.lower(), "K", P, "eval300", (ARM,), split, d))
    ctl = plus(cc.plan("3", cv.CMODE, cv.CK, split))
    assert len(lns) == 1 and cv.groups(lns, split) == cv.groups(ctl, split), \
        f"{line}: not controls line 3's Plus cases, order and groups"
    return lns


def lines(split):
    """The runner lines the reading reads, {name: (d, line)}: line 2's frozen C2 Plus line (G-auto and VS1.0, one
    Base, its first 108 Plus cases), line 3's Plus lines (G-auto, d 10 / 20) and the K lines (VS1.0, d 10 / 20)."""
    ctl = {ln["id"]: ln for L in "23" for ln in cc.plan(L, cv.CMODE, cv.CK, split)}
    return ({"C2d0": (0, ctl[f"C2_{P}_eval300_plus_d0"])}
            | {f"C3d{d}": (d, ctl[f"C3_{P}_eval300_plus_d{d}"]) for d in (10, 20)}
            | {L: (d, plan(L, split)[0]) for L, d in LINES.items()})


def boot(*parts, reps=REPS):
    """The journal's C: the paired difference x - y in pp (every pair weighs one) and its cluster bootstrap by source
    task, e2.diff_pp's scheme (generator seed 0, percentiles 2.5, 5, 97.5) at reps resamples. parts: (sign, cv.pairs's
    pairs) per d; with two, the signed sum (T = D(20) - D(0)) and the clusters resampled jointly: a task's pairs at
    both d move together. Raw floats."""
    keys = sorted({p[1] for _, prs in parts for p in prs})
    pick = np.random.default_rng(0).integers(len(keys), size=(reps, len(keys)))
    est, bs, npr = 0.0, 0.0, []
    for sign, prs in parts:
        s, n = defaultdict(float), Counter()
        for _, c, a, b in prs:
            s[c] += float(a) - float(b)
            n[c] += 1
        s, n = np.array([s[k] for k in keys]), np.array([n[k] for k in keys])
        est += sign * float(100 * s.sum() / n.sum())
        bs = bs + sign * 100 * s[pick].sum(1) / n[pick].sum(1)
        npr.append(int(n.sum()))
    lo2, lo5, hi = map(float, np.percentile(bs, [2.5, 5, 97.5]))
    return {"diff_pp": est, "lo2.5": lo2, "lo5_one_sided": lo5, "hi97.5": hi,
            "pairs": npr[0] if len(npr) == 1 else npr, "clusters": len(keys)}


def read(new, ctrl, names, split):
    """The journal's checks before reading (each K line's VS1.0 present on >= 99 % of its 187 cases, and so, as the
    siblings' controls completeness, the controls lines' G-auto and VS1.0 the pairs read; invalid pairs <= 2 % per d,
    a missing record not one; conf equal per d but arms, gpu, kbar), then
    D(d) = Gauto - VS1.0 (d 0: the 108; d 10, 20: the 187 and the 108), T = D(20) - D(0) and D(10) - D(0) on the 108
    and the reading of T. new: grid_all_K.jsonl; ctrl: controls lines 2 and 3 (their Plus lines read here). Pairs:
    cv.pairs, one Dynamic episode per side and case, the same seed, both valid, the two Bases' e equal (d 0: the one
    Base on both sides). Records off the plan are counted, never read."""
    lns = lines(split)
    want = {x: L for L, (_, ln) in lns.items() for x in cc.lkeys([ln])}
    by = cv.unique(new + ctrl, cc.lk)
    used = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    comp = {}
    for L, (_, ln) in lns.items():  # the Dynamic records read: >= 99 % present per line
        exp = {x for x in cc.lkeys([ln]) if x[3] == "dynamic" and x[4] in (ARM, "Gauto")}
        miss = len(exp - used.keys())
        comp[L] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    conf = defaultdict(set)
    for r in used.values():  # conf equal per d but cc.CONF_SKIP (arms, gpu, kbar)
        conf[f"d{r['d']}"].add(json.dumps({f: v for f, v in r["conf"].items() if f not in cc.CONF_SKIP},
                                          sort_keys=True))
    mixed = {g: sorted(v) for g, v in conf.items() if len(v) > 1}
    one = {r["case_id"]: r for r in used.values()}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    of = lambda L, kind, a: {r["case_id"]: r for x, r in used.items() if want[x] == L and x[3:5] == (kind, a)}  # noqa
    c108, c187 = cv.ids(lns["C2d0"][1], split), cv.ids(lns["Kd10"][1], split)
    side = {0: ("C2d0", "C2d0"), 10: ("C3d10", "Kd10"), 20: ("C3d20", "Kd20")}  # d: (Gauto's line, VS1.0's)
    pr = {(d, len(cases)): cv.pairs(of(g, "dynamic", "Gauto"), of(k, "dynamic", ARM), of(g, "base", "none"),
                                    of(k, "base", "none"), clu, cases)
          for d, (g, k) in side.items() for cases in ([c108] if d == 0 else [c187, c108])}
    bad = lambda drop: sum(v for w, v in drop.items() if w != "missing")  # noqa: E731  (missing: the completeness's)
    inv = {f"d{d}": {"pairs": len(pr[d, n][0]) + bad(pr[d, n][1]), "invalid": bad(pr[d, n][1])}
           for d, n in ((0, 108), (10, 187), (20, 187))}
    checks = {"records": len(new), "read": len(used),
              "skipped": sum(bool(r.get("skipped_reason")) for x, r in by.items() if x in want),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for r in new if cc.lk(r) not in want)),
              "complete": comp, "invalid_pairs": {d: v | {"pass": v["invalid"] <= 0.02 * v["pairs"] + TOL}
                                                  for d, v in inv.items()},
              "conf": {"except": cc.CONF_SKIP, "mixed": mixed, "pass": not mixed}}
    checks["pass"] = bool(all(v["pass"] for f in ("complete", "invalid_pairs") for v in checks[f].values())
                          and checks["conf"]["pass"])
    if not checks["pass"]:
        return {"checks": checks}

    tests = {(0, 108): "line 2's frozen 108 Plus cases (one process, one Base)", (10, 187): "187 Plus cases",
             (20, 187): "187 Plus cases", (10, 108): "line 2's 108 Plus cases", (20, 108): "line 2's 108 Plus cases"}
    meas = {f"D({d})" + ("/108" if d and n == 108 else ""): {
        "test": f"pi05, Plus, d = {d}, {t}: Gauto - {ARM} (G(k = 1.0)), pp", **boot((1, pr[d, n][0])),
        "invalid_pairs": bad(pr[d, n][1]), "drops": pr[d, n][1]} for (d, n), t in tests.items()}
    for d in (20, 10):  # the trend on the 108 cases common to all three d; T is the journal's main statistic
        meas["T" if d == 20 else f"D({d})-D(0)"] = {
            "test": f"pi05, Plus, line 2's 108 Plus cases: D({d}) - D(0), pp (clusters: the task, its pairs at d = {d} "
                    "and d = 0 in it, resampled jointly)", **boot((1, pr[d, 108][0]), (-1, pr[0, 108][0]))}
    return {"checks": checks, "measurements": {n: e2.rnd(v) for n, v in meas.items()},
            "reading": READING[meas["T"]["lo5_one_sided"] > 0]}  # the raw bound, before rounding


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 kappa ablation checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


# ------------------------------------------------------------------ selftest

def selftest():
    """The plan on the real split: one Plus line per d, VS1.0 alone, --base-prefix, no kbar, line 3's 187 Plus cases
    in its groups (187 episodes: the Bases are cut at the event), ids K_pi05_eval300_plus_d10 / _d20; the bootstrap;
    the reading on a synthetic pod."""
    split = maxwrap.load_split(cs.SPLIT)
    for line, d in LINES.items():
        (ln,) = plan(line, split)
        assert ln["id"] == f"K_{P}_eval300_plus_d{d}" and ln["phase"] == line.lower() and ln["arms"] == [ARM], ln
        assert "--base-prefix" in ln["args"] and "--kbar" not in ln["args"] and cc.n_of(ln) == 187, ln
    # one part is e2.diff_pp's scheme at 8000 resamples, not at its 2000 (which differ here)
    rng = np.random.default_rng(5)
    prs = [(j, f"t{j % 13}", rng.random() < 0.6, rng.random() < 0.4) for j in range(150)]
    with mock.patch.object(e2, "N_BOOT", REPS):
        assert e2.rnd(boot((1, prs))) == e2.rnd(e2.diff_pp([p[1:] for p in prs]))
    assert REPS == 8000 and e2.rnd(boot((1, prs))) != e2.rnd(e2.diff_pp([p[1:] for p in prs]))
    selftest_read(split)
    print("selftest ok")


def selftest_read(split):
    """The reading on a synthetic pod: controls line 2's frozen Plus line and line 3's Plus lines, the K lines. Success
    by the case's place i among the 187 Plus cases in cycles of 4 (q: the cases of the 4 that succeed, by arm and d),
    G-auto failing beyond the first 108 at d 10, 20 (D on the 187 apart from D on the 108); task clusters from fake
    LIBERO names as c1e4v_summary's selftest; one policy seed per case."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    lns = lines(split)
    plus = cv.ids(lns["Kd10"][1], split)
    i = {c: j for j, c in enumerate(plus)}
    cf = {"brain": P, "revision": "r", "precision": "fp32", "s": 10, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01, "R": 360,
          "split": split["sha256"]}
    q = {("Gauto", 0): 2, (ARM, 0): 3, ("Gauto", 10): 2, (ARM, 10): 2, ("Gauto", 20): 3, (ARM, 20): 1}
    win = lambda a, d, c: i[c] % 4 < q.get((a, d), 0) and (i[c] < 108 or a != "Gauto" or d == 0)  # noqa: E731

    def rec(x, win):  # x: lk
        _, b, c, kind, a, d, _, arms = x
        h = sum(map(ord, c))
        s = list(run.MAX_STEPS)[h % 4]
        return {"brain": b, "case_id": c, "kind": kind, "arm": a, "method": cs.method(a), "d": d,
                "policy_seed": 195 + i[c], "conf": cf | {"d": d, "arms": list(arms), "kbar": 0.306 if "Gauto" in arms
                                                        else None, "gpu": "RTX 4090" + " D" * (a == ARM and d > 0)},
                "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None, "e": 30,
                "prefix_ok": None if kind == "base" else True, "replay_miss": None, "skipped_reason": None,
                "success": None if kind == "base" else bool(win(a, d, c))}

    def world(win):
        new = [rec(x, win) for L in LINES for x in sorted(cc.lkeys([lns[L][1]]))]
        return new, [rec(x, win) for L in ("C2d0", "C3d10", "C3d20") for x in sorted(cc.lkeys([lns[L][1]]))]

    def want(d, n, win=win, out=()):  # D(d) by hand: the mean paired difference over the first n cases, out dropped
        cs_ = [c for c in plus[:n] if c not in out]
        return round(100 * sum(win("Gauto", d, c) - win(ARM, d, c) for c in cs_) / len(cs_), 3)

    new, ctrl = world(win)
    rd = lambda n=new, c=ctrl: read(n, c, names, split)  # noqa: E731
    probe = rec(("m", P, plus[0], "dynamic", "VS0.3", 10, False, ("VS0.3",)), win)  # off the plan
    res = rd(new + [probe] + new[:5], ctrl + ctrl[:5])  # [:5]: identical copies
    ch, ms = res["checks"], res["measurements"]
    assert ch["pass"] and ch["off_plan"] == {f"{P}/VS0.3": 1} and ch["read"] == len(new) + len(ctrl), ch
    assert {L: (v["expected"], v["missing"]) for L, v in ch["complete"].items()} == {
        "C2d0": (216, 0), "C3d10": (187, 0), "C3d20": (187, 0), "Kd10": (187, 0), "Kd20": (187, 0)}, ch["complete"]
    assert ch["invalid_pairs"] == {"d0": {"pairs": 108, "invalid": 0, "pass": True},
                                   "d10": {"pairs": 187, "invalid": 0, "pass": True},
                                   "d20": {"pairs": 187, "invalid": 0, "pass": True}}, ch["invalid_pairs"]
    # D(d): -25 (d 0), 0 and 50 on the 108, apart on the 187; T = 50 - (-25) = 75, D(10) - D(0) = 25
    assert list(ms) == ["D(0)", "D(10)", "D(20)", "D(10)/108", "D(20)/108", "T", "D(10)-D(0)"], list(ms)
    got = {n: ms[n]["diff_pp"] for n in ms}
    assert got == {"D(0)": -25, "D(10)": want(10, 187), "D(20)": want(20, 187), "D(10)/108": 0, "D(20)/108": 50,
                   "T": 75, "D(10)-D(0)": 25} and want(10, 187) == -21.39 and want(20, 187) == 18.182, got
    assert [ms[n]["pairs"] for n in ms] == [108, 187, 187, 108, 108, [108, 108], [108, 108]], ms
    nclu = len({(r["suite"], r["task_name"][:len(r["suite"]) + 3]) for r in ctrl if r["case_id"] in plus[:108]})
    assert ms["D(0)"]["clusters"] == ms["T"]["clusters"] == nclu and all(
        ms[n]["invalid_pairs"] == 0 and ms[n]["drops"] == {} for n in list(ms)[:5]), ms
    t = ms["T"]
    assert 0 < t["lo5_one_sided"] and t["lo2.5"] <= t["lo5_one_sided"] <= 75 <= t["hi97.5"], t
    assert res["reading"] == READING[True] == "преимущество закона З1 над лучшим фиксированным k растёт с задержкой"
    assert not {"pass", "outcome"} & {k for v in ms.values() for k in v}, ms
    cc.exits(lambda: rd(new + [new[3] | {"success": not new[3]["success"]}]), 2)  # one key, other content

    # a flat trend: d 20 as d 0 but one case of the 108 to G-auto (T = 100 / 108 > 0): every resample's T >= 0, its
    # lower bounds 0 (the case's task left out of >= 5 % of them; clusters resampled jointly: d 0 and d 20 cancel)
    # -> not shown; D(10) - D(0) = 0 in every resample
    flat = lambda a, d, c: (a == "Gauto") if d == 20 and c == plus[5] else win(a, 0, c)  # noqa: E731
    r = rd(*world(flat))
    t, t10 = r["measurements"]["T"], r["measurements"]["D(10)-D(0)"]
    assert t["diff_pp"] == round(100 / 108, 3) and r["reading"] == READING[False] == "рост с задержкой не показан", r
    assert t["lo2.5"] == t["lo5_one_sided"] == 0 < t["hi97.5"] and t10 == t10 | dict.fromkeys(
        ("diff_pp", "lo2.5", "lo5_one_sided", "hi97.5"), 0.0), (t, t10)
    # three such cases in three tasks: no flipped task in about 4 % of the resamples, so lo2.5 = 0 < lo5: the one-sided
    # bound reads, shown
    tk = lambda c: (sum(map(ord, c)) % 4, sum(map(ord, c)) // 4 % 10)  # noqa: E731  (the case's fake task)
    three = list({tk(c): c for c in plus[:108] if i[c] % 4 == 0}.values())[:3]
    r = rd(*world(lambda a, d, c: (a == "Gauto") if d == 20 and c in three else win(a, 0, c)))
    t = r["measurements"]["T"]
    assert t["diff_pp"] == round(300 / 108, 3) and t["lo2.5"] == 0 < t["lo5_one_sided"] and r["reading"] == READING[
        True], t

    # invalid pairs: a VS1.0 episode with prefix_ok false (d 10), a line 3 Base with another e (d 20, its G-auto too),
    # another seed (d 0): dropped by reason, counted; a missing K record drops as missing, never invalid
    c1, c2, c3, c4 = plus[1], plus[2], plus[3], plus[150]
    edit = lambda r: (r | {"prefix_ok": False} if (r["case_id"], r["d"], r["arm"]) == (c1, 10, ARM) else  # noqa: E731
                      r | {"e": 31} if (r["case_id"], r["d"], r["conf"]["arms"]) == (c2, 20, ["PPCL", "Gauto"]) else
                      r | {"policy_seed": 1} if (r["case_id"], r["d"], r["arm"]) == (c3, 0, "Gauto") else r)
    r = rd([edit(x) for x in new if (x["case_id"], x["d"], x["arm"]) != (c4, 20, ARM)], [edit(x) for x in ctrl])
    ch, ms = r["checks"], r["measurements"]
    assert ch["pass"] and ch["complete"]["Kd20"]["missing"] == 1 and ch["invalid_pairs"] == {
        "d0": {"pairs": 108, "invalid": 1, "pass": True}, "d10": {"pairs": 187, "invalid": 1, "pass": True},
        "d20": {"pairs": 186, "invalid": 1, "pass": True}}, ch
    assert [(ms[n]["drops"], ms[n]["invalid_pairs"], ms[n]["pairs"]) for n in list(ms)[:5]] == [
        ({"seed": 1}, 1, 107), ({"invalid": 1}, 1, 186), ({"e": 1, "missing": 1}, 1, 185), ({"invalid": 1}, 1, 107),
        ({"e": 1}, 1, 107)], ms
    assert [ms[n]["diff_pp"] for n in ("D(0)", "D(10)", "D(20)", "D(20)/108")] == [
        want(0, 108, out=[c3]), want(10, 187, out=[c1]), want(20, 187, out=[c2, c4]), want(20, 108, out=[c2])], ms

    # the checks before reading, each alone
    k20 = [x for x in new if x["d"] == 20 and x["kind"] == "dynamic"]
    nb = lambda n: [x | {"prefix_ok": False} if x["kind"] == "dynamic" and x["d"] == 10 and i[x["case_id"]] < n  # noqa
                    else x for x in new]  # n VS1.0 episodes at d 10 invalid
    for why, kw, ok in (("complete", dict(n=[x for x in new if x not in k20[:2]]), False),  # 2 of 187 VS1.0 missing
                        ("complete", dict(n=[x for x in new if x not in k20[:1]]), True),
                        ("complete", dict(c=[x for x in ctrl if not (x["d"] == 10 and x["arm"] == "Gauto"
                                                                     and i[x["case_id"]] < 2)]), False),
                        ("invalid_pairs", dict(n=nb(4)), False), ("invalid_pairs", dict(n=nb(3)), True),  # of 187
                        ("conf", dict(n=[x | {"conf": x["conf"] | {"t_ramp": 4}} if x["d"] == 10 else x
                                         for x in new]), False)):
        r = rd(**kw)
        c = r["checks"][why]
        assert r["checks"]["pass"] == ok == ("measurements" in r) and (
            c["pass"] if "pass" in c else all(v["pass"] for v in c.values())) == ok, (why, ok, c)
        if not ok:
            cc.exits(lambda: finish(r), 3)
    r = rd(c=[x | {"prefix_ok": False} if x["d"] == 0 and x["arm"] == "Gauto" and i[x["case_id"]] < 3 else x
              for x in ctrl])  # 3 of the 108 at d 0: above 2 %
    assert r["checks"]["invalid_pairs"]["d0"] == {"pairs": 108, "invalid": 3, "pass": False} and "reading" not in r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_K.jsonl: checks, D(d), T, the reading")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--controls", nargs="+", default=[f"{cc.OUT}/grid_all_C.jsonl"], metavar="JSONL",
                    help="the reading: controls line 2's frozen Plus file and line 3's Plus files")
    ap.add_argument("--out", help="the reading: JSON too")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f) if "kind" in x]  # noqa: E731
    a.read = a.read[a.read[:1] == ["read"]:]  # "read" before the files is optional
    if bool(a.read) + a.plan + a.selftest != 1:
        ap.error("one mode: --plan, --selftest, or [read] JSONL...")
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    if a.selftest:
        selftest()
    elif a.plan:
        print(cc.tsv(plan(a.line)))
    else:
        finish(read(load(a.read), load(a.controls), cs.libero_names(), maxwrap.load_split(cs.SPLIT)), a.out)
