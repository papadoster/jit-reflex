"""C1-E4 cell О12: PPCL on the OFT brains at 12 cm (docs/superpowers/specs/2026-10-20-c1-e4-controls-design.md, its
journal "2026-10-22 — решения владельца после чтения: граница латча на M1 и PPCL на OFT-мозгах (до данных клетки О12)",
item 2), on pod C after a clean mgrid (scripts/gpu_c1e4m.sh POD=C TAIL=1 ogrid): the runner plan per line and the
reading. Reuses scripts/c1e4c_summary.py (mlines, KBAR, lk, tsv), scripts/c1e4v_summary.py (unique, pairs, est) and
scripts/c1e4b_summary.py (est: hi95 of the same resamples); part 2's records (results/c1-e4/part2) are read only; run
from the repository root. R = results/c1-e4/controls/pod_C:
    python scripts/c1e4o_summary.py --plan --line L
        L: O12oftplus O12oft, the runner lines as c1e4c_summary --plan (phase: L in lower case)
    python scripts/c1e4o_summary.py [read] R/grid_all_O.jsonl [--p2 results/c1-e4/part2] [--out R/summary_O.json]
        the checks before reading, then О1 and О2 per brain (an estimate and its 95 % CI, no threshold) and the report;
        a failed check prints only the checks, exit 3. Needs hf-libero (the task clusters)
    python scripts/c1e4o_summary.py --selftest
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4b_summary as cb  # noqa: E402
import c1e4c_summary as cc  # noqa: E402
import c1e4m_summary as cs  # noqa: E402  (puts src/ on sys.path)
import c1e4v_summary as cv  # noqa: E402
from c1e4m_summary import O, OP, TOL, e2, maxwrap, run  # noqa: E402

ARMS = ("PPCL", "PPC", "Gauto")  # the journal: the Base cut at the event, then these in one process
LINES = {"O12oftplus": OP, "O12oft": O}  # the journal's order: OFT+, then OFT
MEAS = {"О1": ("Gauto", "PPCL"), "О2": ("PPCL", "PPC")}  # the journal's measurements, x - y, both on the pod
# part 2's files of each brain (pod B, d 0): "none" through the pod, its own PPC - none and Gauto - PPC, distance_m
P2F = {OP: "pod_B/M1oftplus_oftplus_eval_{}_d0.jsonl", O: "pod_B/M4oft_oft_eval300_{}_d0.jsonl"}
# LIBERO-MAX's scenario id ends -d<draw>: draw 1 is 0.12 m, draw 0 0.06 m (benchmark/max8000/libero_max_8000.json,
# target_relocation); the reading checks it against part 2's distance_m (check "cm12")
FAR = re.compile(r"-d1-p\d+$")
OWNER = "владельцу, без выводов"  # the journal's worst-case bound rule when B >= the gap


def cases(split):
    """The journal's 12 cm cases: eval300's (the first 300 of eval) drawn at 0.12 m, in the set's order; 157 (99 Plus,
    58 PRO), the same for both brains."""
    c = [x for x in run.set_ids(split, "eval300") if FAR.search(x)]
    n = {s: len(v) for s, v in cs.sources(c).items()}
    assert (len(c), n["plus"], n["pro"]) == (157, 99, 58), (len(c), n)
    return c


def plan(line, split=None):
    """The journal's lines, phase line.lower(), in c1e4c_summary.mlines's format: per source PPCL, PPC, Gauto on the
    brain's 12 cm cases at d 0 with --base-prefix and part 2's kbar, the runner's groups of 6 in the set's order."""
    split = split or maxwrap.load_split(cs.SPLIT)
    b = LINES[line]
    return cc.mlines(line.lower(), "O", b, "eval300", ARMS, split, cases=set(cases(split)), kbar=cc.KBAR["m", b])


def load_p2(root=cc.P2):
    """Part 2's records of both brains (P2F, Plus and PRO)."""
    return [x for f in P2F.values() for s in ("plus", "pro") for x in e2.load(Path(root) / f.format(s)) if "kind" in x]


def read(new, p2, names, split):
    """The journal's checks before reading (each line's Dynamic episodes present on >= 99 %, its cut Bases not counted;
    the 157 cases part 2's distance_m 0.12 among eval300, per brain; invalid pairs <= 2 % per measurement, a missing
    record not one; conf equal per brain but cc.CONF_SKIP (arms, gpu, kbar) across the read records, part 2's too;
    every G-auto record's kbar part 2's, cc.KBAR), then О1 = Gauto - PPCL and О2 = PPCL - PPC per brain and the report.
    new: grid_all_O.jsonl; p2: load_p2(). Pairs: cv.pairs, one Dynamic episode per side and case, the same seed, both
    valid, the two Bases' e equal (on the pod one Base, through the pod part 2's against the pod's). Records off the
    plan are counted, never read."""
    c157, e300 = cases(split), set(run.set_ids(split, "eval300"))
    lns = {L: plan(L, split) for L in LINES}
    want = {x: L for L, ln in lns.items() for x in cc.lkeys(ln)}
    by = cv.unique(new, cc.lk)
    used = {x: r for x, r in by.items() if x in want and not r.get("skipped_reason")}
    comp = {}
    for L, ln in lns.items():  # the Dynamic records: >= 99 % present per line (cc.n_of's episodes)
        exp = {x for x in cc.lkeys(ln) if x[3] == "dynamic"}
        miss = len(exp - used.keys())
        comp[L] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    old = [r for r in cv.unique(p2, cc.lk).values() if r["case_id"] in e300 and not r.get("skipped_reason")]
    far = {b: {r["case_id"] for r in old if r["brain"] == b and r["distance_m"] == 0.12} for b in LINES.values()}
    side = {"pod": list(used.values()),  # part 2's read records: the 157, "none", PPC, Gauto and the Base
            "p2": [r for r in old if r["case_id"] in far[r["brain"]] and r["arm"] in ("none", "PPC", "Gauto")]}
    of = lambda s, b, kind, a: {r["case_id"]: r for r in side[s]  # noqa: E731
                                if r["brain"] == b and r["kind"] == kind and r["arm"] == a}
    one = {r["case_id"]: r for r in side["p2"] + side["pod"]}
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    prs = lambda b, x, y, sx="pod", sy="pod": cv.pairs(  # noqa: E731
        of(sx, b, "dynamic", x), of(sy, b, "dynamic", y), of(sx, b, "base", "none"), of(sy, b, "base", "none"), clu,
        c157)
    pr = {(m, b): prs(b, x, y) for b in LINES.values() for m, (x, y) in MEAS.items()}
    bad = lambda drop: sum(v for w, v in drop.items() if w != "missing")  # noqa: E731  (missing: the completeness's)
    conf, kbar = defaultdict(set), Counter()
    for r in side["pod"] + side["p2"]:
        conf[r["brain"]].add(json.dumps({f: v for f, v in r["conf"].items() if f not in cc.CONF_SKIP}, sort_keys=True))
        kbar[f"{r['brain']}/{r['arm']}"] += r["arm"] == "Gauto" and r["conf"]["kbar"] != cc.KBAR["m", r["brain"]]
    mixed, kbar = {g: sorted(v) for g, v in conf.items() if len(v) > 1}, {g: n for g, n in kbar.items() if n}
    checks = {"records": len(new), "read": len(used),
              "skipped": sum(bool(r.get("skipped_reason")) for x, r in by.items() if x in want),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for x, r in by.items() if x not in want)),
              "complete": comp, "cm12": {b: {"part2": len(f), "pass": f == set(c157)} for b, f in far.items()},
              "invalid_pairs": {f"{m}/{b}": {"pairs": len(p) + bad(d), "invalid": bad(d),
                                             "pass": bad(d) <= 0.02 * (len(p) + bad(d)) + TOL}
                                for (m, b), (p, d) in pr.items()},
              "conf": {"except": cc.CONF_SKIP, "mixed": mixed, "kbar_not_part2s": kbar, "pass": not mixed and not kbar}}
    checks["pass"] = bool(all(v["pass"] for f in ("complete", "cm12", "invalid_pairs") for v in checks[f].values())
                          and checks["conf"]["pass"])
    if not checks["pass"]:
        return {"checks": checks}

    meas = {f"{m}/{b}": {"test": f"{b}, M1 12 cm, d = 0, the 157 cases: {MEAS[m][0]} - {MEAS[m][1]}, pp",
                         **e2.rnd(cb.est(p)), "invalid_pairs": bad(d), "drops": d} for (m, b), (p, d) in pr.items()}
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    est = lambda p, d, **kw: {"drops": d, **cv.est(p)} | kw  # noqa: E731
    rep = defaultdict(dict)
    for b in LINES.values():  # the journal's "только отчёт"
        rep["PPCL-none"][b] = est(*prs(b, "PPCL", "none", sy="p2"), **{"через под": True})
        rep["PPC-none"][b] = est(*prs(b, "PPC", "none", sy="p2"), **{"через под": True})
        rep["part2/PPC-none"][b] = est(*prs(b, "PPC", "none", "p2", "p2"))
        gp = prs(b, "Gauto", "PPC", "p2", "p2")
        rep["part2/Gauto-PPC"][b] = est(*gp)
        lt = [r["latch"] for r in of("pod", b, "dynamic", "PPCL").values()]
        rep["latch"][b] = {"n": len(lt), "latched_share": mean([x["chunks"] > 0 for x in lt])} | {
            f"{f}_per_episode": mean([x[f] for x in lt]) for f in ("chunks", "trig", "resets")}
        # the journal's worst-case bound (2026-10-20 "дополнения владельца"): B, the latched share in pp, against part
        # 2's Gauto - PPC on the same 157
        B, gap = 100 * float(np.mean([x["chunks"] > 0 for x in lt])), e2.diff_pp([q[1:] for q in gp[0]])["diff_pp"]
        rep["bound"][b] = {"B_pp": round(B, 3), "gap_part2_Gauto-PPC_pp": round(gap, 3), "B<gap": B < gap - TOL,
                           "sentence": f"латч сработал в {B:.1f}% эпизодов, максимально возможное влияние {B:.1f} п.п.,"
                                       " вывод не меняется" if B < gap - TOL else OWNER}
        for a in ARMS:  # brain calls per episode (c1e4c_summary's calls report)
            g = list(of("pod", b, "dynamic", a).values())
            post = [sum(t >= r["e"] for t, _ in r["plans_in"]) for r in g if r.get("e") is not None]
            rep["calls"][f"{b}/{a}"] = {"n": len(g), "sched": mean([r["calls_sched"] for r in g]),
                                        "trig": mean([r["calls_trig"] for r in g]), "plans_after_event": mean(post)}
        nb, ob = of("pod", b, "base", "none"), of("p2", b, "base", "none")
        rep["base_e_part2"][b] = {"equal": sum(c in ob and ob[c]["e"] == r["e"] for c, r in nb.items()),
                                  "total": len(nb)}
    rep["gpu"] = sorted({str(r["conf"].get("gpu")) for r in side["pod"]})
    return {"checks": checks, "measurements": meas, "report": dict(rep)}


def finish(res, out=None):
    """Print the reading (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 cell O12 checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, nothing read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


# ------------------------------------------------------------------ selftest

def selftest():
    """The plan on the real split: per brain a Plus and a PRO line, PPCL, PPC, Gauto with part 2's kbar and
    --base-prefix on the 12 cm cases of eval300 in the set's order (99 + 58, three episodes each: the Bases are cut
    at the event); a broken 12 cm filter refused; the reading on a synthetic pod."""
    split = maxwrap.load_split(cs.SPLIT)
    c157 = cases(split)
    for line, b in LINES.items():
        lns = plan(line, split)
        assert [(ln["id"], ln["phase"], ln["arms"], cc.n_of(ln)) for ln in lns] == [
            (f"O_{b}_eval300_{s}_d0", line.lower(), list(ARMS), n) for s, n in (("plus", 297), ("pro", 174))], lns
        for ln in lns:
            ids = [c for c in c157 if c.startswith("pro-") == (ln["source"] == "pro")]
            assert ln["args"] == (f"--brain {b} --source {ln['source']} --set eval300 --arms=PPCL,PPC,Gauto --d 0 "
                                  f"--cases {','.join(ids)} --kbar {cc.KBAR['m', b]} --base-prefix"), ln["args"]
            assert cv.ids(ln, split) == ids and all("-d1-" in c for c in ids), ln
        assert cc.tsv(lns).splitlines()[0].split("\t")[:6] == [line.lower(), "c1e4m", b, "plus", lns[0]["id"], "297"]
    with mock.patch.object(sys.modules[__name__], "FAR", re.compile(r"-d0-p\d+$")):  # the 6 cm draw: 143 cases
        try:
            plan("O12oft", split)
            raise SystemExit("a 12 cm filter with 143 cases was accepted")
        except AssertionError:
            pass
    selftest_read(split)
    print("selftest ok")


def selftest_read(split):
    """The reading on a synthetic pod and part 2: success by the case's place i among the 157 in cycles of 4 (q: the
    places of the 4 that succeed, by side, brain and arm), so the differences are exact; part 2's records on all of
    eval300 (distance_m by the draw); the latch on every 10th (OFT+) or every 2nd place (OFT: B >= its gap once part
    2's G-auto is lowered); task clusters from fake LIBERO names as c1e4k_summary's selftest; one policy seed per
    case."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    c157 = cases(split)
    i = {c: j for j, c in enumerate(c157)}
    e300 = run.set_ids(split, "eval300")
    q = {("pod", OP, "PPCL"): 2, ("pod", OP, "PPC"): 1, ("pod", OP, "Gauto"): 3, ("p2", OP, "none"): 1,
         ("p2", OP, "PPC"): 1, ("p2", OP, "Gauto"): 2, ("pod", O, "PPCL"): 1, ("pod", O, "PPC"): 2,
         ("pod", O, "Gauto"): 2, ("p2", O, "none"): 2, ("p2", O, "PPC"): 0, ("p2", O, "Gauto"): 3}
    win = lambda s, b, a, c: i.get(c, 0) % 4 < q.get((s, b, a), 0)  # noqa: E731
    every = {OP: 10, O: 2}

    def rec(s, b, c, kind, a, arms, win):
        h = sum(map(ord, c))
        su = list(run.MAX_STEPS)[h % 4]
        cf = {"brain": b, "revision": "r", "precision": "bf16", "s": 8, "d": 0, "kbar": cc.KBAR["m", b], "t_ramp": 5,
              "k_p": 1.0, "tau_k": 0.01, "R": 256, "arms": list(arms), "split": split["sha256"]}
        r = {"brain": b, "case_id": c, "kind": kind, "arm": a, "method": a, "d": 0, "policy_seed": 195,
             "conf": cf | ({"gpu": "RTX 4090"} if s == "pod" else {}), "suite": su,
             "task_name": f"{su}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None,
             "distance_m": 0.12 if FAR.search(c) else 0.06, "e": 30, "prefix_ok": None if kind == "base" else True,
             "replay_miss": None, "skipped_reason": None, "success": None if kind == "base" else bool(win(s, b, a, c)),
             "calls_sched": 30 + (a == "PPCL"), "calls_trig": 2 * (a != "none"), "plans_in": [[0, 0], [24, 24],
                                                                                               [40, 40]]}
        if a == "PPCL":
            on = i[c] % every[b] == 0
            r["latch"] = {"chunks": 4 * on, "trig": on, "resets": 0}
        return r

    def world(win=win):
        new = [rec("pod", x[1], x[2], x[3], x[4], x[7], win) for L in LINES for x in sorted(cc.lkeys(plan(L, split)))]
        p2 = [rec("p2", b, c, k, a, ("none", "G", "Gauto", "PPC"), win) for b in LINES.values() for c in e300
              for k, a in [("dynamic", a) for a in ("none", "G", "Gauto", "PPC")] + [("base", "none")]]
        return new, p2

    def want(b, x, y, sx="pod", sy="pod", out=()):  # a difference by hand: the mean over the 157 cases, out dropped
        cs_ = [c for c in c157 if c not in out]
        return round(100 * sum(win(sx, b, x, c) - win(sy, b, y, c) for c in cs_) / len(cs_), 3)

    new, p2 = world()
    rd = lambda n=new, p=p2: read(n, p, names, split)  # noqa: E731
    probe = rec("pod", O, c157[0], "dynamic", "VS0.3", ("VS0.3",), win)  # off the plan
    res = rd(new + [probe] + new[:5], p2 + p2[:5])  # [:5]: identical copies
    ch, ms, rp = res["checks"], res["measurements"], res["report"]
    assert ch["pass"] and ch["off_plan"] == {f"{O}/VS0.3": 1} and ch["read"] == len(new) == 2 * 4 * 157, ch
    assert ch["complete"] == {L: {"expected": 471, "missing": 0, "pass": True} for L in LINES}, ch["complete"]
    assert ch["cm12"] == {b: {"part2": 157, "pass": True} for b in LINES.values()}, ch["cm12"]
    assert ch["invalid_pairs"] == {f"{m}/{b}": {"pairs": 157, "invalid": 0, "pass": True}
                                   for b in LINES.values() for m in MEAS}, ch["invalid_pairs"]
    # exact differences, the arms in the journal's order (a swap flips each sign: every О is nonzero here)
    got = {n: v["diff_pp"] for n, v in ms.items()}
    assert got == {f"{m}/{b}": want(b, *MEAS[m]) for b in LINES.values() for m in MEAS} and 0 not in got.values(), got
    assert got == {"О1/oftplus": 24.841, "О2/oftplus": 24.841, "О1/oft": 24.841, "О2/oft": -24.841}, got
    assert all(v["pairs"] == 157 and v["drops"] == {} and v["lo2.5"] <= v["lo5_one_sided"] <= v["diff_pp"]
               <= v["hi95_one_sided"] <= v["hi97.5"] for v in ms.values()), ms
    assert not {"pass", "outcome"} & {k for v in ms.values() for k in v}, ms
    for b in LINES.values():  # through the pod and part 2's own on the same 157
        assert [rp[k][b]["diff_pp"] for k in ("PPCL-none", "PPC-none", "part2/PPC-none", "part2/Gauto-PPC")] == [
            want(b, "PPCL", "none", sy="p2"), want(b, "PPC", "none", sy="p2"), want(b, "PPC", "none", "p2", "p2"),
            want(b, "Gauto", "PPC", "p2", "p2")] and rp["PPCL-none"][b]["через под"], rp
        assert rp["base_e_part2"][b] == {"equal": 157, "total": 157} and rp["calls"][f"{b}/PPCL"] == {
            "n": 157, "sched": 31.0, "trig": 2.0, "plans_after_event": 1.0}, rp
    # the worst-case bound: OFT+ latched on 16 of 157 (10.2 pp) against part 2's gap 24.8: the journal's sentence; OFT
    # on 79 (50.3 pp) against 75.2: the sentence too; OFT's gap moved to 25.5 below: B >= it, the owner's
    assert rp["latch"][OP] == {"n": 157, "latched_share": 0.1019, "chunks_per_episode": 0.4076,
                               "trig_per_episode": 0.1019, "resets_per_episode": 0.0}, rp["latch"]
    assert rp["bound"][OP] == {"B_pp": 10.191, "gap_part2_Gauto-PPC_pp": 24.841, "B<gap": True, "sentence":
                               "латч сработал в 10.2% эпизодов, максимально возможное влияние 10.2 п.п., вывод не "
                               "меняется"}, rp["bound"]
    assert rp["bound"][O]["B_pp"] == 50.318 and rp["bound"][O]["B<gap"], rp["bound"]
    w2 = lambda s, b, a, c: (i.get(c, 0) % 4 < 1) if (s, b, a) == ("p2", O, "Gauto") else win(s, b, a, c)  # noqa
    r = read(*world(w2), names, split)
    assert r["report"]["bound"][O] == {"B_pp": 50.318, "gap_part2_Gauto-PPC_pp": 25.478, "B<gap": False,
                                       "sentence": OWNER}, r["report"]["bound"]
    cc.exits(lambda: rd(new + [new[3] | {"success": not new[3]["success"]}]), 2)  # one key, other content

    # through the pod: part 2's Base with another e drops the pair (e), part 2's "none" with another seed (seed); the
    # pod's pairs keep them; a pod Base with another e than part 2's counts in base_e_part2 and drops there too
    c1, c2, c3 = c157[1], c157[2], c157[3]
    edit = lambda r: (r | {"e": 31} if (r["case_id"], r["kind"], r["brain"]) == (c1, "base", O) else  # noqa: E731
                      r | {"policy_seed": 1} if (r["case_id"], r["kind"], r["brain"]) == (c2, "dynamic", O)
                      and r["arm"] == "none" else r)
    pod3 = lambda r: r | {"e": 29} if (r["case_id"], r["brain"]) == (c3, O) else r  # noqa: E731  (its Base and arms)
    r = rd([pod3(x) for x in new], [edit(x) for x in p2])
    rr = r["report"]
    assert r["checks"]["pass"] and all(r["measurements"][f"{m}/{O}"]["drops"] == {} for m in MEAS), r
    for k, x in (("PPCL-none", "PPCL"), ("PPC-none", "PPC")):
        assert rr[k][O]["drops"] == {"e": 2, "seed": 1} and rr[k][O]["pairs"] == 154 and rr[k][O]["diff_pp"] == \
            want(O, x, "none", sy="p2", out=(c1, c2, c3)), (k, rr[k][O])
    assert rr["part2/PPC-none"][O]["drops"] == {"seed": 1} and rr["part2/Gauto-PPC"][O]["drops"] == {}, rr
    assert rr["base_e_part2"][O] == {"equal": 155, "total": 157} and rr["PPCL-none"][OP]["drops"] == {}, rr

    # the checks before reading, each alone
    dyn = lambda b, a: [x for x in new if x["brain"] == b and x["arm"] == a]  # noqa: E731
    inv = lambda n: [x | {"prefix_ok": False} if x in dyn(O, "PPCL")[:n] else x for x in new]  # noqa: E731
    near = next(c for c in e300 if c not in i)  # a 6 cm case
    dist = lambda b, c, v: [x | {"distance_m": v} if (x["brain"], x["case_id"]) == (b, c) else x  # noqa: E731
                            for x in p2]  # every part 2 record of the case
    for why, kw, ok in (("complete", dict(n=[x for x in new if x["brain"] != O]), False),  # a line not there at all
                        ("complete", dict(n=[x for x in new if x not in dyn(OP, "PPC")[:5]]), False),  # 5 of 471
                        ("complete", dict(n=[x for x in new if x not in dyn(OP, "PPC")[:4]]), True),
                        ("invalid_pairs", dict(n=inv(4)), False), ("invalid_pairs", dict(n=inv(3)), True),  # of 157
                        ("conf", dict(n=[x | {"conf": x["conf"] | {"kbar": 0.2}} if x is dyn(OP, "Gauto")[9] else x
                                         for x in new]), False),
                        ("conf", dict(p=[x | {"conf": x["conf"] | {"kbar": 0.2}} if x["arm"] == "Gauto" else x
                                         for x in p2]), False),
                        ("conf", dict(n=[x | {"conf": x["conf"] | {"kbar": 0.2}} if x["arm"] == "PPC" else x
                                         for x in new]), True),  # kbar: G-auto's only
                        ("conf", dict(n=[x | {"conf": x["conf"] | {"t_ramp": 4}} if x["brain"] == O else x
                                         for x in new]), False),
                        ("conf", dict(p=[x | {"conf": x["conf"] | {"R": 360}} if x["brain"] == OP else x
                                         for x in p2]), False),  # through the pod: part 2's conf too
                        ("cm12", dict(p=dist(OP, c157[7], 0.06)), False), ("cm12", dict(p=dist(O, near, 0.12)), False),
                        ("cm12", dict(p=[x for x in p2 if x["brain"] != O]), False)):  # part 2 of a brain missing
        r = rd(**kw)
        c = r["checks"][why]
        assert r["checks"]["pass"] == ok == ("measurements" in r) and (
            c["pass"] if "pass" in c else all(v["pass"] for v in c.values())) == ok, (why, ok, c)
        if not ok:
            cc.exits(lambda: finish(r), 3)
    r = rd(n=inv(3))
    assert r["checks"]["invalid_pairs"][f"О1/{O}"] == {"pairs": 157, "invalid": 3, "pass": True} and r[
        "measurements"][f"О2/{O}"]["drops"] == {"invalid": 3} and r["measurements"][f"О2/{O}"]["diff_pp"] == want(
        O, "PPCL", "PPC", out=[x["case_id"] for x in dyn(O, "PPCL")[:3]]), r["checks"]
    r = rd(n=[x for x in new if x not in dyn(OP, "PPC")[:1]])  # a missing record drops as missing, never invalid
    assert r["measurements"][f"О2/{OP}"]["drops"] == {"missing": 1} and r["checks"]["invalid_pairs"][
        f"О2/{OP}"] == {"pairs": 156, "invalid": 0, "pass": True}, r


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("read", nargs="*", metavar="[read] JSONL", help="R/grid_all_O.jsonl: checks, О1, О2, the report")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--p2", default=str(cc.P2), help="part 2's records (read only): pod_B's M1 OFT+ and M4 OFT files")
    ap.add_argument("--out", help="the reading: JSON too")
    a = ap.parse_args()
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
        finish(read([x for f in a.read for x in e2.load(f) if "kind" in x], load_p2(a.p2), cs.libero_names(),
                    maxwrap.load_split(cs.SPLIT)), a.out)
