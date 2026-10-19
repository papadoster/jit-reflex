"""C1-E4 controls (PPCL, VS): the probe and its П/С mode (§7.1), the k selection (§5), the runner plan per line
(§6, §7.2, §12), the latched list (§7.2) and the forecast (§12), on the JSON lines of scripts/c1e4m_run.py
(LIBERO-MAX) and scripts/c1e4_run.py (bench). Spec docs/superpowers/specs/2026-10-20-c1-e4-controls-design.md;
part 2's records (results/c1-e4/part2) are read only. Pure numpy, the LeRobot env; run from the repository root.
R = results/c1-e4/controls/pod_C (scripts/gpu_c1e4m.sh POD=C):
    python scripts/c1e4c_summary.py --probe-plan          the probes' runner lines (phase cprobe)
    python scripts/c1e4c_summary.py --probe R/P1_*.jsonl R/P2_*.jsonl R/P3_*.jsonl [--out R/mode.json]
    python scripts/c1e4c_summary.py --select pi05 R/C1_*.jsonl [--out R/k_select.json]   (exit 2: line incomplete)
    python scripts/c1e4c_summary.py --plan --line L [--mode R/mode.json] [--k R/k_select.json]   L: smoke 1-8 c5
        the line's runner lines "phase<TAB>runner<TAB>brain<TAB>source<TAB>id<TAB>n<TAB>args", phase c<L>, n = its
        Dynamic episodes; exit 3 when the line needs a mode or k not there yet. c5 (mode С): --records R/C2_*.jsonl
        R/C3_*.jsonl R/C7_*.jsonl, PPC where PPCL latched; writes latched_bench.json next to mode.json
    python scripts/c1e4c_summary.py --latched R/C2_*.jsonl ...
    python scripts/c1e4c_summary.py --forecast R/tempo.txt [--done R/C[0-9]*.jsonl] [--mode ..] [--k ..]
        hours and $ per line; without mode or k the most expensive case (mode С, kπ != 1.0, kO != kπ)
    python scripts/c1e4c_summary.py --smoke-check R/Csmoke_pi05*.jsonl   exit 0, or 3
    python scripts/c1e4c_summary.py --selftest
"""

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4m_summary as cs  # noqa: E402  (key, mindex, bindex, invalid, mprs, bprs, sources; puts src/ on sys.path)
from c1e4m_summary import O, OP, P, e2, maxwrap, orx, run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs/superpowers/specs/2026-10-20-c1-e4-controls-design.md"
P2 = REPO / "results/c1-e4/part2"  # part 2's records, read only
OUT = "results/c1-e4/controls/pod_C"
VS = ("VS0.1", "VS0.3", "VS1.0")  # gauto.GAgent.VS, the candidates in the tie order of spec §5: 0.3, 0.1, 1.0
TIE = ("VS0.3", "VS0.1", "VS1.0")
TIE_M = 1e-4  # m: candidates within 0.1 mm of the best tie (spec §5)
KBAR = {("m", "pi05"): 0.306, ("m", "oftplus"): 0.121, ("m", "oft"): 0.088, ("b", "pi05"): 0.324}  # part 2's conf
RATE = 0.37  # $/h, card + disk (spec §12)
SAVED = {  # spec §7: part 2's files the probes repeat (the first of each) and the comparisons read
    "pi05_lmax": ["pod_A/M1M2_pi05_eval300_plus_d0.jsonl", "pod_A/M1M2_pi05_eval300_pro_d0.jsonl",
                  "pod_A/D_pi05_eval300_plus_d10.jsonl", "pod_A/D_pi05_eval300_pro_d10.jsonl",
                  "pod_A/D_pi05_eval300_plus_d20.jsonl", "pod_A/D_pi05_eval300_pro_d20.jsonl"],
    "oftplus_lmax": ["pod_B/M1oftplus_oftplus_eval_plus_d0.jsonl", "pod_B/M1oftplus_oftplus_eval_pro_d0.jsonl"],
    "pi05_bench": ["pod_A/Boracle_pi05.jsonl", "pod_A/Bnoisy_pi05.jsonl"],
}
PSET = {"pi05_lmax": "eval300", "oftplus_lmax": "eval"}  # spec §7.1: the --set of the probes' part 2 files
T0 = "{}:{}".format(*orx.TASKS[0])  # the bench probe's and smoke's task
EPS = [(s, t, i) for s, t in orx.TASKS for i in range(10)]  # the bench grid: 26 tasks x inits 0-9
LINES = ("smoke", "1", "2", "3", "4", "5", "6", "7", "8", "c5")
CORE = LINES[1:8]  # spec §12: lines 1-7; 8 is the tail
TEMPO = ("lmax_pi05", "lmax_oftplus", "lmax_oft", "bench_pi05")
PPCL_X = 1.25  # spec §12: PPCL calls more often, +25 % to the tempo
# a LIBERO-MAX Base with --base-prefix, as a share of a full Base episode: cut / full Base batch duration, part 2
# records, review of 184b7d5 (spec §12 says 0.3 flat)
CUT = {(P, "plus"): 0.31, (P, "pro"): 0.44, (OP, "plus"): 0.5, (OP, "pro"): 0.5, (O, "plus"): 0.5, (O, "pro"): 0.5}


def mlines(phase, tag, brain, sset, arms, split, d=0, cases=None, kbar=None, prefix=True):
    """LIBERO-MAX runner lines, one per scene source with cases, id {tag}_{brain}_{set}_{source}_d{d}: cases (a subset
    of the set) in the set's order, the runner's; prefix: --base-prefix (every grid line; never the probe, whose Base
    act_hash is compared with part 2's full one). keys: cs.key of every record the line writes, its Bases too."""
    out = []
    for src, ids in cs.sources([c for c in run.set_ids(split, sset) if cases is None or c in cases]).items():
        args = f"--brain {brain} --source {src} --set {sset} --arms={','.join(arms)} --d {d}"
        if cases is not None:
            args += f" --cases {','.join(ids)}"
        if kbar is not None:
            args += f" --kbar {kbar}"
        if prefix:
            args += " --base-prefix"
        if ids:
            out.append({"phase": phase, "runner": "c1e4m", "brain": brain, "source": src, "arms": list(arms),
                        "id": f"{tag}_{brain}_{sset}_{src}_d{d}", "args": args, "prefix": prefix,
                        "keys": {("m", brain, c, kd, a, d, False) for c in ids
                                 for kd, a in [("base", "none")] + [("dynamic", a) for a in arms]}})
    return out


def blines(phase, ident, arms, cells, eps_arg, eps, kbar=None):
    """The bench line (pi0.5, kind move): eps_arg selects the episodes eps (suite, task, init)."""
    args = f"--brain {P} --kinds move --cells {','.join(cells)} --methods {','.join(arms)} {eps_arg}"
    if kbar is not None:
        args += f" --kbar {kbar}"
    return [{"phase": phase, "runner": "c1e4", "brain": P, "source": "-", "arms": list(arms), "id": ident,
             "args": args, "prefix": False, "keys": cs.bench_keys(P, "move", cells, arms, eps)}]


def n_of(ln):
    """The line's full episodes: a Base cut at its event (--base-prefix) is not one."""
    return sum(not (ln["prefix"] and x[0] == "m" and x[3] == "base") for x in ln["keys"])


def tsv(lns):
    """run_phase's format (scripts/gpu_c1e4m.sh): phase, runner, brain, source, id, n, args."""
    return "\n".join("\t".join(map(str, (ln["phase"], ln["runner"], ln["brain"], ln["source"], ln["id"], n_of(ln),
                                          ln["args"]))) for ln in lns)


def probe_lines(saved, split):
    """spec §7.1: the probes repeat part 2's runs exactly: the first 12 Plus cases (two groups of 6) of its file with
    its arms in their order and its kbar (the batches depend on them), and the bench's first task. saved: {probe:
    the records of its SAVED file, the first}."""
    out = []
    for i, name in enumerate(SAVED, 1):
        rs = [r for r in saved[name] if "kind" in r]
        conf = rs[0]["conf"]
        if name in PSET:
            ids = cs.sources(run.set_ids(split, PSET[name]))["plus"][:12]
            if list(dict.fromkeys(r["case_id"] for r in rs))[:12] != ids:  # part 2 wrote each case's lines together
                print(f"warning: the first 12 cases of {SAVED[name][0]} are not the set's first 12 Plus ids: the probe "
                      "runs the set's (the runner's order)", file=sys.stderr)
            ln = mlines("cprobe", f"P{i}", rs[0]["brain"], PSET[name], conf["arms"], split, cases=ids,
                        kbar=conf["kbar"], prefix=False)
        else:
            ln = blines("cprobe", f"P{i}_{P}", cs.B_ARMS, ("A", "A20"), f"--tasks {T0} --inits 0-9",
                        [x for x in EPS if x[:2] == orx.TASKS[0]], conf["kbar"])
        assert len(ln) == 1, ln
        out.append(ln[0] | {"name": name})
    return out


def probe(new, saved, lns):
    """spec §7.1: per probe, the act_hash of every planned episode against part 2's record of the same key (cs.key:
    brain, case, kind, arm, d; bench brain, cell, arm, suite, task, init). Mode П iff all planned episodes are in both
    and every hash is equal; else С. New records only (conf "gpu"; part 2's have none). LIBERO-MAX: Base e too
    (reported, not a rule)."""
    out = {}
    for ln in lns:
        a = {cs.key(r): r for r in new if "kind" in r and "gpu" in r["conf"] and cs.key(r) in ln["keys"]}
        b = {cs.key(r): r for r in saved[ln["name"]] if "kind" in r and cs.key(r) in ln["keys"]}
        both = a.keys() & b.keys()
        eq = sum(a[x].get("act_hash") is not None and a[x].get("act_hash") == b[x].get("act_hash") for x in both)
        res = {"mode": "P" if eq == len(both) == len(ln["keys"]) else "S", "equal": eq, "total": len(both),
               "missing": len(ln["keys"]) - len(both)}
        if ln["runner"] == "c1e4m":
            base = [x for x in both if x[3] == "base"]
            res |= {"e_equal": sum(a[x].get("e") == b[x].get("e") for x in base), "e_total": len(base)}
        out[ln["name"]] = res | {"gpu": sorted({r["conf"]["gpu"] for r in a.values()})}
    return out


def select(recs, brain, split):
    """spec §5: k of one brain from its M0 line (VS0.1, VS0.3, VS1.0 on the 60 calibration cases, d = 0). A case
    counts when all three Dynamic records are valid (pair kept, event, not skipped) and have a grasp miss; the score is
    the mean grasp_miss_m over those cases, the lowest wins, candidates within TIE_M of it tie and go by TIE. Exit 2
    when a calibration case has no Base record (the line is incomplete) or no case counts."""
    rs = [r for r in recs if "kind" in r and r["brain"] == brain]
    cal = run.set_ids(split, "calib")
    if short := set(cal) - {r["case_id"] for r in rs if r["kind"] == "base"}:
        print(f"!!! {brain}: {len(short)} of {len(cal)} calibration cases without a Base record", file=sys.stderr)
        raise SystemExit(2)
    dyn = {(r["case_id"], r["arm"]): r for r in rs if r["kind"] == "dynamic"}
    ok = lambda r: (r is not None and not cs.invalid(r) and r.get("e") is not None  # noqa: E731
                    and not r.get("skipped_reason") and r.get("grasp_miss_m") is not None)
    valid = [c for c in cal if all(ok(dyn.get((c, v))) for v in VS)]
    if not valid:
        print(f"!!! {brain}: no calibration case valid for all of {VS}", file=sys.stderr)
        raise SystemExit(2)
    score = {v: float(np.mean([dyn[c, v]["grasp_miss_m"] for c in valid])) for v in VS}
    k = next(v for v in TIE if score[v] <= min(score.values()) + TIE_M)
    return {"k": k, "scores": {v: round(s, 6) for v, s in score.items()}, "n_valid": len(valid), "n_cases": len(cal),
            "flag_lt40": len(valid) < 40}


def need(d, name, field):
    """mode.json's or k_select.json's value, or exit 3: the line needs it and it is not there yet."""
    v = ((d or {}).get(name) or {}).get(field)
    if v is None:
        print(f"!!! no {field} of {name} yet (mode.json / k_select.json)", file=sys.stderr)
        raise SystemExit(3)
    assert v in {"mode": ("P", "S"), "k": VS}[field], (name, field, v)
    return v


def plan(line, mode=None, k=None, split=None, records=(), where=None):
    """spec §6, §7.2, §12: the runner lines of one line (LINES), phase c<line>. kπ, kO: k_select's; a mode С probe
    adds G-auto to its lines; in mode П a VS1.0 on M1 d = 0 is part 2's G. c5 (mode С only): PPC where PPCL latched
    (records: lines 2, 3, 7), the bench's pairs written to where/latched_bench.json."""
    split = split or maxwrap.load_split(cs.SPLIT)
    ph = f"c{line}"
    s = lambda name: need(mode, name, "mode") == "S"  # noqa: E731
    if line == "smoke":
        lns = mlines(ph, "Csmoke", P, "eval300", ("PPCL", "VS0.3"), split,
                     cases=cs.sources(run.set_ids(split, "eval300"))["plus"][:6])
        return [lns[0] | {"id": "Csmoke_pi05"}] + blines(
            ph, "Csmoke_bench_pi05", ("VS0.3", "PPCL"), ("A",), f"--tasks {T0} --inits 0-5",
            [(*orx.TASKS[0], i) for i in range(6)])
    if line in ("1", "4"):
        return mlines(ph, f"C{line}", P if line == "1" else OP, "calib", VS, split)
    if line in ("2", "3"):
        kp, sp = need(k, P, "k"), s("pi05_lmax")
        arms = ["PPCL"] + [kp] * (line == "3" or kp != "VS1.0" or sp) + ["Gauto"] * sp
        return [x for d in ((0,) if line == "2" else (10, 20))
                for x in mlines(ph, f"C{line}", P, "eval300", arms, split, d, kbar=KBAR["m", P] if sp else None)]
    if line == "5":
        kp, ko, so = need(k, P, "k"), need(k, OP, "k"), s("oftplus_lmax")
        arms = [a for a in dict.fromkeys((kp, ko)) if so or a != "VS1.0"] + ["Gauto"] * so
        return mlines(ph, "C5", OP, "eval300", arms, split, kbar=KBAR["m", OP] if so else None) if arms else []
    if line in ("6", "7"):
        kp, sb = need(k, P, "k"), s("pi05_bench")
        arms = [f"{kp}@cv"] + ["Gauto@cv"] * sb if line == "6" else ["PPCL", kp] + ["Gauto"] * sb
        return blines(ph, f"C{line}_{P}", arms, ("A20",) if line == "6" else ("A", "A20"), "--inits 0-9", EPS,
                      KBAR["b", P] if sb else None)
    if line == "8":
        return mlines(ph, "C8", O, "eval300", [need(k, P, "k"), "Gauto"], split, kbar=KBAR["m", O])
    assert line == "c5", line
    lt, out = latched(records), []
    if s("pi05_lmax"):
        for d, ids in lt["lmax"].items():
            out += mlines(ph, "C5ppc", P, "eval300", ["PPC"], split, d, cases=set(ids))
    if s("pi05_bench") and lt["bench"]:
        f = Path(where) / "latched_bench.json"
        f.write_text(json.dumps([{"suite": s_, "task": t, "init": i} for s_, t, i in lt["bench"]], indent=1) + "\n")
        out += blines(ph, f"C5ppc_{P}", ["PPC"], ("A", "A20"), f"--pairs-file {f}", list(map(tuple, lt["bench"])))
    return out


def latched(recs):
    """spec §7.2: the cases (LIBERO-MAX, per d) and bench (suite, task, init) where PPCL's latch held a chunk in at
    least one episode; share: [latched, PPCL episodes] per d and on the bench."""
    lm, bench, share = defaultdict(set), set(), defaultdict(lambda: [0, 0])
    for r in recs:
        if "kind" not in r or cs.method(r["arm"]) != "PPCL" or r.get("skipped_reason"):
            continue
        on, lmax = r["latch"]["chunks"] > 0, "case_id" in r
        g = f"lmax_d{r['d']}" if lmax else "bench"
        ids = lm[r["d"]] if lmax else bench  # every d with PPCL records is listed, latched or not
        if on:
            ids.add(r["case_id"] if lmax else (r["suite"], r["task"], r["init"]))
        share[g][0] += on
        share[g][1] += 1
    return {"lmax": {d: sorted(v) for d, v in sorted(lm.items())}, "bench": sorted(map(list, bench)),
            "share": dict(share)}


def read_tempo(path):
    """The last "tempo (s/episode): k=v ..." line (scripts/gpu_c1e4m.sh), as cs.read_tempo without its pod check."""
    hits = re.findall(r"tempo \(s/episode\): (.*)", Path(path).read_text())
    t = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", hits[-1])} if hits else {}
    if missing := [k for k in TEMPO if not t.get(k)]:
        raise SystemExit(f"!!! no tempo for {missing} in {path}: no forecast")
    return {k: t[k] for k in TEMPO}


def assume(mode, k):
    """The forecast's mode and k: the files' where there, else the most expensive case (mode С; kπ not 1.0; kO
    neither kπ nor 1.0)."""
    mode = {n: {"mode": "S"} for n in SAVED} | (mode or {})
    kp = ((k or {}).get(P) or {}).get("k") or "VS0.3"
    ko = ((k or {}).get(OP) or {}).get("k") or next(v for v in VS if v not in (kp, "VS1.0"))
    return mode, {P: {"k": kp}, OP: {"k": ko}}


def forecast(tempo, mode=None, k=None, done=frozenset(), split=None, cut=CUT, rate=RATE):
    """spec §12: per line 1-8 (one GPU, one runner at a time) the full episodes, the equivalent ones (PPCL x PPCL_X, a
    cut Base CUT of its brain and source), hours, $ at rate and the hours left (the keys not in done); the core (1-7)
    and core + tail (8). Setup, probe and smoke are not in it (done before it is read)."""
    mode, k = assume(mode, k)
    rows = {}
    for L in LINES[1:9]:
        n, eq, h, hl = 0, 0.0, 0.0, 0.0
        for ln in plan(L, mode, k, split):
            t = tempo[f"{'lmax' if ln['runner'] == 'c1e4m' else 'bench'}_{ln['brain']}"]
            n += n_of(ln)
            for x in ln["keys"]:
                w = (cut[x[1], ln["source"]] if x[0] == "m" and x[3] == "base" and ln["prefix"]
                     else PPCL_X if cs.method(x[4]) == "PPCL" else 1.0)
                eq += w
                h += w * t / 3600
                hl += w * t / 3600 * (x not in done)
        rows[L] = (n, eq, h, hl)
    tot = lambda n, eq, h, hl: {"episodes": n, "eq": round(eq, 2), "hours": round(h, 2),  # noqa: E731
                                "usd": round(rate * h, 2), "left_hours": round(hl, 2)}
    add = lambda Ls: tot(*(sum(rows[L][j] for L in Ls) for j in range(4)))  # noqa: E731
    return {"usd_per_h": rate, "tempo_s": tempo, "using": {"mode": {n: v["mode"] for n, v in mode.items()},
                                                           "k": {b: v["k"] for b, v in k.items()}},
            "lines": {L: tot(*v) for L, v in rows.items()}, "core": add(CORE), "core_tail": add(LINES[1:9])}


def smoke_check(recs):
    """spec §12's smoke gate: PPCL and VS0.3 each with prefix_ok in >= 5 of the 6 cases. (pass, {arm: [ok, cases]})."""
    n = {}
    for a in ("PPCL", "VS0.3"):
        rs = [r for r in recs if r.get("kind") == "dynamic" and r["arm"] == a]
        n[a] = [len({r["case_id"] for r in rs if r.get("prefix_ok") is True}), len({r["case_id"] for r in rs})]
    return all(ok >= 5 for ok, _ in n.values()), n


def merge(path, d):
    """Write d into the JSON object at path (k_select.json: one brain at a time)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps((json.loads(p.read_text()) if p.exists() else {}) | d, indent=1) + "\n")


def first_rec(path):
    """The first record (a line with "kind") of a JSON lines file."""
    with open(path) as f:
        return next(json.loads(x) for x in f if '"kind"' in x)


def selftest():
    split = maxwrap.load_split(cs.SPLIT)
    A7 = ("none", "G", "Gauto", "PPC", "G@glr", "PPC@glr", "T0")

    def mrec(brain, c, kind, arm, d=0, arms=(), kbar=None, **kw):
        return {"brain": brain, "case_id": c, "kind": kind, "arm": arm, "method": cs.method(arm), "d": d,
                "conf": {"arms": list(arms), "kbar": kbar}, "e": 30, "act_hash": f"{brain}/{c}/{kind}/{arm}/{d}",
                "prefix_ok": None if kind == "base" else True, "replay_miss": None, "skipped_reason": None} | kw

    def brec(s, t, i, cell, arm, **kw):
        return {"brain": P, "kind": "move", "cell": cell, "arm": arm, "method": cs.method(arm), "suite": s, "task": t,
                "init": i, "conf": {"kbar": 0.324}, "act_hash": f"{s}/{t}/{i}/{cell}/{arm}"} | kw

    def case_recs(brain, ids, arms, kbar, d=0):  # part 2's file order: a case's Dynamic lines, its Base last
        return [mrec(brain, c, k, a, d, arms, kbar) for c in ids for k, a in [("dynamic", a) for a in arms]
                + [("base", "none")]]

    # the probe: part 2's runs again; its plan from the saved files (arms, kbar, the first 12 Plus cases)
    plus = {s: cs.sources(run.set_ids(split, s))["plus"] for s in PSET.values()}
    saved = {"pi05_lmax": case_recs(P, plus["eval300"][:13], A7, 0.306),
             "oftplus_lmax": case_recs(OP, plus["eval"][:13], cs.ARMS, 0.121),
             "pi05_bench": [brec(s, t, i, c, a) for s, t in orx.TASKS[:2] for c in ("A", "A20") for a in cs.B_ARMS
                            for i in range(10)] + [brec(*orx.TASKS[0], i, "A20", "Gauto@cv") for i in range(10)]}
    pl = probe_lines(saved, split)
    assert [(ln["name"], ln["phase"], ln["id"], n_of(ln)) for ln in pl] == [
        ("pi05_lmax", "cprobe", "P1_pi05_eval300_plus_d0", 96),
        ("oftplus_lmax", "cprobe", "P2_oftplus_eval_plus_d0", 60),
        ("pi05_bench", "cprobe", "P3_pi05", 80)], pl
    assert pl[0]["args"] == ("--brain pi05 --source plus --set eval300 --arms=none,G,Gauto,PPC,G@glr,PPC@glr,T0 --d 0 "
                             f"--cases {','.join(plus['eval300'][:12])} --kbar 0.306"), pl[0]["args"]
    assert "--set eval --arms=none,G,Gauto,PPC --d 0 --cases " in pl[1]["args"]
    assert pl[1]["args"].endswith("--kbar 0.121")
    assert pl[2]["args"] == ("--brain pi05 --kinds move --cells A,A20 --methods none,Gauto,Glead,PPC --tasks "
                             "libero_spatial:0 --inits 0-9 --kbar 0.324"), pl[2]["args"]
    assert all("--base-prefix" not in ln["args"] for ln in pl)  # a cut Base's act_hash is not part 2's
    assert tsv(pl).splitlines()[0].split("\t")[:6] == ["cprobe", "c1e4m", P, "plus", "P1_pi05_eval300_plus_d0", "96"]
    swapped = saved | {"pi05_lmax": saved["pi05_lmax"][8:16] + saved["pi05_lmax"][:8] + saved["pi05_lmax"][16:]}
    with contextlib.redirect_stderr(io.StringIO()) as err:
        assert probe_lines(swapped, split) == pl  # the runner's order is the set's: the plan keeps it
    assert "warning" in err.getvalue(), err.getvalue()
    gpu = lambda rs: [r | {"conf": r["conf"] | {"gpu": "NVIDIA GeForce RTX 4090"}} for r in rs]  # noqa: E731
    new = gpu([r for rs in saved.values() for r in rs])
    res = probe(new, saved, pl)
    assert res["pi05_lmax"] == {"mode": "P", "equal": 96, "total": 96, "missing": 0, "e_equal": 12, "e_total": 12,
                                "gpu": ["NVIDIA GeForce RTX 4090"]}, res
    assert res["oftplus_lmax"]["mode"] == "P" and res["pi05_bench"] == {
        "mode": "P", "equal": 80, "total": 80, "missing": 0, "gpu": ["NVIDIA GeForce RTX 4090"]}, res
    flip = [r | {"act_hash": "x"} if r["arm"] == "T0" and r.get("case_id") == plus["eval300"][3] else r for r in new]
    res = probe(flip, saved, pl)
    assert res["pi05_lmax"]["mode"] == "S" and res["pi05_lmax"]["equal"] == 95 and res["oftplus_lmax"]["mode"] == "P"
    gone = [r for r in new if not (r.get("init") == 7 and r.get("cell") == "A20" and r["arm"] == "PPC")]
    res = probe(gone, saved, pl)
    assert res["pi05_bench"] == {"mode": "S", "equal": 79, "total": 79, "missing": 1, "gpu": [
        "NVIDIA GeForce RTX 4090"]} and res["pi05_lmax"]["mode"] == "P", res
    res = probe([r for rs in saved.values() for r in rs], saved, pl)  # part 2's own records passed as the probe
    assert all(v["mode"] == "S" and v["missing"] == n_of(ln) for v, ln in zip(res.values(), pl)), res
    unhashed = [r | {"act_hash": None} for r in saved["pi05_lmax"]]  # no hash on either side is no evidence
    assert probe(gpu(unhashed), saved | {"pi05_lmax": unhashed}, pl)["pi05_lmax"]["mode"] == "S"
    e_off = [r | {"e": 31} if r["kind"] == "base" and r.get("case_id") == plus["eval300"][0] else r for r in new]
    assert probe(e_off, saved, pl)["pi05_lmax"]["e_equal"] == 11

    # the k selection (spec §5): mean grasp miss over the cases valid for all three, lowest wins, 0.1 mm ties by TIE
    cal = run.set_ids(split, "calib")

    def m0(miss, brain=P, edit=lambda j, v: {}):
        return [mrec(brain, c, "dynamic", v, arms=VS, grasp_miss_m=miss[v] + 0.001 * (j % 3)) | edit(j, v)
                for j, c in enumerate(cal) for v in VS] + [
            mrec(brain, c, "base", "none", arms=VS, base_cut=True, success=None, grasp_ok=None) for c in cal]
    clear = {"VS0.1": 0.010, "VS0.3": 0.012, "VS1.0": 0.020}
    r = select(m0(clear), P, split)
    assert r["k"] == "VS0.1" and r["n_valid"] == 60 and r["n_cases"] == 60 and not r["flag_lt40"], r
    assert r["scores"] == {"VS0.1": 0.011, "VS0.3": 0.013, "VS1.0": 0.021}, r
    for miss, k in (({"VS0.1": 0.0103, "VS0.3": 0.01005, "VS1.0": 0.0100}, "VS0.3"),  # 0.05 mm: a tie, 0.3 first
                    ({"VS0.1": 0.01005, "VS0.3": 0.0102, "VS1.0": 0.0100}, "VS0.1"),
                    ({"VS0.1": 0.0102, "VS0.3": 0.0102, "VS1.0": 0.0100}, "VS1.0")):  # 0.2 mm: no tie
        assert select(m0(miss), P, split)["k"] == k, (miss, k)
    # a dropped case drops for all three: its VS0.1 miss of 1 m would hand the win to VS0.3
    for why in ({"grasp_miss_m": None}, {"prefix_ok": False}, {"replay_miss": 12}, {"e": None},
                {"skipped_reason": "motion_blur"}):
        r = select(m0(clear, edit=lambda j, v: {"grasp_miss_m": 1.0} if j == 0 and v == "VS0.1"
                      else why if j == 0 and v == "VS1.0" else {}), P, split)
        assert r["k"] == "VS0.1" and r["n_valid"] == 59, (why, r)
    r = select([x for x in m0(clear) if not (x["case_id"] == cal[5] and x["arm"] == "VS0.3")], P, split)
    assert r["n_valid"] == 59, r  # a missing record
    r = select(m0(clear, edit=lambda j, v: {"grasp_miss_m": None} if j < 21 and v == "VS0.3" else {}), P, split)
    assert r["n_valid"] == 39 and r["flag_lt40"] and r["k"] == "VS0.1", r  # fewer than 40: the rule still applies
    r = select(m0(clear) + m0({"VS0.1": 0.03, "VS0.3": 0.001, "VS1.0": 0.03}, OP), P, split)
    assert r["k"] == "VS0.1", r  # the other brain's records do not count
    for rs in ([x for x in m0(clear) if not (x["kind"] == "base" and x["case_id"] == cal[0])],
               m0(clear, edit=lambda j, v: {"grasp_miss_m": None})):  # the line is incomplete / nothing valid
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                select(rs, P, split)
            raise AssertionError("an incomplete selection line was accepted")
        except SystemExit as ex:
            assert ex.code == 2, ex.code

    # the plan per line
    def mk(kp=None, ko=None, sp=None, so=None, sb=None):
        mode = {n: {"mode": m} for n, m in (("pi05_lmax", sp), ("oftplus_lmax", so), ("pi05_bench", sb)) if m}
        return mode, {b: {"k": v} for b, v in ((P, kp), (OP, ko)) if v}

    def arms(line, *a, **kw):
        lns = plan(line, *mk(*a, **kw), split)
        assert len({ln["arms"] == lns[0]["arms"] for ln in lns}) <= 1
        return lns[0]["arms"] if lns else []
    sm = plan("smoke", None, None, split)
    assert [(ln["id"], ln["runner"], n_of(ln)) for ln in sm] == [("Csmoke_pi05", "c1e4m", 12),
                                                                 ("Csmoke_bench_pi05", "c1e4", 12)], sm
    assert sm[0]["args"] == ("--brain pi05 --source plus --set eval300 --arms=PPCL,VS0.3 --d 0 --cases "
                             f"{','.join(plus['eval300'][:6])} --base-prefix"), sm[0]["args"]
    assert sm[1]["args"] == ("--brain pi05 --kinds move --cells A --methods VS0.3,PPCL --tasks libero_spatial:0 "
                             "--inits 0-5"), sm[1]["args"]
    l1 = plan("1", None, None, split)
    assert [(ln["id"], n_of(ln)) for ln in l1] == [("C1_pi05_calib_plus_d0", 126), ("C1_pi05_calib_pro_d0", 54)]
    assert l1[0]["args"] == "--brain pi05 --source plus --set calib --arms=VS0.1,VS0.3,VS1.0 --d 0 --base-prefix"
    assert [ln["id"] for ln in plan("4", None, None, split)] == ["C4_oftplus_calib_plus_d0", "C4_oftplus_calib_pro_d0"]
    assert arms("2", "VS1.0", sp="P") == ["PPCL"]  # VS1.0 = part 2's G in mode П
    l2 = plan("2", *mk("VS1.0", sp="S"), split)
    assert l2[0]["arms"] == ["PPCL", "VS1.0", "Gauto"] and all("--kbar 0.306" in ln["args"] for ln in l2)
    assert [n_of(ln) for ln in l2] == [561, 339] and [ln["id"] for ln in l2] == [
        "C2_pi05_eval300_plus_d0", "C2_pi05_eval300_pro_d0"]
    assert arms("2", "VS0.3", sp="P") == ["PPCL", "VS0.3"] and "--kbar" not in plan("2", *mk("VS0.3", sp="P"),
                                                                                       split)[0]["args"]
    l3 = plan("3", *mk("VS1.0", sp="P"), split)  # no G in D: VS runs in mode П too
    assert [ln["id"] for ln in l3] == ["C3_pi05_eval300_plus_d10", "C3_pi05_eval300_pro_d10",
                                       "C3_pi05_eval300_plus_d20", "C3_pi05_eval300_pro_d20"]
    assert l3[0]["arms"] == ["PPCL", "VS1.0"] and "--d 10 " in l3[0]["args"] and "--kbar" not in l3[0]["args"]
    assert arms("3", "VS0.3", sp="S") == ["PPCL", "VS0.3", "Gauto"]
    assert arms("5", "VS0.3", "VS0.3", so="P") == ["VS0.3"]  # kO = kπ: one VS arm
    assert arms("5", "VS0.3", "VS0.1", so="P") == ["VS0.3", "VS0.1"]
    assert arms("5", "VS1.0", "VS1.0", so="P") == [] and plan("5", *mk("VS1.0", "VS1.0", so="P"), split) == []
    assert arms("5", "VS1.0", "VS0.3", so="P") == ["VS0.3"]
    l5 = plan("5", *mk("VS1.0", "VS1.0", so="S"), split)
    assert l5[0]["arms"] == ["VS1.0", "Gauto"] and "--set eval300" in l5[0]["args"] and "--kbar 0.121" in l5[0]["args"]
    l6 = plan("6", *mk("VS0.3", sb="P"), split)
    assert [(ln["id"], n_of(ln)) for ln in l6] == [("C6_pi05", 260)] and l6[0]["args"] == (
        "--brain pi05 --kinds move --cells A20 --methods VS0.3@cv --inits 0-9"), l6
    l6 = plan("6", *mk("VS0.3", sb="S"), split)
    assert n_of(l6[0]) == 520 and l6[0]["args"].endswith("--methods VS0.3@cv,Gauto@cv --inits 0-9 --kbar 0.324")
    l7 = plan("7", *mk("VS1.0", sb="P"), split)
    assert n_of(l7[0]) == 1040 and "--cells A,A20 --methods PPCL,VS1.0 --inits 0-9" in l7[0]["args"]
    assert arms("7", "VS0.1", sb="S") == ["PPCL", "VS0.1", "Gauto"]
    l8 = plan("8", *mk("VS0.3"), split)  # the tail: always self-contained
    assert l8[0]["arms"] == ["VS0.3", "Gauto"] and "--kbar 0.088" in l8[0]["args"] and sum(map(n_of, l8)) == 600
    full = mk("VS0.3", "VS0.1", "S", "S", "S")
    every = [ln for L in LINES[:-1] for ln in plan(L, *full, split)]
    assert len({ln["id"] for ln in every}) == len(every)  # one --out per runner line
    for L in LINES[:-1]:
        assert {ln["phase"] for ln in plan(L, *full, split)} == {f"c{L}"}
    for ln in every:
        if ln["runner"] == "c1e4m":
            assert "--base-prefix" in ln["args"] and f"--source {ln['source']} " in ln["args"] + " "
            assert all(x[2].startswith("pro-") == (ln["source"] == "pro") for x in ln["keys"])
        assert ("--kbar" in ln["args"]) == bool({"Gauto", "Glead"} & set(map(cs.method, ln["arms"])))
    for L, mo in (("2", mk("VS0.3")), ("2", mk(sp="S")), ("5", mk("VS0.3", so="S")), ("5", mk(ko="VS0.3", so="S")),
                  ("6", mk("VS0.3")), ("7", mk(sb="P")), ("8", mk())):
        try:
            with contextlib.redirect_stderr(io.StringIO()):
                plan(L, *mo, split)
            raise AssertionError(f"line {L} without its mode or k: {mo}")
        except SystemExit as ex:
            assert ex.code == 3, (L, ex.code)

    # the latched list and the PPC pass of mode С (spec §7.2)
    e10 = plus["eval300"][:10]
    lat = lambda n: {"latch": {"trig": n, "chunks": n, "resets": 0}}  # noqa: E731
    pro = cs.sources(run.set_ids(split, "eval300"))["pro"][0]
    recs = ([mrec(P, c, "dynamic", "PPCL", 0) | lat(2 * (c == e10[1])) for c in e10]
            + [mrec(P, c, "dynamic", "PPCL", 10) | lat(c == pro) for c in e10 + [pro]]
            + [mrec(P, c, "dynamic", "PPCL", 20) | lat(0) for c in e10]
            + [mrec(P, e10[2], "dynamic", "PPCL", 20, skipped_reason="x")]  # skipped: no latch field, no episode
            + [mrec(P, c, "dynamic", "VS0.3", 0) | lat(1) for c in e10]  # not PPCL
            + [brec(s, t, i, c, "PPCL") | lat(int(i == 3 and t == 1 and c == "A20")) for s, t in orx.TASKS[:2]
               for c in ("A", "A20") for i in range(10)])
    lt = latched(recs)
    assert lt == {"lmax": {0: [e10[1]], 10: [pro], 20: []}, "bench": [[orx.TASKS[1][0], 1, 3]],
                  "share": {"lmax_d0": [1, 10], "lmax_d10": [1, 11], "lmax_d20": [0, 10], "bench": [1, 40]}}, lt
    with tempfile.TemporaryDirectory() as tmp:
        c5 = plan("c5", *mk(sp="S", sb="S"), split, recs, Path(tmp))
        assert [(ln["id"], ln["arms"], n_of(ln)) for ln in c5] == [
            ("C5ppc_pi05_eval300_plus_d0", ["PPC"], 1), ("C5ppc_pi05_eval300_pro_d10", ["PPC"], 1),
            ("C5ppc_pi05", ["PPC"], 2)], c5
        assert c5[0]["args"].endswith(f"--d 0 --cases {e10[1]} --base-prefix") and c5[0]["phase"] == "cc5"
        assert json.loads((Path(tmp) / "latched_bench.json").read_text()) == [
            {"suite": orx.TASKS[1][0], "task": 1, "init": 3}]
        assert f"--kinds move --cells A,A20 --methods PPC --pairs-file {Path(tmp) / 'latched_bench.json'}" in c5[2][
            "args"], c5[2]["args"]
        assert [ln["id"] for ln in plan("c5", *mk(sp="P", sb="S"), split, recs, Path(tmp))] == ["C5ppc_pi05"]
        assert plan("c5", *mk(sp="P", sb="P"), split, recs, Path(tmp)) == []
        assert plan("c5", *mk(sp="S", sb="S"), split, [x | lat(0) for x in recs], Path(tmp)) == []
        # k_select.json gathers both brains' selections
        merge(Path(tmp) / "k.json", {P: {"k": "VS0.3"}})
        merge(Path(tmp) / "k.json", {OP: {"k": "VS0.1"}})
        assert json.loads((Path(tmp) / "k.json").read_text()) == {P: {"k": "VS0.3"}, OP: {"k": "VS0.1"}}
        tf = Path(tmp) / "tempo.txt"
        tf.write_text("tempo (s/episode): lmax_pi05=1 lmax_oftplus=1 lmax_oft=1 bench_pi05=1\nC1-E4m envs: pi05=6\n"
                      "tempo (s/episode): lmax_pi05=19.5 lmax_oftplus=12.497 lmax_oft=15.9 bench_pi05=14.6\n")
        assert read_tempo(tf) == {"lmax_pi05": 19.5, "lmax_oftplus": 12.497, "lmax_oft": 15.9, "bench_pi05": 14.6}
        tf.write_text("tempo (s/episode): lmax_pi05=19.5 lmax_oftplus=12.5 bench_pi05=14.6\n")
        try:
            read_tempo(tf)
            raise AssertionError("a tempo without lmax_oft was accepted")
        except SystemExit:
            pass

    # the forecast: 36 s per episode everywhere -> hours = equivalent episodes / 100
    t36 = dict.fromkeys(TEMPO, 36.0)
    f = forecast(t36, split=split)  # no mode, no k: the most expensive case
    assert f["using"] == {"mode": {"pi05_lmax": "S", "oftplus_lmax": "S", "pi05_bench": "S"},
                            "k": {P: "VS0.3", OP: "VS0.1"}}, f["using"]
    eq1 = 180 + 42 * CUT[P, "plus"] + 18 * CUT[P, "pro"]  # line 1 by hand: 3 VS arms on 60 cases + 60 cut Bases
    assert f["lines"]["1"] == {"episodes": 180, "eq": round(eq1, 2), "hours": round(eq1 / 100, 2),
                               "usd": round(eq1 / 100 * RATE, 2), "left_hours": round(eq1 / 100, 2)}, f["lines"]["1"]
    eq2 = 300 * (PPCL_X + 2) + 187 * CUT[P, "plus"] + 113 * CUT[P, "pro"]  # PPCL, VS0.3, Gauto
    assert f["lines"]["2"]["eq"] == round(eq2, 2) and f["lines"]["3"]["eq"] == round(2 * eq2, 2)
    assert [f["lines"][L]["episodes"] for L in LINES[1:9]] == [180, 900, 1800, 180, 900, 520, 1560, 600]
    assert f["lines"]["7"]["eq"] == 520 * (PPCL_X + 2) and f["core"]["episodes"] == 6040
    assert f["core_tail"]["episodes"] == 6640 and f["core_tail"]["hours"] == round(
        sum(f["lines"][L]["eq"] for L in LINES[1:9]) / 100, 2)
    f = forecast(t36, *mk("VS1.0", "VS1.0", "P", "P", "P"), split=split)  # every coincidence
    assert [f["lines"][L]["episodes"] for L in LINES[1:9]] == [180, 300, 1200, 180, 0, 260, 1040, 600]
    assert f["lines"]["5"] == {"episodes": 0, "eq": 0, "hours": 0, "usd": 0, "left_hours": 0}
    done = set().union(*(ln["keys"] for ln in plan("1", None, None, split)))
    f = forecast(t36, *full, done=done, split=split)
    assert f["lines"]["1"]["left_hours"] == 0 and f["lines"]["2"]["left_hours"] == f["lines"]["2"]["hours"]
    assert f["core"]["left_hours"] == round(sum(f["lines"][L]["eq"] for L in CORE[1:]) / 100, 2)
    flat = forecast(t36, split=split, cut=dict.fromkeys(CUT, 0.3))
    assert flat["lines"]["1"]["eq"] == 198  # spec §12's 180 (+18)

    # the smoke's gate (spec §12): prefix_ok in >= 5 of 6 cases for PPCL and VS0.3
    sm = [mrec(P, c, "dynamic", a, arms=("PPCL", "VS0.3")) for c in plus["eval300"][:6] for a in ("PPCL", "VS0.3")]
    sm += [mrec(P, c, "base", "none", arms=("PPCL", "VS0.3"), base_cut=True) for c in plus["eval300"][:6]]
    one = lambda a, n: [x | {"prefix_ok": False} if x["arm"] == a and x["case_id"] in plus["eval300"][:n] else x  # noqa
                        for x in sm]
    assert smoke_check(sm) == (True, {"PPCL": [6, 6], "VS0.3": [6, 6]}), smoke_check(sm)
    assert smoke_check(one("PPCL", 1))[0] and smoke_check(one("VS0.3", 1)) == (True, {"PPCL": [6, 6], "VS0.3": [5, 6]})
    assert not smoke_check(one("PPCL", 2))[0] and not smoke_check(one("VS0.3", 2))[0]
    assert not smoke_check([x for x in sm if x["arm"] != "VS0.3"])[0]

    # part 2's files when they are here: kbar and the probes' episodes
    if P2.exists():
        for (r, b), f in {("m", P): SAVED["pi05_lmax"][0], ("m", OP): SAVED["oftplus_lmax"][0],
                          ("m", O): "pod_B/M4oft_oft_eval300_plus_d0.jsonl", ("b", P): SAVED["pi05_bench"][0]}.items():
            assert first_rec(P2 / f)["conf"]["kbar"] == KBAR[r, b], (r, b, f)
        real = probe_lines({n: e2.load(P2 / fs[0]) for n, fs in SAVED.items()}, split)
        assert [n_of(ln) for ln in real] == [96, 60, 80] and real == pl, real
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--probe-plan", action="store_true", help="the probes' runner lines (phase cprobe)")
    g.add_argument("--probe", nargs="+", metavar="JSONL", help="the probes' records -> mode.json")
    g.add_argument("--select", nargs="+", metavar="BRAIN JSONL", help="pi05|oftplus, its M0 records -> k_select.json")
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--latched", nargs="+", metavar="JSONL", help="PPCL records: where the latch held a chunk")
    g.add_argument("--forecast", metavar="TEMPO", help="the tempo file: hours and $ per line")
    g.add_argument("--smoke-check", nargs="+", metavar="JSONL", help="the smoke's LIBERO-MAX records: exit 0, or 3")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--mode", default=f"{OUT}/mode.json", help="mode.json (--plan, --forecast)")
    ap.add_argument("--k", default=f"{OUT}/k_select.json", help="k_select.json (--plan, --forecast)")
    ap.add_argument("--out", help="--probe: default --mode's path; --select: default --k's")
    ap.add_argument("--records", nargs="+", default=[], metavar="JSONL", help="--line c5: lines 2, 3, 7's records")
    ap.add_argument("--done", nargs="+", default=[], metavar="JSONL", help="--forecast: the grid's records so far")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f) if "kind" in x]  # noqa: E731
    jload = lambda f: json.loads(Path(f).read_text()) if Path(f).exists() else None  # noqa: E731
    if a.plan and not a.line:
        ap.error("--plan needs --line")
    if a.line == "c5" and not a.records:
        ap.error("--line c5 needs --records (the PPCL records of lines 2, 3, 7)")
    split = maxwrap.load_split(cs.SPLIT)
    if a.selftest:
        selftest()
    elif a.probe_plan or a.probe:
        saved = {n: e2.load(P2 / fs[0]) for n, fs in SAVED.items()}
        pl = probe_lines(saved, split)
        if a.probe_plan:
            print(tsv(pl))
        else:
            res = probe(load(a.probe), saved, pl)
            Path(a.out or a.mode).parent.mkdir(parents=True, exist_ok=True)
            Path(a.out or a.mode).write_text(json.dumps(res, indent=1) + "\n")
            print(e2.dump(res))
    elif a.select:
        brain, files = a.select[0], a.select[1:]
        if brain not in (P, OP) or not files:
            ap.error("--select pi05|oftplus JSONL...")
        res = {brain: select(load(files), brain, split)}
        merge(a.out or a.k, res)
        print(e2.dump(res))
    elif a.plan:
        if lns := plan(a.line, jload(a.mode), jload(a.k), split, load(a.records), Path(a.mode).parent):
            print(tsv(lns))
    elif a.latched:
        print(e2.dump(latched(load(a.latched))))
    elif a.forecast:
        print(e2.dump(forecast(read_tempo(a.forecast), jload(a.mode), jload(a.k), {cs.key(r) for r in load(a.done)},
                               split)))
    elif a.smoke_check:
        ok, n = smoke_check(load(a.smoke_check))
        print(e2.dump({"prefix_ok_of_cases": n, "pass": ok}))
        print(f"C1-E4 controls smoke: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
