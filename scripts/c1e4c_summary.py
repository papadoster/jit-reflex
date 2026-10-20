"""C1-E4 controls (PPCL, VS): the probe and its П/С mode (§7.1), the k selection (§5), the runner plan per line
(§6, §7.2, §12), the latched list (§7.2) and the forecast (§12), on the JSON lines of scripts/c1e4m_run.py
(LIBERO-MAX) and scripts/c1e4_run.py (bench). Spec docs/superpowers/specs/2026-10-20-c1-e4-controls-design.md;
part 2's records (results/c1-e4/part2) are read only. Pure numpy, the LeRobot env; run from the repository root.
The spec's journal "ветки VS остановлены" (2026-10-20, during the grid, before reading): lines 1, 2, 3, 7 only (4, 5,
6, 8 dropped; line 2's first VS_STOP cases ran with VS, the rest without, line 3 and 7 without), the rule C5 and the
measurement И4 only; the VS records of lines 2+ are on the plan but never read (counted).
R = results/c1-e4/controls/pod_C (scripts/gpu_c1e4m.sh POD=C):
    python scripts/c1e4c_summary.py --probe-plan          the probes' runner lines (phase cprobe)
    python scripts/c1e4c_summary.py --probe R/P1_*.jsonl R/P2_*.jsonl R/P3_*.jsonl [--out R/mode.json]
    python scripts/c1e4c_summary.py --select pi05 R/C1_*.jsonl [--out R/k_select.json]   (exit 2: line incomplete)
    python scripts/c1e4c_summary.py --plan --line L [--mode R/mode.json] [--k R/k_select.json]   L: smoke 1-8 c5
        the line's runner lines "phase<TAB>runner<TAB>brain<TAB>source<TAB>id<TAB>n<TAB>args", phase c<L>, n = its
        Dynamic episodes (4, 5, 6, 8: none); exit 3 when the line needs a mode or k not there yet. c5 (mode С):
        --records R/C2_*.jsonl R/C3_*.jsonl R/C7_*.jsonl (those run), PPC on the runner groups where PPCL latched
    python scripts/c1e4c_summary.py --latched R/C2_*.jsonl ...
    python scripts/c1e4c_summary.py --forecast R/tempo.txt [--done R/C[0-9]*.jsonl] [--mode ..] [--k ..]
        hours and $ per line; without mode or k the most expensive case (mode С, kπ != 1.0, kO != kπ)
    python scripts/c1e4c_summary.py --smoke-check R/Csmoke_pi05*.jsonl   exit 0, or 3
    python scripts/c1e4c_summary.py [grid] R/grid_all_C.jsonl [--mode R/mode.json] [--k R/k_select.json]
            [--p2 results/c1-e4/part2] [--out R/summary.json]
        on the Mac: the checks before reading (§7.3, §8), then the rule C5, the measurement И4 (an estimate and its
        95% CI, no threshold), the claim and the report (§8, §11); a failed check prints only the checks, exit 3.
        grid_all_C.jsonl (the pod's cat of C[1-8]*.jsonl) = lines 1, 2, 3, 7 and the c5 pass; records off the plan (a
        probe's) and VS records are counted, never read; one key twice with other content
        (grid_all_C.jsonl and its C*.jsonl both): exit 2. Needs hf-libero (the task clusters), the LeRobot env
    python scripts/c1e4c_summary.py --selftest
"""

import argparse
import contextlib
import io
import json
import re
import sys
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from unittest import mock

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e4m_summary as cs  # noqa: E402  (key, mindex, bindex, invalid, bprs, sources; puts src/ on sys.path)
from c1e4m_summary import O, OP, P, TOL, e2, maxwrap, orx, run  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
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
EXCLUDES = ("setup, probe and smoke (spec §12: ≈ 2.5 h, ≈ $0.93) and the mode-С PPC pass on latched cases (its size is "
            "known only after lines 2, 3, 7)")  # the forecast's lines 1-8 only
NE = 6  # n-envs of every POD=C runner line (scripts/gpu_c1e4m.sh, spec §7: part 2's): the runner's groups of cases
PROBE = {("m", P): "pi05_lmax", ("m", OP): "oftplus_lmax", ("b", P): "pi05_bench"}  # spec §7.1: whose lines
CONF_SKIP = ("arms", "gpu", "kbar")  # spec §7.3: every other conf field equal in a pair; kbar: VS lines run without it,
# a Gauto or Glead record's is checked against KBAR
# spec journal "ветки VS остановлены": C2, C3, C6, И1-И3 dropped before reading
CLAIMS = {"ЛАТЧ МЕНЯЕТ PPC": "C5"}  # spec §8
MEAS = ("И4",)  # spec §8: was C4; an estimate and its 95% CI, no threshold
EST = ("diff_pp", "lo2.5", "lo5_one_sided", "hi97.5", "pairs", "clusters")  # e2.diff_pp's: a measurement's estimate
DEP = {"И4": "7", "C5": "237"}  # rule -> lines
# spec journal "ветки VS остановлены": per source, line 2's cases done with VS (the set's order) when the owner stopped
# the VS branches; each a multiple of NE (the runner's groups and Base batches go on unchanged) or the whole source
VS_STOP = {"plus": 90, "pro": 0}


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
    """spec §7.1: the probes repeat part 2's runs exactly: the first 12 Plus cases (two groups of 6) of its file, and
    the bench's first task; each with its file's arms in their order and its kbar (the batches depend on them).
    saved: {probe: the records of its SAVED file, the first}."""
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
            ln = blines("cprobe", f"P{i}_{P}", list(dict.fromkeys(r["arm"] for r in rs)), ("A", "A20"),
                        f"--tasks {T0} --inits 0-9", [x for x in EPS if x[:2] == orx.TASKS[0]], conf["kbar"])
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


def plan(line, mode=None, k=None, split=None, records=()):
    """spec §6, §7.2, §12 and its journal "ветки VS остановлены": the runner lines of one line (LINES), phase c<line>;
    4, 5, 6, 8 none (dropped). A mode С probe adds G-auto to its lines. Line 2: per source the first VS_STOP cases as
    they ran (C2, with VS(kπ); in mode П a VS1.0 on M1 d = 0 was part 2's G, not run), the rest without VS (C2_nv: the
    runner's conf, its resume key, holds the arms, so another --out). c5 (mode С only): PPC where PPCL latched
    (records: lines 2, 3, 7), on whole runner groups."""
    split = split or maxwrap.load_split(cs.SPLIT)
    ph = f"c{line}"
    s = lambda name: need(mode, name, "mode") == "S"  # noqa: E731
    if line == "smoke":
        lns = mlines(ph, "Csmoke", P, "eval300", ("PPCL", "VS0.3"), split,
                     cases=cs.sources(run.set_ids(split, "eval300"))["plus"][:6])
        return [lns[0] | {"id": "Csmoke_pi05"}] + blines(
            ph, "Csmoke_bench_pi05", ("VS0.3", "PPCL"), ("A",), f"--tasks {T0} --inits 0-5",
            [(*orx.TASKS[0], i) for i in range(6)])
    if line == "1":
        return mlines(ph, "C1", P, "calib", VS, split)
    if line in ("4", "5", "6", "8"):
        return []
    if line == "2":
        kp, sp = need(k, P, "k"), s("pi05_lmax")
        old = set()
        for src, ids in cs.sources(run.set_ids(split, "eval300")).items():
            assert VS_STOP[src] % NE == 0 or VS_STOP[src] == len(ids), (src, VS_STOP[src], len(ids))
            old |= set(ids[:VS_STOP[src]])
        kb = KBAR["m", P] if sp else None
        new = set(run.set_ids(split, "eval300")) - old
        return (mlines(ph, "C2", P, "eval300", ["PPCL"] + [kp] * (kp != "VS1.0" or sp) + ["Gauto"] * sp, split,
                       cases=old, kbar=kb)
                + mlines(ph, "C2_nv", P, "eval300", ["PPCL"] + ["Gauto"] * sp, split, cases=new, kbar=kb))
    if line == "3":
        sp = s("pi05_lmax")
        return [x for d in (10, 20) for x in mlines(ph, "C3", P, "eval300", ["PPCL"] + ["Gauto"] * sp, split, d,
                                                    kbar=KBAR["m", P] if sp else None)]
    if line == "7":
        sb = s("pi05_bench")
        return blines(ph, f"C7_{P}", ["PPCL"] + ["Gauto"] * sb, ("A", "A20"), "--inits 0-9", EPS,
                      KBAR["b", P] if sb else None)
    assert line == "c5", line
    lt, out = latched(records), []
    if s("pi05_lmax"):  # every runner group (NE consecutive cases of a source, the set's order) with a latched case:
        # the pass's Base batches are the line's, so its Bases' pre-event chunks are too (spec §7)
        grp = [ids[i:i + NE] for ids in cs.sources(run.set_ids(split, "eval300")).values()
               for i in range(0, len(ids), NE)]
        for d, ids in lt["lmax"].items():
            out += mlines(ph, "C5ppc", P, "eval300", ["PPC"], split, d,
                          cases={c for g in grp if set(g) & set(ids) for c in g})
    if s("pi05_bench") and (ts := list(dict.fromkeys((s_, t) for s_, t, _ in lt["bench"]))):
        # every init of a latched task, both cells: the bench runner batches per task and (cell, kind) in slices of
        # n-envs over arms x inits, so this pass's first slice (inits 0-5) is line 7's PPCL slice. ponytail: inits 6-9
        # rode with line 7's next arm there; rebuilding that slice would rerun that arm
        out += blines(ph, f"C5ppc_{P}", ["PPC"], ("A", "A20"),
                      f"--tasks {','.join(f'{s_}:{t}' for s_, t in ts)} --inits 0-9",
                      [(s_, t, i) for s_, t in ts for i in range(10)])
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


def forecast(tempo, mode=None, k=None, done=frozenset(), split=None):
    """spec §12: per line 1-8 (one GPU, one runner at a time) the Dynamic episodes, the records (a LIBERO-MAX line's
    cut Bases too), hours, $ at RATE and the hours left (the keys not in done); the core (1-7) and core + tail (8).
    Lines 4, 5, 6, 8 (dropped) count 0. Setup, probe and smoke are not in it (EXCLUDES)."""
    mode, k = assume(mode, k)
    rows = {}
    for L in LINES[1:9]:
        n, nr, h, hl = 0, 0, 0.0, 0.0
        for ln in plan(L, mode, k, split):
            # every planned record weighs 1: the tempo is seconds per record as the pod measures it (gtempo / full_rate,
            # scripts/gpu_c1e4m.sh), PPCL's extra calls and cut Bases included
            t = tempo[f"{'lmax' if ln['runner'] == 'c1e4m' else 'bench'}_{ln['brain']}"] / 3600
            n += n_of(ln)
            nr += len(ln["keys"])
            h += t * len(ln["keys"])
            hl += t * len(ln["keys"] - done)
        rows[L] = (n, nr, h, hl)
    tot = lambda n, nr, h, hl: {"episodes": n, "records": nr, "hours": round(h, 2),  # noqa: E731
                                "usd": round(RATE * h, 2), "left_hours": round(hl, 2)}
    add = lambda Ls: tot(*(sum(rows[L][j] for L in Ls) for j in range(4)))  # noqa: E731
    return {"usd_per_h": RATE, "tempo_s": tempo, "excludes": EXCLUDES,
            "using": {"mode": {n: v["mode"] for n, v in mode.items()}, "k": {b: v["k"] for b, v in k.items()}},
            "lines": {L: tot(*v) for L, v in rows.items()}, "core": add(CORE), "core_tail": add(LINES[1:9])}


def smoke_check(recs):
    """spec §12's smoke gate: PPCL and VS0.3 each with prefix_ok in >= 5 of the 6 cases. (pass, {arm: [ok, cases]})."""
    n = {}
    for a in ("PPCL", "VS0.3"):
        rs = [r for r in recs if r.get("kind") == "dynamic" and r["arm"] == a]
        n[a] = [len({r["case_id"] for r in rs if r.get("prefix_ok") is True}), len({r["case_id"] for r in rs})]
    return all(ok >= 5 for ok, _ in n.values()), n


def lk(r):
    """The record's key in the plan: cs.key and, LIBERO-MAX, its line's arms (conf): a c5 pass's Base and its line's
    Base share cs.key."""
    return cs.key(r) + (tuple(r["conf"]["arms"]),) if "case_id" in r else cs.key(r)


def lkeys(lns):
    """lk of every episode of the runner lines."""
    return {x + (tuple(ln["arms"]),) if x[0] == "m" else x for ln in lns for x in ln["keys"]}


def mp(mi, clu, brain, a, b, d, cases):
    """cs.mprs by case, Dynamic records only (a cut Base's success is never read): ({case: (task cluster, success a,
    success b)}, invalid pairs). A valid record's e is its Base's (prefix_ok), so a pair whose two e differ drops out
    too: across pods the new Base's e is not part 2's (spec §7.3), or the c5 pass's Base is not its line's."""
    x, y = mi.get((brain, "dynamic", a, d), {}), mi.get((brain, "dynamic", b, d), {})
    out, bad = {}, 0
    for c in sorted(x.keys() & y.keys() & cases):
        if cs.invalid(x[c]) or cs.invalid(y[c]) or x[c]["e"] != y[c]["e"]:
            bad += 1
        else:
            out[c] = (clu[c], x[c]["success"], y[c]["success"])
    return out, bad


def ppc_of(ppcl, ppc):
    """spec §7.2, mode С: PPC's episode is the c5 pass's where it ran, else PPCL's own, which is PPC's when the latch
    never held a chunk."""
    if ppc is not None:
        return ppc
    assert ppcl["latch"]["chunks"] == 0, f"a latched PPCL episode without its PPC: {cs.key(ppcl)}"
    return ppcl


def apart(d, thr, up, down, same):
    """spec §8, two-sided: up if diff >= thr and lo2.5 > 0, down if diff <= -thr and hi97.5 < 0, else same."""
    if d is None:
        return None
    return (up if d["diff_pp"] >= thr - TOL and d["lo2.5"] > 0 else
            down if d["diff_pp"] <= TOL - thr and d["hi97.5"] < 0 else same)


def grid(new, saved, mode, k, names, split):
    """spec §7.3, §8, §11 and its journal "ветки VS остановлены" on the pod's records (grid_all_C.jsonl: lines 1, 2, 3,
    7 and the c5 pass) and part 2's (saved: {SAVED name: records}; π's are read). The checks before reading; when they
    pass, the rule C5, the measurement И4, the claim and the report. Records off the plan (a probe's) are counted, not
    read; VS records (line 2's first VS_STOP cases) are on the plan and checked, counted, never read. Exit 2 when one
    key comes twice with other content (grid_all_C.jsonl and its C*.jsonl both)."""
    kp = need(k, P, "k")
    md = {n: need(mode, n, "mode") for n in SAVED}
    mo = {P: md["pi05_lmax"]}
    sel = {P: {kp}}  # spec §5: VS-общий, line 2's first cases
    by = {}
    for r in new:
        if by.setdefault(lk(r), r) != r:
            print(f"!!! {lk(r)} recorded twice with other content: pass grid_all_C.jsonl or its C*.jsonl, not both",
                  file=sys.stderr)
            raise SystemExit(2)
    ok = {x: r for x, r in by.items() if not r.get("skipped_reason")}  # a skipped episode counts as missing
    lines = {L: plan(L, mode, k, split) for L in LINES[1:-1]}
    want = {x: L for L, lns in lines.items() for x in lkeys(lns)}
    # c5: PPC where PPCL latched in lines 2, 3, 7
    lines["c5"] = plan("c5", mode, k, split, [r for x, r in ok.items() if want.get(x) in ("2", "3", "7")])
    want |= {x: "c5" for x in lkeys(lines["c5"])}
    used = {x: r for x, r in ok.items() if x in want}
    of = lambda *Ls: [r for x, r in used.items() if want[x] in Ls]  # noqa: E731
    extra = [r for x, r in by.items() if x not in want]
    cards = defaultdict(Counter)  # spec §7.1: a mode П probe holds on its card only: its lines' records ran there
    for x, r in by.items():
        if x in want and (n := PROBE.get(("m" if "case_id" in r else "b", r["brain"]))):
            cards[n][str(r["conf"].get("gpu"))] += 1
    on_probe = {n: {"mode": md[n], "probe": mode[n].get("gpu", []), "records": dict(cards[n]),
                    "pass": md[n] == "S" or set(cards[n]) <= set(mode[n].get("gpu", []))} for n in SAVED}
    comp = {}
    for L, lns in lines.items():  # spec §8: <= 1 % missing per line; c5 none (a latched PPCL is no PPC)
        exp = lkeys(lns)
        miss = len(exp - used.keys())
        ran = miss < len(exp)
        comp[L] = {"expected": len(exp), "missing": miss, "run": ran,
                   "pass": miss == 0 if L == "c5" else not ran or miss <= 0.01 * len(exp) + TOL}
    ksel = {}
    for b, L in ((P, "1"), (OP, "4")):  # spec §8: k_select.json is §5's rule on the M0 lines, on the records the
        rs = [r for x, r in by.items() if want.get(x) == L]  # pod's --select read: a skipped Base too
        try:
            got = select(rs, b, split) if rs else None
        except SystemExit:
            got = None
        ksel[b] = {"file": k.get(b), "recomputed": got, "pass": got == k.get(b)}
    other = Counter(f"{r['brain']}/{r['arm']}" for r in extra
                    if cs.method(r["arm"]) in VS and cs.method(r["arm"]) not in sel.get(r["brain"], ()))
    e300 = set(run.set_ids(split, "eval300"))
    sm = [r for r in saved["pi05_lmax"] if r["case_id"] in e300]  # OFT+, OFT: no line left
    sbase = {(r["brain"], r["case_id"], r["d"]): r for r in sm if r["kind"] == "base"}
    lbase = {(r["brain"], r["case_id"], r["d"]): r for r in of("2", "3") if r["kind"] == "base"}
    em = defaultdict(lambda: [0, 0])
    for x, r in lbase.items():  # spec §7.3: the new Base's e against part 2's, >= 98 % per brain and cell in mode П
        em[f"{x[0]}/d{x[2]}"][0] += x in sbase and sbase[x]["e"] == r["e"]
        em[f"{x[0]}/d{x[2]}"][1] += 1
    e_match = {c: {"equal": a, "total": n, "mode": mo[c.split("/")[0]],
                   "pass": mo[c.split("/")[0]] == "S" or a >= 0.98 * n - TOL} for c, (a, n) in sorted(em.items())}
    # spec §7: in mode П the new Base's pre-event calls ran in part 2's batches (π's fp32 chunk depends on its batch):
    # per case (e equal) its batch rows before e equal part 2's, >= 98 % per brain and cell
    br = defaultdict(lambda: {"calls_equal": 0, "calls": 0, "cases_equal": 0, "cases": 0, "rows_max": 0})
    for x, r in lbase.items():
        s = sbase.get(x)
        if s is None or s["e"] != r["e"]:
            continue
        lim = float("inf") if r["e"] is None else r["e"]
        a, b = ({t: row for t, *row in q.get("batch_rows") or [] if t < lim} for q in (r, s))
        v = br[f"{x[0]}/d{x[2]}"]
        v["calls_equal"] += sum(a.get(t) == row for t, row in b.items())
        v["calls"] += len(b)
        v["cases_equal"] += a == b
        v["cases"] += 1
        v["rows_max"] = max([v["rows_max"]] + [row[0] for q in (a, b) for row in q.values()])
    brows = {c: v | {"mode": mo[c.split("/")[0]], "pass": mo[c.split("/")[0]] == "S"
                     or v["cases_equal"] >= 0.98 * v["cases"] - TOL} for c, v in sorted(br.items())}
    c5_off = sum(lbase.get((r["brain"], r["case_id"], r["d"]), {}).get("e", "-") != r["e"] for r in of("c5")
                 if r.get("kind") == "base")  # its pairs drop out (mp)
    # part 2's records read: mode П C5's and И4's comparators (PPC, Gauto) and "none", mode С "none" only (report,
    # "через под")
    keep = lambda r, m: r["kind"] != "base" and r["arm"] in (  # noqa: E731
        ("none",) if m == "S" else ("none", "Gauto", "PPC"))
    km = [r for r in sm if keep(r, mo[P])]
    kb = [r for r in saved.get("pi05_bench", []) if keep(r, md["pi05_bench"])]
    groups, kbar = defaultdict(set), Counter()
    for r in list(used.values()) + km + kb:  # spec §7.3: conf of the new and the comparator records
        m = "m" if "case_id" in r else "b"
        groups[f"{m}/{r['brain']}/{r.get('d')}"].add(
            json.dumps({f: v for f, v in r["conf"].items() if f not in CONF_SKIP}, sort_keys=True))
        kbar[f"{m}/{r['brain']}/{r['arm']}"] += cs.method(r["arm"]) in ("Gauto", "Glead") and r["conf"].get(
            "kbar") != KBAR[m, r["brain"]]
    mixed = {g: sorted(v) for g, v in groups.items() if len(v) > 1}
    kbar = {g: n for g, n in kbar.items() if n}
    checks = {"records": len(new), "read": len(used), "skipped": len(by) - len(ok),
              "off_plan": dict(Counter(f"{r['brain']}/{r['arm']}" for r in extra)),
              "complete": comp, "k_select": ksel,
              "vs_label": {"selected": {b: sorted(v) for b, v in sel.items()}, "other": dict(other), "pass": not other},
              "e_match": e_match, "batch_rows_pre_event": brows, "on_probe_gpu": on_probe,
              "c5_base_e_mismatch": c5_off, "bases_cut": [
                  sum(bool(r.get("base_cut")) for r in used.values() if r["kind"] == "base"),
                  sum(r["kind"] == "base" for r in used.values())],
              "conf": {"except": CONF_SKIP, "mixed": mixed, "kbar_not_part2s": kbar, "pass": not mixed and not kbar}}
    checks["pass"] = bool(all(v["pass"] for d in (comp, ksel, e_match, brows, on_probe) for v in d.values())
                          and not other and checks["conf"]["pass"])
    if not checks["pass"]:
        return {"checks": checks}

    # spec journal "ветки VS остановлены": lines 2+'s VS records (line 2's first VS_STOP cases) are never read, only
    # counted; line 1 (M0, the calibration cases) is only for k
    vs = [r for r in of(*LINES[2:-1]) if cs.method(r["arm"]) in VS]
    rd = [r for r in of(*LINES[2:-1]) if cs.method(r["arm"]) not in VS]
    newm = [r for r in rd if "case_id" in r]
    mi = cs.mindex([r for r in km + newm if r["kind"] == "dynamic"])
    bi = cs.bindex(kb + [r for r in rd if "case_id" not in r])
    ppc = {cs.key(r): r for r in of("c5") if r["kind"] != "base"}
    if mo[P] == "S":
        for d in (0, 10, 20):
            mi[P, "dynamic", "PPC", d] = {c: ppc_of(r, ppc.get(cs.key(r | {"arm": "PPC"})))
                                          for c, r in mi.get((P, "dynamic", "PPCL", d), {}).items()}
    if md["pi05_bench"] == "S":
        for cell in ("A", "A20"):
            bi[P, "move", cell, "PPC"] = {e: ppc_of(r, ppc.get(cs.key(r | {"arm": "PPC"})))
                                          for e, r in bi.get((P, "move", cell, "PPCL"), {}).items()}
    one = {r["case_id"]: r for r in sm + newm}  # the task clusters: c1e4m_summary.summarize's clu, inline there
    clu = {c: maxwrap.base_task({"case_id": c, "task_suite_name": r["suite"], "task_name": r["task_name"],
                                 "substrate_variant": r["substrate_variant"]}, names) for c, r in one.items()}
    M = lambda b, x, y, d=0, cases=e300: mp(mi, clu, b, x, y, d, cases)  # noqa: E731
    B = lambda cells, x, y: cs.bprs(bi, (P,), cells, x, y)  # noqa: E731  (the bench has no invalid pair)
    R, inv = {}, {}

    def put(name, test, prs, bad, total=None, **fs):  # a measurement (MEAS): no fs, no threshold
        d = e2.diff_pp(prs)
        R[name] = {"test": test, **(e2.rnd(d) or {}), "invalid_pairs": bad} | {
            f: None if d is None else g(d) for f, g in fs.items()}
        inv[name] = bad, len(prs) + bad if total is None else total

    put("И4", "bench oracle A + A20, inits 0-9: Gauto - PPCL", B(("A", "A20"), "Gauto", "PPCL"), 0)
    # C5 on the pairs where PPCL's latch held a chunk (latched()'s flag): elsewhere PPCL is PPC by construction (spec
    # §7.2, §8); the pool of all pairs for reference, and for the invalid-pair limit. The bench by PPCL's record: bprs
    # keeps on its second arm
    pm, lat = [M(P, "PPCL", "PPC", d) for d in (0, 10, 20)], lambda r: r["latch"]["chunks"] > 0  # noqa: E731
    pool = ([(("m", *cl), a, b) for p, _ in pm for cl, a, b in p.values()]
            + [(("b", *cl), a, b) for cl, a, b in B(("A", "A20"), "PPCL", "PPC")])
    on = ([(("m", *cl), a, b) for d, (p, _) in zip((0, 10, 20), pm) for c, (cl, a, b) in p.items()
           if lat(mi[P, "dynamic", "PPCL", d][c])]
          + [(("b", *cl), a, b) for cl, b, a in cs.bprs(bi, (P,), ("A", "A20"), "PPC", "PPCL", keep=lat)])
    nc = len({cl for cl, _, _ in on})
    few = len(on) < 20 or nc < 5  # spec §8: a degenerate interval, not read
    put("C5", "pi05, the pairs of M1 300 + D 600 + bench 520 where PPCL's latch held a chunk: (PPCL - PPC), the 95% "
              "CI without 0; fewer than 20 such pairs or 5 tasks: not read", [] if few else on, sum(b for _, b in pm),
        len(pool) + sum(b for _, b in pm),
        outcome=lambda d: apart(d, 0, "меняет: PPCL лучше", "меняет: PPCL хуже", "не меняет различимо"))
    R["C5"] |= {"latched_pairs": len(on), "latched_clusters": nc, "pool": e2.rnd(e2.diff_pp(pool))} | (
        {"outcome": "не читается: латч почти не включался"} if few else {})
    reads = {"И4": {"Gauto", "PPCL"}, "C5": {"PPCL"}}  # its arms
    for n, Ls in DEP.items():  # a line not run at all (spec §12: waits for the owner) with an arm the rule or
        # measurement reads: not read (a measurement: no estimate)
        if miss := [L for L in Ls if not comp[L]["run"] and reads[n] & {a for ln in lines[L] for a in ln["arms"]}]:
            R[n] |= dict.fromkeys(EST if n in MEAS else ("outcome",)) | {"lines_not_run": miss}
    checks["invalid_pairs"] = {n: {"pairs": t, "invalid": b, "pass": b <= 0.02 * t + TOL} for n, (b, t) in inv.items()}
    if not all(v["pass"] for v in checks["invalid_pairs"].values()):
        checks["pass"] = False
        return {"checks": checks}
    meas = {n: R.pop(n) for n in MEAS}
    claims = {}
    for c, n in CLAIMS.items():
        o = R[n]["outcome"]
        claims[c] = ("НЕ ЧИТАЛОСЬ" if o is None else o.upper() if o.startswith("не читается") else
                     "ДА" if o.startswith("меняет") else "НЕТ")
    return {"checks": checks, "rules": R, "measurements": meas, "claims": claims,
            "report": grid_report(mi, bi, M, B, one, e300, mo, md, used, k, mode, checks) | {"vs_unread": len(vs)}}


def grid_report(mi, bi, M, B, one, e300, mo, md, used, k, mode, checks):
    """spec §8 "только отчёт", §11 and its journal "ветки VS остановлены" (mi, bi hold no VS arm): PPCL - none, Gauto -
    PPCL (π LIBERO-MAX: both on the pod), PPCL - PPC; "через под": a comparison with part 2's records in mode С."""
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    dm = lambda b, x, y, d=0, cases=e300: e2.rnd(e2.diff_pp(list(M(b, x, y, d, cases)[0].values())))  # noqa: E731
    db = lambda cells, x, y: e2.rnd(e2.diff_pp(B(cells, x, y)))  # noqa: E731
    via = lambda m: {"через под": m == "S"}  # noqa: E731
    rep = defaultdict(dict)
    for d in (0, 10, 20):
        rep["ppcl_none"][f"{P}/d{d}"] = (dm(P, "PPCL", "none", d) or {}) | via(mo[P])
        rep["gauto_ppcl"][f"{P}/d{d}"] = dm(P, "Gauto", "PPCL", d)
        rep["ppcl_ppc"][f"{P}/d{d}"] = dm(P, "PPCL", "PPC", d)
    for c in ("A", "A20"):
        rep["ppcl_none"][f"bench/{c}"] = (db((c,), "PPCL", "none") or {}) | via(md["pi05_bench"])
        rep["ppcl_ppc"][f"bench/{c}"] = db((c,), "PPCL", "PPC")
    dist = {c: one[c]["distance_m"] for c in e300 if c in one and one[c]["distance_m"] is not None}
    near, far = {c for c, v in dist.items() if v < 0.09}, {c for c, v in dist.items() if v >= 0.09}
    for x, y in (("PPCL", "none"), ("Gauto", "PPCL"), ("PPCL", "PPC")):
        rep["slices_M1"][f"{P}/{x}-{y}"] = {"6cm": dm(P, x, y, 0, near), "12cm": dm(P, x, y, 0, far)}
    for key, g in sorted(list(mi.items()) + list(bi.items()), key=lambda kv: str(kv[0])):  # per arm and cell
        g, a = list(g.values()), key[2 if key[1] == "dynamic" else 3]
        name = "/".join(map(str, key))
        post = [sum(t >= r["e"] for t, _ in r["plans_in"]) for r in g if r.get("e") is not None]
        rep["calls"][name] = {"n": len(g), "sched": mean([r["calls_sched"] for r in g]),
                              "trig": mean([r["calls_trig"] for r in g]), "plans_after_event": mean(post)}
        if a == "PPCL":
            lt = [r["latch"] for r in g]
            rep["latch"][name] = {"n": len(lt), "latched_share": mean([x["chunks"] > 0 for x in lt])} | {
                f"{f}_per_episode": mean([x[f] for x in lt]) for f in ("chunks", "trig", "resets")}
        if cs.method(a) in ("Gauto", "PPCL"):
            rep["vs_metrics"][name] = {f: mean([r[f] for r in g if r.get(f) is not None])
                                       for f in ("jerk", "path_m", "grasp_miss_m")}
    rep["gpu"] = {b: sorted({str(r["conf"].get("gpu")) for r in used.values() if r["brain"] == b})
                  for b in sorted({r["brain"] for r in used.values()})}
    return dict(rep) | {"k_select": k, "mode": mode, "e_match": checks["e_match"],
                        "batch_rows_pre_event": checks["batch_rows_pre_event"],
                        "c5_base_e_mismatch": checks["c5_base_e_mismatch"], "invalid_pairs": checks["invalid_pairs"]}


def finish(res, out=None):
    """Print the grid's result (only the checks when they fail), write it to out, exit 3 on a failed check."""
    text = e2.dump(res)
    print(text)
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(text + "\n")
    print(f"C1-E4 controls checks before reading: {'PASS' if res['checks']['pass'] else 'FAIL, the rules not read'}")
    if not res["checks"]["pass"]:
        raise SystemExit(3)


def merge(path, d):
    """Write d into the JSON object at path (k_select.json: one brain at a time)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps((json.loads(p.read_text()) if p.exists() else {}) | d, indent=1) + "\n")


def first_rec(path):
    """The first record (a line with "kind") of a JSON lines file."""
    with open(path) as f:
        return next(json.loads(x) for x in f if '"kind"' in x)


def exits(f, code):
    try:
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            f()
        raise AssertionError(f"no exit {code}")
    except SystemExit as ex:
        assert ex.code == code, (ex.code, code)


@mock.patch.dict(VS_STOP, {"plus": 90, "pro": 0})  # the synthetic stop: the owner sets the real one at the stop
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
                            for i in range(10)]}  # Boracle: its arms in part 2's order
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
        exits(lambda: select(rs, P, split), 2)

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
    # line 2 (spec journal "ветки VS остановлены"): the first 90 Plus cases as they ran (the pod's: kπ = 1.0, mode С),
    # the other 97 Plus and all 113 PRO without VS, each case once
    e3 = cs.sources(run.set_ids(split, "eval300"))
    l2 = plan("2", *mk("VS1.0", sp="S"), split)
    assert [(ln["id"], ln["arms"], n_of(ln)) for ln in l2] == [
        ("C2_pi05_eval300_plus_d0", ["PPCL", "VS1.0", "Gauto"], 270),
        ("C2_nv_pi05_eval300_plus_d0", ["PPCL", "Gauto"], 194), ("C2_nv_pi05_eval300_pro_d0", ["PPCL", "Gauto"], 226)]
    assert l2[0]["args"] == ("--brain pi05 --source plus --set eval300 --arms=PPCL,VS1.0,Gauto --d 0 --cases "
                             f"{','.join(e3['plus'][:90])} --kbar 0.306 --base-prefix"), l2[0]["args"]
    assert l2[1]["args"] == ("--brain pi05 --source plus --set eval300 --arms=PPCL,Gauto --d 0 --cases "
                             f"{','.join(e3['plus'][90:])} --kbar 0.306 --base-prefix"), l2[1]["args"]
    assert f" --cases {','.join(e3['pro'])} --kbar 0.306 " in l2[2]["args"]
    assert sorted(x[2] for ln in l2 for x in ln["keys"] if x[3] == "base") == sorted(run.set_ids(split, "eval300"))
    with mock.patch.dict(VS_STOP, plus=0, pro=113):  # none of a source done with VS, all of the other
        assert [(ln["id"], n_of(ln)) for ln in plan("2", *mk("VS1.0", sp="S"), split)] == [
            ("C2_pi05_eval300_pro_d0", 339), ("C2_nv_pi05_eval300_plus_d0", 374)]
    for bad in ({"plus": 91}, {"pro": 112}):  # off the runner's groups and not the whole source
        with mock.patch.dict(VS_STOP, bad):
            try:
                plan("2", *mk("VS1.0", sp="S"), split)
                bad = None
            except AssertionError:
                pass
        assert bad, "a VS_STOP off the runner's groups was accepted"
    assert arms("2", "VS1.0", sp="P") == ["PPCL"]  # VS1.0 = part 2's G in mode П
    assert [ln["arms"] for ln in plan("2", *mk("VS0.3", sp="P"), split)] == [["PPCL", "VS0.3"], ["PPCL"], ["PPCL"]]
    assert "--kbar" not in plan("2", *mk("VS0.3", sp="P"), split)[0]["args"]
    l3 = plan("3", *mk(sp="P"), split)  # no VS: no k needed
    assert [ln["id"] for ln in l3] == ["C3_pi05_eval300_plus_d10", "C3_pi05_eval300_pro_d10",
                                       "C3_pi05_eval300_plus_d20", "C3_pi05_eval300_pro_d20"]
    assert l3[0]["arms"] == ["PPCL"] and "--d 10 " in l3[0]["args"] and "--kbar" not in l3[0]["args"]
    assert arms("3", sp="S") == ["PPCL", "Gauto"] and sum(map(n_of, plan("3", *mk(sp="S"), split))) == 1200
    l7 = plan("7", *mk(sb="P"), split)  # the pod's: bench mode П, the comparators part 2's
    assert [(ln["id"], n_of(ln)) for ln in l7] == [("C7_pi05", 520)] and l7[0]["args"] == (
        "--brain pi05 --kinds move --cells A,A20 --methods PPCL --inits 0-9"), l7
    l7 = plan("7", *mk(sb="S"), split)
    assert n_of(l7[0]) == 1040 and l7[0]["args"].endswith("--methods PPCL,Gauto --inits 0-9 --kbar 0.324")
    full = mk("VS0.3", "VS0.1", "S", "S", "S")
    assert all(plan(L, *full, split) == plan(L, None, None, split) == [] for L in "4568")  # dropped
    every = [ln for L in LINES[:-1] for ln in plan(L, *full, split)]
    assert len({ln["id"] for ln in every}) == len(every)  # one --out per runner line
    for L in LINES[:-1]:
        assert {ln["phase"] for ln in plan(L, *full, split)} <= {f"c{L}"}
    for ln in every:
        if ln["runner"] == "c1e4m":
            assert "--base-prefix" in ln["args"] and f"--source {ln['source']} " in ln["args"] + " "
            assert all(x[2].startswith("pro-") == (ln["source"] == "pro") for x in ln["keys"])
        assert ("--kbar" in ln["args"]) == bool({"Gauto", "Glead"} & set(map(cs.method, ln["arms"])))
    for L, mo in (("2", mk("VS0.3")), ("2", mk(sp="S")), ("3", mk("VS0.3", sb="S")),
                  ("7", mk("VS0.3", sp="S"))):  # without its mode or k
        exits(lambda: plan(L, *mo, split), 3)

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
    # the PPC pass on whole runner groups (spec §7: the line's Base batches): Plus 0-5 for e10[1], PRO 0-5 for pro;
    # the bench: every init of the latched task in both cells
    c5 = plan("c5", *mk(sp="S", sb="S"), split, recs)
    assert [(ln["id"], ln["arms"], n_of(ln)) for ln in c5] == [
        ("C5ppc_pi05_eval300_plus_d0", ["PPC"], 6), ("C5ppc_pi05_eval300_pro_d10", ["PPC"], 6),
        ("C5ppc_pi05", ["PPC"], 20)], c5
    assert c5[0]["args"].endswith(f"--d 0 --cases {','.join(plus['eval300'][:6])} --base-prefix") and c5[0][
        "phase"] == "cc5", c5[0]["args"]
    assert c5[1]["args"].endswith(f"--cases {','.join(cs.sources(run.set_ids(split, 'eval300'))['pro'][:6])} "
                                  "--base-prefix"), c5[1]["args"]
    assert c5[2]["args"] == f"--brain pi05 --kinds move --cells A,A20 --methods PPC --tasks {orx.TASKS[1][0]}:1 " \
                            "--inits 0-9", c5[2]["args"]
    g2 = plan("c5", *mk(sp="S", sb="P"), split, recs + [mrec(P, plus["eval300"][7], "dynamic", "PPCL", 0) | lat(1)])
    assert len(g2[0]["keys"]) == 2 * 12, g2  # a latched case of the second group adds that group
    assert [ln["id"] for ln in plan("c5", *mk(sp="P", sb="S"), split, recs)] == ["C5ppc_pi05"]
    assert plan("c5", *mk(sp="P", sb="P"), split, recs) == []
    assert plan("c5", *mk(sp="S", sb="S"), split, [x | lat(0) for x in recs]) == []
    with tempfile.TemporaryDirectory() as tmp:
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

    # the forecast: 36 s per record everywhere -> hours = records / 100; every record (PPCL, a cut Base) weighs 1
    t36 = dict.fromkeys(TEMPO, 36.0)
    f = forecast(t36, split=split)  # no mode, no k: the most expensive case
    assert f["using"] == {"mode": {"pi05_lmax": "S", "oftplus_lmax": "S", "pi05_bench": "S"},
                            "k": {P: "VS0.3", OP: "VS0.1"}}, f["using"]
    assert f["excludes"] == EXCLUDES and "smoke" in f["excludes"]
    # line 1 by hand: 3 VS arms on the 60 calibration cases + their 60 cut Bases = 240 records; x 36 s = 2.4 h
    assert f["lines"]["1"] == {"episodes": 180, "records": 240, "hours": 2.4, "usd": 0.89, "left_hours": 2.4}, f["lines"]
    # 2: PPCL, VS0.3, Gauto + Base on 90 cases, PPCL, Gauto + Base on 210; 3: PPCL, Gauto + Base, d 10, 20; 7: 2 arms x
    # 2 cells x 260; 4, 5, 6, 8: dropped
    assert [f["lines"][L]["records"] for L in LINES[1:9]] == [240, 990, 1800, 0, 0, 0, 1040, 0]
    assert [f["lines"][L]["episodes"] for L in LINES[1:9]] == [180, 690, 1200, 0, 0, 0, 1040, 0]
    assert f["core"] == {"episodes": 3110, "records": 4070, "hours": 40.7, "usd": round(40.7 * RATE, 2),
                         "left_hours": 40.7} and f["core_tail"]["hours"] == 40.7, f["core"]
    f = forecast(t36, *mk("VS1.0", "VS1.0", "P", "P", "P"), split=split)  # every coincidence: PPCL alone
    assert [f["lines"][L]["episodes"] for L in LINES[1:9]] == [180, 300, 600, 0, 0, 0, 520, 0]
    assert f["lines"]["5"] == {"episodes": 0, "records": 0, "hours": 0, "usd": 0, "left_hours": 0}
    done = set().union(*(ln["keys"] for ln in plan("1", None, None, split)))
    f = forecast(t36, *full, done=done, split=split)
    assert f["lines"]["1"]["left_hours"] == 0 and f["lines"]["2"]["left_hours"] == f["lines"]["2"]["hours"]
    assert f["core"]["left_hours"] == round(sum(f["lines"][L]["records"] for L in CORE[1:]) / 100, 2)

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
    selftest_grid(split, cal)
    print("selftest ok")


def selftest_grid(split, cal):
    """The grid (spec §7.3, §8, §11): part 2's records and the pod's on the plan's keys. Success by the case's place in
    cycles of 100 (pct: the arm's share in %, by (brain, arm, d), (cell, arm) or arm; else 50), so every difference is
    exact; task clusters from c1e4m's fake LIBERO names."""
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    e300 = run.set_ids(split, "eval300")
    pos = {c: j for j, c in enumerate(split["eval"])} | {c: j for j, c in enumerate(cal)}
    bpos = {(s, t, i): i * 26 + j for j, (s, t) in enumerate(orx.TASKS) for i in range(10)}
    cf = {"revision": "r", "precision": "fp32", "s": 10, "R": 360, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01,
          "split": split["sha256"]}
    gpu = {"gpu": "NVIDIA GeForce RTX 4090"}

    def rec(x, pct, new, kk):  # x: lk; new: the pod's (gpu, a cut Base); kk: π's VS label with the least grasp miss
        if x[0] == "m":
            _, b, c, kind, a, d, _, arms = x
            h = sum(map(ord, c))
            s = list(run.MAX_STEPS)[h % 4]
            kbar = KBAR["m", b] if {"Gauto", "Glead"} & set(map(cs.method, arms)) else None  # VS lines: none
            r = {"brain": b, "case_id": c, "kind": kind, "arm": a, "d": d,
                 "conf": cf | {"d": d, "arms": list(arms), "kbar": kbar},
                 "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None,
                 "distance_m": (0.06, 0.12)[h % 2], "e": 30,
                 "success": pos[c] % 100 < pct.get((b, a, d), pct.get(a, 50)),
                 "prefix_ok": None if kind == "base" else True, "replay_miss": None, "skipped_reason": None,
                 "batch_rows": [[0, 6, 1], [10, 6, 1], [30, 6, 1]], "plans_in": [[0, 0], [10, 10], [30, 30 + d]]}
            if kind == "base" and new:
                r |= {"base_cut": True, "success": None, "grasp_ok": None}
        else:
            _, b, kind, cell, a, s, t, i = x
            r = {"brain": b, "kind": kind, "cell": cell, "arm": a, "suite": s, "task": t, "init": i,
                 "conf": {f: cf[f] for f in ("revision", "precision", "t_ramp", "k_p", "tau_k")} | {
                     "kbar": KBAR["b", P], "part2": {"cv_bands": [0.02286, 0.007031], "tau_close": 13}},
                 "success": bpos[s, t, i] % 100 < pct.get((cell, a), pct.get(a, 50)), "plans_in": [[0, 0]]}
        r |= {"method": cs.method(a), "calls_sched": 20, "calls_trig": int(cs.method(a) == "PPCL"), "jerk": 1.0,
              "path_m": 1.0, "grasp_miss_m": 0.01 + 0.01 * (cs.method(a) in VS and a != kk)}
        r["conf"] |= gpu if new else {}
        return r | ({"latch": {"trig": 0, "chunks": 0, "resets": 0}} if cs.method(a) == "PPCL" else {})

    def world(modes, kp, pct=None, edit=lambda r: r):
        """mode.json, k_select.json (select on the generated M0 line: kp wins), the pod's records of every planned
        line (edit: on each) and part 2's of SAVED (300 + 5 OFT+ cases: only π's eval300 is read)."""
        pct = pct or {}
        mode = {n: {"mode": m, "gpu": [gpu["gpu"]]} for n, m in zip(SAVED, modes)}
        mk = lambda x: edit(rec(x, pct, True, kp))  # noqa: E731
        new = [mk(x) for x in lkeys(plan("1", split=split))]
        k = {P: select(new, P, split)}
        new += [mk(x) for L in "237" for x in lkeys(plan(L, mode, k, split))]
        new += [mk(x) for x in lkeys(plan("c5", mode, k, split, new))]
        old = lambda b, ids, arms, d=0: [rec(("m", b, c, kd, a, d, False, tuple(arms)), pct, False, kp)  # noqa: E731
                                         for c in ids for kd, a in [("dynamic", a) for a in arms] + [("base", "none")]]
        bold = lambda cells, arms: [rec(("b", P, "move", c, a, *e), pct, False, kp)  # noqa: E731
                                    for c in cells for a in arms for e in EPS]
        saved = {"pi05_lmax": old(P, e300, cs.ARMS + cs.M2) + old(P, e300, cs.D_ARMS, 10) + old(P, e300, cs.D_ARMS, 20),
                 "oftplus_lmax": old(OP, split["eval"][:305], cs.ARMS),
                 "pi05_bench": bold(("A", "A20"), cs.B_ARMS) + bold(("A20",), cs.NOISY)}
        return new, saved, mode, k

    # the preregistered thresholds (spec §8), directly: equality passes, "> 0" strict
    x = lambda d=0.0, lo2=-9.0, lo5=-9.0, hi=9.0: {"diff_pp": d, "lo2.5": lo2, "lo5_one_sided": lo5,  # noqa: E731
                                                     "hi97.5": hi}
    assert [apart(x(*v), 5, "up", "down", "same") for v in (
        (5.0, 1e-9), (5.0, 0.0), (4.99, 1.0), (-5.0, -9, -9, -1e-9), (-5.0, -9, -9, 0.0), (-4.99, -9, -9, -1.0))] == [
        "up", "same", "same", "down", "same", "same"]

    # the pod's (spec journal "ветки VS остановлены"): π LIBERO-MAX mode С, bench mode П, kπ = 1.0, line 2's first 90
    # Plus cases with VS1.0. И4 (part 2's Gauto - the pod's PPCL) at +18 of 520 (A 52 %, A20 54 %: 6 + 12 episodes).
    # C5: PPCL latched on 15 cases at d 0 (in 6 runner groups, 36 cases, before and after the stop) and 15 bench
    # episodes (A20, 13 tasks); PPC of the c5 pass on those groups, part 2's on the bench (mode П), PPCL itself
    # elsewhere: PPC 40 % against PPCL's 50 %, so + 1 on the 30 latched pairs (40-44 of 100) and, in the pool, on the
    # pass's 24 cases and 2 x 30 bench episodes at 40-49 of 100
    lat = lambda r: r["method"] == "PPCL" and (  # noqa: E731
        pos[r["case_id"]] % 100 in range(40, 45) and r["d"] == 0 if "case_id" in r
        else bpos[r["suite"], r["task"], r["init"]] % 100 in range(40, 45) and r["cell"] == "A20")
    pct = {(P, "Gauto", 0): 53, (P, "Gauto", 10): 61, (P, "Gauto", 20): 65, ("A", "Gauto"): 52, ("A20", "Gauto"): 54,
           "PPC": 40}
    new, saved, mode, k = world("SSP", "VS1.0", pct,
                                lambda r: r | {"latch": {"trig": 1, "chunks": 2, "resets": 0}} if lat(r) else r)
    assert k == {P: k[P]} and k[P]["k"] == "VS1.0" and k[P]["scores"]["VS0.3"] == 0.02, k
    probe1 = rec(("m", P, e300[0], "dynamic", "T0", 0, False, cs.ARMS + cs.M2), pct, True, "VS1.0")
    res = grid(new + [probe1] + new[:5], saved, mode, k, names, split)  # new[:5]: an identical copy is fine
    ch = res["checks"]
    assert ch["pass"] and ch["off_plan"] == {f"{P}/T0": 1}, ch
    assert ch["complete"]["2"] == {"expected": 90 * 4 + 210 * 3, "missing": 0, "run": True, "pass": True}, ch
    assert ch["complete"]["c5"] == {"expected": 36 * 2, "missing": 0, "run": True, "pass": True}, ch  # no bench pass
    assert ch["complete"]["3"]["expected"] == 1800 and ch["complete"]["7"]["expected"] == 520, ch["complete"]
    assert all(ch["complete"][L] == {"expected": 0, "missing": 0, "run": False, "pass": True} for L in "4568")
    r, ms, cl, rep = res["rules"], res["measurements"], res["claims"], res["report"]
    assert list(r) == ["C5"] and list(ms) == ["И4"] and cl == {"ЛАТЧ МЕНЯЕТ PPC": "ДА"}, (r, ms, cl)
    # a measurement: the estimate and its 95% CI only, no verdict and no threshold (spec §8)
    assert all(not {"pass", "superior", "outcome"} & set(v) and not re.search("[<>≤≥]", v["test"]) and {
        "diff_pp", "lo2.5", "hi97.5", "pairs", "clusters", "invalid_pairs"} <= set(v) for v in ms.values()), ms
    assert ms["И4"]["pairs"] == 520 and ms["И4"]["diff_pp"] == round(100 * 18 / 520, 3), ms
    # C5 on the 30 latched pairs (spec §8); the pool of 1420 for reference
    assert r["C5"]["pairs"] == r["C5"]["latched_pairs"] == 30 and r["C5"]["diff_pp"] == 100, r["C5"]
    assert r["C5"]["clusters"] == r["C5"]["latched_clusters"] > 13 and r["C5"]["outcome"] == "меняет: PPCL лучше"
    assert r["C5"]["pool"]["pairs"] == 1420 and r["C5"]["pool"]["diff_pp"] == round(100 * 84 / 1420, 3), r["C5"]
    assert r["C5"]["pool"]["clusters"] == rep["gauto_ppcl"][f"{P}/d0"]["clusters"] + 26, r["C5"]
    # the report: no VS anywhere, the 90 VS1.0 records of line 2 counted only
    assert rep["vs_unread"] == 90 and not [x for f in ("ppcl_none", "gauto_ppcl", "ppcl_ppc", "slices_M1", "calls",
                                                       "latch", "vs_metrics") for x in rep[f] if "VS" in x], rep
    assert {x.split("/")[2 + ("/move/" in x)] for x in rep["vs_metrics"]} == {"Gauto", "PPCL"}, rep["vs_metrics"]
    assert rep["latch"][f"{P}/dynamic/PPCL/0"]["latched_share"] == 0.05 and rep["latch"][f"{P}/move/A20/PPCL"][
        "latched_share"] == round(15 / 260, 4), rep["latch"]
    assert rep["ppcl_none"][f"{P}/d0"]["через под"] and rep["ppcl_none"][f"{P}/d0"]["pairs"] == 300
    assert not rep["ppcl_none"]["bench/A20"]["через под"] and rep["ppcl_none"]["bench/A20"]["pairs"] == 260
    assert [rep["gauto_ppcl"][f"{P}/d{d}"]["diff_pp"] for d in (0, 10, 20)] == [3, 11, 15], rep["gauto_ppcl"]
    assert rep["ppcl_ppc"][f"{P}/d0"]["diff_pp"] == 8 and rep["ppcl_ppc"][f"{P}/d10"]["diff_pp"] == 0, rep["ppcl_ppc"]
    assert rep["ppcl_ppc"]["bench/A"]["diff_pp"] == round(3000 / 260, 3), rep["ppcl_ppc"]
    assert set(rep["slices_M1"]) == {f"{P}/PPCL-none", f"{P}/Gauto-PPCL", f"{P}/PPCL-PPC"} and all(
        sum(v[s]["pairs"] for s in ("6cm", "12cm")) == 300 for v in rep["slices_M1"].values()), rep["slices_M1"]
    assert rep["calls"][f"{P}/dynamic/PPCL/10"]["plans_after_event"] == 1 and rep["gpu"][P] == [gpu["gpu"]]
    assert rep["calls"][f"{P}/dynamic/none/0"]["n"] == 300 and f"{OP}/dynamic/none/0" not in rep["calls"]
    assert rep["batch_rows_pre_event"][f"{P}/d0"]["cases_equal"] == 300
    # the VS records are never read: any success, calls or metric of theirs leaves every read number as it was
    vs_off = [x | {"success": not x["success"], "calls_sched": 99, "jerk": 9.0}
              if x["arm"] == "VS1.0" and x["kind"] == "dynamic" and x["case_id"] in e300 else x for x in new]
    assert vs_off != new and grid(vs_off + [probe1] + vs_off[:5], saved, mode, k, names, split) == res
    ppcl = next(x for x in new if lat(x) and "case_id" in x)
    try:
        ppc_of(ppcl, None)  # a latched PPCL is no PPC
        raise AssertionError("a latched PPCL episode stood in for PPC")
    except AssertionError as ex:
        assert "without its PPC" in str(ex), ex
    assert ppc_of(ppcl | {"latch": {"trig": 0, "chunks": 0, "resets": 0}}, None)["arm"] == "PPCL"
    exits(lambda: grid(new + [new[7] | {"success": not new[7]["success"]}], saved, mode, k, names, split), 2)
    # the checks before reading, each alone: the c5 pass short of one episode; line 7 short of > 1 %; a VS label not
    # the selected one; a conf field (an unread VS record's too); a Gauto's kbar not part 2's
    c5x = next(x for x in new if x["arm"] == "PPC" and "case_id" in x)
    l7 = [x for x in new if x.get("cell") and x["arm"] == "PPCL"][:6]  # 6 of the 520 episodes of line 7
    v1 = next(x for x in new if x["arm"] == "VS1.0" and x.get("case_id") in e300)
    for bad, why in (([x for x in new if x is not c5x], "complete"), ([x for x in new if x not in l7], "complete"),
                     (new + [v1 | {"arm": "VS0.1", "method": "VS0.1"}], "vs_label"),
                     ([x | {"conf": x["conf"] | {"t_ramp": 4}} if x.get("cell") == "A" else x for x in new], "conf"),
                     ([x | {"conf": x["conf"] | {"t_ramp": 4}} if x is v1 else x for x in new], "conf"),
                     ([x | {"conf": x["conf"] | {"kbar": 0.3}} if x["arm"] == "Gauto" else x for x in new], "conf")):
        ch = grid(bad, saved, mode, k, names, split)
        c = ch["checks"][why]
        assert not ch["checks"]["pass"] and "rules" not in ch, why
        assert not c["pass"] if "pass" in c else not all(v["pass"] for v in c.values()), (why, c)
    # the line-2 split: records stopped at 90 Plus cases read against a plan stopped at 96 miss the VS-era keys of
    # cases 90-95 (4 each), and their records without VS (3 each) are off the plan
    with mock.patch.dict(VS_STOP, plus=96):
        ch = grid(new, saved, mode, k, names, split)["checks"]
    assert ch["complete"]["2"] == {"expected": 96 * 4 + 204 * 3, "missing": 24, "run": True, "pass": False} and ch[
        "off_plan"] == {f"{P}/{a}": 6 for a in ("PPCL", "Gauto", "none")}, ch["complete"]["2"]
    with contextlib.redirect_stderr(io.StringIO()):
        ch = grid(new, saved, mode, k | {P: k[P] | {"n_valid": 59}}, names, split)["checks"]
    assert not ch["pass"] and not ch["k_select"][P]["pass"] and ch["k_select"][OP]["pass"], ch["k_select"]
    cb = next(x for x in new if x["kind"] == "base" and x["brain"] == P and x["case_id"] == cal[0])
    ch = grid([x | {"skipped_reason": "motion_blur"} if x is cb else x for x in new], saved, mode, k, names,
              split)["checks"]
    assert ch["pass"] and ch["skipped"] == 1 and ch["k_select"][P]["pass"], ch  # the pod's --select reads it too
    # the invalid-pair limit (spec §8): 28 of C5's 1420 pairs (2 %) are read, 29 stop the reading
    for n_bad, ok in ((28, True), (29, False)):
        off = set(e300[:n_bad])
        res = grid([x | {"prefix_ok": False} if x.get("case_id") in off and x["arm"] == "PPCL" and x["d"] == 0 else x
                    for x in new], saved, mode, k, names, split)
        assert res["checks"]["invalid_pairs"]["C5"] == {"pairs": 1420, "invalid": n_bad, "pass": ok} and (
            "rules" in res) == ok == res["checks"]["pass"], res["checks"]["invalid_pairs"]

    # mode С everywhere, kπ = VS0.3: И4 at -60 of 520 (Gauto 40 % on the pod); the latch never held: C5 not read
    pct = {(P, "Gauto", 0): 44, ("A", "Gauto"): 40, ("A20", "Gauto"): 40}
    new, saved, mode, k = world("SSS", "VS0.3", pct)
    res = grid(new, saved, mode, k, names, split)
    r, ms, cl = res["rules"], res["measurements"], res["claims"]
    assert ms["И4"]["diff_pp"] == round(-100 * 60 / 520, 3) and res["report"]["vs_unread"] == 90, ms
    assert r["C5"]["outcome"] == "не читается: латч почти не включался" and "diff_pp" not in r["C5"], r["C5"]
    assert r["C5"]["latched_pairs"] == r["C5"]["latched_clusters"] == 0 and r["C5"]["pool"]["pairs"] == 1420
    assert r["C5"]["pool"]["diff_pp"] == 0 and cl["ЛАТЧ МЕНЯЕТ PPC"] == "НЕ ЧИТАЕТСЯ: ЛАТЧ ПОЧТИ НЕ ВКЛЮЧАЛСЯ", r
    assert res["checks"]["complete"]["c5"]["expected"] == 0 and res["checks"]["complete"]["7"]["expected"] == 1040
    res = grid([x for x in new if x.get("cell") != "A" or x["arm"] == "PPCL"], saved, mode, k, names, split)
    assert res["checks"]["complete"]["7"] == {"expected": 1040, "missing": 260, "run": True, "pass": False}
    res = grid([x for x in new if "cell" not in x], saved, mode, k, names, split)
    r, ms = res["rules"], res["measurements"]  # line 7 not run at all: И4 and C5 (a part of its pool) are not read
    assert res["checks"]["complete"]["7"]["run"] is False and r["C5"]["outcome"] is None and ms["И4"]["pairs"] is None
    assert ms["И4"]["diff_pp"] is None and ms["И4"]["lines_not_run"] == ["7"] == r["C5"]["lines_not_run"], (r, ms)
    assert res["report"]["gauto_ppcl"][f"{P}/d0"]["diff_pp"] == -6 and res["claims"]["ЛАТЧ МЕНЯЕТ PPC"] == "НЕ ЧИТАЛОСЬ"

    # mode П everywhere, kπ = 1.0: no VS ran (VS1.0 = part 2's G), the comparators part 2's
    pct = {(P, "Gauto", 0): 53}
    new, saved, mode, k = world("PPP", "VS1.0", pct)
    assert not [x for x in new if x.get("case_id") in e300 and cs.method(x["arm"]) in VS]
    res = grid(new, saved, mode, k, names, split)
    r, ms, rep = res["rules"], res["measurements"], res["report"]
    assert rep["vs_unread"] == 0 and rep["gauto_ppcl"][f"{P}/d0"]["diff_pp"] == 3 and ms["И4"]["diff_pp"] == 0, rep
    assert r["C5"]["pool"]["pairs"] == 1420 and r["C5"]["pool"]["diff_pp"] == 0, r
    assert not rep["ppcl_none"][f"{P}/d0"]["через под"]
    # C5 on the latched pairs (spec §8): 20 over 5 tasks are read; 19, or 20 over 4 tasks, are not
    e20 = {(*st, i) for st in orx.TASKS[:5] for i in range(4)}
    for eps, ok in ((e20, True), (e20 - {(*orx.TASKS[0], 0)}, False), ({(*st, i) for st in orx.TASKS[:4]
                                                                          for i in range(5)}, False)):
        c5 = grid([x | {"latch": {"trig": 1, "chunks": 1, "resets": 0}} if x["arm"] == "PPCL" and x.get("cell") == "A20"
                   and (x["suite"], x["task"], x["init"]) in eps else x for x in new], saved, mode, k, names, split)
        assert c5["rules"]["C5"]["latched_pairs"] == len(eps) and c5["claims"]["ЛАТЧ МЕНЯЕТ PPC"] == (
            "НЕТ" if ok else "НЕ ЧИТАЕТСЯ: ЛАТЧ ПОЧТИ НЕ ВКЛЮЧАЛСЯ"), (len(eps), c5["rules"]["C5"])
        assert c5["rules"]["C5"]["outcome"] == ("не меняет различимо" if ok else "не читается: латч почти не включался")
    # line 2 (PPCL only) not run: C5 is not read; И4 is
    res = grid([x for x in new if not (x.get("case_id") in e300 and x["d"] == 0)], saved, mode, k, names, split)
    r, ms = res["rules"], res["measurements"]
    assert res["checks"]["complete"]["2"]["run"] is False and ms["И4"]["pairs"] == 520, ms
    assert r["C5"]["outcome"] is None and r["C5"]["lines_not_run"] == ["2"] and res["claims"]["ЛАТЧ МЕНЯЕТ PPC"] == (
        "НЕ ЧИТАЛОСЬ"), r
    # the new Base's e is not part 2's on 6 of the 300 π cases at d 0: 98 % match, the 6 cross-pod pairs drop; 7 fail
    for n_off, ok in ((6, True), (7, False)):
        off = set(e300[:n_off])
        res = grid(*world("PPP", "VS1.0", pct, lambda x: x | {"e": 31} if x.get("case_id") in off
                          and x["d"] == 0 else x), names, split)
        assert res["checks"]["e_match"][f"{P}/d0"] == {"equal": 300 - n_off, "total": 300, "mode": "P", "pass": ok}
        if ok:
            assert res["rules"]["C5"]["invalid_pairs"] == 6 and res["report"]["ppcl_none"][f"{P}/d0"]["pairs"] == 294
        else:
            assert "rules" not in res
            exits(lambda: finish(res), 3)
    # mode П holds on the probe's card (spec §7.1, mode.json's gpu): a record of its lines on another card stops the
    # reading; in mode С it does not
    for modes, name, ok in (("PPP", "pi05_lmax", False), ("SPP", "pi05_lmax", True), ("PPP", "pi05_bench", False)):
        new, saved, mode, k = world(modes, "VS1.0", pct)
        x0 = next(x for x in new if PROBE.get(("m" if "case_id" in x else "b", x["brain"])) == name)
        res = grid([x | {"conf": x["conf"] | {"gpu": "NVIDIA GeForce RTX 3090"}} if x is x0 else x for x in new],
                   saved, mode, k, names, split)
        assert res["checks"]["on_probe_gpu"][name]["pass"] == ok == ("rules" in res), res["checks"]["on_probe_gpu"]
    # mode П: the new Base's pre-event calls in part 2's batches (spec §7), >= 98 % of the cases per cell
    for n, ok in ((6, True), (7, False)):
        off = set(e300[:n])
        new, saved, mode, k = world("PPP", "VS1.0", pct, lambda x: x | {"batch_rows": [[0, 5, 1]]} if (
            x["kind"] == "base" and x["d"] == 0 and x.get("case_id") in off) else x)
        res = grid(new, saved, mode, k, names, split)
        v = res["checks"]["batch_rows_pre_event"][f"{P}/d0"]
        assert v["pass"] == ok == ("rules" in res) and v["cases"] == 300 and v["cases_equal"] == 300 - n, (n, v)
        if not ok:
            exits(lambda: finish(res), 3)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--probe-plan", action="store_true", help="the probes' runner lines (phase cprobe)")
    g.add_argument("--probe", nargs="+", metavar="JSONL", help="the probes' records -> mode.json")
    g.add_argument("--select", nargs="+", metavar="BRAIN JSONL", help="pi05|oftplus, its M0 records -> k_select.json")
    g.add_argument("--plan", action="store_true", help="--line's runner lines")
    g.add_argument("--latched", nargs="+", metavar="JSONL", help="PPCL records: where the latch held a chunk")
    g.add_argument("--forecast", metavar="TEMPO", help="the tempo file: hours and $ per line")
    g.add_argument("--smoke-check", nargs="+", metavar="JSONL", help="the smoke's LIBERO-MAX records: exit 0, or 3")
    g.add_argument("--selftest", action="store_true")
    ap.add_argument("grid", nargs="*", metavar="[grid] JSONL", help="R/grid_all_C.jsonl: checks, rules, report")
    ap.add_argument("--p2", default=str(P2), help="grid: part 2's records (read only)")
    ap.add_argument("--line", choices=LINES)
    ap.add_argument("--mode", default=f"{OUT}/mode.json", help="mode.json (--plan, --forecast)")
    ap.add_argument("--k", default=f"{OUT}/k_select.json", help="k_select.json (--plan, --forecast)")
    ap.add_argument("--out", help="--probe: default --mode's path; --select: default --k's; grid: the JSON too")
    ap.add_argument("--records", nargs="+", default=[], metavar="JSONL", help="--line c5: lines 2, 3, 7's records")
    ap.add_argument("--done", nargs="+", default=[], metavar="JSONL", help="--forecast: the grid's records so far")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f) if "kind" in x]  # noqa: E731
    jload = lambda f: json.loads(Path(f).read_text()) if Path(f).exists() else None  # noqa: E731
    a.grid = a.grid[a.grid[:1] == ["grid"]:]  # "grid" before the files is optional
    if bool(a.grid) + sum(map(bool, (a.probe_plan, a.probe, a.select, a.plan, a.latched, a.forecast, a.smoke_check,
                                     a.selftest))) != 1:
        ap.error("one mode: an option of the group, or [grid] JSONL...")
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
        if lns := plan(a.line, jload(a.mode), jload(a.k), split, load(a.records)):
            print(tsv(lns))
    elif a.latched:
        print(e2.dump(latched(load(a.latched))))
    elif a.forecast:
        print(e2.dump(forecast(read_tempo(a.forecast), jload(a.mode), jload(a.k), {cs.key(r) for r in load(a.done)},
                               split)))
    elif a.grid:
        saved = {n: load([Path(a.p2) / f for f in fs]) for n, fs in SAVED.items()}
        finish(grid(load(a.grid), saved, jload(a.mode), jload(a.k), cs.libero_names(), split), a.out)
    elif a.smoke_check:
        ok, n = smoke_check(load(a.smoke_check))
        print(e2.dump({"prefix_ok_of_cases": n, "pass": ok}))
        print(f"C1-E4 controls smoke: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
