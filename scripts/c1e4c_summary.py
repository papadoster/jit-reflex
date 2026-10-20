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
        R/C3_*.jsonl R/C7_*.jsonl (those run), PPC on the runner groups where PPCL latched
    python scripts/c1e4c_summary.py --latched R/C2_*.jsonl ...
    python scripts/c1e4c_summary.py --forecast R/tempo.txt [--done R/C[0-9]*.jsonl] [--mode ..] [--k ..]
        hours and $ per line; without mode or k the most expensive case (mode С, kπ != 1.0, kO != kπ)
    python scripts/c1e4c_summary.py --smoke-check R/Csmoke_pi05*.jsonl   exit 0, or 3
    python scripts/c1e4c_summary.py [grid] R/grid_all_C.jsonl [--mode R/mode.json] [--k R/k_select.json]
            [--p2 results/c1-e4/part2] [--out R/summary.json]
        on the Mac: the checks before reading (§7.3, §8), then the rules C2, C3, C5, C6, the measurements И1-И4 (an
        estimate and its 95% CI, no threshold), the claims and the report (§8, §11); a failed check prints only the
        checks, exit 3. grid_all_C.jsonl (the pod's cat of C[1-8]*.jsonl) = lines 1-8
        and the c5 pass; records off the plan (a probe's) are counted, never read; one key twice with other content
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
OFT_SAVED = ["pod_B/M4oft_oft_eval300_plus_d0.jsonl", "pod_B/M4oft_oft_eval300_pro_d0.jsonl"]  # line 8's report only
CONF_SKIP = ("arms", "gpu", "kbar")  # spec §7.3: every other conf field equal in a pair; kbar: VS lines run without it,
# a Gauto or Glead record's is checked against KBAR
CLAIMS = {"ОБЩИЙ k НЕ ПЕРЕНОСИТСЯ": "C6",  # spec §8: each alone, no multiplicity correction
          "ЗАДЕРЖКА РАЗВОДИТ G-AUTO И РЕГУЛЯТОР": "C2",
          "З1 НА ШУМНЫХ ГЛАЗАХ": "C3",  # its outcome, one of three
          "ЛАТЧ МЕНЯЕТ PPC": "C5"}
MEAS = ("И1", "И2", "И3", "И4")  # spec §8: was C1a, C1b, C7, C4; an estimate and its 95% CI, no threshold
EST = ("diff_pp", "lo2.5", "lo5_one_sided", "hi97.5", "pairs", "clusters")  # e2.diff_pp's: a measurement's estimate
DEP = {"И1": "2", "И2": "5", "C2": "23", "C3": "6", "И4": "7", "C5": "237", "C6": "5", "И3": "5"}  # rule -> lines


def show(a):
    """spec "Сокращения": VS(1.0) is G, one name wherever a label is displayed ("VS1.0@cv" is "G@cv"); the records,
    k_select.json and mode.json keep VS1.0."""
    return a and a.replace("VS1.0", "G")


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
    """spec §6, §7.2, §12: the runner lines of one line (LINES), phase c<line>. kπ, kO: k_select's; a mode С probe
    adds G-auto to its lines; in mode П a VS1.0 on M1 d = 0 is part 2's G. c5 (mode С only): PPC where PPCL latched
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
    Setup, probe and smoke are not in it (EXCLUDES)."""
    mode, k = assume(mode, k)
    rows = {}
    for L in LINES[1:9]:
        n, nr, h, hl = 0, 0, 0.0, 0.0
        for ln in plan(L, mode, k, split):
            # every planned record weighs 1: the tempo is seconds per record as the pod measures it (gtempo / full_rate,
            # scripts/gpu_c1e4m.sh), PPCL's extra calls and cut Bases included. The seeded part 2 tempos (plain full
            # episodes: lmax_oftplus 12.497, lmax_oft 12.993) lean high until a line of that brain measures its own (4,
            # 8): conservative
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
    """spec §7.3, §8, §11 on the pod's records (grid_all_C.jsonl: lines 1-8 and the c5 pass) and part 2's (saved: {SAVED
    name or "oft_lmax": records}). The checks before reading; when they pass, the rules C2, C3, C5, C6, the
    measurements И1-И4, the claims and the report. Records off the plan (a probe's) are counted, not read. Exit 2 when
    one key comes twice with other content (grid_all_C.jsonl and its C*.jsonl both). No kO (line 4 not run, spec
    §12): no line 5, И2, И3, C6 not read."""
    kp = need(k, P, "k")
    ko = need(k, OP, "k") if k.get(OP) else None
    md = {n: need(mode, n, "mode") for n in SAVED}
    mo = {P: md["pi05_lmax"], OP: md["oftplus_lmax"], O: "S"}  # OFT: no probe, line 8 is self-contained
    sel = {P: {kp}, O: {kp}, OP: {kp, ko} - {None}}  # spec §5: VS-общий everywhere, VS-подогнанный on OFT+
    by = {}
    for r in new:
        if by.setdefault(lk(r), r) != r:
            print(f"!!! {lk(r)} recorded twice with other content: pass grid_all_C.jsonl or its C*.jsonl, not both",
                  file=sys.stderr)
            raise SystemExit(2)
    ok = {x: r for x, r in by.items() if not r.get("skipped_reason")}  # a skipped episode counts as missing
    lines = {L: plan(L, mode, k, split) if L != "5" or ko else [] for L in LINES[1:-1]}
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
    sm = [r for n, rs in saved.items() if n != "pi05_bench" for r in rs if r["case_id"] in e300]
    sbase = {(r["brain"], r["case_id"], r["d"]): r for r in sm if r["kind"] == "base"}
    lbase = {(r["brain"], r["case_id"], r["d"]): r for r in of("2", "3", "5", "8") if r["kind"] == "base"}
    em = defaultdict(lambda: [0, 0])
    for x, r in lbase.items():  # spec §7.3: the new Base's e against part 2's, >= 98 % per brain and cell in mode П
        em[f"{x[0]}/d{x[2]}"][0] += x in sbase and sbase[x]["e"] == r["e"]
        em[f"{x[0]}/d{x[2]}"][1] += 1
    e_match = {c: {"equal": a, "total": n, "mode": mo[c.split("/")[0]],
                   "pass": mo[c.split("/")[0]] == "S" or a >= 0.98 * n - TOL} for c, (a, n) in sorted(em.items())}
    # spec §7: in mode П the new Base's pre-event calls ran in part 2's batches (π's fp32 chunk depends on its batch):
    # per case (e equal) its batch rows before e equal part 2's, >= 98 % per brain and cell; report only for a brain
    # other than π whose calls are all one row (rows_max 1). OFT+'s last group of a source in the first 300 was full
    # in part 2 (it ran the whole evaluation set): not compared
    last = {c for ids in cs.sources(run.set_ids(split, "eval300")).values() for c in ids[len(ids) // NE * NE:]}
    br = defaultdict(lambda: {"calls_equal": 0, "calls": 0, "cases_equal": 0, "cases": 0, "rows_max": 0})
    for x, r in lbase.items():
        s = sbase.get(x)
        if s is None or s["e"] != r["e"] or x[0] == OP and x[1] in last:
            continue
        lim = float("inf") if r["e"] is None else r["e"]
        a, b = ({t: row for t, *row in q.get("batch_rows") or [] if t < lim} for q in (r, s))
        v = br[f"{x[0]}/d{x[2]}"]
        v["calls_equal"] += sum(a.get(t) == row for t, row in b.items())
        v["calls"] += len(b)
        v["cases_equal"] += a == b
        v["cases"] += 1
        v["rows_max"] = max([v["rows_max"]] + [row[0] for q in (a, b) for row in q.values()])
    brows = {c: v | {"mode": mo[c.split("/")[0]], "pass": mo[c.split("/")[0]] == "S" or (
        c.split("/")[0] != P and v["rows_max"] <= 1) or v["cases_equal"] >= 0.98 * v["cases"] - TOL}
        for c, v in sorted(br.items())}
    c5_off = sum(lbase.get((r["brain"], r["case_id"], r["d"]), {}).get("e", "-") != r["e"] for r in of("c5")
                 if r.get("kind") == "base")  # its pairs drop out (mp)
    # part 2's records read: mode П the comparators of the rules and measurements, mode С "none" and G only (report,
    # "через под"); in mode П a selected VS1.0 on M1 d 0 is part 2's G (spec §2, §5)
    keep = lambda r, m: r["kind"] != "base" and r["arm"] in (  # noqa: E731
        ("none", "G") if m == "S" else ("none", "G", "Gauto", "PPC", "Gauto@cv"))
    km = [r for r in sm if keep(r, mo[r["brain"]])]
    km += [r | {"arm": "VS1.0", "method": "VS1.0"} for r in km if r["arm"] == "G" and r["d"] == 0
           and mo[r["brain"]] == "P" and "VS1.0" in sel[r["brain"]]]
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

    newm = [r for r in of(*LINES[1:-1]) if r.get("case_id") in e300]  # M0's calibration cases: only for k
    mi = cs.mindex([r for r in km + newm if r["kind"] == "dynamic"])
    bi = cs.bindex(kb + [r for r in of(*LINES[1:-1]) if "case_id" not in r])
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
        R[name] = {"test": show(test), **(e2.rnd(d) or {}), "invalid_pairs": bad} | {
            f: None if d is None else g(d) for f, g in fs.items()}
        inv[name] = bad, len(prs) + bad if total is None else total

    vals = lambda pb: (list(pb[0].values()), pb[1])  # noqa: E731
    put("И1", f"pi05, M1, d = 0, the first 300: Gauto - {kp}", *vals(M(P, "Gauto", kp)))
    if kp == "VS1.0":
        R["И1"]["note"] = "k_π = 1.0: the regulator is G (mode П: part 2's), И1 = Q4c on the first 300 (spec §2, §8)"
    put("И2", f"oftplus, M1, d = 0, the first 300: Gauto - {kp}", *vals(M(OP, "Gauto", kp)))
    ps = {d: M(P, "Gauto", kp, d) for d in (0, 10, 20)}
    g = lambda d, c: float(ps[d][0][c][1]) - float(ps[d][0][c][2])  # noqa: E731
    put("C2", f"pi05, the first 300, d = 0, 10, 20: per case mean over d = 10, 20 of (Gauto - {kp}) minus d = 0's; "
              "|Δ| >= 5 and the 95% CI without 0",
        [(clu[c], (g(10, c) + g(20, c)) / 2 - g(0, c), 0.0) for c in sorted(set.intersection(*map(set, (
            p for p, _ in ps.values()))))], sum(b for _, b in ps.values()),
        sum(len(p) + b for p, b in ps.values()), outcome=lambda d: apart(
            d, 5, "расходятся: G-auto − VS растёт с задержкой", "расходятся: G-auto − VS падает с задержкой",
            "не расходятся"))
    put("C3", f"bench A20 @cv, inits 0-9: (Gauto@cv - {kp}@cv) >= +5 and lo2.5 > 0, or <= -5 and hi97.5 < 0",
        B(("A20",), "Gauto@cv", f"{kp}@cv"), 0, outcome=lambda d: apart(
            d, 5, "З1 лучше на шуме", "фиксированный κ лучше на шуме", "не различимы"))
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
    c6 = f"oftplus, M1, d = 0, the first 300: |{ko} - {kp}| >= 5 and the 95% CI without 0"
    if ko == kp:
        R["C6"], inv["C6"] = {"test": show(c6), "invalid_pairs": 0, "outcome": "не расходятся, тот же k"}, (0, 0)
    else:
        put("C6", c6, *vals(M(OP, ko, kp)), outcome=lambda d: apart(
            d, 5, f"расходятся: {show(ko)} лучше", f"расходятся: {show(kp)} лучше", "не расходятся"))
    put("И3", f"oftplus, M1, d = 0, the first 300: Gauto - {ko}", *vals(M(OP, "Gauto", ko)))
    reads = {"И1": {"Gauto", kp}, "И2": {"Gauto", kp}, "C2": {"Gauto", kp}, "C3": {"Gauto@cv", f"{kp}@cv"},
             "И4": {"Gauto", "PPCL"}, "C5": {"PPCL"}, "C6": {ko, kp}, "И3": {"Gauto", ko}}  # its arms
    for n, Ls in DEP.items():  # a line not run at all (spec §12: waits for the owner) with an arm the rule or
        # measurement reads: not read (a measurement: no estimate). Without one it reads part 2's records only (mode П,
        # kπ = 1.0: И1 without line 2). No kO: line 4 not run, so no line 5
        miss = ["4", "5"] if ko is None and "5" in Ls else [
            L for L in Ls if not comp[L]["run"] and reads[n] & {a for ln in lines[L] for a in ln["arms"]}]
        if miss and not (n == "C6" and ko == kp):
            R[n] |= dict.fromkeys(EST if n in MEAS else ("outcome",)) | {"lines_not_run": miss}
    checks["invalid_pairs"] = {n: {"pairs": t, "invalid": b, "pass": b <= 0.02 * t + TOL} for n, (b, t) in inv.items()}
    if not all(v["pass"] for v in checks["invalid_pairs"].values()):
        checks["pass"] = False
        return {"checks": checks}
    meas = {n: R.pop(n) for n in MEAS}
    claims = {}
    for c, n in CLAIMS.items():
        o = R[n]["outcome"]
        claims[c] = ("НЕ ЧИТАЛОСЬ" if o is None else o if n == "C3" else o.upper() if o.startswith("не читается") else
                     "ДА" if o.startswith(("расходятся", "меняет")) else "НЕТ")
    return {"checks": checks, "rules": R, "measurements": meas, "claims": claims,
            "report": grid_report(mi, bi, M, B, one, e300, kp, ko, mo, md, used, k, mode, checks)}


def grid_report(mi, bi, M, B, one, e300, kp, ko, mo, md, used, k, mode, checks):
    """spec §8 "только отчёт" and §11; "через под": a comparison with part 2's records in mode С."""
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731
    dm = lambda b, x, y, d=0, cases=e300: e2.rnd(e2.diff_pp(list(M(b, x, y, d, cases)[0].values())))  # noqa: E731
    db = lambda cells, x, y: e2.rnd(e2.diff_pp(B(cells, x, y)))  # noqa: E731
    via = lambda m: {"через под": m == "S"}  # noqa: E731
    # one name per method (show); on M1 d 0 in mode С part 2's G and the pod's VS1.0 are both G: G(часть 2), G(под C).
    # In mode П that VS1.0 is part 2's G itself (spec §2)
    lab = lambda b, a, d=0: {"G": "G(часть 2)", "VS1.0": "G(под C)"}[a] if (  # noqa: E731
        d == 0 and a in ("G", "VS1.0") and mo[b] == "S" and (b, "dynamic", "VS1.0", 0) in mi) else show(a)
    rep = defaultdict(dict)
    m1 = [(P, "Gauto", kp), (OP, "Gauto", kp), (OP, ko, kp), (OP, "Gauto", ko)]  # every M1 difference, sliced below
    for b, d in ((P, 0), (P, 10), (P, 20), (OP, 0), (O, 0)):
        for a in sorted({a for b2, _, a, d2 in mi if (b2, d2) == (b, d) and (cs.method(a) in VS or a == "PPCL")}):
            rep["vs_none"][f"{b}/d{d}/{lab(b, a, d)}-none"] = (dm(b, a, "none", d) or {}) | via(mo[b])
            m1 += [(b, a, "none")] * (d == 0)
    for c, a in (("A", "PPCL"), ("A", kp), ("A20", "PPCL"), ("A20", kp), ("A20", f"{kp}@cv")):
        rep["vs_none"][f"bench/{c}/{show(a)}-none"] = (db((c,), a, "none") or {}) | via(md["pi05_bench"])
    for d in (0, 10, 20):
        rep["ppcl_ppc"][f"{P}/d{d}"] = dm(P, "PPCL", "PPC", d)
    for c in ("A", "A20"):
        rep["ppcl_ppc"][f"bench/{c}"] = db((c,), "PPCL", "PPC")
    for b, a in ((P, kp), (OP, kp), (OP, ko), (O, kp)):  # fixed κ next to G-auto on M1 d 0; mode П G - G ≡ 0: out
        if a and lab(b, "G") != lab(b, a):
            rep["g_vs"][f"{b}/{lab(b, 'G')}-{lab(b, a)}"] = (dm(b, "G", a) or {}) | via(mo[b])
    rep["oft_gauto_vs"] = dm(O, "Gauto", kp)  # line 8, the tail: one pod
    m1 += [(P, "PPCL", "PPC"), (P, "G", kp), (OP, "G", kp), (OP, "G", ko), (O, "G", kp), (O, "Gauto", kp)]
    rep["cv_cost"] = {f"A20/{show(kp)}@cv-{show(kp)}": db(("A20",), f"{kp}@cv", kp),  # part 2: Gauto@cv - Gauto -10.8
                      "A20/Gauto@cv-Gauto": db(("A20",), "Gauto@cv", "Gauto")}
    dist = {c: one[c]["distance_m"] for c in e300 if c in one and one[c]["distance_m"] is not None}
    near, far = {c for c, v in dist.items() if v < 0.09}, {c for c, v in dist.items() if v >= 0.09}
    for b, x, y in dict.fromkeys(m1):
        if None not in (x, y) and lab(b, x) != lab(b, y):
            rep["slices_M1"][f"{b}/{lab(b, x)}-{lab(b, y)}"] = {"6cm": dm(b, x, y, 0, near),
                                                                "12cm": dm(b, x, y, 0, far)}
    for key, g in sorted(list(mi.items()) + list(bi.items()), key=lambda kv: str(kv[0])):  # per arm and cell
        i = 2 if key[1] == "dynamic" else 3
        g, a = list(g.values()), key[i]
        name = "/".join(map(str, key[:i] + (lab(key[0], a, key[3]) if i == 2 else show(a),) + key[i + 1:]))
        post = [sum(t >= r["e"] for t, _ in r["plans_in"]) for r in g if r.get("e") is not None]
        rep["calls"][name] = {"n": len(g), "sched": mean([r["calls_sched"] for r in g]),
                              "trig": mean([r["calls_trig"] for r in g]), "plans_after_event": mean(post)}
        if a == "PPCL":
            lt = [r["latch"] for r in g]
            rep["latch"][name] = {"n": len(lt), "latched_share": mean([x["chunks"] > 0 for x in lt])} | {
                f"{f}_per_episode": mean([x[f] for x in lt]) for f in ("chunks", "trig", "resets")}
        if cs.method(a) in VS + ("Gauto",):
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
                  ("6", mk("VS0.3")), ("7", mk(sb="P")), ("8", mk())):  # without its mode or k
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
    # 2: PPCL, VS0.3, Gauto + Base on 300; 3: d 10, 20; 5: VS0.3, VS0.1, Gauto + Base; 6: 2 @cv arms on 260; 7: 3 arms
    # x 2 cells x 260; 8: VS0.3, Gauto + Base
    assert [f["lines"][L]["records"] for L in LINES[1:9]] == [240, 1200, 2400, 240, 1200, 520, 1560, 900]
    assert [f["lines"][L]["episodes"] for L in LINES[1:9]] == [180, 900, 1800, 180, 900, 520, 1560, 600]
    assert f["core"] == {"episodes": 6040, "records": 7360, "hours": 73.6, "usd": round(73.6 * RATE, 2),
                         "left_hours": 73.6} and f["core_tail"]["hours"] == 82.6, f["core"]
    f = forecast(t36, *mk("VS1.0", "VS1.0", "P", "P", "P"), split=split)  # every coincidence
    assert [f["lines"][L]["episodes"] for L in LINES[1:9]] == [180, 300, 1200, 180, 0, 260, 1040, 600]
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

    def rec(x, pct, new, kk):  # x: lk; new: the pod's (gpu, a cut Base); kk: the VS label with the least grasp miss
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
              "path_m": 1.0, "grasp_miss_m": 0.01 + 0.01 * (cs.method(a) in VS and a != kk[b])}
        r["conf"] |= gpu if new else {}
        return r | ({"latch": {"trig": 0, "chunks": 0, "resets": 0}} if cs.method(a) == "PPCL" else {})

    def world(modes, kk, pct=None, edit=lambda r: r):
        """mode.json, k_select.json (select on the generated M0 lines: kk wins), the pod's records of every planned
        line (edit: on each) and part 2's of SAVED and M4 OFT (300 + 5 OFT+ cases: only eval300 is read)."""
        pct, kk = pct or {}, {P: kk[0], OP: kk[1], O: kk[0]}
        mode = {n: {"mode": m, "gpu": [gpu["gpu"]]} for n, m in zip(SAVED, modes)}
        mk = lambda x: edit(rec(x, pct, True, kk))  # noqa: E731
        m0 = {b: [mk(x) for x in lkeys(plan(L, split=split))] for b, L in ((P, "1"), (OP, "4"))}
        k = {b: select(rs, b, split) for b, rs in m0.items()}
        new = m0[P] + m0[OP] + [mk(x) for L in "235678" for x in lkeys(plan(L, mode, k, split))]
        new += [mk(x) for x in lkeys(plan("c5", mode, k, split, new))]
        old = lambda b, ids, arms, d=0: [rec(("m", b, c, kd, a, d, False, tuple(arms)), pct, False, kk)  # noqa: E731
                                         for c in ids for kd, a in [("dynamic", a) for a in arms] + [("base", "none")]]
        bold = lambda cells, arms: [rec(("b", P, "move", c, a, *e), pct, False, kk)  # noqa: E731
                                    for c in cells for a in arms for e in EPS]
        saved = {"pi05_lmax": old(P, e300, cs.ARMS + cs.M2) + old(P, e300, cs.D_ARMS, 10) + old(P, e300, cs.D_ARMS, 20),
                 "oftplus_lmax": old(OP, split["eval"][:305], cs.ARMS),
                 "pi05_bench": bold(("A", "A20"), cs.B_ARMS) + bold(("A20",), cs.NOISY),
                 "oft_lmax": old(O, e300, cs.ARMS)}
        return new, saved, mode, k

    # the preregistered thresholds (spec §8), directly: equality passes, "> 0" strict
    x = lambda d=0.0, lo2=-9.0, lo5=-9.0, hi=9.0: {"diff_pp": d, "lo2.5": lo2, "lo5_one_sided": lo5,  # noqa: E731
                                                     "hi97.5": hi}
    assert [apart(x(*v), 5, "up", "down", "same") for v in (
        (5.0, 1e-9), (5.0, 0.0), (4.99, 1.0), (-5.0, -9, -9, -1e-9), (-5.0, -9, -9, 0.0), (-4.99, -9, -9, -1.0))] == [
        "up", "same", "same", "down", "same", "same"]

    # mode С, kπ = kO = VS0.3: И1 at +3; C2 at +10 diverges (d 10: +11, d 20: +15); C3 at +5 exactly (184 - 171 of
    # 260); C6 the same k; C5: PPCL latched on 15 cases at d 0 (in 6 runner groups, 36 cases) and 15 bench episodes
    # (A20, 13 tasks), PPC of the c5 pass on those groups and tasks (both cells), PPCL itself elsewhere: PPC 40 %
    # against PPCL's 50 % there, so + 1 on the 30 latched pairs (40-44 of 100) and, in the pool, on the pass's 24
    # cases and 2 x 24 bench episodes at 40-49 of 100
    lat = lambda r: r["method"] == "PPCL" and (  # noqa: E731
        pos[r["case_id"]] % 100 in range(40, 45) and r["d"] == 0 if "case_id" in r
        else bpos[r["suite"], r["task"], r["init"]] % 100 in range(40, 45) and r["cell"] == "A20")
    pct = {(P, "Gauto", 0): 53, (P, "Gauto", 10): 61, (P, "Gauto", 20): 65, ("A20", "Gauto@cv"): 62,
           ("A20", "VS0.3@cv"): 57, "PPC": 40}
    new, saved, mode, k = world("SSS", ("VS0.3", "VS0.3"), pct,
                                lambda r: r | {"latch": {"trig": 1, "chunks": 2, "resets": 0}} if lat(r) else r)
    assert k[P]["k"] == k[OP]["k"] == "VS0.3" and k[P]["scores"]["VS1.0"] == 0.02, k
    probe1 = rec(("m", P, e300[0], "dynamic", "T0", 0, False, cs.ARMS + cs.M2), pct, True, {P: "VS0.3"})
    res = grid(new + [probe1] + new[:5], saved, mode, k, names, split)  # new[:5]: an identical copy is fine
    ch = res["checks"]
    assert ch["pass"] and ch["off_plan"] == {f"{P}/T0": 1}, ch
    assert ch["complete"]["c5"] == {"expected": 36 * 2 + 13 * 10 * 2, "missing": 0, "run": True, "pass": True}, ch
    assert ch["complete"]["5"]["expected"] == 300 * 3 and ch["e_match"][f"{P}/d0"]["mode"] == "S"
    r, ms = res["rules"], res["measurements"]
    assert list(r) == ["C2", "C3", "C5", "C6"] and sorted(ms) == list(MEAS), (list(r), list(ms))
    # a measurement: the estimate and its 95% CI only, no verdict and no threshold (spec §8)
    assert all(not {"pass", "superior", "outcome"} & set(v) and not re.search("[<>≤≥]", v["test"]) and {
        "diff_pp", "lo2.5", "hi97.5", "pairs", "clusters", "invalid_pairs"} <= set(v) for v in ms.values()), ms
    assert ms["И1"]["diff_pp"] == 3 and ms["И1"]["pairs"] == 300 and ms["И4"]["pairs"] == 520, ms
    assert ms["И4"]["diff_pp"] == 0 and ms["И3"] | {"test": 0} == ms["И2"] | {"test": 0}, ms
    assert r["C2"]["diff_pp"] == 10 and r["C2"]["pairs"] == 300 and r["C2"]["outcome"].startswith("расходятся: G-auto")
    assert r["C3"]["outcome"] == "З1 лучше на шуме" and r["C3"]["pairs"] == 260 and r["C3"]["diff_pp"] == 5, r["C3"]
    # C5 on the 30 latched pairs (spec §8); the pool of 1420 for reference
    assert r["C5"]["pairs"] == r["C5"]["latched_pairs"] == 30 and r["C5"]["diff_pp"] == 100, r["C5"]
    assert r["C5"]["clusters"] == r["C5"]["latched_clusters"] > 13 and r["C5"]["outcome"] == "меняет: PPCL лучше"
    assert r["C5"]["pool"]["pairs"] == 1420 and r["C5"]["pool"]["diff_pp"] == round(100 * 72 / 1420, 3), r["C5"]
    assert r["C5"]["pool"]["clusters"] == ms["И1"]["clusters"] + 26, r["C5"]
    assert r["C6"]["outcome"] == "не расходятся, тот же k", r
    cl = res["claims"]
    assert list(cl) == ["ОБЩИЙ k НЕ ПЕРЕНОСИТСЯ", "ЗАДЕРЖКА РАЗВОДИТ G-AUTO И РЕГУЛЯТОР", "З1 НА ШУМНЫХ ГЛАЗАХ",
                        "ЛАТЧ МЕНЯЕТ PPC"], cl
    assert cl["ЛАТЧ МЕНЯЕТ PPC"] == "ДА" and cl["ЗАДЕРЖКА РАЗВОДИТ G-AUTO И РЕГУЛЯТОР"] == "ДА", cl
    assert cl["ОБЩИЙ k НЕ ПЕРЕНОСИТСЯ"] == "НЕТ" and cl["З1 НА ШУМНЫХ ГЛАЗАХ"] == "З1 лучше на шуме", cl
    rep = res["report"]
    assert rep["latch"][f"{P}/dynamic/PPCL/0"]["latched_share"] == 0.05 and rep["latch"][f"{P}/move/A20/PPCL"][
        "latched_share"] == round(15 / 260, 4), rep["latch"]
    assert rep["vs_none"][f"{P}/d0/VS0.3-none"]["через под"] and rep["vs_none"][f"{P}/d0/VS0.3-none"]["pairs"] == 300
    assert rep["ppcl_ppc"][f"{P}/d0"]["diff_pp"] == 8 and rep["ppcl_ppc"][f"{P}/d10"]["diff_pp"] == 0, rep["ppcl_ppc"]
    assert set(rep["slices_M1"][f"{P}/Gauto-VS0.3"]) == {"6cm", "12cm"} and sum(
        rep["slices_M1"][f"{P}/Gauto-VS0.3"][s]["pairs"] for s in ("6cm", "12cm")) == 300
    assert rep["calls"][f"{P}/dynamic/PPCL/10"]["plans_after_event"] == 1 and rep["gpu"][P] == [gpu["gpu"]]
    assert rep["cv_cost"]["A20/Gauto@cv-Gauto"]["diff_pp"] == round(3400 / 260, 3)
    assert rep["calls"][f"{OP}/dynamic/G/0"]["n"] == 300  # part 2's OFT+ cases past the first 300 are not read
    assert rep["batch_rows_pre_event"][f"{P}/d0"]["cases_equal"] == 300
    ppcl = next(x for x in new if lat(x) and "case_id" in x)
    try:
        ppc_of(ppcl, None)  # a latched PPCL is no PPC
        raise AssertionError("a latched PPCL episode stood in for PPC")
    except AssertionError as ex:
        assert "without its PPC" in str(ex), ex
    assert ppc_of(ppcl | {"latch": {"trig": 0, "chunks": 0, "resets": 0}}, None)["arm"] == "PPCL"
    exits(lambda: grid(new + [new[7] | {"success": not new[7]["success"]}], saved, mode, k, names, split), 2)
    # the checks before reading, each alone: the c5 pass short of one episode; line 7 short of > 1 %; a k_select.json
    # not recomputed from line 1; a VS label not the selected one; a conf field (the bench's @cv dead bands too); a
    # Gauto's kbar not part 2's
    c5x = next(x for x in new if x["arm"] == "PPC" and "case_id" in x)
    l7 = [x for x in new if x.get("cell") and x["arm"] == "PPCL"][:16]  # 16 of the 1560 episodes of line 7
    v3 = next(x for x in new if x["arm"] == "VS0.3" and x.get("case_id") in e300)
    for bad, why in (([x for x in new if x is not c5x], "complete"), ([x for x in new if x not in l7], "complete"),
                     (new + [v3 | {"arm": "VS0.1", "method": "VS0.1",
                                   "conf": v3["conf"] | {"arms": ["PPCL", "VS0.1", "Gauto"]}}], "vs_label"),
                     ([x | {"conf": x["conf"] | {"t_ramp": 4}} if x.get("cell") == "A" else x for x in new], "conf"),
                     ([x | {"conf": x["conf"] | {"part2": {"cv_bands": [0.02, 0.007031], "tau_close": 13}}}
                       if x["arm"] == "VS0.3@cv" else x for x in new], "conf"),
                     ([x | {"conf": x["conf"] | {"kbar": 0.3}} if x["arm"] == "Gauto@cv" else x for x in new], "conf")):
        ch = grid(bad, saved, mode, k, names, split)
        c = ch["checks"][why]
        assert not ch["checks"]["pass"] and "rules" not in ch, why
        assert not c["pass"] if "pass" in c else not all(v["pass"] for v in c.values()), (why, c)
    with contextlib.redirect_stderr(io.StringIO()):
        ch = grid(new, saved, mode, k | {P: k[P] | {"n_valid": 59}}, names, split)["checks"]
    assert not ch["pass"] and not ch["k_select"][P]["pass"] and ch["k_select"][OP]["pass"], ch["k_select"]
    cb = next(x for x in new if x["kind"] == "base" and x["brain"] == P and x["case_id"] == cal[0])
    ch = grid([x | {"skipped_reason": "motion_blur"} if x is cb else x for x in new], saved, mode, k, names,
              split)["checks"]
    assert ch["pass"] and ch["skipped"] == 1 and ch["k_select"][P]["pass"], ch  # the pod's --select reads it too
    # the invalid-pair limit (spec §8), a measurement's too: 6 of И1's 300 pairs (2 %) are read, 7 stop the reading
    for n_bad, ok in ((6, True), (7, False)):
        off = set(e300[:n_bad])
        res = grid([x | {"prefix_ok": False} if x.get("case_id") in off and x["arm"] == "Gauto" and x["brain"] == P
                    and x["d"] == 0 else x for x in new], saved, mode, k, names, split)
        assert res["checks"]["invalid_pairs"]["И1"] == {"pairs": 300, "invalid": n_bad, "pass": ok} and (
            "rules" in res) == ok == res["checks"]["pass"], res["checks"]["invalid_pairs"]

    # mode С, kπ = VS0.3, kO = VS0.1: И1 at -6; no divergence; C3 the other way; C6 diverges at +5 exactly; the
    # latch never held: C5 not read
    pct = {(P, "Gauto", 0): 44, (P, "Gauto", 10): 44, (P, "Gauto", 20): 44, ("A20", "Gauto@cv"): 40,
           (OP, "VS0.1", 0): 55}
    res = grid(*world("SSS", ("VS0.3", "VS0.1"), pct), names, split)
    r, ms, cl = res["rules"], res["measurements"], res["claims"]
    assert ms["И1"]["diff_pp"] == -6 and r["C2"]["diff_pp"] == 0 and ms["И3"]["diff_pp"] == -5, (r, ms)
    assert r["C2"]["outcome"] == "не расходятся" and r["C3"]["outcome"] == "фиксированный κ лучше на шуме", r
    assert r["C6"]["diff_pp"] == 5 and r["C6"]["outcome"] == "расходятся: VS0.1 лучше", r["C6"]
    assert r["C5"]["outcome"] == "не читается: латч почти не включался" and "diff_pp" not in r["C5"], r["C5"]
    assert r["C5"]["latched_pairs"] == r["C5"]["latched_clusters"] == 0 and r["C5"]["pool"]["pairs"] == 1420
    assert r["C5"]["pool"]["diff_pp"] == 0 and cl["ЛАТЧ МЕНЯЕТ PPC"] == "НЕ ЧИТАЕТСЯ: ЛАТЧ ПОЧТИ НЕ ВКЛЮЧАЛСЯ", r
    assert cl["ОБЩИЙ k НЕ ПЕРЕНОСИТСЯ"] == "ДА", cl
    assert res["checks"]["complete"]["c5"]["expected"] == 0
    new, saved, mode, k = world("SSS", ("VS0.3", "VS0.1"), pct)
    res = grid([x for x in new if x.get("cell") != "A" or x["arm"] == "PPCL"], saved, mode, k, names, split)
    assert res["checks"]["complete"]["7"] == {"expected": 1560, "missing": 520, "run": True, "pass": False}
    res = grid([x for x in new if "cell" not in x or x["arm"].endswith("@cv")], saved, mode, k, names, split)
    r, ms = res["rules"], res["measurements"]  # line 7 not run at all: И4 and C5 (a part of its pool) are not read
    assert res["checks"]["complete"]["7"]["run"] is False and r["C5"]["outcome"] is None and ms["И4"]["pairs"] is None
    assert ms["И4"]["diff_pp"] is None and ms["И4"]["lines_not_run"] == ["7"] == r["C5"]["lines_not_run"], (r, ms)
    assert r["C3"]["outcome"] and ms["И1"]["diff_pp"] == -6 and res["claims"]["ЛАТЧ МЕНЯЕТ PPC"] == "НЕ ЧИТАЛОСЬ", r
    # k_select.json with π only (line 4 not run, spec §12): no line 5; И1, И4, C2, C3, C5 are read, И2, И3, C6 are not
    res = grid([x for x in new if x["brain"] != OP], saved, mode, {P: k[P]}, names, split)
    r, ms = res["rules"], res["measurements"]
    assert res["checks"]["pass"] and res["checks"]["complete"]["5"]["run"] is False and res["checks"]["k_select"][OP] \
        == {"file": None, "recomputed": None, "pass": True}, res["checks"]
    assert all(r[n]["outcome"] is not None for n in ("C2", "C3", "C5")) and ms["И1"]["diff_pp"] == -6, r
    assert ms["И4"]["diff_pp"] == 0 and "lines_not_run" not in ms["И4"], ms
    assert all(ms[n]["diff_pp"] is ms[n]["pairs"] is None and ms[n]["lines_not_run"] == ["4", "5"]
               for n in ("И2", "И3")) and r["C6"]["outcome"] is None and r["C6"]["lines_not_run"] == ["4", "5"], ms
    assert res["claims"]["ОБЩИЙ k НЕ ПЕРЕНОСИТСЯ"] == "НЕ ЧИТАЛОСЬ" and not res["report"]["g_vs"].get(f"{OP}/G-None")

    # mode П everywhere, kπ = kO = VS1.0: VS on M1 d 0 is part 2's G (И1 = Q4c on the first 300); И2 at -3; C3 at 0
    pct = {"G": 47, (O, "G", 0): 50, (OP, "Gauto", 0): 44}
    new, saved, mode, k = world("PPP", ("VS1.0", "VS1.0"), pct)
    assert not [x for x in new if x.get("case_id") in e300 and x["arm"] == "VS1.0" and x["d"] == 0
                and x["brain"] != O]
    res = grid(new, saved, mode, k, names, split)
    r, ms, rep = res["rules"], res["measurements"], res["report"]
    assert ms["И1"]["diff_pp"] == 3 and ms["И1"]["pairs"] == 300 and "Q4c" in ms["И1"]["note"], ms["И1"]
    assert ms["И2"]["diff_pp"] == -3 and r["C3"]["outcome"] == "не различимы", (r, ms)
    assert r["C5"]["pool"]["pairs"] == 1420 and r["C5"]["pool"]["diff_pp"] == 0 and r["C2"]["diff_pp"] == -3, r  # G 47
    assert not rep["vs_none"][f"{P}/d0/PPCL-none"]["через под"]
    # one name (spec "Сокращения"): VS1.0 is shown as G; mode П drops G - G; OFT (line 8, mode С) has part 2's G and
    # its own: two names
    shown = [v.get(f) for v in list(r.values()) + list(ms.values()) for f in ("test", "outcome", "note")] + [
        x for f in ("vs_none", "g_vs", "cv_cost", "slices_M1", "calls", "latch", "vs_metrics") for x in rep[f]]
    assert not [x for x in shown if x and "VS1.0" in x] and ms["И1"]["test"].endswith("Gauto - G"), shown
    assert {f"{P}/d0/G-none", f"{P}/d10/G-none", "bench/A20/G@cv-none"} <= set(rep["vs_none"]) and set(
        rep["cv_cost"]) == {"A20/G@cv-G", "A20/Gauto@cv-Gauto"} and f"{P}/dynamic/G/0" in rep["vs_metrics"], rep
    assert set(rep["g_vs"]) == {f"{O}/G(часть 2)-G(под C)"} and rep["g_vs"][f"{O}/G(часть 2)-G(под C)"]["через под"]
    assert {f"{O}/dynamic/G(часть 2)/0", f"{O}/dynamic/G(под C)/0", f"{P}/dynamic/G/0"} <= set(rep["calls"]), rep
    assert f"{P}/Gauto-G" in rep["slices_M1"] and not [x for x in rep["slices_M1"] if x.endswith("/G-G")], rep
    # C5 on the latched pairs (spec §8): 20 over 5 tasks are read; 19, or 20 over 4 tasks, are not
    e20 = {(*st, i) for st in orx.TASKS[:5] for i in range(4)}
    for eps, ok in ((e20, True), (e20 - {(*orx.TASKS[0], 0)}, False), ({(*st, i) for st in orx.TASKS[:4]
                                                                          for i in range(5)}, False)):
        c5 = grid([x | {"latch": {"trig": 1, "chunks": 1, "resets": 0}} if x["arm"] == "PPCL" and x.get("cell") == "A20"
                   and (x["suite"], x["task"], x["init"]) in eps else x for x in new], saved, mode, k, names, split)
        assert c5["rules"]["C5"]["latched_pairs"] == len(eps) and c5["claims"]["ЛАТЧ МЕНЯЕТ PPC"] == (
            "НЕТ" if ok else "НЕ ЧИТАЕТСЯ: ЛАТЧ ПОЧТИ НЕ ВКЛЮЧАЛСЯ"), (len(eps), c5["rules"]["C5"])
        assert c5["rules"]["C5"]["outcome"] == ("не меняет различимо" if ok else "не читается: латч почти не включался")
    # line 2 (PPCL only) not run: И1 and C2 read part 2's d 0 and are read; C5 (PPCL) is not
    res = grid([x for x in new if not (x.get("case_id") in e300 and x["brain"] == P and x["d"] == 0)], saved, mode, k,
               names, split)
    r, ms = res["rules"], res["measurements"]
    assert res["checks"]["complete"]["2"]["run"] is False and ms["И1"]["diff_pp"] == 3 and ms["И1"]["pairs"] == 300, ms
    assert r["C2"]["outcome"] and r["C5"]["outcome"] is None and r["C5"]["lines_not_run"] == ["2"], r
    # the new Base's e is not part 2's on 6 of the 300 π cases at d 0: 98 % match, the 6 cross-pod pairs drop; 7 fail
    for n_off, ok in ((6, True), (7, False)):
        off = set(e300[:n_off])
        res = grid(*world("PPP", ("VS1.0", "VS1.0"), pct, lambda x: x | {"e": 31} if x.get("case_id") in off
                          and x["d"] == 0 and x["brain"] == P else x), names, split)
        assert res["checks"]["e_match"][f"{P}/d0"] == {"equal": 300 - n_off, "total": 300, "mode": "P", "pass": ok}
        if ok:
            assert res["rules"]["C5"]["invalid_pairs"] == 6 and res["measurements"]["И1"]["invalid_pairs"] == 0
            assert res["report"]["vs_none"][f"{P}/d0/PPCL-none"]["pairs"] == 294
        else:
            assert "rules" not in res
            exits(lambda: finish(res), 3)
    # mode П holds on the probe's card (spec §7.1, mode.json's gpu): a record of its lines on another card stops the
    # reading; in mode С it does not
    for modes, name, ok in (("PPP", "pi05_lmax", False), ("SPP", "pi05_lmax", True), ("PPP", "pi05_bench", False)):
        new, saved, mode, k = world(modes, ("VS1.0", "VS1.0"), pct)
        x0 = next(x for x in new if PROBE.get(("m" if "case_id" in x else "b", x["brain"])) == name)
        res = grid([x | {"conf": x["conf"] | {"gpu": "NVIDIA GeForce RTX 3090"}} if x is x0 else x for x in new],
                   saved, mode, k, names, split)
        assert res["checks"]["on_probe_gpu"][name]["pass"] == ok == ("rules" in res), res["checks"]["on_probe_gpu"]
    # mode П: the new Base's pre-event calls in part 2's batches (spec §7), >= 98 % of the cases per brain and cell: π
    # always; OFT+ unless its calls are all one row; OFT+'s last group of a source (full in part 2) not compared
    for b, n, rows, one, ok in ((P, 6, 5, False, True), (P, 7, 5, False, False), (OP, 5, 5, False, True),
                                (OP, 7, 5, False, False), (OP, 300, 1, True, True)):
        off = set(e300[:n])
        new, saved, mode, k = world("PPP", ("VS1.0", "VS0.3"), pct, lambda x: x | {"batch_rows": [[0, rows, 1]]} if (
            x["kind"] == "base" and x["brain"] == b and x["d"] == 0 and x.get("case_id") in off) else x)
        if one:  # part 2's OFT+ calls one row each too, at another step
            saved["oftplus_lmax"] = [x | {"batch_rows": [[5, 1, 1]]} for x in saved["oftplus_lmax"]]
        res = grid(new, saved, mode, k, names, split)
        v = res["checks"]["batch_rows_pre_event"][f"{b}/d0"]
        assert v["pass"] == ok == ("rules" in res) and v["cases"] == (300 if b == P else 294), (b, n, v)
        assert v["cases_equal"] == (0 if one else v["cases"] - n), v
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
        saved = {n: load([Path(a.p2) / f for f in fs]) for n, fs in (SAVED | {"oft_lmax": OFT_SAVED}).items()}
        finish(grid(load(a.grid), saved, jload(a.mode), jload(a.k), cs.libero_names(), split), a.out)
    elif a.smoke_check:
        ok, n = smoke_check(load(a.smoke_check))
        print(e2.dump({"prefix_ok_of_cases": n, "pass": ok}))
        print(f"C1-E4 controls smoke: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
