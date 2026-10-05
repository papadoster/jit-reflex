"""C1-E4 part 2: the pod's plan (§6, §7, §13), its gates, the calibration line (§5), the rules Q4a-Q4d and Q5 with their
verdicts (§9) and the report (§11), applied mechanically to the JSON lines of scripts/c1e4m_run.py (LIBERO-MAX) and
scripts/c1e4_run.py (block B, our bench). Spec docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md. Pure numpy
(matplotlib for --fig); the main summary reads LIBERO's 40 task names from hf-libero (the 33 task clusters,
maxwrap.base_task), so it runs in the LeRobot env without LIBERO-MAX's overlay.
    python scripts/c1e4m_summary.py --plan [--cut "D M2"] [--kbar-line "C1-E4 part 2 calib: ..."]
        the pod's runner lines: phase, runner, brain, source, id, episodes, args (without the line: connect and calib)
    python scripts/c1e4m_summary.py --connect results/c1-e4/part2/pod/connect_*.jsonl   the connect gate: exit 0, or 3
    python scripts/c1e4m_summary.py --calib results/c1-e4/part2/pod/M0_*.jsonl ...Bcal_*.jsonl   the line, or exit 4
    python scripts/c1e4m_summary.py --smoke results/c1-e4/part2/pod/smoke_*.jsonl   the smoke's pair validity: 0, or 3
    python scripts/c1e4m_summary.py --forecast results/c1-e4/part2/pod/tempo.txt [--cut ..] [--rate 0.35]
    python scripts/c1e4m_summary.py results/c1-e4/part2/pod/grid_all.jsonl [--cut ..] [--fig fig.png]
    python scripts/c1e4m_summary.py --selftest
If a read gate fails only the gates are returned, unless the owner's deviation, journaled in the spec before reading, is
passed with --override-gate. Bootstrap and rule helpers are C1-E2's and C1-E3's, as part 1's scripts/c1e4_summary.py.
"""

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import c1e2_summary as e2  # noqa: E402  (also puts src/ on sys.path)
import c1e3_summary as s3  # noqa: E402
import c1e4m_run as run  # noqa: E402  (set_ids: the runner's --set)
import lead  # noqa: E402
import maxwrap  # noqa: E402
import objreflex as orx  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md"
SPLIT, PAIRS = REPO / "results/c1-e4/part2/split.json", REPO / "results/c1-e4/part2/bench_calib_pairs.json"
OFFLINE = REPO / "results/c1-e4/part2_offline.json"
TOL = e2.TOL
P, S, O, OP = "pi05", "smolvla", "oft", "oftplus"
CAL = (P, OP, O, S)  # spec §5: the calibration line's order; then the bench's pi05, smolvla
TABLE2 = {P: 79.7, O: 64.3, S: 26.1}  # spec §3, §9: their Table 2, overall Base (all 8 events), %; OFT+ is not in it
# arXiv 2609.36518v1 (LIBERO-MAX), Table 12 (CC BY 4.0), target relocation: Dynamic - Base success, pp, 1000 pairs each
TABLE12 = {"Cosmos-Policy": -47.5, "π0.5": -34.6, "OpenVLA-OFT": -42.1, "X-VLA": -52.3, "Xiaomi-Robotics-0": -44.0,
           "MolmoAct2": -32.0, "SmolVLA": -23.9, "GR00T N1.7": -52.5, "DM0.5": -39.9, "VLA-JEPA": -40.7,
           "Fast-WAM": -30.4, "HiMem-WAM": -43.7, "Light-WAM": -40.7, "DiT4DiT": -34.7}
ARMS, M2 = ("none", "G", "Gauto", "PPC"), ("G@glr", "PPC@glr", "T0")  # spec §6: M1 / M4 arms; M2 rides on M1's 300
B_ARMS, NOISY, D_ARMS = ("none", "Gauto", "Glead", "PPC"), ("Gauto@cv", "Glead@cv"), ("none", "Gauto", "PPC")
CUT_ORDER = ("D", "M2", "Bnoisy", "M4oft", "B59")  # spec §13 fuse: report blocks only, in this order
RATE, N_MIN, CONNECT_PP = 0.35, maxwrap.N_MIN, 10  # $/h RTX 4090 (spec §13); calibration N >= 45; connect +-10 pp
TEMPO = tuple(f"lmax_{b}" for b in CAL) + ("bench_pi05", "bench_smolvla")
NUM = r"-?\d+\.\d+ \[-?\d+\.\d+, -?\d+\.\d+\] N=\d+"
LINE_RE = re.compile(rf"C1-E4 part 2 calib: kbar_pi05={NUM}( · (bench: )?kbar_[a-z0-9]+={NUM}){{5}}")


def method(arm):
    return arm.partition("@")[0]


def sources(ids):
    """The ids by scene source, in order: LIBERO-MAX's PRO case ids start "pro-" (exactly the cases with a
    substrate_variant: checked on the Mac over all 8000), the runner's source_of."""
    return {s: [i for i in ids if i.startswith("pro-") == (s == "pro")] for s in ("plus", "pro")}


def bench_keys(brain, kind, cells, arms, eps):
    return {("b", brain, kind, c, a, *e) for c in cells for a in arms for e in eps}


def lines(cut=(), kbar=None, split=None):
    """spec §13: the pod's runner lines in run order, without the cut report blocks; without kbar (the journaled
    calibration line, parse_line) only the connect and calib phases. One line = one runner process with its own --out
    (id): a LIBERO-MAX block, brain, set, source and d, or a bench block and brain. Dicts: phase, runner, brain, source,
    id, block, arms, keys (the expected episodes, key()), args."""
    if bad := [c for c in cut if c not in CUT_ORDER]:
        raise SystemExit(f"cut {bad}: not report blocks (spec §13: connect, M0, B-cal, M1, M4 S and B oracle 0-4 are "
                         f"never cut; report blocks {' '.join(CUT_ORDER)})")
    if set(cut) != set(CUT_ORDER[:len(set(cut))]):
        raise SystemExit(f"cut {list(cut)}: report blocks are cut in the order {' '.join(CUT_ORDER)} only (spec §13)")
    split = split or maxwrap.load_split(SPLIT)
    out = []

    def m(phase, block, brain, sset, arms, d=0, tag=None):
        for src, ids in sources(run.set_ids(split, sset)).items():
            args = f"--brain {brain} --source {src} --set {sset} --arms={','.join(arms)} --d {d}"
            if {"Gauto", "Glead"} & set(map(method, arms)):
                args += f" --kbar {kbar['m', brain]}"
            out.append({"phase": phase, "runner": "c1e4m", "brain": brain, "source": src, "block": block, "arms": arms,
                        "id": f"{tag or block}_{brain}_{sset}_{src}_d{d}", "args": args,
                        "keys": {("m", brain, c, k, a, d, not arms) for c in ids
                                 for k, a in [("base", "none")] + [("dynamic", a) for a in arms]}})

    def b(phase, block, brain, kind, cells, arms, inits):
        args = f"--brain {brain} --kinds {kind} --cells {','.join(cells)} --methods {','.join(arms)} "
        if inits == "pairs":  # spec §5: the bench calibration, the first 60 (task, init) pairs of the frozen order
            eps = [(q["suite"], q["task"], q["init"]) for q in maxwrap.load_split(PAIRS)["calib"]]
            args += f"--pairs-file {PAIRS.relative_to(REPO)}"
        else:
            eps = [(s, t, i) for s, t in orx.TASKS for i in inits]
            args += f"--inits {inits[0]}-{inits[-1]}"
        if {"Gauto", "Glead"} & set(map(method, arms)):
            args += f" --kbar {kbar['b', brain]}"
        out.append({"phase": phase, "runner": "c1e4", "brain": brain, "source": "-", "block": block, "arms": arms,
                    "id": f"{block}_{brain}", "args": args, "keys": bench_keys(brain, kind, cells, arms, eps)})

    for br in (P, S, O, OP):
        m("connect", "connect", br, "connect", ())  # spec §6: connect = M3, Base only
    for br in CAL:
        m("calib", "M0", br, "calib", ("Gcal",))
    for br in (P, S):
        b("calib", "Bcal", br, "step", ("A",), ("Gcal",), "pairs")
    if kbar is None:
        return out
    # the grid by priority (spec §13): M1 pi0.5 -> M1 OFT+ -> M4 S -> B oracle -> M4 OFT -> B noisy -> (M2) -> D
    m2 = "M2" not in cut  # M2 shares M1's Base on the first 300 cases: one call (spec §6); its own file id
    m("grid", "M1pi", P, "eval300", ARMS + M2 * m2, tag="M1M2" if m2 else None)
    m("grid", "M1pi", P, "evalrest", ARMS)
    m("grid", "M1oftplus", OP, "eval", ARMS)
    m("grid", "M4S", S, "eval300", ARMS)
    for br in (P, S):
        b("grid", "Boracle", br, "move", ("A", "A20"), B_ARMS, range(5 if "B59" in cut else 10))
    if "M4oft" not in cut:
        m("grid", "M4oft", O, "eval300", ARMS)
    if "Bnoisy" not in cut:
        for br in (P, S):
            b("grid", "Bnoisy", br, "move", ("A20",), NOISY, range(10))
    if "D" not in cut:
        for d in (10, 20):
            m("grid", "D", P, "eval300", D_ARMS, d)
    have = set().union(*(ln["keys"] for ln in out))
    m1 = ((P, split["eval"]), (OP, split["eval"]), (S, split["eval"][:300]))
    never = ({("m", br, c, "dynamic", a, 0, False) for br, ids in m1 for c in ids for a in ARMS}
             | bench_keys(P, "move", ("A", "A20"), B_ARMS, [(s, t, i) for s, t in orx.TASKS for i in range(5)])
             | bench_keys(S, "move", ("A", "A20"), B_ARMS, [(s, t, i) for s, t in orx.TASKS for i in range(5)]))
    assert never <= have, "a never-cut block lost episodes (spec §13)"
    assert all(ln["brain"] not in (O, OP) or "--d 0" in ln["args"] for ln in out), "OFT runs at d = 0 only (spec §4)"
    return out


def fuse(line, k):
    """The fuse block of one planned episode (spec §13): M2's arms ride on M1's first 300 pi0.5 cases; B oracle's
    inits 5-9 are B59."""
    return "M2" if k[4] in M2 else "B59" if line["block"] == "Boracle" and k[-1] >= 5 else line["block"]


def key(r):
    """One episode: LIBERO-MAX (brain, case, kind, arm, d, connect = Base only) or bench (brain, kind, cell, arm, suite,
    task, init). A connect case may also be an evaluation case: connect's Base is its own episode."""
    if "case_id" in r:
        return ("m", r["brain"], r["case_id"], r["kind"], r["arm"], r["d"], not r["conf"]["arms"])
    return ("b", r["brain"], r["kind"], r["cell"], r["arm"], r["suite"], r["task"], r["init"])


def parse_line(line):
    """{(runner, brain): kbar} of the calibration line: "m" (LIBERO-MAX) before "bench:", "b" after."""
    if not line or not LINE_RE.fullmatch(line):
        raise SystemExit(f"not a calibration line: {line!r}")
    head, _, bench = line.partition(" · bench: ")
    kb = {("m", b): float(v) for b, v in re.findall(r"kbar_(\w+)=(-?\d+\.\d+)", head)}
    kb |= {("b", b): float(v) for b, v in re.findall(r"kbar_(\w+)=(-?\d+\.\d+)", bench)}
    if set(kb) != {("m", b) for b in CAL} | {("b", P), ("b", S)}:
        raise SystemExit(f"calibration line without its six brains: {line!r}")
    return kb


def spec_line(text):
    """The journaled calibration line of the spec (the last one), else None; the template of §5 does not match."""
    hits = [m.group(0) for m in LINE_RE.finditer(text)]
    return hits[-1] if hits else None


def mindex(rs):
    idx = defaultdict(dict)
    for r in rs:
        idx[r["brain"], r["kind"], r["arm"], r["d"]][r["case_id"]] = r
    return idx


def bindex(rs):
    idx = defaultdict(dict)
    for r in rs:
        idx[r["brain"], r["kind"], r["cell"], r["arm"]][r["suite"], r["task"], r["init"]] = r
    return idx


def invalid(r):
    return r.get("prefix_ok") is False or r.get("replay_miss") is not None


def mprs(mi, clu, brain, a, b, d=0, cases=None):
    """spec §6, §9: (task cluster, success a, success b) on the same case, one brain, (s, d); a, b: Dynamic arms or
    "Base". Pairs with prefix_ok false or a replay miss drop out. (pairs, invalid pairs)."""
    kd = lambda x: ("base", "none") if x == "Base" else ("dynamic", x)  # noqa: E731
    x, y = mi.get((brain, *kd(a), d), {}), mi.get((brain, *kd(b), d), {})
    out, bad = [], 0
    for c in sorted(x.keys() & y.keys()):
        if cases is None or c in cases:
            if invalid(x[c]) or invalid(y[c]):
                bad += 1
            else:
                out.append((clu[c], x[c]["success"], y[c]["success"]))
    return out, bad


def bprs(bi, brains, cells, a, b, inits=range(10), keep=lambda r: True):
    """(task cluster, success a, success b) on the same bench episode seed (task, init, cell), kind move."""
    out = []
    for br in brains:
        for c in cells:
            x, y = bi.get((br, "move", c, a), {}), bi.get((br, "move", c, b), {})
            out += [(k[:2], x[k]["success"], y[k]["success"]) for k in sorted(x.keys() & y.keys())
                    if k[2] in inits and keep(y[k])]
    return out


# spec §9: rule -> (test, [(brain, a, b)], predicate per brain); Q4c / Q4d pass when they pass for each brain
RULES = {"Q4a": ("pi05, M1, d = 0: (Gauto - none) >= +5 and lo2.5 > 0", [(P, "Gauto", "none")], s3.sup(5)),
         "Q4b": ("oftplus, M1, d = 0: (Gauto - none) >= +5 and lo2.5 > 0", [(OP, "Gauto", "none")], s3.sup(5)),
         "Q4c": ("pi05 and oftplus, M1: one-sided (Gauto - G) >= -2 for each", [(P, "Gauto", "G"), (OP, "Gauto", "G")],
                 s3.ni(-2)),
         "Q4d": ("pi05 and oftplus, M1: one-sided (Gauto - PPC) >= -2 for each",
                 [(P, "Gauto", "PPC"), (OP, "Gauto", "PPC")], s3.ni(-2))}
Q5 = (("bench, move, oracle, pi05 + smolvla, A + A20, inits 0-9 (0-4 if B59 is cut): (Glead - Gauto) >= +3 and "
       "lo2.5 > 0"), s3.sup(3))
VERDICTS = {"ВНЕШНИЙ ТЕСТ: G-AUTO ПОМОГАЕТ": ("connect", "Q4a", "Q4b", "Q4c"),
            "G-AUTO НЕ ХУЖЕ PPC НА ИХ ТЕСТЕ": ("connect", "Q4d"),
            "УПРЕЖДЕНИЕ НА ПЛАВНОМ ДВИЖЕНИИ": ("Q5",)}


def mrule(name, mi, clu):
    test, specs, ok = RULES[name]
    res, bad, n = {"test": test}, 0, 0
    for b, x, y in specs:
        p, k = mprs(mi, clu, b, x, y)
        d = e2.diff_pp(p)
        bad, n = bad + k, n + len(p) + k
        res[b] = (e2.rnd(d) or {}) | {"invalid_pairs": k, "pass": None if d is None else bool(ok(d))}
    ps = [res[b]["pass"] for b, _, _ in specs]
    res["pass"] = False if False in ps else None if None in ps else True  # no pairs: not read
    return res, bad, n


def connect(raw, split):
    """spec §9: Base "none" success on the connect set per brain against their Table 2 overall Base, within +-10 pp
    (pi0.5, OFT, SmolVLA; equality passes); OFT+ report only. A brain with > 1% of its 300 Base episodes missing does
    not pass. (pass, detail)."""
    ids, out = set(split["connect"]), {}
    for b in (P, O, S, OP):
        x = {r["case_id"]: r["success"] for r in raw if r.get("case_id") in ids and r["brain"] == b
             and not r["conf"]["arms"] and r["kind"] == "base" and not r.get("skipped_reason")}
        rate, miss = (100 * float(np.mean(list(x.values()))) if x else None), len(ids - x.keys())
        ok = (None if b not in TABLE2 else rate is not None and abs(rate - TABLE2[b]) <= CONNECT_PP + TOL
              and miss <= 0.01 * len(ids) + TOL)
        out[b] = {"n": len(x), "missing": miss, "base_none": None if rate is None else round(rate, 2),
                  "table2": TABLE2.get(b), "pass": ok}
    return all(v["pass"] is not False for v in out.values()), out


def connect_table(det):
    rows = [(f"{'brain':8} {'n':>4} {'miss':>4} {'Base none %':>11} {'Table 2 %':>9} {'diff':>6}  "
             f"pass (+-{CONNECT_PP} pp)")]
    for b, v in det.items():
        diff = None if v["base_none"] is None or v["table2"] is None else v["base_none"] - v["table2"]
        rows.append(f"{b:8} {v['n']:4} {v['missing']:4} {v['base_none']!s:>11} {v['table2'] or '-'!s:>9} "
                    f"{'-' if diff is None else f'{diff:+.1f}':>6}  "
                    f"{'report only' if v['pass'] is None else v['pass']}")
    return "\n".join(rows)


def samples(recs, order):
    """Valid shadow samples in the given order, one per item (None: no sample), and the items without a record."""
    by = {}
    for r in recs:
        by.setdefault(r["_k"], r)
    sm = [None if k not in by or not by[k].get("shadow") else by[k]["shadow"]["sample"] for k in order]
    return sm, [k for k in order if k not in by]


def calib(raw, split, pairs):
    """spec §5: kbar per brain = the median of the valid shadow samples of the calibration cases (M0, the split's calib
    in order) with the percentile bootstrap band (maxwrap.kbar_line: 2000 reps, 5-95%); the bench's from its frozen 60
    pairs (B-cal). Every calibration case needs its record. With N < 45 for a brain the further cases in order (the
    split's eval; the pairs file's more) give maxwrap.n_extra if their Gcal records are there. (line or None,
    report)."""
    sha, rep, more_ok = split["sha256"], {}, True
    m = [r | {"_k": r["case_id"]} for r in raw if "case_id" in r and r["arm"] == "Gcal" and r["kind"] == "dynamic"
         and r["conf"]["split"] == sha]
    bn = [r | {"_k": (r["suite"], r["task"], r["init"])} for r in raw if "case_id" not in r and r["arm"] == "Gcal"
          and r["kind"] == "step" and r["cell"] == "A"]
    for (tag, b), recs, order, nxt in ([(("m", b), [r for r in m if r["brain"] == b], split["calib"], split["eval"])
                                         for b in CAL]
                                        + [(("b", b), [r for r in bn if r["brain"] == b], *pairs) for b in (P, S)]):
        sm, miss = samples(recs, order)
        if miss:
            raise SystemExit(f"!!! {tag} {b}: no Gcal record for {len(miss)} calibration cases, e.g. {miss[:3]}: rerun "
                             "the calib stage (resume runs the missing ones)")
        more, gone = samples(recs, nxt)
        more = more[:nxt.index(gone[0])] if gone else more  # the further cases in order that have a record
        valid = [x for x in sm if x is not None]
        r = {"N": len(valid), "of": len(order)}
        if valid:
            k = maxwrap.kbar_line(sm)
            r |= {"kbar": round(k[0], 6), "band": [round(k[1], 6), round(k[2], 6)],
                  "kbar_n": {n: round(float(np.median(valid[:n])), 4) for n in range(10, len(valid) + 1, 10)}}
        if len(valid) < N_MIN:
            try:
                r["n_extra"] = maxwrap.n_extra(sm, more)
            except ValueError:
                r["n_extra"], more_ok = None, False
                nxt_name = "the split's eval" if tag == "m" else "the pairs file's more"
                r["need"] = (f"at least {N_MIN - len(valid)} more valid samples: run Gcal on the next cases in order "
                             f"({nxt_name}, all brains of this runner) and --calib again")
        rep[f"{tag}/{b}"] = r
    if any(v["N"] < N_MIN for v in rep.values()):
        ext = {t: max([v.get("n_extra") or 0 for k, v in rep.items() if k[0] == t] or [0]) for t in "mb"}
        rep["extra"] = ext if more_ok else None  # spec §5: the maximum over brains drops out of evaluation for all
        return None, rep
    f = lambda v: f"{v['kbar']:.3f} [{v['band'][0]:.3f}, {v['band'][1]:.3f}] N={v['N']}"  # noqa: E731
    line = ("C1-E4 part 2 calib: " + " · ".join(f"kbar_{b}={f(rep['m/' + b])}" for b in CAL) + " · bench: "
            + " · ".join(f"kbar_{b}={f(rep['b/' + b])}" for b in (P, S)))
    assert parse_line(line)
    return line, rep


def smoke(raw):
    """The pod smoke's validity on calibration debug cases (never read), per brain: invalid Dynamic episodes (prefix_ok
    false or a replay miss) <= 2% (spec §9's pair rule), check (g) ok for every Dynamic "none" at d = 0, and the share
    of Base episodes without an event (report). (pass, detail)."""
    out, ok = {}, True
    for b in sorted({r["brain"] for r in raw if "case_id" in r}):
        rs = list({key(r): r for r in raw if r.get("case_id") and r["brain"] == b
                   and not r.get("skipped_reason")}.values())
        dyn, base = [r for r in rs if r["kind"] == "dynamic"], [r for r in rs if r["kind"] == "base"]
        bad = sum(map(invalid, dyn))
        g = [r["g_check"]["ok"] for r in dyn if r["method"] == "none" and r["d"] == 0 and r["g_check"]]
        out[b] = {"dynamic": len(dyn), "invalid": bad, "invalid_share": round(bad / max(len(dyn), 1), 4),
                  "g_check_ok": [sum(g), len(g)], "base": len(base),
                  "no_event_share": round(float(np.mean([r["e"] is None for r in base])), 4) if base else None,
                  "skipped": sum(1 for r in raw if r.get("brain") == b and r.get("skipped_reason"))}
        ok &= bool(dyn) and bad <= 0.02 * len(dyn) + TOL and all(g) and not out[b]["skipped"]
    return ok, out


def read_tempo(path):
    """The last "tempo (s/episode): k=v ..." line of the smoke's tempo file (scripts/gpu_c1e4m.sh)."""
    hits = re.findall(r"tempo \(s/episode\): (.*)", Path(path).read_text())
    t = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", hits[-1])} if hits else {}
    if missing := [k for k in TEMPO if not t.get(k)]:
        raise SystemExit(f"!!! no tempo for {missing} in {path}: no forecast")
    return t


def forecast(tempo, cut=(), rate=RATE):
    """spec §13: hours and dollars per block on one GPU, one runner process at a time, from seconds per episode per
    brain and runner; the total, the total under each cumulative cut of the fuse order, and under cut."""
    kb = {(r, b): 0.0 for r in "mb" for b in CAL}
    n = defaultdict(Counter)  # fuse block -> tempo key -> episodes
    for ln in lines((), kb):
        tk = f"{'lmax' if ln['runner'] == 'c1e4m' else 'bench'}_{ln['brain']}"
        for k in ln["keys"]:
            n[fuse(ln, k)][tk] += 1
    h = {blk: sum(tempo[t] * c for t, c in v.items()) / 3600 for blk, v in n.items()}
    tot = lambda drop: {"hours": round(sum(v for b, v in h.items() if b not in drop), 2),  # noqa: E731
                        "usd": round(rate * sum(v for b, v in h.items() if b not in drop), 2)}
    return {"usd_per_h": rate, "tempo_s": {k: tempo[k] for k in TEMPO},
            "blocks": {b: {"episodes": sum(n[b].values()), "hours": round(v, 2), "usd": round(rate * v, 2)}
                       for b, v in h.items()},
            "total": tot(()), "cuts": {" ".join(CUT_ORDER[:k]): tot(CUT_ORDER[:k]) for k in range(1, 6)},
            "cut_now": " ".join(cut) or "none", "planned": tot(cut)}


def libero_names():
    """LIBERO's 40 task names (hf-libero, not a LIBERO-plus overlay: maxwrap.base_task checks 10 per suite)."""
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    return {s: bd[s]().get_task_names() for s in run.MAX_STEPS}


def summarize(raw, names, split, kline, cut=(), override=None):
    """The read gates, then the rules, the verdicts and the report (spec §9, §11); kline: the spec's calibration
    line."""
    sha, skipped, stale = split["sha256"], [], 0
    keep = []
    for r in raw:  # a skipped episode (motion blur without Wand) counts as missing; another split's are not read
        if r.get("skipped_reason"):
            skipped.append(key(r))
        elif "case_id" in r and r["conf"]["arms"] and r["conf"]["split"] != sha:
            stale += 1
        else:
            keep.append(r)
    succ = defaultdict(set)
    for r in keep:
        succ[key(r)].add(r["success"])
    if conflict := [k for k, v in succ.items() if len(v) > 1]:
        raise SystemExit(f"{len(conflict)} episodes recorded twice with different success, e.g. {conflict[0]}")
    kb = parse_line(kline) if kline else None
    plan = lines(cut, kb or {(r, b): 0.0 for r in "mb" for b in CAL}, split)
    want = set().union(*(ln["keys"] for ln in plan))
    rs = list({key(r): r for r in keep if key(r) in want}.values())  # the planned episodes only, one record each
    stable = ("revision", "precision", "s", "R", "t_ramp", "k_p", "tau_k")
    for rb in {(k[0], k[1]) for k in map(key, rs)}:
        cf = {json.dumps({k: r["conf"].get(k) for k in stable}) for r in rs if key(r)[:2] == rb}
        if len(cf) > 1:
            raise SystemExit(f"{rb} records mix settings (conf): {sorted(cf)}")
    have = {key(r) for r in rs}
    clu = {r["case_id"]: maxwrap.base_task({"case_id": r["case_id"], "task_suite_name": r["suite"],
                                            "task_name": r["task_name"], "substrate_variant": r["substrate_variant"]},
                                           names) for r in rs if "case_id" in r}
    m = [r for r in rs if "case_id" in r and r["conf"]["arms"]]  # LIBERO-MAX records but connect's
    mi, bi = mindex(m), bindex([r for r in rs if "case_id" not in r])
    c_ok, c_det = connect(rs, split)
    g = [r["g_check"]["ok"] for b in (P, OP) for r in mi.get((b, "dynamic", "none", 0), {}).values() if r["g_check"]]
    blocks = {"M1 pi05": lambda ln, k: ln["block"] == "M1pi" and k[4] not in M2,
              "M1 oftplus": lambda ln, k: ln["block"] == "M1oftplus",
              "B oracle": lambda ln, k: ln["block"] == "Boracle"}
    comp = {}
    for name, f in blocks.items():
        exp = {k for ln in plan for k in ln["keys"] if f(ln, k)}
        miss = len(exp - have)
        comp[name] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    rules, inv = {}, {}
    for name in RULES:
        rules[name], bad, n = mrule(name, mi, clu)
        inv[name] = {"pairs": n, "invalid": bad, "pass": bad <= 0.02 * n + TOL}
    kb_bad = sorted({(rr, b) for r in rs if method(r["arm"]) in ("Gauto", "Glead")
                     for rr, b in [("m" if "case_id" in r else "b", r["brain"])]
                     if kb is None or abs(r["conf"]["kbar"] - kb[rr, b]) > 1e-12})
    gate = {"records": len(raw), "skipped": len(skipped), "other_split": stale, "outside_plan": len(keep) - len(rs),
            "connect": c_det | {"pass": c_ok}, "g_check_M1": {"ok": sum(g), "of": len(g), "pass": all(g)},
            "complete": comp, "invalid_pairs": inv, "calib_line": kline, "kbar_mismatch": kb_bad, "cut": list(cut)}
    gate["pass"] = bool(c_ok and all(g) and all(v["pass"] for v in comp.values())
                        and all(v["pass"] for v in inv.values()) and kline and not kb_bad)
    res = {"read_gate": gate}
    if not gate["pass"]:
        if not override:
            return res  # spec §9: the rules are not read
        gate["override"] = override  # a deviation recorded in the spec journal before reading
    q5 = e2.diff_pp(bprs(bi, (P, S), ("A", "A20"), "Glead", "Gauto", range(5 if "B59" in cut else 10)))
    rules["Q5"] = {"test": Q5[0], **(e2.rnd(q5) or {}), "pass": None if q5 is None else bool(Q5[1](q5))}
    ps = {n: v["pass"] for n, v in rules.items()} | {"connect": c_ok}
    verdicts = {v: "НЕТ" if False in (x := [ps[n] for n in need]) else "НЕ ЧИТАЛОСЬ" if None in x else "ДА"
                for v, need in VERDICTS.items()}
    if "override" in gate:
        verdicts = {v: "ОТСТУПЛЕНИЕ (проверка перед чтением не пройдена, см. read_gate.override): " + x
                    for v, x in verdicts.items()}
    fig = {b: {a: e2.rnd(e2.diff_pp(mprs(mi, clu, b, a, "Base")[0])) for a in ("none", "Gauto")} for b in (P, S, O, OP)}
    return res | {"verdicts": verdicts, "rules": rules, "figure": fig, "report": report(rs, mi, bi, clu, split)}


# ---------------------------------------------------------------- §11 report (only reported, never a rule)

def report(rs, mi, bi, clu, split):
    mean = lambda v: round(float(np.mean(v)), 4) if len(v) else None  # noqa: E731  (never NaN: summary.json stays JSON)
    dpp = lambda p: e2.rnd(e2.diff_pp(p))  # noqa: E731
    by = lambda g, f: {str(v): [mean([r["success"] for r in g if r.get(f) == v]),  # noqa: E731  (success, n)
                                sum(r.get(f) == v for r in g)] for v in sorted({r.get(f) for r in g}, key=str)}
    out = {"success": {}, "dyn_minus_base": {}, "vs_none_invalid_pairs": {}, "by": {}, "connect_by": {}, "delay_D": {},
           "M2": {}, "gauto_vs_ppc": {}, "B": {}}
    for k in sorted(mi, key=str):  # (brain, kind, arm, d)
        g, name = list(mi[k].values()), "/".join(map(str, k))
        out["success"][name] = {
            "n": len(g), "success": mean([r["success"] for r in g]), "grasp": mean([r["grasp_ok"] for r in g]),
            "no_event": mean([r["e"] is None for r in g]),
            "calls": [round(float(np.mean([r[c] for r in g])), 2) for c in ("calls_sched", "calls_trig", "calls_live",
                                                                             "calls_shadow")]}
        out["by"][name] = {f: by(g, f) for f in ("distance_m", "source", "substrate_category")}
        if k[1] == "dynamic":  # their terms: Dynamic - Base on the same case; and the invalid pairs against "none"
            out["dyn_minus_base"][name] = dpp(mprs(mi, clu, k[0], k[2], "Base", k[3])[0])
            out["vs_none_invalid_pairs"][name] = mprs(mi, clu, k[0], k[2], "none", k[3])[1]
    conn = [r for r in rs if r.get("case_id") and not r["conf"]["arms"]]
    for b in (P, S, O, OP):
        g = [r for r in conn if r["brain"] == b]
        out["connect_by"][b] = {f: by(g, f) for f in ("change_type", "source")}
    e300 = set(split["eval"][:300])
    for d in (0, 10, 20):  # block D (pi0.5, the first 300 evaluation cases; d = 0 from M1); not LIBERO-MAX's protocol
        fresh = []  # the first plan observed after the event arrives before the first close command (spec §11)
        for c, r in mi.get((P, "dynamic", "none", d), {}).items():
            if c in e300 and r["e"] is not None:
                arr = [ta for to, ta in r["plans_in"] if to >= r["e"]]
                fresh.append(bool(arr) and (r["t_close"] is None or arr[0] < r["t_close"]))
        out["delay_D"][f"d{d}"] = {f"{a}-none": dpp(mprs(mi, clu, P, a, "none", d, e300)[0]) for a in ("Gauto", "PPC")}
        out["delay_D"][f"d{d}"]["fresh_before_close_none"] = mean(fresh)
    for a, b in (("G@glr", "none"), ("PPC@glr", "none"), ("T0", "none"), ("G@glr", "G"), ("PPC@glr", "PPC")):
        out["M2"][f"{a}-{b}"] = dpp(mprs(mi, clu, P, a, b, 0, e300)[0])
    for b in (P, OP):  # spec §9: G-auto's superiority over PPC (lower > 0) is reported, not a rule
        d = e2.diff_pp(mprs(mi, clu, b, "Gauto", "PPC")[0])
        out["gauto_vs_ppc"][b] = d and e2.rnd(d) | {"superior": d["lo2.5"] > 0}
    for br in (P, S):
        x = {}
        for c in ("A", "A20"):
            for v in lead.SPEEDS:
                fast = lambda r, v=v: r["speed"] == v  # noqa: E731
                x[f"{c}/{v}"] = {f"{a}-{b}": dpp(bprs(bi, (br,), (c,), a, b, keep=fast))
                                 for a, b in (("Glead", "Gauto"), ("PPC", "Gauto"))}
        for a, b in (("Gauto@cv", "none"), ("Glead@cv", "Gauto@cv"), ("Gauto@cv", "Gauto"), ("Glead@cv", "Glead")):
            x[f"A20/{a}-{b}"] = dpp(bprs(bi, (br,), ("A20",), a, b))
        for (b2, kind, c, a), g in sorted(bi.items()):
            if b2 == br and kind == "move":
                x[f"{c}/{a}/lead_on_share"] = mean([r["lead_on_share"] for r in g.values()
                                                    if r.get("lead_on_share") is not None])
                x[f"{c}/{a}/stop"] = dict(Counter(r["move_stop_reason"] for r in g.values()))
        out["B"][br] = x
    cv = json.loads(OFFLINE.read_text())["cv"]["bands"]
    v = [round(1000 * s / lead.HZ, 2) for s in lead.SPEEDS]
    out["ppc_cv_offline"] = {"ppc_v_min_cv_mm_per_step": round(1000 * cv["ppc_v_min_cv"], 2), "speeds_mm_per_step": v,
                             "ppc_cv_sees_motion": 1000 * cv["ppc_v_min_cv"] <= max(v),
                             "note": "PPC@cv is not in the grid: its band is above every speed class, so it equals "
                                     "none by construction (spec §7, §11)"}
    return out


def figure(path, fig):
    """Report only (spec §6): their 14 target-relocation points of Table 12 with their median, and our Dynamic - Base
    of "none" and G-auto per brain with 95% cluster-bootstrap intervals."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplots(figsize=(9, 5.2))
    ys = np.array(list(TABLE12.values()))
    ax.scatter(np.linspace(-0.25, 0.25, len(ys)), np.sort(ys), color="0.55", s=22, zorder=3)
    ax.hlines(float(np.median(ys)), -0.4, 0.4, colors="0.3", linestyles="--")
    ax.text(0.42, float(np.median(ys)), f"median {np.median(ys):.1f}", va="center", fontsize=8)
    for name in ("π0.5", "OpenVLA-OFT", "SmolVLA"):
        ax.annotate(name, (0.27, TABLE12[name]), fontsize=7, color="0.3", va="center")
    for j, b in enumerate((P, S, O, OP), start=1):
        for dx, a, col in ((-0.12, "none", "tab:red"), (0.12, "Gauto", "tab:blue")):
            d = fig[b][a]
            if d:
                ax.errorbar(j + dx, d["diff_pp"], yerr=[[d["diff_pp"] - d["lo2.5"]], [d["hi97.5"] - d["diff_pp"]]],
                            fmt="o", color=col, capsize=3, label=a if j == 1 else None)
    ax.set_xticks(range(5), ["LIBERO-MAX\nTable 12", "π0.5 v044", "SmolVLA", "OpenVLA-OFT", "OFT+"])
    ax.axhline(0, color="0.8", lw=0.8)
    ax.set_ylabel("Dynamic − Base success, pp (target relocation)")
    ax.legend(title="ours", fontsize=8)
    f.text(0.01, 0.01, "Theirs: arXiv 2609.36518v1, Table 12 (CC BY 4.0), all 1000 target-relocation cases. Ours: "
           "940 evaluation cases (M4: SmolVLA, OFT: 300), 95% cluster bootstrap over 33 tasks.\nOur π0.5 is LeRobot "
           "v044 at s = 10 (theirs openpi, Q = 5): a reference only. SmolVLA: the same checkpoint.", fontsize=7,
           va="bottom")
    f.subplots_adjust(bottom=0.2)
    f.savefig(path, dpi=150)


def selftest():
    split = maxwrap.load_split(SPLIT)
    kline = ("C1-E4 part 2 calib: kbar_pi05=0.300 [0.250, 0.350] N=58 · kbar_oftplus=0.100 [0.050, 0.150] N=50 · "
             "kbar_oft=0.050 [0.000, 0.100] N=47 · kbar_smolvla=0.040 [0.010, 0.070] N=55 · bench: kbar_pi05=0.350 "
             "[0.300, 0.400] N=60 · kbar_smolvla=0.030 [0.000, 0.060] N=59")
    kb = parse_line(kline)
    assert kb[("m", P)] == 0.3 and kb[("b", S)] == 0.03 and spec_line(f"x `{kline}`.") == kline
    assert spec_line("`C1-E4 part 2 calib: kbar_pi05=… [lo, hi] N=… · kbar_oftplus=…`") is None  # §5's template

    def blocks(cut=()):
        n = Counter()
        for ln in lines(cut, kb):
            for k in ln["keys"]:
                n[fuse(ln, k)] += 1
        return dict(n)
    full = {"connect": 1200, "M0": 480, "Bcal": 120, "M1pi": 4700, "M2": 900, "M1oftplus": 4700, "M4S": 1500,
            "Boracle": 2080, "B59": 2080, "M4oft": 1500, "Bnoisy": 1040, "D": 2400}  # B oracle 4160 = 2080 + 2080
    assert blocks() == full and sum(full.values()) == 22700, blocks()
    for k in range(6):  # every cumulative cut: the cut blocks go, the never-cut ones stay whole
        assert blocks(CUT_ORDER[:k]) == {b: v for b, v in full.items() if b not in CUT_ORDER[:k]}, k
    for bad in (("M1pi",), ("M2",), ("B59",), ("connect",), ("D", "Bnoisy")):
        try:
            lines(bad, kb)
            raise AssertionError(f"cut {bad} accepted")
        except SystemExit:
            pass
    assert {ln["phase"] for ln in lines()} == {"connect", "calib"} and len(lines()) == 18
    pl = lines((), kb)
    assert [ln["phase"] for ln in pl] == sorted((ln["phase"] for ln in pl), key=["connect", "calib", "grid"].index)
    assert len({ln["id"] for ln in pl}) == len(pl)  # one --out per (block, brain, set, source, d)
    order = list(dict.fromkeys(ln["block"] for ln in pl if ln["phase"] == "grid"))
    assert order == ["M1pi", "M1oftplus", "M4S", "Boracle", "M4oft", "Bnoisy", "D"], order
    for ln in pl:  # plus / pro by the split file: the pro process gets exactly the "pro-" cases
        if ln["runner"] == "c1e4m":
            assert f"--source {ln['source']}" in ln["args"] and f"_{ln['source']}_d" in ln["id"]
            assert all(k[2].startswith("pro-") == (ln["source"] == "pro") for k in ln["keys"])
        assert ("--kbar" in ln["args"]) == bool({"Gauto", "Glead"} & set(map(method, ln["arms"])))
    conn = [ln for ln in pl if ln["block"] == "connect" and ln["brain"] == P]
    npro = sum(i.startswith("pro-") for i in split["connect"])
    assert [len(ln["keys"]) for ln in conn] == [300 - npro, npro] and npro == 93
    assert "--kbar 0.3" in next(ln["args"] for ln in pl if ln["id"].startswith("M1M2_pi05"))
    assert "--kbar 0.03" in next(ln["args"] for ln in pl if ln["id"] == "Boracle_smolvla")
    assert "--arms= --d 0" in conn[0]["args"]  # Base only: one shell word
    assert next(ln for ln in lines(("D", "M2"), kb) if ln["block"] == "M1pi")["id"].startswith("M1pi_pi05_eval300")
    # forecast: 3.6 s per episode everywhere -> hours = episodes / 1000
    fc = forecast(dict.fromkeys(TEMPO, 3.6), ("D", "M2"))
    assert abs(fc["total"]["hours"] - 22.7) < 1e-9 and abs(fc["total"]["usd"] - round(22.7 * RATE, 2)) < 1e-9
    assert fc["cuts"]["D"]["hours"] == 20.3 and fc["cuts"]["D M2 Bnoisy M4oft B59"]["hours"] == 14.78
    assert fc["planned"] == fc["cuts"]["D M2"] and fc["blocks"]["M2"] == {"episodes": 900, "hours": 0.9, "usd": 0.32}
    assert forecast(dict.fromkeys(TEMPO, 3.6) | {"bench_pi05": 7.2})["total"]["hours"] == 25.36  # + 2660 bench pi05
    # rule edges: equality to the threshold passes, "lower > 0" is strict
    for n, t in (("Q4a", 5), ("Q4b", 5)):
        ok = RULES[n][2]
        assert ok({"diff_pp": t, "lo2.5": 1e-6}) and not ok({"diff_pp": t - 1e-6, "lo2.5": 1}), n
        assert not ok({"diff_pp": 99, "lo2.5": 0}), n
    assert Q5[1]({"diff_pp": 3, "lo2.5": 1e-6}) and not Q5[1]({"diff_pp": 3 - 1e-6, "lo2.5": 1})
    assert not Q5[1]({"diff_pp": 50, "lo2.5": 0})
    for n in ("Q4c", "Q4d"):
        assert RULES[n][2]({"lo5_one_sided": -2}) and not RULES[n][2]({"lo5_one_sided": -2 - 1e-6}), n
    assert float(np.median(list(TABLE12.values()))) == -40.7 and len(TABLE12) == 14

    # a synthetic grid: every planned episode, LIBERO-MAX cases on 40 fake tasks (10 per suite)
    names = {s: [f"{s}_t{j}" for j in range(10)] for s in run.MAX_STEPS}
    suites = list(run.MAX_STEPS)
    rng = np.random.default_rng(0)
    u = {}

    def uu(c):  # one number per case: shared by its episodes, so pairs correlate
        return u.setdefault(c, rng.random())

    def recs(fx, cut=(), line=kb):
        out = []
        for ln in lines(cut, line):
            for k in sorted(ln["keys"], key=str):
                if k[0] == "m":
                    _, b, c, kind, a, d, _ = k
                    h = sum(map(ord, c))
                    s = suites[h % 4]
                    out.append({"brain": b, "case_id": c, "kind": kind, "arm": a, "method": method(a), "d": d,
                                "conf": {"arms": list(ln["arms"]), "split": split["sha256"], "revision": "r", "s": 10,
                                         "precision": "fp32", "R": 360, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01,
                                         "kbar": kb.get(("m", b)) if ln["phase"] == "grid" else None},
                                "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None,
                                "source": "pro" if c.startswith("pro-") else "plus", "distance_m": (0.06, 0.12)[h % 2],
                                "change_type": "target_relocation", "substrate_category": "x", "e": 30,
                                "success": bool(fx(k) > uu(c)), "grasp_ok": True, "prefix_ok": None if kind == "base"
                                else True, "replay_miss": None, "skipped_reason": None,
                                "g_check": {"ok": True} if kind == "dynamic" and a == "none" and d == 0 else None,
                                "calls_sched": 20, "calls_trig": 0, "calls_live": 20, "calls_shadow": 0,
                                "plans_in": [[0, 0], [30, 30 + d]], "t_close": 50,
                                "shadow": {"sample": 0.2} if a == "Gcal" else None})
                else:
                    _, b, kind, cell, a, s, t, i = k
                    out.append({"brain": b, "kind": kind, "cell": cell, "arm": a, "method": method(a), "suite": s,
                                "task": t, "init": i, "conf": {"revision": "r", "precision": "fp32", "t_ramp": 5,
                                                               "k_p": 1.0, "tau_k": 0.01, "kbar": kb[("b", b)]},
                                "success": bool(fx(k) > uu((s, t, i, cell))), "speed": lead.SPEEDS[i % 3],
                                "lead_on_share": 0.5 if method(a) == "Glead" else None, "move_stop_reason": "tau",
                                "shadow": {"sample": 0.3} if a == "Gcal" else None})
        return out

    base = {P: 0.797, O: 0.643, S: 0.261, OP: 0.5}
    win = lambda k: (base[k[1]] if k[0] == "m" and k[3] == "base" else  # noqa: E731
                     float(k[4] in ("Gauto",)) if k[0] == "m" else float(k[4] == "Glead"))
    raw = recs(win)
    out = summarize(raw, names, split, kline)
    assert out["read_gate"]["pass"], out["read_gate"]
    r = out["rules"]
    assert all(r[n]["pass"] for n in ("Q4a", "Q4b", "Q4c", "Q4d", "Q5")), r
    assert r["Q4a"][P]["pairs"] == 940 and r["Q4c"][OP]["pairs"] == 940 and r["Q5"]["pairs"] == 1040
    assert r["Q4a"][P]["clusters"] == 40 and r["Q5"]["clusters"] == 26
    assert set(out["verdicts"].values()) == {"ДА"}, out["verdicts"]
    assert out["figure"][S]["Gauto"]["pairs"] == 300 and out["figure"][OP]["none"]["pairs"] == 940
    assert out["report"]["B"][P]["A/0.04"]["Glead-Gauto"]["diff_pp"] == 100
    assert json.dumps(summarize(raw, names, split, kline), sort_keys=True) == json.dumps(out, sort_keys=True)
    p = [(("s", i % 5), bool(i % 3), bool(i % 2)) for i in range(100)]
    assert e2.diff_pp(p) == e2.diff_pp(list(p))  # bootstrap determinism (fixed seed, fixed cluster order)
    # gates: each failure alone stops the reading
    rand = recs(lambda k: base[k[1]] if k[0] == "m" and k[3] == "base" else 0.5)
    assert summarize(rand, names, split, kline)["read_gate"]["pass"]
    fails = {
        "connect": [x | {"success": False} if x.get("case_id") and not x["conf"]["arms"] and x["brain"] == O else x
                    for x in rand],
        "g_check": [x | {"g_check": {"ok": False}} if x.get("g_check") and x["brain"] == OP and x["case_id"] ==
                    split["eval"][5] else x for x in rand],
        "complete": [x for x in rand if not (x["brain"] == OP and x.get("case_id") in split["eval"][:60]
                                             and x["arm"] == "PPC")],
        "invalid": [x | {"prefix_ok": False} if x.get("case_id") in split["eval"][:40] and x["arm"] == "PPC"
                    and x["brain"] == P else x for x in rand],
        "kbar": [x | {"conf": x["conf"] | {"kbar": 0.31}} if x["arm"] == "Gauto" and x["brain"] == P else x
                 for x in rand]}
    for why, rr in fails.items():
        g = summarize(rr, names, split, kline)
        assert not g["read_gate"]["pass"] and "rules" not in g, why
    assert summarize(rand, names, split, None)["read_gate"]["calib_line"] is None
    assert not summarize(rand, names, split, None)["read_gate"]["pass"]
    g = summarize(fails["invalid"], names, split, kline)["read_gate"]["invalid_pairs"]
    assert g["Q4d"]["invalid"] == 40 and g["Q4d"]["pairs"] == 1880 and not g["Q4d"]["pass"] and g["Q4a"]["pass"], g
    ov = summarize(fails["connect"], names, split, kline, override="selftest")
    assert all(v.startswith("ОТСТУПЛЕНИЕ") for v in ov["verdicts"].values())
    assert ov["verdicts"]["G-AUTO НЕ ХУЖЕ PPC НА ИХ ТЕСТЕ"].endswith("НЕТ")  # connect failed
    # 21 of 1040 Q4 pairs missing (> 1%) fails, 9 (<= 1%) passes: the M1 OFT+ block has 4700 episodes
    for n_drop, ok in ((48, False), (47, True)):
        rr = [x for j, x in enumerate(rand) if not (x["brain"] == OP and x["arm"] == "G" and x.get("d") == 0
                                                    and x.get("case_id") in split["eval"][:n_drop])]
        assert summarize(rr, names, split, kline)["read_gate"]["complete"]["M1 oftplus"]["pass"] == ok, n_drop
    # a cut grid: D and M2 are not expected, B59's inits not either
    cutr = recs(win, ("D", "M2", "Bnoisy", "M4oft", "B59"))
    oc = summarize(cutr, names, split, kline, ("D", "M2", "Bnoisy", "M4oft", "B59"))
    assert oc["read_gate"]["pass"] and oc["rules"]["Q5"]["pairs"] == 520, oc["read_gate"]
    assert not summarize(cutr, names, split, kline)["read_gate"]["pass"]  # read as uncut: B oracle 5-9 missing
    # the connection gate at its edges: 300 cases, 69.7 = 79.7 - 10
    c_ids = split["connect"]

    def conn(n):  # Base successes of the 300 per brain
        return [{"brain": b, "case_id": c, "kind": "base", "arm": "none", "conf": {"arms": []}, "success": j < n[b]}
                for b in (P, O, S, OP) for j, c in enumerate(c_ids)]
    good = {P: 210, O: 164, S: 78, OP: 0}  # 70.0, 54.67, 26.0 %: 79.7 - 10 = 69.7, 64.3 - 10 = 54.3
    ok, det = connect(conn(good), split)
    assert ok and det[OP]["pass"] is None and det[P]["base_none"] == 70.0, det
    assert not connect(conn(good | {P: 209}), split)[0] and not connect(conn(good | {S: 109}), split)[0]  # 69.67, 36.33
    TABLE2[P] = 80.0  # equality passes: 70.0 = 80.0 - 10
    try:
        assert connect(conn(good), split)[0]
    finally:
        TABLE2[P] = 79.7
    assert not connect([x for x in conn(good) if x["case_id"] not in c_ids[:4]], split)[0]  # 4 of 300 missing > 1%
    # calibration: the line, its format, the N < 45 path and n_extra over the further cases
    cal = recs(lambda k: 0.5)
    pairs = [[(q["suite"], q["task"], q["init"]) for q in maxwrap.load_split(PAIRS)[w]] for w in ("calib", "more")]
    line, rep = calib(cal, split, pairs)
    assert line == ("C1-E4 part 2 calib: kbar_pi05=0.200 [0.200, 0.200] N=60 · kbar_oftplus=0.200 [0.200, 0.200] N=60 "
                    "· kbar_oft=0.200 [0.200, 0.200] N=60 · kbar_smolvla=0.200 [0.200, 0.200] N=60 · bench: "
                    "kbar_pi05=0.300 [0.300, 0.300] N=60 · kbar_smolvla=0.300 [0.300, 0.300] N=60"), line
    assert LINE_RE.fullmatch(line) and parse_line(line)[("b", P)] == 0.3
    short = [x | {"shadow": {"sample": None}} if x.get("case_id") in split["calib"][:20] and x["brain"] == O
             and x["arm"] == "Gcal" else x for x in cal]
    line, rep = calib(short, split, pairs)
    assert line is None and rep["m/oft"]["N"] == 40 and rep["m/oft"]["n_extra"] is None and rep["extra"] is None
    tmpl = next(x for x in cal if x.get("case_id") and x["arm"] == "Gcal")  # a valid sample on the next 7 cases
    more = [tmpl | {"case_id": c, "brain": b} for c in split["eval"][:7] for b in CAL]
    line, rep = calib(short + more, split, pairs)
    assert line is None and rep["m/oft"]["n_extra"] == 5 and rep["extra"] == {"m": 5, "b": 0}, rep
    try:
        calib([x for x in cal if x.get("case_id") != split["calib"][3]], split, pairs)
        raise AssertionError("a calibration case without its record was accepted")
    except SystemExit:
        pass
    # smoke validity: 1 invalid of 50 passes, 2 of 50 do not; a failed check (g) does not
    sm = [{"brain": P, "case_id": f"c{j}", "kind": "dynamic", "arm": "none", "method": "none", "d": 0, "e": 30,
           "prefix_ok": j != 0, "replay_miss": None, "g_check": {"ok": True}, "conf": {"arms": ["none"]}}
          for j in range(50)] + [{"brain": P, "case_id": f"c{j}", "kind": "base", "arm": "none", "method": "none",
                                  "d": 0, "e": None if j < 5 else 30, "conf": {"arms": ["none"]}} for j in range(50)]
    ok, det = smoke(sm)
    assert ok and det[P]["no_event_share"] == 0.1 and det[P]["invalid"] == 1
    assert not smoke([x | {"replay_miss": 12} if x["case_id"] == "c1" and x["kind"] == "dynamic" else x for x in sm])[0]
    assert not smoke([x | {"g_check": {"ok": False}} if x["case_id"] == "c9" and x["kind"] == "dynamic" else x
                      for x in sm])[0]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="?", help="JSON lines of both runners: connect, calib and grid (grid_all.jsonl)")
    ap.add_argument("--override-gate", help="the spec journal entry that records the owner's deviation")
    ap.add_argument("--plan", action="store_true", help='the pod\'s runner lines: "phase<TAB>runner<TAB>brain<TAB>'
                                                        'source<TAB>id<TAB>n<TAB>args"')
    ap.add_argument("--kbar-line", help="with --plan: the journaled calibration line (the grid's kbar)")
    ap.add_argument("--cut", default="", help='report blocks cut by the §13 fuse, e.g. "D M2"')
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--connect", nargs="+", metavar="JSONL", help="connect records: exit 0 if it passes, else 3")
    mode.add_argument("--calib", nargs="+", metavar="JSONL", help="M0 and B-cal records: the line, or exit 4")
    mode.add_argument("--smoke", nargs="+", metavar="JSONL", help="smoke LIBERO-MAX records: exit 0, or 3")
    ap.add_argument("--forecast", metavar="TEMPO", help="the smoke's tempo file: hours and dollars per block")
    ap.add_argument("--rate", type=float, default=RATE, help="$/h")
    ap.add_argument("--fig", help="with the grid: the figure next to their Table 12 (PNG)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f)]  # noqa: E731
    if a.selftest:
        selftest()
    elif a.plan:
        for ln in lines(a.cut.split(), parse_line(a.kbar_line) if a.kbar_line else None):
            print("\t".join(map(str, (ln["phase"], ln["runner"], ln["brain"], ln["source"], ln["id"], len(ln["keys"]),
                                      ln["args"]))))
    elif a.connect:
        ok, det = connect(load(a.connect), maxwrap.load_split(SPLIT))
        print(connect_table(det))
        print(f"C1-E4 part 2 connect: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    elif a.calib:
        sp = maxwrap.load_split(SPLIT)
        pf = maxwrap.load_split(PAIRS)
        line, rep = calib(load(a.calib), sp, [[(q["suite"], q["task"], q["init"]) for q in pf[w]]
                                              for w in ("calib", "more")])
        print(e2.dump(rep))
        if line is None:
            print(f"!!! N < {N_MIN} for {[k for k, v in rep.items() if k != 'extra' and v['N'] < N_MIN]}: extra cases "
                  f"{rep['extra'] or 'unknown yet (see need)'}. The owner freezes the new split (extra = the maximum "
                  "over brains, maxwrap.freeze_split) and journals it before any evaluation is read (spec §5)")
            raise SystemExit(4)
        print(line)
    elif a.smoke:
        ok, det = smoke(load(a.smoke))
        print(e2.dump(det))
        print(f"C1-E4 part 2 smoke validity: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    elif a.forecast:
        print(e2.dump(forecast(read_tempo(a.forecast), a.cut.split(), a.rate)))
    else:
        if not a.grid:
            ap.error("the grid JSON lines are required")
        src = Path(a.grid)
        res = summarize(e2.load(src), libero_names(), maxwrap.load_split(SPLIT), spec_line(SPEC.read_text()),
                        a.cut.split(), a.override_gate)
        (src.parent / "summary.json").write_text(e2.dump(res) + "\n")
        print(e2.dump(res))
        if a.fig and "figure" in res:
            figure(a.fig, res["figure"])
