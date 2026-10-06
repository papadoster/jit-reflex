"""C1-E4 part 2: the pod's plan (§6, §7, §13), its gates, the calibration line (§5), the rules Q4a-Q4d and Q5 with their
verdicts (§9) and the report (§11), applied mechanically to the JSON lines of scripts/c1e4m_run.py (LIBERO-MAX) and
scripts/c1e4_run.py (block B, our bench). Spec docs/superpowers/specs/2026-10-05-c1-e4-part2-design.md. Pure numpy
(matplotlib for --fig); the main summary reads LIBERO's 40 task names from hf-libero (the 33 task clusters,
maxwrap.base_task), so it runs in the LeRobot env without LIBERO-MAX's overlay. Two pods split by brain (spec §13,
owner 2026-10-06; nothing is cut): A = pi0.5, B = OFT+, OFT, SmolVLA; R = results/c1-e4/part2/pod_A|pod_B.
    python scripts/c1e4m_summary.py --plan --pod A [--kbar-line "C1-E4 part 2 calib A: ..."] [--more "m/pi05" --k 20]
        the pod's runner lines: phase, runner, brain, source, id, episodes, args (without the line: connect and calib;
        --more: calib_more's lines for calib's brains with N < 45)
    python scripts/c1e4m_summary.py --connect R/connect_*.jsonl --pod B   the pod's connect gate: exit 0, or 3
    python scripts/c1e4m_summary.py --calib R/M0*.jsonl R/Bcal*.jsonl --pod B   the pod's line, or exit 4
    python scripts/c1e4m_summary.py --smoke R/smoke/*.jsonl --pod B   the smoke's pair validity: 0, or 3
    python scripts/c1e4m_summary.py [R/*.jsonl] --forecast R/tempo_smoke.txt --pod B [--rate 0.34]   (+ hours left)
    python scripts/c1e4m_summary.py --forecast pod_A/tempo_smoke.txt pod_B/tempo_smoke.txt   both pods and the total
    python scripts/c1e4m_summary.py --split0 R/split0.json   the split as first frozen (connect, calib, calib_more)
    python scripts/c1e4m_summary.py pod_A/grid_all_A.jsonl pod_B/grid_all_B.jsonl [--fig fig.png]
    python scripts/c1e4m_summary.py --selftest
If a read gate fails only the gates are returned, unless the owner's deviation, journaled in the spec before reading, is
passed with --override-gate. Bootstrap and rule helpers are C1-E2's and C1-E3's, as part 1's scripts/c1e4_summary.py.
"""

import argparse
import json
import os
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
CAL = (P, OP, O, S)  # spec §5: the calibration lines' order; then the bench's
POD = {"A": (P,), "B": (OP, O, S)}  # spec §13: every arm of a brain on one pod, so pairs never mix machines
BENCH = {"A": P, "B": S}  # the pod's bench brain (B-cal, block B)
PD = "results/c1-e4/part2/pod_{}"  # the pod's records (scripts/gpu_c1e4m.sh)
TABLE2 = {P: 79.7, O: 64.3, S: 26.1}  # spec §3, §9: their Table 2, overall Base (all 8 events), %; OFT+ is not in it
# arXiv 2609.36518v1 (LIBERO-MAX), Table 12 (CC BY 4.0), target relocation: Dynamic - Base success, pp, 1000 pairs each
TABLE12 = {"Cosmos-Policy": -47.5, "π0.5": -34.6, "OpenVLA-OFT": -42.1, "X-VLA": -52.3, "Xiaomi-Robotics-0": -44.0,
           "MolmoAct2": -32.0, "SmolVLA": -23.9, "GR00T N1.7": -52.5, "DM0.5": -39.9, "VLA-JEPA": -40.7,
           "Fast-WAM": -30.4, "HiMem-WAM": -43.7, "Light-WAM": -40.7, "DiT4DiT": -34.7}
ARMS, M2 = ("none", "G", "Gauto", "PPC"), ("G@glr", "PPC@glr", "T0")  # spec §6: M1 / M4 arms; M2 rides on M1's 300
B_ARMS, NOISY, D_ARMS = ("none", "Gauto", "Glead", "PPC"), ("Gauto@cv", "Glead@cv"), ("none", "Gauto", "PPC")
RATE, N_MIN, CONNECT_PP = 0.34, maxwrap.N_MIN, 10  # $/h RTX 4090 (spec §13); calibration N >= 45; connect +-10 pp
NUM = r"-?\d+\.\d+ \[-?\d+\.\d+, -?\d+\.\d+\] N=\d+"
KB0 = {(r, b): 0.0 for r in "mb" for b in CAL}  # a placeholder kbar: the plan's episodes do not depend on it


def mine(pod):
    """The pod's LIBERO-MAX brains in the calibration line's order (both pods' without a pod)."""
    return [b for b in CAL if pod is None or b in POD[pod]]


def tkeys(pod):
    """The pod's tempo keys: seconds per episode per runner and brain (scripts/gpu_c1e4m.sh smoke)."""
    return [f"lmax_{b}" for b in mine(pod)] + [f"bench_{BENCH[pod]}"]


def line_re(pod):
    """spec §5: the pod's calibration line."""
    return re.compile(f"C1-E4 part 2 calib {pod}: " + " · ".join(f"kbar_{b}={NUM}" for b in mine(pod))
                      + f" · bench: kbar_{BENCH[pod]}={NUM}")


def split0(split):
    """The split as first frozen (extra = 0): calibration and evaluation are one order whatever extra is
    (maxwrap.split_cases), so a split refrozen with extra (spec §5) gives it back. Connect, calib and calib_more run on
    it (their records keep its sha256: M0 never reruns), the grid on the split."""
    order = split["calib"] + split["eval"]
    d = {k: v for k, v in split.items() if k != "sha256"} | {"extra": 0, "calib": order[:split["n_cal"]],
                                                             "eval": order[split["n_cal"]:]}
    return d | {"sha256": maxwrap._sha(d)}


def method(arm):
    return arm.partition("@")[0]


def sources(ids):
    """The ids by scene source, in order: LIBERO-MAX's PRO case ids start "pro-" (exactly the cases with a
    substrate_variant: checked on the Mac over all 8000), the runner's source_of."""
    return {s: [i for i in ids if i.startswith("pro-") == (s == "pro")] for s in ("plus", "pro")}


def bench_keys(brain, kind, cells, arms, eps):
    return {("b", brain, kind, c, a, *e) for c in cells for a in arms for e in eps}


def pod_of(brain):
    return next(p for p, bs in POD.items() if brain in bs)


def lines(kbar=None, split=None, pod=None, more=(), k=20):
    """spec §13: the runner lines in run order, the pod's (both pods' without one); without kbar (the journaled
    calibration lines, parse_line) only the connect and calib phases; more (calib's brains with N < 45, e.g.
    ["m/oft", "b/smolvla"]): calib_more's lines, Gcal on the next k cases in order (spec §5). Connect, calib and
    calib_more run on split0 (the pod's copy as the runner's --split), the grid on the split. One line = one runner
    process with its own --out (id): a LIBERO-MAX block, brain, set, source and d, or a bench block and brain. Dicts:
    phase, runner, brain, source, id, block, arms, keys (the expected episodes, key()), args."""
    split = split or maxwrap.load_split(SPLIT)
    s0, brains, out = split0(split), mine(pod), []

    def m(phase, block, brain, sset, arms, d=0, tag=None, cases=None):
        if brain not in brains:
            return
        for src, ids in sources(cases or run.set_ids(split if phase == "grid" else s0, sset)).items():
            args = f"--brain {brain} --source {src} --set {sset} --arms={','.join(arms)} --d {d}"
            if phase != "grid":
                args += f" --split {PD.format(pod_of(brain))}/split0.json"
            if cases:
                args += f" --cases {','.join(ids)}"
            if {"Gauto", "Glead"} & set(map(method, arms)):
                args += f" --kbar {kbar['m', brain]}"
            if ids:
                out.append({"phase": phase, "runner": "c1e4m", "brain": brain, "source": src, "block": block,
                            "arms": arms, "id": f"{tag or block}_{brain}_{sset}_{src}_d{d}", "args": args,
                            "keys": {("m", brain, c, k, a, d, not arms) for c in ids
                                     for k, a in [("base", "none")] + [("dynamic", a) for a in arms]}})

    def b(phase, block, brain, kind, cells, arms, inits):
        if brain not in brains:
            return
        args = f"--brain {brain} --kinds {kind} --cells {','.join(cells)} --methods {','.join(arms)} "
        if inits in ("calib", "more"):  # spec §5: the bench calibration's frozen pairs in order, then the spare ones
            q = maxwrap.load_split(PAIRS)
            eps = [(x["suite"], x["task"], x["init"]) for x in (q["calib"] if inits == "calib" else q["more"][:k])]
            spare = PD.format(pod_of(brain)) + "/bench_more.json"  # a list: the runner reads a frozen file's calib only
            args += f"--pairs-file {PAIRS.relative_to(REPO) if inits == 'calib' else spare}"
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
        b("calib", "Bcal", br, "step", ("A",), ("Gcal",), "calib")
    for t, br in (x.split("/") for x in more):  # spec §5 calib_more: the next evaluation cases / spare pairs in order
        if t == "m":
            m("more", "M0more", br, "eval", ("Gcal",), cases=s0["eval"][:k])
        else:
            b("more", "Bcalmore", br, "step", ("A",), ("Gcal",), "more")
    if kbar is None:
        return out
    # the grid by priority (spec §13): A M1 pi0.5 (+ M2) -> B oracle -> B noisy -> D; B M1 OFT+ -> M4 S -> B oracle ->
    # B noisy -> M4 OFT. M2 shares M1's Base on the first 300 cases: one call (spec §6), its own file id
    m("grid", "M1pi", P, "eval300", ARMS + M2, tag="M1M2")
    m("grid", "M1pi", P, "evalrest", ARMS)
    m("grid", "M1oftplus", OP, "eval", ARMS)
    m("grid", "M4S", S, "eval300", ARMS)
    for blk, cells, arms in (("Boracle", ("A", "A20"), B_ARMS), ("Bnoisy", ("A20",), NOISY)):
        for br in (P, S):
            b("grid", blk, br, "move", cells, arms, range(10))
    m("grid", "M4oft", O, "eval300", ARMS)
    for d in (10, 20):
        m("grid", "D", P, "eval300", D_ARMS, d)
    assert all(ln["brain"] not in (O, OP) or "--d 0" in ln["args"] for ln in out), "OFT runs at d = 0 only (spec §4)"
    return out


def key(r):
    """One episode: LIBERO-MAX (brain, case, kind, arm, d, connect = Base only) or bench (brain, kind, cell, arm, suite,
    task, init). A connect case may also be an evaluation case: connect's Base is its own episode."""
    if "case_id" in r:
        return ("m", r["brain"], r["case_id"], r["kind"], r["arm"], r["d"], not r["conf"]["arms"])
    return ("b", r["brain"], r["kind"], r["cell"], r["arm"], r["suite"], r["task"], r["init"])


def parse_line(line):
    """{(runner, brain): kbar} of one pod's calibration line: "m" (LIBERO-MAX) before "bench:", "b" after."""
    pod = re.match(r"C1-E4 part 2 calib ([AB]): ", line or "")
    if not pod or not line_re(pod[1]).fullmatch(line):
        raise SystemExit(f"not a calibration line: {line!r}")
    head, _, bench = line.partition(" · bench: ")
    return ({("m", b): float(v) for b, v in re.findall(r"kbar_(\w+)=(-?\d+\.\d+)", head)}
            | {("b", b): float(v) for b, v in re.findall(r"kbar_(\w+)=(-?\d+\.\d+)", bench)})


def spec_lines(text):
    """{pod: its journaled calibration line in the spec (the last one)}; the templates of §5 do not match."""
    return {p: hits[-1].group(0) for p in POD if (hits := list(line_re(p).finditer(text)))}


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
Q5 = ("bench, move, oracle, pi05 + smolvla, A + A20, inits 0-9: (Glead - Gauto) >= +3 and lo2.5 > 0", s3.sup(3))
VERDICTS = {"ВНЕШНИЙ ТЕСТ: G-AUTO ПОМОГАЕТ": ("connect", "Q4a", "Q4b", "Q4c"),
            "G-AUTO НЕ ХУЖЕ PPC НА ИХ ТЕСТЕ": ("connect", "Q4d"),
            "УПРЕЖДЕНИЕ НА ПЛАВНОМ ДВИЖЕНИИ": ("Q5",)}


def mrule(name, mi, clu):
    """(the rule's result, {brain: (invalid pairs, pairs with them)})."""
    test, specs, ok = RULES[name]
    res, inv = {"test": test}, {}
    for b, x, y in specs:
        p, k = mprs(mi, clu, b, x, y)
        d = e2.diff_pp(p)
        inv[b] = k, len(p) + k
        res[b] = (e2.rnd(d) or {}) | {"invalid_pairs": k, "pass": None if d is None else bool(ok(d))}
    ps = [res[b]["pass"] for b, _, _ in specs]
    res["pass"] = False if False in ps else None if None in ps else True  # no pairs: not read
    return res, inv


def connect(raw, split, pod=None):
    """spec §9: Base "none" success on the connect set per brain against their Table 2 overall Base, within +-10 pp
    (pi0.5, OFT, SmolVLA; equality passes); OFT+ report only. A brain with > 1% of its 300 Base episodes missing does
    not pass. The pod's brains (both pods' without one). (pass, detail)."""
    ids, out = set(split["connect"]), {}
    for b in mine(pod):
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


def calib(raw, split, pairs, pod):
    """spec §5: the pod's line. kbar per brain = the median of the valid shadow samples of the calibration cases (M0,
    split0's calib in order) with the percentile bootstrap band (maxwrap.kbar_line: 2000 reps, 5-95%); the bench's from
    its frozen 60 pairs (B-cal). Every calibration case needs its record. With N < 45 for a brain the further cases in
    order (split0's eval, the pairs file's more: calib_more's Gcal records) come in up to N = 45 (maxwrap.n_extra);
    extra = the maximum over the pod's brains per runner (the owner refreezes the split with both pods' maximum).
    (line or None, report; short: the brains calib_more runs, "m/oft" or "b/smolvla")."""
    sha, rep = split["sha256"], {}
    m = [r | {"_k": r["case_id"]} for r in raw if "case_id" in r and r["arm"] == "Gcal" and r["kind"] == "dynamic"
         and r["conf"]["split"] == sha]
    bn = [r | {"_k": (r["suite"], r["task"], r["init"])} for r in raw if "case_id" not in r and r["arm"] == "Gcal"
          and r["kind"] == "step" and r["cell"] == "A"]
    for (tag, b), recs, order, nxt in ([(("m", b), [r for r in m if r["brain"] == b], split["calib"], split["eval"])
                                         for b in mine(pod)]
                                        + [(("b", BENCH[pod]), [r for r in bn if r["brain"] == BENCH[pod]], *pairs)]):
        sm, miss = samples(recs, order)
        if miss:
            raise SystemExit(f"!!! {tag} {b}: no Gcal record for {len(miss)} calibration cases, e.g. {miss[:3]}: rerun "
                             "the calib stage (resume runs the missing ones)")
        more, gone = samples(recs, nxt)
        more = more[:nxt.index(gone[0])] if gone else more  # the further cases in order that have a record
        r = {"N_calib": sum(x is not None for x in sm), "of": len(order), "n_extra": 0}
        if r["N_calib"] < N_MIN:
            try:
                r["n_extra"] = maxwrap.n_extra(sm, more)
            except ValueError:
                r["n_extra"] = None
                r["need"] = (f"{N_MIN - r['N_calib'] - sum(x is not None for x in more)} more valid samples after the "
                             f"{len(more)} further cases with a record: calib_more (a larger K), then calib")
        valid = [x for x in sm + more[:r["n_extra"] or 0] if x is not None]
        r["N"] = len(valid)
        if valid:
            k = maxwrap.kbar_line(valid)
            r |= {"kbar": round(k[0], 6), "band": [round(k[1], 6), round(k[2], 6)],
                  "kbar_n": {n: round(float(np.median(valid[:n])), 4) for n in range(10, len(valid) + 1, 10)}}
        rep[f"{tag}/{b}"] = r
    short = [k for k, v in rep.items() if v["N"] < N_MIN]
    rep["extra"] = None if short else {t: max(v["n_extra"] for k, v in rep.items() if k[0] == t) for t in "mb"}
    rep["short"] = short
    if short:
        return None, rep
    f = lambda v: f"{v['kbar']:.3f} [{v['band'][0]:.3f}, {v['band'][1]:.3f}] N={v['N']}"  # noqa: E731
    line = (f"C1-E4 part 2 calib {pod}: " + " · ".join(f"kbar_{b}={f(rep['m/' + b])}" for b in mine(pod))
            + f" · bench: kbar_{BENCH[pod]}={f(rep['b/' + BENCH[pod]])}")
    assert parse_line(line)
    return line, rep


def smoke(raw, pod):
    """The pod smoke's validity on calibration debug cases (never read), per brain of the pod (each must be there):
    invalid Dynamic episodes (prefix_ok false or a replay miss) <= max(1, 2%) per arm (spec §9's pair rule; one in a
    smoke's 20 is a warning to look at before connect, two fail), check (g) ok for
    every Dynamic "none" at d = 0 with an event, no skipped episode, and the share of Base episodes without an event
    (report). (pass, detail)."""
    out, ok = {}, True
    for b in mine(pod):
        rs = list({key(r): r for r in raw if r.get("case_id") and r["brain"] == b
                   and not r.get("skipped_reason")}.values())
        dyn, base = [r for r in rs if r["kind"] == "dynamic"], [r for r in rs if r["kind"] == "base"]
        inv = {a: [sum(invalid(r) for r in dyn if r["arm"] == a), sum(r["arm"] == a for r in dyn)]
               for a in sorted({r["arm"] for r in dyn})}
        ev = [r for r in dyn if r["arm"] == "none" and r["d"] == 0 and r["e"] is not None]
        g = [r["g_check"]["ok"] for r in ev if r["g_check"]]
        out[b] = {"dynamic": len(dyn), "invalid_by_arm": inv, "g_check_ok_of_with_event": [sum(g), len(g), len(ev)],
                  "base": len(base),
                  "no_event_share": round(float(np.mean([r["e"] is None for r in base])), 4) if base else None,
                  "skipped": sum(1 for r in raw if r.get("brain") == b and r.get("skipped_reason"))}
        out[b]["warn_arms"] = [a for a, (k, n) in inv.items() if 0 < k <= max(1, 0.02 * n)]
        ok &= (bool(dyn) and all(k <= max(1, 0.02 * n) + TOL for k, n in inv.values()) and len(g) == len(ev) and all(g)
               and not out[b]["skipped"])
    return ok, out


def read_tempo(path, pod):
    """The last "tempo (s/episode): k=v ..." line of the pod's tempo file (scripts/gpu_c1e4m.sh)."""
    hits = re.findall(r"tempo \(s/episode\): (.*)", Path(path).read_text())
    t = {k: float(v) for k, v in re.findall(r"(\w+)=([0-9.]+)", hits[-1])} if hits else {}
    if missing := [k for k in tkeys(pod) if not t.get(k)]:
        raise SystemExit(f"!!! no tempo for {missing} in {path}: no forecast for pod {pod}")
    return t


def forecast(tempo, pod, rate=RATE, done=frozenset()):
    """spec §13: the pod's hours and dollars per block (one GPU, one runner process at a time) from seconds per episode
    per runner and brain, the total, and what is left: the planned episodes not in done (the recorded ones)."""
    n, left = defaultdict(Counter), Counter()  # block -> tempo key -> episodes; tempo key -> episodes not recorded
    for ln in lines(KB0, pod=pod):
        tk = f"{'lmax' if ln['runner'] == 'c1e4m' else 'bench'}_{ln['brain']}"
        n[ln["block"]][tk] += len(ln["keys"])
        left[tk] += len(ln["keys"] - done)
    hrs = lambda c: sum(tempo[t] * v for t, v in c.items()) / 3600  # noqa: E731
    tot = lambda c: {"episodes": sum(c.values()), "hours": round(hrs(c), 2), "usd": round(rate * hrs(c), 2)}  # noqa
    return {"pod": pod, "usd_per_h": rate, "tempo_s": {k: tempo[k] for k in tkeys(pod)},
            "blocks": {b: tot(c) for b, c in n.items()}, "total": tot(sum(n.values(), Counter())), "left": tot(left)}


def both(fa, fb):
    """Both pods' forecasts run in parallel: dollars add up, the wall clock is the slower pod's."""
    return {"A": fa, "B": fb, "both": {w: {"usd": round(fa[w]["usd"] + fb[w]["usd"], 2),
                                           "wall_hours": max(fa[w]["hours"], fb[w]["hours"])}
                                       for w in ("total", "left")}}


def libero_names():
    """LIBERO's 40 task names (hf-libero, not a LIBERO-plus overlay: maxwrap.base_task checks 10 per suite)."""
    from libero.libero import benchmark
    bd = benchmark.get_benchmark_dict()
    return {s: bd[s]().get_task_names() for s in run.MAX_STEPS}


def summarize(raw, names, split, klines, override=None):
    """The read gates, then the rules, the verdicts and the report (spec §9, §11); raw: both pods' records; klines:
    {pod: the spec's calibration line}, both pods' needed."""
    sha, sha0, skipped, stale = split["sha256"], split0(split)["sha256"], [], 0
    keep = []
    for r in raw:  # a skipped episode (motion blur without Wand) counts as missing; another split's are not read
        if r.get("skipped_reason"):
            skipped.append(key(r))
        elif "case_id" in r and r["conf"]["split"] != (sha0 if r["conf"]["arms"] in ([], ["Gcal"]) else sha):
            stale += 1  # connect and calibration run on split0, the grid on the split
        else:
            keep.append(r)
    succ = defaultdict(set)
    for r in keep:
        succ[key(r)].add(r["success"])
    if conflict := [k for k, v in succ.items() if len(v) > 1]:
        raise SystemExit(f"{len(conflict)} episodes recorded twice with different success, e.g. {conflict[0]}")
    kb = {k: v for p in sorted(klines) for k, v in parse_line(klines[p]).items()}
    plan = lines(KB0 | kb, split)
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
    ev = [r for b in (P, OP) for r in mi.get((b, "dynamic", "none", 0), {}).values() if r["e"] is not None]
    g = [r["g_check"]["ok"] for r in ev if r["g_check"]]  # (g) on every M1 "none" with an event: never vacuous
    blocks = {"M1 pi05": lambda ln, k: ln["block"] == "M1pi" and k[4] not in M2,
              "M1 oftplus": lambda ln, k: ln["block"] == "M1oftplus",
              "B oracle": lambda ln, k: ln["block"] == "Boracle"}
    comp = {}
    for name, f in blocks.items():
        exp = {k for ln in plan for k in ln["keys"] if f(ln, k)}
        miss = len(exp - have)
        comp[name] = {"expected": len(exp), "missing": miss, "pass": miss <= 0.01 * len(exp) + TOL}
    rules, inv = {}, {}
    for name in RULES:  # spec §9: <= 2% invalid pairs per rule and brain
        rules[name], per = mrule(name, mi, clu)
        inv |= {f"{name}/{b}": {"pairs": n, "invalid": k, "pass": k <= 0.02 * n + TOL} for b, (k, n) in per.items()}
    kb_bad = sorted({(rr, b) for r in rs if method(r["arm"]) in ("Gauto", "Glead")
                     for rr, b in [("m" if "case_id" in r else "b", r["brain"])]
                     if (rr, b) not in kb or abs(r["conf"]["kbar"] - kb[rr, b]) > 1e-12})
    gate = {"records": len(raw), "skipped": len(skipped), "other_split": stale, "outside_plan": len(keep) - len(rs),
            "connect": c_det | {"pass": c_ok},
            "g_check_M1": {"ok": sum(g), "of": len(g), "with_event": len(ev), "pass": len(g) == len(ev) and all(g)},
            "complete": comp, "invalid_pairs": inv, "calib_lines": {p: klines.get(p) for p in POD},
            "kbar_mismatch": kb_bad}
    gate["pass"] = bool(c_ok and gate["g_check_M1"]["pass"] and all(v["pass"] for v in comp.values())
                        and all(v["pass"] for v in inv.values()) and set(klines) == set(POD) and not kb_bad)
    res = {"read_gate": gate}
    if not gate["pass"]:
        if not override:
            return res  # spec §9: the rules are not read
        gate["override"] = override  # a deviation recorded in the spec journal before reading
    q5 = e2.diff_pp(bprs(bi, (P, S), ("A", "A20"), "Glead", "Gauto"))
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
    out = {"success": {}, "dyn_minus_base": {}, "base_kept": {}, "vs_none_invalid_pairs": {}, "by": {},
           "connect_by": {}, "delay_D": {}, "M2": {}, "gauto_vs_ppc": {}, "B": {}}
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
            bs = mi.get((k[0], "base", "none", k[3]), {})  # their terms: the share of Base successes kept
            out["base_kept"][name] = mean([r["success"] for c, r in mi[k].items() if c in bs and bs[c]["success"]])
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
                for f, w in (("grasp_miss_along_m", 1000), ("contact_at_shift", 1), ("pushed_before_fire", 1)):
                    x[f"{c}/{a}/{f}" + "_mm" * (w > 1)] = mean([w * r[f] for r in g.values() if r.get(f) is not None])
        out["B"][br] = x
    rows = [x for r in rs if r.get("case_id") and r["brain"] == S for x in r.get("batch_rows") or []]
    out["batch_rows_smolvla"] = {"entries": len(rows), "rows_mean": mean([x[1] for x in rows]),  # [t, rows, texts]
                                 "texts_mean": mean([x[2] for x in rows]),
                                 "mixed_share": mean([x[2] > 1 for x in rows]),
                                 "note": "one entry per episode and call: a batch of n rows counts n times"}
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
    kla = "C1-E4 part 2 calib A: kbar_pi05=0.300 [0.250, 0.350] N=58 · bench: kbar_pi05=0.350 [0.300, 0.400] N=60"
    klb = ("C1-E4 part 2 calib B: kbar_oftplus=0.100 [0.050, 0.150] N=50 · kbar_oft=0.050 [0.000, 0.100] N=47 · "
           "kbar_smolvla=0.040 [0.010, 0.070] N=55 · bench: kbar_smolvla=0.030 [0.000, 0.060] N=59")
    klines = {"A": kla, "B": klb}
    kb = parse_line(kla) | parse_line(klb)
    assert kb[("m", P)] == 0.3 and kb[("b", S)] == 0.03 and set(kb) == {("m", b) for b in CAL} | {("b", P), ("b", S)}
    assert spec_lines(f"x `{kla}`, y `{klb}`.") == klines
    assert spec_lines("`C1-E4 part 2 calib A: kbar_pi05=… [lo, hi] N=… · bench: kbar_pi05=… [lo, hi] N=…`") == {}
    for bad in (kla.replace("calib A", "calib B"), klb.replace("kbar_oft=", "kbar_ofx="), kla + " · x", None):
        try:
            parse_line(bad)
            raise AssertionError(f"accepted {bad!r}")
        except SystemExit:
            pass

    def blocks(pod=None):
        n = Counter()
        for ln in lines(kb, pod=pod):
            n[ln["block"]] += len(ln["keys"])
        return n
    full = {"connect": 1200, "M0": 480, "Bcal": 120, "M1pi": 5600, "M1oftplus": 4700, "M4S": 1500, "Boracle": 4160,
            "M4oft": 1500, "Bnoisy": 1040, "D": 2400}  # M1pi = M1 4700 + M2 900 (one call on the first 300)
    assert blocks() == full and sum(full.values()) == 22700, blocks()
    assert sum(k[4] in M2 for ln in lines(kb) for k in ln["keys"]) == 900
    na, nb = blocks("A"), blocks("B")  # spec §13: nothing is cut, two pods by brain
    assert na + nb == Counter(full) and sum(na.values()) == 11080 and sum(nb.values()) == 11620, (na, nb)
    assert not {ln["brain"] for ln in lines(kb, pod="A")} & {ln["brain"] for ln in lines(kb, pod="B")}
    assert {b for ln in lines(kb) for b in [ln["brain"]]} == set(CAL) and set(POD["A"] + POD["B"]) == set(CAL)
    for p, kl in klines.items():  # a pod's plan needs its own line only
        assert lines(parse_line(kl), pod=p) == lines(kb, pod=p)
    assert {ln["phase"] for ln in lines()} == {"connect", "calib"} and len(lines()) == 18
    assert len(lines(pod="A")) == 5 and len(lines(pod="B")) == 13
    pl = lines(kb)
    assert [ln["phase"] for ln in pl] == sorted((ln["phase"] for ln in pl), key=["connect", "calib", "grid"].index)
    assert len({ln["id"] for ln in pl}) == len(pl)  # one --out per (block, brain, set, source, d)
    for p, want in (("A", ["M1pi", "Boracle", "Bnoisy", "D"]),
                    ("B", ["M1oftplus", "M4S", "Boracle", "Bnoisy", "M4oft"])):
        grid = [ln for ln in lines(kb, pod=p) if ln["phase"] == "grid"]
        assert list(dict.fromkeys(ln["block"] for ln in grid)) == want, grid
    assert [ln["id"] for ln in lines(kb, pod="A") if ln["phase"] == "grid"][:3] == [
        "M1M2_pi05_eval300_plus_d0", "M1M2_pi05_eval300_pro_d0", "M1pi_pi05_evalrest_plus_d0"]
    for ln in pl:  # plus / pro by the split file: the pro process gets exactly the "pro-" cases
        if ln["runner"] == "c1e4m":
            assert f"--source {ln['source']}" in ln["args"] and f"_{ln['source']}_d" in ln["id"]
            assert all(k[2].startswith("pro-") == (ln["source"] == "pro") for k in ln["keys"])
            assert (f"--split {PD.format(pod_of(ln['brain']))}/split0.json" in ln["args"]) == (ln["phase"] != "grid")
        assert ("--kbar" in ln["args"]) == bool({"Gauto", "Glead"} & set(map(method, ln["arms"])))
    conn = [ln for ln in pl if ln["block"] == "connect" and ln["brain"] == P]
    npro = sum(i.startswith("pro-") for i in split["connect"])
    assert [len(ln["keys"]) for ln in conn] == [300 - npro, npro] and npro == 93
    assert "--kbar 0.3" in next(ln["args"] for ln in pl if ln["id"].startswith("M1M2_pi05"))
    assert "--kbar 0.03" in next(ln["args"] for ln in pl if ln["id"] == "Boracle_smolvla")
    assert "--arms= --d 0" in conn[0]["args"]  # Base only: one shell word
    # a split refrozen with extra (spec §5): connect and calibration stay on split0 (M0 never reruns), the grid moves
    order = split["calib"] + split["eval"]
    rf = {k: v for k, v in split.items() if k != "sha256"} | {"extra": 5, "calib": order[:65], "eval": order[65:]}
    rf["sha256"] = maxwrap._sha(rf)
    assert split0(split) == split and split0(rf) == split
    pr = lines(kb, rf)
    assert [ln["id"] for ln in pr if ln["phase"] != "grid"] == [ln["id"] for ln in pl if ln["phase"] != "grid"]
    assert {k[2] for ln in pr if ln["block"] == "M0" for k in ln["keys"]} == set(split["calib"])
    grid = {k[2] for ln in pr if ln["phase"] == "grid" and ln["runner"] == "c1e4m" for k in ln["keys"]}
    assert grid == set(order[65:])
    assert sum(len(ln["keys"]) for ln in pr if ln["block"] == "M1oftplus") == 935 * 5
    # calib_more: Gcal on the next k cases of split0's eval, or the spare bench pairs, for the pod's short brains only
    mo = [ln for ln in lines(pod="B", more=["m/oft", "b/smolvla"], k=20) if ln["phase"] == "more"]
    cm = [ln for ln in mo if ln["runner"] == "c1e4m"]
    assert {k[2] for ln in cm for k in ln["keys"]} == set(split["eval"][:20]) and {ln["brain"] for ln in mo} == {O, S}
    assert all("--set eval " in ln["args"] and "--cases " in ln["args"] and "split0.json" in ln["args"] for ln in cm)
    bm = next(ln for ln in mo if ln["runner"] == "c1e4")
    assert len(bm["keys"]) == 20 and f"--pairs-file {PD.format('B')}/bench_more.json" in bm["args"], bm["args"]
    assert not [ln for ln in lines(pod="A", more=["m/oft", "b/smolvla"]) if ln["phase"] == "more"]
    # forecast: 3.6 s per episode everywhere -> hours = episodes / 1000
    t36 = {k: 3.6 for p in POD for k in tkeys(p)}
    fa, fb = forecast(t36, "A"), forecast(t36, "B")
    assert fa["total"] == {"episodes": 11080, "hours": 11.08, "usd": round(11.08 * RATE, 2)}
    assert fa["left"] == fa["total"]
    assert fb["total"]["hours"] == 11.62 and fa["blocks"]["D"] == {"episodes": 2400, "hours": 2.4, "usd": 0.82}
    assert forecast(t36 | {"bench_pi05": 7.2}, "A")["total"]["hours"] == 13.74  # + 2660 bench pi0.5 episodes
    done = set().union(*(ln["keys"] for ln in lines(KB0, pod="A") if ln["phase"] != "grid"))
    assert forecast(t36, "A", done=done)["left"]["episodes"] == 11080 - 480
    assert both(fa, fb)["both"]["total"] == {"usd": round(fa["total"]["usd"] + fb["total"]["usd"], 2),
                                             "wall_hours": 11.62}
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

    def recs(fx):
        out = []
        for ln in lines(kb):
            for k in sorted(ln["keys"], key=str):
                if k[0] == "m":
                    _, b, c, kind, a, d, _ = k
                    h = sum(map(ord, c))
                    s = suites[h % 4]
                    out.append({"brain": b, "case_id": c, "kind": kind, "arm": a, "method": method(a), "d": d,
                                "conf": {"arms": list(ln["arms"]), "revision": "r", "s": 10,
                                         "split": (split if ln["phase"] == "grid" else split0(split))["sha256"],
                                         "precision": "fp32", "R": 360, "t_ramp": 5, "k_p": 1.0, "tau_k": 0.01,
                                         "kbar": kb.get(("m", b)) if ln["phase"] == "grid" else None},
                                "suite": s, "task_name": f"{s}_t{h // 4 % 10}_v{h % 7}", "substrate_variant": None,
                                "source": "pro" if c.startswith("pro-") else "plus", "distance_m": (0.06, 0.12)[h % 2],
                                "change_type": "target_relocation", "substrate_category": "x", "e": 30,
                                "success": bool(fx(k) > uu(c)), "grasp_ok": True, "prefix_ok": None if kind == "base"
                                else True, "replay_miss": None, "skipped_reason": None,
                                "g_check": {"ok": True} if kind == "dynamic" and a == "none" and d == 0 else None,
                                "calls_sched": 20, "calls_trig": 0, "calls_live": 20, "calls_shadow": 0,
                                "plans_in": [[0, 0], [30, 30 + d]], "t_close": 50, "batch_rows": [[0, 4, 2]],
                                "shadow": {"sample": 0.2} if a == "Gcal" else None})
                else:
                    _, b, kind, cell, a, s, t, i = k
                    out.append({"brain": b, "kind": kind, "cell": cell, "arm": a, "method": method(a), "suite": s,
                                "task": t, "init": i, "conf": {"revision": "r", "precision": "fp32", "t_ramp": 5,
                                                               "k_p": 1.0, "tau_k": 0.01, "kbar": kb[("b", b)]},
                                "success": bool(fx(k) > uu((s, t, i, cell))), "speed": lead.SPEEDS[i % 3],
                                "lead_on_share": 0.5 if method(a) == "Glead" else None, "move_stop_reason": "tau",
                                "grasp_miss_along_m": 0.002, "shadow": {"sample": 0.3} if a == "Gcal" else None})
        return out

    base = {P: 0.797, O: 0.643, S: 0.261, OP: 0.5}
    win = lambda k: (base[k[1]] if k[0] == "m" and k[3] == "base" else  # noqa: E731
                     float(k[4] in ("Gauto",)) if k[0] == "m" else float(k[4] == "Glead"))
    raw = recs(win)
    out = summarize(raw, names, split, klines)
    assert out["read_gate"]["pass"], out["read_gate"]
    assert out["read_gate"]["g_check_M1"] == {"ok": 1880, "of": 1880, "with_event": 1880, "pass": True}
    r = out["rules"]
    assert all(r[n]["pass"] for n in ("Q4a", "Q4b", "Q4c", "Q4d", "Q5")), r
    assert r["Q4a"][P]["pairs"] == 940 and r["Q4c"][OP]["pairs"] == 940 and r["Q5"]["pairs"] == 1040
    assert r["Q4a"][P]["clusters"] == 40 and r["Q5"]["clusters"] == 26
    assert set(out["verdicts"].values()) == {"ДА"}, out["verdicts"]
    assert out["figure"][S]["Gauto"]["pairs"] == 300 and out["figure"][OP]["none"]["pairs"] == 940
    rep = out["report"]
    assert rep["B"][P]["A/0.04"]["Glead-Gauto"]["diff_pp"] == 100
    assert rep["B"][S]["A20/Glead/grasp_miss_along_m_mm"] == 2
    assert rep["base_kept"]["pi05/dynamic/Gauto/0"] == 1 and rep["batch_rows_smolvla"]["mixed_share"] == 1
    assert json.dumps(summarize(raw, names, split, klines), sort_keys=True) == json.dumps(out, sort_keys=True)
    p = [(("s", i % 5), bool(i % 3), bool(i % 2)) for i in range(100)]
    assert e2.diff_pp(p) == e2.diff_pp(list(p))  # bootstrap determinism (fixed seed, fixed cluster order)
    # gates: each failure alone stops the reading
    rand = recs(lambda k: base[k[1]] if k[0] == "m" and k[3] == "base" else 0.5)
    assert summarize(rand, names, split, klines)["read_gate"]["pass"]
    first = lambda x, b: x.get("g_check") and x["brain"] == b and x["case_id"] == split["eval"][5]  # noqa: E731
    fails = {
        "connect": [x | {"success": False} if x.get("case_id") and not x["conf"]["arms"] and x["brain"] == O else x
                    for x in rand],
        "g_check": [x | {"g_check": {"ok": False}} if first(x, OP) else x for x in rand],
        "g_vacuous": [x | {"g_check": None} if first(x, P) else x for x in rand],  # a "none" with an event, no (g)
        "complete": [x for x in rand if not (x["brain"] == OP and x.get("case_id") in split["eval"][:60]
                                             and x["arm"] == "PPC")],
        "invalid": [x | {"prefix_ok": False} if x.get("case_id") in split["eval"][:25] and x["arm"] == "PPC"
                    and x["brain"] == P else x for x in rand],  # 25 of 940 pi0.5 pairs; pooled 25 of 1880 would pass
        "kbar": [x | {"conf": x["conf"] | {"kbar": 0.31}} if x["arm"] == "Gauto" and x["brain"] == P else x
                 for x in rand]}
    for why, rr in fails.items():
        g = summarize(rr, names, split, klines)
        assert not g["read_gate"]["pass"] and "rules" not in g, why
    assert summarize(fails["g_vacuous"], names, split, klines)["read_gate"]["g_check_M1"]["of"] == 1879
    for kl in ({"A": kla}, {}):
        g = summarize(rand, names, split, kl)["read_gate"]
        assert not g["pass"] and g["calib_lines"]["B"] is None and g["kbar_mismatch"], g["kbar_mismatch"]
    g = summarize(fails["invalid"], names, split, klines)["read_gate"]["invalid_pairs"]
    assert g["Q4d/pi05"] == {"pairs": 940, "invalid": 25, "pass": False} and g["Q4d/oftplus"]["pass"], g
    assert g["Q4a/pi05"]["pass"] and set(g) == {"Q4a/pi05", "Q4b/oftplus", "Q4c/pi05", "Q4c/oftplus", "Q4d/pi05",
                                                 "Q4d/oftplus"}
    ov = summarize(fails["connect"], names, split, klines, override="selftest")
    assert all(v.startswith("ОТСТУПЛЕНИЕ") for v in ov["verdicts"].values())
    assert ov["verdicts"]["G-AUTO НЕ ХУЖЕ PPC НА ИХ ТЕСТЕ"].endswith("НЕТ")  # connect failed
    # 48 of the 4700 M1 OFT+ episodes missing (> 1%) fails, 47 (<= 1%) passes
    for n_drop, ok in ((48, False), (47, True)):
        rr = [x for j, x in enumerate(rand) if not (x["brain"] == OP and x["arm"] == "G" and x.get("d") == 0
                                                    and x.get("case_id") in split["eval"][:n_drop])]
        assert summarize(rr, names, split, klines)["read_gate"]["complete"]["M1 oftplus"]["pass"] == ok, n_drop
    # the connection gate at its edges: 300 cases, 69.7 = 79.7 - 10; per pod its own brains
    c_ids = split["connect"]

    def conn(n):  # Base successes of the 300 per brain
        return [{"brain": b, "case_id": c, "kind": "base", "arm": "none", "conf": {"arms": []}, "success": j < n[b]}
                for b in (P, O, S, OP) for j, c in enumerate(c_ids)]
    good = {P: 210, O: 164, S: 78, OP: 0}  # 70.0, 54.67, 26.0 %: 79.7 - 10 = 69.7, 64.3 - 10 = 54.3
    ok, det = connect(conn(good), split)
    assert ok and det[OP]["pass"] is None and det[P]["base_none"] == 70.0, det
    assert not connect(conn(good | {P: 209}), split)[0] and not connect(conn(good | {S: 109}), split)[0]  # 69.67, 36.33
    assert connect(conn(good | {P: 209}), split, "B")[0] and not connect(conn(good | {P: 209}), split, "A")[0]
    assert set(connect(conn(good), split, "B")[1]) == {OP, O, S}
    assert not connect([x for x in conn(good) if x["brain"] != O], split, "B")[0]  # a brain without records
    TABLE2[P] = 80.0  # equality passes: 70.0 = 80.0 - 10
    try:
        assert connect(conn(good), split)[0]
    finally:
        TABLE2[P] = 79.7
    assert not connect([x for x in conn(good) if x["case_id"] not in c_ids[:4]], split)[0]  # 4 of 300 missing > 1%
    # calibration: the pods' lines, their format, the N < 45 path and n_extra over the further cases (calib_more)
    cal = recs(lambda k: 0.5)
    pairs = [[(q["suite"], q["task"], q["init"]) for q in maxwrap.load_split(PAIRS)[w]] for w in ("calib", "more")]
    la, _ = calib(cal, split, pairs, "A")
    lb, rep = calib(cal, split, pairs, "B")
    assert la == ("C1-E4 part 2 calib A: kbar_pi05=0.200 [0.200, 0.200] N=60 · bench: kbar_pi05=0.300 [0.300, 0.300] "
                  "N=60"), la
    assert lb == ("C1-E4 part 2 calib B: kbar_oftplus=0.200 [0.200, 0.200] N=60 · kbar_oft=0.200 [0.200, 0.200] N=60 · "
                  "kbar_smolvla=0.200 [0.200, 0.200] N=60 · bench: kbar_smolvla=0.300 [0.300, 0.300] N=60"), lb
    assert rep["extra"] == {"m": 0, "b": 0} and rep["short"] == [] and spec_lines(f"{la}\n{lb}") == {"A": la, "B": lb}
    short = [x | {"shadow": {"sample": None}} if x.get("case_id") in split["calib"][:20] and x["brain"] == O
             and x["arm"] == "Gcal" else x for x in cal]
    line, rep = calib(short, split, pairs, "B")
    assert line is None and rep["m/oft"]["N"] == 40 and rep["m/oft"]["n_extra"] is None and rep["extra"] is None
    assert rep["short"] == ["m/oft"] and calib(short, split, pairs, "A")[0] == la  # pod A has no OFT
    tmpl = next(x for x in cal if x.get("case_id") and x["arm"] == "Gcal")  # a valid sample on the next 7 cases
    more = [tmpl | {"case_id": c, "brain": b} for c in split["eval"][:7] for b in CAL]
    line, rep = calib(short + more, split, pairs, "B")
    assert rep["m/oft"]["n_extra"] == 5 and rep["m/oft"]["N"] == 45 and rep["extra"] == {"m": 5, "b": 0}, rep
    assert line == lb.replace("kbar_oft=0.200 [0.200, 0.200] N=60", "kbar_oft=0.200 [0.200, 0.200] N=45"), line
    try:
        calib([x for x in cal if x.get("case_id") != split["calib"][3]], split, pairs, "A")
        raise AssertionError("a calibration case without its record was accepted")
    except SystemExit:
        pass
    # smoke validity per arm: 1 invalid of 50 passes, 2 of 50 do not; a failed or missing check (g) does not; every
    # brain of the pod must be there
    sm = [{"brain": P, "case_id": f"c{j}", "kind": "dynamic", "arm": a, "method": a, "d": 0, "e": 30,
           "prefix_ok": j != 0, "replay_miss": None, "g_check": {"ok": True} if a == "none" else None,
           "conf": {"arms": ["none", "G"]}} for a in ("none", "G") for j in range(50)]
    sm += [{"brain": P, "case_id": f"c{j}", "kind": "base", "arm": "none", "method": "none", "d": 0,
            "e": None if j < 5 else 30, "conf": {"arms": ["none", "G"]}} for j in range(50)]
    ok, det = smoke(sm, "A")
    assert ok and det[P]["no_event_share"] == 0.1 and det[P]["invalid_by_arm"] == {"G": [1, 50], "none": [1, 50]}, det
    assert not smoke([x | {"replay_miss": 12} if x["case_id"] == "c1" and x["arm"] == "G" and x["kind"] == "dynamic"
                      else x for x in sm], "A")[0]
    for g in ({"ok": False}, None):
        assert not smoke([x | {"g_check": g} if x["case_id"] == "c9" and x["arm"] == "none" and x["kind"] == "dynamic"
                          else x for x in sm], "A")[0]
    assert not smoke(sm, "B")[0] and not smoke([], "A")[0]
    s20 = [x for x in sm if int(x["case_id"][1:]) < 20]  # the pod's 20 cases per arm: 1 invalid warns, 2 fail
    ok, det = smoke(s20, "A")
    assert ok and det[P]["warn_arms"] == ["G", "none"], det
    assert not smoke([x | {"prefix_ok": False} if x["case_id"] == "c1" and x["arm"] == "G" else x for x in s20], "A")[0]
    print("selftest ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("grid", nargs="*", help="JSON lines: both pods' grid_all_A.jsonl grid_all_B.jsonl (with "
                                            "--forecast: the pod's records, for the hours left; put them first)")
    ap.add_argument("--override-gate", help="the spec journal entry that records the owner's deviation")
    ap.add_argument("--pod", choices=sorted(POD), help="A (pi0.5) or B (OFT+, OFT, SmolVLA), spec §13")
    ap.add_argument("--plan", action="store_true", help='the pod\'s runner lines: "phase<TAB>runner<TAB>brain<TAB>'
                                                        'source<TAB>id<TAB>n<TAB>args"')
    ap.add_argument("--kbar-line", help="with --plan: the pod's journaled calibration line (the grid's kbar)")
    ap.add_argument("--more", default="", help='with --plan: calib_more\'s brains, calib\'s short ("m/oft b/smolvla")')
    ap.add_argument("--k", type=int, default=20, help="with --more: the next k cases (bench: spare pairs) in order")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--connect", nargs="+", metavar="JSONL", help="connect records: exit 0 if it passes, else 3")
    mode.add_argument("--calib", nargs="+", metavar="JSONL", help="M0, B-cal and calib_more records: the line, or 4")
    mode.add_argument("--smoke", nargs="+", metavar="JSONL", help="smoke LIBERO-MAX records: exit 0, or 3")
    mode.add_argument("--forecast", nargs="+", metavar="TEMPO", help="the pod's tempo file, or pod A's and pod B's")
    mode.add_argument("--split0", metavar="OUT", help="write split0 (connect, calib, calib_more); print its sha256")
    ap.add_argument("--rate", type=float, default=RATE, help="$/h")
    ap.add_argument("--fig", help="with the grid: the figure next to their Table 12 (PNG)")
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    load = lambda fs: [x for f in fs for x in e2.load(f)]  # noqa: E731
    if (a.calib or a.smoke or a.kbar_line or a.more or a.forecast and len(a.forecast) == 1) and not a.pod:
        ap.error("--pod A or B is required here")
    if a.kbar_line and not a.kbar_line.startswith(f"C1-E4 part 2 calib {a.pod}: "):
        ap.error(f"--kbar-line is not pod {a.pod}'s calibration line")
    if a.selftest:
        selftest()
    elif a.plan:
        for ln in lines(parse_line(a.kbar_line) if a.kbar_line else None, pod=a.pod, more=a.more.split(), k=a.k):
            print("\t".join(map(str, (ln["phase"], ln["runner"], ln["brain"], ln["source"], ln["id"], len(ln["keys"]),
                                      ln["args"]))))
    elif a.connect:
        ok, det = connect(load(a.connect), maxwrap.load_split(SPLIT), a.pod)
        print(connect_table(det))
        print(f"C1-E4 part 2 connect{' ' + a.pod if a.pod else ''}: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    elif a.calib:
        pf = maxwrap.load_split(PAIRS)
        line, rep = calib(load(a.calib), split0(maxwrap.load_split(SPLIT)),
                          [[(q["suite"], q["task"], q["init"]) for q in pf[w]] for w in ("calib", "more")], a.pod)
        print(e2.dump(rep))
        if line is None:
            print(f"!!! N < {N_MIN} for {rep['short']} (see need): calib_more runs Gcal on the next cases in order, "
                  "then --calib again (spec §5)")
            print(f"C1-E4 part 2 short {a.pod}: {' '.join(rep['short'])}")
            raise SystemExit(4)
        print(line)
        if any(rep["extra"].values()):
            print(f"C1-E4 part 2 extra {a.pod}: m={rep['extra']['m']} b={rep['extra']['b']}")
            print("!!! the line used further cases in order: on the Mac the owner refreezes the split with extra = the "
                  "maximum of both pods' m (maxwrap.freeze_split; the bench's pairs are never evaluated) and journals "
                  "it with the line before either pod's grid (spec §5)")
    elif a.smoke:
        ok, det = smoke(load(a.smoke), a.pod)
        print(e2.dump(det))
        print(f"C1-E4 part 2 smoke validity {a.pod}: {'PASS' if ok else 'FAIL'}")
        raise SystemExit(0 if ok else 3)
    elif a.forecast:
        done = {key(r) for r in load(a.grid)}
        pods = [a.pod] if len(a.forecast) == 1 else "AB"  # two files: pod A's, then pod B's
        fs = [forecast(read_tempo(t, p), p, a.rate, done) for t, p in zip(a.forecast, pods)]
        print(e2.dump(fs[0] if len(fs) == 1 else both(*fs)))
    elif a.split0:
        s0 = split0(maxwrap.load_split(SPLIT))
        Path(a.split0).write_text(json.dumps(s0, indent=1) + "\n")
        print(maxwrap.load_split(a.split0)["sha256"])
    else:
        if not a.grid:
            ap.error("the grid JSON lines are required")
        res = summarize(load(a.grid), libero_names(), maxwrap.load_split(SPLIT), spec_lines(SPEC.read_text()),
                        a.override_gate)
        out = Path(os.path.commonpath([str(Path(f).resolve().parent) for f in a.grid])) / "summary.json"
        out.write_text(e2.dump(res) + "\n")
        print(e2.dump(res))
        if a.fig and "figure" in res:
            figure(a.fig, res["figure"])
