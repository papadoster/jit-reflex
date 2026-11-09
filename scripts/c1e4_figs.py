"""C1-E4 figures from committed results only (no pod, no new episodes): four panels and, next to the PNG, a JSON with
every drawn value and its source file.

  a. the delay axis: success vs d for Base, "none", PPC, G-auto (pi0.5, part 2's pod A, the first 300 evaluation cases,
     the same cases at every d; all records of those lines, no pair selection);
  b. the controls at d = 10 as paired differences X - G-auto with their 95% cluster-bootstrap intervals, copied from
     the committed summaries the memos quote (a difference no memo holds is not drawn and not computed: G-auto - PPC
     at d = 10 is absent, so PPC is not on the panel). "none": part 2's block D (pod A, both arms there); PPCL, PBVS
     (26 mm gate, Н2), VS-T (Т1): pod C, Plus + PRO; VT-m (B main): pod C, Plus only. The memos' G-auto - X are shown
     as X - G-auto: the paired percentile interval of a sign-flipped difference is the flipped interval;
  c. the placebo: G / P / N / N' success at d = 0 and 10 and P - N in the three phi bands with their 95% (summary_P.json);
  d. the gap Base - G-auto at d = 20 split into after the grasp, at the grasp (the grasp gap itself and the ungrasped
     successes apart) (grasp_split_d20.json).
Success shares in a and c are plain means over a line's records (no pair selection, no intervals); the memos'
differences are paired and filtered. The script checks its numbers against the
committed summaries where those hold the same quantity and stops on a mismatch.

Run from the repository root: python scripts/c1e4_figs.py [--out results/c1-e4/figs/c1e4_figs.png]
"""

import argparse
import glob
import json
from pathlib import Path

A, C = "results/c1-e4/part2/pod_A", "results/c1-e4/controls/pod_C"
P_SUM, SPLIT, P2_SUM = ("results/c1-e4/placebo/pod_P/summary_P.json", "results/c1-e4/part2/grasp_split_d20.json",
                        "results/c1-e4/part2/summary.json")


CS = "results/c1-e4/controls/summary_{}.json"
CONTROLS = (  # name, summary, key path, flip (the summary holds G-auto - X), what it is
    ("нет", P2_SUM, ("report", "delay_D", "d10", "Gauto-none"), True, "часть 2, блок D, под A"),
    ("PPCL", CS.format("C"), ("report", "gauto_ppcl", "pi05/d10"), True, "под C, Plus + PRO"),
    ("PBVS", CS.format("V"), ("measurements", "Н2/d10"), False, "Н2, зона 26 мм, под C, Plus + PRO"),
    ("VS-T", CS.format("T"), ("measurements", "Т1/d10"), False, "Т1, под C, Plus + PRO"),
    ("VT-m", CS.format("M"), ("measurements", "main/d10"), False, "B, главное, под C, только Plus"))


def share(files, kind, arm):
    """success % and n over the records of `kind` / `arm` in `files` (one record per case and seed)."""
    rs = {}
    for f in files:
        for ln in open(f):
            if ln.strip():
                r = json.loads(ln)
                if r["kind"] == kind and r["arm"] == arm and r.get("skipped_reason") is None:
                    rs[(r["case_id"], r.get("seed"))] = bool(r["success"])
    assert rs, (files, kind, arm)
    return {"pct": round(100 * sum(rs.values()) / len(rs), 2), "n": len(rs), "files": sorted(files)}


def values():
    v = {"delay": {}, "controls_d10": {}, "placebo": {}, "split": {}}
    for d in (0, 10, 20):
        fs = glob.glob(f"{A}/M1M2_pi05_eval300_*_d0.jsonl" if d == 0 else f"{A}/D_pi05_eval300_*_d{d}.jsonl")
        v["delay"][d] = {"Base": share(fs, "base", "none"),
                         **{a: share(fs, "dynamic", a) for a in ("none", "PPC", "Gauto")}}
    for name, src, path, flip, note in CONTROLS:
        x = json.load(open(src))
        for k in path:
            x = x[k]
        e = {"diff_pp": x["diff_pp"], "lo2.5": x["lo2.5"], "hi97.5": x["hi97.5"]}
        if flip:  # the memo's G-auto - X as X - G-auto
            e = {"diff_pp": -e["diff_pp"], "lo2.5": -e["hi97.5"], "hi97.5": -e["lo2.5"]}
        v["controls_d10"][name] = {**{k: round(y, 3) for k, y in e.items()}, "pairs": x["pairs"], "source": src,
                                   "key": "/".join(path), "sign_flipped": flip, "note": note}
    ps = json.load(open(P_SUM))
    names = {"Gauto": "G", "GautoSham": "P", "none": "N", "none2": "N′"}
    v["placebo"] = {"success": {c: {names[a]: {"pct": round(100 * x[0], 2), "n": x[1]} for a, x in arms.items()}
                                for c, arms in ps["report"]["success"].items()},
                    "bands_P-N": ps["angle"]["bands_P-N"], "source": P_SUM}
    g = json.load(open(SPLIT))
    a, sp = g["arms"], g["split_base_minus_gauto"]
    cB = a["Base"]["grasped_successes"] / a["Base"]["grasps"]
    n = a["Base"]["n"]
    grasp_gap = 100 * (a["Base"]["grasps"] - a["Gauto"]["grasps"]) / n * cB
    ungrasped = 100 * (a["Base"]["ungrasped_successes"] - a["Gauto"]["ungrasped_successes"]) / n
    v["split"] = {"s_Gauto": a["Gauto"]["s"]["est"], "after_grasp": sp["after_grasp"]["est"],
                  "at_grasp": sp["at_grasp"]["est"], "at_grasp_grasps": round(grasp_gap, 3),
                  "at_grasp_ungrasped": round(ungrasped, 3), "s_Base": a["Base"]["s"]["est"], "gap": sp["gap"]["est"],
                  "ci95": {k: sp[k]["ci95"] for k in ("gap", "after_grasp", "at_grasp")}, "source": SPLIT}
    return v


def check(v):
    """the drawn values against the committed summaries that hold the same quantity."""
    rep = json.load(open(P2_SUM))["report"]["success"]
    for d in (10, 20):  # part 2's report holds d = 10, 20 on these very 300 cases
        for a, k in (("Base", "base/none"), ("none", "dynamic/none"), ("PPC", "dynamic/PPC"), ("Gauto", "dynamic/Gauto")):
            assert abs(v["delay"][d][a]["pct"] - 100 * rep[f"pi05/{k}/{d}"]["success"]) < 0.01, (d, a)
    s = v["split"]
    assert abs(s["s_Gauto"] + s["after_grasp"] + s["at_grasp"] - s["s_Base"]) < 0.01
    assert abs(s["at_grasp_grasps"] + s["at_grasp_ungrasped"] - s["at_grasp"]) < 0.01
    memo = {"нет": (-19.0, -24.7, -13.1), "PPCL": (-11.3, -17.7, -5.4), "PBVS": (-4.7, -11.3, 2.0),
            "VS-T": (-3.3, -8.4, 1.7), "VT-m": (-4.3, -10.2, 1.2)}  # as the memos print them (part 2 §3, controls)
    for k, m in memo.items():
        e = v["controls_d10"][k]
        assert all(abs(round(e[f], 1) - y) < 1e-9 for f, y in zip(("diff_pp", "lo2.5", "hi97.5"), m)), (k, e, m)
    assert all(v["delay"][d][a]["n"] == 300 for d in v["delay"] for a in v["delay"][d])


def draw(v, path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    f, ax = plt.subplot_mosaic([["a", "a", "b", "b"], ["c", "c", "c2", "d"]], figsize=(14, 9.5),
                               width_ratios=[1, 0.6, 0.75, 1.25])
    a = ax["a"]  # the delay axis
    ds = [0, 10, 20]
    for arm, lab, col in (("Base", "Base (без события)", "0.4"), ("none", "нет", "tab:red"), ("PPC", "PPC", "tab:orange"),
                          ("Gauto", "G-auto", "tab:blue")):
        a.plot(ds, [v["delay"][d][arm]["pct"] for d in ds], "o-", color=col, label=lab)
    a.set(xticks=ds, xlabel="задержка d, шагов 20 Гц", ylabel="успех, %", title="a. Ось задержки (π0.5, 300 случаев)")
    a.legend(fontsize=8)
    a = ax["b"]  # the controls at d = 10, paired differences
    cs = v["controls_d10"]
    names = list(cs)
    for i, k in enumerate(names):
        e = cs[k]
        a.errorbar(i, e["diff_pp"], yerr=[[e["diff_pp"] - e["lo2.5"]], [e["hi97.5"] - e["diff_pp"]]], fmt="o",
                   color="tab:red" if k == "нет" else ("tab:orange" if k == "PPCL" else "tab:green"), capsize=4)
        a.text(i + 0.08, e["diff_pp"], f"{e['diff_pp']:+.1f}", fontsize=8, va="center")
    a.axhline(0, color="0.3", lw=1)
    a.set_xticks(range(len(names)), [f"{k}\n{cs[k]['pairs']} пар" for k in names], fontsize=8)
    a.set(xlim=(-0.5, len(names) - 0.3), ylabel="X − G-auto, п.п. [95%]",
          title="b. Контроли при d = 10: разность с G-auto, парно")
    a = ax["c"]  # the placebo
    arms = ["G", "P", "N", "N′"]
    for j, (c, dx) in enumerate((("P0", -0.2), ("P10", 0.2))):
        ys = [v["placebo"]["success"][c][k]["pct"] for k in arms]
        a.bar([i + dx for i in range(4)], ys, width=0.38, label=f"d = {0 if c == 'P0' else 10}",
              color=["tab:blue", "tab:purple", "tab:red", "salmon"], alpha=1 if j == 0 else 0.55)
    a.set_xticks(range(4), ["G (G-auto)", "P (плацебо)", "N (нет)", "N′ (нет, др. зерно)"], fontsize=8)
    a.set(ylabel="успех, %", title="c. Плацебо: успех ветвей")
    a.legend(fontsize=8, loc="upper right")
    b = ax["c2"]
    bands = v["placebo"]["bands_P-N"]
    for i, (k, x) in enumerate(bands.items()):
        b.errorbar(i, x["diff_pp"], yerr=[[x["diff_pp"] - x["lo2.5"]], [x["hi97.5"] - x["diff_pp"]]], fmt="o",
                   color="tab:purple", capsize=3)
    b.axhline(0, color="0.7", lw=0.8)
    b.set_xticks(range(3), ["0–60°", "60–120°", "120–180°"], fontsize=8)
    b.set(xlim=(-0.5, 2.5), ylabel="P − N, п.п. [95%]",
          title="P − N по углу поворота φ\n(d = 0 и 10 вместе)")
    a = ax["d"]  # the split
    s = v["split"]
    parts = [("успех G-auto", s["s_Gauto"], "tab:blue"), ("потеря после захвата", s["after_grasp"], "tab:cyan"),
             ("на захвате: сами захваты", s["at_grasp_grasps"], "tab:gray"),
             ("на захвате: успехи без захвата", s["at_grasp_ungrasped"], "0.8")]
    bottom = 0
    for lab, h, col in parts:
        a.bar(0, h, bottom=bottom, color=col, width=0.5, label=f"{lab}: {h:.1f}")
        bottom += h
    a.bar(1, s["s_Base"], color="0.4", width=0.5, label=f"успех Base: {s['s_Base']:.1f}")
    a.set_xticks([0, 1], ["G-auto + разрыв", "Base"])
    a.set(xlim=(-0.6, 3.6), ylabel="успех, %",
          title=f"d. Разрыв Base − G-auto при d = 20: {s['gap']:.1f} п.п.")
    a.legend(fontsize=7, loc="center right")
    f.text(0.01, 0.005, "Только закоммиченные результаты; значения и файлы — рядом в JSON. a, c: доли по всем записям "
           "строк, без отбора пар.\nb: парные разности и 95% кластерного бутстрэпа из сводок записок; «нет» — часть 2 (под A), "
           "остальные — под C; Plus + PRO, VT-m — только Plus; PPC нет: разности G-auto − PPC при d = 10 в записках нет. "
           "PBVS — зона 26 мм.\n"
           "c: 95% кластерный бутстрэп из summary_P.json. d: измерение, не правило (grasp_split_d20.json).",
           fontsize=7, va="bottom")
    f.tight_layout(rect=(0, 0.045, 1, 1))
    f.savefig(path, dpi=150)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", default="results/c1-e4/figs/c1e4_figs.png")
    out = Path(ap.parse_args().out)
    v = values()
    check(v)
    out.parent.mkdir(parents=True, exist_ok=True)
    draw(v, out)
    out.with_suffix(".json").write_text(json.dumps(v, ensure_ascii=False, indent=1) + "\n")
    print(f"wrote {out} and {out.with_suffix('.json')}")


if __name__ == "__main__":
    main()
