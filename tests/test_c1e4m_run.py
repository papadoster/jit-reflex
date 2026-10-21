"""scripts/c1e4m_run.py: Base then Dynamic with the recorded prefix, the event step, check (g), G's engagement and
Gcal's shadow pair on a toy env (no LIBERO, no brain); a replay miss, no event, success mid-episode, strict JSON, a
failed video; main()'s checks before the brain loads; resume; the pure helpers; and, when the LIBERO-plus env is set
up (env vars of the runner's docstring), per-env np.random in --vector sync."""

import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import c1e4m_run as run  # noqa: E402
import objreflex as orx  # noqa: E402

DELTA = np.array([0.0, -0.06, 0.0])  # along the approach: PPC re-queries (alpha > 1)
CASE = {"case_id": "toy-c0", "policy_seed": 195, "task_suite_name": "toy", "task_name": "toy_task", "task_index": 0,
        "init_state_index": 0, "substrate_category": "toy",
        "scenario": {"change_type": "target_relocation", "change": {"distance_m": 0.06, "direction_xy": [0.0, -1.0]},
                     "trigger": {"type": "on_proximity", "value": "obj", "distance_m": 0.18}}}


def conf(d=0):
    return {"brain": "smolvla", "revision": "x", "precision": "fp32", "s": 10, "d": d, "kbar": 0.038, "t_ramp": 5,
            "k_p": 1.0, "tau_k": 0.01, "R": 360, "arms": [], "split": "x"}


class Toy(run.LocalEnv):
    """EEF integrator x += G_POS a, a still object. As LIBERO-MAX: the first step() after which |EEF - object| <= R_TRIG
    sets the trigger step (policy step after the step, both arms); "intervention" moves the object there, so the
    observation it returns is post-event and e = that policy step. DONE: the step from which it reports success."""

    R_TRIG, DONE = 0.18, None

    def __init__(self):
        super().__init__(0)

    def start(self, case, arm, s):
        self.arm, self.t, self.e, self.trig = arm, 0, None, None
        self.x, self.p = np.array([0.5, 0.35, 0.1]), np.array([0.5, 0.0, 0.0])
        return self._pkg()

    def step(self, a):
        self.x = self.x + orx.G_POS * np.asarray(a)[:3]
        self.t += 1
        if self.trig is None and np.linalg.norm(self.x - self.p) <= self.R_TRIG:
            self.trig = self.t
            if self.arm == "intervention":
                self.p, self.e = self.p + DELTA, self.t
        return self._pkg()

    def _pkg(self):
        obs = {"robot0_eef_pos": self.x.copy(), "obj": self.p.copy(), "arm": self.arm,
               "agentview_image": np.zeros((4, 4, 3), np.uint8)}
        return {"obs": obs, "mat": np.eye(3), "done": self.DONE is not None and self.t >= self.DONE,
                "obj": self.p.copy(), "e": self.e, "trig": self.trig, "t": self.t, "instr": "pick it"}

    def close(self):
        pass


class Brain:
    """A plan toward the object it sees (2 cm above), a third of the way per 10 steps (unclipped near the object),
    constant over the chunk, plus seed noise; logs (arm, seed)."""

    def __init__(self):
        self.log = []

    def __call__(self, obs, envs, instr, seeds, suites):
        out = []
        for o, sd in zip(obs, seeds):
            self.log.append((o["arm"], int(sd)))
            ch = np.zeros((orx.H, 7))
            ch[:, :3] = np.clip((o["obj"] + [0, 0, 0.02] - o["robot0_eef_pos"]) / orx.G_POS / 30, -1, 1)
            ch[:, :3] += 1e-3 * np.random.default_rng(int(sd)).normal(size=3)
            ch[:, 6] = -1.0
            out.append(ch)
        return np.stack(out)


@pytest.fixture(autouse=True)
def toy_suite(monkeypatch):
    monkeypatch.setitem(run.MAX_STEPS, "toy", 60)


def by_arm(recs):
    base = next(r for r in recs if r["kind"] == "base")
    return base, {r["arm"]: r for r in recs if r["kind"] == "dynamic"}


def test_base_then_dynamic_prefix_event_g_shadow():
    brain = Brain()
    recs = run.run_cases([CASE], ["none", "G", "Gauto", "PPC", "Gcal"], conf(), [Toy(), Toy()], brain)
    assert [r["kind"] for r in recs] == ["dynamic"] * 5 + ["base"]  # Base last: resume reads it as the case's end
    base, dyn = by_arm(recs)
    e = base["e"]
    assert e is not None and 0 < e < 60 and base["prefix_ok"] is None and base["g_check"] is None
    for arm, r in dyn.items():
        assert r["e"] == e and r["prefix_ok"] is True and r["replay_miss"] is None, arm
        assert r["steps"] == 60 and r["calls_live"] < r["calls_sched"] + r["calls_trig"], arm
    assert dyn["none"]["g_check"] == {"ok": True, "last": 10 * ((e - 1) // 10), "first": 10 * -(-e // 10)}
    assert dyn["none"]["act_hash"] != base["act_hash"]  # live plans after the event see the moved object
    for arm in ("G", "Gauto", "Gcal"):
        assert dyn[arm]["t_engage"] == e and dyn[arm]["g_on"], arm
    assert dyn["PPC"]["calls_trig"] >= 1 and min(t for t, _ in dyn["PPC"]["plans_in"] if t % 10) >= e
    sh = dyn["Gcal"]["shadow"]
    assert dyn["Gcal"]["calls_shadow"] == 2 and (sh["t0"], sh["t1"]) == (e - 1, e)
    assert np.allclose(sh["d"], DELTA) and sh["sample"] > 0.3  # the fresh plan follows the move (cache: step e - 1)
    assert all(r["shadow"] is None and r["calls_shadow"] == 0 for a, r in dyn.items() if a != "Gcal")
    # the brain is never called on a Dynamic observation before the event (the prefix is Base's chunks); seeds = 195 + t
    assert all(sd >= 195 + e for arm, sd in brain.log if arm == "intervention")
    assert sorted({sd for arm, sd in brain.log if arm == "control"}) == [195 + 10 * k for k in range(6)]
    assert all(b[0] >= e and b[2] == 1 for r in dyn.values() for b in r["batch_rows"]) and base["batch_rows"]
    # strict JSON: the records encode (no NaN, no numpy int / bool); a NaN raises
    assert [json.loads(x) for x in run.lines(recs).splitlines()] == json.loads(json.dumps(recs))
    with pytest.raises(ValueError):
        run.lines([{"x": float("nan")}])


def test_replay_miss_is_flagged():
    """A pre-event call Base did not make (T0 on an object pushed 2 cm at step 3) goes live and is flagged."""

    class Pushed(Toy):
        def step(self, a):
            pkg = super().step(a)
            if self.t == 3:
                self.p = self.p + [0.0, -0.02, 0.0]
                pkg = self._pkg()
            return pkg

    _, dyn = by_arm(run.run_cases([CASE], ["none", "T0"], conf(), [Pushed(), Pushed()], Brain()))
    assert dyn["none"]["replay_miss"] is None and dyn["none"]["prefix_ok"] is True
    assert dyn["T0"]["replay_miss"] > 0 and dyn["T0"]["prefix_ok"] is False and dyn["T0"]["steps"] == 60


def test_no_event_is_a_full_replay():
    class Far(Toy):
        R_TRIG = 0.0

    base, dyn = by_arm(run.run_cases([CASE], ["none", "G", "PPC"], conf(), [Far(), Far()], Brain()))
    assert base["e"] is None and base["steps"] == 60
    for arm, r in dyn.items():
        assert r["e"] is None and r["prefix_ok"] is True and r["replay_miss"] is None, arm
        assert r["calls_live"] == 0 and r["act_hash"] == base["act_hash"] and r["g_check"] is None, arm


def test_done_mid_episode():
    class Done(Toy):
        DONE = 40

    base, dyn = by_arm(run.run_cases([CASE], ["none", "G"], conf(), [Done(), Done()], Brain()))
    assert base["e"] < 40
    assert all(r["success"] is True and r["steps"] == 40 for r in [base, *dyn.values()])
    assert all(r["prefix_ok"] is True for r in dyn.values())


def test_base_prefix_stops_the_base_at_the_event():
    """--base-prefix (controls spec §7): Base ends at its event, the Dynamic arms are those of the run without it."""
    arms, key = ["none", "G", "Gauto", "PPC", "Gcal"], ("act_hash", "success", "steps")
    full = run.run_cases([CASE], arms, conf(), [Toy(), Toy()], Brain())
    brain = Brain()
    recs = run.run_cases([CASE], arms, conf(), [Toy(), Toy()], brain, base_prefix=True)
    (base, dyn), (base0, dyn0) = by_arm(recs), by_arm(full)
    e = base["e"]
    assert e == base0["e"] and e is not None and base0["base_cut"] is False
    assert base["base_cut"] is True and base["success"] is None and base["grasp_ok"] is None and base["steps"] == e
    # the Base's chunk keys are its calls' steps (seed 195 + t): all before e
    assert all(sd < 195 + e for arm, sd in brain.log if arm == "control") and all(t < e for t, *_ in base["batch_rows"])
    for arm, r in dyn.items():
        assert r["prefix_ok"] is True and r["replay_miss"] is None, arm
        assert [r[k] for k in key] == [dyn0[arm][k] for k in key], arm
    assert [json.loads(x) for x in run.lines(recs).splitlines()] == json.loads(json.dumps(recs))


def test_base_prefix_keeps_the_base_call_batches():
    """A batch's Bases are cut together at its last event: until then every Base call has the rows of the run without
    the flag (part 2's, spec §7), so a brain whose chunk depends on the batch size gives the same prefix."""

    class Two(Toy):
        def start(self, case, arm, s):
            self.R_TRIG = 0.18 if case["case_id"] == "toy-c0" else 0.08  # two events far apart
            return super().start(case, arm, s)

    class Batchy(Brain):
        def __call__(self, obs, *a):
            out = super().__call__(obs, *a)
            out[:, :, :3] += 1e-4 * len(obs)
            return out

    cases, key = [CASE, {**CASE, "case_id": "toy-c1"}], ("act_hash", "success", "steps", "e", "prefix_ok")
    full, cut = (run.run_cases(cases, ["none", "G"], conf(), [Two(), Two()], Batchy(), base_prefix=f)
                 for f in (False, True))
    bases = [r for r in cut if r["kind"] == "base"]
    e = [r["e"] for r in bases]
    assert e[0] + 10 < e[1] and all(r["base_cut"] is True and r["steps"] == e[1] for r in bases)
    for r, r0 in zip(cut, full):
        assert (r["case_id"], r["kind"], r["arm"]) == (r0["case_id"], r0["kind"], r0["arm"])
        if r["kind"] == "base":
            assert r["batch_rows"] == [b for b in r0["batch_rows"] if b[0] < e[1]]
        else:
            assert [r[k] for k in key] == [r0[k] for k in key] and r["prefix_ok"] is True, r["arm"]


def test_base_prefix_without_event_runs_the_full_base():
    class Far(Toy):
        R_TRIG = 0.0

    base, dyn = by_arm(run.run_cases([CASE], ["none", "G"], conf(), [Far(), Far()], Brain(), base_prefix=True))
    assert base["e"] is None and base["base_cut"] is False and base["steps"] == 60 and type(base["success"]) is bool
    assert all(r["prefix_ok"] is True and r["act_hash"] == base["act_hash"] for r in dyn.values())


def test_ppcl_record_has_latch_counters():
    """Controls spec §11: a PPCL record carries its latch counters (JSON ints), no other arm has the key."""
    base, dyn = by_arm(run.run_cases([CASE], ["none", "PPC", "PPCL"], conf(), [Toy(), Toy()], Brain()))
    lat = dyn["PPCL"]["latch"]
    assert sorted(lat) == ["chunks", "resets", "trig"] and all(type(v) is int for v in lat.values())
    assert all("latch" not in r for r in (base, dyn["none"], dyn["PPC"]))
    json.dumps(dyn["PPCL"], allow_nan=False)


def test_validate_before_the_model():
    """main()'s checks before the brain loads: every arm builds, s + d <= the brain's chunk, arms on target relocation
    only; --set evalrest is eval[300:]."""
    run.validate(["none", "G", "Gauto", "PPC", "Gcal", "G@glr", "PPC@glr", "T0"], conf(), [CASE])
    for bad in ("Glead", "G@cv", "GautoU", "Gk0T@glr", "nope"):
        with pytest.raises(AssertionError):
            run.validate([bad], conf(), [CASE])
    oft = {**conf(), "brain": "oft", "s": 8}
    run.validate(["none"], oft, [CASE])
    with pytest.raises((AssertionError, KeyError)):
        run.validate(["Gk0"], oft, [CASE])  # no kappa0 for OFT
    with pytest.raises(AssertionError):
        run.validate(["none"], {**oft, "d": 1}, [CASE])  # 8 + 1 > H = 8
    run.validate(["none"], {**conf(), "s": 10, "d": 40}, [CASE])  # 50 = H of the LeRobot brains
    with pytest.raises(AssertionError):
        run.validate(["none"], {**conf(), "d": 41}, [CASE])
    light = {**CASE, "scenario": {**CASE["scenario"], "change_type": "illumination_switch"}}
    run.validate([], conf(), [CASE, light])  # connect: Base only
    with pytest.raises(AssertionError):
        run.validate(["none"], conf(), [CASE, light])
    split = {"calib": ["c"], "eval": [f"e{i}" for i in range(940)]}
    assert run.set_ids(split, "eval300") == split["eval"][:300] and run.set_ids(split, "calib") == ["c"]
    assert run.set_ids(split, "evalrest") == split["eval"][300:] and len(run.set_ids(split, "evalrest")) == 640


def test_make_agent_builds_vs_without_kbar():
    conf = {"t_ramp": 5, "k_p": 1.0, "tau_k": 0.01, "brain": "pi05", "kbar": None, "s": 10, "d": 0}
    for arm in ("VS0.1", "VS0.3", "VS1.0", "PPCL"):
        assert run.make_agent(arm, conf, 0).method == arm


BAND = {"model": "I", "G": [orx.G_POS, orx.G_POS], "B": [0.0, 0.0], "delta": 0.005}  # the Toy arm's own model


def test_pbvs_arms_and_log_traj():
    """PBVS spec §3, §9 item 8: --log-traj only adds the Base's x, p and executed xyz (act_hash unchanged, Dynamic
    records without it); PBVS arms run off conf["pbvs"] on the replayed prefix and carry their counters; a Base-only
    run (--arms '') works."""
    c, arms = {**conf(), "pbvs": BAND}, ["none", "PBVS0.3", "PBVS1.0"]
    (base0, _), (base, dyn) = (by_arm(run.run_cases([CASE], arms, c, [Toy(), Toy()], Brain(), log_traj=f))
                               for f in (False, True))
    assert "traj" not in base0 and all("traj" not in r for r in dyn.values())
    assert base["act_hash"] == base0["act_hash"] and base["steps"] == base0["steps"] == 60
    tr = base["traj"]
    x, p, a = (np.array(tr[k]) for k in "xpa")
    assert x.shape == p.shape == a.shape == (60, 3) and np.allclose(p, [0.5, 0.0, 0.0])
    assert np.allclose(np.diff(x, axis=0), orx.G_POS * a[:-1], atol=2e-6)  # the Toy integrator, rounded to 1e-6
    for arm in ("PBVS0.3", "PBVS1.0"):
        r, st = dyn[arm], dyn[arm]["pbvs"]
        assert r["method"] == arm and r["noise"] == "" and r["prefix_ok"] is True and r["t_engage"] == base["e"], arm
        assert st["steps_acting"] >= 1 and all(type(st[k]) is int for k in ("steps_window", "steps_acting", "flips"))
    assert "pbvs" not in dyn["none"] and dyn["PBVS0.3"]["act_hash"] != dyn["none"]["act_hash"]
    only = run.run_cases([CASE], [], conf(), [Toy()], Brain(), log_traj=True)
    assert [r["kind"] for r in only] == ["base"] and only[0]["act_hash"] == base0["act_hash"] and only[0]["traj"]
    assert [json.loads(x) for x in run.lines(only).splitlines()] == json.loads(json.dumps(only))


def test_validate_pbvs_needs_the_band():
    run.validate(["PBVS0.1", "PBVS0.3", "PBVS1.0"], {**conf(), "pbvs": BAND}, [CASE])
    assert run.make_agent("PBVS1.0", {**conf(), "pbvs": BAND}, 0).method == "PBVS1.0"
    with pytest.raises(KeyError):
        run.validate(["PBVS0.3"], conf(), [CASE])
    for bad in ("PBVS0.3@glr", "PBVS0.5"):
        with pytest.raises(AssertionError):
            run.validate([bad], {**conf(), "pbvs": BAND}, [CASE])


class Closing(Brain):
    """Brain's plan, its gripper closed when the observed EEF is within 8 cm of the point 2 cm above the object."""

    def __call__(self, obs, *a):
        out = super().__call__(obs, *a)
        for o, ch in zip(obs, out):
            ch[:, 6] = 1.0 if np.linalg.norm(o["obj"] + [0, 0, 0.02] - o["robot0_eef_pos"]) < 0.08 else -1.0
        return out


VST_ARMS = ["VST0.3", "VST1.0"]


def test_vst_s_star_from_the_same_runs_base():
    """PBVS spec journal (VS-T): each VST episode's s_star is its Base's xy offset gripper - object at the Base's first
    executed close (the Base's t_close; with --log-traj its x, p), |s_star| its grasp miss; the servo acts after the
    event on the replayed prefix; only VST records carry "vst"."""
    recs = run.run_cases([CASE], ["none", *VST_ARMS], conf(), [Toy(), Toy()], Closing(), log_traj=True)
    base, dyn = by_arm(recs)
    tc = base["t_close"]
    x, p = (np.array(base["traj"][k]) for k in "xp")
    assert tc is not None and base["e"] < tc and base["grasp_miss_m"] > 0.01
    for arm in VST_ARMS:
        r, v = dyn[arm], dyn[arm]["vst"]
        assert np.allclose(v["s_star"], (x[tc] - p[tc])[:2], atol=2e-6) and len(v["s_star"]) == 2, arm
        assert all(round(s, 6) == s and type(s) is float for s in v["s_star"]), arm  # rounded to 1e-6 (JSON floats)
        assert abs(np.linalg.norm(v["s_star"]) - base["grasp_miss_m"]) < 2e-6, arm
        assert r["prefix_ok"] is True and r["t_engage"] == base["e"] and v["steps_acting"] >= 1 and v["no_target"] == 0
        assert r["act_hash"] != dyn["none"]["act_hash"] and r["method"] == arm and r["noise"] == "", arm
        assert all(type(v[k]) is int for k in ("steps_window", "steps_acting", "flips", "no_target")), arm
    assert "vst" not in base and "vst" not in dyn["none"]
    assert [json.loads(x) for x in run.lines(recs).splitlines()] == json.loads(json.dumps(recs))


def test_vst_each_case_takes_its_own_bases_s_star():
    """Two cases in one batch whose Bases close at different offsets (the second arm starts 6 cm off in x): each VST
    episode takes its own case's s*."""

    class Apart(Toy):
        def start(self, case, arm, s):
            pkg = super().start(case, arm, s)
            if case["case_id"] == "toy-c1":
                self.x = self.x + [0.06, 0.0, 0.0]
                pkg = self._pkg()
            return pkg

    cases, got = [CASE, {**CASE, "case_id": "toy-c1"}], {}
    recs = run.run_cases(cases, ["none", *VST_ARMS], conf(), [Apart(), Apart()], Closing(), log_traj=True)
    for c in ("toy-c0", "toy-c1"):
        base, dyn = by_arm([r for r in recs if r["case_id"] == c])
        x, p, tc = np.array(base["traj"]["x"]), np.array(base["traj"]["p"]), base["t_close"]
        for arm in VST_ARMS:
            assert np.allclose(dyn[arm]["vst"]["s_star"], (x[tc] - p[tc])[:2], atol=2e-6), (c, arm)
        got[c] = dyn["VST1.0"]["vst"]["s_star"]
    assert np.linalg.norm(np.subtract(got["toy-c0"], got["toy-c1"])) > 0.01, got


def test_vst_without_a_base_close_is_none():
    """A Base that never closed (Brain keeps the gripper open): s_star None, the VST episode is "none"'s bitwise and its
    window steps count as no_target."""
    base, dyn = by_arm(run.run_cases([CASE], ["none", *VST_ARMS], conf(), [Toy(), Toy()], Brain()))
    assert base["t_close"] is None
    for arm in VST_ARMS:
        v = dyn[arm]["vst"]
        assert v["s_star"] is None and v["no_target"] == v["steps_window"] > 0 and v["steps_acting"] == 0, arm
        assert dyn[arm]["act_hash"] == dyn["none"]["act_hash"], arm


def test_vst_refuses_the_base_prefix(monkeypatch, tmp_path):
    """s_star needs the full Base: VST arms with --base-prefix stop, in run_cases and in main() before the brain
    loads."""
    with pytest.raises(AssertionError, match="base-prefix"):
        run.run_cases([CASE], ["none", "VST0.3"], conf(), [Toy(), Toy()], Closing(), base_prefix=True)
    monkeypatch.setattr(sys, "argv", ["c1e4m_run.py", "--brain", "pi05", "--source", "plus", "--set", "calib", "--arms",
                                      "none,VST1.0", "--base-prefix", "--out", str(tmp_path / "x.jsonl")])
    with pytest.raises(AssertionError, match="VS-T"):
        run.main()
    run.validate(VST_ARMS + ["VST0.1"], conf(), [CASE])  # no --band, no kbar: VS-T builds without them


def test_vst_arms_leave_the_other_arms_bitwise():
    """VST arms added to a run change nothing of the Base or the other arms (PBVS among them)."""
    c, old = {**conf(), "pbvs": BAND}, ["none", "G", "Gauto", "PPC", "PPCL", "PBVS0.3"]
    keys = ("act_hash", "success", "steps", "e", "prefix_ok", "t_engage", "calls_sched", "calls_trig", "t_close",
            "grasp_miss_m", "c_extra", "kappa_log", "plans_in")
    (b0, d0), (b1, d1) = (by_arm(run.run_cases([CASE], arms, c, [Toy(), Toy()], Closing()))
                          for arms in (old, old + VST_ARMS))
    assert [b0[k] for k in keys] == [b1[k] for k in keys]
    for a in old:
        assert [d0[a][k] for k in keys] == [d1[a][k] for k in keys], a
    assert d0["PBVS0.3"]["pbvs"] == d1["PBVS0.3"]["pbvs"] and d0["PPCL"]["latch"] == d1["PPCL"]["latch"]
    assert d1["VST0.3"]["act_hash"] != d1["none"]["act_hash"]


def test_ppcl_glr_on_ppc_glrs_frames():
    """controls spec journal "сводное решение владельца", the noisy cell: make_agent gives PPCL@glr the GLRKalman eyes
    seeded by the case (case_seed), as PPC@glr. On one object path (a jump at 20) both see the same noise frames, and
    the latch reads the tracked object (p_hist is the eyes' output, not the true object). With the latch off (L_th =
    +inf) PPCL@glr is PPC@glr step for step; with it on the frames stay the same."""
    import math

    import gauto

    def walk(arm, off=False):
        ag = run.make_agent(arm, conf(), run.case_seed(CASE["case_id"]))
        if off:
            ag.latch_th = math.inf
        x, p0, acts, seen, true = np.array([0.5, 0.35, 0.1]), np.array([0.5, 0.0, 0.0]), [], [], []
        ch = np.zeros((orx.H, 7))
        ch[:, 1], ch[:, 6] = -0.3, -1.0
        for t in range(60):
            p = p0 + (DELTA if t >= 20 else 0.0)
            ag.observe(t, x, p)
            seen.append(ag.p_hist[t].copy())
            true.append(p)
            if ag.sched.arrive(t):
                ag.on_new_chunk()
            if ag.sched.wants_call(t, ag.trigger(t)):
                ag.sched.issue(t, ch)
                if ag.sched.arrive(t):
                    ag.on_new_chunk()
            acts.append(ag.act(t, ag.sched.action(t)))
            x = x + orx.G_POS * np.clip(acts[-1][:3], -1, 1)
        assert isinstance(ag.tracker, gauto.GLRKalman) and ag.tracker.t_alarm is not None, arm
        return np.array(ag.tracker.meas), np.array(seen), np.array(true), np.array(acts)

    m0, s0, p, a0 = walk("PPC@glr")
    m1, s1, _, a1 = walk("PPCL@glr", off=True)
    assert np.array_equal(m0, m1) and np.array_equal(s0, s1) and np.array_equal(a0, a1)
    assert not np.allclose(s1, p, atol=1e-4)  # the agent's object is the eyes' estimate
    m2, s2, _, _ = walk("PPCL@glr")
    assert np.array_equal(m0, m2)  # the latch on: the same frames (one per observed step)


def test_shim_leaves_gym_without_gymnasium(monkeypatch):
    real = object()
    monkeypatch.setitem(sys.modules, "gym", real)
    monkeypatch.setitem(sys.modules, "gymnasium", None)  # import gymnasium raises ImportError
    run.shim()
    assert sys.modules["gym"] is real


def test_delay_keeps_the_prefix():
    recs = run.run_cases([CASE], ["none", "Gauto"], conf(d=10), [Toy(), Toy()], Brain())
    base, dyn = by_arm(recs)
    assert all(r["prefix_ok"] and r["e"] == base["e"] and r["g_check"] is None for r in dyn.values())
    # d = 10: answers arrive 10 steps after their observation (the first at once)
    assert all(a == t + 10 for t, a in dyn["none"]["plans_in"][1:]) and dyn["none"]["plans_in"][0] == [0, 0]


def test_prefix_broken_is_flagged():
    """A reflex that acts before the event (here: the object pushed away by 1 cm at step 3, G follows) is a broken
    pair: prefix_ok False, the episode still runs."""

    class Pushed(Toy):
        def step(self, a):
            pkg = super().step(a)
            if self.t == 3:
                self.p = self.p + [0.0, -0.01, 0.0]
                pkg = self._pkg()
            return pkg

    recs = run.run_cases([CASE], ["none", "G"], conf(), [Pushed(), Pushed()], Brain())
    _, dyn = by_arm(recs)
    assert dyn["none"]["prefix_ok"] is True and dyn["G"]["prefix_ok"] is False and dyn["G"]["steps"] == 60


def test_skipped_motion_blur_without_wand():
    blur = {**CASE, "case_id": "toy-blur", "task_name": "x_view_0_0_100_0_0_initstate_0_noise_7"}
    recs = run.run_cases([blur], ["none", "G"], conf(), [Toy()], Brain(), wand=False)
    assert [(r["kind"], r["arm"]) for r in recs] == [("dynamic", "none"), ("dynamic", "G"), ("base", "none")]
    assert all(r["skipped_reason"] for r in recs)


def test_helpers():
    assert run.noise_level({"task_name": "a_initstate_0_noise_31"}) == 31 and run.noise_level({"task_name": "a"}) == 0
    assert run.skip_reason({"task_name": "a_noise_10"}, False) and not run.skip_reason({"task_name": "a_noise_11"}, False)
    assert not run.skip_reason({"task_name": "a_noise_3"}, True)
    assert run.case_seed("x") == run.case_seed("x") != run.case_seed("y")
    assert run.source_of(CASE) == "plus" and run.source_of({"substrate_variant": {"benchmark": "LIBERO-PRO"}}) == "pro"
    task = SimpleNamespace(name="toy_task", problem_folder="02_x/bddl/toy", bddl_file="toy_task.bddl")
    assert run.same_task(CASE, task) and not run.same_task({**CASE, "task_name": "toy_task_2"}, task)
    pro = {**CASE, "substrate_variant": {"bddl_file": "02_x/bddl/toy/toy_task.bddl"}}
    assert run.same_task(pro, task) and not run.same_task(pro, SimpleNamespace(**{**vars(task), "problem_folder": "y"}))


def test_video_failure_keeps_the_records(monkeypatch, tmp_path):
    def broken(path, frames):
        raise OSError("disk full")

    monkeypatch.setattr(run, "write_video", broken)
    recs = run.run_cases([CASE], ["none"], conf(), [Toy()], Brain(), video_cases={CASE["case_id"]}, video_dir=tmp_path)
    assert [r["kind"] for r in recs] == ["dynamic", "base"]


def test_resume_drops_a_cut_write(tmp_path):
    out, c = tmp_path / "x.jsonl", conf()
    lines = [{"case_id": "a", "kind": "dynamic", "conf": c}, {"case_id": "a", "kind": "base", "conf": c},
             {"case_id": "b", "kind": "dynamic", "conf": c}]
    whole = "".join(json.dumps(r) + "\n" for r in lines[:2])
    out.write_text(whole + json.dumps(lines[2]) + "\n" + '{"case_id": "b", "ki')
    assert run.load_done(out, c) == {"a"} and out.read_text() == whole
    with pytest.raises(AssertionError):
        run.load_done(out, conf(d=10))


@pytest.mark.skipif("LIBERO-plus" not in os.environ.get("LIBERO_SOURCE_PACKAGE_ROOT", ""),
                    reason="needs the LIBERO-plus env of the runner's docstring")
def test_sync_envs_keep_their_own_np_random():
    """Two envs of one fog case (LIBERO-plus sensor noise 31-40 draws from global np.random) stepped side by side in
    one process see what one env alone sees."""
    run.shim()
    from libero_max.manifest import load_manifest
    cases = load_manifest(run.LMAX / "benchmark/max8000/libero_max_8000.json")["cases"]
    case = next(c for c in cases if c["scenario"]["change_type"] == "target_relocation"
                and 31 <= run.noise_level(c) <= 40)
    a = np.array([0.2, -0.1, 0.0, 0, 0, 0, -1])

    def frames(envs, n=3):
        got = [[env.start(case, "control", 10)["obs"]["agentview_image"]] for env in envs]
        for _ in range(n):
            for g, env in zip(got, envs):
                g.append(env.step(a)["obs"]["agentview_image"])
        for env in envs:
            env.close()
        return got

    alone, (x, y) = frames([run.LocalEnv(128)])[0], frames([run.LocalEnv(128), run.LocalEnv(128)])
    assert all(np.array_equal(p, q) and np.array_equal(p, r) for p, q, r in zip(alone, x, y))
    assert not np.array_equal(alone[1], alone[2])  # the fog redraws per step: the test can fail


def test_fog_map_at_our_render_size(monkeypatch):
    """LIBERO-plus's fog adds a 256-px plasma map; at 360 px it must be that same map, resized (not a crash)."""
    from types import ModuleType
    ew = ModuleType("libero.libero.envs.env_wrapper")
    ew.plasma_fractal = lambda mapsize=256, wibbledecay=3: np.random.uniform(0, 1, (mapsize, mapsize))
    for name, mod in {"libero": ModuleType("libero"), "libero.libero": ModuleType("libero.libero"),
                      "libero.libero.envs": ModuleType("libero.libero.envs"),
                      "libero.libero.envs.env_wrapper": ew}.items():
        monkeypatch.setitem(sys.modules, name, mod)
    sys.modules["libero.libero.envs"].env_wrapper = ew
    orig = ew.plasma_fractal
    run.fog_at(256)
    assert ew.plasma_fractal is orig
    run.fog_at(360)
    run.fog_at(360)  # once only
    np.random.seed(0)
    m = ew.plasma_fractal(wibbledecay=2.0)
    np.random.seed(0)
    m256 = orig()
    assert m.shape == (360, 360) and 0 <= m.min() and m.max() <= 1
    assert abs(m.mean() - m256.mean()) < 0.01  # the same draws, stretched
    assert ew.plasma_fractal(wibbledecay=2.0).shape == (360, 360)


@pytest.mark.skipif("LIBERO-plus" not in os.environ.get("LIBERO_SOURCE_PACKAGE_ROOT", ""),
                    reason="needs the LIBERO-plus scene env (see the runner's docstring)")
def test_fast_glass_is_theirs_bitwise():
    """LIBERO-plus's glass blur and ours: the same frame, the same np.random state after, every severity."""
    run.shim()
    from libero.libero.envs import env_wrapper as ew
    from PIL import Image
    run.fast_glass()
    fast = ew.glass_blur
    run.fast_glass()  # once only
    orig = fast.orig  # theirs (an env started in another test may have swapped it already)
    assert ew.glass_blur is fast and fast.c1e4m and not getattr(orig, "c1e4m", False)
    img = Image.fromarray(np.random.default_rng(1).integers(0, 256, (64, 64, 3), dtype=np.uint8))
    for sev in range(1, 11):
        np.random.seed(sev)
        a, sa = orig(img, severity=sev), np.random.get_state()
        np.random.seed(sev)
        b, sb = fast(img, severity=sev), np.random.get_state()
        assert a.dtype == b.dtype and np.array_equal(a, b) and all(np.array_equal(u, v) for u, v in zip(sa, sb)), sev

