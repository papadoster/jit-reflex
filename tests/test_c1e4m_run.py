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
