"""C1-E4 part 2: our episode runner on LIBERO-MAX (plan docs/superpowers/plans/2026-10-05-c1-e4-part2.md, Task 9).

LIBERO-MAX (~/Desktop/M2R-libero-max @ a1e3cef, no license) is an external dependency: its package is on PYTHONPATH and
this file calls its CosmosInterventionEnv, load_manifest, load_case_task, wrap_case_env, create_libero_env_with_retry;
none of its code is copied here. One brain and one scene source (LIBERO-plus or LIBERO-PRO: different task registries)
per process. Per case:
- Base: LIBERO-MAX's "control" arm, method none on the call conveyor gauto.Conveyor(s, d), live. Kept in memory: the
  chunk of every call (by its observation step), every executed action, the step the event would fire at
  (trigger_observation).
- Dynamic: their "intervention" arm, once per --arms label (gauto.parse_arm). Until the event every call is answered
  with Base's recorded chunk of that step (maxwrap.ChunkReplay; the same (s, d) schedule as Base all episode), from the
  event on the brain is called live. A pre-event call Base did not make (the agent's own trigger) does not raise, as
  LIBERO-MAX's replay would: it goes live and is flagged (replay_miss = its step). prefix_ok: the executed actions
  equal Base's bitwise up to the event, at the same event step, with no replay miss. Spec §6 drops the pairs with
  prefix_ok false or a replay miss; the episode still runs to its end.
- Agent per step as scripts/c1e4_run.py: observe(t, eef, object) -> arrive -> Gcal's shadow pair -> wants_call /
  issue (brain or replay) -> arrive -> act -> step. eef = raw robot0_eef_pos; object = backend.entity_position of the
  trigger entity (the moved object). The event flag is not passed to the agent.
- Gcal (M0): at the agent's engagement t_e, two brain calls with one noise, on the last pre-event observation (step
  e - 1, the one the previous step() returned) and on the current one; gauto.shadow_sample as part 1.
- Time t: their policy step (total_env_steps - warmup_steps); e: their cosmos_query_boundary_step (Base: the would-be
  one), the first policy step whose observation is post-event. Brain noise seed: policy_seed + t (their runners);
  the shadow pair's: policy_seed + t_e.
One JSON line per episode (strict JSON, no NaN); a case's lines are written together, Base last (resume: a case with a
Base line is done). The run ends with "done N of M cases, failed K: <ids>" and exits 1 when a group failed (a rerun
with the same --out retries it).

Run (env set by the caller, workers inherit it). Source LIBERO-plus (PRO: LIBERO-PRO/libero, libero-pro-config):
    export MUJOCO_GL=cgl HF_HUB_OFFLINE=1 TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1
    export LIBERO_SOURCE_PACKAGE_ROOT=~/Desktop/M2R-lmax-deps/LIBERO-plus/libero
    export LIBERO_CONFIG_PATH=~/Desktop/M2R-lmax-deps/libero-plus-config
    export PYTHONPATH=~/Desktop/M2R-libero-max/scripts/libero_source_overlay:~/Desktop/M2R-libero-max/src  # Mac: +:~/Desktop/M2R-lmax-deps/pylib
    python scripts/c1e4m_run.py --brain smolvla --source plus --set calib --arms none,G,Gauto,PPC --kbar 0.038 \
        --d 0 --n-envs 2 --vector sync --device mps --out x.jsonl
Freeze the case split (once, WITHOUT the overlay and the env above: LIBERO task names from hf-libero):
    python scripts/c1e4m_run.py --freeze-split results/c1-e4/part2/split.json
"""

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
import traceback
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import c1e4m_brains as cb  # noqa: E402
import gauto  # noqa: E402
import maxwrap  # noqa: E402
import objreflex as orx  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
LMAX = Path(os.environ.get("LIBERO_MAX_ROOT", "~/Desktop/M2R-libero-max")).expanduser()
# LIBERO-MAX's policy-step limits (their scripts/run_*_persistent_shard.py MAX_STEPS), restated
MAX_STEPS = {"libero_spatial": 220, "libero_object": 280, "libero_goal": 300, "libero_10": 520}
Q = {"pi05": 10, "smolvla": 10, "oft": 8, "oftplus": 8}  # native call period (pi0.5: ours, as part 1)
DUMMY = [0, 0, 0, 0, 0, 0, -1]  # lerobot.envs.libero.get_libero_dummy_action(), their warm-up action
T_RAMP, K_P, TAU_K = 5, 1.0, 0.01  # scripts/c1e4_run.py's defaults
STRICT = json.JSONEncoder(allow_nan=False)  # not json.dumps: openvla-oft's import patches it to pass numpy silently


def lines(recs):
    """Records as strict JSON lines: NaN, infinity, a numpy int / bool / float32 raise (np.float64 is a float)."""
    return "".join(STRICT.encode(r) + "\n" for r in recs)


def shim():
    """Before LIBERO is imported: LIBERO-plus / PRO import old gym for type hints only (gym = gymnasium when gymnasium
    imports, else an installed gym stays); LIBERO-plus's fog uses np.float_ (gone in NumPy 2; LIBERO-MAX's runners set
    it too); Wand (ImageMagick) serves LIBERO-plus's motion blur only. Where Wand is missing (the Mac) the stub of the
    Task 1 scratch scene stands in, and motion-blur cases are skipped. True when Wand is real."""
    try:
        import gymnasium
        sys.modules.setdefault("gym", gymnasium)
    except ImportError:
        pass
    if not hasattr(np, "float_"):
        np.float_ = np.float64
    try:
        import wand.api  # noqa: F401
        import wand.image  # noqa: F401
        return True
    except (ImportError, OSError):
        pass

    class WandLib:
        def __getattr__(self, k):
            f = SimpleNamespace(argtypes=None, restype=None)
            setattr(self, k, f)
            return f

    class WandImage:
        def __init__(self, *a, **k):
            raise RuntimeError("wand stub: ImageMagick motion blur not available")

    w, wa, wi = (ModuleType(n) for n in ("wand", "wand.api", "wand.image"))
    wa.library, wi.Image = WandLib(), WandImage
    sys.modules.update({"wand": w, "wand.api": wa, "wand.image": wi})
    return False


def source_of(case):
    return "pro" if case.get("substrate_variant") else "plus"


def same_task(case, task):
    """The task load_case_task loaded is the case's: LIBERO-plus, the registry's task name (task_index numbers the
    overlay's variants, so another source's registry gives another task); LIBERO-PRO, the BDDL file (load_case_task
    copies the task name from the case, and one name repeats over PRO's categories)."""
    v = case.get("substrate_variant")
    return f"{task.problem_folder}/{task.bddl_file}" == v["bddl_file"] if v else task.name == case["task_name"]


def noise_level(case):
    """LIBERO-plus sensor-noise level (task name ..._noise_N; 1-10 motion blur, needs Wand), else 0."""
    m = re.search(r"_noise_(\d+)$", case.get("task_name") or "")
    return int(m.group(1)) if m else 0


def skip_reason(case, wand):
    return None if wand or not 0 < noise_level(case) <= 10 else "motion blur needs Wand (ImageMagick)"


def case_seed(case_id):
    """The noisy eyes' seed: a stable int from the case id (G@glr and PPC@glr share it)."""
    return int.from_bytes(hashlib.sha256(f"c1e4m/{case_id}".encode()).digest()[:4], "little")


def make_agent(arm, conf, seed):
    """scripts/c1e4_run.py make_agent (copied: Task 6 edits that file), oracle or GLR eyes."""
    method, noise = gauto.parse_arm(arm)
    assert noise in ("", "glr"), f"{arm}: part 2 on LIBERO-MAX runs the oracle and GLR eyes"
    kw = {"t_ramp": conf["t_ramp"], "k_p": conf["k_p"], "tau_k": conf["tau_k"], "K": gauto.K}
    if noise == "glr":
        kw["tracker"] = gauto.GLRKalman(seed, *gauto.NOISE)
    kbar = (gauto.KAPPA0[conf["brain"]] if method in ("Gk0", "Gk0T")
            else conf["kbar"] if method in ("Gauto", "GautoT") else None)
    return gauto.GAgent(method, conf["s"], conf["d"], kbar=kbar, **kw)


def validate(arms, conf, cases):
    """main()'s checks before the brain loads: every arm builds its agent, s + d fits the brain's chunk (the conveyor
    reads a chunk up to s + d - 1 steps after its observation), and Dynamic arms run on target relocation only (the eyes
    watch the trigger entity, the moved object; connect runs Base only)."""
    h = cb.OFTBrain.H if conf["brain"] in cb.OFT else orx.H  # OFT 8; LeRobotBrain asserts chunk_size == orx.H (50)
    assert conf["s"] + conf["d"] <= h, f"s + d = {conf['s'] + conf['d']} > {conf['brain']}'s chunk H = {h}"
    for a in arms:
        make_agent(a, conf, 0)
    bad = [c["case_id"] for c in cases if c["scenario"]["change_type"] != "target_relocation"]
    assert not (arms and bad), f"Dynamic arms {arms} on {len(bad)} cases without target relocation, e.g. {bad[:3]}"


def set_ids(split, name):
    """The case ids of --set: eval300 = the first 300 of eval (M1 + M2 of pi0.5, M4, D), evalrest = the rest of eval
    (M1)."""
    return split["eval"][:300] if name == "eval300" else split["eval"][300:] if name == "evalrest" else split[name]


def git_head(path):
    """The commit of the git checkout holding path (the LIBERO-MAX package this process imports), else "unknown"."""
    try:
        r = subprocess.run(["git", "-C", str(path), "rev-parse", "HEAD"], capture_output=True, text=True)
    except OSError:
        return "unknown"
    return r.stdout.strip() if r.returncode == 0 else "unknown"


def _reseed(v):  # LIBERO-MAX's reseed (np.random, torch)
    import torch
    np.random.seed(v)
    torch.manual_seed(v)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(v)


class LocalEnv:
    """One LIBERO-MAX episode per start(), built fresh as their reference shard does (one env per paired arm), and
    stepped to the end. np.random's global state is kept per env (LIBERO-plus's motion blur, fog and glass blur draw
    from it), so episodes stepped side by side in one process (--vector sync) draw as if each ran alone.
    send / recv: the interface of Remote (a worker process holding one LocalEnv)."""

    def __init__(self, res):
        self.res, self.w, self.env, self.rng, self.suites, self._out = res, None, None, None, {}, None

    def send(self, cmd, *a):
        self._out = getattr(self, cmd)(*a)

    def recv(self):
        return self._out

    def start(self, case, arm, s):
        from libero.libero import benchmark, get_libero_path
        from libero.libero.envs import OffScreenRenderEnv
        from libero_max.cosmos_integration import CosmosInterventionEnv
        from libero_max.env_factory import create_libero_env_with_retry
        from libero_max.pro_runtime import wrap_case_env
        from libero_max.substrate import load_case_task
        self.close()
        task, inits = load_case_task(case, benchmark, self.suites)
        assert same_task(case, task), (f"{case['case_id']}: loaded task {task.name} ({task.problem_folder}/"
                                       f"{task.bddl_file}): another scene source's env (LIBERO_* env vars)?")
        seed = int(case["policy_seed"])

        def factory():
            env = OffScreenRenderEnv(bddl_file_name=os.path.join(get_libero_path("bddl_files"), task.problem_folder,
                                                                 task.bddl_file),
                                     camera_heights=self.res, camera_widths=self.res)
            env.seed(seed)
            return env, task.language

        env, text = create_libero_env_with_retry(factory, policy_seed=seed, reseed=_reseed)
        self.env = env = wrap_case_env(env, case)
        env.seed(seed)
        self.w = w = CosmosInterventionEnv(env=env, task_description=text, scenario=case["scenario"], arm=arm,
                                           trace_path=Path(os.devnull), original_task_index=case["task_index"],
                                           init_state_index=case["init_state_index"])
        w.configure_episode(task_suite_name=case["task_suite_name"], policy_seed=seed, query_interval=s,
                            max_policy_steps=MAX_STEPS[case["task_suite_name"]])
        w.reset()
        obs = w.set_init_state(inits[case["init_state_index"]])
        done = False
        for _ in range(w.warmup_steps):
            obs, _, done, _ = w.step(DUMMY)
        self.rng = np.random.get_state()
        return self._pkg(obs, done)

    def step(self, a):
        np.random.set_state(self.rng)
        obs, _, done, _ = self.w.step(np.asarray(a, float).tolist())
        self.rng = np.random.get_state()
        return self._pkg(obs, done)

    def _pkg(self, obs, done):
        """What the main process needs of one step: the raw observation, the EEF rotation matrix of this step (the
        brain's state reads it; c1e4m_brains takes the matrix in place of an env), the trigger entity's position,
        success, the event step and the would-be one."""
        w = self.w
        return {"obs": obs, "mat": cb.robot_state_of(w), "done": bool(done),
                "obj": np.asarray(w.backend.entity_position(w.scenario["trigger"]["value"]), float),
                "e": w.events[0]["cosmos_query_boundary_step"] if w.events else None,
                "trig": (w.trigger_observation or {}).get("policy_step"), "t": w.total_env_steps - w.warmup_steps,
                "instr": w.runtime.current_instruction}

    def close(self):
        if self.env is not None:
            self.env.close()
        self.w = self.env = None


def _serve(conn, res):
    """Worker process (--vector async): one LocalEnv, commands from the main process."""
    shim()
    env = LocalEnv(res)
    while True:
        cmd, a = conn.recv()
        if cmd == "quit":
            env.close()
            return
        try:
            conn.send((True, getattr(env, cmd)(*a)))
        except Exception:
            conn.send((False, traceback.format_exc()))


class Remote:
    def __init__(self, res):
        ctx = mp.get_context("spawn")  # a fresh interpreter: no torch / GL state inherited from the main process
        self.conn, child = ctx.Pipe()
        self.proc = ctx.Process(target=_serve, args=(child, res), daemon=True)
        self.proc.start()

    def send(self, cmd, *a):
        self.conn.send((cmd, a))

    def recv(self):
        ok, out = self.conn.recv()
        if not ok:
            raise RuntimeError(f"worker {self.proc.pid}:\n{out}")
        return out

    def close(self):
        try:
            self.conn.send(("quit", ()))
            self.proc.join(10)
        except (OSError, EOFError, BrokenPipeError):
            pass
        if self.proc.is_alive():
            self.proc.terminate()


def ask(brain, pkgs, seeds, suites):
    """Brain rows of step packages, each with the EEF rotation matrix of its own step (not a live env)."""
    return brain([p["obs"] for p in pkgs], [p["mat"] for p in pkgs], [p["instr"] for p in pkgs], seeds, suites)


class Episode:
    """One episode, Base (kind "base": arm none, live) or one Dynamic arm (kind "dynamic", base = Base's
    {"chunks", "acts", "e"}), driven step by step from the main process."""

    def __init__(self, case, arm, kind, conf, base=None, video=False):
        self.case, self.arm, self.kind, self.conf, self.base = case, arm, kind, conf, base
        self.seed0, self.suite = int(case["policy_seed"]), case["task_suite_name"]
        self.ag = make_agent(arm, conf, case_seed(case["case_id"]))
        # the event step is set when the event is seen (their replay: while not runtime.applied)
        self.replay = maxwrap.ChunkReplay(base["chunks"], math.inf) if kind == "dynamic" else None
        self.t, self.t_max, self.cur, self.prev = 0, MAX_STEPS[self.suite], None, None
        self.e = self.cache = self.shadow = self.miss = None
        self.chunks, self.xs, self.acts, self.objs, self.plans_in = {}, [], [], [], []
        self.success, self.over, self.n_live, self.wall, self.batch_rows = False, False, 0, None, []
        self.frames = [] if video else None

    def see(self, pkg):
        assert pkg["t"] == self.t, (pkg["t"], self.t)
        self.prev, self.cur = self.cur, pkg
        if self.frames is not None:
            self.frames.append(pkg["obs"]["agentview_image"][::-1, ::-1])
        ev = pkg["e"] if self.kind == "dynamic" else pkg["trig"]
        if self.e is None and ev is not None:
            assert ev == self.t, f"event step {ev} seen at step {self.t}"  # e: the first post-event observation
            self.e = ev
            if self.replay is not None:
                self.replay.event_step = ev
                self.cache = self.prev  # the last pre-event observation (step e - 1)

    def _installed(self):
        self.ag.on_new_chunk()
        self.plans_in.append([self.ag.sched.t_obs, self.t])

    def observe(self):
        """Up to the call: observe, install an arriving plan. True when Gcal's shadow pair is due (its engagement, after
        the event)."""
        ag, t = self.ag, self.t
        ag.observe(t, self.cur["obs"]["robot0_eef_pos"], self.cur["obj"])
        if ag.sched.arrive(t):
            self._installed()
        return ag.method == "Gcal" and self.cache is not None and self.shadow is None and ag.t_e == t

    def set_shadow(self, two):
        ag, tf, t = self.ag, self.e - 1, self.t
        self.shadow = gauto.shadow_sample((tf, ag.x_hist[tf], two[0]), (t, ag.x_hist[t], two[1]),
                                          ag.p_hist[tf], ag.p_hist[t])

    def wants(self):
        return self.ag.sched.wants_call(self.t, self.ag.trigger(self.t))  # trigger(): once per step (PPC measures)

    def replayed(self):
        """Base's chunk for a call before the event; None for a live call. A pre-event call Base did not make (the
        agent's own trigger) does not raise: it goes live and is flagged (replay_miss, its first step), and spec §6
        drops the pair."""
        if self.replay is None or not self.replay.replaying(self.t):
            return None
        if self.t not in self.replay.recorded:
            self.miss = self.t if self.miss is None else self.miss
            return None
        return self.replay.chunk(self.t)

    def issue(self, chunk, live):
        ag, t = self.ag, self.t
        self.n_live += live
        if self.kind == "base":
            self.chunks[t] = chunk
        ag.sched.issue(t, chunk)
        if ag.sched.arrive(t):
            self._installed()

    def act(self):
        ag, t = self.ag, self.t
        a = np.clip(ag.act(t, ag.sched.action(t)), -1, 1)
        self.xs.append(self.cur["obs"]["robot0_eef_pos"])
        self.acts.append(a)
        self.objs.append(self.cur["obj"])
        return a

    def after(self, pkg):
        self.t += 1
        self.success = pkg["done"]
        self.over = self.success or self.t >= self.t_max
        self.see(pkg)

    def record(self):
        ag, conf = self.ag, self.conf
        method = ag.method
        rec = ident(self.case, self.kind, self.arm, conf) | {
            "e": self.e, "success": bool(self.success), "steps": len(self.acts), "wall_s": round(self.wall, 2),
            "calls_sched": ag.sched.n_sched, "calls_trig": ag.sched.n_trig, "calls_live": self.n_live,
            "calls_shadow": 2 * (self.shadow is not None), "shadow": self.shadow, "plans_in": self.plans_in,
            "batch_rows": self.batch_rows,
            "t_engage": ag.t_e, "g_on": ag.g_on, "t_alarm": getattr(ag.tracker, "t_alarm", None),
            "kappa_log": ag.kappa_log, "c_extra": np.round(ag.c, 6).tolist(),
            "act_hash": hashlib.sha256(np.asarray(self.acts).tobytes()).hexdigest()[:16],
            "prefix_ok": None, "replay_miss": self.miss, "g_check": None}
        rec |= orx.episode_metrics(self.xs, self.acts, self.objs)
        # grasp_ok after the event (spec §6): lifted >= 3 cm above its height at step e while closed, from e on (Base:
        # the would-be e; no event: step 0)
        k = self.e or 0
        rec["grasp_ok"] = (k < len(self.acts)
                           and orx.episode_metrics(self.xs[k:], self.acts[k:], self.objs[k:])["grasp_ok"])
        if self.kind == "dynamic":
            b = self.base
            n = len(self.acts) if self.e is None else self.e  # actions of steps 0 .. e - 1
            rec["prefix_ok"] = bool(self.miss is None and self.e == b["e"] and len(b["acts"]) >= n
                                    and np.array_equal(np.asarray(self.acts[:n]), np.asarray(b["acts"][:n])))
            if method == "none" and conf["d"] == 0 and self.e is not None:
                ok, last, first = maxwrap.schedule_check(self.e, conf["s"], self.plans_in, len(self.acts), d=0)
                rec["g_check"] = {"ok": bool(ok), "last": last, "first": first}
        return rec


def ident(case, kind, arm, conf):
    sc = case["scenario"]
    method, noise = gauto.parse_arm(arm)
    return {"case_id": case["case_id"], "brain": conf["brain"], "arm": arm, "method": method, "noise": noise,
            "kind": kind, "s": conf["s"], "d": conf["d"], "R": conf["R"], "source": source_of(case),
            "suite": case["task_suite_name"], "task_name": case.get("task_name"),
            "init_state_index": case["init_state_index"], "policy_seed": case["policy_seed"],
            "change_type": sc["change_type"], "distance_m": sc["change"].get("distance_m"),
            "direction_xy": sc["change"].get("direction_xy"), "substrate_category": case.get("substrate_category"),
            "substrate_variant": case.get("substrate_variant"), "conf": conf, "skipped_reason": None}


def run_batch(eps, envs, brain):
    """Run up to len(envs) episodes side by side to their ends (ponytail: a slot idles until the batch's longest
    episode ends; refill slots one by one if pod time matters). Brain calls are batched over the episodes."""
    t0 = time.time()
    pairs = list(zip(eps, envs))
    for ep, env in pairs:
        env.send("start", ep.case, "control" if ep.kind == "base" else "intervention", ep.conf["s"])
    for ep, env in pairs:
        ep.see(env.recv())
    live = pairs
    while live:
        due = [ep for ep, _ in live if ep.observe()]
        if due:  # two calls with one noise: the last pre-event observation and the current one
            two = ask(brain, [p for ep in due for p in (ep.cache, ep.cur)], [ep.seed0 + ep.t for ep in due for _ in "ab"],
                      [ep.suite for ep in due for _ in "ab"])
            for j, ep in enumerate(due):
                ep.set_shadow(two[2 * j: 2 * j + 2])
                ep.shadow["batch_rows"] = [2 * len(due), len({e.cur["instr"] for e in due})]
        want = [ep for ep, _ in live if ep.wants()]
        got = [ep.replayed() for ep in want]
        calls = [ep for ep, ch in zip(want, got) if ch is None]
        if calls:  # SmolVLA pads the text to the batch's longest instruction: a chunk differs ~1e-5 with the batch
            fresh = iter(ask(brain, [ep.cur for ep in calls], [ep.seed0 + ep.t for ep in calls],
                             [ep.suite for ep in calls]))
            for ep in calls:
                ep.batch_rows.append([ep.t, len(calls), len({e.cur["instr"] for e in calls})])
        for ep, ch in zip(want, got):
            ep.issue(next(fresh) if ch is None else ch, ch is None)
        for ep, env in live:
            env.send("step", ep.act())
        for ep, env in live:
            ep.after(env.recv())
        for ep, _ in live:
            if ep.over:
                ep.wall = time.time() - t0
        live = [(ep, env) for ep, env in live if not ep.over]


def write_video(path, frames):
    import cv2
    h, w = frames[0].shape[:2]
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 20, (w, h))  # LIBERO's control rate, 20 Hz
    for f in frames:
        out.write(cv2.cvtColor(np.ascontiguousarray(f), cv2.COLOR_RGB2BGR))
    out.release()


def run_cases(cases, arms, conf, envs, brain, wand=True, video_cases=(), video_dir=None):
    """Base of every case, then their Dynamic arms, n = len(envs) episodes at a time. The records, per case Dynamic
    first and Base last."""
    n, out, ready = len(envs), {}, []
    for c in cases:
        why = skip_reason(c, wand)
        if why:
            print(f"skip {c['case_id']}: {why}", flush=True)
            out[c["case_id"]] = [ident(c, k, a, conf) | {"skipped_reason": why}
                                 for k, a in [("dynamic", a) for a in arms] + [("base", "none")]]
        else:
            ready.append(c)

    def batches(eps):
        for j in range(0, len(eps), n):
            run_batch(eps[j: j + n], envs, brain)

    bases = [Episode(c, "none", "base", conf, video=c["case_id"] in video_cases) for c in ready]
    batches(bases)
    dyn = [Episode(b.case, a, "dynamic", conf, {"chunks": b.chunks, "acts": b.acts, "e": b.e}, b.frames is not None)
           for b in bases for a in arms]
    batches(dyn)
    for ep in dyn + bases:
        out.setdefault(ep.case["case_id"], []).append(ep.record())
        if ep.frames is not None:
            path = (Path(video_dir) / f"{ep.case['case_id']}__{conf['brain']}__{ep.kind}_{ep.arm}__s{conf['s']}"
                                      f"d{conf['d']}.mp4")
            try:
                write_video(path, ep.frames)
            except Exception as exc:  # a video is for viewing only: the records stay
                print(f"video {path} not written: {exc!r}", flush=True)
    return [r for c in cases for r in out[c["case_id"]]]


def load_done(path, conf):
    """Resume: the case ids with a Base line in path. A case's lines are written together, Base last, so anything after
    the last complete Base line is a cut write: it is dropped first."""
    p = Path(path)
    if not p.exists():
        return set()
    raw, done, keep, pos = p.read_bytes(), set(), 0, 0
    for line in raw.splitlines(keepends=True):
        pos += len(line)
        if not line.endswith(b"\n"):
            break
        r = json.loads(line)
        assert r["conf"] == conf, f"{path} was written with another configuration: {r['conf']} vs {conf}"
        if r["kind"] == "base":
            done.add(r["case_id"])
            keep = pos
    if keep < len(raw):
        print(f"warning: dropping {len(raw) - keep} bytes after the last complete case of {path}", flush=True)
        os.truncate(path, keep)
    return done


def freeze(out, manifest):
    """The frozen case split: maxwrap.freeze_split over the 1000 target-relocation cases, LIBERO's 40 task names from
    standard hf-libero (not the LIBERO-plus overlay's variant names), the connection sample over the whole manifest."""
    import libero
    assert "libero_source_overlay" not in str(libero.__file__), "freeze the split without the overlay (hf-libero names)"
    from libero.libero import benchmark
    sys.path.insert(0, str(LMAX / "src"))
    from libero_max.manifest import load_manifest
    cases = load_manifest(Path(manifest))["cases"]
    target = [c for c in cases if c["scenario"]["change_type"] == "target_relocation"]
    bd = benchmark.get_benchmark_dict()
    names = {s: bd[s]().get_task_names() for s in MAX_STEPS}
    d = maxwrap.freeze_split(target, names, seed=maxwrap.SEED, extra=0, all_cases=cases)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(d, indent=1) + "\n")
    maxwrap.load_split(out)
    print(f"{out}: calib {len(d['calib'])} eval {len(d['eval'])} connect {len(d['connect'])} sha256 {d['sha256']}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--freeze-split", metavar="OUT", help="write the frozen case split and exit")
    p.add_argument("--manifest", default=str(LMAX / "benchmark/max8000/libero_max_8000.json"))
    p.add_argument("--brain", choices=cb.NAMES)
    p.add_argument("--precision", choices=["fp32", "bf16"], help="default fp32 (LeRobot), bf16 (OFT)")
    p.add_argument("--source", choices=["plus", "pro"], help="the scene source of this process (env vars)")
    p.add_argument("--split", default=str(REPO / "results/c1-e4/part2/split.json"))
    p.add_argument("--set", choices=["calib", "eval", "eval300", "evalrest", "connect"])
    p.add_argument("--cases", help="comma-separated case ids, a subset of the set (debugging)")
    p.add_argument("--arms", default="none,G,Gauto,PPC", help="Dynamic arms (gauto.parse_arm); '' = Base only")
    p.add_argument("--s", type=int, help="call period, default the brain's native Q")
    p.add_argument("--d", type=int, default=0, help="answer delay, steps")
    p.add_argument("--kbar", type=float, help="G-auto's frozen kbar for this brain")
    p.add_argument("--video-cases", default="", help="comma-separated case ids: mp4 of all their episodes")
    p.add_argument("--video-dir", help="outside the repository")
    p.add_argument("--n-envs", type=int, default=6)
    p.add_argument("--vector", choices=["async", "sync"], default="async")
    p.add_argument("--device", default="cuda")
    p.add_argument("--out", help="JSON lines")
    p.add_argument("--repeat-check", action="store_true",
                   help="repeat the first brain call (same inputs and seeds) and print whether it matches bitwise")
    args = p.parse_args()
    wand = shim()
    if args.freeze_split:
        return freeze(args.freeze_split, args.manifest)
    assert args.brain and args.source and args.set and args.out, "--brain, --source, --set and --out are required"
    arms = [a for a in args.arms.split(",") if a]
    methods = {gauto.parse_arm(a)[0] for a in arms}
    assert args.kbar is not None or not methods & {"Gauto", "GautoT"}, "G-auto needs --kbar"
    video = set(filter(None, args.video_cases.split(",")))
    assert not video or (args.video_dir and REPO not in Path(args.video_dir).resolve().parents
                         and Path(args.video_dir).resolve() != REPO), "videos need --video-dir outside the repository"
    split = maxwrap.load_split(args.split)
    ids = set_ids(split, args.set)
    rev = (gauto.BRAINS.get(args.brain) or cb.OFT[args.brain])[1]
    precision = args.precision or ("bf16" if args.brain in cb.OFT else "fp32")
    conf = {"brain": args.brain, "revision": rev, "precision": precision, "s": args.s or Q[args.brain], "d": args.d,
            "kbar": args.kbar, "t_ramp": T_RAMP, "k_p": K_P, "tau_k": TAU_K, "R": cb.RES[args.brain], "arms": arms,
            "split": split["sha256"]}
    from libero_max.manifest import load_manifest
    by_id = {c["case_id"]: c for c in load_manifest(Path(args.manifest))["cases"]}
    cases = [by_id[i] for i in ids if source_of(by_id[i]) == args.source]
    if args.cases:
        keep = set(args.cases.split(","))
        assert keep <= set(ids), f"not in --set {args.set}: {sorted(keep - set(ids))}"
        cases = [c for c in cases if c["case_id"] in keep]
    validate(arms, conf, cases)
    import libero_max
    prov = {"lmax_commit": git_head(Path(libero_max.__file__).parent), "source": args.source, "n_envs": args.n_envs,
            "vector": args.vector}  # not in conf: conf is the resume key, these may change between resumed runs
    done = load_done(args.out, conf)
    todo = [c for c in cases if c["case_id"] not in done]
    print(f"{len(todo)} of {len(cases)} {args.source} cases of {args.set} to run, {1 + len(arms)} episodes each; "
          f"split {split['sha256'][:12]}, Wand {'real' if wand else 'stub'}", flush=True)
    if not todo:
        print("done 0 of 0 cases, failed 0:", flush=True)
        return
    if video:
        Path(args.video_dir).mkdir(parents=True, exist_ok=True)
    timer = {"s": 0.0, "calls": 0, "rows": 0}
    model = cb.make_brain(args.brain, args.device, precision)
    repeat = args.repeat_check

    def brain(*a):
        nonlocal repeat
        tb = time.time()
        out = model(*a)
        timer["s"] += time.time() - tb
        timer["calls"] += 1
        timer["rows"] += len(a[3])
        if repeat:  # spec §4.1 (scripts/c1e4_run.py --repeat-check): the same call again must give the same chunk
            repeat, again = False, model(*a)
            print(f"C1-E4m repeat: rows={len(a[3])} max_abs={np.abs(again - out).max():.3e} "
                  f"bitwise={int(np.array_equal(again, out))}", flush=True)
        return out

    def make_envs():
        return [(Remote if args.vector == "async" else LocalEnv)(cb.RES[args.brain]) for _ in range(args.n_envs)]

    envs, t0, n_done, n_try, n_fail, failed = make_envs(), time.time(), 0, 0, 0, []
    try:
        for g in range(0, len(todo), args.n_envs):
            group, tb, n_try = todo[g: g + args.n_envs], time.time(), n_try + 1
            timer.update(s=0.0, calls=0, rows=0)
            try:
                recs = run_cases(group, arms, conf, envs, brain, wand, video, args.video_dir)
                text = lines(r | {"run": prov} for r in recs)
            except Exception as exc:  # a fallen group is skipped here; a rerun with resume retries it
                n_fail, failed = n_fail + 1, failed + [c["case_id"] for c in group]
                print(f"FAILED {[c['case_id'] for c in group]}: {exc!r}", flush=True)
                traceback.print_exc()
                for env in envs:
                    try:
                        env.close()
                    except Exception:
                        pass
                if n_fail >= 5 and n_fail > 0.2 * n_try:
                    print(f"ABORT: systematic failure, {n_fail} of {n_try} groups failed", flush=True)
                    break
                envs = make_envs()
                continue
            with open(args.out, "a") as f:
                f.write(text)
            n_done += len(group)
            rate = (time.time() - t0) / n_done
            print(f"group {g // args.n_envs}: {len(group)} cases {len(recs)} episodes {time.time() - tb:.1f} s, brain "
                  f"{timer['s']:.1f} s {timer['calls']} calls {timer['rows']} rows; {n_done}/{len(todo)} cases, "
                  f"{rate:.1f} s/case, eta {rate * (len(todo) - n_done) / 3600:.2f} h", flush=True)
    finally:
        for env in envs:
            env.close()
    print(f"done {n_done} of {len(todo)} cases, failed {len(failed)}: {','.join(failed)}".rstrip(), flush=True)
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
