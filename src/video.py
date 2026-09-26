"""Side-by-side demo videos: several methods on one level, from the same start with the same action noise.

Each panel is env 0 of eval_flow.eval. After its first episode ends, a panel freezes on its last frame and shows
the outcome (AutoReplay would otherwise restart the level). One episode is an illustration, not evidence.

  kick  report E4: naive, RTC and reflex under kicks (results/video_kick.mp4)
  demo  phase B1 demos a, b, c at d = 3, s = 5 (plan docs/superpowers/plans/2026-09-26-demo-videos.md): the episode
        is picked by a rule fixed before viewing, and the footer gives counts over seeds 0-15 and B1's numbers
"""

import dataclasses
import pathlib
import pickle
import textwrap
from typing import Literal, Sequence

import imageio
import jax
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import pandas as pd
import tyro
from PIL import Image, ImageDraw, ImageFont

import eval_flow
import predictors
import probe
import train_expert

D, S = 3, 5  # the demos use B1's d = 3 slice
B1_CSV = "results/b1/gpu/eval_d3/results.csv"  # B1 at d = 3, s = 5: RTX 4090, seeds 10-12 x 256 episodes
PRESETS = {  # fixed before viewing any episode (see the plan); the last panel is the reflex
    "a": dict(
        level="mjc_swimmer", title="mjc_swimmer, learned world model",
        panels=("realtime", "pred:learned", "reflex:learned"), hero="reflex:learned", rival="pred:learned",
        why="Level picked as B1's strongest for J (reflex - pred, learned model).",
    ),
    "b": dict(
        level="mjc_swimmer", title="mjc_swimmer, physics with 20% parameter errors",
        panels=("pred:phys0.2", "reflex:phys0.2"), hero="reflex:phys0.2", rival="pred:phys0.2",
        why="Same level as video A; with 20% wrong physics it ranks 4th of 12 for J in B1.",
    ),
    "c": dict(
        level="catapult", title="catapult, learned world model",
        panels=("realtime", "reflex:learned"), hero="realtime", rival="reflex:learned",
        why="Level picked as B1's worst for the reflex against RTC.",
    ),
}


def main(
    run_path: str = "checkpoints/bc",
    level_path: str = "worlds/l/mjc_walker.json",
    methods: Sequence[str] = ("naive", "realtime", "reflex"),
    delay: int = 2,
    horizon: int = 6,
    kick_prob: float = 0.02,
    kick_std: float = 1.0788,  # c = 1 x v_med (results/eval/v_med.txt)
    seed: int = 0,
    fps: int = 15,
    out: str = "results/video_kick.mp4",
):
    env, level, policy, env_params, static_env_params = _setup(run_path, level_path)
    font = ImageFont.load_default(size=28)
    panels = []
    for name in methods:
        config = eval_flow.EvalConfig(
            num_evals=1, inference_delay=delay, execute_horizon=horizon, method=eval_flow.METHODS[name],
            kick_prob=kick_prob, kick_std=kick_std,
        )
        info, video = jax.jit(lambda rng: eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params))(
            jax.random.key(seed)
        )
        solved, length = _outcome(info)
        print(_label(name, solved, length))
        panels.append(_panel(np.array(video), name, solved, length, font))
    imageio.mimwrite(out, np.concatenate(panels, axis=2), fps=fps)  # methods side by side
    print(f"saved {out}")


def demo(
    preset: Literal["a", "b", "c"],
    run_path: str = "checkpoints/bc",
    seeds: int = 16,
    fps: int = 15,
    out_dir: str = "results/demo",
    gif_width: int = 0,  # > 0: also write <preset>.gif of the panels (no footer), every 2nd frame, this many px wide
):
    p = PRESETS[preset]
    level_path = f"worlds/l/{p['level']}.json"
    env, level, policy, env_params, static_env_params = _setup(run_path, level_path)
    world_model = None
    if any(spec.endswith(":learned") for spec in p["panels"]):
        with (pathlib.Path(predictors.WM_DIR) / f"{predictors.level_name(level_path)}.pkl").open("rb") as f:
            world_model = pickle.load(f)
    fns = {spec: _rollout(spec, env, level, policy, env_params, static_env_params, world_model) for spec in p["panels"]}
    # pass 1: outcomes only. ponytail: each call also renders its video and drops it, so pass 2 can rerun the very
    # same compiled program; a render-free pass 1 would be a second compile per panel
    outcome = {(spec, k): _outcome(fn(jax.random.key(k))[0]) for spec, fn in fns.items() for k in range(seeds)}
    solved = {spec: np.array([outcome[spec, k][0] for k in range(seeds)]) for spec in p["panels"]}
    seed = pick(solved, p["hero"], p["rival"])
    cond = f"{_name(p['hero'])} solves and {_name(p['rival'])} fails"
    if seed is None:
        rule, seed = f"no seed in 0-{seeds - 1} where {cond}; showing seed 0", 0
    else:
        rule = f"the first of seeds 0-{seeds - 1} where {cond}: seed {seed}"
    end = max(outcome[spec, seed][1] for spec in p["panels"]) + fps  # hold the last frames for 1 s
    font = ImageFont.load_default(size=28)
    panels = []
    for spec, fn in fns.items():  # pass 2: render the picked seed
        info, video = fn(jax.random.key(seed))
        assert _outcome(info) == outcome[spec, seed], f"{spec}: the rerun of seed {seed} differs from pass 1"
        panels.append(_panel(np.array(video), _name(spec), *outcome[spec, seed], font)[:end])
    frames = np.concatenate(panels, axis=2)
    t = b1_table()
    pred = p["panels"][-1].partition(":")[2]
    specs = ("realtime", f"pred:{pred}", f"reflex:{pred}")
    lines = [
        f"{p['title']}; d = {D}, s = {S}; every panel has the same start and action noise.",
        f"Illustration, not evidence. Episode rule, fixed before viewing: {rule}.",
        f"Solved over seeds 0-{seeds - 1}: " + ", ".join(f"{_name(s)} {solved[s].sum()}/{seeds}" for s in p["panels"])
        + ". B1 on this level (768 episodes): "
        + ", ".join(f"{_name(s.partition(':')[0])} {t.at[p['level'], s]:.0f}%" for s in specs) + ".",
        f"{p['why']} Mean over all 12 levels: reflex {(t[specs[2]] - t['realtime']).mean():+.1f} pp over RTC.",
    ]
    footer = _footer(lines, frames.shape[2])
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(str(out / f"{preset}.mp4"), [np.concatenate([f, footer]) for f in frames], fps=fps)
    runs = [(spec, k, *o) for (spec, k), o in outcome.items()]
    pd.DataFrame(runs, columns=["spec", "seed", "solved", "length"]).to_csv(out / f"{preset}.csv", index=False)
    if gif_width:  # every 2nd frame at the same frame time: twice real time
        size = (gif_width, frames.shape[1] * gif_width // frames.shape[2])
        gif = [Image.fromarray(f).resize(size, Image.LANCZOS) for f in frames[::2]]
        gif[0].save(out / f"{preset}.gif", save_all=True, append_images=gif[1:], duration=1000 // fps, loop=0,
                    optimize=True)
    print("\n".join(lines))
    print(f"saved {out}/{preset}.*")


def pick(solved: dict[str, np.ndarray], hero: str, rival: str) -> int | None:
    """The episode rule, fixed before viewing: the first seed where `hero` solves and `rival` fails, else None."""
    hits = np.flatnonzero(solved[hero] & ~solved[rival])
    return int(hits[0]) if hits.size else None


def b1_table() -> pd.DataFrame:
    """B1 solve rate (%) at d = D, s = S per level (rows) and 'method[:predictor]' (columns), 768 episodes each."""
    d = pd.read_csv(B1_CSV)
    d = d[(d.delay == D) & (d.execute_horizon == S)]
    spec = d.method.where(d.predictor == "-", d.method + ":" + d.predictor)
    level = d.level.str.extract(r"/(\w+)\.json$", expand=False)
    return d.returned_episode_solved.groupby([level, spec]).mean().unstack() * 100


def _setup(run_path, level_path):
    """Kinetix env, one level and its BC policy, as in eval_flow.main."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    level = jax.tree.map(lambda x: x[0], train_expert.load_levels([level_path], static_env_params, env_params))
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    policy = probe.make_policy(probe.load_state_dict(run_path, level_path), obs_dim, env.action_space(env_params).shape[0])
    return env, level, policy, env_params, static_env_params


def _rollout(spec, env, level, policy, env_params, static_env_params, world_model):
    """jit(eval_flow.eval) of one 'method[:predictor]' panel at d = D, s = S, no kicks: rng -> (info, video)."""
    method, _, pred = spec.partition(":")
    m = eval_flow.METHODS[method]
    if pred:
        m = dataclasses.replace(m, **eval_flow.parse_predictor(pred))
    config = eval_flow.EvalConfig(num_evals=1, inference_delay=D, execute_horizon=S, method=m)
    return jax.jit(
        lambda rng: eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params, world_model=world_model)
    )


def _outcome(info) -> tuple[bool, int]:
    return bool(info["returned_episode_solved"] > 0), int(info["returned_episode_lengths"])


def _name(spec):
    """'realtime' -> 'RTC', 'reflex:learned' -> 'reflex (learned)'."""
    method, _, pred = spec.partition(":")
    method = "RTC" if method == "realtime" else method
    return f"{method} ({pred})" if pred else method


def _label(name, solved, length):
    return f"{name}: {'solved' if solved else 'failed'} ({length} steps)"


def _panel(video, name, solved, length, font):
    """Upscale Kinetix's 128 px render (nearest neighbour), caption it, freeze it after the first episode ends."""
    scale = max(1, 512 // video.shape[1])
    video = video.repeat(scale, axis=1).repeat(scale, axis=2)
    video[length:] = video[length - 1]
    return np.stack([_caption(f, name if i < length else _label(name, solved, length), font) for i, f in enumerate(video)])


def _caption(frame, text, font):
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((12, 10), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(img)


def _footer(lines, width, size=20):
    """Dark strip under the panels: the lines wrapped to `width`, height a multiple of 16 for the video encoder."""
    font = ImageFont.load_default(size=size)
    lines = [w for line in lines for w in textwrap.wrap(line, width=int(width / (0.6 * size)))]
    height = -(-(len(lines) * (size + 6) + 16) // 16) * 16
    img = Image.new("RGB", (width, height), (24, 24, 24))
    draw = ImageDraw.Draw(img)
    for i, line in enumerate(lines):
        draw.text((12, 8 + i * (size + 6)), line, font=font, fill=(235, 235, 235))
    return np.asarray(img)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"kick": main, "demo": demo})
