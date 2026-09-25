"""Side-by-side demo video: several methods on one level with the same start, action noise and kicks (report, E4).

Each panel is env 0 of eval_flow.eval. After its first episode ends, a panel freezes on its last frame and shows
the outcome (AutoReplay would otherwise restart the level). One episode is an illustration, not evidence.
"""

from typing import Sequence

import imageio
import jax
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import tyro
from PIL import Image, ImageDraw, ImageFont

import eval_flow
import probe
import train_expert


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
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    level = jax.tree.map(lambda x: x[0], train_expert.load_levels([level_path], static_env_params, env_params))
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    policy = probe.make_policy(probe.load_state_dict(run_path, level_path), obs_dim, env.action_space(env_params).shape[0])
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
        video, length = np.array(video), int(info["returned_episode_lengths"])
        scale = max(1, 512 // video.shape[1])  # Kinetix renders downscaled (128 px): upscale, nearest neighbour
        video = video.repeat(scale, axis=1).repeat(scale, axis=2)
        label = f"{name}: {'solved' if info['returned_episode_solved'] > 0 else 'failed'} ({length} steps)"
        print(label)
        video[length:] = video[length - 1]
        panels.append(np.stack([_caption(f, name if i < length else label, font) for i, f in enumerate(video)]))
    imageio.mimwrite(out, np.concatenate(panels, axis=2), fps=fps)  # methods side by side
    print(f"saved {out}")


def _caption(frame, text, font):
    img = Image.fromarray(frame)
    ImageDraw.Draw(img).text((12, 10), text, font=font, fill=(255, 255, 255), stroke_width=3, stroke_fill=(0, 0, 0))
    return np.asarray(img)


if __name__ == "__main__":
    tyro.cli(main)
