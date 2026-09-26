# Демо-видео B1 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Три демо-видео по B1 (d = 3, s = 5) и короткий GIF для README. Эпизод выбирается правилом, записанным и запушенным до просмотра. В подписи — счёт по 16 seed'ам, цифры B1 по уровню и среднее по 12 уровням, чтобы никто не обобщал с одного эпизода или уровня.

**Architecture:** Всё в `src/video.py`:
- старая функция `main` остаётся (видео E4) и становится подкомандой `kick`;
- новая подкоманда `demo --preset a|b|c`:
  - проход 1 — исходы seed'ов 0–15 для каждой панели;
  - правило `pick` выбирает seed;
  - проход 2 рисует выбранный seed той же скомпилированной функцией и сверяет исход с проходом 1;
  - внизу кадра подпись;
  - дополнительно GIF.
- `eval_flow` не трогаем: предсказатель задаётся через `dataclasses.replace(METHODS[m], **parse_predictor(p))`, модель мира передаётся аргументом `world_model`.

**Tech Stack:** JAX 0.4.35 (CPU), Kinetix, numpy, pandas, Pillow, imageio(-ffmpeg), tyro 0.9 (`tyro.extras.subcommand_cli_from_dict`), pytest (`pythonpath = src`), uv.

---

## Правило, записанное до просмотра

Уровни выбраны по данным B1 на срезе d = 3, s = 5, `results/b1/gpu/eval_d3/results.csv`, 768 эпизодов на клетку. Ни одного эпизода к этому моменту не видели.

| видео | уровень | панели | эпизод: первый seed из 0–15, где… | почему этот уровень |
|---|---|---|---|---|
| **A** «`J` — это и есть разница» | mjc_swimmer | RTC · pred (learned) · reflex (learned) | reflex (learned) решил, pred (learned) нет | самый сильный эффект `J` (reflex − pred) с моделью: +29.0 п.п.; следующий car_launch, +19.3 |
| **B** «кривой шар» | mjc_swimmer | pred (phys0.2) · reflex (phys0.2) | reflex (phys0.2) решил, pred (phys0.2) нет | тот же уровень, что в A. С phys0.2 он 4-й из 12 по эффекту `J` (+23.6; первый car_launch, +27.2) |
| **C** честное | catapult | RTC · reflex (learned) | RTC решил, reflex (learned) нет | reflex (learned) сильнее всего проигрывает RTC: −11.4 п.п. |

- **Если такого seed'а нет,** показываем seed 0 и пишем это в подписи.
- **Толчков нет,** как в B1. Модели мира — `results/b1/world_models` (маковские; в B1 были модели с пода, данные и seed'ы те же).
- **«Призрак» `ô` не делаем.** Нужны предсказанные состояния изнутри `eval_flow`, а он заморожен.
- **Поправка к плану в чате.** Там я написал, что swimmer — самый сильный уровень и для phys0.2. Это неверно: он 4-й. B оставлен на swimmer ради прямого сравнения с A, а ранг честно назван в подписи.

**Подпись под панелями (английский, ASCII)**, 4 строки, числа считает код:
1. уровень и предсказатель; d = 3, s = 5; у всех панелей один старт и один шум действий;
2. «Illustration, not evidence» и правило выбора эпизода с номером seed'а;
3. сколько решено из 16 seed'ов на каждой панели и цифры B1 по уровню (768 эпизодов): RTC / pred / reflex;
4. почему выбран уровень и среднее по 12 уровням: reflex минус RTC (+7.7 п.п. с моделью, +3.3 с phys0.2).

## Общие правила (для каждого исполнителя)

- **Запуск.** Все команды — из корня `/Users/alexanderkarpov/Desktop/M2R`. Python — только `uv run --offline …`: интернет мобильный. Тесты: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q …`.
- **Коммиты.** Сообщения на английском, в стиле истории. **Никогда не добавлять строку `Co-Authored-By`.** Без `git push`: пушит контроллер.
- **Долгие прогоны** (дольше ~5 мин) исполнитель не ждёт: пишет команду в отчёт, а запускает её контроллер в фоне.
- **Стиль.** Docstring и комментарии на английском, коротко, как в `src/probe.py`. Лишних абстракций нет. `eval_flow.py` не менять.
- **Публичность.** Никаких упоминаний применений вне метода.

## Карта файлов

| файл | ответственность |
|---|---|
| `src/video.py` | `_setup`, `_panel`, `_outcome`, `_label` вынесены из `main`; новые `PRESETS`, `pick`, `b1_table`, `demo`, `_rollout`, `_name`, `_footer`; подкоманды `kick` и `demo` (задача 2) |
| `tests/test_video.py` (новый) | правило выбора и числа и утверждения подписи против CSV B1 (задача 2) |
| `results/demo/` | `a.mp4`, `b.mp4`, `c.mp4`, `a.gif`, `a.csv`, `b.csv`, `c.csv` — исходы по seed'ам (задача 4, контроллер) |
| `README.md` | GIF и строка под ним, ссылки на b и c, команды (задача 5, контроллер) |

---

### Task 1: Зафиксировать правило публично (контроллер)

- [ ] **Step 1: Коммит плана на ветке `demo-video`**

```bash
git add docs/superpowers/plans/2026-09-26-demo-videos.md
git commit -m "plan: B1 demo videos, episode rule fixed before viewing (swimmer for A and B, catapult for C)"
```

- [ ] **Step 2: Push до любого прогона**

```bash
git push -u origin demo-video
```

### Task 2: `demo` в `src/video.py` (исполнитель)

**Files:**
- Modify: `src/video.py` (весь файл, 69 строк)
- Create: `tests/test_video.py`

- [ ] **Step 1: Написать падающие тесты**

`tests/test_video.py`:

```python
import numpy as np

import video


def test_pick_is_the_first_seed_where_hero_solves_and_rival_fails():
    solved = {"h": np.array([False, True, True, True]), "r": np.array([False, True, False, False])}
    assert video.pick(solved, "h", "r") == 2
    assert video.pick(solved, "r", "h") is None


def test_footer_numbers_and_level_claims_match_b1():
    t = video.b1_table()
    assert t.shape == (12, 11)
    assert t.loc["mjc_swimmer", ["realtime", "pred:learned", "reflex:learned"]].round(1).tolist() == [41.4, 42.2, 71.2]
    assert t.loc["mjc_swimmer", ["pred:phys0.2", "reflex:phys0.2"]].round(1).tolist() == [44.0, 67.6]
    assert t.loc["catapult", ["realtime", "pred:learned", "reflex:learned"]].round(1).tolist() == [41.0, 28.9, 29.6]
    assert round((t["reflex:learned"] - t["realtime"]).mean(), 1) == 7.7  # report section 8
    assert round((t["reflex:phys0.2"] - t["realtime"]).mean(), 1) == 3.3
    j = {p: t[f"reflex:{p}"] - t[f"pred:{p}"] for p in ("learned", "phys0.2")}  # the claims in PRESETS[...]["why"]
    assert j["learned"].idxmax() == "mjc_swimmer"  # a: strongest for J
    assert j["phys0.2"].rank(ascending=False)["mjc_swimmer"] == 4  # b: 4th of 12
    assert (t["reflex:learned"] - t["realtime"]).idxmin() == "catapult"  # c: worst against RTC
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_video.py`
Expected: FAIL, `AttributeError: module 'video' has no attribute 'pick'`

- [ ] **Step 3: Переписать `src/video.py`**

Поведение `main` (видео E4) не меняется: тот же код, только вынесенный в помощники.

```python
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
```

- [ ] **Step 4: Тесты проходят**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_video.py`
Expected: `2 passed`

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q`
Expected: `37 passed`

- [ ] **Step 5: CLI собирается**

Run: `uv run --offline src/video.py demo --help && uv run --offline src/video.py kick --help`
Expected: справка tyro с флагами `--preset {a,b,c}`, `--seeds`, `--out-dir`, `--gif-width` у `demo` и прежними флагами у `kick`. Если tyro пишет флаги с подчёркиванием, а не с дефисом, указать это в отчёте: команды задач 3–5 правит контроллер.

- [ ] **Step 6: Коммит**

```bash
git add src/video.py tests/test_video.py
git commit -m "feat(video): B1 demo presets a/b/c with a pre-registered episode rule, 16-seed counts and B1 numbers in the footer; kick/demo subcommands"
```

В отчёте — команды для задач 3–4. Сами прогоны исполнитель не запускает.

### Task 3: Smoke и push кода (контроллер)

- [ ] **Step 1: Видео E4 не изменилось** (фон, несколько минут)

```bash
uv run --offline src/video.py kick --seed 0 --out "$SCRATCH/video_kick.mp4"
cmp "$SCRATCH/video_kick.mp4" results/video_kick.mp4 && echo SAME
```

Если байты отличаются, сравнить кадры: `imageio.mimread` обоих файлов, `np.abs(a - b).max()` ≤ 2 (шум кодека).

- [ ] **Step 2: Короткий `demo`** (фон)

```bash
uv run --offline src/video.py demo --preset c --seeds 2 --out-dir "$SCRATCH/demo" --gif-width 720
```

Проверить:
- время на эпизод — по нему считается полный прогон;
- подпись: сохранить последний кадр mp4 в PNG и прочитать, вся ли строка помещается в ширину.

Смотрим только подпись. Правило уже запушено, так что просмотр ничего не меняет.

- [ ] **Step 3: Если нужно, поправить ширину переноса** (`0.6 * size` в `_footer`), коммит `fix(video): footer wrap`.

- [ ] **Step 4: Push кода до полного прогона**

```bash
git push
```

### Task 4: Полный прогон (контроллер, фон, ~30–60 мин)

- [ ] **Step 1: Три видео**

```bash
uv run --offline src/video.py demo --preset a --gif-width 720 > results/demo/log_a.txt 2>&1
uv run --offline src/video.py demo --preset b > results/demo/log_b.txt 2>&1
uv run --offline src/video.py demo --preset c > results/demo/log_c.txt 2>&1
```

(`mkdir -p results/demo` перед первым запуском.)

- [ ] **Step 2: Размер GIF ≤ 5 МБ и длительность 5–10 с**

```bash
ls -la results/demo/
uv run --offline python -c "from PIL import Image; g = Image.open('results/demo/a.gif'); print(g.n_frames, g.n_frames * g.info['duration'] / 1000, 's')"
```

Если больше 5 МБ, перезапустить A с `--gif-width 600`. Если длиннее 10 с (эпизоды около 256 шагов), так и оставить: это ≈ 9 с.

### Task 5: README (контроллер, текст показать владельцу до коммита)

- [ ] **Step 1: GIF наверху README, сразу после первого абзаца, и строка под ним.** Числа брать из `results/demo/log_a.txt`:

```markdown
![RTC, pred and reflex with a learned world model on mjc_swimmer](results/demo/a.gif)

*Illustration, not evidence.* `mjc_swimmer`, d = 3, s = 5, learned world model; RTC, pred and reflex from the same start. The episode is the first of seeds 0–15 where the reflex solves and pred fails, a rule fixed before viewing. Over those 16 seeds RTC, pred and reflex solve X, Y and Z. This is B1's strongest level for `J`: 41%, 42% and 71% of 768 episodes there, against a +7.7 pp mean gain of the reflex over RTC across all 12 levels.
```

- [ ] **Step 2: В пункте Demo** добавить ссылки на `results/demo/b.mp4` (кривая физика) и `results/demo/c.mp4` (уровень, где рефлекс проигрывает RTC), по одной фразе со счётом из `log_b.txt` и `log_c.txt`.

- [ ] **Step 3: Блок Reproduce:**
  - `uv run src/video.py kick --seed 0 --out results/video_kick.mp4`;
  - строка `uv run src/video.py demo --preset a --gif-width 720` (b, c аналогично);
  - `# 37 tests`.

- [ ] **Step 4: Показать текст владельцу. После его «ок» — коммит и push**

```bash
git add results/demo README.md
git commit -m "results: B1 demo videos (swimmer learned and wrong physics, catapult where RTC wins), README GIF"
git push
```

Слияние в `main` — по слову владельца.

## Оценка времени

| задача | кто | время |
|---|---|---|
| 1 | контроллер | 5 мин |
| 2 | исполнитель + 2 ревью | ~1 ч |
| 3 | контроллер, фон | ~15 мин |
| 4 | контроллер, фон | ~30–60 мин (112 эпизодов + 7 компиляций) |
| 5 | контроллер | ~15 мин |
