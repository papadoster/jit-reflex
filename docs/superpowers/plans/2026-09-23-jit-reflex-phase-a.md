# JIT Reflex, фаза A — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Проверить на Kinetix, может ли замороженная flow-policy между своими вызовами управлять роботом через собственную линеаризацию вдоль предсказанной траектории. Сначала оффлайн kill-test (E1, на Mac), затем замкнутый цикл (E2, на GPU), в конце write-up.

**Architecture:** M2R — это форк `Physical-Intelligence/real-time-chunking-kinetix` с сохранённой историей upstream.

- `src/reflex.py` — чистые функции рефлекса: якобиан, номинальный прогон, пакет, коррекция.
- `src/probe.py` — оффлайн-проверка E1 и две служебные подкоманды.
- `src/eval_flow.py` — в существующий eval-цикл добавляется новый метод `reflex`.
- `src/plot.py` — графики, таблицы и проверка Gate 2.

Предсказатель в фазе A — форк симулятора без шума, то есть оракул.

**Tech Stack:** Python 3.11, JAX 0.4.35 (CPU на macOS, CUDA на Linux), Flax NNX 0.10.2, Kinetix + Jax2D, tyro, pandas, matplotlib, pytest, uv.

Спек: `docs/superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md`. Раздел «§5» в этом плане означает раздел 5 спека.

**Правила для исполнителя:**

- Все команды запускаются из корня репозитория `/Users/alexanderkarpov/Desktop/M2R`.
- Сообщения коммитов — обычные, **без строки `Co-Authored-By`**.
- Поведение upstream-методов (`naive`, `realtime`, `bid`, `hard_masking`) не меняется.
- На **Gate 1** (Task 9) и **Gate 2** (Task 13) нужно остановиться, показать результат пользователю и дождаться решения.
- Первая компиляция JAX на CPU занимает минуты. Это нормально.

> **Поправка измерения E1** (спек, «Журнал изменений», 2026-09-23): для Tasks 5–7 источник истины — код в репозитории, а не листинги в этом плане (сдвинутый шум `reflex.shifted_noise`, пул `ρ` по `k`, сохранение `summary.csv` после каждого уровня, на 2 теста больше).

---

## Структура файлов

| Путь | Ответственность |
|---|---|
| `pyproject.toml`, `uv.lock`, `.python-version`, `.gitignore` | окружение: CPU-jax на macOS, CUDA на Linux, pytest |
| `src/model.py` (правка) | `FlowPolicy.action_from_noise`: сэмплинг с явно заданным шумом |
| `src/reflex.py` (новый) | `first_action_and_jacobian`, `nominal_obs`, `package`, `correct` |
| `src/probe.py` (новый) | E1: `errors`, `verdict`, сбор состояний, зонд; подкоманды `run`, `cost`, `kick-speed` |
| `src/eval_flow.py` (правка) | CLI-фильтры, несколько seed'ов, `ReflexMethodConfig`, толчки, `solved_length` |
| `src/train_expert.py` (правка) | `KickWrapper` |
| `src/plot.py` (новый) | `probe`, `success`, `table`, `gate2`, `wilson` |
| `src/video.py` (новый) | видео naive vs reflex бок о бок |
| `tests/test_reflex.py` (новый) | все тесты |
| `docs/results/*.md` | записки Gate 1 и Gate 2, отчёт |

---

### Task 1: Окружение — upstream, CPU-jax на Mac, pytest

**Files:**
- Modify: `pyproject.toml`, `.gitignore`
- Create: `.python-version`
- Regenerate: `uv.lock`

- [ ] **Step 1: Влить upstream с его историей**

```bash
git remote add upstream https://github.com/Physical-Intelligence/real-time-chunking-kinetix.git
git fetch upstream
git merge upstream/main --allow-unrelated-histories --no-edit
git submodule update --init
ls
```

Ожидаемый вывод `ls`: `LICENSE README.md docs pyproject.toml src third_party uv.lock worlds`.

- [ ] **Step 2: Сделать зависимости кроссплатформенными**

В `pyproject.toml` замени строку `"jax[cuda12]==0.4.35",` на `"jax==0.4.35",`.

Прямо **над** строкой `[tool.uv.sources]` вставь:

```toml
[tool.uv]
# CUDA only on Linux; macOS gets CPU jax. Overrides also replace Kinetix's own jax[cuda12_pip] pin.
override-dependencies = [
    "jax[cuda12]==0.4.35; sys_platform == 'linux'",
    "jax==0.4.35; sys_platform != 'linux'",
]

```

В конец файла добавь:

```toml

[tool.pytest.ini_options]
pythonpath = ["src"]
testpaths = ["tests"]
```

- [ ] **Step 3: Закрепить Python и пересобрать lock**

```bash
uv python pin 3.11
uv lock
uv sync
uv add --dev pytest
```

Ожидается, что `uv lock` напишет `Resolved N packages`, а `uv sync` завершится без ошибок.

**Если `uv lock` падает** на override, используй запасной путь: верни файлы и не ставь CUDA-пакеты.

```bash
git checkout pyproject.toml uv.lock
uv python pin 3.11
SKIP=$(grep -oE '^name = "(jax-cuda12-[a-z-]+|nvidia-[a-z0-9-]+)"' uv.lock | sed -E 's/^name = "(.*)"/--no-install-package \1/' | tr '\n' ' ')
uv sync $SKIP
uv pip install pytest
```

После этого повтори вставку `[tool.pytest.ini_options]` из Step 2. Дальше в этом варианте каждый `uv run` запускается как `UV_NO_SYNC=1 uv run …`. Запиши это в README (Task 14).

- [ ] **Step 4: Проверить импорт**

```bash
uv run python -c "import jax, flax, kinetix; print(jax.__version__, flax.__version__, jax.devices())"
```

Ожидается: `0.4.35 0.10.2 [CpuDevice(id=0)]`.

- [ ] **Step 5: `.gitignore`**

Допиши в конец `.gitignore`:

```
# M2R
checkpoints/
eval_output/
wandb/
results/smoke*/
```

- [ ] **Step 6: Commit**

```bash
git add pyproject.toml uv.lock .python-version .gitignore
git commit -m "build: CPU jax on macOS, CUDA on Linux, pytest"
```

---

### Task 2: Чекпойнты policies (bc/31, ~148 МБ)

**Files:** `checkpoints/bc/31/policies/*.pkl` (игнорируются git'ом).

- [ ] **Step 1: Скачать 12 файлов из публичного бакета**

```bash
mkdir -p checkpoints/bc/31/policies
for L in grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle chain_lander catcher_v3 trampoline car_launch; do
  curl -fsSL -o checkpoints/bc/31/policies/worlds_l_$L.pkl \
    https://storage.googleapis.com/rtc-assets/bc/31/policies/worlds_l_$L.pkl
done
```

- [ ] **Step 2: Проверить**

```bash
uv run python -c "import glob, os; fs = glob.glob('checkpoints/bc/31/policies/*.pkl'); assert len(fs) == 12 and all(os.path.getsize(f) == 12290914 for f in fs), fs; print('ok', len(fs))"
```

Ожидается: `ok 12`. Коммитить нечего.

---

### Task 3: `eval_flow.py` — выбор методов, задержек, горизонтов и seed'ов из CLI

Upstream всегда прогоняет полный sweep: 4 метода × 24 пары `(d, s)`. На CPU это часы. Нужен фильтр.

**Files:**
- Modify: `src/eval_flow.py` (функция `main` и новые определения над ней)
- Create: `tests/test_reflex.py`

- [ ] **Step 1: Написать падающий тест**

`tests/test_reflex.py`:

```python
import eval_flow


def test_horizons_for():
    assert eval_flow.horizons_for(0, 8, (), False) == [1, 2, 3, 4, 5, 6, 7, 8]
    assert eval_flow.horizons_for(2, 8, (), False) == [2, 3, 4, 5, 6]
    assert eval_flow.horizons_for(2, 8, (), True) == [2, 6]
    assert eval_flow.horizons_for(4, 8, (), True) == [4]
    assert eval_flow.horizons_for(3, 8, (1, 3, 5, 7), False) == [3, 5]
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: FAIL, `AttributeError: module 'eval_flow' has no attribute 'horizons_for'`.

- [ ] **Step 3: Реализация**

В `src/eval_flow.py` прямо над `def main(` вставь:

```python
METHODS = {
    "naive": NaiveMethodConfig(),
    "realtime": RealtimeMethodConfig(),
    "bid": BIDMethodConfig(),
    "hard_masking": RealtimeMethodConfig(prefix_attention_schedule="zeros"),
}


def horizons_for(delay: int, chunk_size: int, horizons: Sequence[int], minmax: bool) -> list[int]:
    """Execute horizons to evaluate at `delay`: explicit list, {min, max}, or upstream's full sweep.

    Keeps max(1, delay) <= s <= chunk_size - delay: upstream asserts s >= d, and s + d > H would execute padding zeros.
    """
    lo, hi = max(1, delay), chunk_size - delay
    if horizons:
        return [s for s in horizons if lo <= s <= hi]
    return sorted({lo, hi}) if minmax else list(range(lo, hi + 1))
```

Сигнатуру `main` приведи к такому виду. Список `level_paths` остаётся upstream'овским, `seed` заменяется на `seeds`, в конце добавляются четыре новых параметра:

```python
def main(
    run_path: str,
    config: EvalConfig = EvalConfig(),
    level_paths: Sequence[str] = (
        "worlds/l/grasp_easy.json",
        "worlds/l/catapult.json",
        "worlds/l/cartpole_thrust.json",
        "worlds/l/hard_lunar_lander.json",
        "worlds/l/mjc_half_cheetah.json",
        "worlds/l/mjc_swimmer.json",
        "worlds/l/mjc_walker.json",
        "worlds/l/h17_unicycle.json",
        "worlds/l/chain_lander.json",
        "worlds/l/catcher_v3.json",
        "worlds/l/trampoline.json",
        "worlds/l/car_launch.json",
    ),
    seeds: Sequence[int] = (0,),
    output_dir: str | None = "eval_output",
    methods: Sequence[str] = ("naive", "realtime", "bid", "hard_masking"),
    delays: Sequence[int] = (0, 1, 2, 3, 4),
    horizons: Sequence[int] = (),
    minmax: bool = False,
):
```

Тело `main` не трогай до строки `rngs = jax.random.split(jax.random.key(seed), len(level_paths))`. Эту строку и всё, что идёт после неё до конца функции (четыре скопированных блока методов и запись CSV), замени на:

```python
    results = collections.defaultdict(list)
    for seed in seeds:
        rngs = jax.random.split(jax.random.key(seed), len(level_paths))
        for inference_delay in delays:
            for execute_horizon in horizons_for(inference_delay, config.model.action_chunk_size, horizons, minmax):
                for name in methods:
                    print(f"{seed=} {name=} {inference_delay=} {execute_horizon=}")
                    c = dataclasses.replace(
                        config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=METHODS[name]
                    )
                    out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts))
                    for i in range(len(level_paths)):
                        for k, v in out.items():
                            results[k].append(v[i])
                        results["seed"].append(seed)
                        results["delay"].append(inference_delay)
                        results["method"].append(name)
                        results["level"].append(level_paths[i])
                        results["execute_horizon"].append(execute_horizon)
    pathlib.Path(output_dir).mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(pathlib.Path(output_dir) / "results.csv", index=False)
```

- [ ] **Step 4: Тест проходит**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `1 passed`.

- [ ] **Step 5: Smoke — upstream-eval на CPU (E0)**

```bash
uv run src/eval_flow.py --run-path checkpoints/bc --level-paths worlds/l/grasp_easy.json \
  --config.num-evals 16 --methods naive --delays 0 --horizons 1 --output-dir results/smoke
cat results/smoke/results.csv
```

Ожидается одна строка данных с `returned_episode_solved > 0`. Компиляция может занять несколько минут.

- [ ] **Step 6: Commit**

```bash
git add src/eval_flow.py tests/test_reflex.py
git commit -m "feat(eval): pick methods, delays, horizons and seeds from the CLI"
```

---

### Task 4: `FlowPolicy.action_from_noise`

Рефлексу нужен тот же шум `z`, что и у чанка, поэтому сэмплинг должен уметь принимать шум явно.

**Files:**
- Modify: `src/model.py` (метод `FlowPolicy.action`)
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Падающий тест**

В начало `tests/test_reflex.py` добавь импорты. Итоговый блок импортов:

```python
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np

import eval_flow
import model as _model
```

В конец файла добавь:

```python
def small_policy(obs_dim=5, action_dim=2, seed=0):
    """Tiny FlowPolicy with every weight randomized (zero-init AdaLN would make the blocks identity)."""
    cfg = _model.ModelConfig(channel_dim=16, channel_hidden_dim=32, token_hidden_dim=8, num_layers=2)
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=cfg, rngs=nnx.Rngs(seed))
    graphdef, state = nnx.split(policy)
    leaves, treedef = jax.tree.flatten(state.to_pure_dict())
    keys = jax.random.split(jax.random.key(seed + 1), len(leaves))
    state.replace_by_pure_dict(
        jax.tree.unflatten(treedef, [x + 0.3 * jax.random.normal(k, x.shape) for x, k in zip(leaves, keys)])
    )
    return nnx.merge(graphdef, state)


def test_action_from_noise_matches_action():
    policy = small_policy()
    key = jax.random.key(0)
    obs = jax.random.normal(jax.random.key(1), (3, 5))
    noise = jax.random.normal(key, (3, policy.action_chunk_size, policy.action_dim))
    np.testing.assert_allclose(policy.action(key, obs, 5), policy.action_from_noise(noise, obs, 5), atol=1e-6)
```

- [ ] **Step 2: Тест падает**

Run: `uv run pytest tests/test_reflex.py -q -k action_from_noise`
Expected: FAIL, `AttributeError: 'FlowPolicy' object has no attribute 'action_from_noise'`.

- [ ] **Step 3: Реализация**

В `src/model.py` замени метод `action` целиком на два метода:

```python
    def action(self, rng: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        noise = jax.random.normal(rng, shape=(obs.shape[0], self.action_chunk_size, self.action_dim))
        return self.action_from_noise(noise, obs, num_steps)

    def action_from_noise(self, noise: jax.Array, obs: jax.Array, num_steps: int) -> jax.Array:
        dt = 1 / num_steps

        def step(carry, _):
            x_t, time = carry
            v_t = self(obs, x_t, time)
            return (x_t + dt * v_t, time + dt), None

        (x_1, _), _ = jax.lax.scan(step, (noise, 0.0), length=num_steps)
        assert x_1.shape == (obs.shape[0], self.action_chunk_size, self.action_dim), x_1.shape
        return x_1
```

- [ ] **Step 4: Тесты проходят**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `2 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/model.py tests/test_reflex.py
git commit -m "feat(model): FlowPolicy.action_from_noise"
```

---

### Task 5: `src/reflex.py` — ядро рефлекса

**Files:**
- Create: `src/reflex.py`
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Падающие тесты**

Добавь `import reflex` в блок импортов и в конец файла:

```python
def test_first_action_and_jacobian_matches_finite_differences():
    policy = small_policy()
    noise = jax.random.normal(jax.random.key(2), (policy.action_chunk_size, policy.action_dim))
    obs = jax.random.normal(jax.random.key(3), (5,))
    a0, jac = reflex.first_action_and_jacobian(policy, noise, obs, 5)
    assert a0.shape == (2,) and jac.shape == (2, 5)

    def f(o):
        return policy.action_from_noise(noise[None], o[None], 5)[0, 0]

    np.testing.assert_allclose(a0, f(obs), atol=1e-6)
    v, eps = jax.random.normal(jax.random.key(4), (5,)), 1e-2
    fd = (f(obs + eps * v) - f(obs - eps * v)) / (2 * eps)
    np.testing.assert_allclose(jac @ v, fd, rtol=1e-2, atol=1e-3)


def test_nominal_obs_rolls_forward():
    def step(state, action):
        return state + action, state + action  # (obs, state)

    obs = reflex.nominal_obs(step, jnp.zeros(2), jnp.ones((3, 2)))
    np.testing.assert_allclose(obs, [[1, 1], [2, 2], [3, 3]])


def test_package_flags():
    policy = small_policy()
    B, H, A, O = 2, policy.action_chunk_size, policy.action_dim, 5
    noise = jax.random.normal(jax.random.key(5), (B, H, A))
    ref = jax.random.normal(jax.random.key(6), (B, H, O))
    chunk = jnp.ones((B, H, A))
    a0, jac = reflex.first_action_and_jacobian(policy, noise[1], ref[1, 3], 5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True)
    assert nom.shape == (B, H, A) and gain.shape == (B, H, A, O)
    np.testing.assert_allclose(nom[1, 3], a0, atol=1e-5)
    np.testing.assert_allclose(gain[1, 3], jac, atol=1e-5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=False)
    assert gain is None
    np.testing.assert_allclose(nom[1, 3], a0, atol=1e-5)

    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=False, feedback=False)
    assert gain is None
    np.testing.assert_allclose(nom, chunk)


def test_correct_is_identity_at_zero_deviation_and_clips():
    nom = jnp.array([0.1, -0.2])
    gain = jnp.array([[10.0, 0.0], [0.0, 1.0]])
    ref = jnp.array([1.0, 2.0])
    np.testing.assert_allclose(reflex.correct(nom, gain, ref, ref, 0.5), nom)
    out = reflex.correct(nom, gain, ref, ref + jnp.array([1.0, 0.1]), 0.5)
    np.testing.assert_allclose(out, [0.1 + 0.5, -0.2 + 0.1], atol=1e-6)


def test_forward_equivalents():
    assert reflex.forward_equivalents("naive") == 5
    assert reflex.forward_equivalents("realtime") == 15
    assert reflex.forward_equivalents("pred") == 45
    assert reflex.forward_equivalents("reflex") == reflex.forward_equivalents("reflex_chunk") == 525
```

- [ ] **Step 2: Тесты падают**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: ошибка сбора, `ModuleNotFoundError: No module named 'reflex'`.

- [ ] **Step 3: Реализация — `src/reflex.py`**

```python
"""JIT Reflex: linearize a frozen chunking flow policy along its predicted trajectory.

Between policy calls the robot acts with  a = nom + clip(gain @ (obs - ref)),  where ref is the predicted
observation, nom = pi(z, ref)[0] (or the chunk's action) and gain = d pi(z, o)[0] / d o at o = ref.
See docs/superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md (section 4).
"""

import jax
import jax.numpy as jnp


def first_action_and_jacobian(policy, noise, obs, num_steps):
    """noise [H, A], obs [O] -> (a0 [A], jac [A, O]): first action of pi(noise, obs) and its obs-Jacobian.

    Reverse mode: A (= 6 in Kinetix) VJPs instead of O (hundreds) JVPs.
    """

    def first_action(o):
        return policy.action_from_noise(noise[None], o[None], num_steps)[0, 0]

    a0, vjp = jax.vjp(first_action, obs)
    (jac,) = jax.vmap(vjp)(jnp.eye(a0.shape[0], dtype=a0.dtype))
    return a0, jac


def nominal_obs(step_fn, state, actions):
    """Noise-free rollout. step_fn(state, action) -> (obs, state); actions [T, ...] -> obs after each action [T, ...]."""

    def body(s, a):
        o, s = step_fn(s, a)
        return s, o

    return jax.lax.scan(body, state, actions)[1]


def package(policy, noise, ref, chunk, num_steps, requery: bool, feedback: bool, batch_size: int = 16):
    """Reflex package for one policy call, in chunk frame.

    noise [B, H, A] (the call's sampling noise), ref [B, H, O] (predicted obs per chunk index), chunk [B, H, A].
    Returns nom [B, H, A] (pi(noise, ref)[0] if requery else chunk) and gain [B, H, A, O] (None without feedback).
    Envs are processed batch_size at a time because Jacobian activations are large.
    """
    if not (requery or feedback):
        return chunk, None

    def per_env(x):
        n, r = x
        if feedback:
            return jax.vmap(lambda o: first_action_and_jacobian(policy, n, o, num_steps))(r)
        return policy.action_from_noise(jnp.broadcast_to(n, (r.shape[0], *n.shape)), r, num_steps)[:, 0], None

    a0, jac = jax.lax.map(per_env, (noise, ref), batch_size=min(batch_size, noise.shape[0]))
    return (a0 if requery else chunk), jac


def correct(nom, gain, ref, obs, max_correction):
    """a = nom + clip(gain @ (obs - ref), +-max_correction); broadcasts over leading dims."""
    delta = jnp.einsum("...ao,...o->...a", gain, obs - ref)
    return nom + jnp.clip(delta, -max_correction, max_correction)


def forward_equivalents(method: str, num_steps: int = 5, chunk_size: int = 8, action_dim: int = 6) -> int:
    """Network evaluations per policy call, one VJP counted as 2 (spec section 4, budget).

    Not counted: the reflex's per-step correction (action_dim x obs_dim MACs, negligible) and the predictor
    (a simulator fork here, a separate model in a real system).
    """
    S, H, A = num_steps, chunk_size, action_dim
    return {
        "naive": S,
        "reflex_off": S,
        "realtime": 3 * S,  # one guidance VJP per flow step
        "hard_masking": 3 * S,
        "bid": 16 * S,  # default n_samples=16, no weak policy
        "pred": S + H * S,  # chunk + pi at H predicted states
        "reflex": S + H * (S + 2 * A * S),  # + A VJPs through the whole flow at each predicted state
        "reflex_chunk": S + H * (S + 2 * A * S),
    }[method]
```

- [ ] **Step 4: Тесты проходят**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `7 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/reflex.py tests/test_reflex.py
git commit -m "feat(reflex): policy Jacobian, nominal rollout, reflex package, correction"
```

---

### Task 6: `src/probe.py` — метрики и правило решения E1

Это чистые функции без среды. Правило решения фиксируется в коде **до** запуска эксперимента (§5, E1).

**Files:**
- Create: `src/probe.py`
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Падающие тесты**

Добавь `import pandas as pd` и `import probe` в блок импортов и в конец файла:

```python
def test_errors_exact_for_linear_oracle():
    K, A, O = 3, 2, 4
    k = jax.random.split(jax.random.key(7), 4)
    jac = jax.random.normal(k[0], (K, A, O))
    a_ref = jax.random.normal(k[1], (K, A))
    nom_obs = jax.random.normal(k[2], (K, O))
    obs = nom_obs + 0.1 * jax.random.normal(k[3], (5, K, O))
    a_star = a_ref + jnp.einsum("kao,nko->nka", jac, obs - nom_obs)  # an oracle that IS linear
    e = probe.errors(a_ref, a_ref, jac, nom_obs, obs, a_star)
    np.testing.assert_allclose(e["lin"], 0, atol=1e-8)
    np.testing.assert_allclose(e["floor"], 0)
    assert e["pred"].shape == (5, K) and float(e["pred"].mean()) > 0


def _summary(rho_by_level, rel=0.5):
    rows = [
        {"level": lvl, "sigma": s, "k": k, "rho": r, "rel": rel}
        for lvl, r in rho_by_level.items()
        for s in (0.1, 0.2)
        for k in range(1, 8)
    ]
    return pd.DataFrame(rows)


def test_verdict_rules():
    assert probe.verdict(_summary({f"l{i}": 0.8 for i in range(12)}))["verdict"] == "GO"
    assert probe.verdict(_summary({f"l{i}": 0.1 for i in range(12)}))["verdict"] == "KILL"
    assert probe.verdict(_summary({f"l{i}": 0.3 for i in range(12)}))["verdict"] == "GRAY"
    seven_good = {f"l{i}": (0.9 if i < 7 else 0.1) for i in range(12)}  # median 0.9 but only 7/12 levels ok
    assert probe.verdict(_summary(seven_good))["verdict"] == "GRAY"
    weak_noise = _summary({f"l{i}": 0.1 for i in range(12)}, rel=0.05)  # deviations barely matter at sigma=0.1
    weak_noise.loc[weak_noise["sigma"] == 0.2, "rho"] = 0.8
    v = probe.verdict(weak_noise)
    assert v["sigma"] == 0.2 and v["verdict"] == "GO"
```

- [ ] **Step 2: Тесты падают**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `ModuleNotFoundError: No module named 'probe'`.

- [ ] **Step 3: Реализация — `src/probe.py`**

```python
"""E1 kill-test: is a frozen flow policy locally linear along its predicted trajectory?

For sampled states x_t: chunk A = pi(z, o_t); predicted obs o^_{t+k} (noise-free fork of the simulator);
noisy obs o_{t+k} (action noise sigma); oracle a* = pi(z, o_{t+k})[0] = what a fresh call would do now.
See docs/superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md (section 5, E1).
"""

import math

import jax.numpy as jnp
import pandas as pd

ERRORS = ("chunk", "pred", "lin", "chunk_lin", "floor", "dev")


def errors(chunk, a_ref, jac, nom_obs, obs, a_star):
    """Squared action errors against the oracle a* (spec E1).

    chunk, a_ref [K, A]; jac [K, A, O]; nom_obs [K, O]; obs [..., K, O]; a_star [..., K, A] -> dict of [..., K].
    """
    delta = obs - nom_obs
    lin = jnp.einsum("kao,...ko->...ka", jac, delta)

    def sq(x):
        return jnp.sum(jnp.square(x), axis=-1)

    return {
        "chunk": sq(chunk - a_star),
        "pred": sq(a_ref - a_star),
        "lin": sq(a_ref + lin - a_star),
        "chunk_lin": sq(chunk + lin - a_star),
        "floor": jnp.broadcast_to(sq(chunk - a_ref), a_star.shape[:-1]),
        "dev": jnp.sqrt(sq(delta)),
    }


def verdict(summary: pd.DataFrame) -> dict:
    """Pre-registered E1 decision rule (spec section 5). summary needs columns level, sigma, k, rho, rel."""
    near = summary[summary["k"].between(1, 4)]
    rel = float(near[near["sigma"] == 0.1]["rel"].mean())
    sigma = 0.1 if rel >= 0.1 else 0.2  # relevance check: deviations must actually change the policy's decisions
    per_level = near[near["sigma"] == sigma].groupby("level")["rho"].mean()
    r, n_ok, n = float(per_level.median()), int((per_level >= 0.3).sum()), len(per_level)
    if r >= 0.5 and n_ok >= math.ceil(2 * n / 3):
        v = "GO"
    elif r < 0.2:
        v = "KILL"
    else:
        v = "GRAY"
    return {"verdict": v, "sigma": sigma, "R": r, "levels_ok": n_ok, "levels": n, "rel": rel}
```

- [ ] **Step 4: Тесты проходят**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `9 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/probe.py tests/test_reflex.py
git commit -m "feat(probe): E1 error metrics and pre-registered verdict rule"
```

---

### Task 7: `src/probe.py` — конвейер зонда и подкоманда `cost`

**Files:**
- Modify: `src/probe.py`

- [ ] **Step 1: Импорты**

Блок импортов `src/probe.py` замени на:

```python
import json
import math
import pathlib
import pickle
import time
from typing import Sequence

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import kinetix.environment.wrappers as wrappers
import numpy as np
import pandas as pd
import tyro

import model as _model
import reflex
import train_expert

LEVELS = (
    "worlds/l/grasp_easy.json",
    "worlds/l/catapult.json",
    "worlds/l/cartpole_thrust.json",
    "worlds/l/hard_lunar_lander.json",
    "worlds/l/mjc_half_cheetah.json",
    "worlds/l/mjc_swimmer.json",
    "worlds/l/mjc_walker.json",
    "worlds/l/h17_unicycle.json",
    "worlds/l/chain_lander.json",
    "worlds/l/catcher_v3.json",
    "worlds/l/trampoline.json",
    "worlds/l/car_launch.json",
)
```

Строка `ERRORS = …` остаётся под импортами.

- [ ] **Step 2: Конвейер — дописать в конец `src/probe.py`**

```python
def setup(level_paths: Sequence[str]):
    """AutoReplay env without noise, env params, stacked levels, obs_dim, action_dim; same config as eval_flow.main."""
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    levels = train_expert.load_levels(level_paths, static_env_params, env_params)
    env = wrappers.AutoReplayWrapper(
        kenv.make_kinetix_env_from_name(
            "Kinetix-Symbolic-Continuous-v1",
            static_env_params=static_env_params.replace(screen_dim=train_expert.SCREEN_DIM),
        )
    )
    level0 = jax.tree.map(lambda x: x[0], levels)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level0, env_params)[0].shape[-1]
    return env, env_params, levels, obs_dim, env.action_space(env_params).shape[0]


def load_state_dict(run_path: str, level_path: str, step: int = -1):
    level_name = level_path.replace("/", "_").replace(".json", "")
    dirs = sorted(
        (p for p in pathlib.Path(run_path).iterdir() if p.is_dir() and p.name.isdigit()), key=lambda p: int(p.name)
    )
    with (dirs[step] / "policies" / f"{level_name}.pkl").open("rb") as f:
        return pickle.load(f)


def make_policy(state_dict, obs_dim: int, action_dim: int):
    policy = _model.FlowPolicy(obs_dim=obs_dim, action_dim=action_dim, config=_model.ModelConfig(), rngs=nnx.Rngs(0))
    graphdef, state = nnx.split(policy)
    state.replace_by_pure_dict(state_dict)
    return nnx.merge(graphdef, state)


def collect(env, env_params, policy, level, key, num_envs: int, horizon: int, sigma: float, num_steps: int):
    """Naive chunked rollouts with Gaussian action noise. Returns (obs, state, alive) at every chunk boundary, [C, E, ...].

    alive = the first episode has not ended yet (later episodes are AutoReplay repeats and are ignored).
    """
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(
        jax.random.split(k_reset, num_envs), level, env_params
    )
    env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def run_chunk(carry, key):
        obs, state, alive = carry
        k_act, k_noise, k_env = jax.random.split(key, 3)
        actions = policy.action(k_act, obs, num_steps)[:, :horizon]
        actions = actions + sigma * jax.random.normal(k_noise, actions.shape)

        def one_step(c, xs):
            obs, state, alive = c
            action, k = xs
            obs, state, _, done, _ = env_step(jax.random.split(k, num_envs), state, action, env_params)
            return (obs, state, alive & ~done), None

        carry, _ = jax.lax.scan(
            one_step, (obs, state, alive), (actions.swapaxes(0, 1), jax.random.split(k_env, horizon))
        )
        return carry, (obs, state, alive)

    n_chunks = env_params.max_timesteps // horizon
    return jax.lax.scan(run_chunk, (obs, state, jnp.ones(num_envs, bool)), jax.random.split(k_run, n_chunks))[1]


def sample(boundaries, key, num_states: int):
    """num_states random alive (obs, state) pairs from collect() output."""
    obs, state, alive = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), boundaries)
    idx = jax.random.choice(key, alive.shape[0], (num_states,), replace=False, p=alive / alive.sum())
    return obs[idx], jax.tree.map(lambda x: x[idx], state)


def probe_one(env, env_params, policy, state, obs, key, sigmas: Sequence[float], num_draws: int, num_steps: int):
    """E1 errors for one state: ERRORS + 'valid', each [S, M, K] (sigmas, draws, offsets k = 1..H-1)."""
    H, A = policy.action_chunk_size, policy.action_dim
    K = H - 1
    k_z, k_nom, k_eps, k_roll = jax.random.split(key, 4)
    z = jax.random.normal(k_z, (H, A))
    chunk = policy.action_from_noise(z[None], obs[None], num_steps)[0]  # [H, A]

    def rollout(actions, key):  # obs after each of the K actions, and "the episode ended by then"
        def one_step(c, xs):
            st, ended = c
            action, k = xs
            o, st, _, done, _ = env.step(k, st, action, env_params)
            return (st, ended | done), (o, ended | done)

        return jax.lax.scan(one_step, (state, jnp.bool_(False)), (actions, jax.random.split(key, K)))[1]

    nom_obs, nom_ended = rollout(chunk[:K], k_nom)  # o^_{t+1..t+K}
    a_ref, jac = jax.vmap(lambda o: reflex.first_action_and_jacobian(policy, z, o, num_steps))(nom_obs)

    S = len(sigmas)
    noise = jax.random.normal(k_eps, (S, num_draws, K, A))
    noisy_actions = chunk[:K] + jnp.asarray(sigmas)[:, None, None, None] * noise
    roll_keys = jax.random.split(k_roll, S * num_draws).reshape(S, num_draws)
    obs_t, ended = jax.vmap(jax.vmap(rollout))(noisy_actions, roll_keys)  # [S, M, K, O], [S, M, K]

    flat = obs_t.reshape(-1, obs_t.shape[-1])
    a_star = policy.action_from_noise(jnp.broadcast_to(z, (flat.shape[0], H, A)), flat, num_steps)[:, 0]
    out = errors(chunk[1:], a_ref, jac, nom_obs, obs_t, a_star.reshape(*obs_t.shape[:-1], A))
    out["valid"] = ~(ended | nom_ended)
    return out


def summarize(level: str, res: dict, sigmas: Sequence[float]) -> pd.DataFrame:
    """Mean errors over valid (state, draw) pairs per sigma and offset k, plus rho, rho_total, rel."""
    valid = res["valid"]  # [N, S, M, K]
    rows = []
    for si, sigma in enumerate(sigmas):
        for k in range(valid.shape[-1]):
            v = valid[:, si, :, k]
            means = {name: float(res[name][:, si, :, k][v].mean()) for name in ERRORS}
            rows.append({"level": level, "sigma": sigma, "k": k + 1, "n": int(v.sum()), **means})
    df = pd.DataFrame(rows)
    df["rho"] = 1 - df["lin"] / df["pred"]
    df["rho_total"] = 1 - df["lin"] / df["chunk"]
    df["rel"] = df["pred"] / df["chunk"]
    return df


def run(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = LEVELS,
    num_envs: int = 64,
    collect_horizon: int = 4,
    num_states: int = 256,
    num_draws: int = 4,
    sigmas: Sequence[float] = (0.05, 0.1, 0.2, 0.4),
    num_flow_steps: int = 5,
    batch_size: int = 16,  # states per vmapped batch; lower it if RAM runs out
    seed: int = 0,
    output_dir: str = "results/probe",
):
    """E1 kill-test (spec section 5): writes summary.csv and verdict.json to output_dir."""
    env, env_params, levels, obs_dim, action_dim = setup(level_paths)
    sigmas = tuple(sigmas)

    @jax.jit
    def probe_level(state_dict, level, key):
        policy = make_policy(state_dict, obs_dim, action_dim)
        k_collect, k_sample, k_probe = jax.random.split(key, 3)
        boundaries = collect(
            env, env_params, policy, level, k_collect, num_envs, collect_horizon,
            train_expert.ACTION_NOISE_STD, num_flow_steps,
        )
        obs, state = sample(boundaries, k_sample, num_states)

        def one(x):
            return probe_one(env, env_params, policy, x[0], x[1], x[2], sigmas, num_draws, num_flow_steps)

        res = jax.lax.map(one, (state, obs, jax.random.split(k_probe, num_states)), batch_size=batch_size)
        return res, boundaries[2].sum()

    frames = []
    for i, level_path in enumerate(level_paths):
        level = jax.tree.map(lambda x: x[i], levels)
        res, n_alive = jax.device_get(
            probe_level(load_state_dict(run_path, level_path), level, jax.random.key(seed + i))
        )
        print(f"{level_path}: {int(n_alive)} alive boundary states")
        assert n_alive >= num_states, "too few alive states: sample() would pick finished episodes; raise --num-envs"
        frames.append(summarize(level_path, res, sigmas))
    summary = pd.concat(frames, ignore_index=True)
    out = pathlib.Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out / "summary.csv", index=False)
    v = verdict(summary)
    (out / "verdict.json").write_text(json.dumps(v, indent=2))
    near = summary[(summary["sigma"] == v["sigma"]) & summary["k"].between(1, 4)]
    print(near.groupby("level")[["rho", "rho_total", "rel", "floor"]].mean().round(3).to_string())
    print(json.dumps(v))


def cost(
    run_path: str = "checkpoints/bc",
    level_path: str = LEVELS[0],
    batch: int = 1,
    num_flow_steps: int = 5,
    delay: int = 2,
    horizon: int = 6,
    repeats: int = 20,
    out: str | None = None,
):
    """Per-call cost of every E2 method: network evaluations (analytic), GFLOP and measured latency (spec section 4).

    GFLOP of one network evaluation comes from cost_analysis on a loop-free call: whole calls contain lax.scan loops,
    whose bodies XLA may count only once. The predictor (a simulator fork here) is not included.
    """
    _, _, _, obs_dim, action_dim = setup([level_path])
    policy = make_policy(load_state_dict(run_path, level_path), obs_dim, action_dim)
    H = policy.action_chunk_size
    key = jax.random.key(0)
    noise = jax.random.normal(key, (batch, H, action_dim))
    obs, ref, prev = jnp.zeros((batch, obs_dim)), jnp.zeros((batch, H, obs_dim)), jnp.zeros((batch, H, action_dim))
    one_eval = jax.jit(lambda o, x: policy(o, x, jnp.zeros(())))
    analysis = one_eval.lower(obs[:1], noise[:1]).compile().cost_analysis()
    gflop_per_eval = (analysis[0] if isinstance(analysis, list) else analysis)["flops"] / 1e9

    def with_package(requery, feedback):
        def call(noise, obs, ref, prev):
            chunk = policy.action_from_noise(noise, obs, num_flow_steps)
            return chunk, reflex.package(policy, noise, ref, chunk, num_flow_steps, requery, feedback)

        return call

    calls = {
        "naive": lambda noise, obs, ref, prev: policy.action_from_noise(noise, obs, num_flow_steps),
        "realtime": lambda noise, obs, ref, prev: policy.realtime_action(
            key, obs, num_flow_steps, prev, delay, H - horizon, "exp", 5.0
        ),
        "pred": with_package(True, False),
        "reflex": with_package(True, True),
        "reflex_chunk": with_package(False, True),
    }
    rows = []
    for name, fn in calls.items():
        f = jax.jit(fn)
        jax.block_until_ready(f(noise, obs, ref, prev))
        start = time.perf_counter()
        for _ in range(repeats):
            jax.block_until_ready(f(noise, obs, ref, prev))
        evals = reflex.forward_equivalents(name, num_flow_steps, H, action_dim)
        rows.append({
            "method": name,
            "batch": batch,
            "evals_per_call": evals,
            "gflop_per_call": evals * gflop_per_eval * batch,
            "ms_per_call": (time.perf_counter() - start) / repeats * 1e3,
        })
    df = pd.DataFrame(rows)
    df["latency_vs_realtime"] = df["ms_per_call"] / df.loc[df["method"] == "realtime", "ms_per_call"].item()
    print(f"one network evaluation (batch 1): {gflop_per_eval:.4f} GFLOP")
    print(df.round(3).to_string(index=False))
    if out:
        pathlib.Path(out).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"run": run, "cost": cost})
```

- [ ] **Step 3: Тесты не сломались**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `9 passed`.

- [ ] **Step 4: Smoke-прогон зонда на одном уровне**

```bash
uv run src/probe.py run --level-paths worlds/l/grasp_easy.json --num-envs 8 --num-states 16 --num-draws 2 \
  --batch-size 8 --output-dir results/smoke-probe
uv run python -c "
import pandas as pd
df = pd.read_csv('results/smoke-probe/summary.csv')
assert len(df) == 4 * 7, len(df)
assert {'chunk', 'pred', 'lin', 'chunk_lin', 'floor', 'dev', 'rho', 'rho_total', 'rel', 'n'} <= set(df.columns)
print(df[['sigma', 'k', 'n', 'rho', 'rel']].to_string())"
```

Ожидается таблица из 28 строк с `n > 0` в большинстве строк. Числа `rho` пока ни о чём не говорят: слишком мало данных.

- [ ] **Step 5: Smoke `cost`**

```bash
uv run src/probe.py cost
```

Ожидается строка `one network evaluation (batch 1): … GFLOP` и таблица из 5 методов. `evals_per_call`: naive 5, realtime 15, pred 45, reflex и reflex_chunk 525.

- [ ] **Step 6: Commit**

```bash
git add src/probe.py
git commit -m "feat(probe): E1 pipeline (collect, sample, probe, summarize) and cost subcommand"
```

---

### Task 8: `src/plot.py` — графики, таблицы, Gate 2

**Files:**
- Create: `src/plot.py`
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Падающие тесты**

Добавь `import plot` в блок импортов и в конец файла:

```python
def test_wilson_interval():
    lo, hi = plot.wilson(np.array([0.5]), np.array([100]))
    assert 0.40 < lo[0] < 0.41 and 0.59 < hi[0] < 0.60


def test_gate2_detects_go(tmp_path):
    rows = []
    for seed in (0, 1, 2):
        for d in (1, 2, 3, 4):
            for s in sorted({max(1, d), 8 - d}):
                for method, rate in (("naive", 0.5), ("realtime", 0.6), ("pred", 0.55), ("reflex", 0.75)):
                    rows.append({"seed": seed, "delay": d, "execute_horizon": s, "method": method,
                                 "level": "l", "returned_episode_solved": rate})
    pd.DataFrame(rows).to_csv(tmp_path / "main.csv", index=False)
    out = plot.gate2(str(tmp_path / "main.csv"), str(tmp_path / "no-kicks*/results.csv"))
    assert out["a"] and out["b"] and out["pred_check"] and not out["c"]
    assert out["decision"] == "GO"
    assert out["b_work_ratio"] > 1 and "fewer policy calls" in out["b_claim"]  # fewer calls != less compute
```

- [ ] **Step 2: Тесты падают**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `ModuleNotFoundError: No module named 'plot'`.

- [ ] **Step 3: Реализация — `src/plot.py`**

```python
"""Figures and tables for E1 (probe) and E2 (closed loop), and the Gate 2 check (spec section 5)."""

import glob
import json
import math
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import tyro  # noqa: E402

import reflex  # noqa: E402


def probe(summary_csv: str = "results/probe/summary.csv", out: str = "results/probe/rho.png"):
    """rho(k) per level, one line per sigma, verdict in the title."""
    df = pd.read_csv(summary_csv)
    levels = sorted(df["level"].unique())
    cols = 4
    rows = math.ceil(len(levels) / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharex=True, sharey=True, squeeze=False)
    for ax, level in zip(axes.flat, levels):
        for sigma, g in df[df["level"] == level].groupby("sigma"):
            ax.plot(g["k"], g["rho"], marker="o", label=f"σ={sigma}")
        ax.axhline(0.5, ls="--", c="gray", lw=0.8)
        ax.axhline(0.2, ls=":", c="gray", lw=0.8)
        ax.set_title(pathlib.Path(level).stem)
        ax.set_ylim(-0.5, 1.05)
    axes.flat[0].legend(fontsize=7)
    v = json.loads((pathlib.Path(summary_csv).parent / "verdict.json").read_text())
    fig.suptitle(
        f"E1: ρ(k) = 1 − e_lin/e_pred · {v['verdict']} (R={v['R']:.2f}, σ={v['sigma']}, "
        f"ok {v['levels_ok']}/{v['levels']})"
    )
    fig.supxlabel("k (steps after the policy call)")
    fig.supylabel("ρ")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f"saved {out}")


def wilson(p, n, z=1.96):
    """95% Wilson interval for proportions p over n trials (arrays)."""
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def success(
    results_glob: str = "results/eval/main/results.csv",
    oracle_glob: str = "results/eval/oracle/results.csv",
    num_evals: int = 256,
    out: str = "results/eval/main/success.png",
):
    """Solve rate (mean over levels and seeds) vs delay, for s = d and s = 8 - d; oracle d=0, s=1 dashed."""
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(results_glob))])
    df["horizon"] = np.where(df["execute_horizon"] == df["delay"].clip(lower=1), "s = d", "s = 8 − d")
    g = df.groupby(["horizon", "method", "delay"])["returned_episode_solved"].agg(["mean", "count"]).reset_index()
    oracle_files = sorted(glob.glob(oracle_glob))
    oracle = pd.concat([pd.read_csv(f) for f in oracle_files])["returned_episode_solved"].mean() if oracle_files else None
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for ax, (horizon, h) in zip(axes, g.groupby("horizon")):
        for method, m in h.groupby("method"):
            lo, hi = wilson(m["mean"].to_numpy(), m["count"].to_numpy() * num_evals)
            ax.errorbar(m["delay"], m["mean"], yerr=[m["mean"] - lo, hi - m["mean"]], marker="o", capsize=3, label=method)
        if oracle is not None:
            ax.axhline(oracle, ls="--", c="black", lw=0.8, label="oracle d=0 s=1")
        ax.set_title(horizon)
        ax.set_xlabel("inference delay d")
    axes[0].set_ylabel("solve rate (mean over levels)")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(g.pivot_table(index=["horizon", "delay"], columns="method", values="mean").round(3).to_string())
    calls = df.drop_duplicates(["delay", "execute_horizon"])[["delay", "execute_horizon"]].copy()
    calls["calls_per_episode"] = np.ceil(256 / calls["execute_horizon"]).astype(int)
    print(calls.sort_values(["delay", "execute_horizon"]).to_string(index=False))
    # work = calls x network evaluations per call: fewer calls must not hide more compute
    df["work_per_episode"] = [
        reflex.forward_equivalents(m) * math.ceil(256 / s) for m, s in zip(df["method"], df["execute_horizon"])
    ]
    print("network evaluations per episode:")
    print(df.pivot_table(index=["delay", "execute_horizon"], columns="method", values="work_per_episode").to_string())
    print(f"saved {out}")


def table(results_glob: str):
    """Mean solve rate and solved-episode length over levels and seeds per (run dir, method, delay, horizon)."""
    df = pd.concat([pd.read_csv(f).assign(run=pathlib.Path(f).parent.name) for f in sorted(glob.glob(results_glob))])
    cols = [c for c in ("returned_episode_solved", "solved_length") if c in df.columns]
    print(df.groupby(["run", "method", "delay", "execute_horizon"])[cols].mean().round(3).to_string())


def gate2(
    main_csv: str = "results/eval/main/results.csv",
    kick_glob: str = "results/eval/kick*[0-9]/results.csv",
) -> dict:
    """Spec Gate 2: conditions (a), (b), (c) and the pred check. Solve rates are means over levels."""
    df = pd.read_csv(main_csv)
    t = df.pivot_table(
        index=["seed", "delay", "execute_horizon"], columns="method", values="returned_episode_solved"
    ).reset_index()
    long = t[(t["delay"] >= 2) & (t["execute_horizon"] == 8 - t["delay"])]
    a_by_seed = (long["reflex"] - long["realtime"]).groupby(long["seed"]).mean()
    a = bool(a_by_seed.mean() >= 0.05 and (a_by_seed > 0).all())
    short = t[t["execute_horizon"] == t["delay"].clip(lower=1)].set_index(["seed", "delay"])["realtime"]
    far = t[t["delay"].between(1, 3) & (t["execute_horizon"] == 8 - t["delay"])].set_index(["seed", "delay"])["reflex"]
    b_gap = float((far - short.reindex(far.index)).mean())
    # (b) counts calls; report the work next to it (spec: fewer calls must not hide more compute)
    calls_saved = float(np.mean([math.ceil(256 / max(1, d)) / math.ceil(256 / (8 - d)) for d in (1, 2, 3)]))
    work_ratio = float(np.mean([
        reflex.forward_equivalents("reflex") * math.ceil(256 / (8 - d))
        / (reflex.forward_equivalents("realtime") * math.ceil(256 / max(1, d)))
        for d in (1, 2, 3)
    ]))
    total_gain = float((t["reflex"] - t["naive"]).mean())
    j_gain = float((t["reflex"] - t["pred"]).mean())
    pred_check = bool(total_gain > 0 and j_gain >= 0.5 * total_gain)
    c_gap = None  # no kick runs; None keeps gate2.json valid JSON
    kick_files = sorted(glob.glob(kick_glob))
    if kick_files:
        k = pd.concat([pd.read_csv(f) for f in kick_files]).pivot_table(
            index=["kick_std", "seed", "execute_horizon"], columns="method", values="returned_episode_solved"
        )
        c_gap = float((k["reflex"] - k[["naive", "realtime"]].max(axis=1)).mean())
    out = {
        "a": a, "a_gap": float(a_by_seed.mean()),
        "b": bool(b_gap >= -0.02), "b_gap": b_gap, "b_calls_saved": calls_saved, "b_work_ratio": work_ratio,
        "c": bool(c_gap is not None and c_gap >= 0.05), "c_gap": c_gap,
        "pred_check": pred_check, "total_gain": total_gain, "j_gain": j_gain,
    }
    if out["b"]:
        out["b_claim"] = f"{calls_saved:.1f}x fewer policy calls at {work_ratio:.1f}x the network evaluations of realtime"
    out["decision"] = "GO" if (out["a"] or out["b"] or out["c"]) and pred_check else "NEGATIVE"
    print(json.dumps(out, indent=2))
    return out


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"probe": probe, "success": success, "table": table, "gate2": gate2})
```

- [ ] **Step 4: Тесты проходят**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `11 passed`.

- [ ] **Step 5: Commit**

```bash
git add src/plot.py tests/test_reflex.py
git commit -m "feat(plot): E1/E2 figures, tables and the Gate 2 check"
```

---

### Task 9: E1 — полный прогон kill-test и **Gate 1 (STOP)**

**Files:**
- Create: `results/probe/{summary.csv,verdict.json,rho.png,log.txt}`, `docs/results/probe.md`

- [ ] **Step 1: Прогон на всех 12 уровнях (Mac CPU, десятки минут)**

```bash
mkdir -p results/probe
uv run src/probe.py run 2>&1 | tee results/probe/log.txt
```

Если не хватает памяти, повтори с `--batch-size 8`. В конце вывода — таблица по уровням и JSON вердикта.

- [ ] **Step 2: График**

```bash
uv run src/plot.py probe
```

Expected: `saved results/probe/rho.png`.

- [ ] **Step 3: Записка `docs/results/probe.md`**

Порядок содержимого:

1. Первая строка — JSON вердикта из `results/probe/verdict.json`, дословно.
2. Команда запуска и параметры: `N = 256`, `M = 4`, `σ ∈ {0.05, 0.1, 0.2, 0.4}`, seed 0.
3. Таблица по уровням из `log.txt`, дословно.
4. Картинка: `![rho](../../results/probe/rho.png)`.
5. 3–6 пунктов с ответами на вопросы:
   - какие уровни проходят (`ρ ≥ 0.3`) и какие нет;
   - как `ρ` меняется с ростом `k` и `σ`;
   - что показывает `rel` (важны ли вообще отклонения);
   - насколько велик `floor` по сравнению с `pred`;
   - floor со сдвинутым шумом: насколько план спорит сам с собой;
   - итоговое решение по правилу из §5.

- [ ] **Step 4: Commit**

```bash
git add results/probe docs/results/probe.md
git commit -m "results: E1 kill-test"
```

- [ ] **Step 5: GATE 1 — остановиться**

Покажи пользователю вердикт, картинку и записку.

- **KILL** → пропусти Tasks 10–13 и переходи к Task 14 (write-up только по E1).
- **GO** → Task 10.
- **GRAY** → решает пользователь.

---

### Task 10: `KickWrapper` и калибровка `v_med` (E2, толчки)

**Files:**
- Modify: `src/train_expert.py` (сразу после класса `NoisyActionWrapper`), `src/probe.py`
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Падающий тест**

Добавь `import train_expert` в блок импортов и в конец файла:

```python
def test_kick_moves_only_active_dynamic_bodies():
    env, env_params, levels, _, _ = probe.setup(["worlds/l/grasp_easy.json"])
    _, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    raw = state.env_state
    kicked = train_expert.KickWrapper(env, prob=1.0, std=1.0).kick(jax.random.key(1), raw)
    for before, after in ((raw.polygon, kicked.polygon), (raw.circle, kicked.circle)):
        moved = np.any(np.asarray(after.velocity != before.velocity), axis=-1)
        dynamic = np.asarray((before.inverse_mass > 0) & before.active)
        np.testing.assert_array_equal(moved, dynamic)
        assert dynamic.any() or before is raw.circle  # the level has at least one dynamic polygon
```

- [ ] **Step 2: Тест падает**

Run: `uv run pytest tests/test_reflex.py -q -k kick`
Expected: FAIL, `AttributeError: module 'train_expert' has no attribute 'KickWrapper'`.

- [ ] **Step 3: Реализация — в `src/train_expert.py` после `NoisyActionWrapper`**

```python
class KickWrapper(wrappers.UnderspecifiedEnvWrapper):
    """E2 perturbation: with probability `prob` per step, add N(0, std^2) velocity to every active dynamic body."""

    def __init__(self, env, prob: float, std: float):
        super().__init__(env)
        self.prob = prob
        self.std = std

    def kick(self, key, state):
        key_on, key_poly, key_circle = jax.random.split(key, 3)
        on = jax.random.bernoulli(key_on, self.prob)

        def push(body, k):
            dynamic = (body.inverse_mass > 0) & body.active
            dv = self.std * jax.random.normal(k, body.velocity.shape) * (on & dynamic)[..., None]
            return body.replace(velocity=body.velocity + dv)

        # ponytail: kicks the agent's own bodies too; restrict to non-agent bodies if "only objects move" matters
        return state.replace(polygon=push(state.polygon, key_poly), circle=push(state.circle, key_circle))

    def step_env(self, key, state, action, params):
        key_kick, key_step = jax.random.split(key)
        return self._env.step_env(key_step, self.kick(key_kick, state), action, params)

    def reset_to_level(self, rng, level, params):
        return self._env.reset_to_level(rng, level, params)

    def action_space(self, params):
        return self._env.action_space(params)
```

- [ ] **Step 4: Подкоманда `kick-speed` в `src/probe.py`**

Перед блоком `if __name__ == "__main__":` вставь:

```python
def kick_speed(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = LEVELS,
    num_envs: int = 32,
    num_flow_steps: int = 5,
    seed: int = 0,
):
    """v_med: median speed of active dynamic bodies in naive d=0, s=1 rollouts (spec E2: kick_std = c * v_med)."""
    env, env_params, levels, obs_dim, action_dim = setup(level_paths)

    @jax.jit
    def speeds(state_dict, level, key):
        policy = make_policy(state_dict, obs_dim, action_dim)
        _, state, alive = collect(
            env, env_params, policy, level, key, num_envs, 1, train_expert.ACTION_NOISE_STD, num_flow_steps
        )
        bodies = (state.env_state.polygon, state.env_state.circle)
        return [
            (jnp.linalg.norm(b.velocity, axis=-1), (b.inverse_mass > 0) & b.active & alive[..., None]) for b in bodies
        ]

    samples = []
    for i, level_path in enumerate(level_paths):
        level = jax.tree.map(lambda x: x[i], levels)
        out = jax.device_get(speeds(load_state_dict(run_path, level_path), level, jax.random.key(seed + i)))
        samples += [speed[mask] for speed, mask in out]
    print(f"v_med = {float(np.median(np.concatenate(samples))):.4f}")
```

Строку CLI замени на:

```python
    tyro.extras.subcommand_cli_from_dict({"run": run, "cost": cost, "kick-speed": kick_speed})
```

- [ ] **Step 5: Тесты проходят**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `12 passed`.

- [ ] **Step 6: Smoke `kick-speed` на одном уровне**

```bash
uv run src/probe.py kick-speed --level-paths worlds/l/grasp_easy.json --num-envs 8
```

Expected: строка `v_med = <положительное число>`.

- [ ] **Step 7: Commit**

```bash
git add src/train_expert.py src/probe.py tests/test_reflex.py
git commit -m "feat(env): KickWrapper perturbations and v_med calibration"
```

---

### Task 11: Метод `reflex` в `eval_flow.py`, толчки, `solved_length`

**Files:**
- Modify: `src/eval_flow.py`: импорты, конфиги, `METHODS`, функция `eval` целиком, `main`.

- [ ] **Step 1: Импорт и конфиги**

После строки `import model as _model` добавь `import reflex`.

Над `class EvalConfig` добавь:

```python
@dataclasses.dataclass(frozen=True)
class ReflexMethodConfig:
    requery: bool = True  # nominal action = pi(z, predicted obs)[0] instead of the chunk's action
    feedback: bool = True  # add clip(J @ (obs - predicted obs)) at every step
    max_correction: float = 1.0
    package_batch: int = 16  # envs per Jacobian batch (memory knob)
```

В `EvalConfig` поле `method` замени на следующее и добавь под ним два поля:

```python
    method: NaiveMethodConfig | RealtimeMethodConfig | BIDMethodConfig | ReflexMethodConfig = NaiveMethodConfig()
    kick_prob: float = 0.0  # E2 perturbations: per-step probability of a velocity kick to all dynamic bodies
    kick_std: float = 0.0
```

В `METHODS` добавь:

```python
    "pred": ReflexMethodConfig(feedback=False),
    "reflex": ReflexMethodConfig(),
    "reflex_chunk": ReflexMethodConfig(requery=False),
    "reflex_off": ReflexMethodConfig(requery=False, feedback=False),  # sanity check: must reproduce naive
```

- [ ] **Step 2: Функция `eval` — заменить целиком**

Ветки `RealtimeMethodConfig` и `BIDMethodConfig` здесь дословно совпадают с upstream.

```python
def eval(
    config: EvalConfig,
    env: kenv.environment.Environment,
    rng: jax.Array,
    level: kenv_state.EnvState,
    policy: _model.FlowPolicy,
    env_params: kenv_state.EnvParams,
    static_env_params: kenv_state.EnvParams,
    weak_policy: _model.FlowPolicy | None = None,
):
    base_env = env
    if config.kick_prob > 0:
        env = train_expert.KickWrapper(env, config.kick_prob, config.kick_std)
    env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(train_expert.NoisyActionWrapper(env))), config.num_evals
    )
    # noise- and kick-free twin with the same state structure: the reflex's (oracle) predictor
    nominal_env = train_expert.BatchEnvWrapper(
        wrappers.LogWrapper(wrappers.AutoReplayWrapper(base_env)), config.num_evals
    )
    is_reflex = isinstance(config.method, ReflexMethodConfig)
    render_video = train_expert.make_render_video(renderer_pixels.make_render_pixels(env_params, static_env_params))
    assert config.execute_horizon >= config.inference_delay, f"{config.execute_horizon=} {config.inference_delay=}"
    d, s = config.inference_delay, config.execute_horizon
    assert s + d <= policy.action_chunk_size, f"{s=} + {d=} > H: padded zero actions would be executed"

    def execute_chunk(carry, _):
        def step(carry, xs):
            rng, obs, env_state = carry
            action, pkg_t = xs
            if pkg_t is not None and "gain" in pkg_t:
                action = reflex.correct(action, pkg_t["gain"], pkg_t["ref"], obs, config.method.max_correction)
            rng, key = jax.random.split(rng)
            next_obs, next_env_state, reward, done, info = env.step(key, env_state, action, env_params)
            return (rng, next_obs, next_env_state), (done, env_state, info)

        rng, obs, env_state, action_chunk, n, pkg = carry
        rng, key = jax.random.split(rng)
        if isinstance(config.method, NaiveMethodConfig):
            next_action_chunk = policy.action(key, obs, config.num_flow_steps)
        elif isinstance(config.method, RealtimeMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            assert (
                config.inference_delay <= policy.action_chunk_size
                and prefix_attention_horizon <= policy.action_chunk_size
            ), f"{config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            print(
                f"{config.execute_horizon=} {config.inference_delay=} {prefix_attention_horizon=} {policy.action_chunk_size=}"
            )
            next_action_chunk = policy.realtime_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.prefix_attention_schedule,
                config.method.max_guidance_weight,
            )
        elif isinstance(config.method, BIDMethodConfig):
            prefix_attention_horizon = policy.action_chunk_size - config.execute_horizon
            if config.method.bid_k is not None:
                assert weak_policy is not None, "weak_policy is required for BID"
            next_action_chunk = policy.bid_action(
                key,
                obs,
                config.num_flow_steps,
                action_chunk,
                config.inference_delay,
                prefix_attention_horizon,
                config.method.n_samples,
                bid_k=config.method.bid_k,
                bid_weak_policy=weak_policy if config.method.bid_k is not None else None,
            )
        elif is_reflex:
            noise = jax.random.normal(key, (obs.shape[0], policy.action_chunk_size, policy.action_dim))
            next_action_chunk = policy.action_from_noise(noise, obs, config.num_flow_steps)  # == policy.action(key)
            # steps t..t+H-1 run the previous package's actions for d steps, then this chunk
            planned = jnp.concatenate([pkg["nom"][:, :d], next_action_chunk[:, d:]], axis=1)
            pred = reflex.nominal_obs(
                lambda st, a: nominal_env.step(key, st, a, env_params)[:2], env_state, planned[:, :-1].swapaxes(0, 1)
            )
            ref = jnp.concatenate([obs[:, None], pred.swapaxes(0, 1)], axis=1)  # predicted obs per chunk index
            nom, gain = reflex.package(
                policy,
                noise,
                ref,
                next_action_chunk,
                config.num_flow_steps,
                config.method.requery,
                config.method.feedback,
                config.method.package_batch,
            )
            new_pkg = {"nom": nom, "ref": ref} | ({} if gain is None else {"gain": gain})
        else:
            raise ValueError(f"Unknown method: {config.method}")

        # we execute `inference_delay` actions from the *previously generated* action chunk, and then the remaining
        # `execute_horizon - inference_delay` actions from the newly generated action chunk
        action_chunk_to_execute = jnp.concatenate([action_chunk[:, :d], next_action_chunk[:, d:s]], axis=1)
        xs_pkg, next_pkg = None, None
        if is_reflex:
            # the package lives in chunk frame and is merged and shifted exactly like the action chunk
            exec_pkg = jax.tree.map(lambda old, new: jnp.concatenate([old[:, :d], new[:, d:s]], axis=1), pkg, new_pkg)
            action_chunk_to_execute = exec_pkg.pop("nom")
            xs_pkg = jax.tree.map(lambda x: x.swapaxes(0, 1), exec_pkg)
            next_pkg = jax.tree.map(lambda x: jnp.concatenate([x[:, s:], jnp.zeros_like(x[:, :s])], axis=1), new_pkg)
        # throw away the first `execute_horizon` actions from the newly generated action chunk, to align it with the
        # correct frame of reference for the next scan iteration
        next_action_chunk = jnp.concatenate(
            [next_action_chunk[:, s:], jnp.zeros((obs.shape[0], s, policy.action_dim))], axis=1
        )
        next_n = jnp.concatenate([n[s:], jnp.zeros(s, dtype=jnp.int32)])
        (rng, next_obs, next_env_state), (dones, env_states, infos) = jax.lax.scan(
            step, (rng, obs, env_state), (action_chunk_to_execute.transpose(1, 0, 2), xs_pkg)
        )
        return (rng, next_obs, next_env_state, next_action_chunk, next_n, next_pkg), (dones, env_states, infos)

    rng, key = jax.random.split(rng)
    obs, env_state = env.reset_to_level(key, level, env_params)
    rng, key = jax.random.split(rng)
    action_chunk = policy.action(key, obs, config.num_flow_steps)  # [batch, horizon, action_dim]
    n = jnp.ones(action_chunk.shape[1], dtype=jnp.int32)
    pkg = None
    if is_reflex:  # the first d steps of the first chunk run open-loop
        pkg = {"nom": action_chunk, "ref": jnp.repeat(obs[:, None], action_chunk.shape[1], axis=1)}
        if config.method.feedback:
            pkg["gain"] = jnp.zeros((*action_chunk.shape, obs.shape[-1]))
    scan_length = math.ceil(env_params.max_timesteps / config.execute_horizon)
    _, (dones, env_states, infos) = jax.lax.scan(
        execute_chunk,
        (rng, obs, env_state, action_chunk, n, pkg),
        None,
        length=scan_length,
    )
    dones, env_states, infos = jax.tree.map(lambda x: x.reshape(-1, *x.shape[2:]), (dones, env_states, infos))
    assert dones.shape[0] >= env_params.max_timesteps, f"{dones.shape=}"
    return_info = {}
    first_done_idx = jnp.argmax(dones, axis=0)  # only consider the first episode of each rollout
    for key in ["returned_episode_returns", "returned_episode_lengths", "returned_episode_solved"]:
        return_info[key] = infos[key][first_done_idx, jnp.arange(config.num_evals)].mean()
    solved = infos["returned_episode_solved"][first_done_idx, jnp.arange(config.num_evals)]
    lengths = infos["returned_episode_lengths"][first_done_idx, jnp.arange(config.num_evals)]
    return_info["solved_length"] = jnp.sum(lengths * solved) / jnp.maximum(jnp.sum(solved), 1)
    for key in ["match"]:
        if key in infos:
            return_info[key] = jnp.mean(infos[key])
    video = render_video(jax.tree.map(lambda x: x[:, 0], env_states))
    return return_info, video
```

- [ ] **Step 3: `main` — параметры рефлекса и новые колонки**

В сигнатуру `main` после `minmax: bool = False,` добавь:

```python
    max_correction: float = 1.0,
    package_batch: int = 16,
```

Внутри цикла `for name in methods:` строку с `c = dataclasses.replace(...)` замени на:

```python
                    method = METHODS[name]
                    if isinstance(method, ReflexMethodConfig):
                        method = dataclasses.replace(method, max_correction=max_correction, package_batch=package_batch)
                    c = dataclasses.replace(
                        config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=method
                    )
```

После строки `results["execute_horizon"].append(execute_horizon)` добавь:

```python
                        results["max_correction"].append(
                            max_correction if isinstance(method, ReflexMethodConfig) else float("nan")
                        )
                        results["kick_std"].append(config.kick_std if config.kick_prob > 0 else 0.0)
```

- [ ] **Step 4: Тесты не сломались**

Run: `uv run pytest tests/test_reflex.py -q`
Expected: `12 passed`.

- [ ] **Step 5: Upstream-методы побитно не изменились**

```bash
uv run --offline python scripts/check_upstream_bitwise.py
```

Expected: каждая строка `IDENTICAL`, код выхода 0. Скрипт сравнивает текущий `src/` с `upstream/main` (`naive`, `realtime`, `bid`, `hard_masking` на уровне policy и `eval()` для `naive` и `realtime`). Любое `DIFFERENT` — ошибка в новом `eval()`: чини её, а не скрипт.

- [ ] **Step 6: Sanity-проверка — `reflex_off` воспроизводит `naive`**

```bash
uv run src/eval_flow.py --run-path checkpoints/bc --level-paths worlds/l/grasp_easy.json \
  --config.num-evals 64 --methods naive reflex_off --delays 2 --horizons 4 --output-dir results/smoke-off
uv run python -c "
import pandas as pd
df = pd.read_csv('results/smoke-off/results.csv').set_index('method')
a, b = df.loc['naive', 'returned_episode_solved'], df.loc['reflex_off', 'returned_episode_solved']
print(a, b)
assert abs(a - b) <= 2 / 64, 'reflex_off must reproduce naive: the package merge/shift is broken'"
```

Ожидается, что оба числа совпадают или отличаются не больше чем на 2 эпизода. Если нет — ищи ошибку в склейке пакета (`exec_pkg` / `next_pkg`), а не подгоняй допуск.

- [ ] **Step 7: Smoke — варианты рефлекса и толчки**

```bash
uv run src/eval_flow.py --run-path checkpoints/bc --level-paths worlds/l/grasp_easy.json \
  --config.num-evals 16 --methods pred reflex reflex_chunk --delays 2 --horizons 4 --output-dir results/smoke-reflex
uv run src/eval_flow.py --run-path checkpoints/bc --level-paths worlds/l/grasp_easy.json \
  --config.num-evals 16 --config.kick-prob 0.02 --config.kick-std 1.0 --methods naive reflex \
  --delays 2 --horizons 4 --output-dir results/smoke-kick
cat results/smoke-reflex/results.csv results/smoke-kick/results.csv
```

Expected: 3 строки и 2 строки. `returned_episode_solved` в `[0, 1]`, нет `nan` (кроме `max_correction` у `naive`).

- [ ] **Step 8: Commit**

```bash
git add src/eval_flow.py
git commit -m "feat(eval): reflex method (pred / reflex / reflex_chunk), kicks, solved_length"
```

---

### Task 12: E2 на GPU

Нужна Linux-машина с одним NVIDIA GPU на 24 ГБ и больше (A100, L40S, 4090) и драйверами CUDA 12. Её адрес даёт пользователь: `export GPU_HOST=user@host`. Оценка — 4–6 GPU-часов.

**Files:** `results/eval/**` (результаты возвращаются на Mac).

- [ ] **Step 1: Перенести репозиторий и поставить окружение**

```bash
rsync -a --exclude .venv --exclude 'results/smoke*' /Users/alexanderkarpov/Desktop/M2R/ "$GPU_HOST":m2r/
ssh "$GPU_HOST"
cd m2r
curl -LsSf https://astral.sh/uv/install.sh | sh && source $HOME/.local/bin/env
uv sync
uv run python -c "import jax; print(jax.devices())"
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax
tmux new -s e2
```

Ожидается `[CudaDevice(id=0)]`. Если стоит больше одного GPU, выстави `CUDA_VISIBLE_DEVICES=0`: число устройств должно делить 12.

- [ ] **Step 2: Основная сетка и эталон**

```bash
uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 \
  --methods naive realtime pred reflex reflex_chunk --delays 1 2 3 4 --minmax --output-dir results/eval/main
uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 \
  --methods naive --delays 0 --horizons 1 --output-dir results/eval/oracle
```

Если процесс падает с OOM, добавь `--package-batch 8`.

- [ ] **Step 3: Абляция `max_correction`**

```bash
for mc in 0.1 0.3 1.0; do
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 \
    --methods reflex --delays 2 --horizons 6 --max-correction $mc --output-dir results/eval/mc$mc
done
```

- [ ] **Step 4: Толчки**

```bash
V=$(uv run src/probe.py kick-speed | awk '/v_med/ {print $3}')
echo "v_med=$V" | tee results/eval/v_med.txt
for c in 0.5 1 2; do
  STD=$(python3 -c "print($c * $V)")
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 \
    --config.kick-prob 0.02 --config.kick-std $STD \
    --methods naive realtime pred reflex reflex_chunk --delays 2 --horizons 2 6 --output-dir results/eval/kick$c
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 0 1 2 \
    --config.kick-prob 0.02 --config.kick-std $STD \
    --methods naive --delays 0 --horizons 1 --output-dir results/eval/kick$c-oracle
done
```

- [ ] **Step 5: Стоимость вызова**

```bash
uv run src/probe.py cost --batch 1 --out results/eval/cost_b1.csv | tee results/eval/cost.txt
uv run src/probe.py cost --batch 256 --out results/eval/cost_b256.csv | tee -a results/eval/cost.txt
```

- [ ] **Step 6: Вернуть результаты на Mac и построить графики**

На Mac:

```bash
rsync -a "$GPU_HOST":m2r/results/eval/ results/eval/
uv run src/plot.py success
uv run src/plot.py table --results-glob 'results/eval/mc*/results.csv'
uv run src/plot.py table --results-glob 'results/eval/kick*/results.csv'
uv run src/plot.py gate2 | tee results/eval/gate2.json
```

- [ ] **Step 7: Commit**

```bash
git add results/eval
git commit -m "results: E2 closed loop, max_correction ablation, kicks"
```

---

### Task 13: Записка E2 и **Gate 2 (STOP)**

**Files:** Create `docs/results/closed-loop.md`.

- [ ] **Step 1: Написать записку**

Порядок содержимого:

1. JSON из `results/eval/gate2.json`, дословно.
2. Картинка: `![success](../../results/eval/main/success.png)`.
3. Три таблицы, дословно: вывод `plot.py success`, таблица абляции, таблица толчков.
4. Строки из `results/eval/cost.txt`.
5. 4–8 пунктов с ответами на вопросы:
   - выполнено ли (a), (b) или (c);
   - выигрывает ли `reflex` у `pred` (то есть работает ли `J`, а не предсказатель);
   - `reflex` против `reflex_chunk`;
   - где рефлекс вредит (уровни, большие `d`);
   - как `max_correction` влияет на результат;
   - сколько стоит вызов по сравнению с обычным чанком;
   - если выполнено (b): формулировка из `b_claim` дословно — во сколько раз меньше вызовов и во сколько раз больше работы на эпизод;
   - `latency_vs_realtime` у reflex при batch 1 (из `cost_b1.csv`). Если `r > 1.5`, сравнение при одинаковом `d` подыгрывает reflex: честная задержка `d' = ⌈r·d⌉`. При ней допустимо только `s ≤ 8 − d'`, поэтому перепрогон — это reflex при `(d', 8 − d')`, и `b_calls_saved` пересчитывается. Предложи пользователю перепрогон на Gate 2.

- [ ] **Step 2: Commit**

```bash
git add docs/results/closed-loop.md
git commit -m "results: E2 notes and Gate 2 decision"
```

- [ ] **Step 3: GATE 2 — остановиться**

Покажи пользователю `decision`, картинку и записку. При GO следующий шаг — отдельный спек и план для E3 (A2C2 и VLASH из bt-kinetix) или для фаз B/C. Их решает пользователь, в этот план они не входят. В любом случае дальше идёт Task 14.

---

### Task 14: E4 — write-up, README, видео

**Files:**
- Create: `src/video.py`, `docs/results/report.md`, `README.md` (новый)
- Move: `README.md` → `README.upstream.md`

- [ ] **Step 1: `src/video.py` (только если был E2)**

```python
"""Side-by-side rollout video (env 0) of several methods on one level, same seed and same kicks (E4 write-up)."""

from typing import Sequence

import imageio
import jax
import kinetix.environment.env as kenv
import kinetix.environment.env_state as kenv_state
import numpy as np
import tyro

import eval_flow
import probe
import train_expert


def main(
    run_path: str = "checkpoints/bc",
    level_path: str = "worlds/l/grasp_easy.json",
    methods: Sequence[str] = ("naive", "reflex"),
    delay: int = 2,
    horizon: int = 6,
    kick_prob: float = 0.0,
    kick_std: float = 0.0,
    seed: int = 0,
    out: str = "results/video.mp4",
):
    static_env_params = kenv_state.StaticEnvParams(**train_expert.LARGE_ENV_PARAMS, frame_skip=train_expert.FRAME_SKIP)
    env_params = kenv_state.EnvParams()
    level = jax.tree.map(lambda x: x[0], train_expert.load_levels([level_path], static_env_params, env_params))
    static_env_params = static_env_params.replace(screen_dim=train_expert.SCREEN_DIM)
    env = kenv.make_kinetix_env_from_name("Kinetix-Symbolic-Continuous-v1", static_env_params=static_env_params)
    obs_dim = jax.eval_shape(env.reset_to_level, jax.random.key(0), level, env_params)[0].shape[-1]
    policy = probe.make_policy(probe.load_state_dict(run_path, level_path), obs_dim, env.action_space(env_params).shape[0])
    videos = []
    for name in methods:
        config = eval_flow.EvalConfig(
            num_evals=1, inference_delay=delay, execute_horizon=horizon, method=eval_flow.METHODS[name],
            kick_prob=kick_prob, kick_std=kick_std,
        )
        info, video = jax.jit(lambda rng: eval_flow.eval(config, env, rng, level, policy, env_params, static_env_params))(
            jax.random.key(seed)
        )
        print(name, {k: round(float(v), 3) for k, v in info.items()})
        videos.append(np.asarray(video))
    imageio.mimwrite(out, np.concatenate(videos, axis=2), fps=15)  # methods side by side
    print(f"saved {out}")


if __name__ == "__main__":
    tyro.cli(main)
```

Запуск (`V` — число из `results/eval/v_med.txt`):

```bash
uv run src/video.py --kick-prob 0.02 --kick-std "$(python3 -c "print(1 * $(cut -d= -f2 results/eval/v_med.txt))")" \
  --out results/video_kick.mp4
```

Expected: `saved results/video_kick.mp4`; слева `naive`, справа `reflex`, одинаковые толчки. Если на `grasp_easy` разница не видна, выбери уровень с наибольшим отрывом `reflex` в таблице толчков и повтори с `--level-path`.

- [ ] **Step 2: Отчёт `docs/results/report.md` (2–3 страницы)**

Разделы:

1. **Гипотеза** — формула из §1 спека.
2. **Метод** — схема из §4 и три варианта.
3. **E1** — вердикт, картинка, 3 главных наблюдения. Отдельно — находка про floor: даже со сдвинутым шумом «переспросить policy» и «продолжать старый план» различаются без всяких отклонений (на `grasp_easy` floor 0.2–0.7, а не ноль; спек, «Журнал изменений»).
4. **E2** — если был: график, `gate2.json`, толчки, стоимость.
5. **Ограничения**, строго так:
   - предсказатель — оракул (форк симулятора);
   - Kinetix — это не VLA, наблюдения здесь символьные, без картинок;
   - один seed в E1.
6. **Что дальше** — B или C, одним абзацем.

- [ ] **Step 3: README**

```bash
git mv README.md README.upstream.md
```

Новый `README.md` с разделами:

1. Два абзаца: что проверяется и почему, с формулой `a = π(ô)[0] + J·(o − ô)`.
2. «Результаты»: 3–5 пунктов из `docs/results/report.md` и ссылки на картинки и видео.
3. «Воспроизведение на Mac»: команды Task 1 Steps 3–4, Task 2, `uv run pytest`, Task 9 Steps 1–2. Если использовался запасной путь из Task 1 Step 3, укажи префикс `UV_NO_SYNC=1`.
4. «Воспроизведение на GPU»: команды Task 12.
5. «Структура»: таблица «Структура файлов» из этого плана.
6. «Основано на»: [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) (MIT, исходный README в `README.upstream.md`), [Kinetix](https://github.com/FLAIROx/Kinetix), [bt-kinetix](https://github.com/TheAyos/bt-kinetix).

- [ ] **Step 4: Проверка и commit**

```bash
uv run pytest -q
git add README.md README.upstream.md docs/results/report.md src/video.py results/video_kick.mp4
git commit -m "docs: phase A report, README, demo video"
```

Если E2 не было, `src/video.py` и видео в коммит не входят.

Expected: все тесты зелёные.

---

## Сроки и объём

| Task | Что | Время |
|---|---|---|
| 1–3 | окружение, чекпойнты, CLI и smoke (E0) | день 1 |
| 4–8 | модель, рефлекс, зонд, графики (~480 строк кода, 11 тестов) | дни 2–3 |
| 9 | прогон E1 и Gate 1 | день 3–4 |
| 10–11 | толчки и метод `reflex` в eval (~120 строк) | дни 5–6 |
| 12–13 | GPU-прогоны (4–6 ч GPU) и Gate 2 | дни 7–9 |
| 14 | отчёт, README, видео | дни 10–12 |
