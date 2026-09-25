# B1: устаревание плана и неидеальный предсказатель — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Прогнать заранее записанный эксперимент B1 (спек `docs/superpowers/specs/2026-09-25-b1-staleness-predictors-design.md`): `reflex` и `rtc_reflex` с оракулом, «кривой физикой» и обученной моделью мира при d = 1 (s = 1…7) и на срезе d = 3 (s = 5), и принять решения по правилам R1–R4.

**Architecture:**
- Пакет рефлекса считается только для исполняемых позиций `d…d+s−1`. Это точное ускорение.
- Новый модуль `src/predictors.py` содержит «кривую физику», модель мира (MLP в пространстве наблюдений) и офлайн-таблицу ошибок прогноза.
- `src/eval_flow.py` выбирает предсказатель через `ReflexMethodConfig.predictor` и пишет столбец `predictor`.
- `src/plot.py b1` применяет правила R1–R4.

**Tech Stack:** JAX 0.4.35, Flax NNX 0.10.2, optax 0.2.4 (уже в зависимостях через `train_expert`), Kinetix (`third_party/kinetix`), jax2d, pandas, tyro, pytest (`pythonpath = src`), uv.

---

## Общие правила (для каждого исполнителя)

- **Запуск.** Все команды — из корня репозитория `/Users/alexanderkarpov/Desktop/M2R`. Python — только через `uv run --offline …`: интернет мобильный, пакеты уже стоят. Для тестов: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q …`.
- **Коммиты.** Сообщения на английском, в стиле истории (`feat(...)`, `fix(...)`, `spec:`, `results:`, `docs:`). **Никогда не добавлять строку `Co-Authored-By`**, это постоянное правило владельца. Коммит — только на своей задаче, без `git push`, пуш делает владелец.
- **Долгие прогоны** (дольше ~5 минут) исполнитель не ждёт: он пишет команду в отчёт, а запускает её контроллер в фоне. В фазе A сторожевой таймер убил исполнителя, который ждал долгий прогон.
- **Стиль кода.** Комментарии и docstring на английском, коротко, как в `src/reflex.py`. Новые функции без лишних абстракций.
- **Не трогать:** `naive`, `realtime`, `bid`, `hard_masking`. После задач 3 и 7 `scripts/check_upstream_bitwise.py` должен оставаться зелёным.
- **Данные B1.** Seed'ы 10, 11, 12 используются **только** в задаче 11 на GPU. В smoke и репетиции — seed 99 или 0.

## Карта файлов

| файл | ответственность |
|---|---|
| `scripts/check_upstream_bitwise.py` | закрепить ссылку на код upstream (задача 1) |
| `src/reflex.py` | `package(..., used=)`, `forward_equivalents(..., positions=)` (задача 2) |
| `src/eval_flow.py` | короткий прогноз и `used` (задача 3); выбор предсказателя, `--predictors`, модели мира, столбец `predictor`, метки времени (задача 7) |
| `src/probe.py` | `cost` считает только исполняемые позиции (задача 3) |
| `src/predictors.py` (новый) | «кривая физика» (задача 4), модель мира и `train` (задача 5), таблица ошибок `errors` (задача 6) |
| `src/plot.py` | `b1`: правила R1–R4, `b1.json`, `b1.png` (задача 8) |
| `tests/test_reflex.py` | тесты задач 2, 7, 8 |
| `tests/test_predictors.py` (новый) | тесты задач 4–6 |
| `scripts/run_b1_rehearsal.sh`, `scripts/gpu_b1.sh`, `.gitignore` | задача 9 |
| `docs/…` | журнал спека (задачи 10–11), записка (задача 12) |

---

### Task 1: проверка побитного совпадения без remote `upstream`

После публикации на GitHub remote `upstream` удалён. Переписывание истории сменило хеши коммитов Physical Intelligence, но не их содержимое. Последний коммит PI в нашей истории — `23e8e2f3da35571ea1385f687d38b711a4d7ad96` («Update README.md», точка ответвления). Проверка должна брать код upstream из этого коммита, тогда она работает и у любого, кто склонирует репозиторий.

**Files:**
- Modify: `scripts/check_upstream_bitwise.py` (docstring строки 1–7, константы около строки 22, `git show` в `main()`)

- [ ] **Step 1: Убедиться, что проверка сейчас падает**

Run: `uv run --offline python scripts/check_upstream_bitwise.py`
Expected: FAIL. `git show upstream/main:src/model.py` завершается с ошибкой, потому что ref `upstream/main` не существует.

- [ ] **Step 2: Закрепить коммит upstream**

В `scripts/check_upstream_bitwise.py` заменить строку docstring
```python
"""Bitwise check that upstream methods are untouched: current src/ vs pristine upstream/main.
```
на
```python
"""Bitwise check that upstream methods are untouched: current src/ vs pristine upstream (UPSTREAM_REF).
```
а строку
```python
`git show upstream/main:src/...`, one from src/, and compares every output exactly (np.array_equal).
```
на
```python
`git show UPSTREAM_REF:src/...`, one from src/, and compares every output exactly (np.array_equal).
```
После строки `UPSTREAM_FILES = ("model", "eval_flow", "train_expert")` добавить:
```python
# the last Physical Intelligence commit in this repository's history (the fork point): works without any remote
UPSTREAM_REF = "23e8e2f3da35571ea1385f687d38b711a4d7ad96"
```
В `main()` заменить
```python
                ["git", "show", f"upstream/main:src/{name}.py"], cwd=root, check=True, capture_output=True, text=True
```
на
```python
                ["git", "show", f"{UPSTREAM_REF}:src/{name}.py"], cwd=root, check=True, capture_output=True, text=True
```

- [ ] **Step 3: Проверить, что проверка зелёная**

Run: `uv run --offline python scripts/check_upstream_bitwise.py`
Expected: every line ends with `IDENTICAL`, exit code 0.

- [ ] **Step 4: Commit**

```bash
git add scripts/check_upstream_bitwise.py
git commit -m "fix(check): read upstream code from the fork-point commit, not the upstream remote"
```

---

### Task 2: пакет только для исполняемых позиций

Цикл оценки читает из пакета вызова только индексы `d … d+s−1`: `new[:, d:s]` сейчас и `new[:, s:s+d]` после сдвига. Пакет считаем только для них и возвращаем в формате чанка `[B, H, …]`, где остальные позиции нулевые.

**Files:**
- Modify: `src/reflex.py` (`package`, `forward_equivalents`)
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Написать падающие тесты**

Добавить в конец `tests/test_reflex.py`:
```python
def test_package_computes_only_used_positions():
    policy = small_policy()
    B, H, A, O = 2, policy.action_chunk_size, policy.action_dim, 5
    noise = jax.random.normal(jax.random.key(10), (B, H, A))
    ref = jax.random.normal(jax.random.key(11), (B, H, O))
    chunk = jnp.ones((B, H, A))
    nom, gain = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True)
    nom_u, gain_u = reflex.package(policy, noise, ref, chunk, 5, requery=True, feedback=True, used=(2, 5))
    assert nom_u.shape == nom.shape and gain_u.shape == gain.shape
    np.testing.assert_allclose(nom_u[:, 2:5], nom[:, 2:5], atol=1e-5)
    np.testing.assert_allclose(gain_u[:, 2:5], gain[:, 2:5], atol=1e-5)
    assert not np.any(nom_u[:, :2]) and not np.any(nom_u[:, 5:]) and not np.any(gain_u[:, 5:])
    nom_c, gain_c = reflex.package(policy, noise, ref, chunk, 5, requery=False, feedback=True, used=(2, 5))
    np.testing.assert_array_equal(nom_c, chunk)  # reflex_chunk keeps the chunk
    np.testing.assert_allclose(gain_c[:, 2:5], gain[:, 2:5], atol=1e-5)


def test_forward_equivalents_counts_only_computed_positions():
    assert reflex.forward_equivalents("reflex", positions=1) == 5 + 1 * (5 + 2 * 6 * 5)  # 70
    assert reflex.forward_equivalents("pred", positions=7) == 5 + 7 * 5  # 40
    assert reflex.forward_equivalents("reflex", positions=None) == 525  # phase A: all H positions
    assert reflex.forward_equivalents("naive", positions=3) == 5  # no package: positions don't matter
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_reflex.py -k "used_positions or computed_positions"`
Expected: FAIL with `TypeError: package() got an unexpected keyword argument 'used'` (и то же для `positions`).

- [ ] **Step 3: Реализация**

В `src/reflex.py` заменить функцию `package` целиком на:
```python
def package(policy, noise, ref, chunk, num_steps, requery: bool, feedback: bool, batch_size: int = 16, used=None):
    """Reflex package for one policy call, in chunk frame.

    noise [B, H, A] (the call's sampling noise z), ref [B, H, O] (predicted obs per chunk index), chunk [B, H, A].
    Returns nom [B, H, A] (nom_j = pi(roll(z, -j), ref_j)[0] if requery else chunk) and gain [B, H, A, O]
    (gain_j = d pi(roll(z, -j), o)[0] / d o at ref_j; None without feedback).
    used = (lo, hi): compute only chunk indices lo..hi-1, the ones eval executes (d..d+s-1, spec B1 3.4); the other
    entries are zeros and are never read. Envs are processed batch_size at a time: Jacobian activations are large.
    """
    if not (requery or feedback):
        return chunk, None
    H = noise.shape[1]
    lo, hi = used or (0, H)

    def per_env(x):
        n, r = shifted_noise(x[0])[lo:hi], x[1][lo:hi]
        if feedback:
            return jax.vmap(lambda nj, o: first_action_and_jacobian(policy, nj, o, num_steps))(n, r)
        return policy.action_from_noise(n, r, num_steps)[:, 0], None

    def full(x):  # back to chunk frame [B, H, ...]
        return jnp.zeros((x.shape[0], H, *x.shape[2:]), x.dtype).at[:, lo:hi].set(x)

    a0, jac = jax.lax.map(per_env, (noise, ref), batch_size=min(batch_size, noise.shape[0]))
    return (full(a0) if requery else chunk), (None if jac is None else full(jac))
```

В `forward_equivalents` заменить сигнатуру, docstring и тело на:
```python
def forward_equivalents(
    method: str, num_steps: int = 5, chunk_size: int = 8, action_dim: int = 6, positions: int | None = None
) -> int:
    """Network evaluations per policy call, one VJP counted as 2 (spec section 4, budget).

    positions: package entries computed per call. None = all H (phase A); eval computes only the s executed ones
    (spec B1 3.4). Not counted: the reflex's per-step correction (action_dim x obs_dim MACs, negligible) and the
    predictor (a simulator fork or a small world model).
    """
    S, H, A = num_steps, chunk_size, action_dim
    P = H if positions is None else positions
    return {
        "naive": S,
        "reflex_off": S,
        "realtime": 3 * S,  # one guidance VJP per flow step
        "hard_masking": 3 * S,
        "bid": 16 * S,  # default n_samples=16, no weak policy
        "pred": S + P * S,  # chunk + pi at P predicted states
        "reflex": S + P * (S + 2 * A * S),  # + A VJPs through the whole flow at each predicted state
        "reflex_chunk": S + P * (S + 2 * A * S),
        "rtc_reflex": 3 * S + P * (S + 2 * A * S),  # E2b: RTC chunk + the same package
        "rtc_reflex_off": 3 * S,
    }[method]
```

- [ ] **Step 4: Прогнать тесты**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_reflex.py`
Expected: all pass (19 tests: 17 old + 2 new). Старые `test_package_flags` и `test_forward_equivalents` не меняются.

- [ ] **Step 5: Commit**

```bash
git add src/reflex.py tests/test_reflex.py
git commit -m "feat(reflex): compute the package only at the executed chunk indices (exact B1 speedup)"
```

---

### Task 3: цикл оценки и стоимость используют только исполняемые позиции

**Files:**
- Modify: `src/eval_flow.py` (ветка `elif is_reflex:` в `execute_chunk`)
- Modify: `src/probe.py` (`cost`)

- [ ] **Step 1: Цикл оценки**

В `src/eval_flow.py` в ветке `elif is_reflex:` заменить блок от строки `planned = …` до закрывающей скобки вызова `reflex.package(...)`:
```python
            planned = jnp.concatenate([pkg["nom"][:, :d], next_action_chunk[:, d:]], axis=1)
            pred = reflex.nominal_obs(
                lambda st, a: nominal_step(key, st, a, env_params)[:2],
                env_state.env_state.env_state,  # BatchEnv/LogWrapper -> AutoReplay -> raw EnvState
                planned[:, :-1].swapaxes(0, 1),
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
```
на
```python
            planned = jnp.concatenate([pkg["nom"][:, :d], next_action_chunk[:, d:]], axis=1)
            n_pred = max(d + s - 1, 1)  # the package reads only chunk indices d..d+s-1 (spec B1 3.4)
            pred = reflex.nominal_obs(
                lambda st, a: nominal_step(key, st, a, env_params)[:2],
                env_state.env_state.env_state,  # BatchEnv/LogWrapper -> AutoReplay -> raw EnvState
                planned[:, :n_pred].swapaxes(0, 1),
            )
            ref = jnp.concatenate([obs[:, None], pred.swapaxes(0, 1)], axis=1)  # predicted obs per chunk index
            ref = jnp.pad(ref, ((0, 0), (0, policy.action_chunk_size - ref.shape[1]), (0, 0)))  # never read past d+s-1
            nom, gain = reflex.package(
                policy,
                noise,
                ref,
                next_action_chunk,
                config.num_flow_steps,
                config.method.requery,
                config.method.feedback,
                config.method.package_batch,
                used=(d, d + s),
            )
```

- [ ] **Step 2: Стоимость**

В `src/probe.py`, функция `cost`:
- в docstring первую строку заменить на `"""Per-call cost of every E2/B1 method: network evaluations (analytic), GFLOP and measured latency.`;
- добавить после docstring строку `used = (delay, delay + horizon)  # the package covers only the executed chunk indices (spec B1 3.4)`;
- в `with_package` заменить `reflex.package(policy, noise, ref, chunk, num_flow_steps, requery, feedback)` на `reflex.package(policy, noise, ref, chunk, num_flow_steps, requery, feedback, used=used)`;
- в лямбде `rtc_reflex` заменить `num_flow_steps, False, True,` на `num_flow_steps, False, True, used=used,`;
- строку `evals = reflex.forward_equivalents(name, num_flow_steps, H, action_dim)` заменить на `evals = reflex.forward_equivalents(name, num_flow_steps, H, action_dim, positions=horizon)`.

- [ ] **Step 3: Быстрые проверки**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q`
Expected: 19 passed.

Run: `uv run --offline python scripts/check_upstream_bitwise.py`
Expected: all `IDENTICAL`.

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline src/probe.py cost --batch 1 --delay 1 --horizon 1 --repeats 3`
Expected: таблица, в которой `evals_per_call` для `reflex` = 70, для `pred` = 10, для `realtime` = 15.

- [ ] **Step 4: Проверка в замкнутом цикле (~50 мин; запускает контроллер в фоне)**

Повторить прогоны фазы A на Mac (d = 2, s = 6, 64 эпизода × 12 уровней, seed 0) для `pred`, `reflex`, `rtc_reflex`:
```bash
UV_OFFLINE=1 JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 uv run --offline src/eval_flow.py --run-path checkpoints/bc \
  --config.num-evals 64 --seeds 0 --methods pred reflex rtc_reflex --delays 2 --horizons 6 \
  --output-dir results/smoke_b1_speedup
```
Потом сравнить со старыми результатами: `pred` и `reflex` — с `results/e2_preview`, `rtc_reflex` — с `results/e2_night/rtc_d2s6`:
```bash
uv run --offline python - <<'EOF'
import pandas as pd
old = pd.concat([pd.read_csv("results/e2_preview/results.csv"), pd.read_csv("results/e2_night/rtc_d2s6/results.csv")])
new = pd.read_csv("results/smoke_b1_speedup/results.csv")
m = old[old["method"].isin(["pred", "reflex", "rtc_reflex"])].merge(new, on=["method", "level"], suffixes=("_old", "_new"))
m["diff"] = ((m["returned_episode_solved_new"] - m["returned_episode_solved_old"]) * 64).round().astype(int)
print(m[["method", "level", "diff"]].to_string(index=False))
assert len(m) == 36 and m["diff"].abs().max() <= 2, "the speedup changed closed-loop outcomes"
print("OK: within +-2 episodes per level")
EOF
```
Expected: `OK: within +-2 episodes per level`. Если расхождение больше, остановиться и разобраться: правило спека §4.1.

- [ ] **Step 5: Commit**

```bash
git add src/eval_flow.py src/probe.py
git commit -m "feat(eval): roll the predictor out only to d+s-1 and build the package at executed indices; cost follows"
```

---

### Task 4: «кривая физика»

**Files:**
- Create: `src/predictors.py`
- Test: `tests/test_predictors.py`

- [ ] **Step 1: Написать падающие тесты**

Создать `tests/test_predictors.py`:
```python
import jax
import jax.numpy as jnp
import numpy as np

import predictors
import probe


def _level_state(level="worlds/l/grasp_easy.json"):
    env, env_params, levels, _, action_dim = probe.setup([level])
    _, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), env_params)
    return env._env, env_params, state.env_state, action_dim  # raw Kinetix env and raw EnvState, as eval uses


def test_phys_factors_are_one_plus_minus_error_and_paired():
    _, _, raw, _ = _level_state()
    f2 = predictors.phys_factors(jax.random.key(1), raw, 0.2)
    f3 = predictors.phys_factors(jax.random.key(1), raw, 0.3)
    for a, b in zip(jax.tree.leaves(f2), jax.tree.leaves(f3)):
        np.testing.assert_allclose(np.abs(np.asarray(a) - 1), 0.2, atol=1e-6)
        np.testing.assert_array_equal(np.sign(np.asarray(a) - 1), np.sign(np.asarray(b) - 1))  # same key: same signs


def test_phys_zero_error_is_the_oracle():
    env, params, raw, A = _level_state()
    f = predictors.phys_factors(jax.random.key(1), raw, 0.0)
    a = jnp.full(A, 0.5)
    o1, s1 = predictors.phys_step(env, jax.random.key(2), raw, a, params, f)
    o0, s0 = env.step_env(jax.random.key(2), raw, a, params)[:2]
    np.testing.assert_array_equal(o1, o0)
    jax.tree.map(np.testing.assert_array_equal, s1, s0)


def test_phys_error_changes_the_motion_but_not_the_parameter_features():
    env, params, raw, A = _level_state()
    f = predictors.phys_factors(jax.random.key(1), raw, 0.3)
    a = jnp.full(A, 0.5)
    s_true = s_wrong = raw
    for _ in range(10):
        o_true, s_true = env.step_env(jax.random.key(2), s_true, a, params)[:2]
        o_wrong, s_wrong = predictors.phys_step(env, jax.random.key(2), s_wrong, a, params, f)
    for x, y in zip(jax.tree.leaves(predictors._params(s_wrong)), jax.tree.leaves(predictors._params(raw))):
        np.testing.assert_array_equal(x, y)  # the true parameters are back in the predicted state
    np.testing.assert_array_equal(o_wrong, env.get_obs(s_wrong))  # and in its observation
    assert not np.allclose(o_wrong, o_true)  # but the bodies moved differently
```

- [ ] **Step 2: Убедиться, что тесты падают**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_predictors.py`
Expected: FAIL with `ModuleNotFoundError: No module named 'predictors'`.

- [ ] **Step 3: Реализация**

Создать `src/predictors.py`:
```python
"""B1 predictors for the reflex: wrong-physics simulator copies and a learned world model, plus their offline errors.

See docs/superpowers/specs/2026-09-25-b1-staleness-predictors-design.md (sections 3 and 4.2).
"""

import jax
import jax.numpy as jnp

BODY_PARAMS = ("inverse_mass", "inverse_inertia", "friction")


def level_name(level_path: str) -> str:
    return level_path.replace("/", "_").replace(".json", "")


def _params(state):
    """The physical parameters a phys predictor gets wrong: (polygon, circle, motor power, thruster power)."""

    def body(b):
        return {k: getattr(b, k) for k in BODY_PARAMS}

    return body(state.polygon), body(state.circle), state.joint.motor_power, state.thruster.power


def _with(state, polygon, circle, motor_power, thruster_power):
    return state.replace(
        polygon=state.polygon.replace(**polygon),
        circle=state.circle.replace(**circle),
        joint=state.joint.replace(motor_power=motor_power),
        thruster=state.thruster.replace(power=thruster_power),
    )


def phys_factors(key, state, error: float):
    """1 + error * (+-1) for every entry of _params(state), with a random sign per entry (body, joint, thruster).

    A fixed key gives every method and every error size the same signs: a paired comparison and a clean dose curve.
    """
    leaves, treedef = jax.tree.flatten(_params(state))
    keys = jax.random.split(key, len(leaves))
    return jax.tree.unflatten(
        treedef, [1 + error * jax.random.rademacher(k, x.shape).astype(x.dtype) for k, x in zip(keys, leaves)]
    )


def phys_step(env, key, state, action, params, factors):
    """One predictor step with wrong physics -> (obs, state), like env.step_env(...)[:2].

    Steps with the parameters x factors, then puts the true parameters back: the symbolic observation contains the
    parameters themselves, and a prediction must differ from the oracle's only through the motion.
    """
    wrong = _with(state, *jax.tree.map(jnp.multiply, _params(state), factors))
    nxt = env.step_env(key, wrong, action, params)[1]
    nxt = _with(nxt, *_params(state))
    return env.get_obs(nxt), nxt
```

- [ ] **Step 4: Прогнать тесты**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_predictors.py`
Expected: 3 passed.

Если `test_phys_error_changes_the_motion_but_not_the_parameter_features` упадёт на последней проверке (движение совпало), значит на `grasp_easy` за 10 шагов искажение не успело сказаться. Замени уровень в этом тесте на `"worlds/l/mjc_walker.json"`: там моторы работают с первого шага. Логику не меняй.

- [ ] **Step 5: Commit**

```bash
git add src/predictors.py tests/test_predictors.py
git commit -m "feat(predictors): wrong-physics predictor step (true parameters kept in the predicted observation)"
```

---

### Task 5: обученная модель мира

**Files:**
- Modify: `src/predictors.py`
- Test: `tests/test_predictors.py`

- [ ] **Step 1: Написать падающий тест**

Добавить в конец `tests/test_predictors.py`:
```python
def test_world_model_learns_linear_dynamics_and_copies_static_dims():
    N, O, A = 2048, 4, 2
    obs = jax.random.normal(jax.random.key(0), (N, O)).at[:, 3].set(5.0)  # dim 3 never moves
    act = jax.random.normal(jax.random.key(1), (N, A))
    B = jnp.array([[1.0, 0.0], [0.0, -1.0], [0.5, 0.5], [0.0, 0.0]])
    nxt = obs + 0.1 * act @ B.T
    wm = predictors.fit(obs, act, nxt, jax.random.key(2), hidden=32, steps=3000, batch=256, lr=3e-3)
    np.testing.assert_array_equal(np.asarray(wm["mask"]), [1, 1, 1, 0])
    assert float(predictors.one_step_nmse(wm, obs, act, nxt)) < 0.05
    roll = predictors.wm_rollout(wm, obs[0], act[:5])
    assert roll.shape == (5, O)
    np.testing.assert_array_equal(np.asarray(roll[:, 3]), 5.0)  # static dims are copied exactly
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_predictors.py -k world_model`
Expected: FAIL with `AttributeError: module 'predictors' has no attribute 'fit'`.

- [ ] **Step 3: Реализация**

В `src/predictors.py` заменить блок импортов на:
```python
import pathlib
import pickle
from typing import Sequence

import jax
import jax.numpy as jnp
import optax
import pandas as pd
import tyro

import probe
import train_expert

WM_DIR = "results/b1/world_models"
```
Строку `BODY_PARAMS = …` оставить после `WM_DIR`. В конец файла добавить:
```python
def init(key, obs_dim: int, action_dim: int, hidden: int):
    sizes = (obs_dim + action_dim, hidden, hidden, obs_dim)
    keys = jax.random.split(key, len(sizes) - 1)
    return [(jax.random.normal(k, (m, n)) / jnp.sqrt(m), jnp.zeros(n)) for k, m, n in zip(keys, sizes[:-1], sizes[1:])]


def _net(wm, obs, action):
    """Normalized one-step change of obs predicted from (obs, action): obs [O], action [A] -> [O]."""
    x = jnp.concatenate([(obs - wm["x_mean"]) / wm["x_std"], action])
    for w, b in wm["layers"][:-1]:
        x = jax.nn.gelu(x @ w + b)
    w, b = wm["layers"][-1]
    return x @ w + b


def apply(wm, obs, action):
    """Next obs. Dims that never move in the training data (mask 0) are copied exactly."""
    return obs + wm["mask"] * (_net(wm, obs, action) * wm["d_std"] + wm["d_mean"])


def wm_rollout(wm, obs, actions):
    """Predicted obs after each action: obs [O], actions [T, A] -> [T, O]."""

    def body(o, a):
        o = apply(wm, o, a)
        return o, o

    return jax.lax.scan(body, obs, actions)[1]


def fit(obs, act, nxt, key, hidden: int = 256, steps: int = 10_000, batch: int = 512, lr: float = 1e-3):
    """World model (spec B1 3.3) from transitions obs [N, O], act [N, A], nxt [N, O]."""
    obs, act, nxt = jnp.asarray(obs), jnp.asarray(act), jnp.asarray(nxt)
    delta = nxt - obs
    x_std, d_std = obs.std(0), delta.std(0)
    mask = (d_std > 1e-6).astype(obs.dtype)
    k_init, k_steps = jax.random.split(key)
    wm = {
        "layers": init(k_init, obs.shape[1], act.shape[1], hidden),
        "x_mean": obs.mean(0),
        "x_std": jnp.where(x_std > 1e-6, x_std, 1.0),
        "d_mean": delta.mean(0) * mask,
        "d_std": jnp.where(mask > 0, d_std, 1.0),
        "mask": mask,
    }
    target = (delta - wm["d_mean"]) / wm["d_std"]
    opt = optax.adam(lr)

    def loss(layers, o, a, t):
        pred = jax.vmap(lambda o_, a_: _net(wm | {"layers": layers}, o_, a_))(o, a)
        return jnp.sum(mask * (pred - t) ** 2) / (mask.sum() * o.shape[0])

    @jax.jit
    def run(layers, keys, obs, act, target):
        def body(carry, k):
            layers, opt_state = carry
            idx = jax.random.randint(k, (batch,), 0, obs.shape[0])
            grads = jax.grad(loss)(layers, obs[idx], act[idx], target[idx])
            updates, opt_state = opt.update(grads, opt_state)
            return (optax.apply_updates(layers, updates), opt_state), None

        return jax.lax.scan(body, (layers, opt.init(layers)), keys)[0][0]

    return wm | {"layers": run(wm["layers"], jax.random.split(k_steps, steps), obs, act, target)}


def one_step_nmse(wm, obs, act, nxt):
    """Mean squared one-step error on the moving dims, in units of their change's std (1.0 = predicting the mean)."""
    pred = jax.vmap(lambda o, a: apply(wm, o, a))(jnp.asarray(obs), jnp.asarray(act))
    z = wm["mask"] * (pred - jnp.asarray(nxt)) / wm["d_std"]
    return jnp.sum(z**2) / (wm["mask"].sum() * pred.shape[0])


def collect_transitions(env, env_params, policy, level, key, num_envs, horizon, sigma, num_steps):
    """(obs, executed action, next obs, valid) at every step of naive chunked rollouts with action noise.

    Like probe.collect, but per step. valid = the first episode is still running and this step did not end it
    (AutoReplay would put the reset observation into next obs). Each output is [C, horizon, E, ...].
    """
    k_reset, k_run = jax.random.split(key)
    obs, state = jax.vmap(env.reset_to_level, in_axes=(0, None, None))(
        jax.random.split(k_reset, num_envs), level, env_params
    )
    env_step = jax.vmap(env.step, in_axes=(0, 0, 0, None))

    def run_chunk(carry, key):
        k_act, k_noise, k_env = jax.random.split(key, 3)
        actions = policy.action(k_act, carry[0], num_steps)[:, :horizon]
        actions = actions + sigma * jax.random.normal(k_noise, actions.shape)

        def one_step(c, xs):
            obs, state, alive = c
            action, k = xs
            nobs, state, _, done, _ = env_step(jax.random.split(k, num_envs), state, action, env_params)
            return (nobs, state, alive & ~done), (obs, action, nobs, alive & ~done)

        return jax.lax.scan(one_step, carry, (actions.swapaxes(0, 1), jax.random.split(k_env, horizon)))

    n_chunks = env_params.max_timesteps // horizon
    return jax.lax.scan(run_chunk, (obs, state, jnp.ones(num_envs, bool)), jax.random.split(k_run, n_chunks))[1]


def train(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = probe.LEVELS,
    num_envs: int = 128,
    horizon: int = 4,
    hidden: int = 256,
    steps: int = 10_000,
    batch: int = 512,
    lr: float = 1e-3,
    num_flow_steps: int = 5,
    seed: int = 100,  # level i uses seed + i: disjoint from every eval seed
    out_dir: str = WM_DIR,
):
    """One world model per level from naive rollouts with action noise (spec B1 3.3); the last 10% of envs validate."""
    env, env_params, levels, obs_dim, action_dim = probe.setup(level_paths)
    out = pathlib.Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def data(state_dict, level, key):
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        return collect_transitions(
            env, env_params, policy, level, key, num_envs, horizon, train_expert.ACTION_NOISE_STD, num_flow_steps
        )

    n_val = max(1, num_envs // 10)
    tr, va = slice(0, num_envs - n_val), slice(num_envs - n_val, num_envs)
    rows = []
    for i, level_path in enumerate(level_paths):
        k_data, k_fit = jax.random.split(jax.random.key(seed + i))
        level = jax.tree.map(lambda x: x[i], levels)
        obs, act, nxt, valid = jax.device_get(data(probe.load_state_dict(run_path, level_path), level, k_data))

        def pick(x, envs):  # valid transitions of these envs, [N, ...]
            return x[:, :, envs].reshape(-1, *x.shape[3:])[valid[:, :, envs].reshape(-1)]

        wm = fit(pick(obs, tr), pick(act, tr), pick(nxt, tr), k_fit, hidden, steps, batch, lr)
        val = float(one_step_nmse(wm, pick(obs, va), pick(act, va), pick(nxt, va)))
        with (out / f"{level_name(level_path)}.pkl").open("wb") as f:
            pickle.dump(jax.device_get(wm), f)
        rows.append({
            "level": level_path, "train": int(valid[:, :, tr].sum()), "val": int(valid[:, :, va].sum()),
            "moving_dims": int(wm["mask"].sum()), "val_nmse": val,
        })
        print(rows[-1])
        pd.DataFrame(rows).to_csv(out / "train_log.csv", index=False)  # after every level
```
В самый конец файла добавить CLI:
```python
if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"train": train})
```

- [ ] **Step 4: Прогнать тесты**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_predictors.py`
Expected: 4 passed.

- [ ] **Step 5: Smoke обучения на одном уровне (~2–4 мин)**

Run:
```bash
UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline src/predictors.py train --level-paths worlds/l/mjc_walker.json \
  --num-envs 32 --steps 300 --out-dir results/smoke_b1/wm
```
Expected:
- напечатана строка `{'level': 'worlds/l/mjc_walker.json', 'train': …, 'val': …, 'moving_dims': …, 'val_nmse': …}` с `train > 1000`, `0 < moving_dims < 679`, конечным `val_nmse`;
- появились файлы `results/smoke_b1/wm/worlds_l_mjc_walker.pkl` и `train_log.csv`.

- [ ] **Step 6: Commit**

```bash
git add src/predictors.py tests/test_predictors.py
git commit -m "feat(predictors): learned world model (per-level MLP on observation deltas) and its training CLI"
```

---

### Task 6: офлайн-таблица ошибок прогноза и калибровка

**Files:**
- Modify: `src/predictors.py`
- Test: `tests/test_predictors.py`

- [ ] **Step 1: Написать падающий тест**

Добавить в конец `tests/test_predictors.py`:
```python
def test_normalized_error_counts_only_moving_dims():
    truth = jnp.zeros((2, 3))
    std = jnp.array([2.0, 1.0, 0.0])  # dim 2 never moves
    pred = truth.at[:, 0].set(2.0).at[:, 2].set(100.0)
    np.testing.assert_allclose(predictors.normalized_error(pred, truth, std), [1.0, 1.0])
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_predictors.py -k normalized`
Expected: FAIL with `AttributeError: module 'predictors' has no attribute 'normalized_error'`.

- [ ] **Step 3: Реализация**

В импорты `src/predictors.py` добавить `import json` (в начало, по алфавиту) и `import reflex` (после `import probe`). Перед блоком `if __name__ == "__main__":` добавить:
```python
def normalized_error(pred, truth, std):
    """L2 distance over the dims that move (std > 1e-6), each in units of its std: [..., O] -> [...]."""
    moving = std > 1e-6
    z = jnp.where(moving, (pred - truth) / jnp.where(moving, std, 1.0), 0.0)
    return jnp.sqrt(jnp.sum(z**2, axis=-1))


def errors(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = probe.LEVELS,
    phys: Sequence[float] = (0.1, 0.2, 0.3),
    world_model_dir: str = WM_DIR,
    num_envs: int = 64,
    num_states: int = 128,
    num_draws: int = 4,
    num_flow_steps: int = 5,
    seed: int = 2000,  # level i uses seed + i: disjoint from world-model data and every eval seed
    out: str = "results/b1/errors.csv",
):
    """Offline prediction error of every predictor vs the noise-free truth (spec B1 4.2).

    err = normalized_error(o^_k, o*_k) for k = 1..H-1 after one policy call; 'noise' is the deviation that action noise
    sigma = 0.1 causes, and ratio = err / noise. Prints the median ratio over levels and the calibration decision.
    """
    env, env_params, levels, obs_dim, action_dim = probe.setup(level_paths)
    base = env._env  # raw Kinetix env: no auto-reset, as in eval's predictor
    sigma = train_expert.ACTION_NOISE_STD

    @jax.jit
    def level_errors(state_dict, level, key, wm):
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        H = policy.action_chunk_size
        k_c, k_s, k_p = jax.random.split(key, 3)
        boundaries = probe.collect(env, env_params, policy, level, k_c, num_envs, 4, sigma, num_flow_steps)
        all_obs, alive = boundaries[0].reshape(-1, obs_dim), boundaries[2].reshape(-1, 1)
        mean = (all_obs * alive).sum(0) / alive.sum()
        std = jnp.sqrt((jnp.square(all_obs - mean) * alive).sum(0) / alive.sum())
        obs, state = probe.sample(boundaries, k_s, num_states)

        def one(x):
            o, st, k = x
            k_z, k_f, k_n = jax.random.split(k, 3)
            z = jax.random.normal(k_z, (1, H, action_dim))
            acts = policy.action_from_noise(z, o[None], num_flow_steps)[0, : H - 1]
            raw = st.env_state

            def true_step(s, a):
                return base.step_env(k, s, a, env_params)[:2]

            truth = reflex.nominal_obs(true_step, raw, acts)  # [K, O]
            preds = {"oracle": truth, "hold": jnp.broadcast_to(o, truth.shape)}
            for p in phys:
                f = phys_factors(k_f, raw, p)  # same key: same signs for every p
                preds[f"phys{p}"] = reflex.nominal_obs(
                    lambda s, a, f=f: phys_step(base, k, s, a, env_params, f), raw, acts
                )
            if wm is not None:
                preds["learned"] = wm_rollout(wm, o, acts)
            noisy = acts + sigma * jax.random.normal(k_n, (num_draws, *acts.shape))
            dev = jax.vmap(lambda a: reflex.nominal_obs(true_step, raw, a))(noisy)  # [M, K, O]
            e = {name: normalized_error(v, truth, std) for name, v in preds.items()}
            e["noise"] = normalized_error(dev, truth, std).mean(0)
            return e

        return jax.lax.map(one, (obs, state, jax.random.split(k_p, num_states)), batch_size=16)

    rows = []
    out_path = pathlib.Path(out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    for i, level_path in enumerate(level_paths):
        f = pathlib.Path(world_model_dir) / f"{level_name(level_path)}.pkl"
        wm = pickle.load(f.open("rb")) if f.exists() else None
        e = jax.device_get(level_errors(
            probe.load_state_dict(run_path, level_path), jax.tree.map(lambda x: x[i], levels),
            jax.random.key(seed + i), wm,
        ))
        for name, v in e.items():  # v [num_states, K]
            rows += [{"level": level_path, "predictor": name, "k": k + 1, "err": float(v[:, k].mean())}
                     for k in range(v.shape[1])]
        print(f"{level_path}: done")
        pd.DataFrame(rows).to_csv(out_path, index=False)  # after every level
    df = pd.DataFrame(rows)
    noise = df[df["predictor"] == "noise"].set_index(["level", "k"])["err"]
    df["ratio"] = df["err"].to_numpy() / noise.reindex(pd.MultiIndex.from_frame(df[["level", "k"]])).to_numpy()
    df.to_csv(out_path, index=False)
    table = df.groupby(["predictor", "k"])["ratio"].median().unstack("k")
    print("median over levels of err / noise:")
    print(table.round(2).to_string())
    mid = sorted(phys)[len(phys) // 2]
    ratio = float(table.loc[f"phys{mid}", 4])
    print(json.dumps({"calibration": {"phys_mid": mid, "ratio_k4": ratio, "double_levels": ratio < 1.0}}))
```
Строку CLI заменить на:
```python
    tyro.extras.subcommand_cli_from_dict({"train": train, "errors": errors})
```

- [ ] **Step 4: Прогнать тесты**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_predictors.py`
Expected: 5 passed.

- [ ] **Step 5: Smoke таблицы на одном уровне (~3–5 мин)**

Модель из задачи 5 лежит в `results/smoke_b1/wm`. Run:
```bash
UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline src/predictors.py errors --level-paths worlds/l/mjc_walker.json \
  --num-states 16 --world-model-dir results/smoke_b1/wm --out results/smoke_b1/errors.csv
```
Expected:
- таблица со строками `hold`, `learned`, `noise`, `oracle`, `phys0.1`, `phys0.2`, `phys0.3` и столбцами k = 1…7;
- у `oracle` ratio = 0.00, у `noise` = 1.00;
- ratio у `phys` растёт с p;
- последняя строка — JSON `{"calibration": {...}}`.

- [ ] **Step 6: Commit**

```bash
git add src/predictors.py tests/test_predictors.py
git commit -m "feat(predictors): offline prediction-error table in units of the action-noise deviation, calibration rule"
```

---

### Task 7: выбор предсказателя в цикле оценки

**Files:**
- Modify: `src/eval_flow.py`
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Написать падающий тест**

Добавить в конец `tests/test_reflex.py`:
```python
def test_parse_predictor():
    assert eval_flow.parse_predictor("oracle") == {"predictor": "oracle"}
    assert eval_flow.parse_predictor("learned") == {"predictor": "learned"}
    assert eval_flow.parse_predictor("phys0.2") == {"predictor": "phys", "phys_error": 0.2}
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_reflex.py -k parse_predictor`
Expected: FAIL with `AttributeError: module 'eval_flow' has no attribute 'parse_predictor'`.

- [ ] **Step 3: Конфигурация и разбор имён**

В `src/eval_flow.py`:
- В импорты добавить `import time` (после `import pickle`) и `import predictors as _predictors` (после `import model as _model`).
- В `ReflexMethodConfig` после строки `rtc: bool = False …` добавить:
```python
    predictor: str = "oracle"  # B1: "oracle" | "phys" (wrong physics) | "learned" (world model), spec B1 3.1
    phys_error: float = 0.0  # B1: relative parameter error of the "phys" predictor
```
- После функции `horizons_for` добавить:
```python
def parse_predictor(name: str) -> dict:
    """CLI predictor name -> ReflexMethodConfig fields: 'oracle', 'learned' or 'phys<p>' (e.g. 'phys0.2')."""
    if name.startswith("phys"):
        return {"predictor": "phys", "phys_error": float(name[4:])}
    assert name in ("oracle", "learned"), f"unknown predictor {name!r}"
    return {"predictor": name}
```

- [ ] **Step 4: Предсказатель внутри `eval`**

- Сигнатуру `eval` дополнить последним параметром:
```python
    weak_policy: _model.FlowPolicy | None = None,
    world_model=None,
):
```
- Сразу после строки `assert s + d <= policy.action_chunk_size, …` добавить:
```python
    phys_key = jax.random.fold_in(rng, 1)  # B1: the same parameter-error signs for every method (paired comparison)

    def predict(key, raw_state, obs, actions):
        """Predicted obs after each planned action [B, T, A] -> [B, T, O] (spec B1 3.1)."""
        m = config.method
        if m.predictor == "learned":
            assert world_model is not None, "predictor 'learned' needs world models: src/predictors.py train"
            return jax.vmap(_predictors.wm_rollout, in_axes=(None, 0, 0))(world_model, obs, actions)
        if m.predictor == "phys":
            factors = _predictors.phys_factors(phys_key, raw_state, m.phys_error)
            step = jax.vmap(functools.partial(_predictors.phys_step, base_env), in_axes=(None, 0, 0, None, 0))

            def fn(st, a):
                return step(key, st, a, env_params, factors)
        else:
            assert m.predictor == "oracle", m.predictor

            def fn(st, a):
                return nominal_step(key, st, a, env_params)[:2]

        return reflex.nominal_obs(fn, raw_state, actions.swapaxes(0, 1)).swapaxes(0, 1)
```
- В ветке `elif is_reflex:` заменить
```python
            pred = reflex.nominal_obs(
                lambda st, a: nominal_step(key, st, a, env_params)[:2],
                env_state.env_state.env_state,  # BatchEnv/LogWrapper -> AutoReplay -> raw EnvState
                planned[:, :n_pred].swapaxes(0, 1),
            )
            ref = jnp.concatenate([obs[:, None], pred.swapaxes(0, 1)], axis=1)  # predicted obs per chunk index
```
на
```python
            # BatchEnv/LogWrapper -> AutoReplay -> raw EnvState
            pred = predict(key, env_state.env_state.env_state, obs, planned[:, :n_pred])  # [B, n_pred, O]
            ref = jnp.concatenate([obs[:, None], pred], axis=1)  # predicted obs per chunk index
```

- [ ] **Step 5: CLI и модели мира в `main`**

В `main`:
- к параметрам после `package_batch: int = 16,` добавить:
```python
    predictors: Sequence[str] = ("oracle",),  # B1: predictors for the reflex methods, see parse_predictor
    world_model_dir: str = _predictors.WM_DIR,
```
- после блока с `weak_state_dicts` (перед `obs_dim = …`) добавить:
```python
    world_models = None
    if "learned" in predictors:
        wms = []
        for level_path in level_paths:
            with (pathlib.Path(world_model_dir) / f"{_predictors.level_name(level_path)}.pkl").open("rb") as f:
                wms.append(pickle.load(f))
        world_models = jax.device_put(jax.tree.map(lambda *x: jnp.array(x), *wms))
```
- в декораторе `shard_map` заменить `in_specs=(None, pspec, pspec, pspec, pspec)` на `in_specs=(None, pspec, pspec, pspec, pspec, pspec)`, а в `jax.vmap` заменить `in_axes=(None, 0, 0, 0, 0)` на `in_axes=(None, 0, 0, 0, 0, 0)`;
- сигнатуру `_eval` заменить на `def _eval(config: EvalConfig, rng: jax.Array, level: kenv_state.EnvState, state_dict, weak_state_dict, world_model):`, а вызов внутри — на `eval_info, _ = eval(config, env, rng, level, policy, env_params, static_env_params, weak_policy, world_model)`;
- цикл `for name in methods:` целиком заменить на:
```python
                for name in methods:
                    method = METHODS[name]
                    is_reflex = isinstance(method, ReflexMethodConfig)
                    for predictor in predictors if is_reflex else ("-",):
                        print(f"{time.strftime('%H:%M:%S')} {seed=} {name=} {predictor=} {inference_delay=} "
                              f"{execute_horizon=}")
                        m = method
                        if is_reflex:
                            m = dataclasses.replace(
                                method, max_correction=max_correction, package_batch=package_batch,
                                **parse_predictor(predictor),
                            )
                        c = dataclasses.replace(
                            config, inference_delay=inference_delay, execute_horizon=execute_horizon, method=m
                        )
                        out = jax.device_get(_eval(c, rngs, levels, state_dicts, weak_state_dicts, world_models))
                        for i in range(len(level_paths)):
                            for k, v in out.items():
                                results[k].append(v[i])
                            results["seed"].append(seed)
                            results["delay"].append(inference_delay)
                            results["method"].append(name)
                            results["predictor"].append(predictor)
                            results["level"].append(level_paths[i])
                            results["execute_horizon"].append(execute_horizon)
                            results["max_correction"].append(max_correction if is_reflex else float("nan"))
                            results["kick_std"].append(config.kick_std if config.kick_prob > 0 else 0.0)
                        # after every config: a crash (or Ctrl-C) keeps the finished ones
                        pd.DataFrame(results).to_csv(pathlib.Path(output_dir) / "results.csv", index=False)
```

- [ ] **Step 6: Тесты и побитная проверка**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q`
Expected: 25 passed (20 в `test_reflex.py` + 5 в `test_predictors.py`).

Run: `uv run --offline python scripts/check_upstream_bitwise.py`
Expected: all `IDENTICAL`.

- [ ] **Step 7: Smoke в замкнутом цикле (~5–10 мин)**

Модель мира для `mjc_walker` лежит в `results/smoke_b1/wm` (задача 5). Run:
```bash
UV_OFFLINE=1 JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 uv run --offline src/eval_flow.py --run-path checkpoints/bc \
  --level-paths worlds/l/mjc_walker.json --config.num-evals 4 --seeds 99 --methods naive reflex rtc_reflex \
  --predictors oracle phys0.0 phys0.3 learned --world-model-dir results/smoke_b1/wm --delays 1 --horizons 2 \
  --output-dir results/smoke_b1/eval
```
Потом:
```bash
uv run --offline python - <<'EOF'
import pandas as pd
df = pd.read_csv("results/smoke_b1/eval/results.csv")
print(df[["method", "predictor", "execute_horizon", "returned_episode_solved", "returned_episode_returns"]])
assert set(df["predictor"]) == {"-", "oracle", "phys0.0", "phys0.3", "learned"}
for m in ("reflex", "rtc_reflex"):
    r = df[df["method"] == m].set_index("predictor")
    assert r.loc["phys0.0", "returned_episode_returns"] == r.loc["oracle", "returned_episode_returns"], f"{m}: phys0 != oracle"
assert df["returned_episode_returns"].notna().all()
print("OK")
EOF
```
Expected: 9 строк (`naive` с `-`, `reflex` и `rtc_reflex` с четырьмя предсказателями каждый), а в конце `OK`. `phys0.0` совпадает с `oracle` точно у обоих методов.

- [ ] **Step 8: Commit**

```bash
git add src/eval_flow.py tests/test_reflex.py
git commit -m "feat(eval): B1 predictors for the reflex (oracle / phys<p> / learned), --predictors, predictor column"
```

---

### Task 8: правила решений B1 (`plot.py b1`)

**Files:**
- Modify: `src/plot.py`
- Test: `tests/test_reflex.py`

- [ ] **Step 1: Написать падающий тест**

Добавить в конец `tests/test_reflex.py`:
```python
def _b1_rows(mid_pred=0.62, mid_extra=0.02, mid_rtc=0.62, mid_d3=0.6, base=(0.5, 0.6)):
    """Synthetic B1 results: d = 1, s = 1..7 (5 predictors for pred / reflex, 3 for rtc_reflex) and d = 3, s = 5."""
    rows = []

    def add(d, seed, s, method, predictor, rate):
        rows.append({"delay": d, "seed": seed, "execute_horizon": s, "method": method, "predictor": predictor,
                     "level": "l", "returned_episode_solved": rate})

    for seed in (10, 11, 12):
        for s in range(1, 8):
            add(1, seed, s, "naive", "-", base[0])
            add(1, seed, s, "realtime", "-", base[1])
            for pr, pred, j in (
                ("oracle", 0.62, 0.01 * s),
                ("phys0.1", 0.62, 0.01 * s + 0.01),
                ("phys0.2", mid_pred, 0.01 * s + mid_extra),
                ("phys0.3", 0.62, 0.01 * s),
                ("learned", 0.4, 0.0),
            ):
                add(1, seed, s, "pred", pr, pred)
                add(1, seed, s, "reflex", pr, pred + j)
            for pr, rate in (("oracle", 0.62), ("phys0.2", mid_rtc), ("learned", 0.55)):
                add(1, seed, s, "rtc_reflex", pr, rate)
        add(3, seed, 5, "naive", "-", base[0] - 0.1)
        add(3, seed, 5, "realtime", "-", base[1] - 0.1)
        for pr, reflex_rate in (("oracle", 0.6), ("phys0.2", mid_d3), ("learned", 0.45)):
            add(3, seed, 5, "pred", pr, 0.5)
            add(3, seed, 5, "reflex", pr, reflex_rate)
            add(3, seed, 5, "rtc_reflex", pr, 0.45)
    return pd.DataFrame(rows)


def test_b1_rules(tmp_path):
    def run(**kw):
        d = tmp_path / str(len(list(tmp_path.iterdir())))
        d.mkdir()
        _b1_rows(**kw).to_csv(d / "results.csv", index=False)
        return plot.b1(str(d / "*.csv"), errors_csv=str(tmp_path / "missing.csv"), out_dir=str(d))

    out = run()  # the oracle passes both slices; phys0.2 passes D1 through reflex; learned fails everywhere
    assert out["p_mid"] == "phys0.2" and out["informative_slices"] == ["d1", "d3"]
    assert out["slices"]["d1"]["phys0.2"]["status"] == "PASS"
    assert out["verdict"] == "SURVIVES" and out["R4_learned"] == "TOO-WEAK"
    assert out["R2_staleness"] and abs(out["R2_gap"] - 0.04) < 1e-9  # J(oracle): s>=5 -> 0.06, s<=3 -> 0.02
    assert out["R3_j_grows_with_error"] and abs(out["R3_gap"] - 0.02) < 1e-9

    out = run(mid_pred=0.5, mid_extra=-0.2, mid_rtc=0.55, mid_d3=0.45)  # phys0.2 below the baseline everywhere
    assert out["verdict"] == "ORACLE-BOUND"

    out = run(mid_rtc=0.605, mid_pred=0.5, mid_extra=0.0, mid_d3=0.45)  # D1: rtc +0.5 pp (<1 pp), reflex < 0
    assert out["slices"]["d1"]["phys0.2"]["status"] == "GRAY" and out["verdict"] == "GRAY"

    out = run(base=(0.9, 0.95))  # nothing beats the baseline even with the oracle
    assert out["informative_slices"] == [] and out["verdict"] == "NO-EDGE" and out["R4_learned"] == "n/a"
```

- [ ] **Step 2: Убедиться, что тест падает**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_reflex.py -k b1_rules`
Expected: FAIL with `AttributeError: module 'plot' has no attribute 'b1'`.

- [ ] **Step 3: Реализация**

В `src/plot.py` первую строку docstring модуля заменить на `"""Figures and tables for E1 (probe), E2 (closed loop) and B1, and the Gate 2 / B1 decision rules."""`. Перед `if __name__ == "__main__":` добавить:
```python
B1_SLICES = {"d1": (1, (5, 6, 7)), "d3": (3, (5,))}  # spec B1 section 5: the rare-call slices behind the verdict


def _slice_status(G: pd.DataFrame, delay: int, horizons, predictor: str):
    """PASS / FAIL / GRAY / MISSING of one slice for one predictor (spec B1 section 5), with per-method details.

    G: reflex-type methods' G, index (delay, seed, predictor, execute_horizon). PASS needs pooled >= +1 pp and > 0 in
    every seed for the same method; FAIL needs pooled <= 0 for every method that ran.
    """
    try:
        g = G.xs((delay, predictor), level=("delay", "predictor"))
    except KeyError:
        return "MISSING", {}
    g = g[g.index.get_level_values("execute_horizon").isin(horizons)].groupby(level="seed").mean()
    detail = {
        m: {"pooled": float(g[m].mean()), "min_seed": float(g[m].min())}
        for m in ("reflex", "rtc_reflex")
        if m in g and len(g) and g[m].notna().all()
    }
    if not detail:
        return "MISSING", {}
    if any(v["pooled"] >= 0.01 and v["min_seed"] > 0 for v in detail.values()):
        return "PASS", detail
    if all(v["pooled"] <= 0 for v in detail.values()):
        return "FAIL", detail
    return "GRAY", detail


def b1(
    results_glob: str = "results/b1/gpu/eval*/results.csv",
    errors_csv: str = "results/b1/gpu/errors.csv",
    out_dir: str = "results/b1/gpu",
) -> dict:
    """B1 verdict and rules R1-R4 (spec B1 section 5), b1.json and b1.png. Solve rates are means over levels.

    G = method - max(naive, realtime) at the same (delay, seed, s); J = reflex - pred at d = 1; p_mid = middle phys.
    """
    df = pd.concat([pd.read_csv(f) for f in sorted(glob.glob(results_glob))])
    lv = df.groupby(["delay", "seed", "method", "predictor", "execute_horizon"])["returned_episode_solved"].mean()
    base = lv.xs("-", level="predictor").unstack("method")  # (delay, seed, s) x {naive, realtime}
    t = lv.drop("-", level="predictor").unstack("method")  # (delay, seed, predictor, s) x {pred, reflex, rtc_reflex}
    best = base[["naive", "realtime"]].max(axis=1).reindex(t.index.droplevel("predictor")).to_numpy()
    G = t.sub(pd.Series(best, index=t.index), axis=0)
    t1 = t.xs(1, level="delay")
    J = t1["reflex"] - t1["pred"]  # (seed, predictor, s)
    phys = sorted(
        (p for p in J.index.get_level_values("predictor").unique() if p.startswith("phys")), key=lambda p: float(p[4:])
    )
    p_mid = phys[len(phys) // 2]
    jo = J.xs("oracle", level="predictor").unstack("execute_horizon")  # seed x s
    stale = jo.loc[:, jo.columns >= 5].mean(axis=1) - jo.loc[:, jo.columns <= 3].mean(axis=1)
    jbar = J.groupby(level=["predictor", "seed"]).mean()
    grow = jbar[p_mid] - jbar["oracle"]

    out = {"p_mid": p_mid, "slices": {}}
    for name, (d, horizons) in B1_SLICES.items():
        out["slices"][name] = {}
        for pr in ("oracle", p_mid, "learned"):
            status, detail = _slice_status(G, d, horizons, pr)
            out["slices"][name][pr] = {"status": status, **detail}
    informative = [n for n in B1_SLICES if out["slices"][n]["oracle"]["status"] == "PASS"]
    mids = [out["slices"][n][p_mid]["status"] for n in informative]
    learned = [out["slices"][n]["learned"]["status"] for n in informative]
    out["informative_slices"] = informative
    out["verdict"] = (
        "NO-EDGE" if not informative
        else "SURVIVES" if "PASS" in mids
        else "ORACLE-BOUND" if all(m == "FAIL" for m in mids)
        else "GRAY"
    )
    out["R4_learned"] = (
        "n/a" if not informative
        else "ENOUGH" if "PASS" in learned
        else "TOO-WEAK" if all(m == "FAIL" for m in learned)
        else "GRAY"
    )
    out |= {
        "R2_staleness": bool(stale.mean() >= 0.03 and (stale > 0).all()),
        "R2_gap": float(stale.mean()),
        "R3_j_grows_with_error": bool(grow.mean() >= 0.01 and (grow > 0).all()),
        "R3_gap": float(grow.mean()),
    }
    out["drop_oracle_to_p_mid"] = {  # report only: the spec's prediction is that rtc_reflex drops more than reflex
        n: {
            m: out["slices"][n]["oracle"][m]["pooled"] - out["slices"][n][p_mid][m]["pooled"]
            for m in ("reflex", "rtc_reflex")
            if m in out["slices"][n]["oracle"] and m in out["slices"][n][p_mid]
        }
        for n in B1_SLICES
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for pr, g in J.groupby(level="predictor"):
        g = g.groupby(level="execute_horizon").mean()
        axes[0].plot(g.index, g * 100, marker="o", label=pr)
    g1 = G.xs(1, level="delay").groupby(level=["predictor", "execute_horizon"]).mean()
    for pr, g in g1.groupby(level="predictor"):
        s = g.index.get_level_values("execute_horizon")
        line = axes[1].plot(s, g["reflex"] * 100, marker="o", label=pr)[0]
        if "rtc_reflex" in g and g["rtc_reflex"].notna().any():
            axes[1].plot(s, g["rtc_reflex"] * 100, ls="--", c=line.get_color())
    axes[0].set_title("d = 1: J = reflex − pred (pp)")
    axes[1].set_title("d = 1: G = method − max(naive, RTC) (pp); dashed: rtc_reflex")
    for ax in axes[:2]:
        ax.axhline(0, c="gray", lw=0.8)
        ax.set_xlabel("execute horizon s")
    axes[0].legend(fontsize=8)
    if pathlib.Path(errors_csv).exists():
        e = pd.read_csv(errors_csv)
        ratio = e[e["k"] == 4].groupby("predictor")["ratio"].median()
        jb = J.groupby(level="predictor").mean()
        common = [p for p in jb.index if p in ratio.index]
        axes[2].scatter(ratio[common], jb[common] * 100)
        for p in common:
            axes[2].annotate(p, (ratio[p], jb[p] * 100), fontsize=8)
        axes[2].set_xlabel("prediction error at k = 4 (units of action-noise deviation)")
    axes[2].set_title("d = 1: mean J over s vs predictor error (pp)")
    fig.suptitle(f"B1: {out['verdict']}")
    fig.tight_layout()
    out_path = pathlib.Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path / "b1.png", dpi=150)
    plt.close(fig)
    print((G.groupby(level=["delay", "predictor", "execute_horizon"]).mean() * 100).round(1).to_string())
    print("network evaluations per step:", {
        m: [round(reflex.forward_equivalents(m, positions=s) / s, 1) for s in range(1, 8)]
        for m in ("naive", "realtime", "pred", "reflex", "rtc_reflex")
    })
    (out_path / "b1.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    return out
```
В `subcommand_cli_from_dict` добавить `"b1": b1`:
```python
    tyro.extras.subcommand_cli_from_dict({"probe": probe, "success": success, "table": table, "gate2": gate2, "b1": b1})
```

- [ ] **Step 4: Прогнать тесты**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q`
Expected: 26 passed.

- [ ] **Step 5: Commit**

```bash
git add src/plot.py tests/test_reflex.py
git commit -m "feat(plot): B1 decision rules R1-R4, b1.json and the J / G / error figure"
```

---

### Task 9: скрипты репетиции и GPU

**Files:**
- Create: `scripts/run_b1_rehearsal.sh`
- Create: `scripts/gpu_b1.sh`
- Modify: `.gitignore`

- [ ] **Step 1: Репетиция на Mac**

Создать `scripts/run_b1_rehearsal.sh`:
```bash
#!/usr/bin/env bash
# B1-P + B1-R on the Mac CPU, fully offline (~2.5 h): world models, the prediction-error table with the calibration
# decision, and a tiny closed-loop grid on dev seed 99 (NOT the B1 seeds 10-12). The closed-loop numbers are
# exploratory: they check the pipeline and decide nothing (spec B1 4.2-4.3). Keep the Mac on power, lid open.
set -euo pipefail
cd "$(dirname "$0")/.."
export UV_OFFLINE=1 WANDB_MODE=disabled JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1
B=results/b1
OUT=$B/rehearsal
mkdir -p "$OUT"
start=$(date +%s)
caffeinate -i uv run src/predictors.py train --out-dir $B/world_models 2>&1 | tee "$OUT/train.txt"
caffeinate -i uv run src/predictors.py errors --world-model-dir $B/world_models --out $B/errors.csv 2>&1 \
  | tee "$OUT/errors.txt"
eval_run() {  # eval_run <dir> <eval_flow args...>
  local dir=$1; shift
  caffeinate -i uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 8 --seeds 99 \
    --world-model-dir $B/world_models --output-dir "$OUT/$dir" "$@" 2>&1 \
    | grep --line-buffered -v prefix_attention_horizon | tee "$OUT/$dir.log"
}
eval_run eval_d3 --methods naive realtime pred reflex rtc_reflex --predictors oracle phys0.2 learned --delays 3 --horizons 5
eval_run eval_d1 --methods naive realtime pred reflex --predictors oracle phys0.2 learned --delays 1 --horizons 1 7
eval_run eval_d1_rtc --methods rtc_reflex --predictors oracle phys0.2 learned --delays 1 --horizons 7
uv run src/plot.py b1 --results-glob "$OUT/eval*/results.csv" --errors-csv $B/errors.csv --out-dir "$OUT" 2>&1 \
  | tee "$OUT/b1.txt"
echo "B1 rehearsal done in $(( ($(date +%s) - start) / 60 )) min"
```

- [ ] **Step 2: GPU**

Создать `scripts/gpu_b1.sh`:
```bash
#!/usr/bin/env bash
# B1 on a Linux NVIDIA GPU host (spec 2026-09-25-b1-staleness-predictors-design.md, section 4.4). Run from the unpacked
# repo root: ./scripts/gpu_b1.sh. The host downloads Kinetix, packages and checkpoints itself and trains its own world
# models (same seeds as on the Mac). Writes results/b1/gpu/** and packs it (without model weights) into b1_results.tgz.
# PHYS = the three calibrated phys levels, middle one = p_mid (spec B1 4.2); PB = the Jacobian batch (4 if tight).
set -uo pipefail
cd "$(dirname "$0")/.."
KINETIX_SHA=cf7453ea103fa0b77348af1a39f689c658161613
if [ ! -d third_party/kinetix/kinetix ]; then
  rm -rf third_party/kinetix
  git clone -q https://github.com/FLAIROx/Kinetix.git third_party/kinetix && git -C third_party/kinetix checkout -q $KINETIX_SHA
fi
command -v uv >/dev/null || { curl -LsSf https://astral.sh/uv/install.sh | sh; source "$HOME/.local/bin/env"; }
uv sync || exit 1
mkdir -p checkpoints/bc/31/policies
for L in grasp_easy catapult cartpole_thrust hard_lunar_lander mjc_half_cheetah mjc_swimmer mjc_walker h17_unicycle chain_lander catcher_v3 trampoline car_launch; do
  f=checkpoints/bc/31/policies/worlds_l_$L.pkl
  [ -f "$f" ] || curl -fsSL -o "$f" "https://storage.googleapis.com/rtc-assets/bc/31/policies/worlds_l_$L.pkl"
done
uv run python -c "import jax; d = jax.devices(); print(d); assert d[0].platform == 'gpu', 'no GPU visible'" || exit 1
export JAX_COMPILATION_CACHE_DIR=$HOME/.cache/jax CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0} PYTHONUNBUFFERED=1
G=results/b1/gpu
PB=${PB:-16}
PHYS=${PHYS:-"0.1 0.2 0.3"}
PMID=$(echo $PHYS | awk '{print $2}')
PREDS="oracle $(for p in $PHYS; do printf 'phys%s ' "$p"; done)learned"
MID="oracle phys$PMID learned"
mkdir -p $G
echo "[$(date +%T)] world models"
uv run src/predictors.py train --out-dir $G/world_models 2>&1 | tee $G/train.txt
echo "[$(date +%T)] prediction errors"
uv run src/predictors.py errors --phys $PHYS --world-model-dir $G/world_models --out $G/errors.csv 2>&1 | tee $G/errors.txt
run() {  # run <dir> <eval_flow args...>; on failure (e.g. out of memory) retry once with a smaller Jacobian batch
  local dir=$1; shift
  echo "[$(date +%T)] $dir"
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 10 11 12 --package-batch $PB \
    --world-model-dir $G/world_models --output-dir $G/$dir "$@" 2>&1 | grep --line-buffered -v prefix_attention_horizon && return
  echo "[$(date +%T)] $dir failed, retrying with --package-batch 4"
  uv run src/eval_flow.py --run-path checkpoints/bc --config.num-evals 256 --seeds 10 11 12 --package-batch 4 \
    --world-model-dir $G/world_models --output-dir $G/$dir "$@" 2>&1 | grep --line-buffered -v prefix_attention_horizon \
    || echo "FAILED $dir"
}
run eval_d3 --methods naive realtime pred reflex rtc_reflex --predictors $MID --delays 3 --horizons 5
run eval_d1 --methods naive realtime pred reflex --predictors $PREDS --delays 1
run eval_d1_rtc --methods rtc_reflex --predictors $MID --delays 1
for s in 1 4 7; do
  uv run src/probe.py cost --batch 1 --delay 1 --horizon $s --out $G/cost_b1_s$s.csv 2>&1 | tee -a $G/cost.txt
done
uv run src/plot.py b1 --results-glob "$G/eval*/results.csv" --errors-csv $G/errors.csv --out-dir $G 2>&1 | tee $G/b1.txt
tar czf b1_results.tgz --exclude='*.pkl' $G && echo "[$(date +%T)] done: b1_results.tgz"
```

- [ ] **Step 3: Права и `.gitignore`**

Run: `chmod +x scripts/run_b1_rehearsal.sh scripts/gpu_b1.sh`

В конец `.gitignore` (после строки `.DS_Store`) добавить:
```
results/**/*.pkl
*.tgz
```

- [ ] **Step 4: Проверка синтаксиса**

Run: `bash -n scripts/run_b1_rehearsal.sh && bash -n scripts/gpu_b1.sh && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/run_b1_rehearsal.sh scripts/gpu_b1.sh .gitignore
git commit -m "feat(scripts): B1 rehearsal on the Mac and the B1 GPU runner"
```

---

### Task 10: репетиция и калибровка (Mac, владелец запускает сам, ≈2.5 ч офлайн)

- [ ] **Step 1: Прогон** (владелец, Mac на зарядке)

```bash
./scripts/run_b1_rehearsal.sh
```
Expected: в конце `B1 rehearsal done in … min`. До этого:
- в `errors.txt` — таблица и JSON `calibration`;
- в `b1.txt` — таблица G, JSON с вердиктом и статусами срезов, без NaN в решённых долях;
- в `results/b1/rehearsal/eval_d3`, `eval_d1`, `eval_d1_rtc` лежат `results.csv`.

- [ ] **Step 2: Калибровка в журнал спека (контроллер)**

Взять `ratio_k4` и `double_levels` из последней строки `results/b1/rehearsal/errors.txt`. В спек (`## Журнал изменений`) добавить запись:
```markdown
### 2026-09-DD — калибровка «кривой физики» (B1-P, до основного прогона)

Медиана по уровням `ratio_4(phys0.2)` = <число>. Правило §4.2: <меньше 1 → уровни 0.2/0.4/0.6, p_mid = 0.4 | не меньше 1 → уровни 0.1/0.2/0.3, p_mid = 0.2>. Точность моделей мира (val_nmse по уровням): <мин…макс>, `ratio_4(learned)` = <число>. Репетиция (seed 99, поисковая) прошла без ошибок.
```
Скрипт `scripts/gpu_b1.sh` не имеет уровней по умолчанию (исправление после финального ревью): откалиброванные уровни передаются через `PHYS` при запуске (задача 11).

- [ ] **Step 3: Commit (контроллер)**

```bash
git add docs/superpowers/specs/2026-09-25-b1-staleness-predictors-design.md scripts/gpu_b1.sh \
  results/b1/errors.csv results/b1/world_models/train_log.csv results/b1/rehearsal
git commit -m "results: B1 calibration and rehearsal (dev seed 99, exploratory); phys levels fixed before the GPU run"
```

---

### Task 11: основной прогон на GPU (владелец, RunPod RTX 4090)

- [ ] **Step 1: Архив кода** (Mac)

```bash
git archive --format=tar.gz -o /tmp/m2r.tgz HEAD
```
Залить на под, как в фазе A (`scp -P <port> /tmp/m2r.tgz root@<host>:`). На поде (`PHYS` обязателен: уровни из калибровки, задача 10; при удвоении — `"0.2 0.4 0.6"`):
```bash
mkdir m2r && tar xzf m2r.tgz -C m2r && cd m2r && tmux new -s b1 'PHYS="0.1 0.2 0.3" ./scripts/gpu_b1.sh 2>&1 | tee b1.log'
```

- [ ] **Step 2: Замер скорости** (через ~20 мин после старта сетки)

Прислать контроллеру строки с метками времени из `b1.log`. Контроллер оценивает полный срок по первым конфигурациям. Первым идёт срез d = 3 (11 конфигураций на seed), потом d = 1: 84 конфигурации `naive`/`realtime`/`pred`/`reflex` и 21 конфигурация `rtc_reflex` на seed. Процесс не останавливать.

- [ ] **Step 3: Забрать результаты** (после `done: b1_results.tgz`)

```bash
scp -P <port> root@<host>:m2r/b1_results.tgz . && tar xzf b1_results.tgz
```
Под удалить. Если в `b1.txt` есть `!!! some eval runs FAILED` или `b1` отказался считать неполную сетку, недостающие конфигурации досчитываются, прежде чем читать вердикт. Если вердикт GRAY, по §5 спека выполняется расширение на seed'ы 13–15; перед ним `b1` дополняется подсчётом seed'ов по срезам (журнал спека). Контроллер проверяет `results/b1/gpu/b1.json` и коммитит:
```bash
git add results/b1/gpu
git commit -m "results: B1 on GPU (RTX 4090, seeds 10-12 x 256 x 12 levels): predictors x s = 1..7, errors, cost"
```

---

### Task 12: записка B1

**Files:**
- Create: `docs/results/b1.md` (по-русски, как `docs/results/closed-loop.md`)
- Modify: спек B1 (строка «Статус» и журнал), `docs/roadmap.md` (статус B1), `docs/results/report.md` и `README.md` (раздел B1 на английском)

- [ ] **Step 1: Записка.** Разделы:
  1. вердикт B1 (срезы D1 и D3, статусы по предсказателям, контроль оракулом) и R1–R4 с числами из `b1.json`; при GRAY — расширение на seed'ы 13–15 по §5 спека до записки;
  2. кривые `J(s)` и `G(s)` по предсказателям (`b1.png`);
  3. таблица ошибок прогноза в единицах «шума» (`errors.csv`, k = 4);
  4. `P(pred)` по предсказателям;
  5. стоимость на шаг и новое отношение задержек r (`cost_b1_s*.csv`);
  6. ограничения;
  7. что дальше по правилу §5 спека.

Все утверждения — только из файлов `results/b1/gpu/*`. Поисковую репетицию подписать как поисковую.
- [ ] **Step 2: Статусы.**
  - В спеке B1 строку «Статус» заменить на `**B1 завершён <дата>** (вердикт: <…>; итог — docs/results/b1.md)` и добавить запись в журнал.
  - В `docs/roadmap.md` под B1 — одна строка с вердиктом и ссылкой.
  - В `docs/results/report.md` — раздел «B1» на английском, 1–2 абзаца, и ссылка из README.
- [ ] **Step 3: Commit**

```bash
git add docs/results/b1.md docs/superpowers/specs/2026-09-25-b1-staleness-predictors-design.md docs/roadmap.md \
  docs/results/report.md README.md
git commit -m "docs: B1 memo (<verdict>), spec status, roadmap and report"
```
