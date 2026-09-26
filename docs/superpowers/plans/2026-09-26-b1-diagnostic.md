# Диагностика B1 — план реализации

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Офлайн-диагностика по спеку `docs/superpowers/specs/2026-09-26-b1-diagnostic-design.md`. Она отвечает на три вопроса:
- Q1: почему обученная модель вредит меньше «кривой физики»;
- Q2: почему `rtc_reflex` с моделью лучше, чем с оракулом;
- Q3: при каком отклонении и на каком шаге (до 31) ломается прикидка `J`.

**Architecture:** Новый файл `src/diag.py`:
- `probe_state` — все величины для одного состояния: план из 4 чанков, настоящие прогоны с шумом, 5 предсказателей, `J` и свежие вызовы, как в E1b;
- `run` — цикл по уровням, сохраняет сырые массивы в `results/b1/diag/raw/*.npz`;
- `summarize` — таблицы, пороги и рисунок из этих массивов.

Код переиспользует `probe.collect/sample/errors/executed`, `reflex.first_action_and_jacobian/shifted_noise` и `predictors.phys_factors/phys_step/wm_rollout/normalized_error`. `eval_flow` не трогаем.

**Tech Stack:** JAX 0.4.35 (CPU), Flax NNX, Kinetix (`third_party/kinetix`), numpy, pandas, matplotlib, tyro, pytest (`pythonpath = src`), uv.

---

## Общие правила (для каждого исполнителя)

- **Запуск.** Все команды — из корня `/Users/alexanderkarpov/Desktop/M2R`. Python — только через `uv run --offline …`: интернет мобильный. Тесты: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q …`.
- **Коммиты.** Сообщения на английском, в стиле истории. **Никогда не добавлять строку `Co-Authored-By`.** Без `git push`: пушит контроллер, когда разрешит владелец.
- **Долгие прогоны** (дольше ~5 мин) исполнитель не ждёт: пишет команду в отчёт, а запускает её контроллер в фоне.
- **Стиль.** Docstring и комментарии на английском, коротко, как в `src/probe.py`. Лишних абстракций нет.
- **Seed'ы.** Полный прогон — `3000 + i`. Smoke — `99`.
- **Публичность.** В коде, коммитах и записках никаких упоминаний космоса и применений: только нейтральное «рост ошибки с горизонтом».

## Карта файлов

| файл | ответственность |
|---|---|
| `src/probe.py` | `errors` получает ещё `chunk_lin_clip` (задача 1) |
| `src/predictors.py` | `alive_std` вынесен из `errors` (задача 1) |
| `src/diag.py` (новый) | помощники (задача 1), `roll`/`chain`/`probe_state` (задача 2), `run`/`summarize`/рисунок (задача 3) |
| `tests/test_reflex.py` | проверка `chunk_lin_clip` (задача 1) |
| `tests/test_diag.py` (новый) | тесты задач 1–3 |
| `.gitignore` | `results/**/*.npz` (задача 3) |
| `results/b1/diag/`, `docs/…` | прогоны и записка (задачи 4–6, контроллер) |

---

### Task 1: помощники

**Files:**
- Modify: `src/probe.py:64-85` (`errors`)
- Modify: `src/predictors.py:228-260` (`errors`: std через новую `alive_std`)
- Create: `src/diag.py`
- Test: `tests/test_reflex.py` (`test_lin_clip_uses_the_e2_bound`), `tests/test_diag.py`

- [ ] **Step 1: Write the failing tests**

В `tests/test_reflex.py`, в конец `test_lin_clip_uses_the_e2_bound`, добавить строку:

```python
    np.testing.assert_allclose(e["chunk_lin_clip"], [1.0 + 0.01], rtol=1e-6)  # chunk = a_ref here
```

Создать `tests/test_diag.py`:

```python
import jax.numpy as jnp
import numpy as np

import diag


def test_row_space_share_and_cosine():
    jac = jnp.array([[2.0, 0.0, 0.0], [0.0, 1.0, 0.0]])  # row space = axes 0 and 1, top direction = axis 0
    row, top = diag.row_space_share(jac, jnp.array([3.0, 4.0, 12.0]))
    np.testing.assert_allclose([row, top], [25.0, 9.0], rtol=1e-5)
    row, _ = diag.row_space_share(jnp.array([[1.0, 1.0, 0.0], [2.0, 2.0, 0.0]]), jnp.array([1.0, -1.0, 5.0]))
    np.testing.assert_allclose(row, 0.0, atol=1e-5)  # rank 1: (1, -1, 5) is orthogonal to the row space
    rows, _ = diag.row_space_share(jnp.stack([jac, 2 * jac]), jnp.ones((4, 2, 3)))  # [K, A, O] with [M, K, O]
    assert rows.shape == (4, 2)
    np.testing.assert_allclose(diag.cosine(jnp.array([1.0, 0.0]), jnp.array([1.0, 1.0])), 2**-0.5, rtol=1e-6)
    assert float(diag.cosine(jnp.zeros(2), jnp.ones(2))) == 0.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_diag.py tests/test_reflex.py -k "row_space or lin_clip_uses"`
Expected: FAIL — `ModuleNotFoundError: No module named 'diag'` и `KeyError: 'chunk_lin_clip'`.

- [ ] **Step 3: Implement**

В `src/probe.py`, функция `errors`: docstring дополнить словами «chunk_lin_clip is the clipped correction on top of the chunk (rtc_reflex-like)». В возвращаемый словарь после `"chunk_lin"` добавить:

```python
        "chunk_lin_clip": sq(act(chunk + jnp.clip(lin, -max_correction, max_correction)) - a_star),
```

`ERRORS` не менять: E1-сводка берёт только перечисленные там ключи.

В `src/predictors.py` перед `def errors(` добавить:

```python
def alive_std(boundaries):
    """Per-dim std of obs over the alive chunk-boundary states of probe.collect output: [O]."""
    obs = boundaries[0].reshape(-1, boundaries[0].shape[-1])
    alive = boundaries[2].reshape(-1, 1)
    mean = (obs * alive).sum(0) / alive.sum()
    return jnp.sqrt((jnp.square(obs - mean) * alive).sum(0) / alive.sum())
```

В `errors` (внутри `level_errors`) три строки

```python
        all_obs, alive = boundaries[0].reshape(-1, obs_dim), boundaries[2].reshape(-1, 1)
        mean = (all_obs * alive).sum(0) / alive.sum()
        std = jnp.sqrt((jnp.square(all_obs - mean) * alive).sum(0) / alive.sum())
```

заменить на `std = alive_std(boundaries)`. Операции те же, поведение не меняется.

Создать `src/diag.py`:

```python
"""B1 diagnostic (exploratory): which prediction error J sees, and at which deviation and horizon the tangent breaks.

For a sampled state: a plan of a few policy chunks, each queried where the noise-free truth arrives; real rollouts add
action noise; every predictor rolls the same plan. At step k the reflex's pi(ô_k) + J·(o_k − ô_k) is compared with a
fresh call pi(o_k), as in E1b. See docs/superpowers/specs/2026-09-26-b1-diagnostic-design.md.
"""

import jax
import jax.numpy as jnp


def row_space_share(jac, e):
    """|e|^2 inside J's row space and along its top right singular vector (divide by |e|^2 for the shares).

    jac [..., A, O], e [..., O] (broadcast) -> ([...], [...]). Singular values below 1e-6 of the largest are rank loss.
    """
    _, s, vt = jnp.linalg.svd(jac, full_matrices=False)
    proj = jnp.einsum("...ao,...o->...a", vt, e)
    return jnp.sum(jnp.where(s > 1e-6 * s[..., :1], proj**2, 0.0), -1), proj[..., 0] ** 2


def cosine(x, y):
    """Cosine over the last axis; 0 where either vector is zero."""
    den = jnp.linalg.norm(x, axis=-1) * jnp.linalg.norm(y, axis=-1)
    return jnp.where(den > 0, jnp.sum(x * y, -1) / jnp.where(den > 0, den, 1.0), 0.0)
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests`
Expected: 31 passed (30 прежних, один из них дополнен, плюс новый `test_row_space_share_and_cosine`).

- [ ] **Step 5: Commit**

```bash
git add src/probe.py src/predictors.py src/diag.py tests/test_reflex.py tests/test_diag.py
git commit -m "feat(diag): row-space share and cosine helpers; errors() gains chunk_lin_clip; alive_std factored out"
```

---

### Task 2: одно состояние — план, прогоны, предсказатели, меры

**Files:**
- Modify: `src/diag.py`
- Test: `tests/test_diag.py`

- [ ] **Step 1: Write the failing test**

Дописать в `tests/test_diag.py`, в начало — импорты:

```python
import jax

import predictors
import probe
from test_reflex import small_policy
```

и тест:

```python
def test_probe_state_oracle_invariants():
    env, params, levels, O, A = probe.setup(["worlds/l/grasp_easy.json"])
    obs, state = env.reset_to_level(jax.random.key(0), jax.tree.map(lambda x: x[0], levels), params)
    policy = small_policy(O, A)
    wm = {  # mask 0: this "model" holds the observation
        "layers": predictors.init(jax.random.key(1), O, A, 8), "x_mean": jnp.zeros(O), "x_std": jnp.ones(O),
        "d_mean": jnp.zeros(O), "d_std": jnp.ones(O), "mask": jnp.zeros(O),
    }
    out = diag.probe_state(
        policy, env._env, params, wm, (0.2,), state.env_state, obs, jax.random.key(2), jnp.ones(O),
        num_draws=2, num_chunks=2, num_steps=2,
    )
    P, M, K = 3, 2, 15  # oracle, phys0.2, learned; draws; k = 1..2*8-1
    assert out["e"].shape == (P, M, K) and out["valid"].shape == (M, K) and out["a_ref"].shape == (P, K, A)
    assert set(diag.SUMS + diag.MEANS) <= set(out)
    np.testing.assert_array_equal(out["e_pred"][0], 0)  # the oracle's prediction is the noise-free truth
    np.testing.assert_allclose(out["e"][0], out["e_noise"][0], rtol=1e-6)
    # chunk 1 was queried at o*_8 with zs[1]: at k = 8 the oracle's re-query must be the plan's own action.
    # A one-step shift of the noise index breaks this equality.
    np.testing.assert_allclose(out["a_ref"][0, 7], out["plan"][8], atol=1e-5)
    assert float(out["e_pred"][1].max()) > 0 and float(out["e_pred"][2].max()) > 0  # phys and the holding model err
    for name in diag.SUMS + diag.MEANS:
        assert np.isfinite(np.asarray(out[name])).all(), name
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_diag.py -k probe_state`
Expected: FAIL — `AttributeError: module 'diag' has no attribute 'probe_state'`.

- [ ] **Step 3: Implement**

В `src/diag.py` расширить импорты:

```python
from typing import Sequence

import jax
import jax.numpy as jnp

import predictors
import probe
import reflex
import train_expert
```

После импортов добавить константы:

```python
# per-pair quantities summed within a level (pooled ratios) and averaged (means); spec section 4
SUMS = ("pred", "lin", "lin_clip", "chunk", "chunk_lin_clip",
        "rs_pred", "top_pred", "d_pred", "rs_noise", "top_noise", "d_noise")
MEANS = ("e", "e_pred", "e_noise", "je", "je_pred", "je_noise", "cos_fix", "cos_spur", "clip")
```

В конец файла:

```python
def roll(step, state, actions):
    """step(state, action) -> (obs, state, done). actions [T, A] -> obs [T, O], ended by then [T], final state."""

    def body(c, a):
        s, ended = c
        o, s, done = step(s, a)
        return (s, ended | done), (o, ended | done)

    (state, _), (obs, ended) = jax.lax.scan(body, (state, jnp.bool_(False)), actions)
    return obs, ended, state


def chain(policy, step, state, obs, zs, num_steps: int):
    """Plan of len(zs) chunks; chunk c = pi(zs[c], o) at the obs where the noise-free truth arrived.

    zs [C, H, A] -> plan [C*H, A], truth obs after each action [C*H, O], ended by then [C*H].
    """
    plan, truth, ends, ended = [], [], [], jnp.bool_(False)
    for z in zs:  # C is small and static
        chunk = policy.action_from_noise(z[None], obs[None], num_steps)[0]
        o, e, state = roll(step, state, chunk)
        plan.append(chunk)
        truth.append(o)
        ends.append(e | ended)
        obs, ended = o[-1], ends[-1][-1]
    return jnp.concatenate(plan), jnp.concatenate(truth), jnp.concatenate(ends)


def probe_state(
    policy, base, params, wm, phys: Sequence[float], raw, obs, key, std, num_draws: int, num_chunks: int,
    num_steps: int,
):
    """Diagnostic of one state (spec sections 3-4).

    Returns SUMS + MEANS, each [P, M, K] (predictors oracle, phys..., learned; draws; k = 1..T-1), plus valid [M, K]
    (common to all predictors), a_ref [P, K, A] and plan [T, A] for the invariant checks. base is the raw Kinetix env
    (no auto-reset), raw its state. Actions are compared as executed (probe.executed), as in E1b.
    """
    H, A = policy.action_chunk_size, policy.action_dim
    k_z, k_f, k_n = jax.random.split(key, 3)
    zs = jax.random.normal(k_z, (num_chunks, H, A))

    def true_step(s, a):
        o, s, _, done, _ = base.step_env(key, s, a, params)
        return o, s, done

    plan, truth, truth_ended = chain(policy, true_step, raw, obs, zs, num_steps)
    T = plan.shape[0]
    K = T - 1
    noisy = plan + train_expert.ACTION_NOISE_STD * jax.random.normal(k_n, (num_draws, T, A))
    real, real_ended, _ = jax.vmap(lambda a: roll(true_step, raw, a))(noisy)  # [M, T, O], [M, T]

    preds = [truth]
    for p in phys:
        f = predictors.phys_factors(k_f, raw, p)  # same key: same signs for every p, as in B1

        def phys_step(s, a, f=f):
            o, s = predictors.phys_step(base, key, s, a, params, f)
            return o, s, jnp.bool_(False)  # a prediction is not cut by the episode's end

        preds.append(roll(phys_step, raw, plan)[0])
    preds.append(predictors.wm_rollout(wm, obs, plan))
    nom = jnp.stack(preds)[:, :K]  # [P, K, O]: ô_k after k actions, k = 1..K
    o, o_star = real[:, :K], truth[:K]  # [M, K, O], [K, O]
    z = jax.vmap(reflex.shifted_noise)(zs).reshape(T, H, A)[1:]  # z[k-1]: row 0 is the noise plan[k] came from
    plan_k = plan[1:]  # the plan's own action at step k: the nominal of rtc_reflex-like corrections (measure 8)
    a_star = policy.action_from_noise(
        jnp.broadcast_to(z, (num_draws, K, H, A)).reshape(-1, H, A), o.reshape(num_draws * K, -1), num_steps
    )[:, 0].reshape(num_draws, K, A)
    valid = ~(real_ended[:, :K] | truth_ended[:K])

    def act(a):
        return probe.executed(a, raw)

    need = act(a_star)

    def per_predictor(n):
        a_ref, jac = jax.vmap(lambda zk, ok: reflex.first_action_and_jacobian(policy, zk, ok, num_steps))(z, n)
        err = probe.errors(plan_k, a_ref, jac, n, o, a_star, act=act)  # [M, K]
        e_pred, e_noise = o_star - n, o - o_star  # [K, O], [M, K, O]
        je = jnp.einsum("kao,mko->mka", jac, o - n)
        je_pred = jnp.einsum("kao,ko->ka", jac, e_pred)
        rs_pred, top_pred = row_space_share(jac, e_pred)
        rs_noise, top_noise = row_space_share(jac, e_noise)
        out = {name: err[name] for name in ("pred", "lin", "lin_clip", "chunk", "chunk_lin_clip")} | {
            "rs_pred": rs_pred, "top_pred": top_pred, "d_pred": jnp.sum(e_pred**2, -1),
            "rs_noise": rs_noise, "top_noise": top_noise, "d_noise": jnp.sum(e_noise**2, -1),
            "e": predictors.normalized_error(n, o, std),
            "e_pred": predictors.normalized_error(n, o_star, std),
            "e_noise": predictors.normalized_error(o_star, o, std),
            "je": jnp.linalg.norm(je, axis=-1),
            "je_pred": jnp.linalg.norm(je_pred, axis=-1),
            "je_noise": jnp.linalg.norm(jnp.einsum("kao,mko->mka", jac, e_noise), axis=-1),
            "cos_fix": cosine(act(a_ref + jnp.clip(je, -1.0, 1.0)) - act(a_ref), need - act(a_ref)),
            "cos_spur": cosine(act(plan_k + jnp.clip(je_pred, -1.0, 1.0)) - act(plan_k), need - act(plan_k)),
            "clip": jnp.mean(jnp.abs(je) > 1.0, -1),
        }
        return jax.tree.map(lambda x: jnp.broadcast_to(x, valid.shape), out), a_ref

    stats, a_ref = jax.vmap(per_predictor)(nom)
    return stats | {"valid": valid, "a_ref": a_ref, "plan": plan}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests`
Expected: 32 passed. Новый тест может идти около минуты: там компиляция на CPU.

Если падает равенство на границе чанка, **не ослаблять допуск**. Ищем, где съехал индекс шума, и сверяем с `probe.probe_one`, где `shifted = reflex.shifted_noise(z)[1:]` стоит рядом с `nom_obs` после k действий.

- [ ] **Step 5: Commit**

```bash
git add src/diag.py tests/test_diag.py
git commit -m "feat(diag): probe_state: chunk-chained plan, noisy truth, every predictor, J vs fresh calls up to k = T-1"
```

---

### Task 3: цикл по уровням, сводки, пороги, рисунок

**Files:**
- Modify: `src/diag.py`, `.gitignore`
- Test: `tests/test_diag.py`

- [ ] **Step 1: Write the failing test**

Дописать в `tests/test_diag.py`, в импорты:

```python
import json

import pandas as pd
```

и тесты:

```python
def _raw(N=20, M=2, K=3, dead_k=None):
    """Synthetic probe_state output stacked over N states: rho_clip = 0.75 everywhere."""
    ones = np.ones((N, 2, M, K))
    r = {x: ones.copy() for x in diag.SUMS + diag.MEANS}
    r["lin_clip"] = 0.25 * ones
    r["e"] = np.random.default_rng(0).uniform(0.1, 5.0, (N, 2, M, K))
    valid = np.ones((N, M, K), bool)
    if dead_k is not None:
        valid[..., dead_k] = False  # every episode ended before this k: the level drops out there
    return r | {"valid": valid, "predictors": np.array(["oracle", "learned"])}


def test_summaries_pool_and_track_level_composition(tmp_path, monkeypatch):
    raws = {"a": _raw(), "b": _raw(dead_k=2)}
    t = pd.concat([diag.level_table(r, lv) for lv, r in raws.items()], ignore_index=True)
    np.testing.assert_allclose(t.loc[t["n"] > 0, "rho_clip"], 0.75)
    cv = diag.curves(t)
    al = cv[(cv["variant"] == "all") & (cv["predictor"] == "oracle")].set_index("k")["levels"]
    assert al.to_dict() == {1: 2, 2: 2, 3: 1}  # level b is gone at k = 3
    sv = cv[(cv["variant"] == "survivors") & (cv["predictor"] == "oracle")]
    assert len(sv) == 3 and (sv["levels"] == 1).all()  # only level a lives to the last k
    np.testing.assert_allclose(diag.near(t, k_max=2).loc["oracle", "rho_clip"], 0.75)
    assert diag.first_below([1, 2, 3], [0.5, 0.2, -0.1], 0.3) == 2.0
    assert diag.first_below([1, 2, 3], [0.5, 0.2, -0.1], 0.0) == 3.0
    assert diag.first_below([1, 2], [0.5, 0.4], 0.3) is None
    monkeypatch.setattr(diag, "MIN_BIN", 1)
    bn = diag.bins(raws, t)
    assert len(bn) == 10 and (bn["levels"] == 2).all()
    np.testing.assert_allclose(bn["rho_clip"], 0.75)
    (tmp_path / "raw").mkdir()
    for lv, r in raws.items():
        np.savez_compressed(tmp_path / "raw" / f"{lv}.npz", **r)
    diag.summarize(str(tmp_path))
    for f in ("summary.csv", "curves.csv", "near.csv", "bins.csv", "check.json", "thresholds.json", "diag.png"):
        assert (tmp_path / f).exists(), f
    assert json.loads((tmp_path / "check.json").read_text())["oracle_rho_clip_k1_4"] == 0.75
```

- [ ] **Step 2: Run test to verify it fails**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests/test_diag.py -k summaries`
Expected: FAIL — `AttributeError: module 'diag' has no attribute 'level_table'`.

- [ ] **Step 3: Implement**

Импорты `src/diag.py` целиком привести к такому виду. matplotlib подключается так же, как в `src/plot.py`:

```python
import json
import pathlib
import pickle
import time
from typing import Sequence

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np
import pandas as pd
import tyro

import predictors
import probe
import reflex
import train_expert

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
```

К константам добавить:

```python
OUT = "results/b1/diag"
MIN_FRAC = 0.3  # spec section 8: a level's point at k counts if >= 30% of its (state, draw) pairs are still valid
MIN_BIN = 30  # spec measure 9: pairs a level needs in an |e| bin
MEASURES = ("rho_clip", "res", "rho_chunk", "share_pred", "top1_pred", "share_noise", "top1_noise", *MEANS)
```

В конец файла:

```python
def run(
    run_path: str = "checkpoints/bc",
    level_paths: Sequence[str] = probe.LEVELS,
    phys: Sequence[float] = (0.1, 0.2, 0.3),
    world_model_dir: str = predictors.WM_DIR,
    num_envs: int = 64,
    num_states: int = 128,
    num_draws: int = 4,
    num_chunks: int = 4,
    num_flow_steps: int = 5,
    batch_size: int = 8,  # states per vmapped batch; lower it if RAM runs out
    seed: int = 3000,  # level i uses seed + i: disjoint from phase A and B1
    out_dir: str = OUT,
):
    """Spec section 3: raw arrays per level to out_dir/raw/<level>.npz (a finished level is skipped), then summarize."""
    env, env_params, levels, obs_dim, action_dim = probe.setup(level_paths)
    base = env._env  # raw Kinetix env: no auto-reset, as the B1 predictors
    phys = tuple(phys)
    names = np.array(["oracle", *(f"phys{p}" for p in phys), "learned"])
    raw_dir = pathlib.Path(out_dir) / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    @jax.jit
    def level_diag(state_dict, level, key, wm):
        policy = probe.make_policy(state_dict, obs_dim, action_dim)
        k_c, k_s, k_p = jax.random.split(key, 3)
        boundaries = probe.collect(
            env, env_params, policy, level, k_c, num_envs, 4, train_expert.ACTION_NOISE_STD, num_flow_steps
        )
        std = predictors.alive_std(boundaries)
        obs, state = probe.sample(boundaries, k_s, num_states)

        def one(x):
            return probe_state(
                policy, base, env_params, wm, phys, x[1].env_state, x[0], x[2], std,
                num_draws, num_chunks, num_flow_steps,
            )

        res = jax.lax.map(one, (obs, state, jax.random.split(k_p, num_states)), batch_size=batch_size)
        return res, boundaries[2].sum()

    for i, level_path in enumerate(level_paths):
        name = predictors.level_name(level_path)
        f = raw_dir / f"{name}.npz"
        if f.exists():
            print(f"{level_path}: {f} exists, skipped", flush=True)
            continue
        with (pathlib.Path(world_model_dir) / f"{name}.pkl").open("rb") as fh:
            wm = pickle.load(fh)
        start = time.time()
        res, n_alive = jax.device_get(level_diag(
            probe.load_state_dict(run_path, level_path), jax.tree.map(lambda x: x[i], levels),
            jax.random.key(seed + i), wm,
        ))
        assert n_alive >= num_states, "too few alive states: raise --num-envs"
        np.savez_compressed(f, predictors=names, **res)
        print(f"{level_path}: done in {time.time() - start:.0f} s", flush=True)
    summarize(out_dir)


def add_ratios(t: pd.DataFrame) -> pd.DataFrame:
    """Pooled ratios of spec section 4 from the SUMS columns (a row may pool one k or several)."""

    def r(a, b):
        return t[a] / t[b].where(t[b] > 0)

    return t.assign(
        rho_clip=1 - r("lin_clip", "pred"), res=r("lin", "pred"), rho_chunk=1 - r("chunk_lin_clip", "chunk"),
        share_pred=r("rs_pred", "d_pred"), top1_pred=r("top_pred", "d_pred"),
        share_noise=r("rs_noise", "d_noise"), top1_noise=r("top_noise", "d_noise"),
    )


def level_table(res: dict, level: str) -> pd.DataFrame:
    """One level per (predictor, k): n and frac of valid pairs, SUMS pooled over them, MEANS averaged, ratios."""
    valid = res["valid"]  # [N, M, K]
    rows = []
    for p, name in enumerate(res["predictors"]):
        for k in range(valid.shape[-1]):
            v = valid[:, :, k]
            row = {"level": level, "predictor": str(name), "k": k + 1, "n": int(v.sum()), "frac": float(v.mean())}
            row |= {x: float(res[x][:, p, :, k][v].sum()) for x in SUMS}
            row |= {x: float(res[x][:, p, :, k][v].mean()) if v.any() else float("nan") for x in MEANS}
            rows.append(row)
    return add_ratios(pd.DataFrame(rows))


def curves(t: pd.DataFrame) -> pd.DataFrame:
    """Median over levels per (predictor, k) in two variants (spec section 4): 'all' = levels kept at this k,
    'survivors' = levels kept at every k. 'levels' = how many levels a median is over."""
    K = t["k"].max()
    kept = t[t["frac"] >= MIN_FRAC]
    n_kept = kept[kept["predictor"] == "oracle"].groupby("level")["k"].nunique()  # the mask is common to predictors
    survivors = n_kept[n_kept == K].index
    out = []
    for variant, d in (("all", kept), ("survivors", kept[kept["level"].isin(survivors)])):
        g = d.groupby(["predictor", "k"])
        c = g[list(MEASURES)].median()
        c["levels"] = g["level"].nunique()
        out.append(c.reset_index().assign(variant=variant))
    return pd.concat(out, ignore_index=True)


def near(t: pd.DataFrame, k_max: int = 7) -> pd.DataFrame:
    """Q1/Q2 table (spec section 5): per predictor, median over levels of values pooled over k = 1..k_max."""
    d = t[(t["k"] <= k_max) & (t["frac"] >= MIN_FRAC)]
    by = d.groupby(["level", "predictor"])
    lv = add_ratios(by[list(SUMS)].sum().join(by[list(MEANS)].mean()).reset_index())
    return lv.groupby("predictor")[list(MEASURES)].median()


def bins(raws: dict, t: pd.DataFrame, n_bins: int = 10) -> pd.DataFrame:
    """Spec measure 9: rho_clip against |e| in deciles of |e| over the pairs of all levels, predictors and kept k."""
    kept = t[(t["predictor"] == "oracle") & (t["frac"] >= MIN_FRAC)].groupby("level")["k"].apply(list)
    pairs = []
    for level, res in raws.items():
        ks = np.asarray(kept.get(level, []), int) - 1
        v = np.broadcast_to(res["valid"][..., ks][:, None], res["e"][..., ks].shape)
        pairs.append(tuple(res[x][..., ks][v] for x in ("e", "lin_clip", "pred")))
    edges = np.quantile(np.concatenate([p[0] for p in pairs]), np.linspace(0, 1, n_bins + 1))
    rows = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        rhos = []
        for e, lin_clip, pred in pairs:
            m = (e >= lo) & ((e < hi) if b < n_bins - 1 else (e <= hi))
            if m.sum() >= MIN_BIN and pred[m].sum() > 0:
                rhos.append(1 - lin_clip[m].sum() / pred[m].sum())
        rows.append({"bin": b, "e_lo": lo, "e_hi": hi, "rho_clip": float(np.median(rhos)) if rhos else np.nan,
                     "levels": len(rhos)})
    return pd.DataFrame(rows)


def first_below(x, y, thr: float):
    """The first x where y < thr (NaN never counts), or None."""
    below = np.asarray(y, float) < thr
    return float(np.asarray(x)[below.argmax()]) if below.any() else None


def figure(cv: pd.DataFrame, bn: pd.DataFrame, path: pathlib.Path):
    fig, (a, b, c) = plt.subplots(1, 3, figsize=(17, 4.5))
    colors = dict(zip(sorted(cv["predictor"].unique()), plt.rcParams["axes.prop_cycle"].by_key()["color"]))
    for (pred, variant), d in cv.groupby(["predictor", "variant"]):
        d = d.sort_values("k")
        a.plot(d["k"], d["rho_clip"], ls="-" if variant == "all" else "--", color=colors[pred],
               label=pred if variant == "all" else None)
        if variant == "all" and pred != "oracle":
            c.plot(d["e_pred"], d["je_pred"], marker=".", color=colors[pred], label=pred)
    lv = cv[(cv["predictor"] == "oracle") & (cv["variant"] == "all")].sort_values("k")
    a2 = a.twinx()
    a2.step(lv["k"], lv["levels"], where="mid", color="gray", lw=0.8)
    a2.set_ylabel("levels at k (solid curves)")
    surv = cv.loc[cv["variant"] == "survivors", "levels"]
    for y in (0.3, 0.0):
        a.axhline(y, c="gray", lw=0.6, ls=":")
        b.axhline(y, c="gray", lw=0.6, ls=":")
    a.set_title(f"ρ_clip vs k (dashed: the {int(surv.max()) if len(surv) else 0} levels alive to k = {cv['k'].max()})")
    a.set_xlabel("k, steps after the call (chunk boundaries at 8, 16, 24)")
    a.legend(fontsize=7)
    mid = (bn["e_lo"] + bn["e_hi"]) / 2
    b.plot(mid, bn["rho_clip"], marker="o")
    for x, y, n in zip(mid, bn["rho_clip"], bn["levels"]):
        b.annotate(str(n), (x, y), fontsize=7)
    b.set_xscale("log")
    b.set_xlabel("|o − ô| (normalized), decile bins; labels = levels")
    b.set_title("ρ_clip vs deviation (all predictors and k)")
    c.set_xlabel("|o* − ô| (normalized)")
    c.set_ylabel("|J·(o* − ô)|")
    c.set_title("prediction error J sees, k = 1…31")
    c.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def summarize(out_dir: str = OUT):
    """Tables, thresholds and figure from out_dir/raw/*.npz (spec sections 4-5)."""
    out = pathlib.Path(out_dir)
    raws = {f.stem: dict(np.load(f)) for f in sorted((out / "raw").glob("*.npz"))}
    t = pd.concat([level_table(r, lv) for lv, r in raws.items()], ignore_index=True)
    cv, nr, bn = curves(t), near(t), bins(raws, t)
    o = t[(t["predictor"] == "oracle") & t["k"].between(1, 4) & (t["frac"] >= MIN_FRAC)].groupby("level")
    rho = float((1 - o["lin_clip"].sum() / o["pred"].sum()).median())
    e_pred = float(t.loc[t["predictor"] == "oracle", "e_pred"].abs().max())
    check = {"oracle_rho_clip_k1_4": rho, "e1b": 0.54, "oracle_e_pred_max": e_pred,
             "ok": abs(rho - 0.54) <= 0.1 and e_pred == 0}
    th = {"k": {}, "e": {f"<{thr}": first_below(bn["e_lo"], bn["rho_clip"], thr) for thr in (0.3, 0.0)}}
    for (variant, pred), d in cv.groupby(["variant", "predictor"]):
        d = d.sort_values("k")
        th["k"][f"{variant}/{pred}"] = {f"<{thr}": first_below(d["k"], d["rho_clip"], thr) for thr in (0.3, 0.0)}
    t.to_csv(out / "summary.csv", index=False)
    cv.to_csv(out / "curves.csv", index=False)
    nr.to_csv(out / "near.csv")
    bn.to_csv(out / "bins.csv", index=False)
    (out / "check.json").write_text(json.dumps(check, indent=2))
    (out / "thresholds.json").write_text(json.dumps(th, indent=2))
    figure(cv, bn, out / "diag.png")
    print(nr.round(3).to_string())
    print(bn.round(3).to_string(index=False))
    print(json.dumps(check))
    print(json.dumps(th))
    if not check["ok"]:
        print("!!! sanity check failed (spec section 5): look for a bug before reading anything")


if __name__ == "__main__":
    tyro.extras.subcommand_cli_from_dict({"run": run, "summarize": summarize})
```

В `.gitignore` после строки `results/**/*.pkl` добавить `results/**/*.npz`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `UV_OFFLINE=1 JAX_PLATFORMS=cpu uv run --offline pytest -q tests`
Expected: 33 passed.

- [ ] **Step 5: Commit**

```bash
git add src/diag.py tests/test_diag.py .gitignore
git commit -m "feat(diag): level loop with resumable raw arrays; pooled tables, per-k curves on all and surviving levels, |e| bins, thresholds, figure"
```

---

### Task 4: smoke (контроллер, ~5–10 мин)

- [ ] **Step 1:** запустить на двух уровнях, seed 99:

```bash
JAX_PLATFORMS=cpu uv run --offline src/diag.py run --level-paths worlds/l/grasp_easy.json worlds/l/mjc_walker.json --num-states 8 --num-draws 2 --seed 99 --out-dir results/smoke_diag
```

`results/smoke*/` уже в `.gitignore`.

- [ ] **Step 2: Проверить:**
  - прогон дошёл до конца, NaN нет;
  - `check.json`: `oracle_e_pred_max` = 0. ρ на 8 состояниях ещё ничего не значит;
  - в `curves.csv` есть оба варианта и столбец `levels`;
  - `diag.png` читается.
- [ ] **Step 3: Оценить полный прогон** по времени **второго** уровня: в первом сидит компиляция. Время растёт примерно в 16 раз, по числу состояний. Главная стоимость — `J`, а их число от числа прогонов почти не зависит, число чанков в smoke то же. Если выходит больше 4 ч, уменьшить `--num-states` до 96 и записать это в журнал спека **до** полного прогона.

### Task 5: полный прогон (Mac, офлайн, в фоне, ~2–3 ч)

- [ ] **Step 1:** контроллер запускает в фоне:

```bash
mkdir -p results/b1/diag && JAX_PLATFORMS=cpu uv run --offline src/diag.py run 2>&1 | tee results/b1/diag/log.txt
```

Если прогон упал, повторный запуск той же командой пропускает готовые уровни.

- [ ] **Step 2: Прочитать `check.json` до всего остального.** Если `ok` = false, данные не читаем: ищем ошибку, чиним, записываем в журнал спека и считаем заново.
- [ ] **Step 3: Коммит результатов** — без `raw/*.npz`, их закрывает `.gitignore`:

```bash
git add results/b1/diag
git commit -m "results: B1 diagnostic (exploratory, Mac, seeds 3000+i)"
```

### Task 6: записка (контроллер, после данных)

- [ ] **Step 1: Русская записка `docs/results/b1-diag.md`.** Ответы на Q1–Q3 по толкованию §5 спека, записанному заранее:
  - порог по `‖e‖` — главный переносимый ответ;
  - порог по k — пессимистичная граница для открытого цикла, по кривой доживших уровней;
  - кривую «все уровни» показывать с числом уровней.
- [ ] **Step 2: Абзац в `docs/results/report.md` §8 (английский).** Показать владельцу до коммита.
- [ ] **Step 3:** в спеке B2+B5, когда его начнём, записать как гипотезы то, что подтвердилось.
- [ ] **Step 4:** коммит после «ок» владельца.
