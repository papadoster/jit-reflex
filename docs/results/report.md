# JIT Reflex: linearizing a frozen action-chunking policy between its calls

*Phase A report · Kinetix (12 levels) · September 2026. Lab notes in Russian: [E1 memo](probe.md), [E2 / Gate 2 memo](closed-loop.md), and the [design spec](../superpowers/specs/2026-09-23-jit-reflex-phase-a-design.md) with its changelog.*

## TL;DR

- **Idea.** A frozen chunking policy (flow matching, chunks of H = 8 actions) is slow to call, so between calls the robot replays a stale plan. The reflex instead acts on the policy's own **local linearization** along the predicted trajectory: `a = π(ô)[0] + clip(J·(o − ô))`. The Jacobian `J` comes from reverse-mode autodiff, and no training is involved.
- **Offline (E1).** The tangent recovers about **half** of what a fresh policy call would change, on **all 12** levels (median ρ = 0.54). This holds once actions are compared the way the environment executes them. A raw-action comparison said KILL, and we traced that to errors in clipped or unused action dimensions.
- **Closed loop (E2):** 3 seeds × 256 episodes × 12 levels on an RTX 4090.
  - With rare calls the reflex **beats Real-Time Chunking (RTC)**: +9.1 pp over delays d = 2…4, and +15.6 pp at d = 4.
  - Under random velocity kicks it beats the better of naive and RTC by **+5.3 pp**.
- **But** most of that advantage comes from **re-querying the policy at the oracle-predicted state**, not from `J`. `J`'s own niche is **rare calls**: +4…+10 pp at s ≥ 5, and +6…+8 pp under kicks. This niche was pre-registered and confirmed on held-out seeds, though only narrowly.
- **The pre-registered Gate 2 is NEGATIVE:** `J` accounted for 20% of the gain, and the bar was 50%. The pre-registered **RTC + reflex** combination also missed its bar.
- **Cost.** A reflex call is 525 network evaluations, 35× RTC, and has **2.07× RTC latency**. In the only latency-fair comparison the benchmark allows (d = 1), the reflex loses.

## 1. Hypothesis

Between two calls of a frozen chunking policy, the policy can be replaced by its own linearization along the predicted trajectory:

```
a_j = π(z_j, ô_j)[0] + clip( J_j · (o_j − ô_j), ±1 ),     J_j = ∂π(z_j, o)[0] / ∂o  at  o = ô_j
```

- `ô_j` is the observation predicted for step j at call time.
- `o_j` is the fresh observation.
- `z_j = roll(z, −j)` is the call's sampling noise, shifted so that its row 0 is the row chunk step j came from (see §3).

The idea is a "reflex": it reacts at the very next control step and needs no training. The closest prior work either trains a corrector (A2C2), uses geometric gains (Pace-and-Path), or trains a local predictor (VLA-ULAP). Taking `K` from the policy's own Jacobian was, to our knowledge, untested.

## 2. Setup and method

- **Benchmark.** [real-time-chunking-kinetix](https://github.com/Physical-Intelligence/real-time-chunking-kinetix) (Physical Intelligence): 12 Kinetix 2D-physics levels, public behavior-cloned flow policies, and Gaussian action noise σ = 0.1. The loop simulates an inference delay `d` and an execute horizon `s`. Baselines are **naive** chunking and **RTC**.
- **Predictor.** A noise-free fork of the simulator, which is an **oracle**. The plan executed over the next H steps is rolled out in this fork.
- **Per call.** The reflex computes the chunk plus, for every chunk index, `ã_j` and `J_j`. Each `J_j` takes 6 VJPs, one per action dimension, versus hundreds of JVPs for the observation dimensions.
- **Variants (ablations):**

| variant | nominal action | feedback |
|---|---|---|
| `pred` | `ã_j` | none (a training-free, oracle VLASH-like re-query) |
| `reflex` | `ã_j` | `J_j` |
| `reflex_chunk` | stale chunk action | `J_j` |
| `rtc_reflex` | RTC chunk | `J_j` |

- **Guards.** A zero-noise invariant check: `reflex_chunk` reproduces `naive` when the prediction is exact. Two bitwise checks: `reflex_off` reproduces `naive`, and `rtc_reflex_off` reproduces RTC. The upstream methods are bit-identical to Physical Intelligence's code (`scripts/check_upstream_bitwise.py`).

## 3. E1: offline kill-test (Mac CPU)

For 256 sampled states per level, a chunk is rolled out twice from each state: noise-free, giving `ô`, and with action noise σ, giving `o`. Two candidates are compared against the oracle `a* = π(z_k, o)[0]`, i.e. what a fresh call would do: the re-query `ã_k` and the tangent `ã_k + J·(o − ô)`. The statistic is `ρ = 1 − e_lin / e_pred`, the share of the "fresh call" correction that `J` recovers. It is pooled over k = 1…4 at σ = 0.1.

| run | seed | action space | median ρ (R) | levels with ρ ≥ 0.3 | verdict |
|---|---|---|---|---|---|
| E1 (pre-registered) | 0 | raw | 0.02 | 5/12 | KILL |
| sensitivity check | 0 | as executed by Kinetix | 0.53 | 12/12 | — |
| **E1b** (pre-registered after E1) | 1000 | as executed, correction clipped ±1 | **0.54** | **12/12** | **GO** |

![E1b](../../results/probe_e1b/rho.png)

What we learned:

1. **Compare what is executed.** Kinetix clips motors to [−1, 1] and thrusters to [0, 1], and ignores action dimensions bound to nothing. In raw space the tangent "overshot" in exactly those directions. For example, it predicted 3.0 where the truth was 1.2, but both execute as 1. On the same states, ρ moved from −2.9…0.77 in raw space to 0.34…0.82 as executed.
2. **The tangent behaves like a tangent.** ρ falls with the deviation size (0.65 / 0.54 / 0.41 / 0.31 at σ = 0.05 / 0.1 / 0.2 / 0.4) and with the step offset k. The typical sample is almost exact (median ρ ≈ 1), and the mean is dominated by rare contact events.
3. **Re-asking the policy is not the same as continuing its plan.** Even with correctly shifted noise and no disturbance, `‖A[k] − ã_k‖² ≈ 0.19`, about 5× the effect of the deviation itself. This was an early hint that the re-query alone would carry much of the benefit.

## 4. E2: closed loop (RTX 4090)

Solve rates in %, averaged over 12 levels and 3 seeds (256 episodes each). Calling the policy every step with no delay reaches 91.6%.

| d, s | naive | pred | RTC | **reflex** | `J` (reflex − pred) | reflex − RTC |
|---|---|---|---|---|---|---|
| 1, 1 | 76.2 | 90.6 | 89.1 | 90.9 | +0.4 | +1.9 |
| 1, 7 | 71.6 | 69.8 | 79.4 | 79.4 | **+9.5** | 0.0 |
| 2, 2 | 70.5 | 87.2 | 84.1 | 87.5 | +0.3 | +3.4 |
| 2, 6 | 64.4 | 68.8 | 74.1 | 77.3 | **+8.5** | +3.2 |
| 3, 3 | 58.4 | 82.4 | 74.8 | 82.6 | +0.2 | +7.8 |
| 3, 5 | 56.0 | 71.7 | 67.4 | 75.8 | **+4.1** | +8.4 |
| 4, 4 | 48.8 | 75.3 | 61.6 | 77.2 | +1.9 | **+15.6** |

![E2](../../results/eval/main/success.png)

**Random kicks** (d = 2; every step with p = 0.02, add N(0, (c·v_med)²) velocity to all dynamic bodies):

| c | s | naive | pred | RTC | **reflex** | `J` | reflex − max(naive, RTC) |
|---|---|---|---|---|---|---|---|
| 0.5 | 2 / 6 | 64.4 / 55.5 | 79.0 / 61.4 | 75.2 / 63.7 | 79.8 / 69.8 | +0.8 / +8.4 | +4.6 / +6.2 |
| 1 | 2 / 6 | 58.9 / 51.3 | 72.6 / 57.0 | 67.4 / 57.6 | 73.1 / 64.0 | +0.4 / +7.0 | +5.6 / +6.3 |
| 2 | 2 / 6 | 52.3 / 47.4 | 62.0 / 50.8 | 59.0 / 51.9 | 62.7 / 57.0 | +0.7 / +6.2 | +3.8 / +5.0 |

**Pre-registered decisions:**

| check | result | |
|---|---|---|
| (a) reflex beats RTC at s = 8−d, d = 2…4, by ≥ 5 pp and in every seed | +9.1 pp, 3/3 seeds | ✅ |
| (b) reflex at s = 8−d within 2 pp of RTC at s = d, i.e. ~4× fewer calls | −5.2 pp | ❌ |
| (c) under kicks, reflex beats max(naive, RTC) by ≥ 5 pp | +5.3 pp | ✅ |
| mandatory: `J` gives ≥ 50% of reflex − naive over the grid | 20% | ❌ |
| **Gate 2** | | **NEGATIVE** |
| E2c (registered on seed 0, tested on seeds 1–2): `J` gives ≥ 50% at s ≥ 5 | 0.53 and 0.57 | ✅ (narrow) |
| E2b: RTC + reflex beats RTC by ≥ 3 pp, in every seed | +2.6 pp | ❌ |

**Where it helps and where it hurts.** The reflex gains most on smooth locomotion and control levels (swimmer and walker, +21…+40 pp over RTC; cartpole; lunar lander). It loses on impact-driven and unstable levels (catapult −11…−13 pp; unicycle −18 pp at d = 2), which are the same levels with the lowest ρ in E1. `J` on top of a stale chunk (`reflex_chunk`) barely helps (+1…+4 pp over naive): the re-query and the feedback work as a pair. The same explains why RTC + reflex falls behind plain reflex.

**Cost.** Per call, the reflex costs 525 network evaluations against 15 for RTC, and 12.7× the work per episode at the (b) comparison. On an RTX 4090 at batch 1 it takes 3.42 ms against 1.65 ms for RTC, so r = 2.07. With latency counted fairly, the reflex's delay becomes `⌈2.07·d⌉`. That is feasible within H = 8 only for d = 1, and there the reflex loses: 75.8 vs 79.4 at s = 8−d, and 82.6 vs 89.1 at s = d. The re-query `pred` has r = 1.07, so its advantage at equal d is latency-fair, but it rests on the oracle predictor.

## 5. How the study was run

Every decision rule was written into the spec before its data existed, and commit timestamps show the order. When a measurement bug or a better metric appeared, the old verdict was kept and the new test was registered *before* running it, on a fresh seed:

- E1 → E1b;
- E2b before any `rtc_reflex` run;
- E2c before seeds 1–2.

Exploratory runs on the Mac (a mini-E2 and an overnight run with kicks and large delay) are labelled as such. They were used only to decide what to spend GPU money on (≈ $10).

## 6. Limitations

- **The predictor is an oracle** (a noise-free simulator fork). At large delays, most of the reflex's edge over RTC comes from knowing the future state. A real robot needs a learned predictor, and that is the main open risk.
- **Kinetix is not a VLA.** Observations are symbolic, not images. There is one checkpoint set and one action-noise level.
- **E1 used one seed per run**, with 256 states × 4 draws per level.
- **The implementation over-computes.** The reflex package computes Jacobians at all 8 chunk positions, but only `s` of them are used, so the cost and latency numbers are upper bounds.

## 7. What next

Phase B is the natural next step, before any hardware (phase C). It has three parts:

1. Replace the oracle with a **learned predictor**, and check whether the re-query and `J` survive prediction error.
2. Move to a real VLA action head (π0-style action expert), where the observation Jacobian passes only through the small action expert.
3. Make the package cheap: only the `s` used positions, and a trust-region gate that falls back to re-planning near contacts, where the tangent breaks down.

The measured niche (rare calls, disturbances, smooth dynamics) says where such a reflex could pay off. Two examples: a cloud-hosted VLA called a few times per second, or a slow model on a fast, smooth manipulator.
