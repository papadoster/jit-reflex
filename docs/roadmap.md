# Research agenda after phase A

*Written 2026-09-25, after phase A closed ([report](results/report.md)) and before any experiment below was run. Each experiment will get its own pre-registration (decision rule, seeds, kill criterion) in `docs/superpowers/specs/` before its data exists, as in phase A.*

Phase A in one line: the reflex pays off when the policy is called rarely (s ≥ 5) or the system is kicked. But most of its closed-loop edge came from re-querying the policy at an **oracle**-predicted state, and a reflex call costs 35× RTC. The agenda removes these crutches one at a time, starting with the one most likely to kill the effect. The claim being built is: *a large policy can think rarely, because between its calls a local feedback controller can be taken from the policy itself, without training.*

## B1. Staleness curve under an imperfect predictor (first)

**Question.** Does `J`'s contribution grow as the plan gets staler, and does it survive prediction error?

**Pre-registered spec:** [B1 design and decision rules](superpowers/specs/2026-09-25-b1-staleness-predictors-design.md) (Russian). It moves the learned model into B1 and makes the absolute `J` contribution the main metric.

**Status: done 2026-09-26.** The verdict is SURVIVES, narrowly, with R4 = ENOUGH. Predictions 1 and 2 held, and `J`'s absolute effect grew with the predictor error (R3). Results: [report §8](results/report.md#8-phase-b1-stale-plans-and-imperfect-predictors) and the [Russian memo](results/b1.md). By §5, the next step is B2 with the learned model. An offline diagnostic followed ([report §8.8](results/report.md#88-offline-diagnostic-exploratory), [Russian note](results/b1-diag.md)): `J` sees less of the learned model's error than of wrong physics, and the tangent's quality depends on the deviation size alone (ρ_clip < 0.3 beyond ‖e‖ ≈ 2.3), which sets B3's thresholds.

**Design.** Fixed delay d = 1, execute horizon s = 1…7. Predictors:
- the oracle;
- the same simulator with physical parameters (masses, friction, motor and thruster strength) off by 10 / 20 / 30%;
- later, a learned one-step dynamics model.

Methods: naive, RTC, pred, reflex. Fresh seeds.

Also plot `J`'s gain against the measured ‖o − ô‖. Staleness matters only through the deviation it produces, so this curve also bounds the trust region of B3.

**Predictions:**
1. With the oracle, reflex − pred grows with s. Phase A's draft of this curve is confounded with d, because it comes from s = d and s = 8 − d: +0.4, +0.3, +0.2, +1.9, +4.1, +8.5, +9.5 pp for s = 1…7.
2. pred degrades as the predictor error grows.
3. **`J`'s share of (reflex − naive) grows with the predictor error.** The correction acts on `o − ô` whatever its cause, so it should repair prediction error as well as disturbances.

**Kill.** At 20% parameter error, reflex ≤ max(naive, RTC) at every s. That would mean the effect lives on the oracle.

## B2. A cheaper reflex package

Phase A computes `J` at all 8 chunk positions, through all 5 flow steps. That is 525 network evaluations and 2.07× RTC latency, which made the latency-fair comparison infeasible beyond d = 1.

**Options:**
- compute only the s positions that are executed;
- take gradients through the last k flow steps only;
- share one `J` across neighbouring positions;
- use a low-rank `J`.

**Not an option:** computing `J·(o − ô)` with a JVP at every control step. A JVP through the flow costs more than a fresh policy call. The per-step work must therefore stay a precomputed matrix–vector product, and all savings have to come from the package.

**Target:** latency ratio r ≤ 1.2 with ≤ 1 pp loss in solve rate. That would make the latency-fair comparison possible at d = 2…4.

**Status: offline selection done 2026-09-26** ([Russian memo](results/b2-offline.md), [spec](superpowers/specs/2026-09-26-b2-offline-design.md)). The pre-registered rule sends **T3** and **M3** to the B2+B5 GPU run:
- **T3:** `J` through the last 3 of 5 flow steps, exact nominal, depth 8 against 10. It keeps 94% of the exact `J`'s offline gain over pred on 12 of 12 levels, and still holds without the top-5% states and with the oracle.
- **M3:** a 3-step flow, depth 6. It passes narrowly and fails both of those checks.

One `J` shared across positions hurts. Interpolating B1's timings gives r ≈ 1.5–1.6 for T3, above the 1.2 target. That rules out a latency-fair grid at d = 3, which needs r ≤ 4/3. M3, at about 1.3–1.4, sits at that edge. Taking `J` off the critical path is the first question for the B2+B5 spec.

**Status: GPU run done 2026-09-27** ([report §10](results/report.md#10-phase-b2b5-the-cheaper-package-in-closed-loop-and-a-price-comparison), [Russian memo](results/b2b5.md), [spec](superpowers/specs/2026-09-26-b2b5-gpu-design.md)). In closed loop T3 and M3 keep the exact reflex's gain; T3 even beats it (+1.8 pp at d = 3). The latency target is missed: measured r is 1.54 for T3 and 1.38 for M3.

## B3. Trust region: when to use the reflex

E1 shows ρ falling as the deviation grows, with the mean dominated by rare contact events. In closed loop the reflex loses on catapult and unicycle.

**Gate:**
- small ‖o − ô‖: nominal action only;
- medium: add the `J` correction;
- large, or a predicted contact change: re-plan early.

**Prediction.** The gated reflex does at least as well as the reflex on every level, and removes the catapult and unicycle losses.

**Data so far (B2+B5, 2026-09-27):** in closed loop 19–44% of executed steps have ‖e‖ > 2.3. There the tangent recovers a small but positive share of the needed correction, and `J` still adds +7.5 pp over pred at d = 3. The gate itself has not been tested.

## B4. Strong baselines and literature review

- **A2C2**, a trained residual corrector, reports +23 pp over RTC on Kinetix. A reimplementation exists in bt-kinetix.
- **VLASH-style** predict-then-query.
- **Expected:** A2C2 wins on solve rate. The honest position then becomes training-free versus trained.
- A systematic literature review, before phase C.

**Status: A2C2 run in B2+B5, 2026-09-27** (report-only baseline, [report §10](results/report.md#10-phase-b2b5-the-cheaper-package-in-closed-loop-and-a-price-comparison)). The bt-kinetix variant is flat at about 76% at every delay and does not beat T3. Its mean hides two collapsed levels; on the other 10 it is +28.5 pp over RTC at d = 4. The expectation above held only in part. A distilled variant, the same head trained on fresh calls of the policy itself, beat every `J` variant in 8 of 9 cells, unevenly across levels.

**Status: A2C2 reproduced, 2026-09-28** (report-only rerun, [report §11](results/report.md#11-a2c2-reproduced-a-report-only-rerun), [Russian memo](results/a2c2-fix.md), [spec](superpowers/specs/2026-09-28-a2c2-fix-design.md)). Trained on the BC policy's own states, labelled by one PPO expert, with the paper's wider network, A2C2 solves about 95% in every cell. It beats T3 by 14–18 pp on every level and seed, and the distilled head too (unequal comparison). **The expectation above now holds:** the honest position is training-free versus trained. `J` needs no data, training or expert; A2C2 needs an expert that solves each level. The literature review is done: [related work](results/related-work.md).

## B5. Compute-matched comparison (only after B2)

Does "call a big policy rarely, plus a cheap reflex" beat "call it often" at equal compute? At phase A cost it does not. At d = 2:
- reflex with s = 6 costs 525 / 6 ≈ 88 network evaluations per step and solves 77.3%;
- RTC with s = 2 costs 7.5 per step and solves 84.1%.

**Status: done 2026-09-27** (in the same GPU run as B2). At honest latency on one GPU no reflex variant fits: computing `J` alongside the next call slows it 2.15×, and M3 misses by 3.8% (NOT FEASIBLE). When one delay is imposed on everyone, T3 beats naive, RTC, 10-step RTC and pred by +5.6…+8.7 pp (PASS). On the latency-fair frontier the distilled head dominates at base delays 2–4; the fixed A2C2 of B4 (2026-09-28) dominates it in turn at every base delay.

## C1. A real VLA action head

In a π0-style model an action expert is conditioned on a VLM. The Jacobian with respect to state-like inputs can pass only through the small action expert. Inputs in order: proprioception, then object pose or state estimate, then a visual latent.

## C2. Hardware

SO-101 arm with a VLA. Objects are moved during execution and latency is added artificially. Compare VLA, RTC and reflex.

## Later: combinations

- RTC + reflex, if they turn out to fix different errors. E2b missed its bar in phase A.
- A2C2 + reflex.
- A small learned local policy + reflex.
