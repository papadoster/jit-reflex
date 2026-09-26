# Research agenda after phase A

*Written 2026-09-25, after phase A closed ([report](results/report.md)) and before any experiment below was run. Each experiment will get its own pre-registration (decision rule, seeds, kill criterion) in `docs/superpowers/specs/` before its data exists, as in phase A.*

Phase A in one line: the reflex pays off when the policy is called rarely (s ≥ 5) or the system is kicked. But most of its closed-loop edge came from re-querying the policy at an **oracle**-predicted state, and a reflex call costs 35× RTC. The agenda removes these crutches one at a time, starting with the one most likely to kill the effect. The claim being built is: *a large policy can think rarely, because between its calls a local feedback controller can be taken from the policy itself, without training.*

## B1. Staleness curve under an imperfect predictor (first)

**Question.** Does `J`'s contribution grow as the plan gets staler, and does it survive prediction error?

**Pre-registered spec:** [B1 design and decision rules](superpowers/specs/2026-09-25-b1-staleness-predictors-design.md) (Russian). It moves the learned model into B1 and makes the absolute `J` contribution the main metric.

**Status: done 2026-09-26.** The verdict is SURVIVES, narrowly, with R4 = ENOUGH. Predictions 1 and 2 held, and `J`'s absolute effect grew with the predictor error (R3). Results: [report §8](results/report.md#8-phase-b1-stale-plans-and-imperfect-predictors) and the [Russian memo](results/b1.md). By §5, the next step is B2 with the learned model.

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

## B3. Trust region: when to use the reflex

E1 shows ρ falling as the deviation grows, with the mean dominated by rare contact events. In closed loop the reflex loses on catapult and unicycle.

**Gate:**
- small ‖o − ô‖: nominal action only;
- medium: add the `J` correction;
- large, or a predicted contact change: re-plan early.

**Prediction.** The gated reflex does at least as well as the reflex on every level, and removes the catapult and unicycle losses.

## B4. Strong baselines and literature review

- **A2C2**, a trained residual corrector, reports +23 pp over RTC on Kinetix. A reimplementation exists in bt-kinetix.
- **VLASH-style** predict-then-query.
- **Expected:** A2C2 wins on solve rate. The honest position then becomes training-free versus trained.
- A systematic literature review, before phase C.

## B5. Compute-matched comparison (only after B2)

Does "call a big policy rarely, plus a cheap reflex" beat "call it often" at equal compute? At phase A cost it does not. At d = 2:
- reflex with s = 6 costs 525 / 6 ≈ 88 network evaluations per step and solves 77.3%;
- RTC with s = 2 costs 7.5 per step and solves 84.1%.

## C1. A real VLA action head

In a π0-style model an action expert is conditioned on a VLM. The Jacobian with respect to state-like inputs can pass only through the small action expert. Inputs in order: proprioception, then object pose or state estimate, then a visual latent.

## C2. Hardware

SO-101 arm with a VLA. Objects are moved during execution and latency is added artificially. Compare VLA, RTC and reflex.

## Later: combinations

- RTC + reflex, if they turn out to fix different errors. E2b missed its bar in phase A.
- A2C2 + reflex.
- A small learned local policy + reflex.
