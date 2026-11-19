# Value-free receding-horizon recovery on Pendulum (protocol)

This is the protocol of [#76](https://github.com/personalrobotics/il-supervision-benchmarks/issues/76). It is
frozen before any start is solved.

**Question.** Does receding-horizon feedback make finite-horizon recovery usable when a learned dynamics model is
locally accurate but accumulates too much open-loop error? Stage 3 ([#73](recovery-learned.md#result)) found
one-step error under 0.1 ε but M-step error of 4–33 ε, and 7 of 106 selected open-loop segments correct.

**Hypothesis.** Replanning from the observed state after every action resets accumulated model error, so validity
should depend on short-horizon model accuracy while multi-step reach is kept.

**Value-free.** The controller uses no reward, cost-to-go, expert query or value function. The demonstrations define
the continuation to recover to. Recovery advantage for the frozen BC is computed afterwards as a diagnostic only.

## The 2 × 2

| Dynamics used for planning | Open loop | Receding horizon |
|---|---|---|
| Exact | Stage 2 (#70) and Stage 3's oracle ceiling (#73), reused | **reference arm** (new) |
| Learned | Stage 3 (#73), reused | **primary arm** (new) |

## Problem (unchanged from #70 and #73)

- **Starts.** Stage 2's 400 Pendulum study starts, read from the release `evidence-recovery-feasibility-stage-2-v1`
  by provenance and content, as in #73. M = 1 is certified infeasible on all 400.
- **Target, tolerances and bounds.** Target y_M = x_(t+M), ε = 0.1 τ_demo, |u| ≤ 3, the |ω| envelope, the ladder
  M ∈ {1, 2, 4, 8, 16, 32, 64}. Pendulum is Markov (W = 1, L = 0): there is no handoff, and no continuation audit
  applies.
- **The learned model** is #73's model 0, loaded from `evidence-recovery-learned-stage-3-v1` and checked against its
  weights hash (`ffdfef73…`), evaluated in float64 as in #73. It is not refitted or updated.

## Receding-horizon execution

For each start x_0 and each M in the ladder, one independent execution from x_0, with M exact environment steps:
- **At step j = 0 … M − 1,** observe the current state x_j from the exact environment and solve the remaining
  problem: M − j actions from x_j that end at y_M, under the arm's planning dynamics. Execute only the first action
  in the exact environment and replan.
- **Shrinking horizon.** The target stays y_M and the horizon shrinks by one each step, so each execution is directly
  comparable with the open-loop segment of the same M.
- **Solver.** Stage 2's LM shooting (u = ū tanh v, 50 iterations, damping 1e-3 ×10/÷10, forward-difference step 1e-6).
  - **j = 0:** Stage 2's 8 restarts (zero, demonstrated, 6 seeded random), keyed `[171, 4, arm, start, M, k]`.
  - **j ≥ 1 (warm start):** 2 restarts: the previous plan with its executed first action dropped, and the remaining
    demonstrated actions a_(t+j) … a_(t+M−1).
  - **The executed action** is the first action of the restart with the smallest predicted terminal residual; ties go
    to the earlier restart.
- **Failure handling.** If the plan's first action is non-finite, the execution stops with failure `solver`. If an
  executed state leaves the envelope, it stops with failure `envelope`. Otherwise it runs all M steps.
- **Interaction budget.** Each execution uses exactly its M exact steps, at most Σ ladder = 127 per start; there are no
  other environment calls. The learned arm never calls the exact map while planning; it only observes the states the
  environment returns.

**Declared success.** An execution is a declared success if its observed final state is within ε of y_M and it did
not stop early. Per start, the arm's recovery is its **shortest** M with a declared success.

**Audit (fresh process, exact dynamics).** Each declared recovery must pass: actions within bounds; every executed
state in the envelope; exact replay of the executed actions reproduces the recorded states bitwise; the replayed
endpoint is within ε; and refined integration (#70's RK45, rtol 1e-10, atol 1e-12) of the executed actions ends
within ε. Handoff and continuation audits do not apply to Pendulum (W = 1).

## Gates

1. **Oracle prerequisite (exact arm).** #70's treatment gate on declared recoveries: at least 20 starts, and at
   least 10% of the starts at one radius. If it fails, stop: the target, solver or replanning formulation is
   inadequate, and the learned arm is not interpreted.
2. **Learned result (learned arm).** Both:
   - #70's treatment gate on its declared recoveries; and
   - at least 80% of its declared recoveries pass the audit.

A negative result ends #76. It does not lead to online model updates, learner training, a reward or a new target.

## Measures (all reported, by radius and by M)

- Declared successes and audited successes per arm, per M and per radius; shortest M per start.
- Failure modes: endpoint miss, `envelope`, `solver`; for declared successes, the first audit failure (bounds,
  envelope, replay, endpoint, refinement).
- Final endpoint miss in units of ε; per-step planning residual; the learned model's one-step prediction error on
  the executed transitions.
- The 2 × 2: each cell's yield by radius, with the open-loop cells read from the Stage 2 and Stage 3 records.
- **Diagnostic only:** the exact advantage Δ_M of each declared recovery for the frozen BC (#73's definition), and how
  many of #73's 190 oracle-ceiling starts each arm recovers.

## Smoke, then the run

The smoke runs both arms on Stage 2's key-99 smoke starts: a byte-identical rerun, a fresh-process audit, injected
faults the audit must refuse for the right reason (an action out of bounds, a forged executed state, a forged
refined endpoint, a forged declared success), and a cost projection of both arms against a 3 h wall ceiling on siddh
with 32 workers. The run waits for the smoke and its own go.
