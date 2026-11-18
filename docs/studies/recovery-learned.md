# Learned-dynamics recovery on Pendulum (protocol)

This is the protocol of [#73](https://github.com/personalrobotics/il-supervision-benchmarks/issues/73), Stage 3 of
[CCIL #171](https://github.com/personalrobotics/CCIL/issues/171). It is frozen before any start below is solved.

**Question.** Stage 2 ([#70](recovery-feasibility.md#result)) found that, under exact dynamics, recovery segments of
M > 1 intervals exist on Pendulum from starts where no one-step segment recovers. Does a learned dynamics model
identify valid members of that set, and are they worth taking?
1. **Construction.** Does the learned model find recovery segments?
2. **Validity.** Do those segments replay successfully under exact dynamics?
3. **Relevance.** Is recovery preferable to continuation?

**Scope.** Pendulum only: it passed Stage 2, and it gives the task outcome and exact dynamics needed to audit all
three questions. PushT's Stage 2 result was solver-inconclusive, and it is not repeated here. No learner is trained,
and no label enters any policy.

The segment, its residuals and its views are those of CCIL
[`docs/recovery-segments.md`](https://github.com/personalrobotics/CCIL/blob/12b2fbf/docs/recovery-segments.md).
Pendulum is Markov, so W = 1 and L = 0: there is no handoff (Stage 2 amendment 3).

## Starts and problem

- **Starts.** The 400 Stage 2 study starts, unchanged: the same 50 anchors, radii r ∈ {0.5, 1, 2, 4}, two directions
  each, keys `[171, 2, 1, …]`. Each start is paired with its Stage 2 record (`evidence-recovery-feasibility-stage-2-v1`).
  The smoke uses Stage 2's key-99 smoke starts.
- **Problem.** Unchanged from Stage 2: target y_M = x_(t+M), ε = 0.1 τ_demo, the evaluator map, |u| ≤ 3, the
  |ω| envelope, the ladder M ∈ {1, 2, 4, 8, 16, 32, 64}. M = 1 is ruled out on all 400 starts by Stage 2's
  Lipschitz-certified sweep.

## The learned model

- **Model.** CCIL's `DynamicsNet` with the `pendulum_cont` recipe's transition-model settings (soft-sampling
  Lipschitz constraint 2.0, two hidden layers of 512, 100 epochs, batch 512, learning rate 1e-3), at CCIL `12b2fbf`.
- **Data.** Stage 2's demonstrations: the r = 0.1 subset of design seed 3321, training split only (8 episodes). The
  validation split (2 episodes) is used for model error, never for training.
- **Ensemble.** K = 5 members, trained identically with seeds `SeedSequence([171, 3, k])`, k = 0 … 4. **Member 0
  constructs**; all five estimate the advantage. Each member's weights are hashed; the run consumes the hashes.

## Construction and validity

- **Construction.** LM shooting with Stage 2's settings (8 restarts: zero, demonstrated, 6 seeded random; 50
  iterations; damping 1e-3, ×10/÷10; forward-difference Jacobian, step 1e-6), through member 0. A segment is
  **model-valid** if its member-0 endpoint is within ε, its actions are in bounds and its member-0 states stay in the
  envelope. Every restart is kept; the first model-valid restart is the constructed segment for (start, M).
- **Validity.** The constructed actions are replayed under the exact evaluator map and under refined integration
  (Stage 2's RK45 at rtol 1e-10, atol 1e-12). A segment is **valid** if both endpoints are within ε, its actions are
  in bounds and its exact states stay in the envelope.

## Relevance

**Task outcome.** The cost is the primary Pendulum outcome, upright occupancy:
c(x_k, u_k) = 1 − 1{|wrap(e_k)| ≤ 8°}, with γ = 1, truncated at the episode's original 500-step horizon. No effort
term is added: earlier Pendulum studies neither optimized nor evaluated effort.

**Advantage.** π is the corrected demonstrator, the policy whose rollouts are the demonstrations. From a start x_0
at anchor time t, with T = 500 and n = T − t remaining steps:
- **continuation:** π from x_0 for n steps;
- **recovery:** the segment's M actions, then π for n − M steps;
- Δ_M = Σ_(k=0)^(n−1) [1_upright(recovery)_k − 1_upright(continuation)_k], counted on the state before each action
  (the evaluator's `score` convention). It is an integer number of upright steps. Δ_M / n, in occupancy units in
  [−1, 1], is reported beside it.

**Exact advantage** Δ_M is computed under the exact map; it is the audit. **Estimated advantage** Δ̂_M^(k) is the
same quantity with both branches rolled through ensemble member k. It uses no exact dynamics. The **conservative
estimate** is Δ̂_M = min_k Δ̂_M^(k).

**Useful label.** A constructed segment is a useful label if it is valid and Δ̂_M ≥ 1 (positive in whole upright
steps). Per start, the label is the useful segment of the **shortest** M; that rule is fixed here, so no horizon is
chosen after outcomes.

**Secondary outcome.** Final two-second hold: whether the last 100 steps of the episode are upright, for each branch.
It is reported and never combined into Δ.

**Sensitivity, descriptive only.** The Gymnasium Pendulum return of each branch. It selects nothing.

## Reference behaviors (checked before any outcome is read)

- **π as its own recovery.** Taking ρ = π for M steps must give Δ_M = 0 and Δ̂_M^(k) = 0 for every member, exactly.
- **Exact-dynamics ceiling.** Stage 2's first valid exact-dynamics segment per (start, M) gets its exact Δ_M. This is
  how often recovery is advantageous at all, and bounds what the learned model can find.
- **Zero action.** The zero-action rollout from x_0 as a reference recovery of each M, with its exact Δ_M.

## Measures (all reported, by radius and by M)

- Construction yield: model-valid (start, M) pairs.
- Validity: valid among model-valid; exact and refined endpoint misses; envelope and bounds violations.
- The shortest valid M per start, and the shortest M with a useful label.
- Exact Δ_M and Δ̂_M (and per member), their sign agreement, and Δ_M of every useful label.
- Final two-second hold, both branches; the Gymnasium return, both branches.
- Model error: one-step error of every member on the validation split, and the M-step endpoint error
  d(f̂^M(x_0, U), F^M(x_0, U)) of every constructed segment, per member.
- Agreement with Stage 2: constructed versus exact-dynamics shortest valid M, per start.

## Decision

Let **U** be the starts with a useful label whose exact Δ_M ≥ 1, and let **P** be the fraction of
(start, M) pairs with a model-valid segment and Δ̂_M ≥ 1 that are also valid with exact Δ_M ≥ 1: the precision of the
filter a user without exact dynamics would apply.

**Stage 3 passes** if both hold:
- |U| ≥ 20;
- P ≥ 0.8.

Otherwise the refusal is the result, and #171 stops before any learner is trained. No ε, ladder, model setting,
solver setting, cost or threshold is changed after any study start is solved.

## Smoke, then the run

The smoke runs the full pipeline on Stage 2's key-99 smoke starts: a byte-identical rerun, a fresh-process audit,
injected faults the audit must refuse for the right reason (an action out of bounds, a forged exact state, a forged
Δ_M, a forged Δ̂_M), the reference behaviors above, and a cost projection of the complete path against a 3 h wall
ceiling on siddh. The run waits for the smoke and its own go.
