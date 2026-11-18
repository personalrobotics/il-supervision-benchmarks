# Learned-dynamics recovery on Pendulum (protocol)

This is the protocol of [#73](https://github.com/personalrobotics/il-supervision-benchmarks/issues/73), Stage 3 of
[CCIL #171](https://github.com/personalrobotics/CCIL/issues/171). It is frozen before any start below is solved.

**Question.** Stage 2 ([#70](recovery-feasibility.md#result)) found that, under exact dynamics, recovery segments of
M > 1 intervals exist on Pendulum from starts where no one-step segment recovers. Can learned dynamics identify a
nonempty, mostly correct subset of the recoveries that improve a fixed learner's continuation?
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
- **Ensemble, with fixed roles.** Five members, each with its own initialization seed `SeedSequence([171, 3, k])`,
  k = 0 … 4.
  - **Model 0 builds the segments.** It is trained on the 8 training episodes.
  - **Models 1–4 are check models.** Each is trained on an episode-level bootstrap of the 8 training episodes (8 drawn
    with replacement by `SeedSequence([171, 3, k, 1])`) with its own initialization.
  - Each member's weights are hashed; the run consumes the hashes.

## Construction and validity

- **Construction.** LM shooting with Stage 2's settings (8 restarts: zero, demonstrated, 6 seeded random; 50
  iterations; damping 1e-3, ×10/÷10; forward-difference Jacobian, step 1e-6), through member 0. A segment is
  **model-valid** if its member-0 endpoint is within ε, its actions are in bounds and its member-0 states stay in the
  envelope. Every restart is kept; the first model-valid restart is the constructed segment for (start, M).
- **Validity.** The constructed actions are replayed under the exact evaluator map and under refined integration
  (Stage 2's RK45 at rtol 1e-10, atol 1e-12). A segment is **valid** if both endpoints are within ε, its actions are
  in bounds and its exact states stay in the envelope.

## The base policy

The advantage is measured for the policy whose failures recovery labels would correct, not for the demonstrator.
π is a **frozen learned policy**: #21 Stage A's provisional BC for design seed 3321
(`policies/bc/s3321/policy.npz`, sha256 `1457a98af553071348dbb1844ceeaeb18d259ba2350636ae3d5e8ef33277e34f`, in the
immutable release `evidence-pendulum-recovery-segments-stage-a-v1`). It was trained on exactly Stage 2's
demonstrations: the r = 0.1 subset and the 8/2 split of design seed 3321. On the task's 100 evaluation starts it
scores upright occupancy 0.326 and final hold 0.0, against the demonstrator's 0.613 and 1.0 (measured before any
Stage 3 start is solved). It is competent at swinging up and fails at capture, so it has failures to correct. It is
not retrained.

## Relevance

**Task outcome.** The cost is the primary Pendulum outcome, upright occupancy:
c(x_k, u_k) = 1 − 1{|wrap(e_k)| ≤ 8°}, with γ = 1, truncated at the episode's original 500-step horizon. No effort
term is added: earlier Pendulum studies neither optimized nor evaluated effort.

**Advantage.** From a start x_0 at anchor time t, with T = 500 and n = T − t remaining steps:
- **continuation:** π from x_0 for n steps;
- **recovery:** the segment's M actions, then π for n − M steps;
- Δ_M = Σ_(k=0)^(n−1) [1_upright(recovery)_k − 1_upright(continuation)_k], counted on the state before each action
  (the evaluator's `score` convention). It is an integer number of upright steps; Δ_M / n, in occupancy units in
  [−1, 1], is reported beside it. This is Δ = V^π(h) − [C_M(h, ρ) + V^π(h_M)] in costs.

Both branches use the same dynamics and the same frozen π. The **exact advantage** Δ_M runs both under the exact map;
it is the audit. The **estimated advantage** Δ̂_M^(k) runs both through check model k. The **ensemble minimum**
Δ̂_M = min over k = 1 … 4 of Δ̂_M^(k) excludes model 0, which built the segment. It is not a certified lower bound;
this stage measures whether it behaves conservatively.

**Two independent horizon selections**, fixed here:
- **Oracle ceiling.** Per start, the shortest M at which a Stage 2 exact-dynamics segment (first valid restart, in
  restart order) is valid with exact Δ_M ≥ 1.
- **Learned selection (deployable).** Per start, the shortest M at which a constructed restart, taken in restart
  order, is model-valid with Δ̂_M ≥ 1. It uses only model 0, the check models and π. The exact outcome never
  influences it and is opened only in the audit.

A selected segment is **correct** if the audit finds it valid with exact Δ_M ≥ 1.

**Secondary outcome.** Final two-second hold: whether the last 100 steps of the episode are upright, for each branch.
It is reported and never combined into Δ.

**Sensitivity, descriptive only.** The Gymnasium Pendulum return of each branch, and the demonstrator-relative
advantage (both branches continued by the demonstrator instead of π). Neither selects anything.

## Reference behaviors (checked before any outcome is read)

- **π as its own recovery.** Executing π for the M recovery steps must give Δ_M = 0 and Δ̂_M^(k) = 0 for every
  member, exactly.
- **Exact-dynamics ceiling.** Stage 2's exact-dynamics segments get their exact Δ_M under π, and the oracle ceiling
  is selected from them. This is how often recovery helps π at all.
- **Zero action.** The zero-action rollout from x_0 as a reference recovery of each M, with its exact Δ_M.

## Measures (all reported, by radius and by M)

- Construction yield: model-valid (start, M) pairs.
- Validity: valid among model-valid; exact and refined endpoint misses; envelope and bounds violations.
- The oracle ceiling's and the learned selection's M per start, and their agreement.
- Exact Δ_M and Δ̂_M (and per check model), their sign agreement, and the exact Δ_M of every selected segment.
- Recall of the learned selection against the oracle ceiling (descriptive).
- Final two-second hold, both branches; the Gymnasium return, both branches.
- Model error: one-step error of every member on the validation split, and the M-step endpoint error
  d(f̂^M(x_0, U), F^M(x_0, U)) of every constructed segment, per member.
- Agreement with Stage 2: constructed versus exact-dynamics shortest valid M, per start.

## Decision

Let C be the oracle ceiling's starts and S the learned selection's starts. **Stage 3 passes** only if all three hold:
- **the ceiling is not empty:** |C| ≥ 20, i.e. at least 20 starts have a valid exact-dynamics segment with exact
  Δ_M ≥ 1;
- **the selection is not empty:** |S| ≥ 20;
- **the selection is mostly correct:** at least 80% of S's selected segments are correct.

Precision is the safety question; the two yield conditions keep the filter from passing by returning almost nothing.
Recall against C is reported, not gated. If Stage 3 fails, the refusal is the result, and #171 stops before any
learner is trained. No ε, ladder, model setting, solver setting, policy, cost or threshold is changed after any
study start is solved.

## Smoke, then the run

The smoke runs the full pipeline on Stage 2's key-99 smoke starts: a byte-identical rerun, a fresh-process audit,
injected faults the audit must refuse for the right reason (an action out of bounds, a forged exact state, a forged
Δ_M, a forged Δ̂_M, a learned selection that differs from its recomputation), the reference behaviors above, and a
cost projection of the complete path against a 3 h wall ceiling on siddh. The run waits for the smoke and its own
go.

## Code and smoke

The code is in [`tools/recovery_learned/`](../../tools/recovery_learned/), with the machine-readable protocol in
`stage-3.json`:
- `run.py` fits the five models and solves and evaluates every start;
- `audit.py` is the fresh-process audit;
- `smoke.py` runs the smoke.

It reuses Stage 2's Pendulum system, LM solver and scheduler. It runs in an environment with the ilsb environment's
pinned packages and CCIL at `12b2fbf` (`prlccil`); the study set refuses another CCIL commit.

Implementation choices the protocol leaves open:
- **Starts.** Each x_0 is read from its Stage 2 start file. Provenance: the file's sha256 is the one Stage 2's record
  lists, and the record's sha256 is the one `stage-3.json` names. Content: the start recomputed from Stage 2's seeds
  agrees within 1e-12 (the last bits differ across hosts).
- **Precision.** The models are CCIL's float32 networks, evaluated in float64 (the float32 weights are exact in
  float64). Stage 2's 1e-6 forward-difference step is below float32 resolution, so a float32 evaluation gives LM a
  noise Jacobian. Every model rollout uses the same float64 evaluation.
- **π as its own recovery.** The reference prefix is π's own actions along its continuation in the same dynamics,
  executed open loop for M steps.
- **Actions** of every branch are clipped to ±3, as the task applies them.
- **The oracle ceiling** reads each M's first valid Stage 2 restart; the ceiling is the shortest M at which that
  segment has exact Δ_M ≥ 1.

**Smoke result**, from `efae412` on siddh with CCIL `12b2fbf`, Stage 2's 40 key-99 Pendulum starts
([`evidence/recovery-learned/smoke.json`](evidence/recovery-learned/smoke.json)):
- **Reproduction.** The second run, with its own model fits, was byte-identical on all 46 files (the five models,
  every start file and the record). The fresh-process audit found nothing.
- **Faults.** Each injected fault was refused with its own reason first: action bounds, a forged exact state, a forged
  Δ_M, a forged Δ̂_M, and a forged selection.
- **References.** π as its own recovery gave Δ_M = 0 under the exact map and every check model, at every M.
- **Projection.** The fit takes 72 s and the complete path 11.9 s per start: 0.06 h of wall time for the study on 32
  workers, within the 3 h ceiling.

These smoke figures are not study outcomes. On the smoke starts, 13 have an oracle-ceiling segment, the learned
selection picked 11, and 1 of those 11 was correct; 430 of 516 model-valid segments missed ε under exact replay.
