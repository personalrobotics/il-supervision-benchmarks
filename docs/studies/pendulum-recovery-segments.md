# Pendulum targeted recovery segments at capture failures: predeclaration, version 1 (#21)

**Status: version 1.** This document is committed and pushed before any code for this study exists, any transition model is fitted, any policy is trained or any rollout is run. It fixes the gates, seed sets, start sets, stop rules and the pilot rule. Any later change is a numbered amendment (see "Amendments"), with its reason, made before the outcome it could affect is read.

This is a small **exploratory mechanism study**. It is an extension beyond the paper's one-step (M = 1) CCIL and is labeled as such everywhere it is reported. A favorable CCIL result is not required; the intended outcome is a clear answer about where the method succeeds or stops. A refusal or null outcome is not evidence that CCIL generally cannot help.

## Question

Can recovery segments generated at the states where a provisional BC policy fails to capture supply correct supervision there, and does that supervision improve a learner beyond receiving the same added training exposure?

- **Exact-dynamics segments** are the mechanism ceiling and audit. They are not available to ordinary users.
- **Learned-dynamics segments** are the deployable path: a transition model fitted to the demonstrations, with an independently initialized check model as a trust gate.

## What is already known (not rerun)

- #18 / PR #20: a CCIL − BC recovery signal (+0.420) that is not attributable to corrective information. Exposure-matched duplication explained more than half of the point estimate, and the legacy epoch-based recipe gave the CCIL and duplication arms 21.1× BC's optimizer updates.
- Capture control (`pendulum-capture-control.md`): labels at BC-visited states in the demonstrator's LQR-branch capture region improve final hold beyond duplication (relabel 1.00, BC 0.40, duplication 0.40).
- CCIL Gate 3A: the one-step generator cannot leave demonstration support at the 20 ms interval.
- CCIL #171: persistent-bias segments started from demonstration windows, M ∈ {1, …, 32}, found no M that is both novel and trusted by two independently fitted models. No learner was trained.

This study tests a different proposal family: segments that **start at observed BC capture failures** and run forward to the demonstration continuation.

## Two questions, kept apart (decided 2026-11-02)

1. **Does targeted corrective supervision work?** The primary mechanism study. Failure states come from rollouts of the provisional BC policy **in the real environment**, and both generators share the same failure states and targets, so learned − exact isolates dynamics-model error. **This study therefore uses unlabeled environment interaction and is not demonstrations-only.** Every report of it says so.
2. **Can the failures be discovered from demonstrations alone?** A Stage A diagnostic only: the provisional BC is rolled out inside the learned model, and the states it fails at are compared with the real failures and replayed exactly. It adds no training arm and has no stop rule. If it cannot reproduce meaningful failure states, that is reported as a separate discovery bottleneck, not as label error.

## Fixed setting

- **Task:** the continuous corrected Pendulum (`ilsb.tasks.pendulum.Task`, `variant="continuous"`), 20 ms control interval, action bound ±3, horizon 500, scorer unchanged. Upright means cos(θ − π) ≥ cos 8°; final hold means upright over each of the last 100 steps of an episode.
- **Data:** the canonical continuous dataset (sha256 `59c332bc…cf43c`) at r = 0.1, with the subset and the 8/2 split rule of the positive controls (`SeedSequence([s, 9])` permutation, `make_split(…, 0.2, s)`). Demonstrations, interval, torque, reward and task difficulty are not changed.
- **Metric:** d(x, y) = √(wrap(e_x − e_y)² + ((ω_x − ω_y)/√g)²), e = wrap(θ − π), periodic in e, as in CCIL Gate 3A and #171.
- **Support scale:** τ_demo(s), the 95th percentile, over the seed's validation-episode states, of d to the nearest training-episode state.
- **Regions:** capture region |e| < 0.1 (the demonstrator LQR branch); approach region |e| < 0.5.
- **Demonstrator** π_D: `ilsb.tasks.pendulum.expert`, used only by the audit gate G7, the `recovers` audit and figures. Failure selection and generation never call it.

### Seeds and start sets (new; frozen)

| Use | Seeds | `eval_seed` (100 starts each) |
|---|---|---|
| Stage A design: failure collection | 3321–3325 | 2401 |
| Stage A design: held-out failures (coverage, discovery) | 3321–3325 | 2402 |
| Stage B pilot: failure collection | 3326–3330 | 2403 |
| Stage B pilot: final hold | 3326–3330 | 2404 |
| Stage B pilot: fresh failures (recovery endpoint) | 3326–3330 | 2405 |

The tool refuses unless the five start sets are pairwise disjoint and share no start with `eval_seed` 2201–2220 or 2301. Design and pilot seeds are disjoint. No seed or start set is added after any outcome is read.

## Provisional BC (per seed)

The study learner (`studies/pendulum-j-r-k/study.yaml`, `learner`, with the synthetic keys removed, as the positive controls' BC), trained on the seed's 8 training episodes with the validation episodes for checkpoint selection and patience stopping. A fit stopped by the step ceiling is a failure.

## Capture failures (real environment)

From the seed's provisional BC, rolled out for 500 steps from a start set:
- A **failure trajectory** enters the approach region (|e| < 0.5 at some step) and does not achieve final hold.
- Its **failure states** are its states in the approach region, thinned to at most 40 per trajectory by `round(linspace)` over the eligible indices. Selection uses no demonstrator or oracle.
- The fraction of failure states from which π_D recovers (`recovers`: |e| ≤ 0.1 and |ω| ≤ 0.5 over the last 100 of 500 π_D steps) is reported as an audit, not used.

## Recovery segments

For each failure state x_f, seed s and horizon M:
- **Horizon ladder:** M ∈ {1, 2, 4, 8, 16}, counted in dataset control transitions (20 ms each). It is never extended.
- **Target (data-derived):** t* is the training-episode index whose state is nearest x_f under d, among windows with t* + M ≤ 499 (ties: lowest episode order, then lowest t). The target is z_tgt = x_{t*+M} and the reference actions are u_ref = (u_{t*}, …, u_{t*+M−1}).
- **Solve:** U ∈ [−3, 3]^M minimizing ‖r(U)‖² + λ‖U − u_ref‖², where r(U) = residual(F_M(x_f, U), z_tgt) is the 2-vector whose norm is d, F_M composes the generator's one-step map M times with re-encoding to (sin θ, cos θ, ω) after every step, and λ = 1e-4.
  - Method: batched projected Levenberg–Marquardt from U = u_ref, central-difference Jacobians (h = 1e-3) through F_M, at most 50 iterations, damping μ from 1e-3 (×1/3 on a decrease, ×3 otherwise), actions clipped to ±3 after every step.
  - **Accepted** when finite and the final ‖r‖ ≤ 0.25·τ_demo(s).
- **Generators:**
  - **exact:** the true map (`ExactModel`), the ceiling;
  - **learned:** the primary transition model;
  - **check:** an independently initialized model, used only by G5.
- **Transition models:** the #171 `dynamics` block unchanged (MLP 512-512, ReLU, soft-sampling Lipschitz 2.0 in model-normalized coordinates, 100 epochs, batch 512, learning rate 1e-3, weight decay 1e-5), fitted on the 8 training episodes; primary with fit seed s, check with s + 100000; one deterministic batched fit group. The held-out M-step error ε_M on the validation episodes is reported, not gated.
- **Rows:** (x_i, u_i), i = 0 … M − 1, where x_0 = x_f and x_i are the generator's own states. Every row reconciles to exactly one segment.

## Stage A: geometry and trust (no policy comparison)

### Gates (frozen; per design seed and M)

Unless stated, a gate is computed on the learned arm.

| # | Gate | Criterion |
|---|---|---|
| G1 | Exact feasibility and bounds | Every accepted segment's actions within ±3. Replayed through the exact map from x_f: ≥ 95% of accepted segments have every one-step error d(F(x_i, u_i), x_{i+1}) ≤ 0.25·τ_demo and arrival error d(F_M(x_f, U), z_tgt) ≤ 0.25·τ_demo. |
| G2 | Arrival | Accepted segments ≥ 0.5 of the failure states. |
| G3 | Coverage of held-out failures | ≥ 0.5 of the held-out design failures (start set 2402, same provisional BC) are within ρ_cov = 0.5·τ_demo of an accepted row state. |
| G4 | Novelty | Median d_demo (to the nearest training-episode state) over accepted row states ≥ τ_demo. |
| G5 | Model agreement | κ_M ≤ 0.25·τ_demo, where κ_M is the 95th percentile over accepted segments of the larger of the check model's worst one-step disagreement max_i d(F̂_check(x_i, u_i), x_{i+1}) and its composed disagreement d(F̂_check,M(x_f, U), z_tgt). |
| G6 | Exact audit of trusted segments | Among accepted segments whose own check disagreement is ≤ 0.25·τ_demo (at least one required), ≥ 95% pass G1's exact replay. |
| G7 | Labels in the capture region | ≥ 100 accepted rows with \|e\| < 0.1; ≥ 90% of them have \|u_i − π_D(x_i)\| ≤ 0.3; and their median \|u_i − π_D(x_i)\| ≤ 0.5 × the median \|π_BC(x_i) − π_D(x_i)\| of the provisional BC at the same states. |

The exact arm is evaluated with G2, G3, G4 and G7 (G1, G5 and G6 hold or are undefined by construction; its replay residual is recorded as a solver check).

### Selection and stop rules (fixed)

- An M **passes** for an arm when every one of that arm's gates holds on ≥ 4 of the 5 design seeds.
- **Selected M:** the smallest M that passes for the learned arm. It is never chosen from learner performance. Stage B's exact arm uses the same M; whether the exact arm passes at that M is recorded.
- **Stop 1:** no M passes for the exact arm → stop Pendulum here; the targeted geometry is inadequate.
- **Stop 2:** some M passes for the exact arm but none for the learned arm → stop; dynamics learning is the bottleneck.
- On either stop, the output is an empty synthetic view with the per-M, per-gate diagnosis. The ladder is not extended and no threshold is relaxed.

### Data-only discovery diagnostic (reported; no stop rule)

- The provisional BC is rolled out for 500 steps from the 2401 starts inside the primary model (re-encoded every step). Model failures use the same failure definition, evaluated on the model trajectory.
- Reported per seed: the number of model-failure trajectories and states; the fraction of the real held-out failures (2402) within ρ_cov of a model-failure state; the exact-replay divergence time, the first step at which the model rollout and the real rollout from the same start differ by d > 0.25·τ_demo.
- **Verdict:** "discovery reproduces the failures" when that fraction is ≥ 0.5 on ≥ 4 of 5 seeds; otherwise "discovery is a separate bottleneck". It does not affect selection or Stage B.

### Stage B step budget (fixed in Stage A, before any Stage B outcome)

S = 2 × the median over the design seeds of the provisional BC's best-validation step, rounded up to a multiple of 1000. Validation runs every S/100 steps. S is written to `stage-a.json`, which is committed before Stage B trains anything.

### Required Stage A artifacts

- A row-level quality file for every proposed label (both arms, every M): seed, M, segment identity (seed, failure index, M, arm), offset i, state, action, solver residual, exact replay one-step and arrival errors, check disagreement, d_demo, region, |u − π_D|, |π_BC − π_D|, acceptance and rejection reason (non-finite, residual, or not reached).
- The accepted rows of the selected M, per seed and arm, as ordinary (state, action) supervision rows, schema-checked: finite, within bounds, on the circle, and reconciled to one segment.
- One phase portrait (e ∈ [−π, π) periodic, ω ∈ [−8, 8], plus a near-upright inset): demonstrations, BC failure trajectories, segment starts, accepted segments, the capture region, exact-versus-learned disagreement, and the model-discovered failures beside the real ones.
- `stage-a.json`: every gate per seed, arm and M, the selection, the stop outcome, the discovery verdict and S, bound by sha256 to every input.

## Stage B: five-seed causal pilot (only if Stage A selects an M)

Per pilot seed 3326–3330: a provisional BC (as above), failures from start set 2403, and exact and learned segments at the selected M.

### Arms (exactly three)

1. **exact:** the exact arm's accepted rows;
2. **learned:** the learned arm's accepted rows;
3. **duplication:** demonstration training rows resampled with replacement.

Matching:
- **Added rows:** N_add(s) = the smaller of the exact and learned arms' accepted-row counts. Each arm takes N_add rows: the two generated arms by a deterministic subsample (`SeedSequence([s, 21])`, without replacement), duplication by resampling. If N_add < 100 on any pilot seed, Stage B stops with "insufficient labels" before training.
- **Training:** identical for the three arms: the study learner's network and optimizer, `synthetic_fraction` 0.5, a fixed `step_budget` S, `checkpoint_selection: validation` on the seed's validation episodes every S/100 steps, no patience, constant learning rate. The tool refuses to evaluate unless the three arms' recorded optimizer steps are equal on every seed. Equal epochs over differently sized datasets are not used.

### Endpoints (primary)

1. **Final hold:** the fraction of the 100 episodes from start set 2404 that achieve final hold.
2. **Recovery success:** the pilot seed's provisional BC is rolled out from start set 2405, and its failure states are taken by the definition above. Each arm is rolled out for 300 steps from each failure state; success is upright over each of the last 100 steps. The per-seed value is the mean over that seed's failure states.

### Contrasts

Paired over the five seeds, mean with a descriptive interval (`union_interval`, 95%):
- learned − duplication: the value of deployable corrective information;
- exact − duplication: the oracle ceiling for corrective information;
- learned − exact: the loss due to dynamics-model error.

### Pilot rule (fixed)

This is a directional pilot, not a confirmatory study. **Advance** to a larger, separately predeclared study only if exact > duplication and learned > duplication on both primary mean endpoints, with the paired difference positive on ≥ 4 of 5 seeds in each of those four comparisons. Otherwise **stop** and report the failure mode. Seeds are not added because a result is unfavorable.

## Reporting

The report keeps apart: geometry (G2–G4), label correctness (G7), model trust (G5–G6), optimization exposure (matched and recorded), and closed-loop behavior (Stage B). It states that the mechanism study uses unlabeled environment interaction, that exact-dynamics performance is not available to ordinary users, and that M > 1 is an extension beyond the paper. No public M > 1 CCIL feature follows unless the pilot advances.

## Running and artifacts

- Stages are atomic and resumable; each record is bound to the sha256 of its inputs and of the tool, and resume refuses a record with another identity. The tool refuses to run from a dirty tree.
- Every long stage logs progress at least every five minutes; the run log starts with the command, commit, dirty flag, host load and package versions, and ends with an explicit exit status.
- Evidence goes in `docs/studies/evidence/pendulum-recovery-segments/`, with reproduction instructions and a clone-only checker that recomputes the gates and decisions from the committed records.

## Amendments

None.

## Not done

Reopening #18 or PR #20; reproducing the paper; tuning the task until CCIL wins; sweeps over M beyond the ladder, σ, data fractions, rewards, torque limits or control rates; claiming exact-dynamics performance is available to users; a public M > 1 CCIL feature.
