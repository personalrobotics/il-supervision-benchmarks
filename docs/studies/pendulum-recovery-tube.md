# Pendulum recovery-tube expansion from existing checkpoints: predeclaration, version 1 (#18)

**Status: version 1 draft. It is frozen by its commit before any evaluator code exists.** The audit, evaluator, fixtures and duplication support are added to the same pull request as later commits. That pull request is merged before any real checkpoint is evaluated, including the capture control. Results go in one later pull request.

This study is **exploratory**. It does not reopen the completed causal ladder (#12), rerun its treatments or make a confirmatory claim. A favorable outcome is not required. An ambiguous outcome is reported as "not established". Seeds are not added, and neither the disturbance grid nor the tube changes after any recovery outcome is read.

## Question

Under bounded physical torque disturbances of held-out corrected-demonstrator trajectories, do existing CCIL policies return to the reference trajectory more often than BC? If they do, is the gain due to the corrective information in the synthetic rows, or to the extra training exposure?

## Outcomes already known (not re-estimated here)

- **Corrected-demonstrator study, continuous variant** (CCIL `docs/experiments/pendulum-m3-closeout.md`):
  - CCIL − BC upright occupancy is +0.208 [0.138, 0.277] at 98.75%, over training seeds 1201–1220.
  - Final hold is 0.702 for CCIL and 0.032 for BC.
  - The discontinuous variant shows no detectable difference: +0.012 [−0.080, 0.104].
- **That study's CCIL labels** use the paper settings:
  - action-noise std 1e-5;
  - 20 labels per transition;
  - a learned transition model.
  - On the exact map, labels at σ ≤ 1e-4 are all returned at initialization (CCIL `docs/experiments/pendulum-correction.md`). The geometry of the learned-model labels has not been measured.
- **One-step label bound** (CCIL `docs/experiments/pendulum-correction.md`):
  - every one-step preimage lies within d_source ≤ τ★ = 0.038357 of its source, for the continuous dynamics at dt = 0.02 and |u| ≤ 3;
  - the observed maximum over the exact-map scan is 0.0383.
- **Capture positive control** (`docs/studies/pendulum-capture-control.md`, seeds 3311–3320): passes.
  - Relabel − Duplication held-out capture action error: −0.612 [−1.050, −0.174].
  - Final hold: +0.602 [+0.234, +0.970].
  - Final hold is 1.00 for relabel, 0.40 for BC and 0.40 for duplication.
- **Pumping-region positive control** (`docs/studies/pendulum-positive-control.md`): not established.
- **Recovery-segment diagnostic** (CCIL `docs/experiments/pendulum-recovery-diagnostic.md`): no admissible M > 1 on any seed. This study keeps M = 1.
- **Response surface** (closed PR #16): its σ was selected by validation loss, and it measured near-identity augmentation. No root result was committed.

## Fixed setting

- **Task:** the continuous corrected Pendulum, θ̈ = −9.81 sin θ + u with |u| ≤ 3 and dt = 0.02. The map is `ilsb.tasks.pendulum_map`, the verified port of CCIL's exact map. The discontinuous variant is out of scope.
- **Demonstrator π_D:** `ilsb.tasks.pendulum.expert`, the corrected primary variant.
- **Coordinates:** e = wrap(θ − π). Distance is d(x, y) = √(wrap(θ_x − θ_y)² + ((ω_x − ω_y)/√g)²), the metric of the earlier Pendulum evidence.
- **Unchanged:** M = 1, the 20 ms action interval, the torque bound, the generators, σ, the architectures and the training recipes.

## Cohorts: eligibility and exclusion

Policies are compared only within a cohort. Eligibility is decided by the audit before any disturbance rollout. A cohort that fails is excluded and reported with the failing identities. It is never retrained.

1. **Capture control** (fitness cohort): BC, duplication and relabel for seeds 3311–3320, from run `20261102T003101Z-capture-control`.
   - **Eligible** only if every file listed in the committed `docs/studies/evidence/pendulum-capture-control/artifact-manifest.json` exists and matches its sha256, and every policy hash matches `capture-control.json`.
   - **If excluded,** the study stops, because the evaluator cannot be validated.
2. **Corrected-demonstrator study** (primary cohort): BC and CCIL, continuous variant, training seeds 1201–1220, from CCIL run `20261029T191145Z-corrected-study`.
   - **Eligible** only if the hash chain reconciles for all 40 units:
     1. the checkpoint's `policy.npz` sha256 equals the `files` entry of its `policy-train` stage manifest;
     2. that stage is one of the units recorded as `ok` in the committed CCIL `study-audit.json`;
     3. its `policy-evaluate` stage consumed that checkpoint hash at noise level 0.0;
     4. that stage produced a rollout whose sha256 equals the entry `continuous/{arm}/s{seed}/0.0` in the committed CCIL `study-evaluate.json`.
   - The training stage's candidate and dataset input hashes are also recorded. The dataset must be `59c332bc…cf43c`.
   - **Exclusion is all-or-nothing.** If any unit fails, the cohort is excluded. Seeds are not subset.
3. **Response surface** (optional and descriptive only): run `20261031T141026Z-j-r-k`.
   - **Evaluated** only if its content-addressed store is internally complete:
     - every stored object's sha256 equals its address;
     - every reference from `root.json`, `manifest.json` and `stages/*.json` resolves to a present object with that hash.
   - **Scope:** only the report's selected σ cell per arm, at the largest r.
   - **Never anchored externally.** No root result was committed before 2026-11-02. A manifest created by this study is therefore reported as internal consistency only, never as equivalent to prior external anchoring.
   - **Use:** never in a decision rule.

## Reference trajectories and anchors (outcome-blind)

- **References:** π_D rolled out for 1,000 steps from the 20 starts of `Task(eval_seed=2301, eval_episodes=20).starts()`.
  - These starts are fresh and held out from every training set. The audit refuses to proceed if any of them equals a start of eval seeds 2201–2220.
  - Every frame's continuation under π_D is the **reference continuation** r.
- **Anchors:** frames t ∈ {0, 50, 100, …, 450} of each reference, giving 200 anchors. There is no random draw.
- **Stratum** of an anchor, from its reference state alone:
  - **upright:** |e| < 0.1, π_D's LQR branch;
  - **swing:** otherwise.

## Disturbances (derived from label geometry; fixed)

- **Nominal displacement grid:** D_j = τ★ · 2^j for j ∈ {−3, …, 4}, that is 0.0048 to 0.614 in d.
  - τ★ is the one-step label bound, so the grid spans displacements inside, at and beyond the largest tube any one-step CCIL label can occupy.
  - Each cohort's measured tube quantiles (see "Geometry audit") are marked on its curves. They classify each grid point as inside (D < q95), edge (q95 ≤ D < 2·q95) or beyond. They never move the grid.
- **Impulse:** for duration k steps, k ∈ {1, 5}, and sign σ_b ∈ {+1, −1}:
  - apply the acceleration clip(π_D(x)) + σ_b·b through the exact map for k steps from the anchor;
  - the disturbance b is not clipped to the torque bound;
  - b = D / √((k·dt)⁴/4 + (k·dt)²/g), the free-flight displacement of a constant acceleration over k·dt in the metric d.
  - At k = 1 and D = τ★, b = 6.0, which is 2·|u|max, the action gap that defines τ★.
- **Achieved displacement:** d(x′, r_{t+k}) is recorded per state. Curves use the nominal D.
- **The undisturbed case** D = 0 starts the policy at the anchor itself. It measures tracking and is reported separately from the curve.
- **Identical inputs:** every arm in every cohort gets the same anchors, impulses and disturbed states, 6,600 initial states per policy. These states are for evaluation only. They select no setting, checkpoint or training data.

## Recovery (fixed)

After the impulse, the evaluated policy controls for H = 300 steps. Its commanded action is clipped to ±3 by `Task.step`.

- **Phase-aligned distance:** δ_j = min over |s| ≤ W of d(x_j, r_{j+s}), with W = 25 steps (0.5 s). This tolerates a phase lag of up to 0.5 s against the reference continuation.
- **Recovered:** some j ≤ 250 has δ ≤ ρ = 0.1 at every step from j to j + 49 (a dwell of 1 s).
  - ρ is the demonstrator's LQR-branch scale and is 2.6·τ★. A recovered rollout has re-entered a neighborhood wider than any one-step label tube.
- **Also recorded per rollout:**
  - time to recovery, j·dt;
  - integrated distance Σ δ_j·dt;
  - final hold for upright anchors (within 8° of upright at each of the last 100 steps);
  - upright occupancy;
  - non-finite actions, saturated fraction and failures.
- **Reference behaviors,** reported on the same inputs (descriptive):
  - π_D itself;
  - the zero action.
  - Aggregate return is never a decision metric.

## Primary outcome

- **Recovery curve:** for each (cohort, arm, seed), the recovery fraction at each nominal D, pooled over anchors, k and sign.
- **AUC:** the mean of the 8 grid-point fractions, equally weighted on the log₂ grid.
- **Stratum curves:** curves and AUC are also reported per stratum.
- **Paired decision statistic:** per-seed AUC differences, with a 95% interval that is the union of BCa and t (`ilsb.analysis.union_interval`). "Above 0" means the interval's lower end is > 0.

## Geometry audit (corrected cohort; descriptive)

This runs on each seed's CCIL `candidates.npz`, accepted rows only, before any disturbance rollout. It reports:

- **Displacement:** d_source quantiles (q50, q95, maximum). Each displacement is split into tangent and transverse components, using the unit tangent of the source's episode at the source frame (central difference in metric coordinates).
- **Distance to the reference episode and to the whole demonstration set,** kept separate. The global nearest-demonstration distance is descriptive and is not a gate.
- **Exact transition residual** d(F(z̃, ã), z_target), and |ã| ≤ 3.
- **Learned-model and certificate fields** that the candidate artifact already carries.
- **Overlap:** for each D, the fraction of disturbed states within q95 of some synthetic state.

## Decision rules (sequential; each gates the next)

1. **Evaluator fitness (capture cohort, upright stratum).**
   - The evaluator is **fit** only if Relabel − Duplication AUC is above 0.
   - Otherwise the study **stops**. It reports that the evaluator does not detect the established capture effect. The corrected cohort is not evaluated.
2. **Recovery signal (corrected cohort, all anchors).**
   - CCIL − BC AUC above 0 means **signal**: go to step 3.
   - Otherwise the result is **"no recovery-tube enlargement detected"**. The study stops and publishes it. No duplication arm is trained.
3. **Attribution (only after a signal).**
   - Train the exposure-matched duplication arm (next section) and evaluate it on the same inputs.
   - CCIL − Duplication AUC above 0 means **supervision-driven**. Otherwise the effect is **not attributable to corrective information**.
   - Duplication − BC (exposure) and CCIL − BC (total recipe) are reported with the same intervals and do not decide.

The tests are sequential and each gates the next, so no multiplicity adjustment is applied. The stratum results, the k and sign breakdowns and the response-surface cohort are descriptive.

## Exposure-matched duplication (step 3 only)

- **Labels:** for each training seed, start from the CCIL arm's `candidates.npz`. Replace every accepted synthetic row's state and action with its source row (x_i*, u_i*).
- **Kept unchanged:**
  - row order and count;
  - acceptance;
  - weights;
  - per-source multiplicity;
  - every other field.
- **Training:** the CCIL policy trainer at the corrected study's frozen CCIL commit `6fbcce5`, with the CCIL arm's stage configuration unchanged, including:
  - run.seed;
  - epochs, batch size and learning-rate schedule;
  - the noise model and noise-mask probability;
  - the demonstration weight;
  - checkpoint selection "last";
  - the device.
- **Determinism:** the original training was not bitwise deterministic (`compute.deterministic: false`), and this arm inherits that, unchanged.
- **Identity:** the replaced candidate file's sha256 and the new stage manifest are recorded.
- **Content:** the arm contains no new state or action information.

## Running and artifacts

- **Tool:** one command performs the read-only audit, the geometry audit and the evaluation, without writing to any input run. A separate subcommand trains the duplication arm, and only when step 2 passes.
- **Execution:** the tool runs on siddh in tmux.
  - The log is in `runs/`, headed by the command, commit, dirty state, host, load and versions.
  - Progress is logged at least every 5 minutes.
  - Per-policy results are written atomically, with `--resume`.
  - The stall threshold is 30 minutes, and a watcher checks for completion, failure and stalls.
- **Tests before merge** use only synthetic or off-protocol fixtures. Examples:
  - π_D as the policy recovers at D = 0 with δ ≡ 0;
  - the zero action fails from swing anchors;
  - a tampered hash excludes its cohort;
  - the duplication rewrite preserves every non-state, non-action field.
- **Committed evidence** goes in `docs/studies/evidence/pendulum-recovery-tube/`: the audit, geometry, per-policy summaries, intervals, decisions, run logs and the phase portraits. Every plotted cell and trajectory links to a machine-readable record.
- **Phase portraits,** on common axes e ∈ [−π, π) and ω ∈ [−8, 8]:
  1. references, anchors and disturbance arrows;
  2. synthetic states and corrective-action arrows, with the tangent and transverse displacement;
  3. recovery and failure maps per arm;
  4. the CCIL − control recovery difference;
  5. recovery versus D, with q50, q95 and τ★ marked.

## Not done

- No new generator, oracle, horizon, σ, seed or data-fraction sweep.
- No M > 1, and no change to any recipe.
- No rescue of the published Pendulum result.
- No duplication as a user-facing CCIL mode.
- No claim of generality beyond Pendulum.
