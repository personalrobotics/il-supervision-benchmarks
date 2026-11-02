# Pendulum recovery-tube expansion from existing checkpoints: predeclaration, version 1 (#18)

**Status: version 1, frozen.** The protocol was approved after review on 2026-11-02 and frozen by the commit that adds this status line. No evaluator code existed at that commit. The audit, evaluator, fixtures and duplication support follow as later commits in the same pull request. That pull request is merged before any real checkpoint is evaluated, including the capture control. Results go in one later pull request. Any later change to this protocol is a numbered amendment with its reason, made before the outcome it could affect is read.

This study is **exploratory**. It does not reopen the completed causal ladder (#12), rerun its treatments or make a confirmatory claim. A favorable outcome is not required. An ambiguous outcome is reported as "not established". After any recovery outcome is read:
- seeds are not added;
- the disturbance set, the dose bins and the tube do not change.

## Question

The study separates three links:
1. **Local corrective support:** where do the existing CCIL labels actually sit around the demonstrations?
2. **Recovery within that support:** do CCIL policies return to the reference trajectory more often than BC from feasible disturbances of that size?
3. **Advantage beyond exposure:** is any gain due to the corrective information in the synthetic rows, or to the extra training exposure?

## Outcomes already known (not re-estimated here)

- **Corrected-demonstrator study, continuous variant** (CCIL `docs/experiments/pendulum-m3-closeout.md`):
  - CCIL − BC upright occupancy is +0.208 [0.138, 0.277] at 98.75%, over training seeds 1201–1220.
  - Final hold is 0.702 for CCIL and 0.032 for BC.
  - The discontinuous variant shows no detectable difference: +0.012 [−0.080, 0.104].
- **That study's CCIL labels** use the paper settings:
  - action-noise std 1e-5;
  - 20 labels per transition;
  - a learned transition model.
  - Their geometry has not been measured.
  - On the exact map, labels at σ ≤ 1e-4 are all returned at initialization (CCIL `docs/experiments/pendulum-correction.md`). That result does not transfer to learned-model labels, whose displacement may come from model error.
- **Theoretical reference τ★ = 0.038357** (CCIL `docs/experiments/pendulum-correction.md`):
  - it bounds d_source for exact one-step preimages of the continuous dynamics at dt = 0.02 with |u| ≤ 3;
  - it does not bound learned-model labels;
  - it is drawn as a reference line only.
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

Policies are compared only within a cohort. Eligibility is decided by the audit before any policy is loaded. A cohort that fails is excluded and reported with the failing identities. It is never retrained.

1. **Capture control** (fitness cohort): BC, duplication and relabel for seeds 3311–3320, from run `20261102T003101Z-capture-control`.
   - **Eligible** only if every file listed in the committed `docs/studies/evidence/pendulum-capture-control/artifact-manifest.json` exists and matches its sha256. That includes the selected-state files holding the held-out capture states. Every policy hash must also match `capture-control.json`.
   - **If excluded,** the study stops, because the evaluator cannot be validated.
2. **Corrected-demonstrator study** (primary cohort): BC and CCIL, continuous variant, training seeds 1201–1220, from CCIL run `20261029T191145Z-corrected-study`.
   - **Eligible** only if the hash chain reconciles for all 40 units:
     1. the checkpoint's `policy.npz` sha256 equals the `files` entry of its `policy-train` stage manifest;
     2. that stage is one of the units recorded as `ok` in the committed CCIL `study-audit.json`;
     3. its `policy-evaluate` stage consumed that checkpoint hash at noise level 0.0;
     4. that stage produced a rollout whose sha256 equals the entry `continuous/{arm}/s{seed}/0.0` in the committed CCIL `study-evaluate.json`.
   - **CCIL inputs:** for each CCIL unit, the training stage's `candidates.npz` must match the hash its manifest records. The dataset must be `59c332bc…cf43c`.
   - **Exclusion is all-or-nothing.** If any unit fails, the cohort is excluded. Seeds are not subset.
3. **Response surface** (optional and descriptive only): run `20261031T141026Z-j-r-k`.
   - **Evaluated** only if its content-addressed store is internally complete:
     - every stored object's sha256 equals its address;
     - every reference from `root.json`, `manifest.json` and `stages/*.json` resolves to a present object with that hash.
   - **Scope:** only the report's selected σ cell per arm, at the largest r, with its own geometry phase.
   - **Never anchored externally.** No root result was committed before 2026-11-02. A manifest created by this study is therefore reported as internal consistency only, never as equivalent to prior external anchoring.
   - **Use:** never in a decision rule.

## Phase 1: geometry and disturbances, frozen before any policy is loaded

Everything in this phase depends only on the reconciled candidates, the dataset, π_D and the exact map. The tool writes `geometry.json`, which holds the measurements, the disturbed states, their achieved displacements, the dose bins and the testability verdict. It is committed and pushed to the results branch before phase 2 starts. Phase 2 refuses to run unless the sha256 of `geometry.json` is passed to it and matches.

### Label geometry (corrected cohort; accepted training-episode rows of each CCIL unit's `candidates.npz`)

- **The tube scale:** d_source = d(z̃, z_source), with q50, q95 and maximum per seed and pooled over the 20 seeds. The pooled values define **q50, q95 and the maximum** used below.
- **Displacement split** into tangent and transverse components, using the unit tangent of the source's episode at the source frame (central difference in metric coordinates).
- **Distance** to the reference episode and to the whole demonstration set, kept separate. The global nearest-demonstration distance is descriptive and is not a gate.
- **Exact transition residual** d(F(z̃, ã), z_target), and |ã| ≤ 3.
- **Learned-model and certificate fields** that the candidate artifact already carries.

### Reference trajectories and anchors (outcome-blind)

- **References:** π_D rolled out for 1,000 steps from the 20 starts of `Task(eval_seed=2301, eval_episodes=20).starts()`.
  - These starts are fresh and held out from every training set. The tool refuses to proceed if any of them equals a start of eval seeds 2201–2220.
  - Every frame's continuation under π_D is the **reference continuation** r.
- **Anchors:** frames t ∈ {0, 50, 100, …, 450} of each reference, giving 200 anchors. There is no random draw.
- **Stratum** of an anchor, from its reference state alone:
  - **demonstrator LQR-branch:** |e| < 0.1. This constrains angle only, not angular velocity;
  - **swing:** otherwise.

### Feasible disturbances (fixed set; bounded actuation)

- **The disturbance:** for k steps from the anchor x_t, the applied action is u = clip(π_D(x) + σ_b·b, −3, 3), stepped through the exact map. The applied action stays in [−3, 3] at every step.
- **The fixed set,** applied identically at every anchor (36 disturbances per anchor):
  - offsets b ∈ {0.1875, 0.375, 0.75, 1.5, 3, 6};
  - durations k ∈ {1, 4, 16} steps;
  - both signs σ_b = ±1.
  - Their free-flight displacements span about 0.0012 to 0.69 in d. These are design values only.
- **Achieved displacement:** Δ = d(x′_{t+k}, r_{t+k}), from the exact map. Every analysis uses Δ, never a nominal value. A disturbance with Δ < 1e-9 (for example, when π_D is already saturated in the push direction) is a **null disturbance**. It is counted and excluded from the dose bins.
- **The undisturbed case** starts the policy at the anchor itself. It measures tracking and is reported separately.
- **Identical inputs:** every arm in the cohort gets the same 7,400 initial states: 200 anchors × (36 disturbances + 1 undisturbed). These states are for evaluation only. They select no setting, checkpoint or training data.

### Dose bins and testability (from the pooled tube scale)

- **Inside:** 0 < Δ ≤ q95.
- **Edge:** q95 < Δ ≤ 2·q95.
- **Beyond** (diagnostic): log₂ bins (2^j·q95, 2^(j+1)·q95] for j ≥ 1, up to the largest Δ. A bin is reported only if it holds at least 20 disturbed states.
- **Testability:** the corrective-tube hypothesis is **testable with these checkpoints** only if both hold:
  - the inside bin holds at least 200 disturbed states;
  - the edge bin holds at least 200 disturbed states.
- **If it is not testable** (for example, because the label tube is effectively zero and no feasible disturbance resolves it), the study reports that, and decision 2 is not made. The full Δ curve is still reported as a diagnostic. That outcome by itself would explain why exposure, rather than corrective information, could account for the aggregate effect.

## Recovery (fixed)

After the disturbance, the evaluated policy controls for H = 300 steps. Its commanded action is clipped to ±3 by `Task.step`.

- **Recovered:** there are a single lag s with |s| ≤ 25 steps (0.5 s) and an onset j ≤ 250 such that d(x_{j+ℓ}, r_{j+ℓ+s}) ≤ ρ for every ℓ = 0, …, 49. ρ = 0.1, the demonstrator's LQR-branch scale.
  - The lag is fixed throughout the 1 s dwell, so it follows one coherent continuation.
  - A lag that would need a reference frame before the reference's first frame is unavailable. This applies to the capture states, whose continuation starts at the state itself.
  - The selected (s, j) is recorded, taking the smallest j and then the smallest |s|, with negative s first on a tie.
- **Also recorded per rollout:**
  - time to recovery, j·dt;
  - integrated distance Σ_j min_{|s|≤25} d(x_j, r_{j+s})·dt (descriptive);
  - final hold for demonstrator LQR-branch anchors (within 8° of upright at each of the last 100 steps);
  - upright occupancy;
  - non-finite actions, saturated fraction and failures.
- **Reference behaviors,** reported on the same inputs (descriptive):
  - π_D itself;
  - the zero action.
  - Aggregate return is never a decision metric.

## Outcomes

- **Primary (corrected cohort): local recovery.** For each (arm, seed), take the recovery fraction in the inside bin and in the edge bin. The statistic is the **log-grid mean** of those two fractions.
- **Diagnostic:**
  - the full recovery curve over all reported bins;
  - its log-grid mean;
  - the undisturbed tracking fraction;
  - the per-stratum, per-k and per-sign breakdowns.
- **Paired statistic:** per-seed differences, with a 95% interval that is the union of BCa and t (`ilsb.analysis.union_interval`). "Above 0" means the interval's lower end is > 0.

## Evaluator fitness (capture cohort)

- **The intervention** is the one on which the capture effect was established: each seed's saved held-out capture states, the `diagnostic/x` array of `select/s{seed}.npz` in the capture run, identity-checked by the audit.
- **Each arm** starts from each such state undisturbed. Its reference continuation r is π_D from that state, and it is scored by the recovery rule above.
- **The statistic** is the per-seed recovery fraction over those states. These states define this cohort's inputs. The anchors and disturbances above are reported for it descriptively only.

## Decision rules (sequential; each gates the next)

1. **Evaluator fitness (capture cohort).**
   - The evaluator is **fit** only if Relabel − Duplication recovery on the held-out capture states is above 0.
   - Otherwise the study **stops**. It reports that the evaluator does not detect the established capture effect. The corrected cohort is not evaluated.
2. **Local recovery signal (corrected cohort; only if testable).**
   - CCIL − BC primary log-grid mean above 0 means **signal**: go to step 3.
   - Otherwise the result is **"no local recovery-tube enlargement detected"**. The study stops and publishes it. No duplication arm is trained.
3. **Attribution (only after a signal).**
   - Train the exposure-matched duplication arm (next section) and evaluate it on the same inputs.
   - CCIL − Duplication primary above 0 means **supervision-driven**. Otherwise the effect is **not attributable to corrective information**.
   - Duplication − BC (exposure) and CCIL − BC (total recipe) are reported with the same intervals and do not decide.

The tests are sequential and each gates the next, so no multiplicity adjustment is applied. The diagnostic curves, the breakdowns and the response-surface cohort do not decide.

## Exposure-matched duplication (step 3 only)

- **The original candidate artifacts stay immutable.**
- **A training-only supervision view** is built per seed from the CCIL unit's reconciled `candidates.npz`:
  - every accepted synthetic slot is replaced by its source row (x_i*, u_i*);
  - slot count, ordering, weights, per-source multiplicity and sampler exposure are preserved.
- **The view carries only the fields the frozen trainer reads for supervision.** Generation-describing fields are not copied: targets, residuals, model measurements and certificates. If the frozen trainer cannot consume a view without such a field, the study stops for a protocol amendment rather than filling it with stale values.
- **Hashing:** the view and its construction manifest are hashed. The manifest records:
  - the source candidates' hash;
  - the mapping from slot to source row;
  - the fields kept and dropped;
  - the code commit.
- **Training:** the CCIL policy trainer at the corrected study's frozen CCIL commit `6fbcce5`, with the CCIL arm's stage configuration unchanged, including:
  - run.seed;
  - epochs, batch size and learning-rate schedule;
  - the noise model and noise-mask probability;
  - the demonstration weight;
  - checkpoint selection "last";
  - the device.
- **Determinism:** the original training was not bitwise deterministic (`compute.deterministic: false`), and this arm inherits that, unchanged.
- **Content:** the arm contains no new state or action information.

## Running and artifacts

- **Order of work:**
  1. Audit (no policy loaded).
  2. Phase 1: geometry, disturbances and bins. Then commit and push `geometry.json`.
  3. Capture fitness.
  4. Corrected-cohort evaluation, only if fit and testable.
  5. Duplication training, only if there is a signal.
  6. Duplication evaluation.
- **Execution:** the tool runs on siddh in tmux. It writes to no input run.
  - The log is in `runs/`, headed by the command, commit, dirty state, host, load and versions.
  - Progress is logged at least every 5 minutes.
  - Per-policy results are written atomically, with `--resume`.
  - The stall threshold is 30 minutes, and a watcher checks for completion, failure and stalls.
- **Tests before merge** use only synthetic or off-protocol fixtures. Examples:
  - π_D as the policy recovers undisturbed with lag 0 and onset 0;
  - a policy that follows a phase-shifted copy of a periodic reference recovers only at a single coherent lag;
  - the zero action fails from swing anchors;
  - every applied disturbance action lies in [−3, 3];
  - a tampered hash excludes its cohort;
  - phase 2 refuses a mismatched geometry hash;
  - the duplication view leaves its source artifact byte-identical and drops generation fields.
- **Committed evidence** goes in `docs/studies/evidence/pendulum-recovery-tube/`: the audit, `geometry.json`, per-policy summaries, intervals, decisions, run logs and the phase portraits. Every plotted cell and trajectory links to a machine-readable record.
- **Phase portraits,** on common axes e ∈ [−π, π) and ω ∈ [−8, 8]:
  1. references, anchors and disturbance end-states;
  2. synthetic states and corrective-action arrows, with the tangent and transverse displacement;
  3. recovery and failure maps per arm;
  4. the CCIL − control recovery difference;
  5. recovery versus achieved Δ, with q50, q95, the maximum and the τ★ reference line marked.

## Not done

- No new generator, oracle, horizon, σ, seed or data-fraction sweep.
- No M > 1, and no change to any recipe.
- No rescue of the published Pendulum result.
- No duplication as a user-facing CCIL mode.
- No claim of generality beyond Pendulum.
