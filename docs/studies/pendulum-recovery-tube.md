# Pendulum recovery-tube expansion from existing checkpoints: predeclaration, version 1 (#18)

**Status: version 1, frozen.** The protocol was approved after review on 2026-11-02 and frozen by the commit that adds this status line. No evaluator code existed at that commit. The audit, evaluator, fixtures and duplication support follow as later commits in the same pull request. That pull request is merged before any real checkpoint is evaluated, including the capture control. Results go in one later pull request. Any later change to this protocol is a numbered amendment (see "Amendments") with its reason, made before the outcome it could affect is read.

This study is **exploratory**. It does not reopen the completed causal ladder (#12), rerun its treatments or make a confirmatory claim. A favorable outcome is not required. An ambiguous outcome is reported as "not established". After any recovery outcome is read:
- seeds are not added;
- the disturbance set, the dose bins and the tube do not change.

## Result (run 20261102T160255Z from `acbc35a`; exploratory)

**Decisions:** fitness **fit**; signal **signal**; attribution **not attributable to corrective information**. The evidence is in `docs/studies/evidence/pendulum-recovery-tube/`: `audit.json`, `geometry.json`, `decision.json`, every per-policy result record (`evaluate/`), the matched pair's manifest and view records (`matched/`), the run logs and scripts, and the phase portraits `pendulum-recovery-tube.svg`.

- **Audit:** the capture and corrected cohorts are eligible, with no failures. The response-surface cohort is excluded: one reference (`plan_sha256`) does not resolve. It is not evaluated.
- **Geometry (frozen and pushed as `95f7964` before any policy was loaded):**
  - The pooled label tube is q50 = 0.00013, q95 = 0.00526 and max = 0.0095. Per-seed q95 ranges from 0.0049 to 0.0064.
  - The tube radius is ρ = q95 = 0.00526, about a seventh of τ★.
  - The study is testable, with 993 edge states.
  - Every applied action lies in [−3, 3].
  - π_D recovers from 0.994 of edge states. The zero action recovers from none.
- **Decision 1, sensitivity gate:** held-out capture recovery at ρ = 0.1 is relabel 1.000, duplication 0.407 and BC 0.523. Relabel − duplication is +0.593, with a 95% interval of [+0.237, +0.950]: **fit**.
- **Decision 2, signal:** edge-bin recovery to ρ = q95 is CCIL 0.421 and BC 0.002. CCIL − BC is +0.420, [+0.215, +0.625]: **signal**.
- **Decision 3, attribution:** edge-bin recovery is replay-CCIL 0.421 and duplication 0.241. Replay-CCIL − duplication is +0.180, [−0.070, +0.430], so the effect is **not attributable to corrective information**. The reported, non-deciding contrasts:
  - **Exposure** (duplication − BC): +0.240, [+0.066, +0.433].
  - **Total** (CCIL − BC): +0.420, [+0.215, +0.625].
  - **Retraining drift** (CCIL − replay-CCIL): exactly 0 on every seed.
    - The 40 jobs trained in one group (bucket 64) under `6fbcce5` on CUDA.
    - Every replay policy's arrays are bitwise identical to the original checkpoint's. The files' hashes differ only in the npz container.
    - Every per-state result array is identical, even though the original run did not request deterministic training.
- **The duplication views** are byte-identical across seeds (`9938dfeb…`). Every seed's CCIL unit accepted all 998,000 synthetic rows: 20 per demonstration row, over the same 49,900 rows. The duplication arms differ only in their training seed.
- **Descriptive: the tube measures tracking more than recovery.**
  - **Undisturbed tracking** at ρ = q95 (staying within the tube from an undisturbed anchor) is CCIL 0.511, duplication 0.245 and BC 0.001.
  - **Edge recovery** (0.421, 0.241 and 0.002) and **inside retention** (0.467, 0.251 and 0.002) follow the same order. Each arm's recovery is roughly flat in Δ from 10⁻³ to 10⁻¹ (portrait panel c).
  - So at this tube, the edge-bin contrast mostly measures how closely a policy reproduces the reference trajectory when undisturbed. It does not measure whether the policy corrects from just outside the tube.
  - The frozen rule makes no adjustment for this. It is reported here, not used to change any decision.
- **Conclusion:** on Pendulum, the corrected-study CCIL policies stay near the demonstrator's trajectories far more often than BC, including after small bounded disturbances.
  - Exposure-matched duplication of the demonstration rows recovers a substantial share of that gain: +0.240 of the +0.420.
  - The remainder, +0.180, is not distinguishable from zero over 20 seeds. The study therefore does not attribute CCIL's advantage to the corrective content of its labels.
- **Scope:** exploratory. M = 1 and dt = 20 ms are unchanged. No seeds, disturbances, bins or tubes were changed after any outcome was read. Per the protocol, no further training follows.

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
   - **Eligible** only if the checkpoint → record → committed-rollout hash chain reconciles for all 40 units, as set out in Amendment 1.
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

### Dose bins and testability (from the pooled tube scale; as amended by Amendment 3)

- **Target tube for the corrected cohort:** radius ρ_c = q95, the pooled label-tube scale. Every bin of this cohort, and the response-surface cohort's descriptive curves, are scored against ρ_c.
- **Inside** (retention, reported separately): 0 < Δ ≤ q95. These states start in the target tube, so success means staying there.
- **Edge** (primary): q95 < Δ ≤ 2·q95. These states start outside the target tube, so success means returning to it.
- **Beyond** (diagnostic): log₂ bins (2^j·q95, 2^(j+1)·q95] for j ≥ 1, up to the largest Δ. A bin is reported only if it holds at least 20 disturbed states.
- **Testability:** the corrective-tube hypothesis is **testable with these checkpoints** only if q95 > 0 and the edge bin holds at least 200 disturbed states.
- **If it is not testable** (for example, because the label tube is effectively zero and no feasible disturbance resolves it), the study reports that, and decision 2 is not made. The full Δ curve is still reported as a diagnostic. That outcome by itself would explain why exposure, rather than corrective information, could account for the aggregate effect.

## Recovery (fixed)

After the disturbance, the evaluated policy controls for H = 300 steps. Its commanded action is clipped to ±3 by `Task.step`.

- **Recovered (as amended by Amendment 2):** there are a single lag s with |s| ≤ 25 steps (0.5 s) and an onset j ≤ 250 such that d(x_m, r_{m+s}) ≤ ρ for every m = j, …, 300.
  - The tube radius ρ is ρ_c = q95 for the corrected, duplication and response-surface cohorts (Amendment 3). It is ρ = 0.1, the demonstrator's LQR-branch scale, for the capture sensitivity gate.
  - The rollout enters the tube and stays there to the end of the horizon, which is at least 50 steps (1 s).
  - The lag is fixed from onset to the end, so the rollout follows one coherent continuation.
  - The selected (s, j) is recorded, taking the smallest j and then the smallest |s|, with negative s first on a tie.
  - A lag that would need a reference frame before the reference's first frame is unavailable. This applies to the capture states, whose continuation starts at the state itself.
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

- **Primary (corrected cohort): local recovery (Amendment 3).** For each (arm, seed), the recovery fraction over the edge bin: states that start outside the target tube ρ_c = q95 and must return to it.
- **Reported separately:** retention, the success fraction over the inside bin.
- **Diagnostic:**
  - the full recovery curve over all reported bins;
  - its log-grid mean;
  - the undisturbed tracking fraction;
  - the per-stratum, per-k and per-sign breakdowns.
- **Paired statistic:** per-seed differences, with a 95% interval that is the union of BCa and t (`ilsb.analysis.union_interval`). "Above 0" means the interval's lower end is > 0.

## Evaluator sensitivity gate (capture cohort; as amended by Amendment 3)

- **This is a new sensitivity gate, not a reproduction.** The capture control established held-out action error on these states and final hold from fresh starts. It did not establish recovery from these states. The gate asks whether this evaluator registers the established capture supervision as recovery.
- **The inputs** are each seed's saved held-out capture states, the `diagnostic/x` array of `select/s{seed}.npz` in the capture run, identity-checked by the audit.
- **Each arm** starts from each such state undisturbed. Its reference continuation r is π_D from that state, and it is scored by the recovery rule above.
- **The statistic** is the per-seed recovery fraction over those states. These states define this cohort's inputs. The anchors and disturbances above are reported for it descriptively only.

## Decision rules (sequential; each gates the next)

1. **Evaluator sensitivity (capture cohort).**
   - The evaluator is **fit** only if Relabel − Duplication recovery on the held-out capture states is above 0.
   - Otherwise the study **stops**. It reports that this evaluator does not register the established capture supervision as recovery. The corrected cohort is not evaluated.
2. **Local recovery signal (corrected cohort; only if testable).**
   - CCIL − BC primary (edge-bin recovery) above 0 means **signal**: go to step 3.
   - Otherwise the result is **"no local recovery-tube enlargement detected"**. The study stops and publishes it. No duplication arm is trained.
3. **Attribution (only after a signal; as amended by Amendment 3).**
   - Train the matched pair (next section): replay-CCIL and duplication, in one group. Evaluate both on the same inputs.
   - Replay-CCIL − Duplication primary above 0 means **supervision-driven**. Otherwise the effect is **not attributable to corrective information**.
   - Reported with the same intervals, without deciding:
     - original CCIL − replay-CCIL (retraining drift);
     - Duplication − BC (exposure);
     - original CCIL − BC (total recipe).

The tests are sequential and each gates the next, so no multiplicity adjustment is applied. The diagnostic curves, the breakdowns and the response-surface cohort do not decide.

## Matched replay and duplication (step 3 only; as amended by Amendment 3)

- **Two arms are trained fresh, together:**
  - **replay-CCIL:** the CCIL arm's supervision, rebuilt read-only and required to reproduce the unit's recorded supervision manifest and training identity;
  - **duplication:** the training-only view below.
- **Matching:** both arms of each seed are trained in one `train_group` call of 40 jobs (bucket 64, as in the original run). They share:
  - the seed, so the same initialization (`torch.manual_seed(seed)`) and the same shuffling keys (seed, epoch) over the same row count;
  - the settings, the device and the determinism flag;
  - the execution shape.

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
- **Training:** the CCIL policy trainer at the corrected study's frozen CCIL commit `6fbcce5`, with the CCIL arm's stage configuration unchanged for both arms, including:
  - run.seed;
  - epochs, batch size and learning-rate schedule;
  - the noise model and noise-mask probability;
  - the demonstration weight;
  - checkpoint selection "last";
  - the device.
- **Determinism:** the original training was not bitwise deterministic (`compute.deterministic: false`). Both fresh arms inherit that, unchanged. Original CCIL − replay-CCIL measures the resulting drift.
- **Content:** the arm contains no new state or action information.

## Running and artifacts

- **Order of work:**
  1. Audit (no policy loaded).
  2. Phase 1: geometry, disturbances and bins. Then commit and push `geometry.json`.
  3. Capture fitness.
  4. Corrected-cohort evaluation, only if fit and testable.
  5. Replay-CCIL and duplication training, only if there is a signal.
  6. Replay-CCIL and duplication evaluation.
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

## Amendments

### Amendment 1 (2026-11-02, before any evaluator code or outcome): the corrected cohort's hash chain

**Reason.** Version 1 said the `policy-evaluate` stage consumed the checkpoint and produced the committed rollout. That is wrong. At CCIL `6fbcce5`, `policy.evaluate` computes offline action-error metrics only. The committed rollouts in `study-evaluate.json` come from the separate evaluate step, `tools/pendulum_eval/corrected_study.py evaluate` at `e1ca5b5`, run as `20261030T013609Z-corrected-eval`. That step loads each checkpoint from the unit record's `policy.train` stage and refuses any checkpoint whose sha256 is not the record's `checkpoint_sha256`. Only the description of the mechanism changes. The all-or-nothing exclusion rule is unchanged.

**The chain, for each of the 40 units (continuous; BC and CCIL; seeds 1201–1220):**
1. **Record:** the run's unit record `results/study/continuous/{arm}/s{seed}.json` has status `ok`. The committed CCIL `study-audit.json` lists the same unit as `ok`.
2. **Checkpoint:** the sha256 of `checkpoints/policy.npz` in the record's `policy.train` stage equals both the stage manifest's `files` entry and the record's `checkpoint_sha256`. That stage directory appears in the committed `study-audit.json` `stages` with `pass: true`.
3. **Rollout:** in the evaluate run, the sha256 of `continuous/{arm}/s{seed}-0.0.npz` equals `rollout_sha256["continuous/{arm}/s{seed}/0.0"]` in the committed CCIL `study-evaluate.json`. The checkpoint-to-rollout link is the evaluate step's refusal rule above. No hash inside the rollout names the checkpoint, and this study does not re-create the rollouts.
4. **CCIL inputs:** for each CCIL unit, the `labels.generate` stage's `labels/candidates.npz` has the sha256 recorded under `inputs.candidates` in the `policy.train` manifest. The dataset input is `59c332bc…cf43c`.

### Amendment 2 (2026-11-02, before any real checkpoint is evaluated): recovery means staying in the tube

**Reason.** The off-protocol self-test (references from eval seed 9001; no study policy) expected the zero action to fail from swing anchors. Under version 1's rule it was credited with recovery from 2 of 3:
- Near the end of energy pumping, the reference's energy is 9.61–9.62 against an upright energy of 9.81, and π_D's action there is 0.03–0.84.
- An unpowered pendulum therefore coasts alongside the reference within ρ for 69 and 109 steps, longer than the 50-step dwell. It then falls: the maximum distance over the horizon is 3.7.

Version 1 asked for a return plus a 1 s stay, not for remaining afterwards. So coasting earned credit for every arm, including policies that then fall.

**Change.** A rollout is recovered only if it stays in the tube at one fixed lag from its onset (j ≤ 250) through step 300, the end of the horizon. Onset, lag, ρ, the lag window, the horizon and the tie rule are unchanged.

**Effect on the self-test.** The zero action now fails from every swing anchor. Every other check passes as before.

### Amendment 3 (2026-11-02, after review; before any real checkpoint is evaluated): primary outcome, attribution pair, sensitivity gate, integrity

**Reasons.**
- **Retention was standing in for recovery.** The corrected cohort's primary bins (Δ ≤ 2·q95) were scored with ρ = 0.1. If q95 ≤ 0.05, every primary state starts inside the success tube, and testability could pass without a single rollout having to return.
- **Attribution was confounded.** The original CCIL checkpoints came from a nondeterministic, mixed bucket-64 group. A freshly trained duplication arm in a differently composed group differs from them by retraining as well as by information content.
- **The capture gate was misdescribed.** Recovery from the held-out capture states was never established, so failing the gate is not a failure to reproduce.
- **Integrity gaps.** Resumed or partial results could reach a decision, and the audit trusted copies of evidence that it did not pin.

**Changes.**
1. **Target tube.** The corrected cohort's target tube is ρ_c = q95.
   - The edge bin, which starts outside ρ_c, gives the primary outcome.
   - The inside bin is retention, reported separately.
   - Testability needs q95 > 0 and at least 200 edge states.
   - The capture sensitivity gate keeps ρ = 0.1.
2. **Matched attribution pair.** Attribution compares replay-CCIL with duplication, trained fresh as a matched pair in one group. Original CCIL − replay-CCIL is reported as retraining drift.
3. **Sensitivity gate.** The capture cohort's gate is a new sensitivity gate (see that section).
4. **Result integrity.**
   - Each per-policy result records an identity: policy hash, geometry hash, audit hash, input hashes and the evaluator source hash. Its JSON records its NPZ's hash.
   - Resume, and every decision, require the exact declared arm × seed matrix. Each result must match its expected identity and have its NPZ hash reconcile.
   - Missing, extra, stale or corrupted results are refused.
   - Every stage refuses a dirty harness tree and any evaluator source hash other than the audit's.
5. **Pinned evidence.** The audit pins the committed CCIL `study-audit.json` and `study-evaluate.json` by sha256. It requires their gates to report:
   - frozen CCIL `6fbcce5`, with no differing CCIL paths;
   - a clean tree and passing gates;
   - the evaluate step's commit `e1ca5b5`;
   - all_pass.
6. **Response-surface completeness.**
   - Environment identity (`scope` and `code`) is not treated as a reference.
   - Every other hash in `root.json`, the run manifest, the stage reports and the identities must resolve, either to a store address, store file or stage file, or to an anchor whose bytes the tool hashes:
     - the dataset file;
     - `digest(root.json)` for the protocol;
     - the spec file at the run's harness commit;
     - the recomputed exact-model identity.
   - Anything unresolved excludes the cohort, and root hashes are no longer accepted merely for being in `root.json`.

## Not done

- No new generator, oracle, horizon, σ, seed or data-fraction sweep.
- No M > 1, and no change to any recipe.
- No rescue of the published Pendulum result.
- No duplication as a user-facing CCIL mode.
- No claim of generality beyond Pendulum.
