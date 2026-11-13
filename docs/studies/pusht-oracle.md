# PushT oracle-dynamics ceiling for one-step CCIL (protocol)

**The oracle.** The oracle arm is a **latent-state-assisted oracle with a fixed history decoder**:
- its labels are exact for the simulator, verified in the demonstration's own contact history;
- each label's previous history slot is a fixed first-order encoding of the solved velocity, not a demonstrated or
  verified realizable history.

This is the protocol of [#43](https://github.com/personalrobotics/il-supervision-benchmarks/issues/43). It is frozen
before any oracle label is generated and before any new learner outcome is read. The machine-readable form is
[`tools/pusht_v2/oracle.json`](../../tools/pusht_v2/oracle.json).

**Question.** If dynamics-model error is removed while the proposals, the source rows, the learner and the synthetic
exposure stay fixed, do CCIL labels help more than duplicating the same demonstrations?

**Status of the evidence.** This is an exploratory mechanism study. The cell (r = 0.25, α = 0.25) and the task were
chosen after the PushT surface outcomes were seen ([pusht-surface.md](pusht-surface.md)), so no result here is
confirmatory evidence for CCIL. Exact dynamics make a label transition-consistent with the demonstrated successor.
They do not make it task-optimal, relevant or harmless to a learner.

## Setting

These are fixed from the surface study:
- **Data:** the r = 0.25 development subset, 41 episodes from subset seed 0, with its frozen normalization.
- **Learner:** IBC on the two-frame history, the 80k cosine schedule, the 20k/40k/80k boundaries with the extension
  rule, selection by held-out InfoNCE on D_V, and evaluation on D_dev (seeds 1000–1099).
- **Cell and seeds:** α = 0.25, which means 64 synthetic rows in each batch of 256, on learner seeds 1 and 2.

D_test is not read.

**Proposals.** The existing r = 0.25 pool is the store entry `c30c0f98…`, used by both Stage 1 and Stage 2. It holds
19,996 proposals, which are CCIL's `noisy_action_fe` actions ã: 4 per transition, at 0.1 training-action standard
deviations. Its files are pinned by the committed inventory ([`evidence/reweight/inventory.json`](evidence/reweight/inventory.json)):
- candidates `6f8c9c97…`;
- selection `ed5beb46…`;
- learned labels `labels.npz` `6024b739…`, 19,982 rows.

For each proposal, the oracle keeps its source transition (t → t + 1) and its action ã, and replaces only the learned
dynamics.

## The simulator, and what the history is

The simulator is gym-pusht 0.1.8 on pymunk 6.11.1, `PushT-v0`. Its step semantics:
- **Agent:** a kinematic body under PD control toward the action, with k_p = 100 and k_v = 20, over 10 substeps of
  dt = 0.01.
- **T:** a free rigid body in a space with no gravity, no damping and no floor friction. It moves only through
  contact.

A step therefore depends on the positions q = (agent x, y, T x, y, θ) and on their velocities
v = (agent vx, vy, T vx, vy, ω). The policy's two-frame history (q_{t−1}, q_t) does not determine v. This is why the
protocol fixes both how the solve uses velocities and how its result becomes a history.

The pre-protocol evidence is [`evidence/pusht-oracle/probe.json`](evidence/pusht-oracle/probe.json), from
[`oracle_probe.py`](../../tools/pusht_v2/oracle_probe.py). It is outcome-free and byte-repeatable, and it covers the
41 subset episodes:
- **Replay.** A demonstration replayed from its first frame with its recorded actions reproduces its recorded frames.
  The reset sets the T's angle before its position, because setting the angle rotates the body about its centre of
  gravity. 4,071 of the 5,040 frames match within 10⁻³ px. The rest fall in the agent's initial transient, whose
  unknown starting velocity decays by half each step, or follow a contact during that transient.
- **Fresh state.** One step from a freshly built state matches the recorded successor, with positions and velocities
  set directly and no contact history:
  - steps without contact: 99th percentile 0.0005 px, maximum 0.014 px;
  - steps with contact: 99th percentile 0.024 px, maximum 0.33 px. pymunk's contact cache is the difference.
- **Velocity substitution.** Reassigning velocities in a live replay keeps its contact cache intact. A second replay
  run in lockstep, with its velocities reassigned before every step, equals the plain replay bitwise on all 4,999
  steps.
- **Spacing.** The demonstrations' frame-to-frame displacement has a 10th percentile of 2.77 px and a median of
  7.85 px.

Distances are the larger of the agent's displacement and the T's largest keypoint displacement, in px.

## Stage A: oracle labels, no training

**The source state.** A proposal's source state is the replayed simulator state at frame t. That state has the
recorded positions q_t and the demonstration's true velocities v_t. A proposal is usable only if frames t and t + 1
are both replay-verified within 10⁻³ px. Otherwise it is rejected as `source_not_verified`.

**Two operators.** A label is solved in one operator and accepted in the other.
- **The fresh operator** steps once under ã from a freshly built state, with no contact history.
- **The actual-history operator** replays the demonstration in one continuous simulation, from its first frame (the
  same reset) to frame t. It then substitutes the label's velocities for the replay's own and steps once under ã. Its
  contact cache is the demonstration's own at t.

A label's residual in either operator is its distance from the recorded successor q_{t+1}. The tolerance is
**τ = 0.05 px**, about twice the 99th percentile of the demonstrations' own fresh-step residual with contact. The
demonstrations' own actual-history step is their replay, within 10⁻³ px.

**The solve,** in the fresh operator.
- **Unknowns:** u = (agent vx, vy, T vx, vy, ℓ·ω) in px/s, at the source positions q_t.
- **Residual:** r = (agent Δx, Δy, T Δx, Δy, ℓ·wrap(Δθ)) in px. Each Δ is the step's result minus q_{t+1}, wrap maps
  an angle to (−π, π], and ℓ = 120.93 px is the largest keypoint radius max_i |LOCAL_i|. So ℓ·Δθ bounds the
  keypoint motion that rotation causes.
- **Method:**
  - Newton's method on r, starting from the demonstration's own u, with at most 20 iterations;
  - J = ∂r/∂u by central finite differences, with a step of 10⁻³ px/s in each component of u;
  - a backtracking line search that halves the step up to 8 times while ‖r‖₂ does not decrease.
  - The solve converges when the distance from q_{t+1} is at most τ.
- **Singular Jacobian:** at any iteration, the 2-norm condition number of J exceeds 10¹⁰. The label is rejected as
  `singular_jacobian`.
- **Failure:** if the solve has not converged after the cap, or produces a non-finite value, the label is rejected as
  `solve_failed`.

**Acceptance,** in the actual-history operator. A converged label is stepped through the actual-history operator, and
it is **exact** only if that step also reaches q_{t+1} within τ. A label that converges in the fresh operator but
misses here is rejected as `fails_actual_history` and counted separately.

Without contact, the T's velocity is then (q_{t+1} − q_t)/0.1 and the agent's is fixed by its linear PD dynamics. The
oracle label differs from the source only where the dynamics allow it to.

**Into the history.** The label's current slot is the source frame's own, q_t, exactly as in the learned labels. Its
previous slot is

    q′_{t−1} = q_{t−1} + (v_t − v′) · 0.1

- **For the T:** this is the previous frame from which the label's velocity carries it to q_t without contact. That is
  exact physics whenever the T was not touched during that step.
- **For the agent:** it keeps the demonstration's own offset between its frame difference and its velocity. At
  v′ = v_t the label is the source exactly.

The T pose decodes to keypoints by the rigid map (`pose.keypoints`). This is a fixed first-order decoding. The
history cannot carry more than that.

**Validity.** Each label is checked on its previous slot, its history invariant and its finiteness, with these
rejection reasons:
- **Previous slot within the demonstrated envelope.** The simulator lets its kinematic agent penetrate the T and
  the walls, and it can push the T past them. The demonstrations' own previous slots in the subset show both: 2,288
  of 5,040 rows have the agent disk inside the T, with a median depth of 0.13 px and a maximum of 22.98 px, and the
  agent disk and T reach from 7.49 to 523.06 px (`envelope` in the probe). A label's previous slot is therefore
  checked against those maxima, never against an ideal non-penetrating geometry:
  - the agent disk (radius 15) penetrates the T's nearer rectangle by at most **22.981 px**. Otherwise
    `beyond_penetration_envelope`;
  - every coordinate of the agent disk and the T's keypoints lies in **[7.4939, 523.0609] px**. Otherwise
    `beyond_extent_envelope`.

  These are envelope checks: no label is less physical than a demonstrated state. They are not physical feasibility.
- **History invariant:** the current slot equals the source's bitwise. Otherwise `invariant`.
- **Finite:** every value is finite. Otherwise `non_finite`.

**The common rows.** A proposal enters the matched pool only when:
- it is a learned label in `labels.npz`;
- its source is verified;
- its oracle label is exact and valid.

All three synthetic arms use exactly these proposals, in candidate-row order:
- learned CCIL with its existing labels;
- oracle CCIL with its oracle labels;
- duplication with the source rows.

**Report, without learner outcomes.** The record (`ilsb.pusht_oracle.stage_a 1`) holds:
- the identities of the code, the simulator, the data, the pool files and the probe;
- yields, and every rejection reason with its count;
- replay residuals and iteration counts;
- the displacement of the previous slot from the source's;
- the distance to the nearest demonstration row of the subset, in the policy's observation;
- the oracle-versus-learned disagreement on the previous slot. The action is the proposal's in both, so the action
  disagreement is 0 by construction, and that is recorded;
- the common count, and the number of passes over it that 20k steps of 64 rows imply.

Per-proposal arrays go to a content-hashed `.npz` kept outside git.

**Stage A's rules**, with s = 2.77 px, the 10th percentile of demonstration spacing:
1. **Fails** if the common pool holds fewer than 64 rows, one synthetic batch. α = 0.25 then cannot be realized
   without resampling.
2. **Fails** if the oracle labels are not novel: the median previous-slot displacement from the source is at most s.
3. **Stops without training** if the oracle and learned labels are indistinguishable: their median previous-slot
   disagreement is at most s. Model error is then not a plausible explanation.
4. **Otherwise passes.**

A failed Stage A is a result. The solver, τ, the decoding, the proposals, the dose and the cell are not changed after
it is seen.

## Stage A result

**Stage A passes.** The oracle labels are novel and distinguishable from the learned ones, and the common pool is large
enough. Stage B may run after separate approval of its paid jobs.

The record is [`evidence/pusht-oracle/stage-a.json`](evidence/pusht-oracle/stage-a.json), from
[`oracle_stage_a.py`](../../tools/pusht_v2/oracle_stage_a.py) at `3ad4320` on siddh, from a clean tree; the record
holds that commit and its source files' SHA-256. Two independent runs wrote it byte for byte.
Its per-proposal arrays stay on siddh, under `runs/20261112T233659Z-oracle-stage-a-a/proposals.npz`, identified by
the content SHA-256 in the record.

**Yield.** 12,743 of the 19,996 proposals have exact, valid oracle labels. Every rejection, by reason:

| reason | proposals |
|---|---|
| source frame or successor not replay-verified | 4,040 |
| the fresh solve did not converge in 20 iterations | 2,597 |
| converged in the fresh operator, missed in the actual history | 308 |
| previous slot beyond the demonstrated penetration | 275 |
| previous slot beyond the demonstrated extent | 33 |
| singular Jacobian, non-finite, invariant | 0 |

The **common pool** has 12,739 proposals: the exact oracle labels that are also learned labels. At α = 0.25, 20k
steps pass over it about 100 times.

**The rules,** with s = 2.77 px:
- **Novelty:** the oracle labels' median previous-slot displacement from the source is 11.4 px, above s.
- **Distinguishable:** oracle and learned labels disagree by a median of 19.1 px, above s. The action is the
  proposal's in both.

**Accuracy.** Accepted labels reach the successor to a median of 5 × 10⁻⁵ px in the actual history, and to at most
0.05 px. The median solve takes 1 iteration and the maximum 9. The demonstrations' own previous slots all lie within
the envelope.

**What the learned model does differently,** on the common rows (medians unless stated):

| | oracle | learned |
|---|---|---|
| previous-slot displacement from the source | 11.4 px | 24.3 px |
| distance to the nearest demonstration (policy observation) | 12.6 px | 38.9 px |
| agent penetration into the T, 90th percentile | 8.9 px | 13.5 px |
| beyond the demonstrated penetration envelope | 0 | 359 of 12,739 |
| beyond the demonstrated extent envelope | 0 | 44 of 12,739 |

For the same proposals, the learned dynamics place their labels about twice as far from the source and three times as
far from the data as the simulator does. Some of those labels are physically less plausible than any demonstrated
state. Whether that difference matters to the learner is Stage B's question.

## Stage B: the matched learner comparison

Stage B runs only if Stage A passes, and only after separate approval of its paid jobs.

**Arms**, on seeds 1 and 2:
- **BC;**
- **duplication**, learned CCIL and oracle CCIL. All three use the common pool's proposal identities, source rows, row
  count and order, with requested synthetic mass α = 0.25, and the same learner, optimizer, stopping rule and
  checkpoint rule.

An existing unit is reused only when its complete training identity matches. Otherwise it is retrained. That means at
most six synthetic-arm units, plus BC if its identity has changed.

**Primary diagnostic:** oracle CCIL − duplication in average maximum coverage, reported separately for each seed. The
oracle ceiling is established only if it is positive on both seeds.

**Also reported:**
- oracle − learned;
- learned − duplication;
- success rate and the checkpoint selections, descriptively.

With two seeds the report gives the paired values and their mean, with no interval and no population claim.

**Decision.**
- **Any of the eight fits is incomplete** (its selected checkpoint is the 80k one): the study is **inconclusive**, and
  no substantive claim is made. This rule is checked first and applies whatever the differences are.
- **Oracle does not beat duplication on both seeds:** stop. **This oracle completion did not help in this cell.** The
  result does not rule out dynamics-model error in general, because the oracle's history slot is a fixed decoding
  whose realizability is not verified. It does leave the generator, the placement, the exposure and the learner
  interaction as the more likely limits.
- **Oracle beats duplication but learned does not:** model fidelity is a demonstrated bottleneck in this
  construction. Any learned-model
  follow-up gets its own protocol.
- **Both beat duplication:** the construction has a positive development signal. Confirmation on untouched seeds
  belongs in a separate issue.

No outcome authorizes a final-test run.

**Cost.** Stage A runs on siddh at no cost. On Tillicum, a 20k unit costs about 0.11 GPU-hours, and every unit at
r = 0.25 stopped at 20k in the surface study.

| | GPU-hours | cost |
|---|---|---|
| expected (8 units, plus the stage jobs) | about 1.2 | about $1.10 |
| worst case (every unit runs to 80k) | about 3.5 | about $3.20 |

### How Stage B runs

The harness runs Stage B from [`tools/pusht_v2/oracle-stage-b.json`](../../tools/pusht_v2/oracle-stage-b.json), which is
the surface study's Stage 2 specification at one cell, (0.25, 0.25), on seeds 1 and 2, with an `oracle` arm and a
`matched` block. Its learner, selection and evaluation are the surface study's, unchanged.

- **The matched rows.** No pool is generated. The harness reads two pinned files from `ILSB_ORACLE_INPUTS`:
  - the learned `labels.npz`, by SHA-256 `6024b739…`;
  - Stage A's `proposals.npz`, by content SHA-256 `b31c2201…`.

  It keeps the 12,739 common proposals in candidate-row order. Each one appears as its learned label, its oracle label
  and its source row. The oracle label's previous slot is Stage A's decoded q′_{t−1}; its current slot and action are the
  source frame's and the proposal's. Before any unit trains, it checks:
  - the counts;
  - that each common proposal's learned label is that proposal's, from the same source;
  - that every source is in the subset;
  - that the oracle labels are finite;
  - that both labels' current slots equal the source frame bitwise;
  - containment by content: no label's current frame is a held-out frame.
- **Pairing.** Duplication, learned CCIL and oracle CCIL draw the same index streams at every step. The audit checks
  this for both CCIL arms.
- **BC is retrained.** The harness implementation is part of every store identity, so the surface study's BC units
  do not match. Stage B therefore trains all 8 units.

**Smoke.** [`evidence/pusht-oracle/stage-b-smoke.json`](evidence/pusht-oracle/stage-b-smoke.json) records the whole
chain on [`oracle-stage-b-smoke.json`](../../tools/pusht_v2/oracle-stage-b-smoke.json): seed 1, boundaries
500/1000/2000, 4 evaluation episodes. It ran on siddh at `e3b228d`, from a clean tree, and every stage passed. The
matched rows passed their checks: 12,739 common rows, and no held-out current frame in either arm. All three
synthetic arms drew identical index streams. Their fits reach the smoke's 2k ceiling, so they are incomplete, and the
report correctly returned **inconclusive**, with oracle not established. It records no outcome values. The scheduler's failure and resume path
is unchanged from the surface study's smoke.

**Exact maximum cost** on Tillicum (`gpu-h200`, `normal` QoS, $0.90 per GPU-hour, 1 GPU per job), by requested time:

| job | requested | maximum |
|---|---|---|
| preflight | 1 × 0:20 | $0.30 |
| units (array of 8, `%8`) | 8 × 1:00 | $7.20 |
| train (collect), audit, evaluate, evaluation_audit, report | 1 × 0:45 | $0.68 |
| **total** | 9.08 GPU-hours | **$8.18** |

A unit's 80k worst case is about 0.45 hours on an H200, inside its 1-hour limit. The jobs bill on elapsed time, so the
expected cost is about $1.10. A failed job is not resubmitted without a separate decision.

## Stage B result

**Stop. This oracle completion did not help in this cell.** Oracle CCIL scored below duplication on both seeds, so the
oracle ceiling is not established. All eight fits were complete, so the fail-closed rule did not apply.

The result does not rule out dynamics-model error in general: the oracle's history slot is a fixed decoding whose
realizability is not verified. It does leave the generator, the placement, the exposure and the learner interaction as
the more likely limits.

Average maximum coverage on D_dev (100 starts), at (r, α) = (0.25, 0.25):

| seed | BC | duplication | learned CCIL | oracle CCIL | oracle − duplication | oracle − learned | learned − duplication |
|---|---|---|---|---|---|---|---|
| 1 | 0.408 | 0.422 | 0.478 | 0.416 | −0.005 | −0.062 | +0.056 |
| 2 | 0.485 | 0.457 | 0.429 | 0.428 | −0.029 | −0.001 | −0.029 |
| mean | | | | | −0.017 | −0.031 | +0.014 |

- **Learned CCIL:** above duplication on seed 1 and below it on seed 2. With two seeds this is the same reversal as in
  the surface study, not a signal.
- **Success rates** (%, seed 1 / seed 2): BC 9 / 6, duplication 4 / 15, learned CCIL 5 / 6, oracle CCIL 3 / 5.
- **Selections:** every unit selected its 2.5k checkpoint and stopped at the first boundary (20k).

With two seeds these are paired values and their mean, with no interval and no population claim.

**The record** is [`evidence/pusht-oracle/stage-b.json`](evidence/pusht-oracle/stage-b.json). It holds:
- the commit `ffb376a` (the merge of #47), from a clean clone on Tillicum;
- each stage record's SHA-256;
- every unit's identity and store entry;
- the per-seed values;
- the compute.

Every stage passed, and the evaluation repeat was within tolerance. Stage B used 0.89 H200 GPU-hours, which cost
$0.80 against its $8.18 ceiling.

## Not in this study

- changing CCIL's product code or its quality schema;
- another weighting or routing statistic;
- searching over r, α, the noise, the solver settings or the learners;
- multi-step recovery segments;
- treating exact dynamics as deployable;
- reopening the surface study;
- reading D_test.
