# PushT oracle-dynamics ceiling for one-step CCIL (protocol)

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
- **Spacing.** The demonstrations' frame-to-frame displacement has a 10th percentile of 2.77 px and a median of
  7.85 px.

Distances are the larger of the agent's displacement and the T's largest keypoint displacement, in px.

## Stage A: oracle labels, no training

**The source state.** A proposal's source state is the replayed simulator state at frame t. That state has the
recorded positions q_t and the demonstration's true velocities v_t. A proposal is usable only if frames t and t + 1
are both replay-verified within 10⁻³ px. Otherwise it is rejected as `source_not_verified`.

**Exact dynamics.** One step from a fresh state (as above) under ã. A label's residual is its distance from the
recorded successor q_{t+1}. The label is exact when that residual is at most **τ = 0.05 px**: about twice the
99th percentile of the demonstrations' own fresh-step residual with contact.

**The solve.**
- **Unknowns:** the velocities v′ at the source positions q_t, in five dimensions: agent (vx, vy), T (vx, vy) and ω.
- **Equations:** five, the successor positions q_{t+1}.
- **Method:**
  - Newton's method, starting from the demonstration's v_t, with at most 20 iterations;
  - a central finite-difference Jacobian, with a step of 10⁻³ px/s or rad/s in each component;
  - a backtracking line search that halves the step up to 8 times whenever the residual does not decrease.
- **Failure:** the label is rejected as `solve_failed` if the residual exceeds τ after the cap, if any value is
  non-finite, or if the Jacobian is singular.

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
- **Previous slot inside the arena:** the agent disk (radius 15) and the T inside the walls' interior, x and y in
  [7, 504]. Otherwise `outside_arena`.
- **No overlap:** the agent disk and the T polygons do not overlap. Otherwise `interpenetration`.
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
- **Oracle does not beat duplication on both seeds:** stop. Dynamics-model accuracy is not the main explanation in this
  cell; the limit lies in the generator, the placement, the exposure or the learner interaction.
- **Oracle beats duplication but learned does not:** model fidelity is a demonstrated bottleneck. Any learned-model
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

The exact maximum is stated before submission.

## Not in this study

- changing CCIL's product code or its quality schema;
- another weighting or routing statistic;
- searching over r, α, the noise, the solver settings or the learners;
- multi-step recovery segments;
- treating exact dynamics as deployable;
- reopening the surface study;
- reading D_test.
