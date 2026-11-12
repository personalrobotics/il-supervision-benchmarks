# Validation reweighting, Stage A: a read-only retrospective diagnostic

This is the Stage A protocol of [#40](https://github.com/personalrobotics/il-supervision-benchmarks/issues/40). It was
frozen in #41 before any label-level utility was computed, and its result is recorded below. Stage A trains no policy,
runs no simulator, reads no final-test set and uses no paid compute. The machine-readable form is
[`tools/reweight_stage_a.json`](../../tools/reweight_stage_a.json).

## Result

**Stage A fails both directional checks. By the frozen rule, the investigation ends in #40, and no Stage B is opened.**
- **F1 fails.** The helpful capture relabels score above their demonstration comparator on 5 of 10 seeds; the rule
  needs 8.
- **F2 fails, in the wrong direction.** The harmful PushT pools score far above their source rows: D = +109 at
  r = 0.1 and +263 at r = 0.5.
- **The reported-only cases agree with this failure.**
  - Pumping relabels score positive on 7 of 10 seeds, more often than capture. This is the limit stated in advance:
    the statistic measures held-out action fit, which the pumping relabels did improve.
  - Every other PushT case is positive: r = 0.25 on both seeds, r = 1, and all five v2.2 seeds.

The first-order validation utility, as frozen, does not separate the known helpful supervision from the known harmful
supervision on these artifacts. This is a statement about this statistic on uncertified and oracle labels. It says
nothing about certified labels, which no artifact has, or about other routing rules.

**Observation (post hoc, not a basis for continuing).** On PushT, CCIL labels have a smaller positive fraction than
their sources in every case, and a lower signed mean alignment in every case except r = 0.25 on seed 1. Their alignments are
larger in magnitude in both directions, so the clipped utility max(0, g) keeps their large positive terms and inflates
their mean. Searching for a different statistic after seeing outcomes is a non-goal of #40, so none is proposed.

**Mass.** On every PushT pool about half the rows are positive. A batch of 64 or more synthetic rows therefore
almost surely holds a positive row, and the realizable mass equals the requested α to four decimals. Utility routing
would not have reduced synthetic exposure.

**Record.** [`evidence/reweight/stage-a.json`](evidence/reweight/stage-a.json) (`ilsb.reweight.stage_a 1`) holds the
following, with one entry per case and seed:
- the commit `c9ba2b4`, clean;
- the hashes of the inventory and the specification;
- each checkpoint's SHA-256 and the content SHA-256 of its rows and g files;
- the aggregates, the mass, both checks and the decision.

The per-row files stay on siddh, under `runs/20261112T203149Z-stage-a`, identified by those content hashes.

**Verification.**
- The forward-mode g agrees with independent reverse-mode per-row gradients to FP32 precision, at a relative error
  of at most 5 × 10⁻⁶. Rows were checked in both learners.
- Recomputing g on a second run gives bit-identical values in all 30 cases. A recomputation on the other GPU was also
  identical.
- The report regenerates to the same SHA-256.
- Nothing trained, no simulator ran, D_test was not read, and no paid compute was used.

## Scope

The inventory ([`evidence/reweight/inventory.json`](evidence/reweight/inventory.json)) finds **no historical artifact
with full CCIL evidence**:

| source | label class | what the labels carry |
|---|---|---|
| Pendulum pumping control | oracle | demonstrator relabels at BC-visited states; no CCIL selection, weight, certificate or error ever existed |
| Pendulum capture control | oracle | the same |
| PushT v2.2 | partial | CCIL selection by acceptance alone (`admit: accepted`, `data_only`, dataset class `uncertified`): a mask and a recorded weight of 1.0, but no certificates and no finite error |
| PushT surface, Stages 1 and 2 | partial | the same, one pool per r |

Stage A is therefore scoped to **uncertified, acceptance-selected CCIL labels against oracle relabels**. It can test
whether the utility statistic has the intended direction on known helpful and harmful supervision. It cannot test the
trust constraint of #40, because no row in these artifacts is certified. That question is recorded as untestable from
existing artifacts.

The missing fields are a result. No certificate is regenerated, no weight is inferred from configuration, and no
row is assigned a classification it does not have. An oracle relabel is trusted for this retrospective check only,
and it is never described as CCIL-certified.

## The statistic

For a learner checkpoint θ, a meta-validation set V of real demonstration rows, and a synthetic row i with recorded
weight c_i and per-row training loss ℓ_i:

```
g_i = < ∇θ L_V(θ), ∇θ ℓ_i(θ) >         (signed alignment)
u_i = c_i · max(0, g_i)                  (utility, #40's first-order statistic without the common factor η)
```

- **Computation.** g_i is the directional derivative of ℓ_i(θ + t·∇L_V) at t = 0. It is exact per row and comes from
  one forward-mode pass (`torch.func.jvp`) along the fixed direction ∇L_V. No per-example gradient is materialized.
  The unit of weighting is the row.
- **Losses.** Each is the learner's own training loss, in its own units:
  - Pendulum: `vector_bc`'s squared action error, in its min/max-scaled action units, with the policy's frozen
    scalers;
  - PushT: IBC's InfoNCE, as the frozen selection scorer computes it, with negatives fixed per row from seed 0.
- **Numerics.** Strict FP32 (TF32 off), with the network in evaluation mode (dropout off).
- **c_i** is the weight the artifact records. That is 1.0 on every selected PushT row and 1.0 on every oracle relabel,
  the weight it was trained with. It is never changed.

## Checkpoints and roles

θ is the **BC policy's selected checkpoint** for the case and seed: the learner before any synthetic row. The meta-
validation set V is real demonstration data that the unit at θ never trained on and that never selected its
checkpoint:

| case | θ | V | synthetic rows | comparator rows |
|---|---|---|---|---|
| Pendulum capture (seeds 3311–3320) | BC, per seed | the 90 corrected-demonstrator episodes outside the seed's 10-episode subset | the seed's collection relabels | the seed's subset demonstration rows (duplication's draw population) |
| Pendulum pumping (seeds 3301–3310) | the same | the same | the same | the same |
| PushT surface r = 0.1, 0.25, 0.5 | BC at r, Stage 1 seed 1 (and Stage 2 seed 2 at r = 0.25) | the D_X episodes outside the subset at r | the pool's selected labels that trained (the `labels.npz` rows) | each label's source row (`src_row`), once per label: the duplication view |

- **Not in the directional check:** PushT r = 1 and v2.2. Every D_X episode trained them, and D_V selected their
  checkpoints, so no clean V exists. They are reported with V = D_V and marked as selection-contaminated.
- **Final test.** D_test (seeds 5000–5099) and every Pendulum final start set are not read.
- **Joins.** v2.2's CCIL mask covers all 206 episodes (100,871 rows). The rows that trained are the arm's recorded
  source index (80,604), and Stage A joins on that, never on the mask.

## Directional checks (frozen)

For each case and seed, D = mean u_i over the synthetic rows − mean u over the comparator rows (with c = 1 for
comparator rows).

- **F1, helpful (capture):** D > 0 on at least 8 of the 10 seeds. The capture relabels improved closed-loop
  behaviour.
- **F2, harmful (PushT r = 0.1 and r = 0.5, seed 1):** D ≤ 0 for both. At those r, CCIL was below duplication at
  every α.
- **Stage A passes** exactly when F1 and F2 both hold. Only then may a Stage B protocol PR be opened. Otherwise the
  investigation ends inside #40.

**Reported, not checked:**
- **Pumping.** Its relabels reduced held-out action error but did not beat duplication in closed loop. A
  validation-loss statistic is expected to rate them positively. If it does, that is the statistic's stated limit:
  it measures held-out action fit, not closed-loop value.
- **PushT r = 0.25,** seeds 1 and 2, whose Stage 1 ordering did not replicate.
- **PushT r = 1 and v2.2.**

## Report schema

One artifact per run, `ilsb.reweight.stage_a 1`:
- **identities:**
  - the inventory's SHA-256;
  - per case and seed: the checkpoint files, the label and selection files, the split or subset, and V's episode list;
  - the code commit and environment.
- **per row** (an `.npz` per case and seed, with its SHA-256 in the report):
  - the source identity and the recorded CCIL fields as written: mask, weight, error (NaN where missing) and the
    dataset-level evidence class;
  - g_i and u_i.
- **aggregates** per case, seed and cohort (r, or the Pendulum control):
  - the counts and fractions of positive (g > 0), zero (g ≤ 0) and conflicting rows. A source transition is
    conflicting when its labels have both signs; PushT has K = 4 labels per transition;
  - the mean u, the mean g, and D.
- **mass:** for each recorded α (Pendulum 0.5 at batch 512; PushT 0.25, 0.5 and 0.8 at batch 256):
  - the requested synthetic rows per batch b;
  - the probability that a batch of b rows drawn without replacement has no positive row, C(N − P, b) / C(N, b);
  - the expected realized mass α · (1 − that probability).
- **checks:** F1 and F2 as defined above, and the decision.

A rerun from the recorded identities must reproduce the report byte for byte. Rounding-level values are not recorded
as such: g_i is stored at full precision in the `.npz`, and the report holds sums and signs.

## Hosts

siddh holds the Pendulum runs and the PushT data source. The five surface BC checkpoints Stage A uses (r = 0.1, 0.25, 0.5 and 1 on seed 1, and r = 0.25
on seed 2; 12.8 MB each) and the five surface pools are copied from Tillicum to a run directory on siddh. Each copy is verified against the inventory's
hashes before use. Nothing is copied into the repository. Stage A runs on siddh's GPUs at no cost.

## Reproducing the inventory

On each host, read-only:

    python tools/reweight_inventory.py probe --kind pendulum-control --root <run> --name pendulum-capture --study capture-control
    python tools/reweight_inventory.py probe --kind pendulum-control --root <run> --name pendulum-pumping --study positive-control
    python tools/reweight_inventory.py probe --kind pusht-v22 --root <v2.2 root> --name pusht-v22 --study v22
    python tools/reweight_inventory.py probe --kind pusht-surface --root <stage1 root> --name pusht-surface-stage1 --study stage1
    python tools/reweight_inventory.py probe --kind pusht-surface --root <stage2 root> --name pusht-surface-stage2 --study stage2

Then, in this repository:

    python tools/reweight_inventory.py combine <the five parts>

The combine step checks every part against the committed records and exits non-zero on any mismatch.
