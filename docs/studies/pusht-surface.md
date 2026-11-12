# PushT surface: CCIL against duplication over real data and synthetic exposure (protocol)

This is the protocol of [#5](https://github.com/personalrobotics/il-supervision-benchmarks/issues/5). It is frozen
before any study policy trains. The PushT v2 result ([pusht.md](pusht.md)) is one measured operating point: all the
real data, about 80% synthetic exposure. That run is not repeated.

The machine-readable protocol is [`surface.json`](../../tools/pusht_v2/surface.json) (Stage 1) and
[`surface.yaml`](../../tools/pusht_v2/surface.yaml) (CCIL). The harness implementation is
[`src/ilsb/studies/pusht_surface.py`](../../src/ilsb/studies/pusht_surface.py); the learner is
[`tools/pusht_v2/surface_learner`](../../tools/pusht_v2/surface_learner).

## Question

**Primary:** at a fixed real-data fraction r, is there a synthetic minibatch fraction α at which CCIL beats
source-matched duplication on closed-loop coverage?

**Also asked:**
- Does the useful dose depend on r?
- Is CCIL most useful where the dynamics model is adequate but BC has not saturated?
- When a large dose hurts, does a smaller one help?

No monotonicity in r or α is assumed.

The estimand is the paired difference in average maximum coverage between CCIL and duplication. CCIL − BC and
duplication − BC are reported alongside.

## Data roles

All rows are those of `lerobot/pusht_keypoints` at revision `ace8c161`, derived exactly as in v2.2. Each episode's
first frame is dropped, and each position feature becomes the two-frame history [x_{t−1}, x_t]. The rows are built by
[`source.py`](../../tools/pusht_v2/source.py) and fixed by their content digest `333a9b17…`. They are identical to
v2.2's derived dataset.

| role | definition | size |
|---|---|---|
| D_X | v2.2's 165 training episodes | 20,335 frames |
| D_V | v2.2's 41 held-out episodes: checkpoint selection and offline diagnostics only | 5,109 frames |
| D_dev | `lerobot-eval` starts, seeds 1000–1099 (v2.2's) | 100 |
| D_test | starts with seeds 5000–5099, never evaluated before Stage 3 | 100 |

**Subsets.** For r ∈ {0.10, 0.25, 0.50, 1.00}, the subset is the first round-half-up(r · 165) episodes of one
permutation of D_X, drawn from `SeedSequence([subset_seed, 9])`. That gives 17, 41, 83 and 165 episodes, nested in r.
Development uses subset seed 0. A Stage 3 cell with r < 1 uses subset seed 1.

At each r, the same subset fits the dynamics, generates the labels and trains every arm. Nothing at any r reads D_V,
D_dev or D_test, except selection, which reads D_V.

## Labels (fixed recipe, not a study axis)

The recipe is PushT v2's: one-step CCIL in the T's rigid pose history, decoded back to keypoints.
- The dynamics model is a spectral residual MLP with the history shift structured, fitted on the subset's episodes
  alone.
- Generation is `noisy_action_fe`, with K = 4 proposals per transition, noise of 0.1 training-action standard
  deviations, and M = 1.
- Every accepted label is curated with weight 1.

The CCIL commit is the one ilsb pins (`a995ff9`).

**Checks.** Each subset's pool is generated once, before any policy trains. It fails the run on any of:
- a label whose residual exceeds 10⁻⁴;
- a label whose history invariant or rigidity does not hold in both slots;
- a decode drift above the round-trip tolerance;
- a proposal count other than exactly 4 × the subset's transitions;
- a label sourced outside the subset;
- a label whose current frame equals a held-out frame (checked by content);
- a usable fraction below 0.5;
- a median previous-slot displacement below 0.5 px, meaning the labels do not differ meaningfully from their
  duplication view.

**Reported, not gated:**
- the yield;
- the residuals;
- the previous-slot displacement from the source, in px;
- the distance to the nearest real row of the subset;
- the fraction of labels outside the frozen normalization range;
- the dynamics model's fit;
- each unit's reuse of its unique rows.

## Arms and exposure

For a minibatch of B = 256 rows, the learner draws exactly b_s = round-half-up(α · B) synthetic rows and B − b_s real
rows.

| α | 0 | 0.25 | 0.50 | 0.80 |
|---|---|---|---|---|
| synthetic per batch | 0 | 64 | 128 | 205 |
| realized fraction | 0 | 0.25 | 0.50 | 0.80078 |

**Sampling.** The real and synthetic rows are two streams. Each is drawn without replacement within an epoch, from its
own permutation `SeedSequence([seed, stream, epoch])`, so a step's rows are a function of (seed, step) alone.

**Arms**, at each r:
- **BC** (α = 0): real rows only. It is trained once per learner seed, and every dose is compared with it.
- **Duplication** (α > 0): the synthetic slots hold, in order, each CCIL label's source demonstration row.
- **CCIL** (α > 0): the same slots hold the labels.

Because duplication and CCIL use the same stream and seed, every one of their batches draws the same source rows. The
two arms differ only in the synthetic rows' content.

**Normalization.** Every arm at r uses one min/max computed from that subset's real rows only, then frozen. The labels
never influence the baseline.

**Audit.** The audit regenerates every unit's index stream from the rule, with an implementation independent of the
learner's. It reconciles that stream with the recorded digests and counts for every 2,500-step segment. It then checks
that duplication and CCIL drew identical streams.

## Learner and checkpoint selection

**Learner.** The IBC policy of v2.2's gate, unchanged:
- an energy MLP with 4 × 1024 layers;
- InfoNCE against 1024 uniform negatives;
- min/max normalization;
- AdamW at 10⁻⁴ with warmup 500;
- compiled bf16 autocast over fp32 weights;
- no clipping.

**Training step.** The step is `lerobot-train`'s own: LeRobot 0.6.1's `update_policy`, with the policy, processors,
optimizer, scheduler and checkpoints all built by LeRobot's factories. Only the minibatch source differs.
`lerobot-train` itself can provide neither fixed-α batches nor an exact resumable continuation. Fed `lerobot-train`'s
own sampler order, the learner reproduces `lerobot-train`'s weights byte for byte; this check is part of the smoke
evidence below.

**Schedule.** The cosine decays to 0 over a fixed 80k-step horizon for every unit, so extending a unit continues it
exactly. Checkpoints are saved every 2,500 steps, each with its training state.

**Selection.** At each boundary, the checkpoint with the least mean held-out InfoNCE on D_V is selected; ties go to the
earlier step. The negatives are drawn once, from seed 0, and shared by every checkpoint and arm. Simulator returns
never select a checkpoint or extend a unit.

**Extension.** Train to 20k:
- if the 20k checkpoint is selected, extend to 40k;
- if the 40k checkpoint is then selected, extend to 80k;
- if the 80k checkpoint is selected, the fit is incomplete and its cell is ineligible.

The rule is the same for every arm, so realized steps may differ between units. Compute is provenance, not a study
axis.

**Recomputation.** A killed unit resumes from its newest checkpoint with an identical continuation; the smoke evidence
below verifies this. The audit recomputes every selection and every extension from the saved validation histories. It
also re-scores each selected checkpoint from disk.

## Evaluation

The evaluator is unchanged `lerobot-eval` on the history environment, with 100 episodes from the declared starts.

**Metric.** Average maximum coverage. Success (coverage ≥ 0.95) is secondary.

**References on D_dev**, from v2.2:
- holding still scores 0.0620;
- replaying the demonstrations scores 0.9464.

The evaluation audit recomputes every score from its episodes. It re-evaluates one unit, and that unit's episodes must
repeat within 10⁻¹² (max reward) and 10⁻⁹ (sum reward).

## Stages and stop rules

The study stops at the first rule that applies.

**Stage 1, the one-seed development surface.**
- **Design:** learner seed 1, subset seed 0, D_dev; every r, with BC once and duplication and CCIL at each α > 0.
  That is 28 units. Nothing is reused: no v2.2 unit matches this loop, schedule, sampler or normalization.
- **Learner competence:** if BC at r = 1 does not reach 0.5 of the hold-to-demonstrator gap, the surface is
  uninformative and the study stops.
- **Eligibility:** a cell (r, α) is eligible when CCIL > duplication and CCIL > BC, and all three units' fits are
  complete. If no cell is eligible, the study stops and the surface is published as a negative development result.
- **Ranking:** eligible cells are ranked by CCIL − duplication. Effects within 0.005 of the best count as
  indistinguishable and are ordered by lower α, then lower r.

**Stage 2, replication.**
- **Design:** only the top two cells, on learner seed 2, with BC at their r.
- **Eligibility:** a cell stays eligible only if, on both development seeds, CCIL > duplication and CCIL > BC. If none
  does, the study stops.
- **Selection:** the cell with the largest mean CCIL − duplication, by the same tie rule.

**Stage 3, confirmation.**
- **Design:** the selected cell with BC, duplication and CCIL, on learner seeds 11–15 and D_test. At r < 1 it uses
  subset seed 1, so its dynamics and labels are fitted anew.
- **Signal rule:** a corrective-information signal requires a positive mean CCIL − duplication effect, and CCIL above
  duplication on at least 4 of the 5 paired seeds.
- **Reported whatever the outcome:**
  - the paired interval (`ilsb.analysis.paired`, the union of BCa and t);
  - all five values;
  - CCIL − BC and duplication − BC;
  - success;
  - the selections;
  - the realized exposure;
  - the label diagnostics.

No second confirmatory cell is tried.

Stage 1 is descriptive and may choose what runs next. Only Stage 3 is confirmatory. A positive result is scoped to this
PushT regime, this IBC learner and this one-step recipe. A null or negative result is equally complete.

## Compute

**Hosts.** Policy training runs on Tillicum (H200), as a Slurm array of `ilsb unit` tasks with one GPU each. Everything
else of a run, CCIL included, runs in jobs on the same cluster, because the store scope names the execution
environment. siddh hosts the smoke and debugging.

**Budget.** The declared ceiling for Stage 1 is 24 GPU-hours, about $21.60 at $0.90 per H200-hour. It was set before
any measurement. One off-study timing unit fixes the projection:
- learner seed 99, r = 1, α = 0.8;
- trained 20k steps and never evaluated or reused.

If the projection exceeds the ceiling, the protocol is amended before Stage 1 starts.

| measurement | value |
|---|---|
| timing unit, 20k steps | pending |
| projected Stage 1 GPU-hours | pending |

## Pre-run evidence

[`evidence/pusht-surface/smoke.json`](evidence/pusht-surface/smoke.json) comes from
[`surface_smoke.py`](../../tools/pusht_v2/surface_smoke.py) on siddh. It records no outcomes. It covers:
- the whole chain at r = 0.1 and α = 0.8, with BC, duplication and CCIL, from preflight through the report;
- the pool's checks;
- the extension rule;
- exposure reconciliation and pairing;
- re-scored selections;
- the failure path: a learner killed with SIGKILL stops the run promptly, names its unit and leaves no entry for it,
  and the rerun resumes to the weights of the uninterrupted run;
- the `lerobot-train` equivalence.

To reproduce it, on a host with two GPUs and the learner environment:

    ILSB_LEROBOT_PYTHON=<env>/bin/python ILSB_PUSHT_DATASET=<pusht_keypoints at ace8c161> \
        uv run python tools/pusht_v2/surface_smoke.py --work <new dir>

## Running

The learner environment is [`lerobot-env.txt`](../../tools/pusht_v2/surface_learner/lerobot-env.txt), with the env
plugin, the IBC plugin and the learner of this tree installed. The harness checks the installed plugins and learner by
content.

On Tillicum, [`slurm/setup.sh`](../../tools/pusht_v2/slurm/setup.sh) builds both environments at the protocol commit on
the login node. [`slurm/timing.sbatch`](../../tools/pusht_v2/slurm/timing.sbatch) is the timing unit.
