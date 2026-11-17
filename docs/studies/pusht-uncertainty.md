# PushT policy-ensemble uncertainty (protocol)

This is the protocol of [#62](https://github.com/personalrobotics/il-supervision-benchmarks/issues/62). Stage A was
frozen before any uncertainty score was computed; its machine-readable form is
[`tools/pusht_uncertainty/stage-a.json`](../../tools/pusht_uncertainty/stage-a.json).

**Question.** Does disagreement among competent Diffusion Policies trained on the same demonstrations predict where a
policy fails, so that it could pick which recovery windows to train on without new learner interaction?

**Why.** Static label geometry did not define a correction-specific domain ([pusht-refiner.md](pusht-refiner.md)), and
rollouts from demonstration prefixes almost never reproduced the policy's failures
([pusht-recovery.md](pusht-recovery.md#b2-result)). The next data-only signal comes from the learner itself.

## Stages

- **Stage A** (this protocol): the two existing r = 0.25 policies, no training. It decides only whether a third member
  is trained. **It failed, and the study stopped** ([result](#stage-a-result)).
- **Stage B**: a third member, and the Stage A rule verified on fresh starts.
- **Stage C**: uncertainty-selected oracle recovery against the base, matched self-distillation and uniform selection.

Each later stage runs only if the one before passes, and freezes its own rules in its own protocol PR. A failed stage
is terminal and published. If the uncertainty signal fails, the next method collects learner rollouts interactively.
D_test (starts 5000–5099) is not read in this issue.

## Stage A

**Status of the evidence.** Stage A is exploratory. Its starts are the development starts, whose outcomes B1 already
reported, so it can only justify Stage B, not support a claim.

### Inputs and inventory

The members are B1's two competent bases ([stage-b-base.json](evidence/pusht-recovery/stage-b-base.json)): seed 1
(model `b4e025aa…`) and seed 2 (model `fdc8cf40…`). Both were trained on the same 41-episode r = 0.25 subset, and their
normalization stats are byte-identical, so their actions compare in one normalized space. They are not changed.

B1's records keep each episode's summed and maximum reward, success and start seed, and videos of the first 10
episodes. They keep no observations, states, actions or policy outputs. Stage A therefore makes **one read-only
evaluation rerun** from the pinned checkpoints. Nothing is reconstructed from summaries or videos.

### The rerun

Each member drives its own rollouts on the development starts 1000–1099. An episode starts at
`env.reset(seed = start)`, as lerobot-eval seeds it. It runs at least 80 steps, even past success, and then ends at
success or after 300 steps. The policy is driven as
lerobot-eval drives it, through B2's driver ([`stage_b2.py`](../../tools/pusht_recovery/stage_b2.py)). Torch is seeded
with `SeedSequence([62, driver, start, chunk])` before each 16-action chunk, so any chunk can be regenerated on its
own. Every step's observation, simulator state, action and coverage, and every chunk's outputs, are saved.

Each driver's average maximum coverage and success are reported next to B1's, as a measurement of the simulator's
cross-process variability (#53), not a gate. Failure labels come from the rerun alone. Both score an episode up to its
first success, where lerobot-eval ends it; steps run past success only supply decision points.

### Precision

*Amended before any development outcome, in review of the Stage A code.* The policies run in LeRobot's numerical
mode: TF32 cuDNN convolutions and strict FP32 matmuls. These are PyTorch's defaults, which lerobot-train and
lerobot-eval leave unchanged, so B1's checkpoints were trained and evaluated this way. B2's driver forced strict FP32
convolutions instead; that mode is atypical for these checkpoints, and it projected about 15 GPU-hours for Stage A.
Deterministic algorithms, the shared noise, the 16 samples and the full audit are unchanged.

Before the amendment, member 1's sampling was compared in both modes, at the reset observations of starts 3401 and
3402, with batches of 1 and 16:
- **Action difference:** at most 2.5e-4 in normalized actions (range [-1, 1]); median 4.3e-5.
- **Speed:** 0.35 s against 1.15 s per call at batch 1, and 0.40 s against 1.51 s at batch 16.
- **Repeatability:** both modes repeated bit for bit across processes.

**Bitwise reproducibility is required within the pinned environment, not across hardware.** The environment is siddh's
NVIDIA RTX PRO 6000 Blackwell Max-Q, driver 595.99.02, CUDA 13.0, PyTorch 2.11.0+cu130, cuDNN 9.19.0 and LeRobot 0.6.1.
Each run records its environment, and the audit refuses to compare across another.

### Uncertainty

Diffusion Policy is stochastic by design, so variation among one model's samples is not epistemic uncertainty. At each
chunk boundary of a driver's rollout (steps 0, 8, 16, …), both members generate S = 16 chunks from the driver's
two-frame observation. Before each member's batch, torch is seeded with `SeedSequence([62, 0, driver, start, chunk])`,
so both members denoise the same 16 noise draws. These probe chunks are never executed.

Actions are compared in the normalized space, over the executed slots (horizon indices 1–8) and both action
dimensions, with population variances. For members m and samples s:

- **Between members:** U_epi(x) = mean over s of Var_m[a_m(x; ε_s)].
- **Within a member:** U_ale(x) = mean over m of Var_s[a_m(x; ε_s)].

Each is averaged over the 8 slots and 2 dimensions. Every decision point's U_epi, U_ale and their ratio are recorded.

### Labels, aggregation and decision

- **Failure:** the driver's rerun episode does not reach gym_pusht success (coverage > 0.95) within 300 steps.
- **Early window:** the decision points at steps 0, 8, …, 72: exactly 10 per episode, since every episode runs at least
  80 steps. A failed episode runs all 300 steps and a successful one stops early, so a whole-episode mean would
  confound disagreement with duration and with stalled states. The window also asks whether disagreement is high
  before a failure is settled.
- **Episode score:** the mean U_epi over the early window.
- **Discrimination:** the rank AUC of failed over successful episodes (Mann–Whitney, ties counted half), per driver.

**Pass:** on each driver separately, the episode-level AUC of U_epi is at least 0.70, and it exceeds the AUC of U_ale
under the same aggregation by at least 0.05. The margin requires the signal to come from disagreement between members,
not from ordinary sampling spread. It is about the sampling standard error of an AUC near 0.70 with 100 episodes, so a
nominal advantage does not pass. Stage A refuses as uninformative if a driver has fewer than 10 failed or 10 successful
episodes. **If Stage A fails, it is published and the study stops.** No other formula, noise schedule, window, horizon or
threshold is tried.

Reported descriptively, not as gates:
- U_epi and U_ale by outcome;
- the state-level AUC in the early window;
- U_epi by chunk index over the whole episode, failed against successful;
- U_epi in successful episodes by steps remaining before success;
- the whole-episode mean's AUC.

### Audit and smoke

- **Audit:** in a fresh process, every driver chunk and probe batch is regenerated bit for bit from the saved
  observations and seeds. U_epi, U_ale, the labels, the aggregation and every AUC are recomputed and must be identical.
  An independent replay of the saved actions is reported as a measurement, not an equality gate (#50).
- **Smoke:** before the run, on starts 3401–3410, which no study uses:
  - a member probed against itself with the shared noise gives U_epi = 0 exactly;
  - with the probe enabled and disabled, the driver's executed chunks and saved trajectory are byte-identical;
  - a rerun is byte-identical;
  - the audit passes;
  - injected faults are refused: a changed probe output, a changed label, a moved decision point.

### Implementation

[`tools/pusht_uncertainty/probe.py`](../../tools/pusht_uncertainty/probe.py) reruns one driver with the probe and writes
its trajectories, probe outputs and statistics. Its `audit` mode is the fresh-process audit, and `report` combines the
two drivers' runs into the decision.
- LeRobot's sampler returns only the executed slots (horizon indices 1–8), so those are what the probe saves.
- Each member's 16 samples are one batch.
- Recorded reward is rounded to 1e-9 like coverage, since it is coverage / 0.95 and inherits shapely's last-digit drift.

[`tools/pusht_uncertainty/smoke_probe.py`](../../tools/pusht_uncertainty/smoke_probe.py) ran on starts 3401–3410 from a
clean clone of `0e757e6`, with driver 1 and both members. Its record is
[`evidence/pusht-uncertainty/smoke.json`](evidence/pusht-uncertainty/smoke.json).
- **Rerun:** two runs wrote byte-identical records.
- **Probe invariance:** with the probe disabled, the trajectories were identical.
- **Identity control:** member 1 against a separately loaded copy of itself gave U_epi = 0 exactly at all 326 decision
  points.
- **Audit:** every chunk and probe batch regenerated bit for bit; the recomputed coverage agreed within 5e-10; the
  simulator replay matched exactly.
- **Faults:** each was refused for the right reason: a changed probe output, a flipped label, a moved decision point.

**Compute:** siddh's two GPUs, one driver each, unpaid. The ceiling is 6 GPU-hours.
- **Measured:** the smoke's probe run took 390.7 s for 326 decision points (1.2 s each), and its audit as long.
- **Projected:** about 4.4 GPU-hours for both drivers' runs and audits at B1's failure rates, and 5.1 if every episode
  fails.
- **Strict FP32:** the same probe run took 1,466 s, 3.7 times as long.

Starts 2000–2199, which no earlier stage uses, are reserved for Stage B.

### Stage A result

**Stop: the frozen separation rule fails on both drivers.** Ran 2026-11-17 from a clean clone of `5432f8e`. Record:
[`evidence/pusht-uncertainty/stage-a.json`](evidence/pusht-uncertainty/stage-a.json).

| Driver | Failed / successful | AUC U_epi | AUC U_ale | Margin | Pass |
|---|---|---|---|---|---|
| 1 | 76 / 24 | 0.48 | 0.44 | 0.05 | no |
| 2 | 75 / 25 | 0.58 | 0.45 | 0.12 | no |

- **The failure is not caused by imbalance or an uninformative set.** Both drivers have far more than 10 episodes of
  each class.
- **Two-policy disagreement does not reliably identify consequential failures.** Driver 1 is at chance and driver 2 is
  well below 0.70.
- **The members disagree, but not about failure.** In the early window, U_epi is a median of 19 and 17 times U_ale.
  Its median is the same for failed and successful episodes on driver 1 (3.5e-4 each), and 3.9e-4 against 2.3e-4 on
  driver 2. The policies can agree while both are wrong, and disagree during benign variation.
- **Some information is transient.** The state-level AUC in the early window is 0.60 and 0.64. That is not enough to
  select labels without a post-hoc redesign, which the protocol forbids.
- **The whole-episode AUC (0.62 and 0.68) does not count.** It is confounded with duration, which is why the early
  window was frozen.

**Audit:** in a fresh process for each driver, every chunk and probe batch regenerated bit for bit, and the statistics
reproduced. Coverage recomputed from the saved states agreed within 5e-10, and the simulator replay matched exactly
([driver 1](evidence/pusht-uncertainty/stage-a-audit-d1.json), [driver 2](evidence/pusht-uncertainty/stage-a-audit-d2.json)).

**Against B1 (a measurement, not a gate):**

| Driver | Rerun coverage / success | B1 coverage / success | Episodes with the other outcome |
|---|---|---|---|
| 1 | 0.669 / 24% | 0.661 / 34% | 30 |
| 2 | 0.638 / 25% | 0.654 / 25% | 20 |

Average coverage agrees, but the success of individual episodes often differs between processes. B1's own repeat
differed on 9 episodes (#53). Labels at the success threshold are therefore partly noise, which limits any predictor's
AUC. It does not explain driver 1's AUC at chance.

**Compute:** 4.5 GPU-hours on siddh's two GPUs, within the 6 GPU-hour ceiling. Each probe run took about 68
minutes, and each audit about 67.

**Next.** No third member is trained, and no other uncertainty summary is tried. The next method observes learner
failure directly: roll the deployed learner out from environment starts, find the states or short windows that precede
actual failure, and ask whether CCIL can provide valid recovery supervision there.
