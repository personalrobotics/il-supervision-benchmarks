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
  is trained.
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
`env.reset(seed = start)`, as lerobot-eval seeds it, and ends at success or after 300 steps. The policy is driven as
lerobot-eval drives it, through B2's driver ([`stage_b2.py`](../../tools/pusht_recovery/stage_b2.py)). Torch is seeded
with `SeedSequence([62, driver, start, chunk])` before each 16-action chunk, so any chunk can be regenerated on its
own. Every step's observation, simulator state, action and coverage, and every chunk's outputs, are saved.

Each driver's average maximum coverage and success are reported next to B1's, as a measurement of the simulator's
cross-process variability (#53), not a gate. Failure labels come from the rerun alone.

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
- **Early window:** the decision points before step 80, the first 10 chunks. A failed episode runs all 300 steps and a
  successful one stops early, so a whole-episode mean would confound disagreement with duration and with stalled
  states. The window also asks whether disagreement is high before a failure is settled.
- **Episode score:** the mean U_epi over the early window.
- **Discrimination:** the rank AUC of failed over successful episodes (Mann–Whitney, ties counted half), per driver.

**Pass:** on each driver separately, the episode-level AUC of U_epi is at least 0.70 and exceeds the AUC of U_ale under
the same aggregation. The second condition requires the signal to come from disagreement between members, not from
ordinary sampling spread. Stage A refuses as uninformative if a driver has fewer than 10 failed or 10 successful
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
  - a rerun is byte-identical;
  - the audit passes;
  - injected faults are refused: a changed probe output, a changed label, a moved decision point.

**Compute:** siddh's two GPUs, one driver each, unpaid. The ceiling is 6 GPU-hours.

Starts 2000–2199, which no earlier stage uses, are reserved for Stage B.
