# PushT recovery windows for Diffusion Policy (protocol)

This is the protocol of [#52](https://github.com/personalrobotics/il-supervision-benchmarks/issues/52). Stage A was
frozen before any recovery window was built from the study's proposals; its machine-readable form is
[`tools/pusht_recovery/stage-a.json`](../../tools/pusht_recovery/stage-a.json). The revised Stage B
([below](#stage-b-revised-protocol)) was frozen before its base policy trained; its machine-readable form is
[`tools/pusht_recovery/stage-b.json`](../../tools/pusht_recovery/stage-b.json).

**Question.** Does a one-step CCIL correction help a competent temporal learner when it is written as a physically
consistent recovery window, a history and an action sequence, rather than as an isolated state-action row?

**Status of the evidence.** This is an exploratory mechanism study. Its cell (r = 0.25, α = 0.25) and its oracle
proposals come from studies whose outcomes were seen ([pusht-oracle.md](pusht-oracle.md)), so no result here is
confirmatory evidence for CCIL.

## Stages

- **Stage A** (this protocol) builds the windows and audits them. No policy is trained. If the yield below is not
  met, Stage A refuses, the refusal is published, and the study stops. The window and replay rules are not relaxed
  after the result.
- **Stage B**, as first planned, would have trained on Stage A's windows. It was superseded before its protocol was
  frozen: Stage A's sources were not chosen from where a learner deviates. The revised Stage B builds oracle recovery
  windows at a competent policy's own deviation states, and compares them with self-distillation at the same states.
  Stage A's 1,865 windows are the construction's feasibility evidence, not a training pool. The revised Stage B
  stopped at B2 ([result](#b2-result)): too few deviation states for the B3 yield.
- **Stage C**, learned-dynamics recovery at the same states, runs only if Stage B passes.

D_test (seeds 5000–5099) is not read by any stage before the last.

## Inputs

- **Data:** `lerobot/pusht_keypoints` at `ace8c161`, as the PushT studies' source rows (content `333a9b17…`). Stage A
  reads the r = 0.25 development subset only: 41 training episodes from subset seed 0, 5,040 source rows.
- **Proposals:** #43's oracle Stage A run (content `b31c2201…`) over the store entry `c30c0f98…`'s 19,996 proposals
  (candidates `6f8c9c97…`). Its 12,743 accepted proposals are the population. Each carries a proposal action ã at a
  source frame t and the oracle's first-order decoded previous slot.
- **Simulator:** gym_pusht 0.1.8 with pymunk 6.11.1. Its states repeat bit for bit across processes; only the
  coverage reward can differ in its last digit, and Stage A does not read the reward.

## The recovery window

A window anchored at source frame t has Diffusion Policy's shape: two observations and 16 action slots, of which
slots 1–8 are executed.

- **Observations** [o′_{t−1}, o_t]. o_t is the source frame's own observation. o′_{t−1} is the agent position and T
  keypoints of a predecessor state P.
- **Actions.** Slot 0 is a preceding action b. Slot 1 is the correction ã. Slot k ≥ 2 is the demonstrated action of
  frame t + k − 1, or the episode's last action beyond its end, as LeRobot pads.
- **One simulated chain.** P is loaded into a fresh simulator, and then, in that same live space, b, ã and the
  demonstrated actions are stepped in turn. o′_{t−1} is the chain's first state, o_t is the recorded frame the chain
  reaches within 0.05 px, and every supervised action is one the chain applied. The oracle's decoded previous slot is
  not used as an observation; it is only where the solve starts.
- **Solve.** The unknowns are P's positions and velocities and b, 12 values in pixels. The residuals are the chain's
  positions after b against frame t and after ã against frame t + 1, 10 values. Gauss–Newton takes minimum-norm
  least-squares steps from P = (the decoded previous slot, the replayed source velocities at t − 1) and b = the
  demonstrated action of frame t − 1. The residual is rank-deficient by design: when nothing touches the T its
  velocity dies within one control step, so its position after b and after ã is the same equation. Converged means
  both distances are within 0.05 px, the oracle's tolerance.
- **Loss mask.** Slots 0 and 1 are supervised. Slot k ≥ 2 is supervised while the chain's positions at frames
  t + 1 … t + k − 1, where the demonstrated actions are applied, are each within 1.0 px of the recorded frames and the
  frame lies in the episode. The supervised slots are a prefix. Masked slots keep the demonstrated values, because the
  denoising network still reads them.
- **Provenance.** Each window records its episode, anchor frame and proposal, and each action slot records its source
  frame, or that it is the synthetic history action or the correction.

### Acceptance, in this order

| check | rule |
|---|---|
| predecessor | frame t − 1 is a source row |
| anchor | frame t is a LeRobot anchor (t ≤ T − 1 − 7) |
| solve | the iterate stays finite, the Jacobian is finite and non-zero, and Gauss–Newton converges within 20 iterations, with up to 8 halvings per step |
| finite | every chain state is finite |
| action bounds | every supervised action (b, ã and the supervised demonstrated actions) is inside [0, 512]² |
| b change | ‖b − a_{t−1}‖ ≤ the 99.9th percentile of the subset's consecutive action changes |
| speed | P's agent speed, T speed and ℓ\|ω\| each ≤ its maximum over the subset's replay-verified states |
| envelope | P and every chain state where a supervised action is applied lie within #43's penetration (22.98 px) and extent ([7.49, 523.06] px) envelope |
| continuation | the executed chunk, slots 1–8, is supervised |

One window is kept per source anchor: of its accepted proposals, the one picked by
`default_rng(SeedSequence([52, source row]))`.

## Yield and decision

Stage B is planned at r = 0.25 and α = 0.25, where α is the fraction of each epoch's training windows that are
synthetic. The subset has n_real = 4,794 real Diffusion Policy windows (`ilsb.temporal`), so Stage B needs
n_synthetic = round(α / (1 − α) · n_real) = 1,598 windows.

**Stage A passes when it keeps at least 1,598 windows, each from a distinct source anchor, from at least 30 of the
41 episodes. Otherwise it refuses and the study stops.**

Real windows supervise all 16 slots, padded ones included (LeRobot's default for Diffusion Policy); a recovery window
supervises only its mask's prefix. The record states both masses, so Stage B can match the duplication arm's
supervised mass to the recovery windows' exactly.

## Audit and record

- **Record:** `docs/studies/evidence/pusht-recovery/stage-a.json`. It holds the counts by reason, the selected windows
  per episode, the history, rejoin and continuation residuals, the supervised-slot distribution, the exact supervised
  loss mass against the real windows', novelty (o′_{t−1} against the source frame t − 1, against the oracle's decoded
  slot, and against the nearest real history), the limits, the input and output digests, and the commit.
- **Windows:** `windows.npz`, kept by content digest on siddh. Stage B consumes it.
- **Audit:** [`tools/pusht_recovery/audit.py`](../../tools/pusht_recovery/audit.py) runs in a fresh process. It
  recomputes the limits and the window counts from the inputs; replays every accepted proposal's chain and requires it
  bit for bit; recomputes its mask and every check; recomputes the selection; builds each selected window's slots
  independently (source frames from `ilsb.temporal`'s offsets, actions from the chain and the source rows); refuses
  any observation equal to a held-out (D_V) row; and recomputes the window digests and the decision. It does not
  re-solve rejected proposals; the byte-identical second run covers them.
- **Reproduction:** two clean runs of Stage A must write a byte-identical record.

## Stage A result

**Stage A passed: 1,865 windows from distinct source anchors in 37 of the 41 episodes, against the required 1,598
from 30.** Stage B may be proposed in its own protocol PR. The record is
[`evidence/pusht-recovery/stage-a.json`](evidence/pusht-recovery/stage-a.json), from commit `4696cb0` on siddh.

| proposals | count |
|---|---|
| oracle-accepted population | 12,743 |
| accepted as windows | 2,751 |
| short continuation (slots 1–8 not all within 1 px) | 4,777 |
| solve did not converge | 4,424 |
| anchor in LeRobot's dropped frames | 709 |
| speed / action bounds / b change / penetration / extent | 42 / 25 / 7 / 6 / 2 |

- **Windows.** 1,689 of the 1,865 selected windows supervise all 16 slots; the rest supervise 9–15. They carry 58,286
  supervised action values, against 32 per real window.
- **Physical consistency.** The history and rejoin residuals are at most 0.0498 and 0.0499 px (medians below 1e-9 px), and
  the supervised continuation stays within 1.0 px of the recording (median maximum 0.66 px).
- **Novelty.** The predecessor's observation is a median 5.2 px from the source frame t − 1 (90th percentile 7.6 px)
  and 5.4 px from the nearest real history, and a median 0.36 px from the oracle's first-order decoded slot.
- **Audit and reproduction.** The fresh-process audit found no problems. A second clean clone wrote a byte-identical
  record (sha256 `c06c6387…`) and `windows.npz` (sha256 `559ade49…`, content `0778c3c2…`), kept on siddh.

## Smoke

[`tools/pusht_recovery/smoke.py`](../../tools/pusht_recovery/smoke.py) runs the construction on the first two training
episodes outside the subset, with proposals made the way #43's were. It says nothing about Stage A's yield. It
requires a byte-identical rerun, a clean fresh-process audit, and a refusal, for the right reason, of each injected
fault: an action out of bounds, a predecessor inside the T, a demonstrated slot from another episode, a held-out
observation, a wrong loss mask, a wrong source frame, a wrong selection, a wrong limit, a wrong window count and an
altered source. Its record is [`evidence/pusht-recovery/smoke.json`](evidence/pusht-recovery/smoke.json).

## Reproduce

```
L=$ILSB_LEROBOT_PYTHON   # gym_pusht 0.1.8, pymunk 6.11.1
$L -I tools/pusht_recovery/smoke.py --source SOURCE.npz --work WORK
$L -I tools/pusht_recovery/stage_a.py --source SOURCE.npz --candidates CANDIDATES.npz --proposals PROPOSALS.npz --run RUN
$L -I tools/pusht_recovery/audit.py --source SOURCE.npz --candidates CANDIDATES.npz --proposals PROPOSALS.npz --run RUN
```

SOURCE.npz is built by [`tools/pusht_v2/source.py`](../../tools/pusht_v2/source.py); CANDIDATES.npz and PROPOSALS.npz
are #43's, identified by the digests above.

## Stage B (revised protocol)

**Question.** At the r = 0.25 coordinate, do oracle recovery windows built at a competent Diffusion Policy's own
deviation states improve closed-loop coverage over the frozen base, and over self-distillation at the same states? The
oracle isolates whether the state is relevant and the window representation from learned-dynamics error. Stage C
replaces the oracle with CCIL only if this passes.

This protocol was frozen before the base trained. Nothing in it was chosen from a policy return. B2–B4's code lands in
its own PRs, implementing these rules with a smoke on non-study data, before each stage runs.

### B0: the deviation geometry

[`tools/pusht_recovery/support.py`](../../tools/pusht_recovery/support.py) reads only the subset's demonstrations. Its
record is [`evidence/pusht-recovery/support.json`](evidence/pusht-recovery/support.json); two clean clones of `5cea5e9`
wrote it byte for byte. A state is Diffusion Policy's two-step observation as 18 points. The distance is the largest
per-point distance, in px.

**Rejected: cross-episode proximity.**

| px | 10% | 50% | 90% | 99% |
|---|---|---|---|---|
| a demonstration state to the nearest state of another episode | 11.3 | 32.6 | 81.6 | 137.2 |
| that pair, one executed chunk (8 steps) later | 18.4 | 55.1 | 135.6 | 229.9 |

With 41 episodes, the nearest other episode measures ordinary trajectory diversity. It identifies neither phase nor
failure. A correct demonstration diverges from its nearest-neighbour "continuation" by a median 55 px within one chunk,
so a recovery target taken from another episode would be arbitrary. A support threshold at the 99th percentile would be
137 px.

**Adopted: paired-prefix deviation.** The policy is rolled out from a demonstration's own recorded state and compared
with the same episode's continuation, which is unambiguous.
- **Threshold:** τ = 29.76 px, the 99th percentile of one demonstration step (median 8.6 px) plus the largest replay
  error (0.001 px) over the 4,071 replay-verified states.
- **Anchors:** every eighth row of each episode, from row 8 to 25 rows before its end, where the replay is verified and
  the recorded continuation makes progress. Progress means c_D > c_0: c_0 is the anchor's coverage of the goal, and
  c_D the highest coverage over the episode's rows from the anchor on, both by gym_pusht's coverage of the recorded T
  pose. That gives 398 anchors in 37 episodes; every replay-verified anchor makes progress, by a median 0.70.
- **Demonstrated progress, not success.** No subset demonstration reaches gym_pusht's success threshold: their highest
  recorded coverage runs from 0.79 to 0.90 (median 0.857). The target is therefore the progress the demonstration
  made, not the environment's 0.95 criterion.

The study targets learner rollouts that depart materially from a demonstration and fail to realize the progress that
demonstration achieved.

### B1: a competent base

The specification is [`tools/pusht_recovery/stage-b-base.json`](../../tools/pusht_recovery/stage-b-base.json), run by
the #50 harness. It is the Diffusion Policy reference ([pusht-diffusion.md](pusht-diffusion.md)) with its training
roles cut to the subset: 41 episodes, 5,081 frames and 4,794 windows; the 41 held-out episodes are for the loss
diagnostic only. The harness preflight asserts these counts. Everything else is inherited: 100k steps at batch 64, the
final checkpoint, seeds 1 and 2, `lerobot-eval` on starts 1000–1099.

**Competence:** each seed's average maximum coverage is at least 0.504, half the hold-still (0.062) to
demonstration-replay (0.946) gap. The specification enforces it per seed (`acceptance.min_per_seed`), so a strong seed
cannot cover a weak one. If either seed misses it, the study stops and the result is published. No recovery
label is built for an incompetent learner.

### B1 result

**B1 passed: both seeds are competent.** The run is from commit `836e9ac`, with a clean tree, on siddh. Every
harness stage passed, the audits included. The record is
[`evidence/pusht-recovery/stage-b-base.json`](evidence/pusht-recovery/stage-b-base.json), written by
[`reference_record.py`](../../tools/lerobot_diffusion/reference_record.py) from the run root.

| Seed | Average max coverage | Success | Selected checkpoint |
|---|---|---|---|
| 1 | 0.661 | 34% | 100k |
| 2 | 0.654 | 25% | 100k |

Both clear the 0.504 per-seed threshold, and both leave room below the full-data reference's 0.955.
- **Held-out loss:** as in the reference, it rose from 0.081 at 10k to 0.314 (seed 1) and 0.309 (seed 2) at 100k. The
  final checkpoint is selected by the frozen rule.
- **Repeat evaluation:** seed 1's repeat scored 0.660 against 0.661 and changed 9 episodes' success; the largest
  per-episode change in maximum reward was 0.067. This is a measurement of the simulator's cross-process variability,
  not a gate.
- **Compute:** each unit trained 100k steps with a peak of 5.8 GB of GPU memory.

### B2: discover deviations

For each seed and each of the 398 anchors:
1. **Restore** the anchor row's replayed full state, fresh. The policy's observation queue holds the row's two
   recorded frames.
2. **Roll out** the frozen base for the episode's remaining rows plus 50 steps, with its sampling seeded by
   `SeedSequence([52, seed, episode, row])`.
3. **Missed progress:** c_π, the rollout's highest coverage, is below c_D. Only such rollouts contribute.
4. **Deviation:** D(k) is the distance from the rollout's state k steps on to the nearest state of the same episode's
   continuation. The selected state is the first k ≥ 1 with D(k) > τ, at most one per rollout. The continuation row
   attaining D(k) is j*, and j* + 8 must be a row of the episode. A rollout that tracks the demonstration but falls
   just short never crosses τ, and so gives no state.

Every rollout's c_0, c_D, c_π and shortfall c_D − c_π are recorded, with its selected state if any. No expert or oracle
action takes part. The selected states' identities and full simulator states are written and hashed before any
recovery target is built.

**Audit.** Every rollout's states, full simulator states, observations, actions and policy outputs are saved and
hashed.
- The base's outputs are reproduced exactly on the saved observations, with the saved sampling seeds.
- Coverage (within 1e-9), D(k), missed progress and the selection are recomputed from the saved states in a fresh
  process. The selection must be identical.
- An independent simulator replay of the saved actions is reported as a measurement of variability, not an equality
  gate: PushT contact trajectories need not repeat across processes (#50).

**Implementation.** [`tools/pusht_recovery/stage_b2.py`](../../tools/pusht_recovery/stage_b2.py) runs discovery for
one seed, and its `audit` mode is the fresh-process audit.
- **Driving the policy.** The policy is driven as `lerobot-eval` drives it: `preprocess_observation`, then the
  checkpoint's own processors. The previous recorded frame is queued first, so the first chunk sees a real history.
- **Seeding.** Each 16-action chunk is generated after seeding torch with `SeedSequence([52, seed, episode, row,
  chunk])`, so any chunk can be regenerated on its own.
- **Coverage.** Coverage is kept to 1e-9, because shapely's coverage can differ in its last digits between processes.

[`tools/pusht_recovery/smoke_b2.py`](../../tools/pusht_recovery/smoke_b2.py) runs it on the first two training
episodes outside the subset, so it says nothing about B2's yield. Its record is
[`evidence/pusht-recovery/smoke-b2.json`](evidence/pusht-recovery/smoke-b2.json), from a clean clone of `dbdb046` with
B1's seed-1 base.
- **Rerun:** two runs wrote byte-identical records.
- **Audit:** every chunk regenerated bit for bit; the recomputed coverage agreed within 5e-10; the simulator replay
  matched exactly.
- **Faults:** each was refused for the right reason: a changed action, a changed coverage, a moved selection and
  altered trajectories.
- **Selection:** 10 of the 25 smoke rollouts yielded a state.

### B2 result

**B2 stops: both seeds select far fewer states than the B3 yield, so no recovery window is built, and B3 and B4 do
not run.** The run is from a clean clone of `6a64b6b` on siddh, with B1's checkpoints (seed 1 `b4e025aa…`, seed 2
`fdc8cf40…`). Each seed's discovery and fresh-process audit took about 3.7 hours on one GPU. The record is
[`evidence/pusht-recovery/stage-b2.json`](evidence/pusht-recovery/stage-b2.json); it pins the full run records by
sha256.

| Seed | Rollouts | c_π ≥ c_D | Missed progress | Selected | Episodes | Yield (100 from 20) |
|---|---|---|---|---|---|---|
| 1 | 398 | 390 | 8 | 8 | 1 | not met |
| 2 | 398 | 395 | 3 | 2 | 2 | not met |

- **Restored at a demonstration state, the base nearly always matches or beats the demonstration.** The median
  shortfall c_D − c_π is −0.087 (seed 1) and −0.075 (seed 2). The base's median c_π is 0.944 and 0.933, against a
  median c_D of 0.859, and 167 and 109 of its rollouts pass gym_pusht's 0.95 success threshold, which no
  demonstration in the subset reaches.
- **Where it misses.** Seed 1's 8 misses all come from episode 36 (anchor rows 16–96). Every one crosses τ and is
  selected, with shortfalls of 0.023–0.081. Seed 2's 3 misses are in episodes 36, 39 and 78. Two are selected; the
  third (episode 36, row 104) is not eligible, because j* + 8 lies past the episode's last row.
- **Audit:** both seeds passed. Every chunk regenerated exactly, the recomputed coverage agreed within 5e-10, the
  selection was identical, and the simulator replay of the saved actions matched to 0.0 px.

This does not show that the base is robust: it succeeds on only 34% and 25% of the development starts (B1). It shows
that the subset's demonstration prefixes are not where the base fails, so paired-prefix discovery from these anchors
cannot supply recovery states at the protocol's yield. The stop rule was frozen before B2 ran and is not relaxed.

### B3: oracle recovery windows

Each window is built with Stage A's machinery:
- **Observations:** the rollout's states at k − 1 and k. Both are physically realized.
- **Slot 0:** the action the policy applied at k − 1.
- **Slots 1–8:** recovery actions solved with Stage A's solve settings (Gauss–Newton with minimum-norm steps; 0.05 px,
  20 iterations, finite-difference step 1e-3, 8 halvings). They start from the demonstrated actions of rows
  j* … j* + 7, and must bring the agent and the T onto row j* + 8.
- **Slots 9–15:** the demonstrated actions from row j* + 8 on, supervised while the chain stays within 1.0 px.
- **Checks:** Stage A's, unchanged: action bounds, finiteness, the speed and contact envelopes, provenance, an
  independent rebuild audit, a byte-identical rerun.

**Yield:** at least 100 accepted windows per seed, from at least 20 episodes, or the study stops. There is at most one
window per anchor, so at most 398 per seed. A rejected state is never replaced.

### B4: matched comparison

All three arms start from each seed's byte-identical base checkpoint:
1. **Base:** the checkpoint itself.
2. **Self-distillation:** the windows' observations, with the base's own sampled chunk as the target (seed
   `SeedSequence([52, seed, state])`).
3. **Oracle recovery:** the same observations, with the B3 recovery chunk as the target.

The two fine-tuned arms share the state identities, row order, slot 0, mask, batches, optimizer, schedule, checkpoint
and evaluation starts. They differ only in the values of slots 1–15.

**Fine-tuning:**
- 10,000 steps at batch 64: 48 real subset windows and 16 synthetic, each stream drawn without replacement per epoch;
- a fresh AdamW with LeRobot's defaults, peak lr 1e-5, 100 warmup steps, cosine to 0;
- real windows supervise every slot, as the base trained; synthetic windows supervise their mask
  (`do_mask_loss_for_padding: true`);
- the final checkpoint.

**Decision:** pass only if, on both seeds, oracle recovery's average maximum coverage on starts 1000–1099 exceeds both
the base's and self-distillation's. All raw values, paired differences and success are reported. With two seeds this
is a directional development gate. If it fails, the study stops, and no deviation threshold, horizon, exposure or
schedule is searched.

**Compute:** siddh's two GPUs, at no cost.
- B1: about an hour per seed, at the 0.037 s/step measured in #50.
- B4: about 10 minutes per unit.
- Ceiling: 12 GPU-hours.

The final-test starts (5000–5099) are not read.
