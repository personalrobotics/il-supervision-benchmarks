# PushT keypoints v2: does one-step CCIL help an IBC policy on a history state?

**Status: protocol frozen, not yet run.** The commit that adds this text, with
[`tools/pusht_v2`](../../tools/pusht_v2/study.py), freezes the study before any
of its policies trains. The machine-readable protocol is
[`protocol.json`](../../tools/pusht_v2/protocol.json); the CCIL configuration is
[`pusht.yaml`](../../tools/pusht_v2/pusht.yaml). Controlling issue:
personalrobotics/CCIL#30.

## Question

Does one-step CCIL supervision help an IBC policy on `lerobot/pusht_keypoints`
beyond the same amount of duplicated demonstration supervision? The primary
contrast is **CCIL − duplication**; CCIL − BC is secondary. Improvement is
measured, not required: a null or unfavorable result completes the study.

## Design

The [M=1 pilot](pusht-m1-pilot.md) found no evidence either way. Its
single-frame policies stayed near the hold-still floor. v2 differs from it in
these respects only:

1. **The policy's state is the two-frame keypoint history**
   x̄_t = [x_{t−1}, x_t].
   - **Data:** [`derive.py`](../../tools/pusht_v2/derive.py).
   - **Simulator:** the env plugin `lerobot_env_pusht_history`.
   - **Adapter check:** [`check.py`](../../tools/pusht_v2/check.py).
2. **IBC downstream** ([`lerobot_policy_ibc`](../../tools/pusht_v2/ibc_plugin/lerobot_policy_ibc/modeling_ibc.py)).
   It passed its [competency gate](pusht-v2-ibc-gate.md): 0.8213 coverage,
   0.859 of the hold-to-demonstrator gap. Two earlier learners failed: the
   pilot's single-frame ACT and a [velocity-aware ACT](pusht-v2-gate.md).
3. **CCIL generates in physical coordinates and the labels are decoded.** Each
   frame's T pose (x, y, θ) is recovered from its keypoints by the rigid fit,
   with θ unwrapped per episode ([`pose.py`](../../tools/pusht_v2/pose.py)).
   - **Model:** CCIL fits the structured history model (CCIL#196: the history
     shift exact, only the newest slot learned) on the pose history
     [q_{t−1}, q_t], q = (agent x, y, T x, y, θ).
   - **Labels:** it solves M = 1 preimages there (`noisy_action_fe`).
   - **Decoding:** each label decodes to the keypoint history the policy reads,
     [agent̃_{t−1}, agent_t, keypoints(T̃_{t−1}), keypoints(T_t)] with its
     action. The current halves are the source frame's own.

   The policy and its passed gate are unchanged.

**Why the pose coordinates** ([the full account](pusht-keypoints-vs-pose.md)). Generating directly on the keypoint history
failed:
- **Low yield:** Newton converged on 1% of proposals.
- **No valid root:** every converged label deformed the previous frame's T by
  about 100 px.
- **Damping didn't fix it:** a damped solve only plateaued above the tolerance.

The learned keypoint model has no exact, physically valid preimage
([evidence](evidence/pusht-v2/development/preimage-probe/probe.json)). In
pose coordinates a frozen smoke gate passed: 99.92% of proposals usable, with no
residual, history-invariant or rigidity violation, and a 5 × 5 previous-slot
Jacobian of median condition 275
([evidence](evidence/pusht-v2/development/pose-probe/pose-probe.json)).

## Fixed setting

- **Data.** `lerobot/pusht_keypoints` at `ace8c161`, as the keypoint history
  (206 episodes, 25,444 frames) and the pose history with the same rows.
- **Split.** CCIL's split, which is the pilot's and the gate's: 41 episodes
  held out, 165 for training (20,335 frames).
- **Labels.** CCIL `a995ff9`, [`pusht.yaml`](../../tools/pusht_v2/pusht.yaml):
  - **Model:** the structured spectral residual MLP (256 × 256, constraint 3.0,
    50 epochs), fitted on the training episodes only.
  - **Generation:** `noisy_action_fe`, 4 proposals per transition, noise 0.1
    training-action standard deviations, M = 1.
  - **Curation:** every accepted label is curated with weight 1.
- **Correctness: assertions, not yield thresholds.** Every selected label must
  pass all three; one violation stops the run as a defect:
  - **residual:** ‖h_G + f̂(h_G, ã) − target‖ ≤ 10⁻⁴;
  - **history invariant:** newest slots equal to the source's q_t within
    ε_struct = max(10 · 10⁻⁴, 10 ε₃₂ max(1, ‖q_t‖∞));
  - **rigidity:** a decoded T as rigid as the demonstrations' in both slots,
    within 4.14 × 10⁻⁴ px (10 × their largest deviation).
- **Yield and round trip.** Separately, the run stops if fewer than 50% of the
  proposals are selected, or if the stored poses fail to decode to the
  keypoints within 4.14 × 10⁻⁴ px.
- **Arms**, each a LeRobot v3.0 dataset of the training episodes plus:
  - **bc**: nothing;
  - **ccil**: one one-frame episode per decoded label;
  - **duplication**: for each decoded label, in order, a one-frame copy of its
    source frame. These are the same rows as ccil's, matched by source-index
    hash.
- **Policy.** IBC as in the gate, through unchanged `lerobot-train`.
- **Training.** 200,000 steps at batch 256 for every arm and seed, a checkpoint
  every 20,000, seeds 1–5.
- **Checkpoint selection under a common ceiling.** Each unit's checkpoint with
  the lowest held-out InfoNCE. The negatives are fixed from seed 0 and shared by
  every checkpoint and arm. Ties go to the earlier checkpoint, and simulator
  returns are never used. In the gate, this rule picked the first checkpoint on
  every seed.
- **Evaluation.** Unchanged `lerobot-eval`, env `pusht_history`, 100 episodes
  from start seed 1000. The audit's repeat evaluation requires seeds and
  successes to repeat exactly, and rewards within 1e-12 (max) and 1e-9
  (summed).

## Outcomes and analysis

- **Primary:** average maximum coverage, CCIL − duplication, paired over five
  seeds.
- **Secondary:** CCIL − BC on coverage, and both contrasts on success rate.
- **Interval:** `ilsb.analysis.paired`, the union of a BCa bootstrap and a
  paired t interval at 0.95, descriptive. Every seed's values are reported.
- **Signal rule:** a confirmatory study is considered only if the mean primary
  contrast is above 0 and the primary contrast is positive on at least 4 of the
  5 seeds.

**What the CCIL arm adds.** A valid preimage label sits at a demonstrated
current state with a different previous frame: an alternative arrival
velocity, for both the agent and the T. It does not add broad positional
coverage. In the smoke run, a label's previous frame moved by 14.6 px (median).

## Execution

From a clean checkout of the freezing commit, on `siddh`, in a LeRobot 0.6.1
environment with this commit's two plugins installed:

```bash
uv run python tools/pusht_v2/study.py --root runs/<timestamp>-pusht-v2 \
    --source <lerobot/pusht_keypoints at ace8c161> --lerobot-python <that environment>/bin/python
```

The stages are `derive → labels → arms → train → select → evaluate → audit →
report`. A smoke run from `9a69f4d` (200 steps, one seed, four episodes)
passed every stage before the freeze. It selected 100,871 of 100,952
proposals with no violations, and its three arms reconciled.

## Result

Not yet run.
