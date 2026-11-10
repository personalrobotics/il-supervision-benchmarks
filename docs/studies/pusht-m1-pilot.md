# PushT keypoints: does one-step CCIL help an unchanged LeRobot policy? (pilot)

**Status: protocol version 1, frozen.** The commit that adds this file, with
[`tools/pusht_pilot`](../../tools/pusht_pilot/run.py), freezes the protocol;
it is pushed before any policy of the pilot is trained. The machine-readable
protocol is [`protocol.json`](../../tools/pusht_pilot/protocol.json); the
CCIL configuration is [`pusht.yaml`](../../tools/pusht_pilot/pusht.yaml).
Controlling issue: personalrobotics/CCIL#30.

## Question

Does one-step CCIL supervision help an unchanged LeRobot ACT policy on
`lerobot/pusht_keypoints` beyond adding the same amount of duplicated
demonstration supervision? The primary contrast is **CCIL − duplication**;
CCIL − BC is secondary. Policy improvement is measured, not required: a null
or unfavorable result completes the pilot.

## Outcomes already known

- `ccil diagnose` on this dataset (CCIL#171 comment) found that one-step labels
  do not leave the demonstrations' global support (displacement 0.11–0.53
  τ_demo). This pilot does not require it: the labels are local corrections,
  and global novelty is reported, not gated.
- Nothing about any policy of this pilot was observed before the freeze; the
  smoke runs below train 200 steps and are not results.

## Fixed setting

- **Data.** `lerobot/pusht_keypoints` at revision `ace8c161a68b…`, converted
  by CCIL; the state is `observation.state` (2) and
  `observation.environment_state` (16).
- **Labels.** CCIL `56b64b4`, [`pusht.yaml`](../../tools/pusht_pilot/pusht.yaml)
  (CCIL's `docs/pusht.md`): `noisy_action`, 4 proposals per transition, noise
  0.1 training-action standard deviations, recovery length 1, a spectrally
  constrained residual MLP fitted on the training episodes only, CCIL's
  split (41 of 206 episodes held out). One label set; every accepted label is
  selected with weight 1.
- **Arms**, each one LeRobot v3.0 dataset of the 165 training episodes
  (20,500 frames) plus:
  - **bc**: nothing;
  - **ccil**: CCIL's export, one one-frame episode per label (81,263);
  - **duplication**: for each CCIL label, in the same order, a one-frame copy
    of its source demonstration frame (x_t, u_t). It has the same frame
    count, the same added-row count and the same multiset of source frames as
    ccil (checked by hash in the `arms` stage).

  Each dataset carries its own normalization statistics, as an unchanged
  trainer computes them.
- **Policy.** LeRobot 0.6.1 ACT through unchanged `lerobot-train`:
  `n_obs_steps 1`, `chunk_size 1`, `n_action_steps 1`, `use_vae false`, every
  other setting at its default. A one-frame episode is then consumed exactly
  as a demonstration frame (CCIL `m4-export` evidence).
- **Training.** 100,000 steps at batch 64 for every arm and seed (so every
  unit sees 6.4 M samples: about 63 epochs of ccil and duplication, 312 of
  bc), a checkpoint every 10,000 steps, seeds 1–5, GPU.
- **Checkpoint selection under a common ceiling.** Each unit's checkpoint with
  the lowest mean L2 action error over every frame of the 41 held-out real
  episodes (ties to the earlier step), by a read-only evaluator. Simulator
  returns are never used for selection.
- **Evaluation.** Unchanged `lerobot-eval`, `pusht`, observation type
  `environment_state_agent_pos`, 100 episodes from start seed 1000
  (episode i is seeded 1000 + i), synchronous environments (LeRobot 0.6.1's
  asynchronous workers do not import `gym_pusht`).

## Outcomes and analysis

- **Primary:** average maximum coverage (`avg_max_reward`), CCIL − duplication,
  paired over the five seeds.
- **Secondary:** CCIL − BC on coverage; both contrasts on success rate
  (`pc_success`, coverage ≥ 0.95).
- **Interval:** `ilsb.analysis.paired`, the union of a BCa bootstrap and a
  paired t interval at 0.95, descriptive. Every seed's values are reported.
- **Signal rule:** a separately frozen confirmatory study is considered only
  if the mean primary contrast is above 0 **and** the primary contrast is
  positive on at least 4 of the 5 seeds. Otherwise the pilot ends.

## Execution

From a clean checkout of the freezing commit, on `siddh` (2 × RTX PRO 6000
Blackwell), with LeRobot 0.6.1 (`dataset`, `pusht` and `training` extras) in
its own environment:

```bash
uv run python tools/pusht_pilot/run.py --root runs/<timestamp>-pusht-m1-pilot \
    --source <lerobot/pusht_keypoints at ace8c161> --lerobot-python <lerobot env>/bin/python
```

Stages: `labels → arms → train → select → evaluate → audit → report`. The
training scheduler runs four units per GPU and fails the stage promptly,
naming the unit, if a worker exits or its log stalls for 15 minutes. The
audit re-verifies every artifact, reconciles the arms and the trainers'
exposure, and re-evaluates one unit to confirm the starts are fixed.

**Smoke and failure-path runs (before the freeze, 200 steps, seed 1).**
- Every stage ran end to end on `siddh`.
- A rerun on the same root reused every verified stage.
- **Failure path:** a worker killed with SIGKILL mid-training stopped the
  parent within 6 s, naming the unit. It wrote a failure record listing the
  stopped siblings, marked no unit complete and kept the partial outputs; the
  resume then completed.
- **Reproducibility:** two roots reproduced each other's label export, final
  weights, selection scores and every evaluated episode, bit for bit, on this
  host.

## Result

Not yet run.
