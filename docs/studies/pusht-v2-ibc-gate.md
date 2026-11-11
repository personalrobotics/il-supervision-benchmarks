# PushT keypoints v2: IBC BC feasibility gate

**Status: run; the gate passed** ([Result](#result)), and passed again under
v2.1's faster settings ([rerun](#rerun-result)). The commit that adds this file, with
[`tools/pusht_v2`](../../tools/pusht_v2/gate.py), freezes the gate before any of
its policies trains. The machine-readable protocol is
[`gate.json`](../../tools/pusht_v2/gate.json). Controlling issue:
personalrobotics/CCIL#30.

## Why

Two earlier learners failed on PushT:
- **The [M=1 pilot](pusht-m1-pilot.md):** single-frame ACT on a state with no
  velocity.
- **The [velocity-aware ACT gate](pusht-v2-gate.md):** it predicted held-out
  actions at about 10 px, but closed loop it never moved the T.

The current learner removes both mechanisms the PushT literature relies on:
- **Observation history:** one observation is not Markov.
- **A multimodal action model:** a point regressor averages PushT's several
  valid pushes, and that averaged action drifts closed loop however low its
  offline error.

Implicit behavioral cloning restores both and still predicts one action. It is
an energy MLP over [o_{t−1}, o_t, a_t], with the action chosen by sampling and
refinement. That is the interface CCIL's one-step labels need. Diffusion
reached the demonstrator through this pipeline (0.948), but it predicts
16-step action chunks, which do not align with one-step labels. This gate asks
only whether IBC can learn PushT here. It is a baseline-capability gate, not a
CCIL result.

## The state

For the agent position and the T's 8 keypoints, x̄_t = [x_{t−1}, x_t]:
- `observation.state` grows from 2 to 4 values, and
  `observation.environment_state` from 16 to 32.
- The history lives inside these two features. LeRobot policies read only
  them, and a CCIL one-frame synthetic row then carries its own history.
  LeRobot's own `n_obs_steps 2` would pad a one-frame episode with a copy of
  itself.

The same state goes to BC, to CCIL's dynamics model and to the final policy:
- **Data:** [`derive.py`](../../tools/pusht_v2/derive.py) drops each episode's
  first frame.
- **Simulator:** the env plugin
  [`lerobot_env_pusht_history`](../../tools/pusht_v2/env_plugin/lerobot_env_pusht_history.py)
  carries the previous observation for an unchanged `lerobot-eval`. A reset is
  at rest, so its frame is its own predecessor.
- **Adapter check:** [`check.py`](../../tools/pusht_v2/check.py) must pass
  before training (smoke run):
  - the plugin reproduces every derived row exactly;
  - replayed demonstrations match to 4.5e-5 px (median episode);
  - every start is at rest;
  - LeRobot's discovery yields 4 + 32 observations.

## Learner

[`lerobot_policy_ibc`](../../tools/pusht_v2/ibc_plugin/lerobot_policy_ibc/modeling_ibc.py)
is a LeRobot 0.6.1 policy plugin, so `lerobot-train` and `lerobot-eval` run
unchanged. It ports the published PushT low-dimensional IBC
(real-stanford/diffusion_policy `5ba07ac`, `train_ibc_dfo_lowdim_workspace`
with `pusht_lowdim`; `IbcDfoLowdimPolicy` on its default paths):
- **Model:** a 4 × 1024 ReLU MLP energy with dropout 0.1.
- **Training:** InfoNCE of the demonstrated action (jittered by 1e-4) against
  1024 uniform negatives.
- **Inference:** 1024 samples, 5 iterations of resampling with 0.03 noise,
  then one action drawn from the last distribution.
- **Normalization and optimizer:** min/max normalization to [−1, 1]; AdamW
  (1e-4, betas 0.95/0.999, weight decay 1e-6); cosine to 0 after 500 warmup
  steps; no gradient clipping.

## Fixed setting

- **Split:** the pilot's CCIL split; BC trains on 165 episodes (20,335 frames).
- **Training:** 200,000 steps at batch 256, about the reference's optimizer
  steps. A checkpoint is saved every 20,000. Seeds 1–3.
  - Measured on siddh before the freeze: 0.076 s/step and 12.1 GB per unit
    alone on one GPU.
- **Selection:** each seed's checkpoint with the lowest mean held-out InfoNCE
  ([`select_infonce.py`](../../tools/pusht_v2/select_infonce.py)): eval mode,
  every frame of the 41 held-out episodes, and 1024 uniform negatives drawn
  once from seed 0 and shared by every checkpoint. Ties go to the earlier
  step. Simulator returns are never used. The sampled action's held-out L2 is
  recorded as a diagnostic only, since it penalizes a different valid push.
- **Evaluation:** unchanged `lerobot-eval`, env `pusht_history`, 100 episodes
  from start seed 1000, synchronous environments.

## Gate

Let R be the mean over the three seeds of average maximum coverage at each
seed's selected checkpoint.

**Pass iff (R − 0.0620) / (0.9464 − 0.0620) ≥ 0.5, that is R ≥ 0.5042.**

- **Pass:** freeze v2 and run bc, duplication and ccil with this learner and
  state. M=1, one-frame labels, one-action supervision, and the transition
  model fitted on the training episodes only. Nothing else changes.
- **Fail:** no CCIL run.

## Execution

From a clean checkout of the freezing commit, on `siddh`, in a LeRobot 0.6.1
environment (`dataset`, `pusht`, `training` extras) with this commit's two
plugins installed
(`uv pip install tools/pusht_v2/env_plugin tools/pusht_v2/ibc_plugin`):

```bash
uv run python tools/pusht_v2/gate.py --root runs/<timestamp>-pusht-v2-ibc-gate \
    --source <lerobot/pusht_keypoints at ace8c161> --lerobot-python <that environment>/bin/python
```

The runner refuses an installed plugin that differs from the commit's. A smoke
run (200 steps, one seed, 4 episodes) exercised every stage before the freeze.

## Result

**The gate passed.** The run is root `20261111T044357-pusht-v2-ibc-gate` on
`siddh`, from a clean checkout of `b41e352`; both plugins were installed from
that commit, and the adapter check passed. All three units trained 200,000
steps. Records: [`evidence/pusht-v2-ibc-gate`](evidence/pusht-v2-ibc-gate).

| seed | selected step | held-out InfoNCE | coverage (100 starts) | success |
|---|---|---|---|---|
| 1 | 20k | 1.254 | 0.7988 | 29% |
| 2 | 20k | 1.295 | 0.8348 | 36% |
| 3 | 20k | 1.272 | 0.8304 | 18% |
| **mean** | | | **0.8213** | |

(0.8213 − 0.0620) / (0.9464 − 0.0620) = **0.859** of the hold-to-demonstrator
gap; the gate needs 0.5. Single-step IBC on the two-frame history state learns
PushT. The single-frame ACT of the pilot and the velocity-aware ACT gate did not.

**Selection picked the first checkpoint on every seed.** Held-out InfoNCE
rose monotonically after 20k (1.25–1.30 to 2.73–2.78 at 200k), while closed-loop
coverage kept improving. Monitoring peeks, taken outside the gate's root on the
first 50 starts (hold-still scores 0.0486 there), show it. They are not the
gate's evaluation and decided nothing:

| seed | 20k | 40k | 60k | 80k | 100k | 120k | 140k | 160k | 180k | 200k |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 0.756 | 0.887 | 0.938 | 0.923 | 0.931 | 0.918 | 0.923 | 0.930 | 0.926 | 0.943 |
| 2 | 0.774 | 0.869 | 0.850 | 0.920 | 0.879 | 0.895 | 0.888 | 0.883 | 0.871 | 0.925 |
| 3 | 0.857 | 0.825 | 0.845 | 0.916 | 0.862 | 0.919 | 0.924 | 0.886 | 0.895 | 0.890 |

A rising held-out InfoNCE is the energy landscape sharpening around the
training actions, not worse control. The predeclared selection is kept for v2
unchanged: it never uses simulator returns, and it is applied identically to
every arm.

## Rerun under v2.1's settings (predeclared)

The 200k schedule spent about 90% of each unit's steps past the checkpoint that
selection kept. v2's first full run was therefore stopped before any outcome,
and v2 takes faster settings:
- **Ceiling:** 50,000 steps, with a checkpoint every 5,000.
- **Compilation:** the energy network is compiled in training.
- **Precision:** bf16 autocast over float32 weights.

Compilation leaves the arithmetic unchanged; bf16 changes it. Measured on
`siddh` ([`speed`](evidence/pusht-v2/development/speed/speed.json)), the
settings run 2.24× faster per step, 0.034 s against 0.076. A shorter schedule
also changes the cosine learning rate at every step. So the gate is rerun
under exactly these settings before v2 uses them
([`gate_fast.json`](../../tools/pusht_v2/gate_fast.json), with
`lerobot_policy_ibc` 1.1.0). Everything else is unchanged:
- the state and the split;
- the learner's method;
- held-out InfoNCE selection;
- the 100-episode evaluation;
- three seeds;
- the pass rule, R ≥ 0.5042.

A smoke run of these settings passed every stage before the freeze.

**Evaluation repeats only to rounding.** Before v2's freeze, one selected
checkpoint was evaluated twice under the same seed (10 episodes). Every success
repeated, and rewards agreed to 2.2e-16 (maximum) and 1.4e-14 (summed). v2's
audit predeclares those tolerances.

### Rerun result

**The gate passed again.** The run is root
`20261111T170711-pusht-v2-ibc-gate-fast` on `siddh`, from a clean checkout of
`28388ee`. Both plugins were installed from that commit, and the adapter check
passed. All three units trained 50,000 steps. Seed 2, alone on a GPU, took
1,860 s; seeds 1 and 3, sharing a GPU, took 3,721 s each. Records:
[`evidence/pusht-v2-ibc-gate/fast-run`](evidence/pusht-v2-ibc-gate/fast-run).

| seed | selected step | held-out InfoNCE | coverage (100 starts) | success |
|---|---|---|---|---|
| 1 | 10k | 1.155 | 0.8672 | 28% |
| 2 | 10k | 1.175 | 0.8494 | 22% |
| 3 | 10k | 1.168 | 0.7149 | 20% |
| **mean** | | | **0.8105** | |

(0.8105 − 0.0620) / (0.9464 − 0.0620) = **0.846** of the gap; the gate needs
0.5. The 200k run gave 0.859. Held-out InfoNCE again bottomed early, at 10k on
every seed (1.155–1.175), and rose to 1.50–1.54 at 50k. v2 runs under these
settings.
