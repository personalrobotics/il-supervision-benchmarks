# PushT keypoints v2: BC feasibility gate

**Status: stopped before its decision; the learner changed** ([Result](#result)).
The commit that adds this file, with
[`tools/pusht_v2`](../../tools/pusht_v2/gate.py), freezes the gate before any of
its policies trains. The machine-readable protocol is
[`gate.json`](../../tools/pusht_v2/gate.json). Controlling issue:
personalrobotics/CCIL#30.

## Why

The [M=1 pilot](pusht-m1-pilot.md) gave its policy and CCIL's transition model a
state without velocity. The agent is PD-controlled toward its action, so that
state is not Markov, and every arm stayed near the hold-still floor. v2 makes
the state approximately Markov. That is a benchmark data-adapter correction,
not a CCIL change. This gate asks only whether single-step BC can learn PushT
from that state. It is a baseline-capability gate, not a CCIL result.

## The velocity-aware state

For the agent position and the T's 8 keypoints,
x̄_t = [x_t, (x_t − x_{t−1}) / Δt], with Δt = 0.1 s:

- `observation.state` grows from 2 to 4 values, and
  `observation.environment_state` from 16 to 32. The velocities stay inside
  these two features because LeRobot policies read only them.
- **Data:** [`derive.py`](../../tools/pusht_v2/derive.py) drops each episode's
  first frame rather than invent its velocity (206 episodes, 25,444 frames).
- **Simulator:** the env plugin
  [`lerobot_env_pusht_velocity`](../../tools/pusht_v2/plugin/lerobot_env_pusht_velocity.py)
  computes the same feature from the previous observation. `lerobot-eval`
  imports it unchanged, as it does every `lerobot_env_*` package. A reset is at
  rest, so its velocity is 0.
- **Adapter check:** [`check.py`](../../tools/pusht_v2/check.py) runs before
  training, and must pass:
  - the plugin's arithmetic on the recorded positions reproduces every derived
    row (smoke run: 0 px, 3.1e-5 px/s);
  - the demonstrations replayed through the wrapped simulator reproduce them
    (median episode 4.5e-5 px, 5.3e-4 px/s);
  - every evaluation start is at rest;
  - LeRobot's own plugin discovery yields 4 + 32 observations.

## Fixed setting

The pilot's setting, with three changes:
- **State:** the velocity-aware state above.
- **Normalization:** `ENV` is normalized (mean/std) alongside `STATE` and
  `ACTION`. LeRobot's ACT default leaves the keypoints raw; that was the
  pilot's secondary handicap.
- **Arms and seeds:** bc only, seeds 1–3.

Unchanged from the pilot:
- **Split:** the pilot's CCIL split; BC trains on the other 165 episodes,
  20,335 frames.
- **Policy:** single-frame ACT (`n_obs_steps 1`, `chunk_size 1`,
  `n_action_steps 1`, `use_vae false`).
- **Training and selection:** 100,000 steps at batch 64, a checkpoint every
  10,000; held-out action-error selection; simulator returns are never used.
- **Evaluation:** 100 episodes from start seed 1000.

## Gate

Let R be the mean over the three seeds of average maximum coverage at each
seed's selected checkpoint. The references are the pilot's measured hold-still
(0.0620) and demonstration-replay (0.9464) coverage on the same metric.

**Pass iff (R − 0.0620) / (0.9464 − 0.0620) ≥ 0.5, that is R ≥ 0.5042.**

- **Pass:** freeze v2 and run bc, duplication and ccil on this state. M=1,
  one-frame labels, single-step actions, the three-arm comparison, and the
  transition model fitted on the training episodes only.
- **Fail:** no CCIL run. CCIL labels are not forced into a chunked
  diffusion-policy format. A multimodal single-step learner outside CCIL comes
  first, to separate partial observability from action multimodality.

## Execution

From a clean checkout of the freezing commit, on `siddh`, in a LeRobot 0.6.1
environment (`dataset`, `pusht`, `training` extras) with this commit's plugin
installed (`uv pip install tools/pusht_v2/plugin`):

```bash
uv run python tools/pusht_v2/gate.py --root runs/<timestamp>-pusht-v2-gate \
    --source <lerobot/pusht_keypoints at ace8c161> --lerobot-python <that environment>/bin/python
```

Stages: `derive → train → select → evaluate → decide`. The runner reuses the
pilot's supervised scheduler and resume, and refuses an installed plugin that
differs from the commit's. A smoke run (200 steps, one seed, 4 episodes)
exercised every stage and the resume before the freeze.

## Result

**Stopped at about 80,000 of 100,000 steps; the gate never decided.** The run
(root `20261111T041007-pusht-v2-gate`, from a clean checkout of `4b25746`) was
interrupted by decision. The interrupt reached the scheduler: it stopped all
three units, wrote its failure record, and marked none complete. Records:
[`evidence/pusht-v2-gate`](evidence/pusht-v2-gate).

The adapter check passed on the frozen commit. The stop followed monitoring
peeks taken outside the gate's root: every saved checkpoint was scored on the
held-out frames and on the first 50 evaluation starts. These peeks are not
the gate's selection or evaluation.

| seed | held-out error, 10k → 80k | 50-episode coverage, 10k–80k |
|---|---|---|
| 1 | 12.96 → 10.12 px | 0.047–0.084 (one checkpoint above 0.048) |
| 2 | 13.99 → 9.62 px | 0.047–0.061 |
| 3 | 10.99 → 11.03 px (70k) | 0.047–0.048 |

On those 50 starts, holding still scores 0.0486, and the gate needs 0.5042
(over 100 starts).

- **The state works offline.** Held-out error fell from the pilot's 17.7–20.1 px
  to about 10 px, close to extrapolating the agent's last displacement (9.0 px).
- **Closed loop stays at the floor.** Two diagnostics on the 10k checkpoints
  explain why it is not the evaluation path:
  - fed the same demonstration frames from dataset rows and from the wrapped
    simulator, bc-s2's actions agree to a median of 3.4e-5 px;
  - rolled out from 10 starts, bc-s3 commands a target that pulls further
    ahead of the agent each step, and the agent wanders 155–2,398 px without
    moving the T.

**Decision.** More training of this learner is unlikely to help. Its state is
now velocity-aware, but a point-regression policy still averages PushT's
multiple valid pushes, and that averaged action drifts in closed loop.
Normalization cannot repair either problem. The v2 learner becomes an
implicit (energy-based) BC policy over the two-frame history with one-action
inference, the published IBC configuration for PushT. Its own BC gate is
predeclared before any CCIL arm runs.
