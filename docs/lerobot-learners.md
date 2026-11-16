# LeRobot learners

A study can train a policy with LeRobot's own implementation instead of one of ilsb's learners. The first such learner is LeRobot's Diffusion Policy (#50). ilsb adds a thin adapter and copies none of LeRobot's code. The model, its training loop, checkpointing, normalization and evaluation are those of the pinned release, driven through `lerobot-train` and `lerobot-eval`. ilsb owns:
- the declared configuration;
- the training and held-out datasets;
- the temporal windows' contract;
- the selection rule;
- the audits;
- the report.

The code is in three files:
- [`src/ilsb/learners/lerobot_diffusion.py`](../src/ilsb/learners/lerobot_diffusion.py) is the adapter.
- [`src/ilsb/temporal.py`](../src/ilsb/temporal.py) is the window contract.
- [`src/ilsb/studies/lerobot_policy.py`](../src/ilsb/studies/lerobot_policy.py) is the study that runs the learner on a LeRobot dataset and environment.

The LeRobot side runs in its own environment through [`src/ilsb/learners/lerobot_runtime.py`](../src/ilsb/learners/lerobot_runtime.py). That file imports neither ilsb nor ccil, so the harness and learner environments stay separate.

## The temporal window

A temporal learner trains on windows, not rows. The contract is stated independently of any supervision method, so a later study that builds windows from synthetic labels has a definition to meet.

A window is anchored at frame t of one episode of T frames:

| Part | Definition |
|---|---|
| Observation history | frames t + o for o = 1 − n_obs_steps, …, 0, oldest first |
| Future actions | the actions at t + a for a = 1 − n_obs_steps, …, horizon − n_obs_steps, in order |
| Padding | an offset outside the episode reads its nearest frame (the first or the last) and is flagged |
| Loss mask | every action slot when padded slots are supervised (`do_mask_loss_for_padding: false`, LeRobot's default), otherwise the unpadded slots only. Observation padding is never masked. |
| Anchors | every t ≤ T − 1 − drop_n_last_frames (LeRobot's `EpisodeAwareSampler`) |
| Provenance | the anchor's episode, and the frame of every slot |

A window never crosses an episode boundary, by construction. Rows that do not form episodes are refused before any window is built: an episode's rows must be contiguous and its frames must be 0 … T − 1 in order. Preflight builds every window of both roles from the dataset's rows and compares it, slot by slot, with what `LeRobotDataset` gives the policy: the anchors, every observation and action value, and every padding flag.

For the PushT reference (n_obs_steps 2, horizon 16, n_action_steps 8, drop_n_last_frames 7), each episode contributes T − 7 windows. Each window has one padded observation slot at t = 0, and its padded action slots are the first action at t = 0 plus up to seven repeats of the last action at the end.

## Selecting the learner

A specification with `task.kind: lerobot` selects the study, and its `learner.kind` selects the learner. Another task uses Diffusion Policy by declaring its own dataset, roles, environment and starts. No harness code changes.

```json
{
 "study": "my-task-diffusion",
 "task": {"kind": "lerobot", "env": {"type": "pusht", "obs_type": "environment_state_agent_pos"}},
 "data": {"repo_id": "lerobot/pusht_keypoints", "revision": "ace8c161…",
          "files": {"data/chunk-000/file-000.parquet": "8b4d97…", "meta/info.json": "3a6d3d…", "…": "…"}},
 "roles": {"train": [0, 1, …], "held_out": [7, 9, …]},
 "learner": {"kind": "lerobot_diffusion", "lerobot": "0.6.1",
             "policy": {"n_obs_steps": 2, "horizon": 16, "n_action_steps": 8, "drop_n_last_frames": 7,
                        "do_mask_loss_for_padding": false,
                        "normalization_mapping": {"VISUAL": "MEAN_STD", "STATE": "MIN_MAX", "ENV": "MIN_MAX",
                                                  "ACTION": "MIN_MAX"}},
             "steps": 100000, "batch_size": 64, "save_freq": 10000, "log_freq": 1000, "num_workers": 4,
             "selection": {"rule": "last", "seed": 0, "batch_size": 256}},
 "seeds": [1, 2],
 "evaluation": {"start_seed": 1000, "n_episodes": 100, "batch_size": 50, "device": "cuda"},
 "acceptance": {"metric": "avg_max_reward", "min_mean": 0.85},
 "expected": {"train_episodes": 165, "held_out_episodes": 41, "train_frames": 20500, "held_out_frames": 5150,
              "train_windows": 19345, "held_out_windows": 4863, "units": 2},
 "compute": {"device": "cpu", "learner_device": "cuda", "stall_minutes": 15}
}
```

- **`task.env`** holds `lerobot-eval`'s `--env.*` arguments. They must name an environment whose observations match the dataset's features.
- **`data.files`** pins every file of the dataset directory (`ILSB_LEROBOT_DATASET`) by SHA-256. No file may be missing and none may be extra. Hidden directories, such as the Hub client's `.cache`, are ignored.
- **`roles`** are disjoint lists of the dataset's episode indices. Episodes in neither role are unused.
- **`learner.policy`** holds `DiffusionConfig` fields, passed to `lerobot-train` as `--policy.<key>`.
  - The four window fields, `do_mask_loss_for_padding` and `normalization_mapping` are required.
  - `type`, `device` and `push_to_hub` are set by the adapter.
  - Any other field is passed verbatim. The fixture, for example, sets `down_dims`.
- **`learner.selection.rule`** is one of two rules:
  - `last`: the final checkpoint;
  - `min_held_out_loss`: the checkpoint with the least held-out loss, with ties going to the earlier step.

  The held-out loss of every checkpoint is recorded under either rule.
- **`expected`** holds every count the protocol implies, asserted exactly in preflight.
- **`compute.device`** is the harness's device, which is the CPU. `learner_device` is where training, scoring and evaluation run.

Every field is checked for its exact presence and type before anything is built. A malformed specification is refused.

## Data roles and normalization

`lerobot-train` always normalizes with the training dataset's own `meta/stats.json`. So the study gives it a dataset that holds the training episodes only. It is made by LeRobot's `split_dataset`, which aggregates the kept episodes' own statistics. Preflight checks three things about each role's dataset:
- its rows are exactly the source rows of its episodes;
- its statistics' minima and maxima are its own rows';
- its windows are the contract's.

The audit checks that every checkpoint's normalizer and unnormalizer hold the training episodes' statistics. The held-out episodes are read only for the held-out loss. On PushT this matters: the held-out episodes extend the state and keypoint ranges, so the full dataset's statistics differ from the training episodes'.

## Determinism, resume and reuse

Training runs with deterministic cuDNN and cuBLAS (`--cudnn_deterministic=true`, `CUBLAS_WORKSPACE_CONFIG=:4096:8`). Under that setting, these properties were measured with LeRobot 0.6.1:
- **Repeats:** a repeat with the same seed gives the same parameters bit for bit, on siddh's RTX PRO 6000 and on the CPU.
- **Resume on CUDA:** an interrupted run, resumed through `lerobot-train --resume` from its last checkpoint, reaches the uninterrupted run's parameters bit for bit. That includes its data order, optimizer, scheduler and RNG state.
- **Resume on the CPU:** an interrupted run does not reach them. The segment after the resume diverges, probably because the diffusion noise is drawn from the CPU generator, which the data loader also draws from. So an interrupted CPU unit restarts from step 0.
- **Workers:** `num_workers` changes speed, not the batches. At the reference's scale, 0 and 4 workers give the same held-out losses to the last digit.

A unit records the steps it resumed from and the steps it restarted from. Without the deterministic setting, a repeat with the same seed already differs, by up to 4e-7 in a parameter after 60 steps.

The held-out loss is the policy's own training loss (`DiffusionPolicy.forward`) over every anchored held-out window, in anchor order. Its diffusion noise and timesteps come from a generator seeded once. It is computed in strict FP32, with no autocast and no TF32. So the audit re-scores the selected checkpoint in a fresh process and requires the same number. Evaluation is `lerobot-eval` with deterministic cuDNN, and the evaluation audit requires a repeat evaluation to reproduce every episode exactly.

Units are store entries like every other study's. A complete unit is reused under an equal identity: the split, the learner configuration, the seed, the device, the code and the environment.

## Environment

The LeRobot environment is installed from the pinned freeze [`tools/lerobot_diffusion/lerobot-env.txt`](../tools/lerobot_diffusion/lerobot-env.txt): LeRobot 0.6.1 with diffusers 0.39.0, and torch 2.11.0 for CUDA 13.

```bash
uv venv --python 3.12 envs/lerobot-diffusion
uv pip install --python envs/lerobot-diffusion/bin/python -r tools/lerobot_diffusion/lerobot-env.txt
export ILSB_LEROBOT_PYTHON=$PWD/envs/lerobot-diffusion/bin/python
export ILSB_LEROBOT_DATASET=<the dataset directory>
uv run python -m ilsb run SPEC --root RUNS/<study> --gpus 0,1
```

The study's scope records the environment's distributions, Python, PyTorch, CUDA and GPU. The environment must hold exactly the pinned LeRobot release.

## Evidence

`tools/diffusion_fixture.py --work NEW_DIR --lerobot-python PYTHON` is the compact behavior-level fixture, written to `docs/fixtures/evidence/diffusion-fixture.json`. It uses 6 training and 2 held-out PushT episodes, and a small Diffusion Policy with the reference's window that trains for 200 steps on the CPU, two seeds. Its docstring maps each protected failure to its check:
- the wiring of every stage;
- the windows against the learner's, and the contract's refusals;
- normalization isolation, including a checkpoint carrying the full dataset's statistics, which is refused;
- the selection and the re-scored checkpoint;
- an interrupted unit, killed after its first checkpoint, that reaches the uninterrupted unit's weights bit for bit: restarted on the CPU, and resumed with `--device cuda`;
- rerun and shared-store reuse;
- refusals of overlapping roles, an unpinned LeRobot, an altered dataset and corrupted weights.

CI runs it from the built wheel, against the pinned LeRobot environment on CPU PyTorch. Full training is never a CI job. The same fixture with `--device cuda` (two data-loading workers, the pinned CUDA environment) is written to `docs/fixtures/evidence/diffusion-fixture-cuda.json`; it is where the interrupted unit resumes rather than restarts. Both records were made on siddh from a clean clone of the commit that adds them.

The PushT reference, a competent policy on the canonical split and starts, is [docs/studies/pusht-diffusion.md](studies/pusht-diffusion.md).
