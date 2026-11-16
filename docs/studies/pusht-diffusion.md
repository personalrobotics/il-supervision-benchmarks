# PushT: a Diffusion Policy reference

This reference shows that ilsb's LeRobot Diffusion Policy learner (#50, [docs/lerobot-learners.md](../lerobot-learners.md)) trains and evaluates a competent PushT policy on the repository's canonical split and starts. It compares no supervision methods, and it makes no claim that Diffusion Policy is better than IBC.

## Protocol

The specification is [`tools/lerobot_diffusion/pusht-reference.json`](../../tools/lerobot_diffusion/pusht-reference.json). It was fixed before any reference unit trained.

- **Data:** `lerobot/pusht_keypoints` at revision `ace8c161`, with every file pinned by SHA-256. It has 206 episodes: the agent position (2) and the T's 8 keypoints (16) per frame, at 10 fps.
- **Roles:** the canonical split of the PushT studies ([pusht.md](pusht.md)).
  - 165 training episodes (20,500 frames) train the policy and supply its normalization statistics.
  - 41 held-out episodes (5,150 frames) are scored for the held-out loss only.
- **Learner:** LeRobot 0.6.1's Diffusion Policy.
  - **Window:** two observation steps, a prediction horizon of 16, eight executed actions, and the last 7 frames of each episode dropped as anchors. That gives 19,345 training windows and 4,863 held-out windows.
  - **Loss mask:** padded action slots are supervised (`do_mask_loss_for_padding: false`, LeRobot's default).
  - **Normalization:** min/max for the state, the keypoints and the action.
  - **Everything else:** LeRobot's defaults. These include the 1-D temporal U-Net (512, 1024, 2048), 100 DDPM steps, AdamW at 1e-4 with 500 warmup steps, and a cosine decay over the run.
- **Training:** 100k steps at batch 64, with a checkpoint every 10k steps.
  - It is deterministic: deterministic cuDNN and cuBLAS, data loading in 4 workers.
  - Seeds are 1 and 2, one GPU each.
- **Selection:** the final checkpoint (`rule: last`).
  - Diffusion Policy's denoising loss is a weak guide to closed-loop quality, and the anchor below used the final checkpoint.
  - The held-out loss of every checkpoint is recorded as a diagnostic, with fixed noise (seed 0) in strict FP32.
- **Evaluation:** unchanged `lerobot-eval` on `pusht` with `environment_state_agent_pos` observations, 100 episodes from start seeds 1000–1099 (the canonical starts), in batches of 50. The policy keeps its own two-step observation queue.
- **Evaluation audit:** each rollout's scores are recomputed from its episodes, and its arguments, starts and checkpoint are checked. Seed 1 is re-evaluated and the repeat's deviation recorded. The PushT simulator does not repeat contact-rich trajectories ([lerobot-learners.md](../lerobot-learners.md#determinism-resume-and-reuse)), so the repeat is a measurement, not a gate.
- **Acceptance:** the mean over the seeds of the average maximum coverage is at least 0.85.

### Anchor

An earlier stock LeRobot Diffusion Policy run reached 0.948 average maximum coverage and 71% success. It had:
- the same window and normalization;
- one seed, 100k steps at batch 64;
- the final checkpoint, with no selection;
- training on the M=1 pilot's BC dataset;
- non-deterministic cuDNN.

Its record is `docs/studies/evidence/pusht-m1-pilot/reference/diffusion/` at tag [`pusht-evidence-archive-v1`](https://github.com/personalrobotics/il-supervision-benchmarks/tree/pusht-evidence-archive-v1). It is an anchor, not a value to reproduce bit for bit.

### Compute

The run uses siddh's two RTX PRO 6000 GPUs, one unit per GPU, at no cost. A target-scale timing unit measured the step time and memory before the run: 0.037 s per step with 4 data-loading workers, and 5.8 GB of GPU memory.

## Result

**The reference is met: the mean average maximum coverage is 0.955, against the 0.85 threshold.**

The run is from commit `428a141`, with a clean tree, on siddh. Every stage passed. The record is [`evidence/pusht-diffusion/reference.json`](evidence/pusht-diffusion/reference.json), regenerated from the run root by [`tools/lerobot_diffusion/reference_record.py`](../../tools/lerobot_diffusion/reference_record.py).

| Seed | Average max coverage | Success | Selected checkpoint |
|---|---|---|---|
| 1 | 0.933 | 70% | 100k |
| 2 | 0.977 | 73% | 100k |
| Mean | **0.955** | 71.5% | |

The result is in line with the 0.948 / 71% anchor. The record lists the start seed of each episode that did not succeed: 30 for seed 1 and 27 for seed 2.

- **Preflight:**
  - every count matched the declaration: 165 and 41 episodes, 20,500 and 5,150 frames, 19,345 and 4,863 windows;
  - every window `LeRobotDataset` gave the policy matched the contract's, slot by slot;
  - the training dataset's statistics were its own rows'.
- **Held-out loss:** it rose steadily after the first checkpoint, from 0.040 at 10k to 0.170 at 100k (seed 1), and similarly for seed 2.
  - The final checkpoint, the one the protocol selects, has the highest held-out denoising loss and a competent closed-loop policy.
  - This is the weak link between Diffusion Policy's denoising loss and its closed-loop quality that led the protocol to fix `rule: last` before training.
- **Audit:** each checkpoint's `train_config` and normalizer statistics were the unit's. The selected checkpoint's held-out loss reproduced in a fresh process.
- **Repeat evaluation:** seed 1's repeat gave the same coverage and success, with every episode's success unchanged and per-episode rewards within 6e-14. Because the PushT simulator does not repeat contact-rich trajectories, this agreement is a measurement, not a guarantee. A repeat can move individual episodes.
- **Compute:** each unit trained 100k steps in 3,815 s, with a peak of 5.8 GB of GPU memory and 3.9 GB resident. Neither unit was interrupted.

The store entries, logs and run root are on siddh under `~/code/ilsb-dp-ref-428a141/runs/20261116T041219Z-pusht-diffusion-reference`.
