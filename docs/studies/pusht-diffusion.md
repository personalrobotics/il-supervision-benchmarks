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
- **Evaluation audit:** each rollout's scores are recomputed. Seed 1 is re-evaluated, and every episode's success must be identical, its max reward within 1e-12 and its sum reward within 1e-9. These are the PushT studies' repeat tolerances.
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

The run uses siddh's two RTX PRO 6000 GPUs, one unit per GPU, at no cost. A target-scale timing unit measured the step time and memory before the run (see the result).

## Result

The result is recorded after the run.
