"""LeRobot's Diffusion Policy as an ilsb learner (ilsb#50): a thin adapter, not a second implementation.

The model, its training loop, checkpointing, normalization and evaluation are LeRobot's own (`lerobot-train` and
`lerobot-eval` of the pinned release, driven by ilsb/learners/lerobot_runtime.py in the LeRobot environment). The
adapter declares and binds what a study needs to reproduce and audit a unit:

  learner:
    kind: lerobot_diffusion
    lerobot: "0.6.1"                  the pinned release; the environment must hold exactly it
    policy:                           DiffusionConfig fields, passed to lerobot-train as --policy.<key>
      n_obs_steps, horizon, n_action_steps, drop_n_last_frames   the temporal window (required: ilsb.temporal)
      do_mask_loss_for_padding                                   false supervises padded action slots (required)
      normalization_mapping                                      per feature type (required)
      ...                                                        any other DiffusionConfig field, verbatim
    steps, batch_size, save_freq, log_freq                        lerobot-train's own arguments (save_freq divides steps)
    num_workers                                                   its data-loading processes (the sampler's order is a
                                                                  function of the seed and epoch alone, so this changes
                                                                  speed, not the batches)
    selection: {rule, seed, batch_size}
      rule last                the final checkpoint (the held-out loss of every checkpoint is recorded as a diagnostic)
      rule min_held_out_loss   the checkpoint with the least held-out loss; ties go to the earlier step
      seed, batch_size         the held-out loss's fixed noise generator and batch (lerobot_runtime.score)

Training is deterministic (cuDNN and cuBLAS), so an interrupted unit reaches the parameters an uninterrupted run
reaches: on CUDA by lerobot-train's own resume, on the CPU (where that resume is not exact) by a restart; both are
recorded. The dataset is the study's: a LeRobot dataset
holding the training episodes only, so lerobot-train's normalization statistics (always the dataset's own) are the
training episodes' (lerobot_runtime.split).
"""
from __future__ import annotations

from ..temporal import WindowSpec

KIND = "lerobot_diffusion"
POLICY_TYPE = "diffusion"  # LeRobot's registered name of the policy
REQUIRED_POLICY = ("n_obs_steps", "horizon", "n_action_steps", "drop_n_last_frames", "do_mask_loss_for_padding",
                   "normalization_mapping")
FORBIDDEN_POLICY = ("type", "device", "push_to_hub", "repo_id", "pretrained_path")  # set by the adapter or the study
RULES = ("last", "min_held_out_loss")


class LearnerError(ValueError):
    pass


def _int(x, lo=1) -> bool:
    return not isinstance(x, bool) and isinstance(x, int) and x >= lo


def check(lr: dict) -> dict:
    """The learner block, exactly: every field present with its type, before anything is built."""
    need = {"kind", "lerobot", "policy", "steps", "batch_size", "save_freq", "log_freq", "num_workers", "selection"}
    if set(lr) != need:
        raise LearnerError(f"learner keys {sorted(lr)} differ from {sorted(need)}")
    if lr["kind"] != KIND or not isinstance(lr["lerobot"], str):
        raise LearnerError(f"learner.kind must be {KIND} and learner.lerobot a version string")
    p = lr["policy"]
    if not isinstance(p, dict) or any(k not in p for k in REQUIRED_POLICY) or any(k in p for k in FORBIDDEN_POLICY):
        raise LearnerError(f"learner.policy must set {list(REQUIRED_POLICY)} and none of {list(FORBIDDEN_POLICY)}")
    if not isinstance(p["normalization_mapping"], dict):
        raise LearnerError("learner.policy.normalization_mapping must map feature types to modes")
    window_spec(lr)
    if not all(_int(lr[k]) for k in ("steps", "batch_size", "save_freq", "log_freq")) or lr["steps"] % lr["save_freq"]:
        raise LearnerError("steps, batch_size, save_freq and log_freq must be positive integers, save_freq dividing "
                           "steps (so the final step is a checkpoint)")
    if not _int(lr["num_workers"], 0):
        raise LearnerError("num_workers must be an integer >= 0")
    s = lr["selection"]
    if set(s) != {"rule", "seed", "batch_size"} or s["rule"] not in RULES or not _int(s["seed"], 0) \
            or not _int(s["batch_size"]):
        raise LearnerError(f"learner.selection is {{rule: one of {list(RULES)}, seed: int >= 0, batch_size: int >= 1}}")
    return lr


def window_spec(lr: dict) -> WindowSpec:
    p = lr["policy"]
    if not isinstance(p["do_mask_loss_for_padding"], bool):
        raise LearnerError("learner.policy.do_mask_loss_for_padding must be a bool")
    try:
        return WindowSpec(p["n_obs_steps"], p["horizon"], p["n_action_steps"], p["drop_n_last_frames"],
                          supervise_padding=not p["do_mask_loss_for_padding"])
    except ValueError as e:
        raise LearnerError(f"learner.policy: {e}") from None


def checkpoint_steps(lr: dict) -> list[int]:
    return list(range(lr["save_freq"], lr["steps"] + 1, lr["save_freq"]))


def select(history: list[dict], rule: str) -> int:
    """The selected checkpoint's step, from the recorded history alone."""
    if rule == "last":
        return max(h["step"] for h in history)
    best = min(h["held_out_loss"] for h in history)
    return min(h["step"] for h in history if h["held_out_loss"] == best)


def train_config_contract(lr: dict, seed: int, device: str) -> dict:
    """What a checkpoint's own train_config.json must record for it to be this unit's (paths excluded)."""
    return {"steps": lr["steps"], "batch_size": lr["batch_size"], "save_freq": lr["save_freq"],
            "num_workers": lr["num_workers"], "seed": seed,
            "cudnn_deterministic": True, "policy": {**lr["policy"], "type": POLICY_TYPE, "device": device}}
