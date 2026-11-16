"""The learner-facing temporal example (ilsb#50), independent of any supervision method.

A temporal learner trains on windows, not rows. A window is anchored at one frame t of one episode and holds
  observations  the frames t + o for each observation offset o (1 - n_obs_steps, ..., 0), oldest first;
  actions       the actions at t + a for each action offset a (1 - n_obs_steps, ..., horizon - n_obs_steps), in order.
An offset that falls outside the anchor's episode is padding: the row used is the episode's nearest frame (its first or
last), and the slot is flagged. That is LeRobot's rule (LeRobotDataset clamps a delta index to the episode and returns
`<key>_is_pad`), stated here so the harness can recompute and check every window from the rows alone.

The loss mask says which action slots train the learner: every slot when padded slots are supervised (they repeat
the episode's last action, LeRobot's default for Diffusion Policy), otherwise only the slots that are not padding.
Observation padding is never masked; a padded observation repeats the episode's first frame.

Anchors: every frame t with t <= T - 1 - drop_n_last_frames, in an episode of T frames (LeRobot's EpisodeAwareSampler
rule). A window never crosses an episode boundary, by construction. Rows that do not form episodes (an episode's rows
not contiguous, or its frames not 0..T-1 in order) are refused before any window is built, because a window over them
would join unrelated rows. Every window records its provenance: the source episode and the frame of each slot.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass

import numpy as np


class WindowError(ValueError):
    """Rows that cannot form windows, or a window that does not satisfy the contract."""


@dataclass(frozen=True)
class WindowSpec:
    n_obs_steps: int
    horizon: int
    n_action_steps: int
    drop_n_last_frames: int
    supervise_padding: bool

    def __post_init__(self):
        for k in ("n_obs_steps", "horizon", "n_action_steps", "drop_n_last_frames"):
            v = getattr(self, k)
            if isinstance(v, bool) or not isinstance(v, int) or v < (0 if k == "drop_n_last_frames" else 1):
                raise WindowError(f"{k} must be an integer >= {0 if k == 'drop_n_last_frames' else 1}, not {v!r}")
        if not isinstance(self.supervise_padding, bool):
            raise WindowError("supervise_padding must be a bool")
        if self.n_action_steps > self.horizon - self.n_obs_steps + 1:
            raise WindowError("the executed actions must lie within the predicted horizon, from the current frame on")

    @property
    def obs_offsets(self) -> np.ndarray:
        return np.arange(1 - self.n_obs_steps, 1)

    @property
    def action_offsets(self) -> np.ndarray:
        return np.arange(1 - self.n_obs_steps, 1 - self.n_obs_steps + self.horizon)

    def as_dict(self) -> dict:
        return {"n_obs_steps": self.n_obs_steps, "horizon": self.horizon, "n_action_steps": self.n_action_steps,
                "drop_n_last_frames": self.drop_n_last_frames, "supervise_padding": self.supervise_padding,
                "obs_offsets": self.obs_offsets.tolist(), "action_offsets": self.action_offsets.tolist()}


def episodes(episode_index: np.ndarray, frame_index: np.ndarray) -> list[tuple[int, int, int]]:
    """(episode, first row, row count) of each episode, in row order; refuses rows that do not form episodes."""
    ep, fr = np.asarray(episode_index).reshape(-1), np.asarray(frame_index).reshape(-1)
    if len(ep) != len(fr) or not len(ep):
        raise WindowError("episode and frame indices must be equal, non-empty columns")
    cut = np.flatnonzero(np.diff(ep) != 0) + 1
    starts, stops = np.r_[0, cut], np.r_[cut, len(ep)]
    out = [(int(ep[a]), int(a), int(b - a)) for a, b in zip(starts, stops)]
    seen = [e for e, _, _ in out]
    if len(set(seen)) != len(seen):
        raise WindowError(f"episode rows are not contiguous: {sorted({e for e in seen if seen.count(e) > 1})[:5]} "
                          "appear in more than one run of rows")
    for e, a, n in out:
        if not np.array_equal(fr[a:a + n], np.arange(n)):
            raise WindowError(f"episode {e}'s frames are not 0..{n - 1} in order")
    return out


def windows(episode_index: np.ndarray, frame_index: np.ndarray, spec: WindowSpec) -> dict:
    """Every anchored window of the rows, as row indices into them, with its padding, loss mask and provenance."""
    parts = {k: [] for k in ("anchor", "episode", "obs_rows", "obs_pad", "action_rows", "action_pad")}
    for e, a, n in episodes(episode_index, frame_index):
        t = np.arange(max(0, n - spec.drop_n_last_frames))
        if not len(t):
            continue
        for name, off in (("obs", spec.obs_offsets), ("action", spec.action_offsets)):
            raw = t[:, None] + off[None, :]
            parts[f"{name}_rows"].append(a + np.clip(raw, 0, n - 1))
            parts[f"{name}_pad"].append((raw < 0) | (raw > n - 1))
        parts["anchor"].append(a + t)
        parts["episode"].append(np.full(len(t), e))
    if not parts["anchor"]:
        raise WindowError("no episode is long enough for one anchored window")
    w = {k: np.concatenate(v) for k, v in parts.items()}
    w["loss_mask"] = np.ones_like(w["action_pad"]) if spec.supervise_padding else ~w["action_pad"]
    return w


def verify(w: dict, episode_index: np.ndarray, frame_index: np.ndarray, spec: WindowSpec) -> None:
    """A window set satisfies the contract over these rows: each slot's row is the clamped offset frame of the
    anchor's own episode, its padding flag is exactly the out-of-episode offsets, and the loss mask follows the rule.
    Checked from the rows' episode and frame columns, so a window that joins two episodes is refused."""
    ep, fr = np.asarray(episode_index).reshape(-1), np.asarray(frame_index).reshape(-1)
    length = {e: n for e, _, n in episodes(ep, fr)}
    anchor = np.asarray(w["anchor"])
    if not np.array_equal(ep[anchor], w["episode"]):
        raise WindowError("a window's recorded episode is not its anchor's")
    n = np.array([length[int(e)] for e in w["episode"]])
    if np.any(fr[anchor] > n - 1 - spec.drop_n_last_frames):
        raise WindowError("an anchor lies within the dropped last frames of its episode")
    for name, off in (("obs", spec.obs_offsets), ("action", spec.action_offsets)):
        rows, pad = np.asarray(w[f"{name}_rows"]), np.asarray(w[f"{name}_pad"])
        if rows.shape != (len(anchor), len(off)) or pad.shape != rows.shape:
            raise WindowError(f"{name} windows have shape {rows.shape}, not {(len(anchor), len(off))}")
        if np.any(ep[rows] != np.asarray(w["episode"])[:, None]):
            raise WindowError(f"an {name} slot draws a row of another episode")
        raw = fr[anchor][:, None] + off[None, :]
        if not np.array_equal(fr[rows], np.clip(raw, 0, n[:, None] - 1)) or not np.array_equal(
                pad, (raw < 0) | (raw > n[:, None] - 1)):
            raise WindowError(f"an {name} slot is not its clamped offset frame, or its padding flag is wrong")
    want = np.ones_like(w["action_pad"]) if spec.supervise_padding else ~np.asarray(w["action_pad"])
    if not np.array_equal(w["loss_mask"], want):
        raise WindowError("the loss mask does not follow the padding rule")


def digest(w: dict) -> str:
    """A window set by content: every array's name, dtype, shape and bytes, in name order."""
    h = hashlib.sha256()
    for k in sorted(w):
        a = np.ascontiguousarray(w[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode() + a.tobytes())
    return h.hexdigest()


def summary(w: dict) -> dict:
    return {"windows": int(len(w["anchor"])), "episodes": int(len(np.unique(w["episode"]))),
            "obs_padded_slots": int(np.sum(w["obs_pad"])), "action_padded_slots": int(np.sum(w["action_pad"])),
            "supervised_action_slots": int(np.sum(w["loss_mask"])), "sha256": digest(w)}
