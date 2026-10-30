"""The Pendulum task adapter: transition map, evaluation starts, real-data subsets, scorer and an exact model.

The task is the evaluator's discrete transition map (pendulum_map.step), with the commanded action clipped to the
task's action bound before it is applied. Demonstrations are ccil.trajectories files (the format of the canonical
Pendulum datasets); everything here is task semantics, with no supervision method and no learner.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from . import pendulum_map

UPRIGHT_COS = float(np.cos(np.deg2rad(8.0)))  # dm_control's swing-up criterion: within 8 degrees of upright
HOLD_STEPS = 100  # the final two seconds


@dataclass(frozen=True)
class Task:
    variant: str = "continuous"  # or "discontinuous": velocity flips near 30 and 60 degrees from hanging down
    action_bound: float = 3.0
    horizon: int = 500
    eval_episodes: int = 100
    eval_seed: int = 2201  # fixed final-test starts, shared by every unit

    def __post_init__(self):
        if self.variant not in ("continuous", "discontinuous"):
            raise ValueError(f"variant must be continuous or discontinuous, got {self.variant!r}")
        if not self.action_bound > 0 or self.horizon < HOLD_STEPS or self.eval_episodes < 1:
            raise ValueError("action_bound > 0, horizon >= 100 and eval_episodes >= 1 are required")

    def identity(self) -> dict:
        return {"task": "pendulum", "map": "ilsb.tasks.pendulum_map/1 (CCIL tools/vector_env/pendulum.py port)",
                **asdict(self)}

    def step(self, states: np.ndarray, actions: np.ndarray):
        """(next states, flipped, applied actions) for N states (N, 3) and commanded actions (N,) or (N, 1)."""
        applied = np.clip(np.asarray(actions, np.float64).reshape(len(states)), -self.action_bound, self.action_bound)
        nxt, flipped = pendulum_map.step(np.asarray(states, np.float64), applied, self.variant == "discontinuous")
        return nxt, flipped, applied

    def starts(self) -> np.ndarray:
        """The fixed final-test starts: theta ~ U(-1, 1) about hanging down, theta_dot ~ U(-0.5, 0.5)."""
        d = np.random.default_rng(np.random.SeedSequence([self.eval_seed, 7])).uniform(
            [-1.0, -0.5], [1.0, 0.5], size=(self.eval_episodes, 2))
        return np.stack([np.sin(d[:, 0]), np.cos(d[:, 0]), d[:, 1]], axis=1)

    def rollouts(self, policies: list) -> list[dict]:
        """Every policy (a callable on (N, 3) observations) from the fixed starts, all stepped together."""
        P, E, T = len(policies), self.eval_episodes, self.horizon
        states = np.zeros((P, E, T + 1, 3))
        states[:, :, 0] = self.starts()
        commanded, applied = np.zeros((P, E, T)), np.zeros((P, E, T))
        flipped = np.zeros((P, E, T), bool)
        for t in range(T):
            for i, pol in enumerate(policies):
                commanded[i, :, t] = np.asarray(pol(states[i, :, t]), np.float64).reshape(E)
            if not np.isfinite(commanded[:, :, t]).all():
                raise FloatingPointError(f"a policy commanded a non-finite action at step {t}")
            nxt, fl, ap = self.step(states[:, :, t].reshape(-1, 3), commanded[:, :, t].reshape(-1))
            states[:, :, t + 1], flipped[:, :, t], applied[:, :, t] = nxt.reshape(P, E, 3), fl.reshape(P, E), \
                ap.reshape(P, E)
        return [{"states": states[i], "commanded": commanded[i], "applied": applied[i], "flipped": flipped[i]}
                for i in range(P)]

    def score(self, rollout: dict) -> dict:
        """Per-seed task metrics from a rollout: means over its episodes of the per-episode values."""
        s, a = rollout["states"][:, :-1], rollout["applied"]
        v = np.cos(np.arctan2(s[..., 0], s[..., 1]) - np.pi)
        up = v >= UPRIGHT_COS
        return {"upright_occupancy": float(up.mean(axis=1).mean()),  # per episode, then over episodes (CCIL's order)
                "normalized_height": float(((1 + v) / 2).mean(axis=1).mean()),
                "final_hold": float(up[:, -HOLD_STEPS:].all(axis=1).mean()),
                "normalized_effort": float(((a / self.action_bound) ** 2).mean(axis=1).mean()),
                "saturated_fraction": float((np.abs(rollout["commanded"]) >= self.action_bound).mean())}


class ExactModel:
    """The task's own transition map as a transition model: delta(z, a) = F(z, a) - z (the exact-dynamics arm)."""

    def __init__(self, task: Task):
        self.task = task
        self.sha256 = hashlib.sha256(json.dumps({"model": "exact", **task.identity()}, sort_keys=True).encode()
                                     ).hexdigest()

    def delta(self, z: np.ndarray, a: np.ndarray) -> np.ndarray:
        nxt, _, _ = self.task.step(z, np.asarray(a, np.float64).reshape(len(z)))
        return nxt - np.asarray(z, np.float64)


def subset(dataset: Path, fraction: float, seed: int, out: Path) -> dict:
    """The first round(fraction * N) episodes of SeedSequence([seed, 9])'s permutation, written to out: nested in
    fraction for a fixed seed (v6's subset rule). Returns the subset's identity."""
    from ccil.schema import TrajectoryDataset, read_trajectories, write_trajectories
    ds = read_trajectories(dataset)
    n = len(ds.episodes)
    k = round(fraction * n)
    if not 1 <= k <= n:
        raise ValueError(f"fraction {fraction} of {n} episodes keeps {k}")
    keep = np.random.default_rng(np.random.SeedSequence([seed, 9])).permutation(n)[:k]
    write_trajectories(TrajectoryDataset([ds.episodes[i] for i in keep], {**ds.meta, "subset": {
        "fraction": fraction, "seed": seed, "episodes": [int(i) for i in keep],
        "rule": "first round(fraction * N) of default_rng(SeedSequence([seed, 9])).permutation(N)"}}), out)
    return {"episodes": int(k), "of": n}


# --- the corrected demonstrator, for compact fixtures only (studies use canonical datasets by hash) ---------------
def expert(states: np.ndarray, bound: float = 3.0) -> np.ndarray:
    """The corrected Pendulum demonstrator (CCIL tools/pendulum_eval/expert.py, primary variant): LQR within 0.1 rad
    of upright, energy shaping elsewhere, clamped."""
    theta = np.arctan2(states[..., 0], states[..., 1])
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    delta, w = theta - np.pi, states[..., 2]
    lqr, energy = -20.11 * delta - 7.08 * w, -w * (0.5 * w ** 2 - 9.81 * np.cos(theta) - 9.81)
    return np.clip(np.where(np.abs(delta) < 0.1, lqr, energy), -bound, bound)


def demonstrations(task: Task, episodes: int, steps: int, seed: int, out: Path) -> None:
    """A compact demonstration file from the corrected demonstrator under the task's map (float32 actions, applied
    as stored)."""
    from ccil.schema import Episode, TrajectoryDataset, write_trajectories
    d = np.random.default_rng(np.random.SeedSequence([seed, 7])).uniform([-1.0, -0.5], [1.0, 0.5], size=(episodes, 2))
    s = np.stack([np.sin(d[:, 0]), np.cos(d[:, 0]), d[:, 1]], axis=1)
    obs, act = [s], []
    for _ in range(steps):
        a = expert(s, task.action_bound).astype(np.float32)
        s, _, _ = task.step(s, a.astype(np.float64))
        obs.append(s)
        act.append(a)
    obs, act = np.stack(obs, 1), np.stack(act + [np.zeros(episodes, np.float32)], 1)
    write_trajectories(TrajectoryDataset([Episode(f"e{i}", {"state": obs[i]}, act[i][:, None]) for i in range(episodes)],
                                         {"generator": "ilsb fixture demonstrations", "seed": seed}), out)
