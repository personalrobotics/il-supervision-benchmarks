#!/usr/bin/env python3
"""What each sigma of a study's grid produces under the task's exact dynamics: a diagnostic, never a selector.

    uv run python tools/sigma_geometry.py SPEC [--out FILE]

sigma is a hyperparameter of CCIL, selected per (r, K, seed) from validation data (ilsb.run.select); choosing it
from exact dynamics would make the learned method oracle-assisted. This tool trains nothing and computes no return.
On the specification's dataset, the study's generator (Newton preimage, K_max draws, the study's validators) runs
under the exact transition map at every sigma of the grid, with a non-study seed. It reports acceptance, rejections by
reason, and the displacement |z_G - z| in normalized task coordinates (each state coordinate over its standard
deviation across the demonstration states), beside the data's own spacing: the distance from each demonstration
state to the nearest state of another episode, and to the nearest other state at all.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ilsb.run import load_spec
from ilsb.store import sha256_file
from ilsb.tasks.pendulum import ExactModel, Task

SEED = 9401  # non-study


def nearest(states: np.ndarray, episode: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(distance to the nearest state of another episode, distance to the nearest other state) for every state."""
    other = np.empty(len(states))
    for e in np.unique(episode):
        mine = episode == e
        other[mine] = cKDTree(states[~mine]).query(states[mine])[0]
    anywhere = cKDTree(states).query(states, k=2)[0][:, 1]
    return other, anywhere


def quantiles(x: np.ndarray) -> dict:
    return {f"q{int(q * 100):02d}": float(np.quantile(x, q)) for q in (0.1, 0.5, 0.9)}


def main() -> int:
    from ccil.core import (ActionBounds, Finite, GenerationFailed, IdentityRepresentation, NoisyActionPreimage,
                           SupportRadius, generate_labels)
    from ccil.schema import read_trajectories

    ap = argparse.ArgumentParser()
    ap.add_argument("spec", type=Path)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    spec = load_spec(a.spec)
    data = Path(spec["data"]["path"])
    data = data if data.is_absolute() else a.spec.resolve().parent / data
    if sha256_file(data) != spec["data"]["sha256"]:
        raise SystemExit(f"refused: {data} is not the declared dataset")
    g, task = spec["generation"], Task(**spec["task"])
    k_max = max(spec["synthetic_doses"])
    ds = read_trajectories(data)
    model, rep = ExactModel(task), IdentityRepresentation("state")

    z = {e.episode_id: e.observations["state"].astype(np.float64) for e in ds.episodes}
    src = np.concatenate([s[:-1] for s in z.values()])
    ep = np.concatenate([np.full(len(s) - 1, i) for i, s in enumerate(z.values())])
    scale = src.std(axis=0)
    d_other, d_any = nearest(src / scale, ep)

    validators = [Finite(), GenerationFailed(), SupportRadius(g["support_radius"]),
                  ActionBounds([-task.action_bound], [task.action_bound])]
    rows = []
    for sigma in g["sigma_grid"]:
        c = generate_labels(ds, rep, model, NoisyActionPreimage(k_max, sigma, g["max_iterations"], g["tolerance"]),
                            validators, run_id="sigma-geometry", seed=SEED, inputs={"transition_model": model.sha256})
        origin = np.stack([z[e][f] for e, f in zip(c.source_episode_id.astype(str), c.source_frame)])
        dz = np.linalg.norm((c.observations["state"].astype(np.float64) - origin) / scale, axis=1)[c.accepted]
        rows.append({"sigma": sigma, "candidates": len(c.actions), "acceptance": float(c.accepted.mean()),
                     "feasible_at_min_acceptance": bool(c.accepted.mean() >= g["min_acceptance"]),
                     "rejection_by_reason": dict(zip(c.reason_codes, np.asarray(c.reasons).mean(axis=0).tolist())),
                     "displacement_norm": quantiles(dz) if len(dz) else None})
    from importlib import metadata
    direct = json.loads(metadata.distribution("ccil").read_text("direct_url.json") or "{}")
    ev = {"inputs": {"spec_sha256": sha256_file(a.spec), "dataset_sha256": spec["data"]["sha256"],
                     "transitions": len(src), "exact_model_sha256": model.sha256, "seed": SEED, "k_max": k_max,
                     "ccil_commit": (direct.get("vcs_info") or {}).get("commit_id"),
                     "tool_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()},
          "normalization": {"coordinates": ["sin_theta", "cos_theta", "theta_dot"], "scale": scale.tolist()},
          "spacing": {"nearest_state_of_another_episode": quantiles(d_other), "nearest_other_state": quantiles(d_any)},
          "grid": rows}
    text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
