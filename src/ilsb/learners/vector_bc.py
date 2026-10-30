"""The vector_bc_v2 learner adapter: CCIL's batched policy trainer, the single learner of the first study.

Every unit of one (arm, r, K) cell trains in one group (identical settings; each seed its own supervision, shuffle
and initialization). Validation-based checkpoint selection, patience termination, the synthetic sampling fraction
and the realized-exposure record are the trainer's own (ccil.training.vector_bc_v2 at the pinned commit). Resume is
per unit: a unit whose policy entry verifies is never retrained.
"""
from __future__ import annotations

import json
from pathlib import Path


def train(jobs: list[tuple], settings: dict, device: str) -> list[tuple]:
    """jobs: (train supervision, validation supervision, seed). Returns [(policy, record)] in job order."""
    from ccil.training import vector_bc_v2 as v2
    return v2.train_group([v2.Job(t, s, validation=v) for t, v, s in jobs], settings, device=device,
                          deterministic=True)


def save(policy, record: dict, d: Path) -> None:
    policy.save(d / "policy.npz")
    (d / "training.json").write_text(json.dumps(record, sort_keys=True, default=str) + "\n")


def load(d: Path, device: str = "cpu"):
    """The policy's action function on (N, D) observations."""
    from ccil.training.vector_bc import VectorBCPolicy
    return VectorBCPolicy.load(d / "policy.npz", device=device)[0].predict
