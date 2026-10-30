"""The CCIL supervision adapter: dynamics fits, one K_max candidate set per (subset, dynamics), nested K views, and
the training and validation supervision of each arm, all through CCIL's public API at the pinned commit.

Arms: "bc" (the demonstrations; K = 0), "ccil_learned" (labels under a transition model fitted on the subset's
training episodes) and "ccil_exact" (labels under the task's own transition map). Every artifact is a store entry
whose identity names its inputs by hash.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from ..store import Store, sha256_file

ARMS = ("bc", "ccil_learned", "ccil_exact")
REPRESENTATION = "state"


def _rep():
    from ccil.core import IdentityRepresentation
    return IdentityRepresentation(REPRESENTATION)


def fit_dynamics(store: Store, subset: Path, split: Path, settings: dict, seed: int, device: str) -> Path:
    """A transition model fitted on the subset's training episodes (ccil.models.mlp_batched.fit_group)."""
    import json

    from ccil.core import transitions
    from ccil.models.mlp_batched import FitJob, fit_group
    from ccil.schema import read_trajectories
    ident = {"kind": "dynamics", "subset": sha256_file(subset), "split": sha256_file(split), "settings": settings,
             "seed": seed, "fit_episodes": "train", "device": device}

    def write(d: Path):
        batch = transitions(read_trajectories(subset), _rep())
        train = set(json.loads(split.read_text())["train"])
        keep = np.array([e in train for e in batch.episode_id])
        [(model, record)] = fit_group([FitJob(batch.z[keep], batch.a[keep], batch.z_next[keep], seed)], settings,
                                      device=device, deterministic=True)
        model.save(d / "model.npz")
        (d / "fit.json").write_text(json.dumps(record, sort_keys=True, default=str) + "\n")
    return store.ensure("dynamics", ident, write)[0]


def generate(store: Store, subset: Path, model, model_sha256: str, generation: dict, seed: int) -> Path:
    """The K_max candidate set over the whole subset (the split later keeps the training episodes' rows)."""
    from ccil.core import (ActionBounds, Finite, GenerationFailed, NoisyAction, NoisyActionPreimage, SupportRadius,
                           generate_labels)
    from ccil.schema import read_trajectories, write_candidates
    ident = {"kind": "candidates", "subset": sha256_file(subset), "transition_model": model_sha256,
             "generation": generation, "seed": seed}

    def write(d: Path):
        g = generation
        gen = (NoisyActionPreimage(g["k_max"], g["action_noise_std"], max_iterations=g["max_iterations"],
                                   tolerance=g["tolerance"])
               if g["generator"] == "noisy_action_preimage" else NoisyAction(g["k_max"], g["action_noise_std"]))
        validators = [Finite(), GenerationFailed(), SupportRadius(g["support_radius"]),
                      ActionBounds([-g["action_bound"]], [g["action_bound"]])]
        cands = generate_labels(read_trajectories(subset), _rep(), model, gen, validators, run_id="candidates",
                                seed=seed, inputs={"transition_model": model_sha256, "dataset": sha256_file(subset)})
        write_candidates(cands, d / "candidates.npz")
    return store.ensure("candidates", ident, write)[0]


def view(store: Store, candidates: Path, k: int) -> Path:
    """The candidates of a generation at labels_per_transition = k, as a nested view of the K_max set."""
    from ccil.core import nested_view
    from ccil.schema import read_candidates, write_candidates
    parent = sha256_file(candidates)
    ident = {"kind": "view", "candidates": parent, "k": k}
    return store.ensure("view", ident, lambda d: write_candidates(nested_view(read_candidates(candidates), k, parent),
                                                                  d / "candidates.npz"))[0]


def supervision(arm: str, subset: Path, split: Path, candidates: Path | None, part: str):
    """The arm's supervision for a split part: demonstrations, plus the view's usable labels for CCIL arms."""
    from ccil.supervision import build_supervision
    method = "bc" if arm == "bc" or part == "validation" else "ccil"
    return build_supervision(subset, method, representation=_rep(), split=split, part=part,
                             candidates=candidates if method == "ccil" else None)
