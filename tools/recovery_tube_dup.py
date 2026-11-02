#!/usr/bin/env python3
"""The recovery-tube study's matched replay-CCIL and duplication pair (docs/studies/pendulum-recovery-tube.md,
decision 3 only, as amended by amendment 3).

    cd CCIL_WORKTREE_AT_6fbcce5 && uv run --extra torch python IL_SUPERVISION_BENCHMARKS/tools/recovery_tube_dup.py \\
        --audit OUT/audit.json --decision OUT/decision.json --out OUT/matched

It runs only under ccil at the corrected study's frozen commit 6fbcce5 (checked), and only after decision 2 found a
signal. For each training seed it rebuilds the CCIL arm's supervision from that unit's reconciled dataset and
candidates, read-only, and requires it to reproduce both the unit's recorded supervision manifest and the training
identity recorded in its checkpoint: that rebuilt supervision is the replay-CCIL arm. The duplication view replaces
every synthetic row's observation and action with its source demonstration row's, keeping row count, order, weights,
source labels (so the sampler's exposure is unchanged), episode, frame and candidate row. Generation-describing fields
(targets, residuals, measurements, certificates) are never copied: the trainer's handoff is (observations, actions,
weights) plus the source labels. The candidate files are never written. Both arms of every seed train in one
train_group call (replay jobs, then duplication jobs; 40 jobs, bucket 64 as in the original run), with the CCIL arm's
settings, seeds (so initialization and shuffling match within a seed), device and determinism flag. duplication_view
and selftest import no ccil, so the harness selftest exercises them.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

FROZEN_CCIL = "6fbcce543e8a1ebdf243063c179dd3af82d5db8a"
SEEDS = tuple(range(1201, 1221))
KEPT = ["weights", "source", "episode_id", "frame", "candidate_row", "classification"]
DROPPED = ["target_observations", "measurements", "measurement_specs", "reasons", "certificates"]
VIEW_RULE = "every synthetic row's observation and action replaced by its source demonstration row's"


def duplication_view(source: np.ndarray, episode_id: np.ndarray, frame: np.ndarray) -> np.ndarray:
    """For every supervision row, the row whose observation and action it takes: itself for a demonstration row; for
    a synthetic row, the demonstration row with the same (episode, frame). KeyError if a source is not a training
    demonstration row."""
    demo = np.flatnonzero(source == "demonstration")
    key = {(str(e), int(f)): int(i) for i, e, f in zip(demo, episode_id[demo], frame[demo])}
    idx = np.arange(len(source))
    syn = np.flatnonzero(source == "synthetic")
    idx[syn] = [key[(str(e), int(f))] for e, f in zip(episode_id[syn], frame[syn])]
    return idx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_npz(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


def selftest() -> dict:
    """Synthetic: two episodes, three demonstration rows each, two synthetic rows per source; plus a candidate-like
    file that the view must leave byte-identical."""
    ep = np.array(["a"] * 3 + ["b"] * 3 + ["a", "a", "b", "b"])
    fr = np.array([0, 1, 2, 0, 1, 2, 1, 1, 2, 0])
    src = np.array(["demonstration"] * 6 + ["synthetic"] * 4)
    obs = np.arange(30, dtype=float).reshape(10, 3)
    act = np.arange(10, dtype=float)[:, None]
    w = np.array([1.0] * 6 + [0.5, 0.25, 0.5, 0.75])
    with tempfile.TemporaryDirectory() as td:
        cand = Path(td) / "candidates.npz"
        np.savez(cand, observations=obs[6:] + 0.01, target=obs[6:], meas=np.ones(4))
        before = sha256_file(cand)
        np.load(cand)["observations"]  # the view reads its source; it never writes it
        idx = duplication_view(src, ep, fr)
        unchanged = sha256_file(cand) == before
    view = {"observations": obs[idx], "actions": act[idx], "weights": w, "source": src}
    want = [0, 1, 2, 3, 4, 5, 1, 1, 5, 3]
    ok = {"source_rows": idx.tolist() == want,
          "synthetic_rows_equal_their_sources": bool((view["observations"][6:] == obs[[1, 1, 5, 3]]).all()
                                                     and (view["actions"][6:] == act[[1, 1, 5, 3]]).all()),
          "count_order_weights_sources_kept": bool(len(idx) == 10 and (view["weights"] == w).all()
                                                   and (view["source"] == src).all()),
          "no_generation_fields": not set(view) & {"target", "target_observations", "meas", "measurements"},
          "source_file_unchanged": unchanged}
    try:
        duplication_view(np.array(["synthetic"]), np.array(["z"]), np.array([0]))
        ok["unknown_source_refused"] = False
    except KeyError:
        ok["unknown_source_refused"] = True
    return {**ok, "pass": all(ok.values())}


def main() -> int:
    import ccil
    from ccil.core import IdentityRepresentation
    from ccil.pipelines.policy_harness import settings_of
    from ccil.supervision import Supervision, build_supervision
    from ccil.training import vector_bc_v2
    from ccil.training.vector_bc import VectorBCPolicy

    ap = argparse.ArgumentParser()
    ap.add_argument("--audit", type=Path, required=True)
    ap.add_argument("--decision", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", help="off-protocol smoke only; the study uses the CCIL arm's compute.device")
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="off-protocol smoke only")
    a = ap.parse_args()
    src_root = Path(ccil.__file__).resolve().parents[2]
    git = lambda *x, cwd=src_root: subprocess.run(["git", *x], cwd=cwd, capture_output=True, text=True,  # noqa: E731
                                                  check=True).stdout.strip()
    if git("rev-parse", "HEAD") != FROZEN_CCIL or git("status", "--porcelain", "src"):
        raise SystemExit(f"refused: ccil at {src_root} is not a clean checkout of {FROZEN_CCIL}")
    if json.loads(a.decision.read_text()).get("signal", {}).get("outcome") != "signal":
        raise SystemExit("refused: decision 2 found no signal; the matched pair is not trained")
    aud = json.loads(a.audit.read_text())["corrected"]
    if not aud["eligible"]:
        raise SystemExit("refused: the corrected cohort is not eligible")
    if (a.out / "manifest.json").exists():
        raise SystemExit(f"refused: {a.out}/manifest.json exists")
    (a.out / "views").mkdir(parents=True, exist_ok=True)
    tool = git("rev-parse", "HEAD", cwd=Path(__file__).resolve().parent)
    replay, dups, settings, views, flags, devices, idents = [], [], None, {}, set(), set(), {}
    seeds = tuple(a.seeds)
    for s in seeds:
        u = aud["units"][f"continuous/ccil/s{s}"]
        stage = Path(u["stage"])
        v = {k: d["value"] for k, d in json.loads((stage / "config.json").read_text())["values"].items()}
        need = {"policy.method": "ccil", "certificate.profile": None, "policy.exclude_validation_episodes": False,
                "policy.checkpoint_selection": "last", "policy.trainer_version": "2", "run.seed": s}
        if any(v.get(k) != want for k, want in need.items()):
            raise SystemExit(f"refused: s{s}'s stage configuration is not the one this view reproduces")
        if sha256_file(Path(u["candidates"])) != u["candidates_sha256"]:
            raise SystemExit(f"refused: s{s} candidates changed since the audit")
        sup = build_supervision(Path(u["dataset"]), "ccil", representation=IdentityRepresentation(
            v["representation.modality"]), candidates=Path(u["candidates"]),
            demonstration_weight=v["policy.demonstration_weight"])
        if sup.manifest() != json.loads((stage / "supervision.json").read_text()):
            raise SystemExit(f"refused: s{s}'s rebuilt supervision differs from its recorded manifest")
        st = settings_of(v)
        meta = VectorBCPolicy.load(u["checkpoint"])[1]
        if vector_bc_v2.training_identity_v2(sup, st, s, None, None)["sha256"] != meta["training_identity_sha256"]:
            raise SystemExit(f"refused: s{s}'s rebuilt supervision does not reproduce the CCIL training identity")
        if settings is not None and st != settings:
            raise SystemExit("refused: the CCIL units' training settings differ")
        settings = st
        flags.add(bool(v["compute.deterministic"]))
        devices.add(v["compute.device"])
        idents[s] = meta["training_identity_sha256"]
        idx = duplication_view(sup.source, sup.episode_id, sup.frame)
        vp = a.out / "views" / f"s{s}.npz"
        atomic_npz(vp, observations=sup.observations[idx], actions=sup.actions[idx], source_row=idx,
                   **{k: getattr(sup, k) for k in KEPT})
        man = {**sup.manifest(), "duplication_view": {"rule": VIEW_RULE, "view_sha256": sha256_file(vp)}}
        construction = {"seed": s, "source_candidates_sha256": u["candidates_sha256"],
                        "source_checkpoint_sha256": u["checkpoint_sha256"], "view_sha256": man["duplication_view"][
                            "view_sha256"], "fields_kept": ["observations (replaced)", "actions (replaced)", *KEPT],
                        "fields_dropped": DROPPED, "synthetic_rows": int((sup.source == "synthetic").sum()),
                        "rows": int(len(idx)), "ccil_commit": FROZEN_CCIL, "tool_commit": tool}
        atomic_json(a.out / "views" / f"s{s}.json", construction)
        views[s] = construction
        f = np.load(vp)
        view = Supervision(f["observations"], f["actions"], f["weights"], f["source"], f["episode_id"], f["frame"],
                           f["candidate_row"], f["classification"], man)
        replay.append(vector_bc_v2.Job(sup, s, None, None))
        dups.append(vector_bc_v2.Job(view, s, None, None))
        print("view", s, json.dumps(construction), flush=True)
    if len(flags) != 1 or len(devices) != 1:
        raise SystemExit("refused: the CCIL units differ in compute.deterministic or compute.device")
    device = a.device or devices.pop()
    trained = vector_bc_v2.train_group(replay + dups, settings, device=device, deterministic=flags.pop())
    pols, buckets = {"replay_ccil": {}, "duplication": {}}, set()
    for (arm, s, job), (policy, record) in zip([("replay_ccil", s, j) for s, j in zip(seeds, replay)]
                                               + [("duplication", s, j) for s, j in zip(seeds, dups)], trained):
        if arm == "replay_ccil" and record["training_identity_sha256"] != idents[s]:
            raise SystemExit(f"refused: replay s{s} is not the CCIL unit's training problem")
        d = a.out / "policies" / arm / f"s{s}"
        d.mkdir(parents=True, exist_ok=True)
        policy.save(d / "policy.npz", None, record["epochs_run"],
                    {"supervision": job.supervision.manifest()["inputs"], "seed": s, "method": arm,
                     "trainer_version": record.get("trainer_version", "1"),
                     "execution_sha256": record.get("execution", {}).get("execution_sha256"),
                     "training_identity_sha256": record["training_identity_sha256"]})
        atomic_json(d / "training.json", record)
        bucket = record.get("execution", {}).get("bucket")
        buckets.add(bucket)
        pols[arm][str(s)] = {"sha256": sha256_file(d / "policy.npz"), "bucket": bucket,
                             "training_identity_sha256": record["training_identity_sha256"],
                             **({"view_sha256": views[s]["view_sha256"]} if arm == "duplication" else {})}
    atomic_json(a.out / "manifest.json", {
        "policies": pols, "settings": settings, "device": device, "ccil_commit": FROZEN_CCIL, "tool_commit": tool,
        "seeds": list(seeds), "buckets": sorted(buckets),
        "group": "one train_group call: replay_ccil jobs in seed order, then duplication jobs in seed order",
        "note": "the original CCIL checkpoints trained in the runner's mixed group and were not bitwise deterministic; "
                "original CCIL - replay_ccil is reported as retraining drift, and attribution is replay_ccil - "
                "duplication"})
    print("trained", sum(len(v) for v in pols.values()), "matched policies, buckets", sorted(buckets), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
