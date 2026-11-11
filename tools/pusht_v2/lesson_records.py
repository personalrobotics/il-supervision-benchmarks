"""Compact records for the PushT coordinates lesson (docs/studies/pusht-keypoints-vs-pose.md), from two label runs.

    uv run python tools/pusht_v2/lesson_records.py KEYPOINT_LABELS POSE_LABELS OUT_JSON

KEYPOINT_LABELS is the v2 smoke run's labels directory when CCIL generated on the keypoint history (root
20261111T135807-pusht-v2-smoke, ilsb f750f98); POSE_LABELS is the pose probe's run directory (root
20261111T150215-pose-probe, ilsb 57244ff). Both hold a CCIL runs/ tree (prepared dataset, structured checkpoint,
noisy_action_fe candidates) from the same 100,952 proposals. Records:
- examples: three accepted labels from each, drawn with seed 0, with the previous-frame T of the label and of its
  source frame as 8 keypoints (pose labels decoded with pose.keypoints);
- conditioning: percentiles of the condition number of d(newest next slot)/d(previous slot) at the source states of
  2,000 proposals drawn with seed 0 (18 x 18 for keypoints, 5 x 5 for pose).
"""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose  # noqa: E402

SPACES = {"keypoint": {"names": ("state", "state.environment_state"), "prev": np.r_[0:2, 4:20], "new": np.r_[2:4, 20:36]},
          "pose": {"names": ("state", "state.pose"), "prev": np.r_[0:2, 4:7], "new": np.r_[2:4, 7:10]}}
PCT = (10, 25, 50, 75, 90, 100)


def load(labels: Path, names):
    from ccil.models.mlp import MLPTransitionModel
    from ccil.schema import read_candidates, read_trajectories
    runs = labels / "runs"
    ds = read_trajectories(next(runs.glob("data-prepare-*")) / "data" / "dataset.npz")
    c = read_candidates(next(runs.glob("labels-generate-*")) / "labels" / "candidates.npz", ds)
    model = MLPTransitionModel.load(next(runs.glob("transition-fit-*")) / "checkpoints" / "transition_model.npz")
    z = np.concatenate([np.asarray(c.observations[n], np.float64) for n in names], 1)
    by_id = {e.episode_id: np.concatenate([e.observations[n] for n in names], 1) for e in ds.episodes}
    src = np.stack([by_id[e][f] for e, f in zip(np.asarray(c.source_episode_id), np.asarray(c.source_frame))])
    return z, src.astype(np.float64), np.asarray(c.actions, np.float64), np.asarray(c.accepted, bool), model


def previous_t(space, rows):
    """The previous-frame T of state rows as (n, 8, 2) keypoints."""
    kp = rows[:, 4:20] if space == "keypoint" else pose.keypoints(rows[:, 4:7])
    return kp.reshape(len(rows), 8, 2)


def main(kp_labels: Path, pose_labels: Path, out: Path) -> None:
    doc = {"sources": {"keypoint": "root 20261111T135807-pusht-v2-smoke (ilsb f750f98, CCIL a995ff9)",
                       "pose": "root 20261111T150215-pose-probe (ilsb 57244ff, CCIL a995ff9)"},
           "examples": {}, "conditioning": {}}
    for space, labels in (("keypoint", kp_labels), ("pose", pose_labels)):
        s = SPACES[space]
        z, src, a, acc, model = load(labels, s["names"])
        rng = np.random.default_rng(0)
        pick = np.sort(rng.choice(np.flatnonzero(acc), 3, replace=False))
        doc["examples"][space] = [{"candidate_row": int(i), "label_previous_t": previous_t(space, z[i:i + 1])[0].tolist(),
                                   "source_previous_t": previous_t(space, src[i:i + 1])[0].tolist()} for i in pick]
        sample = np.random.default_rng(0).choice(len(z), 2000, replace=False)
        J = model.jacobian_z(src[sample], a[sample])[:, s["new"]][:, :, s["prev"]]
        sv = np.linalg.svd(J, compute_uv=False)
        cond = sv[:, 0] / sv[:, -1]
        doc["conditioning"][space] = {"size": int(len(s["prev"])), "percentiles": {str(p): float(np.percentile(cond, p))
                                                                                  for p in PCT}}
        print(space, doc["conditioning"][space])
    out.write_text(json.dumps(doc, indent=1) + "\n")


if __name__ == "__main__":
    main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]))
