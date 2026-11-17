"""#52 Stage B: the demonstrated-support threshold tau, from the r = 0.25 subset's demonstrations alone (read-only).

    LEROBOT_PYTHON tools/pusht_recovery/support.py --source SOURCE.npz [--out docs/studies/evidence/pusht-recovery/support.json]

A demonstration state is one source row: Diffusion Policy's two-step observation [frame k, frame k + 1], as 18 points
(the agent and the T's 8 keypoints, in each frame). The distance between two states is the largest Euclidean distance
between corresponding points, in px. For every subset state, the distance to the nearest state of the other 40
episodes is taken; tau is the 99th percentile of those distances. A learner-visited state farther than tau from every
subset state is outside the demonstrated support (pusht-recovery.md, Stage B). No policy, simulator, rollout or held-out
row is read. Float64 on the CPU, so a rerun on the same host is byte-identical.
"""
import argparse
import hashlib
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "pusht_v2"))
import source as source_module  # noqa: E402
from oracle_probe import subset  # noqa: E402

BASE = json.loads((ROOT / "tools/pusht_v2/surface.json").read_text())
SOURCE_CONTENT = "333a9b17ebfc6657591eb2b14a7f314200dd2ae1c743691802a319c1746e561d"
R, SUBSET_SEED, QUANTILE = 0.25, 0, 99
EXPECTED = {"subset_episodes": 41, "subset_states": 5040}
QS = (10, 50, 90, 99, 100)


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def points(state: np.ndarray, env: np.ndarray) -> np.ndarray:
    """(N, 4) agent history and (N, 32) keypoint history -> (N, 18, 2) points, float64."""
    return np.concatenate([state.reshape(-1, 2, 2), env.reshape(-1, 16, 2)], 1).astype(np.float64)


def nearest(q: np.ndarray, p: np.ndarray, chunk: int = 16) -> np.ndarray:
    out = np.empty(len(q))
    for i in range(0, len(q), chunk):
        out[i:i + chunk] = np.sqrt(((q[i:i + chunk, None] - p[None]) ** 2).sum(-1)).max(-1).min(1)
    return out


def held_out(job):
    e, pts, ep = job
    return nearest(pts[ep == e], pts[ep != e])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/pusht-recovery/support.json")
    p.add_argument("--workers", type=int, default=min(41, os.cpu_count() or 1))
    args = p.parse_args()
    if git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; the support record is made from a clean commit")
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if source_module.content_sha256(src) != SOURCE_CONTENT:
        sys.exit("refused: the source rows are not the protocol's by content")
    eps = subset(BASE["roles"]["train"], R, SUBSET_SEED)
    if set(eps) & set(BASE["roles"]["held_out"]):
        sys.exit("refused: the subset overlaps the held-out episodes")
    rows = np.flatnonzero(np.isin(src["episode"], eps))
    counts = {"subset_episodes": len(eps), "subset_states": len(rows)}
    if counts != EXPECTED:
        sys.exit(f"refused: the counts {counts} are not the protocol's")
    pts, ep = points(src["state"][rows], src["env"][rows]), src["episode"][rows]
    with ProcessPoolExecutor(args.workers) as ex:
        d = np.concatenate(list(ex.map(held_out, [(e, pts, ep) for e in eps])))
    out = {
        "schema": "ilsb.pusht_recovery.support 1",
        "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                 "files_sha256": {f: sha(ROOT / f) for f in ("tools/pusht_recovery/support.py", "tools/pusht_v2/source.py",
                                                             "tools/pusht_v2/oracle_probe.py", "tools/pusht_v2/surface.json")}},
        "inputs": {"source_content_sha256": SOURCE_CONTENT, "r": R, "subset_seed": SUBSET_SEED, "subset_episodes": eps},
        "counts": counts,
        "state": "18 points: agent and 8 T keypoints, frames k and k + 1 (one source row)",
        "distance": "largest per-point Euclidean distance, px",
        "held_out_episode_nearest_px": {str(q): float(v) for q, v in zip(QS, np.percentile(d, QS))},
        "tau_px": float(np.percentile(d, QUANTILE)),
        "tau_rule": f"the {QUANTILE}th percentile of each subset state's distance to the nearest state of the other episodes",
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "tau_px": out["tau_px"], "quantiles": out["held_out_episode_nearest_px"]}))


if __name__ == "__main__":
    main()
