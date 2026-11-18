"""#70 protocol input: PushT's tau_demo on the current frame, by #52 B0's rule (read-only, recovery-feasibility.md).

    LEROBOT_PYTHON tools/recovery_feasibility/tau.py --source SOURCE.npz [--out docs/studies/evidence/recovery-feasibility/tau.json]

B0's tau (29.764 px, support.json) was measured on the 18-point two-frame state. A recovery segment cannot move the
history frame, so the study's terminal reach, radius and M = 1 certificate use the 9 current-frame points (the agent
and the T's 8 keypoints at frame t), and tau is refrozen on them by the same rule: the 99th percentile of one
demonstration step of the r = 0.25 subset plus the largest replay error over its replay-verified states. The replay
error is B0's, already a current-frame distance. Only the subset's demonstrations are read; no start is solved.
"""
import argparse
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/pusht_recovery"))
import support as S  # noqa: E402

A = S.A
CURRENT = [1, *range(10, 18)]  # of S.points: 0 prev agent, 1 agent, 2-9 prev T keypoints, 10-17 T keypoints
B0_TAU_18 = 29.764080044321236


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/recovery-feasibility/tau.json")
    p.add_argument("--workers", type=int, default=min(41, os.cpu_count() or 1))
    args = p.parse_args()
    if git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; the tau record is made from a clean commit")
    A.simulator_check()
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if A.source_module.content_sha256(src) != S.SOURCE_CONTENT:
        sys.exit("refused: the source rows are not the protocol's by content")
    eps = A.subset(A.BASE["roles"]["train"], S.R, S.SUBSET_SEED)
    if set(eps) & set(A.BASE["roles"]["held_out"]):
        sys.exit("refused: the subset overlaps the held-out episodes")
    rows = np.flatnonzero(np.isin(src["episode"], eps))
    counts = {"subset_episodes": len(eps), "subset_states": len(rows)}
    if counts != S.EXPECTED:
        sys.exit(f"refused: the counts {counts} are not the protocol's")
    pts, ep = S.points(src["state"][rows], src["env"][rows]), src["episode"][rows]

    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    act = src["action"].astype(np.float64)
    with ProcessPoolExecutor(args.workers) as ex:
        err = np.concatenate(list(ex.map(S.replay_errors, [(cur[rows[ep == e]], act[rows[ep == e]]) for e in eps])))
    verified = err <= A.REPLAY_TOL

    step18 = np.concatenate([S.dist(pts[ep == e][1:], pts[ep == e][:-1]) for e in eps])
    tau18 = float(np.percentile(step18, S.STEP_QUANTILE) + err[verified].max())
    if tau18 != B0_TAU_18:
        sys.exit(f"refused: B0's rule gives {tau18!r} on 18 points, not support.json's {B0_TAU_18!r}")
    p9 = pts[:, CURRENT]
    step9 = np.concatenate([S.dist(p9[ep == e][1:], p9[ep == e][:-1]) for e in eps])
    tau9 = float(np.percentile(step9, S.STEP_QUANTILE) + err[verified].max())

    out = {
        "schema": "ilsb.recovery_feasibility.tau 1",
        "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                 "files_sha256": {f: A.sha(ROOT / f) for f in (
                     "tools/recovery_feasibility/tau.py", "tools/pusht_recovery/support.py",
                     "tools/pusht_recovery/stage_a.py", "tools/pusht_v2/source.py")}},
        "simulator": A.SPEC["simulator"],
        "inputs": {"source_content_sha256": S.SOURCE_CONTENT, "r": S.R, "subset_seed": S.SUBSET_SEED,
                   "subset_episodes": eps},
        "counts": counts,
        "state": "9 points: the agent and the T's 8 keypoints at the current frame",
        "distance": "largest per-point Euclidean distance, px",
        "rule": f"the {S.STEP_QUANTILE}th percentile of one demonstration step plus the largest replay error over "
                "the replay-verified states (#52 B0)",
        "demonstration_step_px": S.quantiles(step9),
        "replay_verified_states": int(verified.sum()),
        "replay_error_max_px": float(err[verified].max()),
        "b0_tau_18_point_px": tau18,
        "tau_px": tau9,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("tau_px", "b0_tau_18_point_px", "replay_verified_states")}))


if __name__ == "__main__":
    main()
