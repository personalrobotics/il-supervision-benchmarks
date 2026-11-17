"""#52 Stage B0: deviation geometry from the r = 0.25 subset's demonstrations alone (read-only, pusht-recovery.md).

    LEROBOT_PYTHON tools/pusht_recovery/support.py --source SOURCE.npz [--out docs/studies/evidence/pusht-recovery/support.json]

A demonstration state is one source row: Diffusion Policy's two-step observation [frame k, frame k + 1], as 18 points
(the agent and the T's 8 keypoints, in each frame). The distance between two states is the largest Euclidean distance
between corresponding points, in px.

Two things are recorded.
- The rejected rule: cross-episode proximity. Each subset state against the nearest state of the other 40 episodes,
  and the two episodes' states one executed chunk (8 steps) later. This measures ordinary trajectory diversity, not
  phase or failure.
- The adopted rule: paired-prefix deviation. A rollout restored at a demonstration's recorded state is compared with
  the same episode's continuation. The deviation threshold tau is the 99th percentile of one demonstration step (the
  distance between consecutive states of an episode) plus the largest replay error over the replay-verified states.
  The anchors are the replay-verified rows at every eighth row of each episode, from row 8 to row n - 25, that are
  not yet a success and whose recorded continuation (the later rows) reaches success. Success is gym_pusht's own:
  the T's coverage of the goal above 0.95, computed from each row's recorded T pose by the environment's coverage.

No policy, rollout, held-out row or evaluation start is read. Distances are float64 on the CPU, and the replay
is gym_pusht's deterministic simulator, so a rerun on the same host is byte-identical.
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
sys.path.insert(0, str(HERE))
import stage_a as A  # noqa: E402

SOURCE_CONTENT = A.SPEC["inputs"]["source_content_sha256"]
R, SUBSET_SEED = 0.25, 0
CHUNK, STEP_QUANTILE, SUCCESS = 8, 99, 0.95
ANCHOR = {"stride": 8, "first_row": 8, "rows_before_end": 25}
EXPECTED = {"subset_episodes": 41, "subset_states": 5040}
QS = (10, 50, 90, 99, 100)


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def points(state: np.ndarray, env: np.ndarray) -> np.ndarray:
    """(N, 4) agent history and (N, 32) keypoint history -> (N, 18, 2) points, float64."""
    return np.concatenate([state.reshape(-1, 2, 2), env.reshape(-1, 16, 2)], 1).astype(np.float64)


def dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.sqrt(((a - b) ** 2).sum(-1)).max(-1)


def quantiles(x: np.ndarray) -> dict:
    return {str(q): float(v) for q, v in zip(QS, np.percentile(x, QS))}


def cross_episode(job):
    """For one episode's states: the nearest other-episode state, and both episodes' states CHUNK steps later."""
    e, pts, ep, pos, n_of = job
    ok = pos + CHUNK < np.array([n_of[x] for x in ep])
    q, o = np.flatnonzero((ep == e) & ok), np.flatnonzero((ep != e) & ok)
    d0, d8 = np.empty(len(q)), np.empty(len(q))
    for i in range(0, len(q), 16):
        qi = q[i:i + 16]
        d = np.sqrt(((pts[qi, None] - pts[None, o]) ** 2).sum(-1)).max(-1)
        j = o[d.argmin(1)]
        d0[i:i + 16], d8[i:i + 16] = d.min(1), dist(pts[qi + CHUNK], pts[j + CHUNK])
    return d0, d8


def replay_errors(job):
    cur, act = job
    return A.replay(A.make_env(), cur, act)[1]


def coverages(job):
    """gym_pusht's coverage of each recorded row's T pose (the agent does not enter it)."""
    env, out = A.make_env(), []
    for x in job:
        A.load_fresh(env, ((x[0], x[1]), (0.0, 0.0), (x[2], x[3]), float(x[4]), (0.0, 0.0), 0.0))
        out.append(env._get_coverage())
    return np.array(out)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/pusht-recovery/support.json")
    p.add_argument("--workers", type=int, default=min(41, os.cpu_count() or 1))
    args = p.parse_args()
    if git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; the support record is made from a clean commit")
    A.simulator_check()
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if A.source_module.content_sha256(src) != SOURCE_CONTENT:
        sys.exit("refused: the source rows are not the protocol's by content")
    eps = A.subset(A.BASE["roles"]["train"], R, SUBSET_SEED)
    if set(eps) & set(A.BASE["roles"]["held_out"]):
        sys.exit("refused: the subset overlaps the held-out episodes")
    rows = np.flatnonzero(np.isin(src["episode"], eps))
    counts = {"subset_episodes": len(eps), "subset_states": len(rows)}
    if counts != EXPECTED:
        sys.exit(f"refused: the counts {counts} are not the protocol's")
    pts, ep = points(src["state"][rows], src["env"][rows]), src["episode"][rows]
    n_of = {e: int((ep == e).sum()) for e in eps}
    pos = np.concatenate([np.arange(n_of[e]) for e in eps])  # rows are in (episode, frame) order

    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    act = src["action"].astype(np.float64)
    with ProcessPoolExecutor(args.workers) as ex:
        cross = list(ex.map(cross_episode, [(e, pts, ep, pos, n_of) for e in eps]))
        errs = list(ex.map(replay_errors, [(cur[rows[ep == e]], act[rows[ep == e]]) for e in eps]))
        covs = list(ex.map(coverages, [cur[rows[ep == e]] for e in eps]))
    d0, d8 = np.concatenate([a for a, _ in cross]), np.concatenate([b for _, b in cross])

    step = np.concatenate([dist(pts[ep == e][1:], pts[ep == e][:-1]) for e in eps])
    err = np.concatenate(errs)
    verified = err <= A.REPLAY_TOL
    tau = float(np.percentile(step, STEP_QUANTILE) + err[verified].max())
    grid = [(e, k, errs_e[k] <= A.REPLAY_TOL, cov_e[k] <= SUCCESS, bool((cov_e[k + 1:] > SUCCESS).any()))
            for e, errs_e, cov_e in zip(eps, errs, covs)
            for k in range(ANCHOR["first_row"], n_of[e] - ANCHOR["rows_before_end"] + 1, ANCHOR["stride"])]
    anchors = [(int(e), int(k)) for e, k, ok, before, reached in grid if ok and before and reached]
    anchor_arr = np.array(anchors, np.int64).reshape(-1, 2)
    filters = {"grid": len(grid), "replay_verified": int(sum(bool(g[2]) for g in grid)),
               "and_not_yet_success": int(sum(bool(g[2] and g[3]) for g in grid)),
               "and_continuation_succeeds": len(anchors)}
    succeeding = [int(e) for e, cov_e in zip(eps, covs) if (cov_e > SUCCESS).any()]

    out = {
        "schema": "ilsb.pusht_recovery.support 3",
        "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
                 "files_sha256": {f: A.sha(ROOT / f) for f in (
                     "tools/pusht_recovery/support.py", "tools/pusht_recovery/stage_a.py",
                     "tools/pusht_recovery/stage-a.json", "tools/pusht_v2/source.py", "tools/pusht_v2/oracle_probe.py",
                     "tools/pusht_v2/oracle_stage_a.py", "tools/pusht_v2/surface.json")}},
        "simulator": A.SPEC["simulator"],
        "inputs": {"source_content_sha256": SOURCE_CONTENT, "r": R, "subset_seed": SUBSET_SEED, "subset_episodes": eps},
        "counts": counts,
        "state": "18 points: agent and 8 T keypoints, frames k and k + 1 (one source row)",
        "distance": "largest per-point Euclidean distance, px",
        "rejected_cross_episode": {
            "nearest_other_episode_px": quantiles(d0),
            "same_pair_after_one_chunk_px": quantiles(d8),
            "tau_if_99th_percentile_px": float(np.percentile(d0, 99)),
            "states": int(len(d0)),
        },
        "paired_prefix": {
            "demonstration_step_px": quantiles(step),
            "replay_verified_states": int(verified.sum()),
            "replay_error_max_px": float(err[verified].max()),
            "tau_px": tau,
            "tau_rule": f"the {STEP_QUANTILE}th percentile of one demonstration step plus the largest replay error over "
                        "the replay-verified states",
            "anchor_rule": {**ANCHOR, "replay_verified": True, "anchor_not_success": True,
                            "continuation_succeeds": True, "success": f"coverage > {SUCCESS} (gym_pusht)"},
            "anchor_filters": filters,
            "episodes_whose_recording_succeeds": len(succeeding),
            "anchors": len(anchors),
            "anchor_episodes": int(len(np.unique(anchor_arr[:, 0]))),
            "anchors_sha256": hashlib.sha256(anchor_arr.tobytes()).hexdigest(),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"counts": counts, "rejected": out["rejected_cross_episode"], "paired": {
        k: out["paired_prefix"][k] for k in ("tau_px", "anchors", "anchor_episodes", "replay_verified_states")}}))


if __name__ == "__main__":
    main()
