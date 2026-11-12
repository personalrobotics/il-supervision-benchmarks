"""v2's pose-space smoke gate: CCIL generates in the T's physical pose, labels decode to rigid keypoints (no policy).

    uv run python tools/pusht_v2/pose_probe.py SOURCE SPEC_JSON OUT_DIR

SOURCE is lerobot/pusht_keypoints at the protocol's revision. Each frame's state is q = (agent x, y, T x, y, theta), the
T's pose recovered from its keypoints by the rigid fit (pose.py) with theta unwrapped along the episode, and a history
row is [q_{t-1}, q_t] with the action from x_t: derive.py's alignment, so ccil's split and rows match the keypoint
history dataset's. Then:
1. round trip: the stored (float32) poses decode to every frame's recorded keypoints within the spec's tolerance;
2. the pinned CCIL generates on the pose history (pose.yaml: v2's configuration with representation.modality
   state+state.pose, the structured spectral MLP on the training episodes, noisy_action_fe, M = 1) and curates;
3. for every proposal: accepted (converged, in bounds), residual ||h_G + f_hat(h_G, a~) - target|| recomputed from the
   checkpoint, the history invariant (newest slots equal the source's q_t within eps_struct), and the decoded T's
   rigidity in both slots against the demonstrations' pair distances; usable = all of them;
4. the conditioning of d(newest next slot)/d(previous slot), 5 x 5, at a sample of source states.
The gate passes iff the round trip holds, usable / proposals >= the spec's minimum, and no accepted label breaks the
residual, the invariant or the rigidity. Writes OUT_DIR/pose-probe.json.
"""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import yaml

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose  # noqa: E402
from preimage_probe import rigidity_error  # noqa: E402

EPS32 = float(np.finfo(np.float32).eps)
NEW, PREV = np.r_[2:4, 7:10], np.r_[0:2, 4:7]  # state [agent_{t-1}(2), agent_t(2)], state.pose [T_{t-1}(3), T_t(3)]


def pose_history(source: Path, out: Path) -> dict:
    from ccil.schema import Episode, TrajectoryDataset, write_trajectories
    fps = float(json.loads((source / "meta" / "info.json").read_text())["fps"])
    t = pq.read_table(sorted((source / "data").rglob("*.parquet"))).to_pydict()
    ep, fr = np.array(t["episode_index"]), np.array(t["frame_index"])
    eps, worst = [], 0.0
    for e in sorted(set(ep.tolist())):
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(fr[rows])]
        agent = np.array([t["observation.state"][i] for i in rows], np.float32)
        kp = np.array([t["observation.environment_state"][i] for i in rows], np.float32)
        q = pose.episode_poses(kp.astype(np.float64)).astype(np.float32)
        worst = max(worst, float(np.abs(pose.keypoints(q.astype(np.float64)) - kp.astype(np.float64)).max()))
        act = np.array([t["action"][i] for i in rows[1:]], np.float32)
        eps.append(Episode(str(e), {"state": np.concatenate([agent[:-1], agent[1:]], 1),
                                    "state.pose": np.concatenate([q[:-1], q[1:]], 1)},
                           act, timestamps=np.arange(len(act)) / fps))
    write_trajectories(TrajectoryDataset(eps), out)
    return {"episodes": len(eps), "frames": int(sum(len(x.actions) for x in eps)), "round_trip_max_px": worst}


def main(source: Path, spec_path: Path, out: Path) -> int:
    spec = json.loads(spec_path.read_text())
    out.mkdir(parents=True)
    built = pose_history(source, out / "pose.npz")
    cfg = yaml.safe_load((HERE / "pusht.yaml").read_text())
    cfg["data"]["path"] = str((out / "pose.npz").resolve())
    cfg["representation"]["modality"] = "state+state.pose"
    (out / "pose.yaml").write_text(yaml.safe_dump(cfg, sort_keys=False))
    for args in (("generate", "--config", "pose.yaml"), ("curate", "runs/generate-result.json")):
        r = subprocess.run([sys.executable, "-m", "ccil", *args], cwd=out, capture_output=True, text=True)
        (out / f"{args[0]}.txt").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise SystemExit(f"ccil {args[0]} exited {r.returncode}:\n{r.stderr[-1500:]}")

    from ccil.models.mlp import MLPTransitionModel
    from ccil.schema import read_candidates, read_trajectories
    runs = out / "runs"
    ds = read_trajectories(next(runs.glob("data-prepare-*")) / "data" / "dataset.npz")
    c = read_candidates(next(runs.glob("labels-generate-*")) / "labels" / "candidates.npz", ds)
    model = MLPTransitionModel.load(next(runs.glob("transition-fit-*")) / "checkpoints" / "transition_model.npz")
    names = ("state", "state.pose")
    z = np.concatenate([np.asarray(c.observations[n], np.float64) for n in names], 1)
    tgt = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in names], 1)
    a = np.asarray(c.actions, np.float64)
    by_id = {e.episode_id: np.concatenate([e.observations[n] for n in names], 1) for e in ds.episodes}
    src = np.stack([by_id[e][f] for e, f in zip(np.asarray(c.source_episode_id), np.asarray(c.source_frame))])
    src = src.astype(np.float64)
    accepted = np.asarray(c.accepted, bool)
    reasons, codes = np.asarray(c.reasons), list(c.reason_codes)
    rejected = {k: int(reasons[:, i].astype(bool).sum()) for i, k in enumerate(codes) if reasons[:, i].any()}
    tol = spec["root_tolerance"]
    resid = np.linalg.norm(z + model.delta(z, a) - tgt, axis=1)
    err = np.abs(z[:, NEW] - src[:, NEW]).max(1)
    eps = np.maximum(10 * tol, 10 * EPS32 * np.maximum(1.0, np.abs(src[:, NEW]).max(1)))
    reference = np.asarray(spec["rigidity"]["reference_pair_distances_px"])
    decoded = np.concatenate([pose.keypoints(z[:, 4:7]), pose.keypoints(z[:, 7:10])], 1)
    rig = rigidity_error(decoded, reference)
    resid_bad, inv_bad = accepted & (resid > tol), accepted & (err > eps)
    rig_bad = accepted & (rig > spec["rigidity"]["tolerance_px"])
    usable = accepted & ~resid_bad & ~inv_bad & ~rig_bad
    sample = np.random.default_rng(spec["sample_seed"]).choice(len(z), spec["conditioning_sample"], replace=False)
    J = model.jacobian_z(src[sample], a[sample])[:, NEW][:, :, PREV]
    sv = np.linalg.svd(J, compute_uv=False)
    q = lambda x: {str(p): float(np.percentile(x, p)) for p in (10, 50, 90, 100)} if len(x) else None  # noqa: E731
    doc = {"spec": spec, "dataset": built, "proposals": int(len(z)), "accepted": int(accepted.sum()),
           "rejected_by_reason": rejected, "usable": int(usable.sum()), "usable_fraction": float(usable.mean()),
           "violations": {"residual": int(resid_bad.sum()), "invariant": int(inv_bad.sum()),
                          "rigidity": int(rig_bad.sum())},
           "residual_accepted": q(resid[accepted]), "invariant_error_accepted": q(err[accepted]),
           "rigidity_error_accepted_px": q(rig[accepted]),
           "displacement_from_source_accepted": q(np.linalg.norm(z - src, axis=1)[accepted]),
           "previous_slot_change_accepted": q(np.abs(z[:, PREV] - src[:, PREV]).max(1)[accepted]),
           "conditioning": {"sample": int(len(sample)), "sigma_max": q(sv[:, 0]), "sigma_min": q(sv[:, -1]),
                            "condition": q(sv[:, 0] / sv[:, -1])}}
    doc["pass"] = bool(built["round_trip_max_px"] <= spec["round_trip_tolerance_px"]
                       and doc["usable_fraction"] >= spec["min_usable_fraction"]
                       and not any(doc["violations"].values()))
    (out / "pose-probe.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: doc[k] for k in ("dataset", "proposals", "accepted", "rejected_by_reason", "usable",
                                          "usable_fraction", "violations", "pass")}, indent=1))
    print("conditioning", doc["conditioning"]["condition"], "residual", doc["residual_accepted"])
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])))
