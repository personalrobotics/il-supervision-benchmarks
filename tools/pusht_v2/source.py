"""The PushT surface study's source rows (ilsb#5), built once from lerobot/pusht_keypoints at the protocol's revision.

    uv run python tools/pusht_v2/source.py DATASET OUT.npz

DATASET is the LeRobot dataset directory at revision ace8c161; its data, episodes, info and tasks files must have the
SHA-256 values below (the ones v2.2 read). Every frame of every episode becomes one derived history row exactly as
derive.py defines it: x_bar_t = [x_{t-1}, x_t] for the agent (state, 4) and the T's keypoints (env, 32), the episode's
first frame dropped, with that frame's action. Each row also carries CCIL's coordinates for it: the agent history
(the same 4 values) and the T's pose history [T_{t-1}, T_t] (6), the pose fitted per frame and theta unwrapped along
each episode (pose.py). OUT.npz's content digest (sha256 over each array's name, dtype, shape and bytes, in name
order) is what the specification declares; the file's own bytes are not (zip metadata carries a timestamp).
"""
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pose  # noqa: E402

FILES = {"data/chunk-000/file-000.parquet": "8b4d978eeeb99254ddead01e79ee70d95a0416e27438ca9438c89a1682598ae0",
         "meta/episodes/chunk-000/file-000.parquet": "cdeb650a03b1a0e321469bbdc6749c8d9749bac2657a7bf64466e21ed40792e1",
         "meta/info.json": "3a6d3d438106d5771b811802a51f9ceeba8b13e83297b01bc467456e2fcdc2e3",
         "meta/tasks.parquet": "5bca7125a8c9e40408f3177487abfc1120dbf535ed61268613627c0311c2404c"}


def content_sha256(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}\0{a.dtype.str}\0{a.shape}\0".encode() + a.tobytes())
    return h.hexdigest()


def build(dataset: Path) -> dict:
    for rel, want in FILES.items():
        got = hashlib.sha256((dataset / rel).read_bytes()).hexdigest()
        if got != want:
            raise SystemExit(f"{dataset / rel} has sha256 {got}, not the protocol's {want}")
    fps = float(json.loads((dataset / "meta" / "info.json").read_text())["fps"])
    t = pq.read_table(dataset / "data" / "chunk-000" / "file-000.parquet").to_pydict()
    ep, fr = np.array(t["episode_index"], np.int64), np.array(t["frame_index"], np.int64)
    idx = np.array(t["index"], np.int64)
    agent = np.array(t["observation.state"], np.float32)
    kp = np.array(t["observation.environment_state"], np.float32)
    act = np.array(t["action"], np.float32)
    out = {k: [] for k in ("episode", "frame", "source_index", "state", "env", "action", "pose")}
    worst = 0.0
    for e in sorted(set(ep.tolist())):
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(fr[rows])]
        if not np.array_equal(fr[rows], np.arange(len(rows))):
            raise SystemExit(f"episode {e}: frame indices are not 0..{len(rows) - 1}")
        q = pose.episode_poses(kp[rows].astype(np.float64)).astype(np.float32)
        worst = max(worst, float(np.abs(pose.keypoints(q.astype(np.float64)) - kp[rows].astype(np.float64)).max()))
        n = len(rows) - 1
        out["episode"].append(np.full(n, e, np.int64))
        out["frame"].append(np.arange(n, dtype=np.int64))  # derived frame k is source frame k + 1
        out["source_index"].append(idx[rows[1:]])
        out["state"].append(np.concatenate([agent[rows[:-1]], agent[rows[1:]]], 1))
        out["env"].append(np.concatenate([kp[rows[:-1]], kp[rows[1:]]], 1))
        out["action"].append(act[rows[1:]])
        out["pose"].append(np.concatenate([q[:-1], q[1:]], 1))
    arrays = {k: np.concatenate(v) for k, v in out.items()}
    arrays["fps"] = np.array(fps)
    arrays["round_trip_max_px"] = np.array(worst)
    return arrays


if __name__ == "__main__":
    arrays = build(Path(sys.argv[1]))
    np.savez(sys.argv[2], **arrays)
    print(json.dumps({"out": sys.argv[2], "rows": int(len(arrays["episode"])),
                      "episodes": int(len(set(arrays["episode"].tolist()))),
                      "round_trip_max_px": float(arrays["round_trip_max_px"]), "content_sha256": content_sha256(arrays)}))
