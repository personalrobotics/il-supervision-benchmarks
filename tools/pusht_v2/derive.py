"""Derive the two-frame history PushT keypoints dataset from lerobot/pusht_keypoints.

    python derive.py SOURCE OUT [--exclude EPISODES_JSON]

Every position feature x_t becomes [x_{t-1}, x_t], for the agent (observation.state, 2 -> 4) and the T's keypoints
(observation.environment_state, 16 -> 32). The first frame of each episode has no x_{t-1} and is dropped, so derived
frame k of episode e is source frame k + 1 with its action. Episodes keep their source index as their ID; --exclude
leaves out the listed episodes (the held-out split, for a training-only dataset). Values are the stored float32 ones,
unchanged. The source's next.* columns are not carried: no policy reads them. OUT is written once
(ccil.lerobot.write_dataset) with OUT.json beside it.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from ccil.lerobot import write_dataset
from ccil.schema import Episode

FEATURES = (("observation.state", "state"), ("observation.environment_state", "state.environment_state"))


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


def augment(x: np.ndarray) -> np.ndarray:
    return np.concatenate([x[:-1], x[1:]], axis=1)


def derive(source: Path, out: Path, exclude=()) -> dict:
    info = json.loads((source / "meta" / "info.json").read_text())
    fps = float(info["fps"])
    t = pq.read_table(sorted((source / "data").rglob("*.parquet"))).to_pydict()
    tasks = pq.read_table(source / "meta" / "tasks.parquet").to_pydict()
    task = dict(zip(tasks["task_index"], tasks["task"]))
    ep, fr = np.array(t["episode_index"], np.int64), np.array(t["frame_index"], np.int64)
    exclude = {int(e) for e in exclude}
    kept = sorted(set(ep.tolist()) - exclude)

    def episodes():
        for e in kept:
            rows = np.flatnonzero(ep == e)
            rows = rows[np.argsort(fr[rows])]
            if not np.array_equal(fr[rows], np.arange(len(rows))):
                raise ValueError(f"episode {e}: frame indices are not 0..{len(rows) - 1}")
            obs = {slot: augment(np.array([t[name][i] for i in rows], np.float32)) for name, slot in FEATURES}
            act = np.array([t["action"][i] for i in rows[1:]], np.float32)
            yield Episode(str(e), obs, act, timestamps=np.arange(len(act)) / fps,
                          task=task[int(t["task_index"][rows[0]])])

    write_dataset(episodes(), out, fps=fps)
    doc = {"source": str(source), "source_sha256": tree(source), "excluded_episodes": sorted(exclude),
           "episodes": len(kept), "frames": int(sum(int((ep == e).sum()) - 1 for e in kept)),
           "definition": "x_bar_t = [x_{t-1}, x_t] for observation.state and observation.environment_state; "
                         "each episode's first frame dropped",
           "out_sha256": tree(out)}
    Path(f"{out}.json").write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    return doc


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("source", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--exclude", type=Path, help="a JSON list of episode indices to leave out")
    a = ap.parse_args()
    d = derive(a.source, a.out, json.loads(a.exclude.read_text()) if a.exclude else ())
    print(f"{a.out}: {d['episodes']} episodes, {d['frames']} frames")
