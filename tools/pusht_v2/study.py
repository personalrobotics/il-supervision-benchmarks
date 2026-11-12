"""PushT keypoints v2 (protocol.json): bc, duplication and ccil on the two-frame history state, IBC downstream.

    python tools/pusht_v2/study.py --root ROOT --source DATA --lerobot-python PY [--smoke] [stages...]

DATA is lerobot/pusht_keypoints at the protocol's revision; PY is a LeRobot 0.6.1 environment with this commit's two
plugins installed (tools/pusht_v2/env_plugin, tools/pusht_v2/ibc_plugin). The policy's state is the keypoint history
[x_{t-1}, x_t]; CCIL generates in the T's rigid pose (pose.py) and its labels are decoded back to keypoints. Stages:
  derive    the keypoint history dataset, full and training-only (derive.py), and the adapter check (check.py)
  labels    the pose history [q_{t-1}, q_t], q = (agent x, y, T x, y, theta), with derive.py's rows; the pinned CCIL
            generates on it (pusht.yaml: a structured spectral MLP fitted on the training episodes, noisy_action_fe,
            M = 1) and curates every accepted label with weight 1
  arms      only labels sourced in a training episode enter an arm (CCIL proposes from every episode; the protocol's
            containment count is checked exactly). Every selected label must have a residual within the root tolerance, newest slots equal to its source's
            q_t within eps_struct = max(10 root_tolerance, 10 eps32 max(1, ||q_t||_inf)), and a decoded T as rigid as
            the demonstrations' in both slots: one violation is a defect and stops the run. Usable labels below the
            protocol's yield floor stop it too. A label decodes to [agent_{t-1}~, agent_t, keypoints(T_{t-1}~),
            keypoints(T_t)] with its action, the current halves the source frame's own. bc is the training episodes;
            ccil adds the decoded labels; duplication adds, for each, in order, a one-frame copy of its source frame.
            Containment is then read back from the written datasets, and again by the audit: demonstrations exactly
            the training episodes, every added row sourced in one, no row's current frame a held-out frame's
  train, select, evaluate, audit, report   as the pilot's runner, with IBC, held-out InfoNCE selection, env
            pusht_history, and the protocol's repeat-evaluation tolerances
A rerun resumes: a verified stage or unit is never recomputed.
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate import Gate, main  # noqa: E402  (gate puts tools/pusht_pilot on the path)
from run import COLUMNS, PilotError, now, sha256, tree, write_json  # noqa: E402  (tools/pusht_pilot/run.py)
import pose  # noqa: E402
from pose_probe import pose_history  # noqa: E402
from preimage_probe import rigidity_error  # noqa: E402

STAGES = ("derive", "labels", "arms", "train", "select", "evaluate", "audit", "report")
EPS32 = float(np.finfo(np.float32).eps)
NAMES = ("state", "state.pose")
NEW = np.r_[2:4, 7:10]  # state [agent_{t-1}(2), agent_t(2)], state.pose [T_{t-1}(3), T_t(3)]


class V2(Gate):
    PROTOCOL = HERE / "protocol.json"
    CONFIG = HERE / "pusht.yaml"
    REPEAT_TOLERANCE = {"max_reward": 1e-12, "sum_reward": 1e-9}  # IBC's sampled actions: rounding-level repeats

    def identity(self) -> dict:
        import importlib.metadata as md
        url = json.loads(md.distribution("ccil").read_text("direct_url.json") or "{}")
        return {**super().identity(), "config_sha256": sha256(self.CONFIG),
                "ccil_commit": url.get("vcs_info", {}).get("commit_id")}

    def check_root(self) -> None:
        ident = self.identity()
        if ident["ccil_commit"] != self.protocol["ccil"]["commit"]:
            raise PilotError(f"ccil is {ident['ccil_commit']}, the protocol pins {self.protocol['ccil']['commit']}")
        path = self.root / "root.json"
        if path.exists():
            old = json.loads(path.read_text())
            if any(old.get(k) != ident[k] for k in ("config_sha256", "ccil_commit")):
                raise PilotError(f"{path} was created under another CCIL or configuration; use a new root")
        super().check_root()

    def derive(self) -> None:
        self.derive_data()

    def ccil(self, *args, cwd) -> str:
        r = subprocess.run([sys.executable, "-m", "ccil", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            raise PilotError(f"ccil {args[0]} exited {r.returncode}:\n{r.stderr[-1500:]}")
        return r.stdout

    def labels(self) -> None:
        if self.marker("labels"):
            return
        self.need("derive")
        d = self.root / "labels"
        if d.exists():
            d.rename(self.root / f"labels.incomplete-{int(time.time())}")
        d.mkdir()
        built = pose_history(self.source, d / "pose.npz")
        if built["round_trip_max_px"] > self.protocol["invariant"]["round_trip_tolerance_px"]:
            raise PilotError(f"poses decode to the keypoints only within {built['round_trip_max_px']} px")
        shutil.copy(self.CONFIG, d / "pusht.yaml")
        self.log(f"labels: pose history of {built['episodes']} episodes, {built['frames']} frames (round trip "
                 f"{built['round_trip_max_px']:.3g} px); generating")
        out = self.ccil("generate", "--config", "pusht.yaml", cwd=d) + self.ccil("curate", "runs/generate-result.json",
                                                                                 cwd=d)
        (d / "summary.txt").write_text(out)
        split = json.loads(next((d / "runs").glob("data-prepare-*/data/split.json")).read_text())
        counts = json.loads((d / "runs" / "generate-result.json").read_text())["counts"]
        write_json(self.root / "labels.json", {"pose_dataset": built, "counts": counts,
                                               "held_out": sorted(int(e) for e in split["validation"]),
                                               "train": sorted(int(e) for e in split["train"]), "completed": now()})
        self.log(f"labels: {counts}")

    def arms(self) -> None:
        import pyarrow.parquet as pq
        from ccil.lerobot import write_dataset
        from ccil.models.mlp import MLPTransitionModel
        from ccil.schema import Episode, read_candidates, read_selection, read_trajectories
        if self.marker("arms"):
            return
        lab, inv = self.need("labels"), self.protocol["invariant"]
        if lab["held_out"] != sorted(self.held_out()):
            raise PilotError("CCIL's split is not the protocol's")
        runs = self.root / "labels" / "runs"
        cfg = {s: json.loads(next(runs.glob(f"{s}-*/config.json")).read_text())["values"]
               for s in ("transition-fit", "labels-generate")}
        study = {"fit_episodes": cfg["transition-fit"]["transition_model.fit_episodes"]["value"],
                 "structure": cfg["transition-fit"]["transition_model.structure"]["value"],
                 "modality": cfg["transition-fit"]["representation.modality"]["value"],
                 "method": cfg["labels-generate"]["generation.method"]["value"]}
        if study != {"fit_episodes": "train", "structure": "history_shift", "modality": "state+state.pose",
                     "method": "noisy_action_fe"}:
            raise PilotError(f"the labels were not made as the protocol fixes: {study}")

        # the selected labels, in pose coordinates
        ds = read_trajectories(next(runs.glob("data-prepare-*")) / "data" / "dataset.npz")
        c = read_candidates(next(runs.glob("labels-generate-*")) / "labels" / "candidates.npz", ds)
        sel = read_selection(next(runs.glob("labels-curate-*")) / "labels" / "selection.npz")
        model = MLPTransitionModel.load(next(runs.glob("transition-fit-*")) / "checkpoints" / "transition_model.npz")
        mask = np.asarray(sel.mask, bool)
        if not np.array_equal(mask, np.asarray(c.accepted, bool)) or (np.asarray(sel.weight)[mask] != 1).any():
            raise PilotError("the curation is not every accepted label with weight 1")
        # CCIL proposes from every episode: only labels sourced in a training episode may enter an arm
        cont = self.protocol["containment"]
        train_ids = set(lab["train"])
        from_train = np.isin(np.asarray(c.source_episode_id).astype(np.int64), sorted(train_ids))
        if train_ids & set(lab["held_out"]) or int(from_train.sum()) != cont["training_proposals"]:
            raise PilotError(f"{int(from_train.sum())} proposals from training episodes, the protocol expects "
                             f"{cont['training_proposals']}")
        mask = mask & from_train
        rows = np.flatnonzero(mask)
        z = np.concatenate([np.asarray(c.observations[n], np.float64) for n in NAMES], 1)[rows]
        tgt = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in NAMES], 1)[rows]
        act = np.asarray(c.actions)[rows]
        src_ep, src_fr = np.asarray(c.source_episode_id)[rows], np.asarray(c.source_frame)[rows]
        by_id = {e.episode_id: np.concatenate([e.observations[n] for n in NAMES], 1) for e in ds.episodes}
        q_src = np.stack([by_id[e][f] for e, f in zip(src_ep, src_fr)]).astype(np.float64)

        # correctness: every selected label (one violation stops the run) ...
        tol = inv["root_tolerance"]
        resid = np.linalg.norm(z + model.delta(z, act.astype(np.float64)) - tgt, axis=1)
        err = np.abs(z[:, NEW] - q_src[:, NEW]).max(1)
        eps = np.maximum(10 * tol, 10 * EPS32 * np.maximum(1.0, np.abs(q_src[:, NEW]).max(1)))
        kp_prev, kp_cur = pose.keypoints(z[:, 4:7]), pose.keypoints(z[:, 7:10])
        rig = rigidity_error(np.concatenate([kp_prev, kp_cur], 1), np.asarray(inv["rigidity"]["reference_pair_distances_px"]))
        bad = {"residual": resid > tol, "invariant": err > eps, "rigidity": rig > inv["rigidity"]["tolerance_px"]}
        q = lambda x: {str(p): float(np.percentile(x, p)) for p in (50, 90, 100)} if len(x) else None  # noqa: E731
        stats = {"proposals": int(from_train.sum()), "proposals_all_episodes": int(len(mask)), "selected": int(len(rows)),
                 "usable_fraction": float(len(rows) / from_train.sum()), "violations": {k: int(v.sum()) for k, v in bad.items()},
                 "residual": q(resid), "invariant_error": q(err), "rigidity_error_px": q(rig),
                 "previous_slot_change": q(np.abs(z[:, np.r_[0:2, 4:7]] - q_src[:, np.r_[0:2, 4:7]]).max(1)),
                 "yield_floor": inv["min_usable_fraction"], "yield": lab["counts"]}
        self.log(f"arms: {stats['selected']} of {stats['proposals']} proposals selected "
                 f"({stats['usable_fraction']:.4f}); violations {stats['violations']}")
        if any(stats["violations"].values()) or stats["usable_fraction"] < inv["min_usable_fraction"]:
            write_json(self.root / f"arms-failure-{int(time.time())}.json", {"labels": stats, "at": now()})
            raise PilotError(f"the labels fail the protocol's correctness or yield: {stats['violations']}, "
                             f"usable {stats['usable_fraction']:.4f}")

        # ... decoded onto the keypoint history the policy reads; the current halves are the source frame's own
        full = self.root / "derived" / "full"
        t = pq.read_table(sorted((full / "data").rglob("*.parquet"))).to_pydict()
        st, env = np.array(t["observation.state"], np.float32), np.array(t["observation.environment_state"], np.float32)
        acts, gidx = np.array(t["action"], np.float32), np.array(t["index"], np.int64)
        ep, fr = np.array(t["episode_index"], np.int64), np.array(t["frame_index"], np.int64)
        tasks = pq.read_table(full / "meta" / "tasks.parquet").to_pydict()
        task = dict(zip(tasks["task_index"], tasks["task"]))
        tidx = np.array(t["task_index"], np.int64)
        fps = float(json.loads((full / "meta" / "info.json").read_text())["fps"])
        row_of = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(ep, fr))}
        src_row = np.array([row_of[(int(e), int(f))] for e, f in zip(src_ep, src_fr)], np.int64)
        drift = max(float(np.abs(env[src_row][:, 16:] - kp_cur).max()), float(np.abs(st[src_row][:, 2:] - z[:, 2:4]).max()))
        if drift > inv["round_trip_tolerance_px"]:
            raise PilotError(f"the labels' sources are not the keypoint history's rows ({drift} px)")
        lab_state = np.concatenate([z[:, 0:2], st[src_row][:, 2:4]], 1).astype(np.float32)
        lab_env = np.concatenate([kp_prev, env[src_row][:, 16:32]], 1).astype(np.float32)
        held = set(self.held_out())
        order = sorted(set(ep.tolist()) - held)
        demo_rows = {e: np.flatnonzero(ep == e)[np.argsort(fr[ep == e])] for e in order}

        def build(arm, added):
            """The training episodes, then one one-frame episode per added (state, env, action, columns)."""
            eps_ = [Episode(str(e), {"state": st[r], "state.environment_state": env[r]}, acts[r],
                            timestamps=np.arange(len(r)) / fps, task=task[int(tidx[r[0]])]) for e, r in demo_rows.items()]
            cols = [{"ccil.synthetic": np.zeros(len(r), np.int64), "ccil.source_episode": ep[r], "ccil.source_frame": fr[r],
                     "ccil.source_index": gidx[r], "ccil.candidate_row": np.full(len(r), -1, np.int64)}
                    for r in demo_rows.values()]
            for k, (s, v, a, col) in enumerate(added):
                eps_.append(Episode(f"{arm}.{k}", {"state": s[None], "state.environment_state": v[None]}, a[None],
                                    timestamps=np.zeros(1), task=task[int(tidx[0])]))
                cols.append(col)
            dest = self.root / "arms" / arm
            if dest.exists():
                dest.rename(dest.with_name(f"{arm}.incomplete-{int(time.time())}"))
            dest.parent.mkdir(exist_ok=True)
            write_dataset(iter(eps_), dest, fps=fps, columns=lambda e, _ep: cols[e], single_frame_episodes=True)
            return dest

        def provenance(i):
            r = src_row[i]
            return {"ccil.synthetic": np.ones(1, np.int64), "ccil.source_episode": ep[r:r + 1],
                    "ccil.source_frame": fr[r:r + 1], "ccil.source_index": gidx[r:r + 1],
                    "ccil.candidate_row": np.array([rows[i]], np.int64)}
        n = len(rows)
        paths = {"bc": build("bc", []),
                 "ccil": build("ccil", [(lab_state[i], lab_env[i], act[i].astype(np.float32), provenance(i))
                                        for i in range(n)]),
                 "duplication": build("duplication", [(st[src_row[i]], env[src_row[i]], acts[src_row[i]], provenance(i))
                                                      for i in range(n)])}
        summary = {}
        for arm, path in paths.items():
            tt = pq.read_table(sorted((path / "data").rglob("*.parquet"))).to_pydict()
            s = np.array(tt["ccil.synthetic"], bool)
            srcs = np.sort(np.array(tt["ccil.source_index"], np.int64)[s])
            demo = np.array(tt["ccil.source_index"], np.int64)[~s]
            summary[arm] = {"path": path.relative_to(self.root).as_posix(), "frames": int(len(s)),
                            "demonstration_frames": int((~s).sum()), "added_rows": int(s.sum()),
                            "added_source_index_sha256": hashlib.sha256(srcs.tobytes()).hexdigest(),
                            "demonstration_index_sha256": hashlib.sha256(demo.tobytes()).hexdigest(),
                            "files": tree(path)}
        ok = (summary["duplication"]["added_rows"] == summary["ccil"]["added_rows"] == n
              and summary["duplication"]["added_source_index_sha256"] == summary["ccil"]["added_source_index_sha256"]
              and summary["bc"]["added_rows"] == 0 and len({v["demonstration_index_sha256"] for v in summary.values()}) == 1)
        if not ok:
            raise PilotError(f"the arms do not reconcile: {json.dumps({k: {x: y for x, y in v.items() if x != 'files'} for k, v in summary.items()})}")
        split = self.containment(paths)
        write_json(self.root / "arms.json", {"arms": summary, "labels": stats, "study": study, "containment": split,
                                             "completed": now()})
        self.log("arms: " + ", ".join(f"{k} {v['frames']} frames ({v['added_rows']} added)" for k, v in summary.items()))

    def containment(self, paths: dict) -> dict:
        """The protocol's containment, read back from the written datasets: demonstrations exactly the training episodes,
        every added row sourced in a training episode, and no row's current frame byte-equal to a held-out frame's."""
        import pyarrow.parquet as pq
        held = set(self.held_out())
        current = lambda t: np.ascontiguousarray(np.concatenate(  # noqa: E731
            [np.array(t["observation.state"], np.float32)[:, 2:4],
             np.array(t["observation.environment_state"], np.float32)[:, 16:32]], 1))
        full = pq.read_table(sorted((self.root / "derived" / "full" / "data").rglob("*.parquet"))).to_pydict()
        fe = np.array(full["episode_index"], np.int64)
        train = set(fe.tolist()) - held
        if train != set(self.need("labels")["train"]):
            raise PilotError("the derived dataset's training episodes are not CCIL's")
        held_frames = {r.tobytes() for r in current(full)[np.isin(fe, sorted(held))]}
        out, problems = {}, []
        for arm, path in paths.items():
            t = pq.read_table(sorted((path / "data").rglob("*.parquet"))).to_pydict()
            syn, src = np.array(t["ccil.synthetic"], bool), np.array(t["ccil.source_episode"], np.int64)
            hits = int(sum(r.tobytes() in held_frames for r in current(t)))
            out[arm] = {"demonstration_episodes": len(set(src[~syn].tolist())),
                        "added_from_held_out": int(np.isin(src[syn], sorted(held)).sum()),
                        "rows_with_a_held_out_current_frame": hits}
            if set(src[~syn].tolist()) != train or out[arm]["added_from_held_out"] or hits:
                problems.append(f"{arm}: {out[arm]}")
        if problems:
            write_json(self.root / f"containment-failure-{int(time.time())}.json", {"arms": out, "at": now()})
            raise PilotError("held-out data reached an arm: " + "; ".join(problems))
        return out

    def select(self) -> None:
        super().select()
        import pyarrow.parquet as pq
        fe = np.asarray(pq.read_table(sorted((self.root / "derived" / "full" / "data").rglob("*.parquet")),
                                      columns=["episode_index"])["episode_index"])
        expected = int(np.isin(fe, self.held_out()).sum())
        for arm, seed in self.units():
            scores = json.loads((self.root / "units" / f"{arm}-s{seed}" / "select-scores.json").read_text())
            if scores["frames"] != expected or sorted(scores["episodes"]) != sorted(self.held_out()):
                raise PilotError(f"{arm}-s{seed}: selection scored {scores['frames']} frames, the held-out episodes have "
                                 f"{expected}")

    def audit(self) -> None:
        arms = self.need("arms")["arms"]
        self.containment({arm: self.root / a["path"] for arm, a in arms.items()})
        self.log("audit: containment verified on every arm")
        super().audit()


if __name__ == "__main__":
    sys.exit(main(V2, STAGES))
