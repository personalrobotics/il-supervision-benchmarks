"""PushT keypoints v2 (protocol.json): bc, duplication and ccil on the two-frame history state, IBC downstream.

    python tools/pusht_v2/study.py --root ROOT --source DATA --lerobot-python PY [--smoke] [stages...]

DATA is lerobot/pusht_keypoints at the protocol's revision; PY is a LeRobot 0.6.1 environment with this commit's two
plugins installed (tools/pusht_v2/env_plugin, tools/pusht_v2/ibc_plugin). Stages:
  derive    the history dataset, full and training-only (derive.py), and the adapter check (check.py)
  labels    the pinned CCIL converts the full history dataset, generates (pusht.yaml: a structured spectral MLP fitted on
            the training episodes, noisy_action_fe, M = 1), curates and exports one augmented LeRobot dataset
  arms      asserts that every exported label's newest slots equal its source frame's newest slots within
            eps_struct = max(10 root_tolerance, 10 eps32 max(1, ||x_t||_inf)), the solver's and float32's error alone:
            each exported label already converged and was accepted, so one violation is a defect and stops the run.
            The labels' yield (candidates, accepted, rejections by reason) is reported separately. bc is the training
            episodes; ccil adds the exported labels; duplication adds, for each, in order, a one-frame copy of its
            source frame
  train, select, evaluate, audit, report   as the pilot's runner, with IBC, held-out InfoNCE selection, env
            pusht_history, and the protocol's repeat-evaluation tolerances
A rerun resumes: a verified stage or unit is never recomputed.
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from gate import Gate, main  # noqa: E402  (gate puts tools/pusht_pilot on the path)
from run import COLUMNS, PilotError, now, sha256, tree, write_json  # noqa: E402  (tools/pusht_pilot/run.py)

STAGES = ("derive", "labels", "arms", "train", "select", "evaluate", "audit", "report")
EPS32 = float(np.finfo(np.float32).eps)
CURRENT = {"observation.state": slice(2, 4), "observation.environment_state": slice(16, 32)}  # newest slots


class V2(Gate):
    PROTOCOL = HERE / "protocol.json"
    CONFIG = HERE / "pusht.yaml"
    PRESERVE = ()  # the history dataset carries no next.* columns
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

    def label_source(self) -> Path:
        return self.root / "derived" / "full"

    def derive(self) -> None:
        self.derive_data()

    def arms(self) -> None:
        import pyarrow.parquet as pq
        from ccil.lerobot import write_dataset
        from ccil.schema import Episode
        if self.marker("arms"):
            return
        lab, f = self.need("labels"), self.protocol["invariant"]
        if sorted(int(e) for e in lab["held_out"]) != sorted(self.held_out()):
            raise PilotError("CCIL's split is not the protocol's")
        fit_cfg = json.loads(next((self.root / "labels" / "runs").glob("transition-fit-*/config.json")).read_text())
        gen_cfg = json.loads(next((self.root / "labels" / "runs").glob("labels-generate-*/config.json")).read_text())
        study = {"fit_episodes": fit_cfg["values"]["transition_model.fit_episodes"],
                 "structure": fit_cfg["values"]["transition_model.structure"],
                 "method": gen_cfg["values"]["generation.method"]}
        if study != {"fit_episodes": "train", "structure": "history_shift", "method": "noisy_action_fe"}:
            raise PilotError(f"the labels were not made as the protocol fixes: {study}")
        exp = self.root / lab["export"]
        manifest = json.loads((exp / "ccil" / "export.json").read_text())
        t = pq.read_table(sorted((exp / "data").rglob("*.parquet"))).to_pydict()
        tasks = pq.read_table(exp / "meta" / "tasks.parquet").to_pydict()
        task = dict(zip(tasks["task_index"], tasks["task"]))
        fps = float(json.loads((exp / "meta" / "info.json").read_text())["fps"])
        syn = np.array(t["ccil.synthetic"], bool)
        arr = {k: np.array(t[k], np.float32) for k in ("observation.state", "observation.environment_state", "action")}
        col = {k: np.array(t[k], np.int64) for k in COLUMNS}
        ts, tidx = np.array(t["timestamp"], np.float64), np.array(t["task_index"], np.int64)
        demo_rows, label_rows = np.flatnonzero(~syn), np.flatnonzero(syn)
        if set(col["ccil.source_episode"][demo_rows].tolist()) & set(self.held_out()):
            raise PilotError("the export holds a held-out episode")
        by_index = {int(col["ccil.source_index"][i]): i for i in demo_rows}
        if any(int(col["ccil.source_index"][i]) not in by_index for i in label_rows):
            raise PilotError("a label's source frame is not in the export")
        src = np.array([by_index[int(col["ccil.source_index"][i])] for i in label_rows], np.int64)

        # the structural invariant: a label h_G = [x', x''] reaching its target [x_t, x_{t+1}] has x'' = x_t
        newest = lambda rows: np.concatenate([arr[k][rows][:, s].astype(np.float64) for k, s in CURRENT.items()], 1)  # noqa: E731
        err = np.abs(newest(label_rows) - newest(src)).max(1)
        eps = np.maximum(10 * f["root_tolerance"], 10 * EPS32 * np.maximum(1.0, np.abs(newest(src)).max(1)))
        bad = np.flatnonzero(err > eps)
        q = lambda x: {str(p): float(np.percentile(x, p)) for p in (50, 90, 100)} if len(x) else None  # noqa: E731
        result = json.loads((self.root / "labels" / "runs" / "generate-result.json").read_text())
        invariant = {"labels": int(len(label_rows)), "violations": int(len(bad)), "error_px": q(err),
                     "eps_struct_px": q(eps), "yield": result["counts"]}
        self.log(f"arms: history invariant over {invariant['labels']} exported labels: error p50 "
                 f"{invariant['error_px']['50']:.3g}, max {invariant['error_px']['100']:.3g} px; "
                 f"{len(bad)} violations; yield {result['counts']}")
        if len(bad):
            write_json(self.root / f"arms-failure-{int(time.time())}.json",
                       {"invariant": invariant, "violating_candidate_rows": col["ccil.candidate_row"][label_rows[bad]].tolist(),
                        "at": now()})
            raise PilotError(f"{len(bad)} exported labels break the history invariant: a defect, not a yield")
        kept, kept_src = label_rows, src

        episodes, order = {}, []
        for i in demo_rows:
            e = int(col["ccil.source_episode"][i])
            if e not in episodes:
                episodes[e] = []
                order.append(e)
            episodes[e].append(i)

        def episode(name, rows, timestamps):
            return Episode(name, {"state": arr["observation.state"][rows],
                                  "state.environment_state": arr["observation.environment_state"][rows]},
                           arr["action"][rows], timestamps=timestamps, task=task[int(tidx[rows[0]])])

        def build(arm, added):
            """The training episodes, then one one-frame episode per (row, provenance row) in added."""
            demos = [np.array(episodes[e]) for e in order]
            eps_ = [episode(str(e), r, ts[r]) for e, r in zip(order, demos)]
            eps_ += [episode(f"{arm}.{k}", np.array([r]), np.zeros(1)) for k, (r, _) in enumerate(added)]
            prov = demos + [np.array([p]) for _, p in added]
            dest = self.root / "arms" / arm
            if dest.exists():
                dest.rename(dest.with_name(f"{arm}.incomplete-{int(time.time())}"))
            dest.parent.mkdir(exist_ok=True)

            def columns(e, ep):
                out = {k: col[k][prov[e]] for k in COLUMNS}
                if e >= len(demos):  # an added row: synthetic 1, and the label it stands for
                    label = added[e - len(demos)][1]
                    out = {**out, "ccil.synthetic": np.ones(1, np.int64),
                           "ccil.candidate_row": col["ccil.candidate_row"][label:label + 1],
                           "ccil.source_index": col["ccil.source_index"][label:label + 1]}
                return out
            write_dataset(iter(eps_), dest, fps=fps, columns=columns, single_frame_episodes=True)
            return dest

        paths = {"bc": build("bc", []), "ccil": build("ccil", list(zip(kept, kept))),
                 "duplication": build("duplication", list(zip(kept_src, kept)))}
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
        ok = (summary["duplication"]["added_rows"] == summary["ccil"]["added_rows"] == len(kept)
              and summary["duplication"]["added_source_index_sha256"] == summary["ccil"]["added_source_index_sha256"]
              and summary["bc"]["added_rows"] == 0 and len({v["demonstration_index_sha256"] for v in summary.values()}) == 1)
        if not ok:
            raise PilotError(f"the arms do not reconcile: {json.dumps({k: {x: y for x, y in v.items() if x != 'files'} for k, v in summary.items()})}")
        write_json(self.root / "arms.json", {"arms": summary, "invariant": invariant, "study": study,
                                             "export_counts": manifest["counts"], "completed": now()})
        self.log("arms: " + ", ".join(f"{k} {v['frames']} frames ({v['added_rows']} added)" for k, v in summary.items()))


if __name__ == "__main__":
    sys.exit(main(V2, STAGES))
