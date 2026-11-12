#!/usr/bin/env python3
"""The PushT M=1 pilot (CCIL #30): BC, exposure-matched duplication and CCIL through unchanged LeRobot commands.

    uv run python tools/pusht_pilot/run.py --root ROOT --source DATA --lerobot-python PY [--smoke] [STAGE ...]

The protocol is tools/pusht_pilot/protocol.json (docs/studies/pusht.md); this file only carries it out.
Stages, in order, each finished by an atomically written marker that later stages verify:
  labels    CCIL (the pinned ccil) converts DATA, generates, curates and exports one augmented LeRobot dataset;
  arms      the three training datasets: ccil (the export), bc (its training episodes alone) and duplication (the
            training episodes plus, for each CCIL label in order, a one-frame copy of its source demonstration frame);
  train     every (arm, seed) unit through an unchanged lerobot-train, under a supervised bounded scheduler;
  select    each unit's checkpoint with the lowest held-out real-episode action error (read-only, LeRobot side);
  evaluate  each selected checkpoint through an unchanged lerobot-eval on the shared starts;
  audit     every marker re-verified; arms, trainer exposure and evaluation starts reconciled;
  report    per-seed results, the paired contrasts and the signal rule.
A rerun resumes: a verified unit is never recomputed; an incomplete one is moved aside, kept, and redone. Progress
goes to ROOT/progress.log, at every stage boundary and at least every five minutes during training.
--smoke shrinks training and evaluation to exercise every stage, and is recorded in the root identity.
"""
import argparse
import hashlib
import json
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
PROTOCOL = HERE / "protocol.json"
CONFIG = HERE / "pusht.yaml"
STAGES = ("labels", "arms", "train", "select", "evaluate", "audit", "report")
COLUMNS = ("ccil.synthetic", "ccil.source_episode", "ccil.source_frame", "ccil.source_index", "ccil.candidate_row")
SMOKE = {"steps": 200, "save_freq": 100, "seeds": [1], "n_episodes": 4, "batch_size": 2}


class PilotError(RuntimeError):
    """A check of the pilot failed; the stage stops and its evidence is kept."""


def sha256(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def tree(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): sha256(p) for p in sorted(root.rglob("*")) if p.is_file()}


def write_json(path: Path, doc) -> None:
    tmp = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=True, default=str) + "\n")
    os.replace(tmp, path)


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Pilot:
    PROTOCOL = PROTOCOL  # a study built on this runner (tools/pusht_v2) names its own
    SELECTOR = (HERE / "lerobot_select.py", "mean_l2")  # the held-out scorer and its score, lower is better
    CONFIG = CONFIG
    PRESERVE = ("next.reward", "next.done", "next.success")  # source columns CCIL's conversion keeps as metadata
    REPEAT_TOLERANCE = {"max_reward": 1e-12}  # the audit's repeat evaluation; every other episode field is exact

    def __init__(self, root: Path, source: Path, lerobot_python: Path, smoke: bool):
        self.root, self.source, self.py, self.smoke = root, source, lerobot_python, smoke
        self.protocol = json.loads(self.PROTOCOL.read_text())
        t = dict(self.protocol["training"])
        e = dict(self.protocol["evaluation"])
        if smoke:
            t.update(steps=SMOKE["steps"], save_freq=SMOKE["save_freq"], seeds=SMOKE["seeds"])
            e.update(n_episodes=SMOKE["n_episodes"], batch_size=SMOKE["batch_size"])
        self.training, self.evaluation = t, e
        self.bin = lerobot_python.parent
        root.mkdir(parents=True, exist_ok=True)

    # --- identity and progress ---------------------------------------------------------------------------------
    def log(self, msg: str) -> None:
        with open(self.root / "progress.log", "a") as f:
            f.write(f"{now()} {msg}\n")
            f.flush()
        print(msg, flush=True)

    def identity(self) -> dict:
        git = lambda *a: subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()  # noqa: E731
        import importlib.metadata as md
        ccil_url = json.loads(md.distribution("ccil").read_text("direct_url.json") or "{}")
        freeze = subprocess.run([str(self.py), "-m", "pip", "freeze"], capture_output=True, text=True).stdout or \
            subprocess.run(["uv", "pip", "freeze", "--python", str(self.py)], capture_output=True, text=True).stdout
        gpus = subprocess.run(["nvidia-smi", "--query-gpu=index,name,memory.total", "--format=csv,noheader"],
                              capture_output=True, text=True).stdout.strip() if shutil.which("nvidia-smi") else None
        return {"protocol_sha256": sha256(self.PROTOCOL), "config_sha256": sha256(self.CONFIG), "smoke": self.smoke,
                "ilsb_commit": git("rev-parse", "HEAD"),
                "ilsb_dirty": bool(git("status", "--porcelain", "--", "src", "tools", "pyproject.toml", "uv.lock")),
                "ccil_commit": ccil_url.get("vcs_info", {}).get("commit_id"),
                "lerobot_freeze_sha256": hashlib.sha256(freeze.encode()).hexdigest(),
                "lerobot": next((ln.split("==")[1] for ln in freeze.splitlines() if ln.startswith("lerobot==")), None),
                "source_sha256": tree(self.source), "host": platform.node(), "gpus": gpus,
                "python": platform.python_version(), "numpy": np.__version__}

    def check_root(self) -> None:
        ident = self.identity()
        if ident["ccil_commit"] != self.protocol["ccil"]["commit"]:
            raise PilotError(f"ccil is {ident['ccil_commit']}, the protocol pins {self.protocol['ccil']['commit']}")
        if ident["lerobot"] != self.protocol["lerobot"]["version"]:
            raise PilotError(f"lerobot is {ident['lerobot']}, the protocol pins {self.protocol['lerobot']['version']}")
        path = self.root / "root.json"
        if path.exists():
            old = json.loads(path.read_text())
            keep = ("protocol_sha256", "config_sha256", "smoke", "ilsb_commit", "ccil_commit", "lerobot_freeze_sha256",
                    "source_sha256", "host")
            diff = [k for k in keep if old.get(k) != ident.get(k)]
            if diff:
                raise PilotError(f"{path} was created under another identity ({diff}); use a new root")
        else:
            if ident["ilsb_dirty"] and not self.smoke:
                raise PilotError("the pilot runs from a clean, committed ilsb checkout")
            write_json(path, {**ident, "created": now(), "command": sys.argv})
            self.log(f"root created: ilsb {ident['ilsb_commit'][:12]}, ccil {ident['ccil_commit'][:12]}, "
                     f"lerobot {ident['lerobot']}, smoke {self.smoke}, gpus {ident['gpus']}")

    def marker(self, name: str) -> dict | None:
        p = self.root / f"{name}.json"
        return json.loads(p.read_text()) if p.exists() else None

    def need(self, name: str) -> dict:
        m = self.marker(name)
        if m is None:
            raise PilotError(f"stage {name} has not completed; run it first")
        return m

    # --- stages ----------------------------------------------------------------------------------------------------
    def ccil(self, *args, cwd):
        r = subprocess.run([sys.executable, "-m", "ccil", *args], cwd=cwd, capture_output=True, text=True)
        if r.returncode:
            raise PilotError(f"ccil {' '.join(args[:2])} exited {r.returncode}:\n{r.stderr[-1500:]}")
        return r.stdout

    def labels(self) -> None:
        from ccil.lerobot import convert
        d = self.root / "labels"
        if self.marker("labels"):
            return
        if d.exists():
            d.rename(self.root / f"labels.incomplete-{int(time.time())}")
        d.mkdir()
        convert(self.label_source(), d / "pusht.npz", d / "conversion.json", preserve=list(self.PRESERVE))
        shutil.copy(self.CONFIG, d / "pusht.yaml")
        self.log("labels: converted; generating")
        gen = self.ccil("generate", "--config", "pusht.yaml", cwd=d)
        cur = self.ccil("curate", "runs/generate-result.json", cwd=d)
        result = sorted((d / "runs").glob("curate-*.json"))
        if len(result) != 1:
            raise PilotError(f"expected one curate result, found {len(result)}")
        self.ccil("export", "lerobot", str(result[0].relative_to(d)), "--conversion", "conversion.json", "--out",
                  "export", "--revision", self.protocol["dataset"]["revision"], cwd=d)
        manifest = json.loads((d / "export" / "ccil" / "export.json").read_text())
        (d / "summary.txt").write_text(gen + cur)
        write_json(self.root / "labels.json", {"export": "labels/export", "manifest_sha256": sha256(d / "export" / "ccil" / "export.json"),
                                               "counts": manifest["counts"], "held_out": manifest["episodes"]["held_out"],
                                               "train": manifest["episodes"]["train"], "curate_result": result[0].name,
                                               "inputs": manifest["inputs"], "completed": now()})
        self.log(f"labels: {manifest['counts']}")

    def arms(self) -> None:
        import pyarrow.parquet as pq
        from ccil.lerobot import write_dataset
        from ccil.schema import Episode
        if self.marker("arms"):
            return
        lab = self.need("labels")
        exp = self.root / lab["export"]
        manifest = json.loads((exp / "ccil" / "export.json").read_text())
        t = pq.read_table(sorted((exp / "data").rglob("*.parquet"))).to_pydict()
        info = json.loads((exp / "meta" / "info.json").read_text())
        tasks = pq.read_table(exp / "meta" / "tasks.parquet").to_pydict()
        task = dict(zip(tasks["task_index"], tasks["task"]))
        syn = np.array(t["ccil.synthetic"], bool)
        arr = {k: np.array(t[k], np.float32) for k in ("observation.state", "observation.environment_state", "action")}
        col = {k: np.array(t[k], np.int64) for k in COLUMNS}
        ts, tidx = np.array(t["timestamp"], np.float64), np.array(t["task_index"], np.int64)
        demo_rows = np.flatnonzero(~syn)
        by_index = {int(col["ccil.source_index"][i]): i for i in demo_rows}
        episodes, order = {}, []
        for i in demo_rows:
            e = int(col["ccil.source_episode"][i])
            if e not in episodes:
                episodes[e] = []
                order.append(e)
            episodes[e].append(i)

        def demo_episode(e):
            rows = np.array(episodes[e])
            return Episode(str(e), {"state": arr["observation.state"][rows],
                                    "state.environment_state": arr["observation.environment_state"][rows]},
                           arr["action"][rows], timestamps=ts[rows], task=task[int(tidx[rows[0]])])

        def frame_episode(name, i):
            return Episode(name, {"state": arr["observation.state"][i:i + 1],
                                  "state.environment_state": arr["observation.environment_state"][i:i + 1]},
                           arr["action"][i:i + 1], timestamps=np.zeros(1), task=task[int(tidx[i])])

        label_rows = np.flatnonzero(syn)
        dup_rows = []
        for i in label_rows:  # each label's source demonstration frame, in label order
            src = int(col["ccil.source_index"][i])
            if src not in by_index:
                raise PilotError(f"label {int(col['ccil.candidate_row'][i])} has no source frame {src} in the export")
            dup_rows.append(by_index[src])

        def columns_for(rows_of):
            def columns(e, ep):
                rows = rows_of[e]
                out = {k: col[k][rows] for k in COLUMNS}
                return out
            return columns

        arms = {}
        for arm in ("bc", "duplication"):
            dest = self.root / "arms" / arm
            if dest.exists():
                dest.rename(dest.with_name(f"{arm}.incomplete-{int(time.time())}"))
            dest.parent.mkdir(exist_ok=True)
            demos = [demo_episode(e) for e in order]
            rows_of = [np.array(episodes[e]) for e in order]
            eps = list(demos)
            if arm == "duplication":
                for k, (i, j) in enumerate(zip(label_rows, dup_rows)):
                    eps.append(frame_episode(f"dup.{k}", j))
                    # provenance of the duplicate: synthetic 1 (an added row), its source, and the label it matches
                    rows_of.append(np.array([j]))
                cols = columns_for(rows_of)

                def columns(e, ep, cols=cols, n=len(demos)):
                    out = cols(e, ep)
                    if e >= n:
                        i = label_rows[e - n]
                        out = {**out, "ccil.synthetic": np.ones(1, np.int64),
                               "ccil.candidate_row": col["ccil.candidate_row"][i:i + 1]}
                    return out
            else:
                columns = columns_for(rows_of)
            write_dataset(iter(eps), dest, fps=float(info["fps"]), columns=columns, single_frame_episodes=True)
            arms[arm] = dest
        arms["ccil"] = exp
        # Reconciliation: equal added rows and source multisets for duplication and ccil; identical demonstrations.
        summary = {}
        for arm, path in arms.items():
            tt = pq.read_table(sorted((path / "data").rglob("*.parquet"))).to_pydict()
            s = np.array(tt["ccil.synthetic"], bool)
            src = np.sort(np.array(tt["ccil.source_index"], np.int64)[s])
            demo = np.array(tt["ccil.source_index"], np.int64)[~s]
            summary[arm] = {"path": path.relative_to(self.root).as_posix(), "frames": int(len(s)),
                            "demonstration_frames": int((~s).sum()), "added_rows": int(s.sum()),
                            "added_source_index_sha256": hashlib.sha256(src.tobytes()).hexdigest(),
                            "demonstration_index_sha256": hashlib.sha256(demo.tobytes()).hexdigest(),
                            "files": tree(path)}
        ok = (summary["duplication"]["added_rows"] == summary["ccil"]["added_rows"] == manifest["counts"]["synthetic_rows"]
              and summary["duplication"]["added_source_index_sha256"] == summary["ccil"]["added_source_index_sha256"]
              == manifest["synthetic_source_index_sha256"]
              and summary["bc"]["added_rows"] == 0
              and len({v["demonstration_index_sha256"] for v in summary.values()}) == 1)
        if not ok:
            raise PilotError(f"the arms do not reconcile: {json.dumps({k: {x: y for x, y in v.items() if x != 'files'} for k, v in summary.items()})}")
        write_json(self.root / "arms.json", {"arms": summary, "completed": now()})
        self.log("arms: " + ", ".join(f"{k} {v['frames']} frames ({v['added_rows']} added)" for k, v in summary.items()))

    def units(self):
        return [(arm, s) for s in self.training["seeds"] for arm in self.protocol["arms"]]

    def train_cmd(self, arm, seed, out):
        t, p = self.training, self.protocol["policy"]
        root = self.root / self.need("arms")["arms"][arm]["path"]
        return [str(self.bin / "lerobot-train"), f"--dataset.repo_id=local/pusht-{arm}", f"--dataset.root={root}",
                f"--policy.type={p['type']}", f"--policy.n_obs_steps={p['n_obs_steps']}",
                f"--policy.chunk_size={p['chunk_size']}", f"--policy.n_action_steps={p['n_action_steps']}",
                f"--policy.use_vae={str(p['use_vae']).lower()}", "--policy.push_to_hub=false",
                f"--policy.device={t['device']}", f"--steps={t['steps']}", f"--batch_size={t['batch_size']}",
                f"--save_freq={t['save_freq']}", "--log_freq=1000", f"--num_workers={t['num_workers']}",
                f"--seed={seed}", f"--output_dir={out}", "--wandb.enable=false",
                *([f"--policy.normalization_mapping={json.dumps(p['normalization_mapping'])}"]
                  if "normalization_mapping" in p else [])]

    def unit_done(self, arm, seed) -> dict | None:
        p = self.root / "units" / f"{arm}-s{seed}" / "train.json"
        if not p.exists():
            return None
        m = json.loads(p.read_text())
        for c in m["checkpoints"]:
            if sha256(self.root / c["weights"]) != c["sha256"]:
                raise PilotError(f"{arm}-s{seed}: checkpoint {c['step']} changed since training")
        return m

    def finish_unit(self, arm, seed, out: Path, log: Path, seconds: float) -> None:
        ckpts = []
        for d in sorted((out / "checkpoints").iterdir()):
            if d.name == "last" or not d.name.isdigit():
                continue
            w = d / "pretrained_model" / "model.safetensors"
            if not w.exists():
                raise PilotError(f"{arm}-s{seed}: checkpoint {d.name} has no weights")
            ckpts.append({"step": int(d.name), "path": (d / "pretrained_model").relative_to(self.root).as_posix(),
                          "weights": w.relative_to(self.root).as_posix(), "sha256": sha256(w)})
        want = list(range(self.training["save_freq"], self.training["steps"] + 1, self.training["save_freq"]))
        if [c["step"] for c in ckpts] != want:
            raise PilotError(f"{arm}-s{seed}: checkpoints {[c['step'] for c in ckpts]}, expected {want}")
        last = [ln for ln in log.read_text().splitlines() if "step:" in ln and "smpl:" in ln]
        write_json(out.parent / "train.json", {"arm": arm, "seed": seed, "checkpoints": ckpts, "seconds": seconds,
                                               "command": self.train_cmd(arm, seed, out), "last_log": last[-1:] if last else [],
                                               "samples": self.training["steps"] * self.training["batch_size"],
                                               "completed": now()})

    def train(self) -> None:
        self.need("arms")
        todo = [u for u in self.units() if self.unit_done(*u) is None]
        if not todo:
            return
        slots = [g for g in self.protocol["compute"]["gpus"] for _ in range(self.protocol["compute"]["concurrency_per_gpu"])]
        running, failed, done = {}, [], len(self.units()) - len(todo)
        stall = 60 * self.protocol["compute"]["stall_minutes"]
        last_report = 0.0
        self.log(f"train: {len(todo)} units to run, {done} verified; {len(slots)} slots on GPUs {sorted(set(slots))}")
        try:
            while todo or running:
                while todo and len(running) < len(slots):
                    arm, seed = todo.pop(0)
                    unit = self.root / "units" / f"{arm}-s{seed}"
                    out = unit / "train"
                    if out.exists():  # an incomplete earlier attempt: keep it for diagnosis, start clean
                        out.rename(unit / f"train.incomplete-{int(time.time())}")
                    unit.mkdir(parents=True, exist_ok=True)
                    busy = [r["gpu"] for r in running.values()]
                    gpu = min((g for g in sorted(set(slots)) if busy.count(g) < slots.count(g)), key=busy.count)
                    log = unit / f"train-{int(time.time())}.log"
                    env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu)}
                    proc = subprocess.Popen(self.train_cmd(arm, seed, out), stdout=open(log, "w"),
                                            stderr=subprocess.STDOUT, env=env, cwd=unit, start_new_session=True)
                    running[(arm, seed)] = {"proc": proc, "gpu": gpu, "log": log, "out": out, "start": time.time()}
                    self.log(f"train: started {arm}-s{seed} on GPU {gpu}, pid {proc.pid}, log {log.relative_to(self.root)}")
                time.sleep(5 if self.smoke else 30)
                for key, r in list(running.items()):
                    rc = r["proc"].poll()
                    if rc is None:
                        if time.time() - r["log"].stat().st_mtime > stall:
                            failed.append((key, f"no log progress for {self.protocol['compute']['stall_minutes']} min"))
                        continue
                    del running[key]
                    if rc != 0:
                        failed.append((key, f"lerobot-train exited {rc}"))
                        continue
                    self.finish_unit(*key, r["out"], r["log"], time.time() - r["start"])
                    done += 1
                    self.log(f"train: completed {key[0]}-s{key[1]} in {time.time() - r['start']:.0f} s ({done}/{len(self.units())})")
                if failed:
                    raise PilotError("; ".join(f"{a}-s{s}: {why}" for (a, s), why in failed))
                if time.time() - last_report > (60 if self.smoke else 300):
                    last_report = time.time()
                    mem = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"],
                                         capture_output=True, text=True).stdout.strip().replace("\n", "; ") \
                        if shutil.which("nvidia-smi") else "n/a"
                    self.log(f"train: running {sorted(f'{a}-s{s}' for a, s in running)}, queued {len(todo)}, "
                             f"completed {done}/{len(self.units())}; GPU memory {mem}")
        except BaseException as err:
            for key, r in running.items():  # stop the siblings: their partial outputs are never marked complete
                try:
                    os.killpg(r["proc"].pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
            for key, r in running.items():
                r["proc"].wait()
            write_json(self.root / f"train-failure-{int(time.time())}.json",
                       {"error": str(err), "stopped": sorted(f"{a}-s{s}" for a, s in running), "at": now()})
            self.log(f"train: FAILED: {err}")
            raise

    def held_out(self) -> list:
        return self.need("labels")["held_out"]

    def label_source(self) -> Path:
        return self.source

    def selection_source(self) -> Path:
        return self.source

    def select(self) -> None:
        held = self.held_out()
        for arm, seed in self.units():
            unit = self.root / "units" / f"{arm}-s{seed}"
            if (unit / "select.json").exists():
                continue
            m = self.unit_done(arm, seed)
            if m is None:
                raise PilotError(f"{arm}-s{seed} has not finished training")
            (unit / "held_out.json").write_text(json.dumps([int(e) for e in held]))
            (unit / "checkpoints.json").write_text(json.dumps({str(c["step"]): str(self.root / c["path"]) for c in m["checkpoints"]}))
            r = subprocess.run([str(self.py), str(self.SELECTOR[0]), str(self.selection_source()), str(unit / "held_out.json"),
                                str(unit / "checkpoints.json"), str(unit / "select-scores.json")],
                               capture_output=True, text=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            if r.returncode:
                raise PilotError(f"{arm}-s{seed}: selection exited {r.returncode}:\n{r.stderr[-1500:]}")
            scores = json.loads((unit / "select-scores.json").read_text())
            key = self.SELECTOR[1]
            best = min(sorted(scores[key], key=int), key=lambda k: scores[key][k])  # ties: earlier step
            chosen = next(c for c in m["checkpoints"] if str(c["step"]) == best)
            write_json(unit / "select.json", {"scores": scores, "selected": chosen, "completed": now()})
            self.log(f"select: {arm}-s{seed} step {best} (held-out {key} {scores[key][best]:.4f})")

    def evaluate(self) -> None:
        e = self.evaluation
        for arm, seed in self.units():
            unit = self.root / "units" / f"{arm}-s{seed}"
            if (unit / "eval.json").exists():
                continue
            sel = json.loads((unit / "select.json").read_text())["selected"]
            if sha256(self.root / sel["weights"]) != sel["sha256"]:
                raise PilotError(f"{arm}-s{seed}: the selected checkpoint changed since training")
            out = unit / "eval"
            if out.exists():
                out.rename(unit / f"eval.incomplete-{int(time.time())}")
            cmd = [str(self.bin / "lerobot-eval"), f"--policy.path={self.root / sel['path']}", f"--env.type={e['env']}",
                   f"--env.obs_type={e['obs_type']}", f"--eval.n_episodes={e['n_episodes']}",
                   f"--eval.batch_size={e['batch_size']}", f"--eval.use_async_envs={str(e['async_envs']).lower()}",
                   f"--seed={e['start_seed']}",
                   f"--policy.device={self.training['device']}", f"--output_dir={out}"]
            r = subprocess.run(cmd, capture_output=True, text=True, cwd=unit, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
            (unit / "eval.log").write_text(r.stdout + r.stderr)
            if r.returncode:
                raise PilotError(f"{arm}-s{seed}: lerobot-eval exited {r.returncode}:\n{r.stderr[-1500:]}")
            eps = per_episode(json.loads((out / "eval_info.json").read_text()), e["start_seed"])
            if len(eps) != e["n_episodes"]:
                raise PilotError(f"{arm}-s{seed}: {len(eps)} evaluated episodes, expected {e['n_episodes']}")
            res = {"avg_max_reward": float(np.mean([ep["max_reward"] for ep in eps])),
                   "pc_success": float(np.mean([bool(ep["success"]) for ep in eps]) * 100)}
            write_json(unit / "eval.json", {"episodes": eps, **res, "command": cmd,
                                            "eval_info_sha256": sha256(out / "eval_info.json"), "completed": now()})
            self.log(f"evaluate: {arm}-s{seed} avg_max_reward {res['avg_max_reward']:.4f}, pc_success {res['pc_success']:.1f}")

    def audit(self) -> None:
        arms = self.need("arms")["arms"]
        problems = []
        for arm, a in arms.items():
            now_files = tree(self.root / a["path"])
            if now_files != a["files"]:
                problems.append(f"arm {arm}: its files changed since they were built")
        exposure, starts = {}, set()
        for arm, seed in self.units():
            unit = self.root / "units" / f"{arm}-s{seed}"
            m = self.unit_done(arm, seed)
            ev = json.loads((unit / "eval.json").read_text())
            if sha256(unit / "eval" / "eval_info.json") != ev["eval_info_sha256"]:
                problems.append(f"{arm}-s{seed}: eval_info.json changed")
            starts.add(tuple(ep["seed"] for ep in ev["episodes"]))
            exposure[f"{arm}-s{seed}"] = {"samples": m["samples"], "dataset_frames": arms[arm]["frames"],
                                          "epochs": m["samples"] / arms[arm]["frames"], "last_log": m["last_log"]}
        # Fixed starts: re-evaluate the first unit with its own command. Every episode's seed, success and summed
        # reward must reproduce exactly; its max reward to 1e-12 (protocol amendment 1: the last bit can differ).
        arm0, seed0 = self.units()[0]
        unit0 = self.root / "units" / f"{arm0}-s{seed0}"
        first = json.loads((unit0 / "eval.json").read_text())
        again = unit0 / f"eval-repeat-{int(time.time())}"
        cmd = [c if not c.startswith("--output_dir=") else f"--output_dir={again}" for c in first["command"]]
        r = subprocess.run(cmd, capture_output=True, text=True, cwd=unit0, env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        repeat = per_episode(json.loads((again / "eval_info.json").read_text()), self.evaluation["start_seed"]) \
            if r.returncode == 0 else []
        tol = self.REPEAT_TOLERANCE
        exact = [k for k in ("index", "seed", "sum_reward", "success") if k not in tol]
        max_diff = max((abs(a["max_reward"] - b["max_reward"]) for a, b in zip(repeat, first["episodes"])), default=None)
        reproduced = (len(repeat) == len(first["episodes"])
                      and all(abs(a[k] - b[k]) <= t for a, b in zip(repeat, first["episodes"]) for k, t in tol.items())
                      and all(a[k] == b[k] for a, b in zip(repeat, first["episodes"]) for k in exact))
        if not reproduced:
            problems.append(f"re-evaluating {arm0}-s{seed0} did not reproduce its episodes (exit {r.returncode})")
        if len({v["samples"] for v in exposure.values()}) != 1:
            problems.append("units saw different numbers of samples")
        if len(starts) != 1:
            problems.append("units were evaluated from different starts")
        if arms["duplication"]["frames"] != arms["ccil"]["frames"]:
            problems.append("duplication and ccil have different frame counts")
        write_json(self.root / "audit.json", {"problems": problems, "exposure": exposure, "completed": now(),
                                              "repeat_evaluation": {"unit": f"{arm0}-s{seed0}", "reproduced": reproduced,
                                                                    "max_reward_max_abs_diff": max_diff},
                                              "pass": not problems})
        if problems:
            raise PilotError("audit: " + "; ".join(problems))
        self.log("audit: passed")

    def report(self) -> None:
        sys.path.insert(0, str(REPO / "src"))
        from ilsb.analysis import paired
        if not self.need("audit")["pass"]:
            raise PilotError("the audit did not pass")
        a = self.protocol["analysis"]
        res = {}
        for arm, seed in self.units():
            ev = json.loads((self.root / "units" / f"{arm}-s{seed}" / "eval.json").read_text())
            sel = json.loads((self.root / "units" / f"{arm}-s{seed}" / "select.json").read_text())
            res.setdefault(arm, {})[seed] = {"avg_max_reward": ev["avg_max_reward"], "pc_success": ev["pc_success"],
                                            "selected_step": sel["selected"]["step"]}
        min_pairs = 1 if self.smoke else a["min_pairs"]

        def contrast(metric, spec):
            x, y = (s.strip() for s in spec.split("-"))
            out = paired({s: v[metric] for s, v in res[x].items()}, {s: v[metric] for s, v in res[y].items()},
                         a["confidence"], min_pairs)
            out["per_seed"] = {s: res[x][s][metric] - res[y][s][metric] for s in sorted(res[x])}
            return {"metric": metric, "contrast": spec, **out}
        primary = contrast(a["primary"]["metric"], a["primary"]["contrast"])
        positive = sum(v > 0 for v in primary["per_seed"].values())
        signal_ = primary["mean"] is not None and primary["mean"] > 0 and positive >= (4 if not self.smoke else 1)
        report = {"results": res, "primary": primary, "secondary": [contrast(s["metric"], s["contrast"]) for s in a["secondary"]],
                  "signal": {"rule": a["signal"], "positive_seeds": positive, "met": bool(signal_)},
                  "smoke": self.smoke, "completed": now()}
        write_json(self.root / "report.json", report)
        self.log(f"report: primary {primary['contrast']} on {primary['metric']}: mean {primary['mean']}, "
                 f"interval {primary.get('interval')}, positive on {positive} seeds; signal {'met' if signal_ else 'not met'}")


def per_episode(info, start_seed: int) -> list:
    """Each episode of a one-task lerobot-eval eval_info.json (LeRobot 0.6.1's per_task format). Episode i is reset
    with seed start_seed + i (lerobot_eval.py, rollout seeding), which the file does not record; the audit checks
    that the starts are fixed by re-evaluating one unit."""
    tasks = info.get("per_task") or []
    if len(tasks) != 1:
        raise PilotError(f"expected one evaluated task, found {len(tasks)}")
    m = tasks[0]["metrics"]
    return [{"index": i, "seed": start_seed + i, "max_reward": float(mx), "sum_reward": float(sm), "success": bool(ok)}
            for i, (mx, sm, ok) in enumerate(zip(m["max_rewards"], m["sum_rewards"], m["successes"]))]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, required=True)
    ap.add_argument("--source", type=Path, required=True, help="the LeRobot dataset at the protocol's revision")
    ap.add_argument("--lerobot-python", type=Path, required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("stages", nargs="*", default=list(STAGES))
    a = ap.parse_args()
    p = Pilot(a.root.resolve(), a.source.resolve(), a.lerobot_python, a.smoke)
    try:
        p.check_root()
        for s in STAGES:
            if s in a.stages:
                p.log(f"stage {s}: start")
                getattr(p, s)()
                write_json(p.root / f"stage-{s}.json", {"stage": s, "completed": now()}) if s in ("train", "select", "evaluate") else None
                p.log(f"stage {s}: done")
    except PilotError as e:
        p.log(f"STOPPED: {e}")
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
