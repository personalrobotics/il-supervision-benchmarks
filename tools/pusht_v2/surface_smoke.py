"""The PushT surface study's smoke evidence (ilsb#5): the whole chain on a small specification, outcome-free.

    ILSB_LEROBOT_PYTHON=PY ILSB_PUSHT_DATASET=DIR uv run python tools/pusht_v2/surface_smoke.py --work NEW_DIR \
        [--out docs/studies/evidence/pusht-surface/smoke.json]

Needs two GPUs (slots 0,1). It runs surface-smoke.json (r = 0.1; BC, duplication and CCIL at alpha = 0.8; boundaries
500/1000/2000 every 250 steps; 4 evaluation episodes) and records:
  chain        every stage of root A, from preflight to report, passing; the pool's checks; each unit's selection,
               extension, exposure and index digests; the audit's re-scores
  failure      root B: one learner is killed (SIGKILL) after its first checkpoint; the scheduler must fail promptly,
               name that unit, stop its siblings and leave no entry for it; the rerun must resume and give every unit
               the weights, selection and index digests of root A
  equivalence  lerobot-train itself and this learner fed lerobot-train's sampler order, 200 steps on the r = 0.1 real
               rows: identical weights
No coverage or success value is recorded: closed-loop outcomes of a 2k-step smoke say nothing and are left out.
Every recorded value is a function of the commit, the host's GPU model and the seeds; a rerun on the same host
reproduces the file byte for byte.
"""
import argparse
import hashlib
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
SPEC = REPO / "tools" / "pusht_v2" / "surface-smoke.json"
PY = os.environ.get("ILSB_LEROBOT_PYTHON", "")


def sha(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ilsb(*args, check=True) -> subprocess.CompletedProcess:
    r = subprocess.run([sys.executable, "-m", "ilsb", *map(str, args)], capture_output=True, text=True, cwd=REPO)
    if check and r.returncode:
        raise SystemExit(f"ilsb {' '.join(map(str, args))} exited {r.returncode}:\n{r.stdout[-2000:]}{r.stderr[-2000:]}")
    return r


def stage(root: Path, name: str) -> dict:
    return json.loads((root / "stages" / f"{name}.json").read_text())


def units_of(root: Path, store: Path) -> dict:
    """Each unit's deterministic record: weights, selection, extension, exposure and index digests."""
    out = {}
    for key, row in stage(root, "train")["units"].items():
        e = store / "policy" / row["entry"]
        res = json.loads((e / "result.json").read_text())
        h = res["history"]
        out[key] = {"model_sha256": sha(e / "model.safetensors"), "outcome": res["outcome"], "reached": res["reached"],
                    "selected_step": res["selected_step"], "per_batch": res["per_batch"], "rows": res["rows"],
                    "exposure": res["exposure"],
                    "infonce": {s: v["infonce"] for s, v in sorted(h["checkpoints"].items(), key=lambda x: int(x[0]))},
                    "index_sha256": {s: v["index_sha256"] for s, v in sorted(h["segments"].items(),
                                                                             key=lambda x: int(x[0]))}}
    return out


def chain(work: Path) -> dict:
    root = work / "A"
    r = ilsb("run", SPEC, "--root", root, "--gpus", "0,1", "--allow-dirty", check=False)
    done = json.loads(r.stdout.strip().splitlines()[-1]) if r.stdout.strip() else {}
    pre, au, ea = stage(root, "preflight"), stage(root, "audit"), stage(root, "evaluation_audit")
    pool = pre["pools"]["r0.1"]
    return {"stages": done, "all_pass": r.returncode == 0,
            "preflight": {"subsets": pre["subsets"], "real_rows": pre["real_rows"],
                          "held_out_frames": pre["held_out_frames"], "planned_units": pre["planned_units"]},
            "pool": {k: pool[k] for k in ("proposals", "transitions", "selected", "usable_fraction", "violations",
                                          "held_out_current_frames", "decode_drift_px", "residual",
                                          "previous_slot_displacement_px", "nearest_real_distance",
                                          "outside_frozen_range_fraction", "ccil_counts", "problems")},
            "audit": {"all_pass": au["all_pass"], "problems": au["problems"]},
            "evaluation_audit": {"all_pass": ea["all_pass"], "problems": ea["problems"],
                                 "repeat_max_deviation": ea["repeat"]["max_deviation"]},
            "units": units_of(root, root / "store")}


def failure(work: Path, reference: dict) -> dict:
    """Kill one learner after its first checkpoint; the run must stop, name it, keep nothing of it, then resume."""
    root = work / "B"
    ilsb("stage", "preflight", SPEC, "--root", root, "--allow-dirty")
    proc = subprocess.Popen([sys.executable, "-m", "ilsb", "stage", "train", str(SPEC), "--root", str(root),
                             "--gpus", "0,1", "--allow-dirty"], cwd=REPO, stdout=subprocess.PIPE, text=True)
    victim, pid = None, None
    while victim is None and proc.poll() is None:
        time.sleep(1)
        for hb in sorted((root / "work").glob("*/work/history.json")):
            meta = hb.parent / "heartbeat.json"
            if meta.exists() and json.loads(hb.read_text())["checkpoints"]:
                victim, pid = hb.parents[1], json.loads(meta.read_text())["pid"]
                break
    if victim is None:
        raise SystemExit("no unit reached a checkpoint before the train stage ended")
    ppid = int(Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()[1])  # its `ilsb unit` parent
    os.kill(pid, signal.SIGKILL)
    killed_at = time.time()
    proc.wait()
    stopped_s = time.time() - killed_at
    tr = stage(root, "train")
    starts = [ln.split() for ln in (root / "progress.log").read_text().splitlines() if " start " in ln]
    key = next(w[2] for w in starts if w[-1] == f"{ppid})")
    entries = {p.name for p in (root / "store" / "policy").iterdir()} if (root / "store" / "policy").exists() else set()
    victim_result = (victim / "result.json").exists()
    resumed = ilsb("run", SPEC, "--root", root, "--gpus", "0,1", "--allow-dirty", check=False)
    again = units_of(root, root / "store") if resumed.returncode == 0 else {}
    same = {k: again.get(k, {}).get(f) == v[f] for k, v in reference.items()
            for f in ("model_sha256", "selected_step", "reached", "index_sha256", "infonce")}
    return {"killed": key, "failed_units": sorted(tr["failed"]), "failed_unit_named": sorted(tr["failed"]) == [key],
            "train_stage_passed": tr["all_pass"], "stopped_within_60_s": stopped_s < 60,
            "killed_unit_has_no_result": not victim_result,
            "store_entries_after_failure": len(entries), "resume_all_pass": resumed.returncode == 0,
            "resumed_from_checkpoint": any("resumed at step" in p.read_text()
                                           for p in (root / "work").glob("*/work/progress.log")),
            "resumed_units_equal_root_A": all(same.values()) and len(again) == len(reference)}


def equivalence(work: Path) -> dict:
    """lerobot-train and this learner, 200 steps on the r = 0.1 real rows in lerobot-train's own order."""
    from ccil.lerobot import write_dataset
    from ccil.schema import Episode
    root = work / "A"
    src = np.load(root / "source.npz")
    eps = stage(root, "preflight")["subsets"]["r0.1"]
    ds = work / "equivalence" / "dataset"
    episodes = []
    for e in eps:
        r = np.flatnonzero(src["episode"] == e)
        episodes.append(Episode(str(e), {"state": src["state"][r], "state.environment_state": src["env"][r]},
                                src["action"][r], timestamps=np.arange(len(r)) / float(src["fps"]),
                                task="Push the T-shaped block onto the target."))
    write_dataset(iter(episodes), ds, fps=float(src["fps"]))
    spec = json.loads(SPEC.read_text())
    lr, pol = spec["learner"], spec["learner"]["policy"]
    seed, steps = 99, 200
    out = work / "equivalence" / "lerobot-train"
    args = [f"--policy.{k}={str(v).lower() if isinstance(v, bool) else v}" for k, v in pol.items()]
    proc = subprocess.Popen([str(Path(PY).parent / "lerobot-train"), "--dataset.repo_id=local/pusht-equivalence",
                             f"--dataset.root={ds}", "--policy.type=ibc", *args,
                             "--policy.scheduler_decay_steps=80000", "--policy.push_to_hub=false",
                             "--policy.device=cuda", "--steps=80000", f"--batch_size={lr['batch_size']}",
                             f"--save_freq={steps}", "--log_freq=50", "--num_workers=4", f"--seed={seed}",
                             f"--output_dir={out}", "--wandb.enable=false"],
                            env=os.environ | {"CUDA_VISIBLE_DEVICES": "0"}, stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)
    mark = out / "checkpoints" / f"{steps:06d}" / "training_state" / "training_step.json"
    while not mark.exists() and proc.poll() is None:
        time.sleep(1)
    time.sleep(3)
    proc.kill()
    proc.wait()
    unit = work / "equivalence" / "learner"
    unit.mkdir(parents=True)
    val = np.flatnonzero(np.isin(src["episode"], spec["roles"]["held_out"]))
    real = np.flatnonzero(np.isin(src["episode"], eps))
    np.savez(unit / "rows.npz", **{f"real_{p}": src[p][real] for p in ("state", "env", "action")},
             **{f"val_{p}": src[p][val] for p in ("state", "env", "action")},
             syn_state=np.zeros((0, 4), np.float32), syn_env=np.zeros((0, 32), np.float32),
             syn_action=np.zeros((0, 2), np.float32))
    (unit / "unit.json").write_text(json.dumps({
        "seed": seed, "alpha": 0.0, "batch_size": lr["batch_size"], "horizon": 80000,
        "boundaries": [steps, 2 * steps, 80000], "cadence": steps, "log_every": 50, "policy": pol,
        "features_root": str(ds), "rows": "rows.npz", "stats": None, "selection": lr["selection"],
        "order": "lerobot", "stop_at": steps}))
    subprocess.run([PY, "-m", "pusht_surface_learner", "train", str(unit)], check=True,
                   env=os.environ | {"CUDA_VISIBLE_DEVICES": "1"}, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    theirs = sha(out / "checkpoints" / f"{steps:06d}" / "pretrained_model" / "model.safetensors")
    ours = sha(unit / "work" / "checkpoints" / f"{steps:06d}" / "pretrained_model" / "model.safetensors")
    return {"steps": steps, "seed": seed, "real_rows": int(len(real)), "lerobot_train_model_sha256": theirs,
            "learner_model_sha256": ours, "identical": theirs == ours}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    a.work = a.work.resolve()  # unit.json paths are resolved against each unit's own directory
    if a.work.exists():
        raise SystemExit(f"{a.work} exists; use a new directory")
    if not PY:
        raise SystemExit("set ILSB_LEROBOT_PYTHON")
    a.work.mkdir(parents=True)
    probe = subprocess.run([PY, "-c", "import torch; print(torch.cuda.get_device_name())"], capture_output=True,
                           text=True).stdout.strip()
    rec = {"spec_sha256": sha(SPEC), "gpu": probe}
    rec["chain"] = chain(a.work)
    rec["failure"] = failure(a.work, rec["chain"]["units"])
    rec["equivalence"] = equivalence(a.work)
    rec["all_pass"] = (rec["chain"]["all_pass"] and rec["chain"]["audit"]["all_pass"]
                       and rec["chain"]["evaluation_audit"]["all_pass"] and rec["failure"]["failed_unit_named"]
                       and not rec["failure"]["train_stage_passed"] and rec["failure"]["stopped_within_60_s"]
                       and rec["failure"]["killed_unit_has_no_result"] and rec["failure"]["resume_all_pass"]
                       and rec["failure"]["resumed_from_checkpoint"] and rec["failure"]["resumed_units_equal_root_A"]
                       and rec["equivalence"]["identical"])
    text = json.dumps(rec, indent=1, sort_keys=True) + "\n"
    (a.work / "smoke.json").write_text(text)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(json.dumps({"all_pass": rec["all_pass"], "out": str(a.out or a.work / "smoke.json")}))
    return 0 if rec["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
