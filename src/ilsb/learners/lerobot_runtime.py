"""The LeRobot side of ilsb's LeRobot learners (ilsb#50). It runs in the pinned LeRobot environment, by path:

    $ILSB_LEROBOT_PYTHON -I .../ilsb/learners/lerobot_runtime.py COMMAND ...

It imports only the standard library, NumPy, PyTorch and LeRobot (never ilsb or ccil), so the harness environment and
the learner environment stay separate. Every command is LeRobot's own mechanism, driven from a declared file:
  split    SOURCE OUT ROLES.json        LeRobot's split_dataset: OUT/train and OUT/held_out, each with the statistics of
                                        its own episodes only (aggregated from their per-episode statistics)
  windows  ROOT POLICY.json OUT.npz     every anchored window LeRobotDataset gives the policy's delta indices (POLICY
                                        is its type and config fields), as
                                        values and padding flags, and the anchors of lerobot-train's sampler
  train    UNIT_DIR                     lerobot-train on UNIT_DIR/unit.json; an interrupted unit resumes from its last
                                        checkpoint on CUDA and restarts on the CPU (RESUME_EXACT); then each
                                        checkpoint's held-out loss (score) in history.json
  score    CHECKPOINT HELD_OUT SEED BATCH DEVICE OUT.json
                                        one checkpoint's held-out loss
  stats    CHECKPOINT OUT.json          the statistics a checkpoint's normalizer and unnormalizer steps hold
  evaluate ARGS...                      lerobot-eval with deterministic cuDNN and cuBLAS
"""
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

DETERMINISTIC_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
# Where lerobot-train's own resume was measured to reach the uninterrupted run's parameters bit for bit. On the CPU it
# does not (the resumed segment diverges; the diffusion noise is drawn from the CPU generator there, which the data
# loader also draws from), so an interrupted CPU unit restarts from step 0 instead.
RESUME_EXACT = {"cuda": True, "cpu": False}


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def log(work: Path, msg: str) -> None:
    with open(work / "progress.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --- data -------------------------------------------------------------------------------------------------------------
def split(source: str, out: str, roles_path: str) -> None:
    from lerobot.datasets.dataset_tools import split_dataset
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    roles = json.loads(Path(roles_path).read_text())
    ds = LeRobotDataset("source", root=source)
    got = split_dataset(ds, {"train": roles["train"], "held_out": roles["held_out"]}, output_dir=out)
    print(json.dumps({k: {"episodes": v.num_episodes, "frames": v.num_frames} for k, v in got.items()}))


def policy_config(policy: dict):
    """The policy's LeRobot config (only its delta indices and anchor rule are read from it here)."""
    from lerobot.policies import make_policy_config
    kw = {k: (tuple(v) if isinstance(v, list) else v) for k, v in policy.items()
          if k not in ("type", "normalization_mapping")}
    return make_policy_config(policy["type"], **kw)


def anchors(meta, drop_n_last_frames: int) -> list:
    """lerobot-train's sampler indices over a whole dataset (EpisodeAwareSampler, unshuffled): the window anchors."""
    from lerobot.datasets import EpisodeAwareSampler
    s = EpisodeAwareSampler(meta.episodes["dataset_from_index"], meta.episodes["dataset_to_index"],
                            drop_n_last_frames=drop_n_last_frames, shuffle=False)
    return sorted(int(i) for i in s)


def windows(root: str, policy_path: str, out: str) -> None:
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    cfg = policy_config(json.loads(Path(policy_path).read_text()))
    meta = LeRobotDatasetMetadata("local/windows", root=root)
    ds = LeRobotDataset("local/windows", root=root, delta_timestamps=resolve_delta_timestamps(cfg, meta))
    keys = sorted(k for k in ds.meta.features if k.startswith("observation.") or k == "action")
    idx = anchors(ds.meta, cfg.drop_n_last_frames)
    rows = {k: [] for k in ["anchor", "episode", "frame"] + keys + [f"{k}_is_pad" for k in keys]}
    for i in idx:
        item = ds[i]
        rows["anchor"].append(int(item["index"]))
        rows["episode"].append(int(item["episode_index"]))
        rows["frame"].append(int(item["frame_index"]))
        for k in keys:
            rows[k].append(item[k].numpy())
            rows[f"{k}_is_pad"].append(item[f"{k}_is_pad"].numpy())
    np.savez(out, **{k: np.asarray(v) for k, v in rows.items()})


# --- training ---------------------------------------------------------------------------------------------------------
def cli_value(v) -> str:
    return json.dumps(v) if isinstance(v, (list, dict)) else str(v).lower() if isinstance(v, bool) else str(v)


def train_command(unit: dict, out: Path, resume: bool) -> list:
    py = [sys.executable, "-m", "lerobot.scripts.lerobot_train"]
    if resume:
        return py + [f"--config_path={out / 'checkpoints' / 'last' / 'pretrained_model' / 'train_config.json'}",
                     "--resume=true", "--cudnn_deterministic=true"]
    return py + ["--dataset.repo_id=local/train", f"--dataset.root={unit['train_root']}",
                 f"--policy.type={unit['policy_type']}",
                 *(f"--policy.{k}={cli_value(v)}" for k, v in unit["policy"].items()),
                 f"--policy.device={unit['device']}", "--policy.push_to_hub=false",
                 f"--steps={unit['steps']}", f"--batch_size={unit['batch_size']}", f"--save_freq={unit['save_freq']}",
                 f"--log_freq={unit['log_freq']}", f"--num_workers={unit['num_workers']}", f"--seed={unit['seed']}",
                 f"--output_dir={out}", "--wandb.enable=false", "--cudnn_deterministic=true"]


def checkpoints(out: Path) -> list:
    d = out / "checkpoints"
    return sorted(int(p.name) for p in d.iterdir() if p.name.isdigit()) if d.exists() else []


def gpu_memory_mib(pid: int) -> int:
    try:
        r = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True)
    except FileNotFoundError:  # no NVIDIA driver: a CPU unit
        return 0
    if r.returncode:
        return 0
    return sum(int(m) for p, m in (line.split(", ") for line in r.stdout.strip().splitlines() if ", " in line)
               if int(p) == pid)


def rss_mib(pid: int) -> int:
    try:
        for line in Path(f"/proc/{pid}/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) // 1024
    except OSError:
        pass
    return 0


def train(unit_dir: str) -> None:
    unit_dir = Path(unit_dir)
    unit = json.loads((unit_dir / "unit.json").read_text())
    work = unit_dir / "work"
    work.mkdir(exist_ok=True)
    out = work / "train"
    done = checkpoints(out)
    if done and done[-1] == unit["steps"]:
        log(work, f"training complete at step {done[-1]}")
    else:
        resume = bool(done) and RESUME_EXACT[unit["device"]]
        if done and not resume:
            restarts = json.loads((work / "restarts.json").read_text()) if (work / "restarts.json").exists() else []
            write_json(work / "restarts.json", restarts + [done[-1]])
            for f in work.glob("score-*.json"):
                f.unlink()  # scores of checkpoints the restart rewrites
            log(work, f"restarting from step 0 (an exact resume is not available on {unit['device']})")
        if out.exists() and not resume:
            shutil.rmtree(out)  # nothing to resume from, or no exact resume
        state = json.loads((work / "resumes.json").read_text()) if (work / "resumes.json").exists() else []
        if resume:
            state.append(done[-1])
            write_json(work / "resumes.json", state)
        log(work, f"lerobot-train {'resumes from step ' + str(done[-1]) if resume else 'starts'}")
        start = time.time()
        with open(work / "train.log", "a") as f:
            p = subprocess.Popen(train_command(unit, out, resume), stdout=f, stderr=subprocess.STDOUT,
                                 env={**os.environ, **DETERMINISTIC_ENV})
        peak = json.loads((work / "peak.json").read_text()) if (work / "peak.json").exists() else {"gpu_mib": 0,
                                                                                                    "rss_mib": 0}
        last_log, seen = 0.0, set(checkpoints(out))
        while p.poll() is None:
            time.sleep(30)
            peak = {"gpu_mib": max(peak["gpu_mib"], gpu_memory_mib(p.pid)), "rss_mib": max(peak["rss_mib"],
                                                                                           rss_mib(p.pid))}
            tail = (work / "train.log").read_text(errors="replace")[-20000:]
            steps = re.findall(r"\|\s*(\d+)/\d+ \[", tail) or re.findall(r"step:(\S+)", tail)  # tqdm, or log line
            write_json(work / "heartbeat.json", {"time": time.time(), "step": steps[-1] if steps else None,
                                                 "peak": peak})
            new = set(checkpoints(out)) - seen
            if new or time.time() - last_log > 300:
                seen |= new
                log(work, f"step {steps[-1] if steps else 0} checkpoints {sorted(seen)} elapsed "
                          f"{time.time() - start:.0f}s peak gpu {peak['gpu_mib']} MiB rss {peak['rss_mib']} MiB")
                last_log = time.time()
        import resource  # the child's exact peak resident set (Linux reports kilobytes)
        peak["rss_mib"] = max(peak["rss_mib"], resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss // 1024)
        write_json(work / "peak.json", peak)
        if p.returncode:
            log(work, f"lerobot-train exited {p.returncode}")
            sys.exit(f"lerobot-train exited {p.returncode}; see {work / 'train.log'}")
        log(work, f"lerobot-train finished in {time.time() - start:.0f}s")
    history = []
    for step in checkpoints(out):
        ck = out / "checkpoints" / f"{step:06d}" / "pretrained_model"
        s = unit["selection"]
        res = work / f"score-{step:06d}.json"
        if not res.exists():
            score(str(ck), unit["held_out_root"], s["seed"], s["batch_size"], unit["device"], str(res))
        history.append({"step": step, **json.loads(res.read_text()),
                        "model_sha256": sha256_file(ck / "model.safetensors")})
        log(work, f"held-out loss at step {step}: {history[-1]['held_out_loss']}")
    write_json(work / "history.json", {"checkpoints": history,
                                       "resumed_from": json.loads((work / "resumes.json").read_text())
                                       if (work / "resumes.json").exists() else [],
                                       "restarted_from": json.loads((work / "restarts.json").read_text())
                                       if (work / "restarts.json").exists() else [],
                                       "peak": json.loads((work / "peak.json").read_text())
                                       if (work / "peak.json").exists() else None})


def score(checkpoint: str, held_out: str, seed: int, batch: int, device: str, out: str) -> None:
    """The checkpoint's mean held-out loss: its own training loss (LeRobot's DiffusionPolicy.forward) over every
    anchored window of the held-out episodes, in anchor order, with the diffusion noise and timesteps drawn from a
    generator seeded once. Strict FP32 (no autocast, no TF32) with deterministic cuDNN and cuBLAS, so a fresh process
    re-scoring the saved checkpoint gives the same number."""
    import torch
    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.policies import get_policy_class, make_pre_post_processors
    os.environ.update(DETERMINISTIC_ENV)
    torch.backends.cudnn.deterministic, torch.backends.cudnn.benchmark = True, False
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)
    policy = get_policy_class(json.loads((Path(checkpoint) / "config.json").read_text())["type"]).from_pretrained(
        checkpoint)
    policy.config.device = device
    policy.to(device).eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=checkpoint,
                                      preprocessor_overrides={"device_processor": {"device": device}})
    meta = LeRobotDatasetMetadata("local/held_out", root=held_out)
    ds = LeRobotDataset("local/held_out", root=held_out, delta_timestamps=resolve_delta_timestamps(policy.config, meta))
    idx = anchors(ds.meta, policy.config.drop_n_last_frames)
    torch.manual_seed(int(seed))
    tot = 0.0
    with torch.no_grad(), torch.autocast(device_type=torch.device(device).type, enabled=False):
        for i in range(0, len(idx), batch):
            b = torch.utils.data.default_collate([ds[j] for j in idx[i:i + batch]])
            loss, _ = policy.forward(pre(b))
            tot += float(loss.double()) * len(idx[i:i + batch])
    write_json(Path(out), {"held_out_loss": tot / len(idx), "held_out_windows": len(idx)})


def stats(checkpoint: str, out: str) -> None:
    from safetensors.numpy import load_file
    got = {f.name: {k: v.astype(np.float64).reshape(-1).tolist() for k, v in load_file(f).items()}
           for f in sorted(Path(checkpoint).glob("policy_*_normalizer_processor.safetensors"))
           + sorted(Path(checkpoint).glob("policy_*_unnormalizer_processor.safetensors"))}
    write_json(Path(out), got)


def evaluate(argv: list) -> None:
    """lerobot-eval itself, with cuDNN restricted to deterministic algorithms and cuBLAS's deterministic workspace.
    A full-size GPU evaluation still repeats only to the last digits of its rewards (docs/lerobot-learners.md), which
    the evaluation audit's declared tolerance covers."""
    import torch
    from lerobot.scripts import lerobot_eval
    os.environ.update(DETERMINISTIC_ENV)
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True)
    sys.argv = ["lerobot-eval", *argv]
    lerobot_eval.main()


if __name__ == "__main__":
    cmd, args = sys.argv[1], sys.argv[2:]
    if cmd == "split":
        split(*args)
    elif cmd == "windows":
        windows(*args)
    elif cmd == "train":
        train(*args)
    elif cmd == "score":
        score(args[0], args[1], int(args[2]), int(args[3]), args[4], args[5])
    elif cmd == "stats":
        stats(*args)
    elif cmd == "evaluate":
        evaluate(args)
    else:
        sys.exit(f"unknown command {cmd}")
