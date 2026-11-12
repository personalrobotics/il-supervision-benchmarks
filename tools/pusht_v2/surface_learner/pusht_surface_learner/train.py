"""One unit's training: lerobot-train's step on fixed-fraction minibatches, checkpointed, resumable and extended.

UNIT_DIR/unit.json declares the unit (the harness writes it):
  seed, alpha, batch_size          the learner seed, the synthetic fraction and B
  horizon, boundaries, cadence     the cosine schedule's length (the same for every boundary, so extending continues
                                   the run exactly), the step boundaries [20k, 40k, 80k] and the checkpoint cadence
  policy                           IBCConfig arguments
  features_root                    a LeRobot dataset with the history features (its metadata only: feature shapes)
  rows, stats                      rows.npz (real_*, syn_*, val_* of state, env, action) and the frozen min/max
  selection                        {negatives, seed, batch}: held-out InfoNCE on the val_* rows
  order                            null, or "lerobot" (the equivalence check only: real rows in lerobot-train's
                                   EpisodeAwareSampler order over features_root, batches as its DataLoader forms them)
  stop_at                          optional: stop after this step whatever the rule (the equivalence check and smoke)

Work goes to UNIT_DIR/work: checkpoints/<step>/{pretrained_model, training_state} at every cadence step (the training
state of all but the newest is pruned), history.json (each checkpoint's held-out InfoNCE, the exposure of each segment
between checkpoints with its index digest, timings), heartbeat.json and progress.log. A rerun resumes from the newest
checkpoint. At each boundary the checkpoint with the least held-out InfoNCE is selected (ties: the earlier step); the
unit extends to the next boundary only when the boundary's own checkpoint is selected, and selecting the last boundary
is an incomplete fit. UNIT_DIR/result.json and UNIT_DIR/selected/pretrained_model are written last, atomically.
"""
import json
import os
import resource
import shutil
import sys
import time
from pathlib import Path

import numpy as np
import torch

from .sampler import Stream, per_batch

FEATURES = ("observation.state", "observation.environment_state", "action")
PARTS = ("state", "env", "action")


def write_json(path: Path, obj) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True) + "\n")
    os.replace(tmp, path)


def log(work: Path, msg: str) -> None:
    with open(work / "progress.log", "a") as f:
        f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
        f.flush()


def resolved(unit_dir: Path) -> dict:
    """unit.json with its file paths resolved against the unit directory."""
    unit = json.loads((unit_dir / "unit.json").read_text())
    for k in ("rows", "stats", "features_root"):
        if unit.get(k) is not None:
            unit[k] = str((unit_dir / unit[k]).resolve())
    return unit


def stats_of(path: Path) -> dict:
    s = json.loads(path.read_text())
    return {k: {m: np.asarray(v, np.float32) for m, v in s[k].items()} for k in FEATURES}


def build(unit: dict, work: Path):
    """Everything lerobot-train builds before its loop, in its order (lerobot_train.train)."""
    from accelerate import Accelerator
    from accelerate.utils import DistributedDataParallelKwargs
    from lerobot.configs.default import DatasetConfig, WandBConfig
    from lerobot.configs.train import TrainPipelineConfig
    from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
    from lerobot.optim.factory import make_optimizer_and_scheduler
    from lerobot.policies import make_policy, make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins
    from lerobot.utils.random_utils import set_seed
    register_third_party_plugins()
    from lerobot_policy_ibc.configuration_ibc import IBCConfig

    argv, sys.argv = sys.argv, sys.argv[:1]  # TrainPipelineConfig.validate reads the command line
    try:
        cfg = TrainPipelineConfig(
            dataset=DatasetConfig(repo_id="local/pusht-history", root=str(unit["features_root"])),
            policy=IBCConfig(**unit["policy"], scheduler_decay_steps=unit["horizon"], device="cuda",
                             push_to_hub=False),
            output_dir=work / "never-written", steps=unit["horizon"], batch_size=unit["batch_size"],
            seed=unit["seed"], num_workers=0, save_freq=unit["cadence"], log_freq=unit["log_every"],
            wandb=WandBConfig(enable=False))
        cfg.validate()
    finally:
        sys.argv = argv
    accelerator = Accelerator(step_scheduler_with_optimizer=False,
                              mixed_precision={"bfloat16": "bf16", "float32": "no"}[cfg.policy.dtype],
                              kwargs_handlers=[DistributedDataParallelKwargs(find_unused_parameters=True)])
    set_seed(cfg.seed, accelerator=accelerator)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    meta = LeRobotDatasetMetadata("local/pusht-history", root=unit["features_root"])
    policy = make_policy(cfg=cfg.policy, ds_meta=meta, rename_map=cfg.rename_map)
    stats = meta.stats if unit.get("order") == "lerobot" else stats_of(Path(unit["stats"]))
    preprocessor, postprocessor = make_pre_post_processors(policy_cfg=cfg.policy, pretrained_path=None,
                                                           dataset_stats=stats)
    optimizer, scheduler = make_optimizer_and_scheduler(cfg, policy)
    return cfg, accelerator, meta, policy, preprocessor, postprocessor, optimizer, scheduler


def lerobot_batches(features_root: str, seed: int, batch: int, steps: int) -> list:
    """lerobot-train's real-row batches over a dataset: EpisodeAwareSampler epochs, chunked with drop_last False."""
    from lerobot.datasets import EpisodeAwareSampler
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
    ds = LeRobotDataset("local/pusht-history", root=features_root)
    sampler = EpisodeAwareSampler(ds.meta.episodes["dataset_from_index"], ds.meta.episodes["dataset_to_index"],
                                  episode_indices_to_use=ds.episodes, drop_n_last_frames=0, shuffle=True, seed=seed,
                                  absolute_to_relative_idx=ds.absolute_to_relative_idx)
    out = []
    while len(out) < steps:
        idx = list(iter(sampler))
        out += [np.asarray(idx[i:i + batch], np.int64) for i in range(0, len(idx), batch)]
    return out[:steps]


def infonce(policy, preprocessor, rows: dict, negatives: torch.Tensor, batch: int) -> float:
    """Mean held-out InfoNCE in eval mode, strict FP32 (no autocast, no TF32: training enables TF32 matmuls, so a
    score taken mid-training would otherwise differ from the same checkpoint's score in a fresh process), fixed
    negatives; draws no random numbers."""
    was, tf32 = policy.training, torch.backends.cuda.matmul.allow_tf32
    policy.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    tot, n = 0.0, 0
    with torch.no_grad(), torch.autocast("cuda", enabled=False):
        for i in range(0, len(rows["action"]), batch):
            b = preprocessor({k: rows[p][i:i + batch] for k, p in zip(FEATURES, PARTS)})
            obs = policy.observation(b)
            act = b["action"][:, 0] if b["action"].dim() == 3 else b["action"]
            tot += float(policy.infonce(obs, act, negatives[i:i + batch]).double().sum())
            n += len(obs)
    policy.train(was)
    torch.backends.cuda.matmul.allow_tf32 = tf32
    return tot / n


def to_device(arrs: dict, prefix: str) -> dict:
    out = {p: torch.from_numpy(np.ascontiguousarray(arrs[f"{prefix}_{p}"], np.float32)).cuda() for p in PARTS}
    out["action"] = out["action"][:, None, :]  # action_delta_indices [0]: (N, 1, 2)
    return out


def checkpoints(work: Path) -> list[int]:
    d = work / "checkpoints"
    return sorted(int(p.name) for p in d.iterdir() if (p / "pretrained_model" / "model.safetensors").exists()
                  and (p / "done").exists()) if d.exists() else []


def selected(history: dict, upto: int) -> int:
    c = [(v["infonce"], int(s)) for s, v in history["checkpoints"].items() if int(s) <= upto]
    return min(c)[1]


def train(unit_dir: Path) -> int:
    from lerobot.common.train_utils import load_training_state, save_checkpoint
    from lerobot.scripts.lerobot_train import update_policy
    from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
    from safetensors.torch import load_file

    unit_dir = Path(unit_dir).resolve()
    unit = resolved(unit_dir)
    if (unit_dir / "result.json").exists():
        print(json.dumps({"unit": str(unit_dir), "result": "complete already"}))
        return 0
    work = unit_dir / "work"
    (work / "checkpoints").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    cfg, acc, meta, policy, pre, post, optimizer, scheduler = build(unit, work)
    hist_path = work / "history.json"
    done = checkpoints(work)
    history = json.loads(hist_path.read_text()) if hist_path.exists() else {"checkpoints": {}, "segments": {}}
    history["checkpoints"] = {s: v for s, v in history["checkpoints"].items() if int(s) in done}
    history["segments"] = {s: v for s, v in history["segments"].items() if int(s) in done}
    step = 0
    if done:  # resume: the newest complete checkpoint's weights, optimizer, scheduler and RNG states
        ck = work / "checkpoints" / f"{done[-1]:06d}"
        policy.load_state_dict(load_file(str(ck / "pretrained_model" / "model.safetensors")))
        step, optimizer, scheduler = load_training_state(ck, optimizer, scheduler)
        log(work, f"resumed at step {step} from {ck.name}")
        history.setdefault("resumes", []).append(step)
    policy, optimizer, scheduler = acc.prepare(policy, optimizer, scheduler)
    inner = acc.unwrap_model(policy)

    arrs = np.load(unit["rows"])
    real, syn, val = to_device(arrs, "real"), to_device(arrs, "syn"), to_device(arrs, "val")
    n_real, n_syn = len(real["action"]), len(syn["action"])
    b_real, b_syn = per_batch(unit["alpha"], unit["batch_size"])
    if (b_syn > 0) != (n_syn > 0):
        raise SystemExit(f"alpha {unit['alpha']} and {n_syn} synthetic rows disagree")
    rs, ss = Stream(n_real, b_real, unit["seed"], 0), Stream(n_syn, b_syn, unit["seed"], 1)
    lerobot_order = None
    if unit.get("order") == "lerobot":
        if b_syn:
            raise SystemExit("the lerobot order is for the equivalence check, with real rows only")
        lerobot_order = lerobot_batches(unit["features_root"], unit["seed"], unit["batch_size"], unit["stop_at"])
    sel = unit["selection"]
    g = torch.Generator().manual_seed(sel["seed"])
    negatives = (torch.rand(len(val["action"]), sel["negatives"], inner.d_act, generator=g) * 2 - 1).cuda()

    metrics = {"loss": AverageMeter("loss", ":.3f"), "grad_norm": AverageMeter("grdn", ":.3f"),
               "lr": AverageMeter("lr", ":0.1e"), "update_s": AverageMeter("updt_s", ":.3f"),
               "dataloading_s": AverageMeter("data_s", ":.3f"), "gpu_mem_gb": AverageMeter("mem_gb", ":.2f")}
    tracker = MetricsTracker(cfg.batch_size, n_real + n_syn, 1, metrics, initial_step=step, accelerator=acc)
    boundaries, cadence = unit["boundaries"], unit["cadence"]
    stop_at = unit.get("stop_at")
    target = next(b for b in boundaries if b > step) if step < boundaries[-1] else boundaries[-1]
    if done and step in boundaries:  # resumed exactly at a boundary: apply its rule before going on
        s_ = selected(history, step)
        target = boundaries[boundaries.index(step) + 1] if s_ == step and step != boundaries[-1] else step
    if stop_at:
        target = min(target, stop_at)
    seg_hash, seg_start, losses = None, step, []
    import hashlib
    timing = history.setdefault("timing", {"segments": []})
    t_seg, first_step_s, peak_gb = time.time(), None, 0.0
    while step < target:
        if seg_hash is None:
            seg_hash, seg_start, losses, t_seg = hashlib.sha256(), step, [], time.time()
        if lerobot_order is not None:
            ri, si = lerobot_order[step], np.zeros(0, np.int64)
        else:
            ri, si = rs.at(step), ss.at(step)
        seg_hash.update(ri.tobytes())
        seg_hash.update(si.tobytes())
        rt, st_ = torch.from_numpy(ri).cuda(), torch.from_numpy(si).cuda()
        batch = {k: torch.cat([real[p][rt], syn[p][st_]]) for k, p in zip(FEATURES, PARTS)}
        batch = pre(batch)
        tracker, out = update_policy(tracker, policy, batch, optimizer, cfg.optimizer.grad_clip_norm, accelerator=acc,
                                     lr_scheduler=scheduler)
        step += 1
        losses.append(float(tracker.loss.val))
        peak_gb = max(peak_gb, float(tracker.gpu_mem_gb.val))  # update_policy resets the peak at every step
        if first_step_s is None:
            first_step_s = time.time() - t0
        if step % 200 == 0:
            write_json(work / "heartbeat.json", {"step": step, "time": time.time(), "pid": os.getpid()})
        if step % unit["log_every"] == 0:
            log(work, f"step {step} loss {np.mean(losses[-unit['log_every']:]):.4f} lr {tracker.lr.val:.3e} "
                      f"{(time.time() - t_seg) / (step - seg_start):.4f} s/step")
        if step % cadence == 0 or step == target:
            ck = work / "checkpoints" / f"{step:06d}"
            if ck.exists():
                shutil.rmtree(ck)
            save_checkpoint(ck, step, cfg, inner, optimizer, scheduler, pre, post, num_processes=1,
                            batch_size=cfg.batch_size)
            score = infonce(inner, pre, val, negatives, sel["batch"])
            seg_s = time.time() - t_seg
            history["checkpoints"][str(step)] = {"infonce": score, "train_loss_mean": float(np.mean(losses))}
            history["segments"][str(step)] = {"start": seg_start, "stop": step, "real_rows": (step - seg_start) * b_real
                                              if lerobot_order is None else int(sum(len(lerobot_order[t])
                                                                                    for t in range(seg_start, step))),
                                              "synthetic_rows": (step - seg_start) * b_syn,
                                              "index_sha256": seg_hash.hexdigest()}
            timing["segments"].append({"stop": step, "seconds": seg_s, "s_per_step": seg_s / (step - seg_start),
                                       "peak_gpu_allocated_gb": peak_gb,
                                       "gpu_reserved_gb": torch.cuda.memory_reserved() / 2**30,
                                       "peak_host_rss_gb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20})
            if first_step_s is not None and "first_step_s" not in timing:
                timing["first_step_s"] = first_step_s
            write_json(hist_path, history)
            (ck / "done").write_text("")
            for old in checkpoints(work)[:-1]:  # keep weights, prune older training states
                shutil.rmtree(work / "checkpoints" / f"{old:06d}" / "training_state", ignore_errors=True)
            log(work, f"checkpoint {step}: held-out InfoNCE {score:.5f}")
            seg_hash = None
            if step == target and not stop_at and step in boundaries:
                s_ = selected(history, step)
                if s_ == step and step != boundaries[-1]:
                    target = boundaries[boundaries.index(step) + 1]
                    log(work, f"boundary {step}: selected {s_}, the boundary itself; extending to {target}")
                else:
                    log(work, f"boundary {step}: selected {s_}")
    reached = step
    s_ = selected(history, reached)
    outcome = "stopped_at" if stop_at else ("incomplete_fit" if s_ == boundaries[-1] else "complete")
    result = {"unit": unit, "reached": reached, "selected_step": s_,
              "selected_infonce": history["checkpoints"][str(s_)]["infonce"], "outcome": outcome,
              "per_batch": {"real": b_real, "synthetic": b_syn}, "rows": {"real": n_real, "synthetic": n_syn},
              "history": history,
              "exposure": {"steps": reached, "real_rows": sum(v["real_rows"] for v in history["segments"].values()),
                           "synthetic_rows": sum(v["synthetic_rows"] for v in history["segments"].values())},
              "environment": {"torch": torch.__version__, "gpu": torch.cuda.get_device_name(),
                              "cuda": torch.version.cuda}}
    sel_dir = unit_dir / "selected.tmp"
    if sel_dir.exists():
        shutil.rmtree(sel_dir)
    shutil.copytree(work / "checkpoints" / f"{s_:06d}" / "pretrained_model", sel_dir / "pretrained_model")
    if (unit_dir / "selected").exists():
        shutil.rmtree(unit_dir / "selected")
    os.replace(sel_dir, unit_dir / "selected")
    write_json(unit_dir / "result.json", result)
    log(work, f"done: reached {reached}, selected {s_}, {outcome}")
    print(json.dumps({"unit": str(unit_dir), "reached": reached, "selected_step": s_, "outcome": outcome}))
    return 0


def rescore(unit_dir: Path, checkpoint: Path, out: Path) -> int:
    """A saved checkpoint's held-out InfoNCE, loaded from disk with its own processors (the audit's check)."""
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    unit = resolved(Path(unit_dir))
    cfg = PreTrainedConfig.from_pretrained(checkpoint)
    policy = get_policy_class(cfg.type).from_pretrained(checkpoint).cuda().eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=str(checkpoint))
    arrs = np.load(unit["rows"])
    val = to_device(arrs, "val")
    sel = unit["selection"]
    g = torch.Generator().manual_seed(sel["seed"])
    negatives = (torch.rand(len(val["action"]), sel["negatives"], policy.d_act, generator=g) * 2 - 1).cuda()
    write_json(Path(out), {"infonce": infonce(policy, pre, val, negatives, sel["batch"]),
                           "frames": int(len(val["action"]))})
    return 0
