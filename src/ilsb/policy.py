"""ilsb policy train: one vector policy on CCIL supervision, moved here from `ccil policy train` (CCIL#168).

    ilsb policy train CONFIG

    labels:            # passed to `ccil labels generate` unchanged
      recipe: NAME     # optional, a CCIL recipe
      config: PATH     # optional, a CCIL configuration file
      set: [KEY=VALUE, ...]
    policy:            # CCIL's former policy.* keys, without the prefix (POLICY_DEFAULTS)
      method: ccil     # ccil, bc or noise_bc
      batch_size: 32
      epochs: 4
    output_dir: DIR    # optional; the labels' run.output_dir by default

CCIL's label stages run first through its public command (data.prepare -> transition.fit -> labels.generate
[-> labels.certify] -> labels.curate ..., each reused when complete). Then policy.train builds the arm's supervision
(ccil.supervision.build_supervision) from their artifacts and trains it with ilsb.learners.vector, and policy.evaluate
reports the same offline metrics `ccil policy train` reported. The seed, device and determinism are the labels
configuration's run.seed, compute.device and compute.deterministic, as they were for `ccil policy train`.

Output: <output_dir>/policy-<12 hex>/, named by the policy's identity (training identity, trainer version, device,
determinism): policy.npz (byte for byte the checkpoint `ccil policy train` wrote), supervision.json, training.json,
evaluation.json, identity.json and manifest.json (each file's SHA-256). It is written whole and renamed into place,
and a complete one with the same identity is reused.

Exit: 0 success; 2 an invalid configuration, a failed label stage, or supervision CCIL refuses.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

NOISY = ("noisy_action", "noisy_action_fe")
CHOICES = {"method": ("ccil", "bc", "noise_bc"), "trainer_version": ("1", "2"),
           "noise_model": ("symmetric_uniform_half_batch", "legacy_uniform_half_batch"),
           "action_scaler": ("min_max", "none"), "observation_scaler": ("standard", "none"),
           "optimizer": ("adam", "adamw"), "lr_schedule": ("constant", "cosine"),
           "checkpoint_selection": ("last", "validation")}
POLICY_DEFAULTS = {"method": "ccil", "trainer_version": "2", "hidden_layers": [256, 256], "learning_rate": 0.001,
                   "batch_size": None, "epochs": None, "observation_scaler": "standard", "action_scaler": "min_max",
                   "balance_noisy_labels": False, "demonstration_weight": 1.0, "exclude_validation_episodes": True,
                   "noise_scale": None, "noise_model": "symmetric_uniform_half_batch", "noise_mask_probability": 0.5,
                   "optimizer": "adam", "weight_decay": 0.0, "lr_schedule": "constant", "warmup_steps": 0,
                   "step_budget": None, "grad_clip_norm": None, "ema_decay": None, "checkpoint_selection": "last",
                   "validation_interval_steps": None}
RECIPE = ("optimizer", "weight_decay", "lr_schedule", "warmup_steps", "step_budget", "grad_clip_norm", "ema_decay",
          "checkpoint_selection", "validation_interval_steps")
V1_RECIPE = {"optimizer": "adam", "weight_decay": 0.0, "lr_schedule": "constant", "warmup_steps": 0,
             "checkpoint_selection": "last"}
FILES = ("policy.npz", "supervision.json", "training.json", "evaluation.json", "identity.json")


class Invalid(Exception):
    """An invalid configuration, a failed label stage, or refused supervision: exit 2."""


def load_config(path: Path) -> dict:
    try:
        cfg = yaml.safe_load(Path(path).read_text()) or {}
    except (OSError, yaml.YAMLError) as e:
        raise Invalid(f"{path}: {e}") from None
    if not isinstance(cfg, dict) or set(cfg) - {"labels", "policy", "output_dir"} or "labels" not in cfg:
        raise Invalid(f"{path}: the configuration has labels, policy and optionally output_dir, got {sorted(cfg)}")
    labels = cfg["labels"] or {}
    if not isinstance(labels, dict) or set(labels) - {"recipe", "config", "set"}:
        raise Invalid("labels takes recipe, config and set (the options of ccil labels generate)")
    if not isinstance(labels.get("set", []), list) or not all(isinstance(s, str) and "=" in s
                                                              for s in labels.get("set", [])):
        raise Invalid("labels.set is a list of KEY=VALUE strings")
    policy = cfg.get("policy") or {}
    if not isinstance(policy, dict) or set(policy) - set(POLICY_DEFAULTS):
        raise Invalid(f"unknown policy keys {sorted(set(policy) - set(POLICY_DEFAULTS))}")
    return {"labels": labels, "policy": {**POLICY_DEFAULTS, **policy}, "output_dir": cfg.get("output_dir")}


def choice_problems(p: dict) -> list[str]:
    return [f"policy.{k} must be one of {list(c)}, got {p[k]!r}" for k, c in CHOICES.items() if p[k] not in c]


def problems(p: dict, labels: dict) -> list[str]:
    """The checks `ccil policy train` made of its policy.* keys, on these keys and the labels' resolved values."""
    out = [f"policy.{k} is required" for k in ("batch_size", "epochs") if p[k] is None]
    if labels["representation.kind"] != "identity":
        out.append("the vector policy trains on representation.kind identity")
    if p["checkpoint_selection"] == "validation" and not (p["exclude_validation_episodes"]
                                                          and labels["data.validation_fraction"]):
        out.append("policy.checkpoint_selection validation needs held-out validation episodes: "
                   "policy.exclude_validation_episodes true and data.validation_fraction > 0")
    if p["warmup_steps"] and p["lr_schedule"] == "constant":
        out.append("policy.warmup_steps applies to lr_schedule cosine; a constant schedule has no warm-up")
    recipe_set = [k for k in RECIPE if p[k] != V1_RECIPE.get(k)]
    if recipe_set and p["trainer_version"] != "2":
        out.append(f"{['policy.' + k for k in recipe_set]} need policy.trainer_version 2")
    if labels.get("certificate.profile") is not None and p["method"] != "ccil":
        out.append(f"certificate.profile certifies labels; policy.method {p['method']} trains on demonstrations only")
    if p["method"] == "noise_bc" and not p["noise_scale"]:
        out.append("policy.method noise_bc requires a positive policy.noise_scale")
    if p["method"] != "noise_bc" and p["noise_scale"] is not None:
        out.append(f"policy.noise_scale applies to policy.method noise_bc, not {p['method']}")
    if p["balance_noisy_labels"] and (p["method"] != "ccil" or labels["generation.method"] not in NOISY):
        out.append("policy.balance_noisy_labels applies to policy.method ccil with noisy-action labels")
    return out


def settings_of(p: dict, labels: dict) -> dict:
    return {"hidden_layers": list(p["hidden_layers"]), "learning_rate": p["learning_rate"],
            "batch_size": p["batch_size"], "epochs": p["epochs"], "observation_scaler": p["observation_scaler"],
            "action_scaler": p["action_scaler"],
            "balance_labels": labels["generation.labels_per_transition"] if p["balance_noisy_labels"] else None,
            **{k: p[k] for k in RECIPE}}


def label_stages(labels: dict) -> dict:
    """Run `ccil labels generate`; return {stage: run directory}."""
    cmd = [sys.executable, "-m", "ccil", "labels", "generate"]
    for flag in ("recipe", "config"):
        if labels.get(flag) is not None:
            cmd += [f"--{flag}", str(labels[flag])]
    for s in labels.get("set", []):
        cmd += ["--set", s]
    r = subprocess.run(cmd, capture_output=True, text=True)
    sys.stdout.write(r.stdout)
    if r.returncode != 0:
        raise Invalid(f"ccil labels generate failed (exit {r.returncode}): {r.stderr.strip()}")
    stages = {}
    for line in r.stdout.splitlines():  # "<stage> wrote|reused <run directory>"
        stage, _, run_dir = line.split(maxsplit=2)
        stages[stage] = Path(run_dir)
    return stages


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _dump(obj, path: Path) -> None:
    path.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str) + "\n")


def _complete(d: Path, identity: dict) -> bool:
    try:
        if json.loads((d / "identity.json").read_text()) != json.loads(json.dumps(identity, sort_keys=True)):
            return False
        manifest = json.loads((d / "manifest.json").read_text())
        return sorted(manifest) == sorted(FILES) and all(_sha256(d / f) == h for f, h in manifest.items())
    except (OSError, ValueError):
        return False


def evaluate(policy, dataset: Path, split: Path | None, modality: str, split_in_training: bool) -> dict:
    """`ccil policy train`'s offline metrics: the policy's squared action error on every demonstration transition."""
    from ccil.core import IdentityRepresentation, transitions
    from ccil.schema import read_trajectories
    batch = transitions(read_trajectories(dataset), IdentityRepresentation(modality))
    sq = ((policy.predict(batch.z) - np.asarray(batch.a, np.float64)) ** 2).sum(axis=1)
    metrics = {"expert_data_error": float(sq.mean()),
               "expert_data_error_definition": "mean over every demonstration transition of sum_j (a - pi(s))_j^2, "
                                               "raw action units (legacy d3rlpy continuous_action_diff_scorer)",
               "transitions": int(len(sq)), "note": "offline metrics only"}
    if split is not None:
        parts = json.loads(split.read_text())
        metrics["validation_episodes_in_training"] = split_in_training
        for part in ("train", "validation"):
            m = np.isin(batch.episode_id, parts[part])
            metrics[f"{part}_action_error"] = float(sq[m].mean()) if m.any() else None
            metrics[f"{part}_transitions"] = int(m.sum())
    return metrics


def train(config_path: Path) -> Path:
    """Run the label stages, then train and evaluate the policy; return its output directory."""
    from ccil.core import IdentityRepresentation
    from ccil.supervision import build_supervision

    from .learners.vector import vector_bc, vector_bc_v2
    from .learners.vector.noise_bc import NoiseBC
    cfg = load_config(config_path)
    p = cfg["policy"]
    bad = choice_problems(p)
    if bad:  # before any label stage runs
        raise Invalid("; ".join(bad))
    stages = label_stages(cfg["labels"])
    labeled = stages["labels.generate"]
    lv = {k: v["value"] for k, v in json.loads((labeled / "config.json").read_text())["values"].items()}
    bad = problems(p, lv)
    if bad:
        raise Invalid("; ".join(bad))
    seed, device, deterministic = lv["run.seed"], lv["compute.device"], lv["compute.deterministic"]
    prepared = stages["data.prepare"]
    dataset, split = prepared / "data" / "dataset.npz", prepared / "data" / "split.json"
    inputs = {"dataset": dataset}
    if split.exists() and p["exclude_validation_episodes"]:
        inputs["split"] = split
    if p["method"] == "ccil":
        inputs["candidates"] = labeled / "labels" / "candidates.npz"
        for stage, key, f in (("labels.certify", "certificates", "certificates.npz"),
                              ("labels.curate", "selection", "selection.npz"),
                              ("labels.calibrate", "calibration", "calibration.json")):
            if stage in stages:
                inputs[key] = stages[stage] / "labels" / f
    rep = IdentityRepresentation(lv["representation.modality"])
    try:
        sup = build_supervision(dataset, p["method"], representation=rep, candidates=inputs.get("candidates"),
                                certificates=inputs.get("certificates"), selection=inputs.get("selection"),
                                calibration=inputs.get("calibration"), split=inputs.get("split"),
                                demonstration_weight=p["demonstration_weight"])
        held = None
        if p["checkpoint_selection"] == "validation":  # the held-out episodes' demonstrations
            held = build_supervision(dataset, "bc", representation=rep, split=split, part="validation",
                                     demonstration_weight=p["demonstration_weight"])
    except ValueError as e:  # SupervisionError, SchemaError
        raise Invalid(str(e)) from None
    noise = None
    if p["method"] == "noise_bc":
        noise = NoiseBC(p["noise_model"], p["noise_scale"], seed=seed, mask_probability=p["noise_mask_probability"],
                        units="standardized observation units (after the observation scaler)")
    settings = settings_of(p, lv)
    if p["trainer_version"] == "2":
        tid = vector_bc_v2.training_identity_v2(sup, settings, seed, noise, held)["sha256"]
    else:
        tid = vector_bc.training_identity(sup, settings, seed, noise)["sha256"]
    identity = {"training_identity_sha256": tid, "trainer_version": p["trainer_version"], "device": device,
                "deterministic": deterministic, "evaluation": {"dataset": _sha256(dataset),
                                                               "split": _sha256(split) if split.exists() else None}}
    out_root = Path(cfg["output_dir"] or lv["run.output_dir"])
    name = "policy-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    final = out_root / name
    if _complete(final, identity):
        print(f"{'policy.train':18s} reused {final}")
        return final
    if final.exists():
        raise Invalid(f"{final} exists but is not a complete policy of this identity; remove it to retrain")
    try:
        if p["trainer_version"] == "2":
            [(policy, record)] = vector_bc_v2.train_group([vector_bc_v2.Job(sup, seed, noise, held)], settings,
                                                          device=device, deterministic=deterministic)
        else:
            policy, record = vector_bc.train_vector_bc(sup, settings, seed=seed, device=device,
                                                       deterministic=deterministic, noise=noise)
    except vector_bc.TrainingError as e:  # settings the supervision cannot train under
        raise Invalid(str(e)) from None
    stage_dir = out_root / f".{name}.tmp-{os.getpid()}"
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)
    try:
        policy.save(stage_dir / "policy.npz", None, record["epochs_run"],
                    {"supervision": sup.manifest()["inputs"], "seed": seed, "method": p["method"],
                     "trainer_version": record.get("trainer_version", "1"),
                     "execution_sha256": record.get("execution", {}).get("execution_sha256"),
                     "training_identity_sha256": record["training_identity_sha256"]})
        _dump(sup.manifest(), stage_dir / "supervision.json")
        _dump(record, stage_dir / "training.json")
        loaded, _, _ = vector_bc.VectorBCPolicy.load(stage_dir / "policy.npz")
        _dump(evaluate(loaded, dataset, split if split.exists() else None, lv["representation.modality"],
                       "split" not in inputs), stage_dir / "evaluation.json")
        _dump(identity, stage_dir / "identity.json")
        _dump({f: _sha256(stage_dir / f) for f in FILES}, stage_dir / "manifest.json")
        os.replace(stage_dir, final)
    finally:
        shutil.rmtree(stage_dir, ignore_errors=True)
    print(f"{'policy.train':18s} wrote  {final}")
    return final


def main(config_path: Path) -> int:
    try:
        train(config_path)
    except Invalid as e:
        print(str(e), file=sys.stderr)
        return 2
    return 0
