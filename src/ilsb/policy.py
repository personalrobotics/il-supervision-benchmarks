"""ilsb policy train: one vector policy on CCIL supervision, moved here from `ccil policy train` (CCIL#168).

    ilsb policy train CONFIG

    schema: ilsb.policy-train
    version: 1
    labels:                  # the options of `ccil labels generate`
      recipe: analytic_double_integrator     # optional
      config: labels.yaml                    # optional
      set: [data.path=rollouts.npz, ...]     # optional
    supervision:
      source: demonstrations_plus_selection  # or demonstrations
      demonstration_weight: 1.0
      exclude_validation_episodes: true
      balance_noisy_labels: false
    policy:
      trainer: vector_bc_v2                  # or vector_bc_v1
      seed: 7
      device: cpu                            # cpu, cuda, cuda:N or mps
      deterministic: true
      hidden_layers: [256, 256]
      batch_size: 32
      epochs: 4
      noise: null                            # NoiseBC: {scale: 0.1, model: ..., mask_probability: 0.5}
    run:
      output_dir: policies

Every field and its default is in SECTIONS. Every field is checked for its exact type and range before CCIL runs
or a learner is built; a malformed configuration exits 2 and writes nothing.

CCIL's label stages run first, through `ccil labels generate --result`, each reused when complete; their artifacts
are read from its result (ccil.labels-result 1.x). The policy's seed, device and determinism are its own, so labels
generated once train any number of policy seeds, or on another device. NoiseBC is the demonstrations source with
policy.noise set. policy.train builds the supervision (ccil.supervision.build_supervision), trains it with
ilsb.learners.vector, and evaluates the offline metrics `ccil policy train` reported.

Output: <run.output_dir>/policy-<12 hex>/, named by the policy's identity (its training identity, trainer, device,
determinism and evaluation data): policy.npz (byte for byte the checkpoint `ccil policy train` wrote for the same
seed and settings), supervision.json, training.json, evaluation.json, identity.json and manifest.json (the
normalized configuration that wrote it and each file's SHA-256). It is written whole and renamed into place, and a
complete one with the same identity is reused.

Exit: 0 success; 2 an invalid configuration, a failed label stage, or supervision or settings the learner refuses.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

SCHEMA, VERSION = "ilsb.policy-train", 1
LABELS_RESULT = ("ccil.labels-result", "1")
REQUIRED = object()
NOISY = ("noisy_action", "noisy_action_fe")
SOURCES = ("demonstrations", "demonstrations_plus_selection")  # CCIL build_supervision sources
FILES = ("policy.npz", "supervision.json", "training.json", "evaluation.json", "identity.json")


class Invalid(Exception):
    """An invalid configuration, a failed label stage, or refused supervision or settings: exit 2."""


# --- field types --------------------------------------------------------------------------------------------------
def integer(lo=None):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, int):
            return f"an integer, got {v!r}"
        return f"an integer >= {lo}, got {v}" if lo is not None and v < lo else None
    return check


def number(lo=None, lo_open=False, hi=None, hi_open=False):
    def check(v):
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v):
            return f"a finite number, got {v!r}"
        if lo is not None and (v <= lo if lo_open else v < lo) or hi is not None and (v >= hi if hi_open else v > hi):
            return (f"a number in {'(' if lo_open else '['}{lo}, {'inf' if hi is None else hi}"
                    f"{')' if hi_open or hi is None else ']'}, got {v}")
        return None
    return check


def boolean(v):
    return None if isinstance(v, bool) else f"true or false, got {v!r}"


def string(v):
    return None if isinstance(v, str) and v else f"a non-empty string, got {v!r}"


def choice(*values):
    return lambda v: None if isinstance(v, str) and v in values else f"one of {list(values)}, got {v!r}"


def device(v):
    return None if isinstance(v, str) and re.fullmatch(r"cpu|mps|cuda(:\d+)?", v) else \
        f"cpu, cuda, cuda:N or mps, got {v!r}"


def widths(v):
    if not isinstance(v, list) or not v or any(integer(1)(w) for w in v):
        return f"a non-empty list of integers >= 1, got {v!r}"
    return None


def assignments(v):
    if not isinstance(v, list) or any(not isinstance(s, str) or not re.fullmatch(r"[a-z_][\w.]*=.*", s) for s in v):
        return f"a list of KEY=VALUE strings, got {v!r}"
    return None


def nullable(check):
    return lambda v: None if v is None else check(v)


NOISE = {"scale": (number(0, lo_open=True), REQUIRED),
         "model": (choice("symmetric_uniform_half_batch", "legacy_uniform_half_batch"), "symmetric_uniform_half_batch"),
         "mask_probability": (number(0, hi=1), 0.5)}


def noise(v):
    if not isinstance(v, dict):
        return f"null or a mapping of {sorted(NOISE)}, got {v!r}"
    return "; ".join(_section(v, NOISE, "policy.noise.")[1]) or None


SECTIONS = {
    "labels": {"recipe": (nullable(string), None), "config": (nullable(string), None), "set": (assignments, [])},
    "supervision": {"source": (choice(*SOURCES), REQUIRED), "demonstration_weight": (number(0), 1.0),
                    "exclude_validation_episodes": (boolean, True), "balance_noisy_labels": (boolean, False)},
    "policy": {"trainer": (choice("vector_bc_v1", "vector_bc_v2"), REQUIRED), "seed": (integer(0), REQUIRED),
               "device": (device, "cpu"), "deterministic": (boolean, False), "hidden_layers": (widths, [256, 256]),
               "learning_rate": (number(0, lo_open=True), 0.001), "batch_size": (integer(1), REQUIRED),
               "epochs": (integer(1), REQUIRED), "observation_scaler": (choice("standard", "none"), "standard"),
               "action_scaler": (choice("min_max", "none"), "min_max"), "noise": (nullable(noise), None),
               # the trainer v2 recipe; each default is v1's protocol
               "optimizer": (choice("adam", "adamw"), "adam"), "weight_decay": (number(0), 0.0),
               "lr_schedule": (choice("constant", "cosine"), "constant"), "warmup_steps": (integer(0), 0),
               "step_budget": (nullable(integer(1)), None),
               "grad_clip_norm": (nullable(number(0, lo_open=True)), None),
               "ema_decay": (nullable(number(0, lo_open=True, hi=1, hi_open=True)), None),
               "checkpoint_selection": (choice("last", "validation"), "last"),
               "validation_interval_steps": (nullable(integer(1)), None)},
    "run": {"output_dir": (string, "policies")},
}
V1_RECIPE = {"optimizer": "adam", "weight_decay": 0.0, "lr_schedule": "constant", "warmup_steps": 0,
             "step_budget": None, "grad_clip_norm": None, "ema_decay": None, "checkpoint_selection": "last",
             "validation_interval_steps": None}


def _section(given, fields: dict, where: str) -> tuple[dict, list[str]]:
    out, problems = {}, []
    if not isinstance(given, dict):
        return out, [f"{where.rstrip('.') or 'the configuration'} must be a mapping, got {given!r}"]
    problems += [f"{where}{k} is not a field" for k in sorted(set(given) - set(fields), key=str)]
    for k, (check, default) in fields.items():
        if k not in given:
            if default is REQUIRED:
                problems.append(f"{where}{k} is required")
            else:
                out[k] = json.loads(json.dumps(default))
            continue
        problem = check(given[k])
        if problem:
            problems.append(f"{where}{k} must be {problem}")
        out[k] = given[k]
    return out, problems


def normalize(raw) -> dict:
    """The configuration with every default filled in, or Invalid naming every malformed field."""
    if not isinstance(raw, dict):
        raise Invalid(f"the configuration must be a mapping, got {raw!r}")
    problems = [f"{k} is not a field" for k in sorted(set(raw) - {"schema", "version", *SECTIONS}, key=str)]
    if raw.get("schema") != SCHEMA:
        problems.append(f"schema must be {SCHEMA}, got {raw.get('schema')!r}")
    if isinstance(raw.get("version"), bool) or raw.get("version") != VERSION:
        problems.append(f"version must be {VERSION}, got {raw.get('version')!r}")
    cfg = {"schema": SCHEMA, "version": VERSION}
    for name, fields in SECTIONS.items():
        if name not in raw and name != "run":
            problems.append(f"{name} is required")
            continue
        cfg[name], p = _section(raw.get(name, {}), fields, f"{name}.")
        problems += p
    if problems:
        raise Invalid("invalid configuration:\n  " + "\n  ".join(problems))
    s, p = cfg["supervision"], cfg["policy"]
    if p["noise"] is not None:
        p["noise"] = _section(p["noise"], NOISE, "")[0]
    problems = []
    if p["noise"] is not None and s["source"] != "demonstrations":
        problems.append("policy.noise (NoiseBC) perturbs demonstrations: it needs supervision.source demonstrations")
    if s["balance_noisy_labels"] and s["source"] != "demonstrations_plus_selection":
        problems.append("supervision.balance_noisy_labels repeats demonstrations against labels: it needs "
                        "supervision.source demonstrations_plus_selection")
    if p["trainer"] == "vector_bc_v1":
        problems += [f"policy.{k} needs policy.trainer vector_bc_v2" for k, v in V1_RECIPE.items() if p[k] != v]
    if p["warmup_steps"] and p["lr_schedule"] != "cosine":
        problems.append("policy.warmup_steps applies to lr_schedule cosine; a constant schedule has no warm-up")
    if p["weight_decay"] and p["optimizer"] != "adamw":
        problems.append("policy.weight_decay with adam would be L2 in the gradient; it needs optimizer adamw")
    if p["validation_interval_steps"] is not None and p["checkpoint_selection"] != "validation":
        problems.append("policy.validation_interval_steps applies to checkpoint_selection validation")
    if p["checkpoint_selection"] == "validation" and not s["exclude_validation_episodes"]:
        problems.append("policy.checkpoint_selection validation needs held-out validation episodes: "
                        "supervision.exclude_validation_episodes true")
    if problems:
        raise Invalid("invalid configuration:\n  " + "\n  ".join(problems))
    return cfg


def load_config(path: Path) -> dict:
    try:
        raw = yaml.safe_load(Path(path).read_text())
    except (OSError, yaml.YAMLError) as e:
        raise Invalid(f"{path}: {e}") from None
    return normalize(raw)


def labels_problems(cfg: dict, values: dict) -> list[str]:
    """The checks that need the labels' resolved configuration."""
    s, p = cfg["supervision"], cfg["policy"]
    out = []
    if values["representation.kind"] != "identity":
        out.append("the vector policy trains on representation.kind identity")
    if p["checkpoint_selection"] == "validation" and not values["data.validation_fraction"]:
        out.append("policy.checkpoint_selection validation needs held-out episodes: data.validation_fraction > 0")
    if values.get("certificate.profile") is not None and s["source"] != "demonstrations_plus_selection":
        out.append("certificate.profile certifies labels; supervision.source demonstrations trains on none")
    if s["balance_noisy_labels"] and values["generation.method"] not in NOISY:
        out.append("supervision.balance_noisy_labels applies to noisy-action labels")
    return out


def label_stages(labels: dict) -> dict:
    """Run `ccil labels generate --result`; return its result (ccil.labels-result 1.x)."""
    with tempfile.TemporaryDirectory() as tmp:
        result = Path(tmp) / "result.json"
        cmd = [sys.executable, "-m", "ccil", "labels", "generate", "--result", str(result)]
        for flag in ("recipe", "config"):
            if labels[flag] is not None:
                cmd += [f"--{flag}", labels[flag]]
        for s in labels["set"]:
            cmd += ["--set", s]
        r = subprocess.run(cmd, capture_output=True, text=True)
        sys.stdout.write(r.stdout)
        if r.returncode != 0:
            raise Invalid(f"ccil labels generate failed (exit {r.returncode}): {r.stderr.strip()}")
        res = json.loads(result.read_text())
    if (res.get("schema"), str(res.get("version", "")).split(".")[0]) != LABELS_RESULT:
        raise Invalid(f"ccil labels generate wrote {res.get('schema')} {res.get('version')}, not "
                      f"{LABELS_RESULT[0]} {LABELS_RESULT[1]}.x")
    return res


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
        files = json.loads((d / "manifest.json").read_text())["files"]
        return sorted(files) == sorted(FILES) and all(_sha256(d / f) == h for f, h in files.items())
    except (OSError, ValueError, KeyError, TypeError):
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
    cfg = load_config(config_path)  # every field checked before CCIL runs
    from ccil.core import IdentityRepresentation
    from ccil.supervision import build_supervision

    from .learners.vector import vector_bc, vector_bc_v2
    from .learners.vector.noise_bc import NoiseBC
    s, p = cfg["supervision"], cfg["policy"]
    res = label_stages(cfg["labels"])
    values = {k: v["value"] for k, v in res["config"]["values"].items()}
    bad = labels_problems(cfg, values)
    if bad:
        raise Invalid("invalid configuration:\n  " + "\n  ".join(bad))
    art = {k: Path(v) for k, v in res["artifacts"].items()}
    dataset, split = art["dataset"], art.get("split")
    perturbation = "noise_bc" if p["noise"] is not None else None
    inputs = {"split": split} if split is not None and s["exclude_validation_episodes"] else {}
    if s["source"] == "demonstrations_plus_selection":
        inputs |= {k: art[k] for k in ("candidates", "certificates", "selection", "calibration") if k in art}
    rep = IdentityRepresentation(values["representation.modality"])
    try:
        sup = build_supervision(dataset, s["source"], perturbation=perturbation, representation=rep, candidates=inputs.get("candidates"),
                                certificates=inputs.get("certificates"), selection=inputs.get("selection"),
                                calibration=inputs.get("calibration"), split=inputs.get("split"),
                                demonstration_weight=s["demonstration_weight"])
        held = None
        if p["checkpoint_selection"] == "validation":  # the held-out episodes' demonstrations
            held = build_supervision(dataset, "demonstrations", representation=rep, split=split, part="validation",
                                     demonstration_weight=s["demonstration_weight"])
    except ValueError as e:  # SupervisionError, SchemaError
        raise Invalid(str(e)) from None
    method = sup.manifest()["method"]  # bc, noise_bc or ccil, as the training identity has always named it
    nz = None
    if p["noise"] is not None:
        nz = NoiseBC(p["noise"]["model"], p["noise"]["scale"], seed=p["seed"],
                     mask_probability=p["noise"]["mask_probability"],
                     units="standardized observation units (after the observation scaler)")
    settings = {"hidden_layers": p["hidden_layers"], "learning_rate": p["learning_rate"],
                "batch_size": p["batch_size"], "epochs": p["epochs"], "observation_scaler": p["observation_scaler"],
                "action_scaler": p["action_scaler"],
                "balance_labels": values["generation.labels_per_transition"] if s["balance_noisy_labels"] else None,
                **{k: p[k] for k in V1_RECIPE}}
    v2 = p["trainer"] == "vector_bc_v2"
    tid = (vector_bc_v2.training_identity_v2(sup, settings, p["seed"], nz, held) if v2
           else vector_bc.training_identity(sup, settings, p["seed"], nz))["sha256"]
    identity = {"training_identity_sha256": tid, "trainer": p["trainer"], "device": p["device"],
                "deterministic": p["deterministic"],
                "evaluation": {"dataset": _sha256(dataset), "split": None if split is None else _sha256(split)}}
    out_root = Path(cfg["run"]["output_dir"])
    name = "policy-" + hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()[:12]
    final = out_root / name
    if _complete(final, identity):
        print(f"{'policy.train':18s} reused {final}")
        return final
    if final.exists():
        raise Invalid(f"{final} exists but is not a complete policy of this identity; remove it to retrain")
    try:
        if v2:
            [(policy, record)] = vector_bc_v2.train_group([vector_bc_v2.Job(sup, p["seed"], nz, held)], settings,
                                                          device=p["device"], deterministic=p["deterministic"])
        else:
            policy, record = vector_bc.train_vector_bc(sup, settings, seed=p["seed"], device=p["device"],
                                                       deterministic=p["deterministic"], noise=nz)
    except vector_bc.TrainingError as e:  # settings this supervision cannot train under
        raise Invalid(str(e)) from None
    stage_dir = out_root / f".{name}.tmp-{os.getpid()}"
    shutil.rmtree(stage_dir, ignore_errors=True)
    stage_dir.mkdir(parents=True)
    try:
        policy.save(stage_dir / "policy.npz", None, record["epochs_run"],
                    {"supervision": sup.manifest()["inputs"], "seed": p["seed"], "method": method,
                     "trainer_version": record.get("trainer_version", "1"),
                     "execution_sha256": record.get("execution", {}).get("execution_sha256"),
                     "training_identity_sha256": record["training_identity_sha256"]})
        _dump(sup.manifest(), stage_dir / "supervision.json")
        _dump(record, stage_dir / "training.json")
        loaded, _, _ = vector_bc.VectorBCPolicy.load(stage_dir / "policy.npz")
        _dump(evaluate(loaded, dataset, split, values["representation.modality"], "split" not in inputs),
              stage_dir / "evaluation.json")
        _dump(identity, stage_dir / "identity.json")
        _dump({"config": cfg, "files": {f: _sha256(stage_dir / f) for f in FILES}}, stage_dir / "manifest.json")
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
