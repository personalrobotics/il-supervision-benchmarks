#!/usr/bin/env python3
"""The learners moved from CCIL (CCIL#168): ilsb trains, loads, evaluates and resumes exactly as CCIL did.

    python tools/learner_check.py --work NEW_DIR [--out docs/fixtures/evidence/learner-migration.json]

The reference is CCIL's own trainers (ccil.training, at the pinned commit), run on the same host. On a 10-episode
Pendulum demonstration file and CCIL's analytic recipe with noisy-action labels, on CPU:
- policy_train: `ilsb policy train` and `ccil policy train` on the same label stages, for BC, NoiseBC and CCIL with
  trainer v1 and v2, and v2 with checkpoint selection on validation episodes, a cosine schedule and EMA. Each pair
  writes the same checkpoint byte for byte, the same supervision manifest and the same offline metrics. A rerun of
  `ilsb policy train` reuses its policy.
- group_training: the study learner's group (three seeds, validation selection, plateau schedule, patience,
  a synthetic fraction) trains to the same parameters, record and training identity under both trainers, and
  every model stops by patience.
- checkpoint_migration: CCIL trains NoiseBC for 3 epochs and writes a resumable checkpoint at epoch 1 (parameters, scalers, Adam
  state, epoch records, NoiseBC trace chain, training identity). ilsb loads it and predicts the same actions, and
  resumes it to epoch 3; CCIL resumes the same checkpoint. The two epoch-3 checkpoints are identical byte for byte.
  Trainer v1 has no learning-rate schedule, and its randomness is keyed by (seed, epoch, step), so no RNG state is
  carried. ilsb also loads the inference checkpoint `ccil policy train` wrote and reproduces its offline metrics,
  and refuses to resume an inference checkpoint or one from another run, as CCIL does.
- installed: whether ilsb ran from an installed wheel rather than a source tree (CI runs it from a clean venv).
The evidence holds no paths or times. Hashes belong to the host; every check compares ilsb with CCIL on that host.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import yaml

LABELS = ["data.path=demos.npz", "run.output_dir=runs", "compute.device=cpu", "compute.deterministic=true",
          "generation.method=noisy_action", "generation.labels_per_transition=2", "generation.action_noise_std=0.05",
          "generation.support_radius=0.5"]
BASE = {"batch_size": 32, "epochs": 3, "hidden_layers": [16, 16]}
CASES = {
    "bc_v1": {"method": "bc", "trainer_version": "1"},
    "noise_bc_v1": {"method": "noise_bc", "trainer_version": "1", "noise_scale": 0.1},
    "ccil_v1": {"method": "ccil", "trainer_version": "1", "balance_noisy_labels": True, "epochs": 4},
    "bc_v2": {"method": "bc"},
    "noise_bc_v2": {"method": "noise_bc", "noise_scale": 0.1},
    "ccil_v2": {"method": "ccil"},
    "ccil_v2_recipe": {"method": "ccil", "checkpoint_selection": "validation", "validation_interval_steps": 5,
                       "lr_schedule": "cosine", "warmup_steps": 2, "ema_decay": 0.9, "optimizer": "adamw",
                       "weight_decay": 0.01},
}
GROUP = {"hidden_layers": [16, 16], "learning_rate": 1e-3, "batch_size": 32, "epochs": 1,
         "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None, "step_budget": 4000,
         "checkpoint_selection": "validation", "validation_interval_steps": 20, "patience": 3,
         "lr_schedule": "plateau", "lr_factor": 0.5, "lr_patience": 2, "min_lr": 2.5e-4, "synthetic_fraction": 0.5,
         "synthetic_coverage_before_stop": True}
V1 = {"hidden_layers": [16, 16], "learning_rate": 1e-3, "batch_size": 32, "epochs": 3, "observation_scaler": "standard",
      "action_scaler": "min_max", "balance_labels": None}


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def untimed(x):
    """A training record without wall-clock times or the compiler's process-dependent cache record."""
    if isinstance(x, dict):
        return {k: untimed(v) for k, v in x.items() if not k.endswith("seconds") and k != "compilation"}
    return [untimed(v) for v in x] if isinstance(x, list) else x


def digest(x) -> str:
    return hashlib.sha256(json.dumps(x, sort_keys=True, default=str).encode()).hexdigest()


def text(v) -> str:
    """A value as `ccil --set` reads it: strings raw, booleans lower case, numbers and lists as JSON."""
    return v if isinstance(v, str) else json.dumps(v)


def run(cmd, cwd) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", *cmd], cwd=cwd, capture_output=True, text=True)


def one(pattern: str, cwd: Path) -> Path:
    [p] = glob.glob(pattern, root_dir=cwd)
    return cwd / p


def policy_train(work: Path) -> dict:
    out = {}
    for name, policy in CASES.items():
        d = work / name
        d.mkdir()
        (d / "demos.npz").symlink_to(work / "demos.npz")
        p = {**BASE, **policy}
        (d / "config.yaml").write_text(yaml.safe_dump({"labels": {"recipe": "analytic_double_integrator",
                                                                  "set": LABELS}, "policy": p,
                                                  "output_dir": "policies"}))
        sets = [s for kv in LABELS + [f"policy.{k}={text(v)}" for k, v in p.items()] for s in ("--set", kv)]
        theirs = run(["ccil", "policy", "train", "--recipe", "analytic_double_integrator", *sets], d)
        mine = run(["ilsb", "policy", "train", "config.yaml"], d)
        again = run(["ilsb", "policy", "train", "config.yaml"], d)
        if theirs.returncode or mine.returncode:
            out[name] = {"pass": False, "ccil": theirs.stderr[-600:], "ilsb": mine.stderr[-600:]}
            continue
        c_dir, i_dir = one("runs/policy-train-*", d), one("policies/policy-*", d)
        c_eval = json.loads((one("runs/policy-evaluate-*", d) / "metrics.json").read_text())["values"]
        i_eval = json.loads((i_dir / "evaluation.json").read_text())
        same_metrics = all(c_eval[k] == v for k, v in i_eval.items() if k != "note")
        same_sup = json.loads((c_dir / "supervision.json").read_text()) == json.loads(
            (i_dir / "supervision.json").read_text())
        ckpt = sha256(c_dir / "checkpoints" / "policy.npz") == sha256(i_dir / "policy.npz")
        reused = "policy.train       reused" in again.stdout and again.returncode == 0
        out[name] = {"checkpoint_identical": ckpt, "supervision_identical": same_sup, "metrics_identical": same_metrics,
                     "rerun_reused": reused, "checkpoint_sha256": sha256(i_dir / "policy.npz"),
                     "expert_data_error": i_eval["expert_data_error"],
                     "pass": ckpt and same_sup and same_metrics and reused}
    return out


def supervision(work: Path):
    from ccil.core import IdentityRepresentation
    from ccil.supervision import build_supervision
    d = work / "ccil_v2"
    data = one("runs/data-prepare-*", d) / "data"
    cands = one("runs/labels-generate-*", d) / "labels" / "candidates.npz"
    rep = IdentityRepresentation("state")
    train = build_supervision(data / "dataset.npz", "ccil", representation=rep, split=data / "split.json",
                              part="train", candidates=cands)
    val = build_supervision(data / "dataset.npz", "bc", representation=rep, split=data / "split.json",
                            part="validation")
    noisy = build_supervision(data / "dataset.npz", "noise_bc", representation=rep, split=data / "split.json",
                              part="train")
    return train, val, noisy, data / "dataset.npz"


def group_training(work: Path) -> dict:
    from ccil.training import vector_bc_v2 as theirs

    from ilsb.learners.vector import vector_bc_v2 as mine
    train, val, _, _ = supervision(work)
    rows = {}
    for name, v2 in (("ccil", theirs), ("ilsb", mine)):
        res = v2.train_group([v2.Job(train, s, validation=val) for s in (1, 2, 3)], GROUP, device="cpu",
                             deterministic=True)
        rows[name] = [{"parameters_sha256": v2.parameters_sha256(p), "training_identity_sha256":
                       r["training_identity_sha256"], "stopping": r["stopping"],
                       "selected_step": (r["validation"] or {}).get("selected_step"),
                       "record_sha256": digest(untimed(r))}
                      for p, r in res]
    same = rows["ccil"] == rows["ilsb"]
    outcomes = [r["stopping"]["outcome"] for r in rows["ilsb"]]
    return {"jobs": 3, "identical": same, "ilsb": rows["ilsb"], "ccil": None if same else rows["ccil"],
            "outcomes": outcomes, "pass": same and outcomes == ["patience"] * 3}


def checkpoint_migration(work: Path) -> dict:
    from ccil.training import noise_bc as c_noise
    from ccil.training import vector_bc as c_bc

    from ilsb.learners.vector import noise_bc as i_noise
    from ilsb.learners.vector import vector_bc as i_bc
    _, _, noisy, dataset = supervision(work)
    d = work / "migration"
    d.mkdir()

    def noise(mod):
        return mod.NoiseBC("symmetric_uniform_half_batch", 0.1, seed=7, mask_probability=0.5,
                           units="standardized observation units (after the observation scaler)")

    def saver(prefix):
        return lambda policy, opt, epoch, state: policy.save(d / f"{prefix}{epoch}.npz", opt, epoch, state)
    # CCIL trains 3 epochs, saving each; its epoch-1 checkpoint is resumed to epoch 3 by CCIL and by ilsb.
    c_bc.train_vector_bc(noisy, V1, seed=7, deterministic=True, noise=noise(c_noise),
                         on_epoch=saver("ccil-uninterrupted-"))
    first = d / "ccil-uninterrupted-1.npz"
    c_bc.train_vector_bc(noisy, V1, seed=7, deterministic=True, noise=noise(c_noise), resume=first,
                         on_epoch=saver("ccil-resumed-"))
    i_bc.train_vector_bc(noisy, V1, seed=7, deterministic=True, noise=noise(i_noise), resume=first,
                         on_epoch=saver("ilsb-resumed-"))
    obs, _, _ = noisy.arrays()
    c_pol, c_meta, c_adam = c_bc.VectorBCPolicy.load(first)
    i_pol, i_meta, i_adam = i_bc.VectorBCPolicy.load(first)
    predictions = np.array_equal(c_pol.predict(obs), i_pol.predict(obs))
    state = c_meta == i_meta and sorted(c_adam) == sorted(i_adam) and all(np.array_equal(c_adam[k], i_adam[k])
                                                                         for k in c_adam)
    resumed = sha256(d / "ilsb-resumed-3.npz") == sha256(d / "ccil-resumed-3.npz")
    uninterrupted = sha256(d / "ilsb-resumed-3.npz") == sha256(d / "ccil-uninterrupted-3.npz")
    # The inference checkpoint `ccil policy train` wrote: loaded and evaluated here, it reproduces CCIL's metrics.
    from ilsb.policy import evaluate
    case = work / "ccil_v1"
    inference = one("runs/policy-train-*", case) / "checkpoints" / "policy.npz"
    c_eval = json.loads((one("runs/policy-evaluate-*", case) / "metrics.json").read_text())["values"]
    data = one("runs/data-prepare-*", case) / "data"
    pol, _, _ = i_bc.VectorBCPolicy.load(inference)
    i_eval = evaluate(pol, data / "dataset.npz", data / "split.json", "state", False)
    evaluated = all(c_eval[k] == v for k, v in i_eval.items() if k != "note")
    refusals = {}
    for name, ckpt, sup, kw in (("inference_checkpoint", inference, noisy, {"noise": noise(i_noise)}),
                                ("another_run", first, noisy, {"noise": noise(i_noise), "seed_": 8})):
        seed = kw.pop("seed_", 7)
        try:
            i_bc.train_vector_bc(sup, V1, seed=seed, deterministic=True, resume=ckpt, **kw)
            refusals[name] = False
        except i_bc.TrainingError as e:
            refusals[name] = str(e).startswith("cannot resume")
    ok = predictions and state and resumed and uninterrupted and evaluated and all(refusals.values())
    return {"loads_with_identical_state": state, "predictions_identical": predictions,
            "resumed_checkpoint_identical": resumed, "resumed_equals_uninterrupted": uninterrupted,
            "epoch1_sha256": sha256(first), "epoch3_sha256": sha256(d / "ilsb-resumed-3.npz"),
            "inference_checkpoint_metrics_identical": evaluated, "refuses_to_resume": refusals, "pass": ok}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True)
    import ilsb
    from ilsb.tasks.pendulum import Task, demonstrations
    demonstrations(Task(variant="continuous", action_bound=3.0, horizon=100, eval_episodes=4, eval_seed=9001),
                   episodes=10, steps=60, seed=5, out=work / "demos.npz")
    ev = {"demonstrations_sha256": sha256(work / "demos.npz"),
          "installed": "site-packages" in Path(ilsb.__file__).resolve().parts}
    ev["policy_train"] = policy_train(work)
    ev["group_training"] = group_training(work)
    ev["checkpoint_migration"] = checkpoint_migration(work)
    ev["all_pass"] = bool(all(c["pass"] for c in ev["policy_train"].values()) and ev["group_training"]["pass"]
                          and ev["checkpoint_migration"]["pass"])
    text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
