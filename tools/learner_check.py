#!/usr/bin/env python3
"""The learners moved from CCIL (CCIL#168): ilsb trains, loads and evaluates as CCIL did, and resumes its v1 checkpoints.

    python tools/learner_check.py --work NEW_DIR --reference REFERENCE_PYTHON
        [--out docs/fixtures/evidence/learner-migration.json]

The reference is CCIL's own trainers (`ccil policy train` and ccil.training), which CCIL removed after
REFERENCE_CCIL (CCIL#168). They run on the same host in a second environment, REFERENCE_PYTHON, holding CCIL at that
commit and the same torch as this one:
    uv venv REF && uv pip install --python REF/bin/python torch==<this torch> \
        "ccil[torch] @ git+https://github.com/personalrobotics/CCIL@<REFERENCE_CCIL>"
ilsb runs here, with its pinned CCIL. On a 10-episode Pendulum demonstration file and CCIL's analytic recipe with
noisy-action labels, on CPU:
- policy_train: `ilsb policy train` (ilsb.policy-train 1) and the reference's `ccil policy train`, with the same
  seed, for BC, NoiseBC and CCIL with trainer v1 and v2, and v2 with checkpoint selection on validation episodes, a
  cosine schedule and EMA. Each CCIL generates its own labels, and each names its own source in their provenance.
  So each pair's labels and checkpoint hold the same arrays, their metadata and supervision manifests differ only in
  the label files' hashes and the identities derived from them, and the offline metrics are the same. A rerun of
  `ilsb policy train` reuses its policy, and its manifest holds the normalized configuration.
- labels_once: two more policy seeds on the same labels reuse every label stage and train two distinct policies.
- malformed_configs: fifteen malformed configurations (wrong schema or version, a boolean or float for an integer, a
  string for a number, a zero width, an unknown field, the old source and trainer names, a missing seed, an unknown
  device, a negative or misplaced NoiseBC, a v2 setting on v1, a malformed labels.set) each exit 2 with their
  message, before CCIL runs or anything is written.
- group_training: on ilsb's labels, the study learner's group (three seeds, validation selection, plateau schedule, patience,
  a synthetic fraction) trains to the same parameters, record and training identity under both trainers, and
  every model stops by patience.
- checkpoint_migration: on ilsb's labels, CCIL trains NoiseBC for 3 epochs with trainer v1 and writes a resumable checkpoint at epoch 1
  (parameters, scalers, Adam state, epoch records, NoiseBC trace chain, training identity). ilsb loads it and predicts the same actions, and
  resumes it to epoch 3 (trainer v1; v2 writes inference checkpoints, which load but do not resume); CCIL resumes
  the same checkpoint. The two epoch-3 checkpoints are identical byte for byte.
  Trainer v1 has no learning-rate schedule, and its randomness is keyed by (seed, epoch, step), so no RNG state is
  carried. ilsb also loads the inference checkpoints `ccil policy train` wrote with v1 and v2 and reproduces their
  offline metrics,
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
# The last CCIL commit with its trainers; the reference environment must hold it.
REFERENCE_CCIL = "8903b247033e72437e805d5bba21133544fd8f70"
# The reference's own label stages, apart from ilsb's.
REFERENCE_LABELS = [s if not s.startswith("run.output_dir=") else "run.output_dir=ccil-runs" for s in LABELS]
SEED = 20261023  # the recipe's run.seed, which `ccil policy train` also trained with
BASE = {"batch_size": 32, "epochs": 3, "hidden_layers": [16, 16]}
# Each case in `ccil policy train`'s policy.* keys; ilsb_config gives the same arm in ilsb.policy-train 1.
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


def run(cmd, cwd, python=sys.executable) -> subprocess.CompletedProcess:
    """Run a module; ilsb runs with FutureWarning as an error, so it uses no deprecated CCIL name."""
    flags = ["-W", "error::FutureWarning"] if cmd[0] == "ilsb" else []
    return subprocess.run([python, *flags, "-m", *cmd], cwd=cwd, capture_output=True, text=True)


def differing(a, b, path="") -> list[str]:
    """The dotted paths at which two JSON values differ."""
    if isinstance(a, dict) and isinstance(b, dict):
        return [p for k in sorted(set(a) | set(b)) for p in differing(a.get(k), b.get(k), f"{path}.{k}")]
    if isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        return [p for i, (x, y) in enumerate(zip(a, b)) for p in differing(x, y, f"{path}[{i}]")]
    return [] if a == b else [path]


def same_arrays(a: Path, b: Path) -> bool:
    """Two .npz files hold the same arrays, apart from their JSON metadata."""
    x, y = np.load(a), np.load(b)
    return sorted(x.files) == sorted(y.files) and all(
        np.array_equal(x[k], y[k], equal_nan=x[k].dtype.kind == "f") for k in x.files if k != "meta")


def meta(path: Path) -> dict:
    return json.loads(str(np.load(path)["meta"]))


# What a checkpoint and its supervision manifest may differ in across the two CCILs: the label files' hashes (each
# CCIL's source enters its labels' provenance) and the identities hashed from them.
LABEL_HASHES = {".supervision.candidates", ".supervision.selection", ".training_identity_sha256", ".execution_sha256"}
MANIFEST_HASHES = {".inputs.candidates", ".inputs.selection"}


def one(pattern: str, cwd: Path) -> Path:
    [p] = glob.glob(pattern, root_dir=cwd)
    return cwd / p


def ilsb_config(ccil_policy: dict, **policy) -> dict:
    """The ilsb.policy-train 1 configuration of a `ccil policy train` case, with the same seed and device."""
    c = dict(ccil_policy)
    method, version = c.pop("method"), c.pop("trainer_version", "2")
    sup = {"source": "demonstrations_plus_selection" if method == "ccil" else "demonstrations"}
    if "balance_noisy_labels" in c:
        sup["balance_noisy_labels"] = c.pop("balance_noisy_labels")
    if method == "noise_bc":
        c["noise"] = {"scale": c.pop("noise_scale")}
    return {"schema": "ilsb.policy-train", "version": 1,
            "labels": {"recipe": "analytic_double_integrator", "set": LABELS}, "supervision": sup,
            "policy": {"trainer": f"vector_bc_v{version}", "seed": SEED, "device": "cpu", "deterministic": True,
                       **c, **policy},
            "run": {"output_dir": "policies"}}


def policy_train(work: Path, ref: str) -> dict:
    out = {}
    for name, policy in CASES.items():
        d = work / name
        d.mkdir()
        (d / "demos.npz").symlink_to(work / "demos.npz")
        p = {**BASE, **policy}
        (d / "config.yaml").write_text(yaml.safe_dump(ilsb_config(p)))
        sets = [s for kv in REFERENCE_LABELS + [f"policy.{k}={text(v)}" for k, v in p.items()] for s in ("--set", kv)]
        theirs = run(["ccil", "policy", "train", "--recipe", "analytic_double_integrator", *sets], d, ref)
        mine = run(["ilsb", "policy", "train", "config.yaml"], d)
        again = run(["ilsb", "policy", "train", "config.yaml"], d)
        if theirs.returncode or mine.returncode:
            out[name] = {"pass": False, "ccil": theirs.stderr[-600:], "ilsb": mine.stderr[-600:]}
            continue
        c_dir, i_dir = one("ccil-runs/policy-train-*", d), one("policies/policy-*", d)
        c_eval = json.loads((one("ccil-runs/policy-evaluate-*", d) / "metrics.json").read_text())["values"]
        i_eval = json.loads((i_dir / "evaluation.json").read_text())
        same_metrics = all(c_eval[k] == v for k, v in i_eval.items() if k != "note")
        labels = [(sorted(glob.glob(f"ccil-runs/{s}", root_dir=d)), sorted(glob.glob(f"runs/{s}", root_dir=d)))
                  for s in ("labels-generate-*/labels/*.npz", "labels-curate-*/labels/*.npz")]
        same_labels = all(len(c) == len(i) and all(same_arrays(d / x, d / y) for x, y in zip(c, i)) for c, i in labels)
        sup_differs = differing(json.loads((c_dir / "supervision.json").read_text()),
                                json.loads((i_dir / "supervision.json").read_text()))
        c_ckpt, i_ckpt = c_dir / "checkpoints" / "policy.npz", i_dir / "policy.npz"
        meta_differs = differing(meta(c_ckpt), meta(i_ckpt))
        ckpt = same_arrays(c_ckpt, i_ckpt) and set(meta_differs) <= LABEL_HASHES
        same_sup = set(sup_differs) <= MANIFEST_HASHES
        reused = "policy.train       reused" in again.stdout and again.returncode == 0
        manifest = json.loads((i_dir / "manifest.json").read_text())["config"]
        normalized = manifest["policy"]["learning_rate"] == 0.001 and manifest["schema"] == "ilsb.policy-train"
        out[name] = {"labels_identical": same_labels, "checkpoint_identical_but_label_hashes": ckpt,
                     "supervision_identical_but_label_hashes": same_sup, "metrics_identical": same_metrics,
                     "checkpoint_metadata_differs": meta_differs, "supervision_differs": sup_differs,
                     "rerun_reused": reused, "manifest_holds_normalized_config": normalized,
                     "checkpoint_sha256": sha256(i_ckpt), "expert_data_error": i_eval["expert_data_error"],
                     "pass": same_labels and ckpt and same_sup and same_metrics and reused and normalized}
    return out


def labels_once(work: Path) -> dict:
    """Labels generated once train several policy seeds: every label stage is reused, each seed its own policy."""
    d = work / "ccil_v2"
    runs, policies = {}, {}
    for seed in (1, 2):
        (d / f"seed{seed}.yaml").write_text(yaml.safe_dump(ilsb_config({**BASE, **CASES["ccil_v2"]}, seed=seed)))
        r = run(["ilsb", "policy", "train", f"seed{seed}.yaml"], d)
        lines = [ln.split() for ln in r.stdout.splitlines()]
        runs[seed] = {"exit": r.returncode, "labels_reused": all(ln[1] == "reused" for ln in lines[:-1]),
                      "policy": lines[-1][1] if lines else None}
        policies[seed] = sha256(Path(d, lines[-1][2], "policy.npz")) if r.returncode == 0 else None
    distinct = None not in policies.values() and policies[1] != policies[2]
    ok = all(v["exit"] == 0 and v["labels_reused"] and v["policy"] == "wrote" for v in runs.values()) and distinct
    return {"runs": runs, "distinct_policies": distinct, "pass": ok}


# Each malformed configuration: (what it changes, the message it must give). Every one exits 2 before CCIL runs.
MALFORMED = {
    "no_schema": (lambda c: c.pop("schema"), "schema must be ilsb.policy-train"),
    "future_version": (lambda c: c.update(version=2), "version must be 1"),
    "bool_for_integer": (lambda c: c["policy"].update(batch_size=True), "policy.batch_size must be an integer"),
    "float_for_integer": (lambda c: c["policy"].update(epochs=4.0), "policy.epochs must be an integer"),
    "string_for_number": (lambda c: c["policy"].update(learning_rate="1e-3"), "policy.learning_rate must be a finite"),
    "zero_width": (lambda c: c["policy"].update(hidden_layers=[16, 0]), "policy.hidden_layers must be a non-empty"),
    "unknown_field": (lambda c: c["policy"].update(method="ccil"), "policy.method is not a field"),
    "old_source_name": (lambda c: c["supervision"].update(source="ccil"), "supervision.source must be one of"),
    "version_string_trainer": (lambda c: c["policy"].update(trainer="2"), "policy.trainer must be one of"),
    "missing_seed": (lambda c: c["policy"].pop("seed"), "policy.seed is required"),
    "unknown_device": (lambda c: c["policy"].update(device="gpu"), "policy.device must be cpu, cuda"),
    "negative_noise": (lambda c: c["policy"].update(noise={"scale": -0.1}), "policy.noise.scale must be a number"),
    "noise_on_labels": (lambda c: c["policy"].update(noise={"scale": 0.1}), "policy.noise (NoiseBC) perturbs"),
    "v2_recipe_on_v1": (lambda c: c["policy"].update(trainer="vector_bc_v1", lr_schedule="cosine", warmup_steps=2),
                        "policy.lr_schedule needs policy.trainer vector_bc_v2"),
    "malformed_set": (lambda c: c["labels"].update(set=["data.path"]), "labels.set must be a list of KEY=VALUE"),
}


def malformed(work: Path) -> dict:
    out = {}
    for name, (change, message) in MALFORMED.items():
        d = work / "malformed" / name
        d.mkdir(parents=True)
        (d / "demos.npz").symlink_to(work / "demos.npz")
        cfg = ilsb_config({**BASE, **CASES["ccil_v2"]})
        change(cfg)
        (d / "config.yaml").write_text(yaml.safe_dump(cfg))
        r = run(["ilsb", "policy", "train", "config.yaml"], d)
        wrote = sorted(p.name for p in d.iterdir() if p.name not in ("demos.npz", "config.yaml"))
        out[name] = {"exit": r.returncode, "message": message in r.stderr, "wrote": wrote,
                     "pass": r.returncode == 2 and message in r.stderr and not wrote and "Traceback" not in r.stderr}
    return out


def supervision(work: Path):
    from ccil.core import IdentityRepresentation
    from ccil.supervision import build_supervision
    d = work / "ccil_v2"
    data = one("runs/data-prepare-*", d) / "data"
    cands = one("runs/labels-generate-*", d) / "labels" / "candidates.npz"
    rep = IdentityRepresentation("state")
    train = build_supervision(data / "dataset.npz", "demonstrations_plus_selection", representation=rep, split=data / "split.json",
                              part="train", candidates=cands)
    val = build_supervision(data / "dataset.npz", "demonstrations", representation=rep, split=data / "split.json",
                            part="validation")
    noisy = build_supervision(data / "dataset.npz", "demonstrations", perturbation="noise_bc", representation=rep,
                              split=data / "split.json", part="train")
    return train, val, noisy, data / "dataset.npz"


def group_rows(v2, work: Path) -> list[dict]:
    train, val, _, _ = supervision(work)
    res = v2.train_group([v2.Job(train, s, validation=val) for s in (1, 2, 3)], GROUP, device="cpu", deterministic=True)
    return [{"parameters_sha256": v2.parameters_sha256(p), "training_identity_sha256": r["training_identity_sha256"],
             "stopping": r["stopping"], "selected_step": (r["validation"] or {}).get("selected_step"),
             "record_sha256": digest(untimed(r))}
            for p, r in res]


def reference_group(work: Path) -> list[dict]:
    from ccil.training import vector_bc_v2
    return group_rows(vector_bc_v2, work)


def group_training(work: Path, ref: str) -> dict:
    from ilsb.learners.vector import vector_bc_v2
    rows = {"ccil": reference(ref, "group", work), "ilsb": group_rows(vector_bc_v2, work)}
    same = rows["ccil"] == rows["ilsb"]
    outcomes = [r["stopping"]["outcome"] for r in rows["ilsb"]]
    return {"jobs": 3, "identical": same, "ilsb": rows["ilsb"], "ccil": None if same else rows["ccil"],
            "outcomes": outcomes, "pass": same and outcomes == ["patience"] * 3}


def noise(mod):
    return mod.NoiseBC("symmetric_uniform_half_batch", 0.1, seed=7, mask_probability=0.5,
                       units="standardized observation units (after the observation scaler)")


def saver(d: Path, prefix: str):
    return lambda policy, opt, epoch, state: policy.save(d / f"{prefix}{epoch}.npz", opt, epoch, state)


def reference_migration(work: Path) -> dict:
    """CCIL trains 3 epochs, saving each, and resumes its epoch-1 checkpoint to epoch 3; it loads that checkpoint."""
    from ccil.training import noise_bc, vector_bc
    _, _, noisy, _ = supervision(work)
    d = work / "migration"
    vector_bc.train_vector_bc(noisy, V1, seed=7, deterministic=True, noise=noise(noise_bc),
                              on_epoch=saver(d, "ccil-uninterrupted-"))
    first = d / "ccil-uninterrupted-1.npz"
    vector_bc.train_vector_bc(noisy, V1, seed=7, deterministic=True, noise=noise(noise_bc), resume=first,
                              on_epoch=saver(d, "ccil-resumed-"))
    pol, meta_, adam = vector_bc.VectorBCPolicy.load(first)
    np.savez(d / "ccil-loaded.npz", predictions=pol.predict(noisy.arrays()[0]), **{f"adam/{k}": v for k, v in adam.items()})
    return meta_


def checkpoint_migration(work: Path, ref: str) -> dict:
    from ilsb.learners.vector import noise_bc as i_noise
    from ilsb.learners.vector import vector_bc as i_bc
    _, _, noisy, dataset = supervision(work)
    d = work / "migration"
    d.mkdir()
    # CCIL trains 3 epochs, saving each; its epoch-1 checkpoint is resumed to epoch 3 by CCIL and by ilsb.
    c_meta = reference(ref, "migration", work)
    first = d / "ccil-uninterrupted-1.npz"
    i_bc.train_vector_bc(noisy, V1, seed=7, deterministic=True, noise=noise(i_noise), resume=first,
                         on_epoch=saver(d, "ilsb-resumed-"))
    obs, _, _ = noisy.arrays()
    loaded = np.load(d / "ccil-loaded.npz")
    c_adam = {k.removeprefix("adam/"): loaded[k] for k in loaded.files if k.startswith("adam/")}
    i_pol, i_meta, i_adam = i_bc.VectorBCPolicy.load(first)
    predictions = np.array_equal(loaded["predictions"], i_pol.predict(obs))
    state = json.loads(json.dumps(i_meta, default=str)) == c_meta and sorted(c_adam) == sorted(i_adam) and all(
        np.array_equal(c_adam[k], i_adam[k]) for k in c_adam)
    resumed = sha256(d / "ilsb-resumed-3.npz") == sha256(d / "ccil-resumed-3.npz")
    uninterrupted = sha256(d / "ilsb-resumed-3.npz") == sha256(d / "ccil-uninterrupted-3.npz")
    # The inference checkpoint `ccil policy train` wrote: loaded and evaluated here, it reproduces CCIL's metrics.
    from ilsb.policy import evaluate
    evaluated = {}
    for case in ("ccil_v1", "ccil_v2"):
        cd = work / case
        inference = one("ccil-runs/policy-train-*", cd) / "checkpoints" / "policy.npz"
        c_eval = json.loads((one("ccil-runs/policy-evaluate-*", cd) / "metrics.json").read_text())["values"]
        data = one("ccil-runs/data-prepare-*", cd) / "data"
        pol, _, _ = i_bc.VectorBCPolicy.load(inference)
        i_eval = evaluate(pol, data / "dataset.npz", data / "split.json", "state", False)
        evaluated[case] = all(c_eval[k] == v for k, v in i_eval.items() if k != "note")
    inference = one("ccil-runs/policy-train-*", work / "ccil_v1") / "checkpoints" / "policy.npz"
    refusals = {}
    for name, ckpt, sup, kw in (("inference_checkpoint", inference, noisy, {"noise": noise(i_noise)}),
                                ("another_run", first, noisy, {"noise": noise(i_noise), "seed_": 8})):
        seed = kw.pop("seed_", 7)
        try:
            i_bc.train_vector_bc(sup, V1, seed=seed, deterministic=True, resume=ckpt, **kw)
            refusals[name] = False
        except i_bc.TrainingError as e:
            refusals[name] = str(e).startswith("cannot resume")
    ok = predictions and state and resumed and uninterrupted and all(evaluated.values()) and all(refusals.values())
    return {"loads_with_identical_state": state, "predictions_identical": predictions,
            "resumed_checkpoint_identical": resumed, "resumed_equals_uninterrupted": uninterrupted,
            "epoch1_sha256": sha256(first), "epoch3_sha256": sha256(d / "ilsb-resumed-3.npz"),
            "inference_checkpoint_metrics_identical": evaluated, "refuses_to_resume": refusals, "pass": ok}


REFERENCE = {"group": reference_group, "migration": reference_migration}


def reference(ref: str, half: str, work: Path):
    """Run one reference half in the reference environment; return its JSON result."""
    r = subprocess.run([ref, "-W", "ignore::FutureWarning", __file__, "--reference-half", half, "--work", str(work)],
                       capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(f"the reference's {half} failed: {r.stderr[-2000:]}")
    return json.loads(r.stdout)


PROBE = ("import json, torch; from importlib import metadata; "
         "d = json.loads(metadata.distribution('ccil').read_text('direct_url.json') or '{}'); "
         "print(json.dumps({'ccil': (d.get('vcs_info') or {}).get('commit_id'), 'torch': torch.__version__}))")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--reference", help="the Python of an environment with CCIL at REFERENCE_CCIL")
    ap.add_argument("--reference-half", choices=REFERENCE, help=argparse.SUPPRESS)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    work = a.work.resolve()
    if a.reference_half:  # run by the reference environment's Python, in the work directory main() made
        print(json.dumps(REFERENCE[a.reference_half](work), default=str))
        return 0
    if not a.reference:
        ap.error("--reference is required")
    import torch
    found = json.loads(subprocess.run([a.reference, "-c", PROBE], capture_output=True, text=True, check=True).stdout)
    if found != {"ccil": REFERENCE_CCIL, "torch": torch.__version__}:
        print(f"the reference environment holds {found}; it needs ccil {REFERENCE_CCIL} and torch {torch.__version__}",
              file=sys.stderr)
        return 2
    work.mkdir(parents=True)
    import ilsb
    from ilsb.tasks.pendulum import Task, demonstrations
    demonstrations(Task(variant="continuous", action_bound=3.0, horizon=100, eval_episodes=4, eval_seed=9001),
                   episodes=10, steps=60, seed=5, out=work / "demos.npz")
    ev = {"demonstrations_sha256": sha256(work / "demos.npz"), "reference_ccil": REFERENCE_CCIL,
          "installed": "site-packages" in Path(ilsb.__file__).resolve().parts}
    ev["policy_train"] = policy_train(work, a.reference)
    ev["labels_once"] = labels_once(work)
    ev["malformed_configs"] = malformed(work)
    ev["group_training"] = group_training(work, a.reference)
    ev["checkpoint_migration"] = checkpoint_migration(work, a.reference)
    ev["all_pass"] = bool(all(c["pass"] for c in ev["policy_train"].values()) and ev["group_training"]["pass"]
                          and ev["labels_once"]["pass"] and all(c["pass"] for c in ev["malformed_configs"].values())
                          and ev["checkpoint_migration"]["pass"])
    text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
