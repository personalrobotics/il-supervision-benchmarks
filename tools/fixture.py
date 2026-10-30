#!/usr/bin/env python3
"""The compact end-to-end fixture of the vertical slice (il-supervision-benchmarks#11).

    uv run python tools/fixture.py --work NEW_DIR [--out docs/fixtures/evidence/vertical-slice-fixture.json]

On a 12-episode demonstration file from the corrected demonstrator (120 steps), a tiny study with every arm
(BC, learned-dynamics CCIL, exact-dynamics CCIL), two real fractions, two synthetic doses and three seeds runs
through the whole state machine, on CPU. Then:
- resume: the same command on the same root reruns no stage and trains nothing;
- reuse: a second study root on the same store trains nothing and reports identically;
- refusals, each on a copy: a wrong dataset hash (preflight), a corrupted rollout (evaluation audit), a mismatched
  store entry (train), and an evaluation before any training (out of order);
- a changed implementation (a copy of the harness source with one added comment) reuses no policy from the store;
- a step ceiling the patience rule never reaches fails the train stage (incomplete fit), and the audit refuses.
The written evidence holds no paths, times or commits, so a rerun on the same host reproduces it byte for byte.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

from ilsb.store import sha256_file
from ilsb.tasks.pendulum import Task, demonstrations

REPO = Path(__file__).resolve().parents[1]
TASK = {"variant": "continuous", "action_bound": 3.0, "horizon": 120, "eval_episodes": 8, "eval_seed": 9001}
SPEC = {
    "study": "vertical-slice-fixture",
    "task": TASK,
    "split": {"validation_fraction": 0.25},
    "real_fractions": [0.5, 1.0],
    "synthetic_doses": [1, 2],
    "arms": ["bc", "ccil_learned", "ccil_exact"],
    "seeds": [1, 2, 3],
    "dynamics": {"hidden_layers": [32, 32], "activation": "relu", "lipschitz_type": "none", "learning_rate": 1e-3,
                 "weight_decay": 0.0, "epochs": 3, "batch_size": 64},
    "generation": {"generator": "noisy_action_preimage", "action_noise_std": 0.05, "support_radius": 0.5,
                   "max_iterations": 30, "tolerance": 1e-8},
    "learner": {"hidden_layers": [32, 32], "learning_rate": 1e-3, "batch_size": 32, "epochs": 1,
                "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None,
                "step_budget": 1500, "checkpoint_selection": "validation", "validation_interval_steps": 20,
                "patience": 3, "synthetic_fraction": 0.5},
    "analysis": {"metric": "upright_occupancy", "confidence": 0.95, "min_pairs": 2,
                 "contrasts": [["ccil_learned", "bc"], ["ccil_exact", "bc"], ["ccil_exact", "ccil_learned"]]},
    "compute": {"device": "cpu"},
}


def ilsb(*args, allow_dirty=False, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-m", "ilsb", *args, *(["--allow-dirty"] if allow_dirty else [])],
                          capture_output=True, text=True, env=env)


def stage(root: Path, name: str) -> dict:
    return json.loads((root / "stages" / f"{name}.json").read_text())


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True)
    dirty = a.allow_dirty
    demonstrations(Task(**TASK), episodes=12, steps=120, seed=5, out=work / "demos.npz")
    spec = {**SPEC, "data": {"path": "demos.npz", "sha256": sha256_file(work / "demos.npz")}}
    (work / "spec.yaml").write_text(yaml.safe_dump(spec, sort_keys=True))
    store, root = work / "store", work / "root"
    ev = {"spec": {k: v for k, v in spec.items() if k != "data"}, "demonstrations_sha256": spec["data"]["sha256"]}

    # 1. The full study.
    p = ilsb("run", str(work / "spec.yaml"), "--root", str(root), "--store", str(store), allow_dirty=dirty)
    ev["first_run"] = {"exit": p.returncode, "stages": json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout else None,
                       "stderr_tail": p.stderr[-400:] if p.returncode else ""}
    if p.returncode:
        print(p.stdout, p.stderr)
        return 1
    tr, rep = stage(root, "train"), stage(root, "report")
    ev["units"] = len(tr["units"])
    ev["trained"] = tr["trained"]
    ev["store_entries"] = {k.name: len(list(k.iterdir())) for k in sorted(store.iterdir()) if k.is_dir()}
    outcomes = {}
    for u in tr["units"].values():
        outcomes[u["stopping"]["outcome"]] = outcomes.get(u["stopping"]["outcome"], 0) + 1
    ev["stopping_outcomes"] = outcomes
    ev["realized_synthetic_row_fractions"] = sorted({u["exposure"]["realized_synthetic_row_fraction"]
                                                     for u in tr["units"].values()})
    ev["audits"] = {s: stage(root, s)["all_pass"] for s in ("audit", "evaluation_audit")}
    ev["report"] = {"cells": {k: v["mean"] for k, v in rep["cells"].items()},
                    "contrasts": {k: {"mean": v["mean"], "interval": v["interval"], "outcome": v["outcome"]}
                                  for k, v in rep["contrasts"].items()}}

    # 2. Resume: the same command reruns no stage and trains nothing.
    p = ilsb("run", str(work / "spec.yaml"), "--root", str(root), "--store", str(store), allow_dirty=dirty)
    ev["resume"] = {"exit": p.returncode, "stages": json.loads(p.stdout.strip().splitlines()[-1])}

    # 3. Reuse across studies: a new root on the same store trains nothing and reports identically.
    root2 = work / "root-reuse"
    p = ilsb("run", str(work / "spec.yaml"), "--root", str(root2), "--store", str(store), allow_dirty=dirty)
    rep2 = stage(root2, "report") if p.returncode == 0 else {}
    ev["reuse"] = {"exit": p.returncode, "trained": stage(root2, "train")["trained"] if p.returncode == 0 else None,
                   "report_identical": {k: rep2.get(k) for k in ("cells", "contrasts")} ==
                   {k: rep.get(k) for k in ("cells", "contrasts")}}

    # 4. Refusals, each on its own copy.
    refusals = {}
    bad = {**spec, "data": {"path": "demos.npz", "sha256": "0" * 64}}
    (work / "spec-wrong-hash.yaml").write_text(yaml.safe_dump(bad, sort_keys=True))
    p = ilsb("run", str(work / "spec-wrong-hash.yaml"), "--root", str(work / "r-hash"), "--store", str(store),
             allow_dirty=dirty)
    refusals["wrong_input_hash"] = {"refused": p.returncode != 0 and "not the declared dataset" in p.stderr + p.stdout}

    root3, store3 = work / "r-corrupt", work / "store-corrupt"
    shutil.copytree(root, root3)
    shutil.copytree(store, store3)
    victim = min((store3 / "rollout").iterdir()) / "rollout.npz"
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    p = ilsb("stage", "evaluation_audit", str(work / "spec.yaml"), "--root", str(root3), "--store", str(store3),
             allow_dirty=dirty)
    ea = stage(root3, "evaluation_audit")
    refusals["corrupted_rollout"] = {"refused": p.returncode != 0 and not ea["all_pass"] and len(ea["problems"]) == 1}
    p = ilsb("stage", "report", str(work / "spec.yaml"), "--root", str(root3), "--store", str(store3),
             allow_dirty=dirty)
    refusals["report_after_failed_audit"] = {"refused": p.returncode != 0 and "evaluation_audit" in p.stderr + p.stdout}

    root4, store4 = work / "r-mismatch", work / "store-mismatch"
    shutil.copytree(store, store4)
    entry = min((store4 / "policy").iterdir())
    ident = json.loads((entry / "identity.json").read_text())
    ident["seed"] = 999
    (entry / "identity.json").write_text(json.dumps(ident, sort_keys=True, separators=(",", ":")) + "\n")
    p = ilsb("run", str(work / "spec.yaml"), "--root", str(root4), "--store", str(store4), allow_dirty=dirty)
    tr4 = stage(root4, "train")
    refusals["mismatched_reuse"] = {"refused": p.returncode != 0 and not tr4["all_pass"]
                                    and "mismatched reuse" in tr4.get("error", "")}

    p = ilsb("stage", "evaluate", str(work / "spec.yaml"), "--root", str(work / "r-order"), "--store", str(store),
             allow_dirty=dirty)
    refusals["out_of_order"] = {"refused": p.returncode != 0 and "needs a completed preflight" in p.stderr + p.stdout}
    ev["refusals"] = refusals

    # 5. A changed implementation: the same study from a copy of the harness whose source differs by one comment
    # trains every unit again on the same store instead of reusing work done by other code.
    impl = work / "impl"
    shutil.copytree(REPO / "src", impl / "src")
    shutil.copy(REPO / "uv.lock", impl / "uv.lock")
    with open(impl / "src" / "ilsb" / "run.py", "a") as f:
        f.write("# implementation changed\n")
    env = {**os.environ, "PYTHONPATH": str(impl / "src")}
    root5 = work / "r-impl"
    for name in ("preflight", "train"):
        p = ilsb("stage", name, str(work / "spec.yaml"), "--root", str(root5), "--store", str(store),
                 allow_dirty=True, env=env)
    tr5 = stage(root5, "train")
    ev["implementation_change"] = {
        "exit": p.returncode, "trained": tr5["trained"], "units": len(tr5["units"]),
        "source_differs": json.loads((root5 / "root.json").read_text())["scope"]["ilsb_source_sha256"]
        != json.loads((root / "root.json").read_text())["scope"]["ilsb_source_sha256"]}

    # 6. A ceiling below any patience stop: every unit is an incomplete fit, the stage fails, and the audit refuses.
    ceiling = {**spec, "learner": {**spec["learner"], "step_budget": 40, "patience": 50}}
    (work / "spec-ceiling.yaml").write_text(yaml.safe_dump(ceiling, sort_keys=True))
    root6 = work / "r-ceiling"
    p = ilsb("run", str(work / "spec-ceiling.yaml"), "--root", str(root6), "--store", str(store), allow_dirty=dirty)
    tr6 = stage(root6, "train")
    p2 = ilsb("stage", "audit", str(work / "spec-ceiling.yaml"), "--root", str(root6), "--store", str(store),
              allow_dirty=dirty)
    ev["incomplete_fit"] = {"run_exit": p.returncode, "train_all_pass": tr6["all_pass"],
                            "incomplete_units": len(tr6["incomplete_fit"]), "units": len(tr6["units"]),
                            "audit_refused": p2.returncode != 0 and "needs a completed train stage" in p2.stderr + p2.stdout}

    ev["all_pass"] = bool(ev["first_run"]["exit"] == 0 and all(ev["audits"].values())
                          and ev["resume"]["exit"] == 0 and set(ev["resume"]["stages"].values()) == {"resumed"}
                          and ev["reuse"]["exit"] == 0 and ev["reuse"]["trained"] == 0 and ev["reuse"]["report_identical"]
                          and all(r["refused"] for r in refusals.values())
                          and ev["implementation_change"]["exit"] == 0 and ev["implementation_change"]["source_differs"]
                          and ev["implementation_change"]["trained"] == ev["implementation_change"]["units"] == ev["units"]
                          and ev["incomplete_fit"]["run_exit"] != 0 and ev["incomplete_fit"]["train_all_pass"] is False
                          and ev["incomplete_fit"]["incomplete_units"] == ev["incomplete_fit"]["units"] == ev["units"]
                          and ev["incomplete_fit"]["audit_refused"])
    text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
