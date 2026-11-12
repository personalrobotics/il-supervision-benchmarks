#!/usr/bin/env python3
"""The compact end-to-end fixture of the vertical slice (il-supervision-benchmarks#11, #34).

    python tools/fixture.py --work NEW_DIR [--out docs/fixtures/evidence/vertical-slice-fixture.json]

It runs from whatever ilsb is importable; CI runs it from the built wheel in a clean environment. On a 12-episode
demonstration file from the corrected demonstrator (120 steps, CPU), the valid artifacts are built once, by one real
study through every stage: BC, learned-dynamics CCIL and exact-dynamics CCIL, one real fraction, one synthetic dose,
one accepted sigma, two seeds (the fewest that give the paired analysis its intervals), scored on the continuous
normalized height. Every other check reuses them or copies of them, and trains only where its failure needs new
numerical work. Each protected failure, and its check:

- a numerical path breaks (dynamics fit, generation, training, evaluation, audits, analysis, report): the study.
  Every stage passes, every arm trains and stops by patience, the train stage's analysis units are the ones
  preflight reported, and every contrast has its BCa and t intervals.
- the design plans the wrong units, seed roles or nesting: the planner. Preflight alone on 2 fractions x 3 doses x
  3 seeds plans 34 units, trains nothing, and matches the expected cells, seed roles, one K_max candidate set per
  (arm, r, seed) and nested subsets.
- CCIL units do not train on the demonstrations plus their synthetic rows: exposure, from the saved records of a
  learned unit and the BC unit on the same data.
- a resume recomputes: the same command reruns no stage.
- a store is not shared exactly: a second root on the store trains nothing and reports identically.
- a wrong input, a corrupted rollout, a report after a failed audit, a mismatched store entry or an out-of-order
  stage is accepted: refusals, each on a copy, none training.
- changed code reuses stored work: the BC cell reuses under the same code and retrains under a copy of the installed
  harness whose source differs by one comment.
- a fit that hits its ceiling counts as a result: the BC cell under a ceiling below any patience stop fails the train
  stage, and the audit refuses.
- an infeasible generation trains, or a coordinate without a feasible sigma passes: a sigma grid with one feasible
  and one infeasible sigma (failure fractions far on either side of max_failure) trains only the feasible one, and
  the infeasible sigma alone leaves every coordinate unselectable and fails the train stage.

The written evidence holds no paths, times or commits, so a rerun on the same host reproduces it byte for byte. It
counts every policy the fixture trained (each check's train stage reports its own).
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

import ilsb
from ilsb.store import sha256_file
from ilsb.tasks.pendulum import Task, demonstrations

TASK = {"variant": "continuous", "action_bound": 3.0, "horizon": 120, "eval_episodes": 8, "eval_seed": 9001}
SIGMA, SIGMA_INFEASIBLE = 0.005, 1.0
LEARNER = {"hidden_layers": [32, 32], "learning_rate": 1e-3, "batch_size": 32, "epochs": 1,
           "observation_scaler": "standard", "action_scaler": "min_max", "balance_labels": None,
           "step_budget": 4000, "checkpoint_selection": "validation", "validation_interval_steps": 20,
           "patience": 3, "lr_schedule": "plateau", "lr_factor": 0.5, "lr_patience": 2, "min_lr": 2.5e-4,
           "synthetic_fraction": 0.5, "synthetic_coverage_before_stop": True}
SPEC = {
    "study": "vertical-slice-fixture",
    "task": TASK,
    "split": {"validation_fraction": 0.25},
    "real_fractions": [1.0],
    "synthetic_doses": [2],
    "arms": ["bc", "ccil_learned", "ccil_exact"],
    "seeds": [1, 2],
    "dynamics": {"hidden_layers": [32, 32], "activation": "relu", "lipschitz_type": "none", "learning_rate": 1e-3,
                 "weight_decay": 0.0, "epochs": 100, "batch_size": 64},
    "generation": {"generator": "noisy_action_preimage", "max_failure": 0.2, "sigma": {"r1.0/k2": SIGMA},
                   "support_radius": 0.03, "max_iterations": 30, "tolerance": 1e-8},
    "learner": LEARNER,
    "analysis": {"metric": "normalized_height", "confidence": 0.95, "min_pairs": 2,
                 "contrasts": [["ccil_learned", "bc"], ["ccil_exact", "ccil_learned"]],
                 "derived": [["ccil_exact", "bc"]]},
    "compute": {"device": "cpu"},
}
# The planner's matrix and the seeds each cell must plan: every seed at K = 0 and K_max, the surface seeds between.
PLANNER = {"real_fractions": [0.5, 1.0], "synthetic_doses": [1, 2, 4], "seeds": [1, 2, 3], "surface_seeds": [1, 2]}
PLANNER_SEEDS = {0: [1, 2, 3], 1: [1, 2], 2: [1, 2], 4: [1, 2, 3]}


def ilsb_cmd(*args, allow_dirty=False, env=None) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, "-W", "error::FutureWarning", "-m", "ilsb", *args,
                           *(["--allow-dirty"] if allow_dirty else [])], capture_output=True, text=True, env=env)


def stage(root: Path, name: str) -> dict:
    return json.loads((root / "stages" / f"{name}.json").read_text())


def write_spec(path: Path, spec: dict) -> Path:
    path.write_text(yaml.safe_dump(spec, sort_keys=True))
    return path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True)
    dirty = a.allow_dirty

    def run(spec_path, root, store, *, name=None, env=None, allow=None):
        cmd = ("stage", name) if name else ("run",)
        return ilsb_cmd(*cmd, str(spec_path), "--root", str(root), "--store", str(store),
                        allow_dirty=dirty if allow is None else allow, env=env)

    demonstrations(Task(**TASK), episodes=12, steps=120, seed=5, out=work / "demos.npz")
    spec = {**SPEC, "data": {"path": "demos.npz", "sha256": sha256_file(work / "demos.npz")}}
    main_spec = write_spec(work / "spec.yaml", spec)
    store, root = work / "store", work / "root"
    ev = {"spec": {k: v for k, v in spec.items() if k != "data"}, "demonstrations_sha256": spec["data"]["sha256"]}
    fits = {}

    # 1. The study: every arm through every stage. These are the valid artifacts every later check starts from.
    p = run(main_spec, root, store)
    ev["study"] = {"exit": p.returncode, "stages": json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout else None}
    if p.returncode:
        print(p.stdout, p.stderr)
        return 1
    tr, rep = stage(root, "train"), stage(root, "report")
    fits["study"] = tr["trained"]
    ev["study"] |= {
        "units": sorted(tr["units"]),
        "stopping_outcomes": sorted({u["stopping"]["outcome"] for u in tr["units"].values()}),
        "audits": {s: stage(root, s)["all_pass"] for s in ("audit", "evaluation_audit")},
        "feasibility": tr["feasibility"], "exact_feasibility": tr["exact_feasibility"],
        "selected_sigmas": rep["selected_sigmas"],
        "store_entries": {k.name: len(list(k.iterdir())) for k in sorted(store.iterdir()) if k.is_dir()},
        "report": {"cells": {k: v["mean"] for k, v in rep["cells"].items()},
                   **{part: {k: {"mean": v["mean"], "bca": v["bca"], "interval": v["interval"], "outcome": v["outcome"]}
                             for k, v in rep[part].items()} for part in ("contrasts", "derived")}}}
    ev["study"]["pass"] = bool(
        set(ev["study"]["stages"].values()) == {"pass"} and all(ev["study"]["audits"].values())
        and ev["study"]["stopping_outcomes"] == ["patience"]
        and {u["arm"] for u in tr["units"].values()} == set(SPEC["arms"])
        and set(stage(root, "preflight")["units"]) == set(tr["analysis"])
        and all(v["bca"] is not None for part in ("contrasts", "derived") for v in rep[part].values()))

    # 2. The planner: preflight alone on a multi-coordinate matrix; its unit plan against the design, nothing trained.
    plan_spec = write_spec(work / "spec-planner.yaml", {**spec, **PLANNER, "generation": {
        **spec["generation"], "sigma": {f"r{r}/k{k}": SIGMA for r in PLANNER["real_fractions"]
                                        for k in PLANNER["synthetic_doses"]}}})
    policies_before = len(list((store / "policy").iterdir()))
    p = run(plan_spec, work / "r-planner", store, name="preflight")
    pf = stage(work / "r-planner", "preflight")
    expected = {f"{arm}/r{r}/k{k}/s{s}" for r in PLANNER["real_fractions"] for arm in SPEC["arms"]
                for k in ([0] if arm == "bc" else PLANNER["synthetic_doses"]) for s in PLANNER_SEEDS[k]}
    k_max = max(PLANNER["synthetic_doses"])
    roles_ok = all(u["seeds"] == ("all" if u["k"] in (0, k_max) else "surface") for u in pf["units"].values())
    views_ok = all(u["views"] == (None if u["arm"] == "bc" else f"{u['arm']}/r{u['r']}/s{u['seed']}/k{k_max}")
                   for u in pf["units"].values())
    real = pf["real_data"]
    nested = all(real[f"r1.0/s{s}"]["episodes"][:len(real[f"r0.5/s{s}"]["episodes"])] == real[f"r0.5/s{s}"]["episodes"]
                 and len(real[f"r0.5/s{s}"]["episodes"]) == 6 for s in PLANNER["seeds"])
    ev["planner"] = {"exit": p.returncode, "planned_units": pf["planned_units"],
                     "units_match_design": set(pf["units"]) == expected, "seed_roles": roles_ok,
                     "one_k_max_set_per_arm_r_seed": views_ok, "subsets_nested": nested,
                     "policies_trained": len(list((store / "policy").iterdir())) - policies_before,
                     "train_stage_written": (work / "r-planner" / "stages" / "train.json").exists()}
    ev["planner"]["pass"] = bool(p.returncode == 0 and pf["planned_units"] == len(expected) == 34 and roles_ok
                                 and views_ok and nested and set(pf["units"]) == expected
                                 and ev["planner"]["policies_trained"] == 0
                                 and not ev["planner"]["train_stage_written"])

    # 3. Exposure, from the saved learner records: the learned unit trains on the BC unit's demonstration rows plus
    # synthetic rows, draws every synthetic row before stopping, and realizes the requested fraction.
    def record(key):
        return json.loads((store / "policy" / tr["units"][key]["entry"] / "training.json").read_text())
    bc, cl = record("bc/r1.0/k0/s1"), record(f"ccil_learned/r1.0/k2/s1/sigma{SIGMA}")
    x = cl["exposure"]
    ev["exposure"] = {"demonstration_rows": cl["row_sources"]["demonstration"], "bc_rows": bc["rows"],
                      "synthetic_rows": cl["row_sources"]["synthetic"], "requested": x["requested_synthetic_fraction"],
                      "realized": x["realized_synthetic_row_fraction"], "within_tolerance": x["within_tolerance"],
                      "all_synthetic_rows_drawn_before_stop": x["synthetic_coverage"]["at_stop"]["all_rows_drawn"]}
    ev["exposure"]["pass"] = bool(
        cl["row_sources"]["demonstration"] == bc["rows"] == bc["row_sources"]["demonstration"]
        and cl["row_sources"]["synthetic"] == x["synthetic_coverage"]["rows"] > 0
        and x["requested_synthetic_fraction"] == LEARNER["synthetic_fraction"] and x["within_tolerance"]
        and x["rows"]["synthetic"] == sum(e["rows_seen"]["synthetic"] for e in cl["epochs"]) > 0
        and x["synthetic_coverage"]["at_stop"]["all_rows_drawn"])

    # 4. Resume: the same command reruns no stage.
    p = run(main_spec, root, store)
    ev["resume"] = {"exit": p.returncode, "stages": json.loads(p.stdout.strip().splitlines()[-1]) if p.stdout else None}
    ev["resume"]["pass"] = p.returncode == 0 and set((ev["resume"]["stages"] or {}).values()) == {"resumed"}

    # 5. Reuse across studies: a new root on the same store trains nothing and reports identically.
    root2 = work / "root-reuse"
    p = run(main_spec, root2, store)
    rep2 = stage(root2, "report") if p.returncode == 0 else {}
    ev["reuse"] = {"exit": p.returncode, "trained": stage(root2, "train")["trained"] if p.returncode == 0 else None,
                   "report_identical": {k: rep2.get(k) for k in ("cells", "contrasts", "derived")} ==
                   {k: rep.get(k) for k in ("cells", "contrasts", "derived")}}
    ev["reuse"]["pass"] = p.returncode == 0 and ev["reuse"]["trained"] == 0 and ev["reuse"]["report_identical"]

    # 6. Refusals, each on its own copy; none trains.
    refusals = {}
    bad = write_spec(work / "spec-wrong-hash.yaml", {**spec, "data": {"path": "demos.npz", "sha256": "0" * 64}})
    p = run(bad, work / "r-hash", store)
    refusals["wrong_input_hash"] = p.returncode != 0 and "not the declared dataset" in p.stderr + p.stdout

    root3, store3 = work / "r-corrupt", work / "store-corrupt"
    shutil.copytree(root, root3)
    shutil.copytree(store, store3)
    victim = min((store3 / "rollout").iterdir()) / "rollout.npz"
    data = bytearray(victim.read_bytes())
    data[len(data) // 2] ^= 0xFF
    victim.write_bytes(bytes(data))
    p = run(main_spec, root3, store3, name="evaluation_audit")
    ea = stage(root3, "evaluation_audit")
    refusals["corrupted_rollout"] = p.returncode != 0 and not ea["all_pass"] and len(ea["problems"]) == 1
    p = run(main_spec, root3, store3, name="report")
    refusals["report_after_failed_audit"] = p.returncode != 0 and "evaluation_audit" in p.stderr + p.stdout

    root4, store4 = work / "r-mismatch", work / "store-mismatch"
    shutil.copytree(store, store4)
    entry = min((store4 / "policy").iterdir())
    ident = json.loads((entry / "identity.json").read_text())
    ident["seed"] = 999
    (entry / "identity.json").write_text(json.dumps(ident, sort_keys=True, separators=(",", ":")) + "\n")
    p = run(main_spec, root4, store4)
    tr4 = stage(root4, "train")
    refusals["mismatched_reuse"] = (p.returncode != 0 and not tr4["all_pass"]
                                    and "mismatched reuse" in tr4.get("error", "")
                                    and len(list((store4 / "policy").iterdir()))
                                    == len(list((store / "policy").iterdir())))

    p = run(main_spec, work / "r-order", store, name="evaluate")
    refusals["out_of_order"] = p.returncode != 0 and "needs a completed preflight" in p.stderr + p.stdout
    ev["refusals"] = refusals

    # 7. A changed implementation: the BC cell, reused under the same code, retrains under a copy of the installed
    # harness whose source differs by one comment.
    bc_only = {**spec, "arms": ["bc"], "analysis": {**spec["analysis"], "contrasts": [], "derived": []}}
    bc_spec = write_spec(work / "spec-bc.yaml", bc_only)
    for name in ("preflight", "train"):
        p = run(bc_spec, work / "r-same", store, name=name)
    same = stage(work / "r-same", "train")
    impl = work / "impl"
    shutil.copytree(Path(ilsb.__file__).resolve().parent, impl / "ilsb", ignore=shutil.ignore_patterns("__pycache__"))
    with open(impl / "ilsb" / "run.py", "a") as f:
        f.write("# implementation changed\n")
    env = {**os.environ, "PYTHONPATH": str(impl)}
    root5 = work / "r-impl"
    for name in ("preflight", "train"):
        p5 = run(bc_spec, root5, store, name=name, env=env, allow=True)
    tr5 = stage(root5, "train")
    fits["implementation_change"] = tr5["trained"]
    ev["implementation_change"] = {
        "same_code": {"exit": p.returncode, "trained": same["trained"], "units": len(same["units"])},
        "changed_code": {"exit": p5.returncode, "trained": tr5["trained"], "units": len(tr5["units"])},
        "source_differs": json.loads((root5 / "root.json").read_text())["scope"].get("ilsb_source_sha256")
        != json.loads((root / "root.json").read_text())["scope"].get("ilsb_source_sha256")}
    ev["implementation_change"]["pass"] = bool(
        p.returncode == 0 and same["trained"] == 0 and same["units"].keys() == tr5["units"].keys()
        and p5.returncode == 0 and tr5["trained"] == len(tr5["units"]) == len(SPEC["seeds"])
        and ev["implementation_change"]["source_differs"])

    # 8. A ceiling below any patience stop: the BC cell is an incomplete fit, the stage fails, and the audit refuses.
    ceiling = write_spec(work / "spec-ceiling.yaml", {**bc_only, "learner": {**LEARNER, "step_budget": 40,
                                                                                     "patience": 50}})
    root6 = work / "r-ceiling"
    p = run(ceiling, root6, store)
    tr6 = stage(root6, "train")
    fits["incomplete_fit"] = tr6["trained"]
    p2 = run(ceiling, root6, store, name="audit")
    ev["incomplete_fit"] = {"run_exit": p.returncode, "train_all_pass": tr6["all_pass"],
                            "incomplete_units": tr6["incomplete_fit"], "units": sorted(tr6["units"]),
                            "audit_refused": p2.returncode != 0
                            and "needs a completed train stage" in p2.stderr + p2.stdout}
    ev["incomplete_fit"]["pass"] = bool(p.returncode != 0 and tr6["all_pass"] is False
                                        and tr6["incomplete_fit"] == sorted(tr6["units"]) and tr6["units"]
                                        and ev["incomplete_fit"]["audit_refused"])

    # 9. Feasibility, with a sigma grid selected per (r, K, seed). The support radius sits between the two sigmas' label
    # distances, so the small sigma fails only by rare solve failures and the large one mostly falls outside the
    # support, far on either side of max_failure. The grid trains only the feasible sigma (the study's units, reused);
    # the infeasible sigma alone leaves every coordinate unselectable and fails the train stage.
    gate = {}
    for name_, sigmas in (("mixed", [SIGMA, SIGMA_INFEASIBLE]), ("infeasible_only", [SIGMA_INFEASIBLE])):
        g_spec = write_spec(work / f"spec-gate-{name_}.yaml", {
            **spec, "arms": ["bc", "ccil_learned"],
            "generation": {**{k: v for k, v in spec["generation"].items() if k != "sigma"}, "sigma_grid": sigmas},
            "analysis": {**spec["analysis"], "contrasts": [["ccil_learned", "bc"]], "derived": []}})
        g_root = work / f"r-gate-{name_}"
        p = run(g_spec, g_root, store)
        t = stage(g_root, "train")
        fits[f"feasibility_{name_}"] = t["trained"]
        gate[name_] = {"exit": p.returncode, "train_all_pass": t["all_pass"], "feasibility": t["feasibility"],
                       "trained_sigmas": sorted({u["sigma"] for u in t["units"].values()
                                                 if u["arm"] == "ccil_learned"}),
                       "trained": t["trained"], "unselectable": t["unselectable"]}
    limit = spec["generation"]["max_failure"]
    expected = {f"ccil_learned/r1.0/s{s_}/sigma{x}": x == SIGMA for s_ in SPEC["seeds"]
                for x in (SIGMA, SIGMA_INFEASIBLE)}
    mixed, alone = gate["mixed"], gate["infeasible_only"]
    gate["pass"] = bool(
        {k_: v["feasible"] for k_, v in mixed["feasibility"].items()} == expected
        and all((v["failure"] <= limit) == v["feasible"] for g in (mixed, alone) for v in g["feasibility"].values())
        and mixed["exit"] == 0 and mixed["trained_sigmas"] == [SIGMA] and mixed["trained"] == 0
        and alone["exit"] != 0 and alone["train_all_pass"] is False and alone["trained_sigmas"] == []
        and alone["trained"] == 0 and alone["unselectable"] == [f"r1.0/k2/s{s_}" for s_ in SPEC["seeds"]])
    ev["feasibility"] = gate

    ev["policy_fits"] = {**fits, "total": sum(fits.values())}
    ev["all_pass"] = bool(all(ev[k]["pass"] for k in ("study", "planner", "exposure", "resume", "reuse",
                                                      "implementation_change", "incomplete_fit", "feasibility"))
                          and all(refusals.values()))
    text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(text)
    return 0 if ev["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
