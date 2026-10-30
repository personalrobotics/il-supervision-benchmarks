"""One study, one state machine: preflight -> train -> audit -> evaluate -> evaluation_audit -> report.

    uv run python -m ilsb run SPEC --root ROOT [--store STORE]     # every stage in order, resuming
    uv run python -m ilsb stage NAME SPEC --root ROOT [--store S]  # one stage (it refuses if out of order)

The run root holds root.json (the protocol identity: the specification, the harness and ccil commits, the lock, and
the scope every store identity carries: the harness source, ccil commit, lock and execution environment),
stages/<stage>.json (each stage's report) and manifest.json (each completed report's SHA-256). A stage runs only
when every predecessor's report passed and still hashes to the manifest's record under the same protocol; a failed
check stops the run, and its report is written with all_pass false. Units are content-addressed store entries, so a
rerun reuses every verified unit and recomputes nothing; the store may be shared by studies.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import platform
import subprocess
import sys
from importlib import metadata
from pathlib import Path

import numpy as np
import yaml

from . import analysis
from .learners import vector_bc
from .store import Store, StoreError, digest, sha256_file, write_json
from .supervision import ccil as sup
from .tasks.pendulum import ExactModel, Task, subset

STAGES = ("preflight", "train", "audit", "evaluate", "evaluation_audit", "report")
REPO = Path(__file__).resolve().parents[2]


class Refused(SystemExit):
    pass


# --- protocol ---------------------------------------------------------------------------------------------------------
def load_spec(path: Path) -> dict:
    spec = yaml.safe_load(Path(path).read_text())
    need = {"study", "task", "data", "split", "real_fractions", "synthetic_doses", "arms", "seeds", "generation",
            "learner", "analysis", "compute"}
    if set(spec) - need - {"dynamics"} or need - set(spec):
        raise Refused(f"refused: specification keys {sorted(spec)} differ from {sorted(need)} (+ dynamics)")
    if not set(spec["arms"]) <= set(sup.ARMS) or "bc" not in spec["arms"]:
        raise Refused(f"refused: arms must include bc and be among {sup.ARMS}")
    if "ccil_learned" in spec["arms"] and "dynamics" not in spec:
        raise Refused("refused: ccil_learned needs dynamics settings")
    if any(not 0 < f <= 1 for f in spec["real_fractions"]) or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                                                                  for k in spec["synthetic_doses"]):
        raise Refused("refused: real fractions must be in (0, 1] and synthetic doses integers >= 1")
    if len(set(spec["seeds"])) != len(spec["seeds"]):
        raise Refused("refused: seeds repeat")
    if "ccil_exact" in spec["arms"] and "ccil_learned" not in spec["arms"]:
        raise Refused("refused: ccil_exact takes its sigma from the learned arm's selection, so it needs ccil_learned")
    g = spec["generation"]
    grid = g.get("sigma_grid")
    if (not isinstance(grid, list) or not grid or grid != sorted(set(grid))
            or any(isinstance(x, bool) or not isinstance(x, (int, float)) or not x > 0 for x in grid)):
        raise Refused("refused: generation.sigma_grid must be a non-empty increasing list of positive numbers")
    if "action_noise_std" in g or not 0 <= g.get("max_failure", -1) <= 1:
        raise Refused("refused: sigma is selected from generation.sigma_grid (no action_noise_std), and "
                      "generation.max_failure must be in [0, 1]")
    if not spec["learner"].get("patience") or spec["learner"].get("checkpoint_selection") != "validation":
        raise Refused("refused: the learner needs the validation stopping rule (patience, checkpoint_selection: "
                      "validation); the step budget is only a failure ceiling")
    return spec


def code_state() -> dict:
    def git(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    try:
        direct = json.loads(metadata.distribution("ccil").read_text("direct_url.json") or "{}")
    except metadata.PackageNotFoundError:
        direct = {}
    lock = REPO / "uv.lock"
    return {"harness_commit": git("rev-parse", "HEAD"), "dirty": git("status", "--porcelain").splitlines(),
            "ccil": {"version": metadata.version("ccil"), "commit": (direct.get("vcs_info") or {}).get("commit_id"),
                     "url": direct.get("url")},
            "uv_lock_sha256": sha256_file(lock) if lock.exists() else None}


def source_sha256() -> str:
    """The harness implementation by content: every source file of the ilsb package, so a dirty tree is named too."""
    h = hashlib.sha256()
    pkg = Path(__file__).resolve().parent
    for f in sorted(pkg.rglob("*.py")):
        h.update(f.relative_to(pkg).as_posix().encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


def environment(device: str) -> dict:
    """The execution environment numerical artifacts belong to; they are not reused outside it."""
    import torch
    env = {"python": platform.python_version(), "system": platform.system(), "machine": platform.machine(),
           "numpy": np.__version__, "torch": torch.__version__, "device": device}
    if device.startswith("cuda"):
        env |= {"cuda": torch.version.cuda, "gpu": torch.cuda.get_device_name(torch.device(device))}
    return env


class Context:
    def __init__(self, spec_path: Path, root: Path, store: Path | None, allow_dirty: bool):
        self.spec_path, self.root = Path(spec_path).resolve(), Path(root).resolve()
        self.spec = load_spec(self.spec_path)
        self.task = Task(**self.spec["task"])
        self.device = self.spec["compute"]["device"]
        state = code_state()
        if state["dirty"] and not allow_dirty:
            raise Refused(f"refused: the harness tree is dirty: {state['dirty'][:5]}")
        if not state["ccil"]["commit"] and not allow_dirty:
            raise Refused("refused: ccil is not installed from a pinned commit, so its implementation is unnamed")
        # Every store identity carries the implementation (harness source, ccil commit, lock) and the environment.
        self.scope = {"ilsb_source_sha256": source_sha256(), "ccil": state["ccil"],
                      "uv_lock_sha256": state["uv_lock_sha256"], "environment": environment(self.device)}
        self.store = Store(Path(store).resolve() if store else self.root / "store", self.scope)
        self.protocol = {"spec": self.spec, "spec_sha256": sha256_file(self.spec_path), "code": state,
                         "scope": self.scope, "allow_dirty": allow_dirty}
        rj = self.root / "root.json"
        if rj.exists():
            if json.loads(rj.read_text()) != json.loads(json.dumps(self.protocol, sort_keys=True, default=str)):
                raise Refused(f"refused: {self.root} was created under a different protocol (specification, code "
                              "or lock); use a new root")
        else:
            write_json(rj, self.protocol)

    # every planned unit, by coordinate
    def cells(self):
        k_doses = sorted(self.spec["synthetic_doses"])
        for r in self.spec["real_fractions"]:
            yield "bc", r, 0
            for arm in self.spec["arms"]:
                if arm != "bc":
                    for k in k_doses:
                        yield arm, r, k

    def dataset(self) -> Path:
        p = Path(self.spec["data"]["path"])
        p = p if p.is_absolute() else self.spec_path.parent / p
        if not p.exists() or sha256_file(p) != self.spec["data"]["sha256"]:
            raise Refused(f"refused: {p} is not the declared dataset {self.spec['data']['sha256']}")
        return p

    def learner_settings(self, arm: str) -> dict:
        s = dict(self.spec["learner"])
        if arm == "bc":
            s.pop("synthetic_fraction", None)  # BC has no synthetic rows; everything else is identical
        return s

    def generation(self, sigma: float) -> dict:
        """sigma is in normalized action units (a fraction of the action bound); the generator takes action units."""
        g = {k: v for k, v in self.spec["generation"].items() if k not in ("sigma_grid", "max_failure")}
        return {**g, "sigma_normalized": sigma, "action_noise_std": sigma * self.task.action_bound,
                "k_max": max(self.spec["synthetic_doses"]), "action_bound": self.task.action_bound}


# --- stage plumbing ---------------------------------------------------------------------------------------------------
def _manifest(ctx) -> dict:
    m = ctx.root / "manifest.json"
    return json.loads(m.read_text()) if m.exists() else {"protocol_sha256": digest(ctx.protocol), "stages": {}}


def require(ctx, stage: str) -> dict:
    """Each predecessor's passing report, unchanged since it was recorded, under this protocol."""
    m = _manifest(ctx)
    if m["protocol_sha256"] != digest(ctx.protocol):
        raise Refused("refused: the manifest belongs to another protocol")
    prior = {}
    for prev in STAGES[:STAGES.index(stage)]:
        f = ctx.root / "stages" / f"{prev}.json"
        if prev not in m["stages"] or not f.exists():
            raise Refused(f"refused: {stage} needs a completed {prev} stage")
        if sha256_file(f) != m["stages"][prev]:
            raise Refused(f"refused: stages/{prev}.json changed after it was recorded")
        rep = json.loads(f.read_text())
        if rep.get("all_pass") is not True:
            raise Refused(f"refused: the {prev} stage did not pass")
        prior[prev] = rep
    return prior


def record(ctx, stage: str, report: dict) -> dict:
    report = {"stage": stage, "protocol_sha256": digest(ctx.protocol), **report}
    f = ctx.root / "stages" / f"{stage}.json"
    write_json(f, report)
    if report.get("all_pass"):
        m = _manifest(ctx)
        m["stages"][stage] = sha256_file(f)
        write_json(ctx.root / "manifest.json", m)
    return report


def data_units(ctx, r: float, seed: int) -> tuple[Path, Path]:
    """The (subset, split) entries of real fraction r for a seed: nested in r, with a seed-keyed episode split."""
    from ccil.data.split import make_split
    from ccil.schema import read_trajectories
    ds = ctx.dataset()
    s_entry = ctx.store.ensure("subset", {"kind": "subset", "dataset": ctx.spec["data"]["sha256"], "fraction": r,
                                          "seed": seed, "rule": "SeedSequence([seed, 9]) permutation prefix"},
                               lambda d: subset(ds, r, seed, d / "subset.npz"))[0] / "subset.npz"
    frac = ctx.spec["split"]["validation_fraction"]

    def write_split(d):
        eps = [e.episode_id for e in read_trajectories(s_entry).episodes]
        (d / "split.json").write_text(json.dumps(make_split(eps, frac, seed), sort_keys=True) + "\n")
    sp = ctx.store.ensure("split", {"kind": "split", "subset": sha256_file(s_entry), "validation_fraction": frac,
                                    "seed": seed}, write_split)[0] / "split.json"
    return s_entry, sp


def full_candidates(ctx, arm: str, r: float, seed: int, sigma: float) -> Path:
    """The K_max candidate set of a CCIL arm at (r, seed, sigma), under the learned or the exact dynamics."""
    subset_p, split_p = data_units(ctx, r, seed)
    if arm == "ccil_exact":
        model = ExactModel(ctx.task)
        model_sha = model.sha256
    else:
        from ccil.core import load_transition_model
        dyn = sup.fit_dynamics(ctx.store, subset_p, split_p, ctx.spec["dynamics"], seed, ctx.device)
        model, model_sha = load_transition_model(dyn / "model.npz", ctx.device), sha256_file(dyn / "model.npz")
    return sup.generate(ctx.store, subset_p, model, model_sha, ctx.generation(sigma), seed) / "candidates.npz"


GENUINE_FAILURES = ("non_finite", "generation_failed", "out_of_support")


def failure(path: Path) -> float:
    """The fraction of candidates that fail genuinely: non-finite, a failed solve, or a state outside the support.
    Proposals are drawn inside the action bounds, so an action-bound rejection is an implementation error."""
    from ccil.schema import read_candidates
    c = read_candidates(path)
    reasons = dict(zip(c.reason_codes, np.asarray(c.reasons).T))
    if reasons.get("action_out_of_bounds", np.zeros(1)).any():
        raise StoreError(f"{path}: a bounded proposal was rejected by the action bound (implementation error)")
    return float(np.any([reasons[r] for r in GENUINE_FAILURES if r in reasons], axis=0).mean())


def unit(ctx, arm, r, k, seed, sigma=None, full=None) -> dict:
    subset_p, split_p = data_units(ctx, r, seed)
    key = f"{arm}/r{r}/k{k}/s{seed}" + (f"/sigma{sigma}" if arm == "ccil_learned" else "")
    return {"key": key, "arm": arm, "r": r, "k": k, "seed": seed, "sigma": sigma, "subset": subset_p,
            "split": split_p, "candidates": None if full is None else sup.view(ctx.store, full, k) / "candidates.npz"}


def plan(ctx, selection: dict | None = None) -> tuple[dict, dict]:
    """The policy groups, each trained together, and every candidate set's genuine-failure fraction with whether it is
    feasible (an infeasible sigma is never trained). Without a selection: BC, and the learned arm at every feasible
    sigma of the grid. With one: the exact arm at each coordinate's selected sigma."""
    sp, g = ctx.spec, ctx.spec["generation"]
    groups, feasibility = {}, {}
    for r in sp["real_fractions"]:
        for seed in sp["seeds"]:
            if selection is None:
                groups.setdefault(("bc", r, 0, None), []).append(unit(ctx, "bc", r, 0, seed))
                for sigma in g["sigma_grid"] if "ccil_learned" in sp["arms"] else ():
                    full = full_candidates(ctx, "ccil_learned", r, seed, sigma)
                    fail = failure(full)
                    feasibility[f"ccil_learned/r{r}/s{seed}/sigma{sigma}"] = {
                        "failure": fail, "feasible": fail <= g["max_failure"]}
                    if fail > g["max_failure"]:
                        continue
                    for k in sorted(sp["synthetic_doses"]):
                        groups.setdefault(("ccil_learned", r, k, sigma), []).append(
                            unit(ctx, "ccil_learned", r, k, seed, sigma, full))
            elif "ccil_exact" in sp["arms"]:
                for k in sorted(sp["synthetic_doses"]):
                    sigma = selection[f"r{r}/k{k}/s{seed}"]["selected_sigma"]
                    full = full_candidates(ctx, "ccil_exact", r, seed, sigma)
                    fail = failure(full)
                    feasibility[f"ccil_exact/r{r}/k{k}/s{seed}/sigma{sigma}"] = {
                        "failure": fail, "feasible": fail <= g["max_failure"]}
                    if fail > g["max_failure"]:
                        continue
                    groups.setdefault(("ccil_exact", r, k, None), []).append(
                        unit(ctx, "ccil_exact", r, k, seed, sigma, full))
    for us in groups.values():  # a policy's identity names its whole training group
        group = sorted([u["seed"], None if u["candidates"] is None else sha256_file(u["candidates"])] for u in us)
        for u in us:
            u["identity"] = policy_identity(ctx, u, group)
    return groups, feasibility


def policy_identity(ctx, u, group) -> dict:
    return {"kind": "policy", "arm": u["arm"], "subset": sha256_file(u["subset"]), "split": sha256_file(u["split"]),
            "candidates": None if u["candidates"] is None else sha256_file(u["candidates"]),
            "settings": ctx.learner_settings(u["arm"]), "seed": u["seed"], "group": group,
            "learner": "ccil.training.vector_bc_v2"}


def validation_mse(u, d: Path) -> float:
    """The selected checkpoint's mean squared action error, in raw action units, on the unit's held-out
    demonstration episodes: comparable across sigma, unlike the trainer's loss in each model's own scaled units."""
    obs, act, _ = sup.supervision(u["arm"], u["subset"], u["split"], None, "validation").arrays()
    err = np.asarray(vector_bc.load(d)(obs), np.float64).reshape(act.shape) - act.astype(np.float64)
    return float((err ** 2).sum(axis=1).mean())


def row_of(u, d: Path) -> dict:
    rec = json.loads((d / "training.json").read_text())
    return {"entry": d.name, "arm": u["arm"], "r": u["r"], "k": u["k"], "seed": u["seed"], "sigma": u["sigma"],
            "policy_sha256": sha256_file(d / "policy.npz"), "stopping": rec["stopping"],
            "exposure": {k_: rec["exposure"][k_] for k_ in ("rows", "realized_synthetic_row_fraction",
                                                             "requested_synthetic_fraction")},
            "selected_step": (rec["validation"] or {}).get("selected_step"), "validation_mse": validation_mse(u, d)}


def select(ctx, rows: dict) -> dict:
    """Per (r, K, seed): the learned arm's sigma with the least validation error; ties go to the smaller sigma."""
    out = {}
    for r in ctx.spec["real_fractions"]:
        for k in sorted(ctx.spec["synthetic_doses"]):
            for seed in ctx.spec["seeds"]:
                c = {x["sigma"]: x["validation_mse"] for x in rows.values()
                     if x["arm"] == "ccil_learned" and (x["r"], x["k"], x["seed"]) == (r, k, seed)}
                out[f"r{r}/k{k}/s{seed}"] = {"candidates": {str(s_): v for s_, v in sorted(c.items())},
                                             "selected_sigma": min(c, key=lambda s_: (c[s_], s_)) if c else None}
    return out


def analysis_units(ctx, selection: dict) -> dict:
    """Each analysis unit's policy row: the learned arm's is the selected sigma's."""
    out = {}
    for arm, r, k in ctx.cells():
        for seed in ctx.spec["seeds"]:
            key_ = f"{arm}/r{r}/k{k}/s{seed}"
            out[key_] = (f"{key_}/sigma{selection[f'r{r}/k{k}/s{seed}']['selected_sigma']}"
                         if arm == "ccil_learned" else key_)
    return out


# --- stages -----------------------------------------------------------------------------------------------------------
def preflight(ctx) -> dict:
    ctx.dataset()
    plan = [f"{a}/r{r}/k{k}/s{s}" for a, r, k in ctx.cells() for s in ctx.spec["seeds"]]
    return {"all_pass": True, "dataset_sha256": ctx.spec["data"]["sha256"], "planned_units": len(plan),
            "plan_sha256": digest(plan), "code": ctx.protocol["code"]}


def train_groups(ctx, groups: dict) -> dict:
    rows = {}
    for (arm, *_), cell in groups.items():
        todo = [u for u in cell if ctx.store.get("policy", u["identity"]) is None]
        if todo:  # the whole group trains together (each identity names the group)
            jobs = [(sup.supervision(arm, u["subset"], u["split"], u["candidates"], "train"),
                     sup.supervision(arm, u["subset"], u["split"], None, "validation"), u["seed"]) for u in cell]
            results = vector_bc.train(jobs, ctx.learner_settings(arm), ctx.device)
            for u, (policy, rec) in zip(cell, results):
                if u in todo:
                    ctx.store.put("policy", u["identity"], lambda d, p=policy, rc=rec: vector_bc.save(p, rc, d))
        for u in cell:
            rows[u["key"]] = {**row_of(u, ctx.store.get("policy", u["identity"])), "reused": u not in todo}
    return rows


def train(ctx) -> dict:
    """BC and the learned arm at every feasible sigma; the selection; then the exact arm at the selected sigmas."""
    groups, feasibility = plan(ctx)
    rows = train_groups(ctx, groups)
    selection = select(ctx, rows) if "ccil_learned" in ctx.spec["arms"] else {}
    unselectable = sorted(k_ for k_, v in selection.items() if v["selected_sigma"] is None)
    exact_feasibility = {}
    if "ccil_exact" in ctx.spec["arms"] and not unselectable:
        g2, exact_feasibility = plan(ctx, selection)
        rows |= train_groups(ctx, g2)
    exact_infeasible = sorted(k_ for k_, v in exact_feasibility.items() if not v["feasible"])
    # The stopping rule must have fired for every unit: reaching the step ceiling is a failed fit, not a result.
    incomplete = sorted(k_ for k_, v in rows.items() if v["stopping"]["outcome"] != "patience")
    ok = not incomplete and not unselectable and not exact_infeasible
    return {"all_pass": ok, "units": rows, "trained": sum(v["reused"] is False for v in rows.values()),
            "feasibility": feasibility, "selection": selection, "unselectable": unselectable,
            "exact_feasibility": exact_feasibility, "exact_infeasible": exact_infeasible, "incomplete_fit": incomplete,
            "analysis": analysis_units(ctx, selection) if ok else None}


def audit(ctx) -> dict:
    """Provenance and selection, recomputed. Every trained unit is a verified store entry, trained on exactly this
    unit's supervision (its training identity, which hashes the supervision manifest with its dataset, split and
    candidate hashes, recomputes from the unit's own inputs), with the weights its record describes and the row the
    train stage reported. Feasibility, the sigma selection and the analysis units recompute to the reported ones."""
    from ccil.training import vector_bc_v2 as v2
    prior = require(ctx, "audit")
    tr = prior["train"]
    problems, rows = {}, {}

    def check(groups):
        for cell in groups.values():
            for u in cell:
                try:
                    d = ctx.store.get("policy", u["identity"])
                    if d is None or tr["units"].get(u["key"], {}).get("entry") != d.name:
                        raise StoreError("missing, or not the entry the train stage recorded")
                    rec = json.loads((d / "training.json").read_text())
                    settings = ctx.learner_settings(u["arm"])
                    t_ = sup.supervision(u["arm"], u["subset"], u["split"], u["candidates"], "train")
                    va = sup.supervision(u["arm"], u["subset"], u["split"], None, "validation")
                    if v2.training_identity_v2(t_, settings, u["seed"], None, va)["sha256"] != \
                            rec["training_identity_sha256"]:
                        raise StoreError("the policy was not trained on this unit's supervision and settings")
                    if v2.parameters_sha256(vector_bc.load(d).__self__) != rec["parameters_sha256"]:
                        raise StoreError("the stored weights are not the weights the training record describes")
                    rows[u["key"]] = row_of(u, d)
                    reported = {k_: v for k_, v in tr["units"][u["key"]].items() if k_ != "reused"}
                    if _json(rows[u["key"]]) != reported:
                        raise StoreError("the recomputed unit row differs from the train stage's")
                except StoreError as e:
                    problems[u["key"]] = str(e)
    groups, feasibility = plan(ctx)
    check(groups)
    selection = select(ctx, rows)
    exact_feasibility = {}
    if "ccil_exact" in ctx.spec["arms"]:
        g2, exact_feasibility = plan(ctx, selection)
        check(g2)
    for name, mine in (("feasibility", feasibility), ("selection", selection), ("exact_feasibility", exact_feasibility),
                       ("analysis", analysis_units(ctx, selection)), ("units", set(rows))):
        theirs = set(tr["units"]) if name == "units" else tr[name]
        if (mine if name == "units" else _json(mine)) != theirs:
            problems[name] = f"the recomputed {name} differs from the train stage's"
    return {"all_pass": not problems, "units": len(tr["units"]), "problems": problems}


def _json(x):
    return json.loads(json.dumps(x, sort_keys=True, default=str))


def evaluate(ctx) -> dict:
    tr = require(ctx, "evaluate")["train"]
    rows = {}
    for key_, row_key in tr["analysis"].items():  # only the analysis units; no unselected sigma is ever evaluated
        pol = ctx.store.root / "policy" / tr["units"][row_key]["entry"]
        ident = {"kind": "rollout", "policy": sha256_file(pol / "policy.npz"), "task": ctx.task.identity()}

        def write(d, pol=pol):
            [ro] = ctx.task.rollouts([vector_bc.load(pol, "cpu")])
            np.savez(d / "rollout.npz", **ro)
            (d / "scores.json").write_text(json.dumps(ctx.task.score(ro), sort_keys=True) + "\n")
        d, reused = ctx.store.ensure("rollout", ident, write)
        rows[key_] = {"entry": d.name, "rollout_sha256": sha256_file(d / "rollout.npz"), "reused": reused,
                        "scores": json.loads((d / "scores.json").read_text())}
    return {"all_pass": True, "units": rows}


def evaluation_audit(ctx) -> dict:
    """Reopen every rollout: hashes, shapes, starts, the action bound, every transition replayed through the map, and
    every score recomputed and compared exactly with the evaluate stage's."""
    prior = require(ctx, "evaluation_audit")
    problems = {}
    starts = ctx.task.starts()
    for k_, row in prior["evaluate"]["units"].items():
        d = ctx.store.root / "rollout" / row["entry"]
        try:
            if not d.exists() or sha256_file(d / "rollout.npz") != row["rollout_sha256"]:
                raise StoreError("rollout missing or changed since evaluation")
            with np.load(d / "rollout.npz", allow_pickle=False) as z:
                ro = {n: z[n] for n in z.files}
            E, T = ctx.task.eval_episodes, ctx.task.horizon
            if ro["states"].shape != (E, T + 1, 3) or ro["commanded"].shape != (E, T):
                raise StoreError("rollout shapes are not the task's")
            if not np.array_equal(ro["states"][:, 0], starts):
                raise StoreError("rollout does not start from the fixed final-test starts")
            if not np.array_equal(ro["applied"], np.clip(ro["commanded"], -ctx.task.action_bound, ctx.task.action_bound)):
                raise StoreError("applied actions are not the commanded actions clipped to the bound")
            nxt, _, _ = ctx.task.step(ro["states"][:, :-1].reshape(-1, 3), ro["commanded"].reshape(-1))
            if not np.array_equal(nxt.reshape(E, T, 3), ro["states"][:, 1:]):
                raise StoreError("transitions do not replay through the task's map")
            if ctx.task.score(ro) != row["scores"]:
                raise StoreError("recomputed scores differ from the evaluate stage's")
        except (StoreError, OSError, ValueError, KeyError) as e:
            problems[k_] = str(e)
    return {"all_pass": not problems, "units": len(prior["evaluate"]["units"]), "problems": problems}


def report(ctx) -> dict:
    prior = require(ctx, "report")
    a = ctx.spec["analysis"]
    scores = prior["evaluate"]["units"]

    def values(arm, r, k):
        return {s: scores[f"{arm}/r{r}/k{k}/s{s}"]["scores"][a["metric"]] for s in ctx.spec["seeds"]}
    cells, contrasts = {}, {}
    for arm, r, k in ctx.cells():
        v = values(arm, r, k)
        cells[f"{arm}/r{r}/k{k}"] = {"mean": float(np.mean(list(v.values()))), "per_seed": v}
    for A, B in a["contrasts"]:
        for r in ctx.spec["real_fractions"]:
            for k in sorted(ctx.spec["synthetic_doses"]):
                kb = 0 if B == "bc" else k
                contrasts[f"{A}-{B}/r{r}/k{k}"] = analysis.paired(values(A, r, k), values(B, r, kb), a["confidence"],
                                                                  a["min_pairs"])
    sel = prior["train"]["selection"]
    chosen = {f"r{r}/k{k}": sorted(sel[f"r{r}/k{k}/s{s_}"]["selected_sigma"] for s_ in ctx.spec["seeds"])
              for r in ctx.spec["real_fractions"] for k in sorted(ctx.spec["synthetic_doses"])} if sel else {}
    return {"all_pass": True, "metric": a["metric"], "cells": cells, "contrasts": contrasts, "selected_sigmas": chosen}


RUNNERS = {"preflight": preflight, "train": train, "audit": audit, "evaluate": evaluate,
           "evaluation_audit": evaluation_audit, "report": report}


def run_stage(ctx, stage: str) -> dict:
    require(ctx, stage)
    try:
        rep = RUNNERS[stage](ctx)
    except StoreError as e:  # a refused reuse or a corrupted entry: the stage fails closed, with its report
        rep = {"all_pass": False, "error": str(e)}
    return record(ctx, stage, rep)


def run_all(ctx) -> dict:
    m = _manifest(ctx)
    done = {}
    for stage in STAGES:
        f = ctx.root / "stages" / f"{stage}.json"
        if stage in m["stages"] and f.exists() and sha256_file(f) == m["stages"][stage]:
            require(ctx, stage)  # resume: a verified completed stage is not rerun
            done[stage] = "resumed"
            continue
        rep = run_stage(ctx, stage)
        done[stage] = "pass" if rep["all_pass"] else "FAIL"
        if not rep["all_pass"]:
            break
    return done


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="ilsb")
    sub_ = ap.add_subparsers(dest="cmd", required=True)
    for name in ("run", "stage"):
        p = sub_.add_parser(name)
        if name == "stage":
            p.add_argument("name", choices=STAGES)
        p.add_argument("spec", type=Path)
        p.add_argument("--root", type=Path, required=True)
        p.add_argument("--store", type=Path)
        p.add_argument("--allow-dirty", action="store_true", help="smoke and fixture runs only; recorded")
    a = ap.parse_args(argv)
    ctx = Context(a.spec, a.root, a.store, a.allow_dirty)
    if a.cmd == "run":
        done = run_all(ctx)
        print(json.dumps(done))
        return 0 if all(v != "FAIL" for v in done.values()) and len(done) == len(STAGES) else 1
    rep = run_stage(ctx, a.name)
    print(json.dumps({"stage": a.name, "all_pass": rep["all_pass"]}))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
