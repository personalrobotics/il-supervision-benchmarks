"""One study, one state machine: preflight -> train -> audit -> evaluate -> evaluation_audit -> report.

    uv run python -m ilsb run SPEC --root ROOT [--store STORE]     # every stage in order, resuming
    uv run python -m ilsb stage NAME SPEC --root ROOT [--store S]  # one stage (it refuses if out of order)
    uv run python -m ilsb policy train CONFIG                      # one vector policy on CCIL supervision (policy.py)

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
from . import policy as policy_cmd
from .learners import vector_bc
from .store import Store, StoreError, digest, sha256_file, write_json
from .supervision import ccil as sup
from .tasks.pendulum import ExactModel, Task, subset

STAGES = ("preflight", "train", "audit", "evaluate", "evaluation_audit", "report")
REPO = Path(__file__).resolve().parents[2]


class Refused(SystemExit):
    pass


# --- protocol ---------------------------------------------------------------------------------------------------------
def study_of(spec: dict):
    """The study module of a specification whose task declares a kind (None: the Pendulum study in this file)."""
    if spec.get("task", {}).get("kind") == "pusht":
        from .studies import pusht_surface
        return pusht_surface
    return None


def load_spec(path: Path) -> dict:
    text = Path(path).read_text()  # a .json specification is JSON (YAML 1.1 would read 1e-06 as a string)
    spec = json.loads(text) if Path(path).suffix == ".json" else yaml.safe_load(text)
    if study_of(spec) is not None:
        return study_of(spec).load_spec(spec)
    need = {"study", "task", "data", "split", "real_fractions", "synthetic_doses", "arms", "seeds", "generation",
            "learner", "analysis", "compute"}
    if set(spec) - need - {"dynamics", "surface_seeds"} or need - set(spec):
        raise Refused(f"refused: specification keys {sorted(spec)} differ from {sorted(need)} "
                      "(+ dynamics, surface_seeds)")
    if not set(spec["arms"]) <= set(sup.ARMS) or "bc" not in spec["arms"]:
        raise Refused(f"refused: arms must include bc and be among {sup.ARMS}")
    if "ccil_learned" in spec["arms"] and "dynamics" not in spec:
        raise Refused("refused: ccil_learned needs dynamics settings")
    if any(not 0 < f <= 1 for f in spec["real_fractions"]) or any(isinstance(k, bool) or not isinstance(k, int) or k < 1
                                                                  for k in spec["synthetic_doses"]):
        raise Refused("refused: real fractions must be in (0, 1] and synthetic doses integers >= 1")
    if len(set(spec["seeds"])) != len(spec["seeds"]):
        raise Refused("refused: seeds repeat")
    surface = spec.get("surface_seeds")
    if surface is not None and (not surface or surface != spec["seeds"][:len(surface)]):
        raise Refused("refused: surface_seeds must be a non-empty prefix of seeds")
    if "ccil_exact" in spec["arms"] and "ccil_learned" not in spec["arms"]:
        raise Refused("refused: ccil_exact takes its sigma from the learned arm's selection, so it needs ccil_learned")
    g = spec["generation"]
    grid, fixed = g.get("sigma_grid"), g.get("sigma")
    if (grid is None) == (fixed is None):
        raise Refused("refused: generation declares exactly one of sigma_grid (selected per (r, K, seed)) and sigma "
                      "(fixed per (r, K))")
    def positive(x):
        return not isinstance(x, bool) and isinstance(x, (int, float)) and x > 0
    if grid is not None and (not isinstance(grid, list) or not grid or grid != sorted(set(grid))
                             or not all(map(positive, grid))):
        raise Refused("refused: generation.sigma_grid must be a non-empty increasing list of positive numbers")
    coords = {f"r{r}/k{k}" for r in spec["real_fractions"] for k in spec["synthetic_doses"]}
    if fixed is not None and (not isinstance(fixed, dict) or set(fixed) != coords
                              or not all(map(positive, fixed.values()))):
        raise Refused(f"refused: generation.sigma must give a positive sigma for exactly the coordinates "
                      f"{sorted(coords)}")
    if "action_noise_std" in g or not 0 <= g.get("max_failure", -1) <= 1:
        raise Refused("refused: sigma is declared in normalized units by sigma_grid or sigma (no action_noise_std), "
                      "and generation.max_failure must be in [0, 1]")
    lr = spec["learner"]
    if (not lr.get("patience") or lr.get("checkpoint_selection") != "validation" or lr.get("lr_schedule") != "plateau"
            or not lr.get("step_budget")):
        raise Refused("refused: the learner needs the convergence-driven rule (checkpoint_selection: validation, "
                      "lr_schedule: plateau, patience) and a step_budget, which is only a failure ceiling")
    if set(spec["arms"]) - {"bc"} and lr.get("synthetic_coverage_before_stop") is not True:
        raise Refused("refused: CCIL arms need synthetic_coverage_before_stop, so no unit stops before drawing every "
                      "selected synthetic row")
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
    def __init__(self, spec_path: Path, root: Path, store: Path | None, allow_dirty: bool, options: dict | None = None):
        self.spec_path, self.root = Path(spec_path).resolve(), Path(root).resolve()
        self.spec = load_spec(self.spec_path)
        self.study, self.options = study_of(self.spec), options or {}
        self.task = Task(**self.spec["task"]) if self.study is None else None
        self.runners = RUNNERS if self.study is None else self.study.RUNNERS
        self.device = self.spec["compute"]["device"]
        state = code_state()
        if state["dirty"] and not allow_dirty:
            raise Refused(f"refused: the harness tree is dirty: {state['dirty'][:5]}")
        if not state["ccil"]["commit"] and not allow_dirty:
            raise Refused("refused: ccil is not installed from a pinned commit, so its implementation is unnamed")
        # Every store identity carries the implementation (harness source, ccil commit, lock) and the environment.
        self.scope = {"ilsb_source_sha256": source_sha256(), "ccil": state["ccil"],
                      "uv_lock_sha256": state["uv_lock_sha256"], "environment": environment(self.device)}
        if self.study is not None:
            self.scope |= self.study.scope(self)
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

    def seeds(self, k: int) -> list:
        """The seeds of dose k: every seed at K = 0 and K_max, the surface seeds (a prefix) at the doses between."""
        if k in (0, max(self.spec["synthetic_doses"])):
            return self.spec["seeds"]
        return self.spec.get("surface_seeds") or self.spec["seeds"]

    def dataset(self) -> Path:
        p = Path(self.spec["data"]["path"])
        p = p if p.is_absolute() else self.spec_path.parent / p
        if not p.exists() or sha256_file(p) != self.spec["data"]["sha256"]:
            raise Refused(f"refused: {p} is not the declared dataset {self.spec['data']['sha256']}")
        return p

    def learner_settings(self, arm: str) -> dict:
        s = dict(self.spec["learner"])
        if arm == "bc":
            for k in ("synthetic_fraction", "synthetic_coverage_before_stop"):
                s.pop(k, None)  # BC has no synthetic rows; everything else is identical
        return s

    def generation(self, sigma: float) -> dict:
        """sigma is in normalized action units (a fraction of the action bound); the generator takes action units."""
        g = {k: v for k, v in self.spec["generation"].items() if k not in ("sigma_grid", "sigma", "max_failure")}
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


def design(ctx) -> list[dict]:
    """The study's design, computed from the specification alone: every analysis unit (arm, r, K, seed) with its seed
    role (every seed at K = 0 and K_max, the surface seeds between) and, for a CCIL unit, the K_max candidate set it
    views (one generation per (arm, r, seed), nested in K). Preflight reports it; the plan, the selection, the
    analysis units and the report consume it."""
    k_max = max(ctx.spec["synthetic_doses"])
    return [{"key": f"{a}/r{r}/k{k}/s{s}", "arm": a, "r": r, "k": k, "seed": s,
             "seeds": "all" if k in (0, k_max) else "surface",
             "views": None if a == "bc" else f"{a}/r{r}/s{s}/k{k_max}"}
            for a, r, k in ctx.cells() for s in ctx.seeds(k)]


def plan(ctx, selection: dict | None = None) -> tuple[dict, dict]:
    """The policy groups of the design's units, each trained together, and every candidate set's genuine-failure
    fraction with whether it is feasible (an infeasible sigma is never trained). Without a selection: BC, and the
    learned arm at every feasible sigma of the grid, or at each coordinate's fixed sigma. With one: the exact arm at
    each coordinate's selected sigma."""
    g = ctx.spec["generation"]
    groups, feasibility = {}, {}
    fixed = g.get("sigma")
    units = design(ctx)
    for r, seed in dict.fromkeys((d["r"], d["seed"]) for d in units):  # each (r, seed)'s generations once
        mine = [d for d in units if (d["r"], d["seed"]) == (r, seed)]
        if selection is None:
            for d in mine:
                if d["arm"] == "bc":
                    groups.setdefault(("bc", r, 0, None), []).append(unit(ctx, "bc", r, 0, seed))
            learned = [d["k"] for d in mine if d["arm"] == "ccil_learned"]
            sigmas = g["sigma_grid"] if fixed is None else sorted({fixed[f"r{r}/k{k}"] for k in learned})
            for sigma in sigmas if learned else ():
                full = full_candidates(ctx, "ccil_learned", r, seed, sigma)
                fail = failure(full)
                feasibility[f"ccil_learned/r{r}/s{seed}/sigma{sigma}"] = {
                    "failure": fail, "feasible": fail <= g["max_failure"]}
                if fail > g["max_failure"]:
                    continue
                for k in learned:
                    if fixed is None or fixed[f"r{r}/k{k}"] == sigma:
                        groups.setdefault(("ccil_learned", r, k, sigma), []).append(
                            unit(ctx, "ccil_learned", r, k, seed, sigma, full))
        else:
            for k in [d["k"] for d in mine if d["arm"] == "ccil_exact"]:
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
            "learner": "ilsb.learners.vector.vector_bc_v2"}


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
    """Per (r, K, seed): the learned arm's sigma with the least validation error; ties go to the smaller sigma. With a
    fixed sigma per (r, K), the only candidate is that sigma (none if it was infeasible)."""
    out = {}
    for d in design(ctx):
        if d["arm"] == "ccil_learned":
            r, k, seed = d["r"], d["k"], d["seed"]
            c = {x["sigma"]: x["validation_mse"] for x in rows.values()
                 if x["arm"] == "ccil_learned" and (x["r"], x["k"], x["seed"]) == (r, k, seed)}
            out[f"r{r}/k{k}/s{seed}"] = {"candidates": {str(s_): v for s_, v in sorted(c.items())},
                                         "selected_sigma": min(c, key=lambda s_: (c[s_], s_)) if c else None}
    return out


def analysis_units(ctx, selection: dict) -> dict:
    """Each analysis unit's policy row: the learned arm's is the selected sigma's."""
    out = {}
    for d in design(ctx):
        coord = f"r{d['r']}/k{d['k']}/s{d['seed']}"
        out[d["key"]] = f"{d['key']}/sigma{selection[coord]['selected_sigma']}" if d["arm"] == "ccil_learned" \
            else d["key"]
    return out


# --- stages -----------------------------------------------------------------------------------------------------------
def preflight(ctx) -> dict:
    """The dataset, and the design without training (the units train consumes), with each (r, seed)'s real-data
    subset and split entries."""
    ctx.dataset()
    units = design(ctx)
    plan = [d["key"] for d in units]
    units = {d["key"]: {k_: v for k_, v in d.items() if k_ != "key"} for d in units}
    from ccil.schema import read_trajectories
    real = {}
    for r in ctx.spec["real_fractions"]:
        for s in ctx.spec["seeds"]:
            subset_p, split_p = data_units(ctx, r, s)
            real[f"r{r}/s{s}"] = {"subset_sha256": sha256_file(subset_p), "split_sha256": sha256_file(split_p),
                                  "episodes": read_trajectories(subset_p).meta["subset"]["episodes"]}
    return {"all_pass": True, "dataset_sha256": ctx.spec["data"]["sha256"], "planned_units": len(plan),
            "plan_sha256": digest(plan), "units": units, "real_data": real, "code": ctx.protocol["code"]}


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
    # Every unit of the design has its policy row; a plan that left one out fails here rather than in a later stage.
    analysis = analysis_units(ctx, selection) if not unselectable else {}
    untrained = sorted(k_ for k_, row in analysis.items() if row not in rows)
    ok = not incomplete and not unselectable and not exact_infeasible and not untrained
    return {"all_pass": ok, "units": rows, "trained": sum(v["reused"] is False for v in rows.values()),
            "feasibility": feasibility, "selection": selection, "unselectable": unselectable,
            "exact_feasibility": exact_feasibility, "exact_infeasible": exact_infeasible, "incomplete_fit": incomplete,
            "untrained": untrained, "analysis": analysis if ok else None}


def audit(ctx) -> dict:
    """Provenance and selection, recomputed. Every trained unit is a verified store entry, trained on exactly this
    unit's supervision (its training identity, which hashes the supervision manifest with its dataset, split and
    candidate hashes, recomputes from the unit's own inputs), with the weights its record describes and the row the
    train stage reported. Feasibility, the sigma selection and the analysis units recompute to the reported ones."""
    from .learners.vector import vector_bc_v2 as v2
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

    units = design(ctx)

    def values(arm, r, k):
        return {d["seed"]: scores[d["key"]]["scores"][a["metric"]] for d in units
                if (d["arm"], d["r"], d["k"]) == (arm, r, k)}
    cells, contrasts, derived = {}, {}, {}
    for arm, r, k in dict.fromkeys((d["arm"], d["r"], d["k"]) for d in units):  # the response surface, descriptive:
        v = values(arm, r, k)                                                   # each cell over its dose's seeds
        cells[f"{arm}/r{r}/k{k}"] = {"mean": float(np.mean(list(v.values()))), "n": len(v), "per_seed": v}
    k_max = max(ctx.spec["synthetic_doses"])
    # The primary contrasts and the derived ones, at K_max over every seed; derived contrasts are not a test family.
    for out, pairs in ((contrasts, a["contrasts"]), (derived, a.get("derived", []))):
        for A, B in pairs:
            for r in ctx.spec["real_fractions"]:
                out[f"{A}-{B}/r{r}/k{k_max}"] = analysis.paired(values(A, r, k_max),
                                                                values(B, r, 0 if B == "bc" else k_max),
                                                                a["confidence"], a["min_pairs"])
    sel = prior["train"]["selection"]
    chosen = {}
    for d in units if sel else ():
        if d["arm"] == "ccil_learned":
            cell = f"r{d['r']}/k{d['k']}"
            chosen.setdefault(cell, []).append(sel[f"{cell}/s{d['seed']}"]["selected_sigma"])
    chosen = {c: sorted(v) for c, v in chosen.items()}
    return {"all_pass": True, "metric": a["metric"], "cells": cells, "contrasts": contrasts, "derived": derived,
            "selected_sigmas": chosen}


RUNNERS = {"preflight": preflight, "train": train, "audit": audit, "evaluate": evaluate,
           "evaluation_audit": evaluation_audit, "report": report}


def run_stage(ctx, stage: str) -> dict:
    require(ctx, stage)
    try:
        rep = ctx.runners[stage](ctx)
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
        p.add_argument("--gpus", default="", help="PushT train stage: comma-separated GPU slots, an id once per "
                                                  "concurrent unit (0,0,1,1); empty collects units trained elsewhere")
    un = sub_.add_parser("unit", help="PushT: train one unit of the design (a Slurm array task)")
    un.add_argument("spec", type=Path)
    un.add_argument("--root", type=Path, required=True)
    un.add_argument("--store", type=Path)
    un.add_argument("--key", required=True)
    un.add_argument("--gpu")
    un.add_argument("--allow-dirty", action="store_true")
    pol = sub_.add_parser("policy").add_subparsers(dest="policy_cmd", required=True)
    pol.add_parser("train", help="one vector policy on CCIL supervision (moved from ccil policy train)",
                   description=policy_cmd.__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
                   ).add_argument("config", type=Path)
    a = ap.parse_args(argv)
    if a.cmd == "policy":
        return policy_cmd.main(a.config)
    if a.cmd == "unit":
        ctx = Context(a.spec, a.root, a.store, a.allow_dirty)
        if ctx.study is None:
            raise Refused("refused: ilsb unit is for the PushT study")
        return ctx.study.train_unit(ctx, a.key, a.gpu)
    ctx = Context(a.spec, a.root, a.store, a.allow_dirty, {"gpus": [g for g in a.gpus.split(",") if g]})
    if a.cmd == "run":
        done = run_all(ctx)
        print(json.dumps(done))
        return 0 if all(v != "FAIL" for v in done.values()) and len(done) == len(STAGES) else 1
    rep = run_stage(ctx, a.name)
    print(json.dumps({"stage": a.name, "all_pass": rep["all_pass"]}))
    return 0 if rep["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
