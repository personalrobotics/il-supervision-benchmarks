"""The PushT surface study (ilsb#5): CCIL against source-matched duplication over real-data fraction r and synthetic
minibatch fraction alpha, with the IBC learner of PushT v2 (docs/studies/pusht-surface.md is the protocol).

The harness owns the data roles, the nested real-data subsets, the frozen normalization, the CCIL label pools and
their structural checks, the design, the store, the audit and the report; it runs on the CPU. Policy training,
held-out selection and closed-loop evaluation run in a LeRobot 0.6.1 environment (ILSB_LEROBOT_PYTHON) with the
plugins and the learner of tools/pusht_v2 installed (tools/pusht_v2/surface_learner); that environment, the installed
plugin sources and the GPU are part of every store identity. ILSB_PUSHT_DATASET is the lerobot/pusht_keypoints
directory the source rows are built from (tools/pusht_v2/source.py), checked against the declared content digest.

Stages: preflight (source, roles, subsets, normalization, pools and their checks, the design: everything before any
policy trains), train (each unit's 20k -> 40k -> 80k life, by the bounded local scheduler or, with --workers 0, only
collected from units trained elsewhere, e.g. a Slurm array of `ilsb unit`), audit (selection, extension, exposure and
pairing recomputed; selected checkpoints re-scored), evaluate (lerobot-eval on the declared starts), evaluation_audit
(scores recomputed, one unit re-evaluated), report (the cells and the stage's frozen rule).
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

from ..store import StoreError, digest, sha256_file, write_json

REPO = Path(__file__).resolve().parents[3]
TOOLS = REPO / "tools" / "pusht_v2"
ARMS = ("bc", "duplication", "ccil")
FEATURES = ("observation.state", "observation.environment_state", "action")
PARTS = ("state", "env", "action")
EPS32 = float(np.finfo(np.float32).eps)
NEW = np.r_[2:4, 7:10]  # newest slots of CCIL's coordinates [agent_{t-1}, agent_t, T_{t-1}, T_t]
LOCAL_PACKAGES = {"lerobot_policy_ibc": TOOLS / "ibc_plugin" / "lerobot_policy_ibc",
                  "lerobot_env_pusht_history": TOOLS / "env_plugin" / "lerobot_env_pusht_history.py",
                  "pusht_surface_learner": TOOLS / "surface_learner" / "pusht_surface_learner"}


class Refused(SystemExit):
    pass


def _pose():
    spec = importlib.util.spec_from_file_location("pusht_pose", TOOLS / "pose.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _source_module():
    spec = importlib.util.spec_from_file_location("pusht_source", TOOLS / "source.py")
    mod = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(TOOLS))
    spec.loader.exec_module(mod)
    return mod


def tree_sha256(path: Path) -> str:
    """A source file or package by content (its .py files, relative names), as the scope names it."""
    path = Path(path)
    files = [path] if path.is_file() else sorted(path.rglob("*.py"))
    h = hashlib.sha256()
    for f in files:
        h.update((f.name if path.is_file() else f.relative_to(path).as_posix()).encode() + b"\0" + f.read_bytes() + b"\0")
    return h.hexdigest()


# --- specification ----------------------------------------------------------------------------------------------------
NEED = {"study", "task", "stage", "data", "roles", "real_fractions", "subset_seed", "synthetic_fractions", "arms",
        "seeds", "generation", "learner", "evaluation", "analysis", "expected", "compute"}


def load_spec(spec: dict) -> dict:
    if set(spec) - NEED - {"cells"} or NEED - set(spec):
        raise Refused(f"refused: specification keys {sorted(spec)} differ from {sorted(NEED)} (+ cells)")
    if spec["arms"] != list(ARMS):
        raise Refused(f"refused: the arms are {list(ARMS)}")
    tr, ho = spec["roles"]["train"], spec["roles"]["held_out"]
    if len(set(tr)) != len(tr) or len(set(ho)) != len(ho) or set(tr) & set(ho):
        raise Refused("refused: the training pool and the held-out episodes must be disjoint lists of distinct ids")
    if any(not 0 < r <= 1 for r in spec["real_fractions"]) or any(not 0 < a < 1 for a in spec["synthetic_fractions"]):
        raise Refused("refused: real fractions in (0, 1], synthetic fractions in (0, 1)")
    if len(set(spec["seeds"])) != len(spec["seeds"]):
        raise Refused("refused: seeds repeat")
    for c in spec.get("cells") or []:
        if c[0] not in spec["real_fractions"] or c[1] not in spec["synthetic_fractions"]:
            raise Refused(f"refused: cell {c} is not a declared (r, alpha)")
    lr = spec["learner"]
    b = lr["boundaries"]
    if b != sorted(b) or b[-1] != lr["horizon"] or any(x % lr["cadence"] for x in b):
        raise Refused("refused: boundaries must increase to the horizon, each a multiple of the cadence")
    if lr.get("stop_at") is not None and not str(spec["study"]).endswith("-timing"):
        raise Refused("refused: learner.stop_at (a fixed step count) is for the timing measurement only")
    return spec


def runtime_python() -> str:
    py = os.environ.get("ILSB_LEROBOT_PYTHON")
    if not py or not Path(py).exists():
        raise Refused("refused: set ILSB_LEROBOT_PYTHON to the LeRobot 0.6.1 environment's python")
    return py


def scope(ctx) -> dict:
    """The learner environment: its distributions (versions, local packages by installed content, which must be this
    tree's), Python and GPU. Folded into every store identity, with the harness's own scope."""
    probe = ("import importlib.metadata as m, importlib.util as u, json, sys, torch\n"
             "d = sorted(f'{x.metadata[\"Name\"]}=={x.version}' for x in m.distributions())\n"
             "loc = {n: (u.find_spec(n).submodule_search_locations or [u.find_spec(n).origin])[0] for n in sys.argv[1:]}\n"
             "g = torch.cuda.get_device_name() if torch.cuda.is_available() else None\n"
             "print(json.dumps({'distributions': d, 'python': sys.version.split()[0], 'torch': torch.__version__,"
             " 'cuda': torch.version.cuda, 'gpu': g, 'local': loc}))")
    r = subprocess.run([runtime_python(), "-c", probe, *LOCAL_PACKAGES], capture_output=True, text=True)
    if r.returncode:
        raise Refused(f"refused: the LeRobot environment did not answer: {r.stderr[-800:]}")
    env = json.loads(r.stdout)
    installed = {n: tree_sha256(Path(p)) for n, p in env.pop("local").items()}
    repo = {n: tree_sha256(p) for n, p in LOCAL_PACKAGES.items()}
    if installed != repo:
        raise Refused(f"refused: the installed plugins or learner differ from this tree's: "
                      f"{sorted(n for n in repo if installed.get(n) != repo[n])}")
    if not env["gpu"]:
        raise Refused("refused: the LeRobot environment sees no GPU")
    return {"lerobot_env": {**env, "distributions_sha256": digest(env.pop("distributions"))},
            "tools_sha256": {"local_packages": repo, "pose.py": tree_sha256(TOOLS / "pose.py"),
                             "source.py": tree_sha256(TOOLS / "source.py")}}


# --- data roles -------------------------------------------------------------------------------------------------------
def source(ctx) -> dict:
    """The derived history rows of every episode (tools/pusht_v2/source.py), checked by content."""
    p = ctx.root / "source.npz"
    if not p.exists():
        ds = os.environ.get("ILSB_PUSHT_DATASET")
        if not ds:
            raise Refused("refused: set ILSB_PUSHT_DATASET to the lerobot/pusht_keypoints directory")
        arrays = _source_module().build(Path(ds))
        tmp = p.with_name(".source.tmp.npz")
        np.savez(tmp, **arrays)
        os.replace(tmp, p)
    with np.load(p) as z:
        arrays = {k: z[k] for k in z.files}
    got = _source_module().content_sha256(arrays)
    if got != ctx.spec["data"]["content_sha256"]:
        raise Refused(f"refused: {p} has content {got}, not the declared {ctx.spec['data']['content_sha256']}")
    return arrays


def roles(ctx, src: dict) -> dict:
    eps = sorted(set(src["episode"].tolist()))
    tr, ho = sorted(ctx.spec["roles"]["train"]), sorted(ctx.spec["roles"]["held_out"])
    if sorted(tr + ho) != eps:
        raise Refused("refused: the training pool and held-out episodes do not partition the source's episodes")
    return {"train": tr, "held_out": ho}


def subset_episodes(train: list, r: float, seed: int) -> list:
    """The first round-half-up(r N) episodes of one permutation of the training pool: nested in r for a seed."""
    perm = np.random.default_rng(np.random.SeedSequence([seed, 9])).permutation(len(train))
    n = int(np.floor(r * len(train) + 0.5))
    return sorted(int(train[i]) for i in perm[:n])


def rows_of(src: dict, episodes) -> np.ndarray:
    """Source row indices of the episodes, in source order (episode, frame)."""
    return np.flatnonzero(np.isin(src["episode"], list(episodes)))


def current_frames(state: np.ndarray, env: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(np.concatenate([state[:, 2:4], env[:, 16:32]], 1).astype(np.float32))


# --- units before training --------------------------------------------------------------------------------------------
def stats_entry(ctx, src, eps) -> Path:
    """The frozen min/max of the subset's real rows: every arm at this subset normalizes with it."""
    rows = rows_of(src, eps)

    def write(d):
        out = {}
        for name, part in zip(FEATURES, PARTS):
            x = src[part][rows].astype(np.float32)
            out[name] = {"min": x.min(0).tolist(), "max": x.max(0).tolist()}
        (d / "stats.json").write_text(json.dumps(out, sort_keys=True) + "\n")
    return ctx.store.ensure("stats", {"kind": "stats", "source": ctx.spec["data"]["content_sha256"],
                                      "episodes": eps, "rule": "min/max over the subset's real rows"}, write)[0]


def pool_entry(ctx, src, eps, held) -> Path:
    """CCIL's labels for one subset: the pinned CCIL fits its dynamics on the subset's episodes alone and generates
    K = 4 proposals per transition (generation.config), every accepted label curated with weight 1; then every selected
    label is checked (residual, history invariant, rigidity, containment) and decoded onto the policy's keypoint
    history. One violation, a yield below the floor or a pool indistinguishable from its duplication view fails."""
    g = ctx.spec["generation"]
    cfg_path = TOOLS / g["config"]
    ident = {"kind": "pool", "source": ctx.spec["data"]["content_sha256"], "episodes": eps,
             "config_sha256": sha256_file(cfg_path), "invariant": g["invariant"], "held_out": held}
    return ctx.store.ensure("pool", ident, lambda d: _make_pool(ctx, src, eps, held, cfg_path, d))[0]


def _make_pool(ctx, src, eps, held, cfg_path, d: Path):
    from ccil.models.mlp import MLPTransitionModel
    from ccil.schema import Episode, TrajectoryDataset, read_candidates, read_selection, read_trajectories, \
        write_trajectories
    from scipy.spatial import cKDTree
    pose, inv = _pose(), ctx.spec["generation"]["invariant"]
    work = ctx.root / "work" / f"pool-{digest(eps)[:16]}-{int(time.time())}"
    work.mkdir(parents=True)
    fps = float(src["fps"])
    episodes = []
    for e in eps:
        r = rows_of(src, [e])
        episodes.append(Episode(str(e), {"state": src["state"][r], "state.pose": src["pose"][r]}, src["action"][r],
                                timestamps=np.arange(len(r)) / fps))
    write_trajectories(TrajectoryDataset(episodes), work / "pose.npz")
    (work / "ccil.yaml").write_text(cfg_path.read_text())
    for args in (["generate", "--config", "ccil.yaml"], ["curate", "runs/generate-result.json"]):
        r = subprocess.run([sys.executable, "-m", "ccil", *args], cwd=work, capture_output=True, text=True)
        (work / f"{args[0]}.log").write_text(r.stdout + r.stderr)
        if r.returncode:
            raise StoreError(f"ccil {args[0]} exited {r.returncode}: {r.stderr[-1200:]}")
    runs = work / "runs"
    one = lambda pat: next(runs.glob(pat))  # noqa: E731
    stage_cfg = {s: json.loads(one(f"{s}-*/config.json").read_text())["values"]
                 for s in ("transition-fit", "labels-generate")}
    made = {"fit_episodes": stage_cfg["transition-fit"]["transition_model.fit_episodes"]["value"],
            "structure": stage_cfg["transition-fit"]["transition_model.structure"]["value"],
            "modality": stage_cfg["transition-fit"]["representation.modality"]["value"],
            "method": stage_cfg["labels-generate"]["generation.method"]["value"]}
    if made != {"fit_episodes": "all", "structure": "history_shift", "modality": "state+state.pose",
                "method": "noisy_action_fe"}:
        raise StoreError(f"the labels were not made as the protocol fixes: {made}")
    ds = read_trajectories(one("data-prepare-*") / "data" / "dataset.npz")
    if sorted(int(e.episode_id) for e in ds.episodes) != eps:
        raise StoreError("CCIL read episodes other than the subset's")
    c = read_candidates(one("labels-generate-*") / "labels" / "candidates.npz", ds)
    sel = read_selection(one("labels-curate-*") / "labels" / "selection.npz")
    model = MLPTransitionModel.load(one("transition-fit-*") / "checkpoints" / "transition_model.npz")
    mask = np.asarray(sel.mask, bool)
    if not np.array_equal(mask, np.asarray(c.accepted, bool)) or (np.asarray(sel.weight)[mask] != 1).any():
        raise StoreError("the curation is not every accepted label with weight 1")
    real = rows_of(src, eps)
    transitions = len(real) - len(eps)
    src_ep = np.asarray(c.source_episode_id).astype(np.int64)
    if len(mask) != 4 * transitions or not np.isin(src_ep, eps).all():
        raise StoreError(f"{len(mask)} proposals, expected 4 x {transitions} transitions, all from the subset")
    rows = np.flatnonzero(mask)
    names = ("state", "state.pose")
    z = np.concatenate([np.asarray(c.observations[n], np.float64) for n in names], 1)[rows]
    tgt = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in names], 1)[rows]
    act = np.asarray(c.actions)[rows]
    sfr = np.asarray(c.source_frame).astype(np.int64)[rows]
    key = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(src["episode"], src["frame"]))}
    src_row = np.array([key[(int(e), int(f))] for e, f in zip(src_ep[rows], sfr)], np.int64)
    q_src = np.concatenate([src["state"][src_row], src["pose"][src_row]], 1).astype(np.float64)
    tol = inv["root_tolerance"]
    resid = np.linalg.norm(z + model.delta(z, act.astype(np.float64)) - tgt, axis=1)
    err = np.abs(z[:, NEW] - q_src[:, NEW]).max(1)
    eps_ = np.maximum(10 * tol, 10 * EPS32 * np.maximum(1.0, np.abs(q_src[:, NEW]).max(1)))
    kp_prev, kp_cur = pose.keypoints(z[:, 4:7]), pose.keypoints(z[:, 7:10])
    ref = np.asarray(inv["rigidity"]["reference_pair_distances_px"])
    rig = np.maximum(np.abs(_pairs(kp_prev) - ref).max(1), np.abs(_pairs(kp_cur) - ref).max(1))
    bad = {"residual": resid > tol, "invariant": err > eps_, "rigidity": rig > inv["rigidity"]["tolerance_px"]}
    # decoded onto the policy's keypoint history; the current halves are the source frame's own
    drift = max(float(np.abs(src["env"][src_row][:, 16:] - kp_cur).max()) if len(rows) else 0.0,
                float(np.abs(src["state"][src_row][:, 2:] - z[:, 2:4]).max()) if len(rows) else 0.0)
    syn_state = np.concatenate([z[:, 0:2], src["state"][src_row][:, 2:4]], 1).astype(np.float32)
    syn_env = np.concatenate([kp_prev, src["env"][src_row][:, 16:32]], 1).astype(np.float32)
    syn_action = act.astype(np.float32)
    # containment, by provenance (above) and by content: no label's current frame is a held-out frame's
    held_frames = {x.tobytes() for x in current_frames(src["state"], src["env"])[rows_of(src, held)]}
    hits = int(sum(x.tobytes() in held_frames for x in current_frames(syn_state, syn_env)))
    # what a label is: its previous slot's physical displacement from its source's (px), its distance to the
    # nearest real row of the subset in the policy's observation, against the source's own distance (0)
    disp = np.maximum(np.abs(syn_state[:, 0:2] - src["state"][src_row][:, 0:2]).max(1),
                      np.abs(syn_env[:, 0:16] - src["env"][src_row][:, 0:16]).max(1))
    obs_real = np.concatenate([src["state"][real], src["env"][real]], 1).astype(np.float64)
    nearest = cKDTree(obs_real).query(np.concatenate([syn_state, syn_env], 1).astype(np.float64))[0]
    q = lambda x: {str(p): float(np.percentile(x, p)) for p in (10, 50, 90, 100)} if len(x) else None  # noqa: E731
    stats = json.loads((stats_entry(ctx, src, eps) / "stats.json").read_text())
    outside = {}
    for name, x in zip(FEATURES, (syn_state, syn_env, syn_action)):
        lo, hi = np.asarray(stats[name]["min"]), np.asarray(stats[name]["max"])
        outside[name] = float(((x < lo) | (x > hi)).any(1).mean()) if len(x) else 0.0
    counts = json.loads((runs / "generate-result.json").read_text())["counts"]
    rec = {"episodes": len(eps), "real_rows": int(len(real)), "transitions": int(transitions),
           "proposals": int(len(mask)), "selected": int(len(rows)), "usable_fraction": float(len(rows) / len(mask)),
           "ccil_counts": counts, "made": made, "violations": {k: int(v.sum()) for k, v in bad.items()},
           "decode_drift_px": drift, "held_out_current_frames": hits,
           "residual": q(resid), "invariant_error": q(err), "rigidity_error_px": q(rig),
           "previous_slot_displacement_px": q(disp), "nearest_real_distance": q(nearest),
           "outside_frozen_range_fraction": outside,
           "model_fit": json.loads(one("transition-fit-*/metrics.json").read_text())}
    problems = [k for k, v in rec["violations"].items() if v]
    if drift > inv["round_trip_tolerance_px"]:
        problems.append(f"decode drift {drift}")
    if hits:
        problems.append(f"{hits} labels carry a held-out current frame")
    if rec["usable_fraction"] < inv["min_usable_fraction"]:
        problems.append(f"usable fraction {rec['usable_fraction']:.4f}")
    if not len(rows) or np.median(disp) < ctx.spec["generation"]["min_label_displacement_px"]:
        problems.append("the labels do not differ meaningfully from their duplication view")
    rec["problems"] = problems
    (d / "labels.json").write_text(json.dumps(rec, indent=1, sort_keys=True, default=float) + "\n")
    if problems:
        write_json(ctx.root / f"pool-failure-{int(time.time())}.json", rec)
        raise StoreError(f"the pool of {len(eps)} episodes fails its checks: {problems}")
    np.savez(d / "labels.npz", syn_state=syn_state, syn_env=syn_env, syn_action=syn_action, src_row=src_row,
             candidate_row=rows.astype(np.int64))
    for f in ("generate-result.json",):
        (d / f).write_bytes((runs / f).read_bytes())


def _pairs(kp16: np.ndarray) -> np.ndarray:
    p = kp16.reshape(len(kp16), 8, 2)
    i, j = np.triu_indices(8, 1)
    return np.linalg.norm(p[:, i] - p[:, j], axis=-1)


# --- design -----------------------------------------------------------------------------------------------------------
def cells(spec) -> list:
    """(r, alpha) coordinates of the treatment arms: the declared cells, or the full grid."""
    if spec.get("cells"):
        return [tuple(c) for c in spec["cells"]]
    return [(r, a) for r in spec["real_fractions"] for a in spec["synthetic_fractions"]]


def design(ctx) -> list[dict]:
    """Every policy unit, from the specification alone: BC at alpha 0 once per (r, seed) of a used r, and duplication
    and CCIL at every cell."""
    cs = cells(ctx.spec)
    rs = [r for r in ctx.spec["real_fractions"] if any(c[0] == r for c in cs)]
    out = []
    for s in ctx.spec["seeds"]:
        for r in rs:
            out.append({"key": f"bc/r{r}/a0/s{s}", "arm": "bc", "r": r, "alpha": 0.0, "seed": s})
            for (rr, a) in cs:
                if rr == r:
                    out += [{"key": f"{arm}/r{r}/a{a}/s{s}", "arm": arm, "r": r, "alpha": a, "seed": s}
                            for arm in ("duplication", "ccil")]
    return out


def data(ctx) -> dict:
    """Source, roles, and per r its subset, normalization and (if any cell uses r) pool entries."""
    src = source(ctx)
    ro = roles(ctx, src)
    out = {"src": src, "roles": ro, "r": {}}
    used = {d["r"] for d in design(ctx) if d["arm"] != "bc"}
    for r in ctx.spec["real_fractions"]:
        eps = subset_episodes(ro["train"], r, ctx.spec["subset_seed"])
        out["r"][r] = {"episodes": eps, "stats": stats_entry(ctx, src, eps),
                       "pool": pool_entry(ctx, src, eps, ro["held_out"]) if r in used else None}
    return out


def features_root(ctx, src) -> Path:
    """A one-episode LeRobot dataset with the history features: the learner's make_policy reads its feature shapes
    (never its rows or statistics). Derived from the source, so it is rewritten identically when absent."""
    from ccil.lerobot import write_dataset
    from ccil.schema import Episode
    root = ctx.root / "features"
    if not (root / "meta" / "info.json").exists():
        r = rows_of(src, [int(src["episode"][0])])
        ep = Episode("0", {"state": src["state"][r], "state.environment_state": src["env"][r]}, src["action"][r],
                     timestamps=np.arange(len(r)) / float(src["fps"]), task="Push the T-shaped block onto the target.")
        tmp = ctx.root / f".features-{os.getpid()}"
        write_dataset(iter([ep]), tmp, fps=float(src["fps"]))
        os.replace(tmp, root)
    return root


def unit_identity(ctx, u, dd) -> dict:
    d = dd["r"][u["r"]]
    return {"kind": "policy", "arm": u["arm"], "r": u["r"], "alpha": u["alpha"], "seed": u["seed"],
            "episodes": d["episodes"], "held_out": dd["roles"]["held_out"],
            "stats": sha256_file(d["stats"] / "stats.json"),
            "pool": None if u["arm"] == "bc" else sha256_file(d["pool"] / "labels.npz"),
            "learner": ctx.spec["learner"]}


def unit_files(ctx, u, dd, work: Path) -> None:
    """unit.json, rows.npz and stats.json of a unit's work directory (deterministic; written once)."""
    if (work / "unit.json").exists():
        return
    work.mkdir(parents=True, exist_ok=True)
    src, d = dd["src"], dd["r"][u["r"]]
    real, val = rows_of(src, d["episodes"]), rows_of(src, dd["roles"]["held_out"])
    arr = {f"real_{p}": src[p][real] for p in PARTS} | {f"val_{p}": src[p][val] for p in PARTS}
    if u["arm"] == "bc":
        arr |= {"syn_state": np.zeros((0, 4), np.float32), "syn_env": np.zeros((0, 32), np.float32),
                "syn_action": np.zeros((0, 2), np.float32)}
    else:
        with np.load(d["pool"] / "labels.npz") as z:
            if u["arm"] == "ccil":
                arr |= {f"syn_{p}": z[f"syn_{p}"] for p in PARTS}
            else:  # duplication: each label's source row, in the labels' order
                arr |= {f"syn_{p}": src[p][z["src_row"]] for p in PARTS}
    np.savez(work / "rows.npz", **arr)
    (work / "stats.json").write_bytes((d["stats"] / "stats.json").read_bytes())
    lr = ctx.spec["learner"]
    write_json(work / "unit.json", {"seed": u["seed"], "alpha": u["alpha"], "batch_size": lr["batch_size"],
                                    "horizon": lr["horizon"], "boundaries": lr["boundaries"], "cadence": lr["cadence"],
                                    "log_every": lr["log_every"], "policy": lr["policy"],
                                    "features_root": str(features_root(ctx, src)), "rows": "rows.npz",
                                    "stats": "stats.json", "selection": lr["selection"], "order": None,
                                    "stop_at": lr.get("stop_at")})


def unit_dir(ctx, ident) -> Path:
    return ctx.root / "work" / digest(ctx.store.full(ident))[:24]


def put_unit(ctx, ident, work: Path) -> Path:
    """The trained unit as a store entry: the result record and the selected checkpoint's files (a LeRobot
    pretrained_model directory that lerobot-eval loads as is)."""
    def write(d):
        (d / "result.json").write_bytes((work / "result.json").read_bytes())
        for f in sorted((work / "selected" / "pretrained_model").iterdir()):
            (d / f.name).write_bytes(f.read_bytes())
    return ctx.store.put("policy", ident, write)


def train_unit(ctx, key: str, gpu: str | None = None) -> int:
    """One unit, in this process's child (for a Slurm array task: `ilsb unit SPEC --root R --key KEY`)."""
    dd = data(ctx)
    u = next(x for x in design(ctx) if x["key"] == key)
    ident = unit_identity(ctx, u, dd)
    if ctx.store.get("policy", ident) is not None:
        return 0
    work = unit_dir(ctx, ident)
    unit_files(ctx, u, dd, work)
    env = os.environ | ({"CUDA_VISIBLE_DEVICES": gpu} if gpu is not None else {})
    with open(work / "learner.log", "a") as log:
        rc = subprocess.run([runtime_python(), "-m", "pusht_surface_learner", "train", str(work)], env=env,
                            stdout=log, stderr=subprocess.STDOUT).returncode
    if rc == 0:
        put_unit(ctx, ident, work)
    return rc


# --- stages -----------------------------------------------------------------------------------------------------------
def preflight(ctx) -> dict:
    """Everything before any policy trains: the source by content, the roles, each r's nested subset, frozen
    normalization and CCIL pool (with its checks), the design and every unit's identity; and the counts the
    protocol implies, asserted exactly."""
    dd = data(ctx)
    src, ro, exp = dd["src"], dd["roles"], ctx.spec["expected"]
    units = design(ctx)
    problems = []
    subsets = {f"r{r}": v["episodes"] for r, v in dd["r"].items()}
    sizes = {k: len(v) for k, v in subsets.items()}
    if sizes != exp["subset_episodes"]:
        problems.append(f"subset sizes {sizes} differ from {exp['subset_episodes']}")
    rs = sorted(dd["r"])
    for a, b in zip(rs, rs[1:]):
        if not set(dd["r"][a]["episodes"]) <= set(dd["r"][b]["episodes"]):
            problems.append(f"r{a} is not nested in r{b}")
    if any(set(v["episodes"]) & set(ro["held_out"]) for v in dd["r"].values()):
        problems.append("a subset holds a held-out episode")
    val = len(rows_of(src, ro["held_out"]))
    if val != exp["held_out_frames"]:
        problems.append(f"{val} held-out frames, expected {exp['held_out_frames']}")
    if len(units) != exp["units"]:
        problems.append(f"{len(units)} units, expected {exp['units']}")
    features_root(ctx, src)
    pools = {f"r{r}": json.loads((v["pool"] / "labels.json").read_text()) for r, v in dd["r"].items() if v["pool"]}
    idents = {u["key"]: digest(ctx.store.full(unit_identity(ctx, u, dd))) for u in units}
    return {"all_pass": not problems, "problems": problems, "source_content_sha256": ctx.spec["data"]["content_sha256"],
            "subsets": subsets, "real_rows": {f"r{r}": int(len(rows_of(src, v["episodes"]))) for r, v in dd["r"].items()},
            "held_out_frames": val, "pools": pools, "planned_units": len(units), "units": idents,
            "plan_sha256": digest([u["key"] for u in units]), "code": ctx.protocol["code"]}


def row(ctx, u, entry: Path, reused: bool) -> dict:
    r = json.loads((entry / "result.json").read_text())
    return {"entry": entry.name, "arm": u["arm"], "r": u["r"], "alpha": u["alpha"], "seed": u["seed"],
            "outcome": r["outcome"], "reached": r["reached"], "selected_step": r["selected_step"],
            "selected_infonce": r["selected_infonce"], "per_batch": r["per_batch"], "rows": r["rows"],
            "exposure": r["exposure"], "reused": reused}


def schedule(ctx, todo: list, gpus: list, dirs: dict) -> dict:
    """Run units as child processes, one per slot (a GPU id may repeat), watching every child's exit and heartbeat.
    A child that dies or stalls stops the run promptly: the others are terminated (their work directories resume),
    and the failure is reported with its unit. Progress is logged at every change and at least every five minutes."""
    stall = 60 * ctx.spec["compute"]["stall_minutes"]
    queue, running, done, failed = list(todo), {}, [], {}
    plog = ctx.root / "progress.log"

    def note(msg):
        with open(plog, "a") as f:
            f.write(f"{time.strftime('%Y-%m-%dT%H:%M:%S')} {msg}\n")
    last_note = 0.0
    while queue or running:
        free_slots = []
        for g in dict.fromkeys(gpus):
            free_slots += [g] * (gpus.count(g) - sum(1 for v in running.values() if v["gpu"] == g))
        while queue and free_slots and not failed:
            key = queue.pop(0)
            g = free_slots.pop(0)
            p = subprocess.Popen([sys.executable, "-m", "ilsb", "unit", str(ctx.spec_path), "--root", str(ctx.root),
                                  "--store", str(ctx.store.root), "--key", key, "--gpu", g]
                                 + (["--allow-dirty"] if ctx.protocol["allow_dirty"] else []),
                                 stdout=subprocess.DEVNULL, stderr=open(ctx.root / "work" / "scheduler.err", "a"),
                                 start_new_session=True)  # its own process group: stopping it stops its learner
            running[key] = {"proc": p, "gpu": g, "start": time.time()}
            note(f"start {key} on gpu {g} (pid {p.pid})")
        time.sleep(5)
        for key, v in list(running.items()):
            rc = v["proc"].poll()
            if rc is not None:
                running.pop(key)
                (done.append(key) if rc == 0 else failed.__setitem__(key, f"exit {rc}"))
                note(f"{'done' if rc == 0 else 'FAILED'} {key} (exit {rc})")
            else:
                hb = _heartbeat(dirs[key])
                if time.time() - max(v["start"], hb) > stall:
                    failed[key] = f"no heartbeat for {stall} s"
                    os.killpg(v["proc"].pid, signal.SIGKILL)
                    note(f"STALLED {key}: killed")
        if failed and running:
            for key, v in running.items():
                os.killpg(v["proc"].pid, signal.SIGTERM)
                note(f"stopping {key}: a sibling failed (its work directory resumes)")
            for v in running.values():
                v["proc"].wait()
            running.clear()
        if failed and not running:
            note(f"stopped: failed {sorted(failed)}; {len(queue)} units not started")
            break
        if time.time() - last_note > 300:
            note(f"queued {len(queue)} running {len(running)} done {len(done)} failed {len(failed)} "
                 f"gpu {_gpu_memory()}")
            last_note = time.time()
    return failed


def _heartbeat(work: Path) -> float:
    """The unit's latest sign of life: its heartbeat (every 200 steps) or progress log (every checkpoint)."""
    times = [p.stat().st_mtime for p in (work / "work" / "heartbeat.json", work / "work" / "progress.log") if p.exists()]
    return max(times) if times else 0.0


def _gpu_memory() -> str:
    r = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader"], capture_output=True,
                       text=True)
    return r.stdout.strip().replace("\n", "; ") if r.returncode == 0 else "n/a"


def train(ctx) -> dict:
    dd = data(ctx)
    units = design(ctx)
    idents = {u["key"]: unit_identity(ctx, u, dd) for u in units}
    todo = [k for k, i in idents.items() if ctx.store.get("policy", i) is None]
    failed = {}
    gpus = ctx.options.get("gpus") or []
    if todo and gpus:
        failed = schedule(ctx, todo, gpus, {k: unit_dir(ctx, i) for k, i in idents.items()})
    rows, missing = {}, []
    for u in units:
        e = ctx.store.get("policy", unit_identity(ctx, u, dd))
        if e is None:
            missing.append(u["key"])
        else:
            rows[u["key"]] = row(ctx, u, e, u["key"] not in todo)
    incomplete = sorted(k for k, v in rows.items() if v["outcome"] == "incomplete_fit")
    ok = not failed and not missing
    return {"all_pass": ok, "units": rows, "trained": sum(not v["reused"] for v in rows.values()),
            "failed": failed, "missing": missing, "incomplete_fit": incomplete}


# The audit's own implementation of the sampler rule (independent of the learner's, so it checks the records).
def stream_digests(seed, n_real, b_real, n_syn, b_syn, segments) -> dict:
    """Each segment's index digest under the declared rule: stream s of n rows is the concatenation of its epochs'
    permutations (SeedSequence([seed, s, epoch])); step t takes positions [t b, (t + 1) b)."""
    def stream(s, n, b, start, stop):
        if b == 0:
            return np.zeros((stop - start, 0), np.int64)
        pos = np.arange(start * b, stop * b, dtype=np.int64)
        out = np.empty(len(pos), np.int64)
        for e in range(int(pos[0] // n), int(pos[-1] // n) + 1):
            m = pos // n == e
            out[m] = np.random.default_rng(np.random.SeedSequence([seed, s, e])).permutation(n)[pos[m] % n]
        return out.reshape(stop - start, b)
    got = {}
    for name, (start, stop) in segments.items():
        real, syn = stream(0, n_real, b_real, start, stop), stream(1, n_syn, b_syn, start, stop)
        h = hashlib.sha256()
        for t in range(stop - start):
            h.update(real[t].tobytes())
            h.update(syn[t].tobytes())
        got[name] = h.hexdigest()
    return got


def audit(ctx) -> dict:
    """Each unit's entry, its selection and extension, its exposure (index digests regenerated from the rule), the
    pairing of duplication and CCIL (identical index streams), the pools' containment, and every selected
    checkpoint's held-out InfoNCE re-scored from disk."""
    from ..run import require
    tr = require(ctx, "audit")["train"]
    dd = data(ctx)
    lr = ctx.spec["learner"]
    problems, streams = {}, {}
    for u in design(ctx):
        key = u["key"]
        try:
            ident = unit_identity(ctx, u, dd)
            e = ctx.store.get("policy", ident)
            if e is None or tr["units"][key]["entry"] != e.name:
                raise StoreError("missing, or not the entry the train stage recorded")
            res = json.loads((e / "result.json").read_text())
            want = json.loads((unit_dir(ctx, ident) / "unit.json").read_text())
            if {k: v for k, v in res["unit"].items() if k not in ("rows", "stats", "features_root")} != \
                    {k: v for k, v in want.items() if k not in ("rows", "stats", "features_root")}:
                raise StoreError("the unit was not trained under its declared settings")
            h = res["history"]
            ck = {int(s): v["infonce"] for s, v in h["checkpoints"].items()}
            if sorted(ck) != list(range(lr["cadence"], res["reached"] + 1, lr["cadence"])):
                raise StoreError("the checkpoints are not every cadence step up to the reached boundary")
            reached = None
            for b in lr["boundaries"]:  # the extension rule, recomputed from the history
                s_ = min((v, s) for s, v in ck.items() if s <= b)[1]
                if s_ != b or b == lr["boundaries"][-1]:
                    reached, sel = b, s_
                    break
            if (reached, sel) != (res["reached"], res["selected_step"]):
                raise StoreError(f"selection or extension recomputes to {(reached, sel)}")
            outcome = "incomplete_fit" if sel == lr["boundaries"][-1] else "complete"
            if outcome != res["outcome"]:
                raise StoreError("the outcome does not follow from the selection")
            b_real, b_syn = res["per_batch"]["real"], res["per_batch"]["synthetic"]
            if b_syn != int(np.floor(u["alpha"] * lr["batch_size"] + 0.5)) or b_real + b_syn != lr["batch_size"]:
                raise StoreError("the per-batch counts are not the declared fraction")
            segs = {s: (v["start"], v["stop"]) for s, v in h["segments"].items()}
            got = stream_digests(u["seed"], res["rows"]["real"], b_real, res["rows"]["synthetic"], b_syn, segs)
            if any(got[s] != h["segments"][s]["index_sha256"] for s in segs):
                raise StoreError("the recorded index streams are not the declared sampler's")
            if res["exposure"] != {"steps": res["reached"], "real_rows": res["reached"] * b_real,
                                   "synthetic_rows": res["reached"] * b_syn}:
                raise StoreError("the exposure record does not reconcile with the steps")
            streams[key] = {s: h["segments"][s]["index_sha256"] for s in segs}
            rr = json.loads(json.dumps(row(ctx, u, e, tr["units"][key]["reused"]), default=float))
            if rr != tr["units"][key]:
                raise StoreError("the recomputed unit row differs from the train stage's")
            out = unit_dir(ctx, ident) / "rescore.json"
            rc = subprocess.run([runtime_python(), "-m", "pusht_surface_learner", "infonce", str(unit_dir(ctx, ident)),
                                 str(e), str(out)], capture_output=True, text=True)
            if rc.returncode:
                raise StoreError(f"re-scoring failed: {rc.stderr[-600:]}")
            again = json.loads(out.read_text())["infonce"]
            if abs(again - res["selected_infonce"]) > 1e-9 * abs(res["selected_infonce"]):
                raise StoreError(f"the stored checkpoint scores {again}, its record {res['selected_infonce']}")
        except (StoreError, KeyError, ValueError, OSError) as ex:
            problems[key] = str(ex)
    for u in design(ctx):  # pairing: duplication and CCIL drew the same positions at every common step
        if u["arm"] == "ccil":
            dk = u["key"].replace("ccil/", "duplication/", 1)
            a, b = streams.get(u["key"], {}), streams.get(dk, {})
            if any(a[s] != b[s] for s in set(a) & set(b)) or not (set(a) & set(b)):
                problems[f"pair {u['key']}"] = "duplication and CCIL index streams differ"
    for r, v in dd["r"].items():
        if v["pool"] is not None:
            with np.load(v["pool"] / "labels.npz") as z:
                if not np.isin(dd["src"]["episode"][z["src_row"]], v["episodes"]).all():
                    problems[f"pool r{r}"] = "a label's source is outside its subset"
    return {"all_pass": not problems, "units": len(tr["units"]), "problems": problems}


def evaluate_unit(ctx, entry: Path, out: Path) -> dict:
    ev = ctx.spec["evaluation"]
    out.mkdir(parents=True, exist_ok=True)
    r = subprocess.run([str(Path(runtime_python()).parent / "lerobot-eval"), f"--policy.path={entry}",
                        f"--env.type={ev['env']}", f"--env.obs_type={ev['obs_type']}",
                        f"--eval.n_episodes={ev['n_episodes']}", f"--eval.batch_size={ev['batch_size']}",
                        "--eval.use_async_envs=false", f"--seed={ev['start_seed']}", "--policy.device=cuda",
                        f"--output_dir={out}"], capture_output=True, text=True)
    (out / "eval.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise StoreError(f"lerobot-eval exited {r.returncode}: {r.stderr[-800:]}")
    return json.loads((out / "eval_info.json").read_text())


def scores_of(info: dict, ev: dict) -> dict:
    """A one-task lerobot-eval eval_info.json (LeRobot 0.6.1's per_task format). Episode i is reset with seed
    start_seed + i, which the file does not record; the evaluation audit's repeat checks the starts are fixed."""
    tasks = info.get("per_task") or []
    if len(tasks) != 1:
        raise StoreError(f"expected one evaluated task, found {len(tasks)}")
    m = tasks[0]["metrics"]
    mx, sm = [float(x) for x in m["max_rewards"]], [float(x) for x in m["sum_rewards"]]
    if len(mx) != ev["n_episodes"] or len(sm) != len(mx) or len(m["successes"]) != len(mx):
        raise StoreError(f"{len(mx)} episodes, the protocol declares {ev['n_episodes']}")
    return {"avg_max_reward": float(np.mean(mx)), "pc_success": 100.0 * float(np.mean([bool(x) for x in m["successes"]])),
            "max_reward": mx, "sum_reward": sm}


def evaluate(ctx) -> dict:
    from ..run import require
    tr = require(ctx, "evaluate")["train"]
    ev = ctx.spec["evaluation"]
    rows = {}
    for key, u in tr["units"].items():
        entry = ctx.store.root / "policy" / u["entry"]
        ident = {"kind": "rollout", "policy": u["entry"], "evaluation": ev}

        def write(d, entry=entry, key=key):
            info = evaluate_unit(ctx, entry, ctx.root / "work" / "eval" / f"{key.replace('/', '_')}-{int(time.time())}")
            (d / "eval_info.json").write_text(json.dumps(info, sort_keys=True) + "\n")
            (d / "scores.json").write_text(json.dumps(scores_of(info, ev), sort_keys=True) + "\n")
        d, reused = ctx.store.ensure("rollout", ident, write)
        sc = json.loads((d / "scores.json").read_text())
        rows[key] = {"entry": d.name, "reused": reused, "avg_max_reward": sc["avg_max_reward"],
                     "pc_success": sc["pc_success"]}
    return {"all_pass": True, "units": rows}


def evaluation_audit(ctx) -> dict:
    """Every rollout's scores recomputed from its episodes, on the declared starts; the first unit re-evaluated, its
    episodes repeating within the declared tolerances."""
    from ..run import require
    prior = require(ctx, "evaluation_audit")
    ev = ctx.spec["evaluation"]
    problems = {}
    for key, row_ in prior["evaluate"]["units"].items():
        d = ctx.store.root / "rollout" / row_["entry"]
        try:
            sc = scores_of(json.loads((d / "eval_info.json").read_text()), ev)
            if sc != json.loads((d / "scores.json").read_text()) or sc["avg_max_reward"] != row_["avg_max_reward"]:
                raise StoreError("recomputed scores differ from the evaluate stage's")
        except (StoreError, OSError, KeyError) as ex:
            problems[key] = str(ex)
    key = next(iter(prior["evaluate"]["units"]))
    entry = ctx.store.root / "policy" / prior["train"]["units"][key]["entry"]
    again = scores_of(evaluate_unit(ctx, entry, ctx.root / "work" / "eval-repeat" / str(int(time.time()))), ev)
    first = json.loads((ctx.store.root / "rollout" / prior["evaluate"]["units"][key]["entry"] / "scores.json"
                        ).read_text())
    tol = ev["repeat_tolerance"]
    dev = {m: float(np.max(np.abs(np.subtract(again[m], first[m])))) for m in ("max_reward", "sum_reward")}
    if dev["max_reward"] > tol["max_reward"] or dev["sum_reward"] > tol["sum_reward"]:
        problems["repeat"] = f"{key} re-evaluated with deviations {dev}"
    return {"all_pass": not problems, "units": len(prior["evaluate"]["units"]), "problems": problems,
            "repeat": {"unit": key, "max_deviation": dev}}


def report(ctx) -> dict:
    """The response surface and the stage's frozen rule (docs/studies/pusht-surface.md)."""
    from ..run import require
    prior = require(ctx, "report")
    a, m = ctx.spec["analysis"], ctx.spec["analysis"]["metric"]
    sc = {k: v[m] for k, v in prior["evaluate"]["units"].items()}
    tr = prior["train"]["units"]
    complete = {k: v["outcome"] == "complete" for k, v in tr.items()}
    seeds = ctx.spec["seeds"]
    out_cells = {}
    for r, al in cells(ctx.spec):
        per = {}
        for s in seeds:
            c, d_, b = f"ccil/r{r}/a{al}/s{s}", f"duplication/r{r}/a{al}/s{s}", f"bc/r{r}/a0/s{s}"
            per[s] = {"ccil": sc[c], "duplication": sc[d_], "bc": sc[b], "ccil_minus_duplication": sc[c] - sc[d_],
                      "ccil_minus_bc": sc[c] - sc[b], "duplication_minus_bc": sc[d_] - sc[b],
                      "complete": complete[c] and complete[d_] and complete[b]}
        out_cells[f"r{r}/a{al}"] = {"r": r, "alpha": al, "per_seed": per,
                                    "eligible": all(v["ccil_minus_duplication"] > 0 and v["ccil_minus_bc"] > 0
                                                    and v["complete"] for v in per.values()),
                                    "mean_ccil_minus_duplication": float(np.mean(
                                        [v["ccil_minus_duplication"] for v in per.values()]))}
    ref = a["references"]
    bc_full = {s: sc[f"bc/r1.0/a0/s{s}"] for s in seeds if f"bc/r1.0/a0/s{s}" in sc}
    competence = {s: (v - ref["hold_still"]) / (ref["demonstrations"] - ref["hold_still"]) for s, v in bc_full.items()}
    competent = bool(competence) and all(v >= a["competence_fraction"] for v in competence.values())
    eligible = [k for k, v in out_cells.items() if v["eligible"]]
    tie = a["tie_tolerance"]

    def rank(keys):
        """Largest mean CCIL - duplication first; effects within the tie tolerance of the best order by lower alpha,
        then lower r."""
        best = max(out_cells[k]["mean_ccil_minus_duplication"] for k in keys)
        top = [k for k in keys if best - out_cells[k]["mean_ccil_minus_duplication"] <= tie]
        top.sort(key=lambda k: (out_cells[k]["alpha"], out_cells[k]["r"]))
        rest = [k for k in keys if k not in top]
        return top + (rank(rest) if rest else [])
    if ctx.spec["stage"] == 1:
        if not competent:
            decision = {"next": "stop", "why": "BC at r = 1 is below the competence fraction: the surface is "
                                               "uninformative about CCIL"}
        elif not eligible:
            decision = {"next": "stop", "why": "no cell has CCIL above both duplication and BC"}
        else:
            decision = {"next": "stage 2", "cells": rank(eligible)[:2]}
    else:
        decision = {"next": "see the protocol", "eligible": rank(eligible) if eligible else []}
    return {"all_pass": True, "metric": m, "cells": out_cells, "bc_competence": competence, "competent": competent,
            "eligible": eligible, "decision": decision,
            "incomplete_fit": prior["train"]["incomplete_fit"]}


RUNNERS = {"preflight": preflight, "train": train, "audit": audit, "evaluate": evaluate,
           "evaluation_audit": evaluation_audit, "report": report}
