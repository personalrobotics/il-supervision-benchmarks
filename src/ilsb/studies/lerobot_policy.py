"""A LeRobot learner on a LeRobot dataset and environment (ilsb#50): one learner, its seeds, its evaluation.

A specification with `task.kind: lerobot` selects this study; its `learner.kind` selects the learner adapter
(ilsb.learners.lerobot_diffusion). Nothing here is task-specific: the dataset (a LeRobot v3.0 directory pinned by
its files' hashes, ILSB_LEROBOT_DATASET), the episode roles, the LeRobot environment and the evaluation starts are
all declared. Training, held-out scoring and evaluation run in the pinned LeRobot environment (ILSB_LEROBOT_PYTHON),
which, with its GPU, is part of every store identity.

Stages:
  preflight         the dataset by its files; the roles; the training and held-out datasets (LeRobot's split, each
                    with its own episodes' statistics, checked against the rows); every window of both, built by
                    ilsb.temporal from the rows and checked slot by slot against what LeRobotDataset gives the
                    learner; the design; every count the protocol implies, asserted exactly
  train             each seed's unit: lerobot-train (resumed from its last checkpoint if interrupted), each
                    checkpoint's held-out loss, the declared selection
  audit             each unit's checkpoint against its record: its train_config, its normalization statistics (the
                    training episodes'), its weights' hash, the selection recomputed from the history, and the
                    selected checkpoint's held-out loss re-scored in a fresh process
  evaluate          lerobot-eval of each selected checkpoint on the declared starts
  evaluation_audit  each rollout's scores recomputed from its episodes, its arguments and starts the protocol's, its
                    checkpoint the audited one; one unit re-evaluated and the repeat's deviation recorded (a measurement:
                    a closed-loop rollout does not repeat where the simulator does not, docs/lerobot-learners.md)
  report            per seed and mean, the acceptance threshold, every episode that did not succeed
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tarfile
import time
from pathlib import Path

import numpy as np

from .. import temporal
from ..learners import lerobot_diffusion
from ..store import StoreError, digest, sha256_file, write_json
from .pusht_surface import Refused, schedule, tree_sha256

RUNTIME = Path(__file__).resolve().parents[1] / "learners" / "lerobot_runtime.py"
LEARNERS = {lerobot_diffusion.KIND: lerobot_diffusion}
FEATURE_PREFIX = "observation."
DETERMINISTIC_ENV = {"CUBLAS_WORKSPACE_CONFIG": ":4096:8", "PYTHONHASHSEED": "0"}
NEED = {"study", "task", "data", "roles", "learner", "seeds", "evaluation", "acceptance", "expected", "compute"}


# --- specification ----------------------------------------------------------------------------------------------------
def load_spec(spec: dict) -> dict:
    if set(spec) != NEED:
        raise Refused(f"refused: specification keys {sorted(spec)} differ from {sorted(NEED)}")
    t, d = spec["task"], spec["data"]
    if set(t) != {"kind", "env"} or not isinstance(t["env"], dict) or "type" not in t["env"]:
        raise Refused("refused: task is {kind: lerobot, env: {type, ...lerobot-eval --env.* arguments}}")
    if set(d) != {"repo_id", "revision", "files"} or not isinstance(d["files"], dict) or not d["files"]:
        raise Refused("refused: data is {repo_id, revision, files: {relative path: sha256}}")
    tr, ho = spec["roles"].get("train"), spec["roles"].get("held_out")
    if set(spec["roles"]) != {"train", "held_out"} or not tr or not ho:
        raise Refused("refused: roles are non-empty train and held_out episode lists")
    for ids in (tr, ho):
        if any(isinstance(e, bool) or not isinstance(e, int) for e in ids) or len(set(ids)) != len(ids):
            raise Refused("refused: role episodes are distinct integers")
    if set(tr) & set(ho):
        raise Refused(f"refused: episodes {sorted(set(tr) & set(ho))[:5]} are both training and held-out")
    lr = spec["learner"]
    if not isinstance(lr, dict) or lr.get("kind") not in LEARNERS:
        raise Refused(f"refused: learner.kind must be one of {sorted(LEARNERS)}")
    try:
        LEARNERS[lr["kind"]].check(lr)
    except ValueError as e:
        raise Refused(f"refused: {e}") from None
    s = spec["seeds"]
    if not s or len(set(s)) != len(s) or any(isinstance(x, bool) or not isinstance(x, int) for x in s):
        raise Refused("refused: seeds are distinct integers")
    ev = spec["evaluation"]
    if set(ev) != {"start_seed", "n_episodes", "batch_size", "device"} or ev["n_episodes"] % ev["batch_size"]:
        raise Refused("refused: evaluation is {start_seed, n_episodes, batch_size (dividing n_episodes), device}")
    a = spec["acceptance"]
    if a.get("metric") != "avg_max_reward" or not set(a) - {"metric"} \
            or not set(a) <= {"metric", "min_mean", "min_per_seed"}:
        raise Refused("refused: acceptance is {metric: avg_max_reward, min_mean and/or min_per_seed}")
    exp = {"train_episodes", "held_out_episodes", "train_frames", "held_out_frames", "train_windows",
           "held_out_windows", "units"}
    if set(spec["expected"]) != exp:
        raise Refused(f"refused: expected counts are {sorted(exp)}")
    c = spec["compute"]
    if set(c) != {"device", "learner_device", "stall_minutes"} or c["device"] != "cpu" \
            or c["learner_device"] not in ("cuda", "cpu") or ev["device"] != c["learner_device"]:
        raise Refused("refused: compute is {device: cpu (the harness), learner_device: cuda or cpu (training, scoring "
                      "and evaluation), stall_minutes}, and evaluation.device is the learner device")
    return spec


def learner(ctx):
    return LEARNERS[ctx.spec["learner"]["kind"]]


def runtime_python() -> str:
    py = os.environ.get("ILSB_LEROBOT_PYTHON")
    if not py or not Path(py).exists():
        raise Refused("refused: set ILSB_LEROBOT_PYTHON to the pinned LeRobot environment's python")
    return py


def runtime(*args, env=None, **kw) -> subprocess.CompletedProcess:
    return subprocess.run([runtime_python(), "-I", str(RUNTIME), *map(str, args)], capture_output=True, text=True,
                          env={**os.environ, **DETERMINISTIC_ENV, **(env or {})}, **kw)


def scope(ctx) -> dict:
    """The learner environment (its distributions, Python, PyTorch, the device it trains on) and the runtime file."""
    probe = ("import importlib.metadata as m, json, sys, torch\n"
             "d = sorted(f'{x.metadata[\"Name\"]}=={x.version}' for x in m.distributions())\n"
             "g = torch.cuda.get_device_name() if torch.cuda.is_available() else None\n"
             "print(json.dumps({'distributions': d, 'lerobot': m.version('lerobot'), 'python': sys.version.split()[0],"
             " 'torch': torch.__version__, 'cuda': torch.version.cuda, 'gpu': g}))")
    r = subprocess.run([runtime_python(), "-I", "-c", probe], capture_output=True, text=True)
    if r.returncode:
        raise Refused(f"refused: the LeRobot environment did not answer: {r.stderr[-800:]}")
    env = json.loads(r.stdout)
    if env["lerobot"] != ctx.spec["learner"]["lerobot"]:
        raise Refused(f"refused: the LeRobot environment holds lerobot {env['lerobot']}, the protocol pins "
                      f"{ctx.spec['learner']['lerobot']}")
    if ctx.spec["compute"]["learner_device"] == "cuda" and not env["gpu"]:
        raise Refused("refused: the LeRobot environment sees no GPU")
    if ctx.spec["compute"]["learner_device"] == "cpu":
        env["gpu"] = None  # a CPU study's numbers do not depend on a GPU that happens to be present
    return {"lerobot_env": {**env, "distributions_sha256": digest(env.pop("distributions"))},
            "runtime_sha256": tree_sha256(RUNTIME)}


# --- data -------------------------------------------------------------------------------------------------------------
def dataset(ctx) -> Path:
    """The declared LeRobot dataset directory, by every file's hash (no file missing, none extra; a hidden directory,
    such as the Hub client's .cache, is not part of the dataset)."""
    p = os.environ.get("ILSB_LEROBOT_DATASET")
    if not p:
        raise Refused("refused: set ILSB_LEROBOT_DATASET to the declared LeRobot dataset directory")
    root = Path(p).resolve()
    got = {f.relative_to(root).as_posix(): f for f in sorted(root.rglob("*"))
           if f.is_file() and not any(part.startswith(".") for part in f.relative_to(root).parts)}
    want = ctx.spec["data"]["files"]
    if set(got) != set(want) or any(sha256_file(got[k]) != v for k, v in want.items()):
        bad = sorted(set(got) ^ set(want)) or sorted(k for k in want if sha256_file(got[k]) != want[k])
        raise Refused(f"refused: {root} is not the declared dataset (differs at {bad[:5]})")
    return root


def read_rows(root: Path) -> dict:
    """Every row of a LeRobot v3.0 dataset's data files, in index order: columns as arrays."""
    import pyarrow.parquet as pq
    tables = [pq.read_table(f) for f in sorted((root / "data").rglob("*.parquet"))]
    cols = {}
    for name in tables[0].column_names:
        parts = [t.column(name).to_pylist() for t in tables]
        cols[name] = np.asarray([x for p in parts for x in p])
    order = np.argsort(cols["index"], kind="stable")
    return {k: v[order] for k, v in cols.items()}


def feature_keys(rows: dict) -> list:
    return sorted(k for k in rows if k.startswith(FEATURE_PREFIX) or k == "action")


def split_entry(ctx, source: Path) -> Path:
    """LeRobot's split of the declared dataset into the training and held-out datasets, one tar of both."""
    roles = ctx.spec["roles"]

    def write(d):
        tmp = ctx.root / "work" / f"split-{os.getpid()}"
        shutil.rmtree(tmp, ignore_errors=True)
        tmp.mkdir(parents=True)
        (tmp / "roles.json").write_text(json.dumps(roles))
        r = runtime("split", source, tmp / "out", tmp / "roles.json")
        if r.returncode:
            raise StoreError(f"LeRobot's split failed: {r.stderr[-1500:]}")
        with tarfile.open(d / "split.tar", "w") as tf:
            for f in sorted((tmp / "out").rglob("*")):
                if f.is_file():
                    tf.add(f, arcname=f.relative_to(tmp / "out").as_posix())
        shutil.rmtree(tmp)
    return ctx.store.ensure("split", {"kind": "split", "data": ctx.spec["data"]["files"], "roles": roles}, write)[0]


def split_root(ctx, entry: Path) -> Path:
    """The split's datasets, extracted once under the run root from the verified entry."""
    root = ctx.root / "data" / entry.name
    if not (root / ".complete").exists():
        tmp = root.with_name(f".{entry.name}-{os.getpid()}")
        shutil.rmtree(tmp, ignore_errors=True)
        with tarfile.open(entry / "split.tar") as tf:
            tf.extractall(tmp, filter="data")
        (tmp / ".complete").write_text("")
        shutil.rmtree(root, ignore_errors=True)
        os.replace(tmp, root)
    return root


def check_stats(root: Path, rows: dict) -> list:
    """A dataset's meta/stats.json min and max equal its own rows' (so normalization reads its episodes only)."""
    st = json.loads((root / "meta" / "stats.json").read_text())
    problems = []
    for k in feature_keys(rows):
        x = np.stack(rows[k]).astype(np.float32)
        for m, f in (("min", np.min), ("max", np.max)):
            if not np.array_equal(np.asarray(st[k][m], np.float32).reshape(-1), f(x, 0).reshape(-1)):
                problems.append(f"{root.name}: {k} {m} is not its rows' {m}")
    return problems


def check_windows(ctx, root: Path, rows: dict, spec: temporal.WindowSpec) -> tuple[dict, list]:
    """The contract's windows over the rows, verified, against LeRobotDataset's: the same anchors, and in every slot
    the same values and padding flags."""
    w = temporal.windows(rows["episode_index"], rows["frame_index"], spec)
    temporal.verify(w, rows["episode_index"], rows["frame_index"], spec)
    (ctx.root / "work").mkdir(parents=True, exist_ok=True)  # a reused split made none
    out = ctx.root / "work" / f"windows-{root.name}.npz"
    policy = {"type": learner(ctx).POLICY_TYPE, **ctx.spec["learner"]["policy"]}
    (ctx.root / "work" / "policy.json").write_text(json.dumps(policy))
    r = runtime("windows", root, ctx.root / "work" / "policy.json", out)
    if r.returncode:
        raise StoreError(f"LeRobot windows failed: {r.stderr[-1500:]}")
    problems = []
    with np.load(out) as z:
        lr = {k: z[k] for k in z.files}
    if not np.array_equal(lr["anchor"], rows["index"][w["anchor"]]) or not np.array_equal(lr["episode"], w["episode"]):
        return w, [f"{root.name}: LeRobot's anchors differ from the contract's"]
    for k in feature_keys(rows):
        x = np.stack(rows[k]).astype(np.float32)
        x = x.reshape(len(x), -1)
        part = "action" if k == "action" else "obs"
        want, pad = x[w[f"{part}_rows"]], w[f"{part}_pad"]
        got = lr[k].reshape(want.shape)
        if not np.array_equal(got, want):
            problems.append(f"{root.name}: {k} windows differ from the contract's")
        if not np.array_equal(lr[f"{k}_is_pad"], pad):
            problems.append(f"{root.name}: {k} padding flags differ from the contract's")
    return w, problems


def data(ctx) -> dict:
    source = dataset(ctx)
    entry = split_entry(ctx, source)
    root = split_root(ctx, entry)
    return {"source": source, "split": entry, "train": root / "train", "held_out": root / "held_out"}


# --- units ------------------------------------------------------------------------------------------------------------
def design(ctx) -> list[dict]:
    return [{"key": f"{ctx.spec['learner']['kind']}/s{s}", "seed": s} for s in ctx.spec["seeds"]]


def unit_identity(ctx, u, dd) -> dict:
    return {"kind": "policy", "split": dd["split"].name, "learner": ctx.spec["learner"], "seed": u["seed"],
            "device": ctx.spec["compute"]["learner_device"]}


def unit_dir(ctx, ident) -> Path:
    return ctx.root / "work" / digest(ctx.store.full(ident))[:24]


def unit_files(ctx, u, dd, work: Path) -> None:
    if (work / "unit.json").exists():
        return
    work.mkdir(parents=True, exist_ok=True)
    lr = ctx.spec["learner"]
    write_json(work / "unit.json", {"seed": u["seed"], "policy_type": learner(ctx).POLICY_TYPE,
                                    "policy": lr["policy"], "steps": lr["steps"],
                                    "batch_size": lr["batch_size"], "save_freq": lr["save_freq"],
                                    "log_freq": lr["log_freq"], "num_workers": lr["num_workers"],
                                    "device": ctx.spec["compute"]["learner_device"],
                                    "train_root": str(dd["train"]), "held_out_root": str(dd["held_out"]),
                                    "selection": {k: lr["selection"][k] for k in ("seed", "batch_size")}})


def put_unit(ctx, ident, work: Path) -> Path:
    """The trained unit: its record and the selected checkpoint's files (a pretrained_model directory lerobot-eval
    loads as is)."""
    hist = json.loads((work / "work" / "history.json").read_text())
    lr = ctx.spec["learner"]
    step = learner(ctx).select(hist["checkpoints"], lr["selection"]["rule"])
    ck = work / "work" / "train" / "checkpoints" / f"{step:06d}" / "pretrained_model"
    rec = next(h for h in hist["checkpoints"] if h["step"] == step)
    result = {"selected_step": step, "rule": lr["selection"]["rule"], "selected_held_out_loss": rec["held_out_loss"],
              "selected_model_sha256": rec["model_sha256"], "checkpoints": hist["checkpoints"],
              "resumed_from": hist["resumed_from"], "restarted_from": hist["restarted_from"], "peak": hist["peak"]}

    def write(d):
        write_json(d / "result.json", result)
        for f in sorted(ck.iterdir()):
            shutil.copyfile(f, d / f.name)
    return ctx.store.put("policy", ident, write)


def train_unit(ctx, key: str, gpu: str | None = None) -> int:
    """One unit (`ilsb unit SPEC --root R --key KEY`): train, score, select, store."""
    dd = data(ctx)
    u = next(x for x in design(ctx) if x["key"] == key)
    ident = unit_identity(ctx, u, dd)
    if ctx.store.get("policy", ident) is not None:
        return 0
    work = unit_dir(ctx, ident)
    unit_files(ctx, u, dd, work)
    env = {**os.environ, **DETERMINISTIC_ENV, **slot_env(gpu)}
    with open(work / "learner.log", "a") as log:
        rc = subprocess.run([runtime_python(), "-I", str(RUNTIME), "train", str(work)], env=env, stdout=log,
                            stderr=subprocess.STDOUT).returncode
    if rc == 0:
        put_unit(ctx, ident, work)
    return rc


# --- stages -----------------------------------------------------------------------------------------------------------
def preflight(ctx) -> dict:
    dd = data(ctx)
    exp, roles = ctx.spec["expected"], ctx.spec["roles"]
    spec = learner(ctx).window_spec(ctx.spec["learner"])
    src = read_rows(dd["source"])
    problems = []
    eps = set(np.unique(src["episode_index"]).tolist())
    if not set(roles["train"]) | set(roles["held_out"]) <= eps:
        problems.append("a role names an episode the dataset does not hold")
    counts, windows = {}, {}
    for role in ("train", "held_out"):
        rows = read_rows(dd[role])
        want = sorted(roles[role])
        # LeRobot's split renumbers episodes 0..n-1 in the source's order: map them back and check every row
        src_rows = np.flatnonzero(np.isin(src["episode_index"], want))
        if len(rows["index"]) != len(src_rows) or not np.array_equal(
                np.asarray(want)[rows["episode_index"]], src["episode_index"][src_rows]):
            problems.append(f"{role}: its rows are not the source rows of its episodes")
        else:
            for k in feature_keys(src):
                if not np.array_equal(np.stack(rows[k]), np.stack(src[k][src_rows])):
                    problems.append(f"{role}: {k} differs from the source rows of its episodes")
        problems += check_stats(dd[role], rows)
        try:
            w, wp = check_windows(ctx, dd[role], rows, spec)
        except temporal.WindowError as e:
            problems.append(f"{role}: {e}")
            continue
        problems += wp
        counts[role] = {"episodes": len(want), "frames": int(len(rows["index"]))}
        windows[role] = temporal.summary(w)
    if counts and windows and len(problems) == 0:
        got = {"train_episodes": counts["train"]["episodes"], "held_out_episodes": counts["held_out"]["episodes"],
               "train_frames": counts["train"]["frames"], "held_out_frames": counts["held_out"]["frames"],
               "train_windows": windows["train"]["windows"], "held_out_windows": windows["held_out"]["windows"],
               "units": len(design(ctx))}
        if got != exp:
            problems.append(f"counts {got} differ from the declared {exp}")
    units = design(ctx)
    return {"all_pass": not problems, "problems": problems, "window_spec": spec.as_dict(), "counts": counts,
            "windows": windows, "unused_episodes": len(eps - set(roles["train"]) - set(roles["held_out"])),
            "split_entry": dd["split"].name, "units": {u["key"]: digest(ctx.store.full(unit_identity(ctx, u, dd)))
                                                       for u in units},
            "code": ctx.protocol["code"]}


def train(ctx) -> dict:
    dd = data(ctx)
    units = design(ctx)
    idents = {u["key"]: unit_identity(ctx, u, dd) for u in units}
    todo = [k for k, i in idents.items() if ctx.store.get("policy", i) is None]
    failed, gpus = {}, ctx.options.get("gpus") or []
    if todo and gpus:
        failed = schedule(ctx, todo, gpus, {k: unit_dir(ctx, i) for k, i in idents.items()})
    rows, missing = {}, []
    for u in units:
        e = ctx.store.get("policy", idents[u["key"]])
        if e is None:
            missing.append(u["key"])
            continue
        r = json.loads((e / "result.json").read_text())
        rows[u["key"]] = {"entry": e.name, "seed": u["seed"], "selected_step": r["selected_step"],
                          "selected_held_out_loss": r["selected_held_out_loss"],
                          "selected_model_sha256": r["selected_model_sha256"], "resumed_from": r["resumed_from"],
                          "restarted_from": r["restarted_from"],
                          "peak": r["peak"], "reused": u["key"] not in todo}
    return {"all_pass": not failed and not missing, "units": rows, "failed": failed, "missing": missing}


def audit(ctx) -> dict:
    from ..run import require
    prior = require(ctx, "audit")
    dd = data(ctx)
    lr = ctx.spec["learner"]
    mod = learner(ctx)
    train_stats = json.loads((dd["train"] / "meta" / "stats.json").read_text())
    problems = {}
    for u in design(ctx):
        row_ = prior["train"]["units"][u["key"]]
        try:
            e = ctx.store.get("policy", unit_identity(ctx, u, dd))
            if e is None or e.name != row_["entry"]:
                raise StoreError("the unit's entry is not the one the train stage recorded")
            r = json.loads((e / "result.json").read_text())
            steps = [h["step"] for h in r["checkpoints"]]
            if steps != mod.checkpoint_steps(lr):
                raise StoreError(f"checkpoints at {steps}, the protocol saves at {mod.checkpoint_steps(lr)}")
            if mod.select(r["checkpoints"], lr["selection"]["rule"]) != r["selected_step"]:
                raise StoreError("the selection does not recompute from the history")
            if sha256_file(e / "model.safetensors") != r["selected_model_sha256"]:
                raise StoreError("the stored weights are not the selected checkpoint's")
            problems_ = check_checkpoint(ctx, e, mod.train_config_contract(lr, u["seed"], ctx.spec["compute"]["learner_device"]),
                                         train_stats, dd["train"])
            if problems_:
                raise StoreError("; ".join(problems_))
            out = ctx.root / "work" / f"rescore-{e.name[:16]}.json"
            s = lr["selection"]
            rr = runtime("score", e, dd["held_out"], s["seed"], s["batch_size"], ctx.spec["compute"]["learner_device"],
                         out, env=gpu_env(ctx))
            if rr.returncode:
                raise StoreError(f"re-scoring failed: {rr.stderr[-800:]}")
            if json.loads(out.read_text())["held_out_loss"] != r["selected_held_out_loss"]:
                raise StoreError("the selected checkpoint's held-out loss does not reproduce in a fresh process")
        except (StoreError, OSError, KeyError, ValueError) as ex:
            problems[u["key"]] = str(ex)
    return {"all_pass": not problems, "units": len(design(ctx)), "problems": problems}


def check_checkpoint(ctx, ckpt: Path, contract: dict, train_stats: dict, train_root: Path) -> list:
    """A checkpoint is this unit's: its train_config records the declared training (paths aside) on the training
    dataset, and its normalizer and unnormalizer hold the training episodes' statistics."""
    problems = []
    tc = json.loads((ckpt / "train_config.json").read_text())
    for k, v in contract.items():
        if k == "policy":
            for pk, pv in v.items():
                got = tc["policy"].get(pk)
                if json.loads(json.dumps(got)) != json.loads(json.dumps(pv)):
                    problems.append(f"train_config policy.{pk} is {got!r}, the unit declares {pv!r}")
        elif tc.get(k) != v:
            problems.append(f"train_config {k} is {tc.get(k)!r}, the unit declares {v!r}")
    # The training dataset is named by its split entry (extracted at <root>/data/<entry>/train), not by where a run
    # root happened to put it, so a unit reused from another root's store checks the same way.
    trained_on = Path(tc["dataset"]["root"])
    if (trained_on.name, trained_on.parent.name) != (train_root.name, train_root.parent.name) \
            or tc["dataset"].get("episodes") is not None:
        problems.append("the checkpoint was not trained on the whole training dataset of this split")
    out = ctx.root / "work" / f"stats-{os.getpid()}-{ckpt.name[:16]}.json"
    r = runtime("stats", ckpt, out)
    if r.returncode:
        return problems + [f"reading the checkpoint's statistics failed: {r.stderr[-800:]}"]
    held = json.loads(out.read_text())
    if not any("_normalizer_" in f for f in held) or not any("_unnormalizer_" in f for f in held):
        problems.append("the checkpoint lacks normalizer or unnormalizer statistics")
    compared = set()
    for f, st in held.items():
        for name, arr in st.items():
            feat, stat = name.rsplit(".", 1)
            if stat not in ("min", "max"):
                continue
            compared.add(feat)
            if feat not in train_stats or not np.array_equal(np.asarray(arr, np.float32),
                                                             np.asarray(train_stats[feat][stat], np.float32)
                                                             .reshape(-1)):
                problems.append(f"{f}: {name} is not the training episodes' {stat}")
    min_max = {k for k, m in contract["policy"]["normalization_mapping"].items() if m == "MIN_MAX"}
    if min_max and not compared:
        problems.append("the checkpoint's statistics hold no min/max to check against the training episodes'")
    return problems


def slot_env(slot: str | None) -> dict:
    """A scheduler slot: a GPU id, or `cpu` (no GPU visible)."""
    return {} if slot is None else {"CUDA_VISIBLE_DEVICES": "" if slot == "cpu" else slot}


def gpu_env(ctx) -> dict:
    gpus = ctx.options.get("gpus") or []
    return slot_env(gpus[0] if gpus else None)


def eval_args(ctx) -> list:
    """lerobot-eval's arguments, the policy and output paths aside. It resets episode i with seed start_seed + i
    (batch b's environments take start_seed + b * batch_size onward), so the starts are these arguments."""
    ev, env = ctx.spec["evaluation"], ctx.spec["task"]["env"]
    return [*(f"--env.{k}={v}" for k, v in env.items()), f"--eval.n_episodes={ev['n_episodes']}",
            f"--eval.batch_size={ev['batch_size']}", "--eval.use_async_envs=false", f"--seed={ev['start_seed']}",
            f"--policy.device={ev['device']}"]


def evaluate_unit(ctx, entry: Path, out: Path) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    r = runtime("evaluate", f"--policy.path={entry}", *eval_args(ctx), f"--output_dir={out}", env=gpu_env(ctx))
    out.mkdir(parents=True, exist_ok=True)
    (out / "eval.log").write_text(r.stdout + r.stderr)
    if r.returncode:
        raise StoreError(f"lerobot-eval exited {r.returncode}: {r.stderr[-800:]}")
    return json.loads((out / "eval_info.json").read_text())


def scores_of(info: dict, ev: dict) -> dict:
    """A one-task lerobot-eval eval_info.json (LeRobot 0.6.1's per_task format): each episode's max and sum reward and
    success. The file does not record the episodes' seeds; episode i's is start_seed + i (eval_args)."""
    tasks = info.get("per_task") or []
    if len(tasks) != 1:
        raise StoreError(f"expected one evaluated task, found {len(tasks)}")
    m = tasks[0]["metrics"]
    mx, sm, ok = [float(x) for x in m["max_rewards"]], [float(x) for x in m["sum_rewards"]], [bool(x) for x in
                                                                                             m["successes"]]
    if len(mx) != ev["n_episodes"] or len(sm) != len(mx) or len(ok) != len(mx):
        raise StoreError(f"{len(mx)} episodes, the protocol declares {ev['n_episodes']}")
    seeds = list(range(ev["start_seed"], ev["start_seed"] + ev["n_episodes"]))
    return {"avg_max_reward": float(np.mean(mx)), "pc_success": 100.0 * float(np.mean(ok)), "max_reward": mx,
            "sum_reward": sm, "success": ok, "start_seed": seeds}


def evaluate(ctx) -> dict:
    from ..run import require
    tr = require(ctx, "evaluate")["train"]
    ev = ctx.spec["evaluation"]
    rows = {}
    for key, u in tr["units"].items():
        entry = ctx.store.root / "policy" / u["entry"]

        def write(d, entry=entry, key=key):
            info = evaluate_unit(ctx, entry, ctx.root / "work" / "eval" / f"{key.replace('/', '_')}-{int(time.time())}")
            (d / "eval_info.json").write_text(json.dumps(info, sort_keys=True) + "\n")
            (d / "eval_args.json").write_text(json.dumps(eval_args(ctx)) + "\n")
            (d / "scores.json").write_text(json.dumps(scores_of(info, ev), sort_keys=True) + "\n")
        d, reused = ctx.store.ensure("rollout", {"kind": "rollout", "policy": u["entry"], "evaluation": ev,
                                                 "env": ctx.spec["task"]["env"]}, write)
        sc = json.loads((d / "scores.json").read_text())
        rows[key] = {"entry": d.name, "reused": reused, "avg_max_reward": sc["avg_max_reward"],
                     "pc_success": sc["pc_success"]}
    return {"all_pass": True, "units": rows}


def evaluation_audit(ctx) -> dict:
    from ..run import require
    prior = require(ctx, "evaluation_audit")
    ev = ctx.spec["evaluation"]
    problems = {}
    for key, row_ in prior["evaluate"]["units"].items():
        d = ctx.store.root / "rollout" / row_["entry"]
        try:
            if json.loads((d / "identity.json").read_text())["policy"] != prior["train"]["units"][key]["entry"]:
                raise StoreError("the rollout is not of the unit's audited checkpoint")
            if json.loads((d / "eval_args.json").read_text()) != eval_args(ctx):
                raise StoreError("the rollout was not evaluated with the protocol's arguments and starts")
            sc = scores_of(json.loads((d / "eval_info.json").read_text()), ev)
            if sc != json.loads((d / "scores.json").read_text()) or sc["avg_max_reward"] != row_["avg_max_reward"]:
                raise StoreError("recomputed scores differ from the evaluate stage's")
        except (StoreError, OSError, KeyError) as ex:
            problems[key] = str(ex)
    key = next(iter(prior["evaluate"]["units"]))
    entry = ctx.store.root / "policy" / prior["train"]["units"][key]["entry"]
    again = scores_of(evaluate_unit(ctx, entry, ctx.root / "work" / "eval-repeat" / str(int(time.time()))), ev)
    first = json.loads((ctx.store.root / "rollout" / prior["evaluate"]["units"][key]["entry"] / "scores.json")
                       .read_text())
    repeat = {"unit": key, "identical": again == first,
              "avg_max_reward": {"first": first["avg_max_reward"], "repeat": again["avg_max_reward"]},
              "pc_success": {"first": first["pc_success"], "repeat": again["pc_success"]},
              "episodes_with_other_success": int(np.sum(np.not_equal(again["success"], first["success"]))),
              "max_episode_deviation": {m: float(np.max(np.abs(np.subtract(again[m], first[m]))))
                                        for m in ("max_reward", "sum_reward")}}
    return {"all_pass": not problems, "units": len(prior["evaluate"]["units"]), "problems": problems,
            "repeat": repeat}


def acceptance(a: dict, mean: float, per_seed: dict) -> dict:
    """The declared thresholds and whether they are met: the seeds' mean, and every seed on its own."""
    out, met = {}, True
    if "min_mean" in a:
        out["min_mean"], met = a["min_mean"], met and mean >= a["min_mean"]
    if "min_per_seed" in a:
        out["min_per_seed"] = a["min_per_seed"]
        met = met and all(v[a["metric"]] >= a["min_per_seed"] for v in per_seed.values())
    return {**out, "met": met}


def report(ctx) -> dict:
    from ..run import require
    prior = require(ctx, "report")
    ev, tr, a = prior["evaluate"]["units"], prior["train"]["units"], ctx.spec["acceptance"]
    per_seed, failed = {}, {}
    for key, row_ in ev.items():
        sc = json.loads((ctx.store.root / "rollout" / row_["entry"] / "scores.json").read_text())
        per_seed[key] = {"avg_max_reward": sc["avg_max_reward"], "pc_success": sc["pc_success"],
                         "selected_step": tr[key]["selected_step"], "checkpoint": tr[key]["entry"],
                         "model_sha256": tr[key]["selected_model_sha256"]}
        failed[key] = [{"start_seed": s, "max_reward": m} for s, m, ok in
                       zip(sc["start_seed"], sc["max_reward"], sc["success"]) if not ok]
    mean = float(np.mean([v[a["metric"]] for v in per_seed.values()]))
    root = json.loads((ctx.root / "root.json").read_text())
    return {"all_pass": True, "metric": a["metric"], "per_seed": per_seed,
            "mean": mean, "mean_pc_success": float(np.mean([v["pc_success"] for v in per_seed.values()])),
            "acceptance": acceptance(a, mean, per_seed),
            "failed_episodes": failed, "learner": ctx.spec["learner"],
            "upstream": {k: root["scope"]["lerobot_env"][k] for k in ("lerobot", "torch", "cuda", "gpu",
                                                                      "distributions_sha256")}}


RUNNERS = {"preflight": preflight, "train": train, "audit": audit, "evaluate": evaluate,
           "evaluation_audit": evaluation_audit, "report": report}
