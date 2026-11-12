"""#40 Stage A: the read-only retrospective utility diagnostic (docs/studies/reweight-stage-a.md, frozen in #41).

    python tools/reweight_stage_a.py prepare --run RUN --inputs INPUTS.json      (harness environment)
    python tools/reweight_stage_a.py compute --run RUN [--device cuda]           (Pendulum: harness environment;
    LEROBOT_PYTHON tools/reweight_stage_a.py compute --run RUN --pusht            PushT: the LeRobot environment)
    python tools/reweight_stage_a.py report --run RUN [--out docs/studies/evidence/reweight/stage-a.json]

prepare  verifies every input against the committed inventory's hashes and writes, per case and seed,
         RUN/cases/<case>/s<seed>/rows.npz: the checkpoint path, V, the synthetic rows with the CCIL fields as
         recorded (mask, weight, error; NaN where the artifact has none), their source transitions, and the
         comparator rows. INPUTS.json names where each artifact sits on this host.
compute  g_i = <grad L_V, grad l_i> at the case's BC checkpoint for every synthetic and comparator row: one
         forward-mode JVP along grad L_V, exact per row, strict FP32, evaluation mode. Writes g.npz beside rows.npz.
report   the aggregates, the realizable mass, F1, F2 and the decision; per-row files are identified by content SHA-256.
No policy trains, no simulator runs, and no final-test data is read.
"""
import argparse
import hashlib
import json
import math
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")  # deterministic cuBLAS, set before CUDA starts
REPO = Path(__file__).resolve().parents[1]
SPEC = json.loads((REPO / "tools/reweight_stage_a.json").read_text())
INVENTORY = REPO / SPEC["inventory"]
FEATURES = ("observation.state", "observation.environment_state", "action")
PUSHT_NEGATIVES, PUSHT_SEED, CHUNK = SPEC["statistic"]["pusht_negatives"], 0, 64


def sha(path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def need(path, digest: str) -> Path:
    path = Path(path)
    if not path.is_file() or sha(path) != digest:
        sys.exit(f"refused: {path} is missing or does not match the inventory ({digest[:12]})")
    return path


def content_sha(path: Path) -> str:
    """SHA-256 of an .npz by content (names, dtypes, shapes, bytes): a zip's timestamps are not content."""
    h = hashlib.sha256()
    with np.load(path) as z:
        for k in sorted(z.files):
            a = np.ascontiguousarray(z[k])
            h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode() + a.tobytes())
    return h.hexdigest()


def write_npz(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def locate(case_dir: Path, path: Path, digest: str) -> None:
    """Where the verified checkpoint sits on this host (not content: rows.npz holds its identity)."""
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "checkpoint.json").write_text(json.dumps({"path": str(path.resolve()), "sha256": digest}) + "\n")


def checkpoint_of(case_dir: Path) -> str:
    c = json.loads((case_dir / "checkpoint.json").read_text())
    with np.load(case_dir / "rows.npz") as z:
        if str(z["checkpoint_sha256"]) != c["sha256"]:
            sys.exit(f"refused: {case_dir} names another checkpoint")
    f = Path(c["path"]) / "model.safetensors" if Path(c["path"]).is_dir() else Path(c["path"])
    if sha(f) != c["sha256"]:
        sys.exit(f"refused: {f} no longer matches its identity")
    return c["path"]


# --- prepare (harness environment) ------------------------------------------------------------------------------------
def prepare_pendulum(run: Path, inv: dict, name: str, root: Path, data: Path, seeds: list):
    from ccil.schema import read_trajectories
    src = inv["sources"][name]
    eps = {e.episode_id: e for e in read_trajectories(data).episodes}
    for s in seeds:
        d = root / "data" / f"s{s}"
        need(d / "subset.npz", src["labels"]["data"][f"s{s}"]["subset_sha256"])
        split = json.loads(need(d / "split.json", src["labels"]["data"][f"s{s}"]["split_sha256"]).read_text())
        pol = need(root / "policies/bc" / f"s{s}" / "policy.npz", src["policies"]["sha256"][f"bc/s{s}"])
        sel = np.load(need(root / "select" / f"s{s}.npz", src["labels"]["relabels"][f"s{s}"]["sha256"]))
        val_eps = sorted((k for k in eps if k not in split["episode_ids"]), key=int)
        obs = lambda ids: np.concatenate([np.asarray(eps[k].observations["state"], np.float64) for k in ids])  # noqa
        act = lambda ids: np.concatenate([np.asarray(eps[k].actions, np.float64).reshape(len(eps[k].actions), -1)
                                          for k in ids])  # noqa: E731
        n = len(sel["collection/a"])
        locate(run / "cases" / name / f"s{s}", pol, sha(pol))
        write_npz(run / "cases" / name / f"s{s}" / "rows.npz", checkpoint_sha256=np.array(sha(pol)),
                  val_obs=obs(val_eps), val_act=act(val_eps), val_episodes=np.array(val_eps),
                  syn_obs=sel["collection/x"], syn_act=sel["collection/a"].reshape(n, -1),
                  syn_transition=np.arange(n), syn_mask=np.ones(n, bool), syn_weight=np.ones(n),
                  syn_error=np.full(n, np.nan),
                  cmp_obs=obs(split["train"]), cmp_act=act(split["train"]), cmp_weight=np.ones(len(obs(split["train"]))))


def pusht_source(path: Path) -> dict:
    sys.path.insert(0, str(REPO / "tools/pusht_v2"))
    import source as source_module
    with np.load(path) as z:
        arrays = {k: z[k] for k in z.files}
    want = json.loads((REPO / "tools/pusht_v2/surface.json").read_text())["data"]["content_sha256"]
    if source_module.content_sha256(arrays) != want:
        sys.exit("refused: the PushT source rows are not the protocol's by content")
    return arrays


def pusht_rows(src: dict, rows) -> dict:
    return {"state": src["state"][rows].astype(np.float32), "env": src["env"][rows].astype(np.float32),
            "act": src["action"][rows].astype(np.float32)}


def prepare_surface(run: Path, inv: dict, src: dict, stage: str, base: Path):
    """base holds the copied store entries: policy/<entry>/..., pool/<entry>/..., selection/<entry>.npz."""
    from ilsb.studies.pusht_surface import rows_of, subset_episodes
    spec = json.loads((REPO / "tools/pusht_v2/surface.json").read_text())
    s_inv = inv["sources"][f"pusht-surface-{stage}"]
    seed = {"stage1": 1, "stage2": 2}[stage]
    for r, pool in s_inv["labels"].items():
        case = f"pusht-surface-{r}"
        if case not in SPEC["cases"] or seed not in SPEC["cases"][case]["seeds"]:
            continue
        rv = float(r[1:])
        unit = s_inv["units"]["detail"][f"bc/r{rv}/a0/s{seed}"]
        ck = base / "policy" / unit["entry"]
        for f, h in unit["checkpoint"].items():
            need(ck / f, h["sha256"])
        lab = np.load(need(base / "pool" / pool["entry"] / "labels.npz", pool["labels_npz_sha256"]))
        sel = np.load(need(base / "selection" / f"{pool['entry']}.npz", pool["ccil"]["selection"]["sha256"]))
        eps = subset_episodes(spec["roles"]["train"], rv, spec["subset_seed"])
        if len(eps) != spec["expected"]["subset_episodes"][r]:
            sys.exit(f"refused: the subset at {r} has {len(eps)} episodes")
        if rv < 1.0:
            val_eps = sorted(set(spec["roles"]["train"]) - set(eps))
        else:
            val_eps = sorted(spec["roles"]["held_out"])  # selection-contaminated: reported only
        cand, srow = lab["candidate_row"], lab["src_row"]
        cmp = pusht_rows(src, srow)
        val = pusht_rows(src, rows_of(src, val_eps))
        locate(run / "cases" / case / f"s{seed}", ck, unit["checkpoint"]["model.safetensors"]["sha256"])
        write_npz(run / "cases" / case / f"s{seed}" / "rows.npz",
                  checkpoint_sha256=np.array(unit["checkpoint"]["model.safetensors"]["sha256"]),
                  val_state=val["state"], val_env=val["env"], val_act=val["act"], val_episodes=np.array(val_eps),
                  syn_state=lab["syn_state"], syn_env=lab["syn_env"], syn_act=lab["syn_action"],
                  syn_transition=srow, syn_candidate_row=cand, syn_mask=sel["mask"][cand],
                  syn_weight=sel["weight"][cand], syn_error=sel["error"][cand],
                  cmp_state=cmp["state"], cmp_env=cmp["env"], cmp_act=cmp["act"], cmp_weight=np.ones(len(srow)))


def column(table, name: str) -> np.ndarray:
    """A parquet column as an array: list columns as (N, D)."""
    c = table.column(name).combine_chunks()
    if hasattr(c, "flatten"):
        return np.asarray(c.flatten().to_numpy(zero_copy_only=False)).reshape(len(c), -1)
    return np.asarray(c.to_numpy(zero_copy_only=False))


def prepare_v22(run: Path, inv: dict, src: dict, root: Path):
    import pyarrow.parquet as pq
    v = inv["sources"]["pusht-v22"]
    arms = json.loads(need(root / "arms.json", v["records"]["sha256"]["arms.json"]).read_text())["arms"]
    tabs = {}
    for arm in ("ccil", "duplication"):
        f = root / "arms" / arm / "data/chunk-000/file-000.parquet"
        t = pq.read_table(need(f, arms[arm]["files"]["data/chunk-000/file-000.parquet"]))
        keep = t.column("ccil.synthetic").to_numpy(zero_copy_only=False).astype(bool)
        tabs[arm] = {k: column(t, k)[keep] for k in (*FEATURES, "ccil.source_index", "ccil.candidate_row")}
    ev = v["labels"]["all"]["ccil"]
    sel = np.load(need(root / "labels/runs" / ev["selection"]["path"], ev["selection"]["sha256"]))
    syn, dup = tabs["ccil"], tabs["duplication"]
    if len(syn["action"]) != arms["ccil"]["added_rows"] or not np.array_equal(syn["ccil.source_index"],
                                                                               dup["ccil.source_index"]):
        sys.exit("refused: the v2.2 arms' added rows are not source-matched as recorded")
    stack = lambda t, k: t[k].astype(np.float32)  # noqa: E731
    cand = syn["ccil.candidate_row"].astype(np.int64)
    spec = json.loads((REPO / "tools/pusht_v2/surface.json").read_text())
    from ilsb.studies.pusht_surface import rows_of
    val_eps = sorted(spec["roles"]["held_out"])
    val = pusht_rows(src, rows_of(src, val_eps))
    for s in SPEC["cases"]["pusht-v22"]["seeds"]:
        u = v["units"]["detail"][f"bc-s{s}"]
        ck = root / "units" / f"bc-s{s}" / "train/checkpoints" / f"{u['selected_step']:06d}" / "pretrained_model"
        for f, h in u["checkpoint"].items():
            need(ck / f, h)
        locate(run / "cases" / "pusht-v22" / f"s{s}", ck, u["checkpoint"]["model.safetensors"])
        write_npz(run / "cases" / "pusht-v22" / f"s{s}" / "rows.npz",
                  checkpoint_sha256=np.array(u["checkpoint"]["model.safetensors"]),
                  val_state=val["state"], val_env=val["env"], val_act=val["act"], val_episodes=np.array(val_eps),
                  syn_state=stack(syn, FEATURES[0]), syn_env=stack(syn, FEATURES[1]), syn_act=stack(syn, FEATURES[2]),
                  syn_transition=syn["ccil.source_index"].astype(np.int64), syn_candidate_row=cand,
                  syn_mask=sel["mask"][cand], syn_weight=sel["weight"][cand], syn_error=sel["error"][cand],
                  cmp_state=stack(dup, FEATURES[0]), cmp_env=stack(dup, FEATURES[1]), cmp_act=stack(dup, FEATURES[2]),
                  cmp_weight=np.ones(len(dup["action"])))


def prepare(run: Path, inputs: dict):
    inv = json.loads(INVENTORY.read_text())
    if inv["problems"]:
        sys.exit("refused: the inventory records problems")
    for case, name in (("pendulum-capture", "pendulum-capture"), ("pendulum-pumping", "pendulum-pumping")):
        prepare_pendulum(run, inv, name, Path(inputs[name]), Path(inputs["pendulum_data"]), SPEC["cases"][case]["seeds"])
    src = pusht_source(Path(inputs["pusht_source"]))
    for stage in ("stage1", "stage2"):
        prepare_surface(run, inv, src, stage, Path(inputs[f"surface_{stage}"]))
    prepare_v22(run, inv, src, Path(inputs["pusht_v22"]))
    print(json.dumps({"prepared": sorted(str(p.parent.relative_to(run)) for p in (run / "cases").glob("*/*/rows.npz"))}))


# --- compute ----------------------------------------------------------------------------------------------------------
def strict(torch):
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def directional(torch, module, params, v, batches):
    """Per-row jvp of module(*batch) (per-row losses) along v at params, concatenated over batches, as float64."""
    from torch.func import functional_call, jvp
    out = []
    for b in batches:
        _, dl = jvp(lambda p: functional_call(module, p, b), (params,), (v,))
        out.append(dl.detach().double().cpu().numpy())
    return np.concatenate(out)


def grad_val(torch, module, params, batches, n):
    from torch.func import functional_call
    p = {k: x.detach().clone().requires_grad_(True) for k, x in params.items()}
    total = None
    for b in batches:
        loss = functional_call(module, p, b).sum() / n
        gs = torch.autograd.grad(loss, list(p.values()))
        total = gs if total is None else tuple(a + c for a, c in zip(total, gs))
    return {k: g.detach() for k, g in zip(p, total)}


def compute_pendulum(case_dir: Path, device: str):
    import torch
    from ilsb.learners.vector.vector_bc import VectorBCPolicy
    strict(torch)
    z = np.load(case_dir / "rows.npz")
    pol = VectorBCPolicy.load(checkpoint_of(case_dir), device=device)[0]

    class Loss(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.net = pol.net

        def forward(self, x, a):  # vector_bc's per-row loss: mean_j (pi(s) - a)_j^2 in scaled action space
            return ((self.net(pol.scale_obs(x)) - pol.scale_act(a)) ** 2).mean(-1)
    m = Loss().to(device).eval()
    params = {k: p.detach() for k, p in m.named_parameters()}
    t = lambda x: torch.as_tensor(x, dtype=torch.float32, device=device)  # noqa: E731
    chunks = lambda o, a: [(t(o[i:i + 4096]), t(a[i:i + 4096])) for i in range(0, len(o), 4096)]  # noqa: E731
    v = grad_val(torch, m, params, chunks(z["val_obs"], z["val_act"]), len(z["val_obs"]))
    return {"syn_g": directional(torch, m, params, v, chunks(z["syn_obs"], z["syn_act"])),
            "cmp_g": directional(torch, m, params, v, chunks(z["cmp_obs"], z["cmp_act"]))}


def compute_pusht(case_dir: Path):
    import torch
    from lerobot.policies.factory import get_policy_class, make_pre_post_processors
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    strict(torch)
    z = np.load(case_dir / "rows.npz")
    ck = checkpoint_of(case_dir)
    policy = get_policy_class(PreTrainedConfig.from_pretrained(ck).type).from_pretrained(ck).cuda().eval()
    pre, _ = make_pre_post_processors(policy.config, pretrained_path=ck)

    class Loss(torch.nn.Module):  # IBC's InfoNCE per row, as the frozen selection scorer computes it
        def __init__(self):
            super().__init__()
            self.p = policy

        def forward(self, obs, act, neg):
            return self.p.infonce(obs, act, neg)
    m = Loss().eval()
    params = {k: p.detach() for k, p in m.named_parameters()}

    def batches(part):
        n = len(z[f"{part}_act"])
        g = torch.Generator().manual_seed(PUSHT_SEED)
        neg = torch.rand(n, PUSHT_NEGATIVES, policy.d_act, generator=g) * 2 - 1
        for i in range(0, n, CHUNK):
            b = pre({FEATURES[0]: torch.as_tensor(z[f"{part}_state"][i:i + CHUNK]).cuda(),
                     FEATURES[1]: torch.as_tensor(z[f"{part}_env"][i:i + CHUNK]).cuda(),
                     FEATURES[2]: torch.as_tensor(z[f"{part}_act"][i:i + CHUNK]).cuda()})
            act = b["action"][:, 0] if b["action"].dim() == 3 else b["action"]
            yield policy.observation(b).float(), act.float(), neg[i:i + CHUNK].cuda()
    with torch.autocast("cuda", enabled=False):
        v = grad_val(torch, m, params, batches("val"), len(z["val_act"]))
        return {"syn_g": directional(torch, m, params, v, batches("syn")),
                "cmp_g": directional(torch, m, params, v, batches("cmp"))}


def compute(run: Path, pusht: bool, device: str):
    for d in sorted((run / "cases").glob("*/*")):
        if d.parent.name.startswith("pusht") != pusht or (d / "g.npz").exists():
            continue
        g = compute_pusht(d) if pusht else compute_pendulum(d, device)
        write_npz(d / "g.npz", **g)
        print(json.dumps({"case": f"{d.parent.name}/{d.name}", "syn": len(g["syn_g"]), "cmp": len(g["cmp_g"])}),
              flush=True)


# --- report (harness environment) -------------------------------------------------------------------------------------
def p_none(n: int, p: int, b: int) -> float:
    """C(N - P, b) / C(N, b): a b-row batch drawn without replacement holds no positive row."""
    if n - p < b:
        return 0.0
    lg = math.lgamma
    return math.exp(lg(n - p + 1) - lg(n - p - b + 1) - lg(n + 1) + lg(n - b + 1))


def summarize(case: str, z, g) -> dict:
    syn_g, cmp_g = g["syn_g"], g["cmp_g"]
    u = z["syn_weight"] * np.maximum(0.0, syn_g)
    cu = z["cmp_weight"] * np.maximum(0.0, cmp_g)
    tr = z["syn_transition"]
    pos = syn_g > 0
    order = np.argsort(tr, kind="stable")
    groups = np.split(pos[order], np.flatnonzero(np.diff(tr[order])) + 1)
    multi = [x for x in groups if len(x) > 1]
    n, p = len(u), int(pos.sum())
    mass = {}
    m = SPEC["mass"]["pendulum" if case.startswith("pendulum") else "pusht"]
    for a in m["alpha"]:
        b = int(math.floor(a * m["batch"] + 0.5))
        p0 = p_none(n, p, b)
        mass[str(a)] = {"requested_rows": b, "p_no_positive": p0, "expected_realized_alpha": a * (1 - p0)}
    return {"synthetic": {"rows": n, "positive": p, "zero": n - p, "mean_u": float(u.mean()), "mean_g": float(syn_g.mean()),
                          "transitions": len(groups), "transitions_with_several_rows": len(multi),
                          "conflicting_transitions": int(sum(1 for x in multi if x.any() and not x.all())),
                          "recorded": {"mask_rows": int(z["syn_mask"].sum()),
                                       "weights": sorted({float(w) for w in np.unique(z["syn_weight"])}),
                                       "finite_error_rows": int(np.isfinite(z["syn_error"]).sum())}},
            "comparator": {"rows": len(cu), "positive": int((cmp_g > 0).sum()), "mean_u": float(cu.mean()),
                           "mean_g": float(cmp_g.mean())},
            "D": float(u.mean() - cu.mean()), "validation_rows": int(len(z["val_act"])),
            "validation_episodes": [str(e) for e in z["val_episodes"]], "mass": mass}


def report(run: Path) -> dict:
    out = {}
    for d in sorted((run / "cases").glob("*/*")):
        z, g = np.load(d / "rows.npz"), np.load(d / "g.npz")
        out.setdefault(d.parent.name, {})[d.name] = {
            "rows_content_sha256": content_sha(d / "rows.npz"), "g_content_sha256": content_sha(d / "g.npz"),
            "checkpoint_sha256": str(z["checkpoint_sha256"]), **summarize(d.parent.name, z, g)}
    checks = {}
    for k, c in SPEC["checks"].items():
        if k == "pass":
            continue
        ds = [s["D"] for case in c["cases"] for s in out[case].values()]
        checks[k] = {"D": ds, "holds": (sum(x > 0 for x in ds) >= 8 and len(ds) == 10) if k == "F1"
                     else all(x <= 0 for x in ds)}
    passed = checks["F1"]["holds"] and checks["F2"]["holds"]
    head = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
    dirty = subprocess.run(["git", "-C", str(REPO), "status", "--porcelain"], capture_output=True, text=True).stdout
    return {"schema": "ilsb.reweight.stage_a 1", "commit": head, "dirty": bool(dirty.strip()),
            "inventory_sha256": sha(INVENTORY), "spec_sha256": sha(REPO / "tools/reweight_stage_a.json"),
            "cases": out, "checks": checks,
            "decision": "pass: a Stage B protocol PR may be opened" if passed else "stop: the investigation ends in #40"}


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("prepare")
    a.add_argument("--run", type=Path, required=True)
    a.add_argument("--inputs", type=Path, required=True)
    b = sub.add_parser("compute")
    b.add_argument("--run", type=Path, required=True)
    b.add_argument("--pusht", action="store_true")
    b.add_argument("--device", default="cuda")
    c = sub.add_parser("report")
    c.add_argument("--run", type=Path, required=True)
    c.add_argument("--out", type=Path, required=True)
    args = p.parse_args()
    if args.cmd == "prepare":
        prepare(args.run.resolve(), json.loads(args.inputs.read_text()))
    elif args.cmd == "compute":
        compute(args.run.resolve(), args.pusht, args.device)
    else:
        rep = report(args.run.resolve())
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(rep, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"checks": {k: v["holds"] for k, v in rep["checks"].items()}, "decision": rep["decision"]}))


if __name__ == "__main__":
    main()
