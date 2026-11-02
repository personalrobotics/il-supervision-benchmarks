#!/usr/bin/env python3
"""The Pendulum recovery-tube study (docs/studies/pendulum-recovery-tube.md, version 1 with amendments 1-3). Exploratory.

    uv run python tools/recovery_tube.py audit    --out DIR [--capture RUN] [--corrected RUN --corrected-eval RUN
                                                  --ccil-evidence DIR] [--response-surface RUN]
    uv run python tools/recovery_tube.py geometry --out DIR [--resume]
    uv run python tools/recovery_tube.py evaluate --out DIR --geometry-sha256 SHA --cohort COHORT [--resume]
    uv run python tools/recovery_tube.py decide   --out DIR
    uv run python tools/recovery_tube.py selftest --out FILE.json

audit reads the input runs without writing to them, pins the evidence it trusts, and decides each cohort's eligibility
before any policy is loaded. geometry (phase 1) measures the corrected cohort's label tube, builds the references,
anchors and feasible disturbances, and freezes the target tube, dose bins and testability in geometry.json and
geometry.npz; it loads no policy. evaluate (phase 2) refuses unless geometry.json has the given sha256; COHORT is
capture, corrected, matched (replay-CCIL and duplication) or response_surface. decide applies the sequential decision
rules to the exact declared arm x seed matrix. Every stage but selftest refuses a dirty tree and, after audit, any
evaluator source other than the audit's. Each per-policy result carries its identity (policy, geometry, audit, inputs,
evaluator source) and its NPZ's hash; resume and decide refuse missing, extra, stale or corrupted results. selftest
runs the synthetic and off-protocol fixtures and writes their checks; it reads no real checkpoint.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
import positive_control as pc  # noqa: E402

from ilsb.analysis import union_interval  # noqa: E402
from ilsb.store import digest, sha256_file  # noqa: E402
from ilsb.tasks.pendulum import UPRIGHT_COS, ExactModel, Task, expert  # noqa: E402

PROTOCOL = "docs/studies/pendulum-recovery-tube.md (version 1, amendments 1-3)"
G, DT = 9.81, 0.02
TAU_STAR = 0.038357  # the one-step exact-preimage bound (CCIL pendulum-correction.md); a reference line only
REF_SEED, N_REF, REF_STEPS, ANCHOR_FRAMES = 2301, 20, 1000, tuple(range(0, 500, 50))
EARLIER_EVAL_SEEDS = tuple(range(2201, 2221))
OFFSETS, DURATIONS, SIGNS = (0.1875, 0.375, 0.75, 1.5, 3.0, 6.0), (1, 4, 16), (1, -1)
H, ONSET_MAX, LAG, RHO_CAPTURE, LQR = 300, 250, 25, 0.1, 0.1  # the shortest stay is H - ONSET_MAX = 50 steps
LAGS = np.array([0] + [v for i in range(1, LAG + 1) for v in (-i, i)])  # tie order: smallest |s|, negative first
NULL_DELTA, MIN_EDGE, MIN_REPORT, CONFIDENCE = 1e-9, 200, 20, 0.95
DATA_SHA256 = pc.DATA_SHA256
CORRECTED_SEEDS, CORRECTED_ARMS, MATCHED_ARMS = tuple(range(1201, 1221)), ("bc", "ccil"), ("replay_ccil", "duplication")
RS_ARMS, RS_K = ("bc", "ccil_exact", "ccil_learned"), {"bc": 0, "ccil_exact": 20, "ccil_learned": 20}
FROZEN_CCIL, EVALUATE_COMMIT = "6fbcce543e8a1ebdf243063c179dd3af82d5db8a", "e1ca5b5b14b6541981c8e178f088472d115de858"
CCIL_EVIDENCE_SHA256 = {"study-audit.json": "14bea907899e100c2dacc018061e1c6efbeea621f29d0fbbd16fe78f2813487e",
                        "study-evaluate.json": "6a1f154b55a79524f8db2ad4b9cb542b8a8519d197d8f778c7a726845ed3e19d"}
TOOL_FILES = ("tools/recovery_tube.py", "tools/recovery_tube_dup.py", "tools/positive_control.py",
              "src/ilsb/tasks/pendulum.py", "src/ilsb/tasks/pendulum_map.py", "src/ilsb/analysis.py", "src/ilsb/store.py")
HOME = Path("/home/siddh/code")
DEFAULTS = {"capture": HOME / "il-supervision-benchmarks/runs/20261102T003101Z-capture-control",
            "corrected": HOME / "CCIL/runs/20261029T191145Z-corrected-study",
            "corrected_eval": HOME / "CCIL/runs/20261030T013609Z-corrected-eval",
            "ccil_evidence": HOME / "CCIL/docs/fixtures/evidence/pendulum-corrected-demonstrator",
            "response_surface": HOME / "il-supervision-benchmarks/runs/20261031T141026Z-j-r-k"}
CAPTURE_EVIDENCE = pc.REPO / "docs/studies/evidence/pendulum-capture-control"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
log, Heartbeat, atomic_json, atomic_npz = pc.log, pc.Heartbeat, pc.atomic_json, pc.atomic_npz


class Refused(SystemExit):
    pass


# --- coordinates ----------------------------------------------------------------------------------------------------
def theta(s):
    return np.arctan2(s[..., 0], s[..., 1])


def wrap(a):
    return np.mod(a + np.pi, 2 * np.pi) - np.pi


def dist(x, y):
    """d(x, y) = sqrt(wrap(theta_x - theta_y)^2 + ((omega_x - omega_y) / sqrt(g))^2)."""
    return np.sqrt(wrap(theta(x) - theta(y)) ** 2 + ((x[..., 2] - y[..., 2]) / np.sqrt(G)) ** 2)


def tree(states):
    """A periodic KD-tree in (theta, omega / sqrt(g)), whose Euclidean distance is d."""
    return cKDTree(_emb(states), boxsize=[2 * np.pi, 1e6])


def _emb(states):
    return np.stack([np.mod(theta(states), 2 * np.pi), states[..., 2] / np.sqrt(G) + 5e5], -1).reshape(-1, 2)


def upright(s):
    return np.cos(theta(s) - np.pi) >= UPRIGHT_COS


# --- code-state identity ----------------------------------------------------------------------------------------------
def tool_sha256() -> str:
    return digest({f: sha256_file(pc.REPO / f) for f in TOOL_FILES})


def require_clean():
    if pc.git("status", "--porcelain"):
        raise Refused("refused: the harness tree is dirty; studies run from a clean, committed tree")


def require_audited_tool(out: Path) -> dict:
    aud = json.loads((out / "audit.json").read_text())
    if aud["tool_sha256"] != tool_sha256():
        raise Refused("refused: the evaluator source differs from the one that wrote audit.json")
    return aud


# --- audit (read-only) ----------------------------------------------------------------------------------------------
def audit_capture(run: Path, manifest: dict, report: dict) -> dict:
    bad = [k for k, h in manifest["artifacts_sha256"].items()
           if not (run / k).is_file() or sha256_file(run / k) != h]
    bad += [f"policy {k}" for k, h in report["policies"].items()
            if manifest["artifacts_sha256"].get(f"policies/{k}/policy.npz") != h]
    return {"run": str(run), "eligible": not bad and report.get("status") == "completed", "failures": bad,
            "seeds": report.get("qualifying_seeds", []), "arms": list(pc.ARMS), "policy_sha256": report["policies"],
            "held_out_sha256": {str(s): manifest["artifacts_sha256"].get(f"select/s{s}.npz")
                                for s in report.get("qualifying_seeds", [])}}


def evidence_problems(audit_p: Path, eval_p: Path) -> list[str]:
    """The committed CCIL evidence, pinned by hash, with the gates amendment 3 requires."""
    out = [f"{p.name} is not the pinned committed file" for p in (audit_p, eval_p)
           if not p.is_file() or sha256_file(p) != CCIL_EVIDENCE_SHA256[p.name]]
    if out:
        return out
    for p in (audit_p, eval_p):
        d = json.loads(p.read_text())
        g, proc = d.get("gate", {}), d.get("process", {})
        if not (g.get("ccil_frozen") == FROZEN_CCIL and g.get("differing_from_frozen_ccil") == [] and g.get("dirty") == []
                and g.get("ok") is True and g.get("real") is True and g.get("tooling", {}).get("ok") is True
                and proc.get("dirty") == [] and d.get("all_pass") is True):
            out.append(f"{p.name}: gate, clean state or all_pass is not as required")
    ev = json.loads(eval_p.read_text())
    if ev["gate"].get("head") != EVALUATE_COMMIT or ev["process"].get("commit") != EVALUATE_COMMIT:
        out.append(f"study-evaluate.json is not from the evaluate step's commit {EVALUATE_COMMIT}")
    return out


def audit_corrected(run: Path, eval_run: Path, evidence: Path) -> dict:
    """Amendment 1's chain for every continuous BC and CCIL unit, against the pinned evidence; all-or-nothing."""
    bad = evidence_problems(evidence / "study-audit.json", evidence / "study-evaluate.json")
    units = {}
    if bad:
        return {"run": str(run), "eligible": False, "failures": bad, "units": units}
    study_audit = json.loads((evidence / "study-audit.json").read_text())
    study_eval = json.loads((evidence / "study-evaluate.json").read_text())
    root = run.resolve()
    for arm in CORRECTED_ARMS:
        for s in CORRECTED_SEEDS:
            key, f = f"continuous/{arm}/s{s}", []
            rec_p = run / "results/study/continuous" / arm / f"s{s}.json"
            if not rec_p.is_file():
                bad.append(f"{key}: no unit record")
                continue
            rec = json.loads(rec_p.read_text())
            if rec.get("status") != "ok" or study_audit["statuses"].get(key) != "ok":
                f.append("status is not ok in the record and the committed audit")
            stage = Path(rec["stages"]["policy.train"]).resolve()
            inside = root in stage.parents
            if not inside:
                f.append("policy.train stage is outside the run")
            man = json.loads((stage / "manifest.json").read_text()) if (stage / "manifest.json").is_file() else {}
            ck = stage / "checkpoints/policy.npz"
            ck_sha = sha256_file(ck) if ck.is_file() else None
            if not (ck_sha and ck_sha == man.get("files", {}).get("checkpoints/policy.npz", {}).get("sha256")
                    == rec.get("checkpoint_sha256")):
                f.append("checkpoint, stage manifest and record hashes differ")
            if not (inside and study_audit["stages"].get(str(stage.relative_to(root)), {}).get("pass")):
                f.append("policy.train stage is not a passing stage of the committed audit")
            roll = eval_run / "continuous" / arm / f"s{s}-0.0.npz"
            if not roll.is_file() or sha256_file(roll) != study_eval["rollout_sha256"].get(f"{key}/0.0"):
                f.append("evaluate rollout does not match the committed rollout hash")
            inputs = man.get("inputs", {})
            if inputs.get("dataset", {}).get("sha256") != DATA_SHA256:
                f.append("dataset input is not the canonical continuous dataset")
            unit = {"checkpoint": str(ck), "checkpoint_sha256": ck_sha, "stage": str(stage),
                    "dataset": inputs.get("dataset", {}).get("path")}
            if arm == "ccil":
                cand = Path(rec["stages"]["labels.generate"]) / "labels/candidates.npz"
                want = inputs.get("candidates", {}).get("sha256")
                if not cand.is_file() or sha256_file(cand) != want:
                    f.append("candidates do not match the policy.train input hash")
                unit.update(candidates=str(cand), candidates_sha256=want)
            units[key] = unit
            bad += [f"{key}: {x}" for x in f]
    return {"run": str(run), "eval_run": str(eval_run), "eligible": not bad, "failures": bad, "units": units,
            "evidence_sha256": CCIL_EVIDENCE_SHA256, "seeds": list(CORRECTED_SEEDS), "arms": list(CORRECTED_ARMS)}


def _references(obj, path=()):
    """(path, hash) for every 64-hex string, skipping environment identity (scope, code)."""
    if isinstance(obj, str):
        if HEX64.match(obj):
            yield path, obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("scope", "code"):
                continue
            if HEX64.match(k):
                yield path, k
            yield from _references(v, (*path, k))
    elif isinstance(obj, list):
        for v in obj:
            yield from _references(v, path)


def audit_store(run: Path, anchors: dict | None = None) -> dict:
    """Internal completeness of a content-addressed run; never external anchoring. Every reference outside the
    environment identity must resolve to a store address, a store file, a stage file, digest(root.json) (the protocol)
    or an anchor the tool computed from bytes it read (the dataset file, the spec at the harness commit, the exact
    model). Root hashes are not accepted for being in root.json."""
    problems, addresses, file_shas = [], set(), set()
    for kind in sorted((run / "store").iterdir()):
        for e in sorted(kind.iterdir()):
            ident = json.loads((e / "identity.json").read_text())
            if digest(ident) != e.name:
                problems.append(f"{kind.name}/{e.name}: address is not the identity digest")
            files = json.loads((e / "manifest.json").read_text())
            present = {p.name for p in e.iterdir()} - {"identity.json", "manifest.json"}
            if present != set(files):
                problems.append(f"{kind.name}/{e.name}: files differ from its manifest")
            for name, m in files.items():
                if (e / name).is_file() and sha256_file(e / name) != m["sha256"]:
                    problems.append(f"{kind.name}/{e.name}/{name}: sha256 differs from its manifest")
                file_shas.add(m["sha256"])
            addresses.add(e.name)
    stage_shas = set()
    for name, h in json.loads((run / "manifest.json").read_text())["stages"].items():
        p = run / "stages" / f"{name}.json"
        got = sha256_file(p) if p.is_file() else None
        stage_shas.add(got)
        if got != h:
            problems.append(f"stage {name}: sha256 differs from the run manifest")
    root = json.loads((run / "root.json").read_text())
    known = addresses | file_shas | stage_shas | {digest(root)} | set((anchors or {}).values())
    unresolved = {}
    for src in [run / "root.json", run / "manifest.json", *sorted((run / "stages").glob("*.json")),
                *sorted((run / "store").glob("*/*/identity.json"))]:
        for path, h in _references(json.loads(src.read_text())):
            if h not in known:
                unresolved.setdefault("/".join(re.sub(r"\d", "#", p) for p in path), set()).add(h)
    problems += [f"unresolved reference at {k} ({len(v)} distinct)" for k, v in sorted(unresolved.items())]
    units = {}
    train = json.loads((run / "stages/train.json").read_text())
    r_max = max(u["r"] for u in train["units"].values())
    for arm in RS_ARMS:
        for key, u in train["units"].items():
            if u["arm"] == arm and u["r"] == r_max and u["k"] == RS_K[arm] and key in train["analysis"].values():
                p = run / "store/policy" / u["entry"] / "policy.npz"
                if not p.is_file() or sha256_file(p) != u["policy_sha256"]:
                    problems.append(f"{key}: policy file does not match the train stage")
                units[key] = {"arm": arm, "seed": u["seed"], "sigma": u["sigma"], "policy": str(p),
                              "policy_sha256": u["policy_sha256"], "entry": u["entry"]}
    return {"run": str(run), "eligible": not problems, "failures": problems[:200], "failure_count": len(problems),
            "units": units, "r": r_max, "anchors": anchors or {},
            "anchoring": "internal consistency only; no root result was committed before 2026-11-02, so this is not "
                         "prior external anchoring"}


def store_anchors(run: Path) -> dict:
    """Hashes the tool computes from bytes it reads, for audit_store."""
    root = json.loads((run / "root.json").read_text())
    out = {"exact_model": ExactModel(Task()).sha256}
    if sha256_file(pc.DATA) == DATA_SHA256:
        out["dataset"] = DATA_SHA256
    commit = root.get("code", {}).get("harness_commit")
    spec = subprocess.run(["git", "show", f"{commit}:studies/pendulum-j-r-k/study.yaml"], cwd=pc.REPO,
                          capture_output=True) if commit else None
    if spec is not None and spec.returncode == 0:
        out["spec"] = hashlib.sha256(spec.stdout).hexdigest()
    return out


def stage_audit(a) -> dict:
    out = {"protocol": PROTOCOL, "code_commit": pc.git("rev-parse", "HEAD"), "tool_sha256": tool_sha256()}
    cap = a.capture or DEFAULTS["capture"]
    out["capture"] = audit_capture(cap, json.loads((CAPTURE_EVIDENCE / "artifact-manifest.json").read_text()),
                                   json.loads((CAPTURE_EVIDENCE / "capture-control.json").read_text()))
    log("capture cohort eligible:", out["capture"]["eligible"], out["capture"]["failures"][:5])
    out["corrected"] = audit_corrected(a.corrected or DEFAULTS["corrected"],
                                       a.corrected_eval or DEFAULTS["corrected_eval"],
                                       a.ccil_evidence or DEFAULTS["ccil_evidence"])
    log("corrected cohort eligible:", out["corrected"]["eligible"], out["corrected"]["failures"][:5])
    rs = a.response_surface or DEFAULTS["response_surface"]
    out["response_surface"] = audit_store(rs, store_anchors(rs)) if rs.exists() else {
        "run": str(rs), "eligible": False, "failures": ["run not present"]}
    log("response-surface cohort internally complete:", out["response_surface"]["eligible"],
        out["response_surface"]["failures"][:5])
    return out


# --- phase 1: references, anchors, feasible disturbances, label geometry, dose bins ----------------------------------
def references(seed=REF_SEED, n=N_REF, steps=REF_STEPS, check_disjoint=True):
    starts = Task(eval_seed=seed, eval_episodes=n).starts()
    if check_disjoint:
        for e in EARLIER_EVAL_SEEDS:
            if (pc.phys(starts)[:, None] == pc.phys(Task(eval_seed=e).starts())[None]).all(-1).any():
                raise Refused(f"refused: reference start set {seed} shares a start with eval seed {e}")
    task, x, out = Task(), starts, [starts]
    for _ in range(steps):
        x = task.step(x, expert(x))[0]
        out.append(x)
    return np.stack(out, 1)


def disturbances(refs, frames=ANCHOR_FRAMES):
    """Every anchor x = refs[i, t]: the undisturbed case, then each (k, b, sign) with u = clip(pi_D(x) + sign b, -3, 3)
    for k steps. Returns the initial states, their reference index and start frame, the design, and the largest applied
    action magnitude."""
    task = Task()
    idx = [(i, t) for i in range(len(refs)) for t in frames]
    rows = {"x0": [refs[i, t] for i, t in idx], "ref": [i for i, _ in idx], "t0": [t for _, t in idx],
            "anchor_t": [t for _, t in idx], "b": [0.0] * len(idx), "k": [0] * len(idx), "sign": [0] * len(idx),
            "delta": [0.0] * len(idx)}
    umax = 0.0
    for k in DURATIONS:
        for b in OFFSETS:
            for sg in SIGNS:
                x = np.array([refs[i, t] for i, t in idx])
                for _ in range(k):
                    u = np.clip(expert(x) + sg * b, -3.0, 3.0)
                    umax = max(umax, float(np.abs(u).max()))
                    x = task.step(x, u)[0]
                end = np.array([refs[i, t + k] for i, t in idx])
                rows["x0"] += list(x)
                rows["ref"] += [i for i, _ in idx]
                rows["t0"] += [t + k for _, t in idx]
                rows["anchor_t"] += [t for _, t in idx]
                rows["b"] += [b] * len(idx)
                rows["k"] += [k] * len(idx)
                rows["sign"] += [sg] * len(idx)
                rows["delta"] += list(dist(x, end))
    out = {k: np.asarray(v) for k, v in rows.items()}
    out["stratum"] = np.where(np.abs(wrap(theta(refs[out["ref"], out["anchor_t"]]) - np.pi)) < LQR, "lqr_branch",
                              "swing")
    return out, umax


def label_geometry(cands_path, dataset_path) -> tuple[dict, np.ndarray, np.ndarray]:
    """Descriptive label geometry of the accepted rows; returns (summary, d_source, synthetic states)."""
    from ccil.schema import read_candidates, read_trajectories
    ds = read_trajectories(dataset_path)
    c = read_candidates(cands_path)
    acc = np.flatnonzero(c.accepted)
    z = np.asarray(c.observations["state"][acc], np.float64)
    a = np.asarray(c.actions[acc], np.float64).reshape(len(acc), -1)[:, 0]
    tgt = np.asarray(c.target_observations["state"][acc], np.float64)
    ep_ids, fr = c.source_episode_id[acc].astype(str), np.asarray(c.source_frame[acc])
    xs, tan = np.full_like(z, np.nan), np.full((len(acc), 2), np.nan)
    own = np.full(len(acc), np.nan)
    demo = []
    for e in ds.episodes:
        X = np.asarray(e.observations["state"], np.float64)
        demo.append(X)
        m = np.flatnonzero(ep_ids == e.episode_id)
        if not len(m):
            continue
        xs[m] = X[fr[m]]
        lo, hi = np.maximum(fr[m] - 1, 0), np.minimum(fr[m] + 1, len(X) - 1)
        v = np.stack([wrap(theta(X[hi]) - theta(X[lo])), (X[hi, 2] - X[lo, 2]) / np.sqrt(G)], -1)
        n = np.linalg.norm(v, axis=1, keepdims=True)
        tan[m] = np.where(n > 0, v / np.where(n > 0, n, 1), np.nan)
        own[m] = tree(X).query(_emb(z[m]))[0]
    if np.isnan(xs).any():
        raise Refused("refused: a candidate's source episode is not in the dataset")
    d_source = dist(z, xs)
    disp = np.stack([wrap(theta(z) - theta(xs)), (z[:, 2] - xs[:, 2]) / np.sqrt(G)], -1)
    along = (disp * tan).sum(1)
    across = np.linalg.norm(disp - along[:, None] * tan, axis=1)
    whole = tree(np.concatenate(demo)).query(_emb(z))[0]
    task, resid = Task(), np.empty(len(acc))
    for b in range(0, len(acc), 200_000):
        nxt = task.step(z[b:b + 200_000], a[b:b + 200_000])[0]
        resid[b:b + 200_000] = dist(nxt, tgt[b:b + 200_000])
    q = lambda v: {"q50": float(np.nanquantile(v, 0.5)), "q95": float(np.nanquantile(v, 0.95)),  # noqa: E731
                   "max": float(np.nanmax(v))} if np.isfinite(v).any() else None
    summary = {"candidates_sha256": sha256_file(cands_path), "rows": int(len(c.accepted)), "accepted": int(len(acc)),
               "d_source": q(d_source), "d_source_zero_fraction": float((d_source == 0).mean()),
               "tangent_abs": q(np.abs(along)), "transverse": q(across), "tangent_undefined": int(np.isnan(along).sum()),
               "distance_to_source_episode": q(own), "distance_to_demonstrations": q(whole),
               "exact_residual": q(resid), "action_within_bound": float((np.abs(a) <= 3).mean()),
               "measurements": {k: q(np.asarray(v, np.float64)[acc]) for k, v in sorted(c.measurements.items())}}
    return summary, d_source, z


def dose_bins(delta, q95):
    """inside (0, q95], edge (q95, 2 q95], then beyond_j (2^j q95, 2^(j+1) q95] for j >= 1; null below NULL_DELTA."""
    lab = np.full(len(delta), "", dtype=object)
    lab[delta < NULL_DELTA] = "null"
    pos = delta >= NULL_DELTA
    lab[pos & (delta <= q95)] = "inside"
    lab[pos & (delta > q95) & (delta <= 2 * q95)] = "edge"
    far = pos & (delta > 2 * q95)
    j = np.floor(np.log2(np.where(far, delta, 1) / q95 - 1e-12)).astype(int)
    lab[far] = np.char.add("beyond_", j[far].astype(str))
    return lab.astype(str)


def stage_geometry(out: Path, resume: bool):
    aud = require_audited_tool(out)
    path = out / "geometry.json"
    if path.exists():
        raise Refused(f"refused: {path} exists; phase 1 is frozen")
    refs = references()
    dist_, umax = disturbances(refs)
    doc = {"protocol": PROTOCOL, "audit_sha256": sha256_file(out / "audit.json"), "tool_sha256": tool_sha256(),
           "references": {"eval_seed": REF_SEED, "episodes": N_REF, "steps": REF_STEPS},
           "anchor_frames": list(ANCHOR_FRAMES), "offsets": list(OFFSETS), "durations": list(DURATIONS),
           "applied_action_max_abs": umax, "tau_star_reference": TAU_STAR,
           "recovery": {"horizon": H, "onset_max": ONSET_MAX, "lag": LAG, "rho_capture": RHO_CAPTURE},
           "code_commit": pc.git("rev-parse", "HEAD")}
    arrays = {"refs": refs, **dist_}
    disturbed = dist_["k"] > 0
    if aud["corrected"]["eligible"]:
        per, d_all, zs = {}, [], {}
        (out / "geometry").mkdir(exist_ok=True)
        with Heartbeat("label geometry"):
            for s in CORRECTED_SEEDS:
                u = aud["corrected"]["units"][f"continuous/ccil/s{s}"]
                p = out / "geometry" / f"s{s}.npz"
                if p.exists() and not resume:
                    raise Refused(f"refused: {p} exists (an interrupted run takes --resume)")
                if not p.exists():
                    summary, d_src, z = label_geometry(Path(u["candidates"]), Path(u["dataset"]))
                    atomic_npz(p, d_source=d_src, z=z, summary=json.dumps(summary))
                    log("geometry", s, json.dumps(summary["d_source"]))
                f = np.load(p)
                per[str(s)], zs[s] = json.loads(str(f["summary"])), f["z"]
                if per[str(s)]["candidates_sha256"] != u["candidates_sha256"]:
                    raise Refused(f"refused: {p} is not from the audited candidates")
                d_all.append(f["d_source"])
        d_all = np.concatenate(d_all)
        tube = {"q50": float(np.quantile(d_all, 0.5)), "q95": float(np.quantile(d_all, 0.95)),
                "max": float(d_all.max()), "rows": int(len(d_all))}
        if tube["q95"] > 0:
            bins = dose_bins(dist_["delta"], tube["q95"])
        else:
            bins = np.where(dist_["delta"] >= NULL_DELTA, "beyond_inf", "null")
        bins = np.where(disturbed, bins, "undisturbed")
        counts = {b: int((bins == b).sum()) for b in sorted(set(bins))}
        overlap = {}
        if tube["q95"] > 0:
            for b in sorted(set(bins) - {"undisturbed", "null"}):
                m = bins == b
                overlap[b] = float(np.mean([(tree(zs[s]).query(_emb(dist_["x0"][m]))[0] <= tube["q95"]).mean()
                                            for s in CORRECTED_SEEDS]))
        doc.update(label_geometry={"per_seed": per, "pooled": tube},
                   rho_corrected=tube["q95"] if tube["q95"] > 0 else None, bins=counts,
                   testable=bool(tube["q95"] > 0 and counts.get("edge", 0) >= MIN_EDGE), overlap_within_q95=overlap,
                   testability_rule=f"q95 > 0 and the edge bin holds >= {MIN_EDGE} states")
        arrays["bin"] = bins
    else:
        doc.update(label_geometry=None, rho_corrected=None, testable=False,
                   reason="the corrected cohort is excluded by the audit")
        arrays["bin"] = np.where(disturbed, "unbinned", "undisturbed")
    if aud["response_surface"]["eligible"]:
        rs, run = {}, Path(aud["response_surface"]["run"])
        for key, u in aud["response_surface"]["units"].items():
            if u["arm"] == "bc":
                continue
            ident = json.loads((Path(u["policy"]).parent / "identity.json").read_text())
            cand = run / "store/candidates" / ident["candidates"] / "candidates.npz"
            sub = json.loads((cand.parent / "identity.json").read_text())["subset"]
            data = next((run / "store/subset" / sub).glob("*.npz"))
            rs[key] = label_geometry(cand, data)[0]["d_source"]
        doc["response_surface_label_geometry"] = rs
    gz = out / "geometry.npz"
    atomic_npz(gz, **arrays)
    doc["geometry_npz_sha256"] = sha256_file(gz)
    atomic_json(path, doc)
    log("phase 1 frozen: geometry.json", sha256_file(path), "testable:", doc["testable"], doc.get("bins"))


# --- phase 2: rollouts and recovery -----------------------------------------------------------------------------------
def rollout(pol, x0, steps=H):
    """The policy from each state for steps; non-finite actions mark the rollout failed and apply 0 thereafter."""
    task, N = Task(), len(x0)
    S, C, bad = np.zeros((N, steps + 1, 3)), np.zeros((N, steps)), np.zeros(N, bool)
    S[:, 0] = x0
    for t in range(steps):
        u = np.asarray(pol(S[:, t]), np.float64).reshape(N)
        bad |= ~np.isfinite(u)
        u = np.where(bad, 0.0, u)
        C[:, t] = u
        S[:, t + 1] = task.step(S[:, t], u)[0]
    return S, C, bad


def recovery(S, R, off, rho, chunk=400):
    """Fixed-lag recovery (amendment 2) of rollouts S (N, H+1, 3) against reference continuations R (N, L, 3), each
    rollout starting at index off[n] of its reference. Recovered: one lag |s| <= LAG and an onset j <= ONSET_MAX with
    d(x_m, r_{m+s}) <= rho for every m = j..H, so the rollout enters the tube and stays to the horizon's end; a lag
    needing a frame outside R is unavailable."""
    N, L = len(S), R.shape[1]
    rec, onset, lag = np.zeros(N, bool), np.full(N, -1), np.zeros(N, int)
    integ = np.zeros(N)
    j = np.arange(H + 1)
    for b in range(0, N, chunk):
        n = np.arange(b, min(b + chunk, N))
        idx = off[n, None, None] + j[None, :, None] + LAGS[None, None, :]
        valid = (idx >= 0) & (idx < L)
        D = dist(S[n, :, None, :], R[n[:, None, None], np.clip(idx, 0, L - 1)])
        D = np.where(valid, D, np.inf)
        integ[n] = np.where(np.isfinite(D.min(2)), D.min(2), 0).sum(1) * DT
        stay = np.flip(np.cumprod(np.flip(D <= rho, 1), 1), 1).astype(bool)  # stay[n, j, s]: in the tube from j to H
        win = stay[:, :ONSET_MAX + 1]
        anyj = win.any(2)
        has = anyj.any(1)
        first = np.argmax(anyj, 1)
        rec[n] = has
        onset[n] = np.where(has, first, -1)
        lag[n] = np.where(has, LAGS[np.argmax(win[np.arange(len(n)), first], 1)], 0)
    return rec, onset, lag, integ


def evaluate_states(pol, x0, R, off, rho):
    S, C, bad = rollout(pol, x0)
    rec, onset, lag, integ = recovery(S, R, off, rho)
    rec &= ~bad
    up = upright(S[:, 1:])
    return {"recovered": rec, "onset": onset, "lag": lag, "integrated": integ, "failed": bad,
            "final_hold": up[:, -pc.HOLD:].all(1), "occupancy": up.mean(1),
            "saturated": (np.abs(C) >= 3).mean(1)}


def _mean(v):
    return float(v.mean()) if len(v) else None


def summarize(r: dict, g: dict) -> dict:
    """Per-policy summary over the frozen inputs: success per bin, stratum, k and sign; the primary edge-bin recovery
    and the inside-bin retention (amendment 3)."""
    rec, bins = r["recovered"], g["bin"]
    out = {"undisturbed_tracking": _mean(rec[bins == "undisturbed"]), "failed": int(r["failed"].sum()),
           "saturated_fraction": float(r["saturated"].mean()), "bins": {}, "strata": {}, "k": {}, "sign": {}}
    for b in sorted(set(bins) - {"undisturbed", "null"}):
        m = bins == b
        if m.sum() >= MIN_REPORT or b in ("inside", "edge"):
            out["bins"][b] = {"n": int(m.sum()), "recovered": _mean(rec[m]),
                              "onset_mean_s": _mean(r["onset"][m & rec] * DT), "integrated_mean": _mean(r["integrated"][m])}
    dis = g["k"] > 0
    for st in ("lqr_branch", "swing"):
        m = dis & (g["stratum"] == st)
        out["strata"][st] = {"n": int(m.sum()), "recovered": _mean(rec[m]),
                             "final_hold": _mean(r["final_hold"][m]) if st == "lqr_branch" else None}
    for k in DURATIONS:
        out["k"][str(k)] = _mean(rec[g["k"] == k])
    for sg in SIGNS:
        out["sign"][str(sg)] = _mean(rec[g["sign"] == sg])
    out["primary_edge_recovery"] = out["bins"].get("edge", {}).get("recovered")
    out["inside_retention"] = out["bins"].get("inside", {}).get("recovered")
    full = [v["recovered"] for v in out["bins"].values() if v["recovered"] is not None]
    out["full_curve_log_grid_mean"] = float(np.mean(full)) if full else None
    return out


def load_geometry(out: Path, want: str) -> dict:
    gj = out / "geometry.json"
    if not gj.is_file() or sha256_file(gj) != want:
        raise Refused(f"refused: {gj} is not the frozen phase-1 record {want}")
    doc = json.loads(gj.read_text())
    if sha256_file(out / "geometry.npz") != doc["geometry_npz_sha256"]:
        raise Refused("refused: geometry.npz does not match geometry.json")
    f = np.load(out / "geometry.npz")
    return {"doc": doc, "sha256": want, **{k: f[k] for k in f.files}}


def pi_d_continuations(x, steps=H + LAG + 1):
    task, s, out = Task(), np.array(x, np.float64), [np.array(x, np.float64)]
    for _ in range(steps):
        s = task.step(s, expert(s))[0]
        out.append(s)
    return np.stack(out, 1)


def expected_units(aud: dict, cohort: str, out: Path) -> dict:
    """The declared arm x seed matrix of a cohort: {(arm, seed): (policy path, policy sha256)}."""
    if cohort == "capture":
        run = Path(aud["capture"]["run"])
        return {(arm, s): (run / "policies" / arm / f"s{s}" / "policy.npz", aud["capture"]["policy_sha256"][f"{arm}/s{s}"])
                for s in aud["capture"]["seeds"] for arm in pc.ARMS}
    if cohort == "corrected":
        return {(arm, s): (Path(u["checkpoint"]), u["checkpoint_sha256"]) for arm in CORRECTED_ARMS
                for s in CORRECTED_SEEDS for u in [aud["corrected"]["units"][f"continuous/{arm}/s{s}"]]}
    if cohort == "matched":
        man = json.loads((out / "matched/manifest.json").read_text())
        return {(arm, s): (out / "matched/policies" / arm / f"s{s}" / "policy.npz",
                           man["policies"][arm][str(s)]["sha256"]) for arm in MATCHED_ARMS for s in CORRECTED_SEEDS}
    if cohort == "response_surface":
        return {(u["arm"], u["seed"]): (Path(u["policy"]), u["policy_sha256"])
                for u in aud["response_surface"]["units"].values()}
    raise Refused(f"unknown cohort {cohort}")


def cohort_rho(cohort: str, g: dict) -> float:
    """The target tube: q95 wherever a measured tube exists (the capture anchors are descriptive), else 0.1."""
    return g["doc"]["rho_corrected"] if g["doc"].get("rho_corrected") is not None else RHO_CAPTURE


def result_identity(aud_sha, g, cohort, arm, seed, policy_sha, rho, held_sha=None) -> dict:
    return {"protocol": PROTOCOL, "cohort": cohort, "arm": arm, "seed": seed, "policy_sha256": policy_sha,
            "geometry_sha256": g["sha256"], "audit_sha256": aud_sha, "tool_sha256": tool_sha256(),
            "inputs": {"geometry_npz_sha256": g["doc"]["geometry_npz_sha256"], "held_out_capture_sha256": held_sha},
            "rho": rho}


def verified_result(d: Path, name: str, identity: dict) -> dict | None:
    """The stored result for this identity, None if absent; refused if stale or corrupted."""
    j, z = d / f"{name}.json", d / f"{name}.npz"
    if not j.exists():
        return None
    doc = json.loads(j.read_text())
    if doc.get("identity") != json.loads(json.dumps(identity)):
        raise Refused(f"refused: {j} was written for a different identity (stale)")
    if not z.is_file() or sha256_file(z) != doc.get("npz_sha256"):
        raise Refused(f"refused: {z} does not match the hash recorded in {j.name} (corrupted)")
    return doc


def write_result(d: Path, name: str, identity: dict, arrays: dict, summary: dict):
    atomic_npz(d / f"{name}.npz", **arrays)
    atomic_json(d / f"{name}.json", {"identity": identity, "npz_sha256": sha256_file(d / f"{name}.npz"), **summary})


def stage_evaluate(out: Path, cohort: str, want: str, resume: bool):
    from ccil.training.vector_bc import VectorBCPolicy
    aud = require_audited_tool(out)
    aud_sha = sha256_file(out / "audit.json")
    g = load_geometry(out, want)
    eligible = {"capture": aud["capture"]["eligible"], "corrected": aud["corrected"]["eligible"],
                "matched": aud["corrected"]["eligible"], "response_surface": aud["response_surface"]["eligible"]}
    if not eligible[cohort]:
        raise Refused(f"refused: the {cohort} cohort is not eligible")
    if cohort != "capture" and g["doc"]["rho_corrected"] is None:
        raise Refused("refused: there is no measured target tube (q95 is 0 or the corrected cohort is excluded)")
    if cohort in ("corrected", "matched"):
        dec = json.loads((out / "decision.json").read_text()) if (out / "decision.json").exists() else {}
        if dec.get("fitness", {}).get("outcome") != "fit":
            raise Refused("refused: the sensitivity gate has not passed (run capture, then decide)")
        if not g["doc"]["testable"]:
            raise Refused("refused: the corrective-tube hypothesis is not testable with these checkpoints")
        if cohort == "matched" and dec.get("signal", {}).get("outcome") != "signal":
            raise Refused("refused: no recovery signal; the matched pair is not trained or evaluated")
    rho = cohort_rho(cohort, g)
    d = out / "evaluate" / cohort
    d.mkdir(parents=True, exist_ok=True)
    R_anchor, off = g["refs"][g["ref"]], g["t0"]
    ref_id = result_identity(aud_sha, g, cohort, "reference", 0, None, rho)
    if verified_result(d, "reference", ref_id) is None:
        ref = {name: summarize(evaluate_states(pol, g["x0"], R_anchor, off, rho), g)
               for name, pol in (("pi_D", expert), ("zero", lambda x: np.zeros(len(x))))}
        write_result(d, "reference", ref_id, {"rho": np.array(rho)}, {"behaviors": ref})
    held = {}
    with Heartbeat(f"evaluate {cohort}"):
        for (arm, s), (path, sha) in sorted(expected_units(aud, cohort, out).items()):
            held_sha = aud["capture"]["held_out_sha256"][str(s)] if cohort == "capture" else None
            ident = result_identity(aud_sha, g, cohort, arm, s, sha, rho, held_sha)
            name = f"{arm}-s{s}"
            if (d / f"{name}.json").exists() and not resume:
                raise Refused(f"refused: {d / name}.json exists (an interrupted run takes --resume)")
            if verified_result(d, name, ident) is not None:
                continue
            if sha256_file(path) != sha:
                raise Refused(f"refused: {path} does not match its audited hash")
            pol = VectorBCPolicy.load(path, "cpu")[0].predict
            r = evaluate_states(pol, g["x0"], R_anchor, off, rho)
            summ, arrays = {"arm": arm, "seed": s, "anchors": summarize(r, g)}, dict(r)
            if cohort == "capture":
                hp = Path(aud["capture"]["run"]) / "select" / f"s{s}.npz"
                if sha256_file(hp) != held_sha:
                    raise Refused(f"refused: {hp} does not match its audited hash")
                if s not in held:
                    x = np.load(hp)["diagnostic/x"]
                    held[s] = (x, pi_d_continuations(x))
                x, R = held[s]
                hc = evaluate_states(pol, x, R, np.zeros(len(x), int), RHO_CAPTURE)
                summ["held_out_capture"] = {"n": int(len(x)), "rho": RHO_CAPTURE,
                                            "recovered": float(hc["recovered"].mean()), "failed": int(hc["failed"].sum())}
                arrays.update({f"held_out/{k}": v for k, v in hc.items()})
            write_result(d, name, ident, arrays, summ)
            log("evaluated", cohort, arm, s)


# --- decisions ------------------------------------------------------------------------------------------------------
def load_results(out: Path, cohort: str, aud: dict, g: dict) -> dict:
    """{(arm, seed): result} for exactly the declared matrix; refused on missing, extra, stale or corrupted results."""
    d = out / "evaluate" / cohort
    want = expected_units(aud, cohort, out)
    present = ({p.stem for p in d.glob("*.json")} - {"reference"}) if d.exists() else set()
    names = {f"{arm}-s{s}" for arm, s in want}
    if present != names:
        raise Refused(f"refused: {cohort} results are not the declared matrix (missing {sorted(names - present)[:5]}, "
                      f"extra {sorted(present - names)[:5]})")
    aud_sha, rho = sha256_file(out / "audit.json"), cohort_rho(cohort, g)
    res = {}
    for (arm, s), (_, sha) in want.items():
        held_sha = aud["capture"]["held_out_sha256"][str(s)] if cohort == "capture" else None
        res[(arm, s)] = verified_result(d, f"{arm}-s{s}", result_identity(aud_sha, g, cohort, arm, s, sha, rho, held_sha))
    return res


def paired(res: dict, a: str, b: str, field) -> dict:
    seeds = sorted({s for arm, s in res if arm == a} & {s for arm, s in res if arm == b})
    va, vb = {s: field(res[(a, s)]) for s in seeds}, {s: field(res[(b, s)]) for s in seeds}
    if not seeds or any(v is None for v in [*va.values(), *vb.values()]):
        return {"seeds": seeds, "interval": None, "above_zero": False, "note": "a value is undefined"}
    iv = union_interval([va[s] - vb[s] for s in seeds], CONFIDENCE)
    lo = iv["interval"][0] if iv.get("interval") else None
    return {**iv, "seeds": seeds, "above_zero": bool(lo is not None and lo > 0)}


def stage_decide(out: Path):
    aud = require_audited_tool(out)
    g = load_geometry(out, sha256_file(out / "geometry.json"))
    if g["doc"]["audit_sha256"] != sha256_file(out / "audit.json"):
        raise Refused("refused: geometry.json was not frozen from this audit")
    dec = {"protocol": PROTOCOL, "geometry_sha256": g["sha256"], "audit_sha256": sha256_file(out / "audit.json"),
           "tool_sha256": tool_sha256()}
    cap = load_results(out, "capture", aud, g)
    fit = paired(cap, "relabel", "duplication", lambda d: d["held_out_capture"]["recovered"])
    dec["fitness"] = {**fit, "contrast": "relabel - duplication, held-out capture recovery (rho = 0.1); a new "
                                         "sensitivity gate", "outcome": "fit" if fit["above_zero"] else "not fit: stop"}
    prim = lambda d: d["anchors"]["primary_edge_recovery"]  # noqa: E731
    if dec["fitness"]["outcome"] == "fit" and not g["doc"]["testable"]:
        dec["signal"] = {"outcome": "not testable with these checkpoints", "bins": g["doc"].get("bins")}
    elif dec["fitness"]["outcome"] == "fit" and (out / "evaluate/corrected").exists():
        cor = load_results(out, "corrected", aud, g)
        sig = paired(cor, "ccil", "bc", prim)
        dec["signal"] = {**sig, "contrast": "ccil - bc, edge-bin recovery to rho = q95",
                         "outcome": "signal" if sig["above_zero"] else "no local recovery-tube enlargement detected"}
        if dec["signal"]["outcome"] == "signal" and (out / "evaluate/matched").exists():
            mat = load_results(out, "matched", aud, g)
            both = {**mat, **cor}
            att = paired(mat, "replay_ccil", "duplication", prim)
            dec["attribution"] = {**att, "contrast": "replay_ccil - duplication",
                                  "outcome": "supervision-driven" if att["above_zero"]
                                  else "not attributable to corrective information",
                                  "retraining_drift_ccil_minus_replay": paired(both, "ccil", "replay_ccil", prim),
                                  "exposure_duplication_minus_bc": paired(both, "duplication", "bc", prim),
                                  "total_ccil_minus_bc": paired(cor, "ccil", "bc", prim)}
    atomic_json(out / "decision.json", dec)
    log("decision", json.dumps({k: v.get("outcome") for k, v in dec.items() if isinstance(v, dict)}))


# --- selftest: synthetic and off-protocol fixtures only --------------------------------------------------------------
def _refuses(fn) -> bool:
    try:
        fn()
    except Refused:
        return True
    return False


def selftest() -> dict:
    checks = {}
    a = np.array([[np.sin(np.pi - 1e-3), np.cos(np.pi - 1e-3), 0.0]])
    b = np.array([[np.sin(-np.pi + 1e-3), np.cos(-np.pi + 1e-3), 0.0]])
    checks["metric_wraps_theta"] = bool(abs(dist(a, b)[0] - 2e-3) < 1e-9)

    refs = references(seed=9001, n=3, steps=REF_STEPS, check_disjoint=False)  # off-protocol references
    g, umax = disturbances(refs, frames=(0, 150, 300))
    checks["disturbance_actions_within_bound"] = bool(umax <= 3.0)
    checks["disturbance_count"] = len(g["x0"]) == 9 * (1 + len(OFFSETS) * len(DURATIONS) * 2)
    und = g["k"] == 0
    r = evaluate_states(expert, g["x0"][und], refs[g["ref"][und]], g["t0"][und], RHO_CAPTURE)
    checks["pi_D_undisturbed_recovers_at_lag0_onset0"] = bool(r["recovered"].all() and (r["lag"] == 0).all()
                                                             and (r["onset"] == 0).all())
    sw = und & (g["anchor_t"] >= 150) & (np.abs(wrap(theta(refs[g["ref"], g["anchor_t"]]) - np.pi)) >= LQR)
    rz = evaluate_states(lambda x: np.zeros(len(x)), g["x0"][sw], refs[g["ref"][sw]], g["t0"][sw], RHO_CAPTURE)
    checks["zero_action_fails_from_swing_anchors"] = {"n": int(sw.sum()), "recovered": int(rz["recovered"].sum()),
                                                      "pass": bool(sw.sum() > 0 and not rz["recovered"].any())}

    # a periodic synthetic reference (period 40) and rollouts that follow it at fixed or alternating lags
    P, Lr = 40, 800
    ph = 2 * np.pi * np.arange(Lr) / P
    Rp = np.stack([np.sin(ph), np.cos(ph), np.full(Lr, 2 * np.pi / P / DT)], -1)
    jj = np.arange(H + 1)
    S = np.stack([Rp[100 + jj + 7], Rp[100 + jj + np.where((jj // 20) % 2 == 0, 5, -5)], Rp[100 + jj + 5]])
    S[2, :60, 2] += 5.0  # off-tube in omega for 60 steps, then on the reference at lag 5
    rec, onset, lag, _ = recovery(S, np.repeat(Rp[None], 3, 0), np.array([100, 100, 100]), RHO_CAPTURE)
    checks["fixed_lag_follows_one_continuation"] = bool(rec[0] and lag[0] == 7 and onset[0] == 0)
    checks["alternating_lag_is_not_recovery"] = bool(not rec[1])
    checks["late_onset_recorded"] = bool(rec[2] and onset[2] == 60 and lag[2] == 5)
    early = Rp[(jj - 3) % P][None]  # follows lag -3 from the start, which needs frames before the reference's first
    rec_e, onset_e, lag_e, _ = recovery(early, Rp[None], np.array([0]), RHO_CAPTURE)
    checks["unavailable_negative_lags_at_reference_start"] = bool(rec_e[0] and onset_e[0] == 3 and lag_e[0] == -3)
    leave = Rp[100 + jj][None].copy()
    leave[0, 280:, 2] += 5.0  # on the reference for 280 steps, then leaves before the horizon's end
    checks["leaving_before_horizon_end_is_not_recovery"] = bool(
        not recovery(leave, Rp[None], np.array([100]), RHO_CAPTURE)[0][0])
    tight = Rp[100 + jj][None].copy()
    tight[0, :, 2] += 0.2  # 0.064 from the reference: inside a tube of 0.1, outside a tube of 0.05
    checks["tube_radius_is_honored"] = bool(recovery(tight, Rp[None], np.array([100]), 0.1)[0][0]
                                            and not recovery(tight, Rp[None], np.array([100]), 0.05)[0][0])

    bins = dose_bins(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.5, 9.0]), 1.0)
    checks["dose_bins"] = bins.tolist() == ["null", "inside", "inside", "edge", "edge", "beyond_1", "beyond_2",
                                            "beyond_3"]
    gs = {"bin": np.array(["undisturbed", "inside", "inside", "edge", "edge"]), "k": np.array([0, 1, 1, 4, 4]),
          "sign": np.array([0, 1, -1, 1, -1]), "stratum": np.array(["swing"] * 5)}
    rs = {"recovered": np.array([True, True, False, False, True]), "onset": np.array([0, 0, -1, -1, 10]),
          "integrated": np.zeros(5), "failed": np.zeros(5, bool), "final_hold": np.zeros(5, bool),
          "saturated": np.zeros(5)}
    sm = summarize(rs, gs)
    checks["primary_is_edge_recovery_retention_separate"] = bool(sm["primary_edge_recovery"] == 0.5
                                                                 and sm["inside_retention"] == 0.5)

    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # capture-style manifest: a tampered file excludes the cohort
        (td / "policies/bc/s1").mkdir(parents=True)
        (td / "policies/bc/s1/policy.npz").write_bytes(b"policy")
        (td / "select").mkdir()
        (td / "select/s1.npz").write_bytes(b"states")
        man = {"artifacts_sha256": {k: sha256_file(td / k) for k in ("policies/bc/s1/policy.npz", "select/s1.npz")}}
        rep = {"status": "completed", "qualifying_seeds": [1],
               "policies": {"bc/s1": man["artifacts_sha256"]["policies/bc/s1/policy.npz"]}}
        ok = audit_capture(td, man, rep)["eligible"]
        (td / "select/s1.npz").write_bytes(b"tampered")
        checks["capture_tamper_excludes"] = bool(ok and not audit_capture(td, man, rep)["eligible"])
        # pinned evidence: an unpinned copy excludes the corrected cohort before anything else is read
        (td / "ev").mkdir()
        (td / "ev/study-audit.json").write_text("{}")
        (td / "ev/study-evaluate.json").write_text("{}")
        unpinned = audit_corrected(td, td, td / "ev")
        checks["unpinned_evidence_excludes"] = bool(not unpinned["eligible"] and len(unpinned["failures"]) == 2
                                                    and all("pinned" in f for f in unpinned["failures"]))
        # geometry freeze: phase 2 refuses a mismatched record
        (td / "geometry.json").write_text("{}")
        checks["phase2_refuses_mismatched_geometry"] = _refuses(lambda: load_geometry(td, "0" * 64))
        # results: exact identity and NPZ reconciliation
        rd = td / "res"
        rd.mkdir()
        ident = {"cohort": "x", "arm": "a", "seed": 1, "policy_sha256": "p"}
        write_result(rd, "a-s1", ident, {"v": np.arange(3)}, {"s": 1})
        checks["result_identity_verifies"] = verified_result(rd, "a-s1", ident) is not None
        checks["stale_result_refused"] = _refuses(lambda: verified_result(rd, "a-s1", {**ident, "policy_sha256": "q"}))
        atomic_npz(rd / "a-s1.npz", v=np.arange(4))
        checks["corrupted_result_refused"] = _refuses(lambda: verified_result(rd, "a-s1", ident))
        # decisions need the exact declared matrix: two of ten seeds cannot decide, nor can an extra result
        out = td / "partial"
        (out / "evaluate/capture").mkdir(parents=True)
        aud = {"capture": {"run": str(td), "seeds": list(range(1, 11)),
                           "policy_sha256": {f"{arm}/s{s}": "p" for arm in pc.ARMS for s in range(1, 12)},
                           "held_out_sha256": {str(s): "h" for s in range(1, 12)}}}
        atomic_json(out / "audit.json", {})
        gg = {"sha256": "g", "doc": {"geometry_npz_sha256": "n", "rho_corrected": None}}

        def put(arm, s):
            idn = result_identity(sha256_file(out / "audit.json"), gg, "capture", arm, s, "p", RHO_CAPTURE, "h")
            write_result(out / "evaluate/capture", f"{arm}-s{s}", idn, {"v": np.zeros(1)},
                         {"held_out_capture": {"recovered": 1.0 if arm == "relabel" else 0.0}})
        for arm in pc.ARMS:
            for s in (1, 2):
                put(arm, s)
        checks["partial_matrix_cannot_decide"] = _refuses(lambda: load_results(out, "capture", aud, gg))
        for arm in pc.ARMS:
            for s in range(3, 12):
                put(arm, s)
        checks["extra_result_cannot_decide"] = _refuses(lambda: load_results(out, "capture", aud, gg))
        for arm in pc.ARMS:
            (out / f"evaluate/capture/{arm}-s11.json").unlink()
        checks["exact_matrix_decides"] = len(load_results(out, "capture", aud, gg)) == 30
        # store completeness: an unresolved root reference, and a corrupted entry, exclude the cohort
        e = td / "rs/store/policy"
        ident_s = {"kind": "policy", "seed": 1}
        d = e / digest(ident_s)
        d.mkdir(parents=True)
        (d / "identity.json").write_text(json.dumps(ident_s))
        (d / "policy.npz").write_bytes(b"p")
        (d / "manifest.json").write_text(json.dumps({"policy.npz": {"sha256": sha256_file(d / "policy.npz")}}))
        (td / "rs/stages").mkdir()
        (td / "rs/stages/train.json").write_text(json.dumps({"units": {"bc/r1.0/k0/s1": {
            "arm": "bc", "r": 1.0, "k": 0, "seed": 1, "sigma": None, "entry": d.name,
            "policy_sha256": sha256_file(d / "policy.npz")}}, "analysis": {"bc/r1.0/k0/s1": "bc/r1.0/k0/s1"}}))
        (td / "rs/manifest.json").write_text(json.dumps({"stages": {"train": sha256_file(td / "rs/stages/train.json")}}))
        (td / "rs/root.json").write_text(json.dumps({"code": {"lock": "a" * 64}, "spec_sha256": "b" * 64}))
        unanchored = audit_store(td / "rs")["eligible"]
        anchored = audit_store(td / "rs", {"spec": "b" * 64})["eligible"]
        checks["store_root_reference_must_resolve"] = bool(not unanchored and anchored)
        (d / "policy.npz").write_bytes(b"q")
        checks["store_corruption_detected"] = bool(not audit_store(td / "rs", {"spec": "b" * 64})["eligible"])

    import recovery_tube_dup as dup
    checks["duplication_view"] = dup.selftest()
    flat = [v["pass"] if isinstance(v, dict) else v for v in checks.values()]
    return {"checks": checks, "all_pass": all(bool(v) for v in flat)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audit", "geometry", "evaluate", "decide", "selftest"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cohort", choices=["capture", "corrected", "matched", "response_surface"])
    ap.add_argument("--geometry-sha256")
    for k in ("capture", "corrected", "corrected_eval", "ccil_evidence", "response_surface"):
        ap.add_argument("--" + k.replace("_", "-"), type=Path)
    a = ap.parse_args()
    t0 = time.time()
    if a.step == "selftest":
        doc = selftest()
        atomic_json(a.out, doc)
        log("selftest", "all pass" if doc["all_pass"] else json.dumps(doc["checks"]))
        return 0 if doc["all_pass"] else 1
    require_clean()
    if a.step == "audit":
        if (a.out / "audit.json").exists():
            raise Refused(f"refused: {a.out}/audit.json exists")
        a.out.mkdir(parents=True, exist_ok=True)
        atomic_json(a.out / "audit.json", stage_audit(a))
    elif a.step == "geometry":
        stage_geometry(a.out, a.resume)
    elif a.step == "evaluate":
        if not (a.cohort and a.geometry_sha256):
            raise Refused("evaluate needs --cohort and --geometry-sha256")
        stage_evaluate(a.out, a.cohort, a.geometry_sha256, a.resume)
    else:
        stage_decide(a.out)
    log(a.step, "done in", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
