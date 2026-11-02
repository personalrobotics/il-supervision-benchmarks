#!/usr/bin/env python3
"""The Pendulum recovery-tube study (docs/studies/pendulum-recovery-tube.md, version 1 with amendments 1-2). Exploratory.

    uv run python tools/recovery_tube.py audit    --out DIR [--capture RUN] [--corrected RUN --corrected-eval RUN
                                                  --ccil-evidence DIR] [--response-surface RUN]
    uv run python tools/recovery_tube.py geometry --out DIR [--resume]
    uv run python tools/recovery_tube.py evaluate --out DIR --geometry-sha256 SHA --cohort COHORT [--resume]
    uv run python tools/recovery_tube.py decide   --out DIR
    uv run python tools/recovery_tube.py selftest --out FILE.json

audit reads the input runs without writing to them and decides each cohort's eligibility before any policy is loaded.
geometry (phase 1) measures the corrected cohort's label tube, builds the references, anchors and feasible
disturbances, and freezes the dose bins and the testability verdict in geometry.json and geometry.npz; it loads no
policy. evaluate (phase 2) refuses unless geometry.json has the given sha256; COHORT is capture, corrected,
duplication or response_surface. decide applies the sequential decision rules. selftest runs the synthetic and
off-protocol fixtures and writes their checks; it reads no real checkpoint. Each per-seed or per-policy artifact is
written atomically and skipped on --resume when present.
"""
from __future__ import annotations

import argparse
import json
import re
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
from ilsb.tasks.pendulum import UPRIGHT_COS, Task, expert  # noqa: E402

G, DT = 9.81, 0.02
TAU_STAR = 0.038357  # the one-step exact-preimage bound (CCIL pendulum-correction.md); a reference line only
REF_SEED, N_REF, REF_STEPS, ANCHOR_FRAMES = 2301, 20, 1000, tuple(range(0, 500, 50))
EARLIER_EVAL_SEEDS = tuple(range(2201, 2221))
OFFSETS, DURATIONS, SIGNS = (0.1875, 0.375, 0.75, 1.5, 3.0, 6.0), (1, 4, 16), (1, -1)
H, ONSET_MAX, DWELL, LAG, RHO, LQR = 300, 250, 50, 25, 0.1, 0.1  # DWELL = H - ONSET_MAX: the shortest stay
LAGS = np.array([0] + [v for i in range(1, LAG + 1) for v in (-i, i)])  # tie order: smallest |s|, negative first
NULL_DELTA, MIN_BIN, MIN_REPORT, CONFIDENCE = 1e-9, 200, 20, 0.95
DATA_SHA256 = pc.DATA_SHA256
CORRECTED_SEEDS, CORRECTED_ARMS = tuple(range(1201, 1221)), ("bc", "ccil")
RS_ARMS, RS_K = ("bc", "ccil_exact", "ccil_learned"), {"bc": 0, "ccil_exact": 20, "ccil_learned": 20}
HOME = Path("/home/siddh/code")
DEFAULTS = {"capture": HOME / "il-supervision-benchmarks/runs/20261102T003101Z-capture-control",
            "corrected": HOME / "CCIL/runs/20261029T191145Z-corrected-study",
            "corrected_eval": HOME / "CCIL/runs/20261030T013609Z-corrected-eval",
            "ccil_evidence": HOME / "CCIL/docs/fixtures/evidence/pendulum-corrected-demonstrator",
            "response_surface": HOME / "il-supervision-benchmarks/runs/20261031T141026Z-j-r-k"}
CAPTURE_EVIDENCE = pc.REPO / "docs/studies/evidence/pendulum-capture-control"
HEX64 = re.compile(r"^[0-9a-f]{64}$")
log, Heartbeat, atomic_json, atomic_npz = pc.log, pc.Heartbeat, pc.atomic_json, pc.atomic_npz


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


# --- audit (read-only) ----------------------------------------------------------------------------------------------
def audit_capture(run: Path, manifest: dict, report: dict) -> dict:
    bad = [k for k, h in manifest["artifacts_sha256"].items()
           if not (run / k).is_file() or sha256_file(run / k) != h]
    bad += [f"policy {k}" for k, h in report["policies"].items()
            if manifest["artifacts_sha256"].get(f"policies/{k}/policy.npz") != h]
    return {"run": str(run), "eligible": not bad and report.get("status") == "completed", "failures": bad,
            "seeds": report.get("qualifying_seeds", []), "arms": list(pc.ARMS)}


def audit_corrected(run: Path, eval_run: Path, study_audit: dict, study_eval: dict) -> dict:
    """Amendment 1's chain for every continuous BC and CCIL unit; all-or-nothing."""
    units, bad = {}, []
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
            if root not in stage.parents:
                f.append("policy.train stage is outside the run")
            man = json.loads((stage / "manifest.json").read_text()) if (stage / "manifest.json").is_file() else {}
            ck = stage / "checkpoints/policy.npz"
            ck_sha = sha256_file(ck) if ck.is_file() else None
            if not (ck_sha and ck_sha == man.get("files", {}).get("checkpoints/policy.npz", {}).get("sha256")
                    == rec.get("checkpoint_sha256")):
                f.append("checkpoint, stage manifest and record hashes differ")
            if not study_audit["stages"].get(str(stage.relative_to(root)) if root in stage.parents else "",
                                             {}).get("pass"):
                f.append("policy.train stage is not a passing stage of the committed audit")
            roll = eval_run / "continuous" / arm / f"s{s}-0.0.npz"
            if not roll.is_file() or sha256_file(roll) != study_eval["rollout_sha256"].get(f"{key}/0.0"):
                f.append("evaluate rollout does not match the committed rollout hash")
            unit = {"checkpoint": str(ck), "checkpoint_sha256": ck_sha, "stage": str(stage)}
            if man.get("inputs", {}).get("dataset", {}).get("sha256") != DATA_SHA256:
                f.append("dataset input is not the canonical continuous dataset")
            unit["dataset"] = man.get("inputs", {}).get("dataset", {}).get("path")
            if arm == "ccil":
                cand = Path(rec["stages"]["labels.generate"]) / "labels/candidates.npz"
                want = man.get("inputs", {}).get("candidates", {}).get("sha256")
                if not cand.is_file() or sha256_file(cand) != want:
                    f.append("candidates do not match the policy.train input hash")
                unit.update(candidates=str(cand), candidates_sha256=want)
            units[key] = unit
            bad += [f"{key}: {x}" for x in f]
    return {"run": str(run), "eval_run": str(eval_run), "eligible": not bad, "failures": bad, "units": units,
            "seeds": list(CORRECTED_SEEDS), "arms": list(CORRECTED_ARMS)}


def audit_store(run: Path) -> dict:
    """Internal completeness of a content-addressed run; never external anchoring."""
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
    stage_shas = {}
    for name, h in json.loads((run / "manifest.json").read_text())["stages"].items():
        p = run / "stages" / f"{name}.json"
        stage_shas[name] = sha256_file(p) if p.is_file() else None
        if stage_shas[name] != h:
            problems.append(f"stage {name}: sha256 differs from the run manifest")
    rooted = set(_hexes(json.loads((run / "root.json").read_text())))
    known = addresses | file_shas | set(stage_shas.values()) | rooted
    for src in [*(run / "stages").glob("*.json"), *(run / "store").glob("*/*/identity.json")]:
        missing = sorted(set(_hexes(json.loads(src.read_text()))) - known)
        problems += [f"{src.relative_to(run)}: unresolved reference {h}" for h in missing[:3]]
    train = json.loads((run / "stages/train.json").read_text())
    units = {}
    r_max = max(u["r"] for u in train["units"].values())
    for arm in RS_ARMS:
        for key, u in train["units"].items():
            if u["arm"] == arm and u["r"] == r_max and u["k"] == RS_K[arm] and key in train["analysis"].values():
                p = run / "store/policy" / u["entry"] / "policy.npz"
                if not p.is_file() or sha256_file(p) != u["policy_sha256"]:
                    problems.append(f"{key}: policy file does not match the train stage")
                units[key] = {"arm": arm, "seed": u["seed"], "sigma": u["sigma"], "policy": str(p),
                              "entry": u["entry"]}
    return {"run": str(run), "eligible": not problems, "failures": problems[:200], "failure_count": len(problems),
            "units": units, "r": r_max, "anchoring": "internal consistency only; no root result was committed before "
                                                     "2026-11-02, so this is not prior external anchoring"}


def _hexes(obj):
    if isinstance(obj, str):
        if HEX64.match(obj):
            yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _hexes(k)
            yield from _hexes(v)
    elif isinstance(obj, list):
        for v in obj:
            yield from _hexes(v)


def stage_audit(a) -> dict:
    out = {"protocol": "docs/studies/pendulum-recovery-tube.md (version 1, amendments 1-2)",
           "code_commit": pc.git("rev-parse", "HEAD"), "code_dirty": bool(pc.git("status", "--porcelain"))}
    cap = a.capture or DEFAULTS["capture"]
    out["capture"] = audit_capture(cap, json.loads((CAPTURE_EVIDENCE / "artifact-manifest.json").read_text()),
                                   json.loads((CAPTURE_EVIDENCE / "capture-control.json").read_text()))
    log("capture cohort eligible:", out["capture"]["eligible"], out["capture"]["failures"][:5])
    ev = a.ccil_evidence or DEFAULTS["ccil_evidence"]
    out["corrected"] = audit_corrected(a.corrected or DEFAULTS["corrected"], a.corrected_eval or DEFAULTS["corrected_eval"],
                                       json.loads((ev / "study-audit.json").read_text()),
                                       json.loads((ev / "study-evaluate.json").read_text()))
    log("corrected cohort eligible:", out["corrected"]["eligible"], out["corrected"]["failures"][:5])
    rs = a.response_surface or DEFAULTS["response_surface"]
    out["response_surface"] = audit_store(rs) if rs.exists() else {"run": str(rs), "eligible": False,
                                                                     "failures": ["run not present"]}
    log("response-surface cohort internally complete:", out["response_surface"]["eligible"])
    return out


# --- phase 1: references, anchors, feasible disturbances, label geometry, dose bins ----------------------------------
def references(seed=REF_SEED, n=N_REF, steps=REF_STEPS, check_disjoint=True):
    starts = Task(eval_seed=seed, eval_episodes=n).starts()
    if check_disjoint:
        for e in EARLIER_EVAL_SEEDS:
            if (pc.phys(starts)[:, None] == pc.phys(Task(eval_seed=e).starts())[None]).all(-1).any():
                raise RuntimeError(f"reference start set {seed} shares a start with eval seed {e}")
    task, x, out = Task(), starts, [starts]
    for _ in range(steps):
        x = task.step(x, expert(x))[0]
        out.append(x)
    return np.stack(out, 1)


def disturbances(refs, frames=ANCHOR_FRAMES):
    """Every anchor x refs[i, t]: the undisturbed case, then each (k, b, sign) with u = clip(pi_D(x) + sign b, -3, 3)
    for k steps. Returns the initial states, their reference index and start frame, and the design; applied actions'
    extreme magnitude is returned for the bound check."""
    task = Task()
    rows = {k: [] for k in ("x0", "ref", "t0", "anchor_t", "b", "k", "sign", "delta")}
    umax = 0.0
    for i in range(len(refs)):
        for t in frames:
            x = refs[i, t]
            rows["x0"].append(x), rows["ref"].append(i), rows["t0"].append(t), rows["anchor_t"].append(t)
            rows["b"].append(0.0), rows["k"].append(0), rows["sign"].append(0), rows["delta"].append(0.0)
    for k in DURATIONS:
        for b in OFFSETS:
            for sg in SIGNS:
                idx = [(i, t) for i in range(len(refs)) for t in frames]
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
        raise RuntimeError("a candidate's source episode is not in the dataset")
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
    aud = json.loads((out / "audit.json").read_text())
    path = out / "geometry.json"
    if path.exists():
        raise SystemExit(f"refused: {path} exists; phase 1 is frozen")
    refs = references()
    dist_, umax = disturbances(refs)
    doc = {"references": {"eval_seed": REF_SEED, "episodes": N_REF, "steps": REF_STEPS},
           "anchor_frames": list(ANCHOR_FRAMES), "offsets": list(OFFSETS), "durations": list(DURATIONS),
           "applied_action_max_abs": umax, "tau_star_reference": TAU_STAR, "recovery": {
               "horizon": H, "onset_max": ONSET_MAX, "dwell": DWELL, "lag": LAG, "rho": RHO},
           "code_commit": pc.git("rev-parse", "HEAD"), "code_dirty": bool(pc.git("status", "--porcelain"))}
    arrays = {"refs": refs, **{k: v for k, v in dist_.items()}}
    disturbed = dist_["k"] > 0
    if aud["corrected"]["eligible"]:
        per, d_all, zs = {}, [], {}
        (out / "geometry").mkdir(exist_ok=True)
        with Heartbeat("label geometry"):
            for s in CORRECTED_SEEDS:
                u = aud["corrected"]["units"][f"continuous/ccil/s{s}"]
                p = out / "geometry" / f"s{s}.npz"
                if not (resume and p.exists()):
                    summary, d_src, z = label_geometry(Path(u["candidates"]), Path(u["dataset"]))
                    atomic_npz(p, d_source=d_src, z=z, summary=json.dumps(summary))
                    log("geometry", s, json.dumps(summary["d_source"]))
                f = np.load(p)
                per[str(s)], zs[s] = json.loads(str(f["summary"])), f["z"]
                d_all.append(f["d_source"])
        d_all = np.concatenate(d_all)
        tube = {"q50": float(np.quantile(d_all, 0.5)), "q95": float(np.quantile(d_all, 0.95)),
                "max": float(d_all.max()), "rows": int(len(d_all))}
        bins = dose_bins(dist_["delta"], tube["q95"]) if tube["q95"] > 0 else np.where(
            disturbed & (dist_["delta"] >= NULL_DELTA), "beyond_inf", "null")
        bins = np.where(disturbed, bins, "undisturbed")
        counts = {b: int((bins == b).sum()) for b in sorted(set(bins))}
        testable = counts.get("inside", 0) >= MIN_BIN and counts.get("edge", 0) >= MIN_BIN
        overlap = {}
        for b in sorted(set(bins) - {"undisturbed", "null"}):
            m = bins == b
            overlap[b] = float(np.mean([(tree(zs[s]).query(_emb(dist_["x0"][m]))[0] <= tube["q95"]).mean()
                                        for s in CORRECTED_SEEDS]))
        doc.update(label_geometry={"per_seed": per, "pooled": tube}, bins=counts, testable=testable,
                   overlap_within_q95=overlap, testability_rule=f"inside and edge each hold >= {MIN_BIN} states")
        arrays["bin"] = bins
    else:
        doc.update(label_geometry=None, testable=False, reason="the corrected cohort is excluded by the audit")
        arrays["bin"] = np.where(disturbed, "unbinned", "undisturbed")
    if aud["response_surface"]["eligible"]:
        rs = {}
        for key, u in aud["response_surface"]["units"].items():
            if u["arm"] == "bc":
                continue
            ident = json.loads((Path(u["policy"]).parent / "identity.json").read_text())
            run = Path(aud["response_surface"]["run"])
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


def recovery(S, R, off, chunk=400):
    """Fixed-lag recovery (amendment 2) of rollouts S (N, H+1, 3) against reference continuations R (N, L, 3), each
    rollout starting at index off[n] of its reference. Recovered: one lag |s| <= LAG and an onset j <= ONSET_MAX with
    d(x_m, r_{m+s}) <= RHO for every m = j..H, so the rollout enters the tube and stays to the horizon's end (at least
    DWELL steps); a lag needing a frame outside R is unavailable."""
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
        stay = np.flip(np.cumprod(np.flip(D <= RHO, 1), 1), 1).astype(bool)  # stay[n, j, s]: in the tube from j to H
        win = stay[:, :ONSET_MAX + 1]
        anyj = win.any(2)
        has = anyj.any(1)
        first = np.argmax(anyj, 1)
        rec[n] = has
        onset[n] = np.where(has, first, -1)
        lag[n] = np.where(has, LAGS[np.argmax(win[np.arange(len(n)), first], 1)], 0)
    return rec, onset, lag, integ


def evaluate_states(pol, x0, R, off):
    S, C, bad = rollout(pol, x0)
    rec, onset, lag, integ = recovery(S, R, off)
    rec &= ~bad
    up = upright(S[:, 1:])
    return {"recovered": rec, "onset": onset, "lag": lag, "integrated": integ, "failed": bad,
            "final_hold": up[:, -pc.HOLD:].all(1), "occupancy": up.mean(1),
            "saturated": (np.abs(C) >= 3).mean(1)}


def summarize(r: dict, g: dict) -> dict:
    """Per-policy summary over the frozen inputs: recovery per bin, stratum, k and sign; the primary log-grid mean."""
    rec, bins = r["recovered"], g["bin"]
    out = {"undisturbed_tracking": float(rec[bins == "undisturbed"].mean()), "failed": int(r["failed"].sum()),
           "saturated_fraction": float(r["saturated"].mean()), "bins": {}, "strata": {}, "k": {}, "sign": {}}
    for b in sorted(set(bins) - {"undisturbed", "null"}):
        m = bins == b
        if m.sum() >= MIN_REPORT or b in ("inside", "edge"):
            out["bins"][b] = {"n": int(m.sum()), "recovered": float(rec[m].mean()) if m.any() else None,
                              "onset_mean_s": float(r["onset"][m & rec].mean() * DT) if (m & rec).any() else None,
                              "integrated_mean": float(r["integrated"][m].mean()) if m.any() else None}
    dis = g["k"] > 0
    for st in ("lqr_branch", "swing"):
        m = dis & (g["stratum"] == st)
        out["strata"][st] = {"n": int(m.sum()), "recovered": float(rec[m].mean()) if m.any() else None,
                             "final_hold": float(r["final_hold"][m].mean()) if st == "lqr_branch" and m.any() else None}
    for k in DURATIONS:
        out["k"][str(k)] = float(rec[g["k"] == k].mean())
    for sg in SIGNS:
        out["sign"][str(sg)] = float(rec[g["sign"] == sg].mean())
    loc = [out["bins"].get(b, {}).get("recovered") for b in ("inside", "edge")]
    out["primary_log_grid_mean"] = float(np.mean(loc)) if all(v is not None for v in loc) else None
    full = [v["recovered"] for v in out["bins"].values() if v["recovered"] is not None]
    out["full_curve_log_grid_mean"] = float(np.mean(full)) if full else None
    return out


def load_geometry(out: Path, want: str) -> dict:
    gj = out / "geometry.json"
    if not gj.is_file() or sha256_file(gj) != want:
        raise SystemExit(f"refused: {gj} is not the frozen phase-1 record {want}")
    doc = json.loads(gj.read_text())
    if sha256_file(out / "geometry.npz") != doc["geometry_npz_sha256"]:
        raise SystemExit("refused: geometry.npz does not match geometry.json")
    f = np.load(out / "geometry.npz")
    return {"doc": doc, **{k: f[k] for k in f.files}}


def pi_d_continuations(x, steps=H + LAG + 1):
    task, s, out = Task(), np.array(x, np.float64), [np.array(x, np.float64)]
    for _ in range(steps):
        s = task.step(s, expert(s))[0]
        out.append(s)
    return np.stack(out, 1)


def cohort_policies(aud: dict, cohort: str, out: Path):
    """(arm, seed, loader) for every policy of an eligible cohort."""
    from ccil.training.vector_bc import VectorBCPolicy
    if cohort == "capture":
        run = Path(aud["capture"]["run"])
        return [(arm, s, lambda arm=arm, s=s: pc.policy(run, arm, s)) for s in aud["capture"]["seeds"] for arm in pc.ARMS]
    if cohort == "corrected":
        return [(arm, s, lambda u=aud["corrected"]["units"][f"continuous/{arm}/s{s}"]: VectorBCPolicy.load(
            u["checkpoint"], "cpu")[0].predict) for s in CORRECTED_SEEDS for arm in CORRECTED_ARMS]
    if cohort == "duplication":
        man = json.loads((out / "duplication/manifest.json").read_text())
        res = []
        for s in CORRECTED_SEEDS:
            p = out / "duplication/policies" / f"s{s}" / "policy.npz"
            if sha256_file(p) != man["policies"][str(s)]["sha256"]:
                raise SystemExit(f"refused: duplication policy s{s} does not match its manifest")
            res.append(("duplication", s, lambda p=p: VectorBCPolicy.load(p, "cpu")[0].predict))
        return res
    if cohort == "response_surface":
        return [(u["arm"], u["seed"], lambda p=u["policy"]: VectorBCPolicy.load(p, "cpu")[0].predict)
                for u in aud["response_surface"]["units"].values()]
    raise SystemExit(f"unknown cohort {cohort}")


def stage_evaluate(out: Path, cohort: str, want: str, resume: bool):
    aud = json.loads((out / "audit.json").read_text())
    g = load_geometry(out, want)
    gate = {"capture": aud["capture"]["eligible"], "corrected": aud["corrected"]["eligible"],
            "duplication": aud["corrected"]["eligible"], "response_surface": aud["response_surface"]["eligible"]}
    if not gate[cohort]:
        raise SystemExit(f"refused: the {cohort} cohort is not eligible")
    if cohort in ("corrected", "duplication"):
        dec = json.loads((out / "decision.json").read_text()) if (out / "decision.json").exists() else {}
        if dec.get("fitness", {}).get("outcome") != "fit":
            raise SystemExit("refused: evaluator fitness has not passed (run capture, then decide)")
        if not g["doc"]["testable"]:
            raise SystemExit("refused: the corrective-tube hypothesis is not testable with these checkpoints")
        if cohort == "duplication" and dec.get("signal", {}).get("outcome") != "signal":
            raise SystemExit("refused: no recovery signal; the duplication arm is not trained or evaluated")
    d = out / "evaluate" / cohort
    d.mkdir(parents=True, exist_ok=True)
    R_anchor, off = g["refs"][g["ref"]], g["t0"]
    if not (d / "reference.json").exists():
        ref = {}
        for name, pol in (("pi_D", expert), ("zero", lambda x: np.zeros(len(x)))):
            r = evaluate_states(pol, g["x0"], R_anchor, off)
            ref[name] = summarize(r, g)
        atomic_json(d / "reference.json", ref)
    held = {}
    with Heartbeat(f"evaluate {cohort}"):
        for arm, s, load in cohort_policies(aud, cohort, out):
            p = d / f"{arm}-s{s}.json"
            if p.exists():
                if not resume:
                    raise SystemExit(f"refused: {p} exists (an interrupted run takes --resume)")
                continue
            pol = load()
            r = evaluate_states(pol, g["x0"], R_anchor, off)
            summ = {"arm": arm, "seed": s, "anchors": summarize(r, g)}
            if cohort == "capture":
                if s not in held:
                    x = np.load(Path(aud["capture"]["run"]) / "select" / f"s{s}.npz")["diagnostic/x"]
                    held[s] = (x, pi_d_continuations(x))
                x, R = held[s]
                hc = evaluate_states(pol, x, R, np.zeros(len(x), int))
                summ["held_out_capture"] = {"n": int(len(x)), "recovered": float(hc["recovered"].mean()),
                                            "failed": int(hc["failed"].sum())}
            atomic_npz(d / f"{arm}-s{s}.npz", **r)
            atomic_json(p, summ)
            log("evaluated", cohort, arm, s)


# --- decisions ------------------------------------------------------------------------------------------------------
def paired(a: dict, b: dict) -> dict:
    seeds = sorted(set(a) & set(b))
    iv = union_interval([a[s] - b[s] for s in seeds], CONFIDENCE)
    lo = iv["interval"][0] if iv.get("interval") else None
    return {**iv, "seeds": seeds, "above_zero": bool(lo is not None and lo > 0)}


def per_seed(out: Path, cohort: str, arm: str, field) -> dict:
    vals = {}
    for p in sorted((out / "evaluate" / cohort).glob(f"{arm}-s*.json")):
        v = field(json.loads(p.read_text()))
        if v is not None:
            vals[int(p.stem.split("-s")[-1])] = v
    return vals


def stage_decide(out: Path):
    g = json.loads((out / "geometry.json").read_text())
    dec = {"protocol": "docs/studies/pendulum-recovery-tube.md (version 1, amendments 1-2)",
           "geometry_sha256": sha256_file(out / "geometry.json")}
    hc = lambda d: d.get("held_out_capture", {}).get("recovered")  # noqa: E731
    fit = paired(per_seed(out, "capture", "relabel", hc), per_seed(out, "capture", "duplication", hc))
    dec["fitness"] = {**fit, "contrast": "relabel - duplication, held-out capture recovery",
                      "outcome": "fit" if fit["above_zero"] else "not fit: stop"}
    if dec["fitness"]["outcome"] == "fit" and g["testable"] and (out / "evaluate/corrected").exists():
        prim = lambda d: d["anchors"]["primary_log_grid_mean"]  # noqa: E731
        sig = paired(per_seed(out, "corrected", "ccil", prim), per_seed(out, "corrected", "bc", prim))
        dec["signal"] = {**sig, "contrast": "ccil - bc, primary log-grid mean (inside, edge)",
                         "outcome": "signal" if sig["above_zero"] else "no local recovery-tube enlargement detected"}
        if dec["signal"]["outcome"] == "signal" and (out / "evaluate/duplication").exists():
            dup, ccil, bc = (per_seed(out, c, a, prim) for c, a in (("duplication", "duplication"),
                                                                     ("corrected", "ccil"), ("corrected", "bc")))
            att = paired(ccil, dup)
            dec["attribution"] = {**att, "contrast": "ccil - duplication",
                                  "outcome": "supervision-driven" if att["above_zero"]
                                  else "not attributable to corrective information",
                                  "exposure_duplication_minus_bc": paired(dup, bc),
                                  "total_ccil_minus_bc": paired(ccil, bc)}
    elif not g["testable"]:
        dec["signal"] = {"outcome": "not testable with these checkpoints", "bins": g.get("bins")}
    atomic_json(out / "decision.json", dec)
    log("decision", json.dumps({k: v.get("outcome") for k, v in dec.items() if isinstance(v, dict)}))


# --- selftest: synthetic and off-protocol fixtures only --------------------------------------------------------------
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
    r = evaluate_states(expert, g["x0"][und], refs[g["ref"][und]], g["t0"][und])
    checks["pi_D_undisturbed_recovers_at_lag0_onset0"] = bool(r["recovered"].all() and (r["lag"] == 0).all()
                                                             and (r["onset"] == 0).all())
    sw = und & (g["anchor_t"] >= 150) & (np.abs(wrap(theta(refs[g["ref"], g["anchor_t"]]) - np.pi)) >= LQR)
    rz = evaluate_states(lambda x: np.zeros(len(x)), g["x0"][sw], refs[g["ref"][sw]], g["t0"][sw])
    checks["zero_action_fails_from_swing_anchors"] = {"n": int(sw.sum()), "recovered": int(rz["recovered"].sum()),
                                                      "pass": bool(sw.sum() > 0 and not rz["recovered"].any())}

    # a periodic synthetic reference (period 40) and rollouts that follow it at fixed or alternating lags
    P, Lr = 40, 800
    ph = 2 * np.pi * np.arange(Lr) / P
    Rp = np.stack([np.sin(ph), np.cos(ph), np.full(Lr, 2 * np.pi / P / DT)], -1)
    off = np.array([100, 100, 100])
    S = np.zeros((3, H + 1, 3))
    jj = np.arange(H + 1)
    S[0] = Rp[100 + jj + 7]
    alt = np.where((jj // 20) % 2 == 0, 5, -5)  # switches lag every 20 steps: no fixed lag holds for 50
    S[1] = Rp[100 + jj + alt]
    S[2] = Rp[100 + jj + 5]
    S[2, :60, 2] += 5.0  # off-tube in omega for 60 steps, then on the reference at lag 5
    rec, onset, lag, _ = recovery(S, np.repeat(Rp[None], 3, 0), off)
    checks["fixed_lag_follows_one_continuation"] = bool(rec[0] and lag[0] == 7 and onset[0] == 0)
    checks["alternating_lag_is_not_recovery"] = bool(not rec[1])
    checks["late_onset_recorded"] = bool(rec[2] and onset[2] == 60 and lag[2] == 5)
    early = Rp[(jj - 3) % P][None]  # follows lag -3 from the start, which needs frames before the reference's first
    rec_e, onset_e, lag_e, _ = recovery(early, Rp[None], np.array([0]))
    checks["unavailable_negative_lags_at_reference_start"] = bool(rec_e[0] and onset_e[0] == 3 and lag_e[0] == -3)

    bins = dose_bins(np.array([0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 4.5, 9.0]), 1.0)
    checks["dose_bins"] = bins.tolist() == ["null", "inside", "inside", "edge", "edge", "beyond_1", "beyond_2",
                                            "beyond_3"]

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
        # geometry freeze: phase 2 refuses a mismatched record
        (td / "geometry.json").write_text("{}")
        try:
            load_geometry(td, "0" * 64)
            checks["phase2_refuses_mismatched_geometry"] = False
        except SystemExit:
            checks["phase2_refuses_mismatched_geometry"] = True
        # store completeness: a corrupted entry is reported
        e = td / "rs/store/policy"
        ident = {"kind": "policy", "seed": 1}
        d = e / digest(ident)
        d.mkdir(parents=True)
        (d / "identity.json").write_text(json.dumps(ident))
        (d / "policy.npz").write_bytes(b"p")
        (d / "manifest.json").write_text(json.dumps({"policy.npz": {"sha256": sha256_file(d / "policy.npz")}}))
        (td / "rs/stages").mkdir()
        (td / "rs/stages/train.json").write_text(json.dumps({"units": {"bc/r1.0/k0/s1": {
            "arm": "bc", "r": 1.0, "k": 0, "seed": 1, "sigma": None, "entry": d.name,
            "policy_sha256": sha256_file(d / "policy.npz")}}, "analysis": {"bc/r1.0/k0/s1": "bc/r1.0/k0/s1"}}))
        (td / "rs/manifest.json").write_text(json.dumps({"stages": {"train": sha256_file(td / "rs/stages/train.json")}}))
        (td / "rs/root.json").write_text("{}")
        good = audit_store(td / "rs")["eligible"]
        (d / "policy.npz").write_bytes(b"q")
        checks["store_corruption_detected"] = bool(good and not audit_store(td / "rs")["eligible"])

    import recovery_tube_dup as dup
    checks["duplication_view"] = dup.selftest()
    flat = [v["pass"] if isinstance(v, dict) else v for v in checks.values()]
    return {"checks": checks, "all_pass": all(bool(v) for v in flat)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audit", "geometry", "evaluate", "decide", "selftest"])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--cohort", choices=["capture", "corrected", "duplication", "response_surface"])
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
    if a.step == "audit":
        if (a.out / "audit.json").exists():
            raise SystemExit(f"refused: {a.out}/audit.json exists")
        a.out.mkdir(parents=True, exist_ok=True)
        doc = stage_audit(a)
        atomic_json(a.out / "audit.json", doc)
    elif a.step == "geometry":
        stage_geometry(a.out, a.resume)
    elif a.step == "evaluate":
        if not (a.cohort and a.geometry_sha256):
            raise SystemExit("evaluate needs --cohort and --geometry-sha256")
        stage_evaluate(a.out, a.cohort, a.geometry_sha256, a.resume)
    else:
        stage_decide(a.out)
    log(a.step, "done in", round(time.time() - t0, 1), "s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
