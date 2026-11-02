#!/usr/bin/env python3
"""The Pendulum missing-state positive control (docs/studies/pendulum-positive-control.md, version 1.1).

    uv run python tools/positive_control.py --out DIR [--device cuda] [--resume]

Stages, each written atomically under DIR and skipped on --resume when present: data (subset and split per seed), bc
(the BC group, also the collection policies), select (BC rollouts from the collection and diagnostic starts, the
off-support recovery states and their checks), arms (the duplication and relabel groups), evaluate (held-out action
error and the final closed loop), then positive-control.json with the four intervals and the decision. The run is
complete only when positive-control.json exists. This tests the missing-state-supervision mechanism; it is not
evidence for CCIL.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

from ilsb.analysis import union_interval
from ilsb.learners import vector_bc
from ilsb.store import sha256_file
from ilsb.supervision.ccil import _rep
from ilsb.tasks.pendulum import Task, expert, subset

REPO = Path(__file__).resolve().parents[1]
SPEC = REPO / "studies/pendulum-j-r-k/study.yaml"
DATA = REPO / "data/pendulum_cont_corrected.npz"
DATA_SHA256 = "59c332bc829db9c9da40d9c1bd9ebbf808965aca6a167be48dc0f2ade52cf43c"
SEEDS = tuple(range(3301, 3311))
R, VALIDATION_FRACTION = 0.1, 0.2
COLLECTION, DIAGNOSTIC, FINAL = 2202, 2203, 2204
BUDGET, MIN_STATES, MIN_SEEDS = 4000, 100, 8
LQR_WINDOW, RECOVERY_STEPS, HOLD, HOLD_E, HOLD_W = 0.1, 500, 100, 0.1, 0.5
MANIFOLD_TOL, CONFIDENCE, G = 6.7e-7, 0.95, 9.81
HEARTBEAT = 300
ARMS = ("bc", "duplication", "relabel")


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


class Heartbeat:
    """Logs that a long stage is alive at least every HEARTBEAT seconds."""

    def __init__(self, label):
        self.label, self.t0, self.stop = label, time.time(), threading.Event()

    def __enter__(self):
        def run():
            while not self.stop.wait(HEARTBEAT):
                log(self.label, f"running, {time.time() - self.t0:.0f} s")
        threading.Thread(target=run, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self.stop.set()
        log(self.label, f"done, {time.time() - self.t0:.0f} s")


def git(*args) -> str:
    return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()


def atomic_json(path: Path, obj):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=float) + "\n")
    os.replace(tmp, path)


def atomic_npz(path: Path, **arrays):
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, **arrays)
    os.replace(tmp, path)


def learner(arm: str) -> dict:
    import yaml
    s = dict(yaml.safe_load(SPEC.read_text())["learner"])
    if arm == "bc":
        for k in ("synthetic_fraction", "synthetic_coverage_before_stop"):
            s.pop(k)
    return s


# ---- geometry: the physical metric, periodic in e (CCIL docs/experiments/pendulum-correction.md) ----

def phys(s):
    """(e + pi in [0, 2 pi), omega / sqrt(g) shifted into a box that never wraps): the tree's coordinates."""
    return np.stack([np.mod(np.arctan2(s[..., 0], s[..., 1]), 2 * np.pi), s[..., 2] / np.sqrt(G) + 1e3], -1)


def tree(states):
    return cKDTree(phys(states), boxsize=[2 * np.pi, 1e9])


def e_of(s):
    """e = wrap(theta - pi), the angle from upright."""
    return np.mod(np.arctan2(s[..., 0], s[..., 1]), 2 * np.pi) - np.pi


def drift(s):
    """|sin^2 + cos^2 - 1|: the exact map carries sin and cos separately, so long fast rollouts drift."""
    return np.abs(s[..., 0] ** 2 + s[..., 1] ** 2 - 1)


def recovers(task: Task, states: np.ndarray) -> np.ndarray:
    """pi_D from each state for RECOVERY_STEPS through the exact map: |e| <= 0.1 and |omega| <= 0.5 over the last HOLD."""
    s, ok = np.array(states, np.float64), np.ones(len(states), bool)
    for t in range(RECOVERY_STEPS):
        s = task.step(s, expert(s))[0]
        if t >= RECOVERY_STEPS - HOLD:
            ok &= (np.abs(e_of(s)) <= HOLD_E) & (np.abs(s[:, 2]) <= HOLD_W)
    return ok


# ---- stages ----

def stage_data(out: Path):
    from ccil.data.split import make_split
    from ccil.schema import read_trajectories
    for s in SEEDS:
        d = out / "data" / f"s{s}"
        if (d / "split.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        subset(DATA, R, s, d / "subset.npz")
        eps = [e.episode_id for e in read_trajectories(d / "subset.npz").episodes]
        atomic_json(d / "split.json", make_split(eps, VALIDATION_FRACTION, s))
    log("data ready", len(SEEDS), "seeds")


def supervision(out: Path, s: int, part: str):
    from ccil.supervision import build_supervision
    d = out / "data" / f"s{s}"
    return build_supervision(d / "subset.npz", "bc", representation=_rep(), split=d / "split.json", part=part)


def with_extra(base, obs: np.ndarray, act: np.ndarray, kind: str):
    """base's demonstrations, then the extra rows as the synthetic block (the trainer's mixing contract)."""
    n = len(obs)
    cat = lambda a, b: np.concatenate([a, np.asarray(b, a.dtype)])  # noqa: E731
    return dataclasses.replace(
        base, observations=cat(base.observations, obs), actions=cat(base.actions, act.reshape(n, -1)),
        weights=cat(base.weights, np.ones(n)), source=cat(base.source, np.full(n, "synthetic")),
        episode_id=cat(base.episode_id, np.full(n, kind)), frame=cat(base.frame, np.arange(n)),
        candidate_row=cat(base.candidate_row, np.full(n, -1)), classification=cat(base.classification, np.full(n, kind)),
        manifest_={**base.manifest(), "extra_rows": {"kind": kind, "rows": n}})


def train(out: Path, arm: str, jobs: list, device: str):
    """One arm's group; jobs (seed, train supervision, validation supervision). Policies saved per seed."""
    todo = [j for j in jobs if not (out / "policies" / arm / f"s{j[0]}" / "policy.npz").exists()]
    if not todo:
        log(arm, "policies resumed")
        return
    with Heartbeat(f"train {arm} ({len(todo)} seeds)"):
        res = vector_bc.train([(t, v, s) for s, t, v in todo], learner(arm), device)
    for (s, _, _), (pol, rec) in zip(todo, res):
        if rec["stopping"]["outcome"] != "patience":  # the study's rule: the ceiling is a failed fit
            raise RuntimeError(f"{arm} s{s}: training stopped by {rec.get('stopping')}, not patience")
        d = out / "policies" / arm / f"s{s}"
        tmp = d.with_name(d.name + ".tmp")
        tmp.mkdir(parents=True, exist_ok=True)
        vector_bc.save(pol, rec, tmp)
        os.replace(tmp, d)
    log(arm, "trained", [s for s, _, _ in todo])


def policy(out: Path, arm: str, s: int):
    return vector_bc.load(out / "policies" / arm / f"s{s}", "cpu")


def stage_select(out: Path):
    from ccil.schema import read_trajectories
    starts = {k: Task(eval_seed=k).starts() for k in (COLLECTION, DIAGNOSTIC, FINAL)}
    for a in starts:
        for b in starts:
            if a < b and (phys(starts[a])[:, None] == phys(starts[b])[None]).all(-1).any():
                raise RuntimeError(f"start sets {a} and {b} share a start")
    for s in SEEDS:
        path = out / "select" / f"s{s}.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        split = json.loads((out / "data" / f"s{s}" / "split.json").read_text())
        eps = {e.episode_id: e.observations["state"].astype(np.float64)
               for e in read_trajectories(out / "data" / f"s{s}" / "subset.npz").episodes}
        t = tree(np.concatenate([eps[i] for i in split["train"]]))
        tau = float(np.quantile(t.query(phys(np.concatenate([eps[i] for i in split["validation"]])))[0], 0.95))
        pol, rec = policy(out, "bc", s), {"tau_demo": tau}
        sets = {}
        for name, seed, salt in (("collection", COLLECTION, 13), ("diagnostic", DIAGNOSTIC, 17)):
            states = Task(eval_seed=seed).rollouts([pol])[0]["states"].reshape(-1, 3)
            d = t.query(phys(states))[0]
            off = (d >= tau) & (np.abs(e_of(states)) >= LQR_WINDOW)
            idx = np.flatnonzero(off)
            ok = recovers(Task(), states[idx])
            keep = idx[ok]
            if len(keep) > BUDGET:
                keep = np.sort(np.random.default_rng(np.random.SeedSequence([s, salt])).choice(keep, BUDGET, False))
            x = states[keep]
            a = expert(x)
            nxt = Task().step(x, a)[0]
            H = 0.5 * x[:, 2] ** 2 - G * (1 - np.cos(e_of(x)))
            sets[name] = {"x": x, "a": a, "d_demo": d[keep]}
            rec[name] = {"visited": int(len(states)), "off_support": int(len(idx)),
                         "demonstrator_recovers": int(ok.sum()), "selected": int(len(keep)),
                         "bounds_ok": bool((np.abs(a) <= 3).all()),
                         "transitions_finite_no_added_drift": bool(np.isfinite(nxt).all() and (
                             np.abs(drift(nxt) - drift(x)) <= MANIFOLD_TOL).all()),
                         "state_drift_max": float(drift(x).max()) if len(x) else None,
                         "pumping": int((H < 0).sum()), "overshoot": int((H >= 0).sum()),
                         "d_demo_q05_q50_q95": np.quantile(d[keep], [0.05, 0.5, 0.95]).tolist() if len(keep) else None}
        atomic_npz(path, **{f"{n}/{k}": v for n, v in sets.items() for k, v in v.items()}, record=json.dumps(rec))
        log("select", s, json.dumps({k: (v if not isinstance(v, dict) else v["selected"]) for k, v in rec.items()}))


def selection(out: Path, s: int):
    f = np.load(out / "select" / f"s{s}.npz")
    return f, json.loads(str(f["record"]))


def stage_arms(out: Path, device: str, qualifying: list):
    jobs = {"duplication": [], "relabel": []}
    for s in qualifying:
        f, _ = selection(out, s)
        tr, va = supervision(out, s, "train"), supervision(out, s, "validation")
        m = len(f["collection/x"])
        dup = np.random.default_rng(np.random.SeedSequence([s, 19])).integers(0, len(tr.observations), m)
        jobs["duplication"].append((s, with_extra(tr, tr.observations[dup], tr.actions[dup], "duplicate"), va))
        jobs["relabel"].append((s, with_extra(tr, f["collection/x"], f["collection/a"], "relabel"), va))
    for arm in ("duplication", "relabel"):
        train(out, arm, jobs[arm], device)


def stage_evaluate(out: Path, qualifying: list) -> dict:
    task = Task(eval_seed=FINAL)
    res = {}
    for s in qualifying:
        path = out / "evaluate" / f"s{s}.json"
        if path.exists():
            res[s] = json.loads(path.read_text())
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        f, _ = selection(out, s)
        pols = [policy(out, arm, s) for arm in ARMS]
        rolls = task.rollouts(pols)
        r = {}
        for arm, pol, roll in zip(ARMS, pols, rolls):
            err = np.abs(np.clip(np.asarray(pol(f["diagnostic/x"])).reshape(-1), -3, 3) - f["diagnostic/a"])
            v = np.cos(np.arctan2(roll["states"][:, :-1, 0], roll["states"][:, :-1, 1]) - np.pi)
            r[arm] = {"action_error": float(err.mean()), **task.score(roll),
                      "per_start_occupancy": (v >= np.cos(np.deg2rad(8.0))).mean(1).tolist()}
            atomic_npz(out / "evaluate" / f"s{s}-{arm}-rollout.npz", states=roll["states"], applied=roll["applied"])
        atomic_json(path, r)
        res[s] = r
        log("evaluate", s, {a: (round(r[a]["action_error"], 4), round(r[a]["upright_occupancy"], 4)) for a in ARMS})
    return res


def decide(res: dict) -> dict:
    seeds = sorted(res)
    out, ok = {}, True
    for other in ("bc", "duplication"):
        for metric, want in (("action_error", "decreases"), ("upright_occupancy", "increases")):
            iv = union_interval([res[s]["relabel"][metric] - res[s][other][metric] for s in seeds], CONFIDENCE)
            lo, hi = iv["interval"]
            outcome = "increases" if lo > 0 else "decreases" if hi < 0 else "no_detectable_difference"
            out[f"relabel-{other}/{metric}"] = {**iv, "outcome": outcome, "required": want}
            ok &= outcome == want
    out["decision"] = "established" if ok else "not established"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if sha256_file(DATA) != DATA_SHA256:
        raise SystemExit("refused: the dataset is not the canonical continuous dataset")
    if a.out.exists() and not (a.resume and not (a.out / "positive-control.json").exists()):
        raise SystemExit(f"refused: {a.out} exists" + ("" if a.resume else " (an incomplete run takes --resume)"))
    a.out.mkdir(parents=True, exist_ok=a.resume)
    t0 = time.time()
    stage_data(a.out)
    train(a.out, "bc", [(s, supervision(a.out, s, "train"), supervision(a.out, s, "validation")) for s in SEEDS],
          a.device)
    with Heartbeat("select"):
        stage_select(a.out)
    qualifying = [s for s in SEEDS if selection(a.out, s)[1]["collection"]["selected"] >= MIN_STATES]
    records = {s: selection(a.out, s)[1] for s in SEEDS}
    base = {"predeclaration": "docs/studies/pendulum-positive-control.md (version 1.1)",
            "code_commit": git("rev-parse", "HEAD"), "code_dirty": bool(git("status", "--porcelain")),
            "data_sha256": DATA_SHA256, "seeds": list(SEEDS), "qualifying_seeds": qualifying, "selection": records,
            "learner": learner("relabel")}
    if len(qualifying) < MIN_SEEDS:
        doc = {**base, "status": "stopped",
               "reason": f"{len(qualifying)} of {len(SEEDS)} seeds have >= {MIN_STATES} off-support recovery states"}
        atomic_json(a.out / "positive-control.json", doc)
        log("stopped:", doc["reason"])
        return 1
    stage_arms(a.out, a.device, qualifying)
    with Heartbeat("evaluate"):
        res = stage_evaluate(a.out, qualifying)
    doc = {**base, "status": "completed", "outcomes": {str(s): {arm: {k: v for k, v in r[arm].items()
                                                                      if k != "per_start_occupancy"} for arm in ARMS}
                                                       for s, r in res.items()},
           "contrasts": decide(res), "seconds": round(time.time() - t0, 1),
           "policies": {f"{arm}/s{s}": hashlib.sha256((a.out / "policies" / arm / f"s{s}" / "policy.npz").read_bytes())
                        .hexdigest() for arm in ARMS for s in qualifying}}
    doc["decision"] = doc["contrasts"]["decision"]
    atomic_json(a.out / "positive-control.json", doc)
    log("decision", doc["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
