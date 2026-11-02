#!/usr/bin/env python3
"""The Pendulum capture-region positive control (docs/studies/pendulum-capture-control.md, version 1). Exploratory.

    uv run python tools/capture_control.py --out DIR [--device cuda] [--resume]

Stages, each written atomically under DIR and skipped on --resume when present: data, bc (also the collection
policies), select (BC rollouts from the collection and diagnostic starts, the capture-region states, thinned per
episode, and their checks), arms (duplication and relabel), evaluate (held-out capture action error and the final closed
loop), then capture-control.json with the intervals and the decision. The run is complete only when
capture-control.json exists. The learner, training, supervision and geometry come unchanged from
tools/positive_control.py. This is not evidence for CCIL.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

import positive_control as pc
from ilsb.analysis import union_interval
from ilsb.store import sha256_file
from ilsb.tasks.pendulum import Task, expert, subset

SEEDS = tuple(range(3311, 3321))
COLLECTION, DIAGNOSTIC, FINAL = 2205, 2206, 2207
EARLIER = (2201, 2202, 2203, 2204)  # the study's and the pumping-region control's start sets
PER_EPISODE, MIN_STATES, MIN_SEEDS = 40, 100, 8
ARMS = pc.ARMS
log, Heartbeat = pc.log, pc.Heartbeat


def stage_data(out: Path):
    from ccil.data.split import make_split
    from ccil.schema import read_trajectories
    for s in SEEDS:
        d = out / "data" / f"s{s}"
        if (d / "split.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        subset(pc.DATA, pc.R, s, d / "subset.npz")
        eps = [e.episode_id for e in read_trajectories(d / "subset.npz").episodes]
        pc.atomic_json(d / "split.json", make_split(eps, pc.VALIDATION_FRACTION, s))
    log("data ready", len(SEEDS), "seeds")


def capture_states(rollout_states: np.ndarray):
    """(selected frames as (episode, frame) pairs, LQR-branch count, captured count): the capture region, thinned to
    PER_EPISODE frames per rollout at round(linspace(0, n - 1, PER_EPISODE)), in time order."""
    E, T = rollout_states.shape[:2]
    flat = rollout_states.reshape(-1, 3)
    lqr = np.flatnonzero(np.abs(pc.e_of(flat)) < pc.LQR_WINDOW)
    captured = lqr[pc.recovers(Task(), flat[lqr])]
    keep = []
    for ep in range(E):
        idx = captured[captured // T == ep]
        if len(idx) > PER_EPISODE:
            idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, PER_EPISODE)).astype(int))]
        keep.append(idx)
    return np.concatenate(keep), len(lqr), len(captured)


def stage_select(out: Path):
    starts = {k: Task(eval_seed=k).starts() for k in (*EARLIER, COLLECTION, DIAGNOSTIC, FINAL)}
    for a in starts:
        for b in starts:
            if a < b and (pc.phys(starts[a])[:, None] == pc.phys(starts[b])[None]).all(-1).any():
                raise RuntimeError(f"start sets {a} and {b} share a start")
    for s in SEEDS:
        path = out / "select" / f"s{s}.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        pol, rec, sets = pc.policy(out, "bc", s), {}, {}
        for name, seed in (("collection", COLLECTION), ("diagnostic", DIAGNOSTIC)):
            states = Task(eval_seed=seed).rollouts([pol])[0]["states"]
            keep, n_lqr, n_cap = capture_states(states)
            x = states.reshape(-1, 3)[keep]
            a = expert(x)
            nxt = Task().step(x, a)[0]
            sets[name] = {"x": x, "a": a, "frame": keep}
            q = lambda v: np.quantile(v, [0.05, 0.5, 0.95]).tolist() if len(v) else None  # noqa: E731
            rec[name] = {"visited": int(states.shape[0] * states.shape[1]), "lqr_branch": n_lqr, "captured": n_cap,
                         "selected": int(len(keep)), "episodes_contributing": int(len(np.unique(keep // states.shape[1]))),
                         "bounds_ok": bool((np.abs(a) <= 3).all()),
                         "transitions_finite_no_added_drift": bool(np.isfinite(nxt).all() and (
                             np.abs(pc.drift(nxt) - pc.drift(x)) <= pc.MANIFOLD_TOL).all()),
                         "state_drift_max": float(pc.drift(x).max()) if len(x) else None,
                         "e_q05_q50_q95": q(pc.e_of(x)), "omega_q05_q50_q95": q(x[:, 2])}
        pc.atomic_npz(path, **{f"{n}/{k}": v for n, v in sets.items() for k, v in v.items()}, record=json.dumps(rec))
        log("select", s, json.dumps({k: (v["lqr_branch"], v["captured"], v["selected"]) for k, v in rec.items()}))


def selection(out: Path, s: int):
    f = np.load(out / "select" / f"s{s}.npz")
    return f, json.loads(str(f["record"]))


def stage_arms(out: Path, device: str, qualifying: list):
    jobs = {"duplication": [], "relabel": []}
    for s in qualifying:
        f, _ = selection(out, s)
        tr, va = pc.supervision(out, s, "train"), pc.supervision(out, s, "validation")
        m = len(f["collection/x"])
        dup = np.random.default_rng(np.random.SeedSequence([s, 19])).integers(0, len(tr.observations), m)
        jobs["duplication"].append((s, pc.with_extra(tr, tr.observations[dup], tr.actions[dup], "duplicate"), va))
        jobs["relabel"].append((s, pc.with_extra(tr, f["collection/x"], f["collection/a"], "relabel"), va))
    for arm in ("duplication", "relabel"):
        pc.train(out, arm, jobs[arm], device)


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
        pols = [pc.policy(out, arm, s) for arm in ARMS]
        r = {}
        for arm, pol, roll in zip(ARMS, pols, task.rollouts(pols)):
            err = np.abs(np.clip(np.asarray(pol(f["diagnostic/x"])).reshape(-1), -3, 3) - f["diagnostic/a"])
            r[arm] = {"action_error": float(err.mean()), **task.score(roll)}
            pc.atomic_npz(out / "evaluate" / f"s{s}-{arm}-rollout.npz", states=roll["states"], applied=roll["applied"])
        pc.atomic_json(path, r)
        res[s] = r
        log("evaluate", s, {a: (round(r[a]["action_error"], 4), round(r[a]["final_hold"], 4)) for a in ARMS})
    return res


def decide(res: dict) -> dict:
    """Required: Relabel - Duplication, action error decreases and final hold increases. Everything else is reported."""
    seeds = sorted(res)
    out, ok = {}, True
    for other in ("duplication", "bc"):
        for metric, want in (("action_error", "decreases"), ("final_hold", "increases"),
                             ("upright_occupancy", "increases")):
            iv = union_interval([res[s]["relabel"][metric] - res[s][other][metric] for s in seeds], pc.CONFIDENCE)
            lo, hi = iv["interval"]
            outcome = "increases" if lo > 0 else "decreases" if hi < 0 else "no_detectable_difference"
            required = other == "duplication" and metric != "upright_occupancy"
            out[f"relabel-{other}/{metric}"] = {**iv, "outcome": outcome, "required": want if required else None}
            ok &= outcome == want or not required
    out["decision"] = "passes" if ok else "not established"
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    a = ap.parse_args()
    if sha256_file(pc.DATA) != pc.DATA_SHA256:
        raise SystemExit("refused: the dataset is not the canonical continuous dataset")
    if a.out.exists() and not (a.resume and not (a.out / "capture-control.json").exists()):
        raise SystemExit(f"refused: {a.out} exists" + ("" if a.resume else " (an incomplete run takes --resume)"))
    a.out.mkdir(parents=True, exist_ok=a.resume)
    t0 = time.time()
    stage_data(a.out)
    pc.train(a.out, "bc", [(s, pc.supervision(a.out, s, "train"), pc.supervision(a.out, s, "validation"))
                           for s in SEEDS], a.device)
    with Heartbeat("select"):
        stage_select(a.out)
    records = {s: selection(a.out, s)[1] for s in SEEDS}
    qualifying = [s for s in SEEDS if records[s]["collection"]["selected"] >= MIN_STATES]
    base = {"predeclaration": "docs/studies/pendulum-capture-control.md (version 1)", "exploratory": True,
            "code_commit": pc.git("rev-parse", "HEAD"), "code_dirty": bool(pc.git("status", "--porcelain")),
            "data_sha256": pc.DATA_SHA256, "seeds": list(SEEDS), "qualifying_seeds": qualifying, "selection": records,
            "learner": pc.learner("relabel")}
    if len(qualifying) < MIN_SEEDS:
        doc = {**base, "status": "stopped",
               "reason": f"{len(qualifying)} of {len(SEEDS)} seeds have >= {MIN_STATES} capture-region states"}
        pc.atomic_json(a.out / "capture-control.json", doc)
        log("stopped:", doc["reason"])
        return 1
    stage_arms(a.out, a.device, qualifying)
    with Heartbeat("evaluate"):
        res = stage_evaluate(a.out, qualifying)
    doc = {**base, "status": "completed", "outcomes": {str(s): r for s, r in res.items()},
           "contrasts": decide(res), "seconds": round(time.time() - t0, 1),
           "policies": {f"{arm}/s{s}": hashlib.sha256((a.out / "policies" / arm / f"s{s}" / "policy.npz").read_bytes())
                        .hexdigest() for arm in ARMS for s in qualifying}}
    doc["decision"] = doc["contrasts"]["decision"]
    pc.atomic_json(a.out / "capture-control.json", doc)
    log("decision", doc["decision"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
