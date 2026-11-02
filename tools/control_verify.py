#!/usr/bin/env python3
"""Bind a positive-control report to the exact bytes it summarizes, and recompute it from them (read-only).

    uv run python tools/control_verify.py --kind pumping|capture --run DIR --work NEW_DIR --out MANIFEST.json

DIR is a completed run of tools/positive_control.py (pumping) or tools/capture_control.py (capture); it is never
written. The verifier:
1. hashes every artifact the report depends on: the subsets and splits, the selected-state files, every policy
   file, the final rollouts and the per-seed evaluation records, and the report itself;
2. checks the policy hashes against the report;
3. recomputes the selection from the run's BC policies into NEW_DIR and requires every array to equal the run's;
4. recomputes the final rollouts, action errors and task scores from the run's policies and the recomputed
   selection, and requires them to equal the run's rollout files, evaluation records and report outcomes exactly;
5. recomputes the intervals and the decision and requires them to equal the report's.
MANIFEST.json records the hashes and each check; the exit status is 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import capture_control  # noqa: E402
import positive_control  # noqa: E402

KINDS = {"pumping": (positive_control, "positive-control.json"), "capture": (capture_control, "capture-control.json")}


def same_npz(a: Path, b: Path) -> bool:
    x, y = np.load(a), np.load(b)
    return sorted(x.files) == sorted(y.files) and all(np.array_equal(x[k], y[k]) for k in x.files)


def strip(r: dict) -> dict:
    return {arm: {k: v for k, v in m.items() if k != "per_start_occupancy"} for arm, m in r.items()}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--kind", choices=sorted(KINDS), required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    mod, report_name = KINDS[a.kind]
    pc = positive_control
    report_p = a.run / report_name
    report = json.loads(report_p.read_text())
    if report.get("status") != "completed":
        raise SystemExit("refused: the report is not a completed run")
    if a.work.exists():
        raise SystemExit(f"refused: {a.work} exists")
    t0 = time.time()
    seeds, qual, arms = mod.SEEDS, report["qualifying_seeds"], pc.ARMS
    files = sorted([*(a.run / "data").rglob("*"), *(a.run / "select").glob("s*.npz"),
                    *(a.run / "policies").rglob("*"), *(a.run / "evaluate").glob("s*")])
    files = [f for f in files if f.is_file() and ".tmp" not in f.name]
    hashes = {str(f.relative_to(a.run)): pc.sha256_file(f) for f in files}
    checks = {"policy_hashes_match_report": all(
        hashes.get(f"policies/{k.split('/')[0]}/{k.split('/')[1]}/policy.npz") == h for k, h in report["policies"].items())}
    pc.log("hashed", len(hashes), "artifacts; policy hashes match the report:", checks["policy_hashes_match_report"])

    # a work directory that reads the run's data and policies and recomputes everything else
    a.work.mkdir(parents=True)
    os.symlink((a.run / "data").resolve(), a.work / "data")
    os.symlink((a.run / "policies").resolve(), a.work / "policies")
    with pc.Heartbeat("recompute selection"):
        mod.stage_select(a.work)
    checks["selection_identical"] = all(same_npz(a.run / "select" / f"s{s}.npz", a.work / "select" / f"s{s}.npz")
                                        for s in seeds)
    sel_records = {str(s): json.loads(str(np.load(a.work / "select" / f"s{s}.npz")["record"])) for s in seeds}
    checks["selection_records_match_report"] = sel_records == json.loads(json.dumps(report["selection"]))
    with pc.Heartbeat("recompute evaluation"):
        res = mod.stage_evaluate(a.work, qual)
    checks["rollouts_identical"] = all(same_npz(a.run / "evaluate" / f"s{s}-{arm}-rollout.npz",
                                                a.work / "evaluate" / f"s{s}-{arm}-rollout.npz")
                                       for s in qual for arm in arms)
    checks["evaluation_records_identical"] = all(
        json.loads((a.run / "evaluate" / f"s{s}.json").read_text()) == json.loads(
            (a.work / "evaluate" / f"s{s}.json").read_text()) for s in qual)
    checks["outcomes_match_report"] = {str(s): strip(r) for s, r in res.items()} == {
        s: strip(r) for s, r in report["outcomes"].items()}
    contrasts = json.loads(json.dumps(mod.decide(res), sort_keys=True, default=float))
    checks["contrasts_and_decision_match_report"] = contrasts == report["contrasts"] and \
        contrasts["decision"] == report["decision"]
    doc = {"kind": a.kind, "run": str(a.run), "report": report_name, "report_sha256": pc.sha256_file(report_p),
           "decision": report["decision"], "artifacts_sha256": hashes, "checks": checks,
           "all_pass": all(checks.values()), "verify_commit": pc.git("rev-parse", "HEAD"),
           "verify_dirty": bool(pc.git("status", "--porcelain")), "seconds": round(time.time() - t0, 1)}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    pc.atomic_json(a.out, doc)
    pc.log("checks", json.dumps(checks), "all pass:", doc["all_pass"])
    return 0 if doc["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
