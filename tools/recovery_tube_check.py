#!/usr/bin/env python3
"""Recheck the Pendulum recovery-tube result from committed evidence (docs/studies/pendulum-recovery-tube.md, "Result").

    uv run python tools/recovery_tube_check.py decisions [--evidence DIR]
    uv run python tools/recovery_tube_check.py replay --run RUN --out FILE.json

decisions needs only a clone. From the committed per-policy summaries in DIR (default
docs/studies/evidence/pendulum-recovery-tube) it checks the exact declared arm x seed matrix of every cohort, that
every record names the committed audit.json and geometry.json, and that each replay-CCIL record's result NPZ hash
equals its original CCIL record's; it then recomputes every contrast with the evaluator's own paired() and requires
each decision block to match decision.json: outcomes, seeds and counts exactly, floats to a relative 1e-12 (math
libraries differ in the last bits across platforms; bitwise_equal reports which blocks are bit-identical here). Exit status 0 only when every check passes.

replay runs where the study's run directory is (it reads the original checkpoints named in RUN/audit.json and the
replay policies in RUN/matched/policies, both read-only) and writes, per seed, both files' sha256, their array keys,
whether every array is equal, and the largest absolute difference.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_tube as rt  # noqa: E402

EVIDENCE = rt.pc.REPO / "docs/studies/evidence/pendulum-recovery-tube"
MATRIX = {"capture": (("bc", "duplication", "relabel"), range(3311, 3321)),
          "corrected": (("bc", "ccil"), range(1201, 1221)),
          "matched": (("duplication", "replay_ccil"), range(1201, 1221))}


def close(a, b) -> bool:
    """Equal, except floats may differ in the last bits across platforms' math libraries (relative 1e-12)."""
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def decisions(ev: Path) -> dict:
    aud_sha, geo_sha = rt.sha256_file(ev / "audit.json"), rt.sha256_file(ev / "geometry.json")
    dec = json.loads((ev / "decision.json").read_text())
    checks = {"decision_names_committed_audit_and_geometry":
              dec["audit_sha256"] == aud_sha and dec["geometry_sha256"] == geo_sha}
    res = {}
    for cohort, (arms, seeds) in MATRIX.items():
        d = ev / "evaluate" / cohort
        want = {f"{a}-s{s}" for a in arms for s in seeds}
        checks[f"{cohort}_exact_matrix"] = {p.stem for p in d.glob("*.json")} - {"reference"} == want
        recs = {(a, s): json.loads((d / f"{a}-s{s}.json").read_text()) for a in arms for s in seeds}
        checks[f"{cohort}_records_name_committed_audit_and_geometry"] = all(
            r["identity"]["audit_sha256"] == aud_sha and r["identity"]["geometry_sha256"] == geo_sha
            and r["identity"]["cohort"] == cohort for r in recs.values())
        res[cohort] = recs
    checks["replay_results_equal_original"] = all(
        res["matched"][("replay_ccil", s)]["npz_sha256"] == res["corrected"][("ccil", s)]["npz_sha256"]
        for s in MATRIX["matched"][1])
    prim = lambda d: d["anchors"]["primary_edge_recovery"]  # noqa: E731
    both, bitwise = {**res["matched"], **res["corrected"]}, {}
    fit = rt.paired(res["capture"], "relabel", "duplication", lambda d: d["held_out_capture"]["recovered"])
    sig = rt.paired(res["corrected"], "ccil", "bc", prim)
    att = rt.paired(res["matched"], "replay_ccil", "duplication", prim)
    redo = {"fitness": fit, "signal": sig, "attribution": att,
            "attribution.retraining_drift_ccil_minus_replay": rt.paired(both, "ccil", "replay_ccil", prim),
            "attribution.exposure_duplication_minus_bc": rt.paired(both, "duplication", "bc", prim),
            "attribution.total_ccil_minus_bc": rt.paired(res["corrected"], "ccil", "bc", prim)}
    for name, block in redo.items():
        top, _, sub = name.partition(".")
        got = dec[top][sub] if sub else dec[top]
        block = json.loads(json.dumps(block, default=float))
        checks[f"{name}_matches_decision"] = all(close(got.get(k), v) for k, v in block.items())
        bitwise[name] = all(got.get(k) == v for k, v in block.items())
    return {"bitwise_equal": bitwise, "evidence": str(ev), "decision_sha256": rt.sha256_file(ev / "decision.json"), "checks": checks,
            "all_pass": all(checks.values())}


def replay(run: Path) -> dict:
    units = json.loads((run / "audit.json").read_text())["corrected"]["units"]
    seeds = {}
    for s in MATRIX["matched"][1]:
        o_p = Path(units[f"continuous/ccil/s{s}"]["checkpoint"])
        r_p = run / "matched/policies/replay_ccil" / f"s{s}" / "policy.npz"
        o, r = np.load(o_p, allow_pickle=False), np.load(r_p, allow_pickle=False)
        keys = sorted(set(o.files) | set(r.files))
        same, worst = sorted(o.files) == sorted(r.files), 0.0
        for k in sorted(set(o.files) & set(r.files)):
            a, b = o[k], r[k]
            eq = a.shape == b.shape and a.dtype == b.dtype and np.array_equal(a, b)
            same &= bool(eq)
            if a.dtype.kind in "fc" and a.shape == b.shape:
                worst = max(worst, float(np.max(np.abs(a - b), initial=0.0)))
        seeds[str(s)] = {"original_sha256": rt.sha256_file(o_p), "replay_sha256": rt.sha256_file(r_p),
                         "original_checkpoint_sha256_in_audit": units[f"continuous/ccil/s{s}"]["checkpoint_sha256"],
                         "keys": keys, "all_arrays_equal": same, "max_abs_difference": worst}
    return {"run": str(run), "seeds": seeds, "all_seeds_equal": all(v["all_arrays_equal"] for v in seeds.values()),
            "check_commit": rt.pc.git("rev-parse", "HEAD"), "check_dirty": bool(rt.pc.git("status", "--porcelain"))}


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("decisions")
    d.add_argument("--evidence", type=Path, default=EVIDENCE)
    r = sub.add_parser("replay")
    r.add_argument("--run", type=Path, required=True)
    r.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    if a.cmd == "decisions":
        doc = decisions(a.evidence)
        print(json.dumps(doc, indent=1))
        return 0 if doc["all_pass"] else 1
    doc = replay(a.run)
    rt.atomic_json(a.out, doc)
    print(json.dumps({k: v for k, v in doc.items() if k != "seeds"}))
    return 0 if doc["all_seeds_equal"] else 1


if __name__ == "__main__":
    sys.exit(main())
