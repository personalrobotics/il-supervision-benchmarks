#!/usr/bin/env python3
"""Recheck the recovery-segment Stage A decision (docs/studies/pendulum-recovery-segments.md) from its artifacts.

    uv run python tools/recovery_segments_check.py --run DIR [--stage-a FILE]

DIR is a Stage A run directory, or the extracted release archive of one. FILE (default DIR/stage-a.json; the committed
docs/studies/evidence/pendulum-recovery-segments/stage-a.json for the published result) is the record to check. The
check verifies the sha256 of every artifact FILE names, recomputes every gate of every seed, M and arm from the
generate and failure files with the tool's own gates(), and the selection and stop outcome with its select(); it also
reconciles every accepted row to one segment, checks the supervision view's schema (finite, within the bound, on the
circle within the exact map's float drift, positive_control.MANIFOLD_TOL) and that FILE is not a smoke run. Gate values must match to a relative 1e-12 and every boolean exactly.
Exit status 0 only when every check passes.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import recovery_segments as rs  # noqa: E402
from ilsb.store import sha256_file  # noqa: E402


def close(a, b) -> bool:
    if isinstance(a, float) and isinstance(b, float):
        return abs(a - b) <= 1e-12 * max(1.0, abs(a), abs(b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(close(a[k], b[k]) for k in a)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(close(x, y) for x, y in zip(a, b))
    return a == b


def check(run: Path, record: Path) -> dict:
    doc = json.loads(record.read_text())
    seeds, ladder = doc["seeds"], doc["ladder"]
    checks = {"not_smoke": doc["smoke"] is False,
              "artifacts_verify": all((run / p).exists() and sha256_file(run / p) == h
                                      for p, h in doc["artifacts_sha256"].items())}
    per = {s: {M: {arm: rs.gates(run, s, M, arm, rs.failures(run, s)[0]["held_out/x"]) for arm in rs.ARMS}
               for M in ladder} for s in seeds}
    redo = json.loads(json.dumps({str(s): {str(M): per[s][M] for M in ladder} for s in seeds}, default=float))
    checks["gates_match"] = close(redo, doc["gates"])
    sel = json.loads(json.dumps(rs.select(per, seeds, ladder), default=float))
    checks["selection_matches"] = all(close(sel[k], doc[k]) for k in sel)
    rows_ok = True
    for s in seeds:
        for M in ladder:
            for arm in rs.ARMS:
                z = np.load(run / "generate" / f"s{s}-M{M}-{arm}.npz")
                acc = z["accepted"]
                x, u = z["states"][acc][:, :-1].reshape(-1, 3), z["U"][acc].reshape(-1)
                rows_ok &= bool(len(x) == M * acc.sum() and np.isfinite(x).all() and np.isfinite(u).all()
                                and (np.abs(u) <= rs.BOUND).all()
                                and (np.abs(x[:, 0] ** 2 + x[:, 1] ** 2 - 1) <= rs.pc.MANIFOLD_TOL).all())
    checks["supervision_rows_schema"] = rows_ok
    return {"record": str(record), "record_sha256": sha256_file(record), "checks": checks,
            "all_pass": all(checks.values())}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--stage-a", type=Path)
    a = ap.parse_args()
    doc = check(a.run, a.stage_a or a.run / "stage-a.json")
    print(json.dumps(doc, indent=1))
    return 0 if doc["all_pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
