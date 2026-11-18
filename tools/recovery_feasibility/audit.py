"""#70 fresh-process audit of one run directory (recovery-feasibility.md "Smoke, then the run").

    PY tools/recovery_feasibility/audit.py --run-dir DIR --workers N INPUT [--out AUDIT.json]

It rebuilds the system from INPUT and, for every start, recomputes the start and its redraws, the eligibility, the
M = 1 certificate and the oracle, then replays every returned segment (all solvers, all restarts) and recomputes its
checks. Each start reports its first refusal in a fixed order:
  start, eligibility, certificate, oracle, then per segment bounds, envelope, replay, terminal, refinement,
  continuation, validity, then selection (first valid segment and ambiguity).
Finally the record is recomputed from the start files and must equal record.json byte for byte. Solvers are not
rerun here; the byte-identical second run covers them. Exit status 0 only with no refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402
import run as R  # noqa: E402

ORDER = ("bounds", "envelope", "replay", "terminal", "refinement", "continuation", "validity")
RUN_DIR = None


def segment_refusal(sy, a: int, x0, M: int, g: dict):
    """The first check on which a stored segment disagrees with its recomputation."""
    re = sy.segment(a, x0, M, g["actions"])
    checks = {
        "bounds": g["checks"]["bounds"] == sy.check_bounds(g["actions"]) == re["checks"]["bounds"],
        "envelope": g["checks"]["envelope"] == sy.check_envelope(g["states"]) == re["checks"]["envelope"],
        "replay": g["states"] == re["states"],
        "terminal": g["miss"] == re["miss"] and g["checks"]["terminal"] == re["checks"]["terminal"],
        "refinement": g["refined_miss"] == re["refined_miss"]
        and g["checks"].get("refinement") == re["checks"].get("refinement"),
        "continuation": g["continuation_px"] == re["continuation_px"]
        and g["checks"].get("continuation") == re["checks"].get("continuation"),
        "validity": g["valid"] == all(g["checks"].values()) == re["valid"],
    }
    return next((k for k in ORDER if not checks[k]), None)


def audit_start(unit: tuple):
    sy = R.SYSTEM
    path = RUN_DIR / "starts" / f"{unit[0]:03d}.json"
    rec = json.loads(path.read_text())
    s, a, r, d = unit
    sid = C.SYSTEMS[sy.name]
    if (rec["start"], rec["anchor"], rec["radius"], rec["direction"]) != (s, a, r, d) or \
            rec["anchor_key"] != list(sy.anchors[a]):
        return s, "start", None
    x0, redraws = C.available(lambda g: sy.propose(a, r, g), C.rng(*C.key(sid, sy.smoke, s)))
    if rec["redraws"] != redraws or rec["available"] != (x0 is not None) or \
            (x0 is not None and {k: rec[k] for k in sy.start_record(x0)} != json.loads(C.dumps(sy.start_record(x0)))):
        return s, "start", None
    if x0 is None:
        return s, None, None
    elig = sy.eligibility(a)
    if rec["eligibility"] != (None if elig is None else json.loads(C.dumps(elig))):
        return s, "eligibility", None
    ok = (lambda M: True) if elig is None else (lambda M: elig["eligible"][str(M)])
    m1 = sy.certificate(a, x0) if ok(1) else None
    if rec["m1"] != (None if m1 is None else json.loads(C.dumps(m1))):
        return s, "certificate", None
    for M in sy.ladder:
        rung = rec["rungs"][str(M)]
        if rung["eligible"] != ok(M):
            return s, "eligibility", M
        if not ok(M):
            continue
        orc, demo = sy.oracle(a, x0, M), sy.demo_replay_miss(a, M)
        if rung["oracle"] != (None if orc is None else json.loads(C.dumps(orc))) or rung["demo_replay_miss"] != demo:
            return s, "oracle", M
        for solver in sy.solvers:
            segs = rung[solver]["segments"]
            for g in segs:
                why = segment_refusal(sy, a, x0, M, g)
                if why:
                    return s, why, M
            if rung[solver]["first_valid"] != C.first_valid(segs) or \
                    rung[solver]["ambiguity"] != C.ambiguity(segs, sy.ubar):
                return s, "selection", M
    return s, None, None


def init(name, smoke, data, run_dir):
    global RUN_DIR
    RUN_DIR = run_dir
    R.init(name, smoke, data)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--quickstart", type=Path)
    p.add_argument("--pendulum", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    ident = json.loads((args.run_dir / "identity.json").read_text())
    name, smoke = ident["system"], ident["set"] == "smoke"
    data = {"double_integrator": args.quickstart, "pendulum": args.pendulum, "pusht": args.source}[name]
    log = C.Log(args.run_dir / "audit.log")
    sy = R.build(name, smoke, data)
    sy.smoke = smoke
    refusals = []
    if json.loads(C.dumps(sy.identity)) != ident["identity"]:
        refusals.append({"start": None, "reason": "identity", "M": None})
    units = C.start_ids(len(sy.anchors))
    out = C.schedule(audit_start, units, args.workers, log, f"audit {name}", init, (name, smoke, data, args.run_dir))
    refusals += [{"start": s, "reason": why, "M": M} for s, why, M in (out[u] for u in units) if why]
    recs = [json.loads((args.run_dir / "starts" / f"{u[0]:03d}.json").read_text()) for u in units]
    stored = json.loads((args.run_dir / "record.json").read_text())
    import hashlib
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    again = R.record(sy, recs, stored["set"], data_sha)
    again["code"] = stored["code"]  # the record's commit is the run's; the recomputation is of its content
    if C.dumps(again) != (args.run_dir / "record.json").read_text():
        refusals.append({"start": None, "reason": "record", "M": None})
    result = {"system": name, "set": ident["set"], "starts": len(units), "refusals": refusals,
              "first_refusal": refusals[0]["reason"] if refusals else None, "ok": not refusals,
              "code": C.code_identity()}
    text = C.dumps(result)
    if args.out:
        C.write_atomic(args.out, text)
    print(text, end="")
    sys.exit(0 if not refusals else 1)


if __name__ == "__main__":
    main()
