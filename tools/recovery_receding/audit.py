"""#76 fresh-process audit of one run directory (recovery-receding.md "Smoke, then the run").

    PY tools/recovery_receding/audit.py --run-dir DIR --workers N --pendulum P --policy POLICY --stage2 STAGE2 \
        --stage3 STAGE3 [--out AUDIT.json]

For every start it recomputes everything but the planning from the stored executed actions and observed states, and
reports the first refusal in a fixed order:
  start; per arm and M: bounds, replay, envelope, endpoint, refinement, declared, model_error, failure_check;
  then recovery, open_loop, and any other field.
Finally record.json is recomputed from the start files and must match byte for byte. The planning is not rerun; the
byte-identical second run covers it. Exit 0 only with no refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receding as R  # noqa: E402
import run as RUN  # noqa: E402

C = R.C
STUDY, RUN_DIR = None, None
EXECUTION = (("bounds", lambda g: [g["actions"], g["audit_checks"]["bounds"]]),
             ("replay", lambda g: [g["states"], g["audit_checks"]["replay"]]),
             ("envelope", lambda g: [g["audit_checks"]["envelope"]]),
             ("endpoint", lambda g: [g["audit_checks"]["endpoint"]]),
             ("refinement", lambda g: [g["refined_miss"], g["audit_checks"]["refinement"]]),
             ("declared", lambda g: [g["complete"], g["observed_miss"], g["declared"], g["first_audit_failure"]]),
             ("model_error", lambda g: [g["model_one_step_error"]]))
START = ("start", "anchor", "anchor_key", "radius", "direction", "x0", "t", "n")


def first_refusal(stored: dict, again: dict, sy) -> tuple:
    if [stored.get(k) for k in START] != [again.get(k) for k in START]:
        return "start", None, None
    for arm in R.ARMS:
        for M in sy.ladder:
            g, h = stored["arms"][arm]["executions"][str(M)], again["arms"][arm]["executions"][str(M)]
            if not g["audit_checks"]["bounds"] == sy.check_bounds(g["actions"]):
                return "bounds", arm, M
            for name, view in EXECUTION:
                if view(g) != view(h):
                    return name, arm, M
            if g["failure"] not in (None, "solver", "envelope") or (g["failure"] is None) != (len(g["actions"]) == M):
                return "failure_check", arm, M
        if stored["arms"][arm]["recovery"] != again["arms"][arm]["recovery"]:
            return "recovery", arm, None
    if stored.get("open_loop") != again.get("open_loop"):
        return "open_loop", None, None
    if stored != again:
        return "other", None, None
    return None, None, None


def audit_start(unit: tuple):
    rec = json.loads((RUN_DIR / "starts" / f"{unit[0]:03d}.json").read_text())
    keep = ("actions", "states", "predicted_miss", "failure")
    executions = {arm: {M: {k: e[k] for k in keep} for M, e in rec["arms"][arm]["executions"].items()}
                  for arm in R.ARMS}
    try:
        again = json.loads(C.dumps(STUDY.evaluate(unit, executions)))
    except ValueError:
        return unit[0], "start", None, None
    why, arm, M = first_refusal(rec, again, STUDY.sy)
    return unit[0], why, arm, M


def init(smoke, pendulum, policy, stage3, stage2, run) -> None:
    global STUDY, RUN_DIR
    RUN_DIR = run
    STUDY = R.Study(smoke, pendulum, policy, stage3, stage2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pendulum", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--stage3", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run = args.run_dir
    ident = json.loads((run / "identity.json").read_text())
    smoke = ident["set"] == "smoke"
    log = C.Log(run / "audit.log")
    refusals = []
    inputs = (smoke, args.pendulum, args.policy, args.stage3, args.stage2, run)
    init(*inputs)  # refuses unless every input is the protocol's
    if ident["identity"] != RUN.build_identity(STUDY.sy, args.pendulum, args.policy, args.stage2, args.stage3,
                                               ident["set"]) | {"ccil_commit": ident["identity"]["ccil_commit"]}:
        refusals.append({"start": None, "reason": "inputs", "arm": None, "M": None})
    units = STUDY.units
    out = C.schedule(audit_start, units, args.workers, log, "audit pendulum receding", init, inputs)
    refusals += [{"start": s, "reason": why, "arm": arm, "M": M} for s, why, arm, M in (out[u] for u in units) if why]
    recs = [json.loads((run / "starts" / f"{u[0]:03d}.json").read_text()) for u in units]
    stored = json.loads((run / "record.json").read_text())
    again = RUN.record(recs, ident["identity"], stored["set"], STUDY.sy.ladder, STUDY.sy.eps)
    again["code"] = stored["code"]  # the record's commit is the run's; the recomputation is of its content
    if C.dumps(again) != (run / "record.json").read_text():
        refusals.append({"start": None, "reason": "record", "arm": None, "M": None})
    result = {"set": ident["set"], "starts": len(units), "refusals": refusals,
              "first_refusal": refusals[0]["reason"] if refusals else None, "ok": not refusals,
              "code": C.code_identity()}
    text = C.dumps(result)
    if args.out:
        C.write_atomic(args.out, text)
    print(text, end="")
    sys.exit(0 if not refusals else 1)


if __name__ == "__main__":
    main()
