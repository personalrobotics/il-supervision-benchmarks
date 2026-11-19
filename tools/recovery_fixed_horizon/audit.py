"""#82 fresh-process audit of one run directory (recovery-fixed-horizon.md).

    PY tools/recovery_fixed_horizon/audit.py --run-dir DIR --workers N --pendulum P --policy POLICY --stage2 S2 \
        --stage3 S3 --stage4 S4 [--out AUDIT.json]

It reruns every unit with the replay planners (the stored plans of every arm) and reports the first refusal in a
fixed order: inputs; per unit: start, then per fixed-H arm trajectory, validity, agreement, handoff, audit, outcome,
then the any-horizon and shortest-feasible arms, and any other field. Finally record.json is recomputed from the unit
files and must match byte for byte. The planning is not rerun; the byte-identical second run covers it. Exit 0 only
with no refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as RUN  # noqa: E402
import fixedh as F  # noqa: E402

C = F.C
STUDY, RUN_DIR = None, None
REC = lambda *keys: lambda a: [[g.get(k) for k in keys] for g in a["recoveries"]]  # noqa: E731
FIXED = (("trajectory", lambda a: [a.get(k) for k in ("actions", "modes", "states", "phases", "distances")]),
         ("validity", lambda a: [[g["entry"], g["predicted_miss"], g["failure"], g["failed_plan"]]
                                 for g in a["recoveries"]]),
         ("agreement", REC("D", "base_actions", "close", "agree")),
         ("handoff", lambda a: [REC("handoff")(a), a.get("transitions")]),
         ("audit", REC("endpoint_miss", "refined_miss", "audit_checks", "declared", "audited")),
         ("outcome", lambda a: [a.get(k) for k in ("outcome", "departure", "bc_first_departure", "base_reproduced",
                                                   "recovered", "upright", "bc_upright")]))
START = ("kind", "unit", "population", "episode", "t", "radius", "x0")


def first_refusal(stored: dict, again: dict):
    if [stored.get(k) for k in START] != [again.get(k) for k in START]:
        return "start"
    for arm in ("fixed_learned", "fixed_exact"):
        if arm in stored["arms"] or arm in again["arms"]:
            a, b = stored["arms"].get(arm), again["arms"].get(arm)
            if a is None or b is None:
                return "arms"
            for name, view in FIXED:
                if view(a) != view(b):
                    return name
    for arm in ("any_horizon", "shortest_feasible"):
        if stored["arms"].get(arm) != again["arms"].get(arm):
            return arm
    return "other" if stored != again else None


def audit_unit(unit):
    rec = json.loads((RUN_DIR / "units" / RUN.name(unit)).read_text())
    try:
        again = json.loads(C.dumps(STUDY.run_unit(unit, rec)))
    except (ValueError, IndexError, KeyError, TypeError):
        return unit, "plans"
    return unit, first_refusal(rec, again)


def init(set_name, pendulum, policy, stage3, stage2, stage4, run) -> None:
    global STUDY, RUN_DIR
    RUN_DIR = run
    RUN.init(set_name, pendulum, policy, stage3, stage2, stage4)
    STUDY = RUN.STUDY


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    for a in ("pendulum", "policy", "stage2", "stage3", "stage4"):
        p.add_argument(f"--{a}", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run = args.run_dir
    ident = json.loads((run / "identity.json").read_text())
    args.set = ident["set"]
    log = C.Log(run / "audit.log")
    refusals = []
    inputs = (args.set, args.pendulum, args.policy, args.stage3, args.stage2, args.stage4, run)
    init(*inputs)  # refuses unless every input is the protocol's
    if ident["identity"] != RUN.build_identity(STUDY, args) | {"ccil_commit": ident["identity"]["ccil_commit"]}:
        refusals.append({"unit": None, "reason": "inputs"})
    units = STUDY.units
    out = C.schedule(audit_unit, units, args.workers, log, "audit pendulum fixed-H", init, inputs)
    refusals += [{"unit": list(u), "reason": why} for u, why in (out[u] for u in units) if why]
    recs = [json.loads((run / "units" / RUN.name(u)).read_text()) for u in units]
    stored = json.loads((run / "record.json").read_text())
    again = RUN.record(recs, ident["identity"], stored["set"], STUDY.tol)
    again["code"] = stored["code"]  # the record's commit is the run's; the recomputation is of its content
    if C.dumps(again) != (run / "record.json").read_text():
        refusals.append({"unit": None, "reason": "record"})
    result = {"set": ident["set"], "units": len(units), "refusals": refusals,
              "first_refusal": refusals[0]["reason"] if refusals else None, "ok": not refusals,
              "code": C.code_identity()}
    text = C.dumps(result)
    if args.out:
        C.write_atomic(args.out, text)
    print(text, end="")
    sys.exit(0 if not refusals else 1)


if __name__ == "__main__":
    main()
