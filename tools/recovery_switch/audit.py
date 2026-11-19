"""#79 fresh-process audit of one run directory (recovery-switch.md "Smoke, then the run").

    PY tools/recovery_switch/audit.py --run-dir DIR --workers N --pendulum P --policy POLICY --stage2 STAGE2 \
        --stage3 STAGE3 --stage4 STAGE4 [--out AUDIT.json]

It recomputes the calibration and compares it byte for byte with calibration.json. Then it reruns every episode with
the replay planner (the stored first-solve plans and executed recovery actions) and reports the first refusal in a
fixed order:
  calibration; inputs; per episode: start, tolerance, plans, trajectory, feasibility, agreement, declared, audit,
  transitions, handoff, outcome (with the departure and the base reproduction), and any other field.
Finally record.json is recomputed from the episode files and must match byte for byte. The planning is not rerun;
the byte-identical second run covers it. Exit 0 only with no refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import run as RUN  # noqa: E402
import switch as W  # noqa: E402

C = W.C
STUDY, RUN_DIR = None, None
SEG = lambda *keys: lambda r: [[g.get(k) for k in keys] for g in r["segments"]]  # noqa: E731
EPISODE = (("start", lambda r: [r.get(k) for k in ("kind", "unit", "population", "episode", "t", "radius", "x0", "n")]),
           ("tolerance", lambda r: [r.get("tolerances_sha256")]),
           ("trajectory", lambda r: [r.get(k) for k in ("actions", "modes", "states", "phases", "distances")]),
           ("feasibility", SEG("k", "phase", "tried", "M")),
           ("agreement", SEG("D", "base_actions", "close", "agree")),
           ("declared", SEG("states", "failure", "endpoint_miss", "declared")),
           ("audit", SEG("refined_miss", "audit_checks", "audited")),
           ("transitions", lambda r: [r.get("transitions")]),
           ("handoff", SEG("handoff")),
           ("outcome", lambda r: [r.get(k) for k in ("outcome", "departure", "bc_first_departure", "base_reproduced",
                                                      "upright", "bc_upright")]))


def first_refusal(stored: dict, again: dict):
    for name, view in EPISODE:
        if view(stored) != view(again):
            return name
    return "other" if stored != again else None


def audit_episode(unit):
    rec = json.loads((RUN_DIR / "episodes" / RUN.name(unit)).read_text())
    try:
        again = json.loads(C.dumps(STUDY.run(unit, W.Replay(rec["segments"]))))
    except (ValueError, IndexError, KeyError, TypeError):
        return unit, "plans"
    return unit, first_refusal(rec, again)


def init(smoke, pendulum, policy, stage3, stage2, stage4, run) -> None:
    global STUDY, RUN_DIR
    RUN_DIR = run
    RUN.init(smoke, pendulum, policy, stage3, stage2, stage4, run)
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
    smoke = ident["set"] == "smoke"
    log = C.Log(run / "audit.log")
    refusals = []
    inputs = (smoke, args.pendulum, args.policy, args.stage3, args.stage2, args.stage4, run)
    init(*inputs)  # refuses unless every input is the protocol's
    log("stage calibration recompute")
    if C.dumps(STUDY.calibrate()) != (run / "calibration.json").read_text():
        refusals.append({"unit": None, "reason": "calibration"})
    want = RUN.build_identity(STUDY, args) | {"ccil_commit": ident["identity"]["ccil_commit"]}
    if ident["identity"] != want:
        refusals.append({"unit": None, "reason": "inputs"})
    units = STUDY.units
    out = C.schedule(audit_episode, units, args.workers, log, "audit pendulum switch", init, inputs)
    refusals += [{"unit": list(u), "reason": why} for u, why in (out[u] for u in units) if why]
    recs = [json.loads((run / "episodes" / RUN.name(u)).read_text()) for u in units]
    stored = json.loads((run / "record.json").read_text())
    again = RUN.record(recs, ident["identity"], stored["set"], RUN.load_tolerances(run))
    again["code"] = stored["code"]  # the record's commit is the run's; the recomputation is of its content
    if C.dumps(again) != (run / "record.json").read_text():
        refusals.append({"unit": None, "reason": "record"})
    result = {"set": ident["set"], "episodes": len(units), "refusals": refusals,
              "first_refusal": refusals[0]["reason"] if refusals else None, "ok": not refusals,
              "code": C.code_identity()}
    text = C.dumps(result)
    if args.out:
        C.write_atomic(args.out, text)
    print(text, end="")
    sys.exit(0 if not refusals else 1)


if __name__ == "__main__":
    main()
