"""#73 fresh-process audit of one Stage 3 run directory (recovery-learned.md "Smoke, then the run").

    PY tools/recovery_learned/audit.py --run-dir DIR --workers N --pendulum P --policy POLICY --stage2 STAGE2 \
        [--out AUDIT.json]

It checks the models against identity.json, then for every start recomputes everything but the solve from the
stored actions and reports the first refusal in a fixed order:
  start; per M and restart: bounds, model_replay, exact_replay, refinement, validity, model_error, delta, delta_hat;
  then references, stage2 (and the oracle ceiling), selection, and any other field.
The selection is also recomputed from the stored segments alone. Finally record.json is recomputed from the start
files and must match byte for byte. The LM solve is not rerun; the byte-identical second run covers it. Exit 0 only
with no refusal.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import learned as S  # noqa: E402
import run as R  # noqa: E402

C = S.C
STUDY, RUN_DIR = None, None
SEGMENT = (("bounds", lambda g, sy: [g["model_checks"]["bounds"], g["exact_checks"]["bounds"],
                                     sy.check_bounds(g["actions"])]),
           ("model_replay", lambda g, sy: [g["model_states"], g["model_miss"], g["model_checks"], g["model_valid"]]),
           ("exact_replay", lambda g, sy: [g["exact_states"], g["exact_miss"]]),
           ("refinement", lambda g, sy: [g["refined_miss"], g["exact_checks"]["refinement"]]),
           ("validity", lambda g, sy: [g["exact_checks"], g["valid"]]),
           ("model_error", lambda g, sy: [g["model_error"]]),
           ("delta", lambda g, sy: [g.get(k) for k in ("delta", "hold", "gym_return", "demonstrator_delta")]),
           ("delta_hat", lambda g, sy: [g.get("delta_hat"), g.get("delta_hat_min")]))


def first_refusal(stored: dict, again: dict, sy) -> tuple:
    if [stored.get(k) for k in ("start", "anchor", "anchor_key", "radius", "direction", "x0", "t", "n")] != \
            [again.get(k) for k in ("start", "anchor", "anchor_key", "radius", "direction", "x0", "t", "n")]:
        return "start", None
    for M in sy.ladder:
        a, b = stored["rungs"][str(M)]["segments"], again["rungs"][str(M)]["segments"]
        if len(a) != len(b):
            return "model_replay", M
        for g, h in zip(a, b):
            if not g["model_checks"]["bounds"] == g["exact_checks"]["bounds"] == sy.check_bounds(g["actions"]):
                return "bounds", M
            for name, view in SEGMENT:
                if view(g, sy) != view(h, sy):
                    return name, M
    for name, keys in (("references", ("pi_reference", "zero_reference", "continuation", "demonstrator_continuation")),
                       ("stage2", ("stage2", "ceiling"))):
        if [stored.get(k) for k in keys] != [again.get(k) for k in keys]:
            return name, None
    if stored.get("selection") != again.get("selection") or stored.get("selection") != STUDY.select(stored["rungs"]):
        return "selection", None
    if stored != again:
        return "other", None
    return None, None


def audit_start(unit: tuple):
    rec = json.loads((RUN_DIR / "starts" / f"{unit[0]:03d}.json").read_text())
    actions = {str(M): [g["actions"] for g in rec["rungs"][str(M)]["segments"]] for M in STUDY.sy.ladder}
    try:
        again = json.loads(C.dumps(STUDY.evaluate(unit, actions)))
    except ValueError:
        return unit[0], "start", None
    why, M = first_refusal(rec, again, STUDY.sy)
    return unit[0], why, M


def init(smoke, pendulum, policy, run, stage2) -> None:
    global STUDY, RUN_DIR
    RUN_DIR = run
    STUDY = S.Study(smoke, pendulum, policy, run, stage2)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pendulum", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    run = args.run_dir
    ident = json.loads((run / "identity.json").read_text())
    smoke = ident["set"] == "smoke"
    log = C.Log(run / "audit.log")
    refusals = []
    identity = ident["identity"]
    if identity["policy_sha256"] != S.sha256(args.policy) or identity["pendulum_sha256"] != S.sha256(args.pendulum) \
            or identity["stage2_record_sha256"] != S.sha256(args.stage2 / "record.json"):
        refusals.append({"start": None, "reason": "inputs", "M": None})
    init(smoke, args.pendulum, args.policy, run, args.stage2)  # refuses unless every model matches its hashes
    units = STUDY.units
    out = C.schedule(audit_start, units, args.workers, log, "audit pendulum stage3", init,
                     (smoke, args.pendulum, args.policy, run, args.stage2))
    refusals += [{"start": s, "reason": why, "M": M} for s, why, M in (out[u] for u in units) if why]
    recs = [json.loads((run / "starts" / f"{u[0]:03d}.json").read_text()) for u in units]
    stored = json.loads((run / "record.json").read_text())
    again = R.record(recs, identity, stored["set"], STUDY.sy.ladder)
    again["code"] = stored["code"]  # the record's commit is the run's; the recomputation is of its content
    if C.dumps(again) != (run / "record.json").read_text():
        refusals.append({"start": None, "reason": "record", "M": None})
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
