"""#76: execute both receding-horizon arms from every Pendulum start (recovery-receding.md, receding.json).

    PY tools/recovery_receding/run.py --set smoke|study --run-dir DIR --workers N \
        --pendulum pendulum_cont_corrected.npz --policy s3321/policy.npz --stage2 STAGE2_DIR --stage3 STAGE3_DIR

STAGE2_DIR and STAGE3_DIR are the Stage 2 Pendulum and #73 run directories for the same set: the study's from the
releases evidence-recovery-feasibility-stage-2-v1 and evidence-recovery-learned-stage-3-v1, or the smokes'. Their
records must be the ones receding.json names; model 0 is #73's, checked by its hashes. Each start is written
atomically to DIR/starts/NNN.json (timing in DIR/timing/), so a rerun resumes and two runs compare byte for byte.
The study set also needs --study-go.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import receding as R  # noqa: E402

C = R.C
STUDY = None


def init(smoke, pendulum, policy, stage3, stage2) -> None:
    global STUDY
    STUDY = R.Study(smoke, pendulum, policy, stage3, stage2)


def work(unit: tuple):
    t0 = time.time()
    rec = STUDY.evaluate(unit, STUDY.solve(unit))
    return rec, time.time() - t0


# --- the record ---------------------------------------------------------------------------------------------------

def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    return {str(p): float(np.percentile(x, p)) for p in (0, 10, 50, 90, 100)} if len(x) else None


def count(xs) -> dict:
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items(), key=lambda kv: (len(kv[0]), kv[0])))


def gate(starts_by_radius: dict, hits: list) -> dict:
    """#70's treatment gate: at least min_starts starts, and min_fraction at some radius."""
    by_r = {r: sum(1 for x in hits if str(x["radius"]) == r) for r in starts_by_radius}
    frac = {r: by_r[r] / n for r, n in starts_by_radius.items()}
    return {"starts": len(hits), "by_radius": by_r, "fraction_by_radius": frac,
            "passes": len(hits) >= R.DEC["min_starts"] and max(frac.values()) >= R.DEC["min_fraction_at_some_radius"]}


def summarize(recs: list, ladder: tuple, eps: float) -> dict:
    radii = {str(r): sum(1 for x in recs if x["radius"] == r) for r in sorted({x["radius"] for x in recs})}
    out = {"starts": len(recs), "starts_by_radius": radii}
    for arm in R.ARMS:
        ex = [(x, M, x["arms"][arm]["executions"][str(M)]) for x in recs for M in ladder]
        rec = [x for x in recs if x["arms"][arm]["recovery"]]
        audited = [x for x in rec if x["arms"][arm]["recovery"]["audited"]]
        ceiling = [x for x in recs if x["open_loop"]["ceiling"]]
        out[arm] = {
            "declared_by_M": {str(M): sum(e["declared"] for _, M_, e in ex if M_ == M) for M in ladder},
            "audited_by_M": {str(M): sum(e["declared"] and e["first_audit_failure"] is None for _, M_, e in ex
                                         if M_ == M) for M in ladder},
            "outcome_by_M": {str(M): count("declared" if e["declared"] else e["failure"] or "endpoint_miss"
                                           for _, M_, e in ex if M_ == M) for M in ladder},
            "recoveries": gate(radii, rec),
            "audited_recoveries": gate(radii, audited),
            "recovery_M": count(x["arms"][arm]["recovery"]["M"] for x in rec),
            "first_audit_failure": count(x["arms"][arm]["recovery"]["first_audit_failure"] for x in rec
                                         if not x["arms"][arm]["recovery"]["audited"]),
            "observed_miss_over_eps_by_M": {str(M): q([e["observed_miss"] / eps for _, M_, e in ex
                                                       if M_ == M and e["complete"]]) for M in ladder},
            "predicted_miss_over_eps_by_M": {str(M): q([p / eps for _, M_, e in ex if M_ == M
                                                        for p in e["predicted_miss"]]) for M in ladder},
            "model_one_step_error_over_eps": q([v / eps for *_, e in ex for v in e["model_one_step_error"]]),
            "delta": q([x["arms"][arm]["recovery"]["delta"] for x in audited]),
            "delta_useful": sum(x["arms"][arm]["recovery"]["delta"] >= R.S3.USEFUL for x in audited),
            "hold": sum(x["arms"][arm]["recovery"]["hold"] for x in audited),
            "ceiling_recovered": [sum(1 for x in ceiling if x in audited), len(ceiling)]}
    out["open_loop"] = {cell: gate(radii, [x for x in recs if x["open_loop"][cell]]) for cell in ("exact", "learned")}
    return out


def decide(s: dict) -> dict:
    ex, le = s["exact"]["recoveries"], s["learned"]["recoveries"]
    frac = s["learned"]["audited_recoveries"]["starts"] / le["starts"] if le["starts"] else None
    out = {"exact_gate": ex["passes"], "learned_gate": le["passes"], "learned_audited_fraction": frac}
    out["learned_interpreted"] = ex["passes"]
    out["passes"] = bool(ex["passes"] and le["passes"] and frac is not None
                         and frac >= R.DEC["min_audited_fraction_learned"])
    return out


def record(recs: list, identity: dict, set_name: str, ladder, eps: float) -> dict:
    s = summarize(recs, ladder, eps)
    return {"schema": "ilsb.recovery_receding 1", "set": set_name,
            "note": "smoke: Stage 2's key-99 starts; not a study outcome" if set_name == "smoke" else "study",
            "code": C.code_identity(), "identity": identity, "summary": s, "decision": decide(s),
            "starts_sha256": {f"{r['start']:03d}": hashlib.sha256(C.dumps(r).encode()).hexdigest() for r in recs}}


def build_identity(sy, pendulum: Path, policy: Path, stage2: Path, stage3: Path, set_name: str) -> dict:
    return {"system": "pendulum", "ccil_commit": R.S3.ccil_commit(), "tau": sy.tau, "epsilon": sy.eps,
            "ladder": list(sy.ladder), "anchors": [list(a) for a in sy.anchors], "pendulum_sha256": R.S3.sha256(pendulum),
            "policy_sha256": R.S3.sha256(policy), "stage2_record_sha256": R.SPEC["stage2"]["record_sha256"][set_name],
            "stage3_record_sha256": R.SPEC["stage3"]["record_sha256"][set_name],
            "model0_weights_sha256": R.SPEC["stage3"]["model0_weights_sha256"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("smoke", "study"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pendulum", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--stage3", type=Path, required=True)
    p.add_argument("--study-go", action="store_true")
    p.add_argument("--limit", type=int, help="development only: the first N starts; no record is written")
    args = p.parse_args()
    smoke = args.set == "smoke"
    if not smoke and not args.study_go:
        sys.exit("refused: the study run needs its own go (--study-go)")
    if not smoke and C.code_identity()["dirty"]:
        sys.exit("refused: the study runs from a clean commit")
    if not smoke and R.S3.ccil_commit() != R.S3.SPEC["models"]["ccil_commit"]:
        sys.exit(f"refused: the study needs CCIL {R.S3.SPEC['models']['ccil_commit']}, not {R.S3.ccil_commit()}")
    run = args.run_dir
    if (run / "EXIT").exists():
        sys.exit(f"refused: {run} is a finished run; never overwrite one")
    (run / "starts").mkdir(parents=True, exist_ok=True)
    (run / "timing").mkdir(exist_ok=True)
    log = C.Log(run / "run.log")
    C.header(log, {"set": args.set, "workers": args.workers, "pendulum": str(args.pendulum),
                   "policy": str(args.policy), "stage2": str(args.stage2), "stage3": str(args.stage3)})
    inputs = (smoke, args.pendulum, args.policy, args.stage3, args.stage2)
    init(*inputs)  # refuses unless every input is the protocol's
    sy = STUDY.sy
    ident_path = run / "identity.json"
    identity = build_identity(sy, args.pendulum, args.policy, args.stage2, args.stage3, args.set)
    if ident_path.exists():
        if json.loads(ident_path.read_text())["identity"] != identity:
            sys.exit(f"refused: {run} was started with other inputs")
    else:
        C.write_atomic(ident_path, C.dumps({"set": args.set, "code": C.code_identity(), "identity": identity}))
    units = STUDY.units[:args.limit]
    todo = [u for u in units if not (run / "starts" / f"{u[0]:03d}.json").exists()]
    log(f"stage execute: {len(units)} starts x 2 arms, {len(units) - len(todo)} already written, "
        f"epsilon {sy.eps!r}")

    def done(u, res):
        rec, secs = res
        C.write_atomic(run / "starts" / f"{u[0]:03d}.json", C.dumps(rec))
        C.write_atomic(run / "timing" / f"{u[0]:03d}.json", C.dumps({"start": u[0], "seconds": secs}))

    t0 = time.time()
    try:
        C.schedule(work, todo, args.workers, log, f"pendulum receding {args.set}", init, inputs, on_done=done)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        (run / "EXIT").write_text("EXIT=1\n")
        raise
    recs = [json.loads((run / "starts" / f"{u[0]:03d}.json").read_text()) for u in units]
    if args.limit is None:
        rec = record(recs, identity, args.set, sy.ladder, sy.eps)
        C.write_atomic(run / "record.json", C.dumps(rec))
        log(f"record {R.S3.sha256(run / 'record.json')} decision " + json.dumps(rec["decision"], sort_keys=True))
        (run / "EXIT").write_text("EXIT=0\n")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
