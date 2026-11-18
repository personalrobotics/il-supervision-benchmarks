"""#73 Stage 3: fit the models, then solve and evaluate every Pendulum start (recovery-learned.md, stage-3.json).

    PY tools/recovery_learned/run.py --set smoke|study --run-dir DIR --workers N \
        --pendulum pendulum_cont_corrected.npz --policy s3321/policy.npz --stage2 STAGE2_PENDULUM_RUN_DIR

STAGE2_PENDULUM_RUN_DIR is Stage 2's Pendulum run directory for the same set: the study's from the release
evidence-recovery-feasibility-stage-2-v1, or the Stage 2 smoke's. Its record.json must be the one stage-3.json
names. The five models are fitted once into DIR/models and identified in DIR/identity.json; each start is written
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
import learned as S  # noqa: E402

C = S.C
STUDY = None


def init(smoke, pendulum, policy, run, stage2) -> None:
    global STUDY
    STUDY = S.Study(smoke, pendulum, policy, run, stage2)


def work(unit: tuple):
    t0 = time.time()
    rec = STUDY.evaluate(unit, STUDY.solve(unit))
    return rec, time.time() - t0


# --- the record ---------------------------------------------------------------------------------------------------

def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    return {str(p): float(np.percentile(x, p)) for p in (0, 10, 50, 90, 100)} if len(x) else None


def summarize(recs: list, ladder: tuple) -> dict:
    radii = sorted({r["radius"] for r in recs})
    by_r = lambda rs, rad: [x for x in rs if x["radius"] == rad]  # noqa: E731
    segs = [(x, M, g) for x in recs for M in ladder for g in x["rungs"][str(M)]["segments"]]
    mv = [(x, M, g) for x, M, g in segs if g["model_valid"]]
    C_ = [x for x in recs if x["ceiling"]]
    S_ = [x for x in recs if x["selection"]]

    def selected(x):
        return x["rungs"][str(x["selection"]["M"])]["segments"][x["selection"]["restart"]]

    correct = [x for x in S_ if selected(x)["valid"] and selected(x)["delta"] >= S.USEFUL]
    out = {"starts": len(recs),
           "construction": {str(M): {"model_valid_restarts": sum(g["model_valid"] for x, M_, g in segs if M_ == M),
                                     "starts_with_model_valid": sum(any(g["model_valid"] for g in x["rungs"][str(M)]
                                                                        ["segments"]) for x in recs)}
                            for M in ladder},
           "validity": {str(M): [sum(g["valid"] for x, M_, g in mv if M_ == M), sum(1 for x, M_, g in mv if M_ == M)]
                        for M in ladder},
           "exact_failures_among_model_valid": {
               c: sum(not g["exact_checks"][c] for *_, g in mv) for c in ("bounds", "envelope", "endpoint", "refinement")},
           "model_error_by_M": {str(M): {str(k): q([g["model_error"][k] for x, M_, g in mv if M_ == M])
                                         for k in range(S.SPEC["models"]["members"])} for M in ladder},
           "delta_model_valid": q([g["delta"] for *_, g in mv]),
           "delta_hat_min_model_valid": q([g["delta_hat_min"] for *_, g in mv]),
           "sign_agreement": [sum((g["delta"] >= S.USEFUL) == (g["delta_hat_min"] >= S.USEFUL) for *_, g in mv),
                              len(mv)],
           "ceiling": {"total": len(C_), "by_radius": {str(r): len(by_r(C_, r)) for r in radii},
                       "M": dict(sorted((str(k), v) for k, v in _count(x["ceiling"]["M"] for x in C_).items()))},
           "selection": {"total": len(S_), "by_radius": {str(r): len(by_r(S_, r)) for r in radii},
                         "M": dict(sorted((str(k), v) for k, v in _count(x["selection"]["M"] for x in S_).items())),
                         "correct": len(correct), "correct_by_radius": {str(r): len(by_r(correct, r)) for r in radii},
                         "valid": sum(selected(x)["valid"] for x in S_),
                         "delta": q([selected(x)["delta"] for x in S_]),
                         "hold": sum(selected(x)["hold"] for x in S_)},
           "recall": [sum(1 for x in C_ if x in correct), len(C_)],
           "continuation": {"upright": q([x["continuation"]["exact"]["upright"] / x["n"] for x in recs]),
                            "hold": sum(x["continuation"]["exact"]["hold"] for x in recs)},
           "pi_reference_nonzero": sum(v["exact"] != 0 or any(m != 0 for m in v["models"])
                                       for x in recs for v in x["pi_reference"].values()),
           "zero_reference_delta": q([v for x in recs for v in x["zero_reference"].values()]),
           "stage2_delta": {str(M): q([x["stage2"][str(M)]["delta"] for x in recs if x["stage2"][str(M)]])
                            for M in ladder}}
    return out


def _count(xs) -> dict:
    out = {}
    for x in xs:
        out[x] = out.get(x, 0) + 1
    return out


def decide(s: dict) -> dict:
    d = S.SPEC["decision"]
    n_sel = s["selection"]["total"]
    precision = s["selection"]["correct"] / n_sel if n_sel else None
    out = {"ceiling_starts": s["ceiling"]["total"], "selected_starts": n_sel, "precision": precision,
           "recall": s["recall"], "references_ok": s["pi_reference_nonzero"] == 0}
    out["passes"] = bool(out["references_ok"] and s["ceiling"]["total"] >= d["min_ceiling_starts"]
                         and n_sel >= d["min_selected_starts"] and precision is not None
                         and precision >= d["min_precision"])
    return out


def record(recs: list, identity: dict, set_name: str, ladder) -> dict:
    s = summarize(recs, ladder)
    return {"schema": "ilsb.recovery_learned.stage3 1", "set": set_name,
            "note": "smoke: Stage 2's key-99 starts; not a study outcome" if set_name == "smoke" else "study",
            "code": C.code_identity(), "identity": identity, "summary": s, "decision": decide(s),
            "starts_sha256": {f"{r['start']:03d}": hashlib.sha256(C.dumps(r).encode()).hexdigest() for r in recs}}


def build_identity(sy, run: Path, pendulum: Path, policy: Path, stage2: Path, set_name: str) -> dict:
    want = S.SPEC["stage2"]["record_sha256"][set_name]
    if S.sha256(stage2 / "record.json") != want:
        sys.exit(f"refused: {stage2}/record.json is not Stage 2's {set_name} record")
    S.load_policy(policy)
    return {"system": "pendulum", "ccil_commit": S.ccil_commit(), "tau": sy.tau, "epsilon": sy.eps, "ladder": list(sy.ladder),
            "anchors": [list(a) for a in sy.anchors], "split": sy.identity["split"],
            "pendulum_sha256": S.sha256(pendulum), "policy_sha256": S.sha256(policy), "stage2_record_sha256": want,
            "models": S.fit_models(sy, run)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("smoke", "study"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--pendulum", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--study-go", action="store_true")
    p.add_argument("--limit", type=int, help="development only: the first N starts; no record is written")
    args = p.parse_args()
    smoke = args.set == "smoke"
    if not smoke and not args.study_go:
        sys.exit("refused: the study run needs its own go (--study-go)")
    if not smoke and C.code_identity()["dirty"]:
        sys.exit("refused: the study runs from a clean commit")
    if not smoke and S.ccil_commit() != S.SPEC["models"]["ccil_commit"]:
        sys.exit(f"refused: the study needs CCIL {S.SPEC['models']['ccil_commit']}, not {S.ccil_commit()}")
    run = args.run_dir
    if (run / "EXIT").exists():
        sys.exit(f"refused: {run} is a finished run; never overwrite one")
    (run / "starts").mkdir(parents=True, exist_ok=True)
    (run / "timing").mkdir(exist_ok=True)
    log = C.Log(run / "run.log")
    C.header(log, {"set": args.set, "workers": args.workers, "pendulum": str(args.pendulum),
                   "policy": str(args.policy), "stage2": str(args.stage2)})
    sy = S.P2.System(smoke, args.pendulum)
    ident_path = run / "identity.json"
    if ident_path.exists():
        identity = json.loads(ident_path.read_text())["identity"]
        S.load_models(run, identity["models"])
    else:
        log("stage fit: 5 transition models, one deterministic CPU group")
        t_fit = time.time()
        identity = build_identity(sy, run, args.pendulum, args.policy, args.stage2, args.set)
        C.write_atomic(run / "timing" / "fit.json", C.dumps({"seconds": time.time() - t_fit}))
        models = S.load_models(run, identity["models"])
        identity["validation_error"] = S.validation_error(sy, models)
        C.write_atomic(ident_path, C.dumps({"set": args.set, "code": C.code_identity(), "identity": identity}))
    log("models " + json.dumps({k: v["weights_sha256"][:12] for k, v in identity["models"].items()}))
    units = C.start_ids(len(sy.anchors))[:args.limit]
    todo = [u for u in units if not (run / "starts" / f"{u[0]:03d}.json").exists()]
    log(f"stage solve: {len(units)} starts, {len(units) - len(todo)} already written, epsilon {sy.eps!r}")

    def done(u, res):
        rec, secs = res
        C.write_atomic(run / "starts" / f"{u[0]:03d}.json", C.dumps(rec))
        C.write_atomic(run / "timing" / f"{u[0]:03d}.json", C.dumps({"start": u[0], "seconds": secs}))

    t0 = time.time()
    try:
        C.schedule(work, todo, args.workers, log, f"pendulum stage3 {args.set}", init,
                   (smoke, args.pendulum, args.policy, run, args.stage2), on_done=done)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        (run / "EXIT").write_text("EXIT=1\n")
        raise
    recs = [json.loads((run / "starts" / f"{u[0]:03d}.json").read_text()) for u in units]
    if args.limit is None:
        rec = record(recs, identity, args.set, sy.ladder)
        C.write_atomic(run / "record.json", C.dumps(rec))
        log(f"record {S.sha256(run / 'record.json')} decision " + json.dumps(rec["decision"], sort_keys=True))
        (run / "EXIT").write_text("EXIT=0\n")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
