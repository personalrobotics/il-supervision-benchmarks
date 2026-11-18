"""#70 Stage 2: solve every start of one system and write its record (recovery-feasibility.md).

    PY tools/recovery_feasibility/run.py --system SYSTEM --set smoke|study --run-dir DIR --workers N INPUT

INPUT is the system's data: --quickstart CCIL/docs/quickstart.md (double_integrator, at the contract commit),
--pendulum pendulum_cont_corrected.npz (pendulum), or --source pusht-source.npz (pusht, under LEROBOT_PYTHON).

Each start's result is written atomically to DIR/starts/NNN.json, with its wall time in DIR/timing/NNN.json, so a
rerun resumes and two runs compare byte for byte on starts/ and record.json. DIR/run.log carries the header and
progress; DIR/EXIT is written last. The study set also needs --study-go: the run has its own authorization.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

SYSTEM = None  # the worker's system, built once per process


def build(name: str, smoke: bool, data: Path):
    if name == "double_integrator":
        import di
        return di.System(smoke, data)
    if name == "pendulum":
        import pendulum
        return pendulum.System(smoke, data)
    import pusht
    return pusht.System(smoke, data)


def init(name: str, smoke: bool, data: Path) -> None:
    global SYSTEM
    SYSTEM = build(name, smoke, data)
    SYSTEM.smoke = smoke


def solve_start(unit: tuple, system=None, smoke: bool | None = None) -> dict:
    """Every step of one start: availability, eligibility, the M = 1 certificate, every M and solver and restart,
    and each returned segment's checks."""
    sy = system or SYSTEM
    smoke = sy.smoke if smoke is None else smoke
    s, a, r, d = unit
    sid = C.SYSTEMS[sy.name]
    x0, redraws = C.available(lambda g: sy.propose(a, r, g), C.rng(*C.key(sid, smoke, s)))
    rec = {"start": s, "anchor": a, "anchor_key": list(sy.anchors[a]), "radius": r, "direction": d,
           "redraws": redraws, "available": x0 is not None}
    if x0 is None:
        return rec
    rec.update(sy.start_record(x0))
    elig = sy.eligibility(a)
    rec["eligibility"] = elig
    ok = (lambda M: True) if elig is None else (lambda M: elig["eligible"][str(M)])
    rec["m1"] = sy.certificate(a, x0) if ok(1) else None
    rungs = {}
    for M in sy.ladder:
        if not ok(M):
            rungs[str(M)] = {"eligible": False}
            continue
        rung = {"eligible": True, "oracle": sy.oracle(a, x0, M), "demo_replay_miss": sy.demo_replay_miss(a, M)}
        seed = lambda k, M=M: C.rng(*C.key(sid, smoke, s, M, k))  # noqa: E731
        for solver in sy.solvers:
            segs = [sy.segment(a, x0, M, U) for U in sy.solve(solver, a, x0, M, seed)]
            rung[solver] = {"segments": segs, "first_valid": C.first_valid(segs),
                            "ambiguity": C.ambiguity(segs, sy.ubar)}
        rungs[str(M)] = rung
    rec["rungs"] = rungs
    return rec


def work(unit: tuple):
    t0 = time.time()
    rec = solve_start(unit)
    return rec, time.time() - t0


# --- the record ---------------------------------------------------------------------------------------------------

def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    return {str(p): float(np.percentile(x, p)) for p in (0, 10, 50, 90, 100)} if len(x) else None


def summarize(name: str, recs: list, eps: float, ladder: tuple, solvers: tuple) -> dict:
    """The measures and the decision inputs (recovery-feasibility.md "Measures", "Decision")."""
    avail = [r for r in recs if r["available"]]
    radii = sorted({r["radius"] for r in recs})
    by_r = lambda rs, rad: [x for x in rs if x["radius"] == rad]  # noqa: E731
    out = {"starts": len(recs), "available": len(avail),
           "unavailable_by_radius": {str(rad): sum(not x["available"] for x in by_r(recs, rad)) for rad in radii},
           "redraws_by_radius": {str(rad): sum(len(x["redraws"]) for x in by_r(recs, rad)) for rad in radii},
           "redraw_reasons": dict(sorted(Counter(w for x in recs for w in x["redraws"]).items()))}
    out["ineligible_pairs_by_M"] = {str(M): sum(not x["rungs"][str(M)]["eligible"] for x in avail) for M in ladder}

    def valid_at(x, M, solver=None):
        rung = x["rungs"][str(M)]
        return rung["eligible"] and any(rung[s]["first_valid"] is not None for s in ((solver,) if solver else solvers))

    I = [x for x in avail if x["m1"] is not None and x["m1"]["infeasible"]]
    V = [x for x in I if any(valid_at(x, M) for M in ladder if M > 1)]
    out["certified_m1_infeasible"] = {"total": len(I), "by_radius": {str(rad): len(by_r(I, rad)) for rad in radii}}
    out["valid_m_gt_1_among_infeasible"] = {"total": len(V), "by_radius": {str(rad): len(by_r(V, rad)) for rad in radii}}
    out["yield"] = {s: {str(rad): {str(M): [sum(valid_at(x, M, s) for x in by_r(avail, rad)),
                                            sum(x["rungs"][str(M)]["eligible"] for x in by_r(avail, rad))]
                                   for M in ladder} for rad in radii} for s in solvers}
    shortest = [min([M for M in ladder if valid_at(x, M)], default=None) for x in avail]
    out["shortest_valid_M"] = dict(sorted(Counter(str(m) for m in shortest).items()))
    segs = [(x["start"], M, s, k, g) for x in avail for M in ladder if x["rungs"][str(M)]["eligible"]
            for s in solvers for k, g in enumerate(x["rungs"][str(M)][s]["segments"])]
    out["terminal_miss_over_eps_by_M"] = {str(M): q([min(g["miss"] for g in x["rungs"][str(M)][s]["segments"]) / eps
                                                     for x in avail if x["rungs"][str(M)]["eligible"] for s in solvers])
                                          for M in ladder}
    # single shooting: every stored state is the reference map's image of its predecessor; the audit's bitwise
    # replay is the check
    out["per_step_defect"] = 0.0
    out["violations"] = dict(sorted(Counter(c for *_, g in segs for c, ok in g["checks"].items() if not ok).items()))
    out["ambiguity"] = {s: {str(M): {"distinct_valid": q([x["rungs"][str(M)][s]["ambiguity"]["distinct_valid"]
                                                         for x in avail if x["rungs"][str(M)]["eligible"]]),
                                     "ties_with_best": q([x["rungs"][str(M)][s]["ambiguity"]["ties_with_best"]
                                                         for x in avail if x["rungs"][str(M)]["eligible"]])}
                            for M in ladder} for s in solvers}
    refined = [abs(g["refined_miss"] - g["miss"]) for *_, g in segs if g["refined_miss"] is not None]
    out["refinement_delta"] = q(refined)
    # "returned" segments: those a solver offers as a solution, i.e. its terminal miss is within epsilon
    returned = [(st, M, s, k, g) for st, M, s, k, g in segs if g["checks"]["terminal"]]
    out["returned_segments"] = len(returned)
    out["returned_rejected"] = [{"start": st, "M": M, "solver": s, "restart": k,
                                 "failed": sorted(c for c, ok in g["checks"].items() if not ok)}
                                for st, M, s, k, g in returned if not g["valid"]]
    if name == "double_integrator":
        recall, contradictions = {}, []
        for s in solvers:
            recall[s] = {}
            for M in ladder:
                if M == 1:
                    continue
                reach = [x for x in avail if x["rungs"][str(M)]["oracle"]["reachable"]]
                recall[s][str(M)] = [sum(valid_at(x, M, s) for x in reach), len(reach)]
        for st, M, s, k, g in segs:
            orc = next(x for x in avail if x["start"] == st)["rungs"][str(M)]["oracle"]
            if g["valid"] and orc["optimum"] > eps:
                contradictions.append({"start": st, "M": M, "solver": s, "restart": k})
        out["recall"] = recall
        out["oracle_contradictions"] = contradictions
        out["m1_exact_infeasible"] = len(I)
    return out


def decide(name: str, s: dict) -> dict:
    """The protocol's per-system verdict (recovery-feasibility.md "Decision")."""
    if name == "double_integrator":
        recall = {k: [sum(v[0] for v in r.values()), sum(v[1] for v in r.values())] for k, r in s["recall"].items()}
        ok = {k: (n == 0 or hit / n >= C.SPEC["decision"]["double_integrator"]["recall_min"]) for k, (hit, n) in
              recall.items()}
        clean = not s["returned_rejected"] and not s["oracle_contradictions"]
        return {"recall": recall, "recall_ok": ok, "every_returned_segment_valid": clean,
                "lm_passes": clean and ok["lm_shooting"], "cem_passes": clean and ok["cem"]}
    t = C.SPEC["decision"]["treatment"]
    I, V = s["certified_m1_infeasible"], s["valid_m_gt_1_among_infeasible"]
    frac = {r: (V["by_radius"][r] / n if n else None) for r, n in I["by_radius"].items()}
    best = max((f for f in frac.values() if f is not None), default=0.0)
    return {"valid_starts": V["total"], "fraction_by_radius": frac,
            "passes": V["total"] >= t["min_valid_starts"] and best >= t["min_fraction_at_some_radius"]}


def record(sy, recs: list, set_name: str, data_sha: str) -> dict:
    s = summarize(sy.name, recs, sy.eps, sy.ladder, sy.solvers)
    starts = {f"{r['start']:03d}": hashlib.sha256(C.dumps(r).encode()).hexdigest() for r in recs}
    return {"schema": "ilsb.recovery_feasibility.stage2 1", "system": sy.name, "set": set_name,
            "note": "smoke: the key-99 anchors, disjoint from the study's; not a study outcome" if set_name == "smoke"
            else "study", "code": C.code_identity(), "input_sha256": data_sha, "identity": sy.identity,
            "summary": s, "decision": decide(sy.name, s), "starts_sha256": starts}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=list(C.SYSTEMS), required=True)
    p.add_argument("--set", choices=("smoke", "study"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    p.add_argument("--quickstart", type=Path)
    p.add_argument("--pendulum", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--study-go", action="store_true")
    p.add_argument("--limit", type=int, help="development only: solve the first N starts; no record is written")
    args = p.parse_args()
    data = {"double_integrator": args.quickstart, "pendulum": args.pendulum, "pusht": args.source}[args.system]
    if data is None:
        sys.exit(f"refused: {args.system} needs its input")
    smoke = args.set == "smoke"
    if not smoke and not args.study_go:
        sys.exit("refused: the study run needs its own go (--study-go)")
    if not smoke and C.code_identity()["dirty"]:
        sys.exit("refused: the study runs from a clean commit")
    run = args.run_dir
    (run / "starts").mkdir(parents=True, exist_ok=True)
    (run / "timing").mkdir(exist_ok=True)
    if (run / "EXIT").exists():
        sys.exit(f"refused: {run} is a finished run; never overwrite one")
    log = C.Log(run / "run.log")
    data_sha = hashlib.sha256(data.read_bytes()).hexdigest()
    C.header(log, {"system": args.system, "set": args.set, "workers": args.workers, "input": str(data),
                   "input_sha256": data_sha})
    sy = build(args.system, smoke, data)
    sy.smoke = smoke
    ident = run / "identity.json"
    text = C.dumps({"system": sy.name, "set": args.set, "code": C.code_identity(), "identity": sy.identity})
    if ident.exists() and ident.read_text() != text:
        sys.exit("refused: the run directory belongs to another code or input identity")
    C.write_atomic(ident, text)
    units = C.start_ids(len(sy.anchors))[:args.limit]
    todo = [u for u in units if not (run / "starts" / f"{u[0]:03d}.json").exists()]
    log(f"stage solve: {len(units)} starts, {len(units) - len(todo)} already written, tau {sy.tau!r}, "
        f"epsilon {sy.eps!r}")

    def done(u, res):
        rec, secs = res
        C.write_atomic(run / "starts" / f"{u[0]:03d}.json", C.dumps(rec))
        C.write_atomic(run / "timing" / f"{u[0]:03d}.json", C.dumps({"start": u[0], "seconds": secs}))

    t0 = time.time()
    try:
        C.schedule(work, todo, args.workers, log, f"{sy.name} {args.set}", init, (args.system, smoke, data),
                   on_done=done)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        (run / "EXIT").write_text("EXIT=1\n")
        raise
    recs = [json.loads((run / "starts" / f"{u[0]:03d}.json").read_text()) for u in units]
    if args.limit is None:
        rec = record(sy, recs, args.set, data_sha)
        C.write_atomic(run / "record.json", C.dumps(rec))
        log(f"record {hashlib.sha256((run / 'record.json').read_bytes()).hexdigest()} decision "
            + json.dumps(rec["decision"], sort_keys=True))
    log(f"done in {time.time() - t0:.0f} s")
    (run / "EXIT").write_text("EXIT=0\n")


if __name__ == "__main__":
    main()
