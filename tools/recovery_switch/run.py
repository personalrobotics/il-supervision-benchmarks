"""#79: calibrate the switch, then run every switched episode (recovery-switch.md, switch.json, split.json).

    PY tools/recovery_switch/run.py --set smoke|study --run-dir DIR --workers N --pendulum pendulum_cont_corrected.npz \
        --policy s3321/policy.npz --stage2 STAGE2_DIR --stage3 STAGE3_DIR --stage4 STAGE4_DIR

STAGE2_DIR, STAGE3_DIR and STAGE4_DIR are the Stage 2 Pendulum, #73 and #76 run directories for the same set: the
study's from their releases, or the smokes'. Calibration runs first and is written to DIR/calibration.json, whose
hash every episode records. Each episode is written atomically
to DIR/episodes/, so a rerun resumes and two runs compare byte for byte. The study set also needs --study-go.
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
import switch as W  # noqa: E402

C = W.C
STUDY = None


def name(unit) -> str:
    return f"{unit[0]}-{unit[1]:04d}.json"


def load_tolerances(run: Path) -> dict:
    text = (run / "calibration.json").read_text()
    tol = json.loads(text)
    tol["sha256"] = hashlib.sha256(text.encode()).hexdigest()
    return tol


def init(smoke, pendulum, policy, stage3, stage2, stage4, run) -> None:
    global STUDY
    STUDY = W.Study(smoke, pendulum, policy, stage3, stage2, stage4)
    STUDY.tol = load_tolerances(run)


def work(unit):
    t0 = time.time()
    rec = STUDY.run(unit, W.Live(STUDY, STUDY.smoke))
    return rec, time.time() - t0


# --- the record ---------------------------------------------------------------------------------------------------

def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    return {str(p): float(np.percentile(x, p)) for p in (0, 10, 50, 90, 100)} if len(x) else None


def count(xs) -> dict:
    out = {}
    for x in xs:
        out[str(x)] = out.get(str(x), 0) + 1
    return dict(sorted(out.items()))


def recovery_episodes(rec: dict) -> list:
    """[(entry k, handoff k or None, segments)] in order."""
    out, segs = [], iter(rec["segments"])
    tr = rec["transitions"]
    for i, t in enumerate(tr):
        if t["to"] != "recover":
            continue
        end = tr[i + 1]["k"] if i + 1 < len(tr) else None
        mine = []
        for g in segs:
            mine.append(g)
            if g["M"] is None or g["handoff"] or g["failure"] or not g["declared"]:
                break
        out.append((t["k"], end, mine))
    return out


def handoff_valid(rec: dict, tol: dict) -> list:
    """Recompute every handoff from the per-step record: the last H steps before it are recovery steps of the same
    recovery episode, each close (D <= delta_in) and agreeing (|base - u| / sigma_u <= alpha), and the handoff state
    is within delta_in of its phase."""
    out = []
    for entry, end, segs in recovery_episodes(rec):
        if end is None:
            continue
        flags = [(d <= tol["delta_in"] and abs(b - u) / tol["sigma_u"] <= tol["alpha"])
                 for g in segs for d, b, u in zip(g["D"], g["base_actions"], g["actions"])]
        modes_ok = all(m == "recover" for m in rec["modes"][end - W.H:end])
        out.append(bool(end - entry >= W.H and modes_ok and len(flags) >= W.H and all(flags[-W.H:])
                        and segs[-1]["endpoint_miss"] <= tol["delta_in"]))
    return out


def summarize(recs: list, tol: dict) -> dict:
    pops = {}
    for pop in ("control", "recover", "unknown"):
        rs = [r for r in recs if r["population"] == pop]
        reps = [(r, e) for r in rs for e in recovery_episodes(r)]
        segs = [g for r in rs for g in r["segments"] if g["M"] is not None]
        retained = [r for r in rs if retained_(r)]
        radii = sorted({r["radius"] for r in rs if r["radius"] is not None})
        pops[pop] = {
            "episodes": len(rs),
            "outcome": count(r["outcome"] for r in rs),
            "in_base_throughout": sum(not r["transitions"] and r["outcome"] == "complete" for r in rs),
            "bc_departed": sum(r["bc_first_departure"] is not None for r in rs),
            "bc_first_departure": q([r["bc_first_departure"] for r in rs if r["bc_first_departure"] is not None]),
            "departed_then_entered": sum(r["bc_first_departure"] is not None and bool(r["transitions"]) for r in rs),
            "departed_then_handed_back": sum(any(t["to"] == "base" for t in r["transitions"]) for r in rs),
            "entered": sum(bool(r["transitions"]) for r in rs),
            "entered_at_step_0": sum(bool(r["transitions"]) and r["transitions"][0]["k"] == 0 for r in rs),
            "retained": len(retained),
            "retained_by_radius": {str(x): sum(1 for r in retained if r["radius"] == x) for x in radii},
            "episodes_by_radius": {str(x): sum(1 for r in rs if r["radius"] == x) for x in radii},
            "recovery_episodes": len(reps),
            "handoffs": sum(e[1] is not None for _, e in reps),
            "handoff_steps": q([e[1] - e[0] for _, e in reps if e[1] is not None]),
            "reentries_after_handoff": sum(max(0, sum(t["to"] == "recover" for t in r["transitions"]) - 1) for r in rs),
            "switches": q([len(r["transitions"]) for r in rs]),
            "segments": len(segs), "declared_segments": sum(g["declared"] for g in segs),
            "declared_not_audited": sum(g["declared"] and not g["audited"] for g in segs),
            "segment_M": count(g["M"] for g in segs),
            "recovery_steps": q([sum(len(g["actions"]) for g in r["segments"]) for r in rs]),
            "upright_complete": q([r["upright"] for r in rs if r["upright"] is not None]),
            "bc_upright": q([r["bc_upright"] for r in rs]),
            "upright_minus_bc_complete": q([r["upright"] - r["bc_upright"] for r in rs if r["upright"] is not None])}
    return pops


def retained_(r: dict) -> bool:
    eps = recovery_episodes(r)
    return bool(eps and eps[0][0] == 0 and eps[0][1] is not None and all(g["declared"] for g in eps[0][2]))


def entry_valid(rec: dict, tol: dict) -> list:
    """Every entry: D > delta_out at its step, and the segment it starts executes a feasible plan."""
    out = []
    for t in rec["transitions"]:
        if t["to"] == "recover":
            g = next((g for g in rec["segments"] if g["k"] == t["k"]), None)
            out.append(bool(g is not None and t["D"] > tol["delta_out"] and g["M"] is not None
                            and g["tried"][-1]["feasible"] and g["tried"][-1]["M"] == g["M"]))
    return out


def decide(s: dict, recs: list, tol: dict) -> dict:
    d = W.SPEC["decision"]
    valid = [v for r in recs for v in handoff_valid(r, tol)]
    entries = [v for r in recs for v in entry_valid(r, tol)]
    abst = set(W.SPEC["abstention"])
    accounted = all((r["outcome"] == "complete" and len(r["actions"]) == r["n"])
                    or (r["outcome"] in abst and len(r["actions"]) < r["n"]) for r in recs) and \
        all(len(r["modes"]) == len(r["actions"]) == len(r["states"]) - 1 for r in recs)
    controls = [r for r in recs if r["population"] == "control"]
    begin = [r for r in controls if r["modes"][:1] == ["base"]]
    out = {"base_first": [len(begin), len(controls)],
           "base_reproduced": [sum(r["base_reproduced"] for r in recs), len(recs)],
           "entries_valid": [sum(entries), len(entries)], "handoffs_valid": [sum(valid), len(valid)],
           "retained": s["recover"]["retained"], "recover_population": s["recover"]["episodes"],
           "declared_not_audited": sum(p["declared_not_audited"] for p in s.values()), "accounted": accounted}
    out["passes"] = bool(controls and len(begin) == len(controls) and out["base_reproduced"][0] == len(recs)
                         and all(entries) and all(valid) and out["retained"] >= d["min_retained"]
                         and out["declared_not_audited"] == 0 and accounted)
    return out


def record(recs: list, identity: dict, set_name: str, tol: dict) -> dict:
    s = summarize(recs, tol)
    return {"schema": "ilsb.recovery_switch 1", "set": set_name,
            "note": "smoke: Stage 2's key-99 starts and calibration-episode controls; not a study outcome"
            if set_name == "smoke" else "study",
            "code": C.code_identity(), "identity": identity, "tolerances": {k: tol[k] for k in (
                "delta_out", "delta_in", "sigma_u", "alpha", "sha256")},
            "summary": s, "decision": decide(s, recs, tol),
            "episodes_sha256": {name((r["kind"], r["unit"]))[:-5]: hashlib.sha256(C.dumps(r).encode()).hexdigest()
                                for r in recs}}


def build_identity(st, args) -> dict:
    s = "smoke" if st.smoke else "study"
    return {"system": "pendulum", "ccil_commit": W.S3.ccil_commit(), "epsilon": st.sy.eps,
            "pendulum_sha256": W.S3.sha256(args.pendulum), "policy_sha256": W.S3.sha256(args.policy),
            "split_sha256": W.S3.sha256(W.HERE / "split.json"),
            "stage2_record_sha256": W.R.SPEC["stage2"]["record_sha256"][s],
            "stage3_record_sha256": W.R.SPEC["stage3"]["record_sha256"][s],
            "stage4_record_sha256": W.SPEC["stage4"]["record_sha256"][s],
            "model0_weights_sha256": W.R.SPEC["stage3"]["model0_weights_sha256"],
            "calibration_episodes": sorted(st.calibration, key=int), "control_episodes": sorted(st.controls, key=int)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--set", choices=("smoke", "study"), required=True)
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=1)
    for a in ("pendulum", "policy", "stage2", "stage3", "stage4"):
        p.add_argument(f"--{a}", type=Path, required=True)
    p.add_argument("--study-go", action="store_true")
    p.add_argument("--limit", type=int, help="development only: the first N episodes; no record is written")
    args = p.parse_args()
    smoke = args.set == "smoke"
    if not smoke and not args.study_go:
        sys.exit("refused: the study run needs its own go (--study-go)")
    if not smoke and C.code_identity()["dirty"]:
        sys.exit("refused: the study runs from a clean commit")
    if not smoke and W.S3.ccil_commit() != W.S3.SPEC["models"]["ccil_commit"]:
        sys.exit(f"refused: the study needs CCIL {W.S3.SPEC['models']['ccil_commit']}, not {W.S3.ccil_commit()}")
    run = args.run_dir
    if (run / "EXIT").exists():
        sys.exit(f"refused: {run} is a finished run; never overwrite one")
    (run / "episodes").mkdir(parents=True, exist_ok=True)
    (run / "timing").mkdir(exist_ok=True)
    log = C.Log(run / "run.log")
    C.header(log, {"set": args.set, "workers": args.workers, **{a: str(getattr(args, a)) for a in (
        "pendulum", "policy", "stage2", "stage3", "stage4")}})
    st = W.Study(smoke, args.pendulum, args.policy, args.stage3, args.stage2, args.stage4)
    identity = build_identity(st, args)
    ident_path = run / "identity.json"
    if ident_path.exists():
        if json.loads(ident_path.read_text())["identity"] != identity:
            sys.exit(f"refused: {run} was started with other inputs")
    else:
        C.write_atomic(ident_path, C.dumps({"set": args.set, "code": C.code_identity(), "identity": identity}))
    if not (run / "calibration.json").exists():
        log(f"stage calibrate: {len(st.calibration)} episodes, {sum(map(len, st.calibration.values()))} starts")
        t0 = time.time()
        C.write_atomic(run / "calibration.json", C.dumps(st.calibrate()))
        C.write_atomic(run / "timing" / "calibration.json", C.dumps({"seconds": time.time() - t0}))
    tol = load_tolerances(run)
    log(f"calibration {tol['sha256']}: " + json.dumps({k: tol[k] for k in ("delta_out", "delta_in", "sigma_u",
                                                                          "alpha")}))
    inputs = (smoke, args.pendulum, args.policy, args.stage3, args.stage2, args.stage4, run)
    units = st.units[:args.limit]
    todo = [u for u in units if not (run / "episodes" / name(u)).exists()]
    log(f"stage episodes: {len(units)} episodes, {len(units) - len(todo)} already written")

    def done(u, res):
        rec, secs = res
        C.write_atomic(run / "episodes" / name(u), C.dumps(rec))
        C.write_atomic(run / "timing" / name(u), C.dumps({"unit": list(u), "seconds": secs}))

    t0 = time.time()
    try:
        C.schedule(work, todo, args.workers, log, f"pendulum switch {args.set}", init, inputs, on_done=done)
    except Exception as exc:
        log(f"FAILED: {type(exc).__name__}: {exc}")
        (run / "EXIT").write_text("EXIT=1\n")
        raise
    recs = [json.loads((run / "episodes" / name(u)).read_text()) for u in units]
    if args.limit is None:
        rec = record(recs, identity, args.set, tol)
        C.write_atomic(run / "record.json", C.dumps(rec))
        log(f"record {W.S3.sha256(run / 'record.json')} decision " + json.dumps(rec["decision"], sort_keys=True))
        (run / "EXIT").write_text("EXIT=0\n")
    log(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
