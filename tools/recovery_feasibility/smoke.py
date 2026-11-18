"""#70 smoke for one system, on its key-99 anchors (recovery-feasibility.md "Smoke, then the run").

    PY tools/recovery_feasibility/smoke.py --system SYSTEM --dir DIR --workers N INPUT
    PY tools/recovery_feasibility/smoke.py --combine SMOKE.json ... --out PROJECTION.json

Two runs in fresh processes (DIR/a, DIR/b) must agree byte for byte on every start file and the record. A fresh-
process audit of DIR/a must find nothing. Each injected fault, in its own copy of DIR/a, must be refused, and the
audit's first refusal must be the fault's own reason. The cost projection times the complete path of every smoke
start (availability, eligibility, the M = 1 certificate, every M, every restart, every check) and scales it to the
study's 400 starts on the study's workers. --combine adds the systems' projections against the 3 h ceiling.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common as C  # noqa: E402

HERE = Path(__file__).resolve().parent
STUDY_STARTS = C.SPEC["starts"]["starts_per_system"]
STUDY_WORKERS = C.SPEC["compute"]["workers"]
CEILING_H = C.SPEC["compute"]["ceiling_wall_hours"]
BOUND_PAST = {"double_integrator": None, "pendulum": 4.0, "pusht": 513.0}  # an action outside each system's bounds
FAULTS = {"double_integrator": ("bounds", "replay", "certificate"),
          "pendulum": ("bounds", "replay", "certificate", "refinement"),
          "pusht": ("bounds", "replay", "certificate", "handoff", "refinement", "continuation")}


def files(d: Path) -> dict:
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in sorted([d / "record.json", *(d / "starts").glob("*.json")])}


def target_segment(d: Path):
    """The first segment of the first start with an eligible rung, preferring a valid one."""
    first = None
    for p in sorted((d / "starts").glob("*.json")):
        rec = json.loads(p.read_text())
        for M, rung in sorted(rec.get("rungs", {}).items(), key=lambda x: int(x[0])):
            for solver, v in sorted((k, v) for k, v in rung.items() if isinstance(v, dict) and "segments" in v):
                for k, g in enumerate(v["segments"]):
                    if g["valid"]:
                        return p, M, solver, k
                    first = first or (p, M, solver, k)
    return first


def inject(src: Path, dst: Path, fault: str, system: str, ubar: float) -> dict:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.log", "EXIT", "timing"))
    if fault == "certificate":
        p = next(p for p in sorted((dst / "starts").glob("*.json")) if json.loads(p.read_text()).get("m1"))
        rec = json.loads(p.read_text())
        rec["m1"]["infeasible"] = not rec["m1"]["infeasible"]
        p.write_text(C.dumps(rec))
        return {"file": p.name}
    p, M, solver, k = target_segment(dst)
    p = dst / "starts" / p.name
    rec = json.loads(p.read_text())
    g = rec["rungs"][M][solver]["segments"][k]
    if fault == "bounds":
        g["actions"][0][0] = BOUND_PAST[system] or ubar + 1.0
    elif fault == "replay":  # a forged terminal state
        g["states"][-1][0] += 1e-3
    elif fault == "refinement":
        g["refined_miss"] = g["refined_miss"] + 10.0 * (g["miss"] + 1.0)
    elif fault == "handoff":
        g["handoff_miss"] = g["handoff_miss"] + 10.0 * (g["miss"] + 1.0)
    elif fault == "continuation":
        g["continuation_px"] = g["continuation_px"] + 1.0
    p.write_text(C.dumps(rec))
    return {"file": p.name, "M": M, "solver": solver, "restart": k, "segment_valid": g["valid"]}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--system", choices=list(C.SYSTEMS))
    p.add_argument("--dir", type=Path)
    p.add_argument("--workers", type=int, default=STUDY_WORKERS)
    p.add_argument("--quickstart", type=Path)
    p.add_argument("--pendulum", type=Path)
    p.add_argument("--source", type=Path)
    p.add_argument("--combine", type=Path, nargs="+")
    p.add_argument("--out", type=Path)
    args = p.parse_args()
    if args.combine:
        parts = {json.loads(f.read_text())["system"]: json.loads(f.read_text()) for f in args.combine}
        if set(parts) != set(C.SYSTEMS) or not all(v["ok"] for v in parts.values()):
            sys.exit("refused: the projection needs a passing smoke of every system")
        total = sum(v["projection"]["study_wall_hours"] for v in parts.values())
        out = {"schema": "ilsb.recovery_feasibility.projection 1", "ceiling_wall_hours": CEILING_H,
               "by_system": {k: v["projection"] for k, v in sorted(parts.items())}, "study_wall_hours": total,
               "within_ceiling": total <= CEILING_H, "code": C.code_identity()}
        C.write_atomic(args.out, C.dumps(out))
        print(C.dumps(out), end="")
        sys.exit(0 if out["within_ceiling"] else 1)

    flag = {"double_integrator": "--quickstart", "pendulum": "--pendulum", "pusht": "--source"}[args.system]
    data = {"double_integrator": args.quickstart, "pendulum": args.pendulum, "pusht": args.source}[args.system]
    inp = [flag, str(data)]
    args.dir.mkdir(parents=True, exist_ok=False)
    log = C.Log(args.dir / "smoke.log")
    C.header(log, {"system": args.system, "workers": args.workers})
    py = [sys.executable]
    for run in ("a", "b"):
        log(f"stage run {run}")
        subprocess.run([*py, str(HERE / "run.py"), "--system", args.system, "--set", "smoke", "--run-dir",
                        str(args.dir / run), "--workers", str(args.workers), *inp], check=True)
    a, b = files(args.dir / "a"), files(args.dir / "b")
    identical = a == b
    log(f"rerun byte-identical: {identical} ({len(a)} files)")
    log("stage audit")
    audit = subprocess.run([*py, str(HERE / "audit.py"), "--run-dir", str(args.dir / "a"), "--workers",
                            str(args.workers), *inp, "--out", str(args.dir / "audit-a.json")])
    audit_ok = audit.returncode == 0 and json.loads((args.dir / "audit-a.json").read_text())["ok"]
    ubar = json.loads((args.dir / "a" / "identity.json").read_text())["identity"]["ubar"]
    faults = {}
    for fault in FAULTS[args.system]:
        log(f"stage fault {fault}")
        d = args.dir / f"fault-{fault}"
        where = inject(args.dir / "a", d, fault, args.system, ubar)
        subprocess.run([*py, str(HERE / "audit.py"), "--run-dir", str(d), "--workers", str(args.workers), *inp,
                        "--out", str(args.dir / f"audit-fault-{fault}.json")])
        res = json.loads((args.dir / f"audit-fault-{fault}.json").read_text())
        faults[fault] = {"injected": where, "first_refusal": res["first_refusal"],
                         "refused_for_the_right_reason": res["first_refusal"] == fault}
    secs = [json.loads(p.read_text())["seconds"] for p in sorted((args.dir / "a" / "timing").glob("*.json"))]
    core_s = sum(secs) * STUDY_STARTS / len(secs)
    projection = {"smoke_starts": len(secs), "smoke_core_seconds": sum(secs),
                  "seconds_per_start": {"mean": sum(secs) / len(secs), "max": max(secs)},
                  "study_core_hours": core_s / 3600, "study_workers": STUDY_WORKERS,
                  "study_wall_hours": core_s / STUDY_WORKERS / 3600, "measured_with_workers": args.workers}
    record = json.loads((args.dir / "a" / "record.json").read_text())
    out = {"schema": "ilsb.recovery_feasibility.smoke 1", "system": args.system, "code": C.code_identity(),
           "rerun_byte_identical": identical, "files_compared": len(a),
           "record_sha256": __import__("hashlib").sha256(a["record.json"]).hexdigest(),
           "audit_ok": audit_ok, "faults": faults, "projection": projection,
           "smoke_summary": record["summary"], "smoke_decision": record["decision"],
           "note": "the key-99 smoke anchors only, disjoint from the study's; not a study outcome"}
    out["ok"] = identical and audit_ok and all(f["refused_for_the_right_reason"] for f in faults.values())
    C.write_atomic(args.dir / "smoke.json", C.dumps(out))
    log(f"smoke ok {out['ok']}: identical {identical}, audit {audit_ok}, faults "
        + json.dumps({k: v["first_refusal"] for k, v in faults.items()}))
    (args.dir / "EXIT").write_text(f"EXIT={0 if out['ok'] else 1}\n")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
