"""#79 smoke (recovery-switch.md "Smoke, then the run").

    PY tools/recovery_switch/smoke.py --dir DIR --workers N --pendulum P --policy POLICY --stage2 STAGE2_SMOKE_DIR \
        --stage3 STAGE3_SMOKE_DIR --stage4 STAGE4_SMOKE_DIR

Stage 2's key-99 starts, and calibration episodes 0 and 1 as controls with the other 43 calibrating. Two runs in fresh
processes (DIR/a, DIR/b) must agree byte for byte on the calibration, every episode file and the record. A fresh-
process audit of DIR/a must find nothing. Each injected fault, in its own copy of DIR/a, must be refused with its own
reason first. The cost projection scales the calibration and each kind's mean episode time to the study's 400 starts
and 180 controls on its workers, against the 3 h ceiling.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import switch as W  # noqa: E402

C = W.C
HERE = Path(__file__).resolve().parent
STUDY = {"start": 400, "control": 180}
WORKERS = W.SPEC["smoke"]["workers"]
CEILING_H = W.SPEC["smoke"]["ceiling_wall_hours"]
FAULTS = ("trajectory", "agreement", "declared", "transitions", "calibration")


def files(d: Path) -> dict:
    paths = [d / "record.json", d / "calibration.json", *sorted((d / "episodes").glob("*.json"))]
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in paths}


def inject(src: Path, dst: Path, fault: str) -> dict:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.log", "EXIT", "timing"))
    if fault == "calibration":
        p = dst / "calibration.json"
        tol = json.loads(p.read_text())
        tol["delta_out"] *= 1.01
        p.write_text(C.dumps(tol))
        return {"file": p.name}
    for p in sorted((dst / "episodes").glob("*.json")):
        rec = json.loads(p.read_text())
        segs = [g for g in rec["segments"] if g["M"] is not None]
        if fault == "trajectory" and "base" in rec["modes"]:
            i = rec["modes"].index("base")
            rec["actions"][i] = rec["actions"][i] + 1e-9
        elif fault == "transitions" and rec["transitions"]:
            rec["transitions"][0]["k"] += 1
        elif fault == "agreement" and any(g["agree"] for g in segs):
            g = next(g for g in segs if g["agree"])
            g["agree"][-1] = not g["agree"][-1]
        elif fault == "declared" and segs:
            segs[0]["declared"] = not segs[0]["declared"]
        else:
            continue
        p.write_text(C.dumps(rec))
        return {"file": p.name}
    raise SystemExit(f"refused: the smoke has no episode to inject {fault} into")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=WORKERS)
    for a in ("pendulum", "policy", "stage2", "stage3", "stage4"):
        p.add_argument(f"--{a}", type=Path, required=True)
    args = p.parse_args()
    inp = [x for a in ("pendulum", "policy", "stage2", "stage3", "stage4") for x in (f"--{a}", str(getattr(args, a)))]
    args.dir.mkdir(parents=True, exist_ok=False)
    log = C.Log(args.dir / "smoke.log")
    C.header(log, {"workers": args.workers})
    py = [sys.executable]
    for run in ("a", "b"):
        log(f"stage run {run}")
        subprocess.run([*py, str(HERE / "run.py"), "--set", "smoke", "--run-dir", str(args.dir / run), "--workers",
                        str(args.workers), *inp], check=True)
    a, b = files(args.dir / "a"), files(args.dir / "b")
    identical = a == b
    log(f"rerun byte-identical: {identical} ({len(a)} files)")
    log("stage audit")
    audit = subprocess.run([*py, str(HERE / "audit.py"), "--run-dir", str(args.dir / "a"), "--workers",
                            str(args.workers), *inp, "--out", str(args.dir / "audit-a.json")])
    audit_ok = audit.returncode == 0 and json.loads((args.dir / "audit-a.json").read_text())["ok"]
    faults = {}
    for fault in FAULTS:
        log(f"stage fault {fault}")
        d = args.dir / f"fault-{fault}"
        where = inject(args.dir / "a", d, fault)
        subprocess.run([*py, str(HERE / "audit.py"), "--run-dir", str(d), "--workers", str(args.workers), *inp,
                        "--out", str(args.dir / f"audit-fault-{fault}.json")])
        res = json.loads((args.dir / f"audit-fault-{fault}.json").read_text())
        faults[fault] = {"injected": where, "first_refusal": res["first_refusal"],
                         "refused_for_the_right_reason": res["first_refusal"] == fault}
    record = json.loads((args.dir / "a" / "record.json").read_text())
    timing = [json.loads(p.read_text()) for p in sorted((args.dir / "a" / "timing").glob("*-*.json"))]
    cal_s = json.loads((args.dir / "a" / "timing" / "calibration.json").read_text())["seconds"]
    mean = {k: [t["seconds"] for t in timing if t["unit"][0] == k] for k in STUDY}
    core_s = sum(STUDY[k] * sum(v) / len(v) for k, v in mean.items() if v)
    wall_h = (cal_s + core_s / WORKERS) / 3600
    projection = {"calibration_seconds": cal_s, "episodes": {k: len(v) for k, v in mean.items()},
                  "seconds_per_episode": {k: {"mean": sum(v) / len(v), "max": max(v)} for k, v in mean.items() if v},
                  "study_core_hours": core_s / 3600, "study_workers": WORKERS, "study_wall_hours": wall_h,
                  "measured_with_workers": args.workers, "ceiling_wall_hours": CEILING_H,
                  "within_ceiling": wall_h <= CEILING_H}
    out = {"schema": "ilsb.recovery_switch.smoke 1", "code": C.code_identity(), "rerun_byte_identical": identical,
           "files_compared": len(a), "record_sha256": hashlib.sha256(a["record.json"]).hexdigest(),
           "audit_ok": audit_ok, "faults": faults, "projection": projection,
           "smoke_tolerances": record["tolerances"], "smoke_summary": record["summary"],
           "smoke_decision": record["decision"],
           "note": "Stage 2's key-99 smoke starts and calibration-episode controls; not a study outcome"}
    out["ok"] = bool(identical and audit_ok and projection["within_ceiling"]
                     and all(f["refused_for_the_right_reason"] for f in faults.values()))
    C.write_atomic(args.dir / "smoke.json", C.dumps(out))
    log(f"smoke ok {out['ok']}: identical {identical}, audit {audit_ok}, projection {wall_h:.2f} h, faults "
        + json.dumps({k: v["first_refusal"] for k, v in faults.items()}))
    (args.dir / "EXIT").write_text(f"EXIT={0 if out['ok'] else 1}\n")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
