"""#76 smoke on Stage 2's key-99 Pendulum starts (recovery-receding.md "Smoke, then the run").

    PY tools/recovery_receding/smoke.py --dir DIR --workers N --pendulum P --policy POLICY --stage2 STAGE2_SMOKE_DIR \
        --stage3 STAGE3_SMOKE_DIR

Two runs in fresh processes (DIR/a, DIR/b) must agree byte for byte on every start file and the record. A fresh-
process audit of DIR/a must find nothing. Each injected fault, in its own copy of DIR/a, must be refused with its own
reason first. The cost projection is the complete path of every smoke start (both arms, every M), scaled to the
study's 400 starts on its workers, against the 3 h ceiling.
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
import receding as R  # noqa: E402

C = R.C
HERE = Path(__file__).resolve().parent
STUDY_STARTS = 400
WORKERS = R.SPEC["smoke"]["workers"]
CEILING_H = R.SPEC["smoke"]["ceiling_wall_hours"]
FAULTS = ("bounds", "replay", "refinement", "declared")


def files(d: Path) -> dict:
    paths = [d / "record.json", *sorted((d / "starts").glob("*.json"))]
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in paths}


def inject(src: Path, dst: Path, fault: str) -> dict:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.log", "EXIT", "timing"))
    for p in sorted((dst / "starts").glob("*.json")):
        rec = json.loads(p.read_text())
        hit = next(((arm, M, g) for arm in R.ARMS for M, g in sorted(rec["arms"][arm]["executions"].items(),
                                                                       key=lambda kv: int(kv[0]))
                    if g["actions"] and (fault != "declared" or not g["declared"])), None)
        if hit is None:
            continue
        arm, M, g = hit
        if fault == "bounds":
            g["actions"][0] = 4.0
        elif fault == "replay":
            g["states"][-1][0] += 1e-3
        elif fault == "refinement":
            g["refined_miss"] += 1
        elif fault == "declared":
            g["declared"] = True
        p.write_text(C.dumps(rec))
        return {"file": p.name, "arm": arm, "M": M}
    raise SystemExit(f"refused: the smoke has no execution to inject {fault} into")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", type=Path, required=True)
    p.add_argument("--workers", type=int, default=WORKERS)
    p.add_argument("--pendulum", type=Path, required=True)
    p.add_argument("--policy", type=Path, required=True)
    p.add_argument("--stage2", type=Path, required=True)
    p.add_argument("--stage3", type=Path, required=True)
    args = p.parse_args()
    inp = ["--pendulum", str(args.pendulum), "--policy", str(args.policy), "--stage2", str(args.stage2),
           "--stage3", str(args.stage3)]
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
    secs = [json.loads(p.read_text())["seconds"] for p in sorted((args.dir / "a" / "timing").glob("[0-9]*.json"))]
    core_s = sum(secs) * STUDY_STARTS / len(secs)
    wall_h = core_s / WORKERS / 3600
    projection = {"smoke_starts": len(secs), "seconds_per_start": {"mean": sum(secs) / len(secs), "max": max(secs)},
                  "study_core_hours": core_s / 3600, "study_workers": WORKERS, "study_wall_hours": wall_h,
                  "measured_with_workers": args.workers, "ceiling_wall_hours": CEILING_H,
                  "within_ceiling": wall_h <= CEILING_H}
    out = {"schema": "ilsb.recovery_receding.smoke 1", "code": C.code_identity(), "rerun_byte_identical": identical,
           "files_compared": len(a), "record_sha256": hashlib.sha256(a["record.json"]).hexdigest(),
           "audit_ok": audit_ok, "faults": faults, "projection": projection,
           "smoke_summary": record["summary"], "smoke_decision": record["decision"],
           "note": "Stage 2's key-99 smoke starts, disjoint from the study's; not a study outcome"}
    out["ok"] = bool(identical and audit_ok and projection["within_ceiling"]
                     and all(f["refused_for_the_right_reason"] for f in faults.values()))
    C.write_atomic(args.dir / "smoke.json", C.dumps(out))
    log(f"smoke ok {out['ok']}: identical {identical}, audit {audit_ok}, projection {wall_h:.2f} h, faults "
        + json.dumps({k: v["first_refusal"] for k, v in faults.items()}))
    (args.dir / "EXIT").write_text(f"EXIT={0 if out['ok'] else 1}\n")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
