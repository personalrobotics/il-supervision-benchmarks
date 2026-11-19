"""#82 smoke (recovery-fixed-horizon.md "Development replay, smoke, then the run").

    PY tools/recovery_fixed_horizon/smoke.py --dir DIR --workers N --pendulum P --policy POLICY --stage2 S2_SMOKE \
        --stage3 S3_SMOKE --stage4 S4_SMOKE

Stage 2's key-99 starts and #79's smoke controls, all four arms; no fresh episode is touched. Two runs in fresh
processes (DIR/a, DIR/b) must agree byte for byte. A fresh-process audit of DIR/a must find nothing. Each injected
fault, in its own copy of DIR/a, must be refused with its own reason first. The cost projection scales each kind's
mean unit time to the study's 400 fresh starts and 20 fresh controls on its workers, against the 3 h ceiling.
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
import fixedh as F  # noqa: E402

C = F.C
HERE = Path(__file__).resolve().parent
STUDY = {"start": 400, "control": 20}
WORKERS = F.SPEC["smoke"]["workers"]
CEILING_H = F.SPEC["smoke"]["ceiling_wall_hours"]
FAULTS = ("trajectory", "agreement", "validity", "handoff")


def files(d: Path) -> dict:
    paths = [d / "record.json", *sorted((d / "units").glob("*.json"))]
    return {p.relative_to(d).as_posix(): p.read_bytes() for p in paths}


def inject(src: Path, dst: Path, fault: str) -> dict:
    shutil.copytree(src, dst, ignore=shutil.ignore_patterns("*.log", "EXIT", "timing"))
    for p in sorted((dst / "units").glob("*.json")):
        rec = json.loads(p.read_text())
        a = rec["arms"]["fixed_learned"]
        recs = [g for g in a["recoveries"] if g["entry"]["feasible"]]
        if fault == "trajectory" and "base" in a["modes"]:
            i = a["modes"].index("base")
            a["actions"][i] += 1e-9
        elif fault == "agreement" and any(g["agree"] for g in recs):
            g = next(g for g in recs if g["agree"])
            g["agree"][-1] = not g["agree"][-1]
        elif fault == "validity" and a["recoveries"]:
            a["recoveries"][0]["entry"]["feasible"] = not a["recoveries"][0]["entry"]["feasible"]
        elif fault == "handoff" and any(g["handoff"] for g in recs):
            next(g for g in recs if g["handoff"])["handoff"] = False
        else:
            continue
        p.write_text(C.dumps(rec))
        return {"file": p.name}
    raise SystemExit(f"refused: the smoke has no unit to inject {fault} into")


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
    timing = [json.loads(p.read_text()) for p in sorted((args.dir / "a" / "timing").glob("*.json"))]
    secs = {k: [t["seconds"] for t in timing if t["unit"][0] == k] for k in STUDY}
    core_s = sum(STUDY[k] * sum(v) / len(v) for k, v in secs.items() if v)
    wall_h = core_s / WORKERS / 3600
    projection = {"units": {k: len(v) for k, v in secs.items()},
                  "seconds_per_unit": {k: {"mean": sum(v) / len(v), "max": max(v)} for k, v in secs.items() if v},
                  "study_core_hours": core_s / 3600, "study_workers": WORKERS, "study_wall_hours": wall_h,
                  "measured_with_workers": args.workers, "ceiling_wall_hours": CEILING_H,
                  "within_ceiling": wall_h <= CEILING_H}
    out = {"schema": "ilsb.recovery_fixed_horizon.smoke 1", "code": C.code_identity(),
           "rerun_byte_identical": identical, "files_compared": len(a),
           "record_sha256": hashlib.sha256(a["record.json"]).hexdigest(), "audit_ok": audit_ok, "faults": faults,
           "projection": projection, "smoke_summary": record["summary"], "smoke_decision": record["decision"],
           "note": "Stage 2's key-99 smoke starts and #79's smoke controls; no fresh episode; not a study outcome"}
    out["ok"] = bool(identical and audit_ok and projection["within_ceiling"]
                     and all(f["refused_for_the_right_reason"] for f in faults.values()))
    C.write_atomic(args.dir / "smoke.json", C.dumps(out))
    log(f"smoke ok {out['ok']}: identical {identical}, audit {audit_ok}, projection {wall_h:.2f} h, faults "
        + json.dumps({k: v["first_refusal"] for k, v in faults.items()}))
    (args.dir / "EXIT").write_text(f"EXIT={0 if out['ok'] else 1}\n")
    sys.exit(0 if out["ok"] else 1)


if __name__ == "__main__":
    main()
