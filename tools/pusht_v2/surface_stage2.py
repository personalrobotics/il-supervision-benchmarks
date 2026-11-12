"""The PushT surface study's Stage 1 record and the Stage 2 specification derived from it (ilsb#5).

    python tools/pusht_v2/surface_stage2.py record --root RUN_ROOT --sacct SACCT.txt \
        [--out docs/studies/evidence/pusht-surface/stage1.json]
    python tools/pusht_v2/surface_stage2.py derive [--record docs/studies/evidence/pusht-surface/stage1.json] \
        [--spec tools/pusht_v2/surface.json] [--out tools/pusht_v2/surface-stage2.json] [--check]

record  compacts a finished Stage 1 run root: every stage report is checked against the run's manifest (SHA-256),
        and the record keeps those hashes, the pool diagnostics, each unit's selection and score, the audits'
        outcomes, the report and the measured compute. SACCT.txt is `sacct -X -P -n -o JobID,State,ElapsedRaw,
        AllocTRES` for the run's jobs.
derive  re-applies the frozen Stage 1 rule (docs/studies/pusht-surface.md) to the record's per-unit scores, written
        independently of the harness's report stage, and refuses unless both agree. It then writes the Stage 2
        specification: the Stage 1 specification with the two selected cells, learner seed 2, and the counts they
        imply. --check compares with the committed file instead of writing it.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RATE_USD_PER_GPU_HOUR = 0.90
STAGES = ("preflight", "train", "audit", "evaluate", "evaluation_audit", "report")


def load(path: Path) -> dict:
    return json.loads(path.read_text())


def dump(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True)


def compute_of(sacct: Path) -> dict:
    jobs, seconds = {}, 0
    for line in sacct.read_text().splitlines():
        if not line.strip():
            continue
        job, state, elapsed, tres = line.split("|")
        if state != "COMPLETED":
            sys.exit(f"job {job} is {state}")
        gpus = int(dict(x.split("=", 1) for x in tres.split(","))["gres/gpu"])
        jobs[job] = {"gpus": gpus, "seconds": int(elapsed)}
        seconds += gpus * int(elapsed)
    return {"jobs": jobs, "gpu_hours": round(seconds / 3600, 4),
            "cost_usd_at_rate": round(seconds / 3600 * RATE_USD_PER_GPU_HOUR, 2),
            "rate_usd_per_gpu_hour": RATE_USD_PER_GPU_HOUR}


def record(root: Path, sacct: Path) -> dict:
    manifest = load(root / "manifest.json")
    stages = {}
    for s in STAGES:
        h = hashlib.sha256((root / "stages" / f"{s}.json").read_bytes()).hexdigest()
        if manifest["stages"].get(s) != h:
            sys.exit(f"stages/{s}.json does not hash to the manifest's record")
        stages[s] = load(root / "stages" / f"{s}.json")
        if not stages[s]["all_pass"]:
            sys.exit(f"stage {s} did not pass")
    pre, tr, ev, rep = stages["preflight"], stages["train"], stages["evaluate"], stages["report"]
    keep = ("episodes", "real_rows", "proposals", "selected", "usable_fraction", "violations",
            "held_out_current_frames", "previous_slot_displacement_px", "outside_frozen_range_fraction")
    units = {k: {"outcome": v["outcome"], "reached": v["reached"], "selected_step": v["selected_step"],
                 "selected_infonce": v["selected_infonce"], "per_batch": v["per_batch"], "train_entry": v["entry"],
                 "avg_max_reward": ev["units"][k]["avg_max_reward"], "pc_success": ev["units"][k]["pc_success"],
                 "evaluation_entry": ev["units"][k]["entry"]}
             for k, v in tr["units"].items()}
    return {"study": "pusht-surface-stage1",
            "protocol_sha256": manifest["protocol_sha256"],
            "spec_sha256": load(root / "root.json")["spec_sha256"],
            "code": pre["code"],
            "stage_sha256": manifest["stages"],
            "preflight": {"source_content_sha256": pre["source_content_sha256"], "plan_sha256": pre["plan_sha256"],
                          "planned_units": pre["planned_units"], "held_out_frames": pre["held_out_frames"],
                          "subset_episodes": {k: len(v) for k, v in pre["subsets"].items()},
                          "real_rows": pre["real_rows"],
                          "pools": {k: {f: v[f] for f in keep} for k, v in pre["pools"].items()}},
            "train": {"incomplete_fit": tr["incomplete_fit"], "failed": tr["failed"], "missing": tr["missing"]},
            "audit": {"units": stages["audit"]["units"], "problems": stages["audit"]["problems"]},
            "evaluation_audit": {"units": stages["evaluation_audit"]["units"],
                                 "problems": stages["evaluation_audit"]["problems"],
                                 "repeat_unit": stages["evaluation_audit"]["repeat"]["unit"]},
            "units": units,
            "report": {k: rep[k] for k in ("metric", "bc_competence", "competent", "eligible", "decision", "cells")},
            "compute": compute_of(sacct)}


def stage1_rule(rec: dict, spec: dict) -> list:
    """The frozen Stage 1 rule from per-unit scores: competence of BC at r = 1, eligibility (CCIL above duplication
    and BC with all three fits complete), and the top two by CCIL - duplication with the tie rule."""
    u, a = rec["units"], spec["analysis"]
    score = {k: v["avg_max_reward"] for k, v in u.items()}
    ref = a["references"]
    bc_full = score["bc/r1.0/a0/s1"]
    if (bc_full - ref["hold_still"]) / (ref["demonstrations"] - ref["hold_still"]) < a["competence_fraction"]:
        return []
    effect = {}
    for r in spec["real_fractions"]:
        for al in spec["synthetic_fractions"]:
            c, d, b = f"ccil/r{r}/a{al}/s1", f"duplication/r{r}/a{al}/s1", f"bc/r{r}/a0/s1"
            if (score[c] > score[d] and score[c] > score[b]
                    and all(u[k]["outcome"] == "complete" for k in (c, d, b))):
                effect[(r, al)] = score[c] - score[d]
    order = []
    while effect:
        best = max(effect.values())
        tied = sorted((k for k, e in effect.items() if best - e <= a["tie_tolerance"]), key=lambda k: (k[1], k[0]))
        order += tied
        for k in tied:
            del effect[k]
    return order[:2]


def derive(rec: dict, spec: dict) -> dict:
    chosen = stage1_rule(rec, spec)
    reported = rec["report"]["decision"]
    if reported.get("next") != "stage 2" or reported["cells"] != [f"r{r}/a{a}" for r, a in chosen]:
        sys.exit(f"the recomputed rule selects {chosen}; the report decided {reported}")
    rs = sorted({r for r, _ in chosen})
    out = dict(spec)
    out["study"] = "pusht-surface-stage2"
    out["stage"] = 2
    out["real_fractions"] = rs
    out["cells"] = [[r, a] for r, a in chosen]
    out["seeds"] = [2]
    out["expected"] = dict(spec["expected"], units=len(rs) + 2 * len(chosen),
                           subset_episodes={f"r{r}": rec["preflight"]["subset_episodes"][f"r{r}"] for r in rs})
    return out


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("record")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--sacct", type=Path, required=True)
    a.add_argument("--out", type=Path, default=Path("docs/studies/evidence/pusht-surface/stage1.json"))
    b = sub.add_parser("derive")
    b.add_argument("--record", type=Path, default=Path("docs/studies/evidence/pusht-surface/stage1.json"))
    b.add_argument("--spec", type=Path, default=Path("tools/pusht_v2/surface.json"))
    b.add_argument("--out", type=Path, default=Path("tools/pusht_v2/surface-stage2.json"))
    b.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.cmd == "record":
        args.out.write_text(dump(record(args.root, args.sacct)) + "\n")
        print(json.dumps({"out": str(args.out)}))
        return
    rec, spec = load(args.record), load(args.spec)
    if hashlib.sha256(args.spec.read_bytes()).hexdigest() != rec["spec_sha256"]:
        sys.exit(f"{args.spec} is not the specification Stage 1 ran")
    text = json.dumps(derive(rec, spec), indent=1)
    if args.check:
        if args.out.read_text() != text:
            sys.exit(f"{args.out} differs from the derived specification")
        print(json.dumps({"check": "identical", "cells": json.loads(text)["cells"]}))
    else:
        args.out.write_text(text)
        print(json.dumps({"out": str(args.out), "cells": json.loads(text)["cells"]}))


if __name__ == "__main__":
    main()
