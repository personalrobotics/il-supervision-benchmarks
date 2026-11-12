"""The PushT surface study's stage records, the Stage 2 specification and the Stage 2 decision (ilsb#5).

    python tools/pusht_v2/surface_stage2.py record --root RUN_ROOT --sacct SACCT.txt --out RECORD
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

    uv run --with matplotlib==3.10.9 python tools/pusht_v2/surface_stage2.py decide [--check]

decide  applies the frozen Stage 2 rule to the Stage 1 and Stage 2 records (seeds 1 and 2), checking seed 2 against
        the Stage 2 report, and writes decision.json (with both records' SHA-256) and the result figure,
        pusht-surface-result.svg. --check regenerates both and compares them byte for byte.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

RATE_USD_PER_GPU_HOUR = 0.90
EV = Path("docs/studies/evidence/pusht-surface")
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
    return {"study": load(root / "root.json")["spec"]["study"],
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


def decide(rec1: dict, rec2: dict, spec2: dict) -> dict:
    """The frozen Stage 2 rule: a cell stays eligible only if CCIL is above duplication and BC, with all three fits
    complete, on both development seeds; none eligible stops the study, otherwise the largest mean CCIL - duplication
    is selected with the tie rule. Seed 2's values are checked against the Stage 2 report."""
    out, eligible = {}, {}
    for r, al in spec2["cells"]:
        per = {}
        for s, rec in (("1", rec1), ("2", rec2)):
            u = rec["units"]
            c, d, b = (u[f"{arm}/r{r}/a{x}/s{s}"] for arm, x in (("ccil", al), ("duplication", al), ("bc", 0)))
            per[s] = {"ccil": c["avg_max_reward"], "duplication": d["avg_max_reward"], "bc": b["avg_max_reward"],
                      "ccil_minus_duplication": c["avg_max_reward"] - d["avg_max_reward"],
                      "ccil_minus_bc": c["avg_max_reward"] - b["avg_max_reward"],
                      "complete": all(x["outcome"] == "complete" for x in (c, d, b))}
        key = f"r{r}/a{al}"
        reported = rec2["report"]["cells"][key]["per_seed"]["2"]
        if any(reported[f] != per["2"][f] for f in ("ccil", "duplication", "bc")):
            sys.exit(f"{key}: seed 2's scores differ from the Stage 2 report")
        ok = all(v["ccil_minus_duplication"] > 0 and v["ccil_minus_bc"] > 0 and v["complete"] for v in per.values())
        mean = (per["1"]["ccil_minus_duplication"] + per["2"]["ccil_minus_duplication"]) / 2
        out[key] = {"per_seed": per, "eligible": ok, "mean_ccil_minus_duplication": mean}
        if ok:
            eligible[key] = (mean, al, r)
    if not eligible:
        decision = {"next": "stop", "why": "no Stage 2 cell has CCIL above both duplication and BC on both "
                                           "development seeds; Stage 3 does not run"}
    else:
        best = max(m for m, _, _ in eligible.values())
        top = sorted((k for k, v in eligible.items() if best - v[0] <= spec2["analysis"]["tie_tolerance"]),
                     key=lambda k: eligible[k][1:])
        decision = {"next": "stage 3", "cell": top[0]}
    return {"study": "pusht-surface-stage2-decision", "records_sha256": {}, "cells": out,
            "eligible": sorted(eligible), "decision": decision}


INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#ffffff"
ABOVE, BELOW, NEUTRAL = "#2a78d6", "#e34948", "#f0efec"   # diverging pair, gray midpoint (dataviz palette)
CELL_COLORS = ("#1baf7a", "#6250d6")                     # categorical slots 3 and 7 (validate_palette: pass)


def draw(rec1: dict, spec1: dict, dec: dict) -> str:
    """Left: the Stage 1 surface of CCIL - duplication. Right: the two candidate cells on seeds 1 and 2."""
    import io
    import matplotlib
    matplotlib.use("svg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm
    from matplotlib.patches import Rectangle
    plt.rcParams.update({"svg.hashsalt": "pusht-surface-result", "svg.fonttype": "none",
                         "font.family": "DejaVu Sans", "font.size": 8, "axes.edgecolor": GRID,
                         "axes.labelcolor": MUTED, "xtick.color": MUTED, "ytick.color": MUTED, "axes.titlesize": 8.5,
                         "axes.titlecolor": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE})
    rs, als = spec1["real_fractions"], spec1["synthetic_fractions"]
    sc = {k: v["avg_max_reward"] for k, v in rec1["units"].items()}
    grid = [[sc[f"ccil/r{r}/a{a}/s1"] - sc[f"duplication/r{r}/a{a}/s1"] for a in als] for r in rs]
    fig, (ax, bx) = plt.subplots(1, 2, figsize=(7.2, 3.0), gridspec_kw={"width_ratios": [1.0, 1.0], "wspace": 0.45})
    cmap = LinearSegmentedColormap.from_list("div", [BELOW, NEUTRAL, ABOVE])
    lim = 0.3
    ax.imshow(grid, cmap=cmap, norm=TwoSlopeNorm(0.0, -lim, lim), origin="lower", aspect="auto")
    for i, row in enumerate(grid):
        for j, v in enumerate(row):
            ax.text(j, i, f"{v:+.3f}", ha="center", va="center", color=INK, fontsize=7.5)
    keys = list(dec["cells"])
    for n, key in enumerate(keys):
        r, a = (float(x[1:]) for x in key.split("/"))
        ax.add_patch(Rectangle((als.index(a) - 0.47, rs.index(r) - 0.47), 0.94, 0.94, fill=False,
                               edgecolor=CELL_COLORS[n], linewidth=2.0))
    ax.set_xticks(range(len(als)), [f"{a:g}" for a in als])
    ax.set_yticks(range(len(rs)), [f"{r:g}" for r in rs])
    ax.set_xlabel("synthetic fraction α")
    ax.set_ylabel("real fraction r")
    ax.set_title("Stage 1 (seed 1): CCIL − duplication", loc="left")
    for side in ax.spines.values():
        side.set_visible(False)
    ax.tick_params(length=0)
    bx.axhline(0, color=MUTED, linewidth=0.8)
    for n, key in enumerate(keys):
        y = [dec["cells"][key]["per_seed"][s]["ccil_minus_duplication"] for s in ("1", "2")]
        bx.plot([1, 2], y, color=CELL_COLORS[n], linewidth=2.0, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        r, a = key.split("/")
        bx.text(2.08, y[1], f"r = {r[1:]}, α = {a[1:]}", va="center", color=INK, fontsize=7.5)
    bx.set_xlim(0.8, 2.75)
    bx.set_xticks([1, 2], ["seed 1\n(Stage 1)", "seed 2\n(Stage 2)"])
    bx.set_ylabel("CCIL − duplication (avg max coverage)")
    bx.set_title("Replication of the two candidates", loc="left")
    bx.grid(axis="y", color=GRID, linewidth=0.6)
    bx.set_axisbelow(True)
    for side in ("top", "right"):
        bx.spines[side].set_visible(False)
    bx.tick_params(length=0)
    buf = io.StringIO()
    fig.savefig(buf, format="svg", metadata={"Date": None, "Creator": None}, bbox_inches="tight", dpi=150)
    plt.close(fig)
    return buf.getvalue()


def write_or_check(path: Path, text: str, check: bool) -> None:
    if check:
        if path.read_text() != text:
            sys.exit(f"{path} differs from the regenerated file")
    else:
        path.write_text(text)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("record")
    a.add_argument("--root", type=Path, required=True)
    a.add_argument("--sacct", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    b = sub.add_parser("derive")
    b.add_argument("--record", type=Path, default=Path("docs/studies/evidence/pusht-surface/stage1.json"))
    b.add_argument("--spec", type=Path, default=Path("tools/pusht_v2/surface.json"))
    b.add_argument("--out", type=Path, default=Path("tools/pusht_v2/surface-stage2.json"))
    b.add_argument("--check", action="store_true")
    c = sub.add_parser("decide")
    c.add_argument("--stage1", type=Path, default=EV / "stage1.json")
    c.add_argument("--stage2", type=Path, default=EV / "stage2.json")
    c.add_argument("--spec", type=Path, default=Path("tools/pusht_v2/surface-stage2.json"))
    c.add_argument("--out", type=Path, default=EV / "decision.json")
    c.add_argument("--figure", type=Path, default=EV / "pusht-surface-result.svg")
    c.add_argument("--stage1-spec", type=Path, default=Path("tools/pusht_v2/surface.json"))
    c.add_argument("--check", action="store_true")
    args = p.parse_args()
    if args.cmd == "decide":
        rec1, rec2, spec2 = load(args.stage1), load(args.stage2), load(args.spec)
        if hashlib.sha256(args.spec.read_bytes()).hexdigest() != rec2["spec_sha256"]:
            sys.exit(f"{args.spec} is not the specification Stage 2 ran")
        dec = decide(rec1, rec2, spec2)
        dec["records_sha256"] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in (args.stage1, args.stage2)}
        write_or_check(args.out, dump(dec) + "\n", args.check)
        write_or_check(args.figure, draw(rec1, load(args.stage1_spec), dec), args.check)
        print(json.dumps({"decision": dec["decision"], "eligible": dec["eligible"], "check": args.check}))
        return
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
