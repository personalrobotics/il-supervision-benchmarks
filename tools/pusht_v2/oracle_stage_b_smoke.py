"""The oracle study's Stage B smoke record (ilsb#43): the whole chain on oracle-stage-b-smoke.json, outcome-free.

    python tools/pusht_v2/oracle_stage_b_smoke.py ROOT [--out docs/studies/evidence/pusht-oracle/stage-b-smoke.json]

ROOT is a finished `ilsb run tools/pusht_v2/oracle-stage-b-smoke.json --root ROOT --gpus 0,1` (ILSB_ORACLE_INPUTS set
to the pinned labels.npz and Stage A's proposals.npz). The record holds each stage's verdict, the matched rows'
checks, and each unit's structure (rows, per-batch counts, exposure, selection, extension, index digests). No coverage,
success or substantive decision of the 2k-step smoke is recorded: they say nothing about Stage B. Its synthetic
fits are incomplete at the shortened ceiling, so the report's fail-closed verdict (inconclusive) is recorded.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--out", type=Path, default=HERE.parents[1] / "docs/studies/evidence/pusht-oracle/stage-b-smoke.json")
    a = p.parse_args()
    st = {s: json.loads((a.root / "stages" / f"{s}.json").read_text())
          for s in ("preflight", "train", "audit", "evaluate", "evaluation_audit", "report")}
    pre, tr = st["preflight"], st["train"]
    units = {}
    for key, row in tr["units"].items():
        res = json.loads((a.root / "store" / "policy" / row["entry"] / "result.json").read_text())
        units[key] = {k: res[k] for k in ("rows", "per_batch", "exposure", "reached", "selected_step", "outcome")} | {
            "index_sha256": {s: v["index_sha256"] for s, v in sorted(res["history"]["segments"].items(),
                                                                     key=lambda x: int(x[0]))}}
    out = {"schema": "ilsb.pusht_oracle.stage_b_smoke 1", "code": pre["code"]["harness_commit"],
           "dirty": bool(pre["code"]["dirty"]), "spec": "tools/pusht_v2/oracle-stage-b-smoke.json",
           "stages": {s: v["all_pass"] for s, v in st.items()},
           "preflight": {"problems": pre["problems"], "real_rows": pre["real_rows"],
                         "held_out_frames": pre["held_out_frames"], "planned_units": pre["planned_units"],
                         "matched": pre["matched"]},
           "audit_problems": st["audit"]["problems"], "evaluation_audit_problems": st["evaluation_audit"]["problems"],
           # the fail-closed rule, exercised: the smoke's synthetic fits reach its shortened ceiling, so the report
           # must be inconclusive (this records the rule's verdict, not an outcome)
           "report_fail_closed": {"outcome": st["report"]["decision"]["outcome"],
                                  "incomplete": st["report"]["decision"].get("incomplete"),
                                  "oracle_established": st["report"]["oracle_established"]},
           "units": units}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps(out["stages"]))


if __name__ == "__main__":
    main()
