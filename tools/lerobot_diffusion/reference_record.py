"""The PushT Diffusion Policy reference's compact record (ilsb#50), from a finished run root.

    python tools/lerobot_diffusion/reference_record.py ROOT [--spec SPEC] \
        [--out docs/studies/evidence/pusht-diffusion/reference.json]

ROOT is a finished `ilsb run SPEC --root ROOT`; SPEC defaults to tools/lerobot_diffusion/pusht-reference.json.
The record holds each stage's verdict, the commit and its cleanliness, the data and window counts, each unit's
checkpoint history (held-out loss and weights' hash), selection, peak memory and stored entry, each unit's evaluation
(average maximum coverage, success, and the start seed of every episode that did not succeed), the repeat evaluation's
measured deviation, and the learner environment's identity. Every value is copied from the run's stage reports and store
entries.
"""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
STAGES = ("preflight", "train", "audit", "evaluate", "evaluation_audit", "report")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("--spec", default="tools/lerobot_diffusion/pusht-reference.json")
    p.add_argument("--out", type=Path,
                   default=HERE.parents[1] / "docs/studies/evidence/pusht-diffusion/reference.json")
    a = p.parse_args()
    st = {s: json.loads((a.root / "stages" / f"{s}.json").read_text()) for s in STAGES}
    pre, tr, rep = st["preflight"], st["train"], st["report"]
    units = {}
    for key, u in tr["units"].items():
        res = json.loads((a.root / "store" / "policy" / u["entry"] / "result.json").read_text())
        units[key] = {"seed": u["seed"], "entry": u["entry"], "selected_step": res["selected_step"],
                      "selected_model_sha256": res["selected_model_sha256"],
                      "checkpoints": [{k: h[k] for k in ("step", "held_out_loss", "model_sha256")}
                                      for h in res["checkpoints"]],
                      "resumed_from": res["resumed_from"], "restarted_from": res["restarted_from"],
                      "peak_mib": res["peak"],
                      "evaluation": {k: rep["per_seed"][key][k] for k in ("avg_max_reward", "pc_success")},
                      "unsuccessful_start_seeds": [e["start_seed"] for e in rep["failed_episodes"][key]]}
    out = {"schema": "ilsb.pusht_diffusion.reference 1", "spec": a.spec,
           "code": pre["code"]["harness_commit"], "dirty": bool(pre["code"]["dirty"]),
           "stages": {s: v["all_pass"] for s, v in st.items()},
           "counts": pre["counts"], "windows": pre["windows"], "window_spec": pre["window_spec"],
           "split_entry": pre["split_entry"], "units": units,
           "mean_avg_max_reward": rep["mean"], "mean_pc_success": rep["mean_pc_success"],
           "acceptance": rep["acceptance"], "repeat": st["evaluation_audit"]["repeat"],
           "upstream": rep["upstream"]}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"stages": out["stages"], "mean": out["mean_avg_max_reward"], "acceptance": out["acceptance"]}))


if __name__ == "__main__":
    main()
