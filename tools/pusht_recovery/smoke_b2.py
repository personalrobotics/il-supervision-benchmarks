"""#52 Stage B2 smoke: discovery and its audit on two training episodes outside the r = 0.25 subset.

    LEROBOT_PYTHON tools/pusht_recovery/smoke_b2.py --source SOURCE.npz --checkpoint CKPT --seed S --work WORK_DIR \
        [--out docs/studies/evidence/pusht-recovery/smoke-b2.json]

Outcome-free for the study: no subset episode is rolled out, so it says nothing about B2's yield. It runs discovery
twice (the records must be byte-identical), audits the first run in a fresh process, and requires the audit to refuse
each injected fault for the right reason: an executed action changed by 1e-3, a coverage value changed by 1e-6, a
selected state's k moved by one, and trajectories that no longer match the record's digest. The record holds counts
and digests only.
"""
import argparse
import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import stage_a as A  # noqa: E402
import stage_b2 as B  # noqa: E402


def faults(args, record, arrs, selection, pol) -> dict:
    i = int(selection["rollout"][0]) if len(selection["rollout"]) else 0
    o = arrs["action_offsets"][i]
    cases = {}
    a = copy.deepcopy(arrs)
    a["action"][o + 3] += 1e-3
    cases["changed_action"] = ("chunks do not regenerate", a, selection, True)
    a = copy.deepcopy(arrs)
    a["coverage"][arrs["coverage_offsets"][i] + 2] += 1e-6
    cases["changed_coverage"] = ("coverage recomputed", a, selection, True)
    s = copy.deepcopy(selection)
    if len(s["rollout"]):
        s["k"][0] += 1
    else:  # no rollout was selected: claim one
        s = {**s, "rollout": np.array([i]), "k": np.array([1]), "j_star": np.array([0])}
    cases["moved_selection"] = ("the selection does not reproduce", arrs, s, True)
    a = copy.deepcopy(arrs)
    a["obs_agent"][arrs["obs_agent_offsets"][i] + 1] += 1.0
    cases["altered_trajectories"] = ("the trajectories are not the record's", a, selection, False)
    out = {}
    for name, (want, a, s, rehash) in cases.items():
        rec = copy.deepcopy(record)
        if rehash:  # a consistent forgery: the digests match, so only the content checks can catch it
            rec["trajectories_content_sha256"], rec["selection_content_sha256"] = A.content_sha(a), A.content_sha(s)
        got = B.audit(args, run=(rec, a, s), only={i}, pol=pol)
        out[name] = {"refused": not got["pass"], "named": any(want in p for p in got["problems"])}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=A.ROOT / "docs/studies/evidence/pusht-recovery/smoke-b2.json")
    args = p.parse_args()
    A.simulator_check()
    base = [sys.executable, "-I", str(HERE / "stage_b2.py")]
    common = ["--source", str(args.source), "--checkpoint", str(args.checkpoint), "--seed", str(args.seed),
              "--episodes", "smoke"]
    for name in ("a", "b"):
        r = subprocess.run(base + common + ["--run", str(args.work / name)], capture_output=True, text=True)
        if r.returncode:
            sys.exit(f"discovery {name} exited {r.returncode}: {r.stderr[-800:]}")
    ra, rb = ((args.work / n / "record.json").read_bytes() for n in "ab")
    fresh = subprocess.run(base + ["audit"] + common + ["--run", str(args.work / "a")], capture_output=True, text=True)
    fresh_out = json.loads(fresh.stdout.strip().splitlines()[-1]) if fresh.stdout.strip() else {"problems": ["no output"]}
    record, arrs, selection = B.load_run(args.work / "a")
    ns = argparse.Namespace(source=args.source, checkpoint=args.checkpoint, episodes="smoke")
    found = faults(ns, record, arrs, selection, B.Policy(args.checkpoint))
    ok = ra == rb and fresh.returncode == 0 and all(f["refused"] and f["named"] for f in found.values())
    out = {"schema": "ilsb.pusht_recovery.smoke_b2 1", "code": record["code"], "seed": args.seed,
           "checkpoint_model_sha256": record["checkpoint_model_sha256"], "episodes": B.episodes_of("smoke"),
           "rollouts": record["rollouts"], "by_reason": record["by_reason"], "selected": record["selected"],
           "trajectories_content_sha256": record["trajectories_content_sha256"],
           "selection_content_sha256": record["selection_content_sha256"],
           "byte_identical_rerun": ra == rb,
           "fresh_process_audit": {"exit": fresh.returncode, **{k: fresh_out.get(k) for k in (
               "problems", "coverage_max_abs_diff", "chunks_mismatched", "simulator_replay_max_px")}},
           "faults": found, "pass": ok}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": ok, "by_reason": record["by_reason"], "faults": found}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
