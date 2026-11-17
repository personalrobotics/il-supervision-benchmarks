"""#62 Stage A smoke: the probe and its audit on starts 3401-3410, which no study uses (stage-a.json).

    LEROBOT_PYTHON tools/pusht_uncertainty/smoke_probe.py --members CKPT1 CKPT2 --work WORK_DIR \
        [--out docs/studies/evidence/pusht-uncertainty/smoke.json]

Outcome-free for the study: no development or final-test start is rolled out. Driver 1 runs the smoke starts twice with
the probe (the records must be byte-identical) and once without it (the trajectories must be identical: the probe
cannot alter the rollout). The first run is audited in a fresh process. Member 1 probed against a second, separately
loaded copy of itself must give U_epi = 0 exactly at every decision point, which shows the noise is shared. The audit
must refuse each injected fault for the right reason: a probe output changed by 1e-3, an episode's label flipped, and a
decision point moved by one step. The record holds counts, timings and digests only.
"""
import argparse
import copy
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import probe as P  # noqa: E402


def faults(args, rec, arrs, probes, pols) -> dict:
    i, k = 0, int(np.flatnonzero(probes["episode"] == 0)[1])
    cases = {}
    p = copy.deepcopy(probes)
    p["output"][k, 1, 0, 0, 0] += 1e-3
    cases["changed_probe_output"] = ("probe batches do not regenerate", rec, p)
    r = copy.deepcopy(rec)
    r["episodes"][i]["success"] = not r["episodes"][i]["success"]
    cases["flipped_label"] = ("the label does not reproduce", r, probes)
    p = copy.deepcopy(probes)
    p["step"][k] += 1
    cases["moved_decision_point"] = ("decision points are not the chunk boundaries", rec, p)
    out = {}
    for name, (want, r, p) in cases.items():
        r = {**r, "probes_content_sha256": P.A.content_sha(p)}  # a consistent forgery: only content checks can catch it
        got = P.audit(args, run=(r, arrs, p), only={i}, pols=pols)
        out[name] = {"refused": not got["pass"], "named": any(want in x for x in got["problems"])}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--members", type=Path, nargs=2, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=P.ROOT / "docs/studies/evidence/pusht-uncertainty/smoke.json")
    args = p.parse_args()
    P.A.simulator_check()
    base = [sys.executable, "-I", str(HERE / "probe.py")]
    common = ["--members", *map(str, args.members)]
    seconds = {}
    for name, extra in (("a", []), ("b", []), ("off", ["--no-probe"])):
        t0 = time.time()
        r = subprocess.run(base + common + ["--driver", "1", "--starts", "smoke", "--run", str(args.work / name), *extra],
                           capture_output=True, text=True)
        seconds[name] = round(time.time() - t0, 1)
        if r.returncode:
            sys.exit(f"run {name} exited {r.returncode}: {r.stderr[-800:]}")
    ra, rb, roff = (json.loads((args.work / n / "record.json").read_text()) for n in ("a", "b", "off"))
    identical = (args.work / "a/record.json").read_bytes() == (args.work / "b/record.json").read_bytes()
    invariant = ra["trajectories_content_sha256"] == roff["trajectories_content_sha256"]
    t0 = time.time()
    fresh = subprocess.run(base + ["audit"] + common + ["--run", str(args.work / "a")], capture_output=True, text=True)
    seconds["audit"] = round(time.time() - t0, 1)
    fresh_out = json.loads(fresh.stdout.strip().splitlines()[-1]) if fresh.stdout.strip() else {"problems": ["no output"]}

    rec, arrs, probes = P.load_run(args.work / "a")
    ns = argparse.Namespace(members=args.members)
    pols = [P.Policy(m) for m in args.members]
    twin = P.Probe([pols[0], P.Policy(args.members[0])])
    worst = 0.0
    for k in range(len(probes["step"])):
        t = P.unpack(arrs, int(probes["episode"][k]))
        step = int(probes["step"][k])
        out = twin((t["obs_agent"][max(step - 1, 0)], t["obs_kp"][max(step - 1, 0)]),
                   (t["obs_agent"][step], t["obs_kp"][step]), int(probes["seed"][k]))
        worst = max(worst, P.u_of(out)[0])
    found = faults(ns, rec, arrs, probes, pols)
    ok = (identical and invariant and fresh.returncode == 0 and worst == 0.0
          and all(f["refused"] and f["named"] for f in found.values()))
    out = {"schema": "ilsb.pusht_uncertainty.smoke 1", "code": rec["code"], "starts": P.starts_of("smoke"),
           "members_model_sha256": rec["members_model_sha256"], "environment": rec["environment"], "driver": 1,
           "decision_points": int(len(probes["step"])), "episodes_steps": [e["steps"] for e in rec["episodes"]],
           "trajectories_content_sha256": rec["trajectories_content_sha256"],
           "probes_content_sha256": rec["probes_content_sha256"],
           "byte_identical_rerun": identical, "probe_cannot_alter_rollout": invariant,
           "identity_control_max_u_epi": worst,
           "fresh_process_audit": {"exit": fresh.returncode, **{k: fresh_out.get(k) for k in (
               "problems", "coverage_max_abs_diff", "chunks_mismatched", "probes_mismatched",
               "simulator_replay_max_px")}},
           "faults": found, "seconds": seconds, "pass": ok}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": ok, "faults": found, "seconds": seconds}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
