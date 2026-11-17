"""#66 Stage A smoke: discovery, reach, validity and the audit on starts 3411-3420, which no study uses (stage-a.json).

    LEROBOT_PYTHON tools/pusht_failure/smoke.py --source SOURCE.npz --checkpoint CKPT --work WORK_DIR \
        [--out docs/studies/evidence/pusht-failure/smoke.json]

Outcome-free for the study: no discovery or evaluation start is rolled out. Base 1 runs the smoke starts twice (the
records must be byte-identical), and the first run is audited in a fresh process. On the smoke starts a window is
attempted at every eligible departure, not only at failures, so the construction is exercised. The audit must refuse
each injected fault for the right reason: a saved action changed by 1e-3, a label flipped, a departure step moved by
one, and a forged recovery window. The record holds counts, timings, digests and the measured cost projection.
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
import failures as F  # noqa: E402

CEILING = float(F.SPEC["compute"]["ceiling_gpu_hours"])


def faults(args, rec, arrs, w, pol, demo) -> dict:
    cases = {}
    a = copy.deepcopy(arrs)
    a["action"][3, 0, 0] += 1e-3  # start 0, step 3, repeat 0
    cases["changed_action"] = ("chunks do not regenerate", rec, a, w, {0})
    r = copy.deepcopy(rec)
    r["table"][0]["success"] = not r["table"][0]["success"]
    cases["flipped_label"] = ("labels, reach or window statuses do not reproduce", r, arrs, w, set())
    r = copy.deepcopy(rec)
    k = next(i for i, x in enumerate(r["table"]) if x["t_e"] > 0)
    r["table"][k]["t_e"] += 1
    cases["moved_departure"] = ("labels, reach or window statuses do not reproduce", r, arrs, w, set())
    x = next(x for x in rec["table"] if x["class"] == "departs_eligible")
    t = F.unpack(arrs, [int(s) for s in arrs["start"]].index(x["start"]))
    forged = F.window(F.A.make_env(), t, x["repeat"], x["t_e"], x["j_star"], demo)
    forged = {**forged, "status": "ok", "start": x["start"], "repeat": x["repeat"], "t_e": x["t_e"]}
    forged["mask"] = np.r_[False, [True] * 8, [False] * 7]
    ww = {k: np.concatenate([w[k], F.windows_arrays([forged])[k]]) for k in w}
    cases["forged_window"] = ("the windows do not rebuild bit for bit", rec, arrs, ww, set())
    out = {}
    for name, (want, r, a, ww, only) in cases.items():
        r = {**r, "trajectories_content_sha256": F.A.content_sha(a), "windows_content_sha256": F.A.content_sha(ww)}
        got = F.audit(args, run=(r, a, ww), pol=pol, demo=demo, only=only)  # a consistent forgery
        out[name] = {"refused": not got["pass"], "named": any(want in p for p in got["problems"])}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=F.ROOT / "docs/studies/evidence/pusht-failure/smoke.json")
    args = p.parse_args()
    F.A.simulator_check()
    base = [sys.executable, "-I", str(HERE / "failures.py")]
    common = ["--source", str(args.source), "--checkpoint", str(args.checkpoint), "--base", "1"]
    seconds = {}
    for name in ("a", "b"):
        t0 = time.time()
        r = subprocess.run(base + common + ["--starts", "smoke", "--run", str(args.work / name)],
                           capture_output=True, text=True)
        seconds[name] = round(time.time() - t0, 1)
        if r.returncode:
            sys.exit(f"run {name} exited {r.returncode}: {r.stderr[-800:]}")
    identical = all((args.work / "a" / f).read_bytes() == (args.work / "b" / f).read_bytes()
                    for f in ("record.json", "trajectories.npz", "windows.npz"))
    t0 = time.time()
    fresh = subprocess.run(base + ["audit"] + common + ["--run", str(args.work / "a")], capture_output=True, text=True)
    seconds["audit"] = round(time.time() - t0, 1)
    fresh_out = json.loads(fresh.stdout.strip().splitlines()[-1]) if fresh.stdout.strip() else {"problems": ["no output"]}

    rec, arrs, w = F.load_run(args.work / "a")
    ns = argparse.Namespace(checkpoint=args.checkpoint, base=1, source=args.source)
    pol, demo = F.Policy(args.checkpoint), F.demonstrations(F.B.load_source(args.source))
    found = faults(ns, rec, arrs, w, pol, demo)
    ok = identical and fresh.returncode == 0 and all(f["refused"] and f["named"] for f in found.values())
    n = len(F.starts_of("smoke"))
    per_base = (seconds["a"] + seconds["audit"]) / n * F.STARTS["discovery"][1]
    out = {"schema": "ilsb.pusht_failure.smoke 1", "code": rec["code"], "starts": F.starts_of("smoke"), "base": 1,
           "model_sha256": rec["model_sha256"], "environment": rec["environment"],
           "trajectories_content_sha256": rec["trajectories_content_sha256"],
           "windows_content_sha256": rec["windows_content_sha256"],
           "rollouts": rec["rollouts"], "rollouts_by_class": {c: sum(x["class"] == c for x in rec["table"])
                                                             for c in F.CLASSES},
           "attempted_windows": {s: sum(x["window"] == s for x in rec["table"]) for s in F.STATUS},
           "byte_identical_rerun": identical,
           "fresh_process_audit": {"exit": fresh.returncode, **{k: fresh_out.get(k) for k in (
               "problems", "chunks_mismatched", "coverage_max_abs_diff", "windows_max_landing_px",
               "simulator_replay_max_px")}},
           "faults": found, "seconds": seconds,
           "projection": {"gpu_hours_per_base": round(per_base / 3600, 2),
                          "gpu_hours_both_bases": round(2 * per_base / 3600, 2),
                          "ceiling_gpu_hours": CEILING, "within_ceiling": 2 * per_base / 3600 <= CEILING},
           "pass": ok}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": ok, "faults": found, "seconds": seconds, "projection": out["projection"]}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
