#!/usr/bin/env python3
"""The compact behavior-level fixture of the LeRobot Diffusion Policy learner (il-supervision-benchmarks#50).

    python tools/diffusion_fixture.py --work NEW_DIR --lerobot-python PYTHON [--dataset DIR] [--out EVIDENCE.json]

It runs from whatever ilsb is importable (CI: the built wheel in a clean environment) against a LeRobot environment
installed from tools/lerobot_diffusion/lerobot-env.txt (CI: CPU PyTorch). Without --dataset, lerobot/pusht_keypoints
is fetched at the reference's revision and checked by its files' hashes. On 6 training and 2 held-out PushT episodes,
a small Diffusion Policy (the reference's temporal window: 2 observations, horizon 16, 8 executed actions) trains for
200 steps on the CPU, two seeds, and is selected by held-out loss and evaluated on two short episodes. Each protected
failure, and its check:

- the adapter's wiring breaks (split, windows, lerobot-train, scoring, selection, lerobot-eval, audits, report): the
  study. Every stage passes and every unit has checkpoints at 100 and 200.
- the learner's windows are not the contract's (anchors, offsets, padding, episode containment): preflight compares
  every window LeRobotDataset gives the policy with ilsb.temporal's, slot by slot, on both roles; a contract whose
  anchor rule differs from the learner's (one fewer dropped frame) is reported; rows whose episodes are not
  contiguous, and a window with a slot moved into another episode, are refused.
- normalization reads held-out episodes: the split's statistics are its rows' (preflight), and the audit refuses a
  checkpoint whose normalizer holds the full dataset's statistics.
- the selection or the checkpoint is not the record's: the audit recomputes the selection from the history,
  re-scores the selected checkpoint in a fresh process, and refuses a store whose selected weights changed.
- an interrupted unit does not continue exactly: a unit killed after its first checkpoint reaches every checkpoint's
  weights bit for bit as the uninterrupted unit did, and records how. On the CPU it restarts (lerobot-train's resume
  is not exact there); with --device cuda (two data-loading workers) it resumes through lerobot-train's own resume.
- a resume or a shared store recomputes: the same command reruns no stage; a second root on the store reuses every
  unit and rollout.
- a wrong input is accepted: overlapping roles, a learner environment that is not the pinned LeRobot, and a dataset
  that is not the declared one are refused before anything trains.

The written evidence holds no paths, times or commits, so a rerun on the same host reproduces it byte for byte.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from ilsb import temporal
from ilsb.learners import lerobot_diffusion
from ilsb.studies import lerobot_policy

REPO = Path(__file__).resolve().parents[1]
REFERENCE = REPO / "tools" / "lerobot_diffusion" / "pusht-reference.json"
EXPECTED = {"train_episodes": 6, "held_out_episodes": 2, "train_frames": 895, "held_out_frames": 303,
            "train_windows": 853, "held_out_windows": 289, "units": 2}


def last_json(p: subprocess.CompletedProcess):
    """The JSON line a command ends with (ilsb run's stage outcomes), or its failure."""
    lines = p.stdout.strip().splitlines()
    return json.loads(lines[-1]) if lines else {"exit": p.returncode, "stderr": p.stderr[-500:]}


def fixture_spec(ref: dict, device: str) -> dict:
    s = json.loads(json.dumps(ref))
    s["study"] = "pusht-diffusion-fixture"
    s["task"]["env"]["episode_length"] = 24
    s["roles"] = {"train": sorted(ref["roles"]["train"])[:6], "held_out": sorted(ref["roles"]["held_out"])[:2]}
    s["learner"].update({"steps": 200, "batch_size": 8, "save_freq": 100, "log_freq": 50,
                         "num_workers": 2 if device == "cuda" else 0,
                         "selection": {"rule": "min_held_out_loss", "seed": 0, "batch_size": 64}})
    s["learner"]["policy"].update({"down_dims": [16, 32], "diffusion_step_embed_dim": 16, "num_inference_steps": 4})
    s["evaluation"] = {"start_seed": 1000, "n_episodes": 2, "batch_size": 2, "device": device}
    s["expected"] = dict(EXPECTED)
    s["compute"] = {"device": "cpu", "learner_device": device, "stall_minutes": 15}
    return s


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--lerobot-python", type=Path, required=True)
    ap.add_argument("--dataset", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    ap.add_argument("--allow-dirty", action="store_true")
    a = ap.parse_args()
    work = a.work.resolve()
    work.mkdir(parents=True)
    ref = json.loads(REFERENCE.read_text())
    if a.dataset is None:
        from huggingface_hub import snapshot_download
        a.dataset = Path(snapshot_download(ref["data"]["repo_id"], repo_type="dataset",
                                           revision=ref["data"]["revision"], local_dir=work / "dataset",
                                           allow_patterns=sorted({f.split("/")[0] + "/**" for f in ref["data"]["files"]})))
    env = {**os.environ, "ILSB_LEROBOT_PYTHON": str(a.lerobot_python.absolute()),  # not resolved: a venv's python
           "ILSB_LEROBOT_DATASET": str(a.dataset.resolve())}
    os.environ.update({k: env[k] for k in ("ILSB_LEROBOT_PYTHON", "ILSB_LEROBOT_DATASET")})  # in-process checks

    def ilsb(*args, env_=None):
        return subprocess.run([sys.executable, "-m", "ilsb", *map(str, args),
                               *(["--allow-dirty"] if a.allow_dirty else [])], capture_output=True, text=True,
                              env=env_ or env)

    def write(path, spec):
        path.write_text(json.dumps(spec, indent=1))
        return path

    def stage(root, name):
        return json.loads((root / "stages" / f"{name}.json").read_text())

    spec = fixture_spec(ref, a.device)
    slots = "0,0" if a.device == "cuda" else "cpu,cpu"
    main_spec = write(work / "spec.json", spec)
    store, root = work / "store", work / "root"
    ev = {"spec": spec}

    # 1. The study: every stage, two seeds on two CPU slots. Its artifacts are the ones the later checks start from.
    p = ilsb("run", main_spec, "--root", root, "--store", store, "--gpus", slots)
    stages = last_json(p)
    if p.returncode:
        print(p.stdout, p.stderr)
        for f in sorted(root.rglob("*.log")):
            print(f, f.read_text()[-3000:])
        return 1
    pre, tr, rep = stage(root, "preflight"), stage(root, "train"), stage(root, "report")
    results = {k: json.loads((store / "policy" / u["entry"] / "result.json").read_text()) for k, u in tr["units"].items()}
    ev["study"] = {
        "stages": stages, "counts": pre["counts"], "windows": pre["windows"], "window_spec": pre["window_spec"],
        "units": {k: {"selected_step": u["selected_step"], "checkpoints": [h["step"] for h in results[k]["checkpoints"]],
                      "model_sha256": [h["model_sha256"] for h in results[k]["checkpoints"]],
                      "resumed_from": results[k]["resumed_from"]} for k, u in tr["units"].items()},
        "evaluation_repeat_identical": stage(root, "evaluation_audit")["repeat"]["identical"],
        "evaluated_episodes": {k: len(v) for k, v in rep["failed_episodes"].items()},
        "acceptance_reported": set(rep["acceptance"]) == {"min_mean", "met"}}
    ev["study"]["pass"] = bool(set(stages.values()) == {"pass"} and len(stages) == 6
                               and all(v["checkpoints"] == [100, 200] for v in ev["study"]["units"].values())
                               and ev["study"]["evaluation_repeat_identical"])

    # 2. The contract's own refusals and its agreement with the learner, on the split's rows.
    ctx = SimpleNamespace(root=root, spec=spec)
    split = root / "data" / pre["split_entry"]
    rows = lerobot_policy.read_rows(split / "train")
    ws = lerobot_diffusion.window_spec(spec["learner"])
    off = temporal.WindowSpec(ws.n_obs_steps, ws.horizon, ws.n_action_steps, ws.drop_n_last_frames - 1,
                              ws.supervise_padding)
    _, problems = lerobot_policy.check_windows(ctx, split / "train", rows, off)
    ep, fr = rows["episode_index"].copy(), rows["frame_index"]
    swapped = ep.copy()
    swapped[[0, -1]] = swapped[[-1, 0]]
    w = temporal.windows(ep, fr, ws)
    moved = {k: v.copy() for k, v in w.items()}
    moved["action_rows"][0, -1] = int(np.flatnonzero(ep != ep[0])[0])

    def refused(f):
        try:
            f()
        except temporal.WindowError:
            return True
        return False
    ev["contract"] = {"other_anchor_rule_reported": any("anchors differ" in x for x in problems),
                      "interleaved_episodes_refused": refused(lambda: temporal.windows(swapped, fr, ws)),
                      "cross_episode_slot_refused": refused(lambda: temporal.verify(moved, ep, fr, ws))}

    # 3. Normalization: a checkpoint whose normalizer holds the full dataset's statistics is refused.
    k1 = next(iter(tr["units"]))
    leak = work / "leaked-checkpoint"
    shutil.copytree(store / "policy" / tr["units"][k1]["entry"], leak)
    full = json.loads((a.dataset / "meta" / "stats.json").read_text())
    subprocess.run([str(a.lerobot_python), "-I", "-c",
                    "import json, sys, numpy as np\nfrom pathlib import Path\n"
                    "from safetensors.numpy import load_file, save_file\n"
                    "full = json.loads(Path(sys.argv[2]).read_text())\n"
                    "for f in Path(sys.argv[1]).glob('policy_*normalizer_processor.safetensors'):\n"
                    "    st = load_file(f)\n"
                    "    save_file({k: (np.asarray(full[k.rsplit('.', 1)[0]][k.rsplit('.', 1)[1]], v.dtype).reshape(v.shape)"
                    " if k.rsplit('.', 1)[1] in ('min', 'max') else v) for k, v in st.items()}, f)\n",
                    leak, a.dataset / "meta" / "stats.json"], check=True, env=env)
    contract = lerobot_diffusion.train_config_contract(spec["learner"], spec["seeds"][0], a.device)
    train_stats = json.loads((split / "train" / "meta" / "stats.json").read_text())
    clean = lerobot_policy.check_checkpoint(ctx, store / "policy" / tr["units"][k1]["entry"], contract, train_stats,
                                            split / "train")
    leaked = lerobot_policy.check_checkpoint(ctx, leak, contract, train_stats, split / "train")
    ev["normalization"] = {"selected_checkpoint_problems": len(clean),
                           "full_dataset_statistics_refused": any("training episodes'" in x for x in leaked),
                           "split_differs_from_full": any(
                               full[k]["min"] != train_stats[k]["min"] for k in ("observation.state",
                                                                                 "observation.environment_state"))}

    # 4. Resume: the same command reruns nothing; a second root on the store reuses every unit and rollout.
    p = ilsb("run", main_spec, "--root", root, "--store", store, "--gpus", slots)
    root2 = work / "root2"
    p2 = ilsb("run", main_spec, "--root", root2, "--store", store, "--gpus", slots)
    ev["reuse"] = {"rerun": last_json(p),
                   "second_root": last_json(p2),
                   "second_root_trained": sum(not u["reused"] for u in stage(root2, "train")["units"].values()),
                   "second_root_evaluated": sum(not u["reused"] for u in stage(root2, "evaluate")["units"].values()),
                   "same_report": {k: v for k, v in stage(root2, "report").items() if k != "protocol_sha256"}
                   == {k: v for k, v in rep.items() if k != "protocol_sha256"}}

    # 5. An interrupted unit: killed after its first checkpoint, resumed by the next run, bit for bit.
    one = write(work / "spec-one.json", {**spec, "seeds": spec["seeds"][:1], "expected": {**EXPECTED, "units": 1}})
    root3, store3 = work / "root-interrupted", work / "store-interrupted"
    p = ilsb("stage", "preflight", one, "--root", root3, "--store", store3)
    key = f"{lerobot_diffusion.KIND}/s{spec['seeds'][0]}"
    proc = subprocess.Popen([sys.executable, "-m", "ilsb", "unit", str(one), "--root", str(root3), "--store",
                             str(store3), "--key", key, "--gpu", "0" if a.device == "cuda" else "cpu",
                             *(["--allow-dirty"] if a.allow_dirty else [])],
                            env=env, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
    killed, deadline = False, time.time() + 900
    while proc.poll() is None and time.time() < deadline:
        last = [p_ for p_ in (root3 / "work").glob("*/work/train/checkpoints/last")]
        if last and os.readlink(last[0]) == "000100":
            os.killpg(proc.pid, signal.SIGKILL)
            killed = True
            break
        time.sleep(0.05)
    proc.wait()
    p = ilsb("run", one, "--root", root3, "--store", store3, "--gpus", slots[:1] if a.device == "cuda" else "cpu")
    r3 = stage(root3, "train")["units"][key] if (root3 / "stages" / "train.json").exists() else None
    res3 = json.loads((store3 / "policy" / r3["entry"] / "result.json").read_text()) if r3 else None
    ev["interrupted"] = {"killed_after_first_checkpoint": killed, "stages": last_json(p),
                         "resumed_from": res3["resumed_from"] if res3 else None,
                         "restarted_from": res3["restarted_from"] if res3 else None,
                         "same_checkpoints_bitwise": bool(res3) and [h["model_sha256"] for h in res3["checkpoints"]]
                         == ev["study"]["units"][key]["model_sha256"]}

    # 6. Refusals before anything trains.
    def refusal(name, sp, env_=None):
        r = ilsb("stage", "preflight", write(work / f"spec-{name}.json", sp), "--root", work / f"root-{name}",
                 "--store", work / f"store-{name}", env_=env_)
        return {"refused": r.returncode != 0 and "refused" in (r.stdout + r.stderr), "trained": (
            work / f"store-{name}" / "policy").exists()}
    overlap = json.loads(json.dumps(spec))
    overlap["roles"]["held_out"] = overlap["roles"]["held_out"] + overlap["roles"]["train"][:1]
    pin = json.loads(json.dumps(spec))
    pin["learner"]["lerobot"] = "0.6.0"
    bad = work / "dataset-altered"
    shutil.copytree(a.dataset, bad, ignore=shutil.ignore_patterns(".cache"))
    with open(bad / "meta" / "info.json", "a") as f:
        f.write("\n")
    ev["refusals"] = {"overlapping_roles": refusal("overlap", overlap), "lerobot_pin": refusal("pin", pin),
                      "altered_dataset": refusal("dataset", spec, {**env, "ILSB_LEROBOT_DATASET": str(bad)})}

    # 7. A store whose selected weights changed: the audit refuses.
    root4, store4 = work / "root-corrupt", work / "store-corrupt"
    shutil.copytree(root, root4)
    shutil.copytree(store, store4)
    w_ = store4 / "policy" / tr["units"][k1]["entry"] / "model.safetensors"
    b = bytearray(w_.read_bytes())
    b[-1] ^= 1
    w_.write_bytes(bytes(b))
    p = ilsb("stage", "audit", main_spec, "--root", root4, "--store", store4, "--gpus", slots.split(",")[0])
    au = stage(root4, "audit")
    ev["refusals"]["corrupted_weights"] = {"refused": p.returncode != 0 and not au["all_pass"],
                                           "units_flagged": sorted(au.get("problems", {}))}

    ok = (ev["study"]["pass"] and all(ev["contract"].values())
          and ev["normalization"]["selected_checkpoint_problems"] == 0
          and ev["normalization"]["full_dataset_statistics_refused"] and ev["normalization"]["split_differs_from_full"]
          and set(ev["reuse"]["rerun"].values()) == {"resumed"} and set(ev["reuse"]["second_root"].values()) == {"pass"}
          and ev["reuse"]["second_root_trained"] == 0 and ev["reuse"]["second_root_evaluated"] == 0
          and ev["reuse"]["same_report"]
          and ev["interrupted"]["killed_after_first_checkpoint"]
          and (ev["interrupted"]["resumed_from"], ev["interrupted"]["restarted_from"])
          == (([100], []) if a.device == "cuda" else ([], [100]))
          and ev["interrupted"]["same_checkpoints_bitwise"]
          and all(v["refused"] and not v.get("trained") for v in ev["refusals"].values())
          and ev["refusals"]["corrupted_weights"]["units_flagged"] == [k1])
    ev["pass"] = bool(ok)
    text = json.dumps(ev, indent=1, sort_keys=True) + "\n"
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(text)
    print(json.dumps({"pass": ev["pass"], "study": ev["study"]["pass"], "contract": ev["contract"],
                      "normalization": ev["normalization"], "reuse": {k: v for k, v in ev["reuse"].items()
                                                                      if k != "same_report"},
                      "same_report": ev["reuse"]["same_report"], "interrupted": ev["interrupted"],
                      "refusals": ev["refusals"]}))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
