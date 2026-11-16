"""#52 Stage A smoke: the construction and its audit on two training episodes outside the r = 0.25 subset.

    LEROBOT_PYTHON tools/pusht_recovery/smoke.py --source SOURCE.npz --work WORK_DIR \
        [--out docs/studies/evidence/pusht-recovery/smoke.json]

Outcome-free for the study: no subset episode is read, so it says nothing about Stage A's yield. The two episodes are
the first two of roles.train that are not in the subset. Their proposals are made here the way #43's were (the
demonstrated action plus Gaussian noise of 0.1 action standard deviations, two per transition; the oracle velocities
solved and accepted by oracle_stage_a.py's own solve and actual-history step, the previous slot decoded the same way).
The smoke then runs Stage A's construction twice (the window arrays and report must be byte-identical), audits it in
a fresh process, and requires the audit to refuse each injected fault for the right reason: an action outside the
bounds, a predecessor inside the T, a demonstrated slot from another episode, a held-out observation, a wrong loss
mask, source frame, selection, limit or window count, and an altered source.
The record holds counts and digests only.
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
from audit import audit  # noqa: E402
from oracle_stage_a import CONTROL_DT, actual_history, solve as oracle_solve  # noqa: E402

N_EPISODES, PER_TRANSITION, NOISE = 2, 2, 0.1


def episodes() -> list:
    sub = set(A.subset(A.BASE["roles"]["train"], A.SPEC["cell"]["r"], A.SPEC["cell"]["subset_seed"]))
    return sorted(e for e in A.BASE["roles"]["train"] if e not in sub)[:N_EPISODES]


def proposals(src: dict, eps: list) -> tuple:
    """#43's proposal and oracle construction on the smoke episodes; returns (population, oracle counts)."""
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    prev = np.concatenate([src["state"][:, 0:2], src["pose"][:, 0:3]], 1).astype(np.float64)
    act = src["action"].astype(np.float64)
    rows_all = np.flatnonzero(np.isin(src["episode"], eps))
    std = act[rows_all].std(0)
    rng = np.random.default_rng(np.random.SeedSequence([52, 1]))
    env, fenv = A.make_env(), A.make_env()
    out = {k: [] for k in ("candidate_row", "episode", "t", "action", "prev")}
    counts = {"proposals": 0, "source_not_verified": 0, "oracle_failed": 0, "oracle_accepted": 0}
    for e in eps:
        rows = np.flatnonzero(src["episode"] == e)
        used, err, states = A.replay(env, cur[rows], act[rows])
        for t in range(len(rows) - 1):
            for _ in range(PER_TRANSITION):
                a = act[rows[t]] + rng.normal(0.0, NOISE * std)
                counts["proposals"] += 1
                if err[t] > A.REPLAY_TOL or err[t + 1] > A.REPLAY_TOL:
                    counts["source_not_verified"] += 1
                    continue
                s = states[t]
                u0 = np.array([*s[1], *s[4], A.L * s[5]])
                st, u, _, _ = oracle_solve(fenv, cur[rows[t]], u0, a, cur[rows[t + 1]])
                if st != "ok" or A.distance(actual_history(env, used, act[rows], t, u, a), cur[rows[t + 1]])[0] > A.TOL:
                    counts["oracle_failed"] += 1
                    continue
                v0, v1 = np.array([*s[1], *s[4], s[5]]), np.array([u[0], u[1], u[2], u[3], u[4] / A.L])
                out["candidate_row"].append(counts["proposals"] - 1)
                out["episode"].append(e)
                out["t"].append(t)
                out["action"].append(a)
                out["prev"].append(prev[rows[t]] + (v0 - v1) * CONTROL_DT)
                counts["oracle_accepted"] += 1
    props = {k: np.array(v, np.float64 if k in ("action", "prev") else np.int64) for k, v in out.items()}
    return props, counts


def load_source(path: Path) -> dict:
    with np.load(path) as z:
        src = {k: z[k] for k in z.files}
    if A.source_module.content_sha256(src) != A.SPEC["inputs"]["source_content_sha256"]:
        sys.exit("refused: the source rows are not the protocol's by content")
    return src


def tamper(src, eps, props, arrays, record) -> dict:
    """Each injected fault on a copy of the run; the audit must name it."""
    rows_of = {e: np.flatnonzero(src["episode"] == e) for e in eps}
    held = np.flatnonzero(np.isin(src["episode"], A.BASE["roles"]["held_out"]))[0]
    sel = arrays["selected"]
    j, i = 0, int(sel[0])
    e, t = int(props["episode"][i]), int(props["t"][i])
    other = [x for x in eps if x != e][0]
    n = len(rows_of[e])
    k_max = min(A.H - 1, n - t)
    env = A.make_env()

    def rechain(a):
        z = a["proposal/z"][i]
        acts = [z[10:12], props["action"][i]] + [src["action"][rows_of[e][t + k - 1]].astype(np.float64)
                                                  for k in range(2, k_max + 1)]
        a["proposal/chain"][i][:k_max + 2] = A.chain(env, z, acts)
        a["window/action"][j][0] = z[10:12]

    faults = {}
    a = copy.deepcopy(arrays)
    a["proposal/z"][i][10] = 600.0
    rechain(a)
    faults["action_out_of_bounds"] = ("a supervised action is outside the action bounds", a)
    a = copy.deepcopy(arrays)
    zt = a["proposal/z"][i]
    a["proposal/z"][i][0:2] = A.pose.keypoints(np.array([[zt[2], zt[3], zt[4] / A.L]]))[0].reshape(8, 2)[0:4].mean(0)
    rechain(a)
    faults["predecessor_inside_t"] = ("penetration envelope", a)
    a = copy.deepcopy(arrays)
    a["window/action"][j][5] = src["action"][rows_of[other][10]]
    faults["slot_from_another_episode"] = ("a window action is not its slot's action", a)
    a = copy.deepcopy(arrays)
    a["window/observation"][j][1] = np.concatenate([src["state"][held, 2:4], src["env"][held, 16:32]]).astype(np.float32)
    faults["held_out_observation"] = ("equals a held-out row", a)
    a = copy.deepcopy(arrays)
    k = A.H - 1
    a["proposal/mask"][i][k] = a["window/loss_mask"][j][k] = not a["window/loss_mask"][j][k]
    faults["wrong_mask"] = ("the loss mask does not follow the rule", a)
    a = copy.deepcopy(arrays)
    a["window/slot_frame"][j][5] += 1
    faults["wrong_slot_frame"] = ("a slot's source frame is not the anchor's offset frame", a)
    a = copy.deepcopy(arrays)
    accepted = np.flatnonzero(a["proposal/status"] == 0)
    a["selected"][0] = [x for x in accepted if x not in set(sel.tolist())][0]
    faults["wrong_selection"] = ("the selection does not reproduce", a)
    out = {}
    records = {}
    r = copy.deepcopy(record)
    r["limits"]["action_change_px"] *= 2
    faults["wrong_limit"], records["wrong_limit"] = ("the record's limits are not the subset's", arrays), r
    r = copy.deepcopy(record)
    r["mass"]["n_synthetic"] -= 1
    faults["wrong_n_synthetic"], records["wrong_n_synthetic"] = ("the record's window counts", arrays), r
    for name, (want, a) in faults.items():
        problems = audit(src, eps, props, a, records.get(name, record))
        out[name] = {"refused": bool(problems), "named": any(want in p for p in problems), "problems": len(problems)}
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--work", type=Path, required=True)
    p.add_argument("--out", type=Path, default=A.ROOT / "docs/studies/evidence/pusht-recovery/smoke.json")
    p.add_argument("--workers", type=int, default=N_EPISODES)
    p.add_argument("--audit-run", type=Path, help=argparse.SUPPRESS)  # the fresh-process audit of a smoke run
    args = p.parse_args()
    A.simulator_check()
    src = load_source(args.source)
    eps = episodes()
    if args.audit_run:
        with np.load(args.audit_run / "props.npz") as z:
            props = {k: z[k] for k in z.files}
        with np.load(args.audit_run / "windows.npz") as z:
            arrays = {k: z[k] for k in z.files}
        problems = audit(src, eps, props, arrays, json.loads((args.audit_run / "report.json").read_text()))
        print(json.dumps({"problems": problems}))
        sys.exit(2 if problems else 0)
    if set(eps) & set(A.subset(A.BASE["roles"]["train"], A.SPEC["cell"]["r"], A.SPEC["cell"]["subset_seed"])):
        sys.exit("refused: a smoke episode is in the subset")
    props, oracle_counts = proposals(src, eps)
    runs = []
    for name in ("a", "b"):
        d = args.work / name
        rpt = A.run(src, eps, props, args.workers, d, smoke=True)
        np.savez(d / "props.npz", **props)
        (d / "report.json").write_text(json.dumps(rpt, indent=1, sort_keys=True) + "\n")
        runs.append(rpt)
    repeat = runs[0] == runs[1]
    fresh = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--source", str(args.source),
                            "--work", str(args.work), "--audit-run", str(args.work / "a")], capture_output=True, text=True)
    fresh_problems = json.loads(fresh.stdout.strip().splitlines()[-1])["problems"] if fresh.stdout.strip() else ["no output"]
    with np.load(args.work / "a" / "windows.npz") as z:
        arrays = {k: z[k] for k in z.files}
    faults = tamper(src, eps, props, arrays, runs[0])
    bad = args.work / "altered-source.npz"
    alt = dict(src)
    alt["action"] = src["action"].copy()
    alt["action"][0, 0] += 1.0
    np.savez(bad, **alt)
    refused = subprocess.run([sys.executable, "-I", str(HERE / "stage_a.py"), "--source", str(bad), "--candidates",
                              str(bad), "--proposals", str(bad), "--run", str(args.work / "never")],
                             capture_output=True, text=True)
    faults["altered_source"] = {"refused": refused.returncode != 0 and not (args.work / "never").exists(),
                                "named": "the source rows are not the protocol's" in refused.stderr + refused.stdout,
                                "problems": 1}
    ok = (repeat and fresh.returncode == 0 and not fresh_problems and runs[0]["proposals"]["anchors"] > 0
          and all(f["refused"] and f["named"] for f in faults.values()))
    out = {"schema": "ilsb.pusht_recovery.smoke 1", "code": A.code_identity(), "episodes": eps,
           "oracle": oracle_counts, "construction": {k: runs[0][k] for k in ("proposals", "mass")},
           "windows_content_sha256": runs[0]["windows_content_sha256"],
           "windows_npz_content_sha256": runs[0]["windows_npz_content_sha256"],
           "byte_identical_rerun": repeat, "fresh_process_audit": {"exit": fresh.returncode, "problems": fresh_problems},
           "faults": faults, "pass": ok}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"pass": ok, "oracle": oracle_counts, "counts": runs[0]["proposals"]["counts"],
                      "anchors": runs[0]["proposals"]["anchors"], "faults": faults, "repeat": repeat}))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
