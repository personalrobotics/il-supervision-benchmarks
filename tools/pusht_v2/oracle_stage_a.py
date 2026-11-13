"""#43 Stage A: latent-state-assisted oracle labels for the r = 0.25 proposals, no training (pusht-oracle.md).

    LEROBOT_PYTHON tools/pusht_v2/oracle_stage_a.py --source SOURCE.npz --candidates CANDIDATES.npz \
        --selection SELECTION.npz --labels LABELS.npz --run RUN_DIR [--out docs/studies/evidence/pusht-oracle/stage-a.json]

Every input is checked against oracle.json's identities. For each proposal (source transition t -> t + 1, action a):
  source   the demonstration's replayed simulator state at t, usable when frames t and t + 1 replay within tolerance
  solve    Newton in the fresh operator for the velocities u at the source positions (oracle.json "solve")
  accept   the actual-history step: the continuous replay to t, u substituted, one step under a, within tolerance
  decode   previous slot q_{t-1} + (v_t - v') * 0.1, current slot the source's own; validity checks
RUN_DIR/proposals.npz holds every proposal's status and arrays; the report holds the identities, the counts by
reason, the diagnostics, the common pool and Stage A's rules. No policy is trained or read.
"""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose  # noqa: E402
import source as source_module  # noqa: E402
from oracle_probe import distance, extent, full_state, load_fresh, make_env, penetration, positions, q, subset  # noqa: E402,E501

SPEC = json.loads((HERE / "oracle.json").read_text())
BASE = json.loads((HERE / "surface.json").read_text())
SOLVE, VALID = SPEC["solve"], SPEC["validity"]
L, TOL, REPLAY_TOL = SOLVE["l_px"], SOLVE["tolerance_px"], SPEC["replay"]["tolerance_px"]
STATUS = ("ok", *VALID["reasons"])
AGENT_R, ENV = VALID["agent_radius"], VALID["envelope"]
CONTROL_DT = 0.1
NEW = [2, 3, 7, 8, 9]  # the current slot in CCIL's pose history [agent prev 2, agent cur 2, T prev 3, T cur 3]


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def need(path: Path, digest: str) -> Path:
    if sha(path) != digest:
        sys.exit(f"refused: {path} does not match oracle.json ({digest[:12]})")
    return path


def content_sha(arrays: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(arrays):
        a = np.ascontiguousarray(arrays[k])
        h.update(f"{k}|{a.dtype.str}|{a.shape}|".encode() + a.tobytes())
    return h.hexdigest()


def r_of(p: np.ndarray, target: np.ndarray) -> np.ndarray:
    """The frozen residual (agent dx, dy, T dx, dy, l * wrap(dtheta)) in px; wrap maps to (-pi, pi]."""
    r = p - target
    w = np.arctan2(np.sin(r[4]), np.cos(r[4]))
    r[4] = L * (np.pi if w == -np.pi else w)
    return r


def state_of(qt: np.ndarray, u: np.ndarray) -> tuple:
    return (tuple(qt[0:2]), (u[0], u[1]), tuple(qt[2:4]), float(qt[4]), (u[2], u[3]), u[4] / L)


def fresh_step(env, qt, u, a) -> np.ndarray:
    load_fresh(env, state_of(qt, u))
    env.step(a)
    return positions(env)


def solve(env, qt, u0, a, target) -> tuple:
    """(status, u, fresh distance, iterations): Newton on r in u from the demonstration's u0."""
    u = u0.copy()
    p = fresh_step(env, qt, u, a)
    r = r_of(p, target)
    for it in range(SOLVE["max_iterations"] + 1):
        if not (np.isfinite(u).all() and np.isfinite(r).all()):
            return "non_finite", u, np.nan, it
        d = float(distance(p, target)[0])
        if d <= TOL:
            return "ok", u, d, it
        if it == SOLVE["max_iterations"]:
            break
        h, J = SOLVE["fd_step"], np.empty((5, 5))
        for k in range(5):
            e = np.zeros(5)
            e[k] = h
            J[:, k] = (r_of(fresh_step(env, qt, u + e, a), target) - r_of(fresh_step(env, qt, u - e, a), target)) / (2 * h)
        if not np.isfinite(J).all() or np.linalg.cond(J) > 1e10:
            return "singular_jacobian", u, d, it
        du, step = np.linalg.solve(J, -r), 1.0
        for _ in range(SOLVE_HALVINGS + 1):
            un = u + step * du
            pn = fresh_step(env, qt, un, a)
            rn = r_of(pn, target)
            if np.isfinite(rn).all() and np.linalg.norm(rn) < np.linalg.norm(r):
                u, p, r = un, pn, rn
                break
            step /= 2
        else:
            return "solve_failed", u, d, it
    return "solve_failed", u, float(distance(p, target)[0]), SOLVE["max_iterations"]


SOLVE_HALVINGS = int(SOLVE["line_search"].split("up to ")[1].split(" ")[0])  # oracle.json: "halve up to 8 times ..."


def replay_reset(env, cur0: np.ndarray) -> np.ndarray:
    """The reset vector whose body pose equals the recorded first frame (oracle.json "replay")."""
    x0 = cur0.copy()
    for _ in range(5):
        used = x0.copy()
        env.reset(options={"reset_to_state": used})
        x0[2:5] += cur0[2:5] - positions(env)[2:5]
    return used


def actual_history(env, used: np.ndarray, act: np.ndarray, t: int, u: np.ndarray, a: np.ndarray) -> np.ndarray:
    env.reset(options={"reset_to_state": used})
    for k in range(t):
        env.step(act[k])
    env.agent.velocity, env.block.velocity, env.block.angular_velocity = (u[0], u[1]), (u[2], u[3]), u[4] / L
    env.step(a)
    return positions(env)


def beyond_penetration(agent: np.ndarray, kp: np.ndarray) -> np.ndarray:
    return penetration(agent, kp, AGENT_R) > ENV["max_penetration_px"]


def beyond_extent(agent: np.ndarray, kp: np.ndarray) -> np.ndarray:
    lo, hi = extent(agent, kp, AGENT_R)
    return (lo < ENV["extent_px"][0]) | (hi > ENV["extent_px"][1])


def nearest(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.empty(len(x))
    rn = (ref ** 2).sum(1)
    for i in range(0, len(x), 2048):
        b = x[i:i + 2048]
        d2 = (b ** 2).sum(1)[:, None] + rn[None] - 2 * b @ ref.T
        out[i:i + 2048] = np.sqrt(np.maximum(d2.min(1), 0))
    return out


def main():
    p = argparse.ArgumentParser()
    for k in ("source", "candidates", "selection", "labels", "run"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--out", type=Path, default=HERE.parents[1] / "docs/studies/evidence/pusht-oracle/stage-a.json")
    p.add_argument("--limit-episodes", type=int, default=None, help="smoke only: the first N subset episodes")
    args = p.parse_args()
    ids = SPEC["proposals"]
    cand = np.load(need(args.candidates, ids["candidates_sha256"]))
    sel = np.load(need(args.selection, ids["selection_sha256"]))
    lab = np.load(need(args.labels, ids["labels_npz_sha256"]))
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if source_module.content_sha256(src) != BASE["data"]["content_sha256"]:
        sys.exit("refused: the source rows are not the protocol's by content")
    eps = subset(BASE["roles"]["train"], SPEC["cell"]["r"], SPEC["cell"]["subset_seed"])
    if len(eps) != BASE["expected"]["subset_episodes"]["r0.25"] or len(cand["actions"]) != ids["proposals"]:
        sys.exit("refused: the subset or the proposal count is not the protocol's")
    key = {(int(e), int(f)): i for i, (e, f) in enumerate(zip(src["episode"], src["frame"]))}
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    prev = np.concatenate([src["state"][:, 0:2], src["pose"][:, 0:3]], 1).astype(np.float64)
    c_ep = cand["source_episode_id"].astype(np.int64)
    c_fr = cand["source_frame"].astype(np.int64)
    z = np.concatenate([cand["obs/state"], cand["obs/state.pose"]], 1).astype(np.float64)
    tgt = np.concatenate([cand["target/state"], cand["target/state.pose"]], 1).astype(np.float64)
    src_row = np.array([key[(e, f)] for e, f in zip(c_ep, c_fr)], np.int64)
    succ_row = np.array([key[(e, f + 1)] for e, f in zip(c_ep, c_fr)], np.int64)
    if (np.abs(z[:, NEW] - cur[src_row]).max() > 1e-3 or np.abs(tgt[:, NEW] - cur[succ_row]).max() > 1e-3):
        sys.exit("refused: the candidates' current and target slots are not the source frame and its successor")
    learned = np.full(len(c_ep), -1, np.int64)
    learned[lab["candidate_row"]] = np.arange(len(lab["candidate_row"]))
    run_eps = eps[:args.limit_episodes] if args.limit_episodes else eps
    n = len(c_ep)
    status = np.full(n, -1, np.int8)
    fresh_d, actual_d, iters = np.full(n, np.nan), np.full(n, np.nan), np.full(n, -1, np.int16)
    u_out, prev_oracle = np.full((n, 5), np.nan), np.full((n, 5), np.nan)
    env, fenv = make_env(), make_env()
    replayed = {}
    for e in run_eps:
        rows = np.flatnonzero(src["episode"] == e)
        act = src["action"][rows].astype(np.float64)
        used = replay_reset(env, cur[rows[0]])
        err, states = [float(distance(positions(env), cur[rows[0]])[0])], [full_state(env)]
        for t in range(len(rows) - 1):
            env.step(act[t])
            err.append(float(distance(positions(env), cur[rows[t + 1]])[0]))
            states.append(full_state(env))
        replayed[e] = (rows, act, used, np.array(err), states)
        for i in np.flatnonzero(c_ep == e):
            t = int(c_fr[i])
            if replayed[e][3][t] > REPLAY_TOL or replayed[e][3][t + 1] > REPLAY_TOL:
                status[i] = STATUS.index("source_not_verified")
                continue
            s = states[t]
            u0 = np.array([*s[1], *s[4], L * s[5]])
            qt, a, target = cur[rows[t]], cand["actions"][i].astype(np.float64), cur[rows[t + 1]]
            st, u, d, it = solve(fenv, qt, u0, a, target)
            fresh_d[i], iters[i], u_out[i] = d, it, u
            if st != "ok":
                status[i] = STATUS.index(st)
                continue
            ad = float(distance(actual_history(env, used, act, t, u, a), target)[0])
            actual_d[i] = ad
            if not np.isfinite(ad) or ad > TOL:
                status[i] = STATUS.index("fails_actual_history")
                continue
            v0 = np.array([*s[1], *s[4], s[5]])
            v1 = np.array([u[0], u[1], u[2], u[3], u[4] / L])
            prev_oracle[i] = prev[rows[t]] + (v0 - v1) * CONTROL_DT
            status[i] = STATUS.index("ok")
    done = status >= 0
    ok0 = done & (status == 0)
    kp = pose.keypoints(prev_oracle[ok0][:, 2:5]) if ok0.any() else np.zeros((0, 16))
    bad_finite = ~np.isfinite(prev_oracle[ok0]).all(1)
    bad_pen = beyond_penetration(prev_oracle[ok0][:, 0:2], kp)
    bad_ext = beyond_extent(prev_oracle[ok0][:, 0:2], kp)
    idx = np.flatnonzero(ok0)
    for j, i in enumerate(idx):  # first failing check, in the frozen order
        for name, bad in (("non_finite", bad_finite), ("beyond_penetration_envelope", bad_pen),
                          ("beyond_extent_envelope", bad_ext)):
            if bad[j]:
                status[i] = STATUS.index(name)
                break
    # the invariant holds by construction (the current slot is copied from the source row); checked bitwise
    exact = done & (status == 0)
    common = exact & (learned >= 0)
    # diagnostics on the common rows
    cr = np.flatnonzero(common)
    src_prev = prev[src_row[cr]]
    lz = z[cr]
    learned_prev = np.concatenate([lz[:, 0:2], lz[:, 4:7]], 1)
    disp_oracle = distance(prev_oracle[cr], src_prev) if len(cr) else np.zeros(0)
    disp_learned = distance(learned_prev, src_prev) if len(cr) else np.zeros(0)
    disagree = distance(prev_oracle[cr], learned_prev) if len(cr) else np.zeros(0)
    obs = lambda pv, cv: np.concatenate([pv[:, 0:2], cv[:, 0:2], pose.keypoints(pv[:, 2:5]),  # noqa: E731
                                         pose.keypoints(cv[:, 2:5])], 1)
    sub_rows = np.flatnonzero(np.isin(src["episode"], eps))
    demo_obs = obs(prev[sub_rows], cur[sub_rows])
    near_oracle = nearest(obs(prev_oracle[cr], cur[src_row[cr]]), demo_obs) if len(cr) else np.zeros(0)
    near_learned = nearest(obs(learned_prev, cur[src_row[cr]]), demo_obs) if len(cr) else np.zeros(0)
    demo_kp = pose.keypoints(prev[sub_rows][:, 2:5])
    s = SPEC["stage_a"]["scale_px"]
    n_common = int(common.sum())
    med_disp = float(np.median(disp_oracle)) if n_common else None
    med_dis = float(np.median(disagree)) if n_common else None
    if n_common < SPEC["stage_a"]["min_common_rows"]:
        decision = "fail: fewer common rows than one synthetic batch"
    elif med_disp <= s:
        decision = "fail: the oracle labels are not novel"
    elif med_dis <= s:
        decision = "stop: oracle and learned labels are indistinguishable at data scale; no training"
    else:
        decision = "pass: Stage B may run after separate approval of its paid jobs"
    arrays = {"candidate_row": np.arange(n), "src_row": src_row, "status": status, "fresh_px": fresh_d,
              "actual_px": actual_d, "iterations": iters, "u": u_out, "prev_oracle": prev_oracle,
              "learned_row": learned, "common": common}
    args.run.mkdir(parents=True, exist_ok=True)
    np.savez(args.run / "proposals.npz", **arrays)
    counts = {k: int((status[done] == STATUS.index(k)).sum()) for k in STATUS}
    import importlib.metadata as md
    git = lambda *a: subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True,  # noqa: E731
                                    check=True).stdout.strip()
    code = {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "files_sha256": {f: sha(HERE / f) for f in ("oracle_stage_a.py", "oracle_probe.py", "pose.py",
                                                        "source.py", "oracle.json", "surface.json")}}
    pen_o = penetration(prev_oracle[cr][:, 0:2], pose.keypoints(prev_oracle[cr][:, 2:5]), AGENT_R)
    pen_l = penetration(learned_prev[:, 0:2], pose.keypoints(learned_prev[:, 2:5]), AGENT_R)
    ext_o, ext_l = beyond_extent(prev_oracle[cr][:, 0:2], pose.keypoints(prev_oracle[cr][:, 2:5])), \
        beyond_extent(learned_prev[:, 0:2], pose.keypoints(learned_prev[:, 2:5]))
    report = {
        "code": code,
        "schema": "ilsb.pusht_oracle.stage_a 1", "smoke": bool(args.limit_episodes),
        "identities": {"oracle_json_sha256": sha(HERE / "oracle.json"), "probe_sha256": SPEC["probe_sha256"],
                       "source_content_sha256": BASE["data"]["content_sha256"], **ids,
                       "simulator": {"gym_pusht": md.version("gym-pusht"), "pymunk": md.version("pymunk")},
                       "proposals_npz_content_sha256": content_sha(arrays)},
        "proposals": {"total": n, "attempted": int(done.sum()), "counts": counts, "exact": int(exact.sum()),
                      "learned_labels": int((learned >= 0).sum()), "exact_without_learned": int((exact & (learned < 0)).sum()),
                      "common": n_common},
        "diagnostics": {"fresh_residual_px": q(fresh_d[exact]), "actual_residual_px": q(actual_d[exact]),
                        "iterations": q(iters[exact]),
                        "converged_fresh_but_failed_actual": counts["fails_actual_history"],
                        "displacement_from_source_px": {"oracle": q(disp_oracle), "learned": q(disp_learned)},
                        "oracle_learned_disagreement_px": q(disagree), "action_disagreement": 0.0,
                        "nearest_demonstration_px": {"oracle": q(near_oracle), "learned": q(near_learned)},
                        "demonstrations_beyond_envelope": {
                            "penetration": int(beyond_penetration(prev[sub_rows][:, 0:2], demo_kp).sum()),
                            "extent": int(beyond_extent(prev[sub_rows][:, 0:2], demo_kp).sum()),
                            "rows": int(len(sub_rows))},
                        "penetration_px": {"oracle": q(pen_o), "learned": q(pen_l)},
                        "common_beyond_envelope": {
                            "oracle": {"penetration": int((pen_o > ENV["max_penetration_px"]).sum()),
                                       "extent": int(ext_o.sum())},
                            "learned": {"penetration": int((pen_l > ENV["max_penetration_px"]).sum()),
                                        "extent": int(ext_l.sum())},
                            "rows": int(len(cr))}},
        "mass": {"alpha": SPEC["cell"]["alpha"], "synthetic_rows_per_batch": 64,
                 "passes_over_common_in_20k_steps": (20000 * 64 / n_common) if n_common else None},
        "rules": {"scale_px": s, "min_common_rows": SPEC["stage_a"]["min_common_rows"],
                  "median_oracle_displacement_px": med_disp, "median_oracle_learned_disagreement_px": med_dis},
        "decision": decision}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"proposals": report["proposals"], "rules": report["rules"], "decision": decision}))


if __name__ == "__main__":
    main()
