"""#52 Stage A: physically consistent recovery windows from the oracle one-step labels, no training (pusht-recovery.md).

    LEROBOT_PYTHON tools/pusht_recovery/stage_a.py --source SOURCE.npz --candidates CANDIDATES.npz \
        --proposals PROPOSALS.npz --run RUN_DIR [--out docs/studies/evidence/pusht-recovery/stage-a.json]

PROPOSALS.npz is #43's oracle Stage A output; its accepted proposals (status ok) are the population. For each one at
source frame t of episode e (proposal action a~, the oracle's decoded previous slot), stage-a.json "solve" finds a
predecessor state P and a preceding action b whose simulated chain P -b-> frame t -a~-> frame t + 1 lands on the
recorded frames, then continues with the demonstrated actions; the window's observations, actions, loss mask and
provenance come from that one chain. RUN_DIR/windows.npz holds every proposal's status and the accepted windows; the
report holds the identities, counts by reason, diagnostics and the yield decision. No policy is trained or read.
"""
import argparse
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools" / "pusht_v2"))
import pose  # noqa: E402
import source as source_module  # noqa: E402
from oracle_probe import distance, extent, full_state, load_fresh, make_env, penetration, positions, q, subset  # noqa: E402,E501
from oracle_stage_a import content_sha, nearest, r_of, replay_reset, sha  # noqa: E402

SPEC = json.loads((HERE / "stage-a.json").read_text())
BASE = json.loads((ROOT / "tools/pusht_v2/surface.json").read_text())
ORACLE = json.loads((ROOT / "tools/pusht_v2/oracle.json").read_text())
SOLVE, MASK, VALID, WIN = SPEC["solve"], SPEC["mask"], SPEC["validity"], SPEC["window"]
L, TOL = ORACLE["solve"]["l_px"], SOLVE["tolerance_px"]
REPLAY_TOL = ORACLE["replay"]["tolerance_px"]
ENV, AGENT_R = ORACLE["validity"]["envelope"], ORACLE["validity"]["agent_radius"]
DT = 0.1
H, N_OBS, N_ACT, DROP = WIN["horizon"], WIN["n_obs_steps"], WIN["n_action_steps"], WIN["drop_n_last_frames"]
STATUS = ("ok", *VALID["reasons"])
LO, HI = VALID["action_bounds"]


def temporal():
    """ilsb.temporal by path: the harness's window contract, numpy only, so it runs in the LeRobot environment."""
    spec = importlib.util.spec_from_file_location("ilsb_temporal", ROOT / "src/ilsb/temporal.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["ilsb_temporal"] = mod  # dataclasses resolves the module by name
    spec.loader.exec_module(mod)
    return mod


def state_of(z: np.ndarray) -> tuple:
    """The fresh state P of the scaled unknowns z (stage-a.json "solve")."""
    return ((z[0], z[1]), (z[5] / DT, z[6] / DT), (z[2], z[3]), z[4] / L, (z[7] / DT, z[8] / DT), z[9] / (L * DT))


def z_of(s: tuple) -> np.ndarray:
    return np.array([*s[0], *s[2], L * s[3], s[1][0] * DT, s[1][1] * DT, s[4][0] * DT, s[4][1] * DT, L * s[5] * DT])


def chain(env, z: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Positions of P, then after each action in turn, in one live space started fresh at P."""
    load_fresh(env, state_of(z))
    out = [positions(env)]
    for a in actions:
        env.step(a)
        out.append(positions(env))
    return np.array(out)


def residual(env, z, a_corr, q_t, q_t1):
    p = chain(env, z, [z[10:12], a_corr])
    return np.concatenate([r_of(p[1], q_t), r_of(p[2], q_t1)]), p


def solve(env, z0, a_corr, q_t, q_t1):
    """(status, z, iterations): Gauss-Newton with minimum-norm steps on the 10 residuals in the 12 unknowns."""
    z = z0.copy()
    r, p = residual(env, z, a_corr, q_t, q_t1)
    halvings, h = SOLVE["line_search"], SOLVE["fd_step"]
    for it in range(SOLVE["max_iterations"] + 1):
        if not (np.isfinite(z).all() and np.isfinite(r).all()):
            return "non_finite", z, it
        if distance(p[1], q_t)[0] <= TOL and distance(p[2], q_t1)[0] <= TOL:
            return "ok", z, it
        if it == SOLVE["max_iterations"]:
            break
        J = np.empty((10, 12))
        for k in range(12):
            e = np.zeros(12)
            e[k] = h
            J[:, k] = (residual(env, z + e, a_corr, q_t, q_t1)[0] - residual(env, z - e, a_corr, q_t, q_t1)[0]) / (2 * h)
        if not np.isfinite(J).all() or not np.any(J):
            return "singular_jacobian", z, it
        dz, step = np.linalg.lstsq(J, -r, rcond=SOLVE["rcond"])[0], 1.0
        for _ in range(halvings + 1):
            zn = z + step * dz
            rn, pn = residual(env, zn, a_corr, q_t, q_t1)
            if np.isfinite(rn).all() and np.linalg.norm(rn) < np.linalg.norm(r):
                z, r, p = zn, rn, pn
                break
            step /= 2
        else:
            return "solve_failed", z, it
    return "solve_failed", z, SOLVE["max_iterations"]


def replay(env, cur: np.ndarray, act: np.ndarray):
    """The episode replayed from its first source row: replay error per row and the full state at each row."""
    used = replay_reset(env, cur[0])
    err, states = [float(distance(positions(env), cur[0])[0])], [full_state(env)]
    for t in range(len(cur) - 1):
        env.step(act[t])
        err.append(float(distance(positions(env), cur[t + 1])[0]))
        states.append(full_state(env))
    return used, np.array(err), states


def speeds(s: tuple) -> np.ndarray:
    return np.array([np.hypot(*s[1]), np.hypot(*s[4]), L * abs(s[5])])


def envelope_bad(p: np.ndarray) -> tuple:
    """(penetration, extent) beyond oracle.json's envelope, for (N, 5) positions."""
    kp = pose.keypoints(p[:, 2:5])
    lo, hi = extent(p[:, 0:2], kp, AGENT_R)
    return (bool((penetration(p[:, 0:2], kp, AGENT_R) > ENV["max_penetration_px"]).any()),
            bool(((lo < ENV["extent_px"][0]) | (hi > ENV["extent_px"][1])).any()))


def episode_speeds(job):
    """Pass 1: the largest speeds over an episode's replay-verified source states."""
    cur, act = job
    _, err, states = replay(make_env(), cur, act)
    ok = [s for s, e in zip(states, err) if e <= REPLAY_TOL]
    return np.max([speeds(s) for s in ok], 0) if ok else np.zeros(3)


def construct(job):
    """Pass 2: every proposal of one episode -> (status, z, iterations, chain positions, deviations, mask)."""
    cur, act, props, limits = job
    env, fenv = make_env(), make_env()
    _, _, states = replay(env, cur, act)
    n = len(cur)
    out = []
    for t, a_corr, prev_oracle in props:
        rec = {"status": None, "z": np.full(12, np.nan), "iterations": -1, "chain": np.full((H + 1, 5), np.nan),
               "dev": np.full(H, np.nan), "mask": np.zeros(H, bool)}
        out.append(rec)
        if t < 1:
            rec["status"] = "no_predecessor"
            continue
        if t > n - DROP - 1:  # LeRobot's anchors: frame t + 1 <= T - 1 - drop, with T = n + 1 dataset frames
            rec["status"] = "anchor_in_dropped_frames"
            continue
        s_prev = states[t - 1]
        z0 = np.concatenate([[*prev_oracle[0:4], L * prev_oracle[4]], z_of(s_prev)[5:10], act[t - 1]])
        st, z, it = solve(fenv, z0, a_corr, cur[t], cur[t + 1])
        rec["z"], rec["iterations"] = z, it
        if st != "ok":
            rec["status"] = st
            continue
        b = z[10:12]
        k_max = min(H - 1, n - t)  # slot k >= 2 applies the action of row t + k - 1
        acts = [b, a_corr] + [act[t + k - 1] for k in range(2, k_max + 1)]
        p = chain(fenv, z, acts)
        rec["chain"][:len(p)] = p
        # dev[j]: the chain at frame t + j (state p[1 + j]) against the recorded row t + j
        for j in range(0, min(len(p) - 1, n - t)):
            rec["dev"][j] = float(distance(p[1 + j], cur[t + j])[0])
        mask = np.zeros(H, bool)
        mask[0] = mask[1] = True
        for k in range(2, k_max + 1):
            if np.all(rec["dev"][1:k] <= MASK["continuation_px"]):
                mask[k] = True
            else:
                break
        rec["mask"] = mask
        supervised = np.array(acts)[mask[:len(acts)]]
        if not np.isfinite(p).all():
            rec["status"] = "non_finite"
        elif not ((supervised >= LO) & (supervised <= HI)).all():
            rec["status"] = "action_out_of_bounds"
        elif np.linalg.norm(b - act[t - 1]) > limits["action_change"]:
            rec["status"] = "b_beyond_action_change"
        elif (speeds(state_of(z)) > limits["speed"]).any():
            rec["status"] = "speed_beyond_envelope"
        else:
            last = int(np.flatnonzero(mask)[-1])
            pen, ext = envelope_bad(p[:last + 1])
            if pen:
                rec["status"] = "beyond_penetration_envelope"
            elif ext:
                rec["status"] = "beyond_extent_envelope"
            elif not mask[MASK["required_slots"]].all():
                rec["status"] = "short_continuation"
            else:
                rec["status"] = "ok"
    return out


def limits_of(src: dict, eps: list, mapper=map) -> dict:
    """stage-a.json validity: the action-change limit and the speed envelope, from the subset's source rows."""
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    act_all = src["action"].astype(np.float64)
    rows_of = {e: np.flatnonzero(src["episode"] == e) for e in eps}
    da = np.concatenate([np.linalg.norm(np.diff(act_all[rows_of[e]], axis=0), axis=1) for e in eps])
    sp = list(mapper(episode_speeds, [(cur[rows_of[e]], act_all[rows_of[e]]) for e in eps]))
    return {"action_change": float(np.percentile(da, VALID["action_change_percentile"])), "speed": np.max(sp, 0)}


def real_windows(src: dict, eps: list) -> tuple:
    """(n_real, n_synthetic): the subset's real Diffusion Policy windows by ilsb.temporal, and Stage B's need."""
    mod = temporal()
    spec = mod.WindowSpec(N_OBS, H, N_ACT, DROP, False)
    lengths = [int((src["episode"] == e).sum()) + 1 for e in eps]  # dataset frames: the rows plus the dropped frame 0
    ep_col = np.concatenate([np.full(n, e) for e, n in zip(eps, lengths)])
    fr_col = np.concatenate([np.arange(n) for n in lengths])
    n_real = int(len(mod.windows(ep_col, fr_col, spec)["anchor"]))
    alpha = SPEC["cell"]["alpha"]
    return n_real, int(np.floor(alpha / (1 - alpha) * n_real + 0.5))


def build(src: dict, eps: list, props: dict, workers: int) -> dict:
    """Every proposal's record and the per-proposal window arrays; props: candidate_row, episode, t, action, prev."""
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    act_all = src["action"].astype(np.float64)
    rows_of = {e: np.flatnonzero(src["episode"] == e) for e in eps}
    sub = np.concatenate([rows_of[e] for e in eps])
    with ProcessPoolExecutor(workers) as pool:
        limits = limits_of(src, eps, pool.map)
        jobs, order = [], []
        for e in eps:
            sel = np.flatnonzero(props["episode"] == e)
            order.append(sel)
            jobs.append((cur[rows_of[e]], act_all[rows_of[e]],
                         [(int(props["t"][i]), props["action"][i].astype(np.float64), props["prev"][i].astype(np.float64))
                          for i in sel], limits))
        results = list(pool.map(construct, jobs))
    n = len(props["candidate_row"])
    out = {"status": np.full(n, -1, np.int8), "z": np.full((n, 12), np.nan), "iterations": np.full(n, -1, np.int16),
           "chain": np.full((n, H + 1, 5), np.nan), "dev": np.full((n, H), np.nan), "mask": np.zeros((n, H), bool)}
    for sel, res in zip(order, results):
        for i, rec in zip(sel, res):
            out["status"][i] = STATUS.index(rec["status"])
            for k in ("z", "iterations", "chain", "dev", "mask"):
                out[k][i] = rec[k]
    return out, limits, sub, rows_of


def windows(src: dict, props: dict, rec: dict, rows_of: dict, accepted: np.ndarray) -> dict:
    """The accepted proposals' windows: float32 observations and actions as LeRobot would hand them to the policy."""
    obs, actions, pad, slot_frame, episode, anchor = [], [], [], [], [], []
    for i in accepted:
        e, t = int(props["episode"][i]), int(props["t"][i])
        rows = rows_of[e]
        n = len(rows)
        z = rec["z"][i]
        kp_prev = pose.keypoints(np.array([[z[2], z[3], z[4] / L]]))[0]
        o_prev = np.concatenate([z[0:2], kp_prev]).astype(np.float32)
        o_t = np.concatenate([src["state"][rows[t], 2:4], src["env"][rows[t], 16:32]]).astype(np.float32)
        a = np.empty((H, 2), np.float32)
        a[0], a[1] = z[10:12], props["action"][i]
        fr = np.full(H, -1, np.int64)
        fr[1] = -2
        pd = np.zeros(H, bool)
        for k in range(2, H):
            j = t + k - 1
            pd[k] = j > n - 1
            a[k] = src["action"][rows[min(j, n - 1)]]
            fr[k] = min(j, n - 1) + 1  # dataset frame of row j
        obs.append([o_prev, o_t])
        actions.append(a)
        pad.append(pd)
        slot_frame.append(fr)
        episode.append(e)
        anchor.append(t + 1)
    return {"observation": np.array(obs, np.float32).reshape(-1, N_OBS, 18),
            "action": np.array(actions, np.float32).reshape(-1, H, 2),
            "loss_mask": rec["mask"][accepted], "action_pad": np.array(pad, bool).reshape(-1, H),
            "slot_frame": np.array(slot_frame, np.int64).reshape(-1, H),
            "episode": np.array(episode, np.int64), "anchor_frame": np.array(anchor, np.int64),
            "candidate_row": props["candidate_row"][accepted].astype(np.int64)}


def select(props: dict, accepted: np.ndarray, src_row: np.ndarray) -> np.ndarray:
    """stage-a.json "selection": one accepted proposal per source anchor."""
    chosen = []
    for r in np.unique(src_row[accepted]):
        cand = np.sort(accepted[src_row[accepted] == r])
        k = int(np.random.default_rng(np.random.SeedSequence([52, int(r)])).integers(len(cand)))
        chosen.append(cand[k])
    return np.array(sorted(chosen), np.int64)


def report(src, eps, props, rec, limits, sub, rows_of, accepted, chosen, w, smoke: bool) -> dict:
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    prev = np.concatenate([src["state"][:, 0:2], src["pose"][:, 0:3]], 1).astype(np.float64)
    n_real, n_syn = real_windows(src, eps)
    done = rec["status"] >= 0
    counts = {k: int((rec["status"][done] == STATUS.index(k)).sum()) for k in STATUS}
    zc = rec["z"][chosen]
    p_prev = zc[:, [0, 1, 2, 3, 4]].copy()
    p_prev[:, 4] /= L
    src_rows = np.array([rows_of[int(props["episode"][i])][int(props["t"][i])] for i in chosen], np.int64)
    disp_source = distance(p_prev, prev[src_rows]) if len(chosen) else np.zeros(0)
    disp_decode = distance(p_prev, props["prev"][chosen].astype(np.float64)) if len(chosen) else np.zeros(0)
    obs = lambda pv, cv: np.concatenate([pv[:, 0:2], cv[:, 0:2], pose.keypoints(pv[:, 2:5]),  # noqa: E731
                                         pose.keypoints(cv[:, 2:5])], 1)
    near = nearest(obs(p_prev, cur[src_rows]), obs(prev[sub], cur[sub])) if len(chosen) else np.zeros(0)
    sel_mask = w["loss_mask"]
    sup_slots = sel_mask.sum(1) if len(chosen) else np.zeros(0)
    # the chain's largest deviation over the frames where supervised continuation actions are applied
    cont = np.array([rec["dev"][i][1:int(np.flatnonzero(rec["mask"][i])[-1])].max() for i in chosen])
    per_ep = {str(e): int((w["episode"] == e).sum()) for e in eps}
    n_eps = int(sum(v > 0 for v in per_ep.values()))
    if len(chosen) >= n_syn and n_eps >= SPEC["yield"]["min_episodes"]:
        decision = "pass: Stage B may be proposed in its own protocol PR"
    else:
        decision = (f"refuse: {len(chosen)} anchors from {n_eps} episodes, against {n_syn} anchors from "
                    f"{SPEC['yield']['min_episodes']} episodes; the study stops")
    return {
        "proposals": {"population": int(len(props["candidate_row"])), "attempted": int(done.sum()), "counts": counts,
                      "accepted": int(len(accepted)), "anchors": int(len(chosen)), "episodes_with_anchors": n_eps,
                      "per_episode": per_ep},
        "limits": {"action_change_px": limits["action_change"], "speed": limits["speed"].tolist()},
        "diagnostics": {
            "history_residual_px": q(rec["dev"][chosen][:, 0]) if len(chosen) else None,
            "rejoin_residual_px": q(rec["dev"][chosen][:, 1]) if len(chosen) else None,
            "continuation_max_px": q(cont) if len(chosen) else None,
            "iterations": q(rec["iterations"][chosen]) if len(chosen) else None,
            "supervised_slots": {str(k): int((sup_slots == k).sum()) for k in range(N_ACT + 1, H + 1)},
            "action_change_px": q(np.linalg.norm(zc[:, 10:12] - np.array(
                [src["action"][r - 1] for r in src_rows], np.float64), axis=1)) if len(chosen) else None,
            "history_from_source_px": q(disp_source), "history_from_oracle_decode_px": q(disp_decode),
            "nearest_real_history_px": q(near)},
        "mass": {"n_real": n_real, "n_synthetic": n_syn,
                 "real_supervised_values_per_window": H * 2,
                 "selected_supervised_values": int(sup_slots.sum() * 2) if len(chosen) else 0,
                 "selected_windows": int(len(chosen))},
        "decision": None if smoke else decision}


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def code_identity() -> dict:
    files = ["tools/pusht_recovery/stage_a.py", "tools/pusht_recovery/stage-a.json", "tools/pusht_v2/oracle_probe.py",
             "tools/pusht_v2/oracle_stage_a.py", "tools/pusht_v2/pose.py", "tools/pusht_v2/source.py",
             "tools/pusht_v2/oracle.json", "tools/pusht_v2/surface.json", "src/ilsb/temporal.py"]
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain")),
            "files_sha256": {f: sha(ROOT / f) for f in files}}


def simulator_check():
    import importlib.metadata as md
    got = {"gym_pusht": md.version("gym-pusht"), "pymunk": md.version("pymunk")}
    want = {k: SPEC["simulator"][k] for k in got}
    if got != want:
        sys.exit(f"refused: simulator {got}, not the protocol's {want}")
    return got


def load_inputs(args):
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if source_module.content_sha256(src) != SPEC["inputs"]["source_content_sha256"]:
        sys.exit("refused: the source rows are not the protocol's by content")
    if sha(args.candidates) != SPEC["inputs"]["candidates_sha256"]:
        sys.exit("refused: the candidates are not the protocol's")
    with np.load(args.proposals) as z:
        orc = {k: z[k] for k in z.files}
    if content_sha(orc) != SPEC["inputs"]["oracle_proposals_content_sha256"]:
        sys.exit("refused: the oracle proposals are not the protocol's by content")
    return src, np.load(args.candidates), orc


def population(src, cand, orc, eps):
    """The oracle-accepted proposals as (candidate_row, episode, t, action, prev), t the row's index in its episode."""
    ok = np.flatnonzero(orc["status"] == 0)
    c_ep = cand["source_episode_id"].astype(np.int64)[ok]
    c_fr = cand["source_frame"].astype(np.int64)[ok]
    if not np.isin(c_ep, eps).all():
        sys.exit("refused: an oracle proposal lies outside the subset")
    first = {e: int(np.flatnonzero(src["episode"] == e)[0]) for e in eps}
    t = np.array([int(r) - first[int(e)] for r, e in zip(orc["src_row"][ok], c_ep)], np.int64)
    if not np.array_equal(src["frame"][orc["src_row"][ok]], c_fr) or not np.array_equal(src["episode"][orc["src_row"][ok]], c_ep):
        sys.exit("refused: the oracle proposals' source rows are not the candidates'")
    return {"candidate_row": ok.astype(np.int64), "episode": c_ep, "t": t,
            "action": cand["actions"][ok].astype(np.float64), "prev": orc["prev_oracle"][ok].astype(np.float64)}


def main():
    p = argparse.ArgumentParser()
    for k in ("source", "candidates", "proposals", "run"):
        p.add_argument(f"--{k}", type=Path, required=True)
    p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/pusht-recovery/stage-a.json")
    p.add_argument("--workers", type=int, default=min(41, os.cpu_count() or 1))
    args = p.parse_args()
    sim = simulator_check()
    src, cand, orc = load_inputs(args)
    eps = subset(BASE["roles"]["train"], SPEC["cell"]["r"], SPEC["cell"]["subset_seed"])
    if set(eps) & set(BASE["roles"]["held_out"]):
        sys.exit("refused: the subset overlaps the held-out episodes")
    ex = SPEC["expected"]
    props = population(src, cand, orc, eps)
    sub_rows = int(np.isin(src["episode"], eps).sum())
    if (len(eps), sub_rows, len(orc["status"]), len(props["candidate_row"])) != (
            ex["subset_episodes"], ex["subset_rows"], ex["proposals"], ex["oracle_accepted"]):
        sys.exit("refused: the subset or the proposal counts are not the protocol's")
    if real_windows(src, eps) != (ex["n_real"], ex["n_synthetic"]):
        sys.exit("refused: the real window count is not the protocol's")
    if git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; Stage A runs from a clean commit")
    if args.run.exists():
        sys.exit(f"refused: {args.run} exists; a run is never overwritten")
    out = run(src, eps, props, args.workers, args.run, smoke=False)
    out = {"schema": "ilsb.pusht_recovery.stage_a 1", "code": code_identity(), "simulator": sim,
           "identities": {**SPEC["inputs"], "stage_a_json_sha256": sha(HERE / "stage-a.json")}, **out}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"proposals": out["proposals"]["counts"], "anchors": out["proposals"]["anchors"],
                      "mass": out["mass"], "decision": out["decision"]}))


def run(src, eps, props, workers, run_dir: Path, smoke: bool) -> dict:
    rec, limits, sub, rows_of = build(src, eps, props, workers)
    accepted = np.flatnonzero(rec["status"] == 0)
    src_row = np.array([rows_of[int(e)][int(t)] for e, t in zip(props["episode"], props["t"])], np.int64)
    chosen = select(props, accepted, src_row)
    w = windows(src, props, rec, rows_of, chosen)
    rpt = report(src, eps, props, rec, limits, sub, rows_of, accepted, chosen, w, smoke)
    arrays = {**{f"proposal/{k}": v for k, v in rec.items()}, "proposal/candidate_row": props["candidate_row"],
              "proposal/episode": props["episode"], "proposal/t": props["t"], "selected": chosen,
              **{f"window/{k}": v for k, v in w.items()}}
    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(run_dir / "windows.npz", **arrays)
    rpt["windows_npz_content_sha256"] = content_sha(arrays)
    rpt["windows_content_sha256"] = content_sha(w)
    return rpt


if __name__ == "__main__":
    main()
