"""#66 Stage A: discovery, reach and validity at a Diffusion Policy's observed failures, no training (pusht-failure.md).

    LEROBOT_PYTHON tools/pusht_failure/failures.py --source SOURCE.npz --checkpoint CKPT --base B --run RUN_DIR \
        [--starts discovery|smoke]
    LEROBOT_PYTHON tools/pusht_failure/failures.py audit --source SOURCE.npz --checkpoint CKPT --base B --run RUN_DIR
    LEROBOT_PYTHON tools/pusht_failure/failures.py report --runs RUN_1 RUN_2 [--out RECORD.json]

A1. Base B drives R = 4 rollouts from each start, in lockstep as one batch: env.reset(seed = start) for every row,
torch seeded with SeedSequence([66, B, start, chunk]) before each 16-action chunk, batch row r is repeat r. The policy
is driven as lerobot-eval drives it (B2's driver) in LeRobot's numerical mode (TF32 convolutions, #62). A row's
episode ends at success or step 300 and is scored up to its first success; the batch steps until every row ends, and
every row's inputs are saved, so any chunk regenerates. A rollout fails consequentially when its maximum scored
coverage is below 0.504; a failure start has at least 3 such repeats.

A2. Each rollout's two-frame state x_t = [o_{t-1}, o_t] (x_0 = [o_0, o_0], as LeRobot fills its queue) is measured as
18 points against the 5,040 subset demonstration states (support.py); d_sup is the largest per-point distance to the
nearest one. t_e is the first t >= 1 with d_sup(x_{t-1}) <= tau < d_sup(x_t); j* is the support row nearest x_{t_e-1}.

A3. At t_e the window observes the policy's two frames there, [o_{t_e-1}, o_{t_e}]. In LeRobot's indexing slot k is
the action at t_e + k - 1, so slot 0 (the action that produced o_{t_e}) is history and masked from the loss, and slots
1-8 are the chunk executed from o_{t_e}. Those 8 actions are solved from the saved full state at t_e onto row j* + 8,
by Gauss-Newton with minimum-norm steps from the demonstrated actions of rows j* .. j* + 7. Slots 9-15 are the
demonstrated actions from row j* + 8 on, supervised as a prefix while the chain stays within 1.0 px. #52's checks
apply. One window per failing rollout; a rejected one is never replaced.

The audit runs in a fresh process: every chunk regenerates bit for bit from the saved observations and seeds; coverage
is recomputed from the saved states; the labels, failure starts, reach and every window's status are recomputed and
must be identical; every accepted window's chain is re-simulated from its saved actions and must land within 0.05 px;
a simulator replay of the saved actions is a measurement, not a gate (#53).
"""
import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
sys.path.insert(0, str(ROOT / "tools/pusht_recovery"))
sys.path.insert(0, str(ROOT / "tools/pusht_uncertainty"))
import probe as P  # noqa: E402
import stage_a as A  # noqa: E402
import stage_b2 as B  # noqa: E402
import support as S  # noqa: E402

SPEC = json.loads((HERE / "stage-a.json").read_text())
D_, REACH, REC, DEC = SPEC["discovery"], SPEC["reach"], SPEC["recovery"], SPEC["decision"]
R_REP, CHUNK, MAX_STEPS, H = D_["repeats"], 8, 300, 16
LOW, MIN_FAIL_REPEATS = D_["consequential_failure"]["max_coverage_below"], 3
TAU = REACH["tau_px"]
SOLVE, TOL, CONT_PX = REC["solve"], REC["solve"]["tolerance_px"], 1.0
STARTS = {k: (SPEC["starts"][k]["start_seed"], SPEC["starts"][k]["n_episodes"]) for k in ("discovery", "smoke")}
COV_DIGITS = B.COV_DIGITS
TRAJ_KEYS = ("obs_agent", "obs_kp", "full", "pos", "action", "coverage", "chunk_seeds")
STATUS = ("ok", "singular_jacobian", "solve_failed", "non_finite", "action_out_of_bounds", "speed_beyond_envelope",
          "beyond_penetration_envelope", "beyond_extent_envelope")
CLASSES = tuple(REACH["classes"])
log = P.log


def git(*a):
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def seed_of(*key) -> int:
    return int(np.random.SeedSequence([66, *key]).generate_state(1, np.uint32)[0])


def check_member(ckpt: Path, base: int) -> str:
    got = A.sha(ckpt / "model.safetensors")
    if got != SPEC["members"][str(base)]["model_sha256"]:
        sys.exit(f"refused: the checkpoint is not base {base}'s ({got[:12]})")
    return got


class Policy(P.Policy):
    """#62's TF32 driver, on a batch of R environments."""

    def batch(self, agent: np.ndarray, kp: np.ndarray) -> dict:
        agent, kp = np.atleast_2d(agent).astype(np.float64), np.atleast_2d(kp).astype(np.float64)
        obs = self.prep({"agent_pos": agent, "environment_state": kp})
        obs["task"] = [""] * len(agent)
        b = self.pre(obs)
        b.pop("action", None)
        return b

    def act(self, agent, kp, seed: int | None) -> np.ndarray:
        from lerobot.utils.constants import ACTION
        with self.torch.inference_mode():
            b = self.batch(agent, kp)
            if len(self.policy._queues[ACTION]) == 0:
                self.torch.manual_seed(seed)
            return self.post(self.policy.select_action(b)).cpu().numpy().astype(np.float64)


def rollouts(pol: Policy, envs: list, base: int, start: int) -> dict:
    """The R repeats of one start, in lockstep. Arrays are (steps + 1, R, ...) for states and (steps, R, ...) for
    actions; end[r] is row r's scored length (its first success, or every step)."""
    for env in envs:
        env.reset(seed=start)
    pol.policy.reset()
    obs = [B.obs_of(env) for env in envs]
    t = {k: [] for k in TRAJ_KEYS}

    def keep_state():
        t["obs_agent"].append([o[0] for o in obs]), t["obs_kp"].append([o[1] for o in obs])
        t["full"].append([B.flat(A.full_state(e)) for e in envs]), t["pos"].append([A.positions(e) for e in envs])

    keep_state()
    end = np.full(len(envs), -1, np.int64)
    for i in range(MAX_STEPS):
        s = None
        if pol.needs_chunk():
            s = seed_of(base, start, len(t["chunk_seeds"]))
            t["chunk_seeds"].append((i, s))
        a = pol.act(np.stack([o[0] for o in obs]), np.stack([o[1] for o in obs]), s)
        cov = []
        for r, env in enumerate(envs):
            _, _, _, _, info = env.step(a[r])
            cov.append(round(float(info["coverage"]), COV_DIGITS))
            if info["is_success"] and end[r] < 0:
                end[r] = i + 1
        obs = [B.obs_of(env) for env in envs]
        t["action"].append(a), t["coverage"].append(cov)
        keep_state()
        if (end > 0).all():
            break
    end[end < 0] = len(t["action"])
    out = {k: np.array(v, np.int64 if k == "chunk_seeds" else np.float64) for k, v in t.items()}
    out["chunk_seeds"] = out["chunk_seeds"].reshape(-1, 2)
    out["end"] = end
    return out


def pack(runs: list) -> dict:
    out = {}
    for key in (*TRAJ_KEYS, "end"):
        parts = [x[key] for x in runs]
        out[key] = np.concatenate(parts)
        out[f"{key}_offsets"] = np.cumsum([0] + [len(p) for p in parts]).astype(np.int64)
    return out


def unpack(arrs: dict, i: int) -> dict:
    return {k: arrs[k][arrs[f"{k}_offsets"][i]:arrs[f"{k}_offsets"][i + 1]] for k in (*TRAJ_KEYS, "end")}


def starts_of(which: str) -> list:
    s0, n = STARTS[which]
    return list(range(s0, s0 + n))


# ---------- support and limits (CPU, from the subset's demonstrations) ----------

def demonstrations(src: dict) -> dict:
    """The support: the subset's 5,040 states as points, with each row's episode position, recorded positions and
    actions; and #52's validity limits."""
    eps = B.episodes_of("all")
    rows = np.flatnonzero(np.isin(src["episode"], eps))
    if len(eps) != S.EXPECTED["subset_episodes"] or len(rows) != S.EXPECTED["subset_states"]:
        sys.exit("refused: the subset is not the protocol's")
    ep = src["episode"][rows]
    n_of = {e: int((ep == e).sum()) for e in eps}
    pos_in = np.concatenate([np.arange(n_of[e]) for e in eps])  # rows are in (episode, frame) order
    cur = np.concatenate([src["state"][:, 2:4], src["pose"][:, 3:6]], 1).astype(np.float64)
    return {"rows": rows, "pts": S.points(src["state"][rows], src["env"][rows]), "pos_in": pos_in,
            "n": np.array([n_of[e] for e in ep]), "cur": cur, "act": src["action"].astype(np.float64),
            "limits": A.limits_of(src, eps)}


def states_of(agent: np.ndarray, kp: np.ndarray) -> np.ndarray:
    """(T + 1, 2), (T + 1, 16) frames -> (T + 1, 18, 2) two-frame states; x_0 repeats frame 0."""
    prev = np.concatenate([[0], np.arange(len(agent) - 1)])
    return S.points(np.concatenate([agent[prev], agent], 1), np.concatenate([kp[prev], kp], 1))


def nearest(x: np.ndarray, sup: np.ndarray) -> tuple:
    """(d_sup, index of the nearest support state) for (N, 18, 2) states."""
    d, j = np.empty(len(x)), np.empty(len(x), np.int64)
    for i in range(0, len(x), 64):
        dd = np.sqrt(((x[i:i + 64, None] - sup[None]) ** 2).sum(-1)).max(-1)
        j[i:i + 64] = dd.argmin(1)
        d[i:i + 64] = dd[np.arange(len(dd)), j[i:i + 64]]
    return d, j


def reach(d: np.ndarray, j: np.ndarray, end: int, demo: dict) -> dict:
    """A rollout's class, t_e and j* from its d_sup over states 0 .. end."""
    d = d[:end + 1]
    inside = d <= TAU
    out = {"class": None, "t_e": -1, "j_star": -1, "d_e": float("nan")}
    if not inside.any():
        out["class"] = "never_in_support"
        return out
    dep = np.flatnonzero(inside[:-1] & ~inside[1:])
    if not len(dep):
        out["class"] = "no_departure"
        return out
    te = int(dep[0]) + 1
    js = int(j[te - 1])
    out.update(t_e=te, j_star=js, d_e=float(d[te]))
    out["class"] = "departs_eligible" if demo["pos_in"][js] + 8 <= demo["n"][js] - 1 else "departs_ineligible"
    return out


# ---------- A3: the recovery window ----------

def chain(env, z: np.ndarray, actions: np.ndarray) -> tuple:
    """Positions and full states from the saved full state z, then after each action, in one fresh live space."""
    A.load_fresh(env, B.unflat(z))
    p, f = [A.positions(env)], [A.full_state(env)]
    for a in actions:
        env.step(a)
        p.append(A.positions(env)), f.append(A.full_state(env))
    return np.array(p), f


def solve(env, z: np.ndarray, u0: np.ndarray, target: np.ndarray) -> tuple:
    """(status, u, iterations): Gauss-Newton with minimum-norm steps on the 5 residuals in the 16 action values."""
    from oracle_stage_a import r_of
    res = lambda v: r_of(chain(env, z, v.reshape(8, 2))[0][-1], target)  # noqa: E731
    u, h = u0.copy(), SOLVE["fd_step"]
    r = res(u)
    for it in range(SOLVE["max_iterations"] + 1):
        if not (np.isfinite(u).all() and np.isfinite(r).all()):
            return "non_finite", u, it
        if A.distance(chain(env, z, u.reshape(8, 2))[0][-1], target)[0] <= TOL:
            return "ok", u, it
        if it == SOLVE["max_iterations"]:
            break
        J = np.empty((5, 16))
        for k in range(16):
            e = np.zeros(16)
            e[k] = h
            J[:, k] = (res(u + e) - res(u - e)) / (2 * h)
        if not np.isfinite(J).all() or not np.any(J):
            return "singular_jacobian", u, it
        du, step = np.linalg.lstsq(J, -r, rcond=SOLVE["rcond"])[0], 1.0
        for _ in range(SOLVE["line_search"] + 1):
            un = u + step * du
            rn = res(un)
            if np.isfinite(rn).all() and np.linalg.norm(rn) < np.linalg.norm(r):
                u, r = un, rn
                break
            step /= 2
        else:
            return "solve_failed", u, it
    return "solve_failed", u, SOLVE["max_iterations"]


def check(env, z: np.ndarray, slots: np.ndarray, js: int, demo: dict) -> tuple:
    """(status, mask, dev): the window's chain from z under slots 1-15, its loss mask and #52's checks.
    The target of slot k >= 9 is row j* + k (slot k applies the action of row j* + k - 1)."""
    j_row = int(demo["rows"][js])
    n_left = int(demo["n"][js] - 1 - demo["pos_in"][js])  # rows after j* in its episode
    p, f = chain(env, z, slots[1:])
    mask, dev = np.zeros(H, bool), np.full(H, np.nan)
    mask[1:9] = True
    for k in range(9, H):
        if k > n_left:
            break
        dev[k] = float(A.distance(p[k], demo["cur"][j_row + k])[0])
        if not (dev[9:k + 1] <= CONT_PX).all():
            break
        mask[k] = True
    last = int(np.flatnonzero(mask)[-1])
    lo, hi = A.LO, A.HI
    if not np.isfinite(p).all():
        return "non_finite", mask, dev
    if not ((slots[mask] >= lo) & (slots[mask] <= hi)).all():
        return "action_out_of_bounds", mask, dev
    if (np.array([A.speeds(s) for s in f[1:last + 1]]) > demo["limits"]["speed"]).any():
        return "speed_beyond_envelope", mask, dev
    pen, ext = A.envelope_bad(p[:last + 1])
    if pen:
        return "beyond_penetration_envelope", mask, dev
    if ext:
        return "beyond_extent_envelope", mask, dev
    return "ok", mask, dev


def window(env, t: dict, r: int, te: int, js: int, demo: dict) -> dict:
    """The recovery window of row r at its departure t_e, matched to support index js."""
    j_row = int(demo["rows"][js])
    z, target = t["full"][te, r], demo["cur"][j_row + 8]
    u0 = demo["act"][j_row:j_row + 8].reshape(-1)
    status, u, it = solve(env, z, u0, target)
    slots = np.empty((H, 2))
    slots[0] = t["action"][te - 1, r]
    slots[1:9] = u.reshape(8, 2)
    last_row = j_row + int(demo["n"][js] - 1 - demo["pos_in"][js])
    for k in range(9, H):  # the demonstrated actions from row j* + 8 on, the episode's last action past its end
        slots[k] = demo["act"][min(j_row + k - 1, last_row)]
    mask, dev = np.zeros(H, bool), np.full(H, np.nan)
    if status == "ok":
        status, mask, dev = check(env, z, slots, js, demo)
    return {"status": status, "iterations": it, "actions": slots, "mask": mask, "dev": dev, "source_row": j_row,
            "obs_agent": t["obs_agent"][te - 1:te + 1, r], "obs_kp": t["obs_kp"][te - 1:te + 1, r]}


# ---------- labels and the whole Stage A computation from saved trajectories ----------

def label(arrs: dict, starts: list, demo: dict, smoke: bool) -> tuple:
    """(per-rollout table, windows): recomputed from the saved trajectories alone."""
    env = A.make_env()
    table, wins = [], []
    for i, start in enumerate(starts):
        t = unpack(arrs, i)
        low = []
        for r in range(R_REP):
            e = int(t["end"][r])
            c = t["coverage"][:e, r]
            success = bool((c > 0.95).any())
            low.append(bool(c.max() < LOW))
            table.append({"start": start, "repeat": r, "success": success, "scored_steps": e,
                          "max_coverage": float(c.max()), "consequential": low[-1]})
        fail_start = sum(low) >= MIN_FAIL_REPEATS
        for r in range(R_REP):
            row = table[-R_REP + r]
            row["failure_start"] = fail_start
            row["unit"] = bool(fail_start and row["consequential"])
            e = row["scored_steps"]
            d, j = nearest(states_of(t["obs_agent"][:e + 1, r], t["obs_kp"][:e + 1, r]), demo["pts"])
            row.update(reach(d, j, e, demo))
            row["d_sup"] = {k: float(v) for k, v in zip(("min", "50", "max"), np.percentile(d, (0, 50, 100)))}
            row["window"] = None
            if row["class"] == "departs_eligible" and (row["unit"] or smoke):
                w = window(env, t, r, row["t_e"], row["j_star"], demo)
                row["window"] = w["status"]
                if w["status"] == "ok":
                    wins.append({**w, "start": start, "repeat": r, "t_e": row["t_e"]})
        if (i + 1) % 20 == 0 or i + 1 == len(starts):
            log(f"labels: {i + 1}/{len(starts)} starts")
    return table, wins


def windows_arrays(wins: list) -> dict:
    shapes = {"actions": (H, 2), "mask": (H,), "dev": (H,), "obs_agent": (2, 2), "obs_kp": (2, 16)}
    out = {k: np.array([w[k] for w in wins], np.int64) for k in ("start", "repeat", "t_e", "source_row", "iterations")}
    for k, shape in shapes.items():
        out[k] = np.array([w[k] for w in wins], bool if k == "mask" else np.float64).reshape(len(wins), *shape)
    return out


def summary(table: list) -> dict:
    units = [x for x in table if x["unit"]]
    fail_starts = sorted({x["start"] for x in table if x["failure_start"]})
    ok = [x for x in units if x["window"] == "ok"]
    ok_starts = sorted({x["start"] for x in ok})
    group = lambda rows: {c: sum(x["class"] == c for x in rows) for c in CLASSES}  # noqa: E731
    by_status = {s: sum(x["window"] == s for x in units) for s in STATUS}
    succ = [x for x in table if x["success"]]
    near = [x for x in table if not x["success"] and not x["consequential"]]
    informative = len(fail_starts) >= DEC["min_failure_starts"]
    passed = bool(informative and len(ok) >= DEC["min_windows"] and len(ok_starts) >= DEC["min_failure_starts"])
    return {
        "rollouts": len(table), "successful": len(succ), "near_miss": len(near),
        "consequential": sum(x["consequential"] for x in table),
        "failure_starts": len(fail_starts), "units": len(units),
        "units_by_class": group(units), "units_by_window_status": by_status,
        "valid_windows": len(ok), "valid_window_starts": len(ok_starts),
        "informative": informative, "pass": passed,
        "descriptive": {"successful_by_class": group(succ), "near_miss_by_class": group(near),
                        "units_d_sup_min": P.q([x["d_sup"]["min"] for x in units]),
                        "successful_d_sup_min": P.q([x["d_sup"]["min"] for x in succ]),
                        "units_d_at_departure": P.q([x["d_e"] for x in units if x["t_e"] > 0])},
    }


# ---------- modes ----------

def discover(args, pol, demo) -> dict:
    envs = [A.make_env() for _ in range(R_REP)]
    starts = starts_of(args.starts)
    runs, t0 = [], time.time()
    log(f"base {args.base}: {len(starts)} starts from {starts[0]}, {R_REP} repeats in lockstep")
    for i, start in enumerate(starts):
        runs.append(rollouts(pol, envs, args.base, start))
        if (i + 1) % 5 == 0 or i + 1 == len(starts):
            log(f"base {args.base}: {i + 1}/{len(starts)} starts, {time.time() - t0:.0f} s")
    arrs = {**pack(runs), "start": np.array(starts, np.int64)}
    seconds = time.time() - t0
    args.run.mkdir(parents=True, exist_ok=False)
    np.savez(args.run / "trajectories.npz", **arrs)  # durable before labelling; the record marks completion
    table, wins = label(arrs, starts, demo, args.starts == "smoke")
    w = windows_arrays(wins)
    np.savez(args.run / "windows.npz", **w)
    return {"schema": "ilsb.pusht_failure.stage_a_run 1",
            "code": {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))},
            "base": args.base, "starts": args.starts, "model_sha256": check_member(args.checkpoint, args.base),
            "environment": P.environment(pol.torch),
            "trajectories_content_sha256": A.content_sha(arrs), "windows_content_sha256": A.content_sha(w),
            "table": table, **summary(table)}, seconds


def load_run(run: Path) -> tuple:
    rec = json.loads((run / "record.json").read_text())
    with np.load(run / "trajectories.npz") as z:
        arrs = {k: z[k] for k in z.files}
    with np.load(run / "windows.npz") as z:
        w = {k: z[k] for k in z.files}
    return rec, arrs, w


def audit(args, run=None, pol=None, demo=None, only=None) -> dict:
    """run: (record, trajectories, windows) in place of RUN_DIR's; only: regenerate and replay these start indices
    alone (the smoke's faults). The labels, reach and windows are always recomputed for every start."""
    rec, arrs, w = run if run is not None else load_run(args.run)
    problems = []
    if A.content_sha(arrs) != rec["trajectories_content_sha256"]:
        problems.append("the trajectories are not the record's")
    if A.content_sha(w) != rec["windows_content_sha256"]:
        problems.append("the windows are not the record's")
    if rec["model_sha256"] != check_member(args.checkpoint, args.base) or rec["base"] != args.base:
        problems.append("the checkpoint is not the record's")
    pol = pol or Policy(args.checkpoint)
    if rec["environment"] != P.environment(pol.torch):
        problems.append("the environment is not the record's: bitwise reproducibility holds only within it")
    demo = demo or demonstrations(B.load_source(args.source))
    starts = [int(s) for s in arrs["start"]]
    env, chunk_bad, worst_cov, worst_replay, t0 = A.make_env(), 0, 0.0, 0.0, time.time()
    for i, start in enumerate(starts):
        if only is not None and i not in only:
            continue
        t = unpack(arrs, i)
        # every chunk, regenerated from the saved observations of all R rows
        for step, s in t["chunk_seeds"]:
            step, s = int(step), int(s)
            prev = max(step - 1, 0)
            pol.start(t["obs_agent"][prev], t["obs_kp"][prev])  # the queue holds [o_prev, o_step] once act adds o_step
            got = [pol.act(t["obs_agent"][step], t["obs_kp"][step], s)]
            for m in range(1, CHUNK):
                if step + m >= len(t["action"]):
                    break
                got.append(pol.act(t["obs_agent"][step + m], t["obs_kp"][step + m], None))
            if not np.array_equal(np.array(got), t["action"][step:step + len(got)]):
                chunk_bad += 1
        for r in range(R_REP):
            cov = []
            for z in t["full"][1:, r]:
                A.load_fresh(env, B.unflat(z))
                cov.append(env._get_coverage())
            worst_cov = max(worst_cov, float(np.abs(np.array(cov) - t["coverage"][:, r]).max()))
            env.reset(seed=start)
            rep = [A.positions(env)]
            for a in t["action"][:, r]:
                env.step(a)
                rep.append(A.positions(env))
            worst_replay = max(worst_replay, float(A.distance(np.array(rep), t["pos"][:, r]).max()))
        if only is None and ((i + 1) % 5 == 0 or i + 1 == len(starts)):
            log(f"audit base {args.base}: {i + 1}/{len(starts)} starts, {time.time() - t0:.0f} s")
    if chunk_bad:
        problems.append(f"{chunk_bad} chunks do not regenerate bit for bit")
    if worst_cov > B.COV_TOL:
        problems.append(f"coverage recomputed from the saved states differs by {worst_cov:.3g}")
    table, wins = label(arrs, starts, demo, rec["starts"] == "smoke")
    if json.dumps(table, sort_keys=True) != json.dumps(rec["table"], sort_keys=True):
        problems.append("the labels, reach or window statuses do not reproduce")
    if A.content_sha(windows_arrays(wins)) != A.content_sha(w):
        problems.append("the windows do not rebuild bit for bit")
    # independent check of every accepted window: its saved slots 1-8 land on row j* + 8 from the saved full state
    landed = 0.0
    for k in range(len(w["start"])):
        i = starts.index(int(w["start"][k]))
        t = unpack(arrs, i)
        z = t["full"][int(w["t_e"][k]), int(w["repeat"][k])]
        p, _ = chain(env, z, w["actions"][k][1:9])
        landed = max(landed, float(A.distance(p[-1], demo["cur"][int(w["source_row"][k]) + 8])[0]))
        if w["mask"][k][0] or not w["mask"][k][1:9].all():
            problems.append(f"window {k}: slot 0 supervised or a recovery slot masked")
    if landed > TOL:
        problems.append(f"an accepted window lands {landed:.3g} px from its target")
    return {"problems": problems, "pass": not problems, "chunks_mismatched": chunk_bad,
            "coverage_max_abs_diff": worst_cov, "windows_max_landing_px": landed,
            "simulator_replay_max_px": worst_replay}


def report(args) -> dict:
    recs = [json.loads((r / "record.json").read_text()) for r in args.runs]
    if [r["base"] for r in recs] != [1, 2] or any(r["starts"] != "discovery" for r in recs):
        sys.exit("refused: report takes base 1's and base 2's discovery runs")
    if len({r["code"]["commit"] for r in recs}) != 1 or any(r["code"]["dirty"] for r in recs):
        sys.exit("refused: the runs are not from one clean commit")
    keys = [k for k in summary(recs[0]["table"])]
    bases = {str(r["base"]): {**{k: r[k] for k in keys}, "model_sha256": r["model_sha256"],
                              "trajectories_content_sha256": r["trajectories_content_sha256"],
                              "windows_content_sha256": r["windows_content_sha256"],
                              "run_files_sha256": {f: A.sha(run / f) for f in ("record.json", "trajectories.npz",
                                                                                "windows.npz")}}
             for r, run in zip(recs, args.runs)}
    uninformative = [b for b, v in bases.items() if not v["informative"]]
    ok = all(v["pass"] for v in bases.values())
    decision = ("refuse: fewer than 20 failure starts on base " + ", ".join(uninformative) if uninformative
                else "pass: Stage B may be proposed" if ok else "stop: too few valid recovery windows")
    return {"schema": "ilsb.pusht_failure.stage_a 1", "code": recs[0]["code"], "environment": recs[0]["environment"],
            "rule": {"max_coverage_below": LOW, "failure_repeats": f"{MIN_FAIL_REPEATS} of {R_REP}", "tau_px": TAU,
                     "min_windows": DEC["min_windows"], "min_failure_starts": DEC["min_failure_starts"]},
            "bases": bases, "pass": ok, "decision": decision}


def main():
    argv = sys.argv[1:]
    mode = argv[0] if argv[:1] and argv[0] in ("audit", "report") else "run"
    argv = argv[1:] if mode != "run" else argv
    p = argparse.ArgumentParser()
    if mode == "report":
        p.add_argument("--runs", type=Path, nargs=2, required=True)
        p.add_argument("--out", type=Path, default=ROOT / "docs/studies/evidence/pusht-failure/stage-a.json")
        args = p.parse_args(argv)
        out = report(args)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
        print(json.dumps({"pass": out["pass"], "decision": out["decision"]}))
        return
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, required=True)
    p.add_argument("--base", type=int, choices=(1, 2), required=True)
    p.add_argument("--run", type=Path, required=True)
    if mode == "run":
        p.add_argument("--starts", choices=tuple(STARTS), default="discovery")
    args = p.parse_args(argv)
    A.simulator_check()
    if mode == "audit":
        out = audit(args)
        print(json.dumps(out))
        sys.exit(0 if out["pass"] else 2)
    if args.starts == "discovery" and git("status", "--porcelain"):
        sys.exit("refused: the tree is dirty; Stage A runs from a clean commit")
    if args.run.exists():
        sys.exit(f"refused: {args.run} exists; a run is never overwritten")
    check_member(args.checkpoint, args.base)
    demo = demonstrations(B.load_source(args.source))
    pol = Policy(args.checkpoint)
    rec, seconds = discover(args, pol, demo)
    (args.run / "record.json").write_text(json.dumps(rec, indent=1, sort_keys=True) + "\n")
    log(json.dumps({"base": args.base, "rollout_seconds": round(seconds, 1),
                    **{k: rec[k] for k in ("failure_starts", "units", "units_by_class", "valid_windows",
                                           "valid_window_starts", "pass")}}))


if __name__ == "__main__":
    main()
