#!/usr/bin/env python3
"""Targeted recovery segments at Pendulum capture failures, Stage A (docs/studies/pendulum-recovery-segments.md, v1).

    uv run python tools/recovery_segments.py --out DIR [--device cuda] [--resume] [--smoke]

Stages, each written atomically under DIR and skipped on --resume when present: data (subset and split per design
seed), bc (the provisional BC group), failures (real-environment BC rollouts from the collection and held-out start
sets, the failure states, the recovers audit), models (primary and check transition models, one fit group), generate
(per seed, M and arm: the forward segments and their row-level quality fields), discovery (the provisional BC inside the
primary model), then stage-a.json with every gate, the selection, the stop outcome, the discovery verdict and the Stage
B step budget S. The run is complete only when stage-a.json exists. Every record carries the run identity (commit,
tool and dataset sha256); --resume refuses a directory with another identity. --smoke runs one non-study seed (3399),
non-study start sets (2498, 2499), M in {1, 2} and 20 starts per set into a directory marked as a smoke run; it reads
no design or pilot data, and its records are never evidence.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import positive_control as pc  # noqa: E402
from ilsb.store import sha256_file  # noqa: E402
from ilsb.tasks.pendulum import UPRIGHT_COS, ExactModel, Task, expert, subset  # noqa: E402

PROTOCOL = "docs/studies/pendulum-recovery-segments.md (version 1)"
DESIGN_SEEDS = tuple(range(3321, 3326))
COLLECTION, HELD_OUT = 2401, 2402
SMOKE_SEED, SMOKE_STARTS = 3399, (2498, 2499)  # never study seeds or study start sets
ALL_NEW_STARTS = (2401, 2402, 2403, 2404, 2405)
EARLIER_STARTS = (*range(2201, 2221), 2301)
LADDER = (1, 2, 4, 8, 16)
ARMS = ("exact", "learned")
APPROACH, CAPTURE, PER_TRAJECTORY, HOLD = 0.5, 0.1, 40, 100
LAMBDA, ITERATIONS, MU0, FD_STEP, BOUND = 1e-4, 50, 1e-3, 1e-3, 3.0
TRUST, RHO_COV, NOVELTY = 0.25, 0.5, 1.0  # fractions of tau_demo
REPLAY_MIN, ARRIVAL_MIN, COVERAGE_MIN, MIN_CAPTURE_ROWS, LABEL_TOL, LABEL_FRACTION, BC_RATIO = (
    0.95, 0.5, 0.5, 100, 0.3, 0.9, 0.5)
PASS_SEEDS, DISCOVERY_MIN = 4, 0.5
CHECK_SEED_OFFSET = 100_000
# CCIL tools/pendulum_recovery/generate.py's DYNAMICS (#171), itself the ilsb study's dynamics block, unchanged
DYNAMICS = {"hidden_layers": [512, 512], "activation": "relu", "lipschitz_type": "soft_sampling",
            "lipschitz_constraint": 2.0, "lipschitz_coordinates": "model_normalized", "soft_penalty_weight": 0.001,
            "soft_sampling_eps": 0.0003, "soft_samples": 5, "learning_rate": 0.001, "weight_decay": 1.0e-05,
            "epochs": 100, "batch_size": 512}
TOOL_FILES = ("tools/recovery_segments.py", "tools/positive_control.py", "src/ilsb/tasks/pendulum.py",
              "src/ilsb/tasks/pendulum_map.py", "src/ilsb/learners/vector_bc.py")
G = 9.81
log, Heartbeat, atomic_json, atomic_npz = pc.log, pc.Heartbeat, pc.atomic_json, pc.atomic_npz


class Refused(SystemExit):
    pass


# ---- geometry (CCIL tools/pendulum_recovery/generate.py: chart, encode, wrap, dist, residual, step, compose) ----

def chart(z):
    return np.arctan2(z[..., 0], z[..., 1]), z[..., 2]


def encode(theta, omega):
    return np.stack([np.sin(theta), np.cos(theta), omega], -1)


def wrap(x):
    return np.mod(x + np.pi, 2 * np.pi) - np.pi


def residual(z, y):
    """The 2-vector [wrap(theta_z - theta_y), (omega_z - omega_y) / sqrt(g)] whose norm is d(z, y)."""
    (tz, wz), (ty, wy) = chart(z), chart(y)
    return np.stack([wrap(tz - ty), (wz - wy) / np.sqrt(G)], -1)


def dist(z, y):
    return np.linalg.norm(residual(z, y), axis=-1)


def e_of(z):
    return wrap(chart(z)[0] - np.pi)


def step(model, z, a):
    """One model step re-encoded through the chart (every state stays on the circle)."""
    nxt = z + model.delta(z, np.asarray(a, np.float64).reshape(-1, 1))
    return encode(*chart(nxt))


def compose(model, z0, U):
    """States (N, M + 1, 3) of actions U (N, M) from z0 (N, 3)."""
    zs, z = [z0], z0
    for i in range(U.shape[1]):
        z = step(model, z, U[:, i])
        zs.append(z)
    return np.stack(zs, 1)


class Exact:
    """The task's exact map as a transition model with the (z, a) -> delta interface of the learned model."""

    def __init__(self):
        self.m = ExactModel(Task())

    def delta(self, z, a):
        return self.m.delta(z, np.asarray(a).reshape(len(z)))


# ---- run identity ----

def identity(smoke: bool) -> dict:
    return {"protocol": PROTOCOL, "commit": pc.git("rev-parse", "HEAD"), "smoke": smoke,
            "data_sha256": pc.DATA_SHA256,
            "tools_sha256": {f: sha256_file(pc.REPO / f) for f in TOOL_FILES}}


def bind(out: Path, ident: dict, resume: bool):
    p = out / "identity.json"
    if p.exists():
        if not resume:
            raise Refused(f"refused: {out} exists (an incomplete run takes --resume)")
        if json.loads(p.read_text()) != ident:
            raise Refused("refused: the run directory was written by another commit, tool or setting")
    else:
        out.mkdir(parents=True, exist_ok=resume)
        atomic_json(p, ident)


# ---- stages ----

def check_starts():
    sets = {k: pc.phys(Task(eval_seed=k).starts()) for k in (*EARLIER_STARTS, *ALL_NEW_STARTS, *SMOKE_STARTS)}
    for a in (*ALL_NEW_STARTS, *SMOKE_STARTS):
        for b in sets:
            if a != b and (sets[a][:, None] == sets[b][None]).all(-1).any():
                raise Refused(f"refused: start set {a} shares a start with {b}")


def stage_data(out: Path, seeds):
    from ccil.data.split import make_split
    from ccil.schema import read_trajectories
    for s in seeds:
        d = out / "data" / f"s{s}"
        if (d / "split.json").exists():
            continue
        d.mkdir(parents=True, exist_ok=True)
        subset(pc.DATA, pc.R, s, d / "subset.npz")
        eps = [e.episode_id for e in read_trajectories(d / "subset.npz").episodes]
        atomic_json(d / "split.json", make_split(eps, pc.VALIDATION_FRACTION, s))
    log("data ready", len(seeds), "seeds")


def demos(out: Path, s: int) -> dict:
    """The seed's training and validation episodes: states (500, 3) and actions (500,), tau_demo, the tree."""
    from ccil.schema import read_trajectories
    d = out / "data" / f"s{s}"
    split = json.loads((d / "split.json").read_text())
    eps = {e.episode_id: e for e in read_trajectories(d / "subset.npz").episodes}
    S = {k: eps[k].observations["state"].astype(np.float64) for k in eps}
    A = {k: eps[k].actions[:, 0].astype(np.float64) for k in eps}
    train = [k for k in eps if k in split["train"]]  # dataset order
    states = np.concatenate([S[k] for k in train])
    tree = pc.tree(states)
    val = np.concatenate([S[k] for k in split["validation"]])
    tau = float(np.quantile(tree.query(pc.phys(val))[0], 0.95))
    return {"split": split, "S": S, "A": A, "train": train, "tree": tree, "tau": tau}


def failure_states(states: np.ndarray, n_keep: int = PER_TRAJECTORY):
    """(failure trajectory mask, flat indices of the thinned failure states) of rollouts (E, T + 1, 3)."""
    E, T1 = states.shape[:2]
    e = np.abs(e_of(states))
    up = np.cos(e) >= UPRIGHT_COS
    hold = up[:, :-1][:, -HOLD:].all(1)  # the scorer's final hold: states 400..499 of a 500-step rollout
    approach = e[:, :-1] < APPROACH
    fail = approach.any(1) & ~hold
    keep = []
    for ep in np.flatnonzero(fail):
        idx = np.flatnonzero(approach[ep])
        if len(idx) > n_keep:
            idx = idx[np.unique(np.round(np.linspace(0, len(idx) - 1, n_keep)).astype(int))]
        keep.append(ep * T1 + idx)
    return fail, (np.concatenate(keep) if keep else np.zeros(0, int))


def stage_failures(out: Path, seeds, episodes: int, sets: tuple):
    for s in seeds:
        path = out / "failures" / f"s{s}.npz"
        if path.exists():
            continue
        path.parent.mkdir(parents=True, exist_ok=True)
        pol, arrays, rec = pc.policy(out, "bc", s), {}, {}
        for name, k in zip(("collection", "held_out"), sets):
            states = Task(eval_seed=k, eval_episodes=episodes).rollouts([pol])[0]["states"]
            fail, idx = failure_states(states)
            x = states.reshape(-1, 3)[idx]
            rec_ok = pc.recovers(Task(), x) if len(x) else np.zeros(0, bool)
            arrays.update({f"{name}/rollouts": states, f"{name}/fail": fail, f"{name}/index": idx, f"{name}/x": x,
                           f"{name}/pi_d_recovers": rec_ok})
            rec[name] = {"eval_seed": k, "episodes": int(len(states)), "failure_trajectories": int(fail.sum()),
                         "failure_states": int(len(x)),
                         "pi_d_recovers_fraction": float(rec_ok.mean()) if len(x) else None}
        atomic_npz(path, **arrays, record=json.dumps(rec))
        log("failures", s, json.dumps(rec))


def failures(out: Path, s: int):
    z = np.load(out / "failures" / f"s{s}.npz")
    return {k: z[k] for k in z.files if k != "record"}, json.loads(str(z["record"]))


def stage_models(out: Path, seeds, D: dict, device: str) -> dict:
    from ccil.models.mlp import MLPTransitionModel
    from ccil.models.mlp_batched import FitJob, fit_group
    d = out / "models"
    paths = {(s, r): d / f"s{s}-{r}.npz" for s in seeds for r in ("primary", "check")}
    if not all(p.exists() for p in paths.values()):
        d.mkdir(parents=True, exist_ok=True)
        jobs, keys = [], []
        for s in seeds:
            u = D[s]
            z = np.concatenate([u["S"][k][:-1] for k in u["train"]])
            a = np.concatenate([u["A"][k][:-1] for k in u["train"]])[:, None]
            zn = np.concatenate([u["S"][k][1:] for k in u["train"]])
            for role, fit_seed in (("primary", s), ("check", s + CHECK_SEED_OFFSET)):
                jobs.append(FitJob(z.astype(np.float32), a.astype(np.float32), zn.astype(np.float32), fit_seed))
                keys.append((s, role))
        with Heartbeat(f"fit {len(jobs)} transition models"):
            res = fit_group(jobs, DYNAMICS, device=device, deterministic=True)
        for key, (model, rec) in zip(keys, res):
            tmp = paths[key].with_suffix(".tmp.npz")
            model.save(tmp)
            tmp.replace(paths[key])
            atomic_json(paths[key].with_suffix(".json"), rec)
    return {k: MLPTransitionModel.load(p, device) for k, p in paths.items()}


def held_out_error(model, u: dict, M: int) -> float:
    """eps_M: q95 of d(F_M(x_t, U_t), x_{t+M}) over every validation window (reported, not gated)."""
    z0, U, zt = [], [], []
    for k in u["split"]["validation"]:
        S, A = u["S"][k], u["A"][k]
        for t in range(len(S) - M):
            z0.append(S[t]), U.append(A[t:t + M]), zt.append(S[t + M])
    return float(np.quantile(dist(compose(model, np.array(z0), np.array(U))[:, -1], np.array(zt)), 0.95))


# ---- the forward segment solver ----

def targets(u: dict, xf: np.ndarray, M: int):
    """Per failure state: the nearest training state with t + M <= 499 (dataset order, then t), its target x_{t+M}
    and its reference actions u_t .. u_{t+M-1}."""
    from scipy.spatial import cKDTree
    S = np.concatenate([u["S"][k][:500 - M] for k in u["train"]])
    tgt = np.concatenate([u["S"][k][M:500] for k in u["train"]])
    ref = np.concatenate([np.stack([u["A"][k][i:i + 500 - M] for i in range(M)], 1) for k in u["train"]])
    _, j = cKDTree(pc.phys(S), boxsize=[2 * np.pi, 1e9]).query(pc.phys(xf))
    return tgt[j], ref[j], j


def solve(model, xf: np.ndarray, zt: np.ndarray, Uref: np.ndarray):
    """Batched projected Levenberg-Marquardt for min ||r(U)||^2 + LAMBDA ||U - Uref||^2 over U in [-3, 3]^M.
    Returns (U, states (N, M + 1, 3), final ||r||)."""
    N, M = Uref.shape
    U = Uref.copy()

    def cost_of(U_):
        st = compose(model, xf, U_)
        r_ = residual(st[:, -1], zt)
        return st, r_, (r_ ** 2).sum(1) + LAMBDA * ((U_ - Uref) ** 2).sum(1)

    states, r, cost = cost_of(U)
    mu = np.full(N, MU0)
    eye = np.eye(M)
    for _ in range(ITERATIONS):
        P = np.concatenate([U[:, None] + FD_STEP * eye[None], U[:, None] - FD_STEP * eye[None]], 1)  # (N, 2M, M)
        end = compose(model, np.repeat(xf, 2 * M, 0), P.reshape(-1, M))[:, -1].reshape(N, 2 * M, 3)
        rp = residual(end[:, :M], zt[:, None])
        rm = residual(end[:, M:], zt[:, None])
        J = ((rp - rm) / (2 * FD_STEP)).transpose(0, 2, 1)  # (N, 2, M)
        g = np.einsum("nkm,nk->nm", J, r) + LAMBDA * (U - Uref)
        H = np.einsum("nkm,nkl->nml", J, J) + (LAMBDA + mu)[:, None, None] * eye
        with np.errstate(all="ignore"):
            dU = np.linalg.solve(H, -g[..., None])[..., 0]
        Un = np.clip(U + np.nan_to_num(dU), -BOUND, BOUND)
        sn, rn, cn = cost_of(Un)
        better = np.isfinite(cn) & (cn < cost)
        U[better], states[better], r[better], cost[better] = Un[better], sn[better], rn[better], cn[better]
        mu = np.where(better, mu / 3, mu * 3)
    return U, states, np.linalg.norm(r, axis=1)


def one_step_errors(model, states, U):
    """max_i d(model(x_i, u_i), x_{i+1}) per segment, and per row."""
    N, M = U.shape
    x = states[:, :-1].reshape(-1, 3)
    per_row = dist(step(model, x, U.reshape(-1)), states[:, 1:].reshape(-1, 3)).reshape(N, M)
    return per_row.max(1), per_row


def stage_generate(out: Path, seeds, ladder, D: dict, models: dict, pols: dict):
    exact = Exact()
    for s in seeds:
        u, (F, _) = D[s], failures(out, s)
        xf = F["collection/x"]
        for M in ladder:
            for arm in ARMS:
                path = out / "generate" / f"s{s}-M{M}-{arm}.npz"
                if path.exists():
                    continue
                path.parent.mkdir(parents=True, exist_ok=True)
                t0 = time.time()
                gen = exact if arm == "exact" else models[(s, "primary")]
                zt, Uref, j = targets(u, xf, M)
                U, states, res = solve(gen, xf, zt, Uref) if len(xf) else (Uref, np.zeros((0, M + 1, 3)), np.zeros(0))
                finite = np.isfinite(states).all((1, 2)) & np.isfinite(U).all(1) & np.isfinite(res)
                accepted = finite & (res <= TRUST * u["tau"])
                reason = np.where(~finite, "non_finite", np.where(accepted, "", "residual"))
                ex_seg, ex_row = one_step_errors(exact, states, U)
                arrival = dist(compose(exact, xf, U)[:, -1], zt) if len(xf) else np.zeros(0)
                ck_seg, _ = one_step_errors(models[(s, "check")], states, U)
                ck_comp = dist(compose(models[(s, "check")], xf, U)[:, -1], zt) if len(xf) else np.zeros(0)
                rows = states[:, :-1].reshape(-1, 3)
                ra = U.reshape(-1)
                pi_d = expert(rows).reshape(-1) if len(rows) else np.zeros(0)
                pi_bc = np.asarray(pols[s](rows), np.float64).reshape(-1) if len(rows) else np.zeros(0)
                atomic_npz(path, x_f=xf, target=zt, target_index=j, U_ref=Uref, U=U, states=states, residual=res,
                           finite=finite, accepted=accepted, reason=reason, exact_one_step=ex_seg,
                           exact_one_step_row=ex_row, exact_arrival=arrival, check_one_step=ck_seg,
                           check_composed=ck_comp,
                           row_d_demo=u["tree"].query(pc.phys(rows))[0] if len(rows) else np.zeros(0),
                           row_e=e_of(rows), row_label_error=np.abs(ra - pi_d), row_bc_error=np.abs(pi_bc - pi_d),
                           tau_demo=u["tau"])
                log("generate", s, f"M={M}", arm, f"accepted {int(accepted.sum())}/{len(xf)}",
                    f"{time.time() - t0:.1f} s")


# ---- gates ----

def gates(out: Path, s: int, M: int, arm: str, held_out: np.ndarray) -> dict:
    z = np.load(out / "generate" / f"s{s}-M{M}-{arm}.npz")
    tau, acc, N = float(z["tau_demo"]), z["accepted"], len(z["accepted"])
    racc = np.repeat(acc, M)
    rows = z["states"][:, :-1].reshape(-1, 3)
    replay_ok = (z["exact_one_step"] <= TRUST * tau) & (z["exact_arrival"] <= TRUST * tau)
    kappa_seg = np.maximum(z["check_one_step"], z["check_composed"])
    bounds = bool((np.abs(z["U"][acc]) <= BOUND).all())
    out_ = {"failure_states": N, "accepted": int(acc.sum()), "tau_demo": tau}
    frac = lambda m: float(m.mean()) if m.size else None  # noqa: E731
    out_["replay_fraction"] = frac(replay_ok[acc])
    out_["G1"] = bool(bounds and acc.any() and out_["replay_fraction"] >= REPLAY_MIN)
    out_["arrival_fraction"] = acc.sum() / N if N else None
    out_["G2"] = bool(N and out_["arrival_fraction"] >= ARRIVAL_MIN)
    if racc.any() and len(held_out):
        d = pc.tree(rows[racc]).query(pc.phys(held_out))[0]
        out_["coverage_fraction"] = float((d <= RHO_COV * tau).mean())
    else:
        out_["coverage_fraction"] = None
    out_["G3"] = bool(out_["coverage_fraction"] is not None and out_["coverage_fraction"] >= COVERAGE_MIN)
    out_["novelty_median_d_demo_over_tau"] = float(np.median(z["row_d_demo"][racc]) / tau) if racc.any() else None
    out_["G4"] = bool(racc.any() and out_["novelty_median_d_demo_over_tau"] >= NOVELTY)
    out_["kappa_over_tau"] = float(np.quantile(kappa_seg[acc], 0.95) / tau) if acc.any() else None
    out_["G5"] = bool(acc.any() and out_["kappa_over_tau"] <= TRUST)
    trusted = acc & (kappa_seg <= TRUST * tau)
    out_["trusted_segments"] = int(trusted.sum())
    out_["trusted_replay_fraction"] = frac(replay_ok[trusted])
    out_["G6"] = bool(trusted.any() and out_["trusted_replay_fraction"] >= REPLAY_MIN)
    cap = racc & (np.abs(z["row_e"]) < CAPTURE)
    le, be = z["row_label_error"][cap], z["row_bc_error"][cap]
    out_.update({"capture_rows": int(cap.sum()), "label_within_tol_fraction": frac(le <= LABEL_TOL),
                 "label_error_median": float(np.median(le)) if cap.any() else None,
                 "bc_error_median": float(np.median(be)) if cap.any() else None})
    out_["G7"] = bool(cap.sum() >= MIN_CAPTURE_ROWS and out_["label_within_tol_fraction"] >= LABEL_FRACTION
                      and out_["label_error_median"] <= BC_RATIO * out_["bc_error_median"])
    out_["exact_replay_residual_max"] = float(z["exact_arrival"][acc].max()) if acc.any() else None
    return out_


ARM_GATES = {"learned": ("G1", "G2", "G3", "G4", "G5", "G6", "G7"), "exact": ("G2", "G3", "G4", "G7")}


def select(per: dict, seeds, ladder) -> dict:
    passes = {arm: {M: sum(all(per[s][M][arm][g] for g in ARM_GATES[arm]) for s in seeds) >= min(PASS_SEEDS, len(seeds))
                    for M in ladder} for arm in ARMS}
    learned = [M for M in ladder if passes["learned"][M]]
    if not any(passes["exact"].values()):
        stop = "stop 1: no M passes for the exact arm; the targeted geometry is inadequate"
    elif not learned:
        stop = "stop 2: the exact arm passes but the learned arm passes no M; dynamics learning is the bottleneck"
    else:
        stop = None
    sel = learned[0] if learned else None
    return {"passes": {arm: {str(M): v for M, v in p.items()} for arm, p in passes.items()}, "selected_M": sel,
            "exact_passes_at_selected_M": None if sel is None else passes["exact"][sel], "stop": stop,
            "stage_b_runs": stop is None}


# ---- data-only discovery diagnostic ----

def stage_discovery(out: Path, seeds, D: dict, models: dict, pols: dict, episodes: int, collection: int) -> dict:
    res = {}
    for s in seeds:
        path = out / "discovery" / f"s{s}.npz"
        u, (F, _) = D[s], failures(out, s)
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            x = Task(eval_seed=collection, eval_episodes=episodes).starts()
            traj = [x]
            for _ in range(Task().horizon):
                a = np.clip(np.asarray(pols[s](x), np.float64).reshape(-1), -BOUND, BOUND)
                x = step(models[(s, "primary")], x, a)
                traj.append(x)
            atomic_npz(path, rollouts=np.stack(traj, 1))
        model_roll = np.load(path)["rollouts"]
        fail, idx = failure_states(model_roll)
        found = model_roll.reshape(-1, 3)[idx]
        held = F["held_out/x"]
        tau = u["tau"]
        cov = (float((pc.tree(found).query(pc.phys(held))[0] <= RHO_COV * tau).mean())
               if len(found) and len(held) else None)
        gap = dist(model_roll, F["collection/rollouts"]) > TRUST * tau
        div = np.where(gap.any(1), gap.argmax(1), gap.shape[1])
        res[str(s)] = {"model_failure_trajectories": int(fail.sum()), "model_failure_states": int(len(found)),
                       "real_failure_trajectories": int(F["collection/fail"].sum()),
                       "held_out_failures_covered": cov,
                       "divergence_step_q05_q50_q95": np.quantile(div, [0.05, 0.5, 0.95]).tolist()}
    ok = sum((r["held_out_failures_covered"] or 0) >= DISCOVERY_MIN for r in res.values())
    verdict = ("discovery reproduces the failures" if ok >= min(PASS_SEEDS, len(seeds))
               else "discovery is a separate bottleneck")
    return {"per_seed": res, "seeds_reproducing": ok, "verdict": verdict}


def artifacts(out: Path) -> dict:
    """sha256 of every stage artifact (the stage directories only; no temporary or foreign file)."""
    files = [p for d in ("data", "policies", "failures", "models", "generate", "discovery")
             for p in (out / d).rglob("*") if p.is_file() and ".tmp" not in str(p.relative_to(out))]
    return {str(p.relative_to(out)): sha256_file(p) for p in sorted(files)}


def step_budget(out: Path, seeds) -> dict:
    sel = [json.loads((out / "policies" / "bc" / f"s{s}" / "training.json").read_text())["validation"]["selected_step"]
           for s in seeds]
    S = int(math.ceil(2 * float(np.median(sel)) / 1000) * 1000)
    return {"best_validation_steps": dict(zip(map(str, seeds), sel)), "S": S, "validation_interval_steps": S // 100}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    if pc.git("status", "--porcelain", "--untracked-files=no"):
        raise Refused("refused: the working tree has uncommitted changes")
    if sha256_file(pc.DATA) != pc.DATA_SHA256:
        raise Refused("refused: the dataset is not the canonical continuous dataset")
    seeds, ladder, episodes, sets = (((SMOKE_SEED,), LADDER[:2], 20, SMOKE_STARTS) if a.smoke
                                     else (DESIGN_SEEDS, LADDER, 100, (COLLECTION, HELD_OUT)))
    ident = identity(a.smoke)
    bind(a.out, ident, a.resume)
    lock = (a.out / ".lock").open("w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        raise Refused(f"refused: another process is running in {a.out}") from None
    if (a.out / "stage-a.json").exists():
        raise Refused("refused: stage-a.json exists; the run is complete")
    t0 = time.time()
    log("run", json.dumps(ident))
    check_starts()
    stage_data(a.out, seeds)
    pc.train(a.out, "bc", [(s, pc.supervision(a.out, s, "train"), pc.supervision(a.out, s, "validation"))
                           for s in seeds], a.device)
    pols = {s: pc.policy(a.out, "bc", s) for s in seeds}
    with Heartbeat("failures"):
        stage_failures(a.out, seeds, episodes, sets)
    D = {s: demos(a.out, s) for s in seeds}
    models = stage_models(a.out, seeds, D, a.device)
    with Heartbeat("generate"):
        stage_generate(a.out, seeds, ladder, D, models, pols)
    with Heartbeat("discovery"):
        disc = stage_discovery(a.out, seeds, D, models, pols, episodes, sets[0])
    per = {s: {M: {arm: gates(a.out, s, M, arm, failures(a.out, s)[0]["held_out/x"]) for arm in ARMS}
               for M in ladder} for s in seeds}
    eps = {str(s): {str(M): held_out_error(models[(s, "primary")], D[s], M) for M in ladder} for s in seeds}
    doc = {**ident, "seeds": list(seeds), "ladder": list(ladder), "start_sets": list(sets),
           "tau_demo": {str(s): D[s]["tau"] for s in seeds},
           "failures": {str(s): failures(a.out, s)[1] for s in seeds},
           "gates": {str(s): {str(M): per[s][M] for M in ladder} for s in seeds},
           "held_out_model_error": eps, **select(per, seeds, ladder), "discovery": disc,
           "stage_b_step_budget": step_budget(a.out, seeds),
           "artifacts_sha256": artifacts(a.out),
           "seconds": round(time.time() - t0, 1)}
    atomic_json(a.out / "stage-a.json", doc)
    log("selection", json.dumps({k: doc[k] for k in ("passes", "selected_M", "stop")}))
    log("discovery", disc["verdict"], "S", doc["stage_b_step_budget"]["S"])
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Refused as e:
        print(e, file=sys.stderr)
        sys.exit(2)
