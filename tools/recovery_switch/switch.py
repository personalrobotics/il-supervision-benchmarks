"""#79 value-free recovery switch on Pendulum: calibration and one switched episode (recovery-switch.md, switch.json).

#76's Study (tools/recovery_receding/receding.py) supplies Stage 2's system and starts, #73's model 0 in float64 and
the frozen BC; #76's receding-horizon controller is reused with its target and demonstrated actions taken from the
episode's phase instead of a Stage 2 anchor (same solver, restarts, warm start and executed-action rule).

One function runs a switched episode with a planner. The live planner solves; the replay planner returns the stored
plans and executed recovery actions. Everything else (states, phase, distances, base actions, closeness, agreement,
feasibility of each tried plan, every decision, the segment audits) is recomputed by the same code, so the audit
re-derives the whole record from the stored plans alone.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "recovery_receding"))  # after this directory: run and audit are #79's
import receding as R  # noqa: E402

C, S3 = R.C, R.S3
SPEC = json.loads((HERE / "switch.json").read_text())
SPLIT = json.loads((HERE / "split.json").read_text())
H = SPEC["handoff_window"]
STEP_BUDGET, ENTRY_BUDGET = SPEC["budgets"]["recovery_steps"], SPEC["budgets"]["entries"]
T = S3.T  # episode step 500
LAST = T - 1  # last continuation index


def key(smoke: bool, *rest) -> list:
    return [171, 5, 2, *([C.SMOKE_KEY] if smoke else []), *rest]


def smoke_sets() -> tuple:
    controls = {e: SPLIT["calibration"][e] for e in ("0", "1")}
    calibration = {e: v for e, v in SPLIT["calibration"].items() if e not in controls}
    return calibration, controls


def no_plan(k: int, p: int, tried: list) -> dict:
    """A search that found no feasible plan: its tried plans, and no executed step."""
    return {"k": k, "phase": p, "tried": tried, "M": None, "actions": [], "predicted_miss": [], "states": [],
            "failure": None, "D": [], "base_actions": [], "close": [], "agree": [], "endpoint_miss": None,
            "refined_miss": None, "declared": False, "audit_checks": None, "audited": False, "handoff": False}


# --- planners -----------------------------------------------------------------------------------------------------

class Live:
    """Solves: the first solve of every tried M, and each replan of an executed segment."""

    def __init__(self, st, smoke: bool):
        self.st, self.smoke = st, smoke

    def first(self, x, y, demo, seed_key) -> np.ndarray:
        sy = self.st.sy
        res = lambda U: sy.residual(self.st.s3.model_rollout(0, x, U)[:, -1], y)  # noqa: E731
        seed = lambda r: C.rng(*key(self.smoke, *seed_key, r))  # noqa: E731
        sols = C.lm(res, len(demo), sy.ubar, C.lm_starts(demo, sy.ubar, seed))
        miss = sy.metric(self.st.s3.model_rollout(0, x, sols)[:, -1], y)
        return sols[int(np.argmin(np.where(np.isfinite(miss), miss, np.inf)))]

    def replan(self, x, y, plan, demo_rest) -> tuple:
        sy = self.st.sy
        res = lambda U: sy.residual(self.st.s3.model_rollout(0, x, U)[:, -1], y)  # noqa: E731
        starts = np.stack([plan[1:], np.clip(demo_rest, -sy.ubar, sy.ubar)])
        sols = C.lm(res, len(demo_rest), sy.ubar, starts)
        miss = sy.metric(self.st.s3.model_rollout(0, x, sols)[:, -1], y)
        best = int(np.argmin(np.where(np.isfinite(miss), miss, np.inf)))
        return sols[best], float(miss[best])


class Replay:
    """Returns the stored plans and executed recovery actions, in order."""

    def __init__(self, segments: list):
        self.tried = [t for g in segments for t in g["tried"]]
        self.steps = [(g["actions"], g["predicted_miss"], g["failure"]) for g in segments if g["M"] is not None]
        self.i = 0

    def first(self, x, y, demo, seed_key) -> np.ndarray:
        t = self.tried.pop(0)
        if t["M"] != len(demo):
            raise ValueError("stored plans do not follow the search order")
        return np.asarray(t["plan"], np.float64)

    def segment_steps(self):
        out = self.steps[self.i]
        self.i += 1
        return out


# --- the study object ------------------------------------------------------------------------------------------

class Study:
    def __init__(self, smoke: bool, pendulum: Path, policy: Path, stage3: Path, stage2: Path, stage4: Path):
        set_name = "smoke" if smoke else "study"
        if S3.sha256(stage4 / "record.json") != SPEC["stage4"]["record_sha256"][set_name]:
            sys.exit(f"refused: {stage4}/record.json is not #76's {set_name} record")
        self.r = R.Study(smoke, pendulum, policy, stage3, stage2)
        self.smoke, self.sy, self.s3 = smoke, self.r.sy, self.r.s3
        self.stage4 = stage4
        self.stage4_starts = json.loads((stage4 / "record.json").read_text())["starts_sha256"]
        from ccil.schema import read_trajectories
        self.held = {e.episode_id: (e.observations["state"].astype(np.float64), e.actions.astype(np.float64)[:, 0])
                     for e in read_trajectories(pendulum).episodes}
        calibration, controls = smoke_sets() if smoke else (SPLIT["calibration"], SPLIT["control"])
        if set(calibration) & set(controls) or (set(calibration) | set(controls)) & set(self.sy.ep):
            sys.exit("refused: the calibration, control and #70 episodes overlap")
        self.calibration, self.controls = calibration, controls
        self.units = [("start", u[0]) for u in self.r.units] + \
                     [("control", int(e) * 4 + i) for e in sorted(controls, key=int) for i in range(4)]
        self.tol = None

    # one-step pieces, single state
    def step(self, x, u: float) -> np.ndarray:
        return self.sy.rollout(x, np.array([[u]]))[0, 1]

    def base(self, x) -> float:
        return float(np.clip(self.s3.pi(x[None])[0], -self.sy.ubar, self.sy.ubar))

    def track(self, X, x, p: int) -> int:
        cand = X[p:min(p + 2, LAST) + 1]
        return p + int(np.argmin(self.sy.metric(x[None], cand)))

    # --- calibration -------------------------------------------------------------------------------------------
    def calibrate(self) -> dict:
        """sigma_u and alpha from the calibration episodes' demonstrated states alone; delta_in = epsilon and
        delta_out = 2 epsilon (switch.json "tolerances")."""
        eps = sorted(self.calibration, key=int)
        U = np.concatenate([self.held[e][1] for e in eps])
        Xs = np.concatenate([self.held[e][0] for e in eps])
        sigma = float(np.std(U))
        pis = np.clip(np.asarray(self.s3.pi(Xs), np.float64), -self.sy.ubar, self.sy.ubar)
        dis = np.abs(pis - np.clip(U, -self.sy.ubar, self.sy.ubar)) / sigma
        return {"episodes": eps, "transitions": int(len(U)), "delta_in": self.sy.eps, "delta_out": 2 * self.sy.eps,
                "sigma_u": sigma, "alpha": float(np.percentile(dis, SPEC["tolerances"]["alpha_percentile"]))}

    # --- episodes ----------------------------------------------------------------------------------------------
    def episode_inputs(self, unit) -> dict:
        kind, uid = unit
        if kind == "start":
            ru = next(u for u in self.r.units if u[0] == uid)
            x0, _ = self.s3.start(ru)
            i, t = self.sy.anchors[ru[1]]
            rec4 = self.stage4_record(uid)
            if rec4["x0"] != C.floats(x0):
                raise ValueError(f"start {uid}'s #76 file does not hold its Stage 2 start")
            pop = "recover" if rec4["arms"]["learned"]["recovery"] else "unknown"
            X, U = self.sy.ep[i]
            return {"kind": kind, "unit": uid, "population": pop, "episode": str(i), "t": int(t), "radius": ru[2],
                    "x0": np.asarray(x0, np.float64), "X": X, "U": U, "seed": [0, uid]}
        e, idx = str(uid // 4), uid % 4
        t = self.controls[e][idx]
        X, U = self.held[e]
        return {"kind": kind, "unit": uid, "population": "control", "episode": e, "t": int(t), "radius": None,
                "x0": X[t].copy(), "X": X, "U": U, "seed": [1, uid]}

    def stage4_record(self, s: int) -> dict:
        path = self.stage4 / "starts" / f"{s:03d}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.stage4_starts[f"{s:03d}"]:
            raise ValueError(f"start {s}'s #76 file is not the one #76's record lists")
        return json.loads(path.read_text())

    def search(self, planner, ep, x, p: int, k: int, n: int, used: int):
        """The entry rule's search from (x, p): (tried, (M, plan)) or (tried, abstention category)."""
        sy, X, U = self.sy, ep["X"], ep["U"]
        fit = [M for M in sy.ladder if p + M <= LAST and k + M <= n]
        if not fit:
            return [], "continuation_end"
        fit = [M for M in fit if used + M <= STEP_BUDGET]
        if not fit:
            return [], "timeout"
        tried = []
        for M in fit:
            plan = planner.first(x, X[p + M], U[p:p + M], [*ep["seed"], k, M])
            pred = self.s3.model_rollout(0, x, plan[None])[0]
            miss = float(sy.metric(pred[-1], X[p + M]))
            feasible = bool(miss <= sy.eps and sy.check_envelope(pred))
            tried.append({"M": M, "plan": C.floats(plan), "predicted_miss": miss,
                          "predicted_envelope": sy.check_envelope(pred), "feasible": feasible})
            if feasible:
                return tried, (M, plan)
        return tried, "no_feasible_plan"

    def bc_alone(self, ep) -> dict:
        """The frozen BC alone from the same start, with the wrapper's own single-state calls and phase tracking."""
        X, x, p, n = ep["X"], ep["x0"], ep["t"], T - ep["t"]
        actions, first, up = [], None, 0
        for k in range(n):
            if k:
                p = self.track(X, x, p)
            if first is None and float(self.sy.metric(x, X[p])) > self.tol["delta_out"]:
                first = k
            up += int(S3.upright(x))
            u = self.base(x)
            actions.append(u)
            x = self.step(x, u)
        return {"actions": actions, "first_departure": first, "upright": up}

    def run(self, unit, planner) -> dict:
        """One switched episode (recovery-switch.md "The wrapper")."""
        sy, tol = self.sy, self.tol
        ep = self.episode_inputs(unit)
        X, t = ep["X"], ep["t"]
        n = T - t
        x, p, k, mode = ep["x0"], t, 0, "base"
        states, actions, modes, phases, dist = [C.floats(x)], [], [], [], []
        transitions, segments, window = [], [], []
        entries, used, outcome, departure, pending = 0, 0, None, None, None
        while k < n:
            if mode == "base":
                if k:
                    p = self.track(X, x, p)
                D = float(sy.metric(x, X[p]))
                if D <= tol["delta_out"]:
                    u = self.base(x)
                    phases.append(p)
                    dist.append(D)
                    actions.append(u)
                    modes.append("base")
                    x = self.step(x, u)
                    states.append(C.floats(x))
                    k += 1
                    continue
                if departure is None:
                    departure = k
                if entries == ENTRY_BUDGET:
                    outcome = "reentry_limit"
                    break
                tried, res = self.search(planner, ep, x, p, k, n, 0)
                if isinstance(res, str):
                    if tried:
                        segments.append(no_plan(k, p, tried))
                    outcome = res
                    break
                entries += 1
                used, window, mode, pending = 0, [], "recover", (tried, res)
                transitions.append({"k": k, "to": "recover", "phase": p, "D": D})
            if pending is None:
                tried, res = self.search(planner, ep, x, p, k, n, used)
                if isinstance(res, str):
                    if tried:
                        segments.append(no_plan(k, p, tried))
                    outcome = res
                    break
                pending = (tried, res)
            tried, (M, plan) = pending
            pending = None
            g = self.segment(planner, x, p, M, plan, X, ep["U"])
            g.update({"k": k, "phase": p, "tried": tried, "M": M})
            segments.append(g)
            for j, u in enumerate(g["actions"]):
                phases.append(p + j)
                dist.append(g["D"][j])
                actions.append(u)
                modes.append("recover")
                states.append(g["states"][j + 1])
            k += len(g["actions"])
            used += len(g["actions"])
            x = np.asarray(g["states"][-1], np.float64)
            if g["failure"]:
                outcome = g["failure"]
                break
            if not g["declared"]:
                outcome = "endpoint_miss"
                break
            window += [c and a for c, a in zip(g["close"], g["agree"])]
            p += M
            g["handoff"] = bool(g["endpoint_miss"] <= tol["delta_in"] and len(window) >= H and all(window[-H:]))
            if g["handoff"]:
                mode = "base"
                transitions.append({"k": k, "to": "base", "phase": p, "D": g["endpoint_miss"]})
        if outcome is None:
            outcome = "complete"
        bc = self.bc_alone(ep)
        c = bc["first_departure"]
        reproduced = (departure == c) and actions[:n if c is None else c] == bc["actions"][:n if c is None else c]
        rec = {k_: v for k_, v in ep.items() if k_ not in ("X", "U", "x0", "seed")}
        rec.update({"x0": C.floats(ep["x0"]), "n": n, "outcome": outcome, "departure": departure,
                    "transitions": transitions, "segments": segments, "actions": actions, "modes": modes,
                    "phases": phases, "distances": dist, "states": states, "tolerances_sha256": tol["sha256"],
                    "bc_first_departure": c, "base_reproduced": bool(reproduced)})
        rec["upright"] = int(S3.upright(np.asarray(states[:len(actions)])).sum()) if outcome == "complete" else None
        rec["bc_upright"] = bc["upright"]
        return rec

    def segment(self, planner, x0, p, M: int, plan, X, U) -> dict:
        """One #76 receding-horizon execution toward X[p + M] from x0, with closeness and agreement per step."""
        sy, tol = self.sy, self.tol
        y = X[p + M]
        replay = planner.segment_steps() if isinstance(planner, Replay) else None
        x, actions, predicted, states, failure = x0, [], [], [C.floats(x0)], None
        D, close, agree, base = [], [], [], []
        for j in range(M):
            if replay is not None:
                if j >= len(replay[0]):
                    failure = "solver" if replay[2] == "solver" else None
                    break
                u, miss = replay[0][j], replay[1][j]
            elif j == 0:
                u, miss = float(plan[0]), float(sy.metric(self.s3.model_rollout(0, x0, plan[None])[0, -1], y))
            else:
                plan, miss = planner.replan(x, y, plan, U[p + j:p + M])
                u = float(plan[0])
            if not np.isfinite(u):
                failure = "solver"
                break
            d = float(sy.metric(x, X[p + j]))
            b = self.base(x)
            D.append(d)
            base.append(b)
            close.append(bool(d <= tol["delta_in"]))
            agree.append(bool(abs(b - u) / tol["sigma_u"] <= tol["alpha"]))
            x = self.step(x, u)
            actions.append(u)
            predicted.append(float(miss))
            states.append(C.floats(x))
            if not sy.inside(x):
                failure = "envelope"
                break
        A = np.asarray(actions, np.float64)
        Xs = np.asarray(states, np.float64)
        complete = failure is None and len(actions) == M
        endpoint = float(sy.metric(Xs[-1], y))
        replayed = sy.rollout(x0, A[None])[0]
        refined = sy.refined_rollout(x0, A)
        checks = {"bounds": sy.check_bounds(A), "envelope": sy.check_envelope(Xs),
                  "replay": bool(replayed.shape == Xs.shape and (replayed == Xs).all()),
                  "endpoint": bool(sy.metric(replayed[-1], y) <= sy.eps),
                  "refinement": bool(sy.metric(refined[-1], y) <= sy.eps)}
        declared = bool(complete and endpoint <= sy.eps)
        return {"actions": actions, "predicted_miss": predicted, "states": states,
                "failure": failure, "D": D, "base_actions": base, "close": close, "agree": agree,
                "endpoint_miss": endpoint, "refined_miss": float(sy.metric(refined[-1], y)), "declared": declared,
                "audit_checks": checks, "audited": bool(declared and all(checks.values())), "handoff": False}
