"""#82 fixed-horizon recovery on Pendulum: the four arms on one start (recovery-fixed-horizon.md, fixedh.json).

#79's Study (tools/recovery_switch/switch.py) supplies the system, model 0, the frozen BC, phase tracking, the base
reproduction check, #79's wrapper (the shortest-feasible arm, unchanged) and #76's segment execution (the any-horizon
arm). The fixed-H wrapper is new: #79's switch with H = 64 replanned every step and never shortened.

Every arm runs through a planner. The live planner solves; the replay planner returns the stored plans. Everything
else is recomputed by the same code, so the audit re-derives each arm from its stored plans alone.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "recovery_switch"))  # after this directory: run and audit are #82's
import switch as W  # noqa: E402

C, S3 = W.C, W.S3
SPEC = json.loads((HERE / "fixedh.json").read_text())
INVENTORY = json.loads((HERE / "inventory.json").read_text())
HORIZON, LAST, T, WINDOW = SPEC["horizon"], W.LAST, W.T, W.H
STEP_BUDGET, ENTRY_BUDGET = W.STEP_BUDGET, W.ENTRY_BUDGET
ARMS = ("any_horizon", "shortest_feasible", "fixed_learned", "fixed_exact")
DYN = {"fixed_learned": 0, "fixed_exact": "exact"}


def key(smoke: bool, *rest) -> list:
    return [171, 6, *([C.SMOKE_KEY] if smoke else []), *rest]


class Live:
    def __init__(self, st, dyn, seed_key):
        self.st, self.dyn, self.seed_key = st, dyn, seed_key

    def solve(self, x, y, starts):
        sy = self.st.sy
        res = lambda U: sy.residual(self.st.rollout(self.dyn, x, U)[:, -1], y)  # noqa: E731
        sols = C.lm(res, starts.shape[1], sy.ubar, starts)
        miss = sy.metric(self.st.rollout(self.dyn, x, sols)[:, -1], y)
        return sols[int(np.argmin(np.where(np.isfinite(miss), miss, np.inf)))]

    def first(self, x, y, demo, k: int):
        seed = lambda r: C.rng(*key(self.st.smoke, *self.seed_key, k, r))  # noqa: E731
        return self.solve(x, y, C.lm_starts(demo, self.st.sy.ubar, seed))

    def replan(self, x, y, prev, demo):
        ubar = self.st.sy.ubar
        shifted = np.append(prev[1:], np.clip(demo[-1], -ubar, ubar))
        return self.solve(x, y, np.stack([shifted, np.clip(demo, -ubar, ubar)]))


class Replay:
    """The stored plans of every recovery episode, in order: the entry plan, each replan, and a failed plan."""

    def __init__(self, recoveries: list):
        self.q = []
        for g in recoveries:
            self.q.append(g["entry"]["plan"])
            self.q += g["plans"][1:]
            if g["failure"] in ("solver", "plan_envelope"):
                self.q.append(g["failed_plan"])

    def _next(self):
        p = self.q.pop(0)
        return np.full(HORIZON, np.nan) if p is None else np.asarray(p, np.float64)

    def first(self, x, y, demo, k):
        return self._next()

    def replan(self, x, y, prev, demo):
        return self._next()


class Study(W.Study):
    def __init__(self, set_name: str, pendulum: Path, policy: Path, stage3: Path, stage2: Path, stage4: Path):
        super().__init__(set_name == "smoke", pendulum, policy, stage3, stage2, stage4)
        self.set_name = set_name
        text = (W.HERE.parent.parent / SPEC["tolerances"]["path"]).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != SPEC["tolerances"]["sha256"]:
            sys.exit("refused: #79's calibration is not the one fixedh.json pins")
        self.tol = json.loads(text) | {"sha256": SPEC["tolerances"]["sha256"]}
        if set_name == "study":
            if INVENTORY["pendulum_sha256"] != S3.sha256(pendulum) or INVENTORY["epsilon"] != self.sy.eps:
                sys.exit("refused: the inventory is not this dataset's")
            self.units = [("fresh", s["start"]) for s in INVENTORY["starts"]] + \
                         [("fresh_control", c["control"]) for c in INVENTORY["controls"]]
        elif set_name == "smoke":
            self.units = [("start", u[0]) for u in self.r.units] + \
                         [("control", int(e) * 4 + i) for e in sorted(self.controls, key=int) for i in range(4)]
        else:
            self.units = [("start", u[0]) for u in self.r.units]
        self.arms = ("fixed_learned",) if set_name == "development" else ARMS
        self._unit = None

    def rollout(self, dyn, x, U) -> np.ndarray:
        return self.sy.rollout(x, U) if dyn == "exact" else self.s3.model_rollout(dyn, x, U)

    # --- episode inputs ---------------------------------------------------------------------------------------
    def episode_inputs(self, unit) -> dict:
        kind, uid = unit
        if kind in ("start", "control"):
            ep = super().episode_inputs(unit)
            ep["kind"] = kind
            return ep
        if kind == "fresh":
            s = INVENTORY["starts"][uid]
            X, U = self.held[s["episode"]]
            return {"kind": kind, "unit": uid, "population": "fresh", "episode": s["episode"], "t": s["t"],
                    "radius": s["radius"], "x0": np.asarray(s["x0"], np.float64), "X": X, "U": U, "seed": [2, uid]}
        c = INVENTORY["controls"][uid]
        X, U = self.held[c["episode"]]
        if c["x0"] != C.floats(X[c["t"]]):
            raise ValueError(f"fresh control {uid} is not its demonstrated state")
        return {"kind": kind, "unit": uid, "population": "fresh_control", "episode": c["episode"], "t": c["t"],
                "radius": None, "x0": X[c["t"]].copy(), "X": X, "U": U, "seed": [3, uid]}

    # --- arm 1: any horizon (#76's learned arm) -----------------------------------------------------------------
    def any_horizon(self, ep, planner_for) -> dict:
        X, U, t, x0 = ep["X"], ep["U"], ep["t"], ep["x0"]
        segs = []
        for M in self.sy.ladder:
            planner = planner_for(M)
            plan = planner.first(x0, X[t + M], U[t:t + M], [9, *ep["seed"], M])
            g = self.segment(planner, x0, t, M, plan, X, U)
            g.update({"M": M, "tried": [{"M": M, "plan": C.floats(plan)}]})
            segs.append(g)
        M = next((g["M"] for g in segs if g["declared"] and g["audited"]), None)
        return {"segments": segs, "recovered_M": M}

    # --- arms 3 and 4: fixed H ----------------------------------------------------------------------------------
    def fixed(self, ep, dyn, planner) -> dict:
        sy, tol = self.sy, self.tol
        X, U, t = ep["X"], ep["U"], ep["t"]
        n = T - t
        x, p, k, mode = ep["x0"], t, 0, "base"
        states, actions, modes, phases, dist = [C.floats(x)], [], [], [], []
        transitions, recoveries, cur, window = [], [], None, []
        entries, outcome, departure, plan = 0, None, None, None

        def validity(plan, x, p):
            if not np.isfinite(plan).all():
                return "solver", None
            pred = self.rollout(dyn, x, plan[None])[0]
            miss = float(sy.metric(pred[-1], X[p + HORIZON]))
            env = sy.check_envelope(pred)
            return (None if env else "plan_envelope"), {"predicted_miss": miss, "predicted_envelope": env}

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
                if p + HORIZON > LAST:
                    outcome = "insufficient_continuation"
                    break
                plan = planner.first(x, X[p + HORIZON], U[p:p + HORIZON], k)
                why, pv = validity(plan, x, p)
                feasible = bool(why is None and pv["predicted_miss"] <= sy.eps)
                cur = {"k": k, "phase": p, "x": C.floats(x),
                       "entry": {"plan": C.floats(plan) if why != "solver" else None, **(pv or {}),
                                 "feasible": feasible},
                       "plans": [], "actions": [], "states": [C.floats(x)], "D": [], "base_actions": [], "close": [],
                       "agree": [], "predicted_miss": [], "failure": None, "failed_plan": None, "handoff": False}
                recoveries.append(cur)
                if not feasible:
                    cur["failure"] = "no_feasible_plan"
                    outcome = "no_feasible_plan"
                    break
                entries += 1
                mode, window = "recover", []
                transitions.append({"k": k, "to": "recover", "phase": p, "D": D})
                cur["plans"].append(C.floats(plan))
                cur["predicted_miss"].append(pv["predicted_miss"])
            # recover: execute the current plan's first action
            u = float(plan[0])
            d = float(sy.metric(x, X[p]))
            b = self.base(x)
            close, agree = bool(d <= tol["delta_in"]), bool(abs(b - u) / tol["sigma_u"] <= tol["alpha"])
            cur["D"].append(d)
            cur["base_actions"].append(b)
            cur["close"].append(close)
            cur["agree"].append(agree)
            window.append(close and agree)
            phases.append(p)
            dist.append(d)
            actions.append(u)
            modes.append("recover")
            x = self.step(x, u)
            states.append(C.floats(x))
            cur["actions"].append(u)
            cur["states"].append(C.floats(x))
            k += 1
            p += 1
            if not sy.inside(x):
                cur["failure"] = outcome = "envelope"
                break
            if float(sy.metric(x, X[p])) <= tol["delta_in"] and len(window) >= WINDOW and all(window[-WINDOW:]):
                cur["handoff"] = True
                mode = "base"
                transitions.append({"k": k, "to": "base", "phase": p, "D": float(sy.metric(x, X[p]))})
                self.audit_recovery(cur, X)
                continue
            if len(cur["actions"]) >= STEP_BUDGET:
                cur["failure"] = outcome = "timeout"
                break
            if k >= n:
                break
            if p + HORIZON > LAST:
                cur["failure"] = outcome = "insufficient_continuation"
                break
            plan = planner.replan(x, X[p + HORIZON], plan, U[p:p + HORIZON])
            why, pv = validity(plan, x, p)
            if why:
                cur["failure"] = outcome = why
                cur["failed_plan"] = C.floats(plan) if why == "plan_envelope" else None
                break
            cur["plans"].append(C.floats(plan))
            cur["predicted_miss"].append(pv["predicted_miss"])
        if outcome is None:
            outcome = "complete" if mode == "base" else "episode_end_in_recovery"
        for g in recoveries:
            g.setdefault("declared", False)
            g.setdefault("audited", False)
        first = recoveries[0] if recoveries else None
        recovered = bool(first and first["k"] == 0 and first["handoff"] and first["audited"])
        return self.finish(ep, n, outcome, departure, transitions, actions, modes, phases, dist, states) | {
            "recoveries": recoveries, "recovered": recovered}

    def audit_recovery(self, g: dict, X) -> None:
        """#76's segment audit on a recovery episode ending in a handoff: entry state to handoff state."""
        sy = self.sy
        A = np.asarray(g["actions"], np.float64)
        Xs = np.asarray(g["states"], np.float64)
        y = X[g["phase"] + len(A)]
        replayed = sy.rollout(Xs[0], A[None])[0]
        refined = sy.refined_rollout(Xs[0], A)
        g["endpoint_miss"] = float(sy.metric(Xs[-1], y))
        g["refined_miss"] = float(sy.metric(refined[-1], y))
        g["audit_checks"] = {"bounds": sy.check_bounds(A), "envelope": sy.check_envelope(Xs),
                             "replay": bool(replayed.shape == Xs.shape and (replayed == Xs).all()),
                             "endpoint": bool(sy.metric(replayed[-1], y) <= sy.eps),
                             "refinement": bool(sy.metric(refined[-1], y) <= sy.eps)}
        g["declared"] = True
        g["audited"] = all(g["audit_checks"].values())

    def finish(self, ep, n, outcome, departure, transitions, actions, modes, phases, dist, states) -> dict:
        bc = self.bc_alone(ep)
        c = bc["first_departure"]
        reproduced = (departure == c) and actions[:n if c is None else c] == bc["actions"][:n if c is None else c]
        return {"n": n, "outcome": outcome, "departure": departure, "transitions": transitions, "actions": actions,
                "modes": modes, "phases": phases, "distances": dist, "states": states,
                "bc_first_departure": c, "base_reproduced": bool(reproduced),
                "upright": int(S3.upright(np.asarray(states[:len(actions)])).sum()) if outcome == "complete" else None,
                "bc_upright": bc["upright"]}

    # --- one unit, every arm ------------------------------------------------------------------------------------
    def run_unit(self, unit, stored: dict | None = None) -> dict:
        """All of this set's arms on one unit: live when stored is None, else replayed from the stored plans."""
        ep = self.episode_inputs(unit)
        self._ep = ep
        rec = {k: v for k, v in ep.items() if k not in ("X", "U", "x0", "seed")}
        rec["x0"] = C.floats(ep["x0"])
        rec["arms"] = {}
        for arm in self.arms:
            if arm == "any_horizon":
                if ep["kind"] in ("control", "fresh_control"):
                    continue
                if stored is None:
                    out = self.any_horizon(ep, lambda M: W.Live(self, self.smoke))
                else:
                    rp = W.Replay(stored["arms"][arm]["segments"])
                    out = self.any_horizon(ep, lambda M: rp)
            elif arm == "shortest_feasible":
                planner = W.Live(self, self.smoke) if stored is None else \
                    W.Replay(stored["arms"][arm]["segments"])
                out = self.run(unit, planner)
                out = {k: v for k, v in out.items() if k not in rec or k == "outcome"}
            else:
                dyn = DYN[arm]
                planner = Live(self, dyn, [4, 0 if dyn == 0 else 1, *ep["seed"]]) if stored is None else \
                    Replay(stored["arms"][arm]["recoveries"])
                out = self.fixed(ep, dyn, planner)
            rec["arms"][arm] = out
        rec["tolerances_sha256"] = self.tol["sha256"]
        return rec
