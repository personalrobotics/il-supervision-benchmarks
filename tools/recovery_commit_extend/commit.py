"""#85 commit-and-extend recovery on Pendulum: the arms on one unit (recovery-commit-extend.md, commit.json).

#82's Study (tools/recovery_fixed_horizon/fixedh.py) supplies the system, model 0, the frozen BC, phase tracking,
#76's segment execution (the any-horizon arm), #82's fixed-H wrapper (the comparator, unchanged), the declared
recovery's audit and the base reproduction check. The commit-and-extend wrapper is new: #79's switch whose recovery
commits to a fixed ladder target, counts its horizon down, and moves to a later target only after the current one
fails. Every commitment and every replan must pass the authority test before any of its actions is executed.

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
sys.path.append(str(HERE.parent / "recovery_fixed_horizon"))  # after this directory: run and audit are #85's
import fixedh as F  # noqa: E402

W, C, S3 = F.W, F.C, F.S3
SPEC = json.loads((HERE / "commit.json").read_text())
INVENTORY = json.loads((HERE / "inventory.json").read_text())
MAX_TARGET, LAST, T, WINDOW = SPEC["max_target"], W.LAST, W.T, W.H
STEP_BUDGET, ENTRY_BUDGET = W.STEP_BUDGET, W.ENTRY_BUDGET
ARMS = tuple(SPEC["arms"])
DYN = {"fixed_learned": 0, "ce_learned": 0, "ce_exact": "exact"}


def key(smoke: bool, *rest) -> list:
    return [171, 7, *([C.SMOKE_KEY] if smoke else []), *rest]


class Live:
    def __init__(self, st, dyn, seed_key):
        self.st, self.dyn, self.seed_key = st, dyn, seed_key

    def solve(self, x, y, starts):
        sy = self.st.sy
        res = lambda U: sy.residual(self.st.rollout(self.dyn, x, U)[:, -1], y)  # noqa: E731
        sols = C.lm(res, starts.shape[1], sy.ubar, starts)
        miss = sy.metric(self.st.rollout(self.dyn, x, sols)[:, -1], y)
        return sols[int(np.argmin(np.where(np.isfinite(miss), miss, np.inf)))]

    def first(self, x, y, demo, k: int, M: int):
        seed = lambda r: C.rng(*key(self.st.smoke, *self.seed_key, k, M, r))  # noqa: E731
        return self.solve(x, y, C.lm_starts(demo, self.st.sy.ubar, seed))

    def replan(self, x, y, prev, demo):
        ubar = self.st.sy.ubar
        return self.solve(x, y, np.stack([prev[1:], np.clip(demo, -ubar, ubar)]))


class Replay:
    """The stored plan of every planner call of every recovery episode, in call order."""

    def __init__(self, recoveries: list):
        self.q = [c["plan"] for g in recoveries for c in g["calls"]]

    def _next(self, H):
        p = self.q.pop(0)
        return np.full(H, np.nan) if p is None else np.asarray(p, np.float64)

    def first(self, x, y, demo, k, M):
        return self._next(len(demo))

    def replan(self, x, y, prev, demo):
        return self._next(len(demo))


class Study(F.Study):
    def __init__(self, set_name: str, pendulum: Path, policy: Path, stage3: Path, stage2: Path, stage4: Path):
        super().__init__("smoke" if set_name == "smoke" else "study", pendulum, policy, stage3, stage2, stage4)
        self.set_name = set_name
        text = (W.HERE.parent.parent / SPEC["tolerances"]["path"]).read_text()
        if hashlib.sha256(text.encode()).hexdigest() != SPEC["tolerances"]["sha256"]:
            sys.exit("refused: #79's calibration is not the one commit.json pins")
        if set_name == "smoke":
            self.arms = {"start": ARMS, "control": ARMS[1:]}
            return
        if INVENTORY["pendulum_sha256"] != S3.sha256(pendulum) or INVENTORY["epsilon"] != self.sy.eps:
            sys.exit("refused: the inventory is not this dataset's")
        self.units = [("fresh", s["start"]) for s in INVENTORY["starts"]] + \
                     [("fresh_control", c["control"]) for c in INVENTORY["controls"]]
        stage = SPEC["stages"][set_name]
        self.arms = {"start": tuple(stage["starts"]), "control": tuple(stage["controls"])}

    # --- episode inputs ---------------------------------------------------------------------------------------
    def episode_inputs(self, unit) -> dict:
        kind, uid = unit
        if kind in ("start", "control"):
            return super().episode_inputs(unit)
        if kind == "fresh":
            s = INVENTORY["starts"][uid]
            X, U = self.held[s["episode"]]
            return {"kind": kind, "unit": uid, "population": "fresh", "episode": s["episode"], "t": s["t"],
                    "radius": s["radius"], "x0": np.asarray(s["x0"], np.float64), "X": X, "U": U, "seed": [4, uid]}
        c = INVENTORY["controls"][uid]
        X, U = self.held[c["episode"]]
        if c["x0"] != C.floats(X[c["t"]]):
            raise ValueError(f"fresh control {uid} is not its demonstrated state")
        return {"kind": kind, "unit": uid, "population": "fresh_control", "episode": c["episode"], "t": c["t"],
                "radius": None, "x0": X[c["t"]].copy(), "X": X, "U": U, "seed": [5, uid]}

    # --- arms 3 and 4: commit and extend ------------------------------------------------------------------------
    def authority(self, dyn, plan, x, y):
        """The authority test: finite actions, predicted states in the envelope, predicted endpoint within epsilon."""
        if not np.isfinite(plan).all():
            return "solver", {}
        pred = self.rollout(dyn, x, plan[None])[0]
        miss = float(self.sy.metric(pred[-1], y))
        env = self.sy.check_envelope(pred)
        why = "plan_envelope" if not env else ("predicted_miss" if miss > self.sy.eps else None)
        return why, {"predicted_miss": miss, "predicted_envelope": env}

    def ce(self, ep, dyn, planner) -> dict:
        sy, tol = self.sy, self.tol
        X, U, t = ep["X"], ep["U"], ep["t"]
        n = T - t
        x, p, k, mode = ep["x0"], t, 0, "base"
        states, actions, modes, phases, dist = [C.floats(x)], [], [], [], []
        transitions, recoveries, cur, window = [], [], None, []
        entries, outcome, departure, plan, M, p0 = 0, None, None, None, None, None

        def call(kind, pl, Mx):
            j = p0 + Mx
            why, pv = self.authority(dyn, pl, x, X[j])
            cur["calls"].append({"k": k, "kind": kind, "M": Mx, "target": j, "H": j - p,
                                 "plan": None if why == "solver" else C.floats(pl), **pv, "passed": why is None,
                                 "failure": why})
            return why

        def commit(after):
            """Commit to the first unexpired target beyond M = after that passes the authority test."""
            for Mx in sy.ladder:
                if Mx <= after or p0 + Mx <= p:
                    continue
                pl = planner.first(x, X[p0 + Mx], U[p:p0 + Mx], k, Mx)
                if call("commit", pl, Mx) is None:
                    cur["targets"].append({"M": Mx, "target": p0 + Mx, "k": k, "end": None})
                    return Mx, pl
            return None, None

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
                if p + MAX_TARGET > LAST:
                    outcome = "insufficient_continuation"
                    break
                p0 = p
                cur = {"k": k, "phase": p, "x": C.floats(x), "calls": [], "targets": [], "actions": [],
                       "states": [C.floats(x)], "D": [], "base_actions": [], "close": [], "agree": [],
                       "failure": None, "handoff": False}
                recoveries.append(cur)
                M, plan = commit(0)
                if M is None:
                    cur["failure"] = outcome = "no_feasible_plan"
                    break
                entries += 1
                mode, window = "recover", []
                transitions.append({"k": k, "to": "recover", "phase": p, "D": D})
            # recover: execute the first action of the plan that passed the authority test
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
                cur["targets"][-1]["end"] = "handoff"
                mode = "base"
                transitions.append({"k": k, "to": "base", "phase": p, "D": float(sy.metric(x, X[p]))})
                self.audit_recovery(cur, X)
                continue
            if len(cur["actions"]) >= STEP_BUDGET:
                cur["failure"] = outcome = "timeout"
                break
            if k >= n:
                break
            if p >= p0 + M:
                cur["targets"][-1]["end"] = "expired"
                M, plan = commit(M)
            else:
                pl = planner.replan(x, X[p0 + M], plan, U[p:p0 + M])
                why = call("replan", pl, M)
                if why is None:
                    plan = pl
                else:
                    cur["targets"][-1]["end"] = why
                    M, plan = commit(M)
            if M is None:
                cur["failure"] = outcome = "targets_exhausted"
                break
        if outcome is None:
            outcome = "complete" if mode == "base" else "episode_end_in_recovery"
        for g in recoveries:
            g.setdefault("declared", False)
            g.setdefault("audited", False)
        first = recoveries[0] if recoveries else None
        recovered = bool(first and first["k"] == 0 and first["handoff"] and first["audited"])
        return self.finish(ep, n, outcome, departure, transitions, actions, modes, phases, dist, states) | {
            "recoveries": recoveries, "recovered": recovered}

    # --- recovery advantage (descriptive) -----------------------------------------------------------------------
    def advantage(self, ep, a: dict) -> dict:
        """#73's measure: upright steps under the wrapper minus under the BC alone, from the start through step 500,
        on the state before each action, and the final two-second hold. An abstained episode is continued by the
        frozen BC from the abstention state for this measure only."""
        t, n = ep["t"], T - ep["t"]
        m = len(a["actions"])
        up = [bool(v) for v in S3.upright(np.asarray(a["states"][:m], np.float64))]
        x = np.asarray(a["states"][m], np.float64)
        for _ in range(m, n):
            up.append(bool(S3.upright(x)))
            x = self.step(x, self.base(x))
        bc, x = [], ep["x0"]
        for _ in range(n):
            bc.append(bool(S3.upright(x)))
            x = self.step(x, self.base(x))
        h = max(0, S3.HOLD_FROM - t)
        return {"continued_by_bc": n - m, "upright": sum(up), "bc_upright": sum(bc), "delta": sum(up) - sum(bc),
                "hold": all(up[h:]), "bc_hold": all(bc[h:])}

    # --- one unit, every arm ------------------------------------------------------------------------------------
    def run_unit(self, unit, stored: dict | None = None) -> dict:
        """This set's arms on one unit: live when stored is None, else replayed from the stored plans."""
        ep = self.episode_inputs(unit)
        self._ep = ep
        rec = {k: v for k, v in ep.items() if k not in ("X", "U", "x0", "seed")}
        rec["x0"] = C.floats(ep["x0"])
        rec["arms"] = {}
        for arm in self.arms["control" if ep["kind"] in ("control", "fresh_control") else "start"]:
            if arm == "any_horizon":
                if stored is None:
                    out = self.any_horizon(ep, lambda M: W.Live(self, self.smoke))
                else:
                    rp = W.Replay(stored["arms"][arm]["segments"])
                    out = self.any_horizon(ep, lambda M: rp)
            else:
                dyn = DYN[arm]
                if arm == "fixed_learned":
                    planner = F.Live(self, 0, [4, 0, *ep["seed"]]) if stored is None else \
                        F.Replay(stored["arms"][arm]["recoveries"])
                    out = self.fixed(ep, 0, planner)
                else:
                    planner = Live(self, dyn, [4, 0 if dyn == 0 else 1, *ep["seed"]]) if stored is None else \
                        Replay(stored["arms"][arm]["recoveries"])
                    out = self.ce(ep, dyn, planner)
                out["advantage"] = self.advantage(ep, out)
            rec["arms"][arm] = out
        rec["tolerances_sha256"] = self.tol["sha256"]
        return rec
