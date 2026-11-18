"""#76 receding-horizon recovery on Pendulum: one start's executions and their evaluation (recovery-receding.md,
receding.json).

#73's Study (tools/recovery_learned/learned.py), built on the #73 run directory of the same set, supplies Stage 2's
system and starts (by provenance and content), model 0 in float64, the frozen BC and the branches for the diagnostic
advantage. Stage 2's LM solver and scheduler are reused unchanged.

A start is executed (the only use of the solver) and then evaluated from its executed actions and observed states
alone, so the audit recomputes everything but the planning. Each (arm, M) is one independent execution of up to M
exact steps from x_0 toward y_M with a shrinking horizon. The exact arm plans through the evaluator map, the learned
arm through model 0; both observe only the states the evaluator map returns.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "recovery_learned"))  # after this directory: run and audit are #76's
import learned as S3  # noqa: E402

C = S3.C
SPEC = json.loads((HERE / "receding.json").read_text())
ARMS = SPEC["arms"]
DEC = SPEC["decision"]
AUDIT = tuple(SPEC["audit"])


def key(smoke: bool, *rest) -> list:
    return [171, 4, *([C.SMOKE_KEY] if smoke else []), *rest]


def stage3_record_ok(stage3: Path, set_name: str) -> bool:
    return S3.sha256(stage3 / "record.json") == SPEC["stage3"]["record_sha256"][set_name]


class Study:
    def __init__(self, smoke: bool, pendulum: Path, policy: Path, stage3: Path, stage2: Path):
        set_name = "smoke" if smoke else "study"
        if S3.sha256(stage2 / "record.json") != SPEC["stage2"]["record_sha256"][set_name]:
            sys.exit(f"refused: {stage2}/record.json is not Stage 2's {set_name} record")
        if not stage3_record_ok(stage3, set_name):
            sys.exit(f"refused: {stage3}/record.json is not #73's {set_name} record")
        m0 = json.loads((stage3 / "identity.json").read_text())["identity"]["models"]["0"]
        if [m0["weights_sha256"], m0["file_sha256"]] != [SPEC["stage3"]["model0_weights_sha256"],
                                                         SPEC["stage3"]["model0_file_sha256"]]:
            sys.exit("refused: model 0 is not #73's")
        self.s3 = S3.Study(smoke, pendulum, policy, stage3, stage2)  # refuses unless every model matches its hashes
        self.smoke, self.sy, self.units = smoke, self.s3.sy, self.s3.units
        self.stage3 = stage3
        self.stage3_starts = json.loads((stage3 / "record.json").read_text())["starts_sha256"]

    def plan_rollout(self, arm: str, x, U) -> np.ndarray:
        """(R, h) actions -> (R, h + 1, 3) states from x under the arm's planning dynamics."""
        return self.sy.rollout(x, U) if arm == "exact" else self.s3.model_rollout(0, x, U)

    def exact_step(self, x, u: float) -> np.ndarray:
        return self.sy.rollout(x, np.array([[u]]))[0, 1]

    # --- execution --------------------------------------------------------------------------------------------
    def execute(self, arm: str, s: int, a: int, x0, M: int) -> dict:
        sy = self.sy
        y, demo = sy.target(a, M), sy.anchor(a)[1][:M]
        x, plan = np.asarray(x0, np.float64), None
        actions, states, predicted, failure = [], [x], [], None
        for j in range(M):
            res = lambda U, x=x: sy.residual(self.plan_rollout(arm, x, U)[:, -1], y)  # noqa: E731
            if j == 0:
                seed = lambda k: C.rng(*key(self.smoke, ARMS[arm], s, M, k))  # noqa: E731
                starts = C.lm_starts(demo, sy.ubar, seed)
            else:
                starts = np.stack([plan[1:], np.clip(demo[j:], -sy.ubar, sy.ubar)])
            sols = C.lm(res, M - j, sy.ubar, starts)
            miss = sy.metric(self.plan_rollout(arm, x, sols)[:, -1], y)
            best = int(np.argmin(np.where(np.isfinite(miss), miss, np.inf)))
            plan = sols[best]
            if not np.isfinite(plan[0]):
                failure = "solver"
                break
            x = self.exact_step(x, float(plan[0]))
            actions.append(float(plan[0]))
            predicted.append(float(miss[best]))
            states.append(x)
            if not sy.inside(x):
                failure = "envelope"
                break
        return {"actions": actions, "states": C.floats(states), "predicted_miss": predicted, "failure": failure}

    def solve(self, unit) -> dict:
        """{arm: {M: execution}}: the planning, the only part the audit does not recompute."""
        s, a, _, _ = unit
        x0, _ = self.s3.start(unit)
        return {arm: {str(M): self.execute(arm, s, a, x0, M) for M in self.sy.ladder} for arm in ARMS}

    # --- evaluation -------------------------------------------------------------------------------------------
    def open_loop(self, unit, rec2: dict) -> dict:
        """The 2 x 2's open-loop cells for this start: Stage 2's shortest valid M (exact) and #73's shortest M with a
        model-valid restart that is valid (learned), read from #73's start file by provenance."""
        s = unit[0]
        path = self.stage3 / "starts" / f"{s:03d}.json"
        if hashlib.sha256(path.read_bytes()).hexdigest() != self.stage3_starts[f"{s:03d}"]:
            raise ValueError(f"start {s}'s #73 file is not the one #73's record lists")
        rec3 = json.loads(path.read_text())
        if rec3["x0"] != rec2["x0"]:
            raise ValueError(f"start {s}'s #73 file does not hold its Stage 2 start")
        ladder = self.sy.ladder
        exact = next((M for M in ladder if (rec2["rungs"].get(str(M)) or {}).get("lm_shooting", {})
                      .get("first_valid") is not None), None)
        learned = next((M for M in ladder if any(g["model_valid"] and g["valid"]
                                                 for g in rec3["rungs"][str(M)]["segments"])), None)
        return {"exact": exact, "learned": learned, "ceiling": rec3["ceiling"]}

    def evaluate(self, unit, executions: dict) -> dict:
        """Everything from the start and its stored executions (actions and observed states)."""
        s, a, r, d = unit
        sy = self.sy
        x0, rec2 = self.s3.start(unit)
        t = sy.anchors[a][1]
        n = S3.T - t
        rec = {"start": s, "anchor": a, "anchor_key": list(sy.anchors[a]), "radius": r, "direction": d,
               "x0": C.floats(x0), "t": int(t), "n": int(n)}
        net0 = self.s3.nets[0]
        arms = {}
        for arm in ARMS:
            ex = {}
            for M in sy.ladder:
                e = executions[arm][str(M)]
                U = np.asarray(e["actions"], np.float64)
                X = np.asarray(e["states"], np.float64)
                y = sy.target(a, M)
                complete = e["failure"] is None and len(U) == M
                observed = float(sy.metric(X[-1], y))
                replay = sy.rollout(x0, U[None])[0]
                refined = sy.refined_rollout(x0, U)
                checks = {"bounds": sy.check_bounds(U), "envelope": sy.check_envelope(X),
                          "replay": bool(replay.shape == X.shape and (replay == X).all()),
                          "endpoint": bool(sy.metric(replay[-1], y) <= sy.eps),
                          "refinement": bool(sy.metric(refined[-1], y) <= sy.eps)}
                one_step = sy.metric(X[:-1] + S3.delta64(net0, X[:-1], U), X[1:]) if len(U) else np.zeros(0)
                ex[str(M)] = {**e, "complete": complete, "observed_miss": observed,
                              "refined_miss": float(sy.metric(refined[-1], y)),
                              "declared": bool(complete and observed <= sy.eps), "audit_checks": checks,
                              "first_audit_failure": next((c for c in AUDIT if not checks[c]), None),
                              "model_one_step_error": C.floats(one_step)}
            M = next((M for M in sy.ladder if ex[str(M)]["declared"]), None)
            recovery = None
            if M is not None:
                g = ex[str(M)]
                b = self.s3.branches("exact", x0, [np.zeros(0), np.asarray(g["actions"])], n)
                recovery = {"M": M, "audited": g["first_audit_failure"] is None,
                            "first_audit_failure": g["first_audit_failure"],
                            "delta": b[1]["upright"] - b[0]["upright"], "hold": b[1]["hold"]}
            arms[arm] = {"recovery": recovery, "executions": ex}
        rec["open_loop"] = self.open_loop(unit, rec2)
        rec["arms"] = arms
        return rec
