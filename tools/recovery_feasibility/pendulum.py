"""#70 Pendulum: a treatment system (recovery-feasibility.md, stage-2.json "pendulum").

The map is ilsb's continuous evaluator map (the task's step, with actions clipped to 3). The M = 1 certificate is the
protocol's Lipschitz-bounded sweep over 6,001 actions; segments are replayed under solve_ivp RK45 at rtol 1e-10,
atol 1e-12 (the refined flow, with the action held over each 0.02 s step).
"""
from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp

import common as C

S = C.SPEC["systems"]["pendulum"]
G = 9.81
SQRT_G = np.sqrt(G)
UBAR = float(S["action_bound"])


def theta(x) -> np.ndarray:
    x = np.asarray(x)
    return np.arctan2(x[..., 0], x[..., 1])


def wrap(d) -> np.ndarray:
    return (d + np.pi) % (2 * np.pi) - np.pi


class System:
    name, solvers, ladder = "pendulum", tuple(S["solvers"]), tuple(S["ladder"])
    refined, continuation = True, False

    def __init__(self, smoke: bool, pendulum: Path):
        from ccil.data.split import make_split
        from ccil.schema import read_trajectories
        from ilsb.tasks import pendulum as P
        if not Path(P.__file__).resolve().is_relative_to(C.ROOT):
            sys.exit(f"refused: ilsb is imported from {P.__file__}, not this checkout (PYTHONPATH=src)")
        if hashlib.sha256(pendulum.read_bytes()).hexdigest() != S["demonstrations"]["sha256"]:
            sys.exit("refused: the Pendulum dataset is not the protocol's")
        self.task = P.Task(variant="continuous")
        with tempfile.TemporaryDirectory() as d:
            P.subset(pendulum, 0.1, 3321, Path(d) / "subset.npz")
            eps = read_trajectories(Path(d) / "subset.npz").episodes
        split = make_split([e.episode_id for e in eps], 0.2, 3321)
        self.ep = {e.episode_id: (e.observations["state"].astype(np.float64), e.actions.astype(np.float64)[:, 0])
                   for e in eps}
        train = np.concatenate([self.ep[i][0] for i in split["train"]])
        val = np.concatenate([self.ep[i][0] for i in split["validation"]])
        self.tau = float(np.percentile([self.metric(v, train).min() for v in val], 95))
        self.eps = C.EPS_OVER_TAU * self.tau
        self.ubar = UBAR
        self.omega_max = 1.5 * float(np.abs(train[:, 2]).max())
        pool = [(i, t) for i in split["train"] for t in range(len(self.ep[i][0]) - max(self.ladder))]
        self.anchors = C.draw_anchors(pool, C.SYSTEMS[self.name], smoke)
        c = S["m1_infeasibility"]
        self.grid = np.linspace(-UBAR, UBAR, c["grid_points"])
        self.h = float(self.grid[1] - self.grid[0])
        self.L_u = self.task_dt * np.exp(np.sqrt(1 + G ** 2) * self.task_dt)
        self.identity = {"split": split, "tau": self.tau, "epsilon": self.eps, "ubar": self.ubar,
                         "omega_envelope": self.omega_max, "metric": S["metric"], "pool": len(pool),
                         "anchors": [list(a) for a in self.anchors], "certificate": {"h": self.h, "L_u": self.L_u}}

    task_dt = 0.02

    # --- the problem ---------------------------------------------------------------------------------------------
    def anchor(self, a: int):
        i, t = self.anchors[a]
        x, u = self.ep[i]
        return x[t], u[t:t + max(self.ladder)], x

    def target(self, a: int, M: int) -> np.ndarray:
        i, t = self.anchors[a]
        return self.ep[i][0][t + M]

    def residual(self, x, y) -> np.ndarray:
        x, y = np.asarray(x), np.asarray(y)
        return np.stack([wrap(theta(x) - theta(y)), (x[..., 2] - y[..., 2]) / SQRT_G], -1)

    def metric(self, x, y) -> np.ndarray:
        return np.sqrt((self.residual(x, y) ** 2).sum(-1))

    def inside(self, x) -> np.ndarray:
        return np.abs(np.asarray(x)[..., 2]) <= self.omega_max

    def propose(self, a: int, r: float, g: np.random.Generator):
        d = g.standard_normal(2)
        d = r * self.tau * d / np.linalg.norm(d)
        x = self.anchor(a)[0]
        th, w = theta(x) + d[0], x[2] + SQRT_G * d[1]
        x0 = np.array([np.sin(th), np.cos(th), w])
        return (x0, None) if self.inside(x0) else (None, "envelope")

    def start_record(self, x0) -> dict:
        return {"x0": C.floats(x0)}

    def rollout(self, x0, U) -> np.ndarray:
        """(B, M) actions -> (B, M + 1, 3) states under the evaluator map."""
        U = np.atleast_2d(U)
        x = np.broadcast_to(np.asarray(x0, np.float64), (len(U), 3)).copy()
        out = [x]
        for k in range(U.shape[1]):
            x, _, _ = self.task.step(x, U[:, k])
            out.append(x)
        return np.stack(out, 1)

    def refined_rollout(self, x0, U) -> np.ndarray:
        """The refined flow, one solve_ivp per held action: (M,) actions -> (M + 1, 3) states."""
        from ilsb.tasks.pendulum_map import field
        x, out = np.asarray(x0, np.float64), [np.asarray(x0, np.float64)]
        for u in np.clip(np.asarray(U, np.float64), -UBAR, UBAR):
            sol = solve_ivp(lambda _, y, u=u: field(y[None], np.array([u]))[0], (0.0, self.task_dt), x, method="RK45",
                            rtol=S["refinement"]["rtol"], atol=S["refinement"]["atol"])
            x = sol.y[:, -1]
            out.append(x)
        return np.array(out)

    # --- certificates --------------------------------------------------------------------------------------------
    def eligibility(self, a: int):
        return None

    def certificate(self, a: int, x0) -> dict:
        """min over the grid of d_refined(F(x0, u), y1) - L_u h / 2 - 2 delta > epsilon."""
        y = self.target(a, 1)
        refined = np.array([self.refined_rollout(x0, [u])[-1] for u in self.grid])
        evaluator = self.rollout(x0, self.grid[:, None])[:, -1]
        delta = float(self.metric(evaluator, refined).max())
        d = self.metric(refined, y)
        bound = float(d.min() - self.L_u * self.h / 2 - 2 * delta)
        return {"kind": "lipschitz_certified_sweep", "min_refined": float(d.min()), "argmin_u": float(self.grid[d.argmin()]),
                "delta": delta, "lower_bound": bound, "infeasible": bool(bound > self.eps)}

    def oracle(self, a: int, x0, M: int):
        return None

    def demo_replay_miss(self, a: int, M: int) -> float:
        x, u, _ = self.anchor(a)
        return float(self.metric(self.rollout(x, u[None, :M])[0, -1], self.target(a, M)))

    # --- solvers and segments ------------------------------------------------------------------------------------
    def solve(self, solver: str, a: int, x0, M: int, seed) -> list:
        y, demo = self.target(a, M), self.anchor(a)[1][:M]
        res = lambda U: self.residual(self.rollout(x0, U)[:, -1], y)  # noqa: E731
        return [u[:, None] for u in C.lm(res, M, self.ubar, C.lm_starts(demo, self.ubar, seed))]

    def segment(self, a: int, x0, M: int, U) -> dict:
        U = np.asarray(U, np.float64)
        y = self.target(a, M)
        states = self.rollout(x0, U[:, 0])[0]
        miss = float(self.metric(states[-1], y))
        refined = float(self.metric(self.refined_rollout(x0, U[:, 0])[-1], y))
        checks = {"bounds": self.check_bounds(U), "envelope": self.check_envelope(states), "terminal": miss <= self.eps,
                  "refinement": refined <= self.eps}
        return {"actions": C.floats(U), "states": C.floats(states), "miss": miss, "refined_miss": refined,
                "continuation_px": None, "checks": checks, "valid": all(checks.values())}

    def check_bounds(self, actions) -> bool:
        return bool((np.abs(np.asarray(actions)) <= self.ubar).all())

    def check_envelope(self, states) -> bool:
        return bool(self.inside(np.asarray(states)[1:]).all())
