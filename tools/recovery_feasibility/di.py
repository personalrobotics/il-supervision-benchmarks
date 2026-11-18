"""#70 double integrator: the solver-calibration system (recovery-feasibility.md, stage-2.json "double_integrator").

The demonstrations are CCIL's quickstart: the `demos.py` block of docs/quickstart.md at the contract commit, checked
by sha256 and run as written. The map is exact, so the M = 1 certificate and the M > 1 reachability oracle are the
bounded least-squares optimum in the metric (scipy.optimize.lsq_linear, method bvls).
"""
from __future__ import annotations

import hashlib
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
from scipy.optimize import lsq_linear

import common as C

S = C.SPEC["systems"]["double_integrator"]
A_MAT, B_VEC = np.array(S["A"]), np.array(S["B"])
BLOCK = re.compile(r"```python quickstart demos.py\n(.*?)```", re.S)
REACHABLE = 0.9  # the oracle's margin: reachable means an optimum <= 0.9 epsilon


def quickstart_demos(quickstart: Path) -> list:
    """Run the hashed quickstart block in a scratch directory and read its episodes."""
    from ccil.schema import read_trajectories
    block = BLOCK.search(quickstart.read_text()).group(1)
    if hashlib.sha256(block.encode()).hexdigest() != S["demonstrations"]["block_sha256"]:
        sys.exit("refused: the quickstart demos.py block is not the protocol's")
    with tempfile.TemporaryDirectory() as d:
        subprocess.run([sys.executable, "-c", block], cwd=d, check=True)
        return read_trajectories(Path(d) / "demos.npz").episodes


class System:
    name, solvers, ladder = "double_integrator", tuple(S["solvers"]), tuple(S["ladder"])
    refined = continuation = False

    def __init__(self, smoke: bool, quickstart: Path):
        from ccil.data.split import make_split
        eps = quickstart_demos(quickstart)
        split = make_split([e.episode_id for e in eps], 0.2, 42)
        self.ep = {e.episode_id: (e.observations["state"].astype(np.float64), e.actions.astype(np.float64)[:, 0])
                   for e in eps}
        train = np.concatenate([self.ep[i][0] for i in split["train"]])
        val = np.concatenate([self.ep[i][0] for i in split["validation"]])
        self.std = train.std(0)
        self.tau = float(np.percentile(np.sqrt((((val[:, None] - train[None]) / self.std) ** 2).sum(-1)).min(1), 95))
        self.eps = C.EPS_OVER_TAU * self.tau
        self.ubar = float(np.abs(np.concatenate([self.ep[i][1] for i in split["train"]])).max())
        lo, hi = train.min(0), train.max(0)
        self.box = (lo - 0.5 * (hi - lo), hi + 0.5 * (hi - lo))
        pool = [(i, t) for i in split["train"] for t in range(len(self.ep[i][0]) - max(self.ladder))]
        self.anchors = C.draw_anchors(pool, C.SYSTEMS[self.name], smoke)
        self.identity = {"split": split, "tau": self.tau, "epsilon": self.eps, "ubar": self.ubar,
                         "std": C.floats(self.std), "envelope": [C.floats(b) for b in self.box],
                         "metric": S["metric"], "pool": len(pool), "anchors": [list(a) for a in self.anchors]}

    # --- the problem ---------------------------------------------------------------------------------------------
    def anchor(self, a: int):
        i, t = self.anchors[a]
        x, u = self.ep[i]
        return x[t], u[t:t + max(self.ladder)], x

    def target(self, a: int, M: int) -> np.ndarray:
        i, t = self.anchors[a]
        return self.ep[i][0][t + M]

    def metric(self, x, y) -> np.ndarray:
        return np.sqrt((((np.asarray(x) - y) / self.std) ** 2).sum(-1))

    def inside(self, x) -> np.ndarray:
        return ((x >= self.box[0]) & (x <= self.box[1])).all(-1)

    def propose(self, a: int, r: float, g: np.random.Generator):
        d = g.standard_normal(2)
        x0 = self.anchor(a)[0] + r * self.tau * d / np.linalg.norm(d) * self.std
        return (x0, None) if self.inside(x0) else (None, "envelope")

    def start_record(self, x0) -> dict:
        return {"x0": C.floats(x0)}

    def rollout(self, x0, U) -> np.ndarray:
        """(B, M) actions -> (B, M + 1, 2) states."""
        U = np.atleast_2d(U)
        x = np.broadcast_to(np.asarray(x0, np.float64), (len(U), 2)).copy()
        out = [x]
        for k in range(U.shape[1]):
            x = x @ A_MAT.T + B_VEC * U[:, k:k + 1]
            out.append(x)
        return np.stack(out, 1)

    # --- certificates and the oracle -----------------------------------------------------------------------------
    def _bls(self, x0, y, M: int) -> dict:
        G = np.stack([np.linalg.matrix_power(A_MAT, M - 1 - k) @ B_VEC for k in range(M)], 1)
        b = y - np.linalg.matrix_power(A_MAT, M) @ x0
        W = 1 / self.std
        res = lsq_linear(W[:, None] * G, W * b, bounds=(-self.ubar, self.ubar), method="bvls")
        return {"optimum": float(np.linalg.norm(W * (G @ res.x - b))), "actions": C.floats(res.x)}

    def eligibility(self, a: int):
        return None

    def certificate(self, a: int, x0) -> dict:
        bls = self._bls(x0, self.target(a, 1), 1)
        return {"kind": "exact", "optimum": bls["optimum"], "infeasible": bool(bls["optimum"] > self.eps)}

    def oracle(self, a: int, x0, M: int) -> dict:
        bls = self._bls(x0, self.target(a, M), M)
        return {"optimum": bls["optimum"], "reachable": bool(bls["optimum"] <= REACHABLE * self.eps)}

    def demo_replay_miss(self, a: int, M: int) -> float:
        x, u, _ = self.anchor(a)
        return float(self.metric(self.rollout(x, u[None, :M])[0, -1], self.target(a, M)))

    # --- solvers and segments ------------------------------------------------------------------------------------
    def solve(self, solver: str, a: int, x0, M: int, seed) -> list:
        y, demo = self.target(a, M), self.anchor(a)[1][:M]
        if solver == "lm_shooting":
            res = lambda U: (self.rollout(x0, U)[:, -1] - y) / self.std  # noqa: E731
            starts = C.lm_starts(demo, self.ubar, seed)
            return [u[:, None] for u in C.lm(res, M, self.ubar, starts)]
        miss = lambda X: self.metric(self.rollout(x0, X[..., 0])[:, -1], y)  # noqa: E731
        return [C.cem(miss, demo[:, None], 0.5 * self.ubar, -self.ubar, self.ubar, seed(k))
                for k in range(C.CEM["restarts"])]

    def segment(self, a: int, x0, M: int, U) -> dict:
        U = np.asarray(U, np.float64)
        states = self.rollout(x0, U[:, 0])[0]
        miss = float(self.metric(states[-1], self.target(a, M)))
        checks = {"bounds": bool((np.abs(U) <= self.ubar).all()), "envelope": bool(self.inside(states[1:]).all()),
                  "terminal": miss <= self.eps}
        return {"actions": C.floats(U), "states": C.floats(states), "miss": miss, "refined_miss": None,
                "continuation_px": None, "checks": checks, "valid": all(checks.values())}

    def check_bounds(self, actions) -> bool:
        return bool((np.abs(np.asarray(actions)) <= self.ubar).all())

    def check_envelope(self, states) -> bool:
        return bool(self.inside(np.asarray(states)[1:]).all())
