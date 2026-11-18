"""#70 PushT: a treatment system (recovery-feasibility.md, stage-2.json "pusht").

The map is gym_pusht's 0.1 s action interval. The metric is the 9 current-frame points (oracle_probe.distance) and
tau is tau.json's. Anchors are the r = 0.25 subset's replay-verified rows; each anchor's full state is its episode
replay's (#52, stage_a.replay). Eligibility reuses #52's 1.0 px continuation tolerance; a segment's continuation
may drift at most 1.0 px beyond its own terminal miss (protocol amendment 2).

Information state (amendment 3): the policy sees W = 2 frames, so a segment's M free actions are followed by L = 1
fixed demonstrated handoff action a_(t+M). The handoff residual is the larger of the two frames' distances,
max(d(x_M, y_M), d(x_(M+1), y_(M+1))), and must be within epsilon, separately from the endpoint. CEM ranks samples by
that residual; the M = 1 sweep (conservative evidence, not a certificate) uses it.

Envelope. A start is built by hand, so it must not penetrate (oracle_probe.penetration <= 0) and its agent and T
centre must lie in [0, 512]^2. During a segment, contact is the simulator's to resolve (the kinematic agent may
penetrate, as in the demonstrations), and every state's agent and T centre must stay in [0, 512]^2.
"""
from __future__ import annotations

import hashlib
import json
import sys

import numpy as np

import common as C

sys.path[:0] = [str(C.ROOT / "tools/pusht_recovery"), str(C.ROOT / "tools/pusht_v2")]
import pose  # noqa: E402
import stage_a as A  # noqa: E402
from oracle_probe import distance, load_fresh, make_env, penetration, positions  # noqa: E402

S = C.SPEC["systems"]["pusht"]
LO, HI = 0.0, 512.0
GRID = np.arange(LO, HI + 1, S["m1_infeasibility"]["grid_px"])
CONT = S["continuation"]
CONT_TOL, CONT_STEPS = CONT["tolerance_px"], CONT["steps"]
UBAR = 256.0  # the ambiguity scale: half the action range
BISECT_GROW, BISECT_ITERS, BISECT_REL = 30, 200, 1e-9


def tau_record() -> float:
    path = C.ROOT / S["tau_record"]
    if hashlib.sha256(path.read_bytes()).hexdigest() != S["tau_record_sha256"]:
        sys.exit("refused: tau.json is not the protocol's")
    tau = json.loads(path.read_text())["tau_px"]
    if tau != S["tau_px"]:
        sys.exit("refused: tau.json and stage-2.json disagree")
    return tau


class System:
    name, solvers, ladder = "pusht", tuple(S["solvers"]), tuple(S["ladder"])
    refined = continuation = True

    def __init__(self, smoke: bool, source):
        A.simulator_check()
        with np.load(source) as z:
            src = {k: z[k] for k in z.files}
        if A.source_module.content_sha256(src) != A.SPEC["inputs"]["source_content_sha256"]:
            sys.exit("refused: the source rows are not the protocol's by content")
        eps = A.subset(A.BASE["roles"]["train"], 0.25, 0)
        if set(eps) & set(A.BASE["roles"]["held_out"]) or len(eps) != 41:
            sys.exit("refused: the subset is not the protocol's 41 training episodes")
        self.tau = tau_record()
        self.eps = C.EPS_OVER_TAU * self.tau
        self.ubar = UBAR
        self.env, self.fine = make_env(), make_env()
        self.fine.dt = S["refinement"]["physics_dt"]
        self.ep, pool = {}, []
        horizon = max(self.ladder) + CONT_STEPS
        for e in eps:
            r = np.flatnonzero(src["episode"] == e)
            cur = np.concatenate([src["state"][r][:, 2:4], src["pose"][r][:, 3:6]], 1).astype(np.float64)
            act = src["action"][r].astype(np.float64)
            _, err, states = A.replay(self.env, cur, act)
            self.ep[int(e)] = (cur, act, states)
            pool += [(int(e), k) for k in range(1, len(r) - horizon) if err[k] <= A.REPLAY_TOL]
        self.anchors = C.draw_anchors(pool, C.SYSTEMS[self.name], smoke)
        self.identity = {"tau": self.tau, "epsilon": self.eps, "ubar": self.ubar, "metric": S["metric"],
                         "pool": len(pool), "anchors": [list(a) for a in self.anchors], "simulator": A.SPEC["simulator"],
                         "refined_substeps": int(1 / (self.fine.dt * self.fine.control_hz))}
        assert self.identity["refined_substeps"] == 40

    # --- the problem ---------------------------------------------------------------------------------------------
    def anchor(self, a: int):
        e, k = self.anchors[a]
        cur, act, states = self.ep[e]
        return states[k], act[k:k + max(self.ladder) + CONT_STEPS], cur

    def target(self, a: int, M: int) -> np.ndarray:
        e, k = self.anchors[a]
        return self.ep[e][0][k + M]

    def metric(self, x, y) -> np.ndarray:
        return distance(x, y)

    def propose(self, a: int, r: float, g: np.random.Generator):
        """Agent (x, y) and T (x, y, L theta) along a uniform direction, scaled by bisection to radius r tau in the
        metric; velocities kept from the anchor's replayed state."""
        s = self.anchor(a)[0]
        z0 = np.array([*s[0], *s[2], A.L * s[3]])
        p0 = np.array([*s[0], *s[2], s[3]])
        d = g.standard_normal(5)
        d /= np.linalg.norm(d)
        goal = r * self.tau
        at = lambda c: z0 + c * d  # noqa: E731
        dist = lambda c: float(distance(np.r_[at(c)[:4], at(c)[4] / A.L], p0)[0])  # noqa: E731
        lo, hi = 0.0, goal
        for _ in range(BISECT_GROW):
            if dist(hi) >= goal:
                break
            lo, hi = hi, 2 * hi
        else:
            return None, "bisection"
        for _ in range(BISECT_ITERS):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if dist(mid) < goal else (lo, mid)
        if abs(dist(hi) - goal) > BISECT_REL * goal:
            return None, "bisection"
        z = at(hi)
        agent, T, th = z[0:2], z[2:4], z[4] / A.L
        if not ((LO <= np.r_[agent, T]) & (np.r_[agent, T] <= HI)).all():
            return None, "workspace"
        if penetration(agent[None], pose.keypoints(np.r_[T, th][None]))[0] > 0:
            return None, "penetration"
        return ((float(agent[0]), float(agent[1])), s[1], (float(T[0]), float(T[1])), float(th), s[4], s[5]), None

    def start_record(self, x0) -> dict:
        return {"x0": [list(v) if isinstance(v, tuple) else v for v in x0]}

    def chain(self, x0, U, env=None) -> np.ndarray:
        """Positions of x0 and after each action, in one live space started fresh at x0."""
        env = env or self.env
        load_fresh(env, x0)
        out = [positions(env)]
        for a in U:
            env.step(np.asarray(a, np.float64))
            out.append(positions(env))
        return np.array(out)

    def inside(self, p) -> np.ndarray:
        p = np.atleast_2d(p)[:, :4]
        return ((LO <= p) & (p <= HI)).all(1)

    # --- eligibility and the certificate --------------------------------------------------------------------------
    def eligibility(self, a: int) -> dict:
        """The anchor's replayed state, restored fresh, under its recorded actions: rows k + 1 ... k + M + 8 within
        1.0 px, per M."""
        e, k = self.anchors[a]
        cur, act, states = self.ep[e]
        n = max(self.ladder) + CONT_STEPS
        dev = distance(self.chain(states[k], act[k:k + n])[1:], cur[k + 1:k + 1 + n])
        return {"deviation_px": C.floats(dev),
                "eligible": {str(M): bool((dev[:M + CONT_STEPS] <= CONT_TOL).all()) for M in self.ladder}}

    def certificate(self, a: int, x0) -> dict:
        """Empirical: no action on the 2 px grid, followed by the handoff action a_(t+1), has a handoff residual
        within 2 epsilon. The endpoint-only minimum is recorded beside it."""
        e, k = self.anchors[a]
        y1, y2, h = self.target(a, 1), self.target(a, 2), self.ep[e][1][k + 1]
        best, arg, best_end = np.inf, None, np.inf
        for ax in GRID:
            for ay in GRID:
                load_fresh(self.env, x0)
                self.env.step(np.array([ax, ay]))
                d1 = float(distance(positions(self.env), y1)[0])
                self.env.step(h)
                d = max(d1, float(distance(positions(self.env), y2)[0]))
                best_end = min(best_end, d1)
                if d < best:
                    best, arg = d, [float(ax), float(ay)]
        return {"kind": "sweep_conservative_empirical", "min_grid": best, "argmin_action": arg,
                "min_grid_endpoint": best_end, "endpoint_infeasible": bool(best_end > 2 * self.eps),
                "infeasible": bool(best > 2 * self.eps)}

    def oracle(self, a: int, x0, M: int):
        return None

    def demo_replay_miss(self, a: int, M: int):
        return None

    # --- solvers and segments ------------------------------------------------------------------------------------
    def handoff(self, a: int, x0, M: int, U, env=None) -> tuple:
        """(endpoint residual, handoff residual) of M free actions U followed by the handoff action a_(t+M)."""
        e, k = self.anchors[a]
        p = self.chain(x0, [*np.asarray(U, np.float64), self.ep[e][1][k + M]], env)[-2:]
        d = distance(p, np.array([self.target(a, M), self.target(a, M + 1)]))
        return float(d[0]), float(d.max())

    def solve(self, solver: str, a: int, x0, M: int, seed) -> list:
        demo = self.anchor(a)[1][:M]
        miss = lambda X: np.array([self.handoff(a, x0, M, U)[1] for U in X])  # noqa: E731
        return [C.cem(miss, demo, S_CEM_STD, LO, HI, seed(k)) for k in range(C.CEM["restarts"])]

    def segment(self, a: int, x0, M: int, U) -> dict:
        e, k = self.anchors[a]
        cur, act, _ = self.ep[e]
        U = np.asarray(U, np.float64)
        y = self.target(a, M)
        states = self.chain(x0, U)
        cont = []
        for j in range(CONT_STEPS):
            self.env.step(act[k + M + j])
            cont.append(positions(self.env))
        miss = float(distance(states[-1], y)[0])
        cont_dev = float(distance(np.array(cont), cur[k + M + 1:k + M + 1 + CONT_STEPS]).max()) - miss
        handoff = max(miss, float(distance(cont[0], cur[k + M + 1])[0]))
        refined = max(self.handoff(a, x0, M, U, self.fine))
        checks = {"bounds": self.check_bounds(U), "envelope": self.check_envelope(states), "terminal": miss <= self.eps,
                  "handoff": handoff <= self.eps, "refinement": refined <= self.eps, "continuation": cont_dev <= CONT_TOL}
        return {"actions": C.floats(U), "states": C.floats(states), "miss": miss, "handoff_miss": handoff,
                "refined_miss": refined, "continuation_px": cont_dev, "checks": checks, "valid": all(checks.values())}

    def check_bounds(self, actions) -> bool:
        U = np.asarray(actions)
        return bool(((LO <= U) & (U <= HI)).all())

    def check_envelope(self, states) -> bool:
        return bool(self.inside(np.asarray(states)[1:]).all())


S_CEM_STD = 64.0  # stage-2.json "cem": initial standard deviation 64 px for PushT
