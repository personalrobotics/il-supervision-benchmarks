"""#70 shared machinery: seeds, anchors, availability, the frozen solvers, atomic records (recovery-feasibility.md).

Numpy only, so both the ilsb environment (DI, Pendulum) and the LeRobot environment (PushT) import it. A system
object supplies the dynamics and the checks; everything here is system-independent.

Seeds. A system's id is SYSTEMS[name]. The study's seeds are the protocol's: anchors SeedSequence([171, 2, sid]),
start s's perturbation SeedSequence([171, 2, sid, s]), and solver restart k at M SeedSequence([171, 2, sid, s, M, k]).
The smoke draws its anchors from the rest of the pool with SeedSequence([171, 2, sid, 99]) and inserts 99 after the
system id in every later key, so no smoke stream equals a study stream.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SPEC = json.loads((HERE / "stage-2.json").read_text())
SYSTEMS = {"double_integrator": 0, "pendulum": 1, "pusht": 2}
SMOKE_KEY = 99
RADII = tuple(float(r) for r in SPEC["starts"]["radii_over_tau"])
DIRECTIONS = SPEC["starts"]["directions_per_radius"]
EPS_OVER_TAU = SPEC["problem"]["epsilon_over_tau"]
MAX_REDRAWS = SPEC["starts"]["availability"]["max_redraws"]
LM, CEM = SPEC["solvers"]["lm_shooting"], SPEC["solvers"]["cem"]
LM_JACOBIAN_STEP = 1e-6  # stage-2.json: "finite difference, step 1e-6"
LM_RANDOM_RESTARTS = 6
AMBIGUITY_RMS, TIE = 0.05, 1e-9
PROGRESS_SECONDS = 60 * SPEC["compute"]["progress_minutes"]
ANCHORS = {"study": SPEC["starts"]["anchors_per_system"], "smoke": SPEC["starts"]["smoke"]["anchors_per_system"]}


def key(sid: int, smoke: bool, *rest) -> list:
    return [171, 2, sid, *([SMOKE_KEY] if smoke else []), *rest]


def rng(*k) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(x) for x in k]))


def draw_anchors(pool: list, sid: int, smoke: bool) -> list:
    """The study's 50 anchors, or the smoke's 5 drawn from the anchors the study does not use."""
    study = [pool[i] for i in rng(171, 2, sid).choice(len(pool), ANCHORS["study"], replace=False)]
    if not smoke:
        return study
    rest = [a for a in pool if a not in study]
    return [rest[i] for i in rng(171, 2, sid, SMOKE_KEY).choice(len(rest), ANCHORS["smoke"], replace=False)]


def start_ids(n_anchors: int) -> list:
    """(start, anchor index, radius, direction), start = (anchor index * radii + radius index) * directions + d."""
    return [((a * len(RADII) + i) * DIRECTIONS + d, a, RADII[i], d)
            for a in range(n_anchors) for i in range(len(RADII)) for d in range(DIRECTIONS)]


def available(propose, g: np.random.Generator):
    """The protocol's one availability rule: up to 1 + MAX_REDRAWS draws from one stream. propose(g) returns
    (start, None) or (None, reason). Returns (start or None, the invalid draws' reasons)."""
    reasons = []
    for _ in range(1 + MAX_REDRAWS):
        start, why = propose(g)
        if start is not None:
            return start, reasons
        reasons.append(why)
    return None, reasons


def lm(residuals, n: int, ubar: float, starts: np.ndarray):
    """LM shooting over v with u = ubar * tanh(v) (stage-2.json "lm_shooting"), all restarts batched.

    residuals(U) maps (B, n) actions to (B, k) metric residuals of the terminal miss. starts is (R, n) initial actions.
    Each iteration is one damped Gauss-Newton trial from a forward-difference Jacobian: accepted if it lowers the
    residual norm (damping / down), else rejected (damping * up); a trial whose damped system is singular in floating
    point is rejected. Returns (R, n) actions: each restart's best.
    """
    h, mu0, up, down = LM_JACOBIAN_STEP, LM["damping"]["initial"], LM["damping"]["up"], LM["damping"]["down"]
    v = np.arctanh(np.clip(starts / ubar, -1 + 1e-12, 1 - 1e-12))
    r_count = len(v)
    mu = np.full(r_count, mu0)
    r = residuals(ubar * np.tanh(v))
    cost = (r * r).sum(1)
    jac_stale = np.ones(r_count, bool)
    J = np.zeros((r_count, r.shape[1], n))
    eye = np.eye(n)
    for _ in range(LM["iterations"]):
        if jac_stale.any():
            rows = np.flatnonzero(jac_stale)
            probe = np.repeat(v[rows], n, axis=0) + h * np.tile(eye, (len(rows), 1))
            rp = residuals(ubar * np.tanh(probe)).reshape(len(rows), n, -1)
            J[rows] = ((rp - r[rows, None, :]) / h).transpose(0, 2, 1)
            jac_stale[:] = False
        JtJ = J.transpose(0, 2, 1) @ J
        A, b = JtJ + mu[:, None, None] * eye, J.transpose(0, 2, 1) @ r[..., None]
        solved = np.ones(r_count, bool)
        try:
            step = -np.linalg.solve(A, b)[..., 0]
        except np.linalg.LinAlgError:  # #76 amendment: a singular damped system is that restart's rejected trial
            step = np.zeros((r_count, n))
            for i in range(r_count):
                try:
                    step[i] = -np.linalg.solve(A[i], b[i])[:, 0]
                except np.linalg.LinAlgError:
                    solved[i] = False
        trial = v + step
        r_new = residuals(ubar * np.tanh(trial))
        cost_new = (r_new * r_new).sum(1)
        ok = solved & np.isfinite(cost_new) & (cost_new < cost)
        v[ok], r[ok], cost[ok] = trial[ok], r_new[ok], cost_new[ok]
        mu = np.where(ok, mu / down, mu * up)
        jac_stale = ok
    return ubar * np.tanh(v)


def lm_starts(demo: np.ndarray, ubar: float, seed) -> np.ndarray:
    """Restart 0 is zero actions, 1 the demonstrated actions, and 2-7 uniform in the bounds from seed(restart)."""
    n = len(demo)
    return np.stack([np.zeros(n), np.clip(demo, -ubar, ubar),
                     *(seed(k).uniform(-ubar, ubar, n) for k in range(2, 2 + LM_RANDOM_RESTARTS))])


def cem(miss, demo: np.ndarray, std0: float, lo, hi, g: np.random.Generator) -> np.ndarray:
    """CEM (stage-2.json "cem") for one restart: miss(U) maps (B, M, d) action sequences to (B,) terminal misses.
    Returns the better of the best sampled sequence over all iterations and the final mean (protocol amendment 1);
    a tie keeps the sample."""
    mean, std = demo.astype(np.float64).copy(), np.full(demo.shape, float(std0))
    best, best_miss = None, np.inf
    for _ in range(CEM["iterations"]):
        x = np.clip(mean + std * g.standard_normal((CEM["population"], *demo.shape)), lo, hi)
        m = miss(x)
        order = np.argsort(m, kind="stable")
        if m[order[0]] < best_miss:
            best, best_miss = x[order[0]].copy(), float(m[order[0]])
        elite = x[order[:CEM["elites"]]]
        mean, std = elite.mean(0), elite.std(0)
    mean = np.clip(mean, lo, hi)
    return mean if float(miss(mean[None])[0]) < best_miss else best


def ambiguity(segments: list, ubar: float) -> dict:
    """Distinct valid solutions across restarts (greedy, in restart order; RMS action difference > 0.05 ubar) and
    the restarts whose terminal miss ties the best within 1e-9."""
    valid = [s for s in segments if s["valid"]]
    distinct = []
    for s in valid:
        a = np.asarray(s["actions"], np.float64)
        if all(np.sqrt(((a - b) ** 2).mean()) > AMBIGUITY_RMS * ubar for b in distinct):
            distinct.append(a)
    misses = [s["miss"] for s in segments if s["miss"] is not None]
    best = min(misses) if misses else None
    return {"valid_restarts": len(valid), "distinct_valid": len(distinct),
            "ties_with_best": sum(abs(m - best) <= TIE for m in misses) if misses else 0}


def first_valid(segments: list):
    for k, s in enumerate(segments):
        if s["valid"]:
            return k
    return None


# --- records and the scheduler ------------------------------------------------------------------------------------

def dumps(obj) -> str:
    return json.dumps(obj, indent=1, sort_keys=True, allow_nan=False) + "\n"


def write_atomic(path: Path, text: str) -> None:
    tmp = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    tmp.write_text(text)
    os.replace(tmp, path)


def git(*a) -> str:
    return subprocess.run(["git", "-C", str(HERE), *a], capture_output=True, text=True, check=True).stdout.strip()


def code_identity() -> dict:
    return {"commit": git("rev-parse", "HEAD"), "dirty": bool(git("status", "--porcelain"))}


def floats(a) -> list:
    return np.asarray(a, np.float64).tolist()


class Log:
    """Timestamped, flushed progress lines."""

    def __init__(self, path: Path):
        self.f = open(path, "a", buffering=1)

    def __call__(self, msg: str) -> None:
        line = f"{time.strftime('%Y-%m-%dT%H:%M:%S%z')} {msg}"
        self.f.write(line + "\n")
        print(line, flush=True)


def schedule(fn, units: list, workers: int, log: Log, label: str, initializer=None, initargs=(), on_done=None) -> dict:
    """Run fn(unit) over units with a bounded pool. The parent sees every completion; a dead worker breaks the pool,
    which raises at once (BrokenProcessPool) with the units still pending named. on_done(unit, result) runs in the
    parent as each unit completes, so finished units are durable before the next. Progress every PROGRESS_SECONDS."""
    t0 = last = time.time()
    done, out = 0, {}
    with ProcessPoolExecutor(workers, initializer=initializer, initargs=initargs) as ex:
        pending = {ex.submit(fn, u): u for u in units}
        while pending:
            finished, _ = wait(pending, timeout=PROGRESS_SECONDS, return_when=FIRST_COMPLETED)
            for f in finished:
                u = pending.pop(f)
                try:
                    out[u] = f.result()
                except Exception:
                    log(f"{label}: unit {u} failed; still pending {sorted(pending.values())}")
                    for g in pending:
                        g.cancel()
                    raise
                if on_done is not None:
                    on_done(u, out[u])
                done += 1
            if time.time() - last >= PROGRESS_SECONDS or not pending:
                last = time.time()
                log(f"{label}: {done}/{len(units)} units, {len(pending)} pending, {last - t0:.0f} s elapsed")
    return out


def header(log: Log, extra: dict) -> None:
    log("command " + " ".join([sys.executable, *sys.argv]))
    log("code " + json.dumps(code_identity()))
    log("host " + json.dumps({"node": os.uname().nodename, "cpus": os.cpu_count(), "load": os.getloadavg(),
                              "python": sys.version.split()[0], "numpy": np.__version__, **extra}))
