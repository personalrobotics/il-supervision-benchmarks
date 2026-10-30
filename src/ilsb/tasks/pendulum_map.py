"""The evaluator's discrete transition map F_0.02 of the Pendulum task (il-supervision-benchmarks#12).

integrate() and step() are a verbatim port of CCIL's batched evaluator, tools/vector_env/pendulum.py at the pinned
CCIL commit: scipy 1.16.3's RK45 vectorized, every state keeping its own adaptive step sequence, with the
discontinuous task's velocity flips. tools/port_check.py checks the port against the original bit for bit. State is
(sin theta, cos theta, theta_dot), theta = pi upright; the action is held over one 0.02 s step.
"""
from __future__ import annotations

import numpy as np

G, LENGTH, DT = 9.81, 1.0, 0.02
RTOL, ATOL = 1e-3, 1e-6
SAFETY, MIN_FACTOR, MAX_FACTOR = 0.9, 0.2, 10.0
ORDER, ERROR_ORDER = 5, 4  # RK45: the error estimator has order 4
C = np.array([0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1])
A = np.array([[0, 0, 0, 0, 0],
              [1 / 5, 0, 0, 0, 0],
              [3 / 40, 9 / 40, 0, 0, 0],
              [44 / 45, -56 / 15, 32 / 9, 0, 0],
              [19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729, 0],
              [9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656]])
B = np.array([35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84])
E = np.array([-71 / 57600, 0, 71 / 16695, -71 / 1920, 17253 / 339200, -22 / 525, 1 / 40])
P = np.array([[1, -8048581381 / 2820520608, 8663915743 / 2820520608, -12715105075 / 11282082432],
              [0, 0, 0, 0],
              [0, 131558114200 / 32700410799, -68118460800 / 10900136933, 87487479700 / 32700410799],
              [0, -1754552775 / 470086768, 14199869525 / 1410260304, -10690763975 / 1880347072],
              [0, 127303824393 / 49829197408, -318862633887 / 49829197408, 701980252875 / 199316789632],
              [0, -282668133 / 205662961, 2019193451 / 616988883, -1453857185 / 822651844],
              [0, 40617522 / 29380423, -110615467 / 29380423, 69997945 / 29380423]])
N_STAGES = 6


def field(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """The canonical vector field on (N, 3) states and (N,) actions."""
    return np.stack([y[:, 2] * y[:, 1], y[:, 2] * -y[:, 0], -(G / LENGTH) * y[:, 0] + a], axis=1)


def _norm(x: np.ndarray) -> np.ndarray:
    """scipy's RMS norm, per row."""
    return np.sqrt((x * x).sum(axis=1)) / np.sqrt(x.shape[1])


def _initial_step(y0, f0, a, t_bound):
    scale = ATOL + np.abs(y0) * RTOL
    d0, d1 = _norm(y0 / scale), _norm(f0 / scale)
    h0 = np.where((d0 < 1e-5) | (d1 < 1e-5), 1e-6, 0.01 * d0 / np.where(d1 == 0, 1, d1))
    h0 = np.minimum(h0, t_bound)
    f1 = field(y0 + h0[:, None] * f0, a)
    d2 = _norm((f1 - f0) / scale) / h0
    tiny = (d1 <= 1e-15) & (d2 <= 1e-15)
    h1 = np.where(tiny, np.maximum(1e-6, h0 * 1e-3),
                  (0.01 / np.where(tiny, 1, np.maximum(d1, d2))) ** (1 / (ERROR_ORDER + 1)))
    return np.minimum(np.minimum(100 * h0, h1), t_bound)


def integrate(y0: np.ndarray, a: np.ndarray, t_bound: float = DT) -> np.ndarray:
    """solve_ivp(RK45, (0, t_bound)).sol(t_bound) for every row of y0 (N, 3) under actions a (N,)."""
    y0, a = np.asarray(y0, np.float64), np.asarray(a, np.float64)
    n = len(y0)
    t = np.zeros(n)
    y = y0.copy()
    f = field(y, a)
    h_abs = _initial_step(y, f, a, t_bound)
    done = np.zeros(n, bool)
    y_old, t_old, h_last = y.copy(), np.zeros(n), np.ones(n)
    Q = np.zeros((n, 3, 4))
    K = np.zeros((N_STAGES + 1, n, 3))
    exponent = -1 / (ERROR_ORDER + 1)
    while not done.all():
        act = np.flatnonzero(~done)
        ta, ya, fa, ha, aa = t[act], y[act], f[act], h_abs[act], a[act]
        min_step = 10 * np.abs(np.nextafter(ta, np.inf) - ta)
        ha = np.clip(ha, min_step, np.inf)
        rejected = np.zeros(len(act), bool)
        accepted = np.zeros(len(act), bool)
        out_y, out_f, out_t, out_h, out_k = (np.empty_like(ya), np.empty_like(fa), np.empty_like(ta), np.empty_like(ta),
                                             np.empty((N_STAGES + 1, len(act), 3)))
        trial = np.arange(len(act))  # rows of act still searching for an accepted step
        while len(trial):
            if (ha[trial] < min_step[trial]).any():
                raise FloatingPointError("RK45 step size fell below the minimum (scipy: TOO_SMALL_STEP)")
            tt, yy, ff, aa_ = ta[trial], ya[trial], fa[trial], aa[trial]
            t_new = np.minimum(tt + ha[trial], t_bound)
            h = t_new - tt
            k = np.empty((N_STAGES + 1, len(trial), 3))
            k[0] = ff
            for s in range(1, N_STAGES):
                dy = np.einsum("s,snd->nd", A[s, :s], k[:s]) * h[:, None]
                k[s] = field(yy + dy, aa_)
            y_new = yy + h[:, None] * np.einsum("s,snd->nd", B, k[:-1])
            f_new = field(y_new, aa_)
            k[-1] = f_new
            scale = ATOL + np.maximum(np.abs(yy), np.abs(y_new)) * RTOL
            err = _norm(np.einsum("s,snd->nd", E, k) * h[:, None] / scale)
            ok = err < 1
            with np.errstate(divide="ignore"):  # err == 0 gives inf, used by neither: grow takes MAX_FACTOR there and
                grow = np.where(err == 0, MAX_FACTOR, np.minimum(MAX_FACTOR, SAFETY * err ** exponent))
                shrink = np.maximum(MIN_FACTOR, SAFETY * err ** exponent)  # shrink applies only where err >= 1
            grow = np.where(rejected[trial], np.minimum(1, grow), grow)
            ha[trial] = np.abs(h) * np.where(ok, grow, shrink)
            acc = trial[ok]
            out_y[acc], out_f[acc], out_t[acc], out_h[acc] = y_new[ok], f_new[ok], t_new[ok], h[ok]
            out_k[:, acc] = k[:, ok]
            accepted[acc] = True
            rejected[trial[~ok]] = True
            trial = trial[~ok]
        y_old[act], t_old[act], h_last[act] = ya, ta, out_h
        Q[act] = np.einsum("snd,sj->ndj", out_k, P)
        t[act], y[act], f[act], h_abs[act] = out_t, out_y, out_f, ha
        done[act] = out_t >= t_bound
    x = (t_bound - t_old) / h_last
    p = np.cumprod(np.repeat(x[:, None], 4, axis=1), axis=1)
    return h_last[:, None] * np.einsum("ndj,nj->nd", Q, p) + y_old


def step(states: np.ndarray, actions: np.ndarray, discontinuous: bool):
    """One environment step for N states: (next states (N, 3), flipped (N,))."""
    nxt = integrate(states, actions)
    flipped = np.zeros(len(nxt), bool)
    if discontinuous:
        theta = np.arctan2(nxt[:, 0], nxt[:, 1])
        flipped = (np.abs(theta - np.pi / 6) < 1e-2) | (np.abs(theta - np.pi / 3) < 1e-2)
        nxt[flipped, 2] = -nxt[flipped, 2]
    return nxt, flipped
