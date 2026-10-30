"""The standard paired analysis: the union of a BCa bootstrap and a paired t-interval of the mean difference.

The same rule as CCIL's predeclared Pendulum analyses (ccil.experiments.analysis.union_interval, fixed resampling
seed and count), so a harness result and a historical one use one statistic. An outcome is "increases" when the
lower endpoint is above 0, "decreases" when the upper endpoint is below 0, and "no_detectable_difference" otherwise;
fewer than min_pairs paired seeds is "inconclusive".
"""
from __future__ import annotations

import warnings

import numpy as np
import scipy.stats

RNG_SEED, RESAMPLES = 20261026, 10000


def union_interval(d, c: float) -> dict:
    d = np.asarray(d, np.float64)
    n = len(d)
    out = {"n": n, "mean": float(d.mean()) if n else None, "confidence": c}
    if n < 2:
        return {**out, "interval": None, "note": "fewer than 2 paired values"}
    half = float(scipy.stats.t.ppf((1 + c) / 2, n - 1)) * float(d.std(ddof=1)) / np.sqrt(n)
    t = [float(d.mean() - half), float(d.mean() + half)]
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore")
        ci = scipy.stats.bootstrap((d,), np.mean, confidence_level=c, n_resamples=RESAMPLES, method="BCa",
                                   rng=np.random.default_rng(RNG_SEED)).confidence_interval
    bca = [float(ci.low), float(ci.high)]
    if np.isfinite(bca).all():
        return {**out, "bca": bca, "t": t, "interval": [min(bca[0], t[0]), max(bca[1], t[1])], "note": None}
    return {**out, "bca": None, "t": t, "interval": t, "note": "BCa undefined; the t-interval alone"}


def paired(a: dict, b: dict, c: float, min_pairs: int) -> dict:
    """A - B over the seeds present in both {seed: value} maps, with its outcome."""
    seeds = sorted(set(a) & set(b))
    res = union_interval([a[s] - b[s] for s in seeds], c)
    iv = res.get("interval")
    if len(seeds) < min_pairs or iv is None:
        outcome = "inconclusive"
    else:
        outcome = "increases" if iv[0] > 0 else "decreases" if iv[1] < 0 else "no_detectable_difference"
    return {**res, "seeds": seeds, "outcome": outcome}
