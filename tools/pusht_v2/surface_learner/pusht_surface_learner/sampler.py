"""The fixed-fraction minibatch sampler: two streams, each drawn without replacement within an epoch.

Stream s (0 real, 1 synthetic) of n rows is the concatenation of its epochs' permutations, epoch e's permutation drawn
from SeedSequence([seed, s, e]); step t of a stream drawing b rows per step takes positions [t b, (t + 1) b) of it. A
step's indices are therefore a function of (seed, step) alone, so a resumed run continues the identical sequence. The
duplication arm's synthetic rows are, in order, the CCIL arm's labels' source rows, so both arms draw the same
positions of the same source multiset at every step.
"""
import hashlib

import numpy as np


class Stream:
    def __init__(self, n: int, per_step: int, seed: int, stream: int):
        self.n, self.b, self.seed, self.stream = n, per_step, seed, stream
        self._perms = {}

    def _perm(self, epoch: int) -> np.ndarray:
        if epoch not in self._perms:
            if len(self._perms) > 4:
                self._perms.pop(min(self._perms))
            rng = np.random.default_rng(np.random.SeedSequence([self.seed, self.stream, epoch]))
            self._perms[epoch] = rng.permutation(self.n)
        return self._perms[epoch]

    def at(self, step: int) -> np.ndarray:
        if self.b == 0:
            return np.zeros(0, np.int64)
        pos = np.arange(step * self.b, (step + 1) * self.b, dtype=np.int64)
        ep, off = pos // self.n, pos % self.n
        return np.array([self._perm(int(e))[o] for e, o in zip(ep, off)], np.int64) if ep[0] != ep[-1] \
            else self._perm(int(ep[0]))[off].astype(np.int64)


def per_batch(alpha: float, batch: int) -> tuple[int, int]:
    """(real, synthetic) rows per batch: b_s = round(alpha * B), half up."""
    b_s = int(np.floor(alpha * batch + 0.5))
    return batch - b_s, b_s


def segment_digest(real: Stream, syn: Stream, start: int, stop: int) -> str:
    """SHA-256 of steps [start, stop)'s indices, each step's real indices then its synthetic ones, as int64."""
    h = hashlib.sha256()
    for t in range(start, stop):
        h.update(real.at(t).tobytes())
        h.update(syn.at(t).tobytes())
    return h.hexdigest()
