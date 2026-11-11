"""Diagnostic only (no labels written): do failed preimage solves converge with more iterations, and how well
conditioned is d g_hat / d(previous frame) at the source state?"""
import numpy as np
from pathlib import Path
from ccil.core.generators import newton_preimage
from ccil.schema import read_candidates, read_trajectories
from ccil.models.mlp import MLPTransitionModel
gen = next(Path("runs").glob("labels-generate-*"))
ds = read_trajectories(next(Path("runs").glob("data-prepare-*")) / "data/dataset.npz")
c = read_candidates(gen / "labels/candidates.npz", ds)
m = MLPTransitionModel.load(next(Path("runs").glob("transition-fit-*")) / "checkpoints/transition_model.npz")
names = ["state", "state.environment_state"]
t = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in names], 1)
a = np.asarray(c.actions, np.float64)
by_id = {ep.episode_id: np.concatenate([ep.observations[n] for n in names], 1) for ep in ds.episodes}
src = np.stack([by_id[e][f] for e, f in zip(np.asarray(c.source_episode_id), np.asarray(c.source_frame))]).astype(np.float64)
reasons, codes = np.asarray(c.reasons), list(c.reason_codes)
failed = np.flatnonzero(reasons[:, codes.index("generation_failed")].astype(bool))
acc = np.flatnonzero(np.asarray(c.accepted, bool))
rng = np.random.default_rng(0)
sf, sa = rng.choice(failed, 2000, replace=False), rng.choice(acc, min(500, len(acc)), replace=False)
for iters in (50, 200, 1000):
    _, fail, res = newton_preimage(m, t[sf], a[sf], src[sf], iters, 1e-4, np.float32)
    print(f"failed sample, {iters} iterations: converged {int((~fail).sum())}/2000; residual p50 {np.median(res):.3g}")
PREV, NEW = np.r_[0:2, 4:20], np.r_[2:4, 20:36]
def cond(rows):
    J = m.jacobian_z(src[rows], a[rows])[:, NEW][:, :, PREV]  # d(newest next) / d(previous frame), 18 x 18
    s = np.linalg.svd(J, compute_uv=False)
    return s[:, 0], s[:, -1]
for name, rows in (("failed", sf[:500]), ("accepted", sa)):
    hi, lo = cond(rows)
    print(f"{name}: sigma_max p50 {np.median(hi):.3g}, sigma_min p50 {np.median(lo):.3g}, condition p50 {np.median(hi / lo):.3g}")
