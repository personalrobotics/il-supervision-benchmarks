"""Diagnostic: why v2 smoke preimage solves fail (split the final residual into agent and T-keypoint parts)."""
import numpy as np
from pathlib import Path
from ccil.schema import read_candidates, read_trajectories
from ccil.models.mlp import MLPTransitionModel
gen = next(Path("runs").glob("labels-generate-*"))
ds = read_trajectories(next(Path("runs").glob("data-prepare-*")) / "data/dataset.npz")
c = read_candidates(gen / "labels/candidates.npz", ds)
m = MLPTransitionModel.load(next(Path("runs").glob("transition-fit-*")) / "checkpoints/transition_model.npz")
names = ["state", "state.environment_state"]
z = np.concatenate([np.asarray(c.observations[n], np.float64) for n in names], 1)
t = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in names], 1)
a = np.asarray(c.actions, np.float64)
acc = np.asarray(c.accepted, bool)
reasons, codes = np.asarray(c.reasons), list(c.reason_codes)
failed = reasons[:, codes.index("generation_failed")].astype(bool)
print("accepted", int(acc.sum()), "failed", int(failed.sum()))
r = z + m.delta(z, a) - t
AG, KP, OLD = np.r_[2:4], np.r_[20:36], np.r_[0:2, 4:20]
q = lambda x: " ".join(f"p{p}={np.percentile(x, p):.3g}" for p in (10, 50, 90))
for name, rows in (("failed", failed), ("accepted", acc)):
    print(f"{name}: |resid agent| {q(np.linalg.norm(r[rows][:, AG], axis=1))}; |resid T kp| "
          f"{q(np.linalg.norm(r[rows][:, KP], axis=1))}; older-slot resid max {np.abs(r[rows][:, OLD]).max():.3g}")
noise = np.asarray(c.measurements["generator.action_noise_norm"], np.float64)
print("action noise norm p50: failed", round(float(np.median(noise[failed])), 3), "accepted", round(float(np.median(noise[acc])), 3))
mv = np.abs(t[:, 20:36] - t[:, 4:20]).max(1)
print("T moving at target (kp displacement > 0.01 px): failed", round(float((mv[failed] > 0.01).mean()), 3),
      "accepted", round(float((mv[acc] > 0.01).mean()), 3), "all", round(float((mv > 0.01).mean()), 3))
