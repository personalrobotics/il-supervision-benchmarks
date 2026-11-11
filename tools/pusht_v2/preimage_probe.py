"""Solver-only feasibility probe for v2's preimage labels (no training, no policy evaluation).

    uv run python tools/pusht_v2/preimage_probe.py rigidity DATASET_NPZ OUT_JSON
    uv run python tools/pusht_v2/preimage_probe.py probe LABELS_DIR SPEC_JSON OUT_JSON

rigidity measures the demonstrations' T shape: for every frame and both history slots, the max over the 28 keypoint
pairs of |pairwise distance - its median over all demonstration frames|. probe_spec.json freezes the rigidity
tolerance from it before the probe runs.

probe takes a v2 labels directory (its transition-fit checkpoint, data-prepare dataset and labels-generate candidates)
and, on a seeded sample of the noisy_action_fe proposals, solves the same equation x + f_hat(x, a~) = target from the
same initialization (the source state) with
- newton: ccil.core.generators.newton_preimage, the generator's own solver;
- damped: a Levenberg damped minimum-norm solve, step (J^T J + lambda I)^-1 J^T r with J = I + df_hat/dx, lambda
  adapted per row (accept: lambda / 3; reject: lambda x 4 and the step is not taken).
Each solution is cast to float32 and its residual recomputed, as the generator does. A row is usable iff it converges
(residual <= the root tolerance), keeps the history invariant (newest slots equal the source frame's within
eps_struct), keeps the T rigid in both slots (within the frozen tolerance) and its action is inside the bounds. Records
per solver: converged and usable fractions, residual and step-size distributions, invariant and rigidity violations
among converged rows. The probe passes iff the damped solver's usable fraction is at least the spec's minimum and no
converged row of it breaks the invariant or the rigidity.
"""
import json
import sys
from pathlib import Path

import numpy as np

EPS32 = float(np.finfo(np.float32).eps)
NAMES = ("state", "state.environment_state")
NEW = np.r_[2:4, 20:36]  # state [x_{t-1}(2), x_t(2)], environment_state [x_{t-1}(16), x_t(16)]
SRC_CUR = NEW


def pair_distances(kp):
    """(N, 16) keypoints -> (N, 28) pairwise distances of the 8 points."""
    p = kp.reshape(len(kp), 8, 2)
    i, j = np.triu_indices(8, 1)
    return np.linalg.norm(p[:, i] - p[:, j], axis=-1)


def rigidity_error(env32, reference):
    """Per row: max over both slots and 28 pairs of |pairwise distance - reference|, (N,)."""
    return np.maximum(np.abs(pair_distances(env32[:, :16]) - reference).max(1),
                      np.abs(pair_distances(env32[:, 16:]) - reference).max(1))


def rigidity(dataset: Path, out: Path) -> None:
    from ccil.data.prepare import load_dataset
    ds = load_dataset(dataset)
    env = np.concatenate([ep.observations["state.environment_state"] for ep in ds.episodes]).astype(np.float64)
    reference = np.median(np.concatenate([pair_distances(env[:, :16]), pair_distances(env[:, 16:])]), axis=0)
    err = rigidity_error(env, reference)
    doc = {"frames": len(env), "reference_pair_distances_px": reference.tolist(),
           "demonstration_error_px": {str(p): float(np.percentile(err, p)) for p in (50, 90, 99, 100)}}
    out.write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc["demonstration_error_px"]))


def damped(model, target, a, init, iterations, tol, lam0=1e-2):
    x, n = np.array(init, np.float64), len(init)
    lam = np.full(n, lam0)
    res = lambda xs, as_, ts: np.linalg.norm(xs + model.delta(xs, as_) - ts, axis=1)  # noqa: E731
    r = res(x, a, target)
    biggest = np.zeros(n)
    eye = np.eye(x.shape[1])
    for _ in range(iterations):
        idx = np.flatnonzero(r > tol)
        if not len(idx):
            break
        xs, as_ = x[idx], a[idx]
        e = xs + model.delta(xs, as_) - target[idx]
        J = eye + model.jacobian_z(xs, as_)
        JT = np.transpose(J, (0, 2, 1))
        A = JT @ J + lam[idx, None, None] * eye
        step = np.linalg.solve(A, (JT @ e[..., None]))[..., 0]
        trial = xs - step
        rt = res(trial, as_, target[idx])
        better = np.isfinite(rt) & (rt < r[idx])
        acc = idx[better]
        x[acc], r[acc] = trial[better], rt[better]
        biggest[acc] = np.maximum(biggest[acc], np.linalg.norm(step[better], axis=1))
        lam[acc] /= 3.0
        lam[idx[~better]] *= 4.0
    stored = x.astype(np.float32)
    return stored, res(stored.astype(np.float64), a, target), biggest


def probe(labels: Path, spec_path: Path, out: Path) -> None:
    from ccil.core.generators import newton_preimage
    from ccil.models.mlp import MLPTransitionModel
    from ccil.schema import read_candidates, read_trajectories
    spec = json.loads(spec_path.read_text())
    runs = labels / "runs"
    ds = read_trajectories(next(runs.glob("data-prepare-*")) / "data" / "dataset.npz")
    c = read_candidates(next(runs.glob("labels-generate-*")) / "labels" / "candidates.npz", ds)
    model = MLPTransitionModel.load(next(runs.glob("transition-fit-*")) / "checkpoints" / "transition_model.npz")
    target = np.concatenate([np.asarray(c.target_observations[n], np.float64) for n in NAMES], 1)
    act = np.asarray(c.actions, np.float64)
    by_id = {ep.episode_id: np.concatenate([ep.observations[n] for n in NAMES], 1) for ep in ds.episodes}
    src = np.stack([by_id[e][f] for e, f in zip(np.asarray(c.source_episode_id), np.asarray(c.source_frame))])
    src = src.astype(np.float64)
    rows = np.random.default_rng(spec["sample_seed"]).choice(len(target), spec["sample"], replace=False)
    t, a, s = target[rows], act[rows], src[rows]
    low, high = np.asarray(spec["action_low"]), np.asarray(spec["action_high"])
    in_bounds = ((a >= low) & (a <= high)).all(1)
    reference = np.asarray(spec["rigidity"]["reference_pair_distances_px"])
    tol, rig_tol = spec["root_tolerance"], spec["rigidity"]["tolerance_px"]
    q = lambda x: {str(p): float(np.percentile(x, p)) for p in (10, 50, 90, 100)} if len(x) else None  # noqa: E731
    results = {}
    for name, solve in (
            ("newton", lambda it: newton_preimage(model, t, a, s, it, tol, np.float32)),
            ("damped", lambda it: damped(model, t, a, s, it, tol))):
        for it in spec["iterations"]:
            out_ = solve(it)
            x = out_[0].astype(np.float64)
            resid = np.linalg.norm(x + model.delta(x, a) - t, axis=1)
            conv = resid <= tol
            err = np.abs(x[:, NEW] - s[:, SRC_CUR]).max(1)
            eps = np.maximum(10 * tol, 10 * EPS32 * np.maximum(1.0, np.abs(s[:, SRC_CUR]).max(1)))
            invariant_bad = conv & (err > eps)
            rig = rigidity_error(x[:, 4:36], reference)
            rigid_bad = conv & (rig > rig_tol)
            usable = conv & ~invariant_bad & ~rigid_bad & in_bounds
            results[f"{name}_{it}"] = {
                "converged_fraction": float(conv.mean()), "usable_fraction": float(usable.mean()),
                "invariant_violations": int(invariant_bad.sum()), "rigidity_violations": int(rigid_bad.sum()),
                "out_of_bounds": int((~in_bounds).sum()), "residual": q(resid),
                "residual_converged": q(resid[conv]), "residual_failed": q(resid[~conv]),
                "displacement_from_source": q(np.linalg.norm(x - s, axis=1)),
                "displacement_converged": q(np.linalg.norm(x - s, axis=1)[conv]),
                "rigidity_error_converged_px": q(rig[conv]), "invariant_error_converged_px": q(err[conv]),
                **({"largest_accepted_step": q(out_[2])} if name == "damped" else {})}
            r = results[f"{name}_{it}"]
            print(f"{name} {it}: converged {r['converged_fraction']:.4f}, usable {r['usable_fraction']:.4f}, "
                  f"invariant {r['invariant_violations']}, rigidity {r['rigidity_violations']}, "
                  f"failed residual p50 {(r['residual_failed'] or {}).get('50')}", flush=True)
    primary = results[spec["primary"]]
    passed = (primary["usable_fraction"] >= spec["min_usable_fraction"] and primary["invariant_violations"] == 0
              and primary["rigidity_violations"] == 0)
    doc = {"spec": spec, "candidates": int(len(target)), "sample": int(len(rows)), "results": results,
           "primary": spec["primary"], "pass": bool(passed)}
    out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(f"probe {'PASSED' if passed else 'FAILED'} on {spec['primary']}")


if __name__ == "__main__":
    if sys.argv[1] == "rigidity":
        rigidity(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        probe(Path(sys.argv[2]), Path(sys.argv[3]), Path(sys.argv[4]))
