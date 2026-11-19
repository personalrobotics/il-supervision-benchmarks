"""#82 fresh start inventory from the frozen selection (recovery-fixed-horizon.md "Fresh starts").

    PYTHONPATH=src PY tools/recovery_fixed_horizon/inventory.py --pendulum pendulum_cont_corrected.npz \
        > tools/recovery_fixed_horizon/inventory.json

Stage 2's Pendulum perturbation and availability rule (tools/recovery_feasibility/pendulum.py) applied to the
selection's anchors, keyed [171, 6, 3, start], with Stage 2's M = 1 certificate toward the anchor's next state. No
controller runs here.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.append(str(HERE.parent / "recovery_feasibility"))
import common as C  # noqa: E402
import pendulum as P2  # noqa: E402

SEL = json.loads((HERE / "selection.json").read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pendulum", type=Path, required=True)
    args = p.parse_args()
    from ccil.schema import read_trajectories
    sy = P2.System(False, args.pendulum)
    ep = {e.episode_id: e.observations["state"].astype(np.float64) for e in read_trajectories(args.pendulum).episodes}
    starts = []
    for s, a, r, d in C.start_ids(len(SEL["anchors"])):
        e, t = SEL["anchors"][a]
        x = ep[e][t]

        def propose(g, x=x, r=r):
            v = g.standard_normal(2)
            v = r * sy.tau * v / np.linalg.norm(v)
            th, w = P2.theta(x) + v[0], x[2] + P2.SQRT_G * v[1]
            x0 = np.array([np.sin(th), np.cos(th), w])
            return (x0, None) if sy.inside(x0) else (None, "envelope")

        x0, redraws = C.available(propose, C.rng(171, 6, 3, s))
        rec = {"start": s, "anchor": a, "episode": e, "t": t, "radius": r, "direction": d, "redraws": redraws,
               "x0": None if x0 is None else C.floats(x0)}
        if x0 is not None:
            y = ep[e][t + 1]
            refined = np.array([sy.refined_rollout(x0, [u])[-1] for u in sy.grid])
            evaluator = sy.rollout(x0, sy.grid[:, None])[:, -1]
            delta = float(sy.metric(evaluator, refined).max())
            dist = sy.metric(refined, y)
            bound = float(dist.min() - sy.L_u * sy.h / 2 - 2 * delta)
            rec["m1"] = {"min_refined": float(dist.min()), "delta": delta, "lower_bound": bound,
                         "infeasible": bool(bound > sy.eps)}
        starts.append(rec)
    controls = [{"control": i, "episode": e, "t": t, "x0": C.floats(ep[e][t])} for i, (e, t) in enumerate(SEL["controls"])]
    out = {"selection_sha256": hashlib.sha256((HERE / "selection.json").read_bytes()).hexdigest(),
           "pendulum_sha256": hashlib.sha256(args.pendulum.read_bytes()).hexdigest(), "tau": sy.tau, "epsilon": sy.eps,
           "starts": starts, "controls": controls,
           "counts": {"starts": len(starts), "available": sum(x["x0"] is not None for x in starts),
                      "m1_infeasible": sum(bool(x.get("m1", {}).get("infeasible")) for x in starts),
                      "controls": len(controls), "anchor_episodes": len({x["episode"] for x in starts})}}
    print(C.dumps(out), end="")


if __name__ == "__main__":
    main()
