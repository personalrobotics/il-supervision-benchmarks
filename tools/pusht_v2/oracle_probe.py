"""Pre-protocol evidence for #43 (docs/studies/pusht-oracle.md): the simulator facts its tolerances rest on.

    LEROBOT_PYTHON tools/pusht_v2/oracle_probe.py --source SOURCE.npz [--out docs/studies/evidence/pusht-oracle/probe.json]

Outcome-free: no label is generated, no policy is read. Over the r = 0.25 development subset's episodes it records
  replay      each episode replayed through gym_pusht from its first frame (a reset whose body pose equals the
              recorded one) with its recorded actions: the per-frame error against the recorded frames
  fresh_step  at every replay-verified frame, one step from a freshly built state (positions and velocities, no
              contact history) against the recorded successor, split by whether the replayed step had contact
  spacing     the demonstrations' frame-to-frame displacement
Distances are in pixels: the larger of the agent's displacement and the T's largest keypoint displacement.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import pose  # noqa: E402
import source as source_module  # noqa: E402

SPEC = json.loads((HERE / "surface.json").read_text())
R, REPLAY_TOL_PX = 0.25, 1e-3


def subset(train: list, r: float, seed: int) -> list:
    """ilsb.studies.pusht_surface.subset_episodes: the first round-half-up(r N) of one permutation."""
    perm = np.random.default_rng(np.random.SeedSequence([seed, 9])).permutation(len(train))
    return sorted(int(train[i]) for i in perm[:int(np.floor(r * len(train) + 0.5))])


def make_env():
    import gymnasium as gym
    import gym_pusht  # noqa: F401
    return gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos").unwrapped


def positions(env) -> np.ndarray:
    return np.array([*env.agent.position, *env.block.position, env.block.angle])


def full_state(env) -> tuple:
    return (tuple(env.agent.position), tuple(env.agent.velocity), tuple(env.block.position), float(env.block.angle),
            tuple(env.block.velocity), float(env.block.angular_velocity))


def load_fresh(env, s: tuple) -> None:
    """A fresh simulator state: a new space, the bodies set directly, no settling step and no contact history. The
    T's angle is set before its position, since setting the angle rotates the body about its centre of gravity."""
    env._setup()
    env.agent.position, env.agent.velocity = s[0], s[1]
    env.block.angle = s[3]
    env.block.position = s[2]
    env.block.velocity, env.block.angular_velocity = s[4], s[5]


def distance(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N, 5) positions (agent x, y, T x, y, theta) -> px: max of the agent's and the T's largest keypoint move."""
    a, b = np.atleast_2d(a), np.atleast_2d(b)
    agent = np.hypot(*(a[:, 0:2] - b[:, 0:2]).T)
    kp = np.abs(pose.keypoints(a[:, 2:5]) - pose.keypoints(b[:, 2:5])).reshape(len(a), 8, 2)
    return np.maximum(agent, np.hypot(kp[..., 0], kp[..., 1]).max(1))


def q(x) -> dict | None:
    x = np.asarray(x, np.float64)
    return {str(p): float(np.percentile(x, p)) for p in (10, 50, 90, 99, 100)} if len(x) else None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=Path, required=True)
    p.add_argument("--out", type=Path, default=HERE.parents[1] / "docs/studies/evidence/pusht-oracle/probe.json")
    args = p.parse_args()
    with np.load(args.source) as z:
        src = {k: z[k] for k in z.files}
    if source_module.content_sha256(src) != SPEC["data"]["content_sha256"]:
        sys.exit("refused: the source rows are not the protocol's by content")
    eps = subset(SPEC["roles"]["train"], R, SPEC["subset_seed"])
    if len(eps) != SPEC["expected"]["subset_episodes"]["r0.25"]:
        sys.exit(f"refused: {len(eps)} subset episodes")
    env, fresh = make_env(), make_env()
    replay_err, verified, fresh_free, fresh_contact, spacing, per_episode = [], 0, [], [], [], {}
    for e in eps:
        r = np.flatnonzero(src["episode"] == e)
        cur = np.concatenate([src["state"][r][:, 2:4], src["pose"][r][:, 3:6]], 1).astype(np.float64)
        act = src["action"][r].astype(np.float64)
        spacing += distance(cur[1:], cur[:-1]).tolist()
        x0 = cur[0].copy()
        for _ in range(5):
            env.reset(options={"reset_to_state": x0})
            x0[2:5] += cur[0][2:5] - positions(env)[2:5]
        err = [float(distance(positions(env), cur[0])[0])]
        for t in range(len(r) - 1):
            s = full_state(env)
            ok_t = err[t] <= REPLAY_TOL_PX
            load_fresh(fresh, s)
            fresh.step(act[t])
            env.step(act[t])
            err.append(float(distance(positions(env), cur[t + 1])[0]))
            if ok_t and err[t + 1] <= REPLAY_TOL_PX:
                d = float(distance(positions(fresh), cur[t + 1])[0])
                (fresh_contact if env.n_contact_points > 0 else fresh_free).append(d)
        err = np.array(err)
        verified += int((err <= REPLAY_TOL_PX).sum())
        replay_err += err.tolist()
        per_episode[str(e)] = {"frames": len(r), "verified": int((err <= REPLAY_TOL_PX).sum()),
                               "first_verified": int(np.argmax(err <= REPLAY_TOL_PX)) if (err <= REPLAY_TOL_PX).any()
                               else None, "max_error_px": float(err.max())}
    import gym_pusht  # noqa: F401
    import importlib.metadata as md
    out = {"schema": "ilsb.pusht_oracle.probe 1", "source_content_sha256": SPEC["data"]["content_sha256"],
           "subset": {"r": R, "seed": SPEC["subset_seed"], "episodes": eps},
           "simulator": {"gym_pusht": md.version("gym-pusht"), "pymunk": md.version("pymunk"),
                         "dt": 0.01, "substeps": 10, "k_p": 100, "k_v": 20, "damping": 0.0},
           "replay": {"tolerance_px": REPLAY_TOL_PX, "frames": len(replay_err), "verified_frames": verified,
                      "error_px": q(replay_err), "episodes": per_episode},
           "fresh_step": {"no_contact_px": q(fresh_free), "contact_px": q(fresh_contact),
                          "steps": {"no_contact": len(fresh_free), "contact": len(fresh_contact)},
                          "exact_no_contact_fraction": float(np.mean(np.array(fresh_free) == 0)) if fresh_free else None},
           "spacing_px": q(spacing)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=1, sort_keys=True) + "\n")
    print(json.dumps({k: out[k] for k in ("fresh_step", "spacing_px")} | {"verified": verified,
                                                                         "frames": len(replay_err)}))


if __name__ == "__main__":
    main()
