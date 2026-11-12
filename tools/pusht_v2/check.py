"""Check that simulator evaluation computes the derived dataset's features; runs in the LeRobot environment, plugin
installed.

    python check.py SOURCE DERIVED OUT_JSON

- identity: the plugin's HistoryObservation, fed the source's recorded positions frame by frame, reproduces every
  derived row exactly;
- replay: each demonstration placed at its first frame's pose (pose.py's rigid fit) and stepped with its recorded
  actions through the wrapped simulator reproduces its derived rows; the median episode's largest error must stay
  below 1e-3 px in both frames (drift on contact-rich episodes is reported, not gated);
- rest: every evaluation start (protocol.json's 100 seeds) resets the agent and the T at rest, so repeating the first frame
  as its own predecessor is the true history;
- discovery: LeRobot's own plugin import finds env type pusht_history, and its vector env observes 4 + 32 values,
  the first observation's two frames equal.
"""
import hashlib
import json
import sys
from pathlib import Path

import lerobot_env_pusht_history as plugin
import numpy as np
import pose
import pyarrow.parquet as pq

EVAL = json.loads((Path(__file__).parent / "protocol.json").read_text())["evaluation"]

KEYS = (("agent_pos", "observation.state"), ("environment_state", "observation.environment_state"))


def frames(root: Path) -> dict:
    t = pq.read_table(sorted((root / "data").rglob("*.parquet"))).to_pydict()
    ep, fr = np.array(t["episode_index"]), np.array(t["frame_index"])
    out = {}
    for e in np.unique(ep):
        rows = np.flatnonzero(ep == e)
        rows = rows[np.argsort(fr[rows])]
        out[int(e)] = {k: np.array([t[k][i] for i in rows], np.float64)
                       for k in ("observation.state", "observation.environment_state", "action")}
    return out


def errors(obs, row):
    """Largest error of one observation against one derived row, in the previous and the current frame."""
    prev = max(float(np.abs(obs[k][:w] - row[c][:w]).max()) for (k, c), w in zip(KEYS, (2, 16)))
    cur = max(float(np.abs(obs[k][w:] - row[c][w:]).max()) for (k, c), w in zip(KEYS, (2, 16)))
    return prev, cur


def main(source, derived, out):
    import gymnasium as gym
    src, der = frames(source), frames(derived)
    env = gym.make(plugin.GYM_ID, obs_type="environment_state_agent_pos")
    w = env
    while not isinstance(w, plugin.HistoryObservation):
        w = w.env
    u = w.env.unwrapped

    ident, replay = [], []
    if len(der) != len(src):
        raise SystemExit(f"{len(der)} derived episodes, {len(src)} source episodes")
    for k, e in enumerate(sorted(src)):  # the derived dataset numbers the same episodes 0.. in source order
        s, d = src[e], der[k]
        n = len(s["action"])
        if len(d["action"]) != n - 1 or not np.array_equal(d["action"], s["action"][1:]):
            raise SystemExit(f"episode {e}: derived actions are not source frames 1..{n - 1}")
        # identity: the wrapper's own arithmetic on the recorded positions
        w.prev = None
        aug = [w.augment({"agent_pos": s["observation.state"][i],
                          "environment_state": s["observation.environment_state"][i]}) for i in range(n)]
        ident.append(np.max([errors(aug[i + 1], {c: d[c][i] for _, c in KEYS}) for i in range(n - 1)], axis=0))
        # replay: the simulator through the wrapper
        env.reset(seed=0)
        x, y, angle = pose.fit(s["observation.environment_state"][:1])[0]
        u.block.angle, u.block.position = float(angle), [float(x), float(y)]
        u.agent.position, u.agent.velocity = s["observation.state"][0].tolist(), (0, 0)
        w.prev = None
        w.augment(u.get_obs())
        worst = (0.0, 0.0)
        for i in range(n - 1):
            obs = w.augment(u.step(s["action"][i])[0])
            worst = tuple(max(a, b) for a, b in zip(worst, errors(obs, {c: d[c][i] for _, c in KEYS})))
        replay.append(worst)

    rest = []
    for i in range(EVAL["n_episodes"]):
        env.reset(seed=EVAL["start_seed"] + i)
        rest.append(max(float(np.abs(np.array(u.agent.velocity)).max()),
                        float(np.abs(np.array(u.block.velocity)).max()), abs(float(u.block.angular_velocity))))

    from lerobot.envs.configs import EnvConfig
    from lerobot.utils.import_utils import register_third_party_plugins
    register_third_party_plugins()
    cfg = EnvConfig.get_choice_class("pusht_history")()
    vec = cfg.create_envs(n_envs=2)["pusht_history"][0]
    obs, _ = vec.reset(seed=[EVAL["start_seed"], EVAL["start_seed"] + 1])
    shapes = {k: list(v.shape) for k, v in obs.items()}
    first_gap = max(float(np.abs(obs["agent_pos"][:, :2] - obs["agent_pos"][:, 2:]).max()),
                    float(np.abs(obs["environment_state"][:, :16] - obs["environment_state"][:, 16:]).max()))

    ident, replay = np.array(ident), np.array(replay)
    q = lambda a: {p: float(np.percentile(a, p)) for p in (50, 90, 100)}  # noqa: E731
    checks = {
        "identity": bool(ident.max() == 0.0),
        "replay": bool(np.median(replay[:, 0]) < 1e-3 and np.median(replay[:, 1]) < 1e-3),
        "rest": bool(max(rest) < 1e-9),
        "discovery": shapes == {"agent_pos": [2, 4], "environment_state": [2, 32]} and first_gap == 0.0,
    }
    doc = {"episodes": len(src), "derived_frames": int(sum(len(d["action"]) for d in der.values())),
           "identity_max_px": {"previous": float(ident[:, 0].max()), "current": float(ident[:, 1].max())},
           "replay_episode_max_px": {"previous": q(replay[:, 0]), "current": q(replay[:, 1])},
           "rest_max_velocity": max(rest), "vector_env_shapes": shapes, "first_frames_gap": first_gap,
           "plugin_sha256": hashlib.sha256(Path(plugin.__file__).read_bytes()).hexdigest(),
           "checks": checks, "pass": all(checks.values())}
    Path(out).write_text(json.dumps(doc, indent=1) + "\n")
    print(json.dumps(doc, indent=1))
    return 0 if doc["pass"] else 1


if __name__ == "__main__":
    sys.exit(main(Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]))
