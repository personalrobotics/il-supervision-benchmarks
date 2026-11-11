"""Reference behaviours for the PushT pilot's metric; runs in the LeRobot environment, after the pilot.

    python reference.py hold OUT_JSON                        a policy that holds the agent still, on the pilot's starts
    python reference.py replay SOURCE OUT_JSON               every demonstration replayed in gym_pusht from its first frame
    python reference.py act SOURCE_ARM OUT_DIR [--policy TYPE] [--steps N] [--train-arg ARG ...]
                                                             a LeRobot policy (default ACT, chunk 100) on the bc arm, then
                                                             evaluated; each --train-arg is passed on to lerobot-train

hold and replay place the pilot's coverage numbers between a do-nothing floor and the demonstrator. replay also checks
that the dataset's observations are gym_pusht's: each first frame's 16 keypoints are fitted by a rigid motion of the
T's own keypoints (the residual tests their order and geometry), the simulator is reset to that pose, its observation
is compared with the recorded one, and the recorded actions are stepped and compared frame by frame with the recording.
act separates the policy design from the pipeline: the same data, schedule and evaluation with ACT's default chunking.
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import gym_pusht  # noqa: F401  registers gym_pusht/PushT-v0
import gymnasium as gym
import numpy as np
import pandas as pd

PROTOCOL = json.loads((Path(__file__).parent / "protocol.json").read_text())
EVAL = PROTOCOL["evaluation"]
EPISODE_STEPS = 300  # LeRobot 0.6.1 PushtEnv.episode_length


def make_env():
    return gym.make("gym_pusht/PushT-v0", obs_type=EVAL["obs_type"], max_episode_steps=EPISODE_STEPS)


def hold(out):
    env, eps = make_env(), []
    for i in range(EVAL["n_episodes"]):
        seed = EVAL["start_seed"] + i
        obs, _ = env.reset(seed=seed)
        still, best, ok = np.array(obs["agent_pos"]), 0.0, False
        for _ in range(EPISODE_STEPS):
            obs, r, term, trunc, info = env.step(still)
            best, ok = max(best, float(r)), ok or bool(info["is_success"])
            if term or trunc:
                break
        eps.append({"seed": seed, "max_reward": best, "success": ok})
    write(out, {"policy": "hold", "episodes": eps, **summary(eps)})


def local_keypoints(env):
    env.reset(seed=0)  # the block's shapes exist after a reset; their vertices are in its own frame
    return np.array([np.array(v) for s in env.unwrapped._block_shapes for v in s.get_vertices()])


def fit_pose(local, kp):
    """Rigid motion (angle, translation) taking the T's local keypoints to the observed ones, and its RMS residual."""
    lc, kc = local.mean(0), kp.mean(0)
    u, _, vt = np.linalg.svd((local - lc).T @ (kp - kc))
    rot = (u @ np.diag([1, np.sign(np.linalg.det(u @ vt))]) @ vt).T
    t = kc - rot @ lc
    resid = float(np.sqrt(np.mean(np.sum((local @ rot.T + t - kp) ** 2, axis=1))))
    return float(np.arctan2(rot[1, 0], rot[0, 0])), t, resid


def replay(source, out):
    df = pd.concat([pd.read_parquet(p) for p in sorted((Path(source) / "data").rglob("*.parquet"))])
    env = make_env()
    local = local_keypoints(env)
    rows = []
    for e, ep in df.sort_values(["episode_index", "frame_index"]).groupby("episode_index"):
        state = np.stack(ep["observation.state"].to_numpy()).astype(np.float64)
        kps = np.stack(ep["observation.environment_state"].to_numpy()).astype(np.float64)
        acts = np.stack(ep["action"].to_numpy()).astype(np.float64)
        rec_reward = ep["next.reward"].to_numpy().astype(np.float64)
        angle, t, resid = fit_pose(local, kps[0].reshape(8, 2))
        # reset_to_state keeps a legacy convention (its docstring: the state set differs from the one passed), so
        # the fitted pose is placed directly, agent at rest, before any physics step.
        env.reset(seed=0)
        u = env.unwrapped
        u.block.angle, u.block.position = angle, t.tolist()
        u.agent.position, u.agent.velocity = state[0].tolist(), (0, 0)
        obs = u.get_obs()
        reset_err = float(np.abs(obs["environment_state"] - kps[0]).max())
        agent_err = float(np.abs(obs["agent_pos"] - state[0]).max())
        kp_err, rewards, ok = [], [], False
        for k in range(len(acts)):
            obs, r, term, _, info = env.unwrapped.step(acts[k])
            rewards.append(float(r))
            ok = ok or bool(info["is_success"])
            if k + 1 < len(kps):
                kp_err.append(float(np.abs(obs["environment_state"] - kps[k + 1]).max()))
        rows.append({"episode": int(e), "frames": len(acts), "fit_rms": resid, "reset_keypoint_err": reset_err,
                     "reset_agent_err": agent_err, "keypoint_err_step1": kp_err[0] if kp_err else None,
                     "keypoint_err_max": max(kp_err, default=None),
                     "recorded_max_reward": float(rec_reward.max()), "replayed_max_reward": max(rewards),
                     "recorded_success": bool(ep["next.success"].any()), "replayed_success": ok})
    col = lambda k: np.array([r[k] for r in rows if r[k] is not None], dtype=float)
    q = lambda k: {p: float(np.percentile(col(k), p)) for p in (50, 90, 100)}
    write(out, {"episodes": rows, "summary": {
        "episodes": len(rows), "fit_rms": q("fit_rms"), "reset_keypoint_err": q("reset_keypoint_err"),
        "keypoint_err_step1": q("keypoint_err_step1"), "keypoint_err_max": q("keypoint_err_max"),
        "recorded_avg_max_reward": float(col("recorded_max_reward").mean()),
        "replayed_avg_max_reward": float(col("replayed_max_reward").mean()),
        "recorded_pc_success": 100 * float(col("recorded_success").mean()),
        "replayed_pc_success": 100 * float(col("replayed_success").mean())}})


def act(arm, out_dir, policy, steps, extra):
    sys.path.insert(0, str(Path(__file__).parent))
    from run import per_episode
    t, bin_ = PROTOCOL["training"], Path(sys.executable).parent
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True)
    train = [str(bin_ / "lerobot-train"), "--dataset.repo_id=local/pusht-bc", f"--dataset.root={arm}",
             f"--policy.type={policy}", "--policy.push_to_hub=false", f"--policy.device={t['device']}",
             f"--steps={steps}", f"--batch_size={t['batch_size']}", f"--save_freq={steps}", "--log_freq=1000",
             f"--num_workers={t['num_workers']}", "--seed=1", f"--output_dir={out_dir / 'train'}", "--wandb.enable=false",
             *extra]
    run(train, out_dir / "train.log")
    model = out_dir / "train" / "checkpoints" / "last" / "pretrained_model"
    ev = [str(bin_ / "lerobot-eval"), f"--policy.path={model}", f"--env.type={EVAL['env']}",
          f"--env.obs_type={EVAL['obs_type']}", f"--eval.n_episodes={EVAL['n_episodes']}",
          f"--eval.batch_size={EVAL['batch_size']}", f"--eval.use_async_envs={str(EVAL['async_envs']).lower()}",
          f"--seed={EVAL['start_seed']}", f"--policy.device={t['device']}", f"--output_dir={out_dir / 'eval'}"]
    run(ev, out_dir / "eval.log")
    eps = per_episode(json.loads((out_dir / "eval" / "eval_info.json").read_text()), EVAL["start_seed"])
    write(out_dir / "result.json", {"policy": policy if extra else f"{policy}-defaults", "train": train, "eval": ev,
                                    "episodes": eps, **summary(eps)})


def run(cmd, log):
    with open(log, "w") as f:
        r = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT)
    if r.returncode:
        sys.exit(f"{cmd[0]} exited {r.returncode}; see {log}")


def summary(eps):
    return {"avg_max_reward": float(np.mean([e["max_reward"] for e in eps])),
            "pc_success": 100 * float(np.mean([e["success"] for e in eps]))}


def write(path, obj):
    Path(path).write_text(json.dumps(obj, indent=1) + "\n")
    print(json.dumps(obj.get("summary") or {k: v for k, v in obj.items() if k != "episodes" and k != "train"}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("hold").add_argument("out")
    p = sub.add_parser("replay")
    p.add_argument("source")
    p.add_argument("out")
    p = sub.add_parser("act")
    p.add_argument("arm")
    p.add_argument("out_dir")
    p.add_argument("--policy", default="act")
    p.add_argument("--steps", type=int, default=PROTOCOL["training"]["steps"])
    p.add_argument("--train-arg", action="append", default=[])
    a = ap.parse_args()
    {"hold": lambda: hold(a.out), "replay": lambda: replay(a.source, a.out),
     "act": lambda: act(a.arm, a.out_dir, a.policy, a.steps, a.train_arg)}[a.cmd]()
