"""Monitoring diagnostic: how far the agent moves in closed loop, and the first actions' offset from the agent."""
import json, sys
import numpy as np, torch, gymnasium as gym
import lerobot_env_pusht_velocity as plugin
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
ckpt = sys.argv[1]
policy = ACTPolicy.from_pretrained(ckpt).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
env = gym.make(plugin.GYM_ID, obs_type="environment_state_agent_pos", max_episode_steps=300)
out = []
for seed in range(1000, 1010):
    obs, _ = env.reset(seed=seed); policy.reset()
    start, path, offsets, best = obs["agent_pos"][:2].copy(), 0.0, [], 0.0
    prev = start.copy()
    for k in range(300):
        b = {kk: v.to("cuda") for kk, v in preprocess_observation({kk: vv[None] for kk, vv in obs.items()}).items()}
        with torch.no_grad():
            a = post(policy.predict_action_chunk(pre(b))[:, 0]).cpu().numpy()[0]
        if k < 5: offsets.append(float(np.linalg.norm(a - obs["agent_pos"][:2])))
        obs, r, term, trunc, _ = env.step(a); best = max(best, float(r))
        path += float(np.linalg.norm(obs["agent_pos"][:2] - prev)); prev = obs["agent_pos"][:2].copy()
        if term or trunc: break
    out.append({"seed": seed, "path_px": round(path, 1), "net_px": round(float(np.linalg.norm(prev - start)), 1),
                "first_offsets_px": [round(x, 1) for x in offsets], "max_reward": round(best, 3)})
for o in out: print(json.dumps(o))
