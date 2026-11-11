"""Monitoring diagnostic: one checkpoint's actions on demonstration frames, from dataset rows vs. from the wrapped
simulator replaying the same demonstration through LeRobot's own observation preprocessing."""
import json, sys
import numpy as np, torch, gymnasium as gym
sys.path.insert(0, sys.argv[1] + "/tools/pusht_pilot")
from reference import fit_pose, local_keypoints
import lerobot_env_pusht_velocity as plugin
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.act.modeling_act import ACTPolicy
from lerobot.policies.factory import make_pre_post_processors
from lerobot.datasets.lerobot_dataset import LeRobotDataset
import pyarrow.parquet as pq
G, ckpt, src = sys.argv[2], sys.argv[3], sys.argv[4]
policy = ACTPolicy.from_pretrained(ckpt).to("cuda").eval()
pre, post = make_pre_post_processors(policy.config, pretrained_path=ckpt)
ds = LeRobotDataset("local/full", root=G + "/derived/full")
t = pq.read_table(sorted(__import__("pathlib").Path(src, "data").rglob("*.parquet"))).to_pydict()
ep = np.array(t["episode_index"])
env = gym.make(plugin.GYM_ID, obs_type="environment_state_agent_pos"); w = env
while not isinstance(w, plugin.VelocityObservation): w = w.env
u = w.env.unwrapped
local = local_keypoints(gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos"))
held = json.load(open(G + "/derived/held_out.json"))
def act(batch):
    with torch.no_grad():
        return post(policy.predict_action_chunk(pre(batch))[:, 0]).cpu().numpy()[0]
rows = {"data_vs_sim_action_px": [], "data_action_err_px": [], "sim_action_err_px": [], "obs_diff": []}
for e in held[:10]:
    r = np.flatnonzero(ep == e)
    s = {k: np.array([t[k][i] for i in r], np.float64) for k in ("observation.state", "observation.environment_state", "action")}
    env.reset(seed=0)
    a, tr, _ = fit_pose(local, s["observation.environment_state"][0].reshape(8, 2))
    u.block.angle, u.block.position = a, tr.tolist()
    u.agent.position, u.agent.velocity = s["observation.state"][0].tolist(), (0, 0)
    w.prev = None; w.augment(u.get_obs())
    k0 = ds.meta.episodes["dataset_from_index"][held.index(e)] if False else None
    idx = [i for i in range(len(ds)) if int(ds.hf_dataset[i]["episode_index"]) == held.index(e) or False]
    for i in range(min(40, len(s["action"]) - 1)):
        obs = w.augment(u.step(s["action"][i])[0])
        sim_b = preprocess_observation({k: v[None] for k, v in obs.items()})
        sim_b = {k: v.to("cuda") for k, v in sim_b.items()}
        row_state = np.concatenate([s["observation.state"][i + 1], (s["observation.state"][i + 1] - s["observation.state"][i]) / 0.1])
        row_env = np.concatenate([s["observation.environment_state"][i + 1], (s["observation.environment_state"][i + 1] - s["observation.environment_state"][i]) / 0.1])
        dat_b = {"observation.state": torch.tensor(row_state[None], dtype=torch.float32, device="cuda"),
                 "observation.environment_state": torch.tensor(row_env[None], dtype=torch.float32, device="cuda")}
        a_sim, a_dat = act(sim_b), act(dat_b)
        rows["obs_diff"].append(float(max((sim_b["observation.state"].cpu().numpy()[0] - row_state).__abs__().max(), (sim_b["observation.environment_state"].cpu().numpy()[0] - row_env).__abs__().max())))
        rows["data_vs_sim_action_px"].append(float(np.linalg.norm(a_sim - a_dat)))
        rows["data_action_err_px"].append(float(np.linalg.norm(a_dat - s["action"][i + 1])))
        rows["sim_action_err_px"].append(float(np.linalg.norm(a_sim - s["action"][i + 1])))
print(json.dumps({k: {"median": float(np.median(v)), "max": float(np.max(v))} for k, v in rows.items()}, indent=1))
