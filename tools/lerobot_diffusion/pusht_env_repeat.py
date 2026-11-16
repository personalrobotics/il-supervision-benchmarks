"""Does gym_pusht reproduce a trajectory bit for bit across processes, given the reset seed and the actions?

    $ILSB_LEROBOT_PYTHON -I tools/lerobot_diffusion/pusht_env_repeat.py   # run it several times; compare the digests

Twenty episodes from reset seeds 1000-1019, 300 steps each, of a fixed seeded action sequence that drives the agent at
the block (so most steps have contact). It prints a digest of every observation and reward, and the number of steps
with contact. With gym-pusht 0.1.8 and pymunk on siddh, three processes printed three different digests: the
simulator does not repeat contact-rich trajectories across processes (docs/lerobot-learners.md).
"""
import hashlib, sys
import numpy as np, gymnasium as gym, gym_pusht  # noqa: F401
env = gym.make("gym_pusht/PushT-v0", obs_type="environment_state_agent_pos")
h, contacts = hashlib.sha256(), 0
for ep in range(20):
    obs, _ = env.reset(seed=1000 + ep)
    rng = np.random.default_rng(ep)
    for t in range(300):  # a random walk of the agent towards the block, so it pushes it
        tgt = obs["environment_state"][:2] if t % 60 < 40 else rng.uniform(100, 400, 2)
        obs, r, term, trunc, info = env.step(np.clip(tgt + rng.normal(0, 20, 2), 0, 512))
        contacts += int(env.unwrapped.n_contact_points > 0)
        h.update(obs["environment_state"].tobytes() + obs["agent_pos"].tobytes() + np.float64(r).tobytes())
print(h.hexdigest()[:16], "steps with contact", contacts)
