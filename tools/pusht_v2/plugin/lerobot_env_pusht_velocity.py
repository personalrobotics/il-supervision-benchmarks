"""gym_pusht with a velocity-aware observation, as a LeRobot 0.6.1 env plugin (env type `pusht_velocity`).

Each position feature x_t is observed as [x_t, (x_t - x_{t-1}) / dt], dt = 1 / fps: `agent_pos` becomes 4 values and
`environment_state` (the T's 8 keypoints) 32, the layout of the derived dataset (tools/pusht_v2/derive.py). They stay
in LeRobot's two observation features because LeRobot policies read only observation.state and
observation.environment_state. A reset places the agent and the T at rest, so the first observation has velocity 0:
x_{-1} = x_0. lerobot-eval imports this module unchanged, as it imports every installed `lerobot_env_*` package.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig, PushtEnv

GYM_ID = "lerobot_env_pusht_velocity/PushT-v0"
KEYS = ("agent_pos", "environment_state")
WIDTHS = {"agent_pos": 2, "environment_state": 16}


class VelocityObservation(gym.Wrapper):
    def __init__(self, env, dt: float):
        super().__init__(env)
        self.dt, self.prev = dt, None
        self.observation_space = spaces.Dict(
            {k: spaces.Box(-np.inf, np.inf, (2 * n,), np.float64) for k, n in WIDTHS.items()})

    def augment(self, obs):
        cur = {k: np.asarray(obs[k], np.float64) for k in KEYS}
        prev = cur if self.prev is None else self.prev
        self.prev = cur
        return {k: np.concatenate([cur[k], (cur[k] - prev[k]) / self.dt]) for k in KEYS}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = None
        return self.augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.augment(obs), reward, terminated, truncated, info


def make(**kwargs):
    from gym_pusht.envs import PushTEnv
    env = PushTEnv(**kwargs)
    return VelocityObservation(env, 1.0 / env.metadata["render_fps"])


if GYM_ID not in gym.registry:
    gym.register(id=GYM_ID, entry_point=make)


@EnvConfig.register_subclass("pusht_velocity")
@dataclass
class PushtVelocityEnv(PushtEnv):
    obs_type: str = "environment_state_agent_pos"

    def __post_init__(self):
        if self.obs_type != "environment_state_agent_pos":
            raise ValueError("pusht_velocity observes environment_state_agent_pos only")
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
        self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(32,))

    @property
    def package_name(self) -> str:
        return "lerobot_env_pusht_velocity"
