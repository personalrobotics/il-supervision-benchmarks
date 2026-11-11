"""gym_pusht with a two-frame history observation, as a LeRobot 0.6.1 env plugin (env type `pusht_history`).

Each position feature x_t is observed as [x_{t-1}, x_t]: `agent_pos` becomes 4 values and `environment_state` (the
T's 8 keypoints) 32, the layout of the derived dataset (tools/pusht_v2/derive.py). The history stays inside LeRobot's
two observation features because LeRobot policies read only observation.state and observation.environment_state,
and because a CCIL one-frame synthetic episode then carries its own history. A reset places the agent and the T at
rest, so the first observation repeats its frame: x_{-1} = x_0. lerobot-eval imports this module unchanged, as it
imports every installed `lerobot_env_*` package.
"""
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
from gymnasium import spaces
from lerobot.configs import FeatureType, PolicyFeature
from lerobot.envs.configs import EnvConfig, PushtEnv

GYM_ID = "lerobot_env_pusht_history/PushT-v0"
KEYS = ("agent_pos", "environment_state")
WIDTHS = {"agent_pos": 2, "environment_state": 16}


class HistoryObservation(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.prev = None
        self.observation_space = spaces.Dict(
            {k: spaces.Box(-np.inf, np.inf, (2 * n,), np.float64) for k, n in WIDTHS.items()})

    def augment(self, obs):
        cur = {k: np.asarray(obs[k], np.float64) for k in KEYS}
        prev = cur if self.prev is None else self.prev
        self.prev = cur
        return {k: np.concatenate([prev[k], cur[k]]) for k in KEYS}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        self.prev = None
        return self.augment(obs), info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        return self.augment(obs), reward, terminated, truncated, info


def make(**kwargs):
    from gym_pusht.envs import PushTEnv
    return HistoryObservation(PushTEnv(**kwargs))


if GYM_ID not in gym.registry:
    gym.register(id=GYM_ID, entry_point=make)


@EnvConfig.register_subclass("pusht_history")
@dataclass
class PushtHistoryEnv(PushtEnv):
    obs_type: str = "environment_state_agent_pos"

    def __post_init__(self):
        if self.obs_type != "environment_state_agent_pos":
            raise ValueError("pusht_history observes environment_state_agent_pos only")
        self.features["agent_pos"] = PolicyFeature(type=FeatureType.STATE, shape=(4,))
        self.features["environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(32,))

    @property
    def package_name(self) -> str:
        return "lerobot_env_pusht_history"
