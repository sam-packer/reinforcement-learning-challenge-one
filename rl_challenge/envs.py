"""MiniGrid environment creation and wrappers."""

import gymnasium as gym
import numpy as np
from minigrid.wrappers import ImgObsWrapper


class NormalizeObsWrapper(gym.ObservationWrapper):
    """Normalize MiniGrid encoded observations to float32 and transpose to CHW.

    MiniGrid observations are NOT pixel images — they are compact encodings where
    each cell has 3 channels: (object_type, color, state) with small integer values
    (typically 0-10). We normalize by dividing by 10.0 (not 255!).
    """

    MINIGRID_MAX_VAL = 10.0  # max possible value in MiniGrid encoding

    def __init__(self, env: gym.Env):
        super().__init__(env)
        old_space = env.observation_space
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(old_space.shape[2], old_space.shape[0], old_space.shape[1]),
            dtype=np.float32,
        )

    def observation(self, obs: np.ndarray) -> np.ndarray:
        return obs.astype(np.float32).transpose(2, 0, 1) / self.MINIGRID_MAX_VAL


def make_env(env_id: str, seed: int | None = None, max_steps: int = 150) -> gym.Env:
    """Create a MiniGrid environment with standard wrappers."""
    env = gym.make(env_id, max_steps=max_steps)
    env = ImgObsWrapper(env)          # Dict obs → image only
    env = NormalizeObsWrapper(env)    # uint8 HWC → float32 CHW
    if seed is not None:
        env.reset(seed=seed)
    return env


def get_obs_shape(env_id: str = "MiniGrid-DoorKey-6x6-v0") -> tuple[int, ...]:
    """Return observation shape (C, H, W) for a given environment."""
    env = make_env(env_id)
    obs, _ = env.reset()
    shape = obs.shape
    env.close()
    return shape


def get_n_actions(env_id: str = "MiniGrid-DoorKey-6x6-v0") -> int:
    """Return number of discrete actions for a given environment."""
    env = make_env(env_id)
    n = env.action_space.n
    env.close()
    return n
