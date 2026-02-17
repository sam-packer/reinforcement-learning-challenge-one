"""Record agent episodes as GIF animations."""

import gymnasium as gym
import imageio
import numpy as np
from minigrid.wrappers import ImgObsWrapper

from .ppo import PPOAgent


def record_agent_gif(
    agent: PPOAgent,
    env_id: str,
    save_path: str,
    max_steps: int = 300,
    seed: int = 42,
    fps: int = 4,
) -> bool:
    """Record one episode of the agent acting and save as GIF.

    Uses render_mode="rgb_array" to capture frames. Observations are manually
    normalized for the agent (matching NormalizeObsWrapper behavior) without
    wrapping the env, so we can still call env.render() for raw pixel frames.

    Returns True if the agent reached the goal (got positive reward).
    """
    env = gym.make(env_id, render_mode="rgb_array", max_steps=max_steps)
    env = ImgObsWrapper(env)

    obs, _ = env.reset(seed=seed)
    frames = [env.render()]

    done = False
    total_reward = 0.0
    while not done:
        # Manually normalize observation for the agent (CHW float32, /10.0)
        obs_normalized = obs.astype(np.float32).transpose(2, 0, 1) / 10.0
        action, _, _ = agent.act(obs_normalized)
        obs, reward, terminated, truncated, _ = env.step(action)
        total_reward += reward
        frames.append(env.render())
        done = terminated or truncated

    env.close()

    imageio.mimsave(save_path, frames, fps=fps, loop=0)
    return total_reward > 0
