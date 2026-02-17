"""Evaluation suite for testing agent generalization across environments."""

import numpy as np
from dataclasses import dataclass, field
from tqdm import tqdm

from .config import Config
from .envs import make_env
from .ppo import PPOAgent


@dataclass
class EvalResult:
    """Results from evaluating an agent on a set of levels."""
    env_id: str
    rewards: list[float] = field(default_factory=list)
    successes: list[bool] = field(default_factory=list)
    lengths: list[int] = field(default_factory=list)

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self.rewards)) if self.rewards else 0.0

    @property
    def std_reward(self) -> float:
        return float(np.std(self.rewards)) if self.rewards else 0.0

    @property
    def success_rate(self) -> float:
        return float(np.mean(self.successes)) if self.successes else 0.0

    @property
    def mean_length(self) -> float:
        return float(np.mean(self.lengths)) if self.lengths else 0.0

    def to_dict(self) -> dict:
        return {
            "env_id": self.env_id,
            "rewards": self.rewards,
            "successes": self.successes,
            "lengths": self.lengths,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "EvalResult":
        return cls(
            env_id=d["env_id"],
            rewards=d["rewards"],
            successes=d["successes"],
            lengths=d["lengths"],
        )


def evaluate_agent(
    agent: PPOAgent,
    env_id: str,
    n_seeds: int = 50,
    episodes_per_seed: int = 3,
    max_steps: int = 150,
    seed_offset: int = 5000,
    desc: str = "Evaluating",
) -> EvalResult:
    """Evaluate agent across a set of level seeds."""
    result = EvalResult(env_id=env_id)

    for i in tqdm(range(n_seeds), desc=desc, leave=False):
        seed = seed_offset + i
        for _ in range(episodes_per_seed):
            env = make_env(env_id, max_steps=max_steps)
            obs, _ = env.reset(seed=seed)
            total_reward = 0.0
            steps = 0
            done = False

            while not done:
                action, _, _ = agent.act(obs)
                obs, reward, terminated, truncated, info = env.step(action)
                total_reward += reward
                steps += 1
                done = terminated or truncated

            env.close()
            result.rewards.append(total_reward)
            result.successes.append(total_reward > 0)
            result.lengths.append(steps)

    return result
