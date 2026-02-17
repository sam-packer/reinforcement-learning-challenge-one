"""Training pipelines: standard PPO and curriculum learning PPO."""

import time
import numpy as np
from tqdm import tqdm

from .config import Config
from .envs import make_env, get_obs_shape, get_n_actions
from .ppo import PPOAgent


class TrainingMetrics:
    """Accumulates and stores training metrics over time."""

    def __init__(self):
        self.steps: list[int] = []
        self.episode_rewards: list[float] = []
        self.episode_lengths: list[int] = []
        self.episode_successes: list[bool] = []
        self.pg_losses: list[float] = []
        self.vf_losses: list[float] = []
        self.entropies: list[float] = []
        self.gae_magnitudes: list[float] = []
        # Running episode tracking
        self._current_reward = 0.0
        self._current_length = 0
        self._reward_window: list[float] = []
        self._success_window: list[bool] = []

    def add_step(self, reward: float, done: bool):
        self._current_reward += reward
        self._current_length += 1
        if done:
            self.episode_rewards.append(self._current_reward)
            self.episode_lengths.append(self._current_length)
            success = self._current_reward > 0
            self.episode_successes.append(success)
            self._reward_window.append(self._current_reward)
            self._success_window.append(success)
            if len(self._reward_window) > 100:
                self._reward_window.pop(0)
                self._success_window.pop(0)
            self._current_reward = 0.0
            self._current_length = 0

    def add_update(self, step: int, update_info: dict):
        self.steps.append(step)
        self.pg_losses.append(update_info["pg_loss"])
        self.vf_losses.append(update_info["vf_loss"])
        self.entropies.append(update_info["entropy"])
        self.gae_magnitudes.append(update_info["gae_magnitude"])

    @property
    def mean_reward(self) -> float:
        return float(np.mean(self._reward_window)) if self._reward_window else 0.0

    @property
    def success_rate(self) -> float:
        return float(np.mean(self._success_window)) if self._success_window else 0.0

    def to_dict(self) -> dict:
        return {
            "steps": self.steps,
            "episode_rewards": self.episode_rewards,
            "episode_lengths": self.episode_lengths,
            "episode_successes": self.episode_successes,
            "pg_losses": self.pg_losses,
            "vf_losses": self.vf_losses,
            "entropies": self.entropies,
            "gae_magnitudes": self.gae_magnitudes,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TrainingMetrics":
        m = cls()
        m.steps = d["steps"]
        m.episode_rewards = d["episode_rewards"]
        m.episode_lengths = d["episode_lengths"]
        m.episode_successes = d["episode_successes"]
        m.pg_losses = d["pg_losses"]
        m.vf_losses = d["vf_losses"]
        m.entropies = d["entropies"]
        m.gae_magnitudes = d["gae_magnitudes"]
        # Rebuild running windows from last 100 episodes
        m._reward_window = m.episode_rewards[-100:]
        m._success_window = m.episode_successes[-100:]
        return m


def _make_agent(config: Config, env_id: str) -> PPOAgent:
    """Create a PPO agent sized for the given environment."""
    obs_shape = get_obs_shape(env_id)
    n_actions = get_n_actions(env_id)
    return PPOAgent(
        obs_shape=obs_shape,
        n_actions=n_actions,
        lr=config.lr,
        gamma=config.gamma,
        gae_lambda=config.gae_lambda,
        clip_range=config.clip_range,
        ent_coef=config.ent_coef,
        vf_coef=config.vf_coef,
        max_grad_norm=config.max_grad_norm,
        n_epochs=config.n_epochs,
        batch_size=config.batch_size,
    )


def train_on_env(
    config: Config,
    env_id: str,
    total_timesteps: int,
    desc: str = "PPO",
    agent: PPOAgent | None = None,
) -> tuple[PPOAgent, TrainingMetrics]:
    """Train a PPO agent on a single environment.

    If agent is provided, continues training that agent (for curriculum).
    Otherwise creates a fresh agent.
    """
    if agent is None:
        agent = _make_agent(config, env_id)

    metrics = TrainingMetrics()
    env = make_env(env_id, max_steps=config.max_episode_steps)
    obs, _ = env.reset()

    total_steps = 0
    start_time = time.time()
    pbar = tqdm(total=total_timesteps, desc=desc, unit="step")

    while total_steps < total_timesteps:
        for _ in range(config.rollout_steps):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.add(obs, action, reward, done, log_prob, value)
            metrics.add_step(reward, done)
            total_steps += 1
            pbar.update(1)

            if done:
                env.close()
                env = make_env(env_id, max_steps=config.max_episode_steps)
                obs, _ = env.reset()
            else:
                obs = next_obs

        _, _, last_value = agent.act(obs)
        update_info = agent.update(last_value)
        metrics.add_update(total_steps, update_info)

        if total_steps % config.log_interval < config.rollout_steps:
            elapsed = time.time() - start_time
            pbar.set_postfix({
                "reward": f"{metrics.mean_reward:.2f}",
                "success": f"{metrics.success_rate:.0%}",
                "entropy": f"{update_info['entropy']:.3f}",
                "fps": f"{total_steps / elapsed:.0f}",
            })

    pbar.close()
    env.close()
    return agent, metrics


def train_curriculum(config: Config) -> tuple[PPOAgent, TrainingMetrics]:
    """Train with curriculum learning: start on 5x5, gradually transition to 6x6.

    Schedule:
      [0, warmup_frac)          -> 100% easy (5x5)
      [warmup_frac, full_hard)  -> linear mix, easy% decreasing to 0%
      [full_hard, 1.0]          -> 100% hard (6x6)

    Same total budget as direct 6x6 training for fair comparison.
    """
    # Both envs have same obs shape (3,7,7) and action space (7)
    agent = _make_agent(config, config.easy_env_id)
    metrics = TrainingMetrics()

    total_timesteps = config.curriculum_timesteps
    warmup_steps = int(total_timesteps * config.curriculum_warmup_frac)
    full_hard_steps = int(total_timesteps * config.curriculum_full_hard_frac)

    # Start on easy env
    current_env_id = config.easy_env_id
    env = make_env(current_env_id, max_steps=config.max_episode_steps)
    obs, _ = env.reset()

    total_steps = 0
    start_time = time.time()
    pbar = tqdm(total=total_timesteps, desc="Curriculum ", unit="step")

    def get_hard_probability(step: int) -> float:
        """Probability of using the hard env at a given step."""
        if step < warmup_steps:
            return 0.0
        elif step >= full_hard_steps:
            return 1.0
        else:
            return (step - warmup_steps) / (full_hard_steps - warmup_steps)

    while total_steps < total_timesteps:
        for _ in range(config.rollout_steps):
            action, log_prob, value = agent.act(obs)
            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            agent.buffer.add(obs, action, reward, done, log_prob, value)
            metrics.add_step(reward, done)
            total_steps += 1
            pbar.update(1)

            if done:
                env.close()
                # Curriculum: pick env based on schedule
                p_hard = get_hard_probability(total_steps)
                if np.random.random() < p_hard:
                    current_env_id = config.hard_env_id
                else:
                    current_env_id = config.easy_env_id
                env = make_env(current_env_id, max_steps=config.max_episode_steps)
                obs, _ = env.reset()
            else:
                obs = next_obs

        _, _, last_value = agent.act(obs)
        update_info = agent.update(last_value)
        metrics.add_update(total_steps, update_info)

        if total_steps % config.log_interval < config.rollout_steps:
            elapsed = time.time() - start_time
            p_hard = get_hard_probability(total_steps)
            pbar.set_postfix({
                "reward": f"{metrics.mean_reward:.2f}",
                "success": f"{metrics.success_rate:.0%}",
                "hard%": f"{p_hard:.0%}",
                "fps": f"{total_steps / elapsed:.0f}",
            })

    pbar.close()
    env.close()
    return agent, metrics
