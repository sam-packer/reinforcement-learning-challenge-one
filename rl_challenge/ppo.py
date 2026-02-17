"""Proximal Policy Optimization (PPO) implementation from scratch."""

import numpy as np
import torch
import torch.nn as nn

from .networks import ActorCritic


class RolloutBuffer:
    """Stores rollout data for PPO updates."""

    def __init__(self):
        self.observations: list[np.ndarray] = []
        self.actions: list[int] = []
        self.rewards: list[float] = []
        self.dones: list[bool] = []
        self.log_probs: list[float] = []
        self.values: list[float] = []

    def add(self, obs, action, reward, done, log_prob, value):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)

    def clear(self):
        self.__init__()

    def __len__(self):
        return len(self.observations)


class PPOAgent:
    """PPO agent with GAE and clipped surrogate objective."""

    def __init__(
        self,
        obs_shape: tuple[int, ...],
        n_actions: int,
        lr: float = 2.5e-4,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
        clip_range: float = 0.2,
        ent_coef: float = 0.01,
        vf_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        n_epochs: int = 4,
        batch_size: int = 64,
        device: str = "cpu",
    ):
        self.gamma = gamma
        self.gae_lambda = gae_lambda
        self.clip_range = clip_range
        self.ent_coef = ent_coef
        self.vf_coef = vf_coef
        self.max_grad_norm = max_grad_norm
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.device = torch.device(device)

        self.network = ActorCritic(obs_shape, n_actions).to(self.device)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=lr, eps=1e-5)
        self.buffer = RolloutBuffer()

    def act(self, obs: np.ndarray) -> tuple[int, float, float]:
        """Select action given observation."""
        obs_tensor = torch.from_numpy(obs).unsqueeze(0).to(self.device)
        return self.network.act(obs_tensor)

    def compute_gae(self, last_value: float) -> tuple[np.ndarray, np.ndarray]:
        """Compute Generalized Advantage Estimation.

        Returns:
            advantages: GAE advantages for each timestep
            returns: discounted returns (advantages + values)
        """
        rewards = np.array(self.buffer.rewards)
        values = np.array(self.buffer.values)
        dones = np.array(self.buffer.dones, dtype=np.float32)
        n_steps = len(rewards)

        advantages = np.zeros(n_steps, dtype=np.float32)
        last_gae = 0.0

        for t in reversed(range(n_steps)):
            next_value = last_value if t == n_steps - 1 else values[t + 1]
            next_non_terminal = 1.0 - (1.0 if t == n_steps - 1 else dones[t + 1])

            # For the current step: if done[t] is True, this transition ended
            # an episode, so next_value should not be bootstrapped
            if dones[t] and t < n_steps - 1:
                next_non_terminal = 0.0
                next_value = 0.0

            delta = rewards[t] + self.gamma * next_value * (1.0 - dones[t]) - values[t]
            last_gae = delta + self.gamma * self.gae_lambda * (1.0 - dones[t]) * last_gae
            advantages[t] = last_gae

        returns = advantages + values
        return advantages, returns

    def update(self, last_value: float) -> dict[str, float]:
        """Run PPO update on collected rollout data.

        Returns dict of training metrics.
        """
        advantages, returns = self.compute_gae(last_value)

        # Convert buffer to tensors (ensure float32 throughout)
        obs = torch.from_numpy(np.array(self.buffer.observations, dtype=np.float32)).to(self.device)
        actions = torch.tensor(self.buffer.actions, dtype=torch.long, device=self.device)
        old_log_probs = torch.tensor(self.buffer.log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.from_numpy(advantages.astype(np.float32)).to(self.device)
        returns_t = torch.from_numpy(returns.astype(np.float32)).to(self.device)

        # Normalize advantages
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)

        n_samples = len(self.buffer)
        total_pg_loss = 0.0
        total_vf_loss = 0.0
        total_entropy = 0.0
        total_approx_kl = 0.0
        n_updates = 0

        for _ in range(self.n_epochs):
            indices = np.random.permutation(n_samples)
            for start in range(0, n_samples, self.batch_size):
                end = min(start + self.batch_size, n_samples)
                batch_idx = indices[start:end]

                b_obs = obs[batch_idx]
                b_actions = actions[batch_idx]
                b_old_lp = old_log_probs[batch_idx]
                b_adv = advantages_t[batch_idx]
                b_ret = returns_t[batch_idx]

                new_log_probs, values, entropy = self.network.evaluate_actions(b_obs, b_actions)

                # PPO clipped surrogate
                ratio = torch.exp(new_log_probs - b_old_lp)
                surr1 = ratio * b_adv
                surr2 = torch.clamp(ratio, 1.0 - self.clip_range, 1.0 + self.clip_range) * b_adv
                pg_loss = -torch.min(surr1, surr2).mean()

                # Value function loss
                vf_loss = nn.functional.mse_loss(values, b_ret)

                # Entropy bonus
                entropy_loss = -entropy.mean()

                # Total loss
                loss = pg_loss + self.vf_coef * vf_loss + self.ent_coef * entropy_loss

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.network.parameters(), self.max_grad_norm)
                self.optimizer.step()

                # Track metrics
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - (new_log_probs - b_old_lp)).mean()
                total_pg_loss += pg_loss.item()
                total_vf_loss += vf_loss.item()
                total_entropy += -entropy_loss.item()
                total_approx_kl += approx_kl.item()
                n_updates += 1

        # Store GAE magnitudes for PLR scoring
        self._last_gae = advantages

        self.buffer.clear()

        return {
            "pg_loss": total_pg_loss / n_updates,
            "vf_loss": total_vf_loss / n_updates,
            "entropy": total_entropy / n_updates,
            "approx_kl": total_approx_kl / n_updates,
            "mean_advantage": float(advantages.mean()),
            "gae_magnitude": float(np.mean(np.abs(advantages))),
        }

    @property
    def last_gae_magnitude(self) -> float:
        """Return mean |GAE| from last update (used for PLR scoring)."""
        if hasattr(self, "_last_gae"):
            return float(np.mean(np.abs(self._last_gae)))
        return 0.0
