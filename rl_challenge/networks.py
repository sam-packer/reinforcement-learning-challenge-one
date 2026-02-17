"""Neural network architectures for PPO actor-critic."""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical


class ActorCritic(nn.Module):
    """CNN-based actor-critic network for MiniGrid partial observations.

    Architecture:
        3-layer CNN encoder (3→16→32→64) with ReLU
        → flatten → shared 256-dim feature layer
        → actor head (policy logits)
        → critic head (state value)
    """

    def __init__(self, obs_shape: tuple[int, ...], n_actions: int):
        super().__init__()
        c, h, w = obs_shape

        self.encoder = nn.Sequential(
            nn.Conv2d(c, 16, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=2, stride=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute flattened feature size dynamically
        with torch.no_grad():
            dummy = torch.zeros(1, c, h, w)
            feat_size = self.encoder(dummy).shape[1]

        self.shared = nn.Sequential(
            nn.Linear(feat_size, 256),
            nn.ReLU(),
        )

        self.actor = nn.Linear(256, n_actions)
        self.critic = nn.Linear(256, 1)

        self._init_weights()

    def _init_weights(self):
        """Orthogonal initialization (standard for PPO)."""
        for module in self.modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                nn.init.orthogonal_(module.weight, gain=np.sqrt(2))
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
        # Scale down actor and critic final layers
        nn.init.orthogonal_(self.actor.weight, gain=0.01)
        nn.init.orthogonal_(self.critic.weight, gain=1.0)

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning policy logits and state value."""
        features = self.shared(self.encoder(obs))
        logits = self.actor(features)
        value = self.critic(features).squeeze(-1)
        return logits, value

    def act(self, obs: torch.Tensor) -> tuple[int, float, float]:
        """Select action, return (action, log_prob, value)."""
        with torch.no_grad():
            logits, value = self(obs)
            dist = Categorical(logits=logits)
            action = dist.sample()
            log_prob = dist.log_prob(action)
        return action.item(), log_prob.item(), value.item()

    def evaluate_actions(
        self, obs: torch.Tensor, actions: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Evaluate actions for PPO update: log_probs, values, entropy."""
        logits, values = self(obs)
        dist = Categorical(logits=logits)
        log_probs = dist.log_prob(actions)
        entropy = dist.entropy()
        return log_probs, values, entropy

    def get_features(self, obs: torch.Tensor) -> torch.Tensor:
        """Extract learned feature representations (for analysis)."""
        with torch.no_grad():
            return self.shared(self.encoder(obs))
