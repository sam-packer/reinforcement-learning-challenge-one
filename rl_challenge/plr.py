"""Prioritized Level Replay (PLR) — adaptive curriculum for generalization.

Based on the insight from Jiang et al. (2021): instead of training on random or
fixed levels, prioritize replaying levels where the agent has the most to learn
(measured by GAE magnitude / value prediction error). This prevents memorization
of easy levels and forces the agent to confront its weaknesses.

Key mechanisms:
  1. Score buffer: each level seed gets a priority score based on learning signal
  2. Staleness bonus: levels not seen recently get a sampling boost
  3. Temperature-controlled sampling: soft prioritization via Boltzmann distribution
  4. New level injection: periodically introduce unseen levels to expand coverage
"""

import numpy as np
from dataclasses import dataclass, field


@dataclass
class LevelRecord:
    """Tracks metadata for a single procedural level."""
    seed: int
    score: float = 0.0         # priority score (mean |GAE|)
    times_played: int = 0
    last_played: int = 0       # global step when last played
    cumulative_reward: float = 0.0
    successes: int = 0
    attempts: int = 0

    @property
    def success_rate(self) -> float:
        return self.successes / max(1, self.attempts)


class PrioritizedLevelReplay:
    """PLR level scheduler that prioritizes high-learning-potential levels."""

    def __init__(
        self,
        seed_pool: list[int],
        buffer_size: int = 100,
        replay_rate: float = 0.5,
        staleness_coeff: float = 0.1,
        temperature: float = 0.1,
        score_ema: float = 0.3,
    ):
        self.seed_pool = list(seed_pool)
        self.buffer_size = buffer_size
        self.replay_rate = replay_rate
        self.staleness_coeff = staleness_coeff
        self.temperature = temperature
        self.score_ema = score_ema

        self.buffer: dict[int, LevelRecord] = {}
        self.unseen_seeds = list(seed_pool)
        np.random.shuffle(self.unseen_seeds)
        self.global_step = 0

        # Analytics
        self.sampling_history: list[tuple[int, int, str]] = []  # (step, seed, reason)

    def sample_level(self) -> int:
        """Sample a level seed using PLR prioritization.

        With probability `replay_rate`, replay a level from the buffer
        (prioritized by score + staleness). Otherwise, introduce a new level.
        """
        self.global_step += 1

        # Decide: replay or explore
        if self.buffer and np.random.random() < self.replay_rate:
            seed = self._sample_from_buffer()
            reason = "replay"
        else:
            seed = self._sample_new_level()
            reason = "new"

        self.sampling_history.append((self.global_step, seed, reason))
        return seed

    def _sample_from_buffer(self) -> int:
        """Sample from buffer using softmax over score + staleness bonus."""
        seeds = list(self.buffer.keys())
        records = [self.buffer[s] for s in seeds]

        scores = np.array([r.score for r in records])
        staleness = np.array([
            self.global_step - r.last_played for r in records
        ], dtype=np.float32)

        # Combine score with staleness bonus
        priorities = scores + self.staleness_coeff * staleness

        # Boltzmann sampling
        priorities = priorities / max(self.temperature, 1e-8)
        priorities -= priorities.max()  # numerical stability
        exp_p = np.exp(priorities)
        probs = exp_p / exp_p.sum()

        idx = np.random.choice(len(seeds), p=probs)
        return seeds[idx]

    def _sample_new_level(self) -> int:
        """Get an unseen level, or fall back to buffer replay."""
        if self.unseen_seeds:
            return self.unseen_seeds.pop()
        elif self.buffer:
            return self._sample_from_buffer()
        else:
            return np.random.choice(self.seed_pool)

    def update_score(
        self,
        seed: int,
        gae_magnitude: float,
        episode_reward: float,
        success: bool,
    ):
        """Update a level's priority score after training on it."""
        if seed not in self.buffer:
            if len(self.buffer) >= self.buffer_size:
                # Evict lowest-scoring level
                worst_seed = min(self.buffer, key=lambda s: self.buffer[s].score)
                if self.buffer[worst_seed].score < gae_magnitude:
                    del self.buffer[worst_seed]
                else:
                    # All levels are higher priority, skip adding
                    return

            self.buffer[seed] = LevelRecord(seed=seed)

        record = self.buffer[seed]
        # EMA update of score
        record.score = (
            self.score_ema * gae_magnitude + (1 - self.score_ema) * record.score
        )
        record.times_played += 1
        record.last_played = self.global_step
        record.cumulative_reward += episode_reward
        record.attempts += 1
        if success:
            record.successes += 1

    def get_stats(self) -> dict:
        """Return PLR buffer statistics for logging."""
        if not self.buffer:
            return {"plr_buffer_size": 0}

        records = list(self.buffer.values())
        scores = [r.score for r in records]
        success_rates = [r.success_rate for r in records]
        times_played = [r.times_played for r in records]

        return {
            "plr_buffer_size": len(self.buffer),
            "plr_mean_score": float(np.mean(scores)),
            "plr_max_score": float(np.max(scores)),
            "plr_min_score": float(np.min(scores)),
            "plr_mean_success_rate": float(np.mean(success_rates)),
            "plr_mean_times_played": float(np.mean(times_played)),
            "plr_unseen_remaining": len(self.unseen_seeds),
        }

    def get_top_levels(self, n: int = 10) -> list[LevelRecord]:
        """Return top-N highest-priority levels."""
        records = sorted(self.buffer.values(), key=lambda r: r.score, reverse=True)
        return records[:n]

    def get_bottom_levels(self, n: int = 10) -> list[LevelRecord]:
        """Return bottom-N lowest-priority levels (easiest for agent)."""
        records = sorted(self.buffer.values(), key=lambda r: r.score)
        return records[:n]
