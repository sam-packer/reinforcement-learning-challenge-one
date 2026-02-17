"""Experiment configuration and hyperparameters."""

from dataclasses import dataclass


@dataclass
class Config:
    # --- Environment ---
    # DoorKey: agent must find a key, pick it up, open a locked door, reach the goal.
    # 5x5 is learnable (~60K steps). 6x6 is nearly impossible from scratch (exploration cliff).
    easy_env_id: str = "MiniGrid-DoorKey-5x5-v0"
    hard_env_id: str = "MiniGrid-DoorKey-8x8-v0"
    max_episode_steps: int = 300

    # --- Training budgets ---
    easy_timesteps: int = 200_000     # 5x5 converges well within this
    hard_timesteps: int = 500_000     # 6x6 still won't learn, proving the failure
    curriculum_timesteps: int = 500_000  # same budget as direct 6x6 for fair comparison

    # --- Curriculum schedule ---
    # fraction of curriculum_timesteps spent purely on 5x5 before mixing in 6x6
    curriculum_warmup_frac: float = 0.30
    # fraction of curriculum_timesteps at which we're 100% on 6x6
    curriculum_full_hard_frac: float = 0.80

    # --- PPO hyperparameters ---
    rollout_steps: int = 256
    n_epochs: int = 4
    batch_size: int = 128
    lr: float = 3e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range: float = 0.2
    ent_coef: float = 0.05
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5

    # --- Evaluation ---
    eval_seeds: int = 50             # number of seeds to evaluate on each env
    eval_episodes_per_seed: int = 3
    eval_seed_offset: int = 5000     # avoid overlap with training

    # --- Output ---
    results_dir: str = "results"
    log_interval: int = 5000
