"""
=============================================================================
  DEEP RL GENERALIZATION CHALLENGE
  Failure Mode: The Exploration Cliff (Poor Generalization Across Scale)
  Mitigation:   Curriculum Learning (5x5 -> 8x8 via gradual transition)
=============================================================================

MiniGrid-DoorKey requires: find key -> pick up -> open locked door -> reach goal.

DoorKey-5x5: PPO learns this in ~60K steps. 100% success.
DoorKey-8x8: PPO gets 0% even after 500K steps. Complete failure.
(DoorKey-6x6 is also learnable; the cliff appears at 8x8 scale.)

Same algorithm, same hyperparameters, one extra row and column.

This is the "exploration cliff" -- a catastrophic failure of generalization
where a tiny increase in environment scale makes the sparse-reward exploration
problem combinatorially harder. The agent cannot discover the key->door->goal
sequence by random exploration in the larger grid.

Mitigation: curriculum learning. Train on 5x5 first (where exploration is
easy), then gradually transition to 8x8. The agent transfers its learned
strategy instead of rediscovering it from scratch.

Phases:
  1. Train PPO on DoorKey-5x5 (200K steps) -- succeeds
  2. Train PPO on DoorKey-8x8 (500K steps) -- fails
  3. Curriculum: 5x5 -> 8x8 transition (500K steps) -- succeeds
  4. Evaluate all three agents on both environments
  5. Generate visualizations and analysis

Usage:
  uv run python main.py              # full experiment (train + eval + plots)
  uv run python main.py --plots-only # regenerate plots from saved metrics
"""

import argparse
import os
import time
import json
import numpy as np
import torch

from rl_challenge.config import Config
from rl_challenge.trainer import train_on_env, train_curriculum, TrainingMetrics
from rl_challenge.evaluator import evaluate_agent, EvalResult
from rl_challenge.visualize import (
    plot_training_curves,
    plot_evaluation_comparison,
    plot_exploration_cliff,
    plot_curriculum_schedule,
    plot_summary_dashboard,
)
from rl_challenge.video_recorder import record_agent_gif

METRICS_FILE = "metrics_data.json"


def print_section(title: str):
    print(f"\n--- {title} ---\n")


def save_metrics(
    config: Config,
    easy_metrics: TrainingMetrics,
    hard_metrics: TrainingMetrics,
    curriculum_metrics: TrainingMetrics,
    results: dict[str, dict[str, EvalResult]],
    easy_time: float,
    hard_time: float,
    curriculum_time: float,
):
    """Save all metrics to disk for later plot regeneration."""
    data = {
        "training_metrics": {
            "easy": easy_metrics.to_dict(),
            "hard": hard_metrics.to_dict(),
            "curriculum": curriculum_metrics.to_dict(),
        },
        "evaluation": {
            name: {
                env_key: results[name][env_key].to_dict()
                for env_key in ["5x5", "8x8"]
            }
            for name in results
        },
        "timing": {
            "easy_time": easy_time,
            "hard_time": hard_time,
            "curriculum_time": curriculum_time,
        },
    }
    path = os.path.join(config.results_dir, METRICS_FILE)
    with open(path, "w") as f:
        json.dump(data, f)
    print(f"  Saved full metrics to {path}")


def load_metrics(config: Config):
    """Load metrics from disk."""
    path = os.path.join(config.results_dir, METRICS_FILE)
    with open(path) as f:
        data = json.load(f)

    easy_metrics = TrainingMetrics.from_dict(data["training_metrics"]["easy"])
    hard_metrics = TrainingMetrics.from_dict(data["training_metrics"]["hard"])
    curriculum_metrics = TrainingMetrics.from_dict(data["training_metrics"]["curriculum"])

    results = {}
    for name in data["evaluation"]:
        results[name] = {
            env_key: EvalResult.from_dict(data["evaluation"][name][env_key])
            for env_key in ["5x5", "8x8"]
        }

    timing = data["timing"]
    return easy_metrics, hard_metrics, curriculum_metrics, results, timing


def generate_plots(
    config: Config,
    easy_metrics: TrainingMetrics,
    hard_metrics: TrainingMetrics,
    curriculum_metrics: TrainingMetrics,
    results: dict[str, dict[str, EvalResult]],
):
    """Generate all visualization plots."""
    plots = [
        ("Training curves", lambda: plot_training_curves(
            easy_metrics, hard_metrics, curriculum_metrics,
            os.path.join(config.results_dir, "01_training_curves.png"))),
        ("Exploration cliff", lambda: plot_exploration_cliff(
            easy_metrics, hard_metrics,
            os.path.join(config.results_dir, "02_exploration_cliff.png"))),
        ("Evaluation comparison", lambda: plot_evaluation_comparison(
            results,
            os.path.join(config.results_dir, "03_evaluation_comparison.png"))),
        ("Curriculum schedule", lambda: plot_curriculum_schedule(
            curriculum_metrics, config.curriculum_warmup_frac,
            config.curriculum_full_hard_frac, config.curriculum_timesteps,
            os.path.join(config.results_dir, "04_curriculum_schedule.png"))),
        ("Summary dashboard", lambda: plot_summary_dashboard(
            results, easy_metrics, hard_metrics, curriculum_metrics,
            os.path.join(config.results_dir, "05_executive_summary.png"))),
    ]
    for i, (name, fn) in enumerate(plots, 1):
        print(f"  [{i}/{len(plots)}] {name}")
        fn()


def main():
    parser = argparse.ArgumentParser(description="Deep RL Generalization Challenge")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from saved metrics (skip training)")
    args = parser.parse_args()

    config = Config()
    os.makedirs(config.results_dir, exist_ok=True)

    if args.plots_only:
        print("\n  Loading saved metrics...")
        easy_metrics, hard_metrics, curriculum_metrics, results, timing = load_metrics(config)
        easy_time = timing["easy_time"]
        hard_time = timing["hard_time"]
        curriculum_time = timing["curriculum_time"]
        print("  Done.\n")

    else:
        print("\nRL Generalization Challenge: Curriculum Learning (5x5 -> 8x8)\n")

        torch.manual_seed(42)
        np.random.seed(42)

        # Phase 1: Train on DoorKey-5x5
        print_section("Training PPO on DoorKey-5x5")
        print(f"  {config.easy_env_id}, {config.easy_timesteps:,} steps")

        t0 = time.time()
        easy_agent, easy_metrics = train_on_env(
            config, config.easy_env_id, config.easy_timesteps, desc="5x5 PPO    ",
        )
        easy_time = time.time() - t0
        print(f"\n  Done in {easy_time:.1f}s -- success rate: {easy_metrics.success_rate:.0%}, mean reward: {easy_metrics.mean_reward:.3f}")

        # Phase 2: Train on DoorKey-8x8
        print_section("Training PPO on DoorKey-8x8 (baseline)")
        print(f"  {config.hard_env_id}, {config.hard_timesteps:,} steps")

        t0 = time.time()
        hard_agent, hard_metrics = train_on_env(
            config, config.hard_env_id, config.hard_timesteps, desc="8x8 PPO    ",
        )
        hard_time = time.time() - t0
        print(f"\n  Done in {hard_time:.1f}s -- success rate: {hard_metrics.success_rate:.0%}, mean reward: {hard_metrics.mean_reward:.3f}")

        # Phase 3: Curriculum learning
        print_section("Curriculum learning (5x5 -> 8x8)")
        print(f"  {config.curriculum_timesteps:,} steps, warmup {config.curriculum_warmup_frac:.0%}, full 8x8 at {config.curriculum_full_hard_frac:.0%}")

        t0 = time.time()
        curriculum_agent, curriculum_metrics = train_curriculum(config)
        curriculum_time = time.time() - t0
        print(f"\n  Done in {curriculum_time:.1f}s -- success rate: {curriculum_metrics.success_rate:.0%}, mean reward: {curriculum_metrics.mean_reward:.3f}")

        # Phase 4: Evaluation
        print_section("Evaluation")

        eval_kwargs = dict(
            n_seeds=config.eval_seeds,
            episodes_per_seed=config.eval_episodes_per_seed,
            max_steps=config.max_episode_steps,
            seed_offset=config.eval_seed_offset,
        )

        agents = {
            "5x5 Agent": easy_agent,
            "Direct 8x8": hard_agent,
            "Curriculum": curriculum_agent,
        }

        results: dict[str, dict[str, EvalResult]] = {}
        for name, agent in agents.items():
            print(f"  Evaluating {name}...")
            results[name] = {
                "5x5": evaluate_agent(agent, config.easy_env_id, desc=f"{name} on 5x5", **eval_kwargs),
                "8x8": evaluate_agent(agent, config.hard_env_id, desc=f"{name} on 8x8", **eval_kwargs),
            }

        print(f"\n  {'Agent':<16} {'5x5 Success':>12} {'8x8 Success':>12} {'8x8 Reward':>12}")
        print(f"  {'-'*52}")
        for name in agents:
            r5 = results[name]["5x5"]
            r6 = results[name]["8x8"]
            print(f"  {name:<16} {r5.success_rate:>11.0%} {r6.success_rate:>11.0%} {r6.mean_reward:>11.4f}")

        print_section("Saving metrics")
        save_metrics(config, easy_metrics, hard_metrics, curriculum_metrics,
                     results, easy_time, hard_time, curriculum_time)

        # Record agent GIFs
        print_section("Recording agent GIFs")

        gifs = [
            ("5x5 agent on 5x5", easy_agent, config.easy_env_id,
             os.path.join(config.results_dir, "5x5_agent_on_5x5.gif")),
            ("Direct 8x8 agent on 8x8", hard_agent, config.hard_env_id,
             os.path.join(config.results_dir, "direct_8x8_agent_on_8x8.gif")),
            ("Curriculum agent on 8x8", curriculum_agent, config.hard_env_id,
             os.path.join(config.results_dir, "curriculum_agent_on_8x8.gif")),
        ]

        for desc, agent, env_id, path in gifs:
            success = record_agent_gif(
                agent, env_id, path,
                max_steps=config.max_episode_steps,
            )
            tag = "ok" if success else "no goal reached"
            print(f"  {desc}: {tag}")

    # Generate plots
    print_section("Generating plots")
    generate_plots(config, easy_metrics, hard_metrics, curriculum_metrics, results)

    # Save results JSON
    json_results = {
        "experiment": "RL generalization: exploration cliff",
        "failure_mode": "DoorKey-5x5 succeeds, DoorKey-8x8 fails",
        "mitigation": "Curriculum learning (5x5 -> 8x8)",
        "training": {
            "easy_5x5": {
                "timesteps": config.easy_timesteps,
                "time_seconds": easy_time,
                "final_success_rate": easy_metrics.success_rate,
                "total_episodes": len(easy_metrics.episode_rewards),
            },
            "hard_8x8": {
                "timesteps": config.hard_timesteps,
                "time_seconds": hard_time,
                "final_success_rate": hard_metrics.success_rate,
                "total_episodes": len(hard_metrics.episode_rewards),
            },
            "curriculum": {
                "timesteps": config.curriculum_timesteps,
                "time_seconds": curriculum_time,
                "final_success_rate": curriculum_metrics.success_rate,
                "total_episodes": len(curriculum_metrics.episode_rewards),
            },
        },
        "evaluation": {
            name: {
                env: {
                    "success_rate": results[name][env].success_rate,
                    "mean_reward": results[name][env].mean_reward,
                    "mean_length": results[name][env].mean_length,
                }
                for env in ["5x5", "8x8"]
            }
            for name in results
        },
    }

    results_path = os.path.join(config.results_dir, "experiment_results.json")
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)

    # Final summary
    print_section("Results")

    r = results
    print(f"  5x5 PPO:     {r['5x5 Agent']['5x5'].success_rate:.0%} on 5x5 ({config.easy_timesteps:,} steps)")
    print(f"  8x8 PPO:     {r['Direct 8x8']['8x8'].success_rate:.0%} on 8x8 ({config.hard_timesteps:,} steps)")
    print(f"  Curriculum:   {r['Curriculum']['8x8'].success_rate:.0%} on 8x8 ({config.curriculum_timesteps:,} steps)")
    print(f"\n  Output saved to {config.results_dir}/\n")


if __name__ == "__main__":
    main()
