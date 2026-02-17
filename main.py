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
    width = 66
    print()
    print("=" * width)
    print(f"  {title}")
    print("=" * width)
    print()


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
    print("  1/5 Training curves...")
    plot_training_curves(
        easy_metrics, hard_metrics, curriculum_metrics,
        os.path.join(config.results_dir, "01_training_curves.png"),
    )

    print("  2/5 Exploration cliff...")
    plot_exploration_cliff(
        easy_metrics, hard_metrics,
        os.path.join(config.results_dir, "02_exploration_cliff.png"),
    )

    print("  3/5 Evaluation comparison...")
    plot_evaluation_comparison(
        results,
        os.path.join(config.results_dir, "03_evaluation_comparison.png"),
    )

    print("  4/5 Curriculum schedule...")
    plot_curriculum_schedule(
        curriculum_metrics,
        config.curriculum_warmup_frac,
        config.curriculum_full_hard_frac,
        config.curriculum_timesteps,
        os.path.join(config.results_dir, "04_curriculum_schedule.png"),
    )

    print("  5/5 Executive summary...")
    plot_summary_dashboard(
        results, easy_metrics, hard_metrics, curriculum_metrics,
        os.path.join(config.results_dir, "05_executive_summary.png"),
    )


def main():
    parser = argparse.ArgumentParser(description="Deep RL Generalization Challenge")
    parser.add_argument("--plots-only", action="store_true",
                        help="Regenerate plots from saved metrics (skip training)")
    args = parser.parse_args()

    config = Config()
    os.makedirs(config.results_dir, exist_ok=True)

    if args.plots_only:
        print("\n  Loading saved metrics from disk...")
        easy_metrics, hard_metrics, curriculum_metrics, results, timing = load_metrics(config)
        easy_time = timing["easy_time"]
        hard_time = timing["hard_time"]
        curriculum_time = timing["curriculum_time"]
        print("  Loaded successfully.\n")

    else:
        print("""
    +==================================================================+
    |  DEEP RL GENERALIZATION CHALLENGE                                |
    |  Failure Mode:  Poor Generalization (Exploration Cliff)          |
    |  Mitigation:    Curriculum Learning (5x5 -> 8x8)                |
    +==================================================================+
        """)

        torch.manual_seed(42)
        np.random.seed(42)

        # ==============================================================
        # PHASE 1: Train on DoorKey-5x5 (should succeed)
        # ==============================================================
        print_section("PHASE 1: Train PPO on DoorKey-5x5 (easy)")
        print(f"  Environment:  {config.easy_env_id}")
        print(f"  Budget:       {config.easy_timesteps:,} steps")
        print()

        t0 = time.time()
        easy_agent, easy_metrics = train_on_env(
            config, config.easy_env_id, config.easy_timesteps, desc="5x5 PPO    ",
        )
        easy_time = time.time() - t0
        print(f"\n  Completed in {easy_time:.1f}s")
        print(f"  Final success rate: {easy_metrics.success_rate:.0%}")
        print(f"  Final mean reward:  {easy_metrics.mean_reward:.3f}")

        # ==============================================================
        # PHASE 2: Train on DoorKey-8x8 (should fail)
        # ==============================================================
        print_section("PHASE 2: Train PPO on DoorKey-8x8 (hard -- expect failure)")
        print(f"  Environment:  {config.hard_env_id}")
        print(f"  Budget:       {config.hard_timesteps:,} steps")
        print(f"  This demonstrates the exploration cliff.")
        print()

        t0 = time.time()
        hard_agent, hard_metrics = train_on_env(
            config, config.hard_env_id, config.hard_timesteps, desc="8x8 PPO    ",
        )
        hard_time = time.time() - t0
        print(f"\n  Completed in {hard_time:.1f}s")
        print(f"  Final success rate: {hard_metrics.success_rate:.0%}")
        print(f"  Final mean reward:  {hard_metrics.mean_reward:.3f}")

        # ==============================================================
        # PHASE 3: Curriculum learning (5x5 -> 8x8 transition)
        # ==============================================================
        print_section("PHASE 3: Curriculum Learning (5x5 -> 8x8)")
        print(f"  Budget:       {config.curriculum_timesteps:,} steps (same as direct 8x8)")
        print(f"  Warmup:       first {config.curriculum_warmup_frac:.0%} on 5x5 only")
        print(f"  Transition:   linear mix from {config.curriculum_warmup_frac:.0%} to {config.curriculum_full_hard_frac:.0%}")
        print(f"  Full 8x8:     from {config.curriculum_full_hard_frac:.0%} onward")
        print()

        t0 = time.time()
        curriculum_agent, curriculum_metrics = train_curriculum(config)
        curriculum_time = time.time() - t0
        print(f"\n  Completed in {curriculum_time:.1f}s")
        print(f"  Final success rate: {curriculum_metrics.success_rate:.0%}")
        print(f"  Final mean reward:  {curriculum_metrics.mean_reward:.3f}")

        # ==============================================================
        # PHASE 4: Evaluate all three agents on both environments
        # ==============================================================
        print_section("PHASE 4: Evaluation")

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

        print("\n  EVALUATION RESULTS:")
        print(f"  {'Agent':<16} {'5x5 Success':>12} {'8x8 Success':>12} {'8x8 Reward':>12}")
        print(f"  {'-'*52}")
        for name in agents:
            r5 = results[name]["5x5"]
            r6 = results[name]["8x8"]
            print(f"  {name:<16} {r5.success_rate:>11.0%} {r6.success_rate:>11.0%} {r6.mean_reward:>11.4f}")

        # Save full metrics to disk
        print_section("Saving Metrics")
        save_metrics(config, easy_metrics, hard_metrics, curriculum_metrics,
                     results, easy_time, hard_time, curriculum_time)

        # ==============================================================
        # PHASE 6: Record agent videos as GIFs
        # ==============================================================
        print_section("PHASE 6: Recording Agent Videos")

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
            status = "success" if success else "no goal reached"
            print(f"  {desc}: {path} ({status})")

    # ==================================================================
    # PHASE 5: Generate visualizations (runs in both modes)
    # ==================================================================
    print_section("PHASE 5: Generating Visualizations")
    generate_plots(config, easy_metrics, hard_metrics, curriculum_metrics, results)

    # ==================================================================
    # Save numerical results summary
    # ==================================================================
    json_results = {
        "experiment": "Deep RL Generalization Challenge -- Exploration Cliff",
        "failure_mode": "Poor generalization: DoorKey-5x5 succeeds, DoorKey-8x8 fails",
        "mitigation": "Curriculum learning: gradual 5x5 -> 8x8 transition",
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

    # ==================================================================
    # Final summary
    # ==================================================================
    print_section("FINAL SUMMARY")

    r = results
    print("  FAILURE MODE: Poor Generalization (Exploration Cliff)")
    print("  -----------------------------------------------------------------")
    print(f"  DoorKey-5x5 PPO: {r['5x5 Agent']['5x5'].success_rate:.0%} success in {config.easy_timesteps:,} steps")
    print(f"  DoorKey-8x8 PPO: {r['Direct 8x8']['8x8'].success_rate:.0%} success in {config.hard_timesteps:,} steps")
    print(f"  Same algorithm. Same hyperparameters. Just 3 extra rows and columns.")
    print()
    print("  MITIGATION: Curriculum Learning")
    print("  -----------------------------------------------------------------")
    print(f"  Curriculum agent on 8x8: {r['Curriculum']['8x8'].success_rate:.0%} success")
    print(f"  Budget: {config.curriculum_timesteps:,} steps (same as direct 8x8)")
    print(f"  Strategy: learn on 5x5 first, gradually transition to 8x8")
    print()
    print("  OUTPUTS:")
    print("  -----------------------------------------------------------------")
    print(f"  {config.results_dir}/experiment_results.json")
    print(f"  {config.results_dir}/metrics_data.json")
    print(f"  {config.results_dir}/01_training_curves.png")
    print(f"  {config.results_dir}/02_exploration_cliff.png")
    print(f"  {config.results_dir}/03_evaluation_comparison.png")
    print(f"  {config.results_dir}/04_curriculum_schedule.png")
    print(f"  {config.results_dir}/05_executive_summary.png")
    print(f"  {config.results_dir}/5x5_agent_on_5x5.gif")
    print(f"  {config.results_dir}/direct_8x8_agent_on_8x8.gif")
    print(f"  {config.results_dir}/curriculum_agent_on_8x8.gif")
    print()


if __name__ == "__main__":
    main()
