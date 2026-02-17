"""Visualization suite for the exploration cliff / curriculum learning experiment."""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from .evaluator import EvalResult
from .trainer import TrainingMetrics


COLORS = {
    "easy":       "#2ecc71",   # green - 5x5 agent
    "hard":       "#e74c3c",   # red - direct 8x8 agent
    "curriculum": "#3498db",   # blue - curriculum agent
    "bg":         "#1a1a2e",
    "panel":      "#16213e",
    "text":       "#ecf0f1",
    "grid":       "#34495e",
    "accent":     "#f39c12",
}


def setup_style():
    plt.rcParams.update({
        "figure.facecolor": COLORS["bg"],
        "axes.facecolor": COLORS["panel"],
        "axes.edgecolor": COLORS["grid"],
        "axes.labelcolor": COLORS["text"],
        "text.color": COLORS["text"],
        "xtick.color": COLORS["text"],
        "ytick.color": COLORS["text"],
        "grid.color": COLORS["grid"],
        "grid.alpha": 0.3,
        "font.size": 12,
        "axes.titlesize": 15,
        "figure.titlesize": 18,
        "legend.facecolor": COLORS["panel"],
        "legend.edgecolor": COLORS["grid"],
        "legend.fontsize": 11,
    })


def smooth(data: list[float], window: int = 20) -> np.ndarray:
    """Exponential moving average smoothing."""
    arr = np.array(data, dtype=float)
    if len(arr) < 2:
        return arr
    alpha = 2.0 / (window + 1)
    result = np.empty_like(arr)
    result[0] = arr[0]
    for i in range(1, len(arr)):
        result[i] = alpha * arr[i] + (1 - alpha) * result[i - 1]
    return result


def plot_training_curves(
    easy_metrics: TrainingMetrics,
    hard_metrics: TrainingMetrics,
    curriculum_metrics: TrainingMetrics,
    save_path: str,
):
    """Plot training curves — each agent gets its own panel for clarity."""
    setup_style()
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Training Curves: Each Agent's Learning Trajectory",
                 fontweight="bold", fontsize=18)

    panels = [
        (easy_metrics, COLORS["easy"], "5x5 Baseline (200K steps)",
         "Learns quickly — 100% success"),
        (hard_metrics, COLORS["hard"], "Direct 8x8 (500K steps)",
         "Never learns — 0% success"),
        (curriculum_metrics, COLORS["curriculum"], "Curriculum 5x5\u21928x8 (500K steps)",
         "Transfers knowledge — 100% success"),
    ]

    for ax, (metrics, color, title, annotation) in zip(axes, panels):
        if metrics.episode_successes:
            data = smooth([float(s) for s in metrics.episode_successes], 50)
            ax.plot(data, color=color, linewidth=2.5, alpha=0.9)
            ax.fill_between(range(len(data)), data, alpha=0.1, color=color)

        ax.set_title(title, fontsize=13, fontweight="bold")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Success Rate")
        ax.set_ylim(-0.05, 1.05)
        ax.grid(True)

        # Add final result annotation
        ax.text(0.5, 0.5, annotation, transform=ax.transAxes,
                ha="center", va="center", fontsize=12, alpha=0.3,
                color=COLORS["text"], fontweight="bold")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_evaluation_comparison(
    results: dict[str, dict[str, EvalResult]],
    save_path: str,
):
    """Bar chart comparing all agents on both environments."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Evaluation: Performance on Unseen Seeds",
                 fontweight="bold", fontsize=16)

    agent_names = list(results.keys())
    agent_colors = [COLORS["easy"], COLORS["hard"], COLORS["curriculum"]]

    for ax_idx, (env_label, env_key) in enumerate([("DoorKey-5x5", "5x5"), ("DoorKey-8x8", "8x8")]):
        ax = axes[ax_idx]
        vals = [results[name][env_key].success_rate for name in agent_names]
        x = np.arange(len(agent_names))
        bars = ax.bar(x, vals, color=agent_colors[:len(agent_names)],
                      alpha=0.85, edgecolor="white", linewidth=0.5, width=0.6)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                    f"{v:.0%}", ha="center", fontweight="bold", fontsize=14,
                    color=COLORS["text"])

        ax.set_title(f"Success Rate on {env_label}", fontsize=14)
        ax.set_xticks(x)
        ax.set_xticklabels(agent_names, fontsize=11)
        ax.set_ylim(0, 1.2)
        ax.set_ylabel("Success Rate")
        ax.grid(True, axis="y")

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_exploration_cliff(
    easy_metrics: TrainingMetrics,
    hard_metrics: TrainingMetrics,
    save_path: str,
):
    """Visualize the exploration cliff: 5x5 and 8x8 side by side."""
    setup_style()
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle("The Exploration Cliff: Same Algorithm, Bigger Grid = Total Failure",
                 fontweight="bold", fontsize=16)

    # Left panel: 5x5 training
    ax = axes[0]
    if easy_metrics.episode_successes:
        data = smooth([float(s) for s in easy_metrics.episode_successes], 50)
        ax.plot(data, color=COLORS["easy"], linewidth=2.5)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=COLORS["easy"])
    ax.set_title("DoorKey-5x5: Succeeds", fontsize=14, fontweight="bold",
                 color=COLORS["easy"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.text(0.95, 0.05, "100%", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=36, fontweight="bold", alpha=0.3, color=COLORS["easy"])

    # Right panel: 8x8 training
    ax = axes[1]
    if hard_metrics.episode_successes:
        data = smooth([float(s) for s in hard_metrics.episode_successes], 50)
        ax.plot(data, color=COLORS["hard"], linewidth=2.5)
        ax.fill_between(range(len(data)), data, alpha=0.15, color=COLORS["hard"])
    ax.set_title("DoorKey-8x8: Fails Completely", fontsize=14, fontweight="bold",
                 color=COLORS["hard"])
    ax.set_xlabel("Episode")
    ax.set_ylabel("Success Rate")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True)
    ax.text(0.95, 0.05, "0%", transform=ax.transAxes, ha="right", va="bottom",
            fontsize=36, fontweight="bold", alpha=0.3, color=COLORS["hard"])

    # Shared annotation
    fig.text(0.5, 0.01, "Same PPO algorithm  |  Same hyperparameters  |  Same training budget  |  Just 3 extra rows and columns",
             ha="center", fontsize=12, alpha=0.7, color=COLORS["accent"], fontweight="bold")

    plt.tight_layout(rect=[0, 0.05, 1, 1])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_curriculum_schedule(
    curriculum_metrics: TrainingMetrics,
    config_warmup_frac: float,
    config_full_hard_frac: float,
    total_timesteps: int,
    save_path: str,
):
    """Visualize the curriculum schedule overlaid with training success."""
    setup_style()
    fig, ax1 = plt.subplots(figsize=(14, 6))

    # Plot success rate on primary y-axis
    if curriculum_metrics.episode_successes:
        data = smooth([float(s) for s in curriculum_metrics.episode_successes], 50)
        ax1.plot(data, color=COLORS["curriculum"], linewidth=2.5, label="Success Rate")
        ax1.fill_between(range(len(data)), data, alpha=0.1, color=COLORS["curriculum"])
    ax1.set_xlabel("Episode", fontsize=13)
    ax1.set_ylabel("Success Rate", color=COLORS["curriculum"], fontsize=13)
    ax1.set_ylim(-0.05, 1.05)
    ax1.tick_params(axis="y", labelcolor=COLORS["curriculum"])

    # Plot curriculum schedule on secondary y-axis
    ax2 = ax1.twinx()
    n_eps = len(curriculum_metrics.episode_successes) if curriculum_metrics.episode_successes else 100
    avg_ep_len = total_timesteps / max(n_eps, 1)
    schedule_x = np.arange(n_eps)
    schedule_y = []
    for ep in schedule_x:
        step = ep * avg_ep_len
        if step < total_timesteps * config_warmup_frac:
            schedule_y.append(0.0)
        elif step >= total_timesteps * config_full_hard_frac:
            schedule_y.append(1.0)
        else:
            frac = (step - total_timesteps * config_warmup_frac) / \
                   (total_timesteps * (config_full_hard_frac - config_warmup_frac))
            schedule_y.append(frac)
    ax2.plot(schedule_x, schedule_y, color=COLORS["accent"], linewidth=2.5,
             linestyle="--", alpha=0.8, label="P(8x8 environment)")
    ax2.set_ylabel("P(8x8 env)", color=COLORS["accent"], fontsize=13)
    ax2.set_ylim(-0.05, 1.05)
    ax2.tick_params(axis="y", labelcolor=COLORS["accent"])

    # Phase annotations
    warmup_ep = int(n_eps * config_warmup_frac)
    full_hard_ep = int(n_eps * config_full_hard_frac)
    ax1.axvline(warmup_ep, color=COLORS["text"], alpha=0.3, linestyle=":")
    ax1.axvline(full_hard_ep, color=COLORS["text"], alpha=0.3, linestyle=":")
    ax1.text(warmup_ep / 2, 1.0, "Warmup\n(5x5 only)", ha="center", va="top",
             fontsize=10, alpha=0.5, color=COLORS["text"])
    ax1.text((warmup_ep + full_hard_ep) / 2, 1.0, "Transition\n(mix 5x5 + 8x8)", ha="center", va="top",
             fontsize=10, alpha=0.5, color=COLORS["text"])
    ax1.text((full_hard_ep + n_eps) / 2, 1.0, "Full 8x8", ha="center", va="top",
             fontsize=10, alpha=0.5, color=COLORS["text"])

    # Combine legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc="center right", fontsize=12)

    ax1.set_title("Curriculum Learning: Gradual 5x5 \u2192 8x8 Transition",
                  fontweight="bold", fontsize=16)
    ax1.grid(True)

    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_summary_dashboard(
    results: dict[str, dict[str, EvalResult]],
    easy_metrics: TrainingMetrics,
    hard_metrics: TrainingMetrics,
    curriculum_metrics: TrainingMetrics,
    save_path: str,
):
    """Single-page executive summary with clear, separated panels."""
    setup_style()
    fig = plt.figure(figsize=(18, 12))
    fig.suptitle("DEEP RL GENERALIZATION CHALLENGE \u2014 EXECUTIVE SUMMARY",
                 fontweight="bold", fontsize=20, y=0.98)

    gs = gridspec.GridSpec(2, 4, hspace=0.4, wspace=0.35)

    agent_names = list(results.keys())
    bar_colors = [COLORS["easy"], COLORS["hard"], COLORS["curriculum"]]

    # --- Top left: 5x5 training curve ---
    ax = fig.add_subplot(gs[0, 0])
    if easy_metrics.episode_successes:
        data = smooth([float(s) for s in easy_metrics.episode_successes], 50)
        ax.plot(data, color=COLORS["easy"], linewidth=2)
        ax.fill_between(range(len(data)), data, alpha=0.1, color=COLORS["easy"])
    ax.set_title("5x5 Training", fontweight="bold", color=COLORS["easy"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Episode")
    ax.grid(True)

    # --- Top center-left: 8x8 training curve ---
    ax = fig.add_subplot(gs[0, 1])
    if hard_metrics.episode_successes:
        data = smooth([float(s) for s in hard_metrics.episode_successes], 50)
        ax.plot(data, color=COLORS["hard"], linewidth=2)
        ax.fill_between(range(len(data)), data, alpha=0.1, color=COLORS["hard"])
    ax.set_title("Direct 8x8 Training", fontweight="bold", color=COLORS["hard"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Episode")
    ax.grid(True)

    # --- Top center-right: curriculum training curve ---
    ax = fig.add_subplot(gs[0, 2])
    if curriculum_metrics.episode_successes:
        data = smooth([float(s) for s in curriculum_metrics.episode_successes], 50)
        ax.plot(data, color=COLORS["curriculum"], linewidth=2)
        ax.fill_between(range(len(data)), data, alpha=0.1, color=COLORS["curriculum"])
    ax.set_title("Curriculum Training", fontweight="bold", color=COLORS["curriculum"])
    ax.set_ylim(-0.05, 1.05)
    ax.set_ylabel("Success Rate")
    ax.set_xlabel("Episode")
    ax.grid(True)

    # --- Top right: 8x8 eval bar chart ---
    ax = fig.add_subplot(gs[0, 3])
    vals = [results[n]["8x8"].success_rate for n in agent_names]
    bars = ax.bar(range(len(agent_names)), vals, color=bar_colors[:len(agent_names)],
                  alpha=0.85, edgecolor="white", width=0.6)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.03,
                f"{v:.0%}", ha="center", fontweight="bold", fontsize=11,
                color=COLORS["text"])
    ax.set_xticks(range(len(agent_names)))
    ax.set_xticklabels(agent_names, fontsize=8, rotation=15)
    ax.set_ylim(0, 1.2)
    ax.set_title("8x8 Eval (Key Result)", fontweight="bold")
    ax.grid(True, axis="y")

    # --- Bottom: Full-width text summary ---
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")

    r = results
    text = (
        "FAILURE MODE: Poor Generalization (Exploration Cliff)\n"
        "\n"
        f"  DoorKey-5x5:  Agent learns key\u2192door\u2192goal in ~60K steps  \u2192  {r['5x5 Agent']['5x5'].success_rate:.0%} success\n"
        f"  DoorKey-8x8:  Agent gets 0% even after 500K steps       \u2192  {r['Direct 8x8']['8x8'].success_rate:.0%} success\n"
        "  Same PPO algorithm. Same hyperparameters. Just 3 extra rows and columns.\n"
        "\n"
        "MITIGATION: Curriculum Learning (5x5 \u2192 8x8 gradual transition)\n"
        "\n"
        f"  Curriculum agent on 5x5:  {r['Curriculum']['5x5'].success_rate:.0%} success\n"
        f"  Curriculum agent on 8x8:  {r['Curriculum']['8x8'].success_rate:.0%} success\n"
        f"  Budget: 500K steps (same as direct 8x8 \u2014 fair comparison)\n"
        "\n"
        "  Strategy: Train on 5x5 first (where exploration is easy), then gradually\n"
        "  transition to 8x8. The agent transfers its learned key\u2192door\u2192goal strategy\n"
        "  instead of rediscovering it from scratch in the larger grid."
    )

    ax.text(0.05, 0.95, text, transform=ax.transAxes, fontsize=13,
            va="top", ha="left", fontfamily="monospace",
            bbox=dict(boxstyle="round,pad=0.8", facecolor=COLORS["panel"],
                      edgecolor=COLORS["accent"], alpha=0.9, linewidth=2))

    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
