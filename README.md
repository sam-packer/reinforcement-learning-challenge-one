# RL Generalization Challenge: The exploration cliff

PPO from scratch, trained on MiniGrid's DoorKey environment. The point of this project is to show how a tiny change in grid size can completely break an RL agent, and how curriculum learning fixes it.

## What's going on

DoorKey is simple: find a key, pick it up, unlock a door, reach the goal. PPO solves the 5x5 grid in about 60K steps with 100% success. On 8x8, the same algorithm with the same hyperparameters gets 0% after 500K steps.

The problem is sparse rewards. On a 5x5 grid, random exploration is enough to stumble into the key-door-goal sequence occasionally, so the agent gets a learning signal. On 8x8, the state space is large enough that random exploration almost never finds the full sequence, so the agent never gets rewarded and never learns anything.

## How the fix works

Train on 5x5 first (where exploration is easy), then gradually mix in 8x8 episodes until the agent is training entirely on the harder grid. The agent carries over what it learned instead of starting blind.

The experiment trains three agents and compares them:

1. PPO on 5x5 (200K steps) - works fine
2. PPO on 8x8 (500K steps) - fails completely
3. Curriculum: 5x5 to 8x8 (500K steps) - works, with the same budget as #2

## Project structure

```
main.py                  # runs the full experiment
rl_challenge/
  config.py              # hyperparameters and experiment settings
  networks.py            # CNN actor-critic network
  ppo.py                 # PPO with GAE, clipped surrogate objective
  trainer.py             # training loops (standard + curriculum)
  evaluator.py           # multi-seed evaluation
  envs.py                # MiniGrid wrappers (normalization, CHW transpose)
  visualize.py           # matplotlib plots
  video_recorder.py      # GIF recording of agent behavior
results/                 # output plots, metrics, and GIFs
```

## Requirements

- Python 3.13
- [uv](https://docs.astral.sh/uv/)
- freetype (`brew install freetype` on macOS, needed by pygame)

## Setup

```bash
uv sync
```

## Usage

Run the full experiment (trains all three agents, evaluates them, generates plots):

```bash
uv run python main.py
```

If you've already trained and just want to regenerate the plots:

```bash
uv run python main.py --plots-only
```

## Output

Everything goes into `results/`:

- `experiment_results.json` - numerical results
- `metrics_data.json` - full training metrics (used by `--plots-only`)
- `01_training_curves.png` - reward and success rate over time
- `02_exploration_cliff.png` - 5x5 vs 8x8 side-by-side
- `03_evaluation_comparison.png` - all three agents on both environments
- `04_curriculum_schedule.png` - the 5x5 to 8x8 transition schedule
- `05_executive_summary.png` - combined dashboard
- `*.gif` - recordings of each agent's behavior

## Implementation notes

No Stable Baselines, no RLlib. PPO is implemented from scratch with clipped surrogate objective, GAE (lambda=0.95), and an entropy bonus. The network is a 3-layer CNN (3 to 16 to 32 to 64 channels) feeding into a shared 256-dim layer that splits into actor and critic heads.

The curriculum schedule spends the first 30% of training purely on 5x5, linearly mixes in 8x8 from 30-80%, and trains on 8x8 only from 80% onward. Evaluation uses 50 held-out seeds with 3 episodes each, separate from training.
