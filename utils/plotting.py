"""Matplotlib plotting utilities for training and evaluation artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np

from env.config import (
    ACTION_CATCH,
    ACTION_DOWN,
    ACTION_LEFT,
    ACTION_RIGHT,
    ACTION_SURFACE,
    ACTION_UP,
    FISH_CONFIG,
    GRID_DEPTH,
    GRID_WIDTH,
    INITIAL_FISH_MASK,
    MAX_GLOBAL_TIME,
    MAX_OXYGEN,
    MOVING_AVERAGE_WINDOW,
    OBSTACLES,
    START_POS,
)
from utils.io_utils import ensure_dir


ACTION_SYMBOLS = {
    ACTION_UP: "^",
    ACTION_DOWN: "v",
    ACTION_LEFT: "<",
    ACTION_RIGHT: ">",
    ACTION_CATCH: "C",
    ACTION_SURFACE: "S",
}


def moving_average(values: Sequence[float], window: int = MOVING_AVERAGE_WINDOW) -> np.ndarray:
    """Return a trailing moving average with a stable length."""

    values = np.asarray(values, dtype=float)
    if values.size == 0:
        return values

    result = np.empty_like(values)
    for idx in range(values.size):
        start = max(0, idx - window + 1)
        result[idx] = np.mean(values[start : idx + 1])
    return result


def plot_training_reward_curve(
    training_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    title: str = "Training Reward Curve",
    window: int = MOVING_AVERAGE_WINDOW,
) -> None:
    """Save a moving-average episode reward curve."""

    rows = list(training_log)
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]

    _start_figure()
    plt.plot(episodes, moving_average(rewards, window), linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    _save_figure(output_path)


def plot_reward_curve_comparison(
    q_learning_log: Sequence[Mapping[str, Any]],
    sarsa_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    window: int = MOVING_AVERAGE_WINDOW,
) -> None:
    """Save a reward-curve comparison for Q-learning and SARSA."""

    _start_figure()
    _plot_log_metric(q_learning_log, "total_reward", "Q-learning", window)
    _plot_log_metric(sarsa_log, "total_reward", "SARSA", window)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Reward Curve Comparison")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save_figure(output_path)


def plot_survival_rate_curve(
    q_learning_log: Sequence[Mapping[str, Any]],
    sarsa_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    window: int = MOVING_AVERAGE_WINDOW,
) -> None:
    """Save a moving-average survival-rate curve."""

    _start_figure()
    _plot_log_metric(q_learning_log, "survived", "Q-learning", window)
    _plot_log_metric(sarsa_log, "survived", "SARSA", window)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Survival Rate")
    plt.ylim(-0.05, 1.05)
    plt.title("Survival Rate Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save_figure(output_path)


def plot_death_rate_curve(
    q_learning_log: Sequence[Mapping[str, Any]],
    sarsa_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    window: int = MOVING_AVERAGE_WINDOW,
) -> None:
    """Save a moving-average death-rate curve."""

    _start_figure()
    _plot_log_metric(q_learning_log, "death", "Q-learning", window)
    _plot_log_metric(sarsa_log, "death", "SARSA", window)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Death Rate")
    plt.ylim(-0.05, 1.05)
    plt.title("Death Rate Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save_figure(output_path)


def plot_policy_map(
    env,
    policy: Mapping[Any, int],
    output_path: str | Path,
    title: str = "Policy Map",
) -> None:
    """Save a representative grid map of greedy actions."""

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_DEPTH - 0.5, -0.5)
    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_DEPTH))
    ax.set_xlabel("x")
    ax.set_ylabel("depth d")
    ax.set_title(title)
    ax.grid(True, color="0.85")

    fish_positions = {fish["position"]: fish for fish in FISH_CONFIG}

    for d in range(GRID_DEPTH):
        for x in range(GRID_WIDTH):
            pos = (x, d)
            state = (x, d, MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK)

            facecolor = "white"
            text = ""
            color = "black"

            if pos in OBSTACLES:
                facecolor = "0.25"
                text = "#"
                color = "white"
            elif pos == START_POS:
                facecolor = "#d9f2d9"
                text = "B"
            elif pos in fish_positions:
                facecolor = "#fce8c8"
                text = f"F{fish_positions[pos]['id']}"
            else:
                action = policy.get(state)
                text = ACTION_SYMBOLS.get(action, ".")

            rect = plt.Rectangle((x - 0.5, d - 0.5), 1, 1, facecolor=facecolor, edgecolor="0.85")
            ax.add_patch(rect)
            ax.text(x, d, text, ha="center", va="center", color=color, fontsize=10)

    _save_figure(output_path)


def plot_trajectory(
    env,
    rollout: Mapping[str, Any],
    output_path: str | Path,
    title: str = "Evaluation Trajectory",
) -> None:
    """Save one rollout trajectory over the grid."""

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_DEPTH - 0.5, -0.5)
    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_DEPTH))
    ax.set_xlabel("x")
    ax.set_ylabel("depth d")
    ax.set_title(title)
    ax.grid(True, color="0.85")

    for x, d in OBSTACLES:
        ax.add_patch(plt.Rectangle((x - 0.5, d - 0.5), 1, 1, facecolor="0.25"))

    for fish in FISH_CONFIG:
        x, d = fish["position"]
        ax.scatter(x, d, marker="s", s=120, color="#f0a202", edgecolor="black", zorder=3)
        ax.text(x, d, f"F{fish['id']}", ha="center", va="center", fontsize=8, zorder=4)

    states = [rollout["trajectory"][0]["state"]] if rollout["trajectory"] else [START_POS + (MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK)]
    states.extend(step["next_state"] for step in rollout["trajectory"])
    xs = [state[0] for state in states]
    ds = [state[1] for state in states]

    ax.plot(xs, ds, color="#1f77b4", linewidth=2, marker="o", markersize=3, zorder=2)
    ax.scatter([START_POS[0]], [START_POS[1]], marker="*", s=180, color="#2ca02c", edgecolor="black", zorder=5)

    catch_points = [
        step["next_state"] for step in rollout["trajectory"] if step.get("caught_fish_id") is not None
    ]
    if catch_points:
        ax.scatter(
            [state[0] for state in catch_points],
            [state[1] for state in catch_points],
            marker="X",
            s=130,
            color="#d62728",
            zorder=6,
            label="Catch",
        )
        ax.legend(loc="lower right")

    _save_figure(output_path)


def plot_oxygen_time_curve(
    rollout: Mapping[str, Any],
    output_path: str | Path,
    title: str = "Oxygen and Time Curve",
) -> None:
    """Save oxygen and global time over one evaluation rollout."""

    rows = list(rollout.get("oxygen_time", []))
    if not rows:
        return

    steps = [int(row["step"]) for row in rows]
    oxygen = [int(row["oxygen"]) for row in rows]
    global_time = [int(row["global_time"]) for row in rows]

    _start_figure()
    plt.plot(steps, oxygen, label="Oxygen", linewidth=2)
    plt.plot(steps, global_time, label="Global Time", linewidth=2)
    plt.xlabel("Step")
    plt.ylabel("Remaining Units")
    plt.title(title)
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save_figure(output_path)


def _plot_log_metric(
    training_log: Sequence[Mapping[str, Any]],
    metric: str,
    label: str,
    window: int,
) -> None:
    """Plot one metric from a training log if rows are available."""

    rows = list(training_log)
    if not rows:
        return

    episodes = [int(row["episode"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.plot(episodes, moving_average(values, window), label=label, linewidth=2)


def _start_figure() -> None:
    """Create a standard figure."""

    plt.figure(figsize=(9, 5))


def _save_figure(output_path: str | Path) -> None:
    """Write the active Matplotlib figure to disk."""

    output_path = Path(output_path)
    ensure_dir(output_path)
    plt.tight_layout()
    plt.savefig(output_path, dpi=160)
    plt.close()


__all__ = [
    "moving_average",
    "plot_training_reward_curve",
    "plot_reward_curve_comparison",
    "plot_survival_rate_curve",
    "plot_death_rate_curve",
    "plot_policy_map",
    "plot_trajectory",
    "plot_oxygen_time_curve",
]
