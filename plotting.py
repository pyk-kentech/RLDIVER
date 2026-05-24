"""Matplotlib plotting utilities for diver experiment results."""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np

from config import (
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
    OBSTACLES,
    START_POS,
)

ACTION_SYMBOLS = {
    ACTION_UP: "^",
    ACTION_DOWN: "v",
    ACTION_LEFT: "<",
    ACTION_RIGHT: ">",
    ACTION_CATCH: "C",
    ACTION_SURFACE: "S",
}


def read_csv_rows(path: str | Path) -> List[Dict[str, Any]]:
    """Read a CSV file into dictionaries."""

    path = Path(path)
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def moving_average(values: Sequence[float], window: int = 100) -> np.ndarray:
    """Return a trailing moving average."""

    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return array
    output = np.empty_like(array)
    for index in range(array.size):
        start = max(0, index - window + 1)
        output[index] = np.mean(array[start : index + 1])
    return output


def plot_training_reward_curve(
    training_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    title: str,
    window: int = 100,
) -> None:
    """Plot one training reward curve."""

    rows = list(training_log)
    if not rows:
        return
    episodes = [int(row["episode"]) for row in rows]
    rewards = [float(row["total_reward"]) for row in rows]
    _new_figure()
    plt.plot(episodes, moving_average(rewards, window), linewidth=2)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title(title)
    plt.grid(True, alpha=0.25)
    _save(output_path)


def plot_reward_curve_comparison(
    sarsa_log: Sequence[Mapping[str, Any]],
    q_learning_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    window: int = 100,
) -> None:
    """Plot SARSA and Q-learning reward curves together."""

    _new_figure()
    _plot_metric(sarsa_log, "total_reward", "SARSA", window)
    _plot_metric(q_learning_log, "total_reward", "Q-learning", window)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Reward")
    plt.title("Reward Curve Comparison")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save(output_path)


def plot_survival_rate_curve(
    sarsa_log: Sequence[Mapping[str, Any]],
    q_learning_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    window: int = 100,
) -> None:
    """Plot moving-average survival rates."""

    _new_figure()
    _plot_metric(sarsa_log, "survived", "SARSA", window)
    _plot_metric(q_learning_log, "survived", "Q-learning", window)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Survival Rate")
    plt.ylim(-0.05, 1.05)
    plt.title("Survival Rate Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save(output_path)


def plot_death_rate_curve(
    sarsa_log: Sequence[Mapping[str, Any]],
    q_learning_log: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    window: int = 100,
) -> None:
    """Plot moving-average death rates."""

    _new_figure()
    _plot_metric(sarsa_log, "death", "SARSA", window)
    _plot_metric(q_learning_log, "death", "Q-learning", window)
    plt.xlabel("Episode")
    plt.ylabel("Moving Average Death Rate")
    plt.ylim(-0.05, 1.05)
    plt.title("Death Rate Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save(output_path)


def plot_policy_map(policy: Mapping[Any, int], output_path: str | Path, title: str) -> None:
    """Plot representative greedy actions for a full-resource state."""

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_DEPTH - 0.5, -0.5)
    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_DEPTH))
    ax.set_xlabel("x")
    ax.set_ylabel("depth")
    ax.set_title(title)
    ax.grid(True, color="0.85")

    fish_by_position = {fish["position"]: fish for fish in FISH_CONFIG}

    for d in range(GRID_DEPTH):
        for x in range(GRID_WIDTH):
            position = (x, d)
            state = (x, d, MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK)
            face = "white"
            text = ACTION_SYMBOLS.get(policy.get(state), ".")
            color = "black"

            if position in OBSTACLES:
                face = "0.25"
                text = "#"
                color = "white"
            elif position == START_POS:
                face = "#d9f2d9"
                text = "D"
            elif position in fish_by_position:
                face = "#ffe6b3"
                text = f"F{fish_by_position[position]['id']}"

            ax.add_patch(
                plt.Rectangle((x - 0.5, d - 0.5), 1, 1, facecolor=face, edgecolor="0.85")
            )
            ax.text(x, d, text, ha="center", va="center", color=color, fontsize=10)

    _save(output_path)


def plot_trajectory(rollout: Mapping[str, Any], output_path: str | Path, title: str) -> None:
    """Plot one evaluation trajectory."""

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.set_xlim(-0.5, GRID_WIDTH - 0.5)
    ax.set_ylim(GRID_DEPTH - 0.5, -0.5)
    ax.set_xticks(range(GRID_WIDTH))
    ax.set_yticks(range(GRID_DEPTH))
    ax.set_xlabel("x")
    ax.set_ylabel("depth")
    ax.set_title(title)
    ax.grid(True, color="0.85")

    for x, d in OBSTACLES:
        ax.add_patch(plt.Rectangle((x - 0.5, d - 0.5), 1, 1, facecolor="0.25"))

    for fish in FISH_CONFIG:
        x, d = fish["position"]
        ax.scatter(x, d, marker="s", s=120, color="#f0a202", edgecolor="black", zorder=3)
        ax.text(
            x + 0.12,
            d - 0.12,
            f"F{fish['id']}",
            ha="left",
            va="bottom",
            fontsize=8,
            zorder=5,
        )

    trajectory = list(rollout.get("trajectory", []))
    if trajectory:
        states = [trajectory[0]["state"]] + [step["next_state"] for step in trajectory]
    else:
        states = [(START_POS[0], START_POS[1], MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK)]

    xs = [state[0] for state in states]
    ds = [state[1] for state in states]
    ax.plot(xs, ds, marker="o", markersize=3, linewidth=2, color="#1f77b4")
    ax.scatter([START_POS[0]], [START_POS[1]], marker="*", s=180, color="#2ca02c", edgecolor="black")

    catches = [step["next_state"] for step in trajectory if step.get("caught_fish_id") is not None]
    if catches:
        ax.scatter(
            [state[0] for state in catches],
            [state[1] for state in catches],
            marker="X",
            s=130,
            color="#d62728",
            label="Catch",
            zorder=4,
        )
        ax.legend()

    _save(output_path)


def plot_oxygen_time_curve(rollout: Mapping[str, Any], output_path: str | Path) -> None:
    """Plot oxygen and global time from one rollout."""

    rows = list(rollout.get("oxygen_time", []))
    if not rows:
        return
    _new_figure()
    plt.plot([row["step"] for row in rows], [row["oxygen"] for row in rows], label="Oxygen")
    plt.plot([row["step"] for row in rows], [row["global_time"] for row in rows], label="Global Time")
    plt.xlabel("Step")
    plt.ylabel("Remaining Units")
    plt.title("Oxygen and Time Curve")
    plt.legend()
    plt.grid(True, alpha=0.25)
    _save(output_path)


def _plot_metric(rows: Sequence[Mapping[str, Any]], metric: str, label: str, window: int) -> None:
    if not rows:
        return
    episodes = [int(row["episode"]) for row in rows]
    values = [float(row[metric]) for row in rows]
    plt.plot(episodes, moving_average(values, window), label=label, linewidth=2)


def _new_figure() -> None:
    plt.figure(figsize=(9, 5))


def _save(output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=160)
    plt.close()
