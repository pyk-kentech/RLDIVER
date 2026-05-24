"""Evaluation utilities for tabular diver policies."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Sequence

import numpy as np

from config import EVAL_EPISODES
from env import ReducedUnderwaterFishingEnv, State

PolicyFn = Callable[[State, Sequence[int]], int]


def rollout_episode(
    policy_fn: PolicyFn,
    seed: int = 42,
    render: bool = False,
) -> Dict[str, Any]:
    """Run one episode and return detailed trajectory data for plotting."""

    env = ReducedUnderwaterFishingEnv(stochastic=True, seed=seed)
    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    fish_value = 0.0
    fish_count = 0
    terminal_reason = None
    trajectory: List[Dict[str, Any]] = []
    oxygen_time = [{"step": 0, "oxygen": state[2], "global_time": state[3]}]
    renders = [env.render_ascii(state)] if render else []

    while not done:
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            terminal_reason = "no_valid_actions"
            break

        action = policy_fn(state, valid_actions)
        if action not in valid_actions:
            action = valid_actions[0]

        next_state, reward, done, info = env.step(action)
        steps += 1
        total_reward += reward

        caught_fish_id = info.get("caught_fish_id")
        if caught_fish_id is not None:
            fish_value += env.fish_by_id[caught_fish_id]["value"]
            fish_count += 1

        trajectory.append(
            {
                "step": steps,
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "event": info.get("event"),
                "caught_fish_id": caught_fish_id,
                "attack": bool(info.get("attack", False)),
            }
        )
        oxygen_time.append(
            {
                "step": steps,
                "oxygen": next_state[2],
                "global_time": next_state[3],
            }
        )

        terminal_reason = info.get("terminal_reason")
        state = next_state
        if render:
            renders.append(env.render_ascii(state))

    all_fish_caught = fish_count == len(env.fish_by_id) and state[0:2] == (0, 0)
    death = terminal_reason in {"death", "failed_surface", "timeout_underwater"}
    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminal_reason": terminal_reason,
        "fish_value": fish_value,
        "num_fish_caught": fish_count,
        "final_state": state,
        "remaining_oxygen": state[2],
        "remaining_time": state[3],
        "survived": int(not death),
        "death": int(death),
        "all_fish_caught": int(all_fish_caught),
        "trajectory": trajectory,
        "oxygen_time": oxygen_time,
        "renders": renders,
    }


def evaluate_policy(
    method: str,
    policy_fn: PolicyFn,
    episodes: int = EVAL_EPISODES,
    seed: int = 42,
) -> Dict[str, Any]:
    """Evaluate a policy and return Section 21.3 metrics as one row."""

    rollouts = [
        rollout_episode(policy_fn, seed=seed + episode)
        for episode in range(episodes)
    ]

    rewards = np.array([item["total_reward"] for item in rollouts], dtype=float)
    survived = np.array([item["survived"] for item in rollouts], dtype=float)
    deaths = np.array([item["death"] for item in rollouts], dtype=float)
    steps = np.array([item["steps"] for item in rollouts], dtype=float)
    fish_values = np.array([item["fish_value"] for item in rollouts], dtype=float)
    fish_counts = np.array([item["num_fish_caught"] for item in rollouts], dtype=float)
    oxygen = np.array([item["remaining_oxygen"] for item in rollouts], dtype=float)
    time = np.array([item["remaining_time"] for item in rollouts], dtype=float)
    all_fish = np.array([item["all_fish_caught"] for item in rollouts], dtype=float)

    return {
        "method": method,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "survival_rate": float(np.mean(survived)),
        "death_rate": float(np.mean(deaths)),
        "avg_steps": float(np.mean(steps)),
        "avg_fish_value": float(np.mean(fish_values)),
        "avg_num_fish_caught": float(np.mean(fish_counts)),
        "avg_remaining_oxygen": float(np.mean(oxygen)),
        "avg_remaining_time": float(np.mean(time)),
        "all_fish_caught_rate": float(np.mean(all_fish)),
    }
