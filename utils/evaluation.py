"""Evaluation utilities for greedy tabular policies.

The functions in this module run stochastic environment rollouts and compute
the exact metrics required by the project specification. A policy is provided
as a callable that receives ``(state, valid_actions)`` and returns one action
ID. This keeps evaluation independent of whether the policy came from Value
Iteration, SARSA, Q-learning, a random baseline, or a heuristic.
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np

from davethediver_rl.env.config import EVAL_EPISODES
from davethediver_rl.env.underwater_env import State

PolicyFn = Callable[[State, Sequence[int]], int]


def rollout_episode(
    env,
    policy_fn: PolicyFn,
    render: bool = False,
) -> Dict[str, Any]:
    """Run one episode and return detailed trajectory data.

    The returned dictionary includes scalar episode outcomes and per-step
    records that plotting functions can use for path, catch-event, oxygen, and
    time visualizations.
    """

    state = env.reset()
    done = False
    total_reward = 0.0
    steps = 0
    fish_value = 0.0
    caught_fish_ids: List[int] = []
    terminal_reason: Optional[str] = None
    trajectory: List[Dict[str, Any]] = []
    oxygen_time: List[Dict[str, int]] = [
        {"step": 0, "oxygen": state[2], "global_time": state[3]}
    ]
    renders: List[str] = []

    if render:
        renders.append(env.render_ascii(state))

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
            caught_fish_ids.append(int(caught_fish_id))
            fish_value += float(env.fish_by_id[caught_fish_id]["value"])

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

    death = terminal_reason in {"death", "failed_surface"}
    all_fish_caught = terminal_reason == "all_fish_caught"
    final_state = state

    return {
        "total_reward": total_reward,
        "steps": steps,
        "terminal_reason": terminal_reason,
        "fish_value": fish_value,
        "num_fish_caught": len(caught_fish_ids),
        "caught_fish_ids": caught_fish_ids,
        "final_state": final_state,
        "remaining_oxygen": final_state[2],
        "remaining_time": final_state[3],
        "survived": int(not death),
        "death": int(death),
        "all_fish_caught": int(all_fish_caught),
        "trajectory": trajectory,
        "oxygen_time": oxygen_time,
        "renders": renders,
    }


def evaluate_policy(
    env,
    policy_fn: PolicyFn,
    episodes: int = EVAL_EPISODES,
    method: str = "policy",
) -> Dict[str, Any]:
    """Evaluate a policy and return one DataFrame-ready summary row."""

    rollouts = [rollout_episode(env, policy_fn) for _ in range(episodes)]

    rewards = np.array([episode["total_reward"] for episode in rollouts], dtype=float)
    survived = np.array([episode["survived"] for episode in rollouts], dtype=float)
    deaths = np.array([episode["death"] for episode in rollouts], dtype=float)
    steps = np.array([episode["steps"] for episode in rollouts], dtype=float)
    fish_values = np.array([episode["fish_value"] for episode in rollouts], dtype=float)
    fish_counts = np.array([episode["num_fish_caught"] for episode in rollouts], dtype=float)
    remaining_time = np.array([episode["remaining_time"] for episode in rollouts], dtype=float)
    all_fish_caught = np.array(
        [episode["all_fish_caught"] for episode in rollouts], dtype=float
    )

    survived_oxygen = [
        episode["remaining_oxygen"] for episode in rollouts if episode["survived"]
    ]
    avg_remaining_oxygen = (
        float(np.mean(survived_oxygen)) if survived_oxygen else 0.0
    )

    return {
        "method": method,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "survival_rate": float(np.mean(survived)),
        "death_rate": float(np.mean(deaths)),
        "avg_steps": float(np.mean(steps)),
        "avg_fish_value": float(np.mean(fish_values)),
        "avg_num_fish_caught": float(np.mean(fish_counts)),
        "avg_remaining_oxygen": avg_remaining_oxygen,
        "avg_remaining_time": float(np.mean(remaining_time)),
        "all_fish_caught_rate": float(np.mean(all_fish_caught)),
    }


__all__ = ["evaluate_policy", "rollout_episode"]
