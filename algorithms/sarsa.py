"""Tabular SARSA for the 8x8 diver MDP."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, List, Tuple

from tqdm import trange

from env import ReducedUnderwaterFishingEnv
from .q_learning import epsilon_greedy_action, new_q_values


def train_sarsa(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    seed: int,
    show_progress: bool,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Train a sparse tabular SARSA agent."""

    env = ReducedUnderwaterFishingEnv(stochastic=True, seed=seed)
    Q = defaultdict(new_q_values)
    log: List[Dict[str, Any]] = []

    iterator = trange(1, episodes + 1, desc="SARSA", disable=not show_progress)
    for episode in iterator:
        state = env.reset()
        action = epsilon_greedy_action(Q, state, env.get_valid_actions(state), epsilon)
        done = False
        total_reward = 0.0
        steps = 0
        fish_value = 0.0
        terminal_reason = None

        while not done:
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            caught_fish_id = info.get("caught_fish_id")
            if caught_fish_id is not None:
                fish_value += env.fish_by_id[caught_fish_id]["value"]

            if done:
                target = reward
                next_action = None
            else:
                next_action = epsilon_greedy_action(
                    Q,
                    next_state,
                    env.get_valid_actions(next_state),
                    epsilon,
                )
                target = reward + gamma * Q[next_state][next_action]

            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
            if next_action is not None:
                action = next_action
            terminal_reason = info.get("terminal_reason")

        death = terminal_reason in {"death", "failed_surface", "timeout_underwater"}
        log.append(
            {
                "episode": episode,
                "total_reward": total_reward,
                "steps": steps,
                "epsilon": epsilon,
                "terminal_reason": terminal_reason,
                "fish_value": fish_value,
                "survived": int(not death),
                "death": int(death),
            }
        )
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        if show_progress:
            iterator.set_postfix(reward=f"{total_reward:.1f}", states=len(Q))

    return Q, log
