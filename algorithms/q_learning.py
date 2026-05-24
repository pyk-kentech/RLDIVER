"""Tabular Q-learning for the 8x8 diver MDP."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np
from tqdm import trange

from config import NUM_ACTIONS
from env import ReducedUnderwaterFishingEnv, State


def new_q_values() -> np.ndarray:
    """Create a pickle-safe zero Q row."""

    return np.zeros(NUM_ACTIONS, dtype=float)


def epsilon_greedy_action(
    Q,
    state: State,
    valid_actions: Sequence[int],
    epsilon: float,
) -> int:
    """Sample an epsilon-greedy action from valid actions only."""

    actions = list(valid_actions)
    if not actions:
        actions = list(range(NUM_ACTIONS))

    if random.random() < epsilon:
        return random.choice(actions)

    q_values = Q[state]
    best_value = max(float(q_values[action]) for action in actions)
    best_actions = [action for action in actions if float(q_values[action]) == best_value]
    return random.choice(best_actions)


def greedy_action_from_q(Q, state: State, valid_actions: Sequence[int]) -> int:
    """Return a greedy action from a Q-table."""

    return epsilon_greedy_action(Q, state, valid_actions, epsilon=0.0)


def train_q_learning(
    episodes: int,
    alpha: float,
    gamma: float,
    epsilon: float,
    epsilon_decay: float,
    min_epsilon: float,
    seed: int,
    show_progress: bool,
) -> Tuple[Any, List[Dict[str, Any]]]:
    """Train a sparse tabular Q-learning agent."""

    env = ReducedUnderwaterFishingEnv(stochastic=True, seed=seed)
    Q = defaultdict(new_q_values)
    log: List[Dict[str, Any]] = []

    iterator = trange(1, episodes + 1, desc="Q-learning", disable=not show_progress)
    for episode in iterator:
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        fish_value = 0.0
        terminal_reason = None

        while not done:
            action = epsilon_greedy_action(Q, state, env.get_valid_actions(state), epsilon)
            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            caught_fish_id = info.get("caught_fish_id")
            if caught_fish_id is not None:
                fish_value += env.fish_by_id[caught_fish_id]["value"]

            if done:
                target = reward
            else:
                next_valid = env.get_valid_actions(next_state)
                best_next = max(float(Q[next_state][a]) for a in next_valid) if next_valid else 0.0
                target = reward + gamma * best_next

            Q[state][action] += alpha * (target - Q[state][action])
            state = next_state
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
