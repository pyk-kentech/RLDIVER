"""Tabular Q-learning for the underwater fishing MDP."""

from __future__ import annotations

import random
from collections import defaultdict
from typing import Any, DefaultDict, Dict, List, Tuple

import numpy as np
from tqdm import trange

from env.config import (
    ALPHA,
    EPSILON_DECAY,
    GAMMA,
    INITIAL_EPSILON,
    MIN_EPSILON,
    NUM_ACTIONS,
    RANDOM_SEED,
    TRAIN_EPISODES,
)
from env.underwater_env import State
from utils.policy import epsilon_greedy_action

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

QTable = DefaultDict[State, np.ndarray]


def _new_q_values() -> np.ndarray:
    """Create a zero-valued action row for a newly visited state.

    A named factory is equivalent to ``lambda: np.zeros(NUM_ACTIONS)`` but can
    be saved with standard ``pickle``.
    """

    return np.zeros(NUM_ACTIONS, dtype=float)


def train_q_learning(
    env,
    episodes: int = TRAIN_EPISODES,
    alpha: float = ALPHA,
    gamma: float = GAMMA,
    epsilon: float = INITIAL_EPSILON,
    epsilon_decay: float = EPSILON_DECAY,
    show_progress: bool = True,
) -> Tuple[QTable, List[Dict[str, Any]]]:
    """Train a sparse tabular Q-learning agent.

    The Q-table is a ``defaultdict`` whose values are one-dimensional NumPy
    arrays of length ``NUM_ACTIONS``. This keeps memory proportional to visited
    states instead of the full theoretical state space.
    """

    Q: QTable = defaultdict(_new_q_values)
    training_log: List[Dict[str, Any]] = []

    episode_iter = trange(
        1,
        episodes + 1,
        desc="Q-learning",
        unit="episode",
        disable=not show_progress,
    )

    for episode in episode_iter:
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        fish_value = 0.0
        terminal_reason = None

        while not done:
            valid_actions = env.get_valid_actions(state)
            action = epsilon_greedy_action(Q, state, valid_actions, epsilon)

            next_state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            caught_fish_id = info.get("caught_fish_id")
            if caught_fish_id is not None:
                fish_value += float(env.fish_by_id[caught_fish_id]["value"])

            if done:
                target = reward
            else:
                valid_next_actions = env.get_valid_actions(next_state)
                if valid_next_actions:
                    best_next = max(
                        float(Q[next_state][next_action])
                        for next_action in valid_next_actions
                    )
                else:
                    best_next = 0.0
                target = reward + gamma * best_next

            Q[state][action] += alpha * (target - Q[state][action])

            state = next_state
            terminal_reason = info.get("terminal_reason")

        death = terminal_reason in {"death", "failed_surface"}
        training_log.append(
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

        epsilon = max(MIN_EPSILON, epsilon * epsilon_decay)

        if show_progress:
            episode_iter.set_postfix(
                reward=f"{total_reward:.1f}",
                epsilon=f"{epsilon:.3f}",
                states=len(Q),
                terminal=terminal_reason,
            )

    return Q, training_log


__all__ = ["train_q_learning"]
