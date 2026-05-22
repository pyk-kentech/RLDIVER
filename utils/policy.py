"""Policy helpers for sparse tabular reinforcement learning.

All functions in this module operate on explicit table lookups. They do not
use function approximation. Action selection strongly prefers the valid action
set supplied by the environment, which prevents wasted exploration on obvious
boundary collisions and invalid catches while still allowing stochastic
epsilon-greedy exploration among meaningful choices.
"""

from __future__ import annotations

import random
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np

from env.config import NUM_ACTIONS
from env.underwater_env import State


def _normalize_valid_actions(valid_actions: Sequence[int]) -> List[int]:
    """Return a usable action list, falling back to all actions if necessary."""

    actions = [int(action) for action in valid_actions]
    if actions:
        return actions
    return list(range(NUM_ACTIONS))


def _random_argmax(action_values: Mapping[int, float]) -> int:
    """Return a random action among the highest-valued actions."""

    max_value = max(action_values.values())
    best_actions = [
        action for action, value in action_values.items() if value == max_value
    ]
    return random.choice(best_actions)


def epsilon_greedy_action(
    Q: Mapping[State, np.ndarray],
    state: State,
    valid_actions: Sequence[int],
    epsilon: float,
) -> int:
    """Choose an action using epsilon-greedy exploration over valid actions.

    The function always samples from ``valid_actions`` when that sequence is not
    empty. This is the intended behavior for this project because invalid moves
    are still modeled in the environment, but repeatedly selecting them during
    exploration slows learning without adding useful state coverage.
    """

    actions = _normalize_valid_actions(valid_actions)
    epsilon = max(0.0, min(1.0, float(epsilon)))

    if random.random() < epsilon:
        return random.choice(actions)

    return greedy_action_from_q(Q, state, actions)


def greedy_action_from_q(
    Q: Mapping[State, np.ndarray], state: State, valid_actions: Sequence[int]
) -> int:
    """Return the greedy Q-table action with random tie-breaking."""

    actions = _normalize_valid_actions(valid_actions)

    if state in Q:
        q_values = Q[state]
    else:
        q_values = np.zeros(NUM_ACTIONS, dtype=float)

    action_values = {action: float(q_values[action]) for action in actions}
    return _random_argmax(action_values)


def greedy_action_from_policy(
    policy: Mapping[State, int], state: State, valid_actions: Sequence[int]
) -> int:
    """Return the policy action if valid, otherwise a safe valid fallback.

    A policy learned or extracted on a subset of reachable states may not
    contain every state encountered during stochastic evaluation. In that case,
    this helper returns a random valid action instead of raising an exception.
    """

    actions = _normalize_valid_actions(valid_actions)
    action = policy.get(state)

    if action in actions:
        return int(action)

    return random.choice(actions)


def q_to_policy(
    Q: Mapping[State, np.ndarray],
    env,
    states: Optional[Iterable[State]] = None,
) -> Dict[State, int]:
    """Convert a sparse Q-table into a deterministic greedy policy dictionary."""

    if states is None:
        states = Q.keys()

    policy: Dict[State, int] = {}
    for state in states:
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            continue
        policy[state] = greedy_action_from_q(Q, state, valid_actions)

    return policy


__all__ = [
    "epsilon_greedy_action",
    "greedy_action_from_q",
    "greedy_action_from_policy",
    "q_to_policy",
]
