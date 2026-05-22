"""Sparse Value Iteration for the underwater fishing MDP."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm, trange

from env.config import (
    ACTIONS,
    GAMMA,
    VALUE_ITERATION_MAX_ITERATIONS,
    VALUE_ITERATION_THETA,
)
from env.underwater_env import State

VTable = DefaultDict[State, float]
Policy = Dict[State, int]


def build_reachable_states(
    env,
    max_states: Optional[int] = None,
    show_progress: bool = True,
) -> List[State]:
    """Discover reachable decision states with BFS from the initial state.

    This function deliberately avoids the full Cartesian state space. It uses a
    ``deque`` for breadth-first expansion and a Python ``set`` for visited
    states, as required by the project specification. Terminal transition
    outcomes are not enqueued because Value Iteration only needs to update
    states where a future action can be selected.
    """

    initial_state = env.reset()
    queue = deque([initial_state])
    visited = {initial_state}
    states: List[State] = []

    progress = tqdm(
        desc="Reachable-state BFS",
        unit="state",
        disable=not show_progress,
    )

    try:
        while queue:
            state = queue.popleft()
            states.append(state)
            progress.update(1)

            # Terminal decision states have no useful outgoing action updates.
            if not env.get_valid_actions(state):
                continue

            for action in range(len(ACTIONS)):
                for _, next_state, _, done, _ in env.get_transitions(state, action):
                    if done:
                        continue
                    if next_state in visited:
                        continue

                    visited.add(next_state)
                    queue.append(next_state)

                    if max_states is not None and len(visited) > max_states:
                        raise MemoryError(
                            "Reachable state limit exceeded. Increase max_states or "
                            "use the reduced Value Iteration configuration from the spec."
                        )

            if show_progress and len(states) % 1000 == 0:
                progress.set_postfix(visited=len(visited), queue=len(queue))
    finally:
        progress.close()

    return states


def value_iteration(
    env,
    gamma: float = GAMMA,
    theta: float = VALUE_ITERATION_THETA,
    max_iterations: int = VALUE_ITERATION_MAX_ITERATIONS,
    states: Optional[Iterable[State]] = None,
    show_progress: bool = True,
) -> Tuple[VTable, Policy, List[State], List[Dict[str, Any]]]:
    """Run sparse Value Iteration over BFS-discovered reachable states.

    Returns:
        ``(V, policy, states, iteration_log)`` where ``V`` is a
        ``defaultdict(float)`` and ``policy`` maps reachable states to greedy
        action IDs.
    """

    if states is None:
        state_list = build_reachable_states(env, show_progress=show_progress)
    else:
        state_list = list(states)

    V: VTable = defaultdict(float)
    iteration_log: List[Dict[str, Any]] = []

    iteration_iter = trange(
        1,
        max_iterations + 1,
        desc="Value Iteration",
        unit="iteration",
        disable=not show_progress,
    )

    for iteration in iteration_iter:
        delta = 0.0
        V_new: VTable = defaultdict(float)

        for state in state_list:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                V_new[state] = 0.0
                continue

            best_value = max(
                _expected_action_value(env, V, state, action, gamma)
                for action in valid_actions
            )
            V_new[state] = best_value
            delta = max(delta, abs(best_value - V[state]))

        V = V_new
        iteration_log.append({"iteration": iteration, "delta": delta})
        if show_progress:
            iteration_iter.set_postfix(delta=f"{delta:.6g}", states=len(state_list))

        if delta < theta:
            break

    policy = extract_policy(env, V, state_list, gamma)
    return V, policy, state_list, iteration_log


def extract_policy(
    env, V: VTable, states: Iterable[State], gamma: float = GAMMA
) -> Policy:
    """Extract a greedy policy from a value table using the transition model."""

    policy: Policy = {}

    for state in states:
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            continue

        best_action = valid_actions[0]
        best_value = _expected_action_value(env, V, state, best_action, gamma)

        for action in valid_actions[1:]:
            action_value = _expected_action_value(env, V, state, action, gamma)
            if action_value > best_value:
                best_action = action
                best_value = action_value

        policy[state] = best_action

    return policy


def _expected_action_value(env, V: VTable, state: State, action: int, gamma: float) -> float:
    """Compute the Bellman expectation for one state-action pair."""

    value = 0.0
    for probability, next_state, reward, done, _ in env.get_transitions(state, action):
        if done:
            target = reward
        else:
            target = reward + gamma * V[next_state]
        value += probability * target
    return value


__all__ = ["build_reachable_states", "value_iteration", "extract_policy"]
