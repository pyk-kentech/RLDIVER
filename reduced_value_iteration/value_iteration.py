"""Sparse Value Iteration for the reduced MDP."""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Any, DefaultDict, Dict, Iterable, List, Optional, Tuple

from tqdm import tqdm, trange

from config import ACTIONS, GAMMA, VALUE_ITERATION_MAX_ITERATIONS, VALUE_ITERATION_THETA
from env import State

VTable = DefaultDict[State, float]
Policy = Dict[State, int]


def build_reachable_states(env, show_progress: bool = True) -> List[State]:
    """Build reachable states with BFS instead of a dense Cartesian product."""

    initial_state = env.reset()
    queue = deque([initial_state])
    visited = {initial_state}
    states: List[State] = []

    progress = tqdm(desc="Reduced reachable-state BFS", unit="state", disable=not show_progress)
    try:
        while queue:
            state = queue.popleft()
            states.append(state)
            progress.update(1)

            if not env.get_valid_actions(state):
                continue

            for action in range(len(ACTIONS)):
                for _, next_state, _, done, _ in env.get_transitions(state, action):
                    if done or next_state in visited:
                        continue
                    visited.add(next_state)
                    queue.append(next_state)

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
    """Run Value Iteration on reachable states."""

    state_list = build_reachable_states(env, show_progress=show_progress) if states is None else list(states)
    V: VTable = defaultdict(float)
    iteration_log: List[Dict[str, Any]] = []

    iterator = trange(
        1,
        max_iterations + 1,
        desc="Reduced Value Iteration",
        unit="iteration",
        disable=not show_progress,
    )
    for iteration in iterator:
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
            iterator.set_postfix(delta=f"{delta:.6g}", states=len(state_list))
        if delta < theta:
            break

    policy = extract_policy(env, V, state_list, gamma)
    return V, policy, state_list, iteration_log


def extract_policy(env, V: VTable, states: Iterable[State], gamma: float = GAMMA) -> Policy:
    """Extract a deterministic greedy policy from V."""

    policy: Policy = {}
    for state in states:
        valid_actions = env.get_valid_actions(state)
        if not valid_actions:
            continue
        policy[state] = max(
            valid_actions,
            key=lambda action: _expected_action_value(env, V, state, action, gamma),
        )
    return policy


def _expected_action_value(env, V: VTable, state: State, action: int, gamma: float) -> float:
    value = 0.0
    for probability, next_state, reward, done, _ in env.get_transitions(state, action):
        target = reward if done else reward + gamma * V[next_state]
        value += probability * target
    return value
