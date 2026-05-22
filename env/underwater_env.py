"""Tabular underwater fishing environment.

The environment implements the finite MDP described in ``davethediver.md``.
It exposes stochastic simulation through ``step`` and an exact sparse model
through ``get_transitions`` for Value Iteration. No function approximation or
dense state-space table is used.
"""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from .config import (
    ACTION_CATCH,
    ACTION_SURFACE,
    ACTION_TO_ID,
    ACTIONS,
    CATCH_BASE_REWARD_MULTIPLIER,
    DEATH_PENALTY,
    FISH_CONFIG,
    GRID_DEPTH,
    GRID_WIDTH,
    INITIAL_FISH_MASK,
    INVALID_ACTION_PENALTY,
    MAX_EPISODE_STEPS,
    MAX_GLOBAL_TIME,
    MAX_OXYGEN,
    MAX_TRACKED_WEIGHT,
    MAX_WEIGHT_LIMIT,
    MOVEMENT_ACTIONS,
    MOVEMENT_REWARD,
    NORMAL_MOVE_COST,
    NUM_ACTIONS,
    OBSTACLES,
    OVERWEIGHT_MOVE_COST,
    RANDOM_SEED,
    START_POS,
    SURFACE_BONUS,
    TIMEOUT_UNDERWATER_PENALTY,
)

State = Tuple[int, int, int, int, int, int]
Transition = Tuple[float, State, float, bool, Dict[str, Any]]


def is_fish_available(mask: int, fish_id: int) -> bool:
    """Return True when the fish bit is still set in the availability mask."""

    return (mask & (1 << fish_id)) != 0


def remove_fish(mask: int, fish_id: int) -> int:
    """Clear one fish bit from the availability mask."""

    return mask & ~(1 << fish_id)


class UnderwaterFishingEnv:
    """Finite tabular MDP for oxygen-limited underwater fishing.

    State format:
        ``(x, d, oxygen, global_time, weight, fish_mask)``

    Action format:
        Integer action IDs matching ``env.config.ACTIONS``.
    """

    def __init__(self, stochastic: bool = True, seed: int = RANDOM_SEED):
        self.stochastic = stochastic
        self.seed = seed
        self.rng = random.Random(seed)

        self.fish_by_id = {fish["id"]: fish for fish in FISH_CONFIG}
        self.fish_by_position = {fish["position"]: fish for fish in FISH_CONFIG}
        self._validate_static_layout()

        self.state: State = self._initial_state()
        self.steps = 0

    def reset(self) -> State:
        """Reset the environment to the initial dive state."""

        self.state = self._initial_state()
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Apply one sampled environment transition.

        This method is used by SARSA, Q-learning, and evaluation rollouts. Fish
        attacks are sampled with the environment's local random number generator.
        """

        if not self._is_known_action(action):
            raise ValueError(f"Unknown action id: {action}")

        next_state, reward, done, info = self._sample_transition(self.state, action)
        self.steps += 1

        if not done and self.steps >= MAX_EPISODE_STEPS:
            done = True
            info = dict(info)
            info["terminal_reason"] = "max_steps"

        self.state = next_state
        return next_state, reward, done, info

    def get_transitions(self, state: State, action: int) -> List[Transition]:
        """Return all possible model transitions for a state-action pair.

        Deterministic actions return a one-element list. A valid catch returns
        two outcomes: no attack and attack, with probabilities defined by fish
        aggression. The returned transitions are suitable for sparse Value
        Iteration over reachable states.
        """

        if not self._is_known_action(action):
            raise ValueError(f"Unknown action id: {action}")

        x, d, oxygen, global_time, weight, fish_mask = state
        fish = self._available_fish_at((x, d), fish_mask)

        if action == ACTION_CATCH and fish is not None:
            aggression = float(fish["aggression"])
            no_attack = self._transition_with_attack_choice(state, action, attack=False)
            attack = self._transition_with_attack_choice(state, action, attack=True)

            transitions: List[Transition] = []
            if aggression < 1.0:
                transitions.append((1.0 - aggression, *no_attack))
            if aggression > 0.0:
                transitions.append((aggression, *attack))
            return transitions

        next_state, reward, done, info = self._transition_with_attack_choice(
            state, action, attack=False
        )
        return [(1.0, next_state, reward, done, info)]

    def get_valid_actions(self, state: State) -> List[int]:
        """Return logically valid actions for policy selection.

        Learning code may still call ``step`` with invalid actions during
        exploration, but epsilon-greedy policies should prefer this filtered
        action set for faster and more stable learning.
        """

        x, d, oxygen, global_time, weight, fish_mask = state
        if self._is_terminal_state(x, d, oxygen, global_time, fish_mask):
            return []

        valid_actions: List[int] = []

        for action, (dx, dd) in MOVEMENT_ACTIONS.items():
            target = (x + dx, d + dd)
            if self._is_valid_cell(target):
                valid_actions.append(action)

        if self._available_fish_at((x, d), fish_mask) is not None:
            valid_actions.append(ACTION_CATCH)

        # The surface action is a meaningful choice anywhere except at base
        # with no carried fish. It may still fail if oxygen or time is too low.
        if not ((x, d) == START_POS and weight == 0):
            valid_actions.append(ACTION_SURFACE)

        return valid_actions

    def render_ascii(self, state: Optional[State] = None) -> str:
        """Render a compact text map of the current grid."""

        if state is None:
            state = self.state

        agent_x, agent_d, _, _, _, fish_mask = state
        rows = []

        for d in range(GRID_DEPTH):
            cells = []
            for x in range(GRID_WIDTH):
                pos = (x, d)
                label = "."

                if pos in OBSTACLES:
                    label = "#"
                elif pos == START_POS:
                    label = "B"

                fish = self._available_fish_at(pos, fish_mask)
                if fish is not None:
                    label = f"F{fish['id']}"

                if (x, d) == (agent_x, agent_d):
                    label = "D"

                cells.append(f"{label:>2}")

            rows.append(" ".join(cells))

        return "\n".join(rows)

    def _sample_transition(
        self, state: State, action: int
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Sample one transition for stochastic simulation."""

        x, d, _, _, _, fish_mask = state
        fish = self._available_fish_at((x, d), fish_mask)

        if action == ACTION_CATCH and fish is not None and self.stochastic:
            attack = self.rng.random() < float(fish["aggression"])
        else:
            attack = False

        return self._transition_with_attack_choice(state, action, attack=attack)

    def _transition_with_attack_choice(
        self, state: State, action: int, attack: bool
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Apply one transition with an explicit fish attack outcome."""

        x, d, oxygen, global_time, weight, fish_mask = state
        info = self._base_info()
        reward: float

        if action in MOVEMENT_ACTIONS:
            dx, dd = MOVEMENT_ACTIONS[action]
            target = (x + dx, d + dd)

            if self._is_valid_cell(target):
                x, d = target
                cost = (
                    OVERWEIGHT_MOVE_COST
                    if weight > MAX_WEIGHT_LIMIT
                    else NORMAL_MOVE_COST
                )
                oxygen -= cost
                global_time -= cost
                reward = MOVEMENT_REWARD
                info["event"] = "move"
                info["oxygen_cost"] = cost
                info["time_cost"] = cost
            else:
                oxygen -= 1
                global_time -= 1
                reward = INVALID_ACTION_PENALTY
                info["event"] = "invalid_move"
                info["oxygen_cost"] = 1
                info["time_cost"] = 1

        elif action == ACTION_CATCH:
            fish = self._available_fish_at((x, d), fish_mask)

            if fish is None:
                oxygen -= 1
                global_time -= 1
                reward = INVALID_ACTION_PENALTY
                info["event"] = "invalid_catch"
                info["oxygen_cost"] = 1
                info["time_cost"] = 1
            else:
                fish_id = int(fish["id"])
                catch_cost = int(fish["health"])
                attack_damage = int(fish["attack_damage"]) if attack else 0

                fish_mask = remove_fish(fish_mask, fish_id)
                weight = min(weight + int(fish["weight"]), MAX_TRACKED_WEIGHT)
                oxygen -= catch_cost + attack_damage
                global_time -= catch_cost
                reward = float(fish["value"]) * CATCH_BASE_REWARD_MULTIPLIER

                info["event"] = "catch"
                info["caught_fish_id"] = fish_id
                info["attack"] = attack
                info["oxygen_cost"] = catch_cost + attack_damage
                info["time_cost"] = catch_cost

        elif action == ACTION_SURFACE:
            result = self._surface_transition(state)
            return result

        else:
            raise ValueError(f"Unknown action id: {action}")

        return self._finalize_transition(
            x=x,
            d=d,
            oxygen=oxygen,
            global_time=global_time,
            weight=weight,
            fish_mask=fish_mask,
            reward=reward,
            info=info,
        )

    def _surface_transition(self, state: State) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Handle the special surface action."""

        x, d, oxygen, global_time, weight, fish_mask = state
        info = self._base_info()
        info["event"] = "surface"

        if (x, d) == START_POS and weight == 0:
            oxygen -= 1
            global_time -= 1
            info["event"] = "invalid_surface"
            info["oxygen_cost"] = 1
            info["time_cost"] = 1
            return self._finalize_transition(
                x=x,
                d=d,
                oxygen=oxygen,
                global_time=global_time,
                weight=weight,
                fish_mask=fish_mask,
                reward=INVALID_ACTION_PENALTY,
                info=info,
            )

        return_cost = d + x
        info["oxygen_cost"] = return_cost
        info["time_cost"] = return_cost

        if oxygen > return_cost and global_time > return_cost:
            x, d = START_POS
            oxygen = MAX_OXYGEN
            global_time -= return_cost
            reward = float(SURFACE_BONUS) if weight > 0 else 0.0
            weight = 0

            return self._finalize_transition(
                x=x,
                d=d,
                oxygen=oxygen,
                global_time=global_time,
                weight=weight,
                fish_mask=fish_mask,
                reward=reward,
                info=info,
            )

        info["event"] = "failed_surface"
        info["terminal_reason"] = "failed_surface"
        next_state = self._make_state(x, d, oxygen, global_time, weight, fish_mask)
        return next_state, float(DEATH_PENALTY), True, info

    def _finalize_transition(
        self,
        x: int,
        d: int,
        oxygen: int,
        global_time: int,
        weight: int,
        fish_mask: int,
        reward: float,
        info: Dict[str, Any],
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Apply terminal-condition priority and clamp state fields."""

        done = False

        if oxygen <= 0:
            reward = float(DEATH_PENALTY)
            done = True
            info["terminal_reason"] = "death"
        elif global_time <= 0 and (x, d) != START_POS:
            reward += TIMEOUT_UNDERWATER_PENALTY
            done = True
            info["terminal_reason"] = "timeout_underwater"
        elif global_time <= 0 and (x, d) == START_POS:
            done = True
            info["terminal_reason"] = "time_finished_at_base"
        elif fish_mask == 0 and (x, d) == START_POS:
            done = True
            info["terminal_reason"] = "all_fish_caught"

        next_state = self._make_state(x, d, oxygen, global_time, weight, fish_mask)
        return next_state, float(reward), done, info

    def _make_state(
        self,
        x: int,
        d: int,
        oxygen: int,
        global_time: int,
        weight: int,
        fish_mask: int,
    ) -> State:
        """Create a finite-range state tuple."""

        return (
            int(x),
            int(d),
            max(0, min(int(oxygen), MAX_OXYGEN)),
            max(0, min(int(global_time), MAX_GLOBAL_TIME)),
            max(0, min(int(weight), MAX_TRACKED_WEIGHT)),
            int(fish_mask) & INITIAL_FISH_MASK,
        )

    def _initial_state(self) -> State:
        """Return the exact initial state from the specification."""

        return (
            START_POS[0],
            START_POS[1],
            MAX_OXYGEN,
            MAX_GLOBAL_TIME,
            0,
            INITIAL_FISH_MASK,
        )

    def _base_info(self) -> Dict[str, Any]:
        """Create a fresh info dictionary with all required keys."""

        return {
            "terminal_reason": None,
            "event": "none",
            "caught_fish_id": None,
            "attack": False,
            "oxygen_cost": 0,
            "time_cost": 0,
        }

    def _available_fish_at(self, position: Tuple[int, int], fish_mask: int) -> Optional[Dict[str, Any]]:
        """Return the available fish at a position, if one exists."""

        fish = self.fish_by_position.get(position)
        if fish is None:
            return None
        if not is_fish_available(fish_mask, int(fish["id"])):
            return None
        return fish

    def _is_valid_cell(self, position: Tuple[int, int]) -> bool:
        """Return True when a cell is inside the grid and not an obstacle."""

        x, d = position
        if x < 0 or x >= GRID_WIDTH:
            return False
        if d < 0 or d >= GRID_DEPTH:
            return False
        return position not in OBSTACLES

    def _is_terminal_state(
        self, x: int, d: int, oxygen: int, global_time: int, fish_mask: int
    ) -> bool:
        """Return True for states where no further decisions are meaningful."""

        if oxygen <= 0:
            return True
        if global_time <= 0:
            return True
        if fish_mask == 0 and (x, d) == START_POS:
            return True
        return False

    def _is_known_action(self, action: int) -> bool:
        """Return True when the action is one of the six defined actions."""

        return isinstance(action, int) and 0 <= action < NUM_ACTIONS

    def _validate_static_layout(self) -> None:
        """Validate the fixed map so configuration errors fail early."""

        if START_POS in OBSTACLES:
            raise ValueError("START_POS must not be an obstacle.")

        seen_positions = set()
        for fish in FISH_CONFIG:
            fish_id = int(fish["id"])
            position = fish["position"]

            if fish_id < 0 or fish_id >= len(FISH_CONFIG):
                raise ValueError(f"Fish id out of mask range: {fish_id}")
            if position in OBSTACLES:
                raise ValueError(f"Fish {fish_id} is placed inside an obstacle.")
            if not self._is_valid_cell(position):
                raise ValueError(f"Fish {fish_id} is outside the valid grid.")
            if position in seen_positions:
                raise ValueError(f"Multiple fish share position {position}.")

            seen_positions.add(position)


__all__ = [
    "State",
    "Transition",
    "UnderwaterFishingEnv",
    "is_fish_available",
    "remove_fish",
]
