"""Final underwater fishing environment for the 8x8 tabular MDP."""

from __future__ import annotations

import random
from typing import Any, Dict, List, Optional, Tuple

from config import (
    ACTION_CATCH,
    ACTION_SURFACE,
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
    """Return True if a fish is still available in the bit mask."""

    return (mask & (1 << fish_id)) != 0


def remove_fish(mask: int, fish_id: int) -> int:
    """Remove one fish from the availability bit mask."""

    return mask & ~(1 << fish_id)


class ReducedUnderwaterFishingEnv:
    """Finite tabular MDP for the underwater fishing task."""

    def __init__(self, stochastic: bool = True, seed: int = RANDOM_SEED):
        self.stochastic = stochastic
        self.rng = random.Random(seed)
        self.fish_by_id = {fish["id"]: fish for fish in FISH_CONFIG}
        self.fish_by_position = {fish["position"]: fish for fish in FISH_CONFIG}
        self._validate_static_layout()
        self.state = self._initial_state()
        self.steps = 0

    def reset(self) -> State:
        """Reset the episode to the initial state."""

        self.state = self._initial_state()
        self.steps = 0
        return self.state

    def step(self, action: int) -> Tuple[State, float, bool, Dict[str, Any]]:
        """Sample one stochastic transition."""

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
        """Return exact transition outcomes for model-based DP."""

        if not self._is_known_action(action):
            raise ValueError(f"Unknown action id: {action}")

        x, d, _, _, _, fish_mask = state
        fish = self._available_fish_at((x, d), fish_mask)
        if action == ACTION_CATCH and fish is not None:
            aggression = float(fish["aggression"])
            no_attack = self._transition_with_attack_choice(state, action, False)
            attack = self._transition_with_attack_choice(state, action, True)
            return [
                (1.0 - aggression, *no_attack),
                (aggression, *attack),
            ]

        next_state, reward, done, info = self._transition_with_attack_choice(
            state, action, False
        )
        return [(1.0, next_state, reward, done, info)]

    def get_valid_actions(self, state: State) -> List[int]:
        """Return logically valid actions for Bellman backups and rollouts."""

        x, d, oxygen, global_time, weight, fish_mask = state
        if self._is_terminal_state(x, d, oxygen, global_time, fish_mask):
            return []

        valid_actions = []
        for action, (dx, dd) in MOVEMENT_ACTIONS.items():
            if self._is_valid_cell((x + dx, d + dd)):
                valid_actions.append(action)

        if self._available_fish_at((x, d), fish_mask) is not None:
            valid_actions.append(ACTION_CATCH)

        if not ((x, d) == START_POS and weight == 0):
            valid_actions.append(ACTION_SURFACE)

        return valid_actions

    def render_ascii(self, state: Optional[State] = None) -> str:
        """Render the map as text."""

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
                if pos == (agent_x, agent_d):
                    label = "D"
                cells.append(f"{label:>2}")
            rows.append(" ".join(cells))
        return "\n".join(rows)

    def _sample_transition(self, state: State, action: int):
        x, d, _, _, _, fish_mask = state
        fish = self._available_fish_at((x, d), fish_mask)
        attack = (
            action == ACTION_CATCH
            and fish is not None
            and self.stochastic
            and self.rng.random() < float(fish["aggression"])
        )
        return self._transition_with_attack_choice(state, action, attack)

    def _transition_with_attack_choice(
        self, state: State, action: int, attack: bool
    ) -> Tuple[State, float, bool, Dict[str, Any]]:
        x, d, oxygen, global_time, weight, fish_mask = state
        info = self._base_info()

        if action in MOVEMENT_ACTIONS:
            dx, dd = MOVEMENT_ACTIONS[action]
            target = (x + dx, d + dd)
            if self._is_valid_cell(target):
                x, d = target
                cost = OVERWEIGHT_MOVE_COST if weight > MAX_WEIGHT_LIMIT else NORMAL_MOVE_COST
                oxygen -= cost
                global_time -= cost
                reward = MOVEMENT_REWARD
                info.update({"event": "move", "oxygen_cost": cost, "time_cost": cost})
            else:
                oxygen -= 1
                global_time -= 1
                reward = INVALID_ACTION_PENALTY
                info.update({"event": "invalid_move", "oxygen_cost": 1, "time_cost": 1})

        elif action == ACTION_CATCH:
            fish = self._available_fish_at((x, d), fish_mask)
            if fish is None:
                oxygen -= 1
                global_time -= 1
                reward = INVALID_ACTION_PENALTY
                info.update({"event": "invalid_catch", "oxygen_cost": 1, "time_cost": 1})
            else:
                fish_id = int(fish["id"])
                catch_cost = int(fish["health"])
                attack_damage = int(fish["attack_damage"]) if attack else 0
                fish_mask = remove_fish(fish_mask, fish_id)
                weight = min(weight + int(fish["weight"]), MAX_TRACKED_WEIGHT)
                oxygen -= catch_cost + attack_damage
                global_time -= catch_cost
                reward = float(fish["value"]) * CATCH_BASE_REWARD_MULTIPLIER
                info.update(
                    {
                        "event": "catch",
                        "caught_fish_id": fish_id,
                        "attack": attack,
                        "oxygen_cost": catch_cost + attack_damage,
                        "time_cost": catch_cost,
                    }
                )

        elif action == ACTION_SURFACE:
            return self._surface_transition(state)
        else:
            raise ValueError(f"Unknown action id: {action}")

        return self._finalize_transition(
            x, d, oxygen, global_time, weight, fish_mask, reward, info
        )

    def _surface_transition(self, state: State):
        x, d, oxygen, global_time, weight, fish_mask = state
        info = self._base_info()
        info["event"] = "surface"

        if (x, d) == START_POS and weight == 0:
            oxygen -= 1
            global_time -= 1
            info.update({"event": "invalid_surface", "oxygen_cost": 1, "time_cost": 1})
            return self._finalize_transition(
                x, d, oxygen, global_time, weight, fish_mask, INVALID_ACTION_PENALTY, info
            )

        return_cost = x + d
        info.update({"oxygen_cost": return_cost, "time_cost": return_cost})
        if oxygen > return_cost and global_time > return_cost:
            reward = float(SURFACE_BONUS) if weight > 0 else 0.0
            return self._finalize_transition(
                START_POS[0],
                START_POS[1],
                MAX_OXYGEN,
                global_time - return_cost,
                0,
                fish_mask,
                reward,
                info,
            )

        info.update({"event": "failed_surface", "terminal_reason": "failed_surface"})
        return self._make_state(x, d, oxygen, global_time, weight, fish_mask), float(DEATH_PENALTY), True, info

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
    ):
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
        return self._make_state(x, d, oxygen, global_time, weight, fish_mask), float(reward), done, info

    def _make_state(self, x, d, oxygen, global_time, weight, fish_mask) -> State:
        return (
            int(x),
            int(d),
            max(0, min(int(oxygen), MAX_OXYGEN)),
            max(0, min(int(global_time), MAX_GLOBAL_TIME)),
            max(0, min(int(weight), MAX_TRACKED_WEIGHT)),
            int(fish_mask) & INITIAL_FISH_MASK,
        )

    def _initial_state(self) -> State:
        return (START_POS[0], START_POS[1], MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK)

    def _base_info(self) -> Dict[str, Any]:
        return {
            "terminal_reason": None,
            "event": "none",
            "caught_fish_id": None,
            "attack": False,
            "oxygen_cost": 0,
            "time_cost": 0,
        }

    def _available_fish_at(self, position, fish_mask):
        fish = self.fish_by_position.get(position)
        if fish is None or not is_fish_available(fish_mask, int(fish["id"])):
            return None
        return fish

    def _is_valid_cell(self, position) -> bool:
        x, d = position
        return 0 <= x < GRID_WIDTH and 0 <= d < GRID_DEPTH and position not in OBSTACLES

    def _is_terminal_state(self, x, d, oxygen, global_time, fish_mask) -> bool:
        return oxygen <= 0 or global_time <= 0 or (fish_mask == 0 and (x, d) == START_POS)

    def _is_known_action(self, action: int) -> bool:
        return isinstance(action, int) and 0 <= action < NUM_ACTIONS

    def _validate_static_layout(self) -> None:
        if START_POS in OBSTACLES:
            raise ValueError("START_POS must not be an obstacle.")
        positions = set()
        for fish in FISH_CONFIG:
            position = fish["position"]
            fish_id = int(fish["id"])
            if fish_id < 0 or fish_id >= len(FISH_CONFIG):
                raise ValueError(f"Fish id out of range: {fish_id}")
            if position in OBSTACLES or not self._is_valid_cell(position):
                raise ValueError(f"Invalid fish position for fish {fish_id}: {position}")
            if position in positions:
                raise ValueError(f"Duplicate fish position: {position}")
            positions.add(position)
