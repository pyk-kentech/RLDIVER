"""Reduced configuration for a tractable Value Iteration baseline.

This folder is intentionally separate from the full project. It keeps the same
state variables and transition logic, but uses a smaller map, fewer fish, lower
oxygen, and lower global time so exact tabular DP can finish on normal course
servers.
"""

GRID_WIDTH = 8
GRID_DEPTH = 8
START_POS = (0, 0)

MAX_OXYGEN = 34
MAX_GLOBAL_TIME = 80

MAX_WEIGHT_LIMIT = 7
MAX_TRACKED_WEIGHT = 14

NORMAL_MOVE_COST = 1
OVERWEIGHT_MOVE_COST = 2

INVALID_ACTION_PENALTY = -10
MOVEMENT_REWARD = -1
CATCH_BASE_REWARD_MULTIPLIER = 1.0
SURFACE_BONUS = 50
DEATH_PENALTY = -1000
TIMEOUT_UNDERWATER_PENALTY = -100

GAMMA = 0.95
VALUE_ITERATION_THETA = 1e-6
VALUE_ITERATION_MAX_ITERATIONS = 1000
EVAL_EPISODES = 200
RANDOM_SEED = 42
MAX_EPISODE_STEPS = 240

# Four fish means a 4-bit availability mask. This is still much smaller than
# the full 6-fish mask, but it preserves a meaningful capture-order problem.
INITIAL_FISH_MASK = 0b1111

ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CATCH = 4
ACTION_SURFACE = 5

ACTIONS = ["up", "down", "left", "right", "catch", "surface"]
NUM_ACTIONS = len(ACTIONS)

MOVEMENT_ACTIONS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Meaningful fish placement:
# F0: safe early fish near the entrance, a natural first target.
# F1: medium-value fish on a short right-side detour.
# F2: deeper high-value fish on the lower-right route.
# F3: farthest fish, valuable but costly to reach and return from.
FISH_CONFIG = [
    {
        "id": 0,
        "name": "Starry Puffer",
        "position": (2, 2),
        "health": 2,
        "aggression": 0.10,
        "attack_damage": 2,
        "weight": 2,
        "value": 20,
    },
    {
        "id": 1,
        "name": "Sheepshead",
        "position": (6, 2),
        "health": 3,
        "aggression": 0.20,
        "attack_damage": 3,
        "weight": 3,
        "value": 40,
    },
    {
        "id": 2,
        "name": "Marlin",
        "position": (5, 5),
        "health": 4,
        "aggression": 0.30,
        "attack_damage": 5,
        "weight": 4,
        "value": 70,
    },
    {
        "id": 3,
        "name": "Tiger Shark",
        "position": (1, 7),
        "health": 5,
        "aggression": 0.45,
        "attack_damage": 6,
        "weight": 5,
        "value": 100,
    },
]

OBSTACLES = {
    (0, 1),
    (1, 1),
    (3, 1),
    (5, 1),
    (6, 1),
    (3, 2),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (6, 3),
    (6, 4),
    (0, 5),
    (1, 5),
    (3, 5),
    (3, 6),
    (4, 6),
    (5, 6),
    (7, 6),
}
