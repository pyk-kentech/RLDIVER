"""Reduced configuration for a tractable Value Iteration baseline.

This folder is intentionally separate from the full project. It keeps the same
state variables and transition logic, but uses a smaller map, fewer fish, lower
oxygen, and lower global time so exact tabular DP can finish on normal course
servers.
"""

GRID_WIDTH = 5
GRID_DEPTH = 5
START_POS = (0, 0)

MAX_OXYGEN = 22
MAX_GLOBAL_TIME = 50

MAX_WEIGHT_LIMIT = 5
MAX_TRACKED_WEIGHT = 10

NORMAL_MOVE_COST = 1
OVERWEIGHT_MOVE_COST = 2

INVALID_ACTION_PENALTY = -10
MOVEMENT_REWARD = -1
CATCH_BASE_REWARD_MULTIPLIER = 1.0
SURFACE_BONUS = 50
DEATH_PENALTY = -500
TIMEOUT_UNDERWATER_PENALTY = -100

GAMMA = 0.95
VALUE_ITERATION_THETA = 1e-6
VALUE_ITERATION_MAX_ITERATIONS = 1000
EVAL_EPISODES = 200
RANDOM_SEED = 42
MAX_EPISODE_STEPS = 120

# Three fish means a 3-bit availability mask. This keeps a nontrivial capture
# order decision while making exact DP tractable on a course server.
INITIAL_FISH_MASK = 0b111

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
# F2: farthest fish that tests whether the policy can still return safely.
FISH_CONFIG = [
    {
        "id": 0,
        "name": "Training Reef Fish",
        "position": (1, 2),
        "health": 2,
        "aggression": 0.10,
        "attack_damage": 2,
        "weight": 2,
        "value": 20,
    },
    {
        "id": 1,
        "name": "Cave Bass",
        "position": (3, 2),
        "health": 3,
        "aggression": 0.20,
        "attack_damage": 3,
        "weight": 3,
        "value": 40,
    },
    {
        "id": 2,
        "name": "Longfin Snapper",
        "position": (0, 4),
        "health": 4,
        "aggression": 0.25,
        "attack_damage": 4,
        "weight": 4,
        "value": 70,
    },
]

OBSTACLES = {
    (0, 1),
    (1, 1),
    (3, 1),
    (1, 3),
    (2, 3),
}
