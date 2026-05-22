"""Configuration constants for the tabular underwater fishing environment.

This module intentionally contains only explicit constants and small lookup
tables. The project specification requires a finite tabular MDP with sparse
value and action-value tables, so no dense state-space arrays are created here.
"""

# Grid geometry.
GRID_WIDTH = 11
GRID_DEPTH = 11
START_POS = (0, 0)

# Resource limits.
MAX_OXYGEN = 60
MAX_GLOBAL_TIME = 360
MAX_WEIGHT_LIMIT = 10
MAX_TRACKED_WEIGHT = 30

# Time and oxygen costs.
NORMAL_MOVE_COST = 1
OVERWEIGHT_MOVE_COST = 2

# Rewards and penalties.
INVALID_ACTION_PENALTY = -10
MOVEMENT_REWARD = -1
CATCH_BASE_REWARD_MULTIPLIER = 1.0
SURFACE_BONUS = 50
DEATH_PENALTY = -500
TIMEOUT_UNDERWATER_PENALTY = -100

# RL defaults.
GAMMA = 0.95
ALPHA = 0.10
INITIAL_EPSILON = 1.0
MIN_EPSILON = 0.05
EPSILON_DECAY = 0.995
TRAIN_EPISODES = 10000
EVAL_EPISODES = 500

# Dynamic Programming defaults.
VALUE_ITERATION_THETA = 1e-6
VALUE_ITERATION_MAX_ITERATIONS = 1000

# Plotting defaults.
MOVING_AVERAGE_WINDOW = 100

# Reproducibility.
RANDOM_SEED = 42

# Simulation-only safety limit. This is not used by the model transitions for
# Value Iteration because the MDP state itself already contains oxygen and time.
MAX_EPISODE_STEPS = 1000

# Fish availability is encoded as a six-bit mask. A set bit means the fish is
# still available; a cleared bit means the fish has already been caught.
INITIAL_FISH_MASK = 0b111111

# Action identifiers used throughout the project.
ACTION_UP = 0
ACTION_DOWN = 1
ACTION_LEFT = 2
ACTION_RIGHT = 3
ACTION_CATCH = 4
ACTION_SURFACE = 5

ACTIONS = ["up", "down", "left", "right", "catch", "surface"]
NUM_ACTIONS = len(ACTIONS)

ACTION_TO_ID = {name: action_id for action_id, name in enumerate(ACTIONS)}
ID_TO_ACTION = {action_id: name for action_id, name in enumerate(ACTIONS)}

MOVEMENT_ACTIONS = {
    ACTION_UP: (0, -1),
    ACTION_DOWN: (0, 1),
    ACTION_LEFT: (-1, 0),
    ACTION_RIGHT: (1, 0),
}

# Fixed fish table. The six fish count is kept to avoid exponential fish-mask
# growth, but positions are arranged on a more maze-like 11x11 ocean map.
FISH_CONFIG = [
    {
        "id": 0,
        "name": "Small Reef Fish",
        "position": (9, 4),
        "health": 2,
        "aggression": 0.10,
        "attack_damage": 2,
        "weight": 2,
        "value": 20,
    },
    {
        "id": 1,
        "name": "Striped Bass",
        "position": (6, 2),
        "health": 3,
        "aggression": 0.20,
        "attack_damage": 3,
        "weight": 3,
        "value": 35,
    },
    {
        "id": 2,
        "name": "Deep Snapper",
        "position": (0, 8),
        "health": 4,
        "aggression": 0.30,
        "attack_damage": 5,
        "weight": 4,
        "value": 55,
    },
    {
        "id": 3,
        "name": "Barracuda",
        "position": (9, 8),
        "health": 5,
        "aggression": 0.40,
        "attack_damage": 6,
        "weight": 5,
        "value": 75,
    },
    {
        "id": 4,
        "name": "Deep Tuna",
        "position": (7, 8),
        "health": 6,
        "aggression": 0.50,
        "attack_damage": 8,
        "weight": 6,
        "value": 95,
    },
    {
        "id": 5,
        "name": "Abyss Shark",
        "position": (5, 6),
        "health": 7,
        "aggression": 0.60,
        "attack_damage": 10,
        "weight": 7,
        "value": 130,
    },
]

# Fixed obstacle cells that Dave cannot enter. Surface cells are kept open.
OBSTACLES = {
    (0, 1),
    (1, 1),
    (2, 1),
    (3, 1),
    (4, 1),
    (5, 1),
    (6, 1),
    (7, 1),
    (9, 1),
    (0, 2),
    (0, 3),
    (1, 3),
    (2, 3),
    (3, 3),
    (4, 3),
    (6, 3),
    (7, 3),
    (8, 3),
    (10, 3),
    (0, 4),
    (4, 4),
    (8, 4),
    (0, 5),
    (1, 5),
    (2, 5),
    (4, 5),
    (5, 5),
    (6, 5),
    (8, 5),
    (9, 5),
    (10, 5),
    (6, 6),
    (2, 7),
    (3, 7),
    (4, 7),
    (6, 7),
    (7, 7),
    (8, 7),
    (10, 7),
    (4, 8),
    (0, 9),
    (1, 9),
    (2, 9),
    (4, 9),
    (5, 9),
    (6, 9),
    (7, 9),
    (8, 9),
    (10, 9),
}
