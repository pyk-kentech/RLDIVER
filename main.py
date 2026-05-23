"""Command-line interface for the Dave the Diver tabular RL project."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np
import pandas as pd

if __package__ is None or __package__ == "":
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from algorithms.q_learning import train_q_learning
from algorithms.sarsa import train_sarsa
from algorithms.value_iteration import value_iteration
from env import UnderwaterFishingEnv
from env.config import (
    ALPHA,
    EPSILON_DECAY,
    EVAL_EPISODES,
    FISH_CONFIG,
    GAMMA,
    GRID_DEPTH,
    GRID_WIDTH,
    INITIAL_FISH_MASK,
    INITIAL_EPSILON,
    MAX_GLOBAL_TIME,
    MAX_OXYGEN,
    MIN_EPSILON,
    OBSTACLES,
    RANDOM_SEED,
    TRAIN_EPISODES,
    VALUE_ITERATION_MAX_ITERATIONS,
    VALUE_ITERATION_THETA,
)
from env.underwater_env import State
from utils.evaluation import evaluate_policy, rollout_episode
from utils.io_utils import ensure_dir, load_pickle, save_csv, save_pickle
from utils.plotting import (
    plot_death_rate_curve,
    plot_oxygen_time_curve,
    plot_policy_map,
    plot_reward_curve_comparison,
    plot_survival_rate_curve,
    plot_training_reward_curve,
    plot_trajectory,
)
from utils.policy import greedy_action_from_policy, greedy_action_from_q, q_to_policy


PROJECT_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROJECT_DIR / "results"


def set_global_seeds(seed: int = RANDOM_SEED) -> None:
    """Set global random seeds for reproducible scripts."""

    random.seed(seed)
    np.random.seed(seed)


def run_value_iteration(args: argparse.Namespace) -> None:
    """Run sparse Value Iteration and save values, policy, and iteration log."""

    env = UnderwaterFishingEnv(stochastic=False, seed=args.seed)
    V, policy, states, iteration_log = value_iteration(
        env,
        gamma=args.gamma,
        theta=args.theta,
        max_iterations=args.max_iterations,
        show_progress=args.show_progress,
    )

    save_pickle(V, RESULTS_DIR / "value_iteration_values.pkl")
    save_pickle(policy, RESULTS_DIR / "value_iteration_policy.pkl")
    save_csv(iteration_log, RESULTS_DIR / "value_iteration_log.csv")

    print(f"Value Iteration finished with {len(states)} reachable states.")
    print(f"Saved value and policy files to {RESULTS_DIR}.")


def run_train_sarsa(args: argparse.Namespace) -> None:
    """Train SARSA and save its Q-table, training log, and greedy policy."""

    env = UnderwaterFishingEnv(stochastic=True, seed=args.seed)
    Q, training_log = train_sarsa(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        show_progress=args.show_progress,
    )
    policy = q_to_policy(Q, env, states=_representative_policy_states())

    save_pickle(Q, RESULTS_DIR / "sarsa_q.pkl")
    save_pickle(policy, RESULTS_DIR / "sarsa_policy.pkl")
    save_csv(training_log, RESULTS_DIR / "sarsa_training_log.csv")

    print(f"SARSA training finished for {args.episodes} episodes.")
    print(f"Visited states in Q-table: {len(Q)}.")


def run_train_q_learning(args: argparse.Namespace) -> None:
    """Train Q-learning and save its Q-table, training log, and greedy policy."""

    env = UnderwaterFishingEnv(stochastic=True, seed=args.seed)
    Q, training_log = train_q_learning(
        env,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_decay=args.epsilon_decay,
        show_progress=args.show_progress,
    )
    policy = q_to_policy(Q, env, states=_representative_policy_states())

    save_pickle(Q, RESULTS_DIR / "q_learning_q.pkl")
    save_pickle(policy, RESULTS_DIR / "q_learning_policy.pkl")
    save_csv(training_log, RESULTS_DIR / "q_learning_training_log.csv")

    print(f"Q-learning training finished for {args.episodes} episodes.")
    print(f"Visited states in Q-table: {len(Q)}.")


def run_evaluate(args: argparse.Namespace) -> None:
    """Evaluate all available policies and save the summary CSV."""

    rows: List[Dict[str, Any]] = []

    for method, policy_fn in _available_policy_functions(args.seed):
        env = UnderwaterFishingEnv(stochastic=True, seed=args.seed)
        row = evaluate_policy(env, policy_fn, episodes=args.eval_episodes, method=method)
        rows.append(row)
        print(f"Evaluated {method}: avg_reward={row['avg_reward']:.2f}")

    if not rows:
        print("No trained policies were found. Train at least one model first.")
        return

    save_csv(rows, RESULTS_DIR / "evaluation_summary.csv")
    pd.DataFrame(rows).to_csv(RESULTS_DIR / "evaluation_summary.csv", index=False)
    print(f"Saved evaluation summary to {RESULTS_DIR / 'evaluation_summary.csv'}.")


def run_plot(args: argparse.Namespace) -> None:
    """Generate training curves, policy maps, and trajectory plots."""

    ensure_dir(RESULTS_DIR)

    q_log = _read_csv_if_exists(RESULTS_DIR / "q_learning_training_log.csv")
    sarsa_log = _read_csv_if_exists(RESULTS_DIR / "sarsa_training_log.csv")

    if q_log:
        plot_training_reward_curve(
            q_log,
            RESULTS_DIR / "q_learning_reward_curve.png",
            title="Q-learning Reward Curve",
        )
    if sarsa_log:
        plot_training_reward_curve(
            sarsa_log,
            RESULTS_DIR / "sarsa_reward_curve.png",
            title="SARSA Reward Curve",
        )
    if q_log and sarsa_log:
        plot_reward_curve_comparison(
            q_log, sarsa_log, RESULTS_DIR / "reward_curve_comparison.png"
        )
        plot_survival_rate_curve(
            q_log, sarsa_log, RESULTS_DIR / "survival_rate_curve.png"
        )
        plot_death_rate_curve(q_log, sarsa_log, RESULTS_DIR / "death_rate_curve.png")

    _plot_policy_and_trajectory_files(args.seed)
    print(f"Saved available plots to {RESULTS_DIR}.")


def run_all(args: argparse.Namespace) -> None:
    """Run Value Iteration, SARSA, Q-learning, evaluation, and plotting."""

    print("[1/5] Running Value Iteration")
    run_value_iteration(args)
    print("[2/5] Training SARSA")
    run_train_sarsa(args)
    print("[3/5] Training Q-learning")
    run_train_q_learning(args)
    print("[4/5] Evaluating policies")
    run_evaluate(args)
    print("[5/5] Generating plots")
    run_plot(args)


def random_policy_fn(state: State, valid_actions: Sequence[int]) -> int:
    """Random baseline policy over valid actions."""

    return random.choice(list(valid_actions))


def greedy_heuristic_policy_fn(state: State, valid_actions: Sequence[int]) -> int:
    """Simple non-RL heuristic that targets fish by value-to-risk score."""

    x, d, oxygen, global_time, weight, fish_mask = state

    if 4 in valid_actions:
        return 4

    candidate_fish = []
    for fish in FISH_CONFIG:
        fish_id = int(fish["id"])
        if not (fish_mask & (1 << fish_id)):
            continue
        fx, fd = fish["position"]
        distance = abs(fx - x) + abs(fd - d)
        risk_cost = distance + fish["health"] + fish["aggression"] * fish["attack_damage"] + 1
        candidate_fish.append((fish["value"] / risk_cost, fish))

    if not candidate_fish:
        return 5 if 5 in valid_actions else valid_actions[0]

    _, target_fish = max(candidate_fish, key=lambda item: item[0])
    tx, td = target_fish["position"]
    return_cost = x + d

    if (oxygen <= return_cost + 3 or global_time <= return_cost + 3) and 5 in valid_actions:
        return 5

    movement_preferences = []
    if tx > x:
        movement_preferences.append(3)
    elif tx < x:
        movement_preferences.append(2)
    if td > d:
        movement_preferences.append(1)
    elif td < d:
        movement_preferences.append(0)

    for action in movement_preferences:
        if action in valid_actions:
            return action

    return valid_actions[0]


def make_policy_dict_fn(policy: Mapping[State, int]) -> Callable[[State, Sequence[int]], int]:
    """Create an evaluation callable from a policy dictionary."""

    return lambda state, valid_actions: greedy_action_from_policy(policy, state, valid_actions)


def make_q_table_fn(Q: Mapping[State, np.ndarray]) -> Callable[[State, Sequence[int]], int]:
    """Create an evaluation callable from a sparse Q-table."""

    return lambda state, valid_actions: greedy_action_from_q(Q, state, valid_actions)


def _available_policy_functions(seed: int) -> List[tuple[str, Callable[[State, Sequence[int]], int]]]:
    """Load trained policies and include optional baselines."""

    policies: List[tuple[str, Callable[[State, Sequence[int]], int]]] = []

    value_policy_path = RESULTS_DIR / "value_iteration_policy.pkl"
    if value_policy_path.exists():
        policies.append(("value_iteration", make_policy_dict_fn(load_pickle(value_policy_path))))

    sarsa_q_path = RESULTS_DIR / "sarsa_q.pkl"
    if sarsa_q_path.exists():
        policies.append(("sarsa", make_q_table_fn(load_pickle(sarsa_q_path))))

    q_learning_q_path = RESULTS_DIR / "q_learning_q.pkl"
    if q_learning_q_path.exists():
        policies.append(("q_learning", make_q_table_fn(load_pickle(q_learning_q_path))))

    random.seed(seed)
    policies.append(("random", random_policy_fn))
    policies.append(("greedy_heuristic", greedy_heuristic_policy_fn))
    return policies


def _plot_policy_and_trajectory_files(seed: int) -> None:
    """Create policy-map and trajectory plots for available learned methods."""

    items = [
        (
            "q_learning",
            RESULTS_DIR / "q_learning_policy.pkl",
            RESULTS_DIR / "policy_map_q_learning.png",
            RESULTS_DIR / "trajectory_q_learning.png",
        ),
        (
            "sarsa",
            RESULTS_DIR / "sarsa_policy.pkl",
            RESULTS_DIR / "policy_map_sarsa.png",
            RESULTS_DIR / "trajectory_sarsa.png",
        ),
        (
            "value_iteration",
            RESULTS_DIR / "value_iteration_policy.pkl",
            RESULTS_DIR / "policy_map_value_iteration.png",
            RESULTS_DIR / "trajectory_value_iteration.png",
        ),
    ]

    for method, policy_path, map_path, trajectory_path in items:
        if not policy_path.exists():
            continue

        policy = load_pickle(policy_path)
        env = UnderwaterFishingEnv(stochastic=True, seed=seed)
        policy_fn = make_policy_dict_fn(policy)

        plot_policy_map(env, policy, map_path, title=f"{method.replace('_', ' ').title()} Policy Map")
        rollout = rollout_episode(env, policy_fn)
        plot_trajectory(
            env,
            rollout,
            trajectory_path,
            title=f"{method.replace('_', ' ').title()} Trajectory",
        )

        if method == "q_learning":
            plot_oxygen_time_curve(
                rollout,
                RESULTS_DIR / "oxygen_time_curve_q_learning.png",
                title="Q-learning Oxygen and Time Curve",
            )


def _read_csv_if_exists(path: Path) -> List[Dict[str, Any]]:
    """Read a CSV file as dictionaries if it exists."""

    if not path.exists():
        return []
    return pd.read_csv(path).to_dict(orient="records")


def _representative_policy_states() -> List[State]:
    """Return states used for policy-map visualization."""

    states: List[State] = []
    for d in range(GRID_DEPTH):
        for x in range(GRID_WIDTH):
            if (x, d) in OBSTACLES:
                continue
            states.append((x, d, MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK))
    return states


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(description="Dave the Diver tabular RL project")
    parser.add_argument(
        "--mode",
        required=True,
        choices=[
            "value_iteration",
            "train_sarsa",
            "train_q_learning",
            "evaluate",
            "plot",
            "all",
        ],
        help="Execution mode.",
    )
    parser.add_argument("--episodes", type=int, default=TRAIN_EPISODES)
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--alpha", type=float, default=ALPHA)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--epsilon", type=float, default=INITIAL_EPSILON)
    parser.add_argument("--epsilon_decay", type=float, default=EPSILON_DECAY)
    parser.add_argument("--min_epsilon", type=float, default=MIN_EPSILON)
    parser.add_argument("--theta", type=float, default=VALUE_ITERATION_THETA)
    parser.add_argument("--max_iterations", type=int, default=VALUE_ITERATION_MAX_ITERATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no_progress",
        dest="show_progress",
        action="store_false",
        help="Disable tqdm progress bars.",
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main() -> None:
    """Run the selected CLI mode."""

    args = parse_args()
    set_global_seeds(args.seed)
    ensure_dir(RESULTS_DIR)

    if args.mode == "value_iteration":
        run_value_iteration(args)
    elif args.mode == "train_sarsa":
        run_train_sarsa(args)
    elif args.mode == "train_q_learning":
        run_train_q_learning(args)
    elif args.mode == "evaluate":
        run_evaluate(args)
    elif args.mode == "plot":
        run_plot(args)
    elif args.mode == "all":
        run_all(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
