"""Command-line runner for the 8x8 tabular diver MDP."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import numpy as np

from algorithms import greedy_action_from_q, train_q_learning, train_sarsa
from config import (
    EVAL_EPISODES,
    FISH_CONFIG,
    GAMMA,
    GRID_DEPTH,
    GRID_WIDTH,
    INITIAL_FISH_MASK,
    MAX_GLOBAL_TIME,
    MAX_OXYGEN,
    RANDOM_SEED,
    VALUE_ITERATION_MAX_ITERATIONS,
    VALUE_ITERATION_THETA,
)
from env import ReducedUnderwaterFishingEnv, State
from evaluation import evaluate_policy, rollout_episode
from plotting import (
    plot_death_rate_curve,
    plot_oxygen_time_curve,
    plot_policy_map,
    plot_reward_curve_comparison,
    plot_survival_rate_curve,
    plot_training_reward_curve,
    plot_trajectory,
    read_csv_rows,
)
from value_iteration import value_iteration

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def policy_action(policy: Mapping[State, int], state: State, valid_actions: Sequence[int]) -> int:
    """Return a policy action or a valid fallback."""

    action = policy.get(state)
    if action in valid_actions:
        return int(action)
    return int(valid_actions[0])


def save_pickle(obj: Any, path: Path) -> None:
    """Save an object with pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: Path) -> Any:
    """Load an object saved with pickle."""

    with path.open("rb") as file:
        return pickle.load(file)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_environment_summary() -> None:
    """Print the current map and fish table in a human-readable form."""

    env = ReducedUnderwaterFishingEnv(stochastic=False, seed=RANDOM_SEED)
    print("Environment: 8x8 oxygen-limited diver MDP")
    print("Map legend: D=Dave/start, #=obstacle, F0-F3=fish")
    print(env.render_ascii())
    print()
    print("Fish table:")
    for fish in FISH_CONFIG:
        print(
            f"  F{fish['id']} {fish['name']}: pos={fish['position']}, "
            f"health={fish['health']}, aggression={fish['aggression']}, "
            f"damage={fish['attack_damage']}, weight={fish['weight']}, "
            f"value={fish['value']}"
        )
    print()


def run_value_iteration(args: argparse.Namespace) -> Dict[str, Any]:
    """Run Value Iteration and save its artifacts."""

    print("Step: Value Iteration")
    print(f"  theta={args.theta}, max_iterations={args.max_iterations}")
    env = ReducedUnderwaterFishingEnv(stochastic=False, seed=args.seed)
    V, policy, states, iteration_log = value_iteration(
        env,
        theta=args.theta,
        max_iterations=args.max_iterations,
        show_progress=True,
    )

    save_pickle(V, RESULTS_DIR / "reduced_value_iteration_values.pkl")
    save_pickle(policy, RESULTS_DIR / "reduced_value_iteration_policy.pkl")
    save_csv(iteration_log, RESULTS_DIR / "reduced_value_iteration_log.csv")

    print(f"  reachable_states={len(states)}")
    print(f"  iterations={len(iteration_log)}")
    print("  saved: results/reduced_value_iteration_values.pkl")
    print("  saved: results/reduced_value_iteration_policy.pkl")
    print("  saved: results/reduced_value_iteration_log.csv")
    print()
    return {"V": V, "policy": policy, "states": states, "log": iteration_log}


def run_sarsa(args: argparse.Namespace) -> Dict[str, Any]:
    """Train SARSA and save its artifacts."""

    print("Step: SARSA")
    print(
        f"  episodes={args.episodes}, alpha={args.alpha}, gamma={args.gamma}, "
        f"epsilon={args.epsilon}, decay={args.epsilon_decay}"
    )
    Q, log = train_sarsa(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_decay,
        args.min_epsilon,
        args.seed,
        True,
    )
    save_pickle(Q, RESULTS_DIR / "reduced_sarsa_q.pkl")
    save_csv(log, RESULTS_DIR / "reduced_sarsa_training_log.csv")
    print(f"  visited_q_states={len(Q)}")
    print("  saved: results/reduced_sarsa_q.pkl")
    print("  saved: results/reduced_sarsa_training_log.csv")
    print()
    return {"Q": Q, "log": log}


def run_q_learning(args: argparse.Namespace) -> Dict[str, Any]:
    """Train Q-learning and save its artifacts."""

    print("Step: Q-learning")
    print(
        f"  episodes={args.episodes}, alpha={args.alpha}, gamma={args.gamma}, "
        f"epsilon={args.epsilon}, decay={args.epsilon_decay}"
    )
    Q, log = train_q_learning(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_decay,
        args.min_epsilon,
        args.seed,
        True,
    )
    save_pickle(Q, RESULTS_DIR / "reduced_q_learning_q.pkl")
    save_csv(log, RESULTS_DIR / "reduced_q_learning_training_log.csv")
    print(f"  visited_q_states={len(Q)}")
    print("  saved: results/reduced_q_learning_q.pkl")
    print("  saved: results/reduced_q_learning_training_log.csv")
    print()
    return {"Q": Q, "log": log}


def run_evaluation(
    args: argparse.Namespace,
    vi_policy: Mapping[State, int] | None = None,
    sarsa_q: Any | None = None,
    q_learning_q: Any | None = None,
) -> List[Dict[str, Any]]:
    """Evaluate available policies and save the summary CSV."""

    print("Step: Evaluation")
    print(f"  episodes={args.eval_episodes}")

    rows = []

    if vi_policy is None and (RESULTS_DIR / "reduced_value_iteration_policy.pkl").exists():
        vi_policy = load_pickle(RESULTS_DIR / "reduced_value_iteration_policy.pkl")
    if sarsa_q is None and (RESULTS_DIR / "reduced_sarsa_q.pkl").exists():
        sarsa_q = load_pickle(RESULTS_DIR / "reduced_sarsa_q.pkl")
    if q_learning_q is None and (RESULTS_DIR / "reduced_q_learning_q.pkl").exists():
        q_learning_q = load_pickle(RESULTS_DIR / "reduced_q_learning_q.pkl")

    if vi_policy is not None:
        rows.append(
            evaluate_policy(
                "reduced_value_iteration",
                lambda state, valid: policy_action(vi_policy, state, valid),
                args.eval_episodes,
                args.seed,
            )
        )
    else:
        print("  skipped VI policy: results/reduced_value_iteration_policy.pkl not found")

    if sarsa_q is not None:
        rows.append(
            evaluate_policy(
                "reduced_sarsa",
                lambda state, valid: greedy_action_from_q(sarsa_q, state, valid),
                args.eval_episodes,
                args.seed,
            )
        )
    else:
        print("  skipped SARSA policy: results/reduced_sarsa_q.pkl not found")

    if q_learning_q is not None:
        rows.append(
            evaluate_policy(
                "reduced_q_learning",
                lambda state, valid: greedy_action_from_q(q_learning_q, state, valid),
                args.eval_episodes,
                args.seed,
            )
        )
    else:
        print("  skipped Q-learning policy: results/reduced_q_learning_q.pkl not found")

    if rows:
        save_csv(rows, RESULTS_DIR / "reduced_evaluation_summary.csv")
        print("  saved: results/reduced_evaluation_summary.csv")
        print("  summary:")
        for row in rows:
            print(
                f"    {row['method']}: avg_reward={row['avg_reward']:.2f}, "
                f"survival={row['survival_rate']:.2f}, "
                f"fish_value={row['avg_fish_value']:.2f}, "
                f"all_fish={row['all_fish_caught_rate']:.2f}"
            )
    else:
        print("  no policies were available to evaluate")
    print()
    return rows


def run_plot(args: argparse.Namespace) -> None:
    """Generate reward curves, survival/death curves, maps, and trajectories."""

    print("Step: Plotting")
    sarsa_log = read_csv_rows(RESULTS_DIR / "reduced_sarsa_training_log.csv")
    q_learning_log = read_csv_rows(RESULTS_DIR / "reduced_q_learning_training_log.csv")

    if sarsa_log:
        plot_training_reward_curve(
            sarsa_log,
            RESULTS_DIR / "sarsa_reward_curve.png",
            "SARSA Reward Curve",
        )
        print("  saved: results/sarsa_reward_curve.png")
    else:
        print("  skipped SARSA reward curve: training log not found")

    if q_learning_log:
        plot_training_reward_curve(
            q_learning_log,
            RESULTS_DIR / "q_learning_reward_curve.png",
            "Q-learning Reward Curve",
        )
        print("  saved: results/q_learning_reward_curve.png")
    else:
        print("  skipped Q-learning reward curve: training log not found")

    if sarsa_log and q_learning_log:
        plot_reward_curve_comparison(
            sarsa_log,
            q_learning_log,
            RESULTS_DIR / "reward_curve_comparison.png",
        )
        plot_survival_rate_curve(
            sarsa_log,
            q_learning_log,
            RESULTS_DIR / "survival_rate_curve.png",
        )
        plot_death_rate_curve(
            sarsa_log,
            q_learning_log,
            RESULTS_DIR / "death_rate_curve.png",
        )
        print("  saved: results/reward_curve_comparison.png")
        print("  saved: results/survival_rate_curve.png")
        print("  saved: results/death_rate_curve.png")

    vi_policy = None
    sarsa_q = None
    q_learning_q = None
    if (RESULTS_DIR / "reduced_value_iteration_policy.pkl").exists():
        vi_policy = load_pickle(RESULTS_DIR / "reduced_value_iteration_policy.pkl")
        plot_policy_map(
            vi_policy,
            RESULTS_DIR / "policy_map_value_iteration.png",
            "Value Iteration Policy Map",
        )
        print("  saved: results/policy_map_value_iteration.png")
    if (RESULTS_DIR / "reduced_sarsa_q.pkl").exists():
        sarsa_q = load_pickle(RESULTS_DIR / "reduced_sarsa_q.pkl")
        sarsa_policy = _representative_policy_from_q(sarsa_q)
        plot_policy_map(sarsa_policy, RESULTS_DIR / "policy_map_sarsa.png", "SARSA Policy Map")
        print("  saved: results/policy_map_sarsa.png")
    if (RESULTS_DIR / "reduced_q_learning_q.pkl").exists():
        q_learning_q = load_pickle(RESULTS_DIR / "reduced_q_learning_q.pkl")
        q_policy = _representative_policy_from_q(q_learning_q)
        plot_policy_map(q_policy, RESULTS_DIR / "policy_map_q_learning.png", "Q-learning Policy Map")
        print("  saved: results/policy_map_q_learning.png")

    if vi_policy is not None:
        rollout = rollout_episode(lambda state, valid: policy_action(vi_policy, state, valid), seed=args.seed)
        plot_trajectory(rollout, RESULTS_DIR / "trajectory_value_iteration.png", "Value Iteration Trajectory")
        print("  saved: results/trajectory_value_iteration.png")
    if sarsa_q is not None:
        rollout = rollout_episode(lambda state, valid: greedy_action_from_q(sarsa_q, state, valid), seed=args.seed)
        plot_trajectory(rollout, RESULTS_DIR / "trajectory_sarsa.png", "SARSA Trajectory")
        print("  saved: results/trajectory_sarsa.png")
    if q_learning_q is not None:
        rollout = rollout_episode(lambda state, valid: greedy_action_from_q(q_learning_q, state, valid), seed=args.seed)
        plot_trajectory(rollout, RESULTS_DIR / "trajectory_q_learning.png", "Q-learning Trajectory")
        plot_oxygen_time_curve(rollout, RESULTS_DIR / "oxygen_time_curve_q_learning.png")
        print("  saved: results/trajectory_q_learning.png")
        print("  saved: results/oxygen_time_curve_q_learning.png")
    print()


def run_all(args: argparse.Namespace) -> None:
    """Run VI, SARSA, Q-learning, and evaluation."""

    print_environment_summary()
    print("Running complete experiment pipeline")
    print()
    vi_result = run_value_iteration(args)
    sarsa_result = run_sarsa(args)
    q_learning_result = run_q_learning(args)
    run_evaluation(
        args,
        vi_policy=vi_result["policy"],
        sarsa_q=sarsa_result["Q"],
        q_learning_q=q_learning_result["Q"],
    )
    run_plot(args)


def _representative_policy_from_q(Q: Any) -> Dict[State, int]:
    """Build a representative full-resource policy map from a Q-table."""

    policy: Dict[State, int] = {}
    env = ReducedUnderwaterFishingEnv(stochastic=False, seed=RANDOM_SEED)
    for d in range(GRID_DEPTH):
        for x in range(GRID_WIDTH):
            state = (x, d, MAX_OXYGEN, MAX_GLOBAL_TIME, 0, INITIAL_FISH_MASK)
            valid_actions = env.get_valid_actions(state)
            if valid_actions:
                policy[state] = greedy_action_from_q(Q, state, valid_actions)
    return policy


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""

    parser = argparse.ArgumentParser(
        description="Run tabular RL experiments on the 8x8 diver MDP."
    )
    parser.add_argument(
        "--mode",
        choices=[
            "all",
            "value_iteration",
            "train_sarsa",
            "train_q_learning",
            "evaluate",
            "plot",
        ],
        default="all",
        help="Which part of the experiment to run. Default: all.",
    )
    parser.add_argument("--episodes", type=int, default=500000)
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--min_epsilon", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=VALUE_ITERATION_THETA)
    parser.add_argument("--max_iterations", type=int, default=VALUE_ITERATION_MAX_ITERATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    return parser.parse_args()


def main() -> None:
    """Run the selected experiment mode."""

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "value_iteration":
        print_environment_summary()
        run_value_iteration(args)
    elif args.mode == "train_sarsa":
        print_environment_summary()
        run_sarsa(args)
    elif args.mode == "train_q_learning":
        print_environment_summary()
        run_q_learning(args)
    elif args.mode == "evaluate":
        print_environment_summary()
        run_evaluation(args)
    elif args.mode == "plot":
        print_environment_summary()
        run_plot(args)
    elif args.mode == "all":
        run_all(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
