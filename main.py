"""Command-line runner for the 8x8 tabular diver MDP.

Current environment summary:
    Dave starts at the surface base `(0, 0)` with 34 oxygen units and
    80 global time units. The map is an 8x8 maze-like ocean. Obstacles force
    route planning, and each fish is placed to create a different decision:

    D  .  .  .  .  .  .  .
    #  #  .  #  .  #  #  .
    .  . F0  #  .  . F1  .
    .  #  #  #  #  .  #  .
    .  .  .  .  .  .  #  .
    #  #  .  #  . F2  .  .
    .  .  .  #  #  #  .  #
    . F3  .  .  .  .  .  .

    F0 Training Reef Fish at (2, 2): safe early target.
    F1 Cave Bass at (6, 2): medium-value right-side detour.
    F2 Longfin Snapper at (5, 5): deeper high-value target.
    F3 Reduced Abyss Shark at (1, 7): farthest high-risk target.

The state is `(x, d, oxygen, global_time, weight, fish_mask)`.
All algorithms are tabular and use sparse dictionaries. No function
approximation is used.
"""

from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence

import numpy as np

from algorithms import greedy_action_from_q, train_q_learning, train_sarsa
from config import (
    EVAL_EPISODES,
    FISH_CONFIG,
    GAMMA,
    RANDOM_SEED,
    VALUE_ITERATION_MAX_ITERATIONS,
    VALUE_ITERATION_THETA,
)
from env import ReducedUnderwaterFishingEnv, State
from value_iteration import value_iteration

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def policy_action(policy: Mapping[State, int], state: State, valid_actions: Sequence[int]) -> int:
    """Return a policy action or a valid fallback."""

    action = policy.get(state)
    if action in valid_actions:
        return int(action)
    return int(valid_actions[0])


def evaluate_policy(
    method: str,
    policy_fn: Callable[[State, Sequence[int]], int],
    episodes: int,
    seed: int,
) -> Dict[str, Any]:
    """Evaluate a policy on the stochastic MDP and return one CSV-ready row."""

    rewards = []
    deaths = []
    steps_list = []
    fish_values = []
    fish_counts = []
    remaining_oxygen = []
    remaining_time = []
    all_fish = []

    for episode in range(episodes):
        env = ReducedUnderwaterFishingEnv(stochastic=True, seed=seed + episode)
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        fish_value = 0.0
        fish_count = 0
        terminal_reason = None

        while not done:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break

            action = policy_fn(state, valid_actions)
            if action not in valid_actions:
                action = valid_actions[0]

            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1

            caught_fish_id = info.get("caught_fish_id")
            if caught_fish_id is not None:
                fish_value += env.fish_by_id[caught_fish_id]["value"]
                fish_count += 1
            terminal_reason = info.get("terminal_reason")

        death = terminal_reason in {"death", "failed_surface"}
        rewards.append(total_reward)
        deaths.append(int(death))
        steps_list.append(steps)
        fish_values.append(fish_value)
        fish_counts.append(fish_count)
        remaining_oxygen.append(state[2])
        remaining_time.append(state[3])
        all_fish.append(1 if terminal_reason == "all_fish_caught" else 0)

    return {
        "method": method,
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "survival_rate": float(1.0 - np.mean(deaths)),
        "death_rate": float(np.mean(deaths)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_fish_value": float(np.mean(fish_values)),
        "avg_num_fish_caught": float(np.mean(fish_counts)),
        "avg_remaining_oxygen": float(np.mean(remaining_oxygen)),
        "avg_remaining_time": float(np.mean(remaining_time)),
        "all_fish_caught_rate": float(np.mean(all_fish)),
    }


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
    print(
        f"  theta={args.theta}, max_iterations={args.max_iterations}, "
        f"progress={'on' if args.show_progress else 'off'}"
    )
    env = ReducedUnderwaterFishingEnv(stochastic=False, seed=args.seed)
    V, policy, states, iteration_log = value_iteration(
        env,
        theta=args.theta,
        max_iterations=args.max_iterations,
        show_progress=args.show_progress,
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
        args.show_progress,
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
        args.show_progress,
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
            "show_map",
        ],
        default="all",
        help="Which part of the experiment to run. Default: all.",
    )
    parser.add_argument("--episodes", type=int, default=10000)
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--alpha", type=float, default=0.10)
    parser.add_argument("--gamma", type=float, default=GAMMA)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon_decay", type=float, default=0.995)
    parser.add_argument("--min_epsilon", type=float, default=0.05)
    parser.add_argument("--theta", type=float, default=VALUE_ITERATION_THETA)
    parser.add_argument("--max_iterations", type=int, default=VALUE_ITERATION_MAX_ITERATIONS)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument(
        "--no_progress",
        dest="show_progress",
        action="store_false",
        help="Disable tqdm progress bars for cleaner logs.",
    )
    parser.set_defaults(show_progress=True)
    return parser.parse_args()


def main() -> None:
    """Run the selected experiment mode."""

    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    if args.mode == "show_map":
        print_environment_summary()
    elif args.mode == "value_iteration":
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
    elif args.mode == "all":
        run_all(args)
    else:
        raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    main()
