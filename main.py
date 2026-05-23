"""Run VI, SARSA, and Q-learning on the 8x8 tabular diver MDP."""

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
    """Evaluate a policy on the reduced stochastic MDP."""

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


def main() -> None:
    """Run all reduced experiments."""

    parser = argparse.ArgumentParser()
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
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)
    show_progress = not args.no_progress

    env = ReducedUnderwaterFishingEnv(stochastic=False, seed=args.seed)
    print("Reduced experiment map:")
    print(env.render_ascii())

    print("[1/4] Reduced Value Iteration")
    V, vi_policy, states, vi_log = value_iteration(
        env,
        theta=args.theta,
        max_iterations=args.max_iterations,
        show_progress=show_progress,
    )
    save_pickle(V, RESULTS_DIR / "reduced_value_iteration_values.pkl")
    save_pickle(vi_policy, RESULTS_DIR / "reduced_value_iteration_policy.pkl")
    save_csv(vi_log, RESULTS_DIR / "reduced_value_iteration_log.csv")
    print(f"Reduced VI states={len(states)}, iterations={len(vi_log)}")

    print("[2/4] Reduced SARSA")
    sarsa_q, sarsa_log = train_sarsa(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_decay,
        args.min_epsilon,
        args.seed,
        show_progress,
    )
    save_pickle(sarsa_q, RESULTS_DIR / "reduced_sarsa_q.pkl")
    save_csv(sarsa_log, RESULTS_DIR / "reduced_sarsa_training_log.csv")

    print("[3/4] Reduced Q-learning")
    q_learning_q, q_learning_log = train_q_learning(
        args.episodes,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_decay,
        args.min_epsilon,
        args.seed,
        show_progress,
    )
    save_pickle(q_learning_q, RESULTS_DIR / "reduced_q_learning_q.pkl")
    save_csv(q_learning_log, RESULTS_DIR / "reduced_q_learning_training_log.csv")

    print("[4/4] Reduced evaluation")
    rows = [
        evaluate_policy(
            "reduced_value_iteration",
            lambda state, valid: policy_action(vi_policy, state, valid),
            args.eval_episodes,
            args.seed,
        ),
        evaluate_policy(
            "reduced_sarsa",
            lambda state, valid: greedy_action_from_q(sarsa_q, state, valid),
            args.eval_episodes,
            args.seed,
        ),
        evaluate_policy(
            "reduced_q_learning",
            lambda state, valid: greedy_action_from_q(q_learning_q, state, valid),
            args.eval_episodes,
            args.seed,
        ),
    ]
    save_csv(rows, RESULTS_DIR / "reduced_evaluation_summary.csv")
    for row in rows:
        print(f"{row['method']}: avg_reward={row['avg_reward']:.2f}")
    print(f"Saved reduced all-method outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
