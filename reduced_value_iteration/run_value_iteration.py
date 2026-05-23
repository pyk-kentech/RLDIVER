"""Run and evaluate the standalone reduced Value Iteration baseline."""

from __future__ import annotations

import argparse
import csv
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from config import ACTIONS, EVAL_EPISODES, RANDOM_SEED
from env import ReducedUnderwaterFishingEnv
from value_iteration import value_iteration

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def save_pickle(obj: Any, path: Path) -> None:
    """Save a Python object with pickle."""

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as file:
        pickle.dump(obj, file, protocol=pickle.HIGHEST_PROTOCOL)


def save_csv(rows: List[Dict[str, Any]], path: Path) -> None:
    """Save dictionaries to CSV."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def greedy_policy_action(policy, state, valid_actions):
    """Return the policy action if available, otherwise a valid fallback."""

    action = policy.get(state)
    if action in valid_actions:
        return action
    return valid_actions[0]


def evaluate_policy(policy, episodes: int = EVAL_EPISODES, seed: int = RANDOM_SEED):
    """Evaluate the reduced Value Iteration policy by stochastic rollouts."""

    rewards = []
    deaths = []
    steps_list = []
    fish_values = []
    all_fish = []

    for episode in range(episodes):
        env = ReducedUnderwaterFishingEnv(stochastic=True, seed=seed + episode)
        state = env.reset()
        done = False
        total_reward = 0.0
        steps = 0
        fish_value = 0.0
        terminal_reason = None

        while not done:
            valid_actions = env.get_valid_actions(state)
            if not valid_actions:
                break
            action = greedy_policy_action(policy, state, valid_actions)
            state, reward, done, info = env.step(action)
            total_reward += reward
            steps += 1
            caught_fish_id = info.get("caught_fish_id")
            if caught_fish_id is not None:
                fish_value += env.fish_by_id[caught_fish_id]["value"]
            terminal_reason = info.get("terminal_reason")

        rewards.append(total_reward)
        deaths.append(1 if terminal_reason in {"death", "failed_surface"} else 0)
        steps_list.append(steps)
        fish_values.append(fish_value)
        all_fish.append(1 if terminal_reason == "all_fish_caught" else 0)

    return {
        "method": "reduced_value_iteration",
        "avg_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "survival_rate": float(1.0 - np.mean(deaths)),
        "death_rate": float(np.mean(deaths)),
        "avg_steps": float(np.mean(steps_list)),
        "avg_fish_value": float(np.mean(fish_values)),
        "all_fish_caught_rate": float(np.mean(all_fish)),
    }


def main() -> None:
    """Run reduced Value Iteration and write result files."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--theta", type=float, default=1e-6)
    parser.add_argument("--max_iterations", type=int, default=1000)
    parser.add_argument("--eval_episodes", type=int, default=EVAL_EPISODES)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    env = ReducedUnderwaterFishingEnv(stochastic=False, seed=args.seed)
    print("Reduced Value Iteration map:")
    print(env.render_ascii())
    print(f"Actions: {ACTIONS}")

    V, policy, states, iteration_log = value_iteration(
        env,
        theta=args.theta,
        max_iterations=args.max_iterations,
        show_progress=not args.no_progress,
    )
    eval_row = evaluate_policy(policy, episodes=args.eval_episodes, seed=args.seed)

    save_pickle(V, RESULTS_DIR / "reduced_value_iteration_values.pkl")
    save_pickle(policy, RESULTS_DIR / "reduced_value_iteration_policy.pkl")
    save_csv(iteration_log, RESULTS_DIR / "reduced_value_iteration_log.csv")
    save_csv([eval_row], RESULTS_DIR / "reduced_value_iteration_eval.csv")

    print(f"Reachable states: {len(states)}")
    print(f"Iterations: {len(iteration_log)}")
    print(f"Evaluation avg_reward: {eval_row['avg_reward']:.2f}")
    print(f"Saved reduced VI outputs to {RESULTS_DIR}")


if __name__ == "__main__":
    main()
