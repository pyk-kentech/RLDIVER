# Dave the Diver Tabular Reinforcement Learning

## Project Description

This project models oxygen-limited underwater fishing as a finite Markov Decision Process. The agent controls Dave, a diver who moves through a discrete ocean grid, catches fish, manages oxygen and global mission time, handles carried weight, and decides when to return safely to the surface.

The implementation uses only tabular Dynamic Programming and tabular Reinforcement Learning. It does not use neural networks, linear function approximation, deep RL, Gymnasium, Stable-Baselines, PyTorch, or any external RL framework.

## MDP Formulation

The state is:

```text
(x, d, oxygen, global_time, weight, fish_mask)
```

where `x` is horizontal position, `d` is depth, `oxygen` is the remaining oxygen for the current dive, `global_time` is the remaining mission time, `weight` is carried fish weight, and `fish_mask` is a six-bit integer showing which fish remain available.

The action space is:

```text
up, down, left, right, catch, surface
```

Movement consumes oxygen and time. Invalid movement and invalid catch actions keep Dave in place but still consume one oxygen unit, one time unit, and receive a penalty. Fish catches give fish value as reward and may trigger stochastic oxygen damage. The `surface` action returns Dave to base if enough oxygen and time remain.

The reward structure includes movement cost, invalid action penalty, fish value, surface bonus, death penalty, and underwater timeout penalty.

## Installation

Create and activate a Python environment, then install dependencies:

```bash
pip install -r requirements.txt
```

The project requires Python with `numpy`, `matplotlib`, `pandas`, and `tqdm`.

## Run Instructions

Run commands from the `RLDIVER` directory:

```bash
python main.py --mode value_iteration
python main.py --mode train_sarsa
python main.py --mode train_q_learning
python main.py --mode evaluate
python main.py --mode plot
python main.py --mode all
```

Useful optional arguments:

```bash
python main.py --mode train_q_learning --episodes 10000
python main.py --mode evaluate --eval_episodes 500
python main.py --mode value_iteration --max_iterations 1000 --theta 1e-6
python main.py --mode all --skip_value_iteration
python main.py --mode value_iteration --max_reachable_states 0
python main.py --mode all --no_progress
```

Progress bars are enabled by default. Use `--no_progress` when running in a
log file or non-interactive environment.

Full Value Iteration can be memory intensive on the full tabular state space.
By default, reachable-state discovery has a safety cap. For normal experiments,
use `--mode all --skip_value_iteration` and compare SARSA and Q-learning. Use
`--max_reachable_states 0` only if the machine has enough memory for full
reachable-state BFS.

## Output Files

Outputs are written to the `results/` directory.

Main model files:

```text
value_iteration_values.pkl
value_iteration_policy.pkl
sarsa_q.pkl
sarsa_policy.pkl
q_learning_q.pkl
q_learning_policy.pkl
```

CSV logs and summaries:

```text
value_iteration_log.csv
sarsa_training_log.csv
q_learning_training_log.csv
evaluation_summary.csv
```

Plots:

```text
q_learning_reward_curve.png
sarsa_reward_curve.png
reward_curve_comparison.png
survival_rate_curve.png
death_rate_curve.png
policy_map_q_learning.png
policy_map_sarsa.png
policy_map_value_iteration.png
trajectory_q_learning.png
trajectory_sarsa.png
trajectory_value_iteration.png
oxygen_time_curve_q_learning.png
```

## Algorithms Implemented

The project implements three core tabular methods:

- Value Iteration as the model-based Dynamic Programming baseline.
- SARSA as the on-policy model-free control baseline.
- Q-learning as the main off-policy model-free control method.

The Q-tables and V-tables are sparse dictionaries. Value Iteration discovers reachable states using BFS instead of allocating the full theoretical state space.

## Result Summary Placeholder

After running training and evaluation, fill this section with values from `results/evaluation_summary.csv`.

Expected qualitative behavior:

- Learned policies should outperform random behavior.
- SARSA may learn a safer policy because it is on-policy.
- Q-learning may learn a more reward-seeking policy because it bootstraps from greedy targets.
- Value Iteration should provide a strong model-based reference when reachable-state computation is tractable.

## Notes on Limitations

The environment is intentionally discretized for tabular RL. Movement, oxygen, time, fish positions, and attacks are simplified compared with a continuous real game environment.

Value Iteration can be computationally expensive because the reachable state space includes position, oxygen, global time, carried weight, and fish mask. If full Value Iteration is too slow, use the reduced configuration described in the project specification and clearly report that choice.

The policy maps use a representative fixed state with full oxygen, full global time, zero carried weight, and all fish available. This is useful for visualization but does not show every possible resource condition.
