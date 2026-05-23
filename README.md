# RLDIVER: 8x8 Tabular Diver MDP

This project implements an oxygen-limited underwater fishing problem as a
finite Markov Decision Process. The current version uses a single tractable
8x8 tabular environment so Value Iteration, SARSA, and Q-learning can all be
run on the same MDP.

No function approximation is used. The project uses explicit tabular state
values, tabular action values, sparse dictionaries, and exact transition
enumeration for Value Iteration.

## File Structure

```text
RLDIVER/
  README.md
  requirements.txt
  config.py
  env.py
  value_iteration.py
  main.py
  run_value_iteration.py
  algorithms/
    __init__.py
    q_learning.py
    sarsa.py
  results/
```

`algorithms/q_learning.py` and `algorithms/sarsa.py` contain the model-free
control implementations. `value_iteration.py` contains the Dynamic Programming
baseline.

## MDP

State:

```text
(x, d, oxygen, global_time, weight, fish_mask)
```

Actions:

```text
up, down, left, right, catch, surface
```

Environment size:

```text
Grid: 8 x 8
Fish count: 4
Fish mask size: 16
Oxygen: 34
Global time: 80
```

Current map:

```text
 D  .  .  .  .  .  .  .
 #  #  .  #  .  #  #  .
 .  . F0  #  .  . F1  .
 .  #  #  #  #  .  #  .
 .  .  .  .  .  .  #  .
 #  #  .  #  . F2  .  .
 .  .  .  #  #  #  .  #
 . F3  .  .  .  .  .  .
```

Fish roles:

- `F0`: safe early target.
- `F1`: right-side detour target.
- `F2`: deeper high-value target.
- `F3`: farthest high-risk, high-value target.

## Installation

```bash
pip install -r requirements.txt
```

## Run Everything

```bash
python main.py
```

This runs:

1. Value Iteration
2. SARSA
3. Q-learning
4. Evaluation of all three policies

Use smaller settings for a quick smoke test:

```bash
python main.py --max_iterations 1 --episodes 3 --eval_episodes 2 --no_progress
```

Run only Value Iteration:

```bash
python run_value_iteration.py
```

Progress bars are enabled by default. Use `--no_progress` for cleaner logs.

## Outputs

All outputs are written to:

```text
results/
```

Main output files:

```text
reduced_value_iteration_values.pkl
reduced_value_iteration_policy.pkl
reduced_value_iteration_log.csv
reduced_sarsa_q.pkl
reduced_sarsa_training_log.csv
reduced_q_learning_q.pkl
reduced_q_learning_training_log.csv
reduced_evaluation_summary.csv
```

## Reference Result

On the current 8x8 MDP, one representative run produced:

```text
Reachable states: 1,320,682
Value Iteration iterations: 57
```

This version is intentionally smaller than the original full 11x11 design so
exact tabular DP can be completed and compared fairly with SARSA and Q-learning
on the same environment.
