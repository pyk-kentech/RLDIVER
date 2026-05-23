# Reduced Value Iteration Baseline

This folder contains a standalone reduced MDP for the Value Iteration baseline.
It is intentionally isolated from the full SARSA/Q-learning project so it can be
deleted without affecting the main implementation.

The reduced environment keeps the same state representation and transition
logic:

```text
(x, d, oxygen, global_time, weight, fish_mask)
```

It reduces the problem size to make exact tabular DP tractable:

- Grid: `6 x 6`
- Fish count: `4`
- Fish mask size: `16`
- Oxygen: `28`
- Global time: `65`

The fish placement is designed so every fish has a purpose:

- `F0`: safe early fish near the entrance.
- `F1`: medium fish on a right-side detour.
- `F2`: deeper high-value fish on the lower-right route.
- `F3`: farthest fish, valuable but costly to reach and return from.

Run:

```bash
cd reduced_value_iteration
python run_value_iteration.py
```

Outputs:

```text
results/reduced_value_iteration_values.pkl
results/reduced_value_iteration_policy.pkl
results/reduced_value_iteration_log.csv
results/reduced_value_iteration_eval.csv
```

Use this result as the DP baseline in the report, while SARSA and Q-learning
remain evaluated on the full environment.

Reference run on the current intermediate reduced MDP:

```text
Reachable states: 510429
Iterations to convergence: 51
Evaluation avg_reward: about 345.6
```
