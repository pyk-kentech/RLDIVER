# RLDIVER: 8x8 Tabular Diver MDP

This project implements an oxygen-limited underwater fishing problem as a
finite Markov Decision Process for **EL5001 Reinforcement Learning Term Project
01: Solve MDP with Model-free RL**. The current version uses a single tractable
8x8 tabular environment so Value Iteration, SARSA, and Q-learning can all be run
and compared on the same MDP.

No function approximation is used. The project uses explicit tabular state
values, tabular action values, sparse dictionaries, and exact transition
enumeration for Value Iteration. It does **not** use a linear model, nonlinear
model, neural network, Deep RL method, or external RL framework.

## Project Motivation

Underwater fishing is a sequential decision-making problem because each action
changes the diver's future options. Moving toward a valuable fish consumes
oxygen and time, catching fish increases carried weight, aggressive fish may
cause damage, and surfacing too late can fail the mission. The best immediate
action is therefore not always the best long-term action.

The agent is the diver. The diver must decide where to move, whether to catch a
fish, and when to surface. Reinforcement learning is suitable because the task
requires learning or computing a policy that balances reward, resource use,
risk, and long-term survival over an episode.

## File Structure

```text
RLDIVER/
  README.md
  requirements.txt
  config.py
  env.py
  value_iteration.py
  evaluation.py
  plotting.py
  main.py
  algorithms/
    __init__.py
    q_learning.py
    sarsa.py
  results/
```

`algorithms/q_learning.py` and `algorithms/sarsa.py` contain the model-free
control implementations. `value_iteration.py` contains the Dynamic Programming
baseline. `evaluation.py` and `plotting.py` generate the evaluation metrics and
figures used in the results.

## MDP Formulation

### Environment

The environment is an 8x8 underwater grid with obstacles, four fish, oxygen,
remaining global time, carried weight, and a base at `(0, 0)`. The diver starts
at the base with `34` oxygen units, `80` remaining global time units, zero
carried weight, and all four fish available.

Environment size:

```text
Grid: 8 x 8
Fish count: 4
Fish mask size: 16
Initial oxygen: 34
Initial global time: 80
Maximum episode steps: 240
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

Fish configuration:

| Fish | Name | Position | Health | Aggression | Attack Damage | Weight | Value |
|---|---|---:|---:|---:|---:|---:|---:|
| F0 | Training Reef Fish | `(2, 2)` | 2 | 0.10 | 2 | 2 | 20 |
| F1 | Cave Bass | `(6, 2)` | 3 | 0.20 | 3 | 3 | 40 |
| F2 | Longfin Snapper | `(5, 5)` | 4 | 0.30 | 5 | 4 | 70 |
| F3 | Reduced Abyss Shark | `(1, 7)` | 5 | 0.45 | 6 | 5 | 100 |

Fish roles:

- `F0`: safe early target.
- `F1`: right-side detour target.
- `F2`: deeper high-value target.
- `F3`: farthest high-risk, high-value target.

### Agent

The agent is Dave, the diver. The agent observes the full tabular state and
chooses one discrete action at each decision step.

### State

```text
(x, d, oxygen, global_time, weight, fish_mask)
```

- `x`: horizontal grid coordinate.
- `d`: depth coordinate.
- `oxygen`: remaining oxygen, clipped to `[0, 34]`.
- `global_time`: remaining global time, clipped to `[0, 80]`.
- `weight`: carried fish weight, clipped to `[0, 14]`.
- `fish_mask`: 4-bit fish availability mask. The initial mask is `0b1111`.

The state is Markovian for this simplified environment because it contains the
information needed to determine future transitions and rewards: position,
remaining resources, carried weight, and which fish are still available. No
additional history is required by the transition function.

### Actions

```text
up, down, left, right, catch, surface
```

The first four actions move the diver by one grid cell when the target cell is
valid. `catch` attempts to catch a fish at the current position. `surface`
attempts to return to the base.

### Reward

The reward function combines fish value, movement cost, invalid-action
penalties, successful surfacing, and failure penalties. Exact values are listed
in the Reward Design section.

### Transition Rules

The environment supports both sampled transitions for model-free learning and
exact transition enumeration for Value Iteration.

- Movement actions change `(x, d)` by one cell if the destination is inside the
  grid and not an obstacle.
- Normal movement costs `1` oxygen and `1` remaining global time.
- If carried weight is greater than `7`, movement costs `2` oxygen and `2`
  remaining global time.
- Invalid movement leaves the diver in place but still costs `1` oxygen and `1`
  remaining global time and gives a `-10` penalty.
- `catch` is valid only when an available fish is at the current position.
- A successful catch removes the fish from `fish_mask`, increases carried
  weight by the fish weight, decreases oxygen by fish health plus possible
  attack damage, decreases remaining global time by fish health, and gives the
  fish value as reward.
- If the fish attacks, the attack occurs with the fish's aggression probability.
  Exact transition enumeration returns both outcomes with probabilities
  `1 - aggression` and `aggression`.
- Invalid catch leaves the diver in place but costs `1` oxygen and `1`
  remaining global time and gives a `-10` penalty.
- `surface` returns the diver to `(0, 0)` when enough oxygen and time remain.
  Its return cost is `x + d`.
- Successful surfacing with positive carried weight gives a `50` surface bonus,
  resets oxygen to `34`, and resets carried weight to `0`.
- Surfacing at `(0, 0)` with zero carried weight is treated as invalid farming:
  it costs `1` oxygen and `1` remaining global time and gives a `-10` penalty.

### Episode Termination Conditions

An episode terminates when:

- oxygen reaches `0`, causing death;
- remaining global time reaches `0` away from the base;
- remaining global time reaches `0` at the base;
- all fish have been caught and the diver is at the base;
- a surface attempt fails because not enough oxygen or time remains;
- `240` episode steps are reached.

## Reward Design

The reward design encourages high-value fish collection while penalizing waste,
unsafe routing, and failure.

| Event | Reward or Penalty |
|---|---:|
| Valid movement | `-1` |
| Invalid movement | `-10` |
| Invalid catch | `-10` |
| Invalid surface at base with zero weight | `-10` |
| Catch fish | fish value times `1.0` |
| Successful surface with positive carried weight | `50` |
| Oxygen death | `-500` |
| Failed surface | `-500` |
| Timeout underwater | additional `-100` |

Movement has a small negative reward, so shorter efficient routes are preferred.
Fish rewards motivate collection, while aggression and oxygen costs make deeper
or more dangerous fish a strategic risk. The surface bonus rewards returning
safely with fish instead of only catching fish and dying underwater. Invalid
actions consume oxygen and time, preventing an agent from exploiting repeated
invalid actions to avoid progress.

## Algorithms Compared

### Value Iteration

Value Iteration is used as the Dynamic Programming baseline. It uses exact
transition enumeration from `get_transitions(state, action)` and performs
Bellman optimality backups over states discovered by reachable-state BFS. The
state table is sparse and stored with `defaultdict(float)`.

Value Iteration is model-based: it uses the transition model directly and does
not learn from sampled episodes.

### SARSA

SARSA is an on-policy model-free control method. It learns action values from
sampled episodes generated by its own epsilon-greedy behavior policy. The update
uses the next action actually selected by the current behavior policy.

The Q-table is sparse and stored with `defaultdict(new_q_values)`, where
`new_q_values()` is a pickle-safe factory for zero-valued action rows.

### Q-learning

Q-learning is an off-policy model-free control method. It also learns from
sampled episodes, but its target uses the greedy next-action value even while
the behavior policy explores with epsilon-greedy action selection.

Both SARSA and Q-learning use only valid actions for epsilon-greedy selection,
which improves learning efficiency without using function approximation.

## Hyperparameters

Important defaults are defined in `config.py` and `main.py`.

| Hyperparameter | Default |
|---|---:|
| Discount factor `gamma` | `0.95` |
| SARSA/Q-learning learning rate `alpha` | `0.10` |
| Initial epsilon | `1.0` |
| Epsilon decay | `0.995` |
| Minimum epsilon | `0.05` |
| Training episodes | `10000` |
| Evaluation episodes | `200` |
| Value Iteration theta | `1e-6` |
| Value Iteration max iterations | `1000` |
| Random seed | `42` |

These values can be overridden from the command line with options such as
`--episodes`, `--eval_episodes`, `--alpha`, `--gamma`, `--epsilon`,
`--epsilon_decay`, `--min_epsilon`, `--theta`, `--max_iterations`, and `--seed`.

## Evaluation Metrics

The evaluation code runs policies in the stochastic environment and writes one
CSV-ready row per method. Metrics include:

- `avg_reward`: average return over evaluation episodes.
- `std_reward`: standard deviation of return.
- `survival_rate`: fraction of episodes not ending in death, failed surface, or
  underwater timeout.
- `death_rate`: fraction of episodes ending in death, failed surface, or
  underwater timeout.
- `all_fish_caught_rate`: strict task success rate, measured as catching all
  fish and returning to the base.
- `avg_steps`: average episode length.
- `avg_fish_value`: average collected fish value.
- `avg_num_fish_caught`: average number of caught fish.
- `avg_remaining_oxygen`: average oxygen remaining at termination.
- `avg_remaining_time`: average global time remaining at termination.

These metrics allow comparison between the exact Value Iteration policy and the
model-free SARSA and Q-learning policies.

## Results and Interpretation

All generated results are written to:

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

Plotting can also generate:

```text
sarsa_reward_curve.png
q_learning_reward_curve.png
reward_curve_comparison.png
survival_rate_curve.png
death_rate_curve.png
policy_map_value_iteration.png
policy_map_sarsa.png
policy_map_q_learning.png
trajectory_value_iteration.png
trajectory_sarsa.png
trajectory_q_learning.png
oxygen_time_curve_q_learning.png
```

The CSV files can be used for result tables. The PNG files can be used for
graphs, policy visualizations, trajectory examples, and presentation slides.
Reward curves show learning progress, survival/death curves show safety trends,
policy maps show representative greedy actions, and trajectory plots show an
example path taken by a learned or computed policy.

## Design Justification

The MDP design is reasonable for the project because the state includes the
necessary information for decision making: location, oxygen, time, carried
weight, and fish availability. The action space matches realistic diver choices:
moving, catching, and surfacing. The reward function reflects the real-world
objective of collecting valuable fish while avoiding inefficient movement,
unsafe decisions, and mission failure.

The oxygen and time constraints create long-term planning requirements. A policy
must decide not only which fish are valuable, but also whether the diver can
reach them, catch them, and return safely. The 8x8 tabular environment is small
enough for exact DP comparison but still captures the key sequential
decision-making challenge.

## Limitations

The environment is simplified. The map layout, fish positions, fish values, and
fish aggression probabilities are fixed. The state space is large but still
handled with tabular sparse dictionaries. Function approximation and neural
networks are intentionally not used because this project focuses on tabular
Dynamic Programming and model-free Reinforcement Learning methods.

## Installation

```bash
pip install -r requirements.txt
```

Dependencies:

```text
numpy
matplotlib
pandas
tqdm
```

## Run Everything

```bash
python main.py
```

Equivalent explicit command:

```bash
python main.py --mode all
```

This runs:

1. Value Iteration
2. SARSA
3. Q-learning
4. Evaluation of all three policies
5. Plot generation for available logs and policies

Progress bars are enabled by default.

## Individual Commands

Run only Value Iteration:

```bash
python main.py --mode value_iteration
```

Train only SARSA:

```bash
python main.py --mode train_sarsa
```

Train only Q-learning:

```bash
python main.py --mode train_q_learning
```

Evaluate all available saved policies:

```bash
python main.py --mode evaluate
```

Generate plots from available saved files:

```bash
python main.py --mode plot
```

## Quick Smoke Test

Use smaller settings for a quick smoke test:

```bash
python main.py --mode value_iteration --max_iterations 1
python main.py --mode train_sarsa --episodes 10
python main.py --mode train_q_learning --episodes 10
python main.py --mode evaluate --eval_episodes 5
python main.py --mode plot
```

These commands check that the environment, algorithms, evaluation, and plotting
pipeline run correctly. The short Value Iteration command is only a smoke test;
it is not intended to produce a converged DP policy.

## Reference Result

On the current 8x8 MDP, one representative run produced:

```text
Reachable states: 1,320,682
Value Iteration iterations: 57
```

This version is intentionally smaller than the original full 11x11 design so
exact tabular DP can be completed and compared fairly with SARSA and Q-learning
on the same environment.

## Reproducibility Checklist

From a fresh clone:

```bash
git clone https://github.com/pyk-kentech/RLDIVER.git
cd RLDIVER
pip install -r requirements.txt
python main.py --mode all
```

After execution, inspect:

```bash
ls results
cat results/reduced_evaluation_summary.csv
```

Open the PNG files in `results/` to review learning curves, policy maps, and
sample trajectories.
