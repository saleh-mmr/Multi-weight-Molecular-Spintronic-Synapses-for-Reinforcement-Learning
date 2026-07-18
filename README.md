# Multi-weight molecular spintronic synapses for reinforcement learning

This repository contains **three related CartPole reinforcement-learning projects** that progressively move from a standard DQN baseline to physically-inspired synaptic weight-update mechanisms.

## Repository layout

| Project | Purpose | Key difference |
| --- | --- | --- |
| `Baseline-RMSProp` | Reference DQN implementation | Uses standard RMSprop optimizer |
| `SignBased Optimizer` | Sign-driven update experiments | Replaces optimizer with custom gradient-sign controllers |
| `Synaptic_Weights` | Multi-task molecular/spintronic synapse simulation | Uses physical crosspoint conductance models and task-specific weight loading |

## Common architecture used across projects

Most projects follow this module layout:

- `envs/`: Gymnasium CartPole
- `network/`: DQN feed-forward Q-network
- `memory/`: replay buffer
- `agents/`: policy + learning logic
- `learning/`: train/test orchestration
- `scripts/`: runnable entry points and analysis tooling
- `utils/config.py`: runtime device and CUDA-debug configuration

## 1) Baseline-RMSProp

**Goal:** provide a clean DQN baseline for comparison.

### Main workflow

- Train/evaluate entry point: `Baseline-RMSProp/scripts/script.py`
- Core learner: `Baseline-RMSProp/learning/trainer.py`
- Agent + optimizer: `Baseline-RMSProp/agents/agent.py` (RMSprop)

### Analysis scripts

- `scripts/plot_weights_heatmap.py`
- `scripts/plot_weight_histograms.py`
- `scripts/calculate_information_content_autocorrelation.py`
- `scripts/analyze_all_weight_layers.py`

## 2) SignBased Optimizer

**Goal:** compare classical and sign-based update rules with the same DQN structure.

### Controllers

- `controller/linear_function_conductance.py`: linear Manhattan-style conductance update
- `controller/logarithmic_function_conductance.py`: logarithmic conductance update
- `controller/sgd_optimizer.py`: simple gradient-descent controller

### Main workflow

- Entry point: `SignBased Optimizer/scripts/script.py`
- Agent selection logic: `SignBased Optimizer/agents/agent.py` (`optimizer_selector`)
- Trainer: `SignBased Optimizer/learning/trainer.py`

## 3) Synaptic_Weights

**Goal:** train one shared network over **three modified CartPole tasks** using a multi-weight synapse model inspired by molecular spintronic behavior.

### Physical device model

- `devices/crosspointParams.py`: conductance/noise parameters
- `devices/crosspointState.py`: per-crosspoint index + noise state
- `devices/magnetoresistiveCrosspoint.py`: P/AP conductance behavior
- `devices/nonMagnetoresistiveCrosspoint.py`: bias conductance branch
- `devices/multiWeightSynapse.py`: combines crosspoints into one multi-weight synapse

### Synaptic controllers

- `controller/synaptic_weight_controller.py`: object-level implementation
- `controller/synaptic_weight_controller_optimize.py`: vectorized high-performance implementation used by the agent

### Training/testing scripts

- `scripts/train/train_script.py`: primary training run
- `scripts/train/grid_search.py`: hyperparameter sweeps
- `scripts/test/cartpole_progressive.py`: stage-wise checkpoint filtering
- `scripts/test/cartpole_swap.py`: swap-specificity verification
- `scripts/test/cartpole_test_script.py`: single pair test
- `scripts/test/multiple_weights_test.py`: bulk checkpoint test
- `scripts/analysis/*.py`: entropy, autocorrelation, heatmaps, histograms, and spectral analyses

## Environment and dependencies

Use Python 3.10+ (3.13 also works with the current code) and install:

```bash
pip install torch gymnasium numpy pandas matplotlib seaborn pytest
```

## How to run

### Baseline RMSProp training

```bash
cd "Baseline-RMSProp/scripts"
python3 script.py
```

### Sign-based optimizer training

```bash
cd "SignBased Optimizer/scripts"
python3 script.py
```

### Synaptic multi-weight training

```bash
cd "Synaptic_Weights/scripts/train"
python3 train_script.py
```

## Notes on reproducibility

- All projects expose explicit seeding in their run scripts.
- Model checkpoints are saved during training and then reused by test/analysis scripts.
- `Synaptic_Weights` stores run artifacts under `scripts/*/three_problems/run_<timestamp>/`.

