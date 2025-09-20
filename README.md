# Bourbon

**Bourbon** is a Python package for Reinforcement Learning (RL), focusing on RL-based training of Large Language Models (LLMs).
It's an experimentation project built on top of PyTorch and the following research papers:

[Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/pdf/2303.11366)
[ReAct: Synergizing Reasoning and Acting in Language Models](https://arxiv.org/pdf/2210.03629)

The focus is to use natural language feedback as a reward signal to train LLMs to 1. solve a task via reasoning and acting, and 2. to improve the performance of LLMs on a given task via verbal self-reflection and to align the model's behavior with human preferences.

[![PyPI](https://img.shields.io/pypi/v/bourbon)](https://pypi.org/project/bourbon/)
<!-- [![Python Version](https://img.shields.io/pypi/pyversions/bourbon?label=Python)](https://pypi.org/project/bourbon/) -->
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/meghdadFar/bourbon/blob/main/LICENSE)

## Quick Start

### Installation

```bash
pip install bourbon
```

### Prerequisites

Before using Bourbon to solve your problem via RL, you need to define:

- **State space**: How your problem states are represented
- **Actions**: What operations your agent can perform  
- **Reward function**: How you assign rewards for actions

## 📖 Core Concepts

### Environment

The first step is mapping your problem to an RL environment. Environments can be:

- **Deterministic**: Same action in same state always produces same result
- **Stochastic**: Actions may have probabilistic outcomes

#### State Representation

States can be represented as vectors of natural numbers `{1, 2, 3, ...}`. Here's a classic grid world example:

<p align="center">
    <img src="docs/figs/rlgrid.png" alt="State space and rewards for each state. The agent is shown in orange, and the goal state is in green." width="400">
</p>

In this 3x3 grid:
- 9 total states (indexed 1-9)
- Agent (orange) navigates to reach the goal (green)
- Goal state provides reward of +10

### Actions

Actions define what operations your RL agent can perform. In the grid example above, the agent has 4 possible actions:
- **LEFT**: Move one cell left
- **RIGHT**: Move one cell right  
- **UP**: Move one cell up
- **DOWN**: Move one cell down

### Rewards

RL agents learn by maximizing future rewards. Bourbon supports:

- **Immediate rewards**: Agent receives feedback after each action
- **Delayed rewards**: Agent receives feedback only at episode end or after action sequences

Design your reward function to guide the agent toward desired behaviors.

## 🎯 Examples

### Featured Notebooks

Explore complete worked examples in the [`notebooks/`](./notebooks) directory:

| Notebook | Description |
|----------|-------------|
| [`multiplication.ipynb`](./notebooks/multiplication.ipynb) | Train an agent to learn multiplication tables |
| [`capitals.ipynb`](./notebooks/capitals.ipynb) | Train an agent to predict country capitals |
| [`wind.ipynb`](./notebooks/wind.ipynb) | Solve the classic windy gridworld problem |

### Quick Example

```python
import bourbon

# Define your environment, actions, and rewards
# Train your agent
agent = bourbon.QLearning(state_space_size=9, action_space_size=4)

# Your training loop here
for episode in range(1000):
    # ... training logic
    pass
```

## 🛠️ Development

### Requirements

- Python 3.10+
- PyTorch 2.0.0
- Additional dependencies listed in `pyproject.toml`

### Project Structure

```
bourbon/
├── bourbon/           # Main package
│   ├── q_learning.py  # Q-learning implementation
│   ├── qtable.py      # Q-table utilities
│   └── steps.py       # Step management
├── notebooks/         # Example notebooks
├── docs/             # Documentation
└── resources/        # Data files
```

## 📚 Documentation

- **Research Papers**: [`docs/articles/`](./docs/articles/)
- **Figures**: [`docs/figs/`](./docs/figs/)
- **Examples**: [`notebooks/`](./notebooks/)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit issues and pull requests.

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔗 Links

- [PyPI Package](https://pypi.org/project/bourbon/)
- [GitHub Repository](https://github.com/meghdadFar/bourbon)