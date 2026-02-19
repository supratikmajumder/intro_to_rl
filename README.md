# RL Training for Code Generation

A comprehensive reinforcement learning project that trains agents to generate Python functions from natural language specifications and test cases. This project demonstrates the application of RL techniques to software engineering tasks, specifically automated code generation.

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Detailed Usage](#detailed-usage)
- [Configuration](#configuration)
- [Key Concepts](#key-concepts)
- [Architecture](#architecture)
- [Extending the Project](#extending-the-project)
- [Troubleshooting](#troubleshooting)
- [Further Reading](#further-reading)

## Overview

This project applies **Deep Q-Network (DQN)** reinforcement learning to learn code generation policies. The agent learns to:

1. **Observe** problem specifications (docstrings and test cases)
2. **Generate** Python code token by token
3. **Receive rewards** based on test passage and code quality
4. **Improve** through trial and error

### Why RL for Code Generation?

Traditional supervised learning approaches require large datasets of (specification, code) pairs. Reinforcement learning offers several advantages:

- **Self-supervised learning**: Only needs problem specifications and test cases
- **Quality optimization**: Can optimize for multiple objectives (correctness, efficiency, readability)
- **Exploration**: Discovers novel solutions beyond training data
- **Continuous improvement**: Can refine solutions through interaction

### Software Engineering Use Case

This project targets a practical software engineering problem: **automated unit test implementation**. Given a function specification and test cases, the agent learns to generate correct Python implementations.

Example Problem:
```python
# Specification
Docstring: "Return the sum of two numbers."
Test Cases: [(2, 3) → 5, (0, 0) → 0, (-1, 1) → 0]

# Agent learns to generate:
def add_numbers(a, b):
    return a + b
```

## Project Structure

```
rl_training/
├── agents/                      # RL agent implementations
│   ├── __init__.py
│   └── dqn_agent.py            # Deep Q-Network agent with experience replay
│
├── environments/                # RL environments
│   ├── __init__.py
│   └── code_gen_env.py         # Code generation environment (OpenAI Gym)
│
├── models/                      # Neural network architectures
│   ├── __init__.py
│   └── code_gen_model.py       # Q-Network and Dueling Q-Network
│
├── rewards/                     # Reward function implementations
│   ├── __init__.py
│   └── code_quality_reward.py  # Multi-objective reward functions
│
├── utils/                       # Utility functions
│   ├── __init__.py
│   └── helpers.py              # Data loading, logging, evaluation
│
├── configs/                     # Configuration files
│   └── train_config.yaml       # Training hyperparameters
│
├── data/                        # Problem datasets
│   └── examples/
│       ├── training_problems.json   # Training problem set
│       └── eval_problems.json       # Evaluation problem set
│
├── experiments/                 # Training outputs (created during training)
│   ├── logs/                   # Training logs and metrics
│   └── models/                 # Saved model checkpoints
│
├── docs/                        # Documentation
│   └── RL_CONCEPTS.md          # Detailed RL concepts explanation
│
├── train.py                     # Main training script
├── evaluate.py                  # Evaluation script
├── requirements.txt             # Python dependencies
├── CLAUDE.md                    # Claude Code guidance
└── README.md                    # This file
```

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) CUDA-capable GPU for faster training

### Setup

1. **Clone or navigate to the repository:**
   ```bash
   cd rl_training
   ```

2. **Create a virtual environment (recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Verify installation:**
   ```bash
   python -c "import torch; import gym; import numpy; print('Installation successful!')"
   ```

## Quick Start

### Train a Model

Train an agent on the default problem set:

```bash
python train.py --config configs/train_config.yaml
```

This will:
- Load 8 training problems (add, is_even, max_of_three, etc.)
- Train a DQN agent for 5000 episodes
- Save checkpoints every 100 episodes
- Log metrics to `experiments/logs/`
- Save the best model to `experiments/models/best_model.pt`

Training takes approximately 30-60 minutes on CPU, 10-20 minutes on GPU.

### Evaluate a Trained Model

Evaluate the trained model on test problems:

```bash
python evaluate.py --model experiments/models/best_model.pt \
                   --problems data/examples/eval_problems.json \
                   --num_episodes 50
```

This will:
- Load the trained model
- Evaluate on 3 test problems (multiply, is_prime, find_min)
- Report success rates and generated code samples
- Show per-problem performance breakdown

## Detailed Usage

### Training Configuration

The training behavior is controlled by `configs/train_config.yaml`. Key parameters:

#### Environment Settings
```yaml
environment:
  max_steps: 50          # Maximum actions per episode
  vocab_size: 5000       # Size of code token vocabulary
```

#### Agent Hyperparameters
```yaml
agent:
  learning_rate: 0.0001  # α: Step size for gradient descent
  gamma: 0.99            # γ: Discount factor for future rewards
  epsilon_start: 1.0     # Initial exploration rate
  epsilon_end: 0.01      # Minimum exploration rate
  epsilon_decay: 0.995   # Decay rate per episode
  buffer_capacity: 50000 # Experience replay buffer size
  batch_size: 64         # Mini-batch size for training
```

**Learning Rate (α)**: Controls how quickly the agent updates its policy. Too high → unstable learning. Too low → slow convergence.

**Gamma (γ)**: Balances immediate vs. future rewards. γ=0 → myopic (only immediate rewards). γ=1 → farsighted (all future rewards equally important).

**Epsilon (ε)**: Exploration rate. High ε → more random exploration. Low ε → more exploitation of learned policy.

#### Reward Function
```yaml
reward:
  type: 'code_quality'      # Options: 'code_quality', 'sparse', 'curriculum'
  correctness_weight: 10.0  # Weight for test passage
  quality_weight: 2.0       # Weight for code quality
  efficiency_weight: 1.0    # Weight for efficiency
  progress_weight: 0.5      # Weight for incremental progress
```

**code_quality**: Multi-objective reward balancing correctness, quality, and efficiency
**sparse**: Binary reward (1 if all tests pass, 0 otherwise)
**curriculum**: Gradually increasing difficulty over training

### Creating Custom Problems

Add problems to `data/examples/training_problems.json`:

```json
{
  "function_name": "fibonacci",
  "docstring": "Return the nth Fibonacci number.",
  "test_cases": [
    [0, 0],
    [1, 1],
    [5, 5],
    [10, 55]
  ],
  "starter_code": ""
}
```

**Format**:
- `function_name`: Name of the function to generate
- `docstring`: Natural language description
- `test_cases`: List of [input, expected_output] pairs
- `starter_code`: Optional template code (usually empty)

### Advanced Training

#### Multi-GPU Training
```bash
# Set device in config
agent:
  device: 'cuda'  # Use GPU

# Train
python train.py --config configs/train_config.yaml
```

#### Resume from Checkpoint
```python
# In train.py, add after agent initialization:
agent.load('experiments/models/checkpoint_ep1000.pt')
```

#### Custom Reward Function

Create a new reward function in `rewards/`:

```python
class MyCustomReward:
    def compute_reward(self, code, test_results, **kwargs):
        # Your custom reward logic
        reward = ...
        return reward
```

Update config:
```yaml
reward:
  type: 'my_custom'  # Add handling in train.py
```

## Configuration

### Hyperparameter Tuning Guide

| Parameter | Effect | Recommended Range |
|-----------|--------|-------------------|
| `learning_rate` | Learning speed | 1e-5 to 1e-3 |
| `gamma` | Future reward importance | 0.95 to 0.99 |
| `epsilon_decay` | Exploration schedule | 0.99 to 0.999 |
| `batch_size` | Training stability | 32 to 128 |
| `hidden_dim` | Model capacity | 256 to 1024 |
| `buffer_capacity` | Experience diversity | 10k to 100k |

### Common Configurations

**Fast Training (for testing)**:
```yaml
training:
  num_episodes: 1000
agent:
  buffer_capacity: 5000
  batch_size: 32
```

**Production Training (best performance)**:
```yaml
training:
  num_episodes: 10000
agent:
  buffer_capacity: 100000
  batch_size: 128
  hidden_dim: 1024
```

## Key Concepts

### Reinforcement Learning Fundamentals

**Agent**: The code generator (neural network) that learns to generate code

**Environment**: The code generation task (problem specification + test execution)

**State (s)**: Current context (problem description + code generated so far)

**Action (a)**: Next code token or operation to add

**Reward (r)**: Feedback signal based on test results and code quality

**Policy (π)**: Strategy for selecting actions given states. Our agent learns π(s) = argmax_a Q(s,a)

**Episode**: One complete attempt at solving a problem (from start to solution or max steps)

### Deep Q-Learning (DQN)

DQN learns a **Q-function** Q(s,a) that estimates expected cumulative reward:

```
Q(s,a) = Expected total reward from taking action a in state s
```

The agent selects actions greedily: `a* = argmax_a Q(s,a)`

**Key Innovation**: Using a neural network to approximate Q(s,a) for large state/action spaces.

**Training**: Minimize temporal difference (TD) error:
```
TD Error = [r + γ * max_a' Q(s',a')] - Q(s,a)
         = target Q-value - current Q-value
```

### Experience Replay

Instead of learning from experiences in order, we:
1. Store experiences (s, a, r, s') in a replay buffer
2. Sample random mini-batches for training
3. Break correlations between consecutive samples
4. Improve sample efficiency (reuse experiences)

### Target Network

Maintain two networks:
- **Policy Network**: Updated every training step
- **Target Network**: Updated every N episodes

Target network provides stable Q-value targets during training, preventing moving target problem.

### Epsilon-Greedy Exploration

Balance exploration vs. exploitation:
- With probability ε: choose random action (explore)
- With probability 1-ε: choose best action (exploit)

ε decays over time: start with high exploration, gradually shift to exploitation.

## Architecture

### Code Generation Environment

The environment (`CodeGenerationEnv`) implements the OpenAI Gym interface:

```python
state = env.reset()                    # Start new episode
action = agent.select_action(state)    # Agent chooses action
next_state, reward, done, info = env.step(action)  # Execute action
```

**State Representation**:
- Problem embedding (256 dims): Encodes docstring and function signature
- Code embedding (256 dims): Encodes current code state
- Metadata (8 dims): Step count, syntax validity, test results

**Action Space**:
- Discrete actions representing code tokens (keywords, operators, identifiers)
- Special operations (indent, dedent, newline, delete)
- Vocabulary size: 5000 tokens

**Reward Signal**:
- Correctness: Tests passed / Total tests
- Quality: Syntax validity, proper structure, reasonable length
- Efficiency: Code complexity, generation speed
- Progress: Incremental improvements

### DQN Agent Architecture

**Q-Network**:
```
Input (520 dims) → FC(512) → ReLU → Dropout → FC(512) → ReLU → Dropout → FC(5000)
                                                                            ↓
                                                                    Q-values for each action
```

**Training Process**:
1. Observe state s from environment
2. Select action a using ε-greedy policy
3. Execute action, observe reward r and next state s'
4. Store transition (s, a, r, s', done) in replay buffer
5. Sample mini-batch from replay buffer
6. Compute target: y = r + γ * max_a' Q_target(s', a')
7. Update policy network: minimize (y - Q_policy(s, a))²
8. Periodically copy policy network → target network

### Reward Function Design

Multi-objective reward balances:

1. **Correctness** (weight: 10.0):
   - Linear: tests_passed / total_tests
   - Bonus: +20 for passing all tests

2. **Code Quality** (weight: 2.0):
   - Syntax validity: +0.5
   - Proper structure: +0.3
   - Reasonable length: +0.2
   - Low complexity: +0.2

3. **Efficiency** (weight: 1.0):
   - Fast generation: -0.1 * (steps / max_steps)
   - Concise code: penalty for excessive length
   - Efficient patterns: bonus for pythonic code

4. **Progress** (weight: 0.5):
   - Syntax improvement: +0.3
   - Structure improvement: +0.2
   - Continuous generation: +0.05

## Extending the Project

### Add New Problem Types

1. Create problem specification in `data/examples/`:
```json
{
  "function_name": "my_function",
  "docstring": "Description of what the function should do",
  "test_cases": [[input, output], ...],
  "starter_code": ""
}
```

2. Add to training or evaluation set

### Implement Advanced RL Algorithms

**Double DQN**: Reduce overestimation bias
```python
# In dqn_agent.py, modify target computation:
best_actions = self.policy_net(next_state_batch).argmax(1)
next_q_values = self.target_net(next_state_batch).gather(1, best_actions)
```

**Prioritized Experience Replay**: Sample important transitions more frequently
```python
# Modify ReplayBuffer to track TD errors and sample proportionally
```

**Dueling DQN**: Already implemented in `models/code_gen_model.py`
```python
# Use DuelingQNetwork instead of QNetwork
from models.code_gen_model import DuelingQNetwork
self.policy_net = DuelingQNetwork(state_dim, action_dim)
```

### Integrate Pre-trained Code Models

Use CodeBERT or GraphCodeBERT for better code embeddings:

```python
# Install transformers
pip install transformers

# In environments/code_gen_env.py:
from transformers import AutoTokenizer, AutoModel

class CodeGenerationEnv:
    def __init__(self, ...):
        self.tokenizer = AutoTokenizer.from_pretrained("microsoft/codebert-base")
        self.code_model = AutoModel.from_pretrained("microsoft/codebert-base")

    def _embed_code(self, code):
        inputs = self.tokenizer(code, return_tensors="pt")
        outputs = self.code_model(**inputs)
        return outputs.last_hidden_state.mean(dim=1).detach().numpy()
```

### Add Visualization Tools

Track training with TensorBoard:

```python
# Install tensorboard
pip install tensorboard

# In train.py:
from torch.utils.tensorboard import SummaryWriter

writer = SummaryWriter('experiments/tensorboard')

# In training loop:
writer.add_scalar('Reward/Episode', episode_reward, episode)
writer.add_scalar('Loss/Train', loss, episode)
writer.add_scalar('SuccessRate/Test', success_rate, episode)

# View in browser:
# tensorboard --logdir=experiments/tensorboard
```

## Troubleshooting

### Common Issues

**Issue**: Agent not learning (reward stays low)
- **Solution**: Reduce learning rate, increase buffer capacity, check reward function
- **Check**: Is epsilon decaying too fast? Agent needs exploration time.

**Issue**: Training is very slow
- **Solution**: Use GPU (`device: 'cuda'` in config), reduce batch size, simplify problems
- **Check**: Is vocab_size too large? Consider reducing to 1000-2000.

**Issue**: Agent generates invalid syntax
- **Solution**: Increase quality_weight in reward config, add syntax error penalty
- **Check**: Are reward weights balanced? Correctness weight might be too high.

**Issue**: Agent overfits to training problems
- **Solution**: Add more diverse problems, use curriculum learning, regularize model
- **Check**: Evaluate on held-out test set regularly.

### Debugging Training

Enable verbose logging:
```yaml
verbose: true
log_frequency: 1  # Log every episode
```

Render episodes to see generated code:
```python
# In train.py, add after env.step():
if episode % 100 == 0:
    env.render()
```

Check gradient flow:
```python
# In dqn_agent.py, after backward():
for name, param in self.policy_net.named_parameters():
    if param.grad is not None:
        print(f"{name}: {param.grad.abs().mean()}")
```

## Further Reading

### RL Resources
- **Sutton & Barto**: "Reinforcement Learning: An Introduction" (2nd Edition)
- **OpenAI Spinning Up**: https://spinningup.openai.com/
- **Deep RL Course**: https://huggingface.co/learn/deep-rl-course/

### Code Generation with RL
- **AlphaCode** (DeepMind): Competitive programming with RL
- **CodeRL** (Meta): Program synthesis with RL
- **RLTF** (OpenAI): RL from human feedback for code

### Related Papers
- **DQN**: Mnih et al. "Playing Atari with Deep RL" (2013)
- **Double DQN**: van Hasselt et al. "Deep RL with Double Q-learning" (2015)
- **Dueling DQN**: Wang et al. "Dueling Network Architectures" (2016)
- **Rainbow**: Hessel et al. "Rainbow: Combining Improvements in Deep RL" (2017)

### Software Engineering + AI
- **GitHub Copilot**: AI pair programmer
- **AlphaCode 2**: Advanced competitive programming
- **CodeT5**: Pre-trained code understanding and generation

---

## License

This project is for educational purposes. Feel free to use and modify for learning RL and code generation.

## Contributing

This is an educational project. Suggestions and improvements welcome!

## Citation

If you use this project in your research or coursework, please reference:
```
RL Training for Code Generation
Educational RL project demonstrating DQN for automated code generation
```

---

**Happy Learning! 🚀**
