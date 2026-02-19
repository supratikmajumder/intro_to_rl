# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Reinforcement Learning (RL) training project** focused on teaching agents to generate Python code from natural language specifications and test cases. The project uses **Deep Q-Networks (DQN)** to learn code generation policies through trial-and-error interaction with a custom OpenAI Gym environment.

### Software Engineering Use Case

The project demonstrates RL applied to automated code generation:
- **Input**: Function specification (docstring) + test cases
- **Output**: Python function implementation that passes all tests
- **Learning**: Agent improves through rewards based on test passage and code quality

## Project Structure

```
rl_training/
├── agents/                  # RL agent implementations (DQN with experience replay)
├── environments/            # Custom Gym environments for code generation
├── models/                  # Neural network architectures (Q-Network, Dueling DQN)
├── rewards/                 # Reward function implementations (multi-objective, sparse, curriculum)
├── utils/                   # Helper functions (data loading, logging, evaluation)
├── configs/                 # YAML configuration files for training
├── data/examples/          # Problem datasets (training and evaluation)
├── experiments/            # Training outputs (logs, model checkpoints) - created during training
├── docs/                   # Documentation (RL concepts guide)
├── train.py               # Main training script
├── evaluate.py            # Model evaluation script
└── requirements.txt       # Python dependencies
```

### Key Files

- **`environments/code_gen_env.py`**: OpenAI Gym environment that presents coding problems, executes generated code, runs tests, and provides rewards
- **`agents/dqn_agent.py`**: DQN agent with experience replay buffer, epsilon-greedy exploration, and target network
- **`models/code_gen_model.py`**: Neural network architectures (QNetwork, DuelingQNetwork) for approximating Q-function
- **`rewards/code_quality_reward.py`**: Multi-objective reward functions balancing correctness, quality, efficiency, and progress
- **`train.py`**: Main training loop orchestrating agent-environment interaction
- **`evaluate.py`**: Evaluation script for testing trained models on held-out problems

## Development Commands

### Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### Training

```bash
# Train with default configuration
python train.py --config configs/train_config.yaml

# Training outputs:
# - Model checkpoints: experiments/models/
# - Training logs: experiments/logs/
# - Metrics plots: experiments/logs/training_curves.png
```

### Evaluation

```bash
# Evaluate trained model on test problems
python evaluate.py --model experiments/models/best_model.pt \
                   --problems data/examples/eval_problems.json \
                   --num_episodes 50

# Compare multiple models
python evaluate.py --compare experiments/models/checkpoint_ep1000.pt \
                             experiments/models/checkpoint_ep2000.pt \
                   --problems data/examples/eval_problems.json
```

### Testing Individual Components

```bash
# Test environment
python environments/code_gen_env.py

# Test DQN agent
python agents/dqn_agent.py

# Test Q-Network
python models/code_gen_model.py

# Test reward functions
python rewards/code_quality_reward.py

# Test utilities
python utils/helpers.py
```

## Architecture Overview

### RL Components

**State (520-dim vector)**:
- Problem embedding (256 dims): Encodes docstring and function signature
- Code embedding (256 dims): Encodes current code state
- Metadata (8 dims): Step count, syntax validity, tests passed, etc.

**Action Space (Discrete 5000)**:
- Each action corresponds to a code token (keywords, operators, identifiers) or operation (indent, dedent, newline)
- Agent selects actions to build code incrementally

**Reward Function**:
- Correctness: Test passage rate (primary objective)
- Quality: Code readability, structure, simplicity
- Efficiency: Generation speed, code conciseness
- Progress: Incremental improvements (dense feedback)

**Episode Flow**:
1. Environment presents a random problem (docstring + test cases)
2. Agent generates code token-by-token
3. After each action, environment checks syntax and runs tests
4. Episode ends when all tests pass or max steps reached
5. Agent receives cumulative reward based on final code quality

### DQN Training

**Network Architecture**:
```
Input (520) → FC(512) → ReLU → Dropout → FC(512) → ReLU → Dropout → FC(5000)
                                                                        ↓
                                                                  Q-values for each action
```

**Training Algorithm**:
1. **Experience Collection**: Agent interacts with environment using ε-greedy policy
2. **Experience Storage**: Transitions (s, a, r, s', done) stored in replay buffer
3. **Mini-batch Sampling**: Random batch sampled from buffer (breaks correlations)
4. **Q-value Update**: Minimize TD error: [r + γ max_a' Q_target(s', a')] - Q_policy(s, a)
5. **Target Network Update**: Copy policy network → target network every N episodes
6. **Epsilon Decay**: Gradually reduce exploration rate

**Key Hyperparameters**:
- Learning rate (α): 0.0001
- Discount factor (γ): 0.99
- Epsilon decay: 0.995 per episode
- Replay buffer: 50,000 transitions
- Batch size: 64
- Target update frequency: Every 10 episodes

## Configuration

Training behavior is controlled by `configs/train_config.yaml`:

### Important Settings

**Environment**:
- `max_steps`: Maximum actions per episode (default: 50)
- `vocab_size`: Size of code token vocabulary (default: 5000)

**Agent**:
- `learning_rate`: Step size for gradient descent (tune for convergence speed)
- `gamma`: Discount factor for future rewards (0.99 = value long-term rewards)
- `epsilon_start/end/decay`: Exploration schedule (start high, decay to low)
- `buffer_capacity`: Experience replay size (larger = more diversity)
- `batch_size`: Mini-batch size (larger = more stable, slower)

**Reward**:
- `type`: 'code_quality' (multi-objective), 'sparse' (binary), or 'curriculum' (progressive)
- `correctness_weight`: Weight for test passage (most important)
- `quality_weight`: Weight for code quality metrics
- `efficiency_weight`: Weight for code efficiency
- `progress_weight`: Weight for incremental progress

**Training**:
- `num_episodes`: Total training episodes (5000 default, increase for harder problems)
- `eval_frequency`: Evaluate every N episodes (50 default)
- `save_frequency`: Save checkpoint every N episodes (100 default)

## Code Patterns and Conventions

### Adding New Problems

Problems are defined in JSON format in `data/examples/`:

```json
{
  "function_name": "function_to_implement",
  "docstring": "Natural language description of what function should do",
  "test_cases": [
    [input_args, expected_output],
    ...
  ],
  "starter_code": ""
}
```

**Test Case Format**:
- Single argument: `[input, expected_output]`
- Multiple arguments: `[[arg1, arg2, ...], expected_output]`
- Expected output can be any JSON-serializable value

### Implementing Custom Reward Functions

Create new reward class in `rewards/`:

```python
class CustomReward:
    def compute_reward(self, current_code, test_results, **kwargs):
        # Your reward logic
        reward = calculate_reward(current_code, test_results)
        return reward
```

Then update `train.py` to use it:

```python
if reward_type == 'custom':
    reward_fn = CustomReward()
```

### Extending the Agent

The DQN agent is modular and extensible:

- **New exploration strategies**: Modify `select_action()` method
- **Different network architectures**: Use `DuelingQNetwork` or create custom architectures in `models/`
- **Prioritized replay**: Extend `ReplayBuffer` to track priorities
- **Multi-step returns**: Modify TD target computation in `train()` method

## Common Development Tasks

### Debugging Training

1. **Monitor metrics**: Check `experiments/logs/training_*.log` for episode-by-episode metrics
2. **Visualize training**: Open `experiments/logs/training_curves.png` to see learning progress
3. **Enable rendering**: Set `render_frequency: 1` in config to see generated code each episode
4. **Check Q-values**: Add logging in `dqn_agent.py::train()` to inspect Q-value magnitudes

### Hyperparameter Tuning

Start with these ranges:
- Learning rate: [1e-5, 1e-3] (too high → divergence, too low → slow learning)
- Gamma: [0.95, 0.99] (higher → values long-term rewards more)
- Epsilon decay: [0.99, 0.999] (slower decay → more exploration)
- Batch size: [32, 128] (larger → more stable but slower)

### Improving Performance

If agent isn't learning well:

1. **Check reward function**: Ensure it provides informative signal
2. **Increase exploration**: Slower epsilon decay or higher epsilon_end
3. **Simplify problems**: Start with easier problems to verify learning
4. **Adjust network capacity**: Try larger hidden_dim (512 → 1024)
5. **More training**: Increase num_episodes (5000 → 10000)
6. **Better code embeddings**: Use pre-trained code models (CodeBERT)

## Important Notes

### State Representation

The current implementation uses simple hand-crafted features for code embeddings. For better performance, consider:
- Pre-trained code models (CodeBERT, GraphCodeBERT)
- AST-based representations
- Syntax-aware embeddings

### Action Space

The action space is simplified (discrete tokens). More sophisticated approaches:
- Hierarchical actions (statement-level, then token-level)
- Continuous action spaces with code transformations
- Pointer networks for copying from input

### Reward Design

Reward shaping is critical. Current multi-objective reward balances correctness and quality, but can be tuned:
- Increase `correctness_weight` if agent generates poor code that passes tests
- Increase `quality_weight` if agent generates correct but messy code
- Add custom metrics (security, performance, style)

## Documentation

- **README.md**: Comprehensive user guide with installation, usage, and concepts
- **docs/RL_CONCEPTS.md**: Detailed explanation of RL fundamentals with project examples
- **Inline documentation**: All modules have extensive docstrings explaining RL concepts

## Dependencies

Core dependencies (see `requirements.txt`):
- `torch>=2.0.0`: Deep learning framework for Q-network
- `gym>=0.26.0`: RL environment interface
- `numpy>=1.24.0`: Numerical operations
- `pyyaml>=6.0`: Configuration file parsing
- `matplotlib>=3.7.0`: Plotting training curves

Optional (commented in requirements.txt):
- `transformers`: For using pre-trained code models
- `tensorboard`: For advanced training visualization
- `wandb`: For experiment tracking
