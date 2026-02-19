# Reinforcement Learning Concepts: A Comprehensive Guide

This document explains reinforcement learning concepts using concrete examples from our code generation project. If you're new to RL, this guide will help you understand how the pieces fit together.

## Table of Contents

1. [Introduction to Reinforcement Learning](#introduction-to-reinforcement-learning)
2. [Core RL Components](#core-rl-components)
3. [The Markov Decision Process (MDP)](#the-markov-decision-process-mdp)
4. [Value Functions and the Bellman Equation](#value-functions-and-the-bellman-equation)
5. [Q-Learning](#q-learning)
6. [Deep Q-Networks (DQN)](#deep-q-networks-dqn)
7. [Experience Replay](#experience-replay)
8. [Exploration vs Exploitation](#exploration-vs-exploitation)
9. [Reward Shaping](#reward-shaping)
10. [Advanced Topics](#advanced-topics)

---

## Introduction to Reinforcement Learning

### What is Reinforcement Learning?

**Reinforcement Learning (RL)** is a paradigm of machine learning where an agent learns to make decisions by interacting with an environment. Unlike supervised learning (where we provide correct answers) or unsupervised learning (where we find patterns), RL learns through **trial and error** guided by **rewards and penalties**.

### The RL Framework

The fundamental RL setup consists of:

```
Agent ←→ Environment

Agent observes STATE
      ↓
Agent takes ACTION
      ↓
Environment gives REWARD
      ↓
Environment transitions to NEW STATE
      ↓
(Repeat)
```

### Why RL for Code Generation?

Traditional approaches to code generation:
- **Supervised Learning**: Requires large datasets of (specification, code) pairs
- **Rule-based**: Brittle, hard to maintain, limited coverage

RL advantages:
- **Self-supervised**: Only needs specifications and test cases (no code examples needed!)
- **Optimizes for outcomes**: Directly optimizes for passing tests, not just mimicking examples
- **Explores novel solutions**: Can discover solutions not in training data
- **Multi-objective**: Can balance correctness, efficiency, readability, etc.

---

## Core RL Components

Let's understand each component using our code generation project as an example.

### 1. Agent

**Definition**: The learner or decision-maker. In our project, the agent is a neural network that learns to generate code.

**In our code**:
```python
# agents/dqn_agent.py
class DQNAgent:
    def __init__(self, state_dim, action_dim, ...):
        self.policy_net = QNetwork(state_dim, action_dim)  # Neural network
        self.target_net = QNetwork(state_dim, action_dim)  # Target network
```

**What it does**:
- Observes the current state (problem specification + code so far)
- Decides what action to take (which code token to add next)
- Learns from experience to improve its decisions

### 2. Environment

**Definition**: The world the agent interacts with. It responds to the agent's actions and provides feedback.

**In our code**:
```python
# environments/code_gen_env.py
class CodeGenerationEnv(gym.Env):
    def reset(self):
        # Start a new problem
        return initial_state

    def step(self, action):
        # Execute action (add code token)
        # Run tests, compute reward
        return next_state, reward, done, info
```

**What it does**:
- Presents problems to solve
- Executes code the agent generates
- Runs test cases
- Provides rewards based on outcomes

### 3. State (s)

**Definition**: A representation of the current situation. It contains all the information the agent needs to make a decision.

**In our code**:
```python
# environments/code_gen_env.py
def _get_observation(self):
    problem_embedding = self._embed_problem()      # What to solve
    code_embedding = self._embed_code()            # Code so far
    metadata = self._get_metadata()                # Step count, syntax validity, etc.

    observation = np.concatenate([problem_embedding, code_embedding, metadata])
    return observation  # 520-dimensional state vector
```

**Example state for "add two numbers"**:
- **Problem embedding**: Vector encoding "Add two numbers and return result"
- **Code embedding**: Vector encoding current code (e.g., "def add_numbers(a, b):")
- **Metadata**: [step=5, code_length=30, syntax_valid=1, tests_passed=0, ...]

### 4. Action (a)

**Definition**: A choice the agent makes that affects the environment.

**In our code**:
```python
# Action space: Discrete(5000)
# Each action ID maps to a code token or operation

def _action_to_token(self, action):
    vocab = ['def ', 'return ', 'if ', 'else:', 'for ', ':\n', '+', '-', ...]
    return vocab[action] if action < len(vocab) else ''
```

**Example actions**:
- Action 0 → "def "
- Action 1 → "return "
- Action 7 → ":\n"
- Action 14 → "+"

### 5. Reward (r)

**Definition**: Immediate feedback signal indicating how good an action was. The agent's goal is to maximize cumulative reward.

**In our code**:
```python
# rewards/code_quality_reward.py
def compute_reward(self, current_code, test_results, ...):
    # Correctness: Did tests pass?
    correctness_reward = (tests_passed / total_tests) * 10.0
    if tests_passed == total_tests:
        correctness_reward += 20.0  # Bonus for complete success

    # Quality: Is code well-written?
    quality_reward = self._quality_reward(current_code)

    # Efficiency: Is code fast/concise?
    efficiency_reward = self._efficiency_reward(current_code)

    total_reward = correctness_reward + quality_reward + efficiency_reward
    return total_reward
```

**Example rewards**:
- All tests pass + clean code: **+30.0**
- 2/4 tests pass, valid syntax: **+5.5**
- Syntax error: **-2.0**
- All tests pass but code too long: **+28.0**

### 6. Policy (π)

**Definition**: The agent's strategy for choosing actions. Maps states to actions.

**Types**:
- **Deterministic**: π(s) → a (always choose the same action for a state)
- **Stochastic**: π(a|s) → probability (sample actions from a distribution)

**In our code**:
```python
# agents/dqn_agent.py
def select_action(self, state, training=True):
    # Epsilon-greedy policy (stochastic during training)
    if training and random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)  # Random action

    # Greedy action (deterministic)
    q_values = self.policy_net(state)
    return q_values.argmax()  # Best action according to Q-values
```

**Example**:
```
State: "Problem: add two numbers, Code: def add_numbers(a, b):"

Q-values: [Q(s, "def ")=-1.2, Q(s, "return ")=5.3, Q(s, "if ")=2.1, ...]

Policy: π(s) = argmax(Q) = "return " (action with highest Q-value)
```

### 7. Episode

**Definition**: A complete sequence of interactions from start to termination.

**In our code**:
```python
# One episode = One attempt to solve a problem

episode_start:
    state = env.reset()  # Get problem: "add two numbers"
    done = False

    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        state = next_state

episode_end (done=True when):
    - All tests pass (success!)
    - Max steps reached (failure)
    - Invalid code (failure)
```

**Example episode**:
```
Step 1: State: "add two numbers, code: ''"
        Action: "def "
        Reward: +0.05 (progress)
        New State: "add two numbers, code: 'def '"

Step 2: State: "add two numbers, code: 'def '"
        Action: "add_numbers"
        Reward: +0.15 (function definition)
        New State: "add two numbers, code: 'def add_numbers'"

...

Step 8: State: "add two numbers, code: 'def add_numbers(a, b):\n    return a + b'"
        Action: DONE
        Reward: +30.0 (all tests pass!)
        Episode ends
```

---

## The Markov Decision Process (MDP)

### Mathematical Framework

RL problems are formalized as **Markov Decision Processes (MDPs)**, defined by:

- **S**: Set of states
- **A**: Set of actions
- **P**: Transition probabilities P(s'|s,a) = probability of reaching state s' from state s via action a
- **R**: Reward function R(s,a,s') = reward for transition
- **γ**: Discount factor (0 ≤ γ ≤ 1)

### The Markov Property

**Definition**: The future depends only on the current state, not on the history.

```
P(s_{t+1} | s_t, a_t, s_{t-1}, a_{t-1}, ...) = P(s_{t+1} | s_t, a_t)
```

**In our project**:
The state includes everything needed to make a decision:
- Problem specification (doesn't change)
- Current code (complete history of actions)
- Metadata (current progress)

We don't need to remember the sequence of actions taken; the current code state captures it all!

### Trajectory (τ)

A trajectory is a sequence of states, actions, and rewards:

```
τ = (s_0, a_0, r_0, s_1, a_1, r_1, ..., s_T, a_T, r_T)
```

**Example trajectory** (solving "add two numbers"):
```
s_0: Problem + empty code
a_0: "def "
r_0: +0.05
s_1: "def "
a_1: "add_numbers(a, b):\n"
r_1: +0.2
s_2: "def add_numbers(a, b):\n"
a_2: "    return a + b"
r_2: +30.0
(Episode ends - all tests pass)
```

---

## Value Functions and the Bellman Equation

### Why Value Functions?

Instead of directly learning a policy, we often learn **value functions** that estimate how good states or state-action pairs are. Then we derive the policy from the value function.

### State-Value Function V(s)

**Definition**: Expected cumulative reward from state s following policy π.

```
V^π(s) = E_π[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s]
       = E_π[Σ_{k=0}^∞ γ^k R_{t+k} | s_t = s]
```

**Intuition**: "How good is it to be in state s (if I follow policy π)?"

**Example**:
```
State: "Problem: add(2,3)→5, Code: 'def add_numbers(a, b):\n    return '"

V(s) ≈ 25.0  (high value - close to solution!)

State: "Problem: add(2,3)→5, Code: 'def add_numbers(a, b):\n    if '"

V(s) ≈ 5.0  (lower value - went down wrong path)
```

### Action-Value Function Q(s, a)

**Definition**: Expected cumulative reward from state s, taking action a, then following policy π.

```
Q^π(s, a) = E_π[R_t + γR_{t+1} + γ²R_{t+2} + ... | s_t = s, a_t = a]
```

**Intuition**: "How good is it to take action a in state s?"

**Example**:
```
State s: "Problem: add(2,3)→5, Code: 'def add_numbers(a, b):\n    '"

Q(s, "return ") = 28.0  (good action!)
Q(s, "if ") = 5.0       (suboptimal)
Q(s, "for ") = 2.0      (bad action)

Best action: argmax_a Q(s, a) = "return "
```

**In our code**:
```python
# models/code_gen_model.py
class QNetwork(nn.Module):
    def forward(self, state):
        # Neural network computes Q(s, a) for all actions
        q_values = self.fc3(x)  # Output: [Q(s,a_0), Q(s,a_1), ..., Q(s,a_n)]
        return q_values

# agents/dqn_agent.py
q_values = self.policy_net(state)  # Get Q(s, a) for all actions
action = q_values.argmax()         # Choose action with highest Q-value
```

### Bellman Equations

**Bellman Equation for V(s)**:
```
V(s) = Σ_a π(a|s) Σ_{s'} P(s'|s,a) [R(s,a,s') + γ V(s')]
```

Intuition: Value of current state = immediate reward + discounted value of next state

**Bellman Equation for Q(s, a)**:
```
Q(s, a) = Σ_{s'} P(s'|s,a) [R(s,a,s') + γ Σ_{a'} π(a'|s') Q(s', a')]
```

For deterministic transitions (our case):
```
Q(s, a) = R(s, a) + γ max_{a'} Q(s', a')
```

**This is the foundation of Q-learning!**

---

## Q-Learning

### The Q-Learning Algorithm

Q-learning learns the optimal Q-function Q*(s, a) without needing to know the transition probabilities P(s'|s,a).

**Update Rule**:
```
Q(s, a) ← Q(s, a) + α [r + γ max_{a'} Q(s', a') - Q(s, a)]
                        \_____________________________/
                                TD Error
```

Where:
- **α**: Learning rate (step size)
- **γ**: Discount factor
- **r**: Reward received
- **s'**: Next state
- **TD Error**: Difference between target and current estimate

**Intuition**:
1. Current estimate: Q(s, a)
2. Better estimate: r + γ max_{a'} Q(s', a')  (immediate reward + best future value)
3. Update toward better estimate

### Temporal Difference (TD) Learning

Q-learning is a **TD learning** method: it learns from the difference between successive estimates.

```
TD Target: y = r + γ max_{a'} Q(s', a')
TD Error: δ = y - Q(s, a)
Update: Q(s, a) ← Q(s, a) + α δ
```

**Example**:
```
State s: "def add_numbers(a, b):\n    "
Action a: "return "
Current Q(s, a) = 15.0

After taking action:
Reward r = 1.0
Next state s': "def add_numbers(a, b):\n    return "
max_{a'} Q(s', a') = 25.0  (best future value)

TD Target: y = 1.0 + 0.99 * 25.0 = 25.75
TD Error: δ = 25.75 - 15.0 = 10.75

Update: Q(s, a) ← 15.0 + 0.1 * 10.75 = 16.075
```

### Off-Policy Learning

Q-learning is **off-policy**: it learns the optimal policy while following a different policy (e.g., ε-greedy).

- **Behavior policy**: ε-greedy (explores randomly sometimes)
- **Target policy**: Greedy (always choose best action)

This allows exploration while learning the optimal policy!

---

## Deep Q-Networks (DQN)

### Why Deep Learning?

Classical Q-learning maintains a table Q(s, a) for every state-action pair.

**Problem**: Code generation has:
- **Huge state space**: Infinite possible problem specifications and partial code states
- **Huge action space**: 5000+ possible code tokens

Solution: **Function approximation** with neural networks!

### DQN Architecture

Instead of a table, use a neural network to approximate Q(s, a):

```
Q(s, a; θ) ≈ Q*(s, a)
```

Where θ are the network parameters (weights).

**In our code**:
```python
# models/code_gen_model.py
class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        self.fc1 = nn.Linear(state_dim, 512)     # 520 → 512
        self.fc2 = nn.Linear(512, 512)           # 512 → 512
        self.fc3 = nn.Linear(512, action_dim)    # 512 → 5000

    def forward(self, state):
        # state: [batch, 520] (problem + code + metadata)
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        q_values = self.fc3(x)  # [batch, 5000] Q-values
        return q_values
```

**Visualization**:
```
Input: State vector [520 dimensions]
  ↓
Hidden Layer 1: [512 neurons] + ReLU activation
  ↓
Hidden Layer 2: [512 neurons] + ReLU activation
  ↓
Output Layer: [5000 Q-values] (one for each action)
  ↓
Action Selection: argmax(Q-values)
```

### Training DQN

**Objective**: Minimize TD error

```
Loss = E[(y - Q(s, a; θ))²]

where y = r + γ max_{a'} Q(s', a'; θ⁻)  (target Q-value)
```

**In our code**:
```python
# agents/dqn_agent.py
def train(self, batch_size):
    # Sample mini-batch from replay buffer
    transitions = self.replay_buffer.sample(batch_size)

    # Compute current Q-values: Q(s, a; θ)
    current_q_values = self.policy_net(states).gather(1, actions)

    # Compute target Q-values: r + γ max_{a'} Q(s', a'; θ⁻)
    with torch.no_grad():
        next_q_values = self.target_net(next_states).max(1)[0]
        target_q_values = rewards + (1 - dones) * self.gamma * next_q_values

    # Compute loss: MSE(target, current)
    loss = F.mse_loss(current_q_values, target_q_values)

    # Backpropagation
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()
```

### Target Network

**Problem**: TD target y = r + γ max_{a'} Q(s', a'; θ) depends on the same parameters θ we're updating!

This creates a moving target → unstable learning.

**Solution**: Use separate **target network** with frozen parameters θ⁻:

```
y = r + γ max_{a'} Q(s', a'; θ⁻)
```

Update θ⁻ periodically (every N episodes) by copying θ.

**In our code**:
```python
# agents/dqn_agent.py
def __init__(self, ...):
    self.policy_net = QNetwork(...)  # Updated every training step
    self.target_net = QNetwork(...)  # Updated every N episodes
    self.target_net.load_state_dict(self.policy_net.state_dict())

def update_target_network(self):
    # Copy policy network → target network
    self.target_net.load_state_dict(self.policy_net.state_dict())
```

**Why it helps**:
- Policy network: changes frequently (unstable but learning)
- Target network: changes slowly (stable targets)
- Result: More stable training!

---

## Experience Replay

### The Problem

Training on consecutive experiences creates problems:

1. **Correlation**: Consecutive states are highly correlated
2. **Non-stationarity**: Data distribution changes as policy improves
3. **Sample inefficiency**: Each experience used only once

### The Solution

**Experience Replay Buffer**: Store past experiences and sample randomly.

**In our code**:
```python
# agents/dqn_agent.py
class ReplayBuffer:
    def __init__(self, capacity=10000):
        self.buffer = deque(maxlen=capacity)  # Circular buffer

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)  # Random sampling!
```

### How It Works

```
Training Loop:
    1. Interact with environment
       → Store experiences in buffer: (s, a, r, s', done)

    2. Sample random mini-batch from buffer
       → Break correlations!

    3. Train on mini-batch
       → Reuse experiences multiple times
```

**Example**:
```
Buffer: [(s_1, a_1, r_1, s'_1), (s_2, a_2, r_2, s'_2), ..., (s_10000, a_10000, r_10000, s'_10000)]

Sample batch_size=64 randomly:
    → Might get experiences from episodes 5, 23, 47, 102, ... (completely uncorrelated!)

Train on this diverse batch
    → More stable learning
```

### Benefits

1. **Breaks temporal correlations**: Random sampling decorrelates experiences
2. **Sample efficiency**: Each experience can be used for multiple gradient updates
3. **Stabilizes learning**: Smooth out fluctuations in data distribution

---

## Exploration vs Exploitation

### The Dilemma

**Exploitation**: Choose actions that are known to be good (maximize immediate reward)

**Exploration**: Try new actions to discover potentially better strategies (might find higher reward later)

**Example**:
```
State: "def add_numbers(a, b):\n    "

Known good action: "return a + b" (Q = 25.0)
Unknown action: "return sum([a, b])" (Q = unknown, might be 30.0 or 5.0)

Exploit: Always choose "return a + b" → safe but might miss better solution
Explore: Sometimes try "return sum([a, b])" → risky but might discover better approach
```

### ε-Greedy Exploration

**Strategy**: With probability ε, choose random action; otherwise, choose best action.

```
Action selection:
    if random() < ε:
        action = random_action()  # Explore
    else:
        action = argmax Q(s, a)   # Exploit
```

**In our code**:
```python
# agents/dqn_agent.py
def select_action(self, state, training=True):
    # Exploration: random action
    if training and random.random() < self.epsilon:
        return random.randint(0, self.action_dim - 1)

    # Exploitation: best action
    q_values = self.policy_net(state)
    return q_values.argmax().item()
```

### ε-Decay Schedule

Start with high exploration, gradually shift to exploitation:

```
ε_t = max(ε_end, ε_start * decay^t)
```

**In our code**:
```python
# agents/dqn_agent.py
def decay_epsilon(self):
    self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

# Example:
# Episode 0: ε = 1.0 (100% exploration)
# Episode 100: ε ≈ 0.6
# Episode 500: ε ≈ 0.08
# Episode 1000: ε = 0.01 (1% exploration, 99% exploitation)
```

**Intuition**:
- **Early training**: Agent knows nothing → explore widely
- **Late training**: Agent has learned good policy → exploit knowledge

---

## Reward Shaping

### The Challenge

Raw rewards are often **sparse** (only get reward at the end) or **delayed** (consequence of action appears much later).

**Example**:
```
Sparse reward:
    Episode starts
    ... 49 steps of code generation ...
    Episode ends
    Reward: +30 if all tests pass, 0 otherwise

Problem: Agent doesn't know which of the 49 actions were good!
```

### Reward Shaping Solution

Provide **intermediate rewards** to guide learning.

**In our code**:
```python
# rewards/code_quality_reward.py
def compute_reward(self, code, test_results, ...):
    reward = 0

    # Immediate feedback for progress
    if syntax_valid and not previous_syntax_valid:
        reward += 0.3  # Reward for fixing syntax!

    if 'def ' in code and 'def ' not in previous_code:
        reward += 0.2  # Reward for adding function definition!

    # Partial credit for tests
    reward += (tests_passed / total_tests) * 10.0  # Not all-or-nothing!

    # Bonus for complete success
    if tests_passed == total_tests:
        reward += 20.0

    return reward
```

### Types of Shaped Rewards

1. **Progress Rewards**: Reward incremental improvements
   ```python
   # Code becomes syntactically valid
   if is_valid_syntax(code) and not is_valid_syntax(previous_code):
       reward += 0.3
   ```

2. **Partial Success Rewards**: Reward partial completion
   ```python
   # Some tests pass (not all-or-nothing)
   reward += (tests_passed / total_tests) * 10.0
   ```

3. **Penalty Avoidance**: Small penalties for bad actions
   ```python
   # Syntax errors
   if not is_valid_syntax(code):
       reward -= 0.5
   ```

4. **Efficiency Rewards**: Reward efficient solutions
   ```python
   # Taking too many steps
   reward -= 0.01 * step_count
   ```

### Reward Shaping Principles

**Good reward shaping**:
- Guides toward solution
- Doesn't change optimal policy
- Provides dense feedback

**Bad reward shaping**:
- Creates local optima (reward hacking)
- Conflicts with main objective
- Makes task easier to exploit than solve

**Example of bad shaping**:
```python
# BAD: Reward code length (agent just adds spaces!)
reward += len(code) * 0.1

# GOOD: Reward reasonable length
if 10 <= len(code) <= 100:
    reward += 0.5
```

---

## Advanced Topics

### 1. Dueling DQN

**Idea**: Separate Q-function into value and advantage:

```
Q(s, a) = V(s) + [A(s, a) - mean(A(s, ·))]
```

- **V(s)**: How good is state s?
- **A(s, a)**: How much better is action a compared to average?

**In our code**:
```python
# models/code_gen_model.py
class DuelingQNetwork(nn.Module):
    def forward(self, state):
        # Shared features
        features = self.shared_network(state)

        # Value stream: V(s)
        value = self.value_stream(features)  # [batch, 1]

        # Advantage stream: A(s, a)
        advantages = self.advantage_stream(features)  # [batch, actions]

        # Combine: Q(s,a) = V(s) + A(s,a) - mean(A)
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        return q_values
```

**When useful**: When many actions have similar values (common in code generation).

### 2. Double DQN

**Problem**: Standard DQN overestimates Q-values (always takes max).

**Solution**: Use policy network to select action, target network to evaluate:

```
Standard DQN:
    y = r + γ max_a Q(s', a; θ⁻)

Double DQN:
    a* = argmax_a Q(s', a; θ)    (policy network selects)
    y = r + γ Q(s', a*; θ⁻)      (target network evaluates)
```

**Implementation**:
```python
# Instead of:
next_q_values = self.target_net(next_states).max(1)[0]

# Use:
best_actions = self.policy_net(next_states).argmax(1)
next_q_values = self.target_net(next_states).gather(1, best_actions)
```

### 3. Prioritized Experience Replay

**Idea**: Sample important experiences more frequently.

Priority based on TD error:
```
priority_i = |δ_i| + ε = |r + γ max_a' Q(s', a') - Q(s, a)| + ε
```

Large TD error → more to learn from → sample more often.

**Implementation**:
```python
class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha=0.6):
        self.buffer = []
        self.priorities = []
        self.alpha = alpha  # Priority exponent

    def push(self, experience, td_error):
        priority = (abs(td_error) + 1e-5) ** self.alpha
        self.buffer.append(experience)
        self.priorities.append(priority)

    def sample(self, batch_size):
        # Sample proportional to priority
        probs = self.priorities / sum(self.priorities)
        indices = np.random.choice(len(self.buffer), batch_size, p=probs)
        return [self.buffer[i] for i in indices]
```

### 4. Curriculum Learning

**Idea**: Start with easy problems, gradually increase difficulty.

**In our code**:
```python
# rewards/code_quality_reward.py
class CurriculumReward:
    def compute_reward(self, code, test_results, episode):
        progress = episode / total_episodes

        # Stage 1 (0-30%): Just reward valid syntax
        if progress < 0.3:
            return 1.0 if is_valid_syntax(code) else 0.0

        # Stage 2 (30-70%): Reward passing any test
        elif progress < 0.7:
            return tests_passed * 1.0

        # Stage 3 (70-100%): Reward passing all tests
        else:
            return 10.0 if tests_passed == total_tests else 0.0
```

**Why it works**: Early exploration when agent is random, high standards when agent is competent.

### 5. Multi-Objective RL

Optimize multiple objectives simultaneously:

```python
# rewards/code_quality_reward.py
total_reward = (
    w_correctness * correctness_reward +    # Tests pass?
    w_quality * quality_reward +            # Clean code?
    w_efficiency * efficiency_reward +      # Fast/concise?
    w_progress * progress_reward            # Improving?
)
```

**Tuning weights**: Balance objectives based on importance.

---

## Summary

### Key Takeaways

1. **RL Framework**: Agent interacts with environment, learns from rewards
2. **MDP**: Mathematical formalization of sequential decision making
3. **Q-Learning**: Learn Q(s, a) to estimate value of state-action pairs
4. **DQN**: Use neural networks to approximate Q-function for large state/action spaces
5. **Experience Replay**: Store and reuse experiences for stable learning
6. **Exploration**: Balance trying new things vs. exploiting knowledge
7. **Reward Shaping**: Guide learning with informative intermediate rewards

### From Theory to Code

Every RL concept has a concrete implementation in our project:

| Concept | Implementation |
|---------|----------------|
| State | `environments/code_gen_env.py::_get_observation()` |
| Action | `environments/code_gen_env.py::_apply_action()` |
| Reward | `rewards/code_quality_reward.py::compute_reward()` |
| Q-Network | `models/code_gen_model.py::QNetwork` |
| Experience Replay | `agents/dqn_agent.py::ReplayBuffer` |
| Training Loop | `train.py::train()` |
| ε-Greedy | `agents/dqn_agent.py::select_action()` |

### Further Exploration

To deepen your understanding:

1. **Experiment with hyperparameters**: Change learning rate, gamma, epsilon decay
2. **Design new reward functions**: What happens with sparse vs. dense rewards?
3. **Try different problems**: More complex code generation tasks
4. **Implement extensions**: Double DQN, prioritized replay, dueling architecture
5. **Visualize learning**: Plot Q-values, action distributions, generated code quality over time

---

**Now you understand RL! Time to experiment and build your own RL agents. 🚀**
