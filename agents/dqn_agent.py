"""
Deep Q-Network (DQN) Agent for Code Generation

This module implements a DQN agent that learns to generate code through
reinforcement learning. DQN combines Q-learning with deep neural networks
to handle large state and action spaces.

Key RL Concepts:
- Q-Learning: Learning action-value function Q(s,a)
- Deep Q-Network: Using neural networks to approximate Q-function
- Experience Replay: Storing and sampling past experiences
- Target Network: Stabilizing learning with separate target Q-network
- Epsilon-Greedy Exploration: Balancing exploration vs exploitation
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque, namedtuple
import random
import sys
import os
from typing import Tuple, List, Optional

# Add parent directory to path for imports when running as script
if __name__ == "__main__":
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


# Define a named tuple for storing experiences (transitions)
# In RL: A transition is (state, action, reward, next_state, done)
Transition = namedtuple('Transition',
                       ('state', 'action', 'reward', 'next_state', 'done'))


class ReplayBuffer:
    """
    Experience Replay Buffer for DQN.

    Experience replay is a key technique in DQN that:
    1. Breaks correlation between consecutive samples (improves stability)
    2. Allows reuse of experiences (improves sample efficiency)
    3. Enables mini-batch training (faster convergence)

    The buffer stores transitions (s, a, r, s', done) and allows sampling
    random mini-batches for training.

    Attributes:
        buffer (deque): Circular buffer storing transitions
        capacity (int): Maximum buffer size
    """

    def __init__(self, capacity: int = 10000):
        """
        Initialize replay buffer.

        Args:
            capacity: Maximum number of transitions to store
        """
        self.buffer = deque(maxlen=capacity)

    def push(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool):
        """
        Store a transition in the buffer.

        Args:
            state: Current state s_t
            action: Action taken a_t
            reward: Reward received r_t
            next_state: Next state s_{t+1}
            done: Whether episode terminated
        """
        self.buffer.append(Transition(state, action, reward, next_state, done))

    def sample(self, batch_size: int) -> List[Transition]:
        """
        Sample a random mini-batch of transitions.

        Random sampling breaks temporal correlations and improves learning.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            batch: List of sampled transitions
        """
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        """Return current buffer size."""
        return len(self.buffer)


class DQNAgent:
    """
    Deep Q-Network Agent for learning to generate code.

    The DQN agent learns a policy π(s) = argmax_a Q(s,a) by:
    1. Approximating Q-function with a neural network
    2. Using experience replay for stable learning
    3. Employing a target network to reduce overestimation
    4. Balancing exploration and exploitation via epsilon-greedy

    The Q-function Q(s,a) estimates the expected return (cumulative reward)
    from taking action a in state s and following the optimal policy thereafter.

    Attributes:
        state_dim (int): Dimension of state space
        action_dim (int): Dimension of action space
        policy_net (nn.Module): Main Q-network for action selection
        target_net (nn.Module): Target Q-network for stability
        optimizer (Optimizer): Optimizer for training policy_net
        replay_buffer (ReplayBuffer): Experience replay buffer
        epsilon (float): Exploration rate for epsilon-greedy policy
    """

    def __init__(self,
                 state_dim: int,
                 action_dim: int,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: float = 0.995,
                 buffer_capacity: int = 10000,
                 device: str = 'cpu'):
        """
        Initialize DQN agent.

        Args:
            state_dim: Dimension of observation space
            action_dim: Number of possible actions
            learning_rate: Learning rate for optimizer (α in literature)
            gamma: Discount factor (γ) - balances immediate vs future rewards
            epsilon_start: Initial exploration rate
            epsilon_end: Minimum exploration rate
            epsilon_decay: Decay rate for epsilon
            buffer_capacity: Size of experience replay buffer
            device: Device for PyTorch ('cpu' or 'cuda')
        """
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.device = torch.device(device)

        # Import the Q-network architecture from models module
        from models.code_gen_model import QNetwork

        # Policy network: The main Q-network used for action selection
        # Q(s, a; θ) where θ are the network parameters
        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)

        # Target network: Separate network for computing target Q-values
        # Q(s, a; θ⁻) where θ⁻ are periodically updated from θ
        # This stabilizes training by providing fixed targets
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()  # Target network is not trained directly

        # Optimizer for training the policy network
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

        # Loss function: Mean Squared Error for Q-value regression
        self.criterion = nn.MSELoss()

        # Experience replay buffer
        self.replay_buffer = ReplayBuffer(capacity=buffer_capacity)

        # Training statistics
        self.training_step = 0
        self.episode_count = 0

    def select_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Select action using epsilon-greedy policy.

        Epsilon-greedy balances exploration and exploitation:
        - With probability ε: choose random action (explore)
        - With probability 1-ε: choose action with highest Q-value (exploit)

        During training, ε decreases over time (annealing schedule):
        - Early training: high ε → more exploration → discover good strategies
        - Late training: low ε → more exploitation → refine best strategies

        Args:
            state: Current state observation
            training: If True, use epsilon-greedy; if False, use greedy policy

        Returns:
            action: Selected action
        """
        # Exploration: random action
        if training and random.random() < self.epsilon:
            return random.randint(0, self.action_dim - 1)

        # Exploitation: choose action with highest Q-value
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            action = q_values.argmax(dim=1).item()

        return action

    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """
        Store a transition in the replay buffer.

        Each transition represents one step of experience:
        (s_t, a_t, r_t, s_{t+1}, done_t)

        Args:
            state: State before action
            action: Action taken
            reward: Reward received
            next_state: State after action
            done: Whether episode terminated
        """
        self.replay_buffer.push(state, action, reward, next_state, done)

    def train(self, batch_size: int = 64) -> Optional[float]:
        """
        Train the agent using a mini-batch from the replay buffer.

        This implements the core DQN update rule:

        1. Sample mini-batch of transitions from replay buffer
        2. For each transition, compute target Q-value:
           y_t = r_t + γ * max_a' Q_target(s_{t+1}, a')   if not terminal
           y_t = r_t                                        if terminal

        3. Update policy network to minimize TD error:
           Loss = MSE(Q_policy(s_t, a_t), y_t)

        This is based on the Bellman equation:
        Q(s,a) = E[r + γ * max_a' Q(s',a')]

        Args:
            batch_size: Number of transitions to sample for training

        Returns:
            loss: Training loss value (None if buffer too small)
        """
        # Don't train until we have enough samples
        if len(self.replay_buffer) < batch_size:
            return None

        # Sample random mini-batch
        transitions = self.replay_buffer.sample(batch_size)
        batch = Transition(*zip(*transitions))

        # Convert batch to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch.done).to(self.device)

        # Compute current Q-values: Q(s_t, a_t; θ)
        # We use gather to select Q-values for actions that were actually taken
        current_q_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute target Q-values: y_t = r_t + γ * max_a' Q(s_{t+1}, a'; θ⁻)
        with torch.no_grad():
            # Get max Q-value for next states from target network
            next_q_values = self.target_net(next_state_batch).max(1)[0]

            # If episode is done, target is just the reward (no future rewards)
            # Otherwise, target is reward + discounted future Q-value
            target_q_values = reward_batch + (1 - done_batch) * self.gamma * next_q_values

        # Compute loss: MSE between current and target Q-values
        # This is the Temporal Difference (TD) error squared
        loss = self.criterion(current_q_values.squeeze(), target_q_values)

        # Optimize the policy network
        self.optimizer.zero_grad()
        loss.backward()

        # Clip gradients to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), max_norm=10.0)

        self.optimizer.step()

        self.training_step += 1

        return loss.item()

    def update_target_network(self):
        """
        Update target network by copying weights from policy network.

        The target network Q(s,a; θ⁻) is periodically synchronized with
        the policy network Q(s,a; θ). This provides stable target values
        during training.

        Without a target network, we would be chasing a moving target:
        - Policy network updates → target values change → unstable learning

        With target network:
        - Target values remain fixed between updates → stable learning
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """
        Decay exploration rate (epsilon).

        As training progresses, reduce exploration and increase exploitation.
        This implements an annealing schedule:

        ε_t = max(ε_end, ε_t-1 * decay)

        Intuition:
        - Early: Explore widely to discover good strategies
        - Late: Exploit learned knowledge to maximize performance
        """
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save(self, filepath: str):
        """
        Save agent state to file.

        Args:
            filepath: Path to save checkpoint
        """
        checkpoint = {
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
        }
        torch.save(checkpoint, filepath)
        print(f"Agent saved to {filepath}")

    def load(self, filepath: str):
        """
        Load agent state from file.

        Args:
            filepath: Path to load checkpoint from
        """
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.training_step = checkpoint['training_step']
        self.episode_count = checkpoint['episode_count']
        print(f"Agent loaded from {filepath}")

    def get_stats(self) -> dict:
        """
        Get current agent statistics.

        Returns:
            stats: Dictionary of agent statistics
        """
        return {
            'epsilon': self.epsilon,
            'training_step': self.training_step,
            'episode_count': self.episode_count,
            'buffer_size': len(self.replay_buffer),
        }


if __name__ == "__main__":
    # Example usage
    print("Testing DQN Agent")
    print("="*70)

    # Create agent
    agent = DQNAgent(
        state_dim=520,  # Match environment observation space
        action_dim=5000,  # Match environment action space
        learning_rate=1e-4,
        gamma=0.99,
        epsilon_start=1.0,
        epsilon_end=0.01,
        epsilon_decay=0.995
    )

    print(f"Agent initialized")
    print(f"State dimension: {agent.state_dim}")
    print(f"Action dimension: {agent.action_dim}")
    print(f"Epsilon: {agent.epsilon}")
    print(f"Device: {agent.device}")

    # Test action selection
    dummy_state = np.random.randn(520)
    action = agent.select_action(dummy_state, training=True)
    print(f"\nSelected action: {action}")

    # Test storing transitions and training
    for i in range(100):
        state = np.random.randn(520)
        action = agent.select_action(state, training=True)
        reward = np.random.randn()
        next_state = np.random.randn(520)
        done = i % 20 == 0

        agent.store_transition(state, action, reward, next_state, done)

    print(f"\nBuffer size: {len(agent.replay_buffer)}")

    # Test training
    loss = agent.train(batch_size=32)
    if loss:
        print(f"Training loss: {loss:.4f}")

    # Test epsilon decay
    agent.decay_epsilon()
    print(f"Epsilon after decay: {agent.epsilon:.4f}")

    # Test target network update
    agent.update_target_network()
    print("Target network updated")

    print("\nDQN Agent test completed!")
