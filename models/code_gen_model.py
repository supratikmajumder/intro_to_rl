"""
Neural Network Models for Code Generation RL Agent

This module defines the neural network architectures used to approximate
the Q-function in Deep Q-Learning. The Q-network learns to predict the
expected return (Q-value) for each action in a given state.

Key Concepts:
- Q-Function: Q(s,a) = Expected cumulative reward from taking action a in state s
- Function Approximation: Using neural networks to handle large state/action spaces
- Deep Learning for RL: Combining reinforcement learning with deep neural networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class QNetwork(nn.Module):
    """
    Q-Network for approximating the action-value function Q(s,a).

    Architecture:
        Input → FC1 → ReLU → Dropout → FC2 → ReLU → Dropout → FC3 → Output

    The network takes a state representation as input and outputs Q-values
    for all possible actions. The agent then selects the action with the
    highest Q-value (greedy policy) or explores randomly (epsilon-greedy).

    Input: State vector (e.g., problem + code embeddings)
    Output: Q-value for each action (e.g., code tokens)

    The network learns through gradient descent to minimize the temporal
    difference (TD) error:
        TD Error = Q(s,a) - [r + γ * max_a' Q(s',a')]

    Attributes:
        fc1, fc2, fc3: Fully connected layers
        dropout1, dropout2: Dropout layers for regularization
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        """
        Initialize Q-Network.

        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of possible actions
            hidden_dim: Number of neurons in hidden layers
        """
        super(QNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Layer 1: State embedding → Hidden representation
        # This layer learns to extract relevant features from the state
        # (problem specification + current code state)
        self.fc1 = nn.Linear(state_dim, hidden_dim)

        # Layer 2: Hidden representation → Deeper representation
        # This layer learns more complex patterns and relationships
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Layer 3: Deep representation → Q-values for each action
        # This layer outputs Q(s,a) for all actions simultaneously
        self.fc3 = nn.Linear(hidden_dim, action_dim)

        # Dropout for regularization (prevent overfitting)
        self.dropout1 = nn.Dropout(0.2)
        self.dropout2 = nn.Dropout(0.2)

        # Initialize weights using Xavier/Glorot initialization
        # This helps with training stability and convergence
        self._initialize_weights()

    def _initialize_weights(self):
        """
        Initialize network weights for stable training.

        Xavier initialization is designed for networks with tanh/sigmoid activations,
        but also works well with ReLU. It helps maintain variance of activations
        and gradients across layers.
        """
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the network.

        This computes Q(s,a) for all actions a given state s.

        Args:
            state: Batch of state vectors, shape (batch_size, state_dim)

        Returns:
            q_values: Q-values for all actions, shape (batch_size, action_dim)
        """
        # Layer 1: State → Hidden
        x = F.relu(self.fc1(state))
        x = self.dropout1(x)

        # Layer 2: Hidden → Hidden
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)

        # Layer 3: Hidden → Q-values
        # Note: No activation on output layer
        # Q-values can be any real number (positive or negative)
        q_values = self.fc3(x)

        return q_values


class DuelingQNetwork(nn.Module):
    """
    Dueling Q-Network architecture for improved Q-value estimation.

    Dueling DQN separates the Q-function into two components:
    1. Value function V(s): Expected return from state s
    2. Advantage function A(s,a): How much better action a is compared to average

    Q(s,a) = V(s) + [A(s,a) - mean(A(s,·))]

    Advantages:
    - Better learning: Separates state value from action advantages
    - Faster convergence: Value function learned from all actions
    - More robust: Less sensitive to action selection

    This is particularly useful when many actions have similar Q-values,
    which is common in code generation where many tokens might be reasonable.

    Reference: Wang et al. "Dueling Network Architectures for Deep RL" (2016)
    """

    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 512):
        """
        Initialize Dueling Q-Network.

        Args:
            state_dim: Dimension of state/observation space
            action_dim: Number of possible actions
            hidden_dim: Number of neurons in hidden layers
        """
        super(DuelingQNetwork, self).__init__()

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Shared feature extraction layers
        # These layers learn a common representation used by both
        # value and advantage streams
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        # Value stream: V(s)
        # Estimates the expected return from being in state s
        self.value_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.value_fc2 = nn.Linear(hidden_dim // 2, 1)

        # Advantage stream: A(s,a)
        # Estimates how much better each action is compared to average
        self.advantage_fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.advantage_fc2 = nn.Linear(hidden_dim // 2, action_dim)

        self.dropout = nn.Dropout(0.2)

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize network weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through dueling architecture.

        Computes Q(s,a) by combining value and advantage streams:
        Q(s,a) = V(s) + [A(s,a) - mean_a A(s,a)]

        The subtraction of mean advantage ensures identifiability:
        it's the unique way to decompose Q into V and A.

        Args:
            state: Batch of state vectors, shape (batch_size, state_dim)

        Returns:
            q_values: Q-values for all actions, shape (batch_size, action_dim)
        """
        # Shared feature extraction
        x = F.relu(self.fc1(state))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        # Value stream: V(s)
        value = F.relu(self.value_fc1(x))
        value = self.value_fc2(value)  # Shape: (batch_size, 1)

        # Advantage stream: A(s,a)
        advantage = F.relu(self.advantage_fc1(x))
        advantage = self.advantage_fc2(advantage)  # Shape: (batch_size, action_dim)

        # Combine value and advantage
        # Subtract mean advantage to ensure identifiability
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))

        return q_values


class CodeEmbedding(nn.Module):
    """
    Neural network for embedding code into a continuous vector space.

    This can be used as a learned encoder for the code state, replacing
    or augmenting hand-crafted features. The embedding network learns to
    extract relevant features from raw code representations.

    In practice, you might use:
    - Pre-trained code models (CodeBERT, GraphCodeBERT, CodeT5)
    - Custom embeddings trained end-to-end with RL
    - Hybrid approaches combining pre-trained and learned components
    """

    def __init__(self, input_dim: int, embedding_dim: int = 256):
        """
        Initialize code embedding network.

        Args:
            input_dim: Dimension of input code representation
            embedding_dim: Dimension of output embedding
        """
        super(CodeEmbedding, self).__init__()

        self.fc1 = nn.Linear(input_dim, 512)
        self.fc2 = nn.Linear(512, embedding_dim)
        self.dropout = nn.Dropout(0.1)

    def forward(self, code_features: torch.Tensor) -> torch.Tensor:
        """
        Encode code features into embedding vector.

        Args:
            code_features: Raw code features, shape (batch_size, input_dim)

        Returns:
            embedding: Code embedding, shape (batch_size, embedding_dim)
        """
        x = F.relu(self.fc1(code_features))
        x = self.dropout(x)
        embedding = torch.tanh(self.fc2(x))  # tanh to normalize embedding
        return embedding


def test_q_network():
    """Test basic Q-Network functionality."""
    print("Testing Q-Network")
    print("="*70)

    state_dim = 520
    action_dim = 5000
    batch_size = 32

    # Create network
    q_net = QNetwork(state_dim, action_dim, hidden_dim=512)

    # Test forward pass
    dummy_states = torch.randn(batch_size, state_dim)
    q_values = q_net(dummy_states)

    print(f"Input shape: {dummy_states.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min():.2f}, {q_values.max():.2f}]")
    print(f"Q-values mean: {q_values.mean():.2f}")
    print(f"Q-values std: {q_values.std():.2f}")

    # Test action selection (argmax)
    best_actions = q_values.argmax(dim=1)
    print(f"Best actions shape: {best_actions.shape}")
    print(f"Sample best actions: {best_actions[:5].tolist()}")

    # Count parameters
    total_params = sum(p.numel() for p in q_net.parameters())
    trainable_params = sum(p.numel() for p in q_net.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


def test_dueling_q_network():
    """Test Dueling Q-Network functionality."""
    print("\n\nTesting Dueling Q-Network")
    print("="*70)

    state_dim = 520
    action_dim = 5000
    batch_size = 32

    # Create network
    dueling_q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=512)

    # Test forward pass
    dummy_states = torch.randn(batch_size, state_dim)
    q_values = dueling_q_net(dummy_states)

    print(f"Input shape: {dummy_states.shape}")
    print(f"Output shape: {q_values.shape}")
    print(f"Q-values range: [{q_values.min():.2f}, {q_values.max():.2f}]")

    # Count parameters
    total_params = sum(p.numel() for p in dueling_q_net.parameters())
    print(f"Total parameters: {total_params:,}")


def compare_networks():
    """Compare standard DQN and Dueling DQN on same inputs."""
    print("\n\nComparing Standard vs Dueling Q-Network")
    print("="*70)

    state_dim = 520
    action_dim = 100  # Smaller for clearer comparison
    batch_size = 4

    # Create both networks
    q_net = QNetwork(state_dim, action_dim, hidden_dim=128)
    dueling_q_net = DuelingQNetwork(state_dim, action_dim, hidden_dim=128)

    # Same input
    dummy_states = torch.randn(batch_size, state_dim)

    # Get Q-values from both
    q_values_standard = q_net(dummy_states)
    q_values_dueling = dueling_q_net(dummy_states)

    print("Standard Q-Network - Sample Q-values:")
    print(q_values_standard[0, :10].detach().numpy())

    print("\nDueling Q-Network - Sample Q-values:")
    print(q_values_dueling[0, :10].detach().numpy())

    print("\nBoth networks produce Q-values, but dueling architecture")
    print("separates value and advantage streams for better learning.")


if __name__ == "__main__":
    test_q_network()
    test_dueling_q_network()
    compare_networks()
    print("\n" + "="*70)
    print("All tests completed!")
