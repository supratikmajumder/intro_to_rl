"""
Utility Functions for RL Training

This module provides helper functions for:
- Data loading and preprocessing
- Logging and visualization
- Evaluation and metrics
- Checkpointing and model management
"""

import json
import yaml
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import os
from datetime import datetime


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to config file

    Returns:
        config: Configuration dictionary
    """
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_problems(problems_path: str) -> List[Dict[str, Any]]:
    """
    Load problem specifications from JSON file.

    Expected format:
    [
        {
            "function_name": "add",
            "docstring": "Add two numbers",
            "test_cases": [[[1, 2], 3], [[0, 0], 0]],
            "starter_code": ""
        },
        ...
    ]

    Args:
        problems_path: Path to problems JSON file

    Returns:
        problems: List of problem dictionaries
    """
    with open(problems_path, 'r') as f:
        problems = json.load(f)
    return problems


def save_problems(problems: List[Dict[str, Any]], output_path: str):
    """
    Save problem specifications to JSON file.

    Args:
        problems: List of problem dictionaries
        output_path: Path to save JSON file
    """
    with open(output_path, 'w') as f:
        json.dump(problems, f, indent=2)
    print(f"Saved {len(problems)} problems to {output_path}")


def set_random_seed(seed: int):
    """
    Set random seed for reproducibility.

    Sets seeds for:
    - Python random module
    - NumPy
    - PyTorch (if available)

    Args:
        seed: Random seed value
    """
    import random
    random.seed(seed)
    np.random.seed(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


class MetricsLogger:
    """
    Logger for training metrics and statistics.

    Tracks metrics over time and provides methods for:
    - Logging episode statistics
    - Computing moving averages
    - Saving metrics to file
    - Plotting training curves
    """

    def __init__(self, log_dir: str = 'experiments/logs'):
        """
        Initialize metrics logger.

        Args:
            log_dir: Directory to save log files
        """
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Metrics storage
        self.episodes = []
        self.rewards = []
        self.losses = []
        self.test_success_rates = []
        self.episode_lengths = []
        self.epsilons = []

        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.log_file = os.path.join(log_dir, f'training_{timestamp}.log')

        # Write header
        with open(self.log_file, 'w') as f:
            f.write("episode,reward,loss,success_rate,length,epsilon\n")

    def log_episode(self,
                    episode: int,
                    reward: float,
                    loss: float = None,
                    success_rate: float = 0.0,
                    length: int = 0,
                    epsilon: float = 0.0):
        """
        Log metrics for a single episode.

        Args:
            episode: Episode number
            reward: Total episode reward
            loss: Training loss (None if not trained)
            success_rate: Test success rate
            length: Episode length (number of steps)
            epsilon: Current exploration rate
        """
        self.episodes.append(episode)
        self.rewards.append(reward)
        self.losses.append(loss if loss is not None else 0.0)
        self.test_success_rates.append(success_rate)
        self.episode_lengths.append(length)
        self.epsilons.append(epsilon)

        # Append to log file
        with open(self.log_file, 'a') as f:
            loss_str = f"{loss:.4f}" if loss is not None else "N/A"
            f.write(f"{episode},{reward:.2f},{loss_str},{success_rate:.2f},{length},{epsilon:.4f}\n")

    def get_recent_average(self, metric: str = 'reward', window: int = 100) -> float:
        """
        Compute moving average of a metric.

        Args:
            metric: Metric name ('reward', 'success_rate', 'length')
            window: Window size for moving average

        Returns:
            average: Moving average value
        """
        if metric == 'reward':
            values = self.rewards
        elif metric == 'success_rate':
            values = self.test_success_rates
        elif metric == 'length':
            values = self.episode_lengths
        else:
            raise ValueError(f"Unknown metric: {metric}")

        if len(values) == 0:
            return 0.0

        recent_values = values[-window:]
        return np.mean(recent_values)

    def plot_metrics(self, save_path: str = None, window: int = 100):
        """
        Plot training metrics.

        Creates a figure with 4 subplots:
        1. Episode rewards (with moving average)
        2. Test success rate
        3. Training loss
        4. Exploration rate (epsilon)

        Args:
            save_path: Path to save plot (None to display)
            window: Window size for moving average
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Training Metrics', fontsize=16)

        # 1. Rewards
        ax = axes[0, 0]
        ax.plot(self.episodes, self.rewards, alpha=0.3, label='Episode Reward')
        if len(self.rewards) >= window:
            moving_avg = self._moving_average(self.rewards, window)
            ax.plot(self.episodes[window-1:], moving_avg, label=f'{window}-Episode MA')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Total Reward')
        ax.set_title('Episode Rewards')
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 2. Success Rate
        ax = axes[0, 1]
        ax.plot(self.episodes, self.test_success_rates, label='Test Success Rate')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Success Rate')
        ax.set_title('Test Success Rate')
        ax.set_ylim(0, 1.1)
        ax.legend()
        ax.grid(True, alpha=0.3)

        # 3. Loss
        ax = axes[1, 0]
        if any(l > 0 for l in self.losses):
            ax.plot(self.episodes, self.losses, alpha=0.5, label='Training Loss')
            ax.set_xlabel('Episode')
            ax.set_ylabel('Loss')
            ax.set_title('Training Loss')
            ax.legend()
            ax.grid(True, alpha=0.3)

        # 4. Epsilon
        ax = axes[1, 1]
        ax.plot(self.episodes, self.epsilons, label='Epsilon')
        ax.set_xlabel('Episode')
        ax.set_ylabel('Exploration Rate')
        ax.set_title('Epsilon Decay')
        ax.legend()
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

    def _moving_average(self, values: List[float], window: int) -> List[float]:
        """Compute moving average."""
        return [np.mean(values[max(0, i-window+1):i+1]) for i in range(window-1, len(values))]

    def save_summary(self, filepath: str):
        """
        Save training summary statistics to file.

        Args:
            filepath: Path to save summary
        """
        summary = {
            'total_episodes': len(self.episodes),
            'final_reward': self.rewards[-1] if self.rewards else 0,
            'max_reward': max(self.rewards) if self.rewards else 0,
            'avg_reward_last_100': self.get_recent_average('reward', 100),
            'final_success_rate': self.test_success_rates[-1] if self.test_success_rates else 0,
            'max_success_rate': max(self.test_success_rates) if self.test_success_rates else 0,
            'avg_episode_length': np.mean(self.episode_lengths) if self.episode_lengths else 0,
        }

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"Training summary saved to {filepath}")


def evaluate_agent(agent, env, num_episodes: int = 20, deterministic: bool = True) -> Dict[str, float]:
    """
    Evaluate trained agent performance.

    Args:
        agent: RL agent
        env: Environment
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy (no exploration)

    Returns:
        metrics: Dictionary with evaluation metrics
    """
    total_rewards = []
    success_rates = []
    episode_lengths = []

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        while not done:
            action = agent.select_action(state, training=not deterministic)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            state = next_state

        total_rewards.append(episode_reward)
        success_rates.append(1.0 if info.get('tests_passed', 0) == info.get('total_tests', 0) else 0.0)
        episode_lengths.append(episode_length)

    metrics = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'mean_success_rate': np.mean(success_rates),
        'mean_episode_length': np.mean(episode_lengths),
    }

    return metrics


def print_training_progress(episode: int,
                           total_episodes: int,
                           reward: float,
                           success_rate: float,
                           epsilon: float,
                           loss: float = None,
                           moving_avg_reward: float = None):
    """
    Print formatted training progress.

    Args:
        episode: Current episode
        total_episodes: Total training episodes
        reward: Episode reward
        success_rate: Test success rate
        epsilon: Current exploration rate
        loss: Training loss (optional)
        moving_avg_reward: Moving average reward (optional)
    """
    progress = episode / total_episodes * 100

    print(f"Episode {episode}/{total_episodes} ({progress:.1f}%)")
    print(f"  Reward: {reward:.2f}", end='')
    if moving_avg_reward is not None:
        print(f" (MA: {moving_avg_reward:.2f})", end='')
    print()
    print(f"  Success Rate: {success_rate:.2%}")
    print(f"  Epsilon: {epsilon:.4f}")
    if loss is not None:
        print(f"  Loss: {loss:.4f}")
    print("-" * 50)


def create_example_problems() -> List[Dict[str, Any]]:
    """
    Create example problem set for training.

    Returns:
        problems: List of problem specifications
    """
    problems = [
        {
            'function_name': 'add_numbers',
            'docstring': 'Return the sum of two numbers.',
            'test_cases': [
                ((2, 3), 5),
                ((0, 0), 0),
                ((-1, 1), 0),
                ((10, -5), 5),
            ],
            'starter_code': ''
        },
        {
            'function_name': 'is_even',
            'docstring': 'Return True if number is even, False otherwise.',
            'test_cases': [
                (2, True),
                (3, False),
                (0, True),
                (-2, True),
                (-3, False),
            ],
            'starter_code': ''
        },
        {
            'function_name': 'max_of_three',
            'docstring': 'Return the maximum of three numbers.',
            'test_cases': [
                ((1, 2, 3), 3),
                ((3, 2, 1), 3),
                ((2, 3, 1), 3),
                ((5, 5, 5), 5),
                ((-1, -2, -3), -1),
            ],
            'starter_code': ''
        },
        {
            'function_name': 'count_vowels',
            'docstring': 'Count the number of vowels in a string.',
            'test_cases': [
                ('hello', 2),
                ('world', 1),
                ('python', 1),
                ('aeiou', 5),
                ('xyz', 0),
            ],
            'starter_code': ''
        },
        {
            'function_name': 'reverse_string',
            'docstring': 'Return the reverse of a string.',
            'test_cases': [
                ('hello', 'olleh'),
                ('world', 'dlrow'),
                ('python', 'nohtyp'),
                ('a', 'a'),
                ('', ''),
            ],
            'starter_code': ''
        },
    ]

    return problems


if __name__ == "__main__":
    # Test utility functions
    print("Testing Utility Functions")
    print("="*70)

    # Test random seed
    set_random_seed(42)
    print("Random seed set to 42")

    # Test metrics logger
    print("\nTesting MetricsLogger")
    logger = MetricsLogger(log_dir='experiments/logs')

    for i in range(100):
        reward = np.random.randn() * 10 + 50 + i * 0.5  # Increasing trend
        success_rate = min(1.0, i / 100.0 + np.random.rand() * 0.2)
        epsilon = max(0.01, 1.0 - i / 100.0)

        logger.log_episode(
            episode=i,
            reward=reward,
            loss=np.random.rand(),
            success_rate=success_rate,
            length=np.random.randint(10, 50),
            epsilon=epsilon
        )

    print(f"Logged {len(logger.episodes)} episodes")
    print(f"Average reward (last 100): {logger.get_recent_average('reward', 100):.2f}")

    # Test create example problems
    print("\nTesting create_example_problems")
    problems = create_example_problems()
    print(f"Created {len(problems)} example problems")
    for i, problem in enumerate(problems):
        print(f"  {i+1}. {problem['function_name']}: {len(problem['test_cases'])} test cases")

    print("\nUtility functions test completed!")
