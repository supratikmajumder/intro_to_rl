"""
Main Training Script for Code Generation RL Agent

This script orchestrates the complete training pipeline:
1. Load configuration and data
2. Initialize environment, agent, and reward function
3. Train agent using RL algorithm (DQN)
4. Log metrics and save checkpoints
5. Evaluate agent performance

Usage:
    python train.py --config configs/train_config.yaml

RL Training Loop:
    For each episode:
        1. Reset environment to get initial state s₀
        2. While episode not done:
            a. Select action using epsilon-greedy policy
            b. Execute action in environment
            c. Observe reward and next state
            d. Store transition in replay buffer
            e. Sample mini-batch and train agent
        3. Update target network periodically
        4. Decay exploration rate (epsilon)
        5. Log metrics and save checkpoints
"""

import argparse
import os
import sys
import numpy as np
import torch

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.code_gen_env import CodeGenerationEnv
from agents.dqn_agent import DQNAgent
from rewards.code_quality_reward import CodeQualityReward, SparseReward, CurriculumReward
from utils.helpers import (
    load_config,
    load_problems,
    set_random_seed,
    MetricsLogger,
    evaluate_agent,
    print_training_progress
)


def train(config_path: str = 'configs/train_config.yaml'):
    """
    Main training function.

    Implements the complete DQN training algorithm:
    1. Initialize environment, agent, and replay buffer
    2. For each episode:
        - Reset environment
        - Collect experience using epsilon-greedy policy
        - Train agent on mini-batches from replay buffer
        - Update target network periodically
    3. Log metrics and save checkpoints

    Args:
        config_path: Path to configuration file
    """
    # Load configuration
    print("="*70)
    print("RL Training for Code Generation")
    print("="*70)
    print(f"\nLoading configuration from {config_path}")
    config = load_config(config_path)

    # Set random seed for reproducibility
    seed = config.get('seed', 42)
    set_random_seed(seed)
    print(f"Random seed set to {seed}")

    # Load training problems
    problems_file = config['data']['problems_file']
    print(f"Loading training problems from {problems_file}")
    training_problems = load_problems(problems_file)
    print(f"Loaded {len(training_problems)} training problems")

    # Initialize environment
    print("\nInitializing environment...")
    env = CodeGenerationEnv(
        problems=training_problems,
        max_steps=config['environment']['max_steps'],
        vocab_size=config['environment']['vocab_size']
    )
    print(f"Environment created:")
    print(f"  Observation space: {env.observation_space.shape}")
    print(f"  Action space: {env.action_space.n}")

    # Initialize agent
    print("\nInitializing DQN agent...")
    agent = DQNAgent(
        state_dim=config['agent']['state_dim'],
        action_dim=config['agent']['action_dim'],
        learning_rate=config['agent']['learning_rate'],
        gamma=config['agent']['gamma'],
        epsilon_start=config['agent']['epsilon_start'],
        epsilon_end=config['agent']['epsilon_end'],
        epsilon_decay=config['agent']['epsilon_decay'],
        buffer_capacity=config['agent']['buffer_capacity'],
        device=config['agent']['device']
    )
    print(f"Agent initialized:")
    print(f"  Learning rate: {config['agent']['learning_rate']}")
    print(f"  Gamma: {config['agent']['gamma']}")
    print(f"  Device: {config['agent']['device']}")

    # Initialize reward function
    reward_type = config['reward']['type']
    print(f"\nInitializing {reward_type} reward function...")
    if reward_type == 'code_quality':
        reward_fn = CodeQualityReward(
            correctness_weight=config['reward']['correctness_weight'],
            quality_weight=config['reward']['quality_weight'],
            efficiency_weight=config['reward']['efficiency_weight'],
            progress_weight=config['reward']['progress_weight']
        )
    elif reward_type == 'sparse':
        reward_fn = SparseReward()
    elif reward_type == 'curriculum':
        reward_fn = CurriculumReward(total_episodes=config['training']['num_episodes'])
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")

    # Initialize metrics logger
    log_dir = config['training']['log_dir']
    os.makedirs(log_dir, exist_ok=True)
    logger = MetricsLogger(log_dir=log_dir)
    print(f"\nMetrics logger initialized (log_dir: {log_dir})")

    # Create checkpoint directory
    checkpoint_dir = config['training']['checkpoint_dir']
    os.makedirs(checkpoint_dir, exist_ok=True)
    print(f"Checkpoint directory: {checkpoint_dir}")

    # Training loop
    print("\n" + "="*70)
    print("Starting training...")
    print("="*70)

    num_episodes = config['training']['num_episodes']
    batch_size = config['agent']['batch_size']
    target_update_freq = config['agent']['target_update_frequency']
    log_frequency = config['training']['log_frequency']
    eval_frequency = config['training']['eval_frequency']
    save_frequency = config['training']['save_frequency']

    best_eval_reward = -float('inf')

    for episode in range(num_episodes):
        # Reset environment for new episode
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False
        previous_code = ""

        # Episode loop
        while not done:
            # Select action using epsilon-greedy policy
            action = agent.select_action(state, training=True)

            # Execute action in environment
            next_state, _, done, info = env.step(action)

            # Compute custom reward using reward function
            if reward_type == 'code_quality':
                reward, _ = reward_fn.compute_reward(
                    current_code=env.current_code,
                    test_results=env.test_results,
                    previous_code=previous_code,
                    step_count=env.step_count,
                    max_steps=env.max_steps
                )
            elif reward_type == 'curriculum':
                reward = reward_fn.compute_reward(
                    code=env.current_code,
                    test_results=env.test_results,
                    episode=episode
                )
            else:  # sparse
                reward = reward_fn.compute_reward(env.test_results)

            # Store transition in replay buffer
            agent.store_transition(state, action, reward, next_state, done)

            # Train agent (if enough samples in buffer)
            loss = None
            if len(agent.replay_buffer) >= config['agent']['min_buffer_size']:
                loss = agent.train(batch_size=batch_size)

            # Update state
            previous_code = env.current_code
            state = next_state
            episode_reward += reward
            episode_length += 1

        # Update target network periodically
        if episode % target_update_freq == 0:
            agent.update_target_network()

        # Decay epsilon
        agent.decay_epsilon()

        # Calculate success rate
        tests_passed = info.get('tests_passed', 0)
        total_tests = info.get('total_tests', 1)
        success_rate = tests_passed / total_tests if total_tests > 0 else 0.0

        # Log metrics
        logger.log_episode(
            episode=episode,
            reward=episode_reward,
            loss=loss,
            success_rate=success_rate,
            length=episode_length,
            epsilon=agent.epsilon
        )

        # Print progress
        if config.get('verbose', True) and episode % log_frequency == 0:
            moving_avg_reward = logger.get_recent_average('reward', window=100)
            print_training_progress(
                episode=episode,
                total_episodes=num_episodes,
                reward=episode_reward,
                success_rate=success_rate,
                epsilon=agent.epsilon,
                loss=loss,
                moving_avg_reward=moving_avg_reward
            )

        # Evaluate agent
        if episode % eval_frequency == 0 and episode > 0:
            print(f"\nEvaluating agent at episode {episode}...")
            eval_metrics = evaluate_agent(
                agent=agent,
                env=env,
                num_episodes=config['evaluation']['num_episodes'],
                deterministic=config['evaluation']['deterministic']
            )
            print(f"  Evaluation results:")
            print(f"    Mean reward: {eval_metrics['mean_reward']:.2f}")
            print(f"    Mean success rate: {eval_metrics['mean_success_rate']:.2%}")
            print(f"    Mean episode length: {eval_metrics['mean_episode_length']:.1f}")

            # Save best model
            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                best_model_path = os.path.join(checkpoint_dir, 'best_model.pt')
                agent.save(best_model_path)
                print(f"    New best model saved! (reward: {best_eval_reward:.2f})")

        # Save checkpoint
        if episode % save_frequency == 0 and episode > 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_ep{episode}.pt')
            agent.save(checkpoint_path)

    # Training completed
    print("\n" + "="*70)
    print("Training completed!")
    print("="*70)

    # Save final model
    final_model_path = os.path.join(checkpoint_dir, 'final_model.pt')
    agent.save(final_model_path)

    # Save training summary
    summary_path = os.path.join(log_dir, 'training_summary.json')
    logger.save_summary(summary_path)

    # Plot training curves
    plot_path = os.path.join(log_dir, 'training_curves.png')
    logger.plot_metrics(save_path=plot_path, window=100)

    # Final evaluation
    print("\nFinal evaluation...")
    final_eval_metrics = evaluate_agent(
        agent=agent,
        env=env,
        num_episodes=config['evaluation']['num_episodes'] * 2,  # More episodes for final eval
        deterministic=True
    )
    print(f"\nFinal evaluation results:")
    print(f"  Mean reward: {final_eval_metrics['mean_reward']:.2f} ± {final_eval_metrics['std_reward']:.2f}")
    print(f"  Mean success rate: {final_eval_metrics['mean_success_rate']:.2%}")
    print(f"  Mean episode length: {final_eval_metrics['mean_episode_length']:.1f}")

    print("\nTraining artifacts saved:")
    print(f"  - Final model: {final_model_path}")
    print(f"  - Best model: {os.path.join(checkpoint_dir, 'best_model.pt')}")
    print(f"  - Training log: {logger.log_file}")
    print(f"  - Training summary: {summary_path}")
    print(f"  - Training curves: {plot_path}")

    return agent, logger


def main():
    """Parse arguments and run training."""
    parser = argparse.ArgumentParser(description='Train RL agent for code generation')
    parser.add_argument(
        '--config',
        type=str,
        default='configs/train_config.yaml',
        help='Path to configuration file'
    )
    args = parser.parse_args()

    # Run training
    train(config_path=args.config)


if __name__ == "__main__":
    main()
