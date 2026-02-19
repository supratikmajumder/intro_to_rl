"""
Evaluation Script for Trained RL Agent

This script evaluates a trained agent on test problems and provides
detailed performance analysis.

Usage:
    python evaluate.py --model experiments/models/best_model.pt \
                       --problems data/examples/eval_problems.json \
                       --num_episodes 50

The script provides:
- Success rate on test problems
- Average reward and episode length
- Detailed problem-by-problem analysis
- Example generated code
"""

import argparse
import os
import sys
import numpy as np
import json

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from environments.code_gen_env import CodeGenerationEnv
from agents.dqn_agent import DQNAgent
from utils.helpers import load_problems, set_random_seed


def evaluate_model(model_path: str,
                   problems_path: str,
                   num_episodes: int = 50,
                   deterministic: bool = True,
                   render: bool = False,
                   verbose: bool = True):
    """
    Evaluate a trained model on test problems.

    Args:
        model_path: Path to saved model checkpoint
        problems_path: Path to evaluation problems JSON
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic policy (no exploration)
        render: Render episodes for visualization
        verbose: Print detailed information

    Returns:
        results: Dictionary with evaluation results
    """
    # Load problems
    if verbose:
        print("="*70)
        print("Evaluating Trained Agent")
        print("="*70)
        print(f"\nLoading problems from {problems_path}")

    problems = load_problems(problems_path)

    if verbose:
        print(f"Loaded {len(problems)} evaluation problems:")
        for i, prob in enumerate(problems):
            print(f"  {i+1}. {prob['function_name']}: {len(prob['test_cases'])} test cases")

    # Initialize environment
    env = CodeGenerationEnv(problems=problems, max_steps=20, vocab_size=1000)

    # Initialize agent
    if verbose:
        print(f"\nInitializing agent and loading model from {model_path}")

    agent = DQNAgent(
        state_dim=520,
        action_dim=1000,
        learning_rate=1e-4,
        gamma=0.99
    )
    agent.load(model_path)

    if verbose:
        stats = agent.get_stats()
        print(f"Agent loaded successfully:")
        print(f"  Training steps: {stats['training_step']}")
        print(f"  Episodes: {stats['episode_count']}")
        print(f"  Epsilon: {stats['epsilon']:.4f}")

    # Evaluation loop
    if verbose:
        print(f"\n{'='*70}")
        print(f"Running {num_episodes} evaluation episodes...")
        print("="*70)

    all_rewards = []
    all_success_rates = []
    all_episode_lengths = []
    problem_results = {}

    for episode in range(num_episodes):
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        done = False

        # Track which problem this episode is evaluating
        problem_name = env.current_problem['function_name']
        if problem_name not in problem_results:
            problem_results[problem_name] = {
                'successes': 0,
                'attempts': 0,
                'total_reward': 0,
                'generated_code': []
            }

        while not done:
            action = agent.select_action(state, training=not deterministic)
            next_state, reward, done, info = env.step(action)

            episode_reward += reward
            episode_length += 1
            state = next_state

        # Record results
        all_rewards.append(episode_reward)
        all_episode_lengths.append(episode_length)

        tests_passed = info.get('tests_passed', 0)
        total_tests = info.get('total_tests', 1)
        success_rate = tests_passed / total_tests if total_tests > 0 else 0.0
        all_success_rates.append(success_rate)

        # Record problem-specific results
        problem_results[problem_name]['attempts'] += 1
        problem_results[problem_name]['total_reward'] += episode_reward
        if success_rate == 1.0:
            problem_results[problem_name]['successes'] += 1

        # Save some example generated code
        if len(problem_results[problem_name]['generated_code']) < 3:
            problem_results[problem_name]['generated_code'].append({
                'code': env.current_code,
                'tests_passed': tests_passed,
                'total_tests': total_tests,
                'reward': episode_reward
            })

        # Render if requested
        if render and episode % 10 == 0:
            env.render()

        # Print progress
        if verbose and episode % 10 == 0:
            print(f"Episode {episode}/{num_episodes} - "
                  f"Problem: {problem_name}, "
                  f"Tests: {tests_passed}/{total_tests}, "
                  f"Reward: {episode_reward:.2f}")

    # Compute aggregate statistics
    results = {
        'num_episodes': num_episodes,
        'mean_reward': np.mean(all_rewards),
        'std_reward': np.std(all_rewards),
        'median_reward': np.median(all_rewards),
        'mean_success_rate': np.mean(all_success_rates),
        'std_success_rate': np.std(all_success_rates),
        'mean_episode_length': np.mean(all_episode_lengths),
        'std_episode_length': np.std(all_episode_lengths),
        'problem_results': problem_results
    }

    # Print summary
    if verbose:
        print("\n" + "="*70)
        print("Evaluation Summary")
        print("="*70)
        print(f"\nOverall Performance:")
        print(f"  Episodes: {num_episodes}")
        print(f"  Mean Reward: {results['mean_reward']:.2f} ± {results['std_reward']:.2f}")
        print(f"  Median Reward: {results['median_reward']:.2f}")
        print(f"  Mean Success Rate: {results['mean_success_rate']:.2%} ± {results['std_success_rate']:.2%}")
        print(f"  Mean Episode Length: {results['mean_episode_length']:.1f} ± {results['std_episode_length']:.1f}")

        print(f"\nPer-Problem Performance:")
        print("-" * 70)
        for problem_name, stats in problem_results.items():
            success_rate = stats['successes'] / stats['attempts'] if stats['attempts'] > 0 else 0
            avg_reward = stats['total_reward'] / stats['attempts'] if stats['attempts'] > 0 else 0

            print(f"\n{problem_name}:")
            print(f"  Attempts: {stats['attempts']}")
            print(f"  Successes: {stats['successes']}/{stats['attempts']} ({success_rate:.1%})")
            print(f"  Average Reward: {avg_reward:.2f}")

            # Show best generated code example
            if stats['generated_code']:
                best_code = max(stats['generated_code'], key=lambda x: x['reward'])
                print(f"  Best Generated Code (tests: {best_code['tests_passed']}/{best_code['total_tests']}):")
                if best_code['code']:
                    for line in best_code['code'].split('\n')[:10]:  # Show first 10 lines
                        print(f"    {line}")
                    line_count = best_code['code'].count('\n')
                    if line_count > 10:
                        print(f"    ... ({line_count - 10} more lines)")
                else:
                    print("    (no code generated)")

    return results


def compare_models(model_paths: list,
                   problems_path: str,
                   num_episodes: int = 50):
    """
    Compare multiple trained models on the same evaluation set.

    Args:
        model_paths: List of paths to model checkpoints
        problems_path: Path to evaluation problems
        num_episodes: Number of episodes per model

    Returns:
        comparison: Dictionary with comparison results
    """
    print("="*70)
    print("Model Comparison")
    print("="*70)

    comparison = {}

    for i, model_path in enumerate(model_paths):
        print(f"\n[{i+1}/{len(model_paths)}] Evaluating {model_path}")
        print("-" * 70)

        results = evaluate_model(
            model_path=model_path,
            problems_path=problems_path,
            num_episodes=num_episodes,
            verbose=False
        )

        model_name = os.path.basename(model_path).replace('.pt', '')
        comparison[model_name] = {
            'mean_reward': results['mean_reward'],
            'mean_success_rate': results['mean_success_rate'],
            'mean_episode_length': results['mean_episode_length']
        }

        print(f"  Mean Reward: {results['mean_reward']:.2f}")
        print(f"  Mean Success Rate: {results['mean_success_rate']:.2%}")

    # Print comparison table
    print("\n" + "="*70)
    print("Comparison Summary")
    print("="*70)
    print(f"\n{'Model':<30} {'Reward':<15} {'Success Rate':<15} {'Avg Length':<15}")
    print("-" * 70)

    for model_name, metrics in comparison.items():
        print(f"{model_name:<30} "
              f"{metrics['mean_reward']:<15.2f} "
              f"{metrics['mean_success_rate']:<15.2%} "
              f"{metrics['mean_episode_length']:<15.1f}")

    return comparison


def main():
    """Parse arguments and run evaluation."""
    parser = argparse.ArgumentParser(description='Evaluate trained RL agent')
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to trained model checkpoint'
    )
    parser.add_argument(
        '--problems',
        type=str,
        default='data/examples/eval_problems.json',
        help='Path to evaluation problems JSON'
    )
    parser.add_argument(
        '--num_episodes',
        type=int,
        default=50,
        help='Number of evaluation episodes'
    )
    parser.add_argument(
        '--deterministic',
        action='store_true',
        default=True,
        help='Use deterministic policy (no exploration)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render episodes for visualization'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed'
    )
    parser.add_argument(
        '--compare',
        nargs='+',
        help='Compare multiple models (provide paths)'
    )
    parser.add_argument(
        '--output',
        type=str,
        help='Path to save evaluation results JSON'
    )

    args = parser.parse_args()

    # Set random seed
    set_random_seed(args.seed)

    # Run evaluation or comparison
    if args.compare:
        results = compare_models(
            model_paths=args.compare,
            problems_path=args.problems,
            num_episodes=args.num_episodes
        )
    else:
        results = evaluate_model(
            model_path=args.model,
            problems_path=args.problems,
            num_episodes=args.num_episodes,
            deterministic=args.deterministic,
            render=args.render
        )

    # Save results if output path specified
    if args.output:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_serializable(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {key: convert_to_serializable(value) for key, value in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            return obj

        results = convert_to_serializable(results)

        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
