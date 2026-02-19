"""
Reward Function Design for Code Generation RL

The reward function is arguably the most critical component of any RL system.
It defines what behavior we want the agent to learn. A poorly designed reward
function can lead to:
- Reward hacking: Agent finds loopholes to maximize reward without solving the task
- Sparse rewards: Agent struggles to learn due to infrequent feedback
- Unintended behavior: Agent optimizes for reward but not our true objective

This module implements various reward functions for code generation, balancing:
1. Correctness (does the code work?)
2. Quality (is the code well-written?)
3. Efficiency (is the code fast/optimal?)
4. Safety (does the code avoid vulnerabilities?)

Key RL Concept - Reward Shaping:
Rather than binary rewards (0 or 1), we provide shaped rewards that guide
learning. For example, partial credit for:
- Passing some tests (even if not all)
- Having valid syntax (even if tests fail)
- Making progress toward the solution

This accelerates learning by providing more informative feedback signals.
"""

import ast
import math
from typing import Dict, List, Tuple, Any
import numpy as np


class CodeQualityReward:
    """
    Multi-objective reward function for code generation.

    Combines multiple reward components:
    1. Correctness: Test passage rate
    2. Code quality: Readability, maintainability, style
    3. Efficiency: Runtime, code length, complexity
    4. Progress: Incremental improvements

    The final reward is a weighted combination:
    R = w₁·R_correctness + w₂·R_quality + w₃·R_efficiency + w₄·R_progress

    Attributes:
        weights (Dict): Weights for each reward component
    """

    def __init__(self,
                 correctness_weight: float = 10.0,
                 quality_weight: float = 2.0,
                 efficiency_weight: float = 1.0,
                 progress_weight: float = 0.5):
        """
        Initialize reward function with component weights.

        Args:
            correctness_weight: Weight for test correctness
            quality_weight: Weight for code quality metrics
            efficiency_weight: Weight for code efficiency
            progress_weight: Weight for incremental progress
        """
        self.weights = {
            'correctness': correctness_weight,
            'quality': quality_weight,
            'efficiency': efficiency_weight,
            'progress': progress_weight,
        }

    def compute_reward(self,
                      current_code: str,
                      test_results: Dict[str, Any],
                      previous_code: str = "",
                      step_count: int = 0,
                      max_steps: int = 50) -> Tuple[float, Dict[str, float]]:
        """
        Compute total reward and component breakdown.

        Args:
            current_code: The generated code
            test_results: Dictionary with test execution results
            previous_code: Code from previous step (for progress tracking)
            step_count: Current step number in episode
            max_steps: Maximum steps per episode

        Returns:
            total_reward: Scalar reward value
            reward_breakdown: Dictionary with individual reward components
        """
        # Compute individual reward components
        correctness_reward = self._correctness_reward(test_results)
        quality_reward = self._quality_reward(current_code, test_results)
        efficiency_reward = self._efficiency_reward(current_code, step_count, max_steps)
        progress_reward = self._progress_reward(current_code, previous_code, test_results)

        # Combine rewards with weights
        total_reward = (
            self.weights['correctness'] * correctness_reward +
            self.weights['quality'] * quality_reward +
            self.weights['efficiency'] * efficiency_reward +
            self.weights['progress'] * progress_reward
        )

        # Return reward and breakdown for analysis
        reward_breakdown = {
            'correctness': correctness_reward,
            'quality': quality_reward,
            'efficiency': efficiency_reward,
            'progress': progress_reward,
            'total': total_reward
        }

        return total_reward, reward_breakdown

    def _correctness_reward(self, test_results: Dict[str, Any]) -> float:
        """
        Reward for test correctness.

        This is the primary objective: generate code that passes tests.

        Reward structure:
        - Linear reward for passing tests: n_passed / n_total
        - Bonus reward for passing all tests: +2.0
        - Penalty for runtime errors: -0.1 per error

        This provides both:
        - Dense feedback: Partial credit for partial success
        - Sparse feedback: Bonus for complete success

        Args:
            test_results: Dictionary with 'passed', 'total', 'errors'

        Returns:
            reward: Correctness reward value
        """
        passed = test_results.get('passed', 0)
        total = test_results.get('total', 1)
        errors = test_results.get('errors', [])

        if total == 0:
            return 0.0

        # Base reward: proportion of tests passed
        base_reward = passed / total

        # Bonus for passing all tests (complete success)
        if passed == total:
            bonus = 2.0
        else:
            bonus = 0.0

        # Penalty for runtime errors
        error_penalty = 0.1 * len(errors)

        reward = base_reward + bonus - error_penalty

        return reward

    def _quality_reward(self, code: str, test_results: Dict[str, Any]) -> float:
        """
        Reward for code quality metrics.

        Good code is not just correct, but also:
        - Readable: Clear variable names, proper formatting
        - Maintainable: Simple, modular, well-structured
        - Idiomatic: Follows language conventions

        Quality metrics:
        1. Syntax validity (+0.5)
        2. Proper structure: function definition, return statement (+0.3)
        3. Reasonable length: Not too short or long (+0.2)
        4. Code complexity: Prefer simpler solutions (+0.2)
        5. Documentation: Has docstring (+0.1)

        Args:
            code: Generated code string
            test_results: Test execution results

        Returns:
            reward: Quality reward value
        """
        reward = 0.0

        # 1. Syntax validity
        if self._check_syntax(code):
            reward += 0.5
        else:
            return 0.0  # No quality reward for invalid syntax

        # 2. Proper structure
        if 'def ' in code:
            reward += 0.15
        if 'return ' in code:
            reward += 0.15

        # 3. Reasonable length (prefer concise but not too short)
        line_count = code.count('\n')
        if 3 <= line_count <= 20:
            reward += 0.2
        elif line_count > 50:
            reward -= 0.1  # Penalty for very long code

        # 4. Code complexity (prefer simpler solutions)
        complexity_score = self._estimate_complexity(code)
        if complexity_score < 5:
            reward += 0.2
        elif complexity_score > 15:
            reward -= 0.1

        # 5. Documentation
        if '"""' in code or "'''" in code:
            reward += 0.1

        return reward

    def _efficiency_reward(self,
                          code: str,
                          step_count: int,
                          max_steps: int) -> float:
        """
        Reward for code efficiency and generation efficiency.

        Two aspects of efficiency:
        1. Code efficiency: Runtime, space complexity, algorithmic optimality
        2. Generation efficiency: Fewer steps to solution

        Args:
            code: Generated code string
            step_count: Current step in episode
            max_steps: Maximum steps allowed

        Returns:
            reward: Efficiency reward value
        """
        reward = 0.0

        # 1. Generation efficiency: Penalize slow solutions
        # Reward faster convergence to solution
        step_ratio = step_count / max_steps
        reward -= 0.1 * step_ratio  # Small penalty for taking many steps

        # 2. Code efficiency: Prefer optimal algorithms
        # Check for efficient patterns vs inefficient patterns

        # Efficient patterns (bonus)
        if 'in ' in code:  # Membership testing
            reward += 0.05
        if 'enumerate(' in code:  # Pythonic iteration
            reward += 0.05
        if 'list comprehension' in code or '[' in code and 'for' in code:
            reward += 0.05

        # Inefficient patterns (penalty)
        if code.count('for') > 3:  # Too many nested loops
            reward -= 0.1
        if 'while True:' in code and 'break' not in code:  # Potential infinite loop
            reward -= 0.2

        # Code length penalty: Prefer concise solutions
        code_length = len(code)
        if code_length > 1000:
            reward -= 0.1

        return reward

    def _progress_reward(self,
                        current_code: str,
                        previous_code: str,
                        test_results: Dict[str, Any]) -> float:
        """
        Reward for making progress toward solution.

        Incremental progress rewards help with sparse reward problems.
        Even if the final solution is far away, we reward steps in the
        right direction:
        - Code becomes syntactically valid
        - More tests pass
        - Code structure improves

        This provides dense feedback to guide exploration.

        Args:
            current_code: Current code state
            previous_code: Previous code state
            test_results: Current test results

        Returns:
            reward: Progress reward value
        """
        reward = 0.0

        # Progress in syntax validity
        current_valid = self._check_syntax(current_code)
        previous_valid = self._check_syntax(previous_code)
        if current_valid and not previous_valid:
            reward += 0.3  # Bonus for fixing syntax

        # Progress in code structure
        current_has_function = 'def ' in current_code
        previous_has_function = 'def ' in previous_code
        if current_has_function and not previous_has_function:
            reward += 0.2

        current_has_return = 'return ' in current_code
        previous_has_return = 'return ' in previous_code
        if current_has_return and not previous_has_return:
            reward += 0.1

        # Small reward for any code generation (encourages exploration)
        if len(current_code) > len(previous_code):
            reward += 0.05

        return reward

    def _check_syntax(self, code: str) -> bool:
        """
        Check if code is syntactically valid.

        Args:
            code: Code string to check

        Returns:
            is_valid: True if syntactically valid
        """
        try:
            ast.parse(code)
            return True
        except SyntaxError:
            return False

    def _estimate_complexity(self, code: str) -> int:
        """
        Estimate cyclomatic complexity of code.

        Cyclomatic complexity measures the number of linearly independent
        paths through the code. Higher complexity → harder to test/maintain.

        Simplified estimation based on control flow keywords.

        Args:
            code: Code string

        Returns:
            complexity: Estimated complexity score
        """
        complexity = 1  # Base complexity

        # Count decision points
        complexity += code.count('if ')
        complexity += code.count('elif ')
        complexity += code.count('for ')
        complexity += code.count('while ')
        complexity += code.count('and ')
        complexity += code.count('or ')
        complexity += code.count('except ')

        return complexity


class SparseReward:
    """
    Sparse reward function: Only reward complete success.

    R = +1 if all tests pass
    R = 0 otherwise

    Advantages:
    - Simple and unambiguous
    - No reward shaping bias

    Disadvantages:
    - Slow learning: Agent gets no feedback until success
    - Exploration challenge: Hard to discover successful strategies
    - Suitable only for simple tasks or with good exploration strategies

    This is useful as a baseline or for tasks where partial solutions
    are not meaningful.
    """

    def compute_reward(self, test_results: Dict[str, Any]) -> float:
        """
        Compute sparse reward.

        Args:
            test_results: Dictionary with test execution results

        Returns:
            reward: 1.0 if all tests pass, 0.0 otherwise
        """
        passed = test_results.get('passed', 0)
        total = test_results.get('total', 1)

        if total > 0 and passed == total:
            return 1.0
        else:
            return 0.0


class CurriculumReward:
    """
    Curriculum learning reward: Adjust difficulty over time.

    Start with easier objectives (sparse rewards, partial credit) and
    gradually increase difficulty (stricter requirements).

    Example curriculum:
    1. Early training: Reward syntax validity
    2. Mid training: Reward passing any test
    3. Late training: Reward passing all tests with quality

    This helps with exploration in early training when agent is random,
    and enforces high standards in late training when agent is competent.

    Curriculum learning often significantly speeds up training for
    complex tasks.
    """

    def __init__(self, total_episodes: int = 10000):
        """
        Initialize curriculum reward.

        Args:
            total_episodes: Total episodes for curriculum schedule
        """
        self.total_episodes = total_episodes
        self.current_episode = 0

    def compute_reward(self,
                      code: str,
                      test_results: Dict[str, Any],
                      episode: int) -> float:
        """
        Compute reward based on curriculum stage.

        Args:
            code: Generated code
            test_results: Test execution results
            episode: Current episode number

        Returns:
            reward: Reward value based on curriculum stage
        """
        self.current_episode = episode
        progress = episode / self.total_episodes

        # Stage 1 (0-30%): Reward syntax validity
        if progress < 0.3:
            try:
                ast.parse(code)
                return 1.0
            except SyntaxError:
                return 0.0

        # Stage 2 (30-70%): Reward passing any test
        elif progress < 0.7:
            passed = test_results.get('passed', 0)
            if passed > 0:
                return float(passed)
            return 0.0

        # Stage 3 (70-100%): Reward passing all tests
        else:
            passed = test_results.get('passed', 0)
            total = test_results.get('total', 1)
            if total > 0 and passed == total:
                return 10.0
            return 0.0


def test_reward_functions():
    """Test various reward function implementations."""
    print("Testing Reward Functions")
    print("="*70)

    # Example test results
    test_results_partial = {
        'passed': 2,
        'total': 4,
        'errors': ['Test 2: Expected 5, got 4']
    }

    test_results_complete = {
        'passed': 4,
        'total': 4,
        'errors': []
    }

    # Example code
    good_code = """
def add_numbers(a, b):
    \"\"\"Add two numbers.\"\"\"
    return a + b
"""

    bad_code = "def add_numbers(a, b) return a + b"  # Syntax error

    # Test CodeQualityReward
    print("\n1. Testing CodeQualityReward")
    print("-" * 70)
    cq_reward = CodeQualityReward()

    reward, breakdown = cq_reward.compute_reward(good_code, test_results_complete)
    print(f"Good code, all tests passed:")
    print(f"  Total reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")

    reward, breakdown = cq_reward.compute_reward(good_code, test_results_partial)
    print(f"\nGood code, partial tests passed:")
    print(f"  Total reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")

    reward, breakdown = cq_reward.compute_reward(bad_code, test_results_partial)
    print(f"\nBad code, partial tests passed:")
    print(f"  Total reward: {reward:.2f}")
    print(f"  Breakdown: {breakdown}")

    # Test SparseReward
    print("\n\n2. Testing SparseReward")
    print("-" * 70)
    sparse_reward = SparseReward()

    reward = sparse_reward.compute_reward(test_results_complete)
    print(f"All tests passed: {reward:.2f}")

    reward = sparse_reward.compute_reward(test_results_partial)
    print(f"Partial tests passed: {reward:.2f}")

    # Test CurriculumReward
    print("\n\n3. Testing CurriculumReward")
    print("-" * 70)
    curriculum_reward = CurriculumReward(total_episodes=1000)

    for episode in [0, 300, 700, 999]:
        reward = curriculum_reward.compute_reward(good_code, test_results_partial, episode)
        print(f"Episode {episode} (stage {episode/1000:.0%}): {reward:.2f}")


if __name__ == "__main__":
    test_reward_functions()
    print("\n" + "="*70)
    print("Reward function tests completed!")
