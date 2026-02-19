"""
Code Generation Environment for Reinforcement Learning

This module defines a custom RL environment for training agents to generate
Python functions from natural language specifications and test cases.

The environment follows the OpenAI Gym interface:
- State: A problem specification (docstring + test cases)
- Action: Generate or modify code tokens/lines
- Reward: Based on test passage, code quality, and efficiency
- Episode: Complete when valid function is generated or max steps reached

Key RL Concepts Demonstrated:
- State Space: Problem specifications represented as embeddings
- Action Space: Discrete actions for code generation (tokens, operations)
- Reward Function: Multi-objective (correctness, quality, efficiency)
- Episode Termination: Success (all tests pass) or failure (max steps/invalid code)
"""

import gym
from gym import spaces
import numpy as np
import ast
import subprocess
import tempfile
import os
from typing import Dict, List, Tuple, Any, Optional


class CodeGenerationEnv(gym.Env):
    """
    Custom RL Environment for learning to generate Python functions.

    This environment simulates the task of writing a Python function given:
    1. A docstring describing what the function should do
    2. A set of test cases the function must pass

    The agent learns to generate correct, efficient, and well-structured code
    through trial and error, receiving rewards for passing tests and writing
    quality code.

    Attributes:
        problem_spec (Dict): Current problem specification
        current_code (str): Code generated so far in current episode
        max_steps (int): Maximum steps per episode
        step_count (int): Current step in episode
        test_results (Dict): Results from running tests
    """

    metadata = {'render.modes': ['human']}

    def __init__(self,
                 problems: List[Dict[str, Any]],
                 max_steps: int = 50,
                 vocab_size: int = 5000):
        """
        Initialize the code generation environment.

        Args:
            problems: List of problem specifications, each containing:
                - 'docstring': Function description
                - 'function_name': Name of function to generate
                - 'test_cases': List of (input, expected_output) tuples
                - 'starter_code': Optional template code
            max_steps: Maximum number of generation steps per episode
            vocab_size: Size of code token vocabulary for action space
        """
        super(CodeGenerationEnv, self).__init__()

        self.problems = problems
        self.max_steps = max_steps
        self.vocab_size = vocab_size

        # Current episode state
        self.current_problem_idx = 0
        self.current_code = ""
        self.step_count = 0
        self.test_results = {}

        # Define action space: discrete actions for code tokens/operations
        # Actions include: code tokens, special operations (indent, dedent, newline)
        self.action_space = spaces.Discrete(vocab_size)

        # Define observation space: embedding of problem specification + current code state
        # We use a flattened representation:
        # - Problem embedding (256 dims)
        # - Code state embedding (256 dims)
        # - Metadata (step count, code length, syntax validity, etc.)
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(520,),  # 256 + 256 + 8 metadata features
            dtype=np.float32
        )

    def reset(self) -> np.ndarray:
        """
        Reset the environment to start a new episode.

        In RL terms:
        - Samples a new problem (initial state s₀)
        - Resets episode counters
        - Returns initial observation

        Returns:
            observation: Initial state representation
        """
        # Select a random problem for this episode
        self.current_problem_idx = np.random.randint(0, len(self.problems))
        self.current_problem = self.problems[self.current_problem_idx]

        # Reset episode state
        self.current_code = self.current_problem.get('starter_code', '')
        self.step_count = 0
        self.test_results = {}

        # Generate initial observation
        observation = self._get_observation()

        return observation

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute one step in the environment.

        In RL terms:
        - Takes action aₜ in state sₜ
        - Transitions to new state sₜ₊₁
        - Computes reward rₜ
        - Determines if episode is done

        Args:
            action: Action to take (code token or operation)

        Returns:
            observation: Next state sₜ₊₁
            reward: Reward rₜ for this transition
            done: Whether episode has terminated
            info: Additional diagnostic information
        """
        self.step_count += 1

        # Apply action to modify current code
        self.current_code = self._apply_action(action)

        # Check if code is syntactically valid
        is_valid_syntax = self._check_syntax()

        # Run tests if syntax is valid
        if is_valid_syntax:
            self.test_results = self._run_tests()
        else:
            self.test_results = {'passed': 0, 'total': 0, 'errors': ['Syntax Error']}

        # Calculate reward based on test results and code quality
        reward = self._calculate_reward()

        # Check termination conditions
        done = self._is_done()

        # Get next observation
        observation = self._get_observation()

        # Additional info for debugging and analysis
        info = {
            'step': self.step_count,
            'tests_passed': self.test_results.get('passed', 0),
            'total_tests': len(self.current_problem['test_cases']),
            'syntax_valid': is_valid_syntax,
            'code_length': len(self.current_code),
            'episode_done': done
        }

        return observation, reward, done, info

    def _get_observation(self) -> np.ndarray:
        """
        Generate observation vector representing current state.

        The observation encodes:
        1. Problem specification (docstring, function signature)
        2. Current code state
        3. Metadata (step count, syntax validity, test results)

        Returns:
            observation: State representation as numpy array
        """
        # In a real implementation, you would:
        # 1. Embed the docstring using a language model (e.g., BERT, CodeBERT)
        # 2. Embed the current code using a code embedding model
        # 3. Concatenate with metadata features

        # For this example, we use placeholder embeddings
        problem_embedding = self._embed_problem()  # 256-dim vector
        code_embedding = self._embed_code()  # 256-dim vector
        metadata = self._get_metadata()  # 8-dim vector

        observation = np.concatenate([problem_embedding, code_embedding, metadata])
        return observation.astype(np.float32)

    def _embed_problem(self) -> np.ndarray:
        """
        Generate embedding for the problem specification.

        In practice, use a pre-trained language model like CodeBERT or
        GraphCodeBERT to encode the docstring and function signature.
        """
        # Placeholder: random embedding based on problem hash
        # In real implementation: use transformer-based encoder
        problem_text = self.current_problem['docstring']
        seed = hash(problem_text) % (2**32)
        np.random.seed(seed)
        embedding = np.random.randn(256)
        return embedding

    def _embed_code(self) -> np.ndarray:
        """
        Generate embedding for the current code state.

        In practice, use a code-specific embedding model or AST-based
        representation to capture code structure and semantics.
        """
        # Placeholder: simple features based on code content
        # In real implementation: use CodeBERT, GraphCodeBERT, or AST embedding
        code_features = np.zeros(256)

        # Simple features: code length, line count, indentation level, etc.
        code_features[0] = len(self.current_code) / 1000.0  # normalized length
        code_features[1] = self.current_code.count('\n') / 50.0  # normalized line count
        code_features[2] = self.current_code.count('def') / 5.0  # function definitions
        code_features[3] = self.current_code.count('return') / 5.0  # return statements
        code_features[4] = self.current_code.count('if') / 10.0  # conditionals
        code_features[5] = self.current_code.count('for') / 10.0  # loops

        return code_features

    def _get_metadata(self) -> np.ndarray:
        """
        Generate metadata features about the current episode state.
        """
        metadata = np.zeros(8)

        metadata[0] = self.step_count / self.max_steps  # normalized step count
        metadata[1] = len(self.current_code) / 1000.0  # normalized code length
        metadata[2] = 1.0 if self._check_syntax() else 0.0  # syntax validity
        metadata[3] = self.test_results.get('passed', 0) / max(len(self.current_problem['test_cases']), 1)
        metadata[4] = 1.0 if self.test_results.get('passed', 0) == len(self.current_problem['test_cases']) else 0.0
        metadata[5] = self.current_code.count('def') / 5.0  # function count
        metadata[6] = self.current_code.count('\n') / 50.0  # line count
        metadata[7] = len(self.test_results.get('errors', [])) / 10.0  # error count

        return metadata

    def _apply_action(self, action: int) -> str:
        """
        Apply an action to modify the current code.

        Actions can be:
        - Code tokens (keywords, identifiers, operators, literals)
        - Special operations (indent, dedent, newline, delete)

        Args:
            action: Integer action ID

        Returns:
            updated_code: Modified code string
        """
        # In a real implementation, you would:
        # 1. Map action ID to actual code token or operation
        # 2. Use a tokenizer/vocabulary to decode action
        # 3. Apply the token to current code with proper formatting

        # Placeholder: simple append operation
        # In real implementation: use proper tokenizer and code formatter
        token = self._action_to_token(action)
        updated_code = self.current_code + token

        return updated_code

    def _action_to_token(self, action: int) -> str:
        """
        Convert action ID to code token or operation.

        This is a simplified placeholder. In practice:
        - Use a proper Python tokenizer
        - Maintain a vocabulary of common code patterns
        - Support structural operations (indent, dedent, etc.)
        """
        # Simplified vocabulary
        vocab = [
            'def ', 'return ', 'if ', 'else:', 'for ', 'in ', 'range(',
            ':\n', '\n', '    ', '(', ')', '[', ']', '+', '-', '*', '/',
            '==', '!=', '<', '>', '<=', '>=', 'and ', 'or ', 'not ',
            'True', 'False', 'None', ', ', ' = ', ' == ', '\n    ',
        ]

        if action < len(vocab):
            return vocab[action]
        else:
            return ''  # No-op for out-of-range actions

    def _check_syntax(self) -> bool:
        """
        Check if current code is syntactically valid Python.

        Returns:
            is_valid: True if code parses without syntax errors
        """
        try:
            ast.parse(self.current_code)
            return True
        except SyntaxError:
            return False

    def _run_tests(self) -> Dict[str, Any]:
        """
        Execute test cases against the generated code.

        This is a critical component for RL reward:
        - Passing tests → positive reward
        - Failing tests → negative reward or zero reward
        - Runtime errors → penalty

        Returns:
            results: Dictionary with test outcomes
        """
        test_cases = self.current_problem['test_cases']
        function_name = self.current_problem['function_name']

        passed = 0
        errors = []

        for i, (inputs, expected_output) in enumerate(test_cases):
            try:
                # Create a temporary namespace and execute the code
                namespace = {}
                exec(self.current_code, namespace)

                # Check if function exists
                if function_name not in namespace:
                    errors.append(f"Test {i}: Function '{function_name}' not defined")
                    continue

                # Call the function with test inputs
                func = namespace[function_name]
                if isinstance(inputs, tuple):
                    actual_output = func(*inputs)
                else:
                    actual_output = func(inputs)

                # Check if output matches expected
                if actual_output == expected_output:
                    passed += 1
                else:
                    errors.append(f"Test {i}: Expected {expected_output}, got {actual_output}")

            except Exception as e:
                errors.append(f"Test {i}: {type(e).__name__}: {str(e)}")

        return {
            'passed': passed,
            'total': len(test_cases),
            'errors': errors,
            'success_rate': passed / len(test_cases) if test_cases else 0.0
        }

    def _calculate_reward(self) -> float:
        """
        Calculate reward for the current state-action transition.

        Reward function design is crucial in RL. Here we use a multi-objective
        reward that balances:

        1. Correctness: Tests passed (primary objective)
        2. Code Quality: Readability, efficiency, simplicity
        3. Progress: Incremental improvements
        4. Penalties: Syntax errors, excessive length, inefficiency

        Reward Formula:
            R = w₁ * test_reward + w₂ * quality_reward - w₃ * penalties

        Returns:
            reward: Scalar reward value
        """
        # Test correctness reward (most important)
        tests_passed = self.test_results.get('passed', 0)
        total_tests = len(self.current_problem['test_cases'])

        if total_tests > 0:
            test_reward = (tests_passed / total_tests) * 10.0
            # Bonus for passing all tests
            if tests_passed == total_tests:
                test_reward += 20.0
        else:
            test_reward = 0.0

        # Code quality reward
        quality_reward = 0.0
        if self._check_syntax():
            quality_reward += 1.0  # Valid syntax

            # Prefer concise code
            code_lines = self.current_code.count('\n')
            if 5 <= code_lines <= 20:
                quality_reward += 1.0
            elif code_lines > 50:
                quality_reward -= 2.0  # Penalty for very long code

            # Reward proper structure (function definition, return statement)
            if 'def ' in self.current_code and 'return ' in self.current_code:
                quality_reward += 0.5

        # Penalties
        penalties = 0.0

        # Syntax error penalty
        if not self._check_syntax():
            penalties += 2.0

        # Step penalty (encourage efficiency)
        penalties += 0.01 * self.step_count

        # Runtime error penalty
        if self.test_results.get('errors'):
            penalties += 0.5 * len(self.test_results['errors'])

        # Final reward
        reward = test_reward + quality_reward - penalties

        return reward

    def _is_done(self) -> bool:
        """
        Determine if the episode should terminate.

        Termination conditions:
        1. Success: All tests pass
        2. Max steps reached
        3. Invalid state: Irrecoverable error

        Returns:
            done: True if episode should end
        """
        # Success: all tests passed
        if self.test_results.get('passed', 0) == len(self.current_problem['test_cases']) and \
           len(self.current_problem['test_cases']) > 0:
            return True

        # Failure: max steps reached
        if self.step_count >= self.max_steps:
            return True

        # Could add additional termination conditions:
        # - Code too long (>1000 characters)
        # - Too many consecutive syntax errors
        # - Timeout on test execution

        return False

    def render(self, mode='human'):
        """
        Render the current environment state for visualization.

        Useful for debugging and understanding agent behavior.
        """
        if mode == 'human':
            print("\n" + "="*70)
            print(f"STEP {self.step_count}/{self.max_steps}")
            print("="*70)
            print("\nPROBLEM:")
            print(self.current_problem['docstring'])
            print("\nCURRENT CODE:")
            print(self.current_code if self.current_code else "(empty)")
            print("\nTEST RESULTS:")
            print(f"  Passed: {self.test_results.get('passed', 0)}/{len(self.current_problem['test_cases'])}")
            if self.test_results.get('errors'):
                print("  Errors:")
                for error in self.test_results['errors'][:3]:  # Show first 3 errors
                    print(f"    - {error}")
            print("="*70)

    def close(self):
        """
        Clean up environment resources.
        """
        pass


if __name__ == "__main__":
    # Example usage and testing
    example_problems = [
        {
            'function_name': 'add_numbers',
            'docstring': 'Add two numbers and return the result.',
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
            'docstring': 'Check if a number is even.',
            'test_cases': [
                (2, True),
                (3, False),
                (0, True),
                (-2, True),
                (-3, False),
            ],
            'starter_code': ''
        }
    ]

    # Create environment
    env = CodeGenerationEnv(problems=example_problems, max_steps=20)

    # Run a test episode
    print("Testing Code Generation Environment")
    print("="*70)

    obs = env.reset()
    print(f"Initial observation shape: {obs.shape}")

    for step in range(5):
        action = env.action_space.sample()  # Random action
        obs, reward, done, info = env.step(action)

        print(f"\nStep {step + 1}:")
        print(f"  Action: {action}")
        print(f"  Reward: {reward:.2f}")
        print(f"  Done: {done}")
        print(f"  Info: {info}")

        if done:
            print("\nEpisode finished!")
            break

    env.render()
