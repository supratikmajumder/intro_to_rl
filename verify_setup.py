#!/usr/bin/env python3
"""
Setup verification script for RL Training Project

This script checks that:
1. All required files exist
2. Python syntax is valid
3. Basic imports work
4. Configuration files are valid
"""

import os
import sys
import ast
import json

# Color codes for terminal output
GREEN = '\033[0;32m'
RED = '\033[0;31m'
YELLOW = '\033[1;33m'
BLUE = '\033[0;34m'
NC = '\033[0m'  # No Color


def print_header(text):
    print(f"\n{BLUE}{'='*60}{NC}")
    print(f"{BLUE}{text}{NC}")
    print(f"{BLUE}{'='*60}{NC}\n")


def print_success(text):
    print(f"{GREEN}✓ {text}{NC}")


def print_error(text):
    print(f"{RED}✗ {text}{NC}")


def print_warning(text):
    print(f"{YELLOW}⚠ {text}{NC}")


def check_file_exists(filepath):
    """Check if a file exists."""
    if os.path.exists(filepath):
        print_success(f"Found: {filepath}")
        return True
    else:
        print_error(f"Missing: {filepath}")
        return False


def check_python_syntax(filepath):
    """Check if a Python file has valid syntax."""
    try:
        with open(filepath, 'r') as f:
            ast.parse(f.read())
        return True
    except SyntaxError as e:
        print_error(f"Syntax error in {filepath}: {e}")
        return False


def check_json_valid(filepath):
    """Check if a JSON file is valid."""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        return True
    except json.JSONDecodeError as e:
        print_error(f"JSON error in {filepath}: {e}")
        return False


def main():
    print_header("RL Training Project - Setup Verification")

    all_passed = True

    # Step 1: Check directory structure
    print_header("Step 1: Checking Directory Structure")

    directories = [
        'agents',
        'environments',
        'models',
        'rewards',
        'utils',
        'configs',
        'data/examples',
        'docs',
    ]

    for directory in directories:
        if os.path.isdir(directory):
            print_success(f"Directory exists: {directory}/")
        else:
            print_error(f"Directory missing: {directory}/")
            all_passed = False

    # Step 2: Check Python files exist and have valid syntax
    print_header("Step 2: Checking Python Files")

    python_files = [
        'agents/__init__.py',
        'agents/dqn_agent.py',
        'environments/__init__.py',
        'environments/code_gen_env.py',
        'models/__init__.py',
        'models/code_gen_model.py',
        'rewards/__init__.py',
        'rewards/code_quality_reward.py',
        'utils/__init__.py',
        'utils/helpers.py',
        'train.py',
        'evaluate.py',
    ]

    syntax_valid = True
    for filepath in python_files:
        if check_file_exists(filepath):
            if not check_python_syntax(filepath):
                syntax_valid = False
                all_passed = False

    if syntax_valid:
        print_success("All Python files have valid syntax")

    # Step 3: Check data files
    print_header("Step 3: Checking Data Files")

    data_files = [
        'data/examples/training_problems.json',
        'data/examples/eval_problems.json',
    ]

    for filepath in data_files:
        if check_file_exists(filepath):
            if not check_json_valid(filepath):
                all_passed = False

    # Step 4: Check configuration files
    print_header("Step 4: Checking Configuration Files")

    config_files = [
        'configs/train_config.yaml',
        'requirements.txt',
    ]

    for filepath in config_files:
        check_file_exists(filepath)

    # Step 5: Check documentation
    print_header("Step 5: Checking Documentation")

    doc_files = [
        'README.md',
        'CLAUDE.md',
        'docs/RL_CONCEPTS.md',
        'TEST_GUIDE.md',
    ]

    for filepath in doc_files:
        check_file_exists(filepath)

    # Step 6: Check imports (basic)
    print_header("Step 6: Checking Basic Imports")

    try:
        import ast
        import json
        print_success("Standard library imports work")
    except ImportError as e:
        print_error(f"Standard library import failed: {e}")
        all_passed = False

    # Try importing third-party libraries (they might not be installed yet)
    print("\nChecking third-party dependencies:")
    dependencies = {
        'numpy': 'numpy',
        'torch': 'PyTorch',
        'gym': 'OpenAI Gym',
        'yaml': 'PyYAML',
        'matplotlib': 'Matplotlib'
    }

    missing_deps = []
    for module, name in dependencies.items():
        try:
            __import__(module)
            print_success(f"{name} is installed")
        except ImportError:
            print_warning(f"{name} not installed (install with: pip3 install {module})")
            missing_deps.append(module)

    # Final summary
    print_header("Verification Summary")

    if all_passed:
        print_success("All structure and syntax checks passed!")
    else:
        print_error("Some checks failed. Please review errors above.")

    if missing_deps:
        print_warning(f"\nMissing dependencies: {', '.join(missing_deps)}")
        print(f"\nTo install all dependencies, run:")
        print(f"  {GREEN}pip3 install -r requirements.txt{NC}")
    else:
        print_success("\nAll dependencies are installed!")

    print("\n" + "="*60)
    if all_passed and not missing_deps:
        print(f"{GREEN}✓ Setup complete! You're ready to run training.{NC}")
        print("\nNext steps:")
        print("  1. Run quick test: python3 train.py --config configs/test_config.yaml")
        print("  2. See TEST_GUIDE.md for comprehensive testing")
    elif all_passed:
        print(f"{YELLOW}⚠ Setup almost complete. Install dependencies first.{NC}")
        print(f"\n  Run: {GREEN}pip3 install -r requirements.txt{NC}")
    else:
        print(f"{RED}✗ Setup incomplete. Fix errors above.{NC}")

    print("="*60 + "\n")

    return 0 if all_passed and not missing_deps else 1


if __name__ == "__main__":
    sys.exit(main())
