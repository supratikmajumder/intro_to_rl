#!/bin/bash
# Comprehensive test runner for RL training project

set -e  # Exit on any error

echo "=========================================="
echo "RL Training Project - Test Suite"
echo "=========================================="
echo ""

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Track results
PASSED=0
FAILED=0

run_test() {
    local test_name=$1
    local test_command=$2

    echo -e "${YELLOW}Running: $test_name${NC}"
    if eval "$test_command" > /dev/null 2>&1; then
        echo -e "${GREEN}✓ PASSED${NC}"
        ((PASSED++))
    else
        echo -e "${RED}✗ FAILED${NC}"
        echo "  Command: $test_command"
        ((FAILED++))
    fi
    echo ""
}

echo "Step 1: Checking Python environment"
echo "----------------------------------------"
python3 --version
echo ""

echo "Step 2: Testing individual components"
echo "----------------------------------------"
echo ""

run_test "Environment Module" "python3 environments/code_gen_env.py"
run_test "Q-Network Model" "python3 models/code_gen_model.py"
run_test "DQN Agent" "python3 agents/dqn_agent.py"
run_test "Reward Functions" "python3 rewards/code_quality_reward.py"
run_test "Utility Functions" "python3 utils/helpers.py"

echo "=========================================="
echo "Test Summary"
echo "=========================================="
echo -e "Passed: ${GREEN}$PASSED${NC}"
echo -e "Failed: ${RED}$FAILED${NC}"
echo ""

if [ $FAILED -eq 0 ]; then
    echo -e "${GREEN}All tests passed! ✓${NC}"
    echo ""
    echo "Next steps:"
    echo "1. Install dependencies: pip3 install -r requirements.txt"
    echo "2. Run quick training: python3 train.py --config configs/test_config.yaml"
    echo "3. See TEST_GUIDE.md for full testing instructions"
    exit 0
else
    echo -e "${RED}Some tests failed. Please check errors above.${NC}"
    exit 1
fi
