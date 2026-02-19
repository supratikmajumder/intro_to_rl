# Testing Guide for RL Training Project

This guide walks you through testing and verifying the entire project.

## Step 1: Environment Setup

### Install Dependencies

```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch; import gym; import numpy; print('✓ All dependencies installed')"
```

## Step 2: Test Individual Components

Run these tests to verify each component works independently:

### Test 1: Environment

```bash
python environments/code_gen_env.py
```

**Expected output**:
- Environment created successfully
- Observation shape: (520,)
- Action space: 5000
- Sample episode runs with random actions
- Test results shown for each step

### Test 2: Q-Network

```bash
python models/code_gen_model.py
```

**Expected output**:
- Q-Network tests pass
- Dueling Q-Network tests pass
- Network comparison shown
- Parameter counts displayed

### Test 3: DQN Agent

```bash
python agents/dqn_agent.py
```

**Expected output**:
- Agent initialized successfully
- Action selection works
- Experience buffer stores transitions
- Training loss computed
- Target network updated

### Test 4: Reward Functions

```bash
python rewards/code_quality_reward.py
```

**Expected output**:
- CodeQualityReward tested with various scenarios
- SparseReward tested
- CurriculumReward tested with different episode stages
- Reward breakdowns shown

### Test 5: Utility Functions

```bash
python utils/helpers.py
```

**Expected output**:
- MetricsLogger tested
- Example problems created
- Logging functionality verified

## Step 3: Quick Training Test (5 minutes)

Create a minimal config for quick testing:

```bash
# Create test config
cat > configs/test_config.yaml << 'EOF'
environment:
  max_steps: 20
  vocab_size: 1000

agent:
  state_dim: 520
  action_dim: 1000
  hidden_dim: 256
  learning_rate: 0.001
  gamma: 0.99
  epsilon_start: 1.0
  epsilon_end: 0.1
  epsilon_decay: 0.99
  buffer_capacity: 1000
  batch_size: 32
  min_buffer_size: 100
  target_update_frequency: 5
  device: 'cpu'

training:
  num_episodes: 50  # Just 50 episodes for quick test
  max_timesteps: 20
  train_frequency: 1
  eval_frequency: 25
  save_frequency: 25
  log_frequency: 10
  log_dir: 'experiments/test_logs'
  checkpoint_dir: 'experiments/test_models'
  save_best_only: false

reward:
  type: 'code_quality'
  correctness_weight: 10.0
  quality_weight: 2.0
  efficiency_weight: 1.0
  progress_weight: 0.5

evaluation:
  num_episodes: 5
  deterministic: true
  render: false

data:
  problems_file: 'data/examples/training_problems.json'
  eval_problems_file: 'data/examples/eval_problems.json'

seed: 42
verbose: true
EOF
```

### Run Quick Training

```bash
python train.py --config configs/test_config.yaml
```

**Expected output**:
- Configuration loaded
- Problems loaded (8 training problems)
- Environment initialized
- Agent initialized
- Training progress printed every 10 episodes
- Evaluation at episode 25
- Final model saved
- Training summary and plots generated

**What to look for**:
- ✓ No errors or crashes
- ✓ Episode rewards change over time (learning happening)
- ✓ Success rate improves (even slightly)
- ✓ Loss values are reasonable (not NaN or infinity)
- ✓ Epsilon decays from 1.0 toward 0.1

**Expected files created**:
```
experiments/
├── test_logs/
│   ├── training_YYYYMMDD_HHMMSS.log
│   ├── training_summary.json
│   └── training_curves.png
└── test_models/
    ├── checkpoint_ep25.pt
    ├── best_model.pt
    └── final_model.pt
```

## Step 4: Evaluate the Trained Model

```bash
python evaluate.py --model experiments/test_models/best_model.pt \
                   --problems data/examples/eval_problems.json \
                   --num_episodes 10
```

**Expected output**:
- Model loaded successfully
- Agent statistics shown
- Evaluation runs on 3 test problems (multiply, is_prime, find_min)
- Per-problem performance breakdown
- Example generated code (even if incomplete)
- Overall success rate and metrics

**Note**: After only 50 episodes, the agent won't solve problems perfectly. This is expected! A full training run (5000 episodes) is needed for good performance.

## Step 5: Visualize Results

### View Training Curves

Open the generated plot:
```bash
# On macOS
open experiments/test_logs/training_curves.png

# On Linux
xdg-open experiments/test_logs/training_curves.png

# On Windows
start experiments/test_logs/training_curves.png
```

**What to look for**:
- Episode rewards plot (should show some trend, even if noisy)
- Success rate plot (may be mostly zero for short training)
- Loss plot (should be non-zero after buffer fills)
- Epsilon decay plot (should decrease smoothly)

### View Training Log

```bash
# View last 20 lines of training log
tail -20 experiments/test_logs/training_*.log

# Or view summary
cat experiments/test_logs/training_summary.json
```

## Step 6: Full Training Run (Optional, ~30-60 minutes)

If the quick test works, try a full training run:

```bash
# Use the default config
python train.py --config configs/train_config.yaml
```

This will train for 5000 episodes and should achieve reasonable performance on simple problems like `add_numbers`, `is_even`, etc.

**Expected performance after full training**:
- Simple problems (add, is_even): 60-80% success rate
- Medium problems (max_of_three, factorial): 30-50% success rate
- Complex problems: Lower success rate (this is a simple baseline agent)

## Troubleshooting

### Issue: Import errors

```bash
# Make sure you're in the right directory
cd /Users/smajumder/WS/Modeling_w_Claude/rl_training

# Check Python path
python -c "import sys; print(sys.path)"

# Verify dependencies
pip list | grep -E "torch|gym|numpy|yaml"
```

### Issue: "No module named 'agents'"

```bash
# Run from project root directory
cd /Users/smajumder/WS/Modeling_w_Claude/rl_training
python train.py --config configs/test_config.yaml
```

### Issue: Training is very slow

This is expected on CPU. For faster training:
1. Reduce `num_episodes` in config (test with 50-100)
2. Reduce `vocab_size` (1000 instead of 5000)
3. Reduce `hidden_dim` (256 instead of 512)
4. Use GPU if available (set `device: 'cuda'`)

### Issue: Rewards stay at zero

This might happen in the first few episodes:
- Buffer needs to fill (`min_buffer_size: 100`)
- Exploration needs time (high epsilon initially)
- Simple problems should show progress after 50-100 episodes

### Issue: NaN or Inf in loss

This indicates a problem. Check:
- Learning rate too high (try 1e-5)
- Gradient explosion (gradient clipping is enabled, but may need tuning)
- Check for divide-by-zero in reward function

## Verification Checklist

Run through this checklist to verify everything works:

- [ ] All dependencies installed without errors
- [ ] Individual component tests pass (5 tests in Step 2)
- [ ] Quick training runs without crashes (Step 3)
- [ ] Model checkpoints are saved
- [ ] Training log file created
- [ ] Training curves plot generated
- [ ] Evaluation script runs without errors (Step 4)
- [ ] Agent shows some learning (rewards change over time)

## Success Criteria

**Minimal (Quick Test)**:
- ✓ All components run without errors
- ✓ Training completes 50 episodes
- ✓ Files are generated (logs, models, plots)
- ✓ Rewards are non-zero and changing

**Good (After 500 episodes)**:
- ✓ Success rate > 10% on simple problems
- ✓ Average reward increasing trend
- ✓ Some problems occasionally solved

**Excellent (After 5000 episodes)**:
- ✓ Success rate > 50% on simple problems (add, is_even)
- ✓ Consistent reward improvement
- ✓ Generated code often syntactically valid

## Next Steps

After verification:

1. **Experiment with hyperparameters**: Modify `configs/train_config.yaml`
2. **Add custom problems**: Edit `data/examples/training_problems.json`
3. **Try different reward functions**: Change `reward.type` in config
4. **Implement improvements**: Add Double DQN, prioritized replay, etc.
5. **Visualize learning**: Use TensorBoard or custom plotting

## Getting Help

If you encounter issues:

1. Check the error message carefully
2. Review `docs/RL_CONCEPTS.md` for understanding
3. Read `README.md` troubleshooting section
4. Enable verbose logging: `verbose: true` in config
5. Run with fewer episodes to debug faster

Happy testing! 🚀
