# RNN Experiments Quick Start Guide

This guide will get you up and running with RNN deformation-based unit classification experiments.

## Installation

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

   This installs:
   - Core scientific Python: numpy, scipy, scikit-learn, matplotlib
   - **PyTorch**: Required for RNN experiments

2. **Verify installation:**
   ```bash
   python scripts/test_integration.py
   ```

   This runs a quick end-to-end test (1-2 minutes) to verify everything works.

---

## Running Your First Experiment

### Single Task (FlipFlop Memory)

```bash
python scripts/test_rnn.py --task flipflop --architecture vanilla --hidden-size 128 --epochs 2000
```

**What this does:**
1. Trains a vanilla RNN (128 units) on the 3-bit flip-flop task for 2000 epochs
2. Extracts hidden state trajectories (50 trials × 200 timesteps)
3. Estimates deformation signals (rotation, contraction, expansion) via PCA + local linear regression
4. Extracts unit features (correlation with each deformation mode)
5. Clusters units using K-means (4 clusters)
6. Compares to baselines (PCA, raw activations)
7. Saves results and figures to `results/rnn_figures/`

**Expected output:**
```
RNN DEFORMATION EXPERIMENT: FLIPFLOP / VANILLA
================================================================

Task: FlipFlop
  Input size: 3
  Output size: 3

Expected dynamics:
  Contraction-dominant: 2^n_bits stable attractors
  Expected: 8 attractor states
  Unit distribution: Integrators (60-80%), Rotators (10-20%), Explorers (5-15%)

Training Vanilla RNN on FlipFlop
  Hidden size: 128
  Epochs: 2000
  Learning rate: 0.001
------------------------------------------------------------
  Epoch  200/2000: Loss=0.0234, Accuracy=94.23%
  ...
  Epoch 2000/2000: Loss=0.0012, Accuracy=98.76%
------------------------------------------------------------
Training complete! Final accuracy: 98.76%

...

UNIT CLASSIFICATION SUMMARY
================================================================

Cluster 0: Integrator (89 units, 69.5%)
  Mean correlation with rotation:    +0.042
  Mean correlation with contraction: +0.687
  Mean correlation with expansion:   -0.123
  → Dominant mode: Contraction (+0.687)

Cluster 1: Rotator (21 units, 16.4%)
  Mean correlation with rotation:    +0.592
  Mean correlation with contraction: +0.089
  Mean correlation with expansion:   +0.134
  → Dominant mode: Rotation (+0.592)

...

Comparing to baseline methods...
  Silhouette scores:
    Deformation method: 0.423
    PCA baseline:       0.287
    Raw activations:    0.198
  Improvement over PCA: +47.4%
  
================================================================
```

**Interpretation:**
- ✓ FlipFlop task is integrator-dominant (69.5% > 60% expected)
- ✓ High task accuracy (98.76%) confirms RNN learned the task
- ✓ Deformation method outperforms baselines significantly (+47% vs PCA)
- ✓ Good clustering quality (silhouette = 0.423)

---

## Multi-Task Comparison

To test the hypothesis that task structure determines unit type distribution:

```bash
python scripts/test_rnn.py --task all --architecture vanilla --hidden-size 128 --epochs 2000
```

This runs all 3 tasks sequentially:
1. **FlipFlop** (stable attractors) → Expect integrators
2. **Cycling Memory** (periodic dynamics) → Expect rotators
3. **Context Integration** (mixed dynamics) → Expect balanced

**Expected cross-task comparison:**
```
CROSS-TASK COMPARISON
================================================================

FLIPFLOP:
  Integrator     :  69.5%
  Rotator        :  16.4%
  Explorer       :   8.6%
  Mixed          :   5.5%

CYCLING:
  Integrator     :  24.2%
  Rotator        :  58.6%
  Explorer       :  12.1%
  Mixed          :   5.1%

CONTEXT:
  Integrator     :  38.3%
  Rotator        :  29.7%
  Explorer       :  23.4%
  Mixed          :   8.6%
================================================================
```

**Interpretation:**
- ✓ Task-specific distributions match predictions
- ✓ FlipFlop → Integrator-dominant
- ✓ Cycling → Rotator-dominant
- ✓ Context → More balanced

---

## Testing Different Architectures

Compare how Vanilla RNN, LSTM, and GRU develop functional specializations:

```bash
# Vanilla RNN
python scripts/test_rnn.py --task flipflop --architecture vanilla --hidden-size 128 --epochs 2000

# LSTM
python scripts/test_rnn.py --task flipflop --architecture lstm --hidden-size 128 --epochs 2000

# GRU
python scripts/test_rnn.py --task flipflop --architecture gru --hidden-size 128 --epochs 2000
```

**Question:** Do different architectures develop similar functional specializations for the same task?

---

## Robustness Testing

Run comprehensive robustness tests:

```bash
python scripts/test_rnn_robustness.py
```

This runs 4 tests (estimated time: 30-60 minutes):

1. **Task Specificity** (3 trials × 3 tasks)
   - Verifies FlipFlop → Integrators, Cycling → Rotators

2. **Architecture Comparison** (3 trials × 3 architectures)
   - Tests consistency across RNN/LSTM/GRU

3. **Clustering Stability** (10 random seeds)
   - Checks if results are robust to initialization

4. **Hidden Size Scaling** (3 trials × 3 sizes)
   - Tests 64, 128, 256 unit networks

**Success criteria:**
- FlipFlop integrator percentage > 60%
- Clustering stability std < 0.05
- Silhouette score > 0.3 for all conditions

---

## Programmatic Usage

For custom experiments, use the Python API:

```python
from scripts.test_rnn import run_single_task_experiment

# Run experiment
results = run_single_task_experiment(
    task_name='flipflop',
    architecture='vanilla',
    hidden_size=128,
    n_epochs=2000,
    n_trials=50,
    trial_length=200,
    save_checkpoint=True,
    verbose=True
)

# Access results
print(f"Task accuracy: {results['final_accuracy']:.2%}")
print(f"Silhouette score: {results['silhouette']:.3f}")

# Unit type distribution
for cid, interp in results['interpretation'].items():
    print(f"{interp['name']}: {interp['percentage']:.1f}%")

# Deformation signals
import matplotlib.pyplot as plt
plt.plot(results['deformation']['rotation'], label='Rotation')
plt.plot(results['deformation']['contraction'], label='Contraction')
plt.plot(results['deformation']['expansion'], label='Expansion')
plt.legend()
plt.show()

# Feature space
features = results['features']
labels = results['labels']
# ... (cluster analysis)
```

---

## Common Issues & Solutions

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution:** Install PyTorch
```bash
pip install torch torchvision torchaudio
```

### Issue: Training accuracy stuck at low value (<80%)

**Possible causes:**
- Too few epochs (try 2000-3000)
- Learning rate too high/low (default 0.001 usually works)
- Network too small (try hidden_size=128 or 256)

**Solution:**
```bash
python scripts/test_rnn.py --task flipflop --hidden-size 256 --epochs 3000 --lr 0.001
```

### Issue: Silhouette score very low (<0.1)

**Possible causes:**
- RNN hasn't developed clear functional specializations
- Network undertrained or too simple
- Task doesn't require specialized dynamics

**Solution:**
1. Check task accuracy first (should be >95%)
2. Try longer training or larger network
3. Try a different task known to require complex dynamics (e.g., Cycling)

### Issue: All units classified as same type

**Possible causes:**
- Network is using homogeneous representation
- Deformation signals are too weak or noisy
- Hidden size too small for redundancy

**Solution:**
- Increase hidden_size (more units allow specialization)
- Check deformation signal ranges (should vary over time)
- Try different task with stronger dynamics

---

## Command-Line Arguments

Full list of `test_rnn.py` arguments:

```
--task            Task name: 'flipflop', 'cycling', 'context', or 'all' (default: flipflop)
--architecture    RNN type: 'vanilla', 'lstm', 'gru' (default: vanilla)
--hidden-size     Number of hidden units (default: 128)
--epochs          Training epochs (default: 2000)
--lr              Learning rate (default: 0.001)
--trials          Number of test trials for trajectory extraction (default: 50)
--trial-length    Length of each trial (default: 200)
--no-save         Don't save trained model checkpoint
--quiet           Minimal output
```

**Examples:**
```bash
# Quick test (fewer epochs)
python scripts/test_rnn.py --task flipflop --epochs 500 --quiet

# Large network
python scripts/test_rnn.py --task cycling --hidden-size 256 --epochs 3000

# All tasks, LSTM architecture
python scripts/test_rnn.py --task all --architecture lstm --hidden-size 128
```

---

## Output Files

All results are saved to `results/`:

### Checkpoints
- `results/checkpoints/{task}_{architecture}_h{hidden_size}.pt`
  - Trained RNN weights
  - Optimizer state
  - Training history

### Figures
- `results/rnn_figures/{task}_{architecture}_distribution.png`
  - Unit type distribution bar plot

### Future Enhancements
- Training curves (loss/accuracy over epochs)
- 3D latent trajectory plots
- Deformation signal timeseries
- Unit activity heatmaps
- Feature space scatter plots

Use `src.rnn_visualization` module to create custom visualizations.

---

## Next Steps

1. **Run integration test** to verify setup:
   ```bash
   python scripts/test_integration.py
   ```

2. **Single task experiment** to understand workflow:
   ```bash
   python scripts/test_rnn.py --task flipflop --epochs 2000
   ```

3. **Multi-task comparison** to test task-specificity hypothesis:
   ```bash
   python scripts/test_rnn.py --task all
   ```

4. **Robustness testing** for publication-ready validation:
   ```bash
   python scripts/test_rnn_robustness.py
   ```

5. **Custom experiments** using Python API for your specific research questions

---

## Citation

If you use this code in your research, please cite:

```
@software{flowy_neuron_id,
  title={Deformation-Based Neuron and RNN Unit Classification},
  author={[Your Name]},
  year={2026},
  url={https://github.com/[your-repo]}
}
```

---

## Support

For issues, questions, or contributions:
- Open an issue on GitHub
- See `docs/RNN_expansion_proposal.md` for detailed technical documentation
- Check `docs/project_synthesis.md` for theoretical background
