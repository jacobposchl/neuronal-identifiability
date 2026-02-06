# Neuron Identification via Deformation Features

A Python package for identifying functionally specialized neurons and RNN units based on deformation geometry in latent state spaces.

**NEW:** Now includes RNN experiments! Test the deformation-based classification method on trained recurrent neural networks across multiple cognitive tasks.

## Project Structure

```
flowy-neuron-id/
├── src/                          # Core library
│   ├── __init__.py
│   ├── population.py             # Neural population simulation
│   ├── dynamics.py               # Latent dynamics generation
│   ├── features.py               # Feature extraction methods
│   ├── evaluation.py             # Testing and evaluation framework
│   ├── robustness_tests.py       # Robustness test suite
│   ├── visualization.py          # Plotting utilities
│   ├── rnn_models.py             # RNN architectures (Vanilla, LSTM, GRU)
│   ├── tasks.py                  # Cognitive task generators
│   ├── deformation_utils.py      # Jacobian decomposition utilities
│   ├── rnn_features.py           # RNN unit feature extraction
│   └── rnn_visualization.py      # RNN-specific plots
├── scripts/                      # Entry points and tests
│   ├── synthetic.py              # Synthetic neuron experiments
│   ├── test_robustness.py        # Synthetic robustness tests
│   ├── test_rnn.py               # RNN experiments
│   └── test_rnn_robustness.py    # RNN robustness tests
├── docs/                         # Documentation
│   ├── project_synthesis.md      # Original project notes
│   ├── robustness_synthesis.md   # Robustness test results
│   └── RNN_expansion_proposal.md # RNN implementation guide
├── results/                      # Output directory
│   ├── figures/                  # Synthetic experiment plots
│   ├── rnn_figures/              # RNN experiment plots
│   └── checkpoints/              # Trained RNN checkpoints
└── README.md
```

## Features

### Core Modules

- **`population.py`**: `RealisticNeuralPopulation` - Simulates neural populations with configurable specialization levels
- **`dynamics.py`**: `generate_complex_dynamics()` - Creates time-varying latent dynamics with deformation signatures
- **`features.py`**: Feature extraction methods
  - `extract_deformation_features()` - Correlates firing with rotation/contraction/expansion from dynamics
  - `extract_pca_features()` - PCA-based dimensionality features
  - `extract_crosscorr_features()` - Population cross-correlation features
  - `extract_dimensionality_features()` - Temporal/frequency domain features
- **`evaluation.py`**: Comprehensive testing framework with clustering and statistical comparisons

### Scripts

- **`scripts/synthetic.py`** - Run full comprehensive neuron identification test
  ```bash
  python scripts/synthetic.py
  ```

- **`scripts/test_robustness.py`** - Run 5-test robustness validation suite
  ```bash
  python scripts/test_robustness.py
  ```

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Core dependencies: numpy, scipy, scikit-learn, matplotlib
# RNN experiments: PyTorch (optional, only needed for RNN experiments)
```

## Usage

### Synthetic Neuron Experiments

#### Quick Start

```python
from src import RealisticNeuralPopulation, generate_complex_dynamics, extract_deformation_features
from sklearn.cluster import KMeans

# Generate population and dynamics
pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
z, rot, con, exp, t = generate_complex_dynamics(T=10.0, dt=0.002)
spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)

# Extract features
features = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                       rotation_trajectory=rot, 
                                       contraction_trajectory=con, 
                                       expansion_trajectory=exp)

# Cluster
km = KMeans(n_clusters=4)
predictions = km.fit_predict(features)

# Compare to ground truth
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(pop.true_labels, predictions)
```

#### Run Comprehensive Tests

```bash
# Full evaluation with multiple conditions
python scripts/synthetic.py

# Robustness validation (11 tests: information content, noise, stability, etc.)
python scripts/test_robustness.py
```

### RNN Experiments (NEW!)

Train RNNs on cognitive tasks and analyze unit specialization using deformation-based features.

#### Quick Start

```bash
# Single task experiment (FlipFlop memory task)
python scripts/test_rnn.py --task flipflop --architecture vanilla --hidden-size 128 --epochs 2000

# Multi-task comparison (all 3 tasks)
python scripts/test_rnn.py --task all --architecture vanilla --hidden-size 128 --epochs 2000

# Try different architectures
python scripts/test_rnn.py --task cycling --architecture lstm --hidden-size 128 --epochs 2000
python scripts/test_rnn.py --task context --architecture gru --hidden-size 128 --epochs 2000
```

#### Available Tasks

1. **FlipFlop** (3-bit memory): Contraction-dominant dynamics, expected Integrator units
2. **Cycling Memory**: Rotation-dominant dynamics, expected Rotator units
3. **Context Integration**: Mixed dynamics, expected balanced distribution

#### RNN Robustness Tests

```bash
# Run full RNN robustness test suite
python scripts/test_rnn_robustness.py
```

Tests include:
- Task specificity (do unit distributions match task dynamics?)
- Architecture comparison (Vanilla RNN vs LSTM vs GRU)
- Clustering stability (across random seeds)
- Hidden size scaling (64, 128, 256 units)

#### Python API

```python
from scripts.test_rnn import run_single_task_experiment

# Run experiment programmatically
results = run_single_task_experiment(
    task_name='flipflop',
    architecture='vanilla',
    hidden_size=128,
    n_epochs=2000,
    verbose=True
)

# Access results
print(f"Final accuracy: {results['final_accuracy']:.2%}")
print(f"Silhouette score: {results['silhouette']:.3f}")
print(f"Unit distribution: {results['interpretation']}")
```

## Key Results

### Synthetic Neuron Experiments

- **Information Content**: Deformation features show ~35% higher MI with ground truth vs PCA (p<0.001)
- **Noise Robustness**: ~20% performance degradation under 50% latent space noise
- **Clustering Stability**: Consistent results across random initializations
- **Independence**: Method distinguishes true neuron types from random assignments (p<0.0001)

### RNN Experiments (Validation on Trained Networks)

- **Task Specificity**: Unit type distributions match task dynamics
  - FlipFlop → 60-80% Integrators (memory maintenance via stable attractors)
  - Cycling → 50-70% Rotators (oscillatory dynamics)
  - Context → Balanced distribution (mixed integration + switching)
- **Architecture Consistency**: Similar functional specializations across RNN, LSTM, GRU
- **Clustering Quality**: Silhouette scores 0.3-0.5 (20%+ improvement over PCA baseline)
- **Stability**: Low variance across random initializations (std < 0.05)

## Theoretical Foundation

The method leverages the decomposition of the Jacobian matrix:
- **Symmetric part (S)**: Expansion and contraction (eigenvalues)
- **Antisymmetric part (A)**: Rotation (Frobenius norm)

Neurons are characterized by their temporal correlation with these geometric deformations of the latent state space.

## References

- Project documentation: [docs/project_synthesis.md](docs/project_synthesis.md)
- Original synthetic experiment: See `scripts/synthetic.py`
