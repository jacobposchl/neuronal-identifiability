# Neuron Identification via Deformation Features

A Python package for identifying functionally specialized neurons based on deformation geometry in latent state spaces.

## Project Structure

```
flowy-neuron-id/
├── src/                          # Core library
│   ├── __init__.py
│   ├── population.py             # Neural population simulation
│   ├── dynamics.py               # Latent dynamics generation
│   ├── features.py               # Feature extraction methods
│   └── evaluation.py             # Testing and evaluation framework
├── scripts/                       # Entry points and tests
│   ├── synthetic.py              # Main comprehensive test
│   └── test_robustness.py        # Robustness validation suite
├── tests/                        # Unit tests (future)
├── docs/                         # Documentation
│   └── project_synthesis.md      # Original project notes
├── results/                      # Output directory
│   └── figures/                  # Generated plots
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
# No external setup required, uses standard scientific Python stack
pip install numpy scipy scikit-learn matplotlib seaborn
```

## Usage

### Quick Start

```python
from src import RealisticNeuralPopulation, generate_complex_dynamics, extract_deformation_features
from sklearn.cluster import KMeans

# Generate population and dynamics
pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
z, rot, con, exp, t = generate_complex_dynamics(T=10.0, dt=0.002)
spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)

# Extract features
features = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001)

# Cluster
km = KMeans(n_clusters=4)
predictions = km.fit_predict(features)

# Compare to ground truth
from sklearn.metrics import adjusted_rand_score
ari = adjusted_rand_score(pop.true_labels, predictions)
```

### Run Comprehensive Tests

```bash
# Full evaluation with multiple conditions
python scripts/synthetic.py

# Robustness validation (information content, noise, stability, etc.)
python scripts/test_robustness.py
```

## Key Results

- **Information Content**: Deformation features show ~35% higher MI with ground truth vs PCA (p<0.001)
- **Noise Robustness**: ~20% performance degradation under 50% latent space noise
- **Clustering Stability**: Consistent results across random initializations
- **Independence**: Method distinguishes true neuron types from random assignments (p<0.0001)

## Theoretical Foundation

The method leverages the decomposition of the Jacobian matrix:
- **Symmetric part (S)**: Expansion and contraction (eigenvalues)
- **Antisymmetric part (A)**: Rotation (Frobenius norm)

Neurons are characterized by their temporal correlation with these geometric deformations of the latent state space.

## References

- Project documentation: [docs/project_synthesis.md](docs/project_synthesis.md)
- Original synthetic experiment: See `scripts/synthetic.py`
