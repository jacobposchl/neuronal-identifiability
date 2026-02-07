# Codebase Reorganization Guide

## New Directory Structure

```
neuronal-identifiability/
├── src/
│   ├── core/              # Core deformation-based identification
│   ├── models/            # Neural network models
│   ├── tasks/             # Cognitive task definitions
│   ├── analysis/          # Analysis tools
│   └── visualization/     # Visualization utilities
├── experiments/
│   ├── synthetic/         # Synthetic neuron experiments
│   └── rnn/              # RNN experiments
├── tests/                 # Unit and integration tests
├── notebooks/            # Jupyter notebooks
├── docs/                 # Documentation
├── results/              # Output directory
├── README.md
├── requirements.txt
└── LICENSE
```

## File Migration Plan

### Current `src/` Files → New Locations

#### To `src/core/`:
- [✓] **population.py** - Neural population simulation (core synthetic neuron functionality)
- [✓] **dynamics.py** - Latent dynamics generation (core dynamics)
- [✓] **features.py** - Basic feature extraction methods
- [✓] **deformation_utils.py** - Jacobian decomposition and deformation estimation utilities
- [✓] **evaluation.py** - Core testing framework

**Rationale**: These are fundamental to the deformation-based identification approach and used by both synthetic and RNN experiments.

#### To `src/models/`:
- [✓] **rnn_models.py** - RNN architectures (Vanilla, LSTM, GRU)
- [✓] **synthetic_rnn.py** - Synthetic RNN generators with known ground truth

**Rationale**: These define neural network architectures.

#### To `src/tasks/`:
- [✓] **tasks.py** - All cognitive task definitions (8 different task classes)

**Rationale**: Task definitions are separate from models and form their own logical unit.

#### To `src/analysis/`:
- [✓] **rnn_features.py** - RNN unit feature extraction and classification
- [✓] **spectral_analysis.py** - Eigenvalue analysis and spectral methods
- [✓] **perturbation.py** - Ablation and perturbation analysis tools
- [✓] **statistical_tests.py** - Statistical testing utilities
- [✓] **robustness_tests.py** - Robustness test suite for synthetic neurons

**Rationale**: These are advanced analysis methods that operate on models/data.

#### To `src/visualization/`:
- [✓] **visualization.py** - General plotting utilities
- [✓] **rnn_visualization.py** - RNN-specific visualization

**Rationale**: All plotting functionality grouped together.

#### Keep in `src/` (root):
- [✓] **__init__.py** - Main package exports (needs updating)

---

### Current `scripts/` Files → New Locations

#### To `experiments/synthetic/`:
- [✓] **synthetic.py** → **basic_experiment.py**
  - Main synthetic neuron identification experiment
  
- [✓] **test_robustness.py** → **robustness_suite.py**
  - Robustness tests for synthetic populations

**Rationale**: These run experiments on synthetic neural populations.

#### To `experiments/rnn/`:
- [✓] **test_rnn.py** → **basic_experiment.py**
  - Main RNN deformation identification experiment
  
- [✓] **test_rnn_robustness.py** → **robustness_suite.py**
  - RNN robustness validation tests
  
- [✓] **test_perturbation.py** → **perturbation_experiments.py**
  - Unit importance and perturbation analysis
  
- [✓] **test_pruning.py** → **pruning_experiments.py**
  - Network compression and pruning studies
  
- [✓] **test_ground_truth.py** → **ground_truth_validation.py**
  - Validation against synthetic RNNs with known ground truth
  
- [✓] **test_comprehensive_analysis.py** → **comprehensive_analysis.py**
  - Full multi-task analysis workflow
  
- [✓] **test_integration.py** → **integration_tests.py** OR delete if redundant
  - Integration tests (review contents)

**Rationale**: These are experimental workflows for RNN analysis.

#### Delete or Move:
- [?] **scripts/__init__.py** - Empty file, can be deleted
- [?] **scripts/__pycache__/** - Already in .gitignore

---

### Files to Keep As-Is:
- **notebooks/RNN_Experiments.ipynb**
- **docs/project_synthesis.md**
- **docs/RNN_expansion_proposal.md**
- **README.md** (will need updating)
- **requirements.txt**
- **LICENSE**

---

## Migration Steps

For each file, we will:
1. Review the file contents
2. Confirm the destination
3. Move the file
4. Update import statements (as needed)
5. Test that nothing breaks

## Post-Migration Tasks

1. Update `src/__init__.py` to export from new submodules
2. Update README.md with new structure
3. Update all import statements in notebooks
4. Verify all experiments still run
5. Remove old `scripts/` directory
6. Consider consolidating duplicate functionality:
   - `dynamics.py::estimate_deformation_from_latents()` 
   - `deformation_utils.py::estimate_deformation_from_latents()`

---

## Notes

- **No function changes** - Only moving files and updating imports
- **Test incrementally** - Move a few files at a time and test
- **Keep backup** - Ensure code is committed to git before major changes
- Files like `robustness_tests.py` (currently in `src/`) are actually experiment suites, not core library code
