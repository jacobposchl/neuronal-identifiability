"""
Neuron identification via deformation-based feature extraction.

This package provides tools for identifying functionally specialized neurons
and RNN units based on deformation geometry in latent state spaces.
"""

__version__ = "0.2.0"

# Core functionality
from .core import (
    RealisticNeuralPopulation,
    generate_complex_dynamics,
    extract_deformation_features,
    extract_pca_features,
    extract_crosscorr_features,
    extract_dimensionality_features,
    decompose_jacobian,
    estimate_deformation_from_rnn,
    smooth_deformation_signals,
    test_single_condition,
    run_comprehensive_test,
)

# Models
from .models import (
    VanillaRNN,
    SimpleLSTM,
    SimpleGRU,
    build_synthetic_rnn,
)

# Tasks
from .tasks import (
    FlipFlopTask,
    CyclingMemoryTask,
    ContextIntegrationTask,
    get_task,
)

# Analysis
from .analysis import (
    extract_rnn_unit_features,
    classify_units,
    interpret_clusters,
    compare_to_baseline,
)

# Visualization
from .visualization import ensure_dirs

__all__ = [
    # Core
    'RealisticNeuralPopulation',
    'generate_complex_dynamics',
    'extract_deformation_features',
    'extract_pca_features',
    'extract_crosscorr_features',
    'extract_dimensionality_features',
    'decompose_jacobian',
    'estimate_deformation_from_rnn',
    'smooth_deformation_signals',
    'test_single_condition',
    'run_comprehensive_test',
    # Models
    'VanillaRNN',
    'SimpleLSTM',
    'SimpleGRU',
    'build_synthetic_rnn',
    # Tasks
    'FlipFlopTask',
    'CyclingMemoryTask',
    'ContextIntegrationTask',
    'get_task',
    # Analysis
    'extract_rnn_unit_features',
    'classify_units',
    'interpret_clusters',
    'compare_to_baseline',
    # Visualization
    'ensure_dirs',
]
