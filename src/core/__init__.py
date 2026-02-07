"""
Core deformation-based neuron identification library.

This module contains the fundamental components for identifying neurons
based on deformation geometry in latent state spaces.

Intended contents:
- Neural population simulation
- Latent dynamics generation
- Deformation feature extraction
- Core evaluation framework
"""

from .population import RealisticNeuralPopulation
from .dynamics import generate_complex_dynamics, estimate_deformation_from_latents
from .features import (
    extract_deformation_features,
    extract_pca_features,
    extract_crosscorr_features,
    extract_dimensionality_features
)
from .deformation_utils import (
    decompose_jacobian,
    compute_jacobian_analytical,
    compute_jacobian_numerical,
    estimate_deformation_from_latents,
    estimate_deformation_from_rnn,
    smooth_deformation_signals,
    detect_discrete_dynamics,
    validate_task_dynamics
)
from .evaluation import test_single_condition, run_comprehensive_test

__all__ = [
    'RealisticNeuralPopulation',
    'generate_complex_dynamics',
    'estimate_deformation_from_latents',
    'extract_deformation_features',
    'extract_pca_features',
    'extract_crosscorr_features',
    'extract_dimensionality_features',
    'decompose_jacobian',
    'compute_jacobian_analytical',
    'compute_jacobian_numerical',
    'estimate_deformation_from_rnn',
    'smooth_deformation_signals',
    'detect_discrete_dynamics',
    'validate_task_dynamics',
    'test_single_condition',
    'run_comprehensive_test',
]
