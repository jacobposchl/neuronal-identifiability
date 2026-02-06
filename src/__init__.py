"""
Neuron identification via deformation-based feature extraction.
"""

from .population import RealisticNeuralPopulation
from .dynamics import generate_complex_dynamics
from .features import (
    extract_deformation_features,
    extract_pca_features,
    extract_crosscorr_features,
    extract_dimensionality_features
)
from .evaluation import test_single_condition, run_comprehensive_test

__version__ = "0.1.0"
__all__ = [
    "RealisticNeuralPopulation",
    "generate_complex_dynamics",
    "extract_deformation_features",
    "extract_pca_features",
    "extract_crosscorr_features",
    "extract_dimensionality_features",
    "test_single_condition",
    "run_comprehensive_test",
]
