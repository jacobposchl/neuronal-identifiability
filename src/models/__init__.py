"""
Neural network model architectures.

This module contains RNN and other neural network model implementations.

Intended contents:
- RNN architectures (Vanilla, LSTM, GRU)
- Synthetic RNN generators
- Model utilities and initialization
"""

from .rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU, count_parameters, initialize_weights
from .synthetic_rnn import (
    generate_integrator_weights,
    generate_rotator_weights,
    generate_explorer_weights,
    generate_mixed_weights,
    build_synthetic_rnn,
    verify_spectral_properties
)

__all__ = [
    'VanillaRNN',
    'SimpleLSTM',
    'SimpleGRU',
    'count_parameters',
    'initialize_weights',
    'generate_integrator_weights',
    'generate_rotator_weights',
    'generate_explorer_weights',
    'generate_mixed_weights',
    'build_synthetic_rnn',
    'verify_spectral_properties',
]
