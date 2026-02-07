"""
Synthetic RNN generation with known ground truth unit types.

Creates RNNs with pre-specified functional unit distributions for validating
the deformation-based classification method. Units are initialized with
spectral properties that enforce specific dynamical behaviors.
"""

import numpy as np
import torch
import torch.nn as nn
from src.models.rnn_models import VanillaRNN


def generate_integrator_weights(input_dim, hidden_dim, n_integrators, bias_value=-0.95):
    """
    Generate weights for integrator units (line attractors).
    
    Integrator units maintain information over time via eigenvalues near 1.
    Implemented via: h_t = (1-α)h_{t-1} + α·input, where α is small.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size
        n_integrators: Number of integrator units to create
        bias_value: Bias term (negative values create leak, positive create growth)
    
    Returns:
        Whh_integrator: (n_integrators, hidden_dim) recurrent weight matrix
        Why_integrator: (n_integrators, input_dim) input weight matrix  
        bias_integrator: (n_integrators,) bias vector
    """
    # Recurrent weights: near-identity for self-connection
    Whh = np.zeros((n_integrators, hidden_dim))
    for i in range(n_integrators):
        Whh[i, i] = 0.98  # Eigenvalue near 1 (stable integrator)
        # Small random off-diagonal for coupling
        Whh[i, :] += np.random.randn(hidden_dim) * 0.01
    
    # Input weights: random projection
    Why = np.random.randn(n_integrators, input_dim) * 0.3
    
    # Bias: slight leak
    bias = np.ones(n_integrators) * bias_value
    
    return Whh, Why, bias


def generate_rotator_weights(input_dim, hidden_dim, n_rotators, frequency=0.3):
    """
    Generate weights for rotator units (oscillators).
    
    Rotator units exhibit periodic dynamics via complex conjugate eigenvalues.
    Implemented via 2D rotation matrices embedded in weight space.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size
        n_rotators: Number of rotator units (must be even for conjugate pairs)
        frequency: Oscillation frequency (radians per timestep)
    
    Returns:
        Whh_rotator: (n_rotators, hidden_dim) recurrent weight matrix
        Why_rotator: (n_rotators, input_dim) input weight matrix
        bias_rotator: (n_rotators,) bias vector
    """
    assert n_rotators % 2 == 0, "n_rotators must be even (conjugate pairs)"
    
    Whh = np.zeros((n_rotators, hidden_dim))
    
    # Create rotation matrices for pairs
    for pair_idx in range(n_rotators // 2):
        i = pair_idx * 2
        j = i + 1
        
        # 2D rotation matrix with frequency ω
        omega = frequency + np.random.randn() * 0.05  # Add variance
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        
        # Embed rotation in recurrent weights
        Whh[i, i] = cos_omega
        Whh[i, j] = -sin_omega
        Whh[j, i] = sin_omega
        Whh[j, j] = cos_omega
        
        # Add coupling to other units
        Whh[i, :] += np.random.randn(hidden_dim) * 0.02
        Whh[j, :] += np.random.randn(hidden_dim) * 0.02
    
    # Input weights
    Why = np.random.randn(n_rotators, input_dim) * 0.2
    
    # Minimal bias
    bias = np.random.randn(n_rotators) * 0.1
    
    return Whh, Why, bias


def generate_explorer_weights(input_dim, hidden_dim, n_explorers, expansion_rate=1.1):
    """
    Generate weights for explorer units (expanding/unstable).
    
    Explorer units have eigenvalues >1, causing expansion unless constrained
    by nonlinearity (tanh saturation).
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size
        n_explorers: Number of explorer units
        expansion_rate: Eigenvalue magnitude (>1 for expansion)
    
    Returns:
        Whh_explorer: (n_explorers, hidden_dim) recurrent weight matrix
        Why_explorer: (n_explorers, input_dim) input weight matrix
        bias_explorer: (n_explorers,) bias vector
    """
    Whh = np.zeros((n_explorers, hidden_dim))
    
    for i in range(n_explorers):
        # Self-connection with expansion
        Whh[i, i] = expansion_rate + np.random.randn() * 0.05
        # Random couplings
        Whh[i, :] += np.random.randn(hidden_dim) * 0.05
    
    # Strong input weights (explorers respond to inputs)
    Why = np.random.randn(n_explorers, input_dim) * 0.5
    
    # Random bias
    bias = np.random.randn(n_explorers) * 0.2
    
    return Whh, Why, bias


def generate_mixed_weights(input_dim, hidden_dim, n_mixed):
    """
    Generate weights for mixed/generic units.
    
    Standard random initialization without specific spectral properties.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size
        n_mixed: Number of mixed units
    
    Returns:
        Whh_mixed: (n_mixed, hidden_dim) recurrent weight matrix
        Why_mixed: (n_mixed, input_dim) input weight matrix
        bias_mixed: (n_mixed,) bias vector
    """
    # Random orthogonal initialization
    Whh = np.random.randn(n_mixed, hidden_dim) * 0.5 / np.sqrt(hidden_dim)
    
    # Random input weights
    Why = np.random.randn(n_mixed, input_dim) * 0.3
    
    # Random bias
    bias = np.random.randn(n_mixed) * 0.1
    
    return Whh, Why, bias


def build_synthetic_rnn(input_dim, hidden_size, output_dim,
                        n_integrators=30, n_rotators=20, 
                        n_explorers=15, n_mixed=None):
    """
    Build VanillaRNN with known unit type distribution.
    
    Creates an RNN where unit types are specified a priori through their
    recurrent weight initialization. This provides ground truth labels for
    validating the deformation-based classification method.
    
    Args:
        input_dim: Input feature dimension
        hidden_size: Total hidden units (must equal sum of unit types)
        output_dim: Output dimension
        n_integrators: Number of integrator units
        n_rotators: Number of rotator units (must be even)
        n_explorers: Number of explorer units
        n_mixed: Number of mixed units (if None, fills remaining)
    
    Returns:
        rnn: VanillaRNN with specialized initialization
        ground_truth_labels: (hidden_size,) array with unit type for each unit
                           0=Integrator, 1=Rotator, 2=Explorer, 3=Mixed
    """
    # Validate inputs
    if n_rotators % 2 != 0:
        n_rotators = n_rotators - 1  # Make even
        print(f"  Warning: n_rotators must be even, adjusted to {n_rotators}")
    
    # Calculate mixed units to fill remaining space
    if n_mixed is None:
        n_mixed = hidden_size - n_integrators - n_rotators - n_explorers
        if n_mixed < 0:
            raise ValueError(f"Unit counts exceed hidden_size: {n_integrators + n_rotators + n_explorers} > {hidden_size}")
    
    total = n_integrators + n_rotators + n_explorers + n_mixed
    if total != hidden_size:
        raise ValueError(f"Unit counts ({total}) must equal hidden_size ({hidden_size})")
    
    print(f"\nBuilding synthetic RNN:")
    print(f"  Hidden size: {hidden_size}")
    print(f"  Integrators: {n_integrators} ({100*n_integrators/hidden_size:.1f}%)")
    print(f"  Rotators:    {n_rotators} ({100*n_rotators/hidden_size:.1f}%)")
    print(f"  Explorers:   {n_explorers} ({100*n_explorers/hidden_size:.1f}%)")
    print(f"  Mixed:       {n_mixed} ({100*n_mixed/hidden_size:.1f}%)")
    
    # Generate weights for each unit type
    Whh_int, Why_int, bias_int = generate_integrator_weights(
        input_dim, hidden_size, n_integrators
    )
    Whh_rot, Why_rot, bias_rot = generate_rotator_weights(
        input_dim, hidden_size, n_rotators
    )
    Whh_exp, Why_exp, bias_exp = generate_explorer_weights(
        input_dim, hidden_size, n_explorers
    )
    Whh_mix, Why_mix, bias_mix = generate_mixed_weights(
        input_dim, hidden_size, n_mixed
    )
    
    # Concatenate all weight matrices
    Whh = np.vstack([Whh_int, Whh_rot, Whh_exp, Whh_mix])  # (hidden_size, hidden_size)
    Why = np.vstack([Why_int, Why_rot, Why_exp, Why_mix])  # (hidden_size, input_dim)
    bias_h = np.concatenate([bias_int, bias_rot, bias_exp, bias_mix])  # (hidden_size,)
    
    # Create ground truth labels
    labels = np.concatenate([
        np.zeros(n_integrators, dtype=int),      # 0 = Integrator
        np.ones(n_rotators, dtype=int),          # 1 = Rotator
        np.ones(n_explorers, dtype=int) * 2,     # 2 = Explorer
        np.ones(n_mixed, dtype=int) * 3          # 3 = Mixed
    ])
    
    # Create RNN and initialize with synthetic weights
    rnn = VanillaRNN(input_dim, hidden_size, output_dim)
    
    # Convert to torch and assign
    # PyTorch RNN weights: weight_ih_l0 is (hidden_size, input_size)
    #                      weight_hh_l0 is (hidden_size, hidden_size)
    with torch.no_grad():
        rnn.rnn.weight_hh_l0.copy_(torch.from_numpy(Whh).float())
        rnn.rnn.weight_ih_l0.copy_(torch.from_numpy(Why).float())
        rnn.rnn.bias_hh_l0.copy_(torch.from_numpy(bias_h).float())
        # Keep default bias_ih
    
    print(f"  Synthetic RNN initialized with ground truth unit types")
    
    return rnn, labels


def verify_spectral_properties(rnn, ground_truth_labels):
    """
    Verify that synthetic RNN has expected spectral properties.
    
    Checks that eigenvalues match expected unit type characteristics:
    - Integrators: Real eigenvalues near 1
    - Rotators: Complex conjugate pairs
    - Explorers: Real eigenvalues > 1
    
    Args:
        rnn: Synthetic RNN
        ground_truth_labels: Ground truth unit type labels
    
    Returns:
        verification_report: Dict with verification statistics
    """
    # Extract recurrent weight matrix
    Whh = rnn.rnn.weight_hh_l0.detach().cpu().numpy().T  # (hidden_size, hidden_size)
    
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(Whh)
    
    # Separate by label
    n_integrators = np.sum(ground_truth_labels == 0)
    n_rotators = np.sum(ground_truth_labels == 1)
    n_explorers = np.sum(ground_truth_labels == 2)
    
    # Integrator eigenvalues (first n_integrators)
    eig_int = eigenvalues[:n_integrators]
    int_real = np.sum(np.abs(eig_int.imag) < 0.1) / len(eig_int)
    int_near_one = np.sum(np.abs(np.abs(eig_int) - 1.0) < 0.1) / len(eig_int)
    
    # Rotator eigenvalues
    eig_rot = eigenvalues[n_integrators:n_integrators + n_rotators]
    rot_complex = np.sum(np.abs(eig_rot.imag) > 0.1) / len(eig_rot)
    
    # Explorer eigenvalues
    eig_exp = eigenvalues[n_integrators + n_rotators:n_integrators + n_rotators + n_explorers]
    exp_large = np.sum(np.abs(eig_exp) > 1.05) / len(eig_exp)
    
    report = {
        'integrator_are_real': int_real,
        'integrator_near_one': int_near_one,
        'rotator_are_complex': rot_complex,
        'explorer_are_large': exp_large,
        'all_eigenvalues': eigenvalues
    }
    
    print(f"\nSpectral Property Verification:")
    print(f"  Integrators: {int_real*100:.1f}% real, {int_near_one*100:.1f}% near |λ|=1")
    print(f"  Rotators:    {rot_complex*100:.1f}% complex")
    print(f"  Explorers:   {exp_large*100:.1f}% |λ| > 1.05")
    
    return report
