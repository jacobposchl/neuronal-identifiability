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


def generate_integrator_weights(input_dim, hidden_dim, n_integrators, start_idx=0, bias_value=-0.95):
    """
    Generate weights for integrator units (line attractors).
    
    Integrator units maintain information over time via eigenvalues near 1.
    Uses spectral construction: A = VΛV^{-1} to guarantee eigenvalues.
    
    CRITICAL: Only couples within integrator block to maintain block-diagonal structure.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size (total, not just integrators)
        n_integrators: Number of integrator units to create
        start_idx: Starting index in full weight matrix (for proper block placement)
        bias_value: Bias term (negative values create leak, positive create growth)
    
    Returns:
        Whh_integrator: (n_integrators, hidden_dim) recurrent weight matrix
        Why_integrator: (n_integrators, input_dim) input weight matrix  
        bias_integrator: (n_integrators,) bias vector
    """
    # Construct matrix with eigenvalues near 1 using spectral decomposition
    # Target eigenvalues: real, near 1 (between 0.95 and 1.0)
    eigenvalues = np.random.uniform(0.95, 1.0, n_integrators)
    
    # Random orthogonal eigenvector matrix
    V = np.linalg.qr(np.random.randn(n_integrators, n_integrators))[0]
    
    # Construct block weight matrix: A = V * diag(eigenvalues) * V^T
    Whh_block = V @ np.diag(eigenvalues) @ V.T
    
    # Embed in full-size weight matrix (zeros outside block)
    Whh = np.zeros((n_integrators, hidden_dim))
    for i in range(n_integrators):
        for j in range(n_integrators):
            Whh[i, start_idx + j] = Whh_block[i, j]
    
    # Input weights: random projection
    Why = np.random.randn(n_integrators, input_dim) * 0.3
    
    # Bias: slight leak
    bias = np.ones(n_integrators) * bias_value
    
    return Whh, Why, bias


def generate_rotator_weights(input_dim, hidden_dim, n_rotators, start_idx=0, frequency=0.3):
    """
    Generate weights for rotator units (oscillators).
    
    Rotator units exhibit periodic dynamics via complex conjugate eigenvalues.
    Implemented via 2D rotation matrices embedded in weight space.
    
    CRITICAL: Only couples within rotator block to maintain block-diagonal structure.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size (total, not just rotators)
        n_rotators: Number of rotator units (must be even for conjugate pairs)
        start_idx: Starting index in full weight matrix (for proper block placement)
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
        local_i = pair_idx * 2
        local_j = local_i + 1
        global_i = start_idx + local_i
        global_j = start_idx + local_j
        
        # 2D rotation matrix with frequency ω
        omega = frequency + np.random.randn() * 0.05  # Add variance
        cos_omega = np.cos(omega)
        sin_omega = np.sin(omega)
        
        # Embed rotation in recurrent weights (at correct global indices)
        Whh[local_i, global_i] = cos_omega
        Whh[local_i, global_j] = -sin_omega
        Whh[local_j, global_i] = sin_omega
        Whh[local_j, global_j] = cos_omega
        
        # Weak coupling to other rotator pairs ONLY (within block)
        for other_pair in range(n_rotators // 2):
            if other_pair != pair_idx:
                other_i = start_idx + other_pair * 2
                other_j = other_i + 1
                Whh[local_i, other_i] += np.random.randn() * 0.02
                Whh[local_i, other_j] += np.random.randn() * 0.02
                Whh[local_j, other_i] += np.random.randn() * 0.02
                Whh[local_j, other_j] += np.random.randn() * 0.02
    
    # Input weights
    Why = np.random.randn(n_rotators, input_dim) * 0.2
    
    # Minimal bias
    bias = np.random.randn(n_rotators) * 0.1
    
    return Whh, Why, bias


def generate_explorer_weights(input_dim, hidden_dim, n_explorers, start_idx=0, expansion_rate=1.1):
    """
    Generate weights for explorer units (expanding/unstable).
    
    Explorer units have eigenvalues >1, causing expansion unless constrained
    by nonlinearity (tanh saturation).
    Uses spectral construction: A = VΛV^{-1} to guarantee eigenvalues.
    
    CRITICAL: Only couples within explorer block to maintain block-diagonal structure.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size (total, not just explorers)
        n_explorers: Number of explorer units
        start_idx: Starting index in full weight matrix (for proper block placement)
        expansion_rate: Eigenvalue magnitude (>1 for expansion)
    
    Returns:
        Whh_explorer: (n_explorers, hidden_dim) recurrent weight matrix
        Why_explorer: (n_explorers, input_dim) input weight matrix
        bias_explorer: (n_explorers,) bias vector
    """
    # Construct matrix with eigenvalues > 1 using spectral decomposition
    # Target eigenvalues: real, > 1 (between 1.05 and 1.2)
    eigenvalues = np.random.uniform(1.05, 1.2, n_explorers)
    
    # Random orthogonal eigenvector matrix
    V = np.linalg.qr(np.random.randn(n_explorers, n_explorers))[0]
    
    # Construct block weight matrix: A = V * diag(eigenvalues) * V^T
    Whh_block = V @ np.diag(eigenvalues) @ V.T
    
    # Embed in full-size weight matrix (zeros outside block)
    Whh = np.zeros((n_explorers, hidden_dim))
    for i in range(n_explorers):
        for j in range(n_explorers):
            Whh[i, start_idx + j] = Whh_block[i, j]
    
    # Strong input weights (explorers respond to inputs)
    Why = np.random.randn(n_explorers, input_dim) * 0.5
    
    # Random bias
    bias = np.random.randn(n_explorers) * 0.2
    
    return Whh, Why, bias


def generate_mixed_weights(input_dim, hidden_dim, n_mixed, start_idx=0):
    """
    Generate weights for mixed/generic units.
    
    Mixed units have eigenvalues scattered around 1 (mixture of behaviors).
    Uses spectral construction: A = VΛV^{-1} with mixed eigenvalue types.
    
    CRITICAL: Only couples within mixed block to maintain block-diagonal structure.
    
    Args:
        input_dim: Input dimensionality
        hidden_dim: Hidden layer size (total, not just mixed)
        n_mixed: Number of mixed units
        start_idx: Starting index in full weight matrix (for proper block placement)
    
    Returns:
        Whh_mixed: (n_mixed, hidden_dim) recurrent weight matrix
        Why_mixed: (n_mixed, input_dim) input weight matrix
        bias_mixed: (n_mixed,) bias vector
    """
    # Construct matrix with mixed eigenvalues using spectral decomposition
    # Eigenvalues scattered around 1: some <1, some ≈1, some >1
    eigenvalues = np.random.uniform(0.8, 1.15, n_mixed)
    
    # Random orthogonal eigenvector matrix
    V = np.linalg.qr(np.random.randn(n_mixed, n_mixed))[0]
    
    # Construct block weight matrix: A = V * diag(eigenvalues) * V^T
    Whh_block = V @ np.diag(eigenvalues) @ V.T
    
    # Embed in full-size weight matrix (zeros outside block)
    Whh = np.zeros((n_mixed, hidden_dim))
    for i in range(n_mixed):
        for j in range(n_mixed):
            Whh[i, start_idx + j] = Whh_block[i, j]
    
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
    
    # Calculate starting indices for each block
    start_int = 0
    start_rot = n_integrators
    start_exp = n_integrators + n_rotators
    start_mix = n_integrators + n_rotators + n_explorers
    
    # Generate weights for each unit type with proper block placement
    Whh_int, Why_int, bias_int = generate_integrator_weights(
        input_dim, hidden_size, n_integrators, start_idx=start_int
    )
    Whh_rot, Why_rot, bias_rot = generate_rotator_weights(
        input_dim, hidden_size, n_rotators, start_idx=start_rot
    )
    Whh_exp, Why_exp, bias_exp = generate_explorer_weights(
        input_dim, hidden_size, n_explorers, start_idx=start_exp
    )
    Whh_mix, Why_mix, bias_mix = generate_mixed_weights(
        input_dim, hidden_size, n_mixed, start_idx=start_mix
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
    
    Checks that eigenvalues of each BLOCK match expected unit type characteristics:
    - Integrators: Real eigenvalues near 1
    - Rotators: Complex conjugate pairs
    - Explorers: Real eigenvalues > 1
    
    CRITICAL FIX: Computes eigenvalues of each block separately (block-diagonal structure).
    
    Args:
        rnn: Synthetic RNN
        ground_truth_labels: Ground truth unit type labels
    
    Returns:
        verification_report: Dict with verification statistics
    """
    # Extract recurrent weight matrix
    Whh = rnn.rnn.weight_hh_l0.detach().cpu().numpy().T  # (hidden_size, hidden_size)
    
    # Get block sizes
    n_integrators = np.sum(ground_truth_labels == 0)
    n_rotators = np.sum(ground_truth_labels == 1)
    n_explorers = np.sum(ground_truth_labels == 2)
    n_mixed = np.sum(ground_truth_labels == 3)
    
    # Extract each block from the weight matrix
    start_int = 0
    start_rot = n_integrators
    start_exp = n_integrators + n_rotators
    start_mix = n_integrators + n_rotators + n_explorers
    
    # Integrator block
    Whh_int = Whh[start_int:start_int+n_integrators, start_int:start_int+n_integrators]
    eig_int = np.linalg.eigvals(Whh_int)
    int_real = np.sum(np.abs(eig_int.imag) < 0.1) / len(eig_int) if len(eig_int) > 0 else 0
    int_near_one = np.sum(np.abs(np.abs(eig_int) - 1.0) < 0.1) / len(eig_int) if len(eig_int) > 0 else 0
    
    # Rotator block
    Whh_rot = Whh[start_rot:start_rot+n_rotators, start_rot:start_rot+n_rotators]
    eig_rot = np.linalg.eigvals(Whh_rot)
    rot_complex = np.sum(np.abs(eig_rot.imag) > 0.1) / len(eig_rot) if len(eig_rot) > 0 else 0
    
    # Explorer block
    Whh_exp = Whh[start_exp:start_exp+n_explorers, start_exp:start_exp+n_explorers]
    eig_exp = np.linalg.eigvals(Whh_exp)
    exp_large = np.sum(np.abs(eig_exp) > 1.05) / len(eig_exp) if len(eig_exp) > 0 else 0
    
    # Verify block-diagonal structure (should have near-zero off-block elements)
    off_block_magnitude = 0
    total_off_block = 0
    
    # Check integrator-rotator coupling
    if n_integrators > 0 and n_rotators > 0:
        off_block_magnitude += np.sum(np.abs(Whh[start_int:start_int+n_integrators, start_rot:start_rot+n_rotators]))
        total_off_block += n_integrators * n_rotators
    
    # Check integrator-explorer coupling
    if n_integrators > 0 and n_explorers > 0:
        off_block_magnitude += np.sum(np.abs(Whh[start_int:start_int+n_integrators, start_exp:start_exp+n_explorers]))
        total_off_block += n_integrators * n_explorers
    
    # Check rotator-explorer coupling
    if n_rotators > 0 and n_explorers > 0:
        off_block_magnitude += np.sum(np.abs(Whh[start_rot:start_rot+n_rotators, start_exp:start_exp+n_explorers]))
        total_off_block += n_rotators * n_explorers
    
    avg_off_block = off_block_magnitude / total_off_block if total_off_block > 0 else 0
    
    report = {
        'integrator_are_real': int_real,
        'integrator_near_one': int_near_one,
        'rotator_are_complex': rot_complex,
        'explorer_are_large': exp_large,
        'block_diagonal_quality': 1.0 - min(1.0, avg_off_block),  # 1.0 = perfect block diagonal
        'integrator_eigenvalues': eig_int,
        'rotator_eigenvalues': eig_rot,
        'explorer_eigenvalues': eig_exp
    }
    
    print(f"\nSpectral Property Verification:")
    print(f"  Integrators: {int_real*100:.1f}% real, {int_near_one*100:.1f}% near |λ|=1")
    print(f"  Rotators:    {rot_complex*100:.1f}% complex")
    print(f"  Explorers:   {exp_large*100:.1f}% |λ| > 1.05")
    print(f"  Block-diagonal quality: {report['block_diagonal_quality']*100:.1f}% (avg off-block weight: {avg_off_block:.4f})")
    
    return report
