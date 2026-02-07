"""
Spectral analysis and eigenvalue-based validation for RNN dynamics.

Provides tools to analyze the eigenvalue spectrum of recurrent weight matrices
and validate that deformation-based unit classifications align with spectral
properties predicted by dynamical systems theory.
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from scipy import stats


def compute_recurrent_eigenvalues(rnn):
    """
    Extract eigenvalues of recurrent weight matrix.
    
    Args:
        rnn: PyTorch RNN model with weight_hh_l0 parameter
    
    Returns:
        eigenvalues: Complex array of eigenvalues
        eigenvectors: Eigenvectors (optional, for mode analysis)
    """
    # Extract recurrent weight matrix
    if hasattr(rnn, 'rnn'):
        # VanillaRNN wrapper
        Whh = rnn.rnn.weight_hh_l0.detach().cpu().numpy().T
    elif hasattr(rnn, 'weight_hh_l0'):
        # Direct RNN module
        Whh = rnn.weight_hh_l0.detach().cpu().numpy().T
    else:
        raise ValueError("RNN structure not recognized")
    
    # Compute eigendecomposition
    eigenvalues, eigenvectors = np.linalg.eig(Whh)
    
    return eigenvalues, eigenvectors


def classify_eigenvalues(eigenvalues, threshold_complex=0.1, threshold_stable=1.05):
    """
    Classify eigenvalues by dynamical type.
    
    Classification criteria:
    - Integrator: Real eigenvalue near 1 (0.90 < |λ| < 1.05, |Im(λ)| < 0.1)
    - Rotator: Complex eigenvalue (|Im(λ)| > 0.1)
    - Explorer: Large eigenvalue (|λ| > 1.05)
    - Contracting: Small eigenvalue (|λ| < 0.90)
    
    Args:
        eigenvalues: Array of complex eigenvalues
        threshold_complex: Imaginary part threshold for complex classification
        threshold_stable: Magnitude threshold for stability
    
    Returns:
        classifications: Array of strings ['integrator', 'rotator', 'explorer', 'contracting']
    """
    mags = np.abs(eigenvalues)
    imags = np.abs(eigenvalues.imag)
    
    classifications = []
    for mag, imag in zip(mags, imags):
        if imag > threshold_complex:
            classifications.append('rotator')
        elif mag > threshold_stable:
            classifications.append('explorer')
        elif mag > 0.90:
            classifications.append('integrator')
        else:
            classifications.append('contracting')
    
    return np.array(classifications)


def compare_eigenvalues_to_deformation(rnn, deformation_labels, label_names=None):
    """
    Validate deformation classification against spectral properties.
    
    Theory prediction:
    - Units classified as 'Rotator' should have complex eigenvalues
    - Units classified as 'Integrator' should have real eigenvalues near 1
    - Units classified as 'Explorer' should have large eigenvalues
    
    Args:
        rnn: RNN model
        deformation_labels: (n_units,) cluster labels from deformation method
        label_names: Dict mapping label indices to names (e.g., {0: 'Rotator', ...})
    
    Returns:
        comparison: Dict with statistics and chi-square test results
    """
    eigenvalues, _ = compute_recurrent_eigenvalues(rnn)
    spectral_types = classify_eigenvalues(eigenvalues)
    
    n_units = len(eigenvalues)
    
    # Build contingency table
    unique_deform_labels = np.unique(deformation_labels)
    unique_spectral = np.unique(spectral_types)
    
    contingency = np.zeros((len(unique_deform_labels), len(unique_spectral)))
    
    for i, deform_label in enumerate(unique_deform_labels):
        for j, spec_type in enumerate(unique_spectral):
            count = np.sum((deformation_labels == deform_label) & (spectral_types == spec_type))
            contingency[i, j] = count
    
    # Chi-square test
    chi2, p_value, dof, expected = stats.chi2_contingency(contingency)
    
    # Compute per-type agreement
    agreement = {}
    if label_names is None:
        label_names = {i: f'Cluster {i}' for i in unique_deform_labels}
    
    for deform_label in unique_deform_labels:
        mask = deformation_labels == deform_label
        spec_counts = {
            'rotator': np.sum((spectral_types == 'rotator') & mask),
            'integrator': np.sum((spectral_types == 'integrator') & mask),
            'explorer': np.sum((spectral_types == 'explorer') & mask),
            'contracting': np.sum((spectral_types == 'contracting') & mask)
        }
        total = np.sum(mask)
        spec_fractions = {k: v/total for k, v in spec_counts.items()} if total > 0 else {}
        
        agreement[label_names.get(deform_label, f'Cluster {deform_label}')] = {
            'counts': spec_counts,
            'fractions': spec_fractions,
            'n_units': total
        }
    
    comparison = {
        'contingency_table': contingency,
        'chi2_statistic': chi2,
        'p_value': p_value,
        'degrees_of_freedom': dof,
        'per_type_agreement': agreement,
        'spectral_types': spectral_types,
        'eigenvalues': eigenvalues
    }
    
    return comparison


def print_spectral_comparison(comparison, label_names=None):
    """
    Print human-readable spectral comparison results.
    
    Args:
        comparison: Output from compare_eigenvalues_to_deformation()
        label_names: Optional dict mapping labels to names
    """
    print("\n" + "="*70)
    print("SPECTRAL VALIDATION OF DEFORMATION CLASSIFICATION")
    print("="*70)
    
    print(f"\nChi-square test: χ² = {comparison['chi2_statistic']:.2f}, " +
          f"p = {comparison['p_value']:.2e}, dof = {comparison['degrees_of_freedom']}")
    
    if comparison['p_value'] < 0.001:
        print("  *** Highly significant association between deformation and spectral types ***")
    elif comparison['p_value'] < 0.05:
        print("  ** Significant association **")
    else:
        print("  No significant association (p > 0.05)")
    
    print("\nPer-Type Spectral Properties:")
    print("-" * 70)
    
    for type_name, stats in comparison['per_type_agreement'].items():
        print(f"\n{type_name} ({stats['n_units']} units):")
        fracs = stats['fractions']
        print(f"  Rotator eigenvalues:     {fracs.get('rotator', 0)*100:5.1f}%")
        print(f"  Integrator eigenvalues:  {fracs.get('integrator', 0)*100:5.1f}%")
        print(f"  Explorer eigenvalues:    {fracs.get('explorer', 0)*100:5.1f}%")
        print(f"  Contracting eigenvalues: {fracs.get('contracting', 0)*100:5.1f}%")
    
    print("="*70)


def plot_eigenvalue_spectrum(eigenvalues, deformation_labels, 
                              label_names=None, save_path=None):
    """
    Plot eigenvalue spectrum in complex plane, colored by deformation type.
    
    Args:
        eigenvalues: Complex eigenvalues
        deformation_labels: Cluster labels from deformation method
        label_names: Dict mapping labels to names
        save_path: Optional path to save figure
    
    Returns:
        fig, ax: Matplotlib figure and axis objects
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Unit circle
    theta = np.linspace(0, 2*np.pi, 100)
    ax.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=2, label='Unit circle')
    
    # Plot eigenvalues by cluster
    unique_labels = np.unique(deformation_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = deformation_labels == label
        eigs = eigenvalues[mask]
        
        if label_names is not None and label in label_names:
            name = label_names[label]
        else:
            name = f'Cluster {label}'
        
        ax.scatter(eigs.real, eigs.imag, 
                  c=[colors[i]], s=100, alpha=0.7, 
                  edgecolors='black', linewidth=1,
                  label=name)
    
    ax.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax.set_xlabel('Real Part', fontsize=14)
    ax.set_ylabel('Imaginary Part', fontsize=14)
    ax.set_title('Eigenvalue Spectrum by Deformation Type', fontsize=16)
    ax.legend(loc='upper right', fontsize=12)
    ax.grid(True, alpha=0.2)
    ax.set_aspect('equal')
    
    # Set reasonable limits
    max_val = max(np.max(np.abs(eigenvalues.real)), np.max(np.abs(eigenvalues.imag))) * 1.1
    ax.set_xlim(-max_val, max_val)
    ax.set_ylim(-max_val, max_val)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved eigenvalue spectrum: {save_path}")
    
    return fig, ax


def compute_lyapunov_spectrum(rnn, hidden_state, n_steps=100):
    """
    Estimate local Lyapunov exponents via finite-time perturbations.
    
    Measures exponential divergence of nearby trajectories, indicating
    chaos, stability, or periodic behavior.
    
    Args:
        rnn: RNN model
        hidden_state: Initial hidden state (n_units,)
        n_steps: Number of timesteps for perturbation growth
    
    Returns:
        lyapunov_exponents: Estimated Lyapunov spectrum
    """
    hidden_state = torch.tensor(hidden_state, dtype=torch.float32)
    h = hidden_state.clone()
    
    # Small random perturbation
    epsilon = 1e-6
    delta_h = torch.randn_like(h) * epsilon
    delta_h = delta_h / torch.norm(delta_h) * epsilon  # Normalize
    
    h_perturbed = h + delta_h
    
    # Evolve both trajectories
    input_zero = torch.zeros(rnn.input_size)
    
    for _ in range(n_steps):
        h = rnn.step(input_zero, h)
        h_perturbed = rnn.step(input_zero, h_perturbed)
    
    # Measure final separation
    final_delta = h_perturbed - h
    growth_rate = torch.log(torch.norm(final_delta) / epsilon) / n_steps
    
    return growth_rate.item()


def detect_fixed_points(rnn, n_random_init=100, tolerance=1e-5, max_iter=1000):
    """
    Find fixed points of RNN dynamics via random initialization + relaxation.
    
    Fixed points satisfy: h* = f(h*, 0)
    
    Args:
        rnn: RNN model
        n_random_init: Number of random initializations
        tolerance: Convergence tolerance
        max_iter: Maximum relaxation iterations
    
    Returns:
        fixed_points: List of detected fixed points
        stability: Stability classification for each fixed point
    """
    fixed_points = []
    stability = []
    
    input_zero = torch.zeros(rnn.input_size)
    
    for _ in range(n_random_init):
        # Random initialization
        h = torch.randn(rnn.hidden_size) * 0.5
        
        # Relaxation
        for iteration in range(max_iter):
            h_next = rnn.step(input_zero, h)
            delta = torch.norm(h_next - h)
            
            if delta < tolerance:
                # Converged - check if unique
                is_unique = True
                for fp in fixed_points:
                    if torch.norm(h - fp) < tolerance * 10:
                        is_unique = False
                        break
                
                if is_unique:
                    fixed_points.append(h.clone())
                    
                    # Compute Jacobian at fixed point
                    from src.core.deformation_utils import compute_jacobian_analytical
                    try:
                        J = compute_jacobian_analytical(rnn, h.numpy(), input_zero.numpy())
                        eigs = np.linalg.eigvals(J)
                        max_eig = np.max(np.abs(eigs))
                        
                        if max_eig < 1.0:
                            stability.append('stable')
                        else:
                            stability.append('unstable')
                    except:
                        stability.append('unknown')
                
                break
            
            h = h_next
    
    return fixed_points, stability


def plot_spectral_summary(rnn, deformation_labels, interpretation, save_path=None):
    """
    Create comprehensive 4-panel spectral analysis figure.
    
    Panels:
    1. Eigenvalue spectrum (complex plane)
    2. Eigenvalue magnitude histogram by type
    3. Imaginary part histogram by type
    4. Contingency heatmap (deformation vs spectral type)
    
    Args:
        rnn: RNN model
        deformation_labels: Cluster labels
        interpretation: Dict from interpret_clusters()
        save_path: Optional save path
    """
    eigenvalues, _ = compute_recurrent_eigenvalues(rnn)
    spectral_types = classify_eigenvalues(eigenvalues)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
    
    # Panel 1: Complex plane
    ax1 = fig.add_subplot(gs[0, 0])
    theta = np.linspace(0, 2*np.pi, 100)
    ax1.plot(np.cos(theta), np.sin(theta), 'k--', alpha=0.3, linewidth=2)
    
    unique_labels = np.unique(deformation_labels)
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = deformation_labels == label
        eigs = eigenvalues[mask]
        name = interpretation[label]['name']
        ax1.scatter(eigs.real, eigs.imag, c=[colors[i]], s=80, alpha=0.6, label=name)
    
    ax1.axhline(0, color='k', linewidth=0.5, alpha=0.3)
    ax1.axvline(0, color='k', linewidth=0.5, alpha=0.3)
    ax1.set_xlabel('Real Part')
    ax1.set_ylabel('Imaginary Part')
    ax1.set_title('Eigenvalue Spectrum')
    ax1.legend()
    ax1.grid(True, alpha=0.2)
    ax1.set_aspect('equal')
    
    # Panel 2: Magnitude histogram
    ax2 = fig.add_subplot(gs[0, 1])
    for i, label in enumerate(unique_labels):
        mask = deformation_labels == label
        mags = np.abs(eigenvalues[mask])
        name = interpretation[label]['name']
        ax2.hist(mags, bins=20, alpha=0.5, label=name, color=colors[i])
    
    ax2.axvline(1.0, color='r', linestyle='--', linewidth=2, label='|λ|=1')
    ax2.set_xlabel('Eigenvalue Magnitude')
    ax2.set_ylabel('Count')
    ax2.set_title('Magnitude Distribution')
    ax2.legend()
    ax2.grid(True, alpha=0.2)
    
    # Panel 3: Imaginary part histogram
    ax3 = fig.add_subplot(gs[1, 0])
    for i, label in enumerate(unique_labels):
        mask = deformation_labels == label
        imags = np.abs(eigenvalues[mask].imag)
        name = interpretation[label]['name']
        ax3.hist(imags, bins=20, alpha=0.5, label=name, color=colors[i])
    
    ax3.set_xlabel('|Imaginary Part|')
    ax3.set_ylabel('Count')
    ax3.set_title('Oscillatory Component')
    ax3.legend()
    ax3.grid(True, alpha=0.2)
    
    # Panel 4: Contingency heatmap
    ax4 = fig.add_subplot(gs[1, 1])
    
    unique_spectral = ['rotator', 'integrator', 'explorer', 'contracting']
    contingency = np.zeros((len(unique_labels), len(unique_spectral)))
    
    for i, deform_label in enumerate(unique_labels):
        for j, spec_type in enumerate(unique_spectral):
            count = np.sum((deformation_labels == deform_label) & (spectral_types == spec_type))
            contingency[i, j] = count
    
    im = ax4.imshow(contingency, cmap='YlOrRd', aspect='auto')
    ax4.set_xticks(range(len(unique_spectral)))
    ax4.set_xticklabels([s.capitalize() for s in unique_spectral], rotation=45, ha='right')
    ax4.set_yticks(range(len(unique_labels)))
    ax4.set_yticklabels([interpretation[l]['name'] for l in unique_labels])
    ax4.set_xlabel('Spectral Type')
    ax4.set_ylabel('Deformation Type')
    ax4.set_title('Contingency Matrix')
    
    # Add count labels
    for i in range(len(unique_labels)):
        for j in range(len(unique_spectral)):
            text = ax4.text(j, i, int(contingency[i, j]),
                           ha="center", va="center", color="black", fontsize=10)
    
    plt.colorbar(im, ax=ax4, label='Count')
    
    plt.suptitle('Spectral Analysis Summary', fontsize=16, y=0.995)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"  Saved spectral summary: {save_path}")
    
    return fig
