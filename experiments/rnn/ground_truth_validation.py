"""
Ground truth validation tests for deformation-based classification.

Tests the deformation method on synthetic RNNs with known unit type distributions.
This provides direct validation that the method can recover ground truth functional
roles when dynamics match theoretical assumptions.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import argparse
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, confusion_matrix

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.synthetic_rnn import build_synthetic_rnn, verify_spectral_properties
from src.tasks import get_task
from src.core.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.analysis.rnn_features import (extract_rnn_unit_features, classify_units, 
                               interpret_clusters, print_cluster_summary)
from src.analysis.spectral_analysis import (compare_eigenvalues_to_deformation, 
                                    print_spectral_comparison,
                                    plot_eigenvalue_spectrum, plot_spectral_summary)
from src.visualization import ensure_dirs
import matplotlib.pyplot as plt


def test_synthetic_recovery(task_name='context', n_epochs=1000, hidden_size=100, 
                             n_integrators=30, n_rotators=20, n_explorers=15,
                             verbose=True):
    """
    Test: Can deformation method recover known unit types from synthetic RNN?
    
    Protocol:
    1. Generate synthetic RNN with ground truth labels
    2. Verify spectral properties match expected types
    3. Train on continuous-dynamics task
    4. Apply deformation classification
    5. Compare to ground truth using ARI, NMI, confusion matrix
    
    Args:
        task_name: Task to train on ('context', 'flipflop', 'cycling')
        n_epochs: Training epochs
        hidden_size: Total hidden units
        n_integrators: Number of integrator units in ground truth
        n_rotators: Number of rotator units in ground truth
        n_explorers: Number of explorer units in ground truth
        verbose: Print detailed output
    
    Returns:
        results: Dict with ground truth comparison metrics
    """
    if verbose:
        print("\n" + "="*70)
        print(f"GROUND TRUTH VALIDATION TEST: {task_name.upper()}")
        print("="*70)
    
    # 1. Create task
    task = get_task(task_name)
    
    # 2. Build synthetic RNN with known unit distribution
    n_mixed = hidden_size - n_integrators - n_rotators - n_explorers
    rnn, ground_truth_labels = build_synthetic_rnn(
        input_dim=task.input_size,
        hidden_size=hidden_size,
        output_dim=task.output_size,
        n_integrators=n_integrators,
        n_rotators=n_rotators,
        n_explorers=n_explorers,
        n_mixed=n_mixed
    )
    
    # 3. Verify spectral properties
    spectral_report = verify_spectral_properties(rnn, ground_truth_labels)
    
    # 4. Train RNN on task
    if verbose:
        print(f"\nTraining synthetic RNN on {task_name}...")
        print(f"  Epochs: {n_epochs}")
    
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001, 
                                   batch_size=32, verbose=verbose)
    
    final_accuracy = history['accuracy'][-1]
    if verbose:
        print(f"  Final accuracy: {final_accuracy:.2f}%")
    
    # 5. Extract deformation features
    if verbose:
        print(f"\nExtracting trajectories...")
    
    hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=50, trial_length=200)
    
    if verbose:
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"\nEstimating deformation...")
    
    rotation_traj, contraction_traj, expansion_traj, latent_traj = estimate_deformation_from_rnn(
        hidden_states, rnn=None, dt=0.01, latent_dim=3, method='pca_then_local'
    )
    
    # Check if deformation succeeded
    if rotation_traj is None:
        if verbose:
            print(f"  WARNING: Deformation estimation failed on {task_name}")
            print(f"  This may indicate task has discrete dynamics")
        return {
            'task': task_name,
            'deformation_valid': False,
            'ground_truth_labels': ground_truth_labels,
            'predicted_labels': None,
            'ari': 0.0,
            'nmi': 0.0
        }
    
    # Smooth signals
    rotation_traj, contraction_traj, expansion_traj = smooth_deformation_signals(
        rotation_traj, contraction_traj, expansion_traj, sigma=5
    )
    
    if verbose:
        print(f"  Deformation ranges:")
        print(f"    Rotation:    [{np.min(rotation_traj):.3f}, {np.max(rotation_traj):.3f}]")
        print(f"    Contraction: [{np.min(contraction_traj):.3f}, {np.max(contraction_traj):.3f}]")
        print(f"    Expansion:   [{np.min(expansion_traj):.3f}, {np.max(expansion_traj):.3f}]")
    
    # 6. Extract features and classify
    if verbose:
        print(f"\nExtracting unit features...")
    
    features = extract_rnn_unit_features(
        hidden_states, rotation_traj, contraction_traj, expansion_traj,
        smooth_sigma=5
    )
    
    if verbose:
        print(f"  Features shape: {features.shape}")
        print(f"  Mean |rotation corr|: {np.mean(np.abs(features[:, 0])):.3f}")
        print(f"  Mean |contraction corr|: {np.mean(np.abs(features[:, 1])):.3f}")
        print(f"  Mean |expansion corr|: {np.mean(np.abs(features[:, 2])):.3f}")
    
    # 7. Cluster units
    n_clusters = 4  # Integrator, Rotator, Explorer, Mixed
    predicted_labels, details = classify_units(features, n_clusters=n_clusters, 
                                               method='kmeans', return_details=True)
    
    interpretation = interpret_clusters(features, predicted_labels)
    
    if verbose:
        print(f"\nClustering results:")
        print(f"  Silhouette score: {details['silhouette']:.3f}")
        print_cluster_summary(interpretation)
    
    # 8. Compare to ground truth
    ari = adjusted_rand_score(ground_truth_labels, predicted_labels)
    nmi = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    
    # Confusion matrix
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
    
    if verbose:
        print(f"\n" + "="*70)
        print(f"GROUND TRUTH COMPARISON")
        print(f"="*70)
        print(f"Adjusted Rand Index: {ari:.3f}")
        print(f"Normalized Mutual Information: {nmi:.3f}")
        print(f"\nConfusion Matrix (rows=ground truth, cols=predicted):")
        print(f"{'':15} {'Cluster 0':>10} {'Cluster 1':>10} {'Cluster 2':>10} {'Cluster 3':>10}")
        
        type_names = ['Integrator', 'Rotator', 'Explorer', 'Mixed']
        for i, name in enumerate(type_names):
            if i < conf_matrix.shape[0]:
                row = ' '.join(f'{conf_matrix[i,j]:10d}' for j in range(conf_matrix.shape[1]))
                print(f"{name:15} {row}")
        print("="*70)
    
    # 9. Spectral validation
    if verbose:
        print(f"\nSpectral validation...")
    
    # Map ground truth labels to names for comparison
    gt_label_names = {0: 'Integrator', 1: 'Rotator', 2: 'Explorer', 3: 'Mixed'}
    spectral_comparison = compare_eigenvalues_to_deformation(
        rnn, ground_truth_labels, label_names=gt_label_names
    )
    
    if verbose:
        print_spectral_comparison(spectral_comparison, gt_label_names)
    
    # 10. Generate visualizations
    ensure_dirs('results/ground_truth_figures')
    
    # Eigenvalue spectrum plot
    fig1, ax1 = plot_eigenvalue_spectrum(
        spectral_comparison['eigenvalues'],
        ground_truth_labels,
        label_names=gt_label_names,
        save_path=f'results/ground_truth_figures/{task_name}_eigenvalues_groundtruth.png'
    )
    plt.close(fig1)
    
    # Spectral summary with deformation predictions
    fig2 = plot_spectral_summary(
        rnn, predicted_labels, interpretation,
        save_path=f'results/ground_truth_figures/{task_name}_spectral_summary.png'
    )
    plt.close(fig2)
    
    # Confusion matrix heatmap
    fig3, ax3 = plt.subplots(figsize=(10, 8))
    im = ax3.imshow(conf_matrix, cmap='Blues', aspect='auto')
    
    ax3.set_xticks(range(n_clusters))
    ax3.set_xticklabels([interp['name'] for interp in interpretation.values()])
    ax3.set_yticks(range(len(type_names)))
    ax3.set_yticklabels(type_names)
    ax3.set_xlabel('Predicted Cluster', fontsize=12)
    ax3.set_ylabel('Ground Truth Type', fontsize=12)
    ax3.set_title(f'Ground Truth Recovery: {task_name.capitalize()} Task\n' + 
                  f'ARI = {ari:.3f}, NMI = {nmi:.3f}', fontsize=14)
    
    # Add count labels
    for i in range(len(type_names)):
        for j in range(n_clusters):
            if i < conf_matrix.shape[0] and j < conf_matrix.shape[1]:
                text = ax3.text(j, i, int(conf_matrix[i, j]),
                               ha="center", va="center", 
                               color="white" if conf_matrix[i,j] > conf_matrix.max()/2 else "black",
                               fontsize=12)
    
    plt.colorbar(im, ax=ax3, label='Count')
    plt.tight_layout()
    plt.savefig(f'results/ground_truth_figures/{task_name}_confusion_matrix.png', 
                dpi=150, bbox_inches='tight')
    plt.close(fig3)
    
    if verbose:
        print(f"\nSaved visualizations to results/ground_truth_figures/")
    
    # Return results
    results = {
        'task': task_name,
        'deformation_valid': True,
        'final_accuracy': final_accuracy,
        'ground_truth_labels': ground_truth_labels,
        'predicted_labels': predicted_labels,
        'ari': ari,
        'nmi': nmi,
        'confusion_matrix': conf_matrix,
        'silhouette': details['silhouette'],
        'spectral_comparison': spectral_comparison,
        'spectral_report': spectral_report,
        'interpretation': interpretation
    }
    
    return results


def run_multi_task_validation(tasks=['context', 'flipflop', 'cycling'], 
                               n_epochs=1000, hidden_size=100, verbose=True):
    """
    Run ground truth validation across multiple tasks.
    
    Tests hypothesis: Deformation method works on continuous tasks,
    struggles on discrete tasks.
    
    Args:
        tasks: List of task names
        n_epochs: Training epochs
        hidden_size: RNN hidden size
        verbose: Print output
    
    Returns:
        all_results: Dict mapping task names to results
    """
    all_results = {}
    
    for task_name in tasks:
        results = test_synthetic_recovery(
            task_name=task_name,
            n_epochs=n_epochs,
            hidden_size=hidden_size,
            verbose=verbose
        )
        all_results[task_name] = results
        
        if verbose:
            print(f"\n" + "-"*70 + "\n")
    
    # Summary comparison
    if verbose:
        print(f"\n" + "="*70)
        print(f"MULTI-TASK VALIDATION SUMMARY")
        print(f"="*70)
        print(f"{'Task':15} {'Deform Valid':15} {'ARI':>8} {'NMI':>8} {'Silhouette':>12}")
        print(f"-"*70)
        
        for task_name, results in all_results.items():
            valid = "✓" if results['deformation_valid'] else "✗"
            ari = results['ari'] if results['deformation_valid'] else 0.0
            nmi = results['nmi'] if results['deformation_valid'] else 0.0
            sil = results.get('silhouette', 0.0)
            
            print(f"{task_name:15} {valid:^15} {ari:8.3f} {nmi:8.3f} {sil:12.3f}")
        
        print(f"="*70)
        
        # Interpretation
        print(f"\nInterpretation:")
        continuous_tasks = [t for t in tasks if all_results[t]['deformation_valid'] 
                           and all_results[t]['ari'] > 0.5]
        if continuous_tasks:
            print(f"  ✓ Method succeeds on continuous tasks: {', '.join(continuous_tasks)}")
        
        discrete_tasks = [t for t in tasks if not all_results[t]['deformation_valid'] 
                         or all_results[t]['ari'] < 0.3]
        if discrete_tasks:
            print(f"  ✗ Method struggles on discrete tasks: {', '.join(discrete_tasks)}")
    
    return all_results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ground truth validation for deformation method')
    parser.add_argument('--task', type=str, default='context',
                       choices=['context', 'flipflop', 'cycling', 'all'],
                       help='Task to test (default: context)')
    parser.add_argument('--multi-task', action='store_true',
                       help='Run multi-task validation (same as --task all)')
    parser.add_argument('--epochs', type=int, default=1000,
                       help='Training epochs (default: 1000)')
    parser.add_argument('--hidden-size', type=int, default=100,
                       help='Hidden units (default: 100)')
    parser.add_argument('--n-integrators', type=int, default=30,
                       help='Number of integrator units (default: 30)')
    parser.add_argument('--n-rotators', type=int, default=20,
                       help='Number of rotator units (default: 20)')
    parser.add_argument('--n-explorers', type=int, default=15,
                       help='Number of explorer units (default: 15)')
    parser.add_argument('--seed', type=int, default=42,
                       help='Random seed (default: 42)')
    
    args = parser.parse_args()
    
    # Set seeds
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Handle multi-task flag
    if args.multi_task or args.task == 'all':
        # Multi-task validation
        results = run_multi_task_validation(
            tasks=['context', 'flipflop', 'cycling'],
            n_epochs=args.epochs,
            hidden_size=args.hidden_size,
            verbose=True
        )
    else:
        # Single task validation
        results = test_synthetic_recovery(
            task_name=args.task,
            n_epochs=args.epochs,
            hidden_size=args.hidden_size,
            n_integrators=args.n_integrators,
            n_rotators=args.n_rotators,
            n_explorers=args.n_explorers,
            verbose=True
        )
        
        print(f"\n" + "="*70)
        print(f"VALIDATION TEST COMPLETE")
        print(f"="*70)
        if results['deformation_valid']:
            print(f"✓ Deformation method recovered ground truth")
            print(f"  ARI: {results['ari']:.3f} (>0.75 = excellent)")
            print(f"  NMI: {results['nmi']:.3f}")
            
            if results['ari'] > 0.75:
                print(f"\n✓✓✓ VALIDATION SUCCESSFUL - Method works as theoretically predicted!")
            elif results['ari'] > 0.50:
                print(f"\n✓ VALIDATION PARTIAL - Method shows promise but imperfect recovery")
            else:
                print(f"\n✗ VALIDATION FAILED - Method does not recover ground truth")
        else:
            print(f"✗ Deformation estimation failed - expected for discrete dynamics tasks")
        print(f"="*70)
