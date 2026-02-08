"""
RNN deformation-based unit classification experiment runner.

Tests the deformation-based neuron identification method on trained RNNs
across multiple cognitive tasks with different dynamical signatures.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU
from src.tasks import FlipFlopTask, CyclingMemoryTask, ContextIntegrationTask, get_task
from src.core.deformation_utils import (estimate_deformation_from_rnn, smooth_deformation_signals,
                                    detect_discrete_dynamics, validate_task_dynamics)
from src.analysis.rnn_features import (extract_rnn_unit_features, classify_units, 
                               interpret_clusters, print_cluster_summary,
                               compare_to_baseline, select_features_by_task_dynamics,
                               select_optimal_clusters)
from src.visualization import ensure_dirs, plot_bar


def run_single_task_experiment(task_name='flipflop', architecture='vanilla',
                                hidden_size=128, n_epochs=2000, lr=0.001,
                                n_trials=50, trial_length=200,
                                save_checkpoint=True, verbose=True):
    """
    Run complete experiment on a single task.
    
    Steps:
    1. Create task and RNN
    2. Train RNN on task
    3. Extract hidden state trajectories
    4. Estimate deformation signals
    5. Extract unit features
    6. Cluster units
    7. Interpret and visualize results
    
    Args:
        task_name: 'flipflop', 'cycling', or 'context'
        architecture: 'vanilla', 'lstm', or 'gru'
        hidden_size: Number of hidden units
        n_epochs: Training epochs
        lr: Learning rate
        n_trials: Number of test trials for trajectory extraction
        trial_length: Length of each trial
        save_checkpoint: Whether to save trained model
        verbose: Print progress
    
    Returns:
        results: Dict with all experimental results
    """
    ensure_dirs('results/rnn_figures')
    
    if verbose:
        print("\n" + "="*70)
        print(f"RNN DEFORMATION EXPERIMENT: {task_name.upper()} / {architecture.upper()}")
        print("="*70)
    
    # 1. Create task
    task = get_task(task_name)
    
    if verbose:
        print(f"\nTask: {task.name}")
        print(f"  Input size: {task.input_size}")
        print(f"  Output size: {task.output_size}")
        print(f"\nExpected dynamics:")
        print(f"  {task.get_expected_dynamics()}")
    
    # 2. Create RNN
    arch_map = {
        'vanilla': VanillaRNN,
        'lstm': SimpleLSTM,
        'gru': SimpleGRU
    }
    
    if architecture not in arch_map:
        raise ValueError(f"Unknown architecture: {architecture}")
    
    rnn = arch_map[architecture](task.input_size, hidden_size, task.output_size)
    
    if verbose:
        n_params = sum(p.numel() for p in rnn.parameters())
        print(f"\nRNN Architecture: {architecture}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Parameters: {n_params:,}")
    
    # 3. Train RNN
    checkpoint_path = f"results/checkpoints/{task_name}_{architecture}_h{hidden_size}.pt" if save_checkpoint else None
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=lr, 
                                   trial_length=100, verbose=verbose,
                                   save_path=checkpoint_path)
    
    final_accuracy = history['accuracy'][-1]
    if verbose:
        print(f"\nTraining Results:")
        print(f"  Final accuracy: {final_accuracy:.2%}")
        print(f"  Final loss: {history['loss'][-1]:.4f}")
    
    # 4. Extract hidden state trajectories
    if verbose:
        print(f"\nExtracting trajectories ({n_trials} trials)...")
    
    hidden_states, inputs, outputs = task.extract_trajectories(
        rnn, n_trials=n_trials, trial_length=trial_length
    )
    
    if verbose:
        print(f"  Hidden states shape: {hidden_states.shape}")
        print(f"  Total timesteps: {hidden_states.shape[1]}")
    
    # 5. Estimate deformation signals
    if verbose:
        print(f"\nEstimating deformation from hidden states...")
    
    rotation_traj, contraction_traj, expansion_traj, latent_traj = estimate_deformation_from_rnn(
        hidden_states, rnn=None, dt=0.01, latent_dim=3, method='pca_then_local'
    )
    
    # Smooth deformation signals (handles None returns)
    rotation_traj, contraction_traj, expansion_traj = smooth_deformation_signals(
        rotation_traj, contraction_traj, expansion_traj, sigma=5
    )
    
    # Check if deformation estimation succeeded
    deformation_valid = (rotation_traj is not None)
    
    if deformation_valid and verbose:
        print(f"  Rotation range: [{np.min(rotation_traj):.3f}, {np.max(rotation_traj):.3f}]")
        print(f"  Contraction range: [{np.min(contraction_traj):.3f}, {np.max(contraction_traj):.3f}]")
        print(f"  Expansion range: [{np.min(expansion_traj):.3f}, {np.max(expansion_traj):.3f}]")
    
    # Detect discrete dynamics
    is_discrete = False
    if latent_traj is not None:
        is_discrete, _, dynamics_info = detect_discrete_dynamics(latent_traj)
        if verbose and is_discrete:
            print(f"  Discrete dynamics detected: {dynamics_info['dynamics_type']}")
    
    # Validate dynamics match task expectations
    if verbose:
        print(f"\nValidating task dynamics...")
    
    valid, issues, suggestions = validate_task_dynamics(
        task_name, 
        (rotation_traj, contraction_traj, expansion_traj), 
        hidden_states,
        latent_traj
    )
    
    if not valid and verbose:
        print(f"  ⚠️  VALIDATION WARNINGS:")
        for issue in issues:
            print(f"    - {issue}")
        if suggestions:
            print(f"  Suggestions:")
            for suggestion in suggestions:
                print(f"    • {suggestion}")
    
    # 6. Extract unit features (task-appropriate)
    if verbose:
        print(f"\nExtracting unit features...")
    
    # First try deformation-based features if valid
    if deformation_valid:
        deformation_features = extract_rnn_unit_features(
            hidden_states, rotation_traj, contraction_traj, expansion_traj,
            smooth_sigma=5
        )
        
        if verbose:
            print(f"  Deformation features shape: {deformation_features.shape}")
            print(f"  Mean |rotation corr|: {np.mean(np.abs(deformation_features[:, 0])):.3f}")
            print(f"  Mean |contraction corr|: {np.mean(np.abs(deformation_features[:, 1])):.3f}")
            print(f"  Mean |expansion corr|: {np.mean(np.abs(deformation_features[:, 2])):.3f}")
    else:
        deformation_features = None
        if verbose:
            print(f"  Deformation features unavailable (estimation failed)")
    
    # Select features based on task dynamics
    task_dynamics = 'discrete' if is_discrete else 'unknown'
    features, method_used = select_features_by_task_dynamics(
        hidden_states, deformation_features, task_dynamics, deformation_valid
    )
    
    # 7. Select optimal number of clusters
    if verbose:
        print(f"\nSelecting optimal number of clusters...")
    
    optimal_k, cluster_scores = select_optimal_clusters(
        features, min_clusters=2, max_clusters=8, 
        method='combined', verbose=verbose
    )
    
    if verbose:
        print(f"  Optimal k: {optimal_k}")
        print(f"  Silhouette scores by k: {dict(zip(cluster_scores['k_values'], [f'{s:.3f}' for s in cluster_scores['silhouette']]))}")
    
    # 8. Cluster units
    if verbose:
        print(f"\nClustering units using {method_used} features (k={optimal_k})...")
    
    labels, details = classify_units(features, n_clusters=optimal_k, method='kmeans', 
                                     return_details=True)
    
    # Interpret clusters with appropriate feature type
    feature_type = 'deformation' if method_used == 'deformation' else 'pca'
    interpretation = interpret_clusters(features, labels, feature_type=feature_type)
    
    if verbose:
        print(f"  Silhouette score: {details['silhouette']:.3f}")
        
        # Check confidence levels
        low_conf_count = sum(1 for interp in interpretation.values() 
                           if interp.get('confidence', 'high') in ['very_low', 'low'])
        if low_conf_count > 0:
            print(f"  ⚠️  Warning: {low_conf_count} clusters have low confidence")
        
        print_cluster_summary(interpretation, feature_type=feature_type)
    
    # 8. Compare to baselines (only if using deformation features)
    if verbose:
        print(f"\nComparing to baseline methods...")
    
    # Use appropriate comparison based on method
    if method_used == 'deformation':
        baseline_comparison = compare_to_baseline(features, labels, hidden_states)
        
        if verbose:
            print(f"  Silhouette scores:")
            print(f"    Deformation method: {baseline_comparison['deformation']['silhouette']:.3f}")
            print(f"    PCA baseline:       {baseline_comparison['pca']['silhouette']:.3f}")
            print(f"    Raw activations:    {baseline_comparison['raw']['silhouette']:.3f}")
            
            deform_score = baseline_comparison['deformation']['silhouette']
            pca_score = baseline_comparison['pca']['silhouette']
            raw_score = baseline_comparison['raw']['silhouette']
            
            improvement_pca = (deform_score - pca_score) / (pca_score + 1e-10) * 100
            improvement_raw = (deform_score - raw_score) / (raw_score + 1e-10) * 100
            
            print(f"  Improvement over PCA: {improvement_pca:+.1f}%")
            print(f"  Improvement over raw: {improvement_raw:+.1f}%")
    else:
        baseline_comparison = None
        if verbose:
            print(f"  Using {method_used} features (baseline comparison not applicable)")
    
    # 10. Visualize results
    if verbose:
        print(f"\nGenerating visualizations...")
    
    # Plot unit type distribution
    plot_path = f"results/rnn_figures/{task_name}_{architecture}_distribution.png"
    
    dist_stats = {
        interp['name']: {
            'mean': interp['percentage'],
            'ci': 0  # No error bars for single experiment
        }
        for cid, interp in interpretation.items()
    }
    
    plot_bar(list(dist_stats.keys()), dist_stats, plot_path, 'Percentage of Units')
    
    if verbose:
        print(f"  Saved: {plot_path}")
    
    # Diagnostic plots
    if deformation_valid and latent_traj is not None:
        import matplotlib.pyplot as plt
        
        # Plot 1: Latent trajectory variance
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Latent PCA trajectories
        ax = axes[0, 0]
        for dim in range(min(3, latent_traj.shape[1])):
            ax.plot(latent_traj[:, dim], alpha=0.7, label=f'PC{dim+1}')
        ax.set_xlabel('Time')
        ax.set_ylabel('Latent Coordinate')
        ax.set_title('Latent Trajectory (PCA)')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Latent variance over time
        ax = axes[0, 1]
        latent_std = np.std(latent_traj, axis=1)
        ax.plot(latent_std)
        ax.axhline(np.mean(latent_std), color='r', linestyle='--', label='Mean')
        ax.set_xlabel('Time')
        ax.set_ylabel('Std Dev across PCs')
        ax.set_title('Latent Variance Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Deformation signal magnitudes
        ax = axes[1, 0]
        ax.plot(np.abs(rotation_traj), label='|Rotation|', alpha=0.7)
        ax.plot(np.abs(contraction_traj), label='|Contraction|', alpha=0.7)
        ax.plot(np.abs(expansion_traj), label='|Expansion|', alpha=0.7)
        ax.set_xlabel('Time')
        ax.set_ylabel('Signal Magnitude')
        ax.set_title('Deformation Signal Magnitudes')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_yscale('log')
        
        # Velocity distribution (if discrete dynamics)
        ax = axes[1, 1]
        if is_discrete and 'velocities' in dynamics_info:
            velocities = dynamics_info['velocities']
            ax.hist(velocities, bins=50, alpha=0.7, edgecolor='black')
            ax.axvline(dynamics_info.get('threshold', 0), color='r', linestyle='--', 
                      label=f"Threshold: {dynamics_info.get('threshold', 0):.3f}")
            ax.set_xlabel('Step Velocity')
            ax.set_ylabel('Frequency')
            ax.set_title(f"Velocity Distribution ({dynamics_info.get('dynamics_type', 'unknown')})")
            ax.legend()
        else:
            velocities = np.sqrt(np.sum(np.diff(latent_traj, axis=0)**2, axis=1))
            ax.hist(velocities, bins=50, alpha=0.7, edgecolor='black')
            ax.set_xlabel('Step Velocity')
            ax.set_ylabel('Frequency')
            ax.set_title('Velocity Distribution (continuous)')
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        diagnostic_path = f"results/rnn_figures/{task_name}_{architecture}_diagnostics.png"
        plt.savefig(diagnostic_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        if verbose:
            print(f"  Saved diagnostics: {diagnostic_path}")
    
    # Compile results
    results = {
        'task_name': task_name,
        'architecture': architecture,
        'hidden_size': hidden_size,
        'n_epochs': n_epochs,
        'final_accuracy': final_accuracy,
        'final_loss': history['loss'][-1],
        'history': history,
        'hidden_states': hidden_states,
        'deformation': {
            'rotation': rotation_traj,
            'contraction': contraction_traj,
            'expansion': expansion_traj,
            'valid': deformation_valid
        },
        'latent_trajectory': latent_traj,
        'dynamics': {
            'is_discrete': is_discrete,
            'info': dynamics_info if is_discrete else None
        },
        'validation': {
            'valid': valid,
            'issues': issues,
            'suggestions': suggestions
        },
        'features': features,
        'feature_method': method_used,
        'labels': labels,
        'optimal_k': optimal_k,
        'cluster_scores': cluster_scores,
        'interpretation': interpretation,
        'silhouette': details['silhouette'],
        'baseline_comparison': baseline_comparison
    }
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70 + "\n")
    
    return results


def run_multi_task_comparison(tasks=['flipflop', 'cycling', 'context', 'mnist',
                                     'parametric', 'matchsample', 'gonogo', 'fsm'],
                               architecture='vanilla', hidden_size=128,
                               n_epochs=2000, verbose=True):
    """
    Compare unit type distributions across multiple tasks.
    
    Hypothesis: Task structure determines functional type distribution
    Tier A (Continuous - deformation should work):
    - Context → Mixed (integration + switching)
    - Parametric → Integrators (line attractor maintenance)
    - MatchSample → Integrators (sample storage + comparison)
    - MNIST → Integrators (visual evidence accumulation)
    
    Tier B (Mixed - borderline):
    - FlipFlop → Integrators (memory maintenance, but discrete)
    - GoNoGo → Integrators + Explorers (integration + discrete decision)
    
    Tier C (Discrete - deformation should fail):
    - Cycling → Rotators (discrete periodic cycling)
    - FSM → Fail (pure discrete state transitions)
    
    Args:
        tasks: List of task names
        architecture: RNN architecture
        hidden_size: Number of hidden units
        n_epochs: Training epochs
        verbose: Print progress
    
    Returns:
        all_results: List of result dicts
    """
    if verbose:
        print("\n" + "="*70)
        print("MULTI-TASK COMPARISON")
        print("="*70)
        print(f"Tasks: {tasks}")
        print(f"Architecture: {architecture}")
        print(f"Hidden size: {hidden_size}")
        print("="*70)
    
    all_results = []
    
    for task_name in tasks:
        results = run_single_task_experiment(
            task_name=task_name,
            architecture=architecture,
            hidden_size=hidden_size,
            n_epochs=n_epochs,
            verbose=verbose
        )
        all_results.append(results)
    
    # Compare distributions
    if verbose:
        print("\n" + "="*70)
        print("CROSS-TASK COMPARISON")
        print("="*70)
        
        for results in all_results:
            print(f"\n{results['task_name'].upper()}:")
            for cid, interp in results['interpretation'].items():
                print(f"  {interp['name']:15s}: {interp['percentage']:5.1f}%")
    
    return all_results


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(description='RNN deformation-based unit classification')
    
    parser.add_argument('--task', type=str, default='flipflop',
                       choices=['flipflop', 'cycling', 'context', 'all'],
                       help='Task to run (or "all" for multi-task comparison)')
    parser.add_argument('--architecture', type=str, default='vanilla',
                       choices=['vanilla', 'lstm', 'gru'],
                       help='RNN architecture')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Training epochs')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--trials', type=int, default=50,
                       help='Number of test trials')
    parser.add_argument('--trial-length', type=int, default=200,
                       help='Length of each trial')
    parser.add_argument('--no-save', action='store_true',
                       help='Do not save checkpoint')
    parser.add_argument('--quiet', action='store_true',
                       help='Minimal output')
    
    args = parser.parse_args()
    
    if args.task == 'all':
        run_multi_task_comparison(
            tasks=['flipflop', 'cycling', 'context'],
            architecture=args.architecture,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            verbose=not args.quiet
        )
    else:
        run_single_task_experiment(
            task_name=args.task,
            architecture=args.architecture,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            lr=args.lr,
            n_trials=args.trials,
            trial_length=args.trial_length,
            save_checkpoint=not args.no_save,
            verbose=not args.quiet
        )


if __name__ == "__main__":
    main()
