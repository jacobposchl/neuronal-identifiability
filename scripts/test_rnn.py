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
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU
from src.tasks import FlipFlopTask, CyclingMemoryTask, ContextIntegrationTask, get_task
from src.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.rnn_features import (extract_rnn_unit_features, classify_units, 
                               interpret_clusters, print_cluster_summary,
                               compare_to_baseline)
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
    
    # Smooth deformation signals
    rotation_traj, contraction_traj, expansion_traj = smooth_deformation_signals(
        rotation_traj, contraction_traj, expansion_traj, sigma=5
    )
    
    if verbose:
        print(f"  Rotation range: [{np.min(rotation_traj):.3f}, {np.max(rotation_traj):.3f}]")
        print(f"  Contraction range: [{np.min(contraction_traj):.3f}, {np.max(contraction_traj):.3f}]")
        print(f"  Expansion range: [{np.min(expansion_traj):.3f}, {np.max(expansion_traj):.3f}]")
    
    # 6. Extract unit features
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
    if verbose:
        print(f"\nClustering units...")
    
    labels, details = classify_units(features, n_clusters=4, method='kmeans', 
                                     return_details=True)
    
    interpretation = interpret_clusters(features, labels)
    
    if verbose:
        print(f"  Silhouette score: {details['silhouette']:.3f}")
        print_cluster_summary(interpretation)
    
    # 8. Compare to baselines
    if verbose:
        print(f"\nComparing to baseline methods...")
    
    baseline_comparison = compare_to_baseline(features, labels, hidden_states)
    
    if verbose:
        print(f"  Silhouette scores:")
        print(f"    Deformation method: {baseline_comparison['deformation']:.3f}")
        print(f"    PCA baseline:       {baseline_comparison['pca']:.3f}")
        print(f"    Raw activations:    {baseline_comparison['raw']:.3f}")
        
        improvement_pca = (baseline_comparison['deformation'] - baseline_comparison['pca']) / \
                         (baseline_comparison['pca'] + 1e-10) * 100
        improvement_raw = (baseline_comparison['deformation'] - baseline_comparison['raw']) / \
                         (baseline_comparison['raw'] + 1e-10) * 100
        
        print(f"  Improvement over PCA: {improvement_pca:+.1f}%")
        print(f"  Improvement over raw: {improvement_raw:+.1f}%")
    
    # 9. Visualize results
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
            'expansion': expansion_traj
        },
        'latent_trajectory': latent_traj,
        'features': features,
        'labels': labels,
        'interpretation': interpretation,
        'silhouette': details['silhouette'],
        'baseline_comparison': baseline_comparison
    }
    
    if verbose:
        print("\n" + "="*70)
        print("EXPERIMENT COMPLETE")
        print("="*70 + "\n")
    
    return results


def run_multi_task_comparison(tasks=['flipflop', 'cycling', 'context', 'mnist'],
                               architecture='vanilla', hidden_size=128,
                               n_epochs=2000, verbose=True):
    """
    Compare unit type distributions across multiple tasks.
    
    Hypothesis: Task structure determines functional type distribution
    - FlipFlop → Integrators (memory maintenance)
    - Cycling → Rotators (oscillatory dynamics)
    - Context → Mixed (integration + switching)
    - MNIST → Integrators (visual evidence accumulation)
    
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
