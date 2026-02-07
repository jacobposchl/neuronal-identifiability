"""
Perturbation analysis test script.

Tests functional importance of deformation-classified unit types through
selective ablation and cross-task transfer experiments.
"""

import sys
from pathlib import Path
import argparse
import numpy as np
import torch
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.rnn_models import VanillaRNN
from src.tasks import get_task
from src.core.deformation_utils import estimate_deformation_from_rnn
from src.analysis.rnn_features import (extract_rnn_unit_features, classify_units, 
                               interpret_clusters, select_features_by_task_dynamics)
from src.analysis.perturbation import (test_unit_importance, cross_task_transfer,
                               progressive_ablation, confidence_guided_pruning,
                               task_specific_importance)
from src.visualization import ensure_dirs


def run_importance_analysis(task_name='context', hidden_size=128, 
                            n_epochs=2000, verbose=True):
    """
    Test functional importance of each unit type via ablation.
    
    Args:
        task_name: Task to analyze
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        verbose: Print output
    
    Returns:
        importance_results: Dict with ablation results
    """
    print("\n" + "="*70)
    print(f"UNIT TYPE IMPORTANCE ANALYSIS: {task_name.upper()}")
    print("="*70)
    
    # 1. Train RNN
    task = get_task(task_name)
    rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
    
    print("\nTraining RNN...")
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
    print(f"Final accuracy: {history['accuracy'][-1]:.2%}")
    
    # 2. Extract and classify units
    print("\nExtracting trajectories and classifying units...")
    hidden_states, inputs, outputs = task.extract_trajectories(rnn, n_trials=50)
    
    # Estimate deformation
    rotation, contraction, expansion = estimate_deformation_from_rnn(
        rnn, inputs, hidden_states, smoothing_window=5)
    
    # Select features by task dynamics
    features = select_features_by_task_dynamics(
        rotation, contraction, expansion, task_name, hidden_states, verbose=False)
    
    if features is None:
        print("\n❌ Task has discrete dynamics - deformation method failed")
        print("Cannot perform perturbation analysis without unit classifications")
        return None
    
    # Cluster units
    unit_labels, cluster_centers = classify_units(features, n_clusters=4)
    interpretation = interpret_clusters(cluster_centers, feature_type='deformation')
    
    # 3. Test importance via ablation
    importance_results = test_unit_importance(
        rnn, task, unit_labels, interpretation, 
        n_test_trials=50, verbose=verbose)
    
    return importance_results


def run_transfer_analysis(task_a='context', task_b='parametric',
                          hidden_size=128, n_epochs=2000, verbose=True):
    """
    Test cross-task transfer of unit classifications.
    
    Args:
        task_a: Training task (where units are classified)
        task_b: Test task (where ablation is tested)
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        verbose: Print output
    
    Returns:
        transfer_results: Dict with cross-task ablation results
    """
    print("\n" + "="*70)
    print(f"CROSS-TASK TRANSFER: {task_a.upper()} → {task_b.upper()}")
    print("="*70)
    
    # 1. Train RNN on Task A
    task_obj_a = get_task(task_a)
    rnn_a = VanillaRNN(task_obj_a.input_size, hidden_size, task_obj_a.output_size)
    
    print(f"\nTraining on {task_a}...")
    rnn_a, history_a = task_obj_a.train_rnn(rnn_a, n_epochs=n_epochs, verbose=False)
    print(f"Final accuracy: {history_a['accuracy'][-1]:.2%}")
    
    # 2. Classify units based on Task A
    print(f"\nClassifying units based on {task_a} dynamics...")
    hidden_states_a, inputs_a, _ = task_obj_a.extract_trajectories(rnn_a, n_trials=50)
    
    rotation_a, contraction_a, expansion_a = estimate_deformation_from_rnn(
        rnn_a, inputs_a, hidden_states_a, smoothing_window=5)
    
    features_a = select_features_by_task_dynamics(
        rotation_a, contraction_a, expansion_a, task_a, hidden_states_a, verbose=False)
    
    if features_a is None:
        print(f"\n❌ {task_a} has discrete dynamics - cannot classify units")
        return None
    
    unit_labels_a, cluster_centers_a = classify_units(features_a, n_clusters=4)
    interpretation_a = interpret_clusters(cluster_centers_a, feature_type='deformation')
    
    # 3. Test transfer to Task B
    task_obj_b = get_task(task_b)
    
    transfer_results = cross_task_transfer(
        rnn_a, task_obj_a, task_obj_b, unit_labels_a, interpretation_a,
        n_test_trials=50, verbose=verbose)
    
    return transfer_results


def run_progressive_ablation_experiment(task_name='context', hidden_size=128,
                                       n_epochs=2000, verbose=True):
    """
    Progressive unit ablation experiment.
    
    Tests prediction: Low-confidence units can be removed with minimal impact.
    
    Args:
        task_name: Task to analyze
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        verbose: Print output
    
    Returns:
        ablation_curves: Dict mapping strategy to ablation curve
    """
    print("\n" + "="*70)
    print(f"PROGRESSIVE ABLATION: {task_name.upper()}")
    print("="*70)
    
    # 1. Train and classify
    task = get_task(task_name)
    rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
    
    print("\nTraining RNN...")
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
    print(f"Final accuracy: {history['accuracy'][-1]:.2%}")
    
    print("\nClassifying units...")
    hidden_states, inputs, _ = task.extract_trajectories(rnn, n_trials=50)
    
    rotation, contraction, expansion = estimate_deformation_from_rnn(
        rnn, inputs, hidden_states, smoothing_window=5)
    
    features = select_features_by_task_dynamics(
        rotation, contraction, expansion, task_name, hidden_states, verbose=False)
    
    if features is None:
        print("\n❌ Discrete dynamics - cannot perform progressive ablation")
        return None
    
    unit_labels, cluster_centers = classify_units(features, n_clusters=4)
    interpretation = interpret_clusters(cluster_centers, feature_type='deformation')
    
    # 2. Progressive ablation - multiple strategies
    strategies = ['low_confidence_first', 'high_confidence_first', 'random']
    ablation_curves = {}
    
    for strategy in strategies:
        print(f"\n--- Strategy: {strategy} ---")
        curve = progressive_ablation(
            rnn, task, unit_labels, interpretation,
            strategy=strategy, n_test_trials=50, verbose=verbose)
        ablation_curves[strategy] = curve
    
    # 3. Plot comparison
    ensure_dirs('results/figures')
    
    plt.figure(figsize=(10, 6))
    for strategy, curve in ablation_curves.items():
        pcts = [c[1] for c in curve]
        accs = [c[2] for c in curve]
        plt.plot(pcts, accs, marker='o', label=strategy, linewidth=2)
    
    plt.xlabel('Percentage of Units Ablated', fontsize=12)
    plt.ylabel('Task Accuracy (%)', fontsize=12)
    plt.title(f'Progressive Ablation: {task_name}', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    save_path = f'results/figures/progressive_ablation_{task_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")
    plt.close()
    
    return ablation_curves


def run_compression_experiment(task_name='context', hidden_size=128,
                               n_epochs=2000, compression_ratios=[0.2, 0.4, 0.6],
                               verbose=True):
    """
    Network compression guided by deformation confidence.
    
    Args:
        task_name: Task to analyze
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        compression_ratios: List of compression ratios to test
        verbose: Print output
    
    Returns:
        compression_results: List of pruning results for each ratio
    """
    print("\n" + "="*70)
    print(f"CONFIDENCE-GUIDED COMPRESSION: {task_name.upper()}")
    print("="*70)
    
    # 1. Train and classify
    task = get_task(task_name)
    rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
    
    print("\nTraining RNN...")
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
    print(f"Final accuracy: {history['accuracy'][-1]:.2%}")
    
    print("\nClassifying units...")
    hidden_states, inputs, _ = task.extract_trajectories(rnn, n_trials=50)
    
    rotation, contraction, expansion = estimate_deformation_from_rnn(
        rnn, inputs, hidden_states, smoothing_window=5)
    
    features = select_features_by_task_dynamics(
        rotation, contraction, expansion, task_name, hidden_states, verbose=False)
    
    if features is None:
        print("\n❌ Discrete dynamics - cannot perform compression")
        return None
    
    unit_labels, cluster_centers = classify_units(features, n_clusters=4)
    interpretation = interpret_clusters(cluster_centers, feature_type='deformation')
    
    # 2. Test multiple compression ratios
    compression_results = []
    
    for ratio in compression_ratios:
        print(f"\n--- Compression ratio: {ratio*100:.0f}% ---")
        results = confidence_guided_pruning(
            rnn, task, unit_labels, interpretation,
            target_compression=ratio, n_test_trials=50, verbose=verbose)
        compression_results.append(results)
    
    # 3. Summary plot
    ensure_dirs('results/figures')
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Accuracy vs compression
    ratios_pct = [r*100 for r in compression_ratios]
    baseline_accs = [r['baseline_accuracy'] for r in compression_results]
    pruned_accs = [r['pruned_accuracy'] for r in compression_results]
    random_accs = [r['random_accuracy'] for r in compression_results]
    
    ax1.plot(ratios_pct, baseline_accs, 'k--', label='Baseline', linewidth=2)
    ax1.plot(ratios_pct, pruned_accs, 'b-o', label='Confidence-guided', linewidth=2)
    ax1.plot(ratios_pct, random_accs, 'r-s', label='Random', linewidth=2)
    ax1.set_xlabel('Compression Ratio (%)', fontsize=12)
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Network Compression Performance', fontsize=13, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Advantage over random
    advantages = [r['advantage_over_random'] for r in compression_results]
    colors = ['green' if a > 0 else 'red' for a in advantages]
    ax2.bar(ratios_pct, advantages, color=colors, alpha=0.7, edgecolor='black')
    ax2.axhline(0, color='black', linestyle='-', linewidth=1)
    ax2.set_xlabel('Compression Ratio (%)', fontsize=12)
    ax2.set_ylabel('Advantage over Random (%)', fontsize=12)
    ax2.set_title('Confidence-Guided vs Random Pruning', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    save_path = f'results/figures/compression_{task_name}.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\nSaved plot: {save_path}")
    plt.close()
    
    return compression_results


def run_multi_task_importance(task_names=['context', 'parametric', 'flipflop'],
                              hidden_size=128, n_epochs=2000, verbose=True):
    """
    Test task-specific importance of unit types.
    
    Trains one RNN, classifies units, then tests ablation across multiple tasks.
    
    Args:
        task_names: List of tasks to test
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        verbose: Print output
    
    Returns:
        importance_matrix: Dict mapping (unit_type, task) -> accuracy_drop
    """
    print("\n" + "="*70)
    print(f"MULTI-TASK UNIT IMPORTANCE MATRIX")
    print("="*70)
    
    # Use first task for training/classification
    primary_task_name = task_names[0]
    
    # 1. Train on primary task
    primary_task = get_task(primary_task_name)
    rnn = VanillaRNN(primary_task.input_size, hidden_size, primary_task.output_size)
    
    print(f"\nTraining on {primary_task_name}...")
    rnn, history = primary_task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
    print(f"Final accuracy: {history['accuracy'][-1]:.2%}")
    
    # 2. Classify units
    print(f"\nClassifying units based on {primary_task_name}...")
    hidden_states, inputs, _ = primary_task.extract_trajectories(rnn, n_trials=50)
    
    rotation, contraction, expansion = estimate_deformation_from_rnn(
        rnn, inputs, hidden_states, smoothing_window=5)
    
    features = select_features_by_task_dynamics(
        rotation, contraction, expansion, primary_task_name, hidden_states, verbose=False)
    
    if features is None:
        print(f"\n❌ {primary_task_name} has discrete dynamics")
        return None
    
    unit_labels, cluster_centers = classify_units(features, n_clusters=4)
    interpretation = interpret_clusters(cluster_centers, feature_type='deformation')
    
    # 3. Create task objects
    tasks = [get_task(name) for name in task_names]
    
    # 4. Test across tasks
    importance_matrix = task_specific_importance(
        rnn, unit_labels, interpretation, tasks,
        n_test_trials=50, verbose=verbose)
    
    return importance_matrix


def main():
    parser = argparse.ArgumentParser(
        description='Perturbation analysis for RNN unit types')
    
    parser.add_argument('--experiment', type=str, default='importance',
                       choices=['importance', 'transfer', 'progressive', 
                               'compression', 'multi-task'],
                       help='Type of perturbation experiment')
    parser.add_argument('--task', type=str, default='context',
                       help='Primary task name')
    parser.add_argument('--task-b', type=str, default='parametric',
                       help='Secondary task for transfer experiments')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='RNN hidden size')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Training epochs')
    parser.add_argument('--compression-ratios', type=float, nargs='+',
                       default=[0.2, 0.4, 0.6],
                       help='Compression ratios to test')
    parser.add_argument('--multi-tasks', type=str, nargs='+',
                       default=['context', 'parametric', 'flipflop'],
                       help='Tasks for multi-task importance')
    
    args = parser.parse_args()
    
    # Run requested experiment
    if args.experiment == 'importance':
        results = run_importance_analysis(
            task_name=args.task,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            verbose=True
        )
    
    elif args.experiment == 'transfer':
        results = run_transfer_analysis(
            task_a=args.task,
            task_b=args.task_b,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            verbose=True
        )
    
    elif args.experiment == 'progressive':
        results = run_progressive_ablation_experiment(
            task_name=args.task,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            verbose=True
        )
    
    elif args.experiment == 'compression':
        results = run_compression_experiment(
            task_name=args.task,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            compression_ratios=args.compression_ratios,
            verbose=True
        )
    
    elif args.experiment == 'multi-task':
        results = run_multi_task_importance(
            task_names=args.multi_tasks,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs,
            verbose=True
        )
    
    print("\n" + "="*70)
    print("EXPERIMENT COMPLETE")
    print("="*70)


if __name__ == '__main__':
    main()
