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
from src.tasks.tasks import (ContextIntegrationTask, ParametricWorkingMemoryTask, 
                               FlipFlopTask, GoNoGoTask)
from src.core.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
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
    rotation, contraction, expansion, _ = estimate_deformation_from_rnn(hidden_states)
    rotation, contraction, expansion = smooth_deformation_signals(rotation, contraction, expansion, sigma=5)
    
    # Check deformation signal quality
    if verbose:
        print(f"  Deformation signal diagnostics:")
        print(f"    Rotation:    range=[{np.min(rotation):.3f}, {np.max(rotation):.3f}], std={np.std(rotation):.3f}")
        print(f"    Contraction: range=[{np.min(contraction):.3f}, {np.max(contraction):.3f}], std={np.std(contraction):.3f}")
        print(f"    Expansion:   range=[{np.min(expansion):.3f}, {np.max(expansion):.3f}], std={np.std(expansion):.3f}")
        
        if np.std(rotation) < 0.1 and np.std(contraction) < 0.1 and np.std(expansion) < 0.1:
            print(f"  ⚠️  WARNING: Deformation signals have very low variance!")
            print(f"     Features will likely be uninformative.")
    
    # Extract deformation features
    deformation_features = extract_rnn_unit_features(hidden_states, rotation, contraction, expansion)
    
    # CRITICAL: Check if features have variance
    feature_stds = np.std(deformation_features, axis=0)
    max_std = np.max(feature_stds)
    
    if max_std < 0.01:
        print(f"\n❌ CRITICAL ERROR: Deformation features have no variance!")
        print(f"   All units have identical features: {np.mean(deformation_features, axis=0)}")
        print(f"   Feature std: {feature_stds}")
        print(f"\n   This means:")
        print(f"   - Deformation signals (R, C, E) are not distinguishing units")
        print(f"   - Clustering will produce arbitrary/meaningless results")
        print(f"   - The method is NOT working for this task")
        print(f"\n   Possible causes:")
        print(f"   1. Task dynamics are too simple (all units do the same thing)")
        print(f"   2. Hidden size too small (no specialization emerges)")
        print(f"   3. Deformation estimation failing (check signal variance)")
        print(f"\n   Debug steps:")
        print(f"   - Check deformation signal ranges (should vary significantly)")
        print(f"   - Try a more complex task (parametric, matchsample)")
        print(f"   - Increase hidden_size to 256+ to encourage specialization")
        return None
    
    # Select features by task dynamics
    features, method_used = select_features_by_task_dynamics(
        hidden_states, deformation_features, task_dynamics='unknown', deformation_valid=True)
    
    if features is None:
        print("\n❌ Task has discrete dynamics - deformation method failed")
        print("Cannot perform perturbation analysis without unit classifications")
        return None
    
    # Test different cluster sizes to find optimal K
    if verbose:
        print(f"\n  Testing different cluster sizes (K=2 to 6):")
        from sklearn.metrics import silhouette_score
        for k in range(2, 7):
            labels_test = classify_units(features, n_clusters=k, return_details=False)
            sil_score = silhouette_score(features, labels_test) if len(np.unique(labels_test)) > 1 else 0
            cluster_sizes = [np.sum(labels_test == i) for i in range(k)]
            max_size = max(cluster_sizes)
            max_pct = 100 * max_size / len(labels_test)
            print(f"    K={k}: silhouette={sil_score:.3f}, largest cluster={max_size:3d} ({max_pct:4.1f}%)")
    
    # Cluster units with K=4
    print(f"\n  Using K=4 clusters for analysis:")
    unit_labels, details = classify_units(features, n_clusters=4, return_details=True)
    cluster_centers = details['centers']
    interpretation = interpret_clusters(features, unit_labels, feature_type='deformation')
    
    # Print clustering diagnostics
    if verbose:
        print(f"\nClustering diagnostics:")
        print(f"  Silhouette score: {details['silhouette']:.3f} (<0.3=poor, >0.5=good)")
        print(f"  Feature statistics:")
        print(f"    Rotation    - mean: {np.mean(features[:, 0]):.3f}, std: {np.std(features[:, 0]):.3f}")
        print(f"    Contraction - mean: {np.mean(features[:, 1]):.3f}, std: {np.std(features[:, 1]):.3f}")
        print(f"    Expansion   - mean: {np.mean(features[:, 2]):.3f}, std: {np.std(features[:, 2]):.3f}")
        
        # Check for uniform features (all same)
        max_feat_std = max(np.std(features[:, 0]), np.std(features[:, 1]), np.std(features[:, 2]))
        if max_feat_std < 0.01:
            print(f"\n  ❌ CRITICAL: Features have no variance (all units identical)!")
            print(f"     Clustering is finding arbitrary structure in noise.")
            print(f"     Results are NOT meaningful - method failed on this task.")
        
        print(f"\n  Cluster distribution:")
        for label in sorted(np.unique(unit_labels)):
            type_name = interpretation[label]['name']
            n_units = np.sum(unit_labels == label)
            pct = 100 * n_units / len(unit_labels)
            print(f"    Cluster {label} ({type_name:20s}): {n_units:3d} units ({pct:5.1f}%)")
        
        if details['silhouette'] < 0.3:
            print(f"\n  ⚠ WARNING: Low silhouette score suggests weak clustering structure!")
            print(f"     Clusters may not represent distinct functional types.")
            print(f"     Perturbation results may be unreliable.")
            print(f"\n  Recommendations:")
            print(f"     - Try a task with clearer dynamics (parametric, matchsample)")
            print(f"     - Increase hidden_size for more specialization")
            print(f"     - Check if features have sufficient variance")
    
    # 3. Test importance via ablation
    importance_results = test_unit_importance(
        rnn, task, unit_labels, interpretation, 
        n_test_trials=50, verbose=verbose)
    
    return importance_results


def make_tasks_compatible(task_a_name, task_b_name):
    """
    Create compatible task instances by adjusting parameters if needed.
    
    Args:
        task_a_name: Name of task A
        task_b_name: Name of task B
    
    Returns:
        (task_a, task_b, modified): Tuple of task instances and whether they were modified
    """
    # Try default tasks first
    task_a = get_task(task_a_name)
    task_b = get_task(task_b_name)
    
    # Check if already compatible
    if (task_a.input_size == task_b.input_size and 
        task_a.output_size == task_b.output_size):
        return task_a, task_b, False
    
    # Known compatible combinations with parameter adjustments
    compatible_pairs = {
        ('context', 'parametric'): (
            ContextIntegrationTask(n_contexts=1),  # input=2, output=1
            ParametricWorkingMemoryTask(),          # input=2, output=1
            "Context(n_contexts=1) + Parametric"
        ),
        ('parametric', 'context'): (
            ParametricWorkingMemoryTask(),          # input=2, output=1
            ContextIntegrationTask(n_contexts=1),  # input=2, output=1
            "Parametric + Context(n_contexts=1)"
        ),
        ('flipflop', 'gonogo'): (
            FlipFlopTask(n_bits=1),  # input=1, output=1
            GoNoGoTask(),             # input=1, output=1
            "FlipFlop(n_bits=1) + GoNoGo"
        ),
        ('gonogo', 'flipflop'): (
            GoNoGoTask(),             # input=1, output=1
            FlipFlopTask(n_bits=1),  # input=1, output=1
            "GoNoGo + FlipFlop(n_bits=1)"
        ),
    }
    
    pair_key = (task_a_name.lower(), task_b_name.lower())
    
    if pair_key in compatible_pairs:
        task_a_compat, task_b_compat, description = compatible_pairs[pair_key]
        return task_a_compat, task_b_compat, description
    
    # No compatible configuration found
    return task_a, task_b, None


def run_transfer_analysis(task_a='context', task_b='parametric',
                          hidden_size=128, n_epochs=2000, verbose=True):
    """
    Test cross-task transfer of unit classifications.
    
    Tests if units classified on Task A also matter for Task B performance.
    Uses the SAME RNN (trained on Task A) and tests on Task B inputs.
    
    Automatically adjusts task parameters to ensure compatibility when possible.
    
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
    
    # Create compatible task instances (auto-adjusts parameters if needed)
    task_obj_a, task_obj_b, modification = make_tasks_compatible(task_a, task_b)
    
    # Check if tasks are compatible
    if (task_obj_a.input_size != task_obj_b.input_size or 
        task_obj_a.output_size != task_obj_b.output_size):
        print(f"\n⚠️  ERROR: Incompatible task dimensions!")
        print(f"   {task_a}: input={task_obj_a.input_size}, output={task_obj_a.output_size}")
        print(f"   {task_b}: input={task_obj_b.input_size}, output={task_obj_b.output_size}")
        print(f"\n   No automatic compatibility fix available for this pair.")
        print(f"\n   Try these compatible pairs:")
        print(f"   - context + parametric (auto-adjusted)")
        print(f"   - flipflop + gonogo (auto-adjusted)")
        print(f"   - Two instances of same task")
        return None
    
    # Notify if tasks were modified
    if modification:
        print(f"\n💡 Auto-adjusted parameters for compatibility:")
        print(f"   Using: {modification}")
        print(f"   Dimensions: input={task_obj_a.input_size}, output={task_obj_a.output_size}")
    
    # 1. Train RNN on Task A
    rnn_a = VanillaRNN(task_obj_a.input_size, hidden_size, task_obj_a.output_size)
    
    print(f"\nTraining on {task_a}...")
    rnn_a, history_a = task_obj_a.train_rnn(rnn_a, n_epochs=n_epochs, verbose=False)
    print(f"Final accuracy: {history_a['accuracy'][-1]:.2%}")
    
    # 2. Classify units based on Task A
    print(f"\nClassifying units based on {task_a} dynamics...")
    hidden_states_a, inputs_a, _ = task_obj_a.extract_trajectories(rnn_a, n_trials=50)
    
    rotation_a, contraction_a, expansion_a, _ = estimate_deformation_from_rnn(hidden_states_a)
    rotation_a, contraction_a, expansion_a = smooth_deformation_signals(rotation_a, contraction_a, expansion_a, sigma=5)
    
    deformation_features_a = extract_rnn_unit_features(hidden_states_a, rotation_a, contraction_a, expansion_a)
    
    # Check feature variance
    feature_stds = np.std(deformation_features_a, axis=0)
    if np.max(feature_stds) < 0.01:
        print(f"\n❌ Features have no variance - method failed on {task_a}")
        return None
    
    features_a, _ = select_features_by_task_dynamics(
        hidden_states_a, deformation_features_a, task_dynamics='unknown', deformation_valid=True)
    
    if features_a is None:
        print(f"\n❌ {task_a} has discrete dynamics - cannot classify units")
        return None
    
    unit_labels_a, details_a = classify_units(features_a, n_clusters=4, return_details=True)
    cluster_centers_a = details_a['centers']
    interpretation_a = interpret_clusters(features_a, unit_labels_a, feature_type='deformation')
    
    # 3. Test transfer to Task B
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
    
    rotation, contraction, expansion, _ = estimate_deformation_from_rnn(hidden_states)
    rotation, contraction, expansion = smooth_deformation_signals(rotation, contraction, expansion, sigma=5)
    
    deformation_features = extract_rnn_unit_features(hidden_states, rotation, contraction, expansion)
    
    # Check feature variance
    feature_stds = np.std(deformation_features, axis=0)
    if np.max(feature_stds) < 0.01:
        print(f"\n❌ Features have no variance - method failed on this task")
        return None
    
    features, _ = select_features_by_task_dynamics(
        hidden_states, deformation_features, task_dynamics='unknown', deformation_valid=True)
    
    if features is None:
        print("\n❌ Discrete dynamics - cannot perform progressive ablation")
        return None
    
    unit_labels, details = classify_units(features, n_clusters=4, return_details=True)
    cluster_centers = details['centers']
    interpretation = interpret_clusters(features, unit_labels, feature_type='deformation')
    
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
    
    rotation, contraction, expansion, _ = estimate_deformation_from_rnn(hidden_states)
    rotation, contraction, expansion = smooth_deformation_signals(rotation, contraction, expansion, sigma=5)
    
    deformation_features = extract_rnn_unit_features(hidden_states, rotation, contraction, expansion)
    
    # Check feature variance
    feature_stds = np.std(deformation_features, axis=0)
    if np.max(feature_stds) < 0.01:
        print(f"\n❌ Features have no variance - method failed on this task")
        return None
    
    features, _ = select_features_by_task_dynamics(
        hidden_states, deformation_features, task_dynamics='unknown', deformation_valid=True)
    
    if features is None:
        print("\n❌ Discrete dynamics - cannot perform compression")
        return None
    
    unit_labels, details = classify_units(features, n_clusters=4, return_details=True)
    cluster_centers = details['centers']
    interpretation = interpret_clusters(features, unit_labels, feature_type='deformation')
    
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
    
    rotation, contraction, expansion, _ = estimate_deformation_from_rnn(hidden_states)
    rotation, contraction, expansion = smooth_deformation_signals(rotation, contraction, expansion, sigma=5)
    
    deformation_features = extract_rnn_unit_features(hidden_states, rotation, contraction, expansion)
    
    # Check feature variance
    feature_stds = np.std(deformation_features, axis=0)
    if np.max(feature_stds) < 0.01:
        print(f"\n❌ Features have no variance - method failed on {primary_task_name}")
        return None
    
    features, _ = select_features_by_task_dynamics(
        hidden_states, deformation_features, task_dynamics='unknown', deformation_valid=True)
    
    if features is None:
        print(f"\n❌ {primary_task_name} has discrete dynamics")
        return None
    
    unit_labels, details = classify_units(features, n_clusters=4, return_details=True)
    cluster_centers = details['centers']
    interpretation = interpret_clusters(features, unit_labels, feature_type='deformation')
    
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
