"""
Perturbation and ablation analysis for RNN functional units.

Tests functional importance of deformation-classified unit types through
selective ablation and cross-task transfer experiments.
"""

import numpy as np
import torch
import copy
from src.analysis.rnn_features import interpret_clusters


def ablate_units(rnn, unit_indices, method='zero'):
    """
    Ablate specific units in RNN.
    
    Args:
        rnn: RNN model
        unit_indices: List of unit indices to ablate
        method: 'zero' (zero weights) or 'noise' (add noise)
    
    Returns:
        ablated_rnn: New RNN with specified units removed
    """
    ablated_rnn = copy.deepcopy(rnn)
    
    if method == 'zero':
        # Zero out all connections to/from ablated units
        with torch.no_grad():
            # Recurrent connections
            ablated_rnn.rnn.weight_hh_l0[:, unit_indices] = 0  # Inputs to ablated units
            ablated_rnn.rnn.weight_hh_l0[unit_indices, :] = 0  # Outputs from ablated units
            
            # Input connections
            ablated_rnn.rnn.weight_ih_l0[unit_indices, :] = 0
            
            # Biases
            ablated_rnn.rnn.bias_hh_l0[unit_indices] = 0
            ablated_rnn.rnn.bias_ih_l0[unit_indices] = 0
    
    elif method == 'noise':
        # Add strong noise to ablated units
        with torch.no_grad():
            noise_scale = 10.0
            ablated_rnn.rnn.weight_hh_l0[:, unit_indices] += torch.randn_like(
                ablated_rnn.rnn.weight_hh_l0[:, unit_indices]) * noise_scale
            ablated_rnn.rnn.weight_hh_l0[unit_indices, :] += torch.randn_like(
                ablated_rnn.rnn.weight_hh_l0[unit_indices, :]) * noise_scale
    
    return ablated_rnn


def test_unit_importance(rnn, task, unit_labels, interpretation, 
                         n_test_trials=50, verbose=True):
    """
    Test functional importance of each unit type via ablation.
    
    Protocol:
    1. For each unit type (Integrator, Rotator, Explorer, Mixed)
    2. Ablate all units of that type
    3. Measure task performance degradation
    
    Args:
        rnn: Trained RNN model
        task: Task object
        unit_labels: (n_units,) cluster labels
        interpretation: Dict from interpret_clusters()
        n_test_trials: Number of trials for testing
        verbose: Print output
    
    Returns:
        importance_results: Dict mapping unit type to performance metrics
    """
    if verbose:
        print("\n" + "="*70)
        print("UNIT TYPE IMPORTANCE ANALYSIS")
        print("="*70)
    
    # Baseline performance (no ablation)
    baseline_accuracy = evaluate_rnn_performance(rnn, task, n_test_trials)
    
    if verbose:
        print(f"\nBaseline accuracy: {baseline_accuracy:.2f}%")
        print(f"\nAblation results:")
        print(f"{'Unit Type':20} {'Remaining':>10} {'Ablated':>10} {'Accuracy':>10} {'Drop':>10}")
        print("-"*70)
    
    importance_results = {}
    
    # Test each cluster type
    unique_labels = np.unique(unit_labels)
    for label in unique_labels:
        # Get unit indices for this type
        unit_indices = np.where(unit_labels == label)[0].tolist()
        n_ablated = len(unit_indices)
        n_remaining = len(unit_labels) - n_ablated
        
        type_name = interpretation[label]['name']
        
        # Ablate these units
        ablated_rnn = ablate_units(rnn, unit_indices, method='zero')
        
        # Test performance
        ablated_accuracy = evaluate_rnn_performance(ablated_rnn, task, n_test_trials)
        
        # Compute drop
        accuracy_drop = baseline_accuracy - ablated_accuracy
        
        importance_results[type_name] = {
            'n_units': n_ablated,
            'baseline_accuracy': baseline_accuracy,
            'ablated_accuracy': ablated_accuracy,
            'accuracy_drop': accuracy_drop,
            'relative_drop': accuracy_drop / baseline_accuracy * 100
        }
        
        if verbose:
            print(f"{type_name:20} {n_remaining:10d} {n_ablated:10d} "
                  f"{ablated_accuracy:9.2f}% {accuracy_drop:9.2f}%")
    
    if verbose:
        print("="*70)
    
    return importance_results


def evaluate_rnn_performance(rnn, task, n_trials=50):
    """
    Evaluate RNN accuracy on task.
    
    Args:
        rnn: RNN model
        task: Task object
        n_trials: Number of test trials
    
    Returns:
        accuracy: Percentage accuracy
    """
    rnn.eval()
    total_correct = 0
    total_samples = 0
    
    with torch.no_grad():
        for _ in range(n_trials):
            # Generate trial
            inputs, targets = task.generate_trial(length=200, batch_size=1)
            
            # Forward pass
            outputs = rnn(inputs)
            
            # Compute accuracy
            accuracy = task._compute_accuracy(outputs, targets)
            total_correct += accuracy
            total_samples += 1
    
    return (total_correct / total_samples) * 100


def cross_task_transfer(rnn_task_a, task_a, task_b, unit_labels, interpretation,
                        n_test_trials=50, verbose=True):
    """
    Test cross-task transfer of unit classifications.
    
    Protocol:
    1. Units classified on Task A (e.g., 'Integrator' units)
    2. Ablate those units
    3. Test on Task B
    4. Does ablation hurt Task B in expected ways?
    
    Example:
    - 'Integrator' units from Context task
    - Ablate them
    - Test on FlipFlop (also integration task)
    - Prediction: Should hurt FlipFlop performance
    
    Args:
        rnn_task_a: RNN trained on Task A
        task_a: Task A object
        task_b: Task B object  
        unit_labels: Cluster labels from Task A
        interpretation: Interpretation dict from Task A
        n_test_trials: Number of test trials
        verbose: Print output
    
    Returns:
        transfer_results: Dict with cross-task ablation results
    """
    if verbose:
        print("\n" + "="*70)
        print(f"CROSS-TASK TRANSFER ANALYSIS")
        print(f"Training Task: {task_a.__class__.__name__}")
        print(f"Test Task: {task_b.__class__.__name__}")
        print("="*70)
    
    # Train on Task B
    rnn_task_b = copy.deepcopy(rnn_task_a)
    if verbose:
        print(f"\nTraining on Task B...")
    
    task_b.train_rnn(rnn_task_b, n_epochs=1000, verbose=False)
    
    # Baseline performance on Task B
    baseline_b = evaluate_rnn_performance(rnn_task_b, task_b, n_test_trials)
    
    if verbose:
        print(f"Baseline accuracy on Task B: {baseline_b:.2f}%")
        print(f"\nAblating Task A unit types, testing on Task B:")
        print(f"{'Unit Type (Task A)':25} {'Accuracy on Task B':>20} {'Drop':>10}")
        print("-"*70)
    
    transfer_results = {}
    
    # Test each unit type from Task A
    unique_labels = np.unique(unit_labels)
    for label in unique_labels:
        # Get unit indices  
        unit_indices = np.where(unit_labels == label)[0].tolist()
        type_name = interpretation[label]['name']
        
        # Ablate these units in Task B RNN
        ablated_rnn_b = ablate_units(rnn_task_b, unit_indices, method='zero')
        
        # Test on Task B
        ablated_accuracy_b = evaluate_rnn_performance(ablated_rnn_b, task_b, n_test_trials)
        accuracy_drop_b = baseline_b - ablated_accuracy_b
        
        transfer_results[type_name] = {
            'baseline_task_b': baseline_b,
            'ablated_task_b': ablated_accuracy_b,
            'drop_task_b': accuracy_drop_b
        }
        
        if verbose:
            print(f"{type_name:25} {ablated_accuracy_b:19.2f}% {accuracy_drop_b:9.2f}%")
    
    if verbose:
        print("="*70)
    
    return transfer_results


def progressive_ablation(rnn, task, unit_labels, interpretation, 
                         strategy='low_confidence_first', n_test_trials=50, verbose=True):
    """
    Progressively ablate units and measure performance degradation.
    
    Tests prediction: Low-confidence units can be removed with minimal impact.
    
    Args:
        rnn: Trained RNN
        task: Task object
        unit_labels: Cluster labels
        interpretation: Interpretation dict
        strategy: 'low_confidence_first', 'high_confidence_first', or 'random'
        n_test_trials: Test trials
        verbose: Print output
    
    Returns:
        ablation_curve: List of (n_ablated, accuracy) tuples
    """
    if verbose:
        print("\n" + "="*70)
        print(f"PROGRESSIVE ABLATION: {strategy}")
        print("="*70)
    
    n_units = len(unit_labels)
    
    # Determine ablation order
    if strategy == 'low_confidence_first':
        # Ablate 'Mixed' and 'Type?' before 'Type'
        confidence_order = {'very_low': 0, 'low': 1, 'high': 2, 'user_specified': 3}
        unit_order = sorted(range(n_units), 
                           key=lambda i: confidence_order.get(
                               interpretation[unit_labels[i]].get('confidence', 'high'), 2))
    
    elif strategy == 'high_confidence_first':
        # Ablate high confidence first
        confidence_order = {'very_low': 2, 'low': 1, 'high': 0, 'user_specified': 0}
        unit_order = sorted(range(n_units),
                           key=lambda i: confidence_order.get(
                               interpretation[unit_labels[i]].get('confidence', 'high'), 0))
    
    elif strategy == 'random':
        unit_order = np.random.permutation(n_units).tolist()
    
    else:
        raise ValueError(f"Unknown strategy: {strategy}")
    
    # Progressive ablation
    ablation_curve = []
    ablated_indices = []
    
    # Test points: every 10% of units
    test_points = list(range(0, n_units + 1, max(1, n_units // 10)))
    if test_points[-1] != n_units:
        test_points.append(n_units)
    
    if verbose:
        print(f"\n{'Units Ablated':15} {'Percentage':>12} {'Accuracy':>12} {'Drop from Baseline':>20}")
        print("-"*70)
    
    baseline_accuracy = None
    
    for n_ablated in test_points:
        # Ablate first n_ablated units in order
        ablated_indices = unit_order[:n_ablated]
        
        if n_ablated == 0:
            # Baseline
            accuracy = evaluate_rnn_performance(rnn, task, n_test_trials)
            baseline_accuracy = accuracy
        else:
            # Ablate units
            ablated_rnn = ablate_units(rnn, ablated_indices, method='zero')
            accuracy = evaluate_rnn_performance(ablated_rnn, task, n_test_trials)
        
        pct_ablated = (n_ablated / n_units) * 100
        drop = baseline_accuracy - accuracy
        
        ablation_curve.append((n_ablated, pct_ablated, accuracy, drop))
        
        if verbose:
            print(f"{n_ablated:15d} {pct_ablated:11.1f}% {accuracy:11.2f}% {drop:19.2f}%")
    
    if verbose:
        print("="*70)
    
    return ablation_curve


def confidence_guided_pruning(rnn, task, unit_labels, interpretation, 
                               target_compression=0.4, n_test_trials=50, verbose=True):
    """
    Network compression guided by deformation confidence levels.
    
    Removes low-confidence 'Mixed' units preferentially.
    
    Args:
        rnn: Trained RNN
        task: Task object
        unit_labels: Cluster labels
        interpretation: Interpretation dict
        target_compression: Fraction of units to remove (e.g., 0.4 = 40% compression)
        n_test_trials: Test trials
        verbose: Print output
    
    Returns:
        pruning_results: Dict with compression results
    """
    if verbose:
        print("\n" + "="*70)
        print(f"CONFIDENCE-GUIDED NETWORK COMPRESSION")
        print(f"Target: {target_compression*100:.0f}% compression")
        print("="*70)
    
    n_units = len(unit_labels)
    n_to_prune = int(n_units * target_compression)
    
    # Baseline
    baseline_accuracy = evaluate_rnn_performance(rnn, task, n_test_trials)
    
    # Sort units by confidence (low to high)
    confidence_order = {'very_low': 0, 'low': 1, 'high': 2, 'user_specified': 3}
    unit_scores = [(i, confidence_order.get(interpretation[unit_labels[i]].get('confidence', 'high'), 2))
                   for i in range(n_units)]
    unit_scores.sort(key=lambda x: x[1])  # Sort by confidence
    
    # Prune lowest-confidence units
    units_to_prune = [idx for idx, _ in unit_scores[:n_to_prune]]
    
    # Ablate
    pruned_rnn = ablate_units(rnn, units_to_prune, method='zero')
    pruned_accuracy = evaluate_rnn_performance(pruned_rnn, task, n_test_trials)
    
    # Compare to random pruning
    random_units = np.random.choice(n_units, n_to_prune, replace=False).tolist()
    random_rnn = ablate_units(rnn, random_units, method='zero')
    random_accuracy = evaluate_rnn_performance(random_rnn, task, n_test_trials)
    
    if verbose:
        print(f"\nBaseline (no pruning):       {baseline_accuracy:.2f}%")
        print(f"Confidence-guided pruning:   {pruned_accuracy:.2f}% (drop: {baseline_accuracy - pruned_accuracy:.2f}%)")
        print(f"Random pruning:              {random_accuracy:.2f}% (drop: {baseline_accuracy - random_accuracy:.2f}%)")
        print(f"\nCompression ratio: {n_to_prune}/{n_units} = {target_compression*100:.0f}%")
        
        if pruned_accuracy > random_accuracy:
            improvement = pruned_accuracy - random_accuracy
            print(f"✓ Confidence-guided pruning outperforms random by {improvement:.2f}%")
        else:
            print(f"✗ Random pruning performed better")
        
        print("="*70)
    
    pruning_results = {
        'baseline_accuracy': baseline_accuracy,
        'pruned_accuracy': pruned_accuracy,
        'random_accuracy': random_accuracy,
        'n_pruned': n_to_prune,
        'compression_ratio': target_compression,
        'units_pruned': units_to_prune,
        'advantage_over_random': pruned_accuracy - random_accuracy
    }
    
    return pruning_results


def task_specific_importance(rnn, unit_labels, interpretation, tasks, 
                              n_test_trials=50, verbose=True):
    """
    Test which unit types are important for which tasks.
    
    Ablates each unit type and tests across multiple tasks to identify
    task-specific functional roles.
    
    Args:
        rnn: Trained RNN (should work on multiple tasks)
        unit_labels: Cluster labels
        interpretation: Interpretation dict
        tasks: List of task objects
        n_test_trials: Test trials per task
        verbose: Print output
    
    Returns:
        importance_matrix: Dict mapping (unit_type, task) -> accuracy_drop
    """
    if verbose:
        print("\n" + "="*70)
        print("TASK-SPECIFIC UNIT IMPORTANCE")
        print("="*70)
    
    importance_matrix = {}
    
    # Get baseline performance on each task
    baselines = {}
    for task in tasks:
        task_name = task.__class__.__name__
        baselines[task_name] = evaluate_rnn_performance(rnn, task, n_test_trials)
    
    if verbose:
        print(f"\nBaseline performance:")
        for task_name, acc in baselines.items():
            print(f"  {task_name:30} {acc:.2f}%")
    
    # Test ablation of each unit type across tasks
    unique_labels = np.unique(unit_labels)
    
    if verbose:
        print(f"\nImportance matrix (accuracy drop %):")
        header = f"{'Unit Type':20}"
        for task in tasks:
            header += f" {task.__class__.__name__[:15]:>15}"
        print(header)
        print("-" * (20 + 16 * len(tasks)))
    
    for label in unique_labels:
        unit_indices = np.where(unit_labels == label)[0].tolist()
        type_name = interpretation[label]['name']
        
        ablated_rnn = ablate_units(rnn, unit_indices, method='zero')
        
        row = f"{type_name:20}"
        for task in tasks:
            task_name = task.__class__.__name__
            ablated_acc = evaluate_rnn_performance(ablated_rnn, task, n_test_trials)
            drop = baselines[task_name] - ablated_acc
            
            importance_matrix[(type_name, task_name)] = drop
            row += f" {drop:15.2f}"
        
        if verbose:
            print(row)
    
    if verbose:
        print("="*70)
    
    return importance_matrix
