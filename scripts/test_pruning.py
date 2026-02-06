"""
Network pruning experiment: Test functional importance of unit types.

This script tests whether units classified by deformation method are functionally
important by removing specific unit types and measuring accuracy recovery.

Experiments:
1. Type-specific pruning: Remove each cluster type (Rotators, Integrators, etc.)
2. Progressive pruning: Remove weakest units by correlation strength
3. Random pruning baseline: Remove random units

Demonstrates practical utility of deformation-based classification.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU
from src.tasks import get_task
from src.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.rnn_features import extract_rnn_unit_features, classify_units, interpret_clusters
from src.visualization import ensure_dirs


def apply_unit_mask(rnn, mask):
    """
    Zero out specified hidden units by masking weights.
    
    Args:
        rnn: VanillaRNN, SimpleLSTM, or SimpleGRU model
        mask: (hidden_size,) boolean array (True = keep, False = prune)
    
    Returns:
        rnn: Modified RNN with pruned units
    """
    mask = torch.tensor(mask, dtype=torch.float32)
    
    if isinstance(rnn, VanillaRNN):
        # Mask recurrent weights (both input and output)
        with torch.no_grad():
            # Input to hidden: zero rows for masked units
            rnn.rnn.weight_hh_l0.data *= mask.unsqueeze(1)
            # Hidden to hidden: zero rows and columns
            rnn.rnn.weight_hh_l0.data *= mask.unsqueeze(0)
            # Hidden to output: zero columns for masked units
            rnn.fc.weight.data *= mask.unsqueeze(0)
    
    elif isinstance(rnn, (SimpleLSTM, SimpleGRU)):
        # For LSTM/GRU, need to handle gate-wise weights
        # This is more complex - simplified version
        with torch.no_grad():
            if isinstance(rnn, SimpleLSTM):
                layer = rnn.lstm
            else:
                layer = rnn.gru
            
            # Get indices of units to keep
            keep_indices = torch.where(mask)[0]
            
            # This is a simplified pruning - just zero the weights
            # Full implementation would properly remove units
            for param_name in ['weight_hh_l0']:
                if hasattr(layer, param_name):
                    param = getattr(layer, param_name)
                    # Zero rows for pruned units (approximate)
                    for i, keep in enumerate(mask):
                        if not keep:
                            # Zero out contributions from this unit
                            param.data[:, i] = 0
            
            # Hidden to output
            rnn.fc.weight.data *= mask.unsqueeze(0)
    
    return rnn


def test_type_specific_pruning(task_name='flipflop', architecture='vanilla',
                                hidden_size=128, n_epochs=2000, finetune_epochs=500,
                                verbose=True):
    """
    Remove each cluster type and measure accuracy drop.
    
    Shows which unit types are most critical for task performance.
    
    Args:
        task_name: Task to test
        architecture: RNN architecture
        hidden_size: Number of hidden units
        n_epochs: Initial training epochs
        finetune_epochs: Fine-tuning epochs after pruning
        verbose: Print progress
    
    Returns:
        results: Dict with accuracy for each pruning condition
    """
    if verbose:
        print("\n" + "="*70)
        print(f"TYPE-SPECIFIC PRUNING: {task_name.upper()}")
        print("="*70)
    
    # Create task and RNN
    task = get_task(task_name)
    arch_map = {'vanilla': VanillaRNN, 'lstm': SimpleLSTM, 'gru': SimpleGRU}
    rnn = arch_map[architecture](task.input_size, hidden_size, task.output_size)
    
    # Train RNN
    if verbose:
        print("\nTraining baseline RNN...")
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                  trial_length=100, verbose=verbose)
    baseline_accuracy = history['accuracy'][-1]
    
    if verbose:
        print(f"Baseline accuracy: {baseline_accuracy:.2%}")
    
    # Extract and cluster
    if verbose:
        print("\nClustering units...")
    hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=50, trial_length=200)
    rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
    rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
    features = extract_rnn_unit_features(hidden_states, rot, con, exp)
    labels, details = classify_units(features, n_clusters=4, return_details=True)
    interpretation = interpret_clusters(features, labels)
    
    # Print cluster distribution
    if verbose:
        print("\nUnit type distribution:")
        for cluster_id, info in interpretation.items():
            print(f"  Cluster {cluster_id} ({info['name']}): {info['n_units']} units ({info['percentage']:.1f}%)")
    
    # Test pruning each cluster type
    results = {'baseline': baseline_accuracy}
    
    for cluster_id in range(4):
        cluster_info = interpretation[cluster_id]
        cluster_name = cluster_info['name']
        
        if verbose:
            print(f"\nPruning {cluster_name} units (Cluster {cluster_id})...")
        
        # Create mask (True = keep, False = prune)
        mask = labels != cluster_id
        n_pruned = np.sum(~mask)
        
        if n_pruned == 0:
            if verbose:
                print(f"  No units in this cluster, skipping")
            results[cluster_name] = baseline_accuracy
            continue
        
        if verbose:
            print(f"  Removing {n_pruned} units ({n_pruned/hidden_size*100:.1f}%)")
        
        # Create fresh RNN and reload trained weights
        rnn_pruned = arch_map[architecture](task.input_size, hidden_size, task.output_size)
        rnn_pruned.load_state_dict(rnn.state_dict())
        
        # Apply mask
        rnn_pruned = apply_unit_mask(rnn_pruned, mask)
        
        # Test immediately after pruning
        inputs, targets = task.generate_trial(batch_size=32)
        with torch.no_grad():
            outputs = rnn_pruned(inputs, return_hidden_states=False)
        immediate_accuracy = task._compute_accuracy(outputs, targets)
        
        if verbose:
            print(f"  Accuracy after pruning: {immediate_accuracy:.2%} "
                  f"(drop: {(baseline_accuracy - immediate_accuracy):.2%})")
        
        # Fine-tune
        if verbose:
            print(f"  Fine-tuning for {finetune_epochs} epochs...")
        
        rnn_pruned, finetune_history = task.train_rnn(
            rnn_pruned, n_epochs=finetune_epochs, lr=0.0001,  # Lower LR for fine-tuning
            trial_length=100, verbose=False
        )
        
        final_accuracy = finetune_history['accuracy'][-1]
        recovery = (final_accuracy - immediate_accuracy) / (baseline_accuracy - immediate_accuracy) * 100
        
        if verbose:
            print(f"  Accuracy after fine-tuning: {final_accuracy:.2%}")
            print(f"  Recovery: {recovery:.1f}%")
        
        results[cluster_name] = {
            'immediate': immediate_accuracy,
            'final': final_accuracy,
            'n_pruned': int(n_pruned),
            'percentage_pruned': n_pruned/hidden_size*100
        }
    
    return results, interpretation


def test_progressive_pruning(task_name='flipflop', architecture='vanilla',
                              hidden_size=128, n_epochs=2000,
                              prune_percentages=[10, 20, 30, 40, 50],
                              verbose=True):
    """
    Progressively prune weakest units and measure accuracy.
    
    Compares three strategies:
    1. Prune by correlation strength (deformation method)
    2. Prune by activation variance (simple baseline)
    3. Prune randomly (control)
    
    Args:
        task_name: Task to test
        architecture: RNN architecture
        hidden_size: Number of hidden units
        n_epochs: Training epochs
        prune_percentages: List of pruning percentages to test
        verbose: Print progress
    
    Returns:
        results: Dict mapping strategy → pruning % → accuracy
    """
    if verbose:
        print("\n" + "="*70)
        print(f"PROGRESSIVE PRUNING: {task_name.upper()}")
        print("="*70)
    
    # Train baseline RNN
    task = get_task(task_name)
    arch_map = {'vanilla': VanillaRNN, 'lstm': SimpleLSTM, 'gru': SimpleGRU}
    rnn = arch_map[architecture](task.input_size, hidden_size, task.output_size)
    
    if verbose:
        print("\nTraining baseline RNN...")
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                  trial_length=100, verbose=verbose)
    baseline_accuracy = history['accuracy'][-1]
    
    # Extract features for ranking
    if verbose:
        print("\nComputing unit rankings...")
    hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=50, trial_length=200)
    rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
    rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
    features = extract_rnn_unit_features(hidden_states, rot, con, exp)
    
    # Strategy 1: Rank by max absolute correlation (deformation strength)
    correlation_strength = np.max(np.abs(features), axis=1)
    correlation_ranking = np.argsort(correlation_strength)  # Ascending (weakest first)
    
    # Strategy 2: Rank by temporal variance (activation variability)
    temporal_variance = np.var(hidden_states, axis=1)
    variance_ranking = np.argsort(temporal_variance)  # Ascending (least variable first)
    
    # Test each pruning percentage
    results = {
        'correlation': {'0': baseline_accuracy},
        'variance': {'0': baseline_accuracy},
        'random': {'0': baseline_accuracy}
    }
    
    for prune_pct in prune_percentages:
        if verbose:
            print(f"\nPruning {prune_pct}% of units...")
        
        n_prune = int(hidden_size * prune_pct / 100)
        
        # Test each strategy
        for strategy in ['correlation', 'variance', 'random']:
            if strategy == 'correlation':
                prune_indices = correlation_ranking[:n_prune]
            elif strategy == 'variance':
                prune_indices = variance_ranking[:n_prune]
            else:  # random
                np.random.seed(42)
                prune_indices = np.random.choice(hidden_size, n_prune, replace=False)
            
            # Create mask
            mask = np.ones(hidden_size, dtype=bool)
            mask[prune_indices] = False
            
            #Create fresh RNN and reload weights
            rnn_pruned = arch_map[architecture](task.input_size, hidden_size, task.output_size)
            rnn_pruned.load_state_dict(rnn.state_dict())
            
            # Apply mask
            rnn_pruned = apply_unit_mask(rnn_pruned, mask)
            
            # Test accuracy
            inputs, targets = task.generate_trial(batch_size=32)
            with torch.no_grad():
                outputs = rnn_pruned(inputs, return_hidden_states=False)
            accuracy = task._compute_accuracy(outputs, targets)
            
            results[strategy][str(prune_pct)] = accuracy
            
            if verbose:
                drop = baseline_accuracy - accuracy
                print(f"  {strategy:12s}: {accuracy:.2%} (drop: {drop:.2%})")
    
    return results


def main():
    """Run all pruning experiments."""
    ensure_dirs('results/rnn_figures')
    
    print("\n" + "="*70)
    print("NETWORK PRUNING EXPERIMENTS")
    print("="*70)
    
    # Test 1: Type-specific pruning
    print("\nTest 1: Type-specific pruning (removing each cluster type)")
    results_type, interpretation = test_type_specific_pruning(
        task_name='flipflop',
        architecture='vanilla',
        hidden_size=128,
        n_epochs=2000,
        finetune_epochs=500,
        verbose=True
    )
    
    # Test 2: Progressive pruning
    print("\nTest 2: Progressive pruning (comparing ranking strategies)")
    results_progressive = test_progressive_pruning(
        task_name='flipflop',
        architecture='vanilla',
        hidden_size=128,
        n_epochs=2000,
        prune_percentages=[10, 20, 30, 40, 50],
        verbose=True
    )
    
    # Summary
    print("\n" + "="*70)
    print("PRUNING SUMMARY")
    print("="*70)
    
    print("\nType-specific pruning:")
    print(f"  Baseline: {results_type['baseline']:.2%}")
    for cluster_name, result in results_type.items():
        if cluster_name != 'baseline' and isinstance(result, dict):
            drop = results_type['baseline'] - result['final']
            print(f"  {cluster_name:15s}: {result['final']:.2%} (drop: {drop:.2%}, pruned {result['percentage_pruned']:.1f}%)")
    
    print("\nProgressive pruning (50% removal):")
    for strategy in ['correlation', 'variance', 'random']:
        acc = results_progressive[strategy]['50']
        drop = results_progressive[strategy]['0'] - acc
        print(f"  {strategy:12s}: {acc:.2%} (drop: {drop:.2%})")
    
    print("\n" + "="*70)
    print("✓ Pruning experiments complete")
    print("="*70)


if __name__ == "__main__":
    main()
