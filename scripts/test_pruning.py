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
    mask_tensor = torch.tensor(mask, dtype=torch.bool)
    
    if isinstance(rnn, VanillaRNN):
        with torch.no_grad():
            # Mask input-to-hidden weights (zero rows for pruned units)
            rnn.rnn.weight_ih_l0.data[~mask_tensor, :] = 0
            
            # Mask hidden-to-hidden weights (zero rows AND columns for pruned units)
            rnn.rnn.weight_hh_l0.data[~mask_tensor, :] = 0  # Zero rows (output)
            rnn.rnn.weight_hh_l0.data[:, ~mask_tensor] = 0  # Zero columns (input)
            
            # Mask biases for pruned units
            if hasattr(rnn.rnn, 'bias_ih_l0') and rnn.rnn.bias_ih_l0 is not None:
                rnn.rnn.bias_ih_l0.data[~mask_tensor] = 0
                rnn.rnn.bias_hh_l0.data[~mask_tensor] = 0
            
            # Mask hidden-to-output weights (zero columns for pruned units)
            rnn.fc.weight.data[:, ~mask_tensor] = 0
    
    elif isinstance(rnn, SimpleLSTM):
        with torch.no_grad():
            # LSTM has 4 gates: input, forget, cell, output
            # weight_ih_l0: (4*hidden_size, input_size)
            # weight_hh_l0: (4*hidden_size, hidden_size)
            hidden_size = rnn.hidden_size
            
            # For each gate, mask the corresponding rows
            for gate in range(4):
                gate_start = gate * hidden_size
                gate_end = (gate + 1) * hidden_size
                gate_mask = mask_tensor
                
                # Mask input-to-hidden for this gate
                rnn.lstm.weight_ih_l0.data[gate_start:gate_end][~gate_mask, :] = 0
                
                # Mask hidden-to-hidden for this gate (rows and columns)
                rnn.lstm.weight_hh_l0.data[gate_start:gate_end][~gate_mask, :] = 0
                rnn.lstm.weight_hh_l0.data[gate_start:gate_end][:, ~gate_mask] = 0
                
                # Mask biases
                if hasattr(rnn.lstm, 'bias_ih_l0') and rnn.lstm.bias_ih_l0 is not None:
                    rnn.lstm.bias_ih_l0.data[gate_start:gate_end][~gate_mask] = 0
                    rnn.lstm.bias_hh_l0.data[gate_start:gate_end][~gate_mask] = 0
            
            # Mask hidden-to-output
            rnn.fc.weight.data[:, ~mask_tensor] = 0
    
    elif isinstance(rnn, SimpleGRU):
        with torch.no_grad():
            # GRU has 3 gates: reset, update, new
            # weight_ih_l0: (3*hidden_size, input_size)
            # weight_hh_l0: (3*hidden_size, hidden_size)
            hidden_size = rnn.hidden_size
            
            # For each gate, mask the corresponding rows
            for gate in range(3):
                gate_start = gate * hidden_size
                gate_end = (gate + 1) * hidden_size
                gate_mask = mask_tensor
                
                # Mask input-to-hidden for this gate
                rnn.gru.weight_ih_l0.data[gate_start:gate_end][~gate_mask, :] = 0
                
                # Mask hidden-to-hidden for this gate (rows and columns)
                rnn.gru.weight_hh_l0.data[gate_start:gate_end][~gate_mask, :] = 0
                rnn.gru.weight_hh_l0.data[gate_start:gate_end][:, ~gate_mask] = 0
                
                # Mask biases
                if hasattr(rnn.gru, 'bias_ih_l0') and rnn.gru.bias_ih_l0 is not None:
                    rnn.gru.bias_ih_l0.data[gate_start:gate_end][~gate_mask] = 0
                    rnn.gru.bias_hh_l0.data[gate_start:gate_end][~gate_mask] = 0
            
            # Mask hidden-to-output
            rnn.fc.weight.data[:, ~mask_tensor] = 0
    
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
                              finetune_epochs=200, n_eval_trials=10,
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
        finetune_epochs: Epochs for fine-tuning after pruning
        n_eval_trials: Number of trials for accuracy evaluation
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
            
            # Create fresh RNN and reload weights
            rnn_pruned = arch_map[architecture](task.input_size, hidden_size, task.output_size)
            rnn_pruned.load_state_dict(rnn.state_dict())
            
            # Apply mask
            rnn_pruned = apply_unit_mask(rnn_pruned, mask)
            
            # Evaluate accuracy over multiple trials (more reliable)
            accuracies = []
            for _ in range(n_eval_trials):
                inputs, targets = task.generate_trial(batch_size=32)
                with torch.no_grad():
                    outputs = rnn_pruned(inputs, return_hidden_states=False)
                acc = task._compute_accuracy(outputs, targets)
                accuracies.append(acc)
            
            immediate_accuracy = np.mean(accuracies)
            
            # Fine-tune to measure recovery
            if finetune_epochs > 0:
                rnn_pruned, _ = task.train_rnn(
                    rnn_pruned, n_epochs=finetune_epochs, lr=0.0001,
                    trial_length=100, verbose=False
                )
                
                # Re-evaluate after fine-tuning
                accuracies = []
                for _ in range(n_eval_trials):
                    inputs, targets = task.generate_trial(batch_size=32)
                    with torch.no_grad():
                        outputs = rnn_pruned(inputs, return_hidden_states=False)
                    acc = task._compute_accuracy(outputs, targets)
                    accuracies.append(acc)
                
                final_accuracy = np.mean(accuracies)
            else:
                final_accuracy = immediate_accuracy
            
            results[strategy][str(prune_pct)] = final_accuracy
            
            if verbose:
                drop = baseline_accuracy - final_accuracy
                if finetune_epochs > 0:
                    recovery = final_accuracy - immediate_accuracy
                    print(f"  {strategy:12s}: {final_accuracy:.2%} (drop: {drop:.2%}, recovery: {recovery:+.2%})")
                else:
                    print(f"  {strategy:12s}: {final_accuracy:.2%} (drop: {drop:.2%})")
    
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
        finetune_epochs=200,
        n_eval_trials=10,
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
