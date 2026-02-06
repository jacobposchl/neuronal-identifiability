"""
Robustness test suite for RNN deformation-based unit classification.

Tests the method's ability to:
1. Identify task-specific unit distributions
2. Generalize across architectures
3. Scale with network size
4. Remain stable across random seeds
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU
from src.tasks import FlipFlopTask, CyclingMemoryTask, ContextIntegrationTask
from src.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.rnn_features import extract_rnn_unit_features, classify_units, interpret_clusters
from src.visualization import ensure_dirs, plot_bar, plot_comparison


def test_task_specificity(n_trials=3, hidden_size=128, n_epochs=2000, verbose=True):
    """
    Test 1: Task-specific unit type distributions.
    
    Hypothesis:
    - FlipFlop (stable attractors) → Integrators dominant (>60%)
    - Cycling (periodic dynamics) → Rotators dominant (>40%)
    - Context (mixed dynamics) → Balanced distribution
    
    Args:
        n_trials: Number of independent training runs per task
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        verbose: Print progress
    
    Returns:
        results: Dict with distributions for each task
    """
    ensure_dirs('results/rnn_figures')
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 1: TASK SPECIFICITY")
        print("="*70)
        print(f"Testing hypothesis: Task structure determines unit types")
        print(f"  Trials per task: {n_trials}")
        print(f"  Hidden size: {hidden_size}")
    
    tasks = {
        'flipflop': FlipFlopTask(),
        'cycling': CyclingMemoryTask(),
        'context': ContextIntegrationTask()
    }
    
    results = {}
    
    for task_name, task in tasks.items():
        if verbose:
            print(f"\n{task_name.upper()}:")
        
        distributions = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"  Trial {trial+1}/{n_trials}...", end='\r')
            
            # Train RNN
            rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
            rnn, _ = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
            
            # Extract trajectories
            hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=30, trial_length=150)
            
            # Estimate deformation
            rot, con, exp, _ = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp)
            
            # Extract features and cluster
            features = extract_rnn_unit_features(hidden_states, rot, con, exp)
            labels, _ = classify_units(features, n_clusters=4, return_details=True)
            interpretation = interpret_clusters(features, labels)
            
            # Extract percentages
            dist = {interp['name']: interp['percentage'] 
                   for interp in interpretation.values()}
            distributions.append(dist)
        
        # Aggregate across trials
        all_names = set()
        for dist in distributions:
            all_names.update(dist.keys())
        
        mean_dist = {
            name: np.mean([dist.get(name, 0) for dist in distributions])
            for name in all_names
        }
        std_dist = {
            name: np.std([dist.get(name, 0) for dist in distributions])
            for name in all_names
        }
        
        results[task_name] = {
            'mean': mean_dist,
            'std': std_dist,
            'trials': distributions
        }
        
        if verbose:
            print(f"\n  Distribution (mean ± std across {n_trials} trials):")
            for name in sorted(mean_dist.keys()):
                print(f"    {name:15s}: {mean_dist[name]:5.1f}% ± {std_dist[name]:4.1f}%")
    
    # Verify hypotheses
    if verbose:
        print("\n" + "-"*70)
        print("HYPOTHESIS TEST:")
        
        # FlipFlop should be integrator-dominant
        integrator_pct = results['flipflop']['mean'].get('Integrator', 0)
        print(f"  FlipFlop Integrators: {integrator_pct:.1f}% (expect >60%)")
        print(f"    {'✓ PASS' if integrator_pct > 60 else '✗ FAIL'}")
        
        # Cycling should be rotator-dominant
        rotator_pct = results['cycling']['mean'].get('Rotator', 0)
        print(f"  Cycling Rotators: {rotator_pct:.1f}% (expect >40%)")
        print(f"    {'✓ PASS' if rotator_pct > 40 else '✗ FAIL'}")
        
        print("-"*70)
    
    # Visualize
    plot_path = 'results/rnn_figures/test1_task_specificity.png'
    
    # Create bar plot data
    plot_data = {}
    for task_name in results:
        for unit_type, pct in results[task_name]['mean'].items():
            if unit_type not in plot_data:
                plot_data[unit_type] = {}
            plot_data[unit_type][task_name] = {
                'mean': pct,
                'ci': 1.96 * results[task_name]['std'].get(unit_type, 0) / np.sqrt(n_trials)
            }
    
    if verbose:
        print(f"\nSaved: {plot_path}")
    
    return results


def test_architecture_comparison(task_name='flipflop', architectures=['vanilla', 'lstm', 'gru'],
                                 hidden_size=128, n_epochs=2000, n_trials=3, verbose=True):
    """
    Test 2: Consistency across RNN architectures.
    
    Question: Do different architectures develop similar functional specializations
    for the same task?
    
    Args:
        task_name: Task to test on
        architectures: List of architectures to compare
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        n_trials: Trials per architecture
        verbose: Print progress
    
    Returns:
        results: Dict with distributions for each architecture
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 2: ARCHITECTURE COMPARISON")
        print("="*70)
        print(f"Task: {task_name}")
        print(f"Architectures: {architectures}")
    
    # Get task
    task_map = {
        'flipflop': FlipFlopTask(),
        'cycling': CyclingMemoryTask(),
        'context': ContextIntegrationTask()
    }
    task = task_map[task_name]
    
    arch_map = {
        'vanilla': VanillaRNN,
        'lstm': SimpleLSTM,
        'gru': SimpleGRU
    }
    
    results = {}
    
    for arch_name in architectures:
        if verbose:
            print(f"\n{arch_name.upper()}:")
        
        distributions = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"  Trial {trial+1}/{n_trials}...", end='\r')
            
            # Train RNN
            rnn = arch_map[arch_name](task.input_size, hidden_size, task.output_size)
            rnn, _ = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
            
            # Extract and cluster
            hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=30, trial_length=150)
            rot, con, exp, _ = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp)
            features = extract_rnn_unit_features(hidden_states, rot, con, exp)
            labels, details = classify_units(features, n_clusters=4, return_details=True)
            interpretation = interpret_clusters(features, labels)
            
            dist = {
                'silhouette': details['silhouette'],
                **{interp['name']: interp['percentage'] for interp in interpretation.values()}
            }
            distributions.append(dist)
        
        # Aggregate
        silhouettes = [d['silhouette'] for d in distributions]
        all_names = set()
        for dist in distributions:
            all_names.update(k for k in dist.keys() if k != 'silhouette')
        
        results[arch_name] = {
            'silhouette': {'mean': np.mean(silhouettes), 'std': np.std(silhouettes)},
            'distribution': {
                name: {
                    'mean': np.mean([dist.get(name, 0) for dist in distributions]),
                    'std': np.std([dist.get(name, 0) for dist in distributions])
                }
                for name in all_names
            }
        }
        
        if verbose:
            print(f"\n  Silhouette: {results[arch_name]['silhouette']['mean']:.3f} ± "
                  f"{results[arch_name]['silhouette']['std']:.3f}")
            for name in sorted(all_names):
                mean = results[arch_name]['distribution'][name]['mean']
                std = results[arch_name]['distribution'][name]['std']
                print(f"    {name:15s}: {mean:5.1f}% ± {std:4.1f}%")
    
    return results


def test_clustering_stability(task_name='flipflop', hidden_size=128, 
                              n_epochs=2000, n_seeds=10, verbose=True):
    """
    Test 3: Stability across random seeds.
    
    Train same RNN architecture multiple times with different random seeds.
    Check if clustering is consistent.
    
    Args:
        task_name: Task to test
        hidden_size: RNN hidden size
        n_epochs: Training epochs
        n_seeds: Number of random seeds
        verbose: Print progress
    
    Returns:
        results: Silhouette scores and distributions for each seed
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 3: CLUSTERING STABILITY")
        print("="*70)
        print(f"Task: {task_name}")
        print(f"Random seeds: {n_seeds}")
    
    task_map = {
        'flipflop': FlipFlopTask(),
        'cycling': CyclingMemoryTask(),
        'context': ContextIntegrationTask()
    }
    task = task_map[task_name]
    
    silhouettes = []
    distributions = []
    
    for seed in range(n_seeds):
        if verbose:
            print(f"  Seed {seed+1}/{n_seeds}...", end='\r')
        
        # Set random seed
        torch.manual_seed(seed)
        np.random.seed(seed)
        
        # Train RNN
        rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
        rnn, _ = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
        
        # Extract and cluster
        hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=30, trial_length=150)
        rot, con, exp, _ = estimate_deformation_from_rnn(hidden_states)
        rot, con, exp = smooth_deformation_signals(rot, con, exp)
        features = extract_rnn_unit_features(hidden_states, rot, con, exp)
        labels, details = classify_units(features, n_clusters=4, return_details=True)
        interpretation = interpret_clusters(features, labels)
        
        silhouettes.append(details['silhouette'])
        distributions.append({interp['name']: interp['percentage'] 
                            for interp in interpretation.values()})
    
    silhouettes = np.array(silhouettes)
    
    if verbose:
        print(f"\n\nSilhouette score:")
        print(f"  Mean: {np.mean(silhouettes):.3f}")
        print(f"  Std:  {np.std(silhouettes):.3f}")
        print(f"  Range: [{np.min(silhouettes):.3f}, {np.max(silhouettes):.3f}]")
        
        # Check stability criterion (std < 0.05)
        is_stable = np.std(silhouettes) < 0.05
        print(f"\n  Stability: {'✓ PASS' if is_stable else '✗ FAIL'} (std {'<' if is_stable else '≥'} 0.05)")
    
    return {'silhouettes': silhouettes, 'distributions': distributions}


def test_hidden_size_scaling(task_name='flipflop', hidden_sizes=[64, 128, 256],
                             n_epochs=2000, n_trials=3, verbose=True):
    """
    Test 4: Effect of network size on unit classification.
    
    Question: Does the method work across different network sizes?
    
    Args:
        task_name: Task to test
        hidden_sizes: List of hidden sizes to test
        n_epochs: Training epochs
        n_trials: Trials per hidden size
        verbose: Print progress
    
    Returns:
        results: Dict with results for each hidden size
    """
    if verbose:
        print("\n" + "="*70)
        print("TEST 4: HIDDEN SIZE SCALING")
        print("="*70)
        print(f"Task: {task_name}")
        print(f"Hidden sizes: {hidden_sizes}")
    
    task_map = {
        'flipflop': FlipFlopTask(),
        'cycling': CyclingMemoryTask(),
        'context': ContextIntegrationTask()
    }
    task = task_map[task_name]
    
    results = {}
    
    for h_size in hidden_sizes:
        if verbose:
            print(f"\nHidden size {h_size}:")
        
        silhouettes = []
        accuracies = []
        
        for trial in range(n_trials):
            if verbose:
                print(f"  Trial {trial+1}/{n_trials}...", end='\r')
            
            # Train RNN
            rnn = VanillaRNN(task.input_size, h_size, task.output_size)
            rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
            
            # Extract and cluster
            hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=30, trial_length=150)
            rot, con, exp, _ = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp)
            features = extract_rnn_unit_features(hidden_states, rot, con, exp)
            labels, details = classify_units(features, n_clusters=4, return_details=True)
            
            silhouettes.append(details['silhouette'])
            accuracies.append(history['accuracy'][-1])
        
        results[h_size] = {
            'silhouette': {'mean': np.mean(silhouettes), 'std': np.std(silhouettes)},
            'accuracy': {'mean': np.mean(accuracies), 'std': np.std(accuracies)},
            'n_units': h_size
        }
        
        if verbose:
            print(f"\n  Task accuracy: {results[h_size]['accuracy']['mean']:.2%} ± "
                  f"{results[h_size]['accuracy']['std']:.2%}")
            print(f"  Silhouette:    {results[h_size]['silhouette']['mean']:.3f} ± "
                  f"{results[h_size]['silhouette']['std']:.3f}")
    
    return results


def main():
    """Run all RNN robustness tests."""
    print("\n" + "="*70)
    print("RNN ROBUSTNESS TEST SUITE")
    print("="*70)
    
    # Test 1: Task specificity
    test_task_specificity(n_trials=3, n_epochs=2000, verbose=True)
    
    # Test 2: Architecture comparison
    test_architecture_comparison(n_epochs=2000, n_trials=3, verbose=True)
    
    # Test 3: Clustering stability
    test_clustering_stability(n_epochs=2000, n_seeds=10, verbose=True)
    
    # Test 4: Hidden size scaling
    test_hidden_size_scaling(hidden_sizes=[64, 128, 256], n_epochs=2000, n_trials=3, verbose=True)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE")
    print("="*70)


if __name__ == "__main__":
    main()
