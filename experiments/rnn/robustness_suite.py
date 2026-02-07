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
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU
from src.tasks import FlipFlopTask, CyclingMemoryTask, ContextIntegrationTask, get_task
from src.core.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.analysis.rnn_features import extract_rnn_unit_features, classify_units, interpret_clusters, compare_to_baseline
from src.visualization import ensure_dirs, plot_bar, plot_comparison
from src.analysis.statistical_tests import compare_methods_with_stats, print_comparison_table


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


def test_statistical_significance(tasks=['flipflop', 'context'], architecture='vanilla',
                                   hidden_size=128, n_epochs=2000, n_seeds=10, verbose=True):
    """
    Test 5: Statistical significance of deformation method vs baselines.
    
    Runs each task multiple times with different random seeds and performs
    rigorous statistical comparisons using:
    - Paired t-tests with effect sizes (Cohen's d)
    - Bootstrap confidence intervals
    - Permutation tests (non-parametric)
    - Multiple comparison corrections (Bonferroni, FDR)
    
    Args:
        tasks: List of task names to test
        architecture: RNN architecture
        hidden_size: Number of hidden units
        n_epochs: Training epochs
        n_seeds: Number of random seeds (10+ recommended)
        verbose: Print progress
    
    Returns:
        results: Dict with statistical comparison results per task
    """
    ensure_dirs('results/rnn_figures')
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 5: STATISTICAL SIGNIFICANCE")
        print("="*70)
        print(f"Tasks: {tasks}")
        print(f"Architecture: {architecture}")
        print(f"Seeds: {n_seeds}")
        print(f"Multiple comparison correction: Bonferroni + FDR")
    
    all_results = {}
    
    for task_name in tasks:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*70}")
        
        # Collect scores across seeds
        method_scores = {
            'deformation': [],
            'pca': [],
            'raw': []
        }
        
        for seed in range(n_seeds):
            if verbose:
                print(f"\n  Seed {seed+1}/{n_seeds}...")
            
            # Set random seed
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            # Create task and RNN
            task = get_task(task_name)
            
            arch_map = {'vanilla': VanillaRNN, 'lstm': SimpleLSTM, 'gru': SimpleGRU}
            rnn = arch_map[architecture](task.input_size, hidden_size, task.output_size)
            
            # Train RNN
            rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                         trial_length=100, verbose=False)
            
            # Extract trajectories
            hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=50, trial_length=200)
            
            # Estimate deformation
            rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
            
            # Extract features
            features = extract_rnn_unit_features(hidden_states, rot, con, exp)
            
            # Cluster with deformation method
            labels_def, details_def = classify_units(features, n_clusters=4, return_details=True)
            method_scores['deformation'].append(details_def['silhouette'])
            
            # Compare to baselines
            baseline_results = compare_to_baseline(hidden_states, labels_def, n_clusters=4)
            method_scores['pca'].append(baseline_results['pca']['silhouette'])
            method_scores['raw'].append(baseline_results['raw']['silhouette'])
            
            if verbose:
                print(f"    Deformation: {details_def['silhouette']:.3f}, "
                      f"PCA: {baseline_results['pca']['silhouette']:.3f}, "
                      f"Raw: {baseline_results['raw']['silhouette']:.3f}")
        
        # Convert to numpy arrays
        for method in method_scores:
            method_scores[method] = np.array(method_scores[method])
        
        # Statistical comparison
        if verbose:
            print(f"\n  Running statistical tests...")
        
        stats_results = compare_methods_with_stats(
            method_scores,
            reference_method='deformation',
            alpha=0.05
        )
        
        all_results[task_name] = {
            'scores': method_scores,
            'statistics': stats_results
        }
        
        # Print comparison table
        if verbose:
            print_comparison_table(stats_results, show_ci=True)
    
    # Overall summary
    if verbose:
        print(f"\n{'='*70}")
        print("OVERALL STATISTICAL SUMMARY")
        print(f"{'='*70}")
        
        for task_name, task_results in all_results.items():
            print(f"\n{task_name.upper()}:")
            stats = task_results['statistics']
            
            for method, comparison in stats['comparisons'].items():
                par = comparison['parametric']
                p_val = par['p_value']
                cohens_d = par['cohens_d']
                effect = comparison['effect_size_interpretation']
                
                from src.analysis.statistical_tests import format_significance_stars
                stars = format_significance_stars(p_val)
                
                improvement = ((par['method1_mean'] - par['method2_mean']) / 
                              par['method2_mean'] * 100)
                
                print(f"  vs {method:12s}: +{improvement:5.1f}%  "
                      f"(d={cohens_d:5.2f}, {effect:10s})  "
                      f"p={p_val:.4f} {stars}")
        
        print(f"\n{'='*70}")
        print("✓ Statistical significance testing complete")
        print(f"{'='*70}")
    
    return all_results


def test_component_ablation(tasks=['flipflop', 'context'], architecture='vanilla',
                             hidden_size=128, n_epochs=2000, n_trials=3, verbose=True):
    """
    Test 6: Component ablation study.
    
    Tests which deformation component (rotation, contraction, expansion) drives
    clustering performance for different tasks.
    
    Hypothesis:
    - FlipFlop (attractors) → Contraction-dominant
    - Cycling (limit cycle) → Rotation-dominant
    - Context (mixed) → All components useful
    
    Args:
        tasks: List of task names
        architecture: RNN architecture
        hidden_size: Number of hidden units
        n_epochs: Training epochs
        n_trials: Number of trials per ablation condition
        verbose: Print progress
    
    Returns:
        results: Dict mapping task → ablation condition → scores
    """
    ensure_dirs('results/rnn_figures')
    
    if verbose:
        print("\n" + "="*70)
        print("TEST 6: COMPONENT ABLATION STUDY")
        print("="*70)
        print(f"Tasks: {tasks}")
        print(f"Ablation conditions: rotation_only, contraction_only, expansion_only, all")
    
    all_results = {}
    
    for task_name in tasks:
        if verbose:
            print(f"\n{'='*70}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*70}")
        
        task_results = {
            'rotation_only': [],
            'contraction_only': [],
            'expansion_only': [],
            'all': []
        }
        
        for trial in range(n_trials):
            if verbose:
                print(f"\n  Trial {trial+1}/{n_trials}...")
            
            # Create task and RNN
            task = get_task(task_name)
            arch_map = {'vanilla': VanillaRNN, 'lstm': SimpleLSTM, 'gru': SimpleGRU}
            rnn = arch_map[architecture](task.input_size, hidden_size, task.output_size)
            
            # Train RNN
            if verbose:
                print(f"    Training...")
            rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                         trial_length=100, verbose=False)
            
            # Extract trajectories
            if verbose:
                print(f"    Extracting trajectories...")
            hidden_states, _, _ = task.extract_trajectories(rnn, n_trials=50, trial_length=200)
            
            # Estimate deformation
            if verbose:
                print(f"    Estimating deformation...")
            rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
            
            # Test each ablation condition
            for condition in ['rotation_only', 'contraction_only', 'expansion_only', 'all']:
                # Create ablated features
                if condition == 'rotation_only':
                    features = extract_rnn_unit_features(hidden_states, rot, 
                                                        np.zeros_like(con), np.zeros_like(exp))
                elif condition == 'contraction_only':
                    features = extract_rnn_unit_features(hidden_states, np.zeros_like(rot), 
                                                        con, np.zeros_like(exp))
                elif condition == 'expansion_only':
                    features = extract_rnn_unit_features(hidden_states, np.zeros_like(rot), 
                                                        np.zeros_like(con), exp)
                else:  # 'all'
                    features = extract_rnn_unit_features(hidden_states, rot, con, exp)
                
                # Cluster
                labels, details = classify_units(features, n_clusters=4, return_details=True)
                task_results[condition].append(details['silhouette'])
            
            if verbose:
                print(f"    R: {task_results['rotation_only'][-1]:.3f}, "
                      f"C: {task_results['contraction_only'][-1]:.3f}, "
                      f"E: {task_results['expansion_only'][-1]:.3f}, "
                      f"All: {task_results['all'][-1]:.3f}")
        
        # Compute statistics
        for condition in task_results:
            scores = np.array(task_results[condition])
            task_results[condition] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'scores': scores
            }
        
        all_results[task_name] = task_results
        
        # Print summary
        if verbose:
            print(f"\n  Results for {task_name}:")
            print(f"    Rotation only:    {task_results['rotation_only']['mean']:.3f} ± {task_results['rotation_only']['std']:.3f}")
            print(f"    Contraction only: {task_results['contraction_only']['mean']:.3f} ± {task_results['contraction_only']['std']:.3f}")
            print(f"    Expansion only:   {task_results['expansion_only']['mean']:.3f} ± {task_results['expansion_only']['std']:.3f}")
            print(f"    All components:   {task_results['all']['mean']:.3f} ± {task_results['all']['std']:.3f}")
            
            # Determine dominant component
            component_means = {
                'Rotation': task_results['rotation_only']['mean'],
                'Contraction': task_results['contraction_only']['mean'],
                'Expansion': task_results['expansion_only']['mean']
            }
            dominant = max(component_means, key=component_means.get)
            print(f"    → Dominant component: {dominant}")
    
    # Overall summary
    if verbose:
        print(f"\n{'='*70}")
        print("ABLATION STUDY SUMMARY")
        print(f"{'='*70}\n")
        
        for task_name, task_results in all_results.items():
            print(f"{task_name.upper()}:")
            
            # Find best single component
            single_components = {
                'Rotation': task_results['rotation_only']['mean'],
                'Contraction': task_results['contraction_only']['mean'],
                'Expansion': task_results['expansion_only']['mean']
            }
            best_component = max(single_components, key=single_components.get)
            best_score = single_components[best_component]
            all_score = task_results['all']['mean']
            
            print(f"  Best single component: {best_component} ({best_score:.3f})")
            print(f"  All components: {all_score:.3f}")
            
            if all_score > best_score:
                improvement = ((all_score - best_score) / best_score) * 100
                print(f"  → Combining components improves by {improvement:.1f}%")
            else:
                print(f"  → Single component sufficient")
            print()
        
        print(f"{'='*70}")
        print("✓ Component ablation study complete")
        print(f"{'='*70}")
    
    return all_results


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
