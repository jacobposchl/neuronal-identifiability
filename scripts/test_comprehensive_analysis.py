"""
Comprehensive Statistical & Advanced Analysis Script

Consolidates all publication-quality validation tests:
1. Statistical significance testing (paired t-tests, bootstrap CI, permutation)
2. Advanced baseline comparison (TDR, selectivity-based methods)
3. Component ablation study (rotation/contraction/expansion contributions)

Runs efficiently by reusing trained models across tests.
"""

import sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import silhouette_score

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.rnn_models import VanillaRNN
from src.tasks import get_task
from src.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.rnn_features import extract_rnn_unit_features, classify_units, compare_to_baseline
from src.statistical_tests import (
    paired_ttest, bootstrap_ci, permutation_test,
    bonferroni_correction, fdr_correction,
    compare_methods_with_stats, print_comparison_table
)
from src.visualization import ensure_dirs


def run_comprehensive_analysis(tasks=['flipflop', 'cycling'], n_seeds=10,
                                hidden_size=128, n_epochs=2000, n_trials=30,
                                verbose=True):
    """
    Run all statistical and advanced analysis tests efficiently.
    
    Args:
        tasks: List of tasks to test
        n_seeds: Number of random seeds for statistical validation
        hidden_size: RNN hidden size
        n_epochs: Training epochs per model
        n_trials: Number of trials for feature extraction
        verbose: Print progress
    
    Returns:
        results: Dict containing all test results
    """
    ensure_dirs('results/rnn_figures')
    
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE STATISTICAL & ADVANCED ANALYSIS")
        print("="*80)
        print(f"\nConfiguration:")
        print(f"  Tasks: {tasks}")
        print(f"  Random seeds: {n_seeds}")
        print(f"  Hidden size: {hidden_size}")
        print(f"  Training epochs: {n_epochs}")
        print(f"  Trials per test: {n_trials}")
    
    all_results = {}
    
    # =================================================================
    # PART 1: STATISTICAL SIGNIFICANCE TESTING
    # =================================================================
    if verbose:
        print("\n" + "="*80)
        print("PART 1: STATISTICAL SIGNIFICANCE TESTING")
        print("="*80)
    
    for task_name in tasks:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*80}")
        
        task = get_task(task_name)
        
        # Collect scores across seeds
        scores = {
            'deformation': [],
            'pca': [],
            'raw': []
        }
        
        for seed in range(n_seeds):
            if verbose:
                print(f"\nSeed {seed+1}/{n_seeds}")
            
            # Set seed
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            # Train RNN
            rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
            rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                         trial_length=100, verbose=False)
            accuracy = history['accuracy'][-1]
            
            if verbose:
                print(f"  Training accuracy: {accuracy:.2%}")
            
            # Extract trajectories
            hidden_states, inputs, targets = task.extract_trajectories(
                rnn, n_trials=n_trials, trial_length=200
            )
            
            # Compute deformation
            rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
            
            # Extract features and cluster
            features = extract_rnn_unit_features(hidden_states, rot, con, exp)
            labels, _ = classify_units(features, n_clusters=4, return_details=True)
            
            # Compare to baselines
            baseline_results = compare_to_baseline(
                features, labels, hidden_states,
                trial_indices=targets if hasattr(task, 'context_index') else None
            )
            
            # Store silhouette scores
            for method_name, result in baseline_results.items():
                if method_name in scores:
                    scores[method_name].append(result['silhouette'])
            
            if verbose:
                print(f"  Deformation: {scores['deformation'][-1]:.3f}")
                print(f"  PCA:         {scores['pca'][-1]:.3f}")
                print(f"  Raw:         {scores['raw'][-1]:.3f}")
        
        # Statistical tests
        if verbose:
            print(f"\n{'='*80}")
            print(f"STATISTICAL ANALYSIS: {task_name.upper()}")
            print(f"{'='*80}")
        
        # Paired t-test: Deformation vs PCA
        ttest_result = paired_ttest(
            scores['deformation'], scores['pca']
        )
        t_stat = ttest_result['t_statistic']
        p_value = ttest_result['p_value']
        effect_size = ttest_result['cohens_d']
        
        if verbose:
            print(f"\nPaired t-test (Deformation vs PCA):")
            print(f"  t-statistic: {t_stat:.3f}")
            print(f"  p-value: {p_value:.4f}")
            print(f"  Cohen's d: {effect_size:.3f}")
        
        # Bootstrap confidence intervals
        deformation_ci = bootstrap_ci(scores['deformation'], n_bootstrap=10000)
        pca_ci = bootstrap_ci(scores['pca'], n_bootstrap=10000)
        
        if verbose:
            print(f"\nBootstrap 95% Confidence Intervals:")
            print(f"  Deformation: [{deformation_ci[0]:.3f}, {deformation_ci[2]:.3f}]")
            print(f"  PCA:         [{pca_ci[0]:.3f}, {pca_ci[2]:.3f}]")
        
        # Permutation test
        perm_result = permutation_test(
            scores['deformation'], scores['pca'], n_permutations=10000
        )
        perm_p_value = perm_result['p_value']
        
        if verbose:
            print(f"\nPermutation test p-value: {perm_p_value:.4f}")
        
        # Multiple comparison correction
        p_values = [p_value, perm_p_value]
        bonf_corrected = bonferroni_correction(p_values)
        fdr_corrected = fdr_correction(p_values)
        
        if verbose:
            print(f"\nMultiple comparison corrections:")
            print(f"  Original p-value:  {p_value:.4f}")
            print(f"  Bonferroni:        {bonf_corrected[0]:.4f}")
            print(f"  FDR (BH):          {fdr_corrected[0]:.4f}")
        
        # Comparison table
        if verbose:
            print(f"\n{'='*80}")
            # Use compare_methods_with_stats for proper table formatting
            method_scores = {
                'deformation': np.array(scores['deformation']),
                'pca': np.array(scores['pca']),
                'raw': np.array(scores['raw'])
            }
            comparison_results = compare_methods_with_stats(
                method_scores, reference_method='deformation', alpha=0.05
            )
            print_comparison_table(comparison_results, show_ci=True)
        
        # Store results
        all_results[f'{task_name}_statistics'] = {
            'scores': scores,
            't_stat': t_stat,
            'p_value': p_value,
            'effect_size': effect_size,
            'ci_deformation': deformation_ci,
            'ci_pca': pca_ci,
            'perm_p_value': perm_p_value,
            'bonferroni': bonf_corrected[0],
            'fdr': fdr_corrected[0]
        }
    
    # =================================================================
    # PART 2: ADVANCED BASELINE COMPARISON
    # =================================================================
    if verbose:
        print("\n" + "="*80)
        print("PART 2: ADVANCED BASELINE COMPARISON")
        print("="*80)
    
    for task_name in tasks:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*80}")
        
        task = get_task(task_name)
        
        # Train single model for baseline comparison
        np.random.seed(42)
        torch.manual_seed(42)
        
        rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
        rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                     trial_length=100, verbose=verbose)
        
        # Extract trajectories
        hidden_states, inputs, targets = task.extract_trajectories(
            rnn, n_trials=n_trials, trial_length=200
        )
        
        # Compute deformation
        rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
        rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
        
        # Extract features and cluster
        features = extract_rnn_unit_features(hidden_states, rot, con, exp)
        labels, _ = classify_units(features, n_clusters=4, return_details=True)
        
        # Compare to ALL baselines (including TDR and selectivity)
        baseline_results = compare_to_baseline(
            features, labels, hidden_states,
            trial_indices=targets if hasattr(task, 'context_index') else None
        )
        
        if verbose:
            print(f"\n{'='*80}")
            print("BASELINE COMPARISON RESULTS")
            print(f"{'='*80}")
            print(f"\n{'Method':<20} {'Silhouette':<12} {'vs Deformation':<15}")
            print("-" * 50)
        
        deformation_score = baseline_results['deformation']['silhouette']
        
        for method_name, result in baseline_results.items():
            score = result['silhouette']
            diff = score - deformation_score
            symbol = "✓" if method_name == 'deformation' else "↓" if diff < 0 else "↑"
            
            if verbose:
                print(f"{method_name:<20} {score:>6.3f}       {symbol} {abs(diff):>6.3f}")
        
        all_results[f'{task_name}_baselines'] = baseline_results
    
    # =================================================================
    # PART 3: COMPONENT ABLATION STUDY
    # =================================================================
    if verbose:
        print("\n" + "="*80)
        print("PART 3: COMPONENT ABLATION STUDY")
        print("="*80)
    
    ablation_summary = {}
    
    for task_name in tasks:
        if verbose:
            print(f"\n{'='*80}")
            print(f"Task: {task_name.upper()}")
            print(f"{'='*80}")
        
        task = get_task(task_name)
        
        # Test each component configuration
        conditions = {
            'rotation_only': [True, False, False],
            'contraction_only': [False, True, False],
            'expansion_only': [False, False, True],
            'all': [True, True, True]
        }
        
        condition_scores = {name: [] for name in conditions}
        
        # Run multiple trials for statistics
        for trial_idx in range(20):  # Reduced from n_trials for speed
            if verbose and trial_idx % 5 == 0:
                print(f"\nTrial {trial_idx+1}/20")
            
            # Train RNN (use offset seed to avoid overlap with Part 1)
            seed = trial_idx + 1000
            np.random.seed(seed)
            torch.manual_seed(seed)
            
            rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
            rnn, _ = task.train_rnn(rnn, n_epochs=n_epochs, lr=0.001,
                                   trial_length=100, verbose=False)
            
            # Extract trajectories
            hidden_states, _, _ = task.extract_trajectories(
                rnn, n_trials=30, trial_length=200
            )
            
            # Compute deformation
            rot, con, exp, latent = estimate_deformation_from_rnn(hidden_states)
            rot, con, exp = smooth_deformation_signals(rot, con, exp, sigma=5)
            
            # Test each condition
            for condition_name, (use_rot, use_con, use_exp) in conditions.items():
                # Build features using proper correlation computation per unit
                from scipy.ndimage import gaussian_filter1d
                
                # Smooth deformation signals
                rot_smooth = gaussian_filter1d(rot, sigma=5) if use_rot else None
                con_smooth = gaussian_filter1d(con, sigma=5) if use_con else None
                exp_smooth = gaussian_filter1d(exp, sigma=5) if use_exp else None
                
                # Extract correlations for each unit
                n_units = hidden_states.shape[0]
                feature_list = []
                
                for unit_idx in range(n_units):
                    unit_activity = hidden_states[unit_idx, :]
                    unit_features = []
                    
                    if use_rot and rot_smooth is not None:
                        rot_corr = np.corrcoef(unit_activity, rot_smooth)[0, 1]
                        unit_features.append(rot_corr)
                    if use_con and con_smooth is not None:
                        con_corr = np.corrcoef(unit_activity, con_smooth)[0, 1]
                        unit_features.append(con_corr)
                    if use_exp and exp_smooth is not None:
                        exp_corr = np.corrcoef(unit_activity, exp_smooth)[0, 1]
                        unit_features.append(exp_corr)
                    
                    if len(unit_features) > 0:
                        feature_list.append(unit_features)
                
                if len(feature_list) == 0:
                    continue
                
                features = np.array(feature_list)  # Shape: (n_units, n_features)
                
                # Cluster
                labels, _ = classify_units(features, n_clusters=4, return_details=True)
                
                # Compute silhouette
                silhouette = silhouette_score(features, labels)
                condition_scores[condition_name].append(silhouette)
        
        # Compute statistics
        if verbose:
            print(f"\n{'='*80}")
            print("COMPONENT ABLATION RESULTS")
            print(f"{'='*80}")
            print(f"\n{'Component':<20} {'Mean ± Std':<20} {'Best?':<10}")
            print("-" * 50)
        
        mean_scores = {name: np.mean(scores) for name, scores in condition_scores.items()}
        best_component = max(mean_scores, key=mean_scores.get)
        
        for condition_name in ['rotation_only', 'contraction_only', 'expansion_only', 'all']:
            scores = condition_scores[condition_name]
            mean = np.mean(scores)
            std = np.std(scores)
            is_best = "✓" if condition_name == best_component else ""
            
            if verbose:
                print(f"{condition_name:<20} {mean:.3f} ± {std:.3f}      {is_best}")
        
        ablation_summary[task_name] = {
            'scores': condition_scores,
            'means': mean_scores,
            'dominant_component': best_component
        }
        
        all_results[f'{task_name}_ablation'] = ablation_summary[task_name]
    
    # =================================================================
    # FINAL SUMMARY
    # =================================================================
    if verbose:
        print("\n" + "="*80)
        print("COMPREHENSIVE ANALYSIS COMPLETE")
        print("="*80)
        print("\nKey Findings:")
        
        for task_name in tasks:
            print(f"\n{task_name.upper()}:")
            
            # Statistical significance
            stats = all_results[f'{task_name}_statistics']
            print(f"  Statistical: p={stats['p_value']:.4f}, d={stats['effect_size']:.2f}")
            
            # Best baseline
            baselines = all_results[f'{task_name}_baselines']
            deform_score = baselines['deformation']['silhouette']
            print(f"  Deformation score: {deform_score:.3f}")
            
            # Dominant component
            ablation = all_results[f'{task_name}_ablation']
            dominant = ablation['dominant_component']
            print(f"  Dominant component: {dominant}")
        
        print("\n" + "="*80)
    
    return all_results


def main():
    """Run comprehensive analysis with publication settings."""
    results = run_comprehensive_analysis(
        tasks=['flipflop', 'cycling'],
        n_seeds=10,
        hidden_size=128,
        n_epochs=2000,
        n_trials=30,
        verbose=True
    )
    
    print("\n✓ All analyses complete. Results ready for publication.")


if __name__ == "__main__":
    main()
