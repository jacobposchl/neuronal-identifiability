"""
Statistical significance testing utilities for RNN deformation experiments.

Provides robust statistical methods for comparing clustering methods across
multiple runs, including:
- Paired t-tests with effect sizes
- Bootstrap confidence intervals
- Permutation tests (non-parametric)
- Multiple comparison corrections (Bonferroni, FDR)
"""

import numpy as np
from scipy import stats
from typing import List, Dict, Tuple, Optional


def paired_ttest(method1_scores: np.ndarray, method2_scores: np.ndarray,
                 alternative: str = 'greater') -> Dict[str, float]:
    """
    Perform paired t-test comparing two methods across multiple runs.
    
    Args:
        method1_scores: Array of scores for method 1 (e.g., deformation)
        method2_scores: Array of scores for method 2 (e.g., PCA baseline)
        alternative: 'greater' (method1 > method2), 'less', or 'two-sided'
    
    Returns:
        results: Dict with 't_statistic', 'p_value', 'cohens_d', 'mean_diff'
    """
    # Convert to numpy arrays
    scores1 = np.asarray(method1_scores)
    scores2 = np.asarray(method2_scores)
    
    # Paired t-test
    t_stat, p_value = stats.ttest_rel(scores1, scores2, alternative=alternative)
    
    # Effect size: Cohen's d for paired samples
    differences = scores1 - scores2
    cohens_d = np.mean(differences) / (np.std(differences, ddof=1) + 1e-10)
    
    # Mean difference
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)
    
    return {
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'mean_diff': float(mean_diff),
        'std_diff': float(std_diff),
        'method1_mean': float(np.mean(scores1)),
        'method2_mean': float(np.mean(scores2)),
        'method1_std': float(np.std(scores1, ddof=1)),
        'method2_std': float(np.std(scores2, ddof=1))
    }


def bootstrap_ci(scores: np.ndarray, confidence_level: float = 0.95,
                 n_iterations: int = 10000, statistic: str = 'mean') -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a statistic.
    
    Args:
        scores: Array of scores
        confidence_level: Confidence level (default: 0.95 for 95% CI)
        n_iterations: Number of bootstrap samples
        statistic: 'mean', 'median', or 'std'
    
    Returns:
        (lower_bound, point_estimate, upper_bound)
    """
    scores = np.asarray(scores)
    
    # Choose statistic function
    if statistic == 'mean':
        stat_func = np.mean
    elif statistic == 'median':
        stat_func = np.median
    elif statistic == 'std':
        stat_func = lambda x: np.std(x, ddof=1)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Point estimate
    point_estimate = stat_func(scores)
    
    # Bootstrap resampling
    n = len(scores)
    bootstrap_stats = np.zeros(n_iterations)
    
    np.random.seed(42)  # Reproducibility
    for i in range(n_iterations):
        # Resample with replacement
        sample = np.random.choice(scores, size=n, replace=True)
        bootstrap_stats[i] = stat_func(sample)
    
    # Compute percentile confidence interval
    alpha = 1 - confidence_level
    lower_percentile = 100 * (alpha / 2)
    upper_percentile = 100 * (1 - alpha / 2)
    
    lower_bound = np.percentile(bootstrap_stats, lower_percentile)
    upper_bound = np.percentile(bootstrap_stats, upper_percentile)
    
    return lower_bound, point_estimate, upper_bound


def permutation_test(group1: np.ndarray, group2: np.ndarray,
                     n_permutations: int = 10000,
                     statistic: str = 'mean_diff') -> Dict[str, float]:
    """
    Non-parametric permutation test for comparing two groups.
    
    Args:
        group1: Scores for group 1
        group2: Scores for group 2
        n_permutations: Number of random permutations
        statistic: 'mean_diff' or 'median_diff'
    
    Returns:
        results: Dict with 'observed_stat', 'p_value', 'permutation_stats'
    """
    group1 = np.asarray(group1)
    group2 = np.asarray(group2)
    
    # Choose statistic
    if statistic == 'mean_diff':
        stat_func = lambda g1, g2: np.mean(g1) - np.mean(g2)
    elif statistic == 'median_diff':
        stat_func = lambda g1, g2: np.median(g1) - np.median(g2)
    else:
        raise ValueError(f"Unknown statistic: {statistic}")
    
    # Observed statistic
    observed_stat = stat_func(group1, group2)
    
    # Combine groups
    combined = np.concatenate([group1, group2])
    n1 = len(group1)
    n = len(combined)
    
    # Permutation distribution
    permutation_stats = np.zeros(n_permutations)
    
    np.random.seed(42)
    for i in range(n_permutations):
        # Random permutation
        permuted = np.random.permutation(combined)
        perm_g1 = permuted[:n1]
        perm_g2 = permuted[n1:]
        
        permutation_stats[i] = stat_func(perm_g1, perm_g2)
    
    # Two-sided p-value
    p_value = np.mean(np.abs(permutation_stats) >= np.abs(observed_stat))
    
    return {
        'observed_stat': float(observed_stat),
        'p_value': float(p_value),
        'permutation_mean': float(np.mean(permutation_stats)),
        'permutation_std': float(np.std(permutation_stats))
    }


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, any]:
    """
    Apply Bonferroni correction for multiple comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Family-wise error rate (default: 0.05)
    
    Returns:
        results: Dict with 'corrected_alpha', 'significant', 'adjusted_p_values'
    """
    p_values = np.asarray(p_values)
    n_comparisons = len(p_values)
    
    # Corrected significance threshold
    corrected_alpha = alpha / n_comparisons
    
    # Adjusted p-values (multiply by number of comparisons, clip at 1.0)
    adjusted_p_values = np.minimum(p_values * n_comparisons, 1.0)
    
    # Determine significance
    significant = adjusted_p_values < alpha
    
    return {
        'corrected_alpha': corrected_alpha,
        'n_comparisons': n_comparisons,
        'significant': significant.tolist(),
        'adjusted_p_values': adjusted_p_values.tolist(),
        'n_significant': int(np.sum(significant))
    }


def fdr_correction(p_values: List[float], alpha: float = 0.05, method: str = 'bh') -> Dict[str, any]:
    """
    Apply False Discovery Rate (FDR) correction for multiple comparisons.
    
    Less conservative than Bonferroni for many comparisons.
    
    Args:
        p_values: List of p-values from multiple tests
        alpha: Desired FDR level (default: 0.05)
        method: 'bh' (Benjamini-Hochberg) or 'by' (Benjamini-Yekutieli)
    
    Returns:
        results: Dict with 'significant', 'adjusted_p_values', 'rejection_threshold'
    """
    from statsmodels.stats.multitest import multipletests
    
    p_values = np.asarray(p_values)
    
    # Apply FDR correction
    if method == 'bh':
        significant, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
    elif method == 'by':
        significant, adjusted_p, _, _ = multipletests(p_values, alpha=alpha, method='fdr_by')
    else:
        raise ValueError(f"Unknown method: {method}. Use 'bh' or 'by'")
    
    return {
        'significant': significant.tolist(),
        'adjusted_p_values': adjusted_p.tolist(),
        'n_significant': int(np.sum(significant)),
        'method': method
    }


def effect_size_interpretation(cohens_d: float) -> str:
    """
    Interpret Cohen's d effect size magnitude.
    
    Args:
        cohens_d: Cohen's d value
    
    Returns:
        interpretation: String description
    """
    abs_d = abs(cohens_d)
    
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def compare_methods_with_stats(method_scores: Dict[str, np.ndarray],
                                reference_method: str = 'deformation',
                                alpha: float = 0.05) -> Dict[str, any]:
    """
    Comprehensive statistical comparison of multiple methods against a reference.
    
    Args:
        method_scores: Dict mapping method names to arrays of scores
        reference_method: Name of reference method (typically 'deformation')
        alpha: Significance level
    
    Returns:
        results: Dict with pairwise comparisons and multiple comparison corrections
    """
    if reference_method not in method_scores:
        raise ValueError(f"Reference method '{reference_method}' not in method_scores")
    
    ref_scores = method_scores[reference_method]
    other_methods = [m for m in method_scores.keys() if m != reference_method]
    
    # Pairwise comparisons
    comparisons = {}
    p_values = []
    
    for method in other_methods:
        # Paired t-test
        ttest_results = paired_ttest(ref_scores, method_scores[method], alternative='greater')
        
        # Permutation test (non-parametric verification)
        perm_results = permutation_test(ref_scores, method_scores[method])
        
        # Bootstrap CIs
        ref_ci = bootstrap_ci(ref_scores)
        method_ci = bootstrap_ci(method_scores[method])
        
        comparisons[method] = {
            'parametric': ttest_results,
            'nonparametric': perm_results,
            'reference_ci': ref_ci,
            'method_ci': method_ci,
            'effect_size_interpretation': effect_size_interpretation(ttest_results['cohens_d'])
        }
        
        p_values.append(ttest_results['p_value'])
    
    # Multiple comparison correction
    bonferroni_results = bonferroni_correction(p_values, alpha=alpha)
    
    # Try FDR correction (requires statsmodels)
    try:
        fdr_results = fdr_correction(p_values, alpha=alpha)
    except ImportError:
        fdr_results = None
    
    return {
        'reference_method': reference_method,
        'comparisons': comparisons,
        'bonferroni': bonferroni_results,
        'fdr': fdr_results,
        'n_methods': len(other_methods)
    }


def format_significance_stars(p_value: float) -> str:
    """
    Convert p-value to significance stars for publication.
    
    Args:
        p_value: P-value from statistical test
    
    Returns:
        stars: '***' (p<0.001), '**' (p<0.01), '*' (p<0.05), 'ns' (not significant)
    """
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def print_comparison_table(results: Dict[str, any], show_ci: bool = True):
    """
    Print formatted table of method comparisons.
    
    Args:
        results: Output from compare_methods_with_stats()
        show_ci: Whether to show bootstrap confidence intervals
    """
    print("\n" + "="*80)
    print("STATISTICAL COMPARISON OF METHODS")
    print("="*80)
    print(f"Reference method: {results['reference_method']}")
    print(f"Comparisons: {results['n_methods']}")
    print()
    
    ref_method = results['reference_method']
    
    # Header
    if show_ci:
        print(f"{'Method':<20} {'Mean ± SD':<20} {'95% CI':<25} {'t-stat':<10} {'p-value':<12} {'Effect':<10} {'Sig':<5}")
        print("-"*105)
    else:
        print(f"{'Method':<20} {'Mean ± SD':<20} {'t-stat':<10} {'p-value':<12} {'Effect':<10} {'Sig':<5}")
        print("-"*80)
    
    # Reference method row
    ref_comp = list(results['comparisons'].values())[0]
    ref_mean = ref_comp['parametric']['method1_mean']
    ref_std = ref_comp['parametric']['method1_std']
    ref_ci = ref_comp['reference_ci']
    
    if show_ci:
        print(f"{ref_method:<20} {ref_mean:.3f} ± {ref_std:.3f}     [{ref_ci[0]:.3f}, {ref_ci[2]:.3f}]         {'—':<10} {'—':<12} {'—':<10} {'—':<5}")
    else:
        print(f"{ref_method:<20} {ref_mean:.3f} ± {ref_std:.3f}     {'—':<10} {'—':<12} {'—':<10} {'—':<5}")
    
    # Other methods
    for i, (method, comp) in enumerate(results['comparisons'].items()):
        par = comp['parametric']
        ci = comp['method_ci']
        
        stars = format_significance_stars(par['p_value'])
        effect = comp['effect_size_interpretation']
        
        if show_ci:
            print(f"{method:<20} {par['method2_mean']:.3f} ± {par['method2_std']:.3f}     "
                  f"[{ci[0]:.3f}, {ci[2]:.3f}]    "
                  f"{par['t_statistic']:>8.2f}  {par['p_value']:>10.4f}  {effect:<10} {stars:<5}")
        else:
            print(f"{method:<20} {par['method2_mean']:.3f} ± {par['method2_std']:.3f}     "
                  f"{par['t_statistic']:>8.2f}  {par['p_value']:>10.4f}  {effect:<10} {stars:<5}")
    
    # Multiple comparison correction
    print()
    print("Multiple comparison corrections:")
    print(f"  Bonferroni-corrected α: {results['bonferroni']['corrected_alpha']:.4f}")
    print(f"  Significant after correction: {results['bonferroni']['n_significant']}/{results['n_methods']}")
    
    if results['fdr'] is not None:
        print(f"  FDR-corrected (BH): {results['fdr']['n_significant']}/{results['n_methods']}")
    
    print("="*80)
    print("Significance: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
    print("Effect size: Cohen's d interpretation (negligible/small/medium/large)")
    print("="*80)
