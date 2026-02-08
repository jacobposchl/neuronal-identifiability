"""
Analysis tools for neural networks and populations.

This module contains advanced analysis methods for understanding
neural network dynamics and properties.

Intended contents:
- RNN unit feature extraction
- Spectral analysis methods
- Perturbation and ablation tools
- Statistical testing utilities
"""

from .rnn_features import (
    extract_rnn_unit_features,
    extract_enhanced_rnn_features,
    classify_units,
    interpret_clusters,
    print_cluster_summary,
    compare_to_baseline,
    tdr_baseline,
    selectivity_baseline,
    select_features_by_task_dynamics,
    select_optimal_clusters
)
from .spectral_analysis import (
    compute_recurrent_eigenvalues,
    classify_eigenvalues,
    compare_eigenvalues_to_deformation,
    print_spectral_comparison,
    plot_eigenvalue_spectrum,
    compute_lyapunov_spectrum,
    detect_fixed_points,
    plot_spectral_summary
)
from .perturbation import (
    ablate_units,
    test_unit_importance,
    evaluate_rnn_performance,
    cross_task_transfer,
    progressive_ablation,
    confidence_guided_pruning,
    task_specific_importance
)
from .statistical_tests import (
    paired_ttest,
    bootstrap_ci,
    permutation_test,
    bonferroni_correction,
    fdr_correction,
    effect_size_interpretation,
    compare_methods_with_stats,
    format_significance_stars,
    print_comparison_table
)

__all__ = [
    'extract_rnn_unit_features',
    'extract_enhanced_rnn_features',
    'classify_units',
    'interpret_clusters',
    'print_cluster_summary',
    'compare_to_baseline',
    'tdr_baseline',
    'selectivity_baseline',
    'select_features_by_task_dynamics',
    'select_optimal_clusters',
    'compute_recurrent_eigenvalues',
    'classify_eigenvalues',
    'compare_eigenvalues_to_deformation',
    'print_spectral_comparison',
    'plot_eigenvalue_spectrum',
    'compute_lyapunov_spectrum',
    'detect_fixed_points',
    'plot_spectral_summary',
    'ablate_units',
    'test_unit_importance',
    'evaluate_rnn_performance',
    'cross_task_transfer',
    'progressive_ablation',
    'confidence_guided_pruning',
    'task_specific_importance',
    'paired_ttest',
    'bootstrap_ci',
    'permutation_test',
    'bonferroni_correction',
    'fdr_correction',
    'effect_size_interpretation',
    'compare_methods_with_stats',
    'format_significance_stars',
    'print_comparison_table',
]
