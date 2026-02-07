"""
Robustness test suite runner.
"""

import sys
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from experiments.synthetic.robustness_tests import (
    test_information_content,
    test_noise_robustness,
    test_clustering_stability,
    test_sufficient_statistics,
    test_independence,
    test_generalization_across_populations,
    test_noise_types,
    test_false_positives,
    test_alternative_deformation_methods,
    test_latent_estimation_noise,
    test_deformation_ablation
)


def main():
    """Run the complete robustness test suite."""
    print("\n" + "="*70)
    print("COMPREHENSIVE ROBUSTNESS TEST SUITE")
    print("="*70)
    
    # Original 5 tests
    test_information_content(n_trials=10)
    test_noise_robustness(n_trials=10)
    test_clustering_stability(n_seeds=30)
    test_sufficient_statistics(n_trials=10)
    test_independence(n_trials=10)
    
    # New generalization tests
    test_generalization_across_populations(n_trials=5)
    test_noise_types(n_trials=5)
    test_false_positives(n_trials=5)
    test_alternative_deformation_methods(n_trials=5)

    # New realism tests
    test_latent_estimation_noise(n_trials=5)
    test_deformation_ablation(n_trials=5)
    
    print("\n" + "="*70)
    print("ALL TESTS COMPLETE - Figures saved to: results/figures/")
    print("="*70)


if __name__ == "__main__":
    main()

