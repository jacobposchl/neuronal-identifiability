"""
Evaluation and testing framework for neuron identification methods
"""

import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from scipy.stats import ttest_rel

from .population import RealisticNeuralPopulation
from .dynamics import generate_complex_dynamics
from .features import (extract_pca_features, extract_crosscorr_features, 
                     extract_dimensionality_features, extract_deformation_features)


def test_single_condition(n_neurons, specialization, T, noise_label, seed):
    """
    Run one test condition
    
    Args:
        n_neurons: Number of neurons to simulate
        specialization: 'low', 'medium', or 'high' specialization level
        T: Total simulation time
        noise_label: Label for noise condition (unused but kept for compatibility)
        seed: Random seed for reproducibility
        
    Returns:
        Tuple of (results_dict, true_labels) where results_dict contains
        clustering scores for each method
    """
    np.random.seed(seed)
    
    # Generate population
    population = RealisticNeuralPopulation(
        n_neurons=n_neurons,
        specialization_level=specialization
    )
    
    # Generate dynamics
    (latent_trajectory, rotation_trajectory, contraction_trajectory,
     expansion_trajectory, t_eval) = generate_complex_dynamics(T=T, dt=0.002)
    
    # Generate spikes
    spike_trains, firing_rates = population.generate_spike_trains(
        latent_trajectory, rotation_trajectory, contraction_trajectory,
        expansion_trajectory, t_eval, dt=0.001
    )
    
    # Extract all features
    feat_pca = extract_pca_features(spike_trains, firing_rates, n_components=5)
    feat_xcorr = extract_crosscorr_features(spike_trains, firing_rates, n_lags=100)
    feat_dim = extract_dimensionality_features(spike_trains, firing_rates)
    feat_deform = extract_deformation_features(spike_trains, firing_rates, 
                                               latent_trajectory, dt=0.001,
                                               rotation_trajectory=rotation_trajectory,
                                               contraction_trajectory=contraction_trajectory,
                                               expansion_trajectory=expansion_trajectory)
    
    # Cluster with each method
    n_clusters = 4
    scaler = StandardScaler()
    
    results = {}
    
    for name, features in [('PCA', feat_pca), ('XCorr', feat_xcorr), 
                          ('Dimensionality', feat_dim), ('Deformation', feat_deform)]:
        feat_norm = scaler.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=seed, n_init=10)
        pred = kmeans.fit_predict(feat_norm)
        ari = adjusted_rand_score(population.true_labels, pred)
        nmi = normalized_mutual_info_score(population.true_labels, pred)
        
        results[name] = {'ari': ari, 'nmi': nmi, 'predictions': pred}
    
    return results, population.true_labels


def run_comprehensive_test():
    """
    Test across multiple conditions:
    - Different specialization levels (how pure functional types are)
    - Different numbers of neurons
    - Multiple random seeds for statistical reliability
    
    Runs comprehensive evaluation with visualization and statistical analysis.
    """
    
    print("="*70)
    print("COMPREHENSIVE NEURON IDENTIFICATION TEST")
    print("="*70)
    
    # Test conditions
    specializations = ['low', 'medium', 'high']
    n_neurons_list = [50, 70]
    n_seeds = 3
    
    all_results = {spec: {n: [] for n in n_neurons_list} 
                  for spec in specializations}
    
    total_tests = len(specializations) * len(n_neurons_list) * n_seeds
    test_count = 0
    
    # Run all tests
    for specialization in specializations:
        print(f"\n{'='*70}")
        print(f"Testing Specialization Level: {specialization.upper()}")
        print('='*70)
        
        for n_neurons in n_neurons_list:
            print(f"\n  Testing with {n_neurons} neurons...")
            
            for seed in range(n_seeds):
                test_count += 1
                print(f"    Trial {seed+1}/{n_seeds}... ({test_count}/{total_tests} total)", end='\r')
                
                results, true_labels = test_single_condition(
                    n_neurons=n_neurons,
                    specialization=specialization,
                    T=15.0,
                    noise_label='standard',
                    seed=seed
                )
                
                all_results[specialization][n_neurons].append(results)
    
    # Aggregate and display results
    print("\n\n" + "="*70)
    print("AGGREGATED RESULTS")
    print("="*70)
    
    methods = ['PCA', 'XCorr', 'Dimensionality', 'Deformation']
    
    for specialization in specializations:
        print(f"\n{'='*70}")
        print(f"Specialization: {specialization.upper()}")
        print('='*70)
        
        for n_neurons in n_neurons_list:
            print(f"\n  {n_neurons} neurons:")
            
            trials = all_results[specialization][n_neurons]
            
            for method in methods:
                aris = [trial[method]['ari'] for trial in trials]
                nmis = [trial[method]['nmi'] for trial in trials]
                
                mean_ari = np.mean(aris)
                std_ari = np.std(aris)
                mean_nmi = np.mean(nmis)
                std_nmi = np.std(nmis)
                
                print(f"    {method:20s} ARI: {mean_ari:.3f} ± {std_ari:.3f}  "
                      f"NMI: {mean_nmi:.3f} ± {std_nmi:.3f}")
    
    # Statistical comparison
    print("\n" + "="*70)
    print("STATISTICAL COMPARISON (Deformation vs Best Baseline)")
    print("="*70)
    
    for specialization in specializations:
        print(f"\n{specialization.upper()} specialization:")
        
        for n_neurons in n_neurons_list:
            trials = all_results[specialization][n_neurons]
            
            # Get ARI scores
            deform_aris = [trial['Deformation']['ari'] for trial in trials]
            pca_aris = [trial['PCA']['ari'] for trial in trials]
            xcorr_aris = [trial['XCorr']['ari'] for trial in trials]
            dim_aris = [trial['Dimensionality']['ari'] for trial in trials]
            
            # Find best baseline for each trial
            best_baseline_aris = []
            for i in range(len(trials)):
                best_baseline_aris.append(max(pca_aris[i], xcorr_aris[i], dim_aris[i]))
            
            # Paired t-test
            if len(deform_aris) > 1:
                t_stat, p_value = ttest_rel(deform_aris, best_baseline_aris)
                
                mean_deform = np.mean(deform_aris)
                mean_baseline = np.mean(best_baseline_aris)
                improvement = (mean_deform - mean_baseline) / (mean_baseline + 1e-10) * 100
                
                sig = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "n.s."
                
                print(f"  {n_neurons} neurons: Deformation {mean_deform:.3f} vs "
                      f"Best Baseline {mean_baseline:.3f} "
                      f"({improvement:+.1f}%, p={p_value:.4f} {sig})")
    
    # Visualization
    print("\nGenerating visualizations...")
    _generate_plots(all_results, specializations, n_neurons_list, methods)
    
    # Final verdict
    print("\n" + "="*70)
    print("FINAL VERDICT")
    print("="*70)
    
    _print_final_verdict(all_results, specializations, n_neurons_list)


def _generate_plots(all_results, specializations, n_neurons_list, methods):
    """Generate and save visualization plots"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, specialization in enumerate(specializations):
        ax_ari = axes[0, idx]
        ax_nmi = axes[1, idx]
        
        for method_idx, method in enumerate(methods):
            aris_by_n = []
            ari_stds = []
            nmis_by_n = []
            nmi_stds = []
            
            for n_neurons in n_neurons_list:
                trials = all_results[specialization][n_neurons]
                aris = [trial[method]['ari'] for trial in trials]
                nmis = [trial[method]['nmi'] for trial in trials]
                
                aris_by_n.append(np.mean(aris))
                ari_stds.append(np.std(aris))
                nmis_by_n.append(np.mean(nmis))
                nmi_stds.append(np.std(nmis))
            
            x = np.arange(len(n_neurons_list))
            colors = ['#3498db', '#e74c3c', '#f39c12', '#2ecc71']
            
            ax_ari.errorbar(x + method_idx*0.2, aris_by_n, yerr=ari_stds,
                           label=method, marker='o', capsize=5, 
                           color=colors[method_idx], linewidth=2)
            ax_nmi.errorbar(x + method_idx*0.2, nmis_by_n, yerr=nmi_stds,
                           label=method, marker='o', capsize=5,
                           color=colors[method_idx], linewidth=2)
        
        ax_ari.set_title(f'{specialization.capitalize()} Specialization - ARI', 
                        fontweight='bold', fontsize=11)
        ax_ari.set_xticks(x + 0.3)
        ax_ari.set_xticklabels(n_neurons_list)
        ax_ari.set_xlabel('Number of Neurons')
        ax_ari.set_ylabel('Adjusted Rand Index')
        ax_ari.legend(fontsize=9)
        ax_ari.grid(alpha=0.3)
        ax_ari.set_ylim([0, 1])
        
        ax_nmi.set_title(f'{specialization.capitalize()} Specialization - NMI',
                        fontweight='bold', fontsize=11)
        ax_nmi.set_xticks(x + 0.3)
        ax_nmi.set_xticklabels(n_neurons_list)
        ax_nmi.set_xlabel('Number of Neurons')
        ax_nmi.set_ylabel('Normalized Mutual Information')
        ax_nmi.legend(fontsize=9)
        ax_nmi.grid(alpha=0.3)
        ax_nmi.set_ylim([0, 1])
    
    plt.tight_layout()
    output_path = 'comprehensive_neuron_test.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved: {output_path}")


def _print_final_verdict(all_results, specializations, n_neurons_list):
    """Analyze and print final verdict"""
    # Count wins across all conditions
    deform_wins = 0
    baseline_wins = 0
    ties = 0
    
    for specialization in specializations:
        for n_neurons in n_neurons_list:
            trials = all_results[specialization][n_neurons]
            
            mean_deform = np.mean([trial['Deformation']['ari'] for trial in trials])
            mean_pca = np.mean([trial['PCA']['ari'] for trial in trials])
            mean_xcorr = np.mean([trial['XCorr']['ari'] for trial in trials])
            mean_dim = np.mean([trial['Dimensionality']['ari'] for trial in trials])
            
            best_baseline = max(mean_pca, mean_xcorr, mean_dim)
            
            if mean_deform > best_baseline + 0.05:
                deform_wins += 1
            elif best_baseline > mean_deform + 0.05:
                baseline_wins += 1
            else:
                ties += 1
    
    total = deform_wins + baseline_wins + ties
    print(f"\nAcross {total} test conditions:")
    print(f"  Deformation wins: {deform_wins} ({deform_wins/total*100:.0f}%)")
    print(f"  Baseline wins:    {baseline_wins} ({baseline_wins/total*100:.0f}%)")
    print(f"  Ties:             {ties} ({ties/total*100:.0f}%)")
    
    if deform_wins > baseline_wins * 1.5:
        print("\n✓✓✓ STRONG EVIDENCE FOR DEFORMATION METHOD")
        print("The flow-based approach consistently outperforms baselines.")
        print("RECOMMENDATION: Strong go-ahead for real data testing.")
    elif deform_wins > baseline_wins:
        print("\n✓ MODERATE EVIDENCE FOR DEFORMATION METHOD")
        print("Deformation features show advantages but not overwhelming.")
        print("RECOMMENDATION: Cautious go-ahead, focus on high-specialization cases.")
    else:
        print("\n✗ INSUFFICIENT EVIDENCE")
        print("Baselines perform as well or better than deformation features.")
        print("RECOMMENDATION: Reconsider approach or improve method.")
