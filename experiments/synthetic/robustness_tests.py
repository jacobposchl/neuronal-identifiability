"""
Robustness testing suite for deformation-based neuron identification.
Contains 11 comprehensive tests evaluating method performance under various conditions.
"""

import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn.decomposition import PCA
from scipy.stats import ttest_rel

from ...src.core.population import RealisticNeuralPopulation
from ...src.core.dynamics import generate_complex_dynamics, estimate_deformation_from_latents
from ...src.core.features import (extract_deformation_features, extract_pca_features,
                       extract_crosscorr_features, extract_dimensionality_features)
from ...src.visualization import plot_bar, plot_line, plot_hist, plot_comparison


def test_information_content(n_trials=10):
    """Test 1: Information content across methods"""
    print("\n" + "="*70)
    print("TEST 1: Information Content")
    print("="*70)
    
    methods = ['Deformation', 'PCA', 'CrossCorr', 'Dimensionality']
    results = {m: [] for m in methods}
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        
        features_dict = {
            'Deformation': extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                                       rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp),
            'PCA': extract_pca_features(spike_trains, firing_rates),
            'CrossCorr': extract_crosscorr_features(spike_trains, firing_rates),
            'Dimensionality': extract_dimensionality_features(spike_trains, firing_rates)
        }
        
        for method, features in features_dict.items():
            scaler = StandardScaler()
            feat_norm = scaler.fit_transform(features)
            km = KMeans(n_clusters=4, random_state=trial, n_init=10)
            mi = normalized_mutual_info_score(pop.true_labels, km.fit_predict(feat_norm))
            results[method].append(mi)
    
    print("\nResults (Mutual Information):")
    print("-" * 70)
    stats = {}
    for method in methods:
        scores = np.array(results[method])
        mean, std = np.mean(scores), np.std(scores)
        ci = 1.96 * std / np.sqrt(len(scores))
        stats[method] = {'mean': mean, 'ci': ci}
        print(f"  {method:20s}: {mean:.4f} ± {ci:.4f}")
    
    # T-tests
    print("\nDeformation vs Others (paired t-tests):")
    for method in methods[1:]:
        t_stat, p = ttest_rel(results['Deformation'], results[method])
        print(f"  {method:20s}: t={t_stat:.2f}, p={p:.4f}")
    
    plot_bar(methods, stats, 'results/figures/01_information_content.png', 'Mutual Information')
    return stats


def test_noise_robustness(n_trials=10):
    """Test 2: Robustness to deformation measurement noise"""
    print("\n" + "="*70)
    print("TEST 2: Noise Robustness (Deformation Measurements)")
    print("="*70)
    
    noise_levels = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    results = {nl: [] for nl in noise_levels}
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        
        for nl in noise_levels:
            # Add noise to deformation trajectories
            rot_noisy = rot + np.random.randn(*rot.shape) * nl * np.std(rot)
            con_noisy = con + np.random.randn(*con.shape) * nl * np.std(con)
            exp_noisy = exp + np.random.randn(*exp.shape) * nl * np.std(exp)
            
            feat = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                               rotation_trajectory=rot_noisy, contraction_trajectory=con_noisy, expansion_trajectory=exp_noisy)
            scaler = StandardScaler()
            km = KMeans(n_clusters=4, random_state=trial, n_init=10)
            ari = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat)))
            results[nl].append(ari)
    
    print("\nResults (ARI vs Noise):")
    print("-" * 70)
    stats = {}
    for nl in noise_levels:
        scores = np.array(results[nl])
        mean, std = np.mean(scores), np.std(scores)
        ci = 1.96 * std / np.sqrt(len(scores))
        stats[nl] = {'mean': mean, 'ci': ci}
        print(f"  Noise {nl:.1%}: {mean:.4f} ± {ci:.4f}")
    
    deg = (stats[0.0]['mean'] - stats[1.0]['mean']) / stats[0.0]['mean'] * 100
    t_stat, p = ttest_rel(results[0.0], results[1.0])
    print(f"\nDegradation (0 pct -> 100 pct): {deg:.1f}% (t={t_stat:.2f}, p={p:.4f})")
    
    plot_line(noise_levels, stats, 'results/figures/02_noise_robustness.png', 'Noise Robustness')
    return stats


def test_clustering_stability(n_seeds=30):
    """Test 3: Stability across K-means random seeds"""
    print("\n" + "="*70)
    print("TEST 3: Clustering Stability")
    print("="*70)
    
    pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
    (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
    spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
    feat = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                       rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
    
    scaler = StandardScaler()
    feat_norm = scaler.fit_transform(feat)
    
    aris = []
    inertias = []
    for seed in range(n_seeds):
        km = KMeans(n_clusters=4, random_state=seed, n_init=1)
        pred = km.fit_predict(feat_norm)
        ari = adjusted_rand_score(pop.true_labels, pred)
        aris.append(ari)
        inertias.append(km.inertia_)
    
    aris = np.array(aris)
    inertias = np.array(inertias)
    ci = 1.96 * np.std(aris) / np.sqrt(len(aris))
    
    print(f"\nResults ({n_seeds} random seeds, n_init=1):")
    print("-" * 70)
    print(f"  Mean ARI:     {np.mean(aris):.4f}")
    print(f"  Std Dev:      {np.std(aris):.4f}")
    print(f"  Range:        [{np.min(aris):.4f}, {np.max(aris):.4f}]")
    print(f"  95% CI:       ± {ci:.4f}")
    print(f"  Inertia range: [{np.min(inertias):.2f}, {np.max(inertias):.2f}]")
    
    plot_hist(aris, 'results/figures/03_stability.png', 'Stability')
    return aris


def test_sufficient_statistics(n_trials=10):
    """Test 4: Dimensionality requirements + feature correlation analysis"""
    print("\n" + "="*70)
    print("TEST 4: Sufficient Statistics")
    print("="*70)
    
    features_list = ["Rotation", "Contraction", "Expansion"]
    results = {f: [] for f in features_list}
    results["All 3D"] = []
    correlations = []
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        feat = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                           rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        
        # Check correlations between features
        corr_matrix = np.corrcoef(feat.T)
        correlations.append(corr_matrix)
        
        for i, fname in enumerate(features_list):
            feat_1d = feat[:, [i]]
            scaler = StandardScaler()
            km = KMeans(n_clusters=4, random_state=trial, n_init=10)
            ari = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat_1d)))
            results[fname].append(ari)
        
        scaler = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat)))
        results["All 3D"].append(ari)
    
    print("\nResults (ARI by Dimensionality):")
    print("-" * 70)
    stats = {}
    for fname in list(features_list) + ["All 3D"]:
        scores = np.array(results[fname])
        mean, std = np.mean(scores), np.std(scores)
        ci = 1.96 * std / np.sqrt(len(scores))
        stats[fname] = {'mean': mean, 'ci': ci}
        print(f"  {fname:20s}: {mean:.4f} ± {ci:.4f}")
    
    # Feature correlation analysis
    print("\nFeature Correlation Analysis:")
    print("-" * 70)
    avg_corr = np.mean(correlations, axis=0)
    print("  Pairwise correlations:")
    for i in range(len(features_list)):
        for j in range(i+1, len(features_list)):
            print(f"    {features_list[i]}-{features_list[j]}: {avg_corr[i,j]:+.4f}")
    
    best_1d = max([stats[f]['mean'] for f in features_list])
    improve = (stats["All 3D"]['mean'] - best_1d) / best_1d * 100
    print(f"\nImprovement (1D -> 3D): {improve:.1f}%")
    if improve > 5:
        print("  Independent features capture meaningful structure")
    elif improve > 1:
        print("  Some benefit from combining features")
    else:
        print("  WARNING: Features still appear redundant!")
    
    plot_bar(list(stats.keys()), stats, 'results/figures/04_sufficient_stats.png', 'ARI')
    return stats


def test_independence(n_trials=10):
    """Test 5: Independence from assumptions"""
    print("\n" + "="*70)
    print("TEST 5: Independence from Assumptions")
    print("="*70)
    
    true_aris, random_aris = [], []
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        feat = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                           rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        
        scaler = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        pred = km.fit_predict(scaler.fit_transform(feat))
        
        true_aris.append(adjusted_rand_score(pop.true_labels, pred))
        random_aris.append(adjusted_rand_score(np.random.randint(0, 4, pop.n_neurons), pred))
    
    true_aris, random_aris = np.array(true_aris), np.array(random_aris)
    ci_true = 1.96 * np.std(true_aris) / np.sqrt(len(true_aris))
    ci_rand = 1.96 * np.std(random_aris) / np.sqrt(len(random_aris))
    
    print("\nResults (ARI vs Ground Truth):")
    print("-" * 70)
    print(f"  True types:   {np.mean(true_aris):.4f} ± {ci_true:.4f}")
    print(f"  Random types: {np.mean(random_aris):.4f} ± {ci_rand:.4f}")
    
    t_stat, p = ttest_rel(true_aris, random_aris)
    diff = np.mean(true_aris - random_aris)
    print(f"\nPaired t-test: t={t_stat:.2f}, p={p:.4f}")
    print(f"Mean difference: {diff:+.4f}")
    
    plot_comparison(true_aris, random_aris, 'results/figures/05_independence.png')
    return {'true': true_aris, 'random': random_aris}


def test_generalization_across_populations(n_trials=5):
    """Test 6: Generalization across population configurations"""
    print("\n" + "="*70)
    print("TEST 6: Generalization Across Populations")
    print("="*70)
    
    configs = [
        (20, 'low'),
        (20, 'high'),
        (60, 'low'),
        (60, 'high'),
        (100, 'medium'),
    ]
    
    results_by_config = {}
    
    for n_neurons, spec in configs:
        print(f"  Config: n={n_neurons}, specialization={spec}...", end='\r')
        aris = []
        
        for trial in range(n_trials):
            pop = RealisticNeuralPopulation(n_neurons=n_neurons, specialization_level=spec)
            (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
            spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
            feat = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                               rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
            
            scaler = StandardScaler()
            km = KMeans(n_clusters=4, random_state=trial, n_init=10)
            ari = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat)))
            aris.append(ari)
        
        mean_ari = np.mean(aris)
        ci = 1.96 * np.std(aris) / np.sqrt(len(aris))
        results_by_config[f"{n_neurons}_{spec}"] = {'mean': mean_ari, 'ci': ci}
    
    print("\nResults (ARI by Population Config):")
    print("-" * 70)
    for config in sorted(results_by_config.keys()):
        m = results_by_config[config]
        print(f"  {config:20s}: {m['mean']:.4f} ± {m['ci']:.4f}")
    
    return results_by_config


def test_noise_types(n_trials=5):
    """Test 7: Robustness to different noise types"""
    print("\n" + "="*70)
    print("TEST 7: Different Noise Types")
    print("="*70)
    
    noise_types = {
        'none': lambda rates: rates,
        'gaussian': lambda rates: rates + np.random.randn(*rates.shape) * 0.5,
        'poisson': lambda rates: np.random.poisson(np.maximum(rates, 0.1)),
        'outliers': lambda rates: rates + np.random.binomial(1, 0.05, rates.shape) * np.random.randn(*rates.shape) * 3,
    }
    
    results_by_noise = {}
    
    for noise_name, noise_fn in noise_types.items():
        print(f"  Testing {noise_name}...", end='\r')
        aris = []
        
        for trial in range(n_trials):
            pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
            (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
            spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
            
            # Apply noise to firing rates
            firing_rates_noisy = noise_fn(firing_rates)
            
            feat = extract_deformation_features(spike_trains, firing_rates_noisy, z, dt=0.001,
                                               rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
            scaler = StandardScaler()
            km = KMeans(n_clusters=4, random_state=trial, n_init=10)
            ari = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat)))
            aris.append(ari)
        
        mean_ari = np.mean(aris)
        ci = 1.96 * np.std(aris) / np.sqrt(len(aris))
        results_by_noise[noise_name] = {'mean': mean_ari, 'ci': ci}
    
    print("\nResults (ARI by Noise Type):")
    print("-" * 70)
    for noise_type in results_by_noise:
        m = results_by_noise[noise_type]
        print(f"  {noise_type:20s}: {m['mean']:.4f} ± {m['ci']:.4f}")
    
    return results_by_noise


def test_false_positives(n_trials=5):
    """Test 8: False positive rate on random data"""
    print("\n" + "="*70)
    print("TEST 8: False Positives (Random Data)")
    print("="*70)
    
    aris_real, aris_random = [], []
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        
        # Real data
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        feat_real = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                                rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        
        scaler = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_real = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat_real)))
        aris_real.append(ari_real)
        
        # Random spike trains (no structure)
        random_spike_trains = np.random.binomial(1, 0.01, spike_trains.shape)
        random_firing_rates = np.random.poisson(2, spike_trains.shape)
        feat_random = extract_deformation_features(random_spike_trains, random_firing_rates, z, dt=0.001,
                                                  rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        
        km_rand = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_random = adjusted_rand_score(pop.true_labels, km_rand.fit_predict(scaler.fit_transform(feat_random)))
        aris_random.append(ari_random)
    
    aris_real = np.array(aris_real)
    aris_random = np.array(aris_random)
    ci_real = 1.96 * np.std(aris_real) / np.sqrt(len(aris_real))
    ci_rand = 1.96 * np.std(aris_random) / np.sqrt(len(aris_random))
    
    print("\nResults (False Positive Rate):")
    print("-" * 70)
    print(f"  Real data:   {np.mean(aris_real):.4f} ± {ci_real:.4f}")
    print(f"  Random data: {np.mean(aris_random):.4f} ± {ci_rand:.4f}")
    
    t_stat, p = ttest_rel(aris_real, aris_random)
    print(f"\nPaired t-test: t={t_stat:.2f}, p={p:.4f}")
    print("  Low false positive rate" if np.mean(aris_random) < 0.05 else "  WARNING: High false positive rate")
    
    return {'real': aris_real, 'random': aris_random}


def test_alternative_deformation_methods(n_trials=5):
    """Test 9: Compare different deformation feature extraction approaches"""
    print("\n" + "="*70)
    print("TEST 9: Alternative Deformation Methods")
    print("="*70)
    
    print("  Comparing feature extraction methods...")
    
    results_by_method = {'eigenvalue_based': [], 'trace_based': [], 'norm_based': []}
    
    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        
        # Method 1: Our deformation-based (rotation, contraction, expansion)
        feat_eig = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                               rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        scaler = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_eig = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat_eig)))
        results_by_method['eigenvalue_based'].append(ari_eig)
        
        # Method 2: Trace-based (sum of features)
        feat_base = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                                rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        feat_trace = np.hstack([
            feat_base[:, :2],
            np.sum(feat_base, axis=1, keepdims=True)
        ])
        scaler2 = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_trace = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler2.fit_transform(feat_trace)))
        results_by_method['trace_based'].append(ari_trace)
        
        # Method 3: Norm-based (Frobenius norms only)
        feat_base = extract_deformation_features(spike_trains, firing_rates, z, dt=0.001,
                                                rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp)
        feat_norm = np.tile(
            np.linalg.norm(feat_base, axis=1, keepdims=True),
            (1, 3)
        )
        scaler3 = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_norm = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler3.fit_transform(feat_norm)))
        results_by_method['norm_based'].append(ari_norm)
    
    print("\nResults (ARI by Method):")
    print("-" * 70)
    stats = {}
    for method in results_by_method:
        scores = np.array(results_by_method[method])
        mean = np.mean(scores)
        ci = 1.96 * np.std(scores) / np.sqrt(len(scores))
        stats[method] = {'mean': mean, 'ci': ci}
        print(f"  {method:20s}: {mean:.4f} ± {ci:.4f}")
    
    return stats


def test_latent_estimation_noise(n_trials=5):
    """Test 10: Sensitivity to latent trajectory estimation noise (PCA-based)."""
    print("\n" + "="*70)
    print("TEST 10: Latent Estimation Noise (PCA Latents)")
    print("="*70)

    results = {'true_latents': [], 'estimated_latents': []}

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)

        # Baseline: true deformation signals
        feat_true = extract_deformation_features(
            spike_trains, firing_rates, z, dt=0.001,
            rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp
        )

        scaler = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_true = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat_true)))
        results['true_latents'].append(ari_true)

        # Estimated latents via PCA on firing rates
        rates_centered = firing_rates.T - np.mean(firing_rates.T, axis=0, keepdims=True)
        pca = PCA(n_components=3)
        z_hat = pca.fit_transform(rates_centered)

        # Estimate deformation from PCA latents
        rot_hat, con_hat, exp_hat = estimate_deformation_from_latents(z_hat, dt=0.001)

        feat_est = extract_deformation_features(
            spike_trains, firing_rates, z, dt=0.001,
            rotation_trajectory=rot_hat, contraction_trajectory=con_hat, expansion_trajectory=exp_hat
        )

        scaler2 = StandardScaler()
        km2 = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_est = adjusted_rand_score(pop.true_labels, km2.fit_predict(scaler2.fit_transform(feat_est)))
        results['estimated_latents'].append(ari_est)

    print("\nResults (ARI by Latent Source):")
    print("-" * 70)
    stats = {}
    for key in results:
        scores = np.array(results[key])
        mean = np.mean(scores)
        ci = 1.96 * np.std(scores) / np.sqrt(len(scores))
        stats[key] = {'mean': mean, 'ci': ci}
        print(f"  {key:20s}: {mean:.4f} ± {ci:.4f}")

    deg = (stats['true_latents']['mean'] - stats['estimated_latents']['mean']) / stats['true_latents']['mean'] * 100
    t_stat, p = ttest_rel(results['true_latents'], results['estimated_latents'])
    print(f"\nDegradation (true -> estimated): {deg:.1f}% (t={t_stat:.2f}, p={p:.4f})")

    return stats


def test_deformation_ablation(n_trials=5):
    """Test 11: Deformation ablation (zeroing each deformation signal)."""
    print("\n" + "="*70)
    print("TEST 11: Deformation Ablation")
    print("="*70)

    feature_names = ['rotation', 'contraction', 'expansion']
    results = {name: {'corr_drop': [], 'ari_drop': []} for name in feature_names}

    for trial in range(n_trials):
        print(f"  Trial {trial+1}/{n_trials}...", end='\r')
        pop = RealisticNeuralPopulation(n_neurons=60, specialization_level='high')
        (z, rot, con, exp, t) = generate_complex_dynamics(T=10.0, dt=0.002)

        # Baseline
        spike_trains, firing_rates = pop.generate_spike_trains(z, rot, con, exp, t, dt=0.001)
        feat_base = extract_deformation_features(
            spike_trains, firing_rates, z, dt=0.001,
            rotation_trajectory=rot, contraction_trajectory=con, expansion_trajectory=exp
        )

        scaler = StandardScaler()
        km = KMeans(n_clusters=4, random_state=trial, n_init=10)
        ari_base = adjusted_rand_score(pop.true_labels, km.fit_predict(scaler.fit_transform(feat_base)))

        for feature_idx, name in enumerate(feature_names):
            rot_a, con_a, exp_a = rot.copy(), con.copy(), exp.copy()
            if name == 'rotation':
                rot_a = np.zeros_like(rot_a)
            elif name == 'contraction':
                con_a = np.zeros_like(con_a)
            else:
                exp_a = np.zeros_like(exp_a)

            spike_trains_a, firing_rates_a = pop.generate_spike_trains(z, rot_a, con_a, exp_a, t, dt=0.001)
            feat_a = extract_deformation_features(
                spike_trains_a, firing_rates_a, z, dt=0.001,
                rotation_trajectory=rot_a, contraction_trajectory=con_a, expansion_trajectory=exp_a
            )

            # Correlation drop for neurons of the ablated type
            mask = pop.true_labels == feature_idx
            if np.any(mask):
                base_corr = np.mean(np.abs(feat_base[mask, feature_idx]))
                ablated_corr = np.mean(np.abs(feat_a[mask, feature_idx]))
                corr_drop = (base_corr - ablated_corr) / (base_corr + 1e-10) * 100
                results[name]['corr_drop'].append(corr_drop)

            # ARI drop
            scaler_a = StandardScaler()
            km_a = KMeans(n_clusters=4, random_state=trial, n_init=10)
            ari_a = adjusted_rand_score(pop.true_labels, km_a.fit_predict(scaler_a.fit_transform(feat_a)))
            ari_drop = (ari_base - ari_a) / (ari_base + 1e-10) * 100
            results[name]['ari_drop'].append(ari_drop)

    print("\nResults (Ablation Impact):")
    print("-" * 70)
    for name in feature_names:
        corr_drop = np.array(results[name]['corr_drop'])
        ari_drop = np.array(results[name]['ari_drop'])
        print(f"  {name:12s}: corr drop {np.mean(corr_drop):.1f}% ± {np.std(corr_drop):.1f}%, "
              f"ARI drop {np.mean(ari_drop):.1f}% ± {np.std(ari_drop):.1f}%")

    return results
