"""
Feature extraction for RNN unit classification based on deformation dynamics.

Adapts the deformation-based feature extraction approach from biological neurons
to work with continuous RNN hidden unit activations.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def select_optimal_clusters(features, min_clusters=2, max_clusters=8, 
                            method='silhouette', verbose=False):
    """
    Determine optimal number of clusters using multiple criteria.
    
    Methods:
    - 'silhouette': Maximize silhouette score (cluster quality)
    - 'elbow': Find elbow in inertia curve (diminishing returns)
    - 'bic': Minimize BIC (Bayesian Information Criterion)
    - 'combined': Weight multiple criteria
    
    Args:
        features: (n_units, 3) feature array
        min_clusters: Minimum number of clusters to test (default: 2)
        max_clusters: Maximum number of clusters to test (default: 8)
        method: Selection method ('silhouette', 'elbow', 'bic', 'combined')
        verbose: Print diagnostic information
    
    Returns:
        optimal_k: Recommended number of clusters
        scores: Dict with all scores for each k value
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    n_units = features.shape[0]
    
    # Limit max_clusters to reasonable values
    max_clusters = min(max_clusters, n_units - 1)
    
    if max_clusters < min_clusters:
        if verbose:
            print(f"  Warning: Not enough units ({n_units}) for cluster selection")
        return min_clusters, {}
    
    # Normalize features
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # Test different cluster numbers
    k_values = range(min_clusters, max_clusters + 1)
    silhouette_scores = []
    inertias = []
    bic_scores = []
    
    for k in k_values:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features_norm)
        
        # Silhouette score (higher is better)
        if len(np.unique(labels)) >= 2:
            try:
                sil_score = silhouette_score(features_norm, labels)
            except:
                sil_score = -1.0
        else:
            sil_score = -1.0
        silhouette_scores.append(sil_score)
        
        # Inertia (for elbow method - lower is better)
        inertias.append(kmeans.inertia_)
        
        # BIC approximation (lower is better)
        # BIC = -2 * log-likelihood + k * log(n)
        # For k-means, use inertia as proxy for -log-likelihood
        bic = kmeans.inertia_ + k * np.log(n_units)
        bic_scores.append(bic)
    
    # Select optimal k based on method
    if method == 'silhouette':
        optimal_k = k_values[np.argmax(silhouette_scores)]
        if verbose:
            print(f"  Silhouette method: k={optimal_k} (score={max(silhouette_scores):.3f})")
    
    elif method == 'elbow':
        # Find elbow using second derivative
        inertias_arr = np.array(inertias)
        # Normalize to [0, 1]
        inertias_norm = (inertias_arr - inertias_arr.min()) / (inertias_arr.max() - inertias_arr.min() + 1e-10)
        # Compute second derivative
        if len(inertias_norm) >= 3:
            second_deriv = np.diff(inertias_norm, n=2)
            # Elbow is where curvature is maximum (most negative second derivative)
            elbow_idx = np.argmin(second_deriv)
            optimal_k = k_values[elbow_idx + 1]  # +1 because diff reduces length
        else:
            optimal_k = min_clusters
        if verbose:
            print(f"  Elbow method: k={optimal_k}")
    
    elif method == 'bic':
        optimal_k = k_values[np.argmin(bic_scores)]
        if verbose:
            print(f"  BIC method: k={optimal_k} (BIC={min(bic_scores):.1f})")
    
    elif method == 'combined':
        # Weight multiple criteria (normalized to [0, 1])
        sil_norm = np.array(silhouette_scores)
        sil_norm = (sil_norm - sil_norm.min()) / (sil_norm.max() - sil_norm.min() + 1e-10)
        
        inertia_norm = np.array(inertias)
        inertia_norm = 1 - (inertia_norm - inertia_norm.min()) / (inertia_norm.max() - inertia_norm.min() + 1e-10)
        
        bic_norm = np.array(bic_scores)
        bic_norm = 1 - (bic_norm - bic_norm.min()) / (bic_norm.max() - bic_norm.min() + 1e-10)
        
        # Combined score (equal weights)
        combined = 0.5 * sil_norm + 0.3 * inertia_norm + 0.2 * bic_norm
        optimal_k = k_values[np.argmax(combined)]
        if verbose:
            print(f"  Combined method: k={optimal_k} (score={max(combined):.3f})")
    
    else:
        raise ValueError(f"Unknown method: {method}")
    
    scores = {
        'k_values': list(k_values),
        'silhouette': silhouette_scores,
        'inertia': inertias,
        'bic': bic_scores
    }
    
    return optimal_k, scores


def extract_rnn_unit_features(hidden_states, rotation_trajectory, 
                               contraction_trajectory, expansion_trajectory,
                               smooth_sigma=5, use_abs=True, normalize=True):
    """
    Extract deformation-based features for RNN units (basic version).
    
    Computes correlation between each unit's activity and the three deformation
    modes (rotation, contraction, expansion) over time.
    
    Key improvements for ground truth recovery:
    - use_abs=True: Uses absolute correlations (ignores polarity artifacts)
    - normalize=True: Normalizes to sum=1 (ratio-based features)
    
    Note: For enhanced features with temporal/contextual resolution, use
    extract_enhanced_rnn_features() instead.
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN hidden unit activations
        rotation_trajectory: (n_timesteps,) rotation magnitude over time
        contraction_trajectory: (n_timesteps,) contraction magnitude over time
        expansion_trajectory: (n_timesteps,) expansion trajectory over time
        smooth_sigma: Gaussian smoothing parameter (0 = no smoothing)
        use_abs: If True, use absolute values of correlations (recommended)
        normalize: If True, normalize features to sum=1 (recommended)
    
    Returns:
        features: (n_units, 3) array where features[i] = [rotation, contraction, expansion]
                  Values represent relative strength of correlation with each mode
    """
    n_units, n_timesteps = hidden_states.shape
    
    # Ensure deformation trajectories match hidden states length
    min_len = min(n_timesteps, len(rotation_trajectory), 
                  len(contraction_trajectory), len(expansion_trajectory))
    
    # Truncate all to same length
    hidden_states = hidden_states[:, :min_len]
    rotation_trajectory = rotation_trajectory[:min_len]
    contraction_trajectory = contraction_trajectory[:min_len]
    expansion_trajectory = expansion_trajectory[:min_len]
    
    # Check if deformation signals have any variance
    deform_std = [np.std(rotation_trajectory), np.std(contraction_trajectory), 
                  np.std(expansion_trajectory)]
    if all(s < 1e-10 for s in deform_std):
        print("  Warning: All deformation signals are constant (std < 1e-10)")
        print("  Features will be all zeros - clustering may be unreliable")
    
    # Optional: smooth hidden states to reduce noise
    if smooth_sigma > 0:
        hidden_states_smooth = np.array([
            gaussian_filter1d(hidden_states[i], sigma=smooth_sigma)
            for i in range(n_units)
        ])
    else:
        hidden_states_smooth = hidden_states
    
    # Extract features for each unit
    features = []
    
    for i in range(n_units):
        unit_activity = hidden_states_smooth[i]
        
        # Compute correlations with each deformation mode
        corr_rotation = np.corrcoef(unit_activity, rotation_trajectory)[0, 1]
        corr_contraction = np.corrcoef(unit_activity, contraction_trajectory)[0, 1]
        corr_expansion = np.corrcoef(unit_activity, expansion_trajectory)[0, 1]
        
        # Handle NaN (constant signals)
        corr_rotation = 0 if np.isnan(corr_rotation) else corr_rotation
        corr_contraction = 0 if np.isnan(corr_contraction) else corr_contraction
        corr_expansion = 0 if np.isnan(corr_expansion) else corr_expansion
        
        # Use absolute values to ignore polarity artifacts
        if use_abs:
            corr_rotation = abs(corr_rotation)
            corr_contraction = abs(corr_contraction)
            corr_expansion = abs(corr_expansion)
        
        # Normalize to ratio-based features (sum to 1)
        if normalize:
            total = corr_rotation + corr_contraction + corr_expansion
            if total > 1e-10:  # Avoid division by zero
                corr_rotation /= total
                corr_contraction /= total
                corr_expansion /= total
            else:
                # If all correlations are ~0, assign equal weight
                corr_rotation = 1/3
                corr_contraction = 1/3
                corr_expansion = 1/3
        
        features.append([corr_rotation, corr_contraction, corr_expansion])
    
    return np.array(features)


def extract_enhanced_rnn_features(hidden_states, rotation_trajectory, 
                                  contraction_trajectory, expansion_trajectory,
                                  task_info, inputs=None,
                                  smooth_sigma=5, use_abs=True, normalize=True):
    """
    Extract enhanced deformation features with temporal and contextual resolution.
    
    Addresses the problem of multiple units correlating with the same deformation mode
    but serving different functional roles by adding:
    - Temporal specificity (early/middle/late trial phases)
    - Task-phase specificity (encoding/integration/decision)
    - Contextual specificity (Context A vs Context B for context tasks)
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN hidden unit activations
        rotation_trajectory: (n_timesteps,) rotation magnitude over time
        contraction_trajectory: (n_timesteps,) contraction magnitude over time
        expansion_trajectory: (n_timesteps,) expansion trajectory over time
        task_info: Dict containing:
            - 'trial_length': length of each trial (e.g., 200)
            - 'n_trials': number of trials (e.g., 50)
            - 'task_name': name of task (e.g., 'context')
            - 'phase_boundaries': Optional dict with phase timing info
        inputs: (n_timesteps, input_size) input sequences (needed for context detection)
        smooth_sigma: Gaussian smoothing parameter
        use_abs: If True, use absolute values of correlations
        normalize: If True, normalize feature groups to sum=1
    
    Returns:
        features: (n_units, n_features) array with:
            - First 3: global R, C, E correlations (backwards compatible)
            - Next 9: temporal R, C, E for early/mid/late periods
            - Additional: task-specific contextual features
    """
    n_units, n_timesteps = hidden_states.shape
    trial_length = task_info.get('trial_length', 200)
    n_trials = task_info.get('n_trials', n_timesteps // trial_length)
    task_name = task_info.get('task_name', 'unknown').lower()
    
    # Ensure trajectories match length
    min_len = min(n_timesteps, len(rotation_trajectory), 
                  len(contraction_trajectory), len(expansion_trajectory))
    
    hidden_states = hidden_states[:, :min_len]
    rotation_trajectory = rotation_trajectory[:min_len]
    contraction_trajectory = contraction_trajectory[:min_len]
    expansion_trajectory = expansion_trajectory[:min_len]
    if inputs is not None:
        inputs = inputs[:min_len]
    
    # Smooth hidden states
    if smooth_sigma > 0:
        hidden_states_smooth = np.array([
            gaussian_filter1d(hidden_states[i], sigma=smooth_sigma)
            for i in range(n_units)
        ])
    else:
        hidden_states_smooth = hidden_states
    
    # Define temporal boundaries (thirds)
    third = min_len // 3
    early_slice = slice(0, third)
    middle_slice = slice(third, 2*third)
    late_slice = slice(2*third, min_len)
    
    all_features = []
    
    for i in range(n_units):
        unit_activity = hidden_states_smooth[i]
        unit_features = []
        
        # === GLOBAL FEATURES (3 features) ===
        # These maintain backwards compatibility
        global_feats = _compute_correlations(
            unit_activity,
            rotation_trajectory,
            contraction_trajectory,
            expansion_trajectory,
            use_abs=use_abs,
            normalize=normalize
        )
        unit_features.extend(global_feats)
        
        # === TEMPORAL FEATURES (9 features) ===
        # Early period
        early_feats = _compute_correlations(
            unit_activity[early_slice],
            rotation_trajectory[early_slice],
            contraction_trajectory[early_slice],
            expansion_trajectory[early_slice],
            use_abs=use_abs,
            normalize=normalize
        )
        unit_features.extend(early_feats)
        
        # Middle period
        middle_feats = _compute_correlations(
            unit_activity[middle_slice],
            rotation_trajectory[middle_slice],
            contraction_trajectory[middle_slice],
            expansion_trajectory[middle_slice],
            use_abs=use_abs,
            normalize=normalize
        )
        unit_features.extend(middle_feats)
        
        # Late period
        late_feats = _compute_correlations(
            unit_activity[late_slice],
            rotation_trajectory[late_slice],
            contraction_trajectory[late_slice],
            expansion_trajectory[late_slice],
            use_abs=use_abs,
            normalize=normalize
        )
        unit_features.extend(late_feats)
        
        # === CONTEXTUAL FEATURES (task-specific) ===
        if task_name == 'context' and inputs is not None:
            # Context task: separate Context A vs Context B
            context_feats = _compute_context_features(
                unit_activity, inputs, trial_length, n_trials,
                rotation_trajectory, contraction_trajectory, expansion_trajectory,
                use_abs=use_abs, normalize=normalize
            )
            unit_features.extend(context_feats)
        
        all_features.append(unit_features)
    
    return np.array(all_features)


def _compute_correlations(activity, rotation, contraction, expansion,
                          use_abs=True, normalize=True):
    """
    Helper function to compute correlations between activity and deformation signals.
    
    Returns:
        [corr_rotation, corr_contraction, corr_expansion]
    """
    if len(activity) == 0 or len(rotation) == 0:
        return [0.33, 0.33, 0.33] if normalize else [0.0, 0.0, 0.0]
    
    # Compute correlations
    corr_r = np.corrcoef(activity, rotation)[0, 1] if len(activity) > 1 else 0.0
    corr_c = np.corrcoef(activity, contraction)[0, 1] if len(activity) > 1 else 0.0
    corr_e = np.corrcoef(activity, expansion)[0, 1] if len(activity) > 1 else 0.0
    
    # Handle NaN
    corr_r = 0.0 if np.isnan(corr_r) else corr_r
    corr_c = 0.0 if np.isnan(corr_c) else corr_c
    corr_e = 0.0 if np.isnan(corr_e) else corr_e
    
    # Use absolute values
    if use_abs:
        corr_r = abs(corr_r)
        corr_c = abs(corr_c)
        corr_e = abs(corr_e)
    
    # Normalize to sum to 1
    if normalize:
        total = corr_r + corr_c + corr_e
        if total > 1e-10:
            corr_r /= total
            corr_c /= total
            corr_e /= total
        else:
            corr_r = corr_c = corr_e = 1/3
    
    return [corr_r, corr_c, corr_e]


def _compute_context_features(activity, inputs, trial_length, n_trials,
                              rotation, contraction, expansion,
                              use_abs=True, normalize=True):
    """
    Compute context-specific features for context integration task.
    
    Context A trials: first 20 timesteps have input[0] > 0.5
    Context B trials: first 20 timesteps have input[1] > 0.5
    
    Returns:
        [contextA_R, contextA_C, contextA_E, contextB_R, contextB_C, contextB_E] (6 features)
    """
    # Reconstruct trial structure
    n_total_timesteps = len(activity)
    actual_n_trials = min(n_trials, n_total_timesteps // trial_length)
    
    # Identify context A vs B trials
    contextA_mask = np.zeros(n_total_timesteps, dtype=bool)
    contextB_mask = np.zeros(n_total_timesteps, dtype=bool)
    
    for trial_idx in range(actual_n_trials):
        trial_start = trial_idx * trial_length
        trial_end = min(trial_start + trial_length, n_total_timesteps)
        
        # Check first 20 timesteps of trial
        check_end = min(trial_start + 20, trial_end)
        if check_end > trial_start and inputs.shape[0] > check_end:
            trial_inputs = inputs[trial_start:check_end]
            
            # Context A: input channel 0 dominant
            if trial_inputs.shape[1] >= 2:
                if np.mean(trial_inputs[:, 0]) > 0.5:
                    contextA_mask[trial_start:trial_end] = True
                # Context B: input channel 1 dominant
                elif np.mean(trial_inputs[:, 1]) > 0.5:
                    contextB_mask[trial_start:trial_end] = True
    
    # Compute correlations separately for each context
    contextA_feats = [0.33, 0.33, 0.33] if normalize else [0.0, 0.0, 0.0]
    contextB_feats = [0.33, 0.33, 0.33] if normalize else [0.0, 0.0, 0.0]
    
    if np.sum(contextA_mask) > 10:  # Need enough samples
        contextA_feats = _compute_correlations(
            activity[contextA_mask],
            rotation[contextA_mask],
            contraction[contextA_mask],
            expansion[contextA_mask],
            use_abs=use_abs,
            normalize=normalize
        )
    
    if np.sum(contextB_mask) > 10:
        contextB_feats = _compute_correlations(
            activity[contextB_mask],
            rotation[contextB_mask],
            contraction[contextB_mask],
            expansion[contextB_mask],
            use_abs=use_abs,
            normalize=normalize
        )
    
    return contextA_feats + contextB_feats


def classify_units(features, n_clusters=4, method='kmeans', return_details=False):
    """
    Classify RNN units based on deformation features.
    
    Automatically handles both basic (n_features=3) and enhanced (n_features>3) feature formats.
    
    Args:
        features: (n_units, n_features) deformation feature array
                  n_features=3: basic [R, C, E] correlations
                  n_features>3: enhanced with temporal/contextual features
        n_clusters: Number of clusters (default 4: rotators, integrators, explorers, mixed)
        method: 'kmeans' or 'argmax' (argmax assigns based on dominant feature)
        return_details: If True, return cluster centers and metrics
    
    Returns:
        labels: (n_units,) cluster assignments
        details: Dict with 'centers', 'silhouette', 'inertia' (if return_details=True)
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    
    n_units = features.shape[0]
    
    if method == 'kmeans':
        # Normalize features
        scaler = StandardScaler()
        features_norm = scaler.fit_transform(features)
        
        # K-means clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels = kmeans.fit_predict(features_norm)
        
        if return_details:
            # Compute metrics
            n_unique_labels = len(np.unique(labels))
            # Silhouette requires at least 2 distinct clusters
            if n_unique_labels >= 2 and n_units > n_clusters:
                try:
                    silhouette = silhouette_score(features_norm, labels)
                except:
                    silhouette = 0.0  # Fallback if computation fails
            else:
                silhouette = 0.0  # Not enough clusters for silhouette
            
            # Get cluster centers in original space
            centers = scaler.inverse_transform(kmeans.cluster_centers_)
            
            details = {
                'centers': centers,
                'silhouette': silhouette,
                'inertia': kmeans.inertia_,
                'scaler': scaler
            }
            return labels, details
        
        return labels
    
    elif method == 'argmax':
        # Simple classification: assign to dominant feature
        # 0 = Rotator, 1 = Integrator, 2 = Explorer
        labels = np.argmax(np.abs(features), axis=1)
        
        # Add 'mixed' category (label 3) for units with no strong preference
        max_corr = np.max(np.abs(features), axis=1)
        labels[max_corr < 0.3] = 3  # Threshold for 'mixed'
        
        if return_details:
            # Compute pseudo-centers (mean features per class)
            centers = np.array([
                np.mean(features[labels == k], axis=0) if np.sum(labels == k) > 0 else np.zeros(3)
                for k in range(n_clusters)
            ])
            
            details = {
                'centers': centers,
                'silhouette': None,
                'inertia': None,
                'scaler': None
            }
            return labels, details
        
        return labels
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'kmeans' or 'argmax'")


def interpret_clusters(features, labels, cluster_names=None, threshold=0.03, strict_threshold=0.10, 
                       feature_type='deformation', normalized=True):
    """
    Interpret cluster assignments with 3-tier confidence classification.
    
    Handles basic (3 features) and enhanced (>3 features) feature formats.
    For enhanced features, identifies temporal and contextual specialization.
    
    For normalized features (recommended):
    - Features sum to 1 (ratio of correlation strengths)
    - Threshold logic based on deviation from uniform (0.33, 0.33, 0.33)
    - Dominant feature > 0.45 = high confidence
    - Dominant feature > 0.38 = low confidence
    - Otherwise = mixed
    
    For raw features (legacy):
    - Uses absolute correlation thresholds (0.03, 0.10)
    
    Args:
        features: (n_units, n_features) deformation feature array
                  n_features=3: basic [R, C, E]
                  n_features=12: enhanced with temporal [global + early/mid/late]
                  n_features=18: enhanced with temporal + contextual [+ contextA/B]
        labels: (n_units,) cluster assignments
        cluster_names: Optional list of names (default: auto-assign based on features)
        threshold: For raw features - minimum correlation (default: 0.03)
        strict_threshold: For raw features - minimum for confident classification (default: 0.10)
        feature_type: 'deformation' (default) or 'pca' - affects interpretation labels
        normalized: If True, assumes features are normalized ratios (sum=1)
    
    Returns:
        interpretation: Dict mapping cluster ID to interpretation dict
    """
    n_clusters = len(np.unique(labels))
    n_features = features.shape[1]
    interpretation = {}
    
    # Detect feature format
    is_enhanced = (n_features > 3)
    has_context = (n_features >= 18)  # 3 global + 9 temporal + 6 context
    
    if feature_type == 'deformation':
        feature_names = ['Rotation', 'Contraction', 'Expansion']
        type_names = ['Rotator', 'Integrator', 'Explorer']
    else:  # PCA or other
        feature_names = ['PC1', 'PC2', 'PC3']
        type_names = ['PC1-dominant', 'PC2-dominant', 'PC3-dominant']
    
    # Adapt thresholds for normalized features
    if normalized:
        # For normalized features summing to 1:
        # - Uniform distribution: (0.33, 0.33, 0.33)
        # - Dominant feature > 0.45 means strong preference (high confidence)
        # - Dominant feature > 0.38 means moderate preference (low confidence)
        # - Otherwise mixed
        weak_threshold = 0.38  # ~15% above uniform
        strong_threshold = 0.45  # ~35% above uniform
    else:
        # Use original thresholds for raw correlations
        weak_threshold = threshold
        strong_threshold = strict_threshold
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        n_units = np.sum(cluster_mask)
        
        if n_units == 0:
            interpretation[cluster_id] = {
                'name': 'Empty',
                'n_units': 0,
                'mean_features': np.zeros(3),
                'dominant_mode': None,
                'confidence': 'none',
                'max_correlation': 0.0
            }
            continue
        
        # Compute mean features for this cluster
        cluster_features = features[cluster_mask]
        mean_features = np.mean(cluster_features, axis=0)
        
        # For enhanced features, analyze temporal and contextual patterns
        if is_enhanced:
            interpretation[cluster_id] = _interpret_enhanced_cluster(
                cluster_id, cluster_features, mean_features, labels,
                feature_type, normalized, has_context,
                weak_threshold, strong_threshold
            )
            continue
        
        # === BASIC FEATURES (original logic) ===
        # For normalized features, already positive; for raw, use absolute
        if normalized:
            abs_features = mean_features  # Already positive
        else:
            abs_features = np.abs(mean_features)
        
        # Determine dominant mode
        dominant_idx = np.argmax(abs_features)
        dominant_value = mean_features[dominant_idx]
        dominant_mode = feature_names[dominant_idx]
        max_value = abs_features[dominant_idx]
        
        # THREE-TIER CLASSIFICATION
        if cluster_names is None:
            if max_value < weak_threshold:
                # Very weak - Mixed type
                name = 'Mixed'
                confidence = 'very_low'
                
            elif max_value < strong_threshold:
                # Weak signal - classify but flag uncertainty
                name = type_names[dominant_idx] + '?'
                confidence = 'low'
                
            else:
                # Strong signal - confident classification
                name = type_names[dominant_idx]
                confidence = 'high'
                
                # Check for multi-modal units (two modes within 20% of each other)
                sorted_abs = np.sort(abs_features)[::-1]
                if len(sorted_abs) > 1 and sorted_abs[1] > 0.8 * sorted_abs[0]:
                    name = name + '+Mixed'
        else:
            name = cluster_names[cluster_id] if cluster_id < len(cluster_names) else f'Cluster {cluster_id}'
            confidence = 'user_specified'
        
        interpretation[cluster_id] = {
            'name': name,
            'n_units': n_units,
            'mean_features': mean_features,
            'dominant_mode': dominant_mode,
            'dominant_value': dominant_value,
            'percentage': 100 * n_units / len(labels),
            'confidence': confidence,
            'max_correlation': max_value
        }
    
    return interpretation


def _interpret_enhanced_cluster(cluster_id, cluster_features, mean_features, all_labels,
                                feature_type, normalized, has_context,
                                weak_threshold, strong_threshold):
    """
    Interpret a cluster using enhanced features (temporal + contextual).
    
    Feature structure:
    - [0:3]: global R, C, E
    - [3:6]: early R, C, E
    - [6:9]: middle R, C, E
    - [9:12]: late R, C, E
    - [12:15]: contextA R, C, E (if has_context)
    - [15:18]: contextB R, C, E (if has_context)
    """
    n_units = len(cluster_features)
    
    if feature_type == 'deformation':
        type_names = ['Rotator', 'Integrator', 'Explorer']
    else:
        type_names = ['PC1-dom', 'PC2-dom', 'PC3-dom']
    
    # Extract feature groups
    global_feats = mean_features[0:3]
    early_feats = mean_features[3:6]
    middle_feats = mean_features[6:9]
    late_feats = mean_features[9:12]
    
    # Determine dominant deformation mode globally
    global_dom_idx = np.argmax(global_feats)
    global_dom_val = global_feats[global_dom_idx]
    base_type = type_names[global_dom_idx]
    
    # Determine temporal specialization
    temporal_strengths = [
        np.max(early_feats),
        np.max(middle_feats),
        np.max(late_feats)
    ]
    temporal_max_idx = np.argmax(temporal_strengths)
    temporal_labels = ['Early', 'Middle', 'Late']
    temporal_label = temporal_labels[temporal_max_idx]
    
    # Check if temporal specialization is strong
    temporal_specialization = temporal_strengths[temporal_max_idx] - np.mean(temporal_strengths)
    has_temporal_spec = (temporal_specialization > 0.05)  # 5% above average
    
    # Contextual analysis (if available)
    context_label = ''
    if has_context:
        contextA_feats = mean_features[12:15]
        contextB_feats = mean_features[15:18]
        
        contextA_strength = np.max(contextA_feats)
        contextB_strength = np.max(contextB_feats)
        
        context_diff = abs(contextA_strength - contextB_strength)
        if context_diff > 0.08:  # Strong context preference
            if contextA_strength > contextB_strength:
                context_label = ' (Context A)'
            else:
                context_label = ' (Context B)'
    
    # Build cluster name
    if global_dom_val < weak_threshold:
        name = 'Mixed'
        confidence = 'very_low'
    elif global_dom_val < strong_threshold:
        if has_temporal_spec:
            name = f'{temporal_label} {base_type}?{context_label}'
        else:
            name = f'{base_type}?{context_label}'
        confidence = 'low'
    else:
        if has_temporal_spec:
            name = f'{temporal_label} {base_type}{context_label}'
        else:
            name = f'{base_type}{context_label}'
        confidence = 'high'
    
    # Build interpretation dict
    interpretation = {
        'name': name,
        'n_units': n_units,
        'mean_features': mean_features,
        'global_features': global_feats,
        'temporal_features': {
            'early': early_feats,
            'middle': middle_feats,
            'late': late_feats
        },
        'dominant_mode': type_names[global_dom_idx],
        'temporal_specialization': temporal_label if has_temporal_spec else 'None',
        'percentage': 100 * n_units / len(all_labels),
        'confidence': confidence,
        'max_correlation': global_dom_val
    }
    
    if has_context:
        interpretation['context_features'] = {
            'contextA': mean_features[12:15],
            'contextB': mean_features[15:18]
        }
        interpretation['context_preference'] = context_label.strip(' ()')
    
    return interpretation


def print_cluster_summary(interpretation, feature_type='deformation', normalized=True):
    """
    Print human-readable summary of cluster interpretation.
    
    Args:
        interpretation: Dict from interpret_clusters()
        feature_type: 'deformation' or 'pca' - affects labels
        normalized: If True, features are normalized ratios (sum=1)
    """
    print("\n" + "="*70)
    print("UNIT CLASSIFICATION SUMMARY")
    if feature_type != 'deformation':
        print(f"(Based on {feature_type.upper()} features - not deformation)")
    if normalized:
        print("(Features are normalized ratios - sum=1, uniform=0.333)")
    print("="*70)
    
    # Get feature labels
    if feature_type == 'deformation':
        if normalized:
            labels = ['rotation ratio', 'contraction ratio', 'expansion ratio']
        else:
            labels = ['rotation', 'contraction', 'expansion']
    else:
        labels = ['PC1', 'PC2', 'PC3']
    
    for cluster_id in sorted(interpretation.keys()):
        info = interpretation[cluster_id]
        
        print(f"\nCluster {cluster_id}: {info['name']} ({info['n_units']} units, {info['percentage']:.1f}%)")
        
        # Check if enhanced features are present
        has_temporal = 'temporal_features' in info
        has_context = 'context_features' in info
        
        if has_temporal or has_context:
            # Enhanced features - show global + temporal + context
            global_feats = info.get('global_features', info['mean_features'][:3])
            print(f"  Global {labels[0]:15s}: {global_feats[0]:.3f}")
            print(f"  Global {labels[1]:15s}: {global_feats[1]:.3f}")
            print(f"  Global {labels[2]:15s}: {global_feats[2]:.3f}")
            
            if has_temporal:
                temporal_spec = info.get('temporal_specialization', 'None')
                if temporal_spec != 'None':
                    print(f"  → Temporal specialization: {temporal_spec}")
                    temporal_data = info['temporal_features']
                    if temporal_spec == 'Early':
                        feats = temporal_data['early']
                    elif temporal_spec == 'Middle':
                        feats = temporal_data['middle']
                    else:
                        feats = temporal_data['late']
                    print(f"     {labels[0]}: {feats[0]:.3f}, {labels[1]}: {feats[1]:.3f}, {labels[2]}: {feats[2]:.3f}")
            
            if has_context:
                context_pref = info.get('context_preference', '')
                if context_pref:
                    print(f"  → Context preference: {context_pref}")
                    context_data = info['context_features']
                    if 'A' in context_pref:
                        feats = context_data['contextA']
                    else:
                        feats = context_data['contextB']
                    print(f"     {labels[0]}: {feats[0]:.3f}, {labels[1]}: {feats[1]:.3f}, {labels[2]}: {feats[2]:.3f}")
        
        else:
            # Basic features - original display
            if normalized:
                print(f"  {labels[0]:20s}: {info['mean_features'][0]:.3f}")
                print(f"  {labels[1]:20s}: {info['mean_features'][1]:.3f}")
                print(f"  {labels[2]:20s}: {info['mean_features'][2]:.3f}")
            else:
                print(f"  Mean correlation with {labels[0]:12s}: {info['mean_features'][0]:+.3f}")
                print(f"  Mean correlation with {labels[1]:12s}: {info['mean_features'][1]:+.3f}")
                print(f"  Mean correlation with {labels[2]:12s}: {info['mean_features'][2]:+.3f}")
            
            if info['dominant_mode'] is not None:
                dom_val = info['mean_features'][np.argmax(np.abs(info['mean_features'][:3]))]
                if normalized:
                    print(f"  → Dominant mode: {info['dominant_mode']} ({dom_val:.3f})")
                else:
                    print(f"  → Dominant mode: {info['dominant_mode']} ({info.get('dominant_value', dom_val):+.3f})")
    
    print("="*70)


def compare_to_baseline(features, labels, hidden_states, trial_indices=None):
    """
    Compare deformation-based clustering to baseline methods.
    
    Baselines:
    1. PCA: PCA on raw activations, then K-means
    2. Raw: K-means directly on raw activation samples
    3. TDR: Targeted Dimensionality Reduction (task-aligned PCA)
    4. Selectivity: Clustering based on unit selectivity indices
    
    Args:
        features: (n_units, 3) deformation features
        labels: (n_units,) cluster assignments from deformation method
        hidden_states: (n_units, n_timesteps) raw RNN activations
        trial_indices: Optional (n_timesteps,) array indicating trial membership
    
    Returns:
        comparison: Dict with silhouette scores and details for each method
    """
    from sklearn.cluster import KMeans
    from sklearn.preprocessing import StandardScaler
    from sklearn.metrics import silhouette_score
    from sklearn.decomposition import PCA
    
    n_units = features.shape[0]
    n_clusters = len(np.unique(labels))
    
    # Normalize deformation features
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    # Baseline 1: PCA on raw activations
    pca = PCA(n_components=3)
    pca_features = pca.fit_transform(hidden_states)
    pca_features_norm = StandardScaler().fit_transform(pca_features)
    
    kmeans_pca = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels_pca = kmeans_pca.fit_predict(pca_features_norm)
    
    # Baseline 2: K-means on raw activations (sample subset of timepoints)
    n_samples = min(1000, hidden_states.shape[1])
    sample_indices = np.linspace(0, hidden_states.shape[1]-1, n_samples, dtype=int)
    raw_features = hidden_states[:, sample_indices]
    raw_features_norm = StandardScaler().fit_transform(raw_features)
    
    kmeans_raw = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
    labels_raw = kmeans_raw.fit_predict(raw_features_norm)
    
    # Baseline 3: TDR (Targeted Dimensionality Reduction)
    try:
        tdr_features = tdr_baseline(hidden_states, trial_indices)
        tdr_features_norm = StandardScaler().fit_transform(tdr_features)
        kmeans_tdr = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels_tdr = kmeans_tdr.fit_predict(tdr_features_norm)
        silhouette_tdr = silhouette_score(tdr_features_norm, labels_tdr) if n_units > n_clusters else 0
    except:
        silhouette_tdr = None
    
    # Baseline 4: Selectivity-based clustering
    try:
        selectivity_features = selectivity_baseline(hidden_states, trial_indices)
        selectivity_features_norm = StandardScaler().fit_transform(selectivity_features)
        kmeans_sel = KMeans(n_clusters=n_clusters, random_state=42, n_init=20)
        labels_sel = kmeans_sel.fit_predict(selectivity_features_norm)
        silhouette_sel = silhouette_score(selectivity_features_norm, labels_sel) if n_units > n_clusters else 0
    except:
        silhouette_sel = None
    
    # Compute silhouette scores
    comparison = {
        'deformation': {'silhouette': silhouette_score(features_norm, labels) if n_units > n_clusters else 0},
        'pca': {'silhouette': silhouette_score(pca_features_norm, labels_pca) if n_units > n_clusters else 0},
        'raw': {'silhouette': silhouette_score(raw_features_norm, labels_raw) if n_units > n_clusters else 0}
    }
    
    if silhouette_tdr is not None:
        comparison['tdr'] = {'silhouette': silhouette_tdr}
    if silhouette_sel is not None:
        comparison['selectivity'] = {'silhouette': silhouette_sel}
    
    return comparison


def tdr_baseline(hidden_states, trial_indices=None, n_components=3):
    """
    Targeted Dimensionality Reduction (TDR) baseline.
    
    Simplified version inspired by Mante et al. 2013.
    NOTE: This is a simplified approximation. True TDR requires explicit task
    variables (stimulus identity, context, etc.) which are not always available.
    This version uses temporal structure as a proxy.
    
    Approach:
    1. Compute variance explained by temporal bins (proxy for task structure)
    2. PCA on variance-normalized activations
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN activations
        trial_indices: (n_timesteps,) array indicating trial membership (optional)
        n_components: Number of PCA components
    
    Returns:
        tdr_features: (n_units, n_components) features
    """
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    
    n_units, n_timesteps = hidden_states.shape
    
    # Simple approach: compute temporal statistics per unit
    # This captures how units vary across task epochs
    features = []
    
    # Divide into temporal bins if no trial indices
    if trial_indices is None:
        n_bins = 10
        bin_size = n_timesteps // n_bins
        trial_indices = np.repeat(np.arange(n_bins), bin_size)[:n_timesteps]
    
    unique_bins = np.unique(trial_indices)
    
    for unit_idx in range(n_units):
        unit_activity = hidden_states[unit_idx, :]
        
        # Compute mean and variance per temporal bin
        bin_means = []
        bin_vars = []
        for bin_id in unique_bins:
            bin_mask = trial_indices == bin_id
            if np.sum(bin_mask) > 0:
                bin_means.append(np.mean(unit_activity[bin_mask]))
                bin_vars.append(np.var(unit_activity[bin_mask]))
        
        # Features: variance across bins (temporal modulation)
        features.append([
            np.var(bin_means),  # Temporal modulation
            np.mean(bin_vars),  # Within-bin variability
            np.mean(np.abs(bin_means)),  # Mean activity
        ])
    
    features = np.array(features)
    
    # Standardize and apply PCA
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    pca = PCA(n_components=min(n_components, features.shape[1]))
    tdr_features = pca.fit_transform(features_norm)
    
    return tdr_features


def selectivity_baseline(hidden_states, trial_indices=None, n_bins=5):
    """
    Selectivity-based clustering baseline.
    
    Computes selectivity indices for each unit:
    - Temporal selectivity: variance across time
    - Amplitude selectivity: mean absolute activation
    - Variance selectivity: standard deviation
    - Peak selectivity: max activation amplitude
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN activations
        trial_indices: (n_timesteps,) array indicating trial membership (optional)
        n_bins: Number of temporal bins for selectivity analysis
    
    Returns:
        selectivity_features: (n_units, 4) selectivity indices
    """
    from scipy.stats import f_oneway
    
    n_units, n_timesteps = hidden_states.shape
    
    features = []
    
    for unit_idx in range(n_units):
        unit_activity = hidden_states[unit_idx, :]
        
        # Feature 1: Temporal selectivity (variance across time)
        temporal_var = np.var(unit_activity)
        
        # Feature 2: Amplitude selectivity (mean absolute activation)
        amplitude = np.mean(np.abs(unit_activity))
        
        # Feature 3: Variance selectivity (coefficient of variation)
        cv = np.std(unit_activity) / (np.abs(np.mean(unit_activity)) + 1e-10)
        
        # Feature 4: Peak selectivity (max absolute activation)
        peak = np.max(np.abs(unit_activity))
        
        features.append([temporal_var, amplitude, cv, peak])
    
    return np.array(features)


def select_features_by_task_dynamics(hidden_states, deformation_features, 
                                      task_dynamics='unknown', deformation_valid=True):
    """
    Choose features based on task dynamics type.
    
    Different tasks require different feature representations:
    - Static/Memory tasks: Use PCA or selectivity (capture state structure)
    - Oscillatory tasks: Use frequency features or deformation
    - Mixed/Unknown: Use deformation if valid, else PCA
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN activations
        deformation_features: (n_units, 3) deformation-based features (or None)
        task_dynamics: 'static', 'oscillatory', 'mixed', 'discrete', or 'unknown'
        deformation_valid: Whether deformation estimation succeeded
    
    Returns:
        features: (n_units, n_features) selected feature array
        method_used: str - name of method used
    """
    from sklearn.decomposition import PCA
    
    n_units = hidden_states.shape[0]
    
    if task_dynamics == 'static' or task_dynamics == 'discrete':
        # For memory/discrete tasks: use PCA (captures state structure)
        print("  Using PCA features (static/discrete dynamics)")
        pca = PCA(n_components=3)
        pca_features = pca.fit_transform(hidden_states)
        return pca_features, 'pca'
    
    elif task_dynamics == 'oscillatory':
        if deformation_valid and deformation_features is not None:
            # Try deformation first for oscillatory tasks
            print("  Using deformation features (oscillatory dynamics)")
            return deformation_features, 'deformation'
        else:
            # Fallback to frequency domain
            print("  Using frequency features (oscillatory dynamics, deformation failed)")
            from scipy.fft import fft
            
            freq_features = []
            for unit in hidden_states:
                fft_vals = np.abs(fft(unit))
                freq_features.append(fft_vals[:50])  # Low frequencies
            return np.array(freq_features), 'frequency'
    
    elif task_dynamics == 'mixed':
        if deformation_valid and deformation_features is not None:
            # Mixed tasks: use deformation
            print("  Using deformation features (mixed dynamics)")
            return deformation_features, 'deformation'
        else:
            print("  Using PCA features (mixed dynamics, deformation failed)")
            pca = PCA(n_components=3)
            return pca.fit_transform(hidden_states), 'pca'
    
    else:  # unknown
        if deformation_valid and deformation_features is not None:
            # Check signal strength
            avg_corr = np.mean(np.abs(deformation_features))
            if avg_corr > 0.05:
                # Strong deformation signals
                print("  Using deformation features (strong signals)")
                return deformation_features, 'deformation'
            else:
                # Weak signals - hybrid approach
                print("  Using hybrid PCA+deformation features (weak deformation signals)")
                pca = PCA(n_components=3)
                pca_features = pca.fit_transform(hidden_states)
                
                # Weight by correlation strength
                hybrid = 0.3 * deformation_features + 0.7 * pca_features
                return hybrid, 'hybrid'
        else:
            # Deformation failed - use PCA
            print("  Using PCA features (deformation unavailable)")
            pca = PCA(n_components=3)
            return pca.fit_transform(hidden_states), 'pca'
