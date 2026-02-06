"""
Feature extraction for RNN unit classification based on deformation dynamics.

Adapts the deformation-based feature extraction approach from biological neurons
to work with continuous RNN hidden unit activations.
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d


def extract_rnn_unit_features(hidden_states, rotation_trajectory, 
                               contraction_trajectory, expansion_trajectory,
                               smooth_sigma=5):
    """
    Extract deformation-based features for RNN units.
    
    Computes correlation between each unit's activity and the three deformation
    modes (rotation, contraction, expansion) over time.
    
    Adapted from features.py::extract_deformation_features() for RNN context:
    - Input: continuous hidden states (not spike trains)
    - No downsampling needed (RNN activations already smooth)
    - Optional smoothing for noise reduction
    - Direct correlation with deformation trajectories
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN hidden unit activations
        rotation_trajectory: (n_timesteps,) rotation magnitude over time
        contraction_trajectory: (n_timesteps,) contraction magnitude over time
        expansion_trajectory: (n_timesteps,) expansion magnitude over time
        smooth_sigma: Gaussian smoothing parameter (0 = no smoothing)
    
    Returns:
        features: (n_units, 3) array where features[i] = [corr_rotation, corr_contraction, corr_expansion]
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
        
        features.append([corr_rotation, corr_contraction, corr_expansion])
    
    return np.array(features)


def classify_units(features, n_clusters=4, method='kmeans', return_details=False):
    """
    Classify RNN units based on deformation features.
    
    Args:
        features: (n_units, 3) deformation feature array
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
            silhouette = silhouette_score(features_norm, labels) if n_units > n_clusters else 0
            
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


def interpret_clusters(features, labels, cluster_names=None, threshold=0.03):
    """
    Interpret cluster assignments by analyzing mean features.
    
    Args:
        features: (n_units, 3) deformation feature array
        labels: (n_units,) cluster assignments
        cluster_names: Optional list of names (default: auto-assign based on features)
        threshold: Minimum absolute correlation to assign a type (default: 0.03)
    
    Returns:
        interpretation: Dict mapping cluster ID to interpretation dict
    """
    n_clusters = len(np.unique(labels))
    interpretation = {}
    
    feature_names = ['Rotation', 'Contraction', 'Expansion']
    
    for cluster_id in range(n_clusters):
        cluster_mask = labels == cluster_id
        n_units = np.sum(cluster_mask)
        
        if n_units == 0:
            interpretation[cluster_id] = {
                'name': 'Empty',
                'n_units': 0,
                'mean_features': np.zeros(3),
                'dominant_mode': None
            }
            continue
        
        # Compute mean features for this cluster
        cluster_features = features[cluster_mask]
        mean_features = np.mean(cluster_features, axis=0)
        
        # Determine dominant mode with threshold
        abs_features = np.abs(mean_features)
        dominant_idx = np.argmax(abs_features)
        dominant_value = mean_features[dominant_idx]
        dominant_mode = feature_names[dominant_idx]
        
        # Check if dominant feature is significantly above threshold
        max_abs_corr = abs_features[dominant_idx]
        
        # Auto-assign name if not provided
        if cluster_names is None:
            # Require minimum absolute correlation to assign specific type
            if max_abs_corr < threshold:
                name = 'Mixed'  # Too weak to classify
            elif dominant_mode == 'Rotation':
                name = 'Rotator'
            elif dominant_mode == 'Contraction':
                name = 'Integrator'
            elif dominant_mode == 'Expansion':
                name = 'Explorer'
            else:
                name = 'Mixed'
        else:
            name = cluster_names[cluster_id] if cluster_id < len(cluster_names) else f'Cluster {cluster_id}'
        
        interpretation[cluster_id] = {
            'name': name,
            'n_units': n_units,
            'mean_features': mean_features,
            'dominant_mode': dominant_mode,
            'dominant_value': dominant_value,
            'percentage': 100 * n_units / len(labels)
        }
    
    return interpretation


def print_cluster_summary(interpretation):
    """
    Print human-readable summary of cluster interpretation.
    
    Args:
        interpretation: Dict from interpret_clusters()
    """
    print("\n" + "="*70)
    print("UNIT CLASSIFICATION SUMMARY")
    print("="*70)
    
    for cluster_id in sorted(interpretation.keys()):
        info = interpretation[cluster_id]
        
        print(f"\nCluster {cluster_id}: {info['name']} ({info['n_units']} units, {info['percentage']:.1f}%)")
        print(f"  Mean correlation with rotation:    {info['mean_features'][0]:+.3f}")
        print(f"  Mean correlation with contraction: {info['mean_features'][1]:+.3f}")
        print(f"  Mean correlation with expansion:   {info['mean_features'][2]:+.3f}")
        
        if info['dominant_mode'] is not None:
            print(f"  → Dominant mode: {info['dominant_mode']} ({info['dominant_value']:+.3f})")
    
    print("="*70)


def compare_to_baseline(features, labels, hidden_states):
    """
    Compare deformation-based clustering to baseline methods.
    
    Args:
        features: (n_units, 3) deformation features
        labels: (n_units,) cluster assignments from deformation method
        hidden_states: (n_units, n_timesteps) raw RNN activations
    
    Returns:
        comparison: Dict with silhouette scores for each method
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
    
    # Compute silhouette scores
    comparison = {
        'deformation': silhouette_score(features_norm, labels) if n_units > n_clusters else 0,
        'pca': silhouette_score(pca_features_norm, labels_pca) if n_units > n_clusters else 0,
        'raw': silhouette_score(raw_features_norm, labels_raw) if n_units > n_clusters else 0
    }
    
    return comparison
