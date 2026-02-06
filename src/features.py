"""
Feature extraction methods for neuron classification
"""

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import correlate
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def extract_pca_features(spike_trains, firing_rates, n_components=5):
    """
    PCA-based features: how neurons load on population PCs
    
    Args:
        spike_trains: (n_neurons, n_timesteps)
        firing_rates: (n_neurons, n_timesteps)
        n_components: number of PCA components
        
    Returns:
        features: (n_neurons, n_components + 1) array
    """
    # Downsample by 10 for efficiency
    downsampled = firing_rates[:, ::10]
    rates_smooth = np.array([gaussian_filter1d(downsampled[i], sigma=5) 
                            for i in range(len(downsampled))])
    
    # PCA on time
    pca = PCA(n_components=n_components)
    pca_proj = pca.fit_transform(rates_smooth.T)
    
    # Features: how much each neuron loads on each PC
    loadings = pca.components_.T  # (n_neurons, n_components)
    
    # Add variance explained by each neuron
    variance_per_neuron = np.var(rates_smooth, axis=1, keepdims=True)
    
    features = np.hstack([loadings, variance_per_neuron])
    return features


def extract_crosscorr_features(spike_trains, firing_rates, n_lags=100):
    """
    Cross-correlation features: how neurons relate to population average
    
    Args:
        spike_trains: (n_neurons, n_timesteps)
        firing_rates: (n_neurons, n_timesteps)
        n_lags: window size for cross-correlation
        
    Returns:
        features: (n_neurons, 2) array [max_correlation, lag]
    """
    # Downsample for efficiency
    downsampled = firing_rates[:, ::10]
    rates_smooth = np.array([gaussian_filter1d(downsampled[i], sigma=5) 
                            for i in range(len(downsampled))])
    
    # Population average
    pop_avg = np.mean(rates_smooth, axis=0)
    
    features = []
    for i in range(len(firing_rates)):
        # Cross-correlation with population (normalized)
        xcorr = correlate(rates_smooth[i], pop_avg, mode='same')
        norm_factor = np.std(rates_smooth[i]) * np.std(pop_avg) * len(pop_avg)
        xcorr = xcorr / norm_factor if norm_factor > 0 else xcorr
        
        # Features from xcorr
        peak_idx = len(xcorr) // 2
        lag_range = slice(peak_idx - n_lags//2, peak_idx + n_lags//2)
        xcorr_window = xcorr[lag_range]
        
        max_corr = np.max(xcorr_window)
        lag_of_max = np.argmax(xcorr_window) - n_lags//2
        
        features.append([max_corr, lag_of_max])
    
    return np.array(features)


def extract_dimensionality_features(spike_trains, firing_rates):
    """
    Features based on dimensionality and variance structure
    
    Args:
        spike_trains: (n_neurons, n_timesteps)
        firing_rates: (n_neurons, n_timesteps)
        
    Returns:
        features: (n_neurons, 6) array [variance, cv, autocorr_short, 
                                        autocorr_long, low_freq_power, high_freq_power]
    """
    # Downsample for efficiency
    downsampled = firing_rates[:, ::10]
    rates_smooth = np.array([gaussian_filter1d(downsampled[i], sigma=5) 
                            for i in range(len(downsampled))])
    
    features = []
    
    for i in range(len(firing_rates)):
        rate = rates_smooth[i]
        
        # Variance and coefficient of variation
        var = np.var(rate)
        cv = np.std(rate) / (np.mean(rate) + 1e-10)
        
        # Temporal autocorrelation (fast vs slow dynamics)
        autocorr_short = np.corrcoef(rate[:-50], rate[50:])[0, 1] if len(rate) > 50 else 0.0
        autocorr_long = np.corrcoef(rate[:-200], rate[200:])[0, 1] if len(rate) > 200 else 0.0
        
        # Handle NaN cases
        autocorr_short = 0.0 if np.isnan(autocorr_short) else autocorr_short
        autocorr_long = 0.0 if np.isnan(autocorr_long) else autocorr_long
        
        # Frequency content (FFT)
        fft = np.fft.fft(rate - np.mean(rate))
        power_spectrum = np.abs(fft[:len(fft)//2])**2
        low_freq_power = np.sum(power_spectrum[:10])
        high_freq_power = np.sum(power_spectrum[10:50])
        
        features.append([var, cv, autocorr_short, autocorr_long, 
                        low_freq_power, high_freq_power])
    
    return np.array(features)


def extract_deformation_features(spike_trains, firing_rates, latent_trajectory, dt,
                               rotation_trajectory=None, contraction_trajectory=None, 
                               expansion_trajectory=None):
    """
    Deformation features from pre-computed dynamics.
    
    Uses analytically-correct deformation metrics from the dynamics generator
    instead of recomputing via finite differences (which introduces numerical errors).
    
    Args:
        spike_trains: (n_neurons, n_timesteps)
        firing_rates: (n_neurons, n_timesteps)
        latent_trajectory: (n_timesteps, latent_dim) latent state positions (unused but kept for API compatibility)
        dt: time step (unused but kept for API compatibility)
        rotation_trajectory: (n_timesteps,) rotation magnitude at each time (from dynamics.py)
        contraction_trajectory: (n_timesteps,) contraction magnitude at each time (from dynamics.py)
        expansion_trajectory: (n_timesteps,) expansion magnitude at each time (from dynamics.py)
        
    Returns:
        features: (n_neurons, 3) array [rotation, contraction, expansion]
                  Each feature is the correlation between neuron firing and that deformation mode
    """
    if rotation_trajectory is None or contraction_trajectory is None or expansion_trajectory is None:
        raise ValueError(
            "Must provide rotation_trajectory, contraction_trajectory, and expansion_trajectory "
            "from generate_complex_dynamics(). These are the analytically-correct deformation metrics."
        )
    
    n_neurons = spike_trains.shape[0]
    
    # Downsample firing rates for efficiency (match 10x downsampling)
    downsampled = firing_rates[:, ::10]
    rates_smooth = np.array([gaussian_filter1d(downsampled[i], sigma=5) 
                            for i in range(n_neurons)])
    
    # Downsample deformation trajectories to match firing rates
    # Firing rates are at dt=0.001, dynamics are at dt=0.002, then we downsample by 10
    # So we need to match the lengths
    n_rates_steps = rates_smooth.shape[1]
    n_dynamics_steps = len(rotation_trajectory)
    
    # Resample dynamics to match firing rates length
    if n_dynamics_steps != n_rates_steps:
        # Simple resampling via linear interpolation
        dynamics_indices = np.linspace(0, n_dynamics_steps - 1, n_rates_steps)
        rotation_resampled = np.interp(dynamics_indices, np.arange(n_dynamics_steps), rotation_trajectory)
        contraction_resampled = np.interp(dynamics_indices, np.arange(n_dynamics_steps), contraction_trajectory)
        expansion_resampled = np.interp(dynamics_indices, np.arange(n_dynamics_steps), expansion_trajectory)
    else:
        rotation_resampled = rotation_trajectory
        contraction_resampled = contraction_trajectory
        expansion_resampled = expansion_trajectory
    
    # Correlate each neuron with each deformation mode
    features = []
    
    for i in range(n_neurons):
        rates_neuron = rates_smooth[i]
        
        # Ensure same length
        min_len = min(len(rates_neuron), len(rotation_resampled))
        rates_neuron = rates_neuron[:min_len]
        rotation_ts = rotation_resampled[:min_len]
        contraction_ts = contraction_resampled[:min_len]
        expansion_ts = expansion_resampled[:min_len]
        
        # Correlate with each deformation mode
        corr_rotation = np.corrcoef(rates_neuron, rotation_ts)[0, 1]
        corr_contraction = np.corrcoef(rates_neuron, contraction_ts)[0, 1]
        corr_expansion = np.corrcoef(rates_neuron, expansion_ts)[0, 1]
        
        # Handle NaN cases (constant signals)
        corr_rotation = 0 if np.isnan(corr_rotation) else corr_rotation
        corr_contraction = 0 if np.isnan(corr_contraction) else corr_contraction
        corr_expansion = 0 if np.isnan(corr_expansion) else corr_expansion
        
        features.append([corr_rotation, corr_contraction, corr_expansion])
    
    return np.array(features)
