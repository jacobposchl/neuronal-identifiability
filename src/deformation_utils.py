"""
Deformation computation utilities for both synthetic and RNN data.

Provides Jacobian decomposition into rotation, contraction, and expansion
components, with support for both analytical (PyTorch) and numerical approaches.
"""

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


def decompose_jacobian(J):
    """
    Decompose Jacobian matrix into rotation, contraction, and expansion.
    
    Mathematical decomposition:
        J = S + A where
        S = 0.5 * (J + J^T) is the symmetric part (stretch/compression)
        A = 0.5 * (J - J^T) is the antisymmetric part (rotation)
    
    Args:
        J: Jacobian matrix, shape (n, n), either numpy array or torch tensor
    
    Returns:
        rotation: Magnitude of rotational component (Frobenius norm of A)
        contraction: Sum of negative eigenvalues of S (compression strength)
        expansion: Sum of positive eigenvalues of S (expansion strength)
    """
    # Handle both numpy and torch
    is_torch = isinstance(J, torch.Tensor)
    if is_torch:
        J_np = J.detach().cpu().numpy()
    else:
        J_np = J
    
    # Symmetric and antisymmetric parts
    S = 0.5 * (J_np + J_np.T)
    A = 0.5 * (J_np - J_np.T)
    
    # Rotation: Frobenius norm of antisymmetric part
    rotation = np.linalg.norm(A, 'fro')
    
    # Contraction/Expansion: eigenvalues of symmetric part
    eigenvalues = np.linalg.eigvalsh(S)
    contraction = -np.sum(eigenvalues[eigenvalues < 0])
    expansion = np.sum(eigenvalues[eigenvalues > 0])
    
    return rotation, contraction, expansion


def compute_jacobian_analytical(rnn, hidden_state, input_vec=None):
    """
    Compute Jacobian using PyTorch automatic differentiation.
    
    For vanilla RNN: dh_next/dh where h_next = f(h, x)
    For LSTM/GRU: Similar but with additional state variables
    
    Args:
        rnn: PyTorch RNN model with step() method
        hidden_state: Current hidden state, shape (hidden_size,) or (batch, hidden_size)
        input_vec: Input vector, shape (input_size,) or (batch, input_size)
                   If None, uses zero input
    
    Returns:
        J: Jacobian matrix (hidden_size, hidden_size) as numpy array
    """
    # Ensure hidden_state is a tensor with grad
    if not isinstance(hidden_state, torch.Tensor):
        hidden_state = torch.tensor(hidden_state, dtype=torch.float32)
    
    hidden_state = hidden_state.detach().clone().requires_grad_(True)
    
    # Default to zero input if not provided
    if input_vec is None:
        input_vec = torch.zeros(rnn.input_size)
    elif not isinstance(input_vec, torch.Tensor):
        input_vec = torch.tensor(input_vec, dtype=torch.float32)
    
    # Handle LSTM (returns tuple)
    if hasattr(rnn, 'lstm'):
        # For LSTM, we compute Jacobian of hidden state only (not cell state)
        # This is a simplification - full analysis would include cell state
        cell_state = torch.zeros_like(hidden_state)
        
        def step_fn(h):
            c = torch.zeros_like(h)
            h_next, _ = rnn.step(input_vec, h, c)
            return h_next
        
        J = torch.autograd.functional.jacobian(step_fn, hidden_state)
    
    # Handle vanilla RNN or GRU
    else:
        def step_fn(h):
            return rnn.step(input_vec, h)
        
        J = torch.autograd.functional.jacobian(step_fn, hidden_state)
    
    return J.detach().cpu().numpy()


def compute_jacobian_numerical(dynamics_fn, state, eps=1e-6):
    """
    Compute Jacobian using finite differences (fallback method).
    
    Args:
        dynamics_fn: Function that takes state and returns next state
        state: Current state, shape (n,)
        eps: Perturbation size for finite differences
    
    Returns:
        J: Jacobian matrix (n, n) as numpy array
    """
    state = np.array(state)
    n = len(state)
    J = np.zeros((n, n))
    
    # Baseline
    f0 = dynamics_fn(state)
    
    # Perturb each dimension
    for i in range(n):
        state_plus = state.copy()
        state_plus[i] += eps
        f_plus = dynamics_fn(state_plus)
        
        J[:, i] = (f_plus - f0) / eps
    
    return J


def estimate_deformation_from_latents(latent_trajectory, dt=0.01, n_samples=200, 
                                      k_neighbors=20, method='local_linear'):
    """
    Estimate deformation signals from latent trajectory.
    
    This mimics a realistic scenario where we must estimate deformation
    from inferred latent dynamics (e.g., from PCA of neural activity).
    
    Args:
        latent_trajectory: (n_timesteps, latent_dim) trajectory in latent space
        dt: Timestep size
        n_samples: Number of points to sample for Jacobian estimation
        k_neighbors: Number of neighbors for local linear regression
        method: 'local_linear' or 'gradient' (gradient is faster but noisier)
    
    Returns:
        rotation_trajectory: (n_timesteps,) rotation magnitude over time
        contraction_trajectory: (n_timesteps,) contraction magnitude over time
        expansion_trajectory: (n_timesteps,) expansion magnitude over time
    """
    n_timesteps, latent_dim = latent_trajectory.shape
    
    # Check for constant or near-constant trajectory (no dynamics)
    trajectory_std = np.std(latent_trajectory, axis=0)
    if np.all(trajectory_std < 1e-6):  # Relaxed from 1e-8
        print("  Warning: Latent trajectory is constant (no dynamics detected)")
        print("  Deformation estimation failed - returning None")
        return None, None, None
    
    # Sample points for Jacobian estimation (expensive to compute at every point)
    sample_indices = np.linspace(0, n_timesteps - 1, min(n_samples, n_timesteps), dtype=int)
    
    rotation_samples = []
    contraction_samples = []
    expansion_samples = []
    
    if method == 'local_linear':
        # Compute velocities
        velocities = np.gradient(latent_trajectory, axis=0) / dt
        
        # Fit nearest neighbors
        nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, n_timesteps))
        nbrs.fit(latent_trajectory)
        
        # Estimate Jacobian at sampled points
        for idx in sample_indices:
            # Find k nearest neighbors
            distances, indices = nbrs.kneighbors([latent_trajectory[idx]])
            
            # Local linear regression
            neighbor_points = latent_trajectory[indices[0]]
            neighbor_velocities = velocities[indices[0]]
            
            delta_z = neighbor_points - latent_trajectory[idx]
            delta_v = neighbor_velocities - velocities[idx]
            
            # Solve for Jacobian: v = J * z
            J = np.zeros((latent_dim, latent_dim))
            for d in range(latent_dim):
                # Least squares fit
                try:
                    J[d, :] = np.linalg.lstsq(delta_z, delta_v[:, d], rcond=None)[0]
                except:
                    J[d, :] = 0  # Fallback if singular
            
            # Decompose
            rotation, contraction, expansion = decompose_jacobian(J)
            rotation_samples.append(rotation)
            contraction_samples.append(contraction)
            expansion_samples.append(expansion)
    
    elif method == 'gradient':
        # Simpler but noisier: estimate Jacobian from gradient of velocity
        velocities = np.gradient(latent_trajectory, axis=0) / dt
        
        for idx in sample_indices:
            if idx > 0 and idx < n_timesteps - 1:
                # Estimate Jacobian from velocity gradient
                dv_dz = np.gradient(velocities[max(0, idx-5):min(n_timesteps, idx+5)], axis=0)
                J = dv_dz[min(5, idx)]  # Central difference
                
                rotation, contraction, expansion = decompose_jacobian(J)
                rotation_samples.append(rotation)
                contraction_samples.append(contraction)
                expansion_samples.append(expansion)
            else:
                rotation_samples.append(0)
                contraction_samples.append(0)
                expansion_samples.append(0)
    
    # Interpolate to full timeline
    sample_times = sample_indices
    full_times = np.arange(n_timesteps)
    
    rotation_trajectory = np.interp(full_times, sample_times, rotation_samples)
    contraction_trajectory = np.interp(full_times, sample_times, contraction_samples)
    expansion_trajectory = np.interp(full_times, sample_times, expansion_samples)
    
    # Check if all signals are zero (estimation failed)
    if (np.all(np.abs(rotation_trajectory) < 1e-10) and 
        np.all(np.abs(contraction_trajectory) < 1e-10) and 
        np.all(np.abs(expansion_trajectory) < 1e-10)):
        print("  Warning: Deformation estimation produced all zeros")
        print("  Deformation estimation failed - returning None")
        return None, None, None
    
    return rotation_trajectory, contraction_trajectory, expansion_trajectory


def estimate_deformation_from_rnn(hidden_states, rnn=None, dt=0.01, 
                                   latent_dim=3, method='pca_then_local'):
    """
    Estimate deformation signals from RNN hidden state trajectories.
    
    This is the main function for RNN deformation analysis. It combines:
    1. Dimensionality reduction (PCA) to get low-dim latent trajectory
    2. Deformation estimation via local linear regression
    
    Args:
        hidden_states: (n_units, n_timesteps) RNN hidden activations
        rnn: Optional RNN model for analytical Jacobian (not used if None)
        dt: Timestep size
        latent_dim: Dimensionality of latent space (default 3 for R, C, E)
        method: 'pca_then_local' (estimate from PCA) or 'analytical' (use RNN Jacobian)
    
    Returns:
        rotation_trajectory: (n_timesteps,) rotation magnitude over time
        contraction_trajectory: (n_timesteps,) contraction magnitude over time
        expansion_trajectory: (n_timesteps,) expansion magnitude over time
        latent_trajectory: (n_timesteps, latent_dim) for visualization
    """
    n_units, n_timesteps = hidden_states.shape
    
    # Diagnostic: check if hidden states have temporal variance
    temporal_variance = np.var(hidden_states, axis=1)  # Variance over time for each unit
    mean_temporal_var = np.mean(temporal_variance)
    if mean_temporal_var < 1e-6:
        print(f"  Warning: Hidden states have very low temporal variance ({mean_temporal_var:.2e})")
        print(f"  RNN may have learned a trivial solution with minimal dynamics")
    
    if method == 'pca_then_local':
        # Project to low-dimensional latent space
        pca = PCA(n_components=latent_dim)
        latent_trajectory = pca.fit_transform(hidden_states.T)  # (n_timesteps, latent_dim)
        
        # Estimate deformation from latent trajectory
        rotation_traj, contraction_traj, expansion_traj = estimate_deformation_from_latents(
            latent_trajectory, dt=dt
        )
        
        return rotation_traj, contraction_traj, expansion_traj, latent_trajectory
    
    elif method == 'analytical' and rnn is not None:
        # Use analytical Jacobian from RNN
        # Sample points for efficiency
        n_samples = min(200, n_timesteps)
        sample_indices = np.linspace(0, n_timesteps - 1, n_samples, dtype=int)
        
        rotation_samples = []
        contraction_samples = []
        expansion_samples = []
        
        for idx in sample_indices:
            # Get hidden state at this time
            h = hidden_states[:, idx]
            
            # Compute Jacobian analytically
            try:
                J = compute_jacobian_analytical(rnn, h)
                rotation, contraction, expansion = decompose_jacobian(J)
            except:
                # Fallback to zeros if computation fails
                rotation, contraction, expansion = 0, 0, 0
            
            rotation_samples.append(rotation)
            contraction_samples.append(contraction)
            expansion_samples.append(expansion)
        
        # Interpolate to full timeline
        full_times = np.arange(n_timesteps)
        rotation_traj = np.interp(full_times, sample_indices, rotation_samples)
        contraction_traj = np.interp(full_times, sample_indices, contraction_samples)
        expansion_traj = np.interp(full_times, sample_indices, expansion_samples)
        
        # Also compute latent for visualization
        pca = PCA(n_components=latent_dim)
        latent_trajectory = pca.fit_transform(hidden_states.T)
        
        return rotation_traj, contraction_traj, expansion_traj, latent_trajectory
    
    else:
        raise ValueError(f"Unknown method: {method}. Use 'pca_then_local' or 'analytical'")


def smooth_deformation_signals(rotation, contraction, expansion, sigma=5):
    """
    Smooth deformation signals with Gaussian filter.
    
    Applies robust scaling before smoothing to handle extreme outliers.
    
    Args:
        rotation, contraction, expansion: (n_timesteps,) deformation trajectories
                                         Can be None if estimation failed
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Smoothed rotation, contraction, expansion trajectories, or None if inputs are None
    """
    # Handle None inputs (failed estimation)
    if rotation is None or contraction is None or expansion is None:
        return None, None, None
    
    from scipy.ndimage import gaussian_filter1d
    from sklearn.preprocessing import RobustScaler
    
    # Apply robust scaling to handle extreme outliers
    scaler = RobustScaler()
    
    # Reshape for scaler (needs 2D)
    rotation_scaled = scaler.fit_transform(rotation.reshape(-1, 1)).flatten()
    contraction_scaled = scaler.fit_transform(contraction.reshape(-1, 1)).flatten()
    expansion_scaled = scaler.fit_transform(expansion.reshape(-1, 1)).flatten()
    
    # Apply smoothing
    rotation_smooth = gaussian_filter1d(rotation_scaled, sigma=sigma)
    contraction_smooth = gaussian_filter1d(contraction_scaled, sigma=sigma)
    expansion_smooth = gaussian_filter1d(expansion_scaled, sigma=sigma)
    
    return rotation_smooth, contraction_smooth, expansion_smooth


def detect_discrete_dynamics(latent_trajectory, threshold=0.1):
    """
    Detect if dynamics are discrete state transitions vs continuous flow.
    
    Discrete dynamics: velocity alternates between ~0 (stable) and high (transition)
    Continuous dynamics: velocity varies smoothly
    
    Args:
        latent_trajectory: (n_timesteps, latent_dim) trajectory
        threshold: Velocity threshold for "in transition"
    
    Returns:
        is_discrete: bool - True if discrete dynamics detected
        velocity_mag: (n_timesteps,) velocity magnitude time series
        diagnostics: Dict with additional info
    \"\"\"\n    velocities = np.gradient(latent_trajectory, axis=0)\n    velocity_mag = np.linalg.norm(velocities, axis=1)
    \n    # Discrete transitions: velocity alternates between ~0 and high\n    high_vel_frac = np.mean(velocity_mag > threshold)\n    \n    # Additional check: bimodal velocity distribution\n    low_vel_frac = np.mean(velocity_mag < threshold/10)\n    \n    # If 5-20% transitioning and 60-95% stable → discrete\n    is_discrete = (0.05 < high_vel_frac < 0.20) and (low_vel_frac > 0.60)\n    \n    diagnostics = {\n        'mean_velocity': np.mean(velocity_mag),\n        'velocity_std': np.std(velocity_mag),\n        'high_vel_fraction': high_vel_frac,\n        'low_vel_fraction': low_vel_frac,\n        'dynamics_type': 'discrete' if is_discrete else 'continuous'\n    }\n    \n    return is_discrete, velocity_mag, diagnostics


def validate_task_dynamics(task_name, deformation_signals, hidden_states, latent_trajectory=None):
    """
    Validate that learned dynamics match task expectations.
    
    Args:
        task_name: Name of task ('flipflop', 'cycling', 'context', etc.)
        deformation_signals: Tuple of (rotation, contraction, expansion) or (None, None, None)
        hidden_states: (n_units, n_timesteps) RNN activations
        latent_trajectory: Optional (n_timesteps, latent_dim) trajectory
    
    Returns:
        valid: bool - True if no major issues  
        issues: List of str - Problems found
        suggestions: List of str - Recommended fixes
    \"\"\"\n    issues = []\n    suggestions = []\n    \n    rot, con, exp = deformation_signals\n    \n    # Check 1: Deformation estimation success\n    if rot is None or con is None or exp is None:\n        issues.append(\"Deformation estimation failed (returned None)\")\n        suggestions.append(\"Network may have learned discrete states instead of continuous dynamics\")\n        suggestions.append(\"Try: Different architecture (GRU/LSTM), increase training variance, or use PCA features\")\n        return False, issues, suggestions\n    \n    # Check 2: Signal strength\n    avg_magnitude = np.mean([np.std(rot), np.std(con), np.std(exp)])\n    if avg_magnitude < 0.01:\n        issues.append(f\"Deformation signals are very weak (std={avg_magnitude:.2e})\")\n        suggestions.append(\"Weak deformation suggests minimal continuous dynamics\")\n        suggestions.append(\"Consider using task-specific features (PCA, selectivity)\")\n    \n    # Check 3: Discrete dynamics detection\n    if latent_trajectory is not None:\n        is_discrete, _, diag = detect_discrete_dynamics(latent_trajectory)\n        if is_discrete:\n            issues.append(f\"Discrete dynamics detected ({diag['dynamics_type']})\")\n            suggestions.append(\"Network uses discrete state-switching rather than continuous flow\")\n            suggestions.append(\"Deformation-based method may not be appropriate for this task\")\n    \n    # Check 4: Task-specific expectations\n    task_lower = task_name.lower()\n    \n    if 'cycling' in task_lower or 'oscillat' in task_lower:\n        # Should have significant rotation\n        if np.std(rot) < 0.1 * max(np.std(con), np.std(exp)):\n            issues.append(\"Expected rotation-dominant dynamics for cycling task\")\n            suggestions.append(\"Network may be using discrete states instead of rotation\")\n            suggestions.append(\"Try: Continuous-time RNN, add recurrent noise, or different task parameters\")\n    \n    elif 'flipflop' in task_lower or 'memory' in task_lower:\n        # Expect stable states (low dynamics during maintenance)\n        total_dynamics = avg_magnitude\n        hidden_variance = np.mean(np.var(hidden_states, axis=1))\n        \n        if total_dynamics > 0.3 * hidden_variance:\n            issues.append(\"Memory task shows high dynamics (expected stable states)\")\n    \n    elif 'context' in task_lower or 'integration' in task_lower:\n        # Expect mixed dynamics with expansion during transitions\n        if np.std(exp) < 0.5 * np.std(con):\n            issues.append(\"Expected significant expansion for context transitions\")\n    \n    valid = len(issues) == 0\n    return valid, issues, suggestions
