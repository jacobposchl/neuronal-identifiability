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
    if np.all(trajectory_std < 1e-8):
        # Constant trajectory - return small random noise instead of zeros
        # This prevents NaN correlations and clustering failures
        print("  Warning: Latent trajectory is constant (no dynamics detected)")
        print("  Returning minimal random deformation signals as fallback")
        np.random.seed(42)
        rotation_traj = np.random.randn(n_timesteps) * 1e-6
        contraction_traj = np.random.randn(n_timesteps) * 1e-6
        expansion_traj = np.random.randn(n_timesteps) * 1e-6
        return rotation_traj, contraction_traj, expansion_traj
    
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
    if (np.all(rotation_trajectory == 0) and 
        np.all(contraction_trajectory == 0) and 
        np.all(expansion_trajectory == 0)):
        print("  Warning: Deformation estimation produced all zeros")
        print("  Returning minimal random signals as fallback")
        np.random.seed(42)
        rotation_trajectory = np.random.randn(n_timesteps) * 1e-6
        contraction_trajectory = np.random.randn(n_timesteps) * 1e-6
        expansion_trajectory = np.random.randn(n_timesteps) * 1e-6
    
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
        print(f\"  Warning: Hidden states have very low temporal variance ({mean_temporal_var:.2e})\")\n        print(f\"  RNN may have learned a trivial solution with minimal dynamics\")\n    
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
    
    Args:
        rotation, contraction, expansion: (n_timesteps,) deformation trajectories
        sigma: Standard deviation of Gaussian kernel
    
    Returns:
        Smoothed rotation, contraction, expansion trajectories
    """
    from scipy.ndimage import gaussian_filter1d
    
    rotation_smooth = gaussian_filter1d(rotation, sigma=sigma)
    contraction_smooth = gaussian_filter1d(contraction, sigma=sigma)
    expansion_smooth = gaussian_filter1d(expansion, sigma=sigma)
    
    return rotation_smooth, contraction_smooth, expansion_smooth
