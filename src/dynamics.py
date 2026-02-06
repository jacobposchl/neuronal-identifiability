"""
Neural dynamics generation with latent state space and deformation metrics
"""

import numpy as np
from scipy.integrate import solve_ivp


def generate_complex_dynamics(T=15.0, dt=0.002):
    """
    Generate more complex, realistic latent dynamics with multiple regimes:
    - Rotation phases (cyclic, oscillatory dynamics)
    - Contraction phases (convergence to attractors, decision commitment)
    - Expansion phases (divergent dynamics, state exploration)
    - Mixed phases (combination of the above)
    
    Args:
        T: Total simulation time
        dt: Time step for integration
        
    Returns:
        Tuple of:
        - latent_trajectory: (n_timesteps, 3) latent state positions
        - rotation_trajectory: (n_timesteps,) rotation magnitude at each time
        - contraction_trajectory: (n_timesteps,) contraction magnitude at each time
        - expansion_trajectory: (n_timesteps,) expansion magnitude at each time
        - t_eval: time points
    """
    
    def vector_field(t, z):
        # Time-varying dynamics (more realistic)
        phase = t % 10
        
        if phase < 2.5:
            # Rotation phase
            r = np.sqrt(z[0]**2 + z[1]**2)
            dr = 0.4 * (1.8 - r)
            omega = 2.5
            return np.array([
                -omega * z[1] + dr * z[0] / (r + 0.1),
                omega * z[0] + dr * z[1] / (r + 0.1),
                -0.6 * z[2]
            ])
        elif phase < 5.0:
            # Contraction phase (decision commitment)
            target = np.array([0.5, 0.5, 0.3])
            return -1.8 * (z - target)
        elif phase < 7.5:
            # Mixed phase (expansion in some dims, contraction in others)
            return np.array([0.6*z[0], -0.6*z[1], -1.2*z[2]])
        else:
            # Return to rotation
            r = np.sqrt(z[0]**2 + z[1]**2)
            dr = 0.3 * (1.5 - r)
            omega = 3.0
            return np.array([
                -omega * z[1] + dr * z[0] / (r + 0.1),
                omega * z[0] + dr * z[1] / (r + 0.1),
                -0.5 * z[2]
            ])
    
    z0 = np.array([1.2, 0.6, 0.4])
    t_eval = np.arange(0, T, dt)
    sol = solve_ivp(vector_field, [0, T], z0, t_eval=t_eval, method='RK45')
    latent_trajectory = sol.y.T
    
    # Compute deformation
    rotation_trajectory = np.zeros(len(t_eval))
    contraction_trajectory = np.zeros(len(t_eval))
    expansion_trajectory = np.zeros(len(t_eval))
    
    for i, t in enumerate(t_eval):
        z = latent_trajectory[i]
        v = vector_field(t, z)
        
        # Numerical Jacobian via finite differences
        eps = 1e-6
        J = np.zeros((3, 3))
        for j in range(3):
            z_plus = z.copy()
            z_plus[j] += eps
            v_plus = vector_field(t, z_plus)
            J[:, j] = (v_plus - v) / eps
        
        # Decompose into symmetric and antisymmetric parts
        S = 0.5 * (J + J.T)  # Symmetric: expansion/contraction
        A = 0.5 * (J - J.T)  # Antisymmetric: rotation
        
        eigenvalues_S = np.linalg.eigvalsh(S)
        
        # Extract deformation metrics
        rotation_trajectory[i] = np.linalg.norm(A, 'fro')
        contraction_trajectory[i] = -np.sum(eigenvalues_S[eigenvalues_S < 0])
        expansion_trajectory[i] = np.sum(eigenvalues_S[eigenvalues_S > 0])
    
    return (latent_trajectory, rotation_trajectory, contraction_trajectory,
            expansion_trajectory, t_eval)


def estimate_deformation_from_latents(latent_trajectory, dt, n_neighbors=20, n_samples=200):
    """Estimate rotation/contraction/expansion from a latent trajectory.

    Uses local linear Jacobian estimation via nearest neighbors, then decomposes
    into symmetric/antisymmetric parts. This mimics a realistic scenario where
    deformation is estimated from inferred latents rather than known dynamics.
    
    Args:
        latent_trajectory: (n_timesteps, latent_dim) array of latent positions
        dt: Time step
        n_neighbors: Number of neighbors for local estimation
        n_samples: Number of sample points for estimation
        
    Returns:
        Tuple of:
        - rotation_full: (n_timesteps,) rotation trajectory
        - contraction_full: (n_timesteps,) contraction trajectory
        - expansion_full: (n_timesteps,) expansion trajectory
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_timesteps, latent_dim = latent_trajectory.shape
    velocities = np.gradient(latent_trajectory, axis=0) / dt

    n_neighbors = min(n_neighbors, n_timesteps)
    nbrs = NearestNeighbors(n_neighbors=n_neighbors).fit(latent_trajectory)

    n_samples = min(n_samples, n_timesteps)
    sample_indices = np.linspace(0, n_timesteps - 1, n_samples, dtype=int)

    rotation_ts = []
    contraction_ts = []
    expansion_ts = []
    sample_times = []

    for idx in sample_indices:
        z = latent_trajectory[idx]
        distances, indices = nbrs.kneighbors([z])

        neighbor_points = latent_trajectory[indices[0]]
        neighbor_velocities = velocities[indices[0]]

        delta_z = neighbor_points - z
        delta_v = neighbor_velocities - velocities[idx]

        J = np.zeros((latent_dim, latent_dim))
        for d in range(latent_dim):
            J[d, :] = np.linalg.lstsq(delta_z, delta_v[:, d], rcond=None)[0]

        S = 0.5 * (J + J.T)
        A = 0.5 * (J - J.T)

        rotation = np.linalg.norm(A, 'fro')
        eigenvalues_S = np.linalg.eigvalsh(S)
        contraction = -np.sum(eigenvalues_S[eigenvalues_S < 0])
        expansion = np.sum(eigenvalues_S[eigenvalues_S > 0])

        rotation_ts.append(rotation)
        contraction_ts.append(contraction)
        expansion_ts.append(expansion)
        sample_times.append(idx)

    sample_times = np.array(sample_times)
    rotation_ts = np.array(rotation_ts)
    contraction_ts = np.array(contraction_ts)
    expansion_ts = np.array(expansion_ts)

    if len(sample_times) < 2:
        return (np.zeros(n_timesteps), np.zeros(n_timesteps), np.zeros(n_timesteps))

    full_times = np.arange(n_timesteps)
    rotation_full = np.interp(full_times, sample_times, rotation_ts)
    contraction_full = np.interp(full_times, sample_times, contraction_ts)
    expansion_full = np.interp(full_times, sample_times, expansion_ts)

    return rotation_full, contraction_full, expansion_full
