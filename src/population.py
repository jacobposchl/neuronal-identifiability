"""
Realistic neural population simulator with mixed functional roles
"""

import numpy as np


class RealisticNeuralPopulation:
    """
    Simulates neurons with realistic properties:
    - Mixed functional roles (not pure types)
    - Variable signal-to-noise ratios
    - Non-stationary tuning
    - Diverse baseline rates and dynamics
    """
    
    def __init__(self, n_neurons=60, specialization_level='medium'):
        """
        Args:
            n_neurons: Total number of neurons
            specialization_level: 'low', 'medium', 'high'
                Controls how specialized neurons are to one function
        """
        self.n_neurons = n_neurons
        self.specialization_level = specialization_level
        
        # Each neuron has weights for: [rotation, contraction, expansion, state_encoding]
        # These determine how much each factor drives the neuron
        if specialization_level == 'high':
            # Neurons are highly specialized (one dominant factor)
            self.functional_weights = self._generate_specialized_weights(sharpness=3.0)
        elif specialization_level == 'medium':
            # Neurons have dominant role but also respond to other factors
            self.functional_weights = self._generate_specialized_weights(sharpness=1.5)
        else:  # 'low'
            # Neurons respond to everything (mixed selectivity)
            self.functional_weights = np.random.randn(n_neurons, 4) * 0.5
        
        # Ground truth labels: which function dominates each neuron
        self.true_labels = np.argmax(self.functional_weights, axis=1)
        
        # Realistic neuron properties
        self.baseline_rates = np.random.uniform(2, 25, n_neurons)  # Hz
        self.gain = np.random.uniform(0.3, 2.5, n_neurons)
        self.noise_level = np.random.uniform(0.5, 2.0, n_neurons)  # Variable SNR
        
        # State encoding: random but structured
        self.state_weights = np.random.randn(n_neurons, 3)  # Encode 3D latent state
        
        # Some neurons have non-stationary properties
        self.adaptation_rate = np.random.uniform(0, 0.1, n_neurons)
        
    def _generate_specialized_weights(self, sharpness=2.0):
        """Generate weights where each neuron has one dominant function"""
        weights = np.zeros((self.n_neurons, 4))
        
        # Assign each neuron to a dominant function
        n_per_type = self.n_neurons // 4
        assignments = []
        for i in range(4):
            assignments.extend([i] * n_per_type)
        # Fill remainder randomly
        while len(assignments) < self.n_neurons:
            assignments.append(np.random.randint(4))
        np.random.shuffle(assignments)
        
        for i, assignment in enumerate(assignments):
            # Dominant weight
            weights[i, assignment] = np.random.uniform(2.0, 4.0) * sharpness
            # Other weights (background)
            for j in range(4):
                if j != assignment:
                    weights[i, j] = np.random.uniform(-0.5, 1.0)
        
        return weights
    
    def compute_firing_rates(self, latent_state, rotation_mag, contraction_mag,
                            expansion_mag, time):
        """
        Compute firing rates with realistic complications:
        - Mixed selectivity
        - Non-stationary tuning
        - State-dependent modulation
        """
        rates = np.zeros(self.n_neurons)
        
        # Normalize deformation signals (to prevent dominance)
        deform_scale = 1.0 / (1 + np.abs(rotation_mag) + np.abs(contraction_mag) + 
                             np.abs(expansion_mag))
        
        for i in range(self.n_neurons):
            # Combine all factors with neuron's weights
            w_rot, w_con, w_exp, w_state = self.functional_weights[i]
            
            # Deformation contributions
            deform_drive = (w_rot * rotation_mag * deform_scale +
                          w_con * contraction_mag * deform_scale +
                          w_exp * expansion_mag * deform_scale)
            
            # State encoding
            state_drive = w_state * np.dot(self.state_weights[i], latent_state)
            
            # Adaptation over time (non-stationarity)
            adaptation = 1.0 - self.adaptation_rate[i] * (1 - np.exp(-time / 10))
            
            # Combined drive
            total_drive = adaptation * (deform_drive + state_drive)
            
            # Transform to rate
            rates[i] = self.baseline_rates[i] + self.gain[i] * total_drive
        
        # Rectify
        rates = np.maximum(rates, 0.1)
        
        return rates
    
    def generate_spike_trains(self, latent_trajectory, rotation_trajectory,
                             contraction_trajectory, expansion_trajectory, 
                             t_eval, dt=0.001):
        """Generate realistic spike trains"""
        n_timesteps = len(latent_trajectory)
        spike_trains = np.zeros((self.n_neurons, n_timesteps))
        firing_rates = np.zeros((self.n_neurons, n_timesteps))
        
        for t in range(n_timesteps):
            rates = self.compute_firing_rates(
                latent_trajectory[t],
                rotation_trajectory[t],
                contraction_trajectory[t],
                expansion_trajectory[t],
                t_eval[t]
            )
            
            firing_rates[:, t] = rates
            
            # Poisson spikes with realistic noise
            spike_probs = rates * dt
            spike_trains[:, t] = (np.random.rand(self.n_neurons) < spike_probs).astype(float)
        
        # Add realistic background noise
        for i in range(self.n_neurons):
            noise_spikes = np.random.poisson(self.noise_level[i] * dt, n_timesteps)
            spike_trains[i] += (noise_spikes > 0).astype(float)
            spike_trains[i] = np.clip(spike_trains[i], 0, 1)
        
        return spike_trains, firing_rates
