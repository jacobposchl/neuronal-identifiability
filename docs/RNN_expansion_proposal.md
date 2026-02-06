# RNN Testing Extension Guide

## Project Overview

### What This Method Does

**Core Idea**: Classify neurons (or RNN units) by their role in generating population dynamics based on geometric properties of the flow in latent space.

**The Three Dynamical Roles**:
1. **Rotators** - Drive cyclic/oscillatory dynamics (preparation, deliberation)
2. **Integrators** - Drive convergence to stable states (decision commitment, memory maintenance)
3. **Explorers** - Drive divergence/expansion (exploration, instability)

**How It Works**:
```
RNN hidden states → Low-dim latent trajectory → Compute Jacobian → Decompose into R(t), C(t), E(t) → Correlate with unit activity → Cluster units
```

### Why Test on RNNs First?

**Advantages over biological data**:
1. ✅ **Ground truth available** - You control the dynamics
2. ✅ **No noise** - Perfect "recordings" of hidden states
3. ✅ **Fast iteration** - No animals, no experiments
4. ✅ **Interpretable** - Can analyze learned representations
5. ✅ **Benchmarkable** - Standard tasks with known solutions

**Critical Test**: Does the method discover meaningful functional specialization in trained networks?

---

## Experimental Design

### Phase 1: Single Task Validation (Start Here)

**Objective**: Prove the method works on one well-understood RNN task.

#### Recommended Task: 3-Bit Flip-Flop Memory

**Why this task?**
- Simple: Store 3 bits in memory
- Known dynamics: Should have attractors (contraction) for each memory state
- Clear roles: Some units should maintain memory (integrators)

**Task Structure**:
```
Input:  [flip bit 1] [flip bit 2] [flip bit 3] [no input...]
Output: [bit 1 value] [bit 2 value] [bit 3 value]
```

**Expected Dynamics**:
- Contraction-dominant (stable fixed points for each state)
- Integrator units should dominate

---

### Phase 2: Multi-Task Comparison (After Phase 1 works)

Test on tasks with different dynamical signatures:

| Task | Expected Dynamics | Expected Unit Types |
|------|-------------------|---------------------|
| **3-Bit Flip-Flop** | Contraction (stable states) | Integrators >> Rotators |
| **Cycling Memory** | Rotation (periodic patterns) | Rotators >> Integrators |
| **Context-Dependent Integration** | Mixed (integrate then switch) | Integrators + Rotators |
| **Go/No-Go Timing** | Expansion then contraction | Explorers → Integrators |

**Hypothesis**: Task structure determines functional type distribution.

---

### Phase 3: Architecture Comparison (Advanced)

Compare unit specialization across architectures:
- Vanilla RNN
- GRU
- LSTM
- Continuous-time RNN (CT-RNN)

**Question**: Do different architectures develop different functional specializations for the same task?

---

## Codebase Extension Plan

### Current Structure
```
src/
├── population.py      # Simulates biological neurons
├── dynamics.py        # Generates latent dynamics
├── features.py        # Feature extraction
└── evaluation.py      # Testing framework

scripts/
├── synthetic.py       # Main test
└── test_robustness.py # Validation suite
```

### New Structure (Add These)
```
src/
├── population.py
├── dynamics.py
├── features.py
├── evaluation.py
└── rnn_tasks.py       # NEW: RNN task definitions

scripts/
├── synthetic.py
├── test_robustness.py
└── test_rnn.py        # NEW: RNN experiments

notebooks/              # NEW: Interactive analysis
└── rnn_analysis.ipynb
```

---

## Implementation Details

### Step 1: Create RNN Task Module (`src/rnn_tasks.py`)

This module should implement:

1. **Task base class** - Defines interface for all tasks
2. **Specific tasks** - Flip-flop, cycling memory, context integration, etc.
3. **RNN training** - Train networks on tasks
4. **Data generation** - Generate trajectories for analysis

**Key Functions**:
```python
class Task:
    def generate_trial(self, length, batch_size):
        """Generate input/target sequences"""
        
    def train_rnn(self, hidden_size, n_epochs):
        """Train RNN on this task"""
        
    def extract_trajectories(self, rnn, n_trials):
        """Get hidden state trajectories from trained RNN"""
```

**Important**: 
- Use **continuous-time RNNs** if possible (easier to compute Jacobian)
- Or use standard RNNs and compute Jacobian numerically

---

### Step 2: Adapt Feature Extraction (`src/features.py`)

**Current limitation**: `extract_deformation_features()` expects:
- Spike trains (discrete)
- Firing rates (smoothed)
- Pre-computed deformation signals

**For RNNs, you have**:
- Hidden states (continuous) - directly analogous to firing rates
- No spikes (not needed)

**Modification needed**:

```python
def extract_deformation_features_rnn(hidden_states, latent_trajectory, dt,
                                     rotation_trajectory=None, 
                                     contraction_trajectory=None,
                                     expansion_trajectory=None):
    """
    RNN version - hidden_states are already continuous, no smoothing needed.
    
    Args:
        hidden_states: (n_units, n_timesteps) - RNN hidden activations
        latent_trajectory: (n_timesteps, latent_dim) - low-dim projection
        dt: timestep
        rotation/contraction/expansion_trajectory: pre-computed deformation
        
    Returns:
        features: (n_units, 3) - correlation with R(t), C(t), E(t)
    """
    n_units = hidden_states.shape[0]
    n_timesteps = hidden_states.shape[1]
    
    # No need to downsample or smooth - RNN activations are already clean
    
    # Ensure deformation trajectories match hidden states length
    min_len = min(n_timesteps, len(rotation_trajectory))
    
    features = []
    for i in range(n_units):
        unit_activity = hidden_states[i, :min_len]
        
        corr_rotation = np.corrcoef(unit_activity, rotation_trajectory[:min_len])[0, 1]
        corr_contraction = np.corrcoef(unit_activity, contraction_trajectory[:min_len])[0, 1]
        corr_expansion = np.corrcoef(unit_activity, expansion_trajectory[:min_len])[0, 1]
        
        # Handle NaN
        corr_rotation = 0 if np.isnan(corr_rotation) else corr_rotation
        corr_contraction = 0 if np.isnan(corr_contraction) else corr_contraction
        corr_expansion = 0 if np.isnan(corr_expansion) else corr_expansion
        
        features.append([corr_rotation, corr_contraction, corr_expansion])
    
    return np.array(features)
```

---

### Step 3: Compute Deformation from RNN Dynamics

**Two approaches**:

#### Approach A: Analytical Jacobian (Best - if using PyTorch/JAX)

```python
import torch

def compute_jacobian_analytical(rnn, hidden_state):
    """
    Compute Jacobian using autograd.
    
    Args:
        rnn: PyTorch RNN model
        hidden_state: (hidden_size,) current state
        
    Returns:
        J: (hidden_size, hidden_size) Jacobian matrix
    """
    hidden_state = torch.tensor(hidden_state, requires_grad=True)
    
    # Get next state
    next_state = rnn.step(hidden_state)  # One RNN step
    
    # Compute Jacobian via autograd
    J = torch.autograd.functional.jacobian(
        lambda h: rnn.step(h), 
        hidden_state
    )
    
    return J.detach().numpy()
```

#### Approach B: Numerical Jacobian (Works for any RNN)

```python
def compute_jacobian_numerical(rnn, hidden_state, eps=1e-6):
    """
    Compute Jacobian via finite differences.
    """
    n = len(hidden_state)
    J = np.zeros((n, n))
    
    # Baseline
    f0 = rnn.step(hidden_state)
    
    # Perturb each dimension
    for i in range(n):
        h_plus = hidden_state.copy()
        h_plus[i] += eps
        f_plus = rnn.step(h_plus)
        
        J[:, i] = (f_plus - f0) / eps
    
    return J
```

**Then use existing decomposition**:
```python
def decompose_jacobian(J):
    """Already implemented in dynamics.py logic"""
    S = 0.5 * (J + J.T)
    A = 0.5 * (J - J.T)
    
    rotation = np.linalg.norm(A, 'fro')
    eigenvalues = np.linalg.eigvalsh(S)
    contraction = -np.sum(eigenvalues[eigenvalues < 0])
    expansion = np.sum(eigenvalues[eigenvalues > 0])
    
    return rotation, contraction, expansion
```

---

### Step 4: Create RNN Test Script (`scripts/test_rnn.py`)

**Main experiment flow**:

```python
#!/usr/bin/env python
"""
Test deformation-based unit classification on RNNs.
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score

from src.rnn_tasks import FlipFlopTask, CyclingMemoryTask
from src.features import extract_deformation_features_rnn


def main():
    # 1. Train RNN on task
    print("Training RNN on 3-bit flip-flop...")
    task = FlipFlopTask(n_bits=3)
    rnn = task.train_rnn(hidden_size=128, n_epochs=2000)
    
    # 2. Generate test trajectories
    print("Generating trajectories...")
    hidden_states, inputs, outputs = task.extract_trajectories(
        rnn, n_trials=50, trial_length=200
    )
    # hidden_states: (n_units=128, n_timesteps=10000)
    
    # 3. Compute latent trajectory (PCA to 3D)
    print("Computing latent space...")
    from sklearn.decomposition import PCA
    pca = PCA(n_components=3)
    latent_traj = pca.fit_transform(hidden_states.T)  # (n_timesteps, 3)
    
    # 4. Estimate deformation signals
    print("Estimating deformation...")
    rotation_traj, contraction_traj, expansion_traj = estimate_deformation(
        latent_traj, rnn, dt=0.01
    )
    
    # 5. Extract unit features
    print("Extracting features...")
    features = extract_deformation_features_rnn(
        hidden_states, latent_traj, dt=0.01,
        rotation_trajectory=rotation_traj,
        contraction_trajectory=contraction_traj,
        expansion_trajectory=expansion_traj
    )
    
    # 6. Cluster units
    print("Clustering units...")
    scaler = StandardScaler()
    features_norm = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=20)
    labels = kmeans.fit_predict(features_norm)
    
    # 7. Analyze results
    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)
    
    for cluster in range(4):
        n_units = np.sum(labels == cluster)
        cluster_features = features[labels == cluster]
        mean_features = np.mean(cluster_features, axis=0)
        
        print(f"\nCluster {cluster} ({n_units} units):")
        print(f"  Mean rotation corr:    {mean_features[0]:+.3f}")
        print(f"  Mean contraction corr: {mean_features[1]:+.3f}")
        print(f"  Mean expansion corr:   {mean_features[2]:+.3f}")
        
        # Assign label
        dominant = np.argmax(np.abs(mean_features))
        labels_map = {0: "Rotator", 1: "Integrator", 2: "Explorer"}
        print(f"  → {labels_map[dominant]}")
    
    # 8. Quality metrics
    silhouette = silhouette_score(features_norm, labels)
    print(f"\nSilhouette score: {silhouette:.3f}")
    
    # 9. Visualize
    plot_rnn_analysis(hidden_states, latent_traj, features, labels,
                     rotation_traj, contraction_traj, expansion_traj)
    
    print("\nSaved: rnn_analysis.png")


def estimate_deformation(latent_traj, rnn, dt=0.01):
    """
    Estimate deformation from latent trajectory.
    Can use either analytical or numerical Jacobian.
    """
    n_timesteps = len(latent_traj)
    
    rotation_traj = []
    contraction_traj = []
    expansion_traj = []
    
    # Sample subset of points (expensive to compute Jacobian at every point)
    sample_indices = np.linspace(0, n_timesteps-1, min(200, n_timesteps), dtype=int)
    
    for idx in sample_indices:
        # Get full hidden state at this time (need to map from latent to hidden)
        # This is tricky - you might need to store full hidden states
        # For now, estimate from local neighborhood in latent space
        
        J = estimate_jacobian_at_point(latent_traj, idx, k_neighbors=20)
        
        # Decompose
        S = 0.5 * (J + J.T)
        A = 0.5 * (J - J.T)
        
        rotation = np.linalg.norm(A, 'fro')
        eigenvalues = np.linalg.eigvalsh(S)
        contraction = -np.sum(eigenvalues[eigenvalues < 0])
        expansion = np.sum(eigenvalues[eigenvalues > 0])
        
        rotation_traj.append(rotation)
        contraction_traj.append(contraction)
        expansion_traj.append(expansion)
    
    # Interpolate to full length
    sample_times = sample_indices
    full_times = np.arange(n_timesteps)
    
    rotation_full = np.interp(full_times, sample_times, rotation_traj)
    contraction_full = np.interp(full_times, sample_times, contraction_traj)
    expansion_full = np.interp(full_times, sample_times, expansion_traj)
    
    return rotation_full, contraction_full, expansion_full


def estimate_jacobian_at_point(trajectory, idx, k_neighbors=20):
    """
    Estimate Jacobian via local linear regression (same as test_robustness.py).
    """
    from sklearn.neighbors import NearestNeighbors
    
    n_timesteps, dim = trajectory.shape
    
    # Compute velocities
    velocities = np.gradient(trajectory, axis=0)
    
    # Find neighbors
    nbrs = NearestNeighbors(n_neighbors=min(k_neighbors, n_timesteps))
    nbrs.fit(trajectory)
    
    distances, indices = nbrs.kneighbors([trajectory[idx]])
    
    # Local linear regression
    neighbor_points = trajectory[indices[0]]
    neighbor_velocities = velocities[indices[0]]
    
    delta_z = neighbor_points - trajectory[idx]
    delta_v = neighbor_velocities - velocities[idx]
    
    # Solve for Jacobian
    J = np.zeros((dim, dim))
    for d in range(dim):
        J[d, :] = np.linalg.lstsq(delta_z, delta_v[:, d], rcond=None)[0]
    
    return J


if __name__ == "__main__":
    main()
```

---

### Step 5: Implement RNN Tasks (`src/rnn_tasks.py`)

**Example: 3-Bit Flip-Flop**

```python
import torch
import torch.nn as nn
import numpy as np


class FlipFlopTask:
    """
    3-bit flip-flop memory task.
    
    Input: 3 channels, each with +1 (flip bit), -1 (flip bit), or 0 (no input)
    Output: 3 channels, each with current bit value (+1 or -1)
    """
    
    def __init__(self, n_bits=3):
        self.n_bits = n_bits
        self.input_size = n_bits
        self.output_size = n_bits
    
    def generate_trial(self, length=200, batch_size=32):
        """Generate random flip-flop sequences."""
        inputs = torch.zeros(batch_size, length, self.n_bits)
        targets = torch.zeros(batch_size, length, self.n_bits)
        
        for b in range(batch_size):
            # Initialize state
            state = torch.randint(0, 2, (self.n_bits,)) * 2 - 1  # Random +1/-1
            
            for t in range(length):
                # Random flip with 5% probability per bit
                for bit in range(self.n_bits):
                    if np.random.rand() < 0.05:
                        flip_value = np.random.choice([1, -1])
                        inputs[b, t, bit] = flip_value
                        state[bit] *= -1  # Flip the bit
                
                targets[b, t] = state
        
        return inputs, targets
    
    def train_rnn(self, hidden_size=128, n_epochs=2000, lr=0.001):
        """Train RNN on flip-flop task."""
        
        # Create RNN
        rnn = SimpleRNN(self.input_size, hidden_size, self.output_size)
        optimizer = torch.optim.Adam(rnn.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        print(f"Training RNN (hidden_size={hidden_size})...")
        
        for epoch in range(n_epochs):
            # Generate batch
            inputs, targets = self.generate_trial(length=100, batch_size=32)
            
            # Forward pass
            optimizer.zero_grad()
            outputs, hidden_states = rnn(inputs)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            if (epoch + 1) % 200 == 0:
                print(f"  Epoch {epoch+1}/{n_epochs}, Loss: {loss.item():.4f}")
        
        print("Training complete!")
        return rnn
    
    def extract_trajectories(self, rnn, n_trials=50, trial_length=200):
        """Extract hidden state trajectories from trained RNN."""
        
        rnn.eval()
        all_hidden_states = []
        all_inputs = []
        all_outputs = []
        
        with torch.no_grad():
            for trial in range(n_trials):
                inputs, targets = self.generate_trial(length=trial_length, batch_size=1)
                outputs, hidden_states = rnn(inputs)
                
                # hidden_states: (batch=1, time, hidden_size)
                all_hidden_states.append(hidden_states[0].numpy())  # (time, hidden_size)
                all_inputs.append(inputs[0].numpy())
                all_outputs.append(outputs[0].numpy())
        
        # Concatenate trials
        hidden_states = np.concatenate(all_hidden_states, axis=0)  # (n_trials*trial_length, hidden_size)
        inputs = np.concatenate(all_inputs, axis=0)
        outputs = np.concatenate(all_outputs, axis=0)
        
        # Transpose to (n_units, n_timesteps)
        hidden_states = hidden_states.T
        
        return hidden_states, inputs, outputs


class SimpleRNN(nn.Module):
    """Simple vanilla RNN."""
    
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.hidden_size = hidden_size
        
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # x: (batch, time, input_size)
        hidden_states, _ = self.rnn(x)  # (batch, time, hidden_size)
        outputs = self.fc(hidden_states)  # (batch, time, output_size)
        
        return outputs, hidden_states
```

---

## Validation Checklist

Before claiming success, verify:

### ✅ Sanity Checks

1. **Clusters are stable**
   - Re-run with different random seeds → similar clusters?
   - Different K-means initializations → consistent?

2. **Clusters are interpretable**
   - Does "integrator" cluster have high contraction correlation?
   - Does "rotator" cluster have high rotation correlation?

3. **Visualization makes sense**
   - Plot trajectory in latent space - does it match task structure?
   - Plot deformation signals - do they align with task phases?

4. **Not just random structure**
   - Shuffle unit labels - does performance drop?
   - Random hidden states - does clustering fail?

---

### ✅ Scientific Validation

1. **Task-specific predictions**
   - Flip-flop → should be dominated by integrators (stable attractors)
   - Cycling memory → should have more rotators (periodic dynamics)
   - Compare actual distributions to predictions

2. **Unit ablation**
   - Remove "integrator" units → task performance drops?
   - Remove "rotator" units → less effect on flip-flop?

3. **Cross-architecture consistency**
   - Train GRU on same task → similar functional types?
   - Or does architecture matter?

4. **Comparison to baselines**
   - PCA loadings
   - K-means on raw activations
   - Does deformation method give more interpretable clusters?

---

## Expected Timeline

### Week 1: Infrastructure
- [ ] Implement `src/rnn_tasks.py` (flip-flop task)
- [ ] Implement `extract_deformation_features_rnn()`
- [ ] Train one RNN and verify it learns task

### Week 2: Core Experiment
- [ ] Compute deformation signals from RNN trajectories
- [ ] Extract features and cluster units
- [ ] Verify clusters are interpretable

### Week 3: Validation
- [ ] Test stability (multiple seeds)
- [ ] Compare to baselines (PCA, raw activations)
- [ ] Try second task (cycling memory)

### Week 4: Analysis & Visualization
- [ ] Create publication-quality figures
- [ ] Compare functional type distributions across tasks
- [ ] Write up results

---

## Success Criteria

**Minimum viable result**:
- Clusters are stable and interpretable
- "Integrator" units have high contraction correlation
- Better silhouette score than PCA baselines

**Strong result**:
- Functional type distribution predicts task structure (flip-flop → integrators)
- Cross-task comparison shows expected differences
- Unit ablation validates functional importance

**Publication-ready result**:
- Works across multiple tasks (3+)
- Works across architectures (RNN, GRU, LSTM)
- Quantitative comparison shows 20%+ improvement over baselines
- Connects to biological predictions

---

## Potential Issues & Solutions

### Issue 1: All units look the same
**Symptom**: K-means finds no clear clusters
**Possible causes**:
- RNN hasn't learned specialized representations (undertrained)
- Task is too simple (all units do everything)
- Hidden size too small (no redundancy)

**Solutions**:
- Train longer / use larger hidden size
- Try more complex task
- Check if RNN actually learned task (test accuracy)

---

### Issue 2: Deformation signals are noisy
**Symptom**: R(t), C(t), E(t) are extremely spiky
**Possible causes**:
- Jacobian estimation is unstable
- Latent space dimension too low

**Solutions**:
- Increase k_neighbors for Jacobian estimation
- Smooth deformation signals with Gaussian filter
- Use higher-dimensional latent space (5D instead of 3D)

---

### Issue 3: RNN dynamics are boring
**Symptom**: Very low rotation and expansion, only contraction
**Possible causes**:
- RNN just converged to stable fixed points (no interesting dynamics)
- Task doesn't require complex dynamics

**Solutions**:
- Try task that requires richer dynamics (context switching, timing)
- Use continuous-time RNN with dt tuning
- Look for transient dynamics (not just steady state)

---

## Next Steps After Initial Success

1. **Extend to more tasks**
   - Implement 5-10 standard cognitive tasks
   - Build task taxonomy by dynamical structure

2. **Compare architectures**
   - RNN vs GRU vs LSTM vs Transformer
   - Do transformers even use rotation/contraction/expansion?

3. **Biological comparison**
   - Get motor cortex Neuropixels data
   - Train RNN on similar task
   - Compare functional type distributions

4. **Tool development**
   - Package as Python library
   - Jupyter notebook tutorials
   - Integration with existing RNN analysis tools

---

## Code Repository Structure (Final)

```
deformation-neuron-classification/
├── src/
│   ├── __init__.py
│   ├── population.py          # Biological neuron simulation
│   ├── dynamics.py             # Latent dynamics generation
│   ├── features.py             # Feature extraction (bio + RNN)
│   ├── evaluation.py           # Testing framework
│   └── rnn_tasks.py           # RNN task definitions
│
├── scripts/
│   ├── synthetic.py            # Biological neuron tests
│   ├── test_robustness.py      # Robustness validation
│   └── test_rnn.py            # RNN experiments
│
├── notebooks/
│   └── rnn_analysis.ipynb     # Interactive RNN analysis
│
├── results/
│   ├── figures/                # Synthetic test plots
│   └── rnn_figures/           # RNN test plots
│
├── docs/
│   ├── project_synthesis.md    # Original concept
│   ├── robustness_synthesis.md # Test results
│   └── rnn_extension_guide.md  # This document
│
└── README.md
```

---

## Key Takeaways

1. **Start simple** - One task (flip-flop), one architecture (vanilla RNN)
2. **Validate thoroughly** - Stability, interpretability, comparison to baselines
3. **Expect iteration** - First attempt might not work, tune hyperparameters
4. **Focus on interpretation** - Can you explain what each cluster does?
5. **Compare to ground truth** - RNNs let you verify if method finds real structure

**The RNN testbed is your proof of concept before expensive biology experiments.**

Good luck! 🚀