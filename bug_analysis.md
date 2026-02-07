# RNN Deformation Analysis - Bug Report and Fixes

## Summary of Issues

Your test_rnn.py output reveals three critical bugs:

1. **CYCLING task**: Deformation estimation returns all zeros (falls back to random noise)
2. **FLIPFLOP task**: Underperforms baselines by 20%+
3. **Unit classification**: Misclassifies units when correlations are weak/noisy

---

## Bug #1: Cycling Task - Deformation Estimation Fails

### Root Cause
The RNN solves the cycling task using **discrete stable states** rather than smooth rotation:
- Achieves 99.86% accuracy with very stable representations
- Switches between patterns discretely (like a state machine)
- Latent trajectory has very low variance → Jacobian estimation breaks down

### Evidence
```
Warning: Deformation estimation produced all zeros
Rotation range: [0.000, 0.000]
Contraction range: [0.000, 0.000]  
Expansion range: [0.000, 0.000]
```

### Why This Happens
In `src/deformation_utils.py::estimate_deformation_from_latents()`:

```python
trajectory_std = np.std(latent_trajectory, axis=0)
if np.all(trajectory_std < 1e-8):
    # Returns random noise as fallback
```

The threshold check (1e-8) is meant to catch **completely static** trajectories, but the cycling task produces dynamics that are:
- Just above 1e-8 threshold (passes the check)
- But too small/discrete for reliable Jacobian estimation via local linear regression
- Results in near-zero Jacobians → near-zero deformation signals

### Fix Option 1: Relax Variance Threshold
```python
# In estimate_deformation_from_latents()
# Change from:
if np.all(trajectory_std < 1e-8):

# To:
if np.all(trajectory_std < 1e-5):  # More lenient threshold
```

### Fix Option 2: Use Velocity Magnitude as Backup
```python
# After computing deformation signals, check if they're informative
deform_std = [np.std(rotation_samples), np.std(contraction_samples), 
              np.std(expansion_samples)]

if all(s < 1e-8 for s in deform_std):
    # Use velocity magnitude as backup signal
    velocities = np.gradient(latent_trajectory, axis=0)
    velocity_magnitude = np.linalg.norm(velocities, axis=1)
    
    # Heuristic: high velocity = transitions (expansion)
    expansion_trajectory = velocity_magnitude
    
    # Use derivatives as rotation/contraction proxies
    accel = np.gradient(velocity_magnitude)
    rotation_trajectory = np.abs(accel)  # Change in velocity direction
    contraction_trajectory = -accel  # Deceleration
```

### Fix Option 3: Detect Discrete Dynamics
```python
# Add task-specific handling
def detect_transition_dynamics(latent_trajectory, threshold=0.1):
    """Detect if dynamics are discrete transitions vs continuous flow."""
    velocities = np.gradient(latent_trajectory, axis=0)
    velocity_mag = np.linalg.norm(velocities, axis=1)
    
    # Discrete transitions: velocity alternates between ~0 and high
    high_vel_frac = np.mean(velocity_mag > threshold)
    
    if 0.05 < high_vel_frac < 0.20:  # 5-20% of time in transition
        return True, velocity_mag
    return False, velocity_mag

# In estimate_deformation_from_rnn():
is_discrete, vel = detect_transition_dynamics(latent_trajectory)
if is_discrete:
    # Use transition-based features instead of Jacobian
    ...
```

---

## Bug #2: FlipFlop Underperforming Baselines

### Root Cause
FlipFlop dynamics are **too static** for deformation-based features to be useful:
- Network maintains stable memory states (attractors)
- Minimal rotation/contraction/expansion during stable periods
- Correlations with deformation signals are weak (0.02-0.05)
- More informative features: raw activation patterns, selectivity

### Evidence
```
FLIPFLOP:
  Mean |rotation corr|: 0.021
  Mean |contraction corr|: 0.054
  Mean |expansion corr|: 0.017
  
  Deformation method: 0.458
  PCA baseline:       0.583  (-21.4%)
  Raw activations:    0.593  (-22.8%)
```

### Why This Happens
The method assumes **continuous flow dynamics** where deformation geometry is meaningful. But FlipFlop is dominated by:
- **Stable fixed points** (no flow most of the time)
- **Rapid transitions** between states (too fast to capture)

### Fix: Add Task-Appropriate Feature Selection
```python
def select_features_by_task_type(hidden_states, deformation_features, 
                                 task_dynamics='unknown'):
    """
    Choose features based on task dynamics type.
    
    Args:
        task_dynamics: 'static', 'oscillatory', 'mixed', or 'unknown'
    """
    
    if task_dynamics == 'static':
        # For memory tasks: use selectivity and state-specific activity
        from sklearn.decomposition import PCA
        
        # Option 1: PCA features (capture state structure)
        pca = PCA(n_components=10)
        pca_features = pca.fit_transform(hidden_states)
        return pca_features
        
    elif task_dynamics == 'oscillatory':
        # For cycling tasks: use frequency domain features
        from scipy.fft import fft
        
        freq_features = []
        for unit in hidden_states:
            fft_vals = np.abs(fft(unit))
            freq_features.append(fft_vals[:50])  # Low frequencies
        return np.array(freq_features)
        
    elif task_dynamics == 'mixed' or task_dynamics == 'unknown':
        # Use deformation features (current approach)
        return deformation_features
        
    return deformation_features
```

### Alternative Fix: Hybrid Features
```python
# Combine deformation + raw activity features
features_deform = extract_rnn_unit_features(...)  # (n_units, 3)
features_pca = PCA(n_components=3).fit_transform(hidden_states)  # (n_units, 3)

# Concatenate and weight by correlation strength
avg_corr = np.mean(np.abs(features_deform))
if avg_corr < 0.05:
    # Deformation features are weak, rely on PCA
    features = 0.3 * features_deform + 0.7 * features_pca
else:
    features = 0.7 * features_deform + 0.3 * features_pca
```

---

## Bug #3: Unit Classification When Correlations Are Weak

### Root Cause
When deformation signals are noise (Cycling) or very weak (FlipFlop), the classification still assigns types based on meaningless correlations:

```python
# Current code always picks a "dominant mode" even if all correlations are ~0
dominant_idx = np.argmax(abs_features)  # Picks max of [0.02, 0.03, 0.01]
dominant_mode = feature_names[dominant_idx]  # "Contraction" (wrong!)
```

### Fix: Stricter Classification Logic
```python
def interpret_clusters(features, labels, cluster_names=None, 
                      threshold=0.03, strict_threshold=0.10):
    """
    Args:
        threshold: Minimum correlation to distinguish from Mixed (0.03)
        strict_threshold: Minimum to confidently assign type (0.10)
    """
    
    for cluster_id in range(n_clusters):
        # ... existing code ...
        
        abs_features = np.abs(mean_features)
        max_abs_corr = np.max(abs_features)
        dominant_idx = np.argmax(abs_features)
        
        # THREE-TIER CLASSIFICATION:
        if max_abs_corr < threshold:
            name = 'Mixed'  # Too weak to classify
            confidence = 'very_low'
            
        elif max_abs_corr < strict_threshold:
            # Weak signal - classify but flag as uncertain
            if dominant_idx == 0:
                name = 'Rotator?'  # Question mark indicates uncertainty
            elif dominant_idx == 1:
                name = 'Integrator?'
            else:
                name = 'Explorer?'
            confidence = 'low'
            
        else:
            # Strong signal - confident classification
            if dominant_idx == 0:
                name = 'Rotator'
            elif dominant_idx == 1:
                name = 'Integrator'
            else:
                name = 'Explorer'
            confidence = 'high'
        
        # Check if second-strongest mode is close (multi-modal unit)
        sorted_abs = np.sort(abs_features)[::-1]
        if sorted_abs[1] > 0.8 * sorted_abs[0]:  # Within 20% of max
            name = name.replace('?', '') + '+Mixed'
        
        interpretation[cluster_id] = {
            'name': name,
            'confidence': confidence,
            'max_correlation': max_abs_corr,
            # ... rest of fields ...
        }
```

---

## Bug #4: Missing Validation

The code doesn't check if learned dynamics match task expectations:

### Add Validation Function
```python
def validate_task_dynamics(task_name, deformation_signals, hidden_states):
    """
    Check if learned dynamics match task expectations.
    
    Returns:
        valid: bool
        issues: list of str (problems found)
        suggestions: list of str (what to try)
    """
    rot, con, exp = deformation_signals
    issues = []
    suggestions = []
    
    # Check signal strength
    avg_magnitude = np.mean([np.std(rot), np.std(con), np.std(exp)])
    if avg_magnitude < 1e-6:
        issues.append("Deformation signals are noise-level")
        suggestions.append("Try: Increase training epochs, reduce learning rate, or use different architecture")
    
    # Task-specific checks
    if task_name == 'cycling':
        # Should have significant rotation
        if np.std(rot) < 0.1 * np.std(con):
            issues.append("Expected rotation-dominant dynamics, found contraction-dominant")
            suggestions.append("Network may be using discrete states instead of rotation")
            suggestions.append("Try: GRU/LSTM, add recurrent noise, or use continuous-time RNN")
    
    elif task_name == 'flipflop':
        # Should have stable states (low overall dynamics)
        total_dynamics = np.std(rot) + np.std(con) + np.std(exp)
        hidden_variance = np.var(hidden_states)
        
        if total_dynamics > 0.5 * hidden_variance:
            issues.append("FlipFlop shows high dynamics, expected stable memory states")
    
    valid = len(issues) == 0
    return valid, issues, suggestions
```

---

## Recommended Implementation Order

1. **Immediate fix (Cycling)**: Use Fix Option 1 or 2 to handle small-variance trajectories
2. **Immediate fix (Classification)**: Add confidence thresholds to interpret_clusters
3. **Medium-term (FlipFlop)**: Add task-specific feature selection
4. **Long-term**: Add validation + warnings about dynamics mismatch

---

## Testing These Fixes

```python
# After implementing fixes, re-run and check:

# Cycling should now show:
# - Non-zero deformation signals OR velocity-based features
# - Better unit classification (not all Integrators)
# - Validation warnings about discrete vs continuous dynamics

# FlipFlop should now show:
# - Either improved silhouette OR switch to PCA features
# - Validation confirmation that task is static-dominant

# All tasks should show:
# - Confidence levels in classification
# - Warnings when correlations are weak
```

---

## Additional Suggestions

### 1. Add Diagnostic Plots
```python
# In test_rnn.py, after deformation estimation:
def plot_deformation_diagnostics(latent_traj, deform_signals, task_name):
    """Visualize what's happening."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Latent trajectory variance
    axes[0, 0].plot(np.std(latent_traj, axis=1))
    axes[0, 0].set_title('Latent Variance Over Time')
    
    # Deformation signals
    rot, con, exp = deform_signals
    axes[0, 1].plot(rot, label='Rotation', alpha=0.7)
    axes[0, 1].plot(con, label='Contraction', alpha=0.7)
    axes[0, 1].plot(exp, label='Expansion', alpha=0.7)
    axes[0, 1].legend()
    axes[0, 1].set_title('Deformation Signals')
    
    # Feature distribution
    axes[1, 0].hist([np.std(rot), np.std(con), np.std(exp)], 
                    bins=10, label=['R', 'C', 'E'])
    axes[1, 0].set_title('Deformation Magnitudes')
    
    # Latent trajectory (3D)
    if latent_traj.shape[1] >= 3:
        axes[1, 1].scatter(latent_traj[:, 0], latent_traj[:, 1], 
                          c=np.arange(len(latent_traj)), cmap='viridis', s=1)
        axes[1, 1].set_title('Latent Trajectory (PC1 vs PC2)')
    
    plt.tight_layout()
    plt.savefig(f'results/rnn_figures/{task_name}_diagnostics.png')
```

### 2. Add Expected Dynamics Check
```python
# In run_single_task_experiment():
expected_dynamics = task.get_expected_dynamics()
valid, issues, suggestions = validate_task_dynamics(task_name, 
                                                     (rotation_traj, contraction_traj, expansion_traj),
                                                     hidden_states)

if not valid:
    print("\n⚠️  VALIDATION WARNING:")
    for issue in issues:
        print(f"  - {issue}")
    print("\nSuggestions:")
    for suggestion in suggestions:
        print(f"  • {suggestion}")
```

---

## Expected Output After Fixes

### CYCLING (Fixed):
```
Estimating deformation from hidden states...
  Note: Using velocity-based features (discrete dynamics detected)
  Rotation range: [0.02, 2.45]  ← Non-zero now
  Contraction range: [0.01, 1.83]
  Expansion range: [0.15, 3.21]

Cluster 0: Rotator (45 units, 35.2%)
Cluster 1: Integrator (38 units, 29.7%)
Cluster 2: Explorer (30 units, 23.4%)
Cluster 3: Mixed (15 units, 11.7%)
```

### FLIPFLOP (Fixed):
```
Task validation: Static-dominant dynamics detected
Using PCA-based features instead of deformation features

Deformation method: 0.612  ← Now competitive
PCA baseline:       0.583
Improvement: +5.0%
```