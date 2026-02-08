# Enhanced RNN Feature Extraction

## Overview

The enhanced feature extraction system adds **temporal** and **contextual** resolution to the basic deformation-based features, enabling finer-grained functional classification of RNN units.

## Problem Solved

**Basic features** (3D: rotation, contraction, expansion correlations) can produce multiple clusters with identical labels but different functional roles. For example:
- Multiple "Explorer" clusters doing different things
- Cannot distinguish Context A vs Context B specialization
- Cannot distinguish early vs late timing

**Enhanced features** add temporal and contextual dimensions to differentiate these cases.

## Quick Start

### Basic Usage (3D features)

```python
from src.analysis import extract_rnn_unit_features, classify_units, interpret_clusters

# Extract basic features
features = extract_rnn_unit_features(
    hidden_states,      # (n_units, n_timesteps)
    rotation_traj,      # (n_timesteps,)
    contraction_traj,   # (n_timesteps,)
    expansion_traj      # (n_timesteps,)
)
# Returns: (n_units, 3) - [R_corr, C_corr, E_corr]

# Cluster and interpret
labels = classify_units(features, n_clusters=4)
interpretation = interpret_clusters(features, labels)
```

### Enhanced Usage (12D or 18D features)

```python
from src.analysis import extract_enhanced_rnn_features, classify_units, interpret_clusters

# Setup task info
task_info = {
    'trial_length': 200,
    'n_trials': 50,
    'task_name': 'context'  # or 'parametric', 'flipflop', etc.
}

# Extract enhanced features
enhanced_features = extract_enhanced_rnn_features(
    hidden_states,      # (n_units, n_timesteps)
    rotation_traj,      # (n_timesteps,)
    contraction_traj,   # (n_timesteps,)
    expansion_traj,     # (n_timesteps,)
    task_info=task_info,
    inputs=inputs       # (n_timesteps, input_size) - needed for context detection
)
# Returns: (n_units, 12) or (n_units, 18) depending on task

# Cluster and interpret (automatically detects enhanced features)
labels = classify_units(enhanced_features, n_clusters=6)
interpretation = interpret_clusters(enhanced_features, labels)
```

## Feature Structure

### Basic Features (3D)
- `[0:3]`: Global R, C, E correlations

### Enhanced Features (12D)
- `[0:3]`: Global R, C, E correlations (backwards compatible)
- `[3:6]`: Early period R, C, E
- `[6:9]`: Middle period R, C, E
- `[9:12]`: Late period R, C, E

### Enhanced Features with Context (18D)
- `[0:12]`: Same as above
- `[12:15]`: Context A R, C, E (only for context task)
- `[15:18]`: Context B R, C, E

## Cluster Labels

Enhanced features produce more descriptive labels:

**Basic features:**
- "Rotator", "Integrator", "Explorer", "Mixed"

**Enhanced features:**
- "Early Integrator", "Late Explorer", "Middle Rotator"
- "Early Explorer (Context A)", "Late Integrator (Context B)"
- Temporal specialization when units are active in specific trial phases
- Context preference when units respond differently to different contexts

## Testing

### Run the example comparison:

```bash
# Compare basic vs enhanced on context task
python test_enhanced_features.py --task context --epochs 500

# Try with parametric task
python test_enhanced_features.py --task parametric --epochs 1000
```

### Expected Output:

```
BASIC FEATURES - Cluster Interpretation
========================================
Cluster 0: Explorer (45 units, 35.2%)
Cluster 1: Explorer (38 units, 29.7%)  # Same label!
Cluster 2: Integrator (32 units, 25.0%)
Cluster 3: Explorer (13 units, 10.2%)  # Same label again!

ENHANCED FEATURES - Cluster Interpretation
===========================================
Cluster 0: Early Explorer (Context A) (45 units, 35.2%)
Cluster 1: Late Explorer (Context B) (38 units, 29.7%)
Cluster 2: Middle Integrator (32 units, 25.0%)
Cluster 3: Early Explorer (Context B) (13 units, 10.2%)
```

Notice how the identical "Explorer" labels are now differentiated!

## When to Use Enhanced Features

### Use enhanced features when:
- ✓ Task has temporal structure (different phases)
- ✓ Context task with multiple input conditions
- ✓ You see duplicate cluster labels with basic features
- ✓ You want to understand timing of unit activity

### Stick with basic features when:
- Task is very simple (static memory)
- Network is small (<64 units)
- You want to compare to published results using basic method

## Backwards Compatibility

All existing code continues to work:
- `extract_rnn_unit_features()` still available (basic features)
- `classify_units()` and `interpret_clusters()` automatically detect feature type
- Enhanced features are **opt-in**, not required

## Implementation Details

### Temporal Division
Splits trajectory into thirds (early/middle/late) and computes separate correlations for each period.

### Context Detection (Context Task)
- Identifies Context A trials: input channel 0 > 0.5 in first 20 timesteps
- Identifies Context B trials: input channel 1 > 0.5 in first 20 timesteps
- Computes separate deformation correlations for each context

### Feature Normalization
Features are normalized within each group (global, early, middle, late, contextA, contextB) to sum to 1, making them ratio-based and comparable.

## Advanced Usage

### Custom Temporal Boundaries

```python
# Modify the function to use custom phase boundaries
# (Currently uses automatic thirds, but you can extend it)
task_info = {
    'trial_length': 200,
    'n_trials': 50,
    'task_name': 'context',
    'phase_boundaries': {  # Not yet implemented, but reserved
        'encoding': (0, 50),
        'integration': (50, 150),
        'decision': (150, 200)
    }
}
```

### Accessing Detailed Interpretation

```python
interpretation = interpret_clusters(enhanced_features, labels)

for cluster_id, info in interpretation.items():
    print(f"Cluster {cluster_id}: {info['name']}")
    
    # Enhanced features provide additional info
    if 'temporal_features' in info:
        print(f"  Temporal specialization: {info['temporal_specialization']}")
        print(f"  Early: {info['temporal_features']['early']}")
        print(f"  Middle: {info['temporal_features']['middle']}")
        print(f"  Late: {info['temporal_features']['late']}")
    
    if 'context_features' in info:
        print(f"  Context preference: {info['context_preference']}")
        print(f"  Context A: {info['context_features']['contextA']}")
        print(f"  Context B: {info['context_features']['contextB']}")
```

## Troubleshooting

### Issue: Enhanced features produce same results as basic

**Cause:** Network may not have developed temporal/contextual specialization

**Solutions:**
- Train longer (increase epochs)
- Increase network size (more units = more specialization)
- Use a more complex task

### Issue: All enhanced features are identical

**Cause:** Deformation signals may not vary temporally/contextually

**Solutions:**
- Check deformation signal variance across time periods
- Ensure task has actual temporal structure
- Verify inputs are being passed correctly for context detection

### Issue: Context features not appearing

**Cause:** Task name not recognized or inputs not provided

**Solutions:**
- Ensure `task_info['task_name'] == 'context'` (exact match, lowercase)
- Pass `inputs` parameter to `extract_enhanced_rnn_features()`
- Check inputs shape: should be `(n_timesteps, input_size)`

## Citation

If you use enhanced features in your research, please cite the original deformation method and note the temporal/contextual enhancement:

```
Enhanced deformation-based features with temporal and contextual resolution
for RNN unit classification, extending the basic method with explicit
modeling of task phase and input condition dependencies.
```
