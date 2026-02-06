"""
Quick integration test for RNN deformation experiment pipeline.

Runs a minimal end-to-end test to verify:
1. RNN training works
2. Deformation computation works
3. Feature extraction works
4. Clustering works
5. All imports are correct
"""

import sys
from pathlib import Path
import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

print("\n" + "="*70)
print("RNN INTEGRATION TEST")
print("="*70)

# Test 1: Imports
print("\n[1/6] Testing imports...")
try:
    from src.rnn_models import VanillaRNN, SimpleLSTM, SimpleGRU
    from src.tasks import FlipFlopTask, CyclingMemoryTask, ContextIntegrationTask
    from src.deformation_utils import (decompose_jacobian, compute_jacobian_analytical,
                                       estimate_deformation_from_rnn, smooth_deformation_signals)
    from src.rnn_features import (extract_rnn_unit_features, classify_units, 
                                  interpret_clusters, compare_to_baseline)
    from src.rnn_visualization import plot_training_curves, plot_latent_trajectory_3d
    from src.visualization import ensure_dirs
    print("  ✓ All imports successful")
except Exception as e:
    print(f"  ✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Create task
print("\n[2/6] Creating FlipFlop task...")
try:
    task = FlipFlopTask(n_bits=3, flip_prob=0.05)
    inputs, targets = task.generate_trial(length=100, batch_size=4)
    print(f"  ✓ Task created: input shape {inputs.shape}, target shape {targets.shape}")
except Exception as e:
    print(f"  ✗ Task creation failed: {e}")
    sys.exit(1)

# Test 3: Create and train RNN (minimal epochs for speed)
print("\n[3/6] Training RNN (100 epochs, minimal)...")
try:
    rnn = VanillaRNN(task.input_size, hidden_size=32, output_size=task.output_size)
    print(f"  RNN parameters: {sum(p.numel() for p in rnn.parameters()):,}")
    
    rnn, history = task.train_rnn(rnn, n_epochs=100, batch_size=16, 
                                   trial_length=50, verbose=False)
    
    final_acc = history['accuracy'][-1]
    final_loss = history['loss'][-1]
    print(f"  ✓ Training complete: accuracy={final_acc:.2%}, loss={final_loss:.4f}")
except Exception as e:
    print(f"  ✗ Training failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 4: Extract trajectories and compute deformation
print("\n[4/6] Extracting trajectories and computing deformation...")
try:
    hidden_states, inputs, outputs = task.extract_trajectories(
        rnn, n_trials=5, trial_length=100
    )
    print(f"  Hidden states shape: {hidden_states.shape}")
    
    rot_traj, con_traj, exp_traj, latent_traj = estimate_deformation_from_rnn(
        hidden_states, rnn=None, dt=0.01, latent_dim=3, method='pca_then_local'
    )
    
    print(f"  ✓ Deformation computed: {len(rot_traj)} timesteps")
    print(f"    Rotation range: [{np.min(rot_traj):.3f}, {np.max(rot_traj):.3f}]")
    print(f"    Contraction range: [{np.min(con_traj):.3f}, {np.max(con_traj):.3f}]")
    print(f"    Expansion range: [{np.min(exp_traj):.3f}, {np.max(exp_traj):.3f}]")
except Exception as e:
    print(f"  ✗ Deformation computation failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 5: Extract features and cluster
print("\n[5/6] Extracting features and clustering...")
try:
    # Smooth deformation
    rot_smooth, con_smooth, exp_smooth = smooth_deformation_signals(
        rot_traj, con_traj, exp_traj, sigma=3
    )
    
    # Extract features
    features = extract_rnn_unit_features(
        hidden_states, rot_smooth, con_smooth, exp_smooth, smooth_sigma=3
    )
    print(f"  Features shape: {features.shape}")
    print(f"  Mean |rotation corr|: {np.mean(np.abs(features[:, 0])):.3f}")
    print(f"  Mean |contraction corr|: {np.mean(np.abs(features[:, 1])):.3f}")
    print(f"  Mean |expansion corr|: {np.mean(np.abs(features[:, 2])):.3f}")
    
    # Cluster
    labels, details = classify_units(features, n_clusters=4, method='kmeans', 
                                     return_details=True)
    interpretation = interpret_clusters(features, labels)
    
    print(f"  ✓ Clustering complete: silhouette={details['silhouette']:.3f}")
    
    for cid, interp in interpretation.items():
        print(f"    {interp['name']:15s}: {interp['n_units']:2d} units ({interp['percentage']:5.1f}%)")
    
except Exception as e:
    print(f"  ✗ Feature extraction/clustering failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Test 6: Baseline comparison
print("\n[6/6] Comparing to baseline methods...")
try:
    baseline_comparison = compare_to_baseline(features, labels, hidden_states)
    
    print(f"  Silhouette scores:")
    print(f"    Deformation: {baseline_comparison['deformation']:.3f}")
    print(f"    PCA:         {baseline_comparison['pca']:.3f}")
    print(f"    Raw:         {baseline_comparison['raw']:.3f}")
    
    improvement_pca = (baseline_comparison['deformation'] - baseline_comparison['pca']) / \
                     (baseline_comparison['pca'] + 1e-10) * 100
    print(f"  ✓ Improvement over PCA: {improvement_pca:+.1f}%")
    
except Exception as e:
    print(f"  ✗ Baseline comparison failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✓ ALL INTEGRATION TESTS PASSED")
print("="*70)
print("\nThe RNN experiment pipeline is working correctly!")
print("\nNext steps:")
print("  1. Run full experiment: python scripts/test_rnn.py --task flipflop --epochs 2000")
print("  2. Compare tasks: python scripts/test_rnn.py --task all")
print("  3. Robustness tests: python scripts/test_rnn_robustness.py")
print("\n")
