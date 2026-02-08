"""
Test script for enhanced RNN feature extraction.

Demonstrates how to use extract_enhanced_rnn_features() to get temporal
and contextual resolution for better unit classification.
"""

import sys
from pathlib import Path
import numpy as np
import torch

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.rnn_models import VanillaRNN
from src.tasks import get_task
from src.core.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.analysis import (extract_rnn_unit_features, extract_enhanced_rnn_features,
                          classify_units, interpret_clusters, print_cluster_summary,
                          select_optimal_clusters)


def compare_basic_vs_enhanced_features(task_name='context', hidden_size=128, n_epochs=500):
    """
    Compare basic vs enhanced feature extraction on a trained RNN.
    
    Demonstrates:
    1. Basic features (3D): global R, C, E correlations
    2. Enhanced features (12D or 18D): temporal + contextual resolution
    3. How enhanced features produce more interpretable clusters
    """
    print("\n" + "="*80)
    print(f"COMPARING BASIC VS ENHANCED FEATURES: {task_name.upper()}")
    print("="*80)
    
    # 1. Train RNN
    print("\n[1] Training RNN...")
    task = get_task(task_name)
    rnn = VanillaRNN(task.input_size, hidden_size, task.output_size)
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
    print(f"    Final accuracy: {history['accuracy'][-1]:.2%}")
    
    # 2. Extract trajectories
    print("\n[2] Extracting trajectories...")
    n_trials = 50
    trial_length = 200
    hidden_states, inputs, outputs = task.extract_trajectories(
        rnn, n_trials=n_trials, trial_length=trial_length
    )
    print(f"    Hidden states shape: {hidden_states.shape}")
    print(f"    Inputs shape: {inputs.shape}")
    
    # 3. Estimate deformation
    print("\n[3] Estimating deformation signals...")
    rotation, contraction, expansion, latent = estimate_deformation_from_rnn(hidden_states)
    rotation, contraction, expansion = smooth_deformation_signals(
        rotation, contraction, expansion, sigma=5
    )
    print(f"    Rotation std: {np.std(rotation):.4f}")
    print(f"    Contraction std: {np.std(contraction):.4f}")
    print(f"    Expansion std: {np.std(expansion):.4f}")
    
    # 4. Extract BASIC features
    print("\n[4] Extracting BASIC features (3D)...")
    basic_features = extract_rnn_unit_features(
        hidden_states, rotation, contraction, expansion
    )
    print(f"    Feature shape: {basic_features.shape}")
    print(f"    Feature range: [{basic_features.min():.3f}, {basic_features.max():.3f}]")
    
    # 5. Extract ENHANCED features
    print("\n[5] Extracting ENHANCED features (12D or 18D)...")
    task_info = {
        'trial_length': trial_length,
        'n_trials': n_trials,
        'task_name': task_name
    }
    enhanced_features = extract_enhanced_rnn_features(
        hidden_states, rotation, contraction, expansion,
        task_info=task_info,
        inputs=inputs
    )
    print(f"    Feature shape: {enhanced_features.shape}")
    print(f"    Feature composition:")
    print(f"      - [0:3]:   Global R, C, E")
    print(f"      - [3:6]:   Early R, C, E")
    print(f"      - [6:9]:   Middle R, C, E")
    print(f"      - [9:12]:  Late R, C, E")
    if enhanced_features.shape[1] >= 18:
        print(f"      - [12:15]: Context A R, C, E")
        print(f"      - [15:18]: Context B R, C, E")
    
    # 6. Cluster with BASIC features
    print("\n[6] Clustering with BASIC features...")
    optimal_k_basic, _ = select_optimal_clusters(basic_features, max_clusters=6)
    labels_basic, details_basic = classify_units(
        basic_features, n_clusters=optimal_k_basic, return_details=True
    )
    interpretation_basic = interpret_clusters(basic_features, labels_basic)
    
    print(f"    Optimal k: {optimal_k_basic}")
    print(f"    Silhouette: {details_basic['silhouette']:.3f}")
    print(f"\n    Cluster distribution:")
    for cid, info in interpretation_basic.items():
        print(f"      {info['name']:15s}: {info['n_units']:3d} units ({info['percentage']:5.1f}%)")
    
    # 7. Cluster with ENHANCED features
    print("\n[7] Clustering with ENHANCED features...")
    optimal_k_enhanced, _ = select_optimal_clusters(enhanced_features, max_clusters=8)
    labels_enhanced, details_enhanced = classify_units(
        enhanced_features, n_clusters=optimal_k_enhanced, return_details=True
    )
    interpretation_enhanced = interpret_clusters(enhanced_features, labels_enhanced)
    
    print(f"    Optimal k: {optimal_k_enhanced}")
    print(f"    Silhouette: {details_enhanced['silhouette']:.3f}")
    print(f"\n    Cluster distribution:")
    for cid, info in interpretation_enhanced.items():
        print(f"      {info['name']:30s}: {info['n_units']:3d} units ({info['percentage']:5.1f}%)")
    
    # 8. Show detailed comparison
    print("\n" + "="*80)
    print("BASIC FEATURES - Cluster Interpretation")
    print("="*80)
    print_cluster_summary(interpretation_basic, feature_type='deformation')
    
    print("\n" + "="*80)
    print("ENHANCED FEATURES - Cluster Interpretation")
    print("="*80)
    print_cluster_summary(interpretation_enhanced, feature_type='deformation')
    
    # 9. Analysis
    print("\n" + "="*80)
    print("COMPARISON ANALYSIS")
    print("="*80)
    
    # Count unique cluster types
    basic_types = set(info['name'].replace('?', '').strip() for info in interpretation_basic.values())
    enhanced_types = set(info['name'].replace('?', '').strip() for info in interpretation_enhanced.values())
    
    print(f"\nUnique cluster types:")
    print(f"  Basic features:    {len(basic_types)} types")
    print(f"  Enhanced features: {len(enhanced_types)} types")
    
    print(f"\nDistinctiveness improvement:")
    improvement = (len(enhanced_types) - len(basic_types)) / max(len(basic_types), 1) * 100
    print(f"  {improvement:+.1f}% more distinct cluster types with enhanced features")
    
    print(f"\nCluster quality:")
    print(f"  Basic silhouette:    {details_basic['silhouette']:.3f}")
    print(f"  Enhanced silhouette: {details_enhanced['silhouette']:.3f}")
    print(f"  Improvement:         {details_enhanced['silhouette'] - details_basic['silhouette']:+.3f}")
    
    print("\n" + "="*80)
    print("KEY INSIGHT:")
    print("="*80)
    if len(enhanced_types) > len(basic_types):
        print("✓ Enhanced features successfully differentiated units that basic features")
        print("  grouped together. This allows finer-grained functional understanding.")
    else:
        print("⚠ Enhanced features did not produce more cluster types.")
        print("  This may indicate:")
        print("  - Task is too simple (no temporal/contextual specialization)")
        print("  - Network hasn't learned specialized representations")
        print("  - Need longer training or larger network")
    
    return {
        'basic_features': basic_features,
        'enhanced_features': enhanced_features,
        'basic_interpretation': interpretation_basic,
        'enhanced_interpretation': interpretation_enhanced,
        'basic_silhouette': details_basic['silhouette'],
        'enhanced_silhouette': details_enhanced['silhouette']
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test enhanced feature extraction')
    parser.add_argument('--task', type=str, default='context',
                       choices=['context', 'parametric', 'flipflop', 'gonogo'],
                       help='Task to test')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='RNN hidden size')
    parser.add_argument('--epochs', type=int, default=500,
                       help='Training epochs')
    
    args = parser.parse_args()
    
    results = compare_basic_vs_enhanced_features(
        task_name=args.task,
        hidden_size=args.hidden_size,
        n_epochs=args.epochs
    )
