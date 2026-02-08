"""
Test if deformation-based clustering can recover ground truth modular structure.

This is a critical sanity check before applying the method to real neural data.
If the method can't recover known artificial modularity, it won't work on biology.
"""

import sys
from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.models.rnn_models import VanillaRNN
from src.tasks import get_task
from src.core.deformation_utils import estimate_deformation_from_rnn, smooth_deformation_signals
from src.analysis import (extract_rnn_unit_features, extract_enhanced_rnn_features,
                          classify_units, interpret_clusters, print_cluster_summary,
                          select_optimal_clusters)
from src.visualization import ensure_dirs


class ModularRNN(nn.Module):
    """
    RNN with enforced modularity to test if deformation clustering can recover structure.
    
    Architecture:
    - N_modules separate groups of units
    - Each module has distinct function (integration, oscillation, decision, memory)
    - Connectivity is block-diagonal with sparse inter-module connections
    - Ground truth labels known for validation
    """
    
    def __init__(self, input_size, hidden_size, output_size, n_modules=4, 
                 inter_module_sparsity=0.9, intra_module_sparsity=0.0):
        """
        Args:
            input_size: Number of inputs
            hidden_size: Total number of hidden units (divided among modules)
            output_size: Number of outputs
            n_modules: Number of functional modules (default: 4)
            inter_module_sparsity: Fraction of connections removed between modules (0.9 = 90% removed)
            intra_module_sparsity: Fraction of connections removed within modules (0 = fully connected)
        """
        super().__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_modules = n_modules
        self.inter_module_sparsity = inter_module_sparsity
        
        # Divide units into modules
        self.module_size = hidden_size // n_modules
        self.module_labels = np.repeat(np.arange(n_modules), self.module_size)
        
        # Adjust for remainder
        remainder = hidden_size - (self.module_size * n_modules)
        if remainder > 0:
            self.module_labels = np.concatenate([
                self.module_labels, 
                np.full(remainder, n_modules - 1)  # Add to last module
            ])
        
        # Standard RNN layers
        self.input_to_hidden = nn.Linear(input_size, hidden_size)
        self.hidden_to_hidden = nn.Linear(hidden_size, hidden_size, bias=False)
        self.hidden_to_output = nn.Linear(hidden_size, output_size)
        
        # Create sparse modular connectivity mask
        self._create_modular_mask(inter_module_sparsity, intra_module_sparsity)
        
        # Initialize with modularity in mind
        self._initialize_modular_weights()
        
        print(f"\nModularRNN initialized:")
        print(f"  Total units: {hidden_size}")
        print(f"  Modules: {n_modules}")
        print(f"  Units per module: {self.module_size} (+{remainder} in last module)")
        print(f"  Inter-module sparsity: {inter_module_sparsity*100:.0f}%")
        print(f"  Intra-module sparsity: {intra_module_sparsity*100:.0f}%")
        
        # Print module sizes
        for i in range(n_modules):
            count = np.sum(self.module_labels == i)
            print(f"  Module {i}: {count} units")
    
    def _create_modular_mask(self, inter_sparsity, intra_sparsity):
        """Create binary mask enforcing modular connectivity."""
        mask = torch.ones(self.hidden_size, self.hidden_size)
        
        for i in range(self.hidden_size):
            for j in range(self.hidden_size):
                module_i = self.module_labels[i]
                module_j = self.module_labels[j]
                
                if module_i == module_j:
                    # Within-module connection
                    if np.random.rand() < intra_sparsity:
                        mask[i, j] = 0.0
                else:
                    # Between-module connection
                    if np.random.rand() < inter_sparsity:
                        mask[i, j] = 0.0
        
        # Register as buffer (not a parameter, but part of model state)
        self.register_buffer('connectivity_mask', mask)
        
        # Calculate actual connectivity
        total_possible = self.hidden_size * self.hidden_size
        total_active = torch.sum(mask).item()
        sparsity = 1 - (total_active / total_possible)
        print(f"  Actual overall sparsity: {sparsity*100:.1f}%")
    
    def _initialize_modular_weights(self):
        """Initialize weights to encourage modular function."""
        # Different initialization scales per module
        with torch.no_grad():
            for module_id in range(self.n_modules):
                module_mask = self.module_labels == module_id
                
                # Input weights - give each module different input emphasis
                scale = 0.5 + (module_id * 0.3)  # Vary by module
                self.input_to_hidden.weight[module_mask] *= scale
                
                # Recurrent weights - different dynamics per module
                if module_id % 2 == 0:
                    # Even modules: stronger self-connections (integration)
                    self.hidden_to_hidden.weight[module_mask, :][:, module_mask] *= 1.5
                else:
                    # Odd modules: weaker self-connections (faster dynamics)
                    self.hidden_to_hidden.weight[module_mask, :][:, module_mask] *= 0.7
    
    def forward(self, x, return_hidden_states=False):
        """
        Forward pass with optional hidden state tracking.
        
        Args:
            x: Input tensor (batch, seq_len, input_size)
            return_hidden_states: If True, return hidden states over time
        
        Returns:
            outputs: (batch, seq_len, output_size)
            hidden_states: (batch, seq_len, hidden_size) if return_hidden_states=True
        """
        batch_size, seq_len, _ = x.shape
        
        # Initialize hidden state
        h = torch.zeros(batch_size, self.hidden_size, device=x.device)
        
        outputs = []
        hidden_states = [] if return_hidden_states else None
        
        for t in range(seq_len):
            # Input contribution
            h_input = self.input_to_hidden(x[:, t, :])
            
            # Recurrent contribution (with modular mask applied)
            h_recurrent = torch.matmul(h, (self.hidden_to_hidden.weight * self.connectivity_mask).t())
            
            # Update hidden state
            h = torch.tanh(h_input + h_recurrent)
            
            # Output
            out = self.hidden_to_output(h)
            outputs.append(out)
            
            if return_hidden_states:
                hidden_states.append(h)
        
        outputs = torch.stack(outputs, dim=1)
        
        if return_hidden_states:
            hidden_states = torch.stack(hidden_states, dim=1)
            return outputs, hidden_states
        
        return outputs
    
    def get_ground_truth_labels(self):
        """Return ground truth module assignments."""
        return self.module_labels.copy()


def test_modular_recovery(task_name='context', hidden_size=128, n_modules=4,
                          inter_module_sparsity=0.9, n_epochs=2000,
                          use_enhanced_features=True):
    """
    Test if deformation clustering can recover ground truth modular structure.
    
    Args:
        task_name: Task to train on
        hidden_size: Number of total hidden units
        n_modules: Number of ground truth modules
        inter_module_sparsity: Sparsity between modules (0.9 = 90% removed)
        n_epochs: Training epochs
        use_enhanced_features: Use enhanced (18D) vs basic (3D) features
    
    Returns:
        results: Dict with recovery metrics
    """
    print("\n" + "="*80)
    print(f"MODULAR NETWORK TEST: {task_name.upper()}")
    print("="*80)
    print(f"Ground truth: {n_modules} functional modules")
    print(f"Inter-module sparsity: {inter_module_sparsity*100:.0f}%")
    print("="*80)
    
    # 1. Create modular RNN
    task = get_task(task_name)
    rnn = ModularRNN(
        task.input_size, hidden_size, task.output_size,
        n_modules=n_modules,
        inter_module_sparsity=inter_module_sparsity,
        intra_module_sparsity=0.0  # Fully connected within modules
    )
    
    ground_truth_labels = rnn.get_ground_truth_labels()
    
    # 2. Train on task
    print(f"\n[TRAINING] on {task_name}...")
    rnn, history = task.train_rnn(rnn, n_epochs=n_epochs, verbose=False)
    print(f"  Final accuracy: {history['accuracy'][-1]:.2%}")
    
    # 3. Extract trajectories
    print(f"\n[EXTRACTING] trajectories...")
    n_trials = 50
    trial_length = 200
    hidden_states, inputs, outputs = task.extract_trajectories(
        rnn, n_trials=n_trials, trial_length=trial_length
    )
    
    # 4. Estimate deformation
    print(f"\n[ESTIMATING] deformation signals...")
    rotation, contraction, expansion, latent = estimate_deformation_from_rnn(hidden_states)
    rotation, contraction, expansion = smooth_deformation_signals(
        rotation, contraction, expansion, sigma=5
    )
    
    # 5. Extract features
    if use_enhanced_features:
        print(f"\n[FEATURES] Extracting ENHANCED features...")
        task_info = {
            'trial_length': trial_length,
            'n_trials': n_trials,
            'task_name': task_name
        }
        features = extract_enhanced_rnn_features(
            hidden_states, rotation, contraction, expansion,
            task_info=task_info, inputs=inputs
        )
        print(f"  Feature shape: {features.shape} (enhanced)")
    else:
        print(f"\n[FEATURES] Extracting BASIC features...")
        features = extract_rnn_unit_features(
            hidden_states, rotation, contraction, expansion
        )
        print(f"  Feature shape: {features.shape} (basic)")
    
    # 6. Cluster with ground truth K
    print(f"\n[CLUSTERING] with K={n_modules} (ground truth)...")
    predicted_labels, details = classify_units(features, n_clusters=n_modules, return_details=True)
    
    # 7. Compute recovery metrics
    print(f"\n[EVALUATING] recovery of ground truth modules...")
    
    # Adjusted Rand Index (ARI): 1.0 = perfect recovery, 0.0 = random, <0 = worse than random
    ari = adjusted_rand_score(ground_truth_labels, predicted_labels)
    
    # Normalized Mutual Information (NMI): 1.0 = perfect, 0.0 = independent
    nmi = normalized_mutual_info_score(ground_truth_labels, predicted_labels)
    
    # Purity: fraction of largest class in each cluster
    purity = compute_purity(ground_truth_labels, predicted_labels)
    
    print(f"\n  Recovery Metrics:")
    print(f"    Adjusted Rand Index (ARI): {ari:.3f}  (1.0 = perfect, 0.0 = random)")
    print(f"    Normalized Mutual Info:    {nmi:.3f}  (1.0 = perfect, 0.0 = independent)")
    print(f"    Purity:                     {purity:.3f}  (1.0 = perfect, 1/{n_modules} = random)")
    print(f"    Silhouette:                 {details['silhouette']:.3f}")
    
    # 8. Confusion matrix
    print(f"\n  Confusion Matrix (rows=ground truth, cols=predicted):")
    confusion = compute_confusion_matrix(ground_truth_labels, predicted_labels, n_modules)
    print_confusion_matrix(confusion, n_modules)
    
    # 9. Interpret clusters
    interpretation = interpret_clusters(features, predicted_labels)
    
    print(f"\n  Predicted Cluster Interpretation:")
    for cid, info in interpretation.items():
        # Find which ground truth module this cluster mostly contains
        cluster_mask = predicted_labels == cid
        gt_modules_in_cluster = ground_truth_labels[cluster_mask]
        if len(gt_modules_in_cluster) > 0:
            most_common_gt = np.bincount(gt_modules_in_cluster).argmax()
            purity_score = np.mean(gt_modules_in_cluster == most_common_gt)
            print(f"    Cluster {cid} ({info['name']:25s}): {info['n_units']:3d} units "
                  f"→ mostly GT Module {most_common_gt} (purity={purity_score:.2f})")
        else:
            print(f"    Cluster {cid} ({info['name']:25s}): {info['n_units']:3d} units (empty)")
    
    # 10. Assessment
    print(f"\n" + "="*80)
    print("ASSESSMENT")
    print("="*80)
    
    if ari > 0.7:
        print("✓ EXCELLENT: Method recovers ground truth modules very well")
        print("  → Should work on real neural data with clear functional types")
    elif ari > 0.4:
        print("✓ GOOD: Method partially recovers ground truth structure")
        print("  → May work on real data but expect some mixing")
    elif ari > 0.1:
        print("⚠ WEAK: Method shows some sensitivity to modularity")
        print("  → Real data results will be noisy, need careful validation")
    else:
        print("✗ FAILED: Method does not recover ground truth modules")
        print("  → Not suitable for real neural data without major modifications")
    
    print(f"\nFeature type: {'Enhanced (temporal+contextual)' if use_enhanced_features else 'Basic (global only)'}")
    
    # Return results
    return {
        'ground_truth_labels': ground_truth_labels,
        'predicted_labels': predicted_labels,
        'ari': ari,
        'nmi': nmi,
        'purity': purity,
        'silhouette': details['silhouette'],
        'features': features,
        'interpretation': interpretation,
        'confusion_matrix': confusion,
        'feature_type': 'enhanced' if use_enhanced_features else 'basic'
    }


def compute_purity(true_labels, pred_labels):
    """Compute purity: fraction of dominant class in each cluster."""
    n_clusters = len(np.unique(pred_labels))
    total_correct = 0
    
    for cluster_id in np.unique(pred_labels):
        cluster_mask = pred_labels == cluster_id
        true_labels_in_cluster = true_labels[cluster_mask]
        
        if len(true_labels_in_cluster) > 0:
            # Most common true label in this cluster
            most_common = np.bincount(true_labels_in_cluster).argmax()
            n_correct = np.sum(true_labels_in_cluster == most_common)
            total_correct += n_correct
    
    return total_correct / len(true_labels)


def compute_confusion_matrix(true_labels, pred_labels, n_classes):
    """Compute confusion matrix."""
    confusion = np.zeros((n_classes, n_classes), dtype=int)
    
    for true_label in range(n_classes):
        for pred_label in range(n_classes):
            confusion[true_label, pred_label] = np.sum(
                (true_labels == true_label) & (pred_labels == pred_label)
            )
    
    return confusion


def print_confusion_matrix(confusion, n_classes):
    """Pretty print confusion matrix."""
    # Header
    header = "    "
    for i in range(n_classes):
        header += f"  C{i} "
    print(header)
    print("    " + "-" * (5 * n_classes))
    
    # Rows
    for i in range(n_classes):
        row = f" M{i} |"
        for j in range(n_classes):
            row += f" {confusion[i, j]:3d} "
        print(row)


def compare_modularity_levels(task_name='context', hidden_size=128, n_epochs=1500):
    """
    Test multiple levels of modularity to see sensitivity.
    
    Tests:
    1. Fully connected (no modularity) - baseline
    2. Weak modularity (50% inter-module sparsity)
    3. Medium modularity (75% inter-module sparsity)
    4. Strong modularity (90% inter-module sparsity)
    5. Very strong modularity (99% inter-module sparsity)
    """
    print("\n" + "="*80)
    print("MODULARITY SENSITIVITY TEST")
    print("="*80)
    print("Testing how well method detects different levels of modularity")
    print("="*80)
    
    sparsity_levels = [0.0, 0.5, 0.75, 0.9, 0.99]
    results_basic = []
    results_enhanced = []
    
    for sparsity in sparsity_levels:
        print(f"\n{'='*80}")
        print(f"Testing {sparsity*100:.0f}% inter-module sparsity")
        print(f"{'='*80}")
        
        # Test with BASIC features
        print("\n--- BASIC FEATURES (3D) ---")
        result_basic = test_modular_recovery(
            task_name=task_name,
            hidden_size=hidden_size,
            n_modules=4,
            inter_module_sparsity=sparsity,
            n_epochs=n_epochs,
            use_enhanced_features=False
        )
        results_basic.append(result_basic)
        
        # Test with ENHANCED features
        print("\n--- ENHANCED FEATURES (18D) ---")
        result_enhanced = test_modular_recovery(
            task_name=task_name,
            hidden_size=hidden_size,
            n_modules=4,
            inter_module_sparsity=sparsity,
            n_epochs=n_epochs,
            use_enhanced_features=True
        )
        results_enhanced.append(result_enhanced)
    
    # Plot comparison
    ensure_dirs('results/figures')
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['ari', 'nmi', 'purity']
    titles = ['Adjusted Rand Index', 'Normalized Mutual Information', 'Purity']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx]
        
        basic_scores = [r[metric] for r in results_basic]
        enhanced_scores = [r[metric] for r in results_enhanced]
        
        ax.plot([s*100 for s in sparsity_levels], basic_scores, 
               'o-', label='Basic (3D)', linewidth=2, markersize=8)
        ax.plot([s*100 for s in sparsity_levels], enhanced_scores, 
               's-', label='Enhanced (18D)', linewidth=2, markersize=8)
        
        ax.set_xlabel('Inter-Module Sparsity (%)', fontsize=11)
        ax.set_ylabel(title, fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.05])
    
    plt.tight_layout()
    save_path = 'results/figures/modularity_sensitivity.png'
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"\n\nSaved comparison plot: {save_path}")
    plt.close()
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY: Feature Type Performance")
    print("="*80)
    
    print("\nBasic Features (3D):")
    for sparsity, result in zip(sparsity_levels, results_basic):
        print(f"  {sparsity*100:5.0f}% sparsity → ARI={result['ari']:.3f}, NMI={result['nmi']:.3f}, Purity={result['purity']:.3f}")
    
    print("\nEnhanced Features (18D):")
    for sparsity, result in zip(sparsity_levels, results_enhanced):
        print(f"  {sparsity*100:5.0f}% sparsity → ARI={result['ari']:.3f}, NMI={result['nmi']:.3f}, Purity={result['purity']:.3f}")
    
    # Best performance
    best_basic_idx = np.argmax([r['ari'] for r in results_basic])
    best_enhanced_idx = np.argmax([r['ari'] for r in results_enhanced])
    
    print(f"\nBest Basic Features:    {sparsity_levels[best_basic_idx]*100:.0f}% sparsity, ARI={results_basic[best_basic_idx]['ari']:.3f}")
    print(f"Best Enhanced Features: {sparsity_levels[best_enhanced_idx]*100:.0f}% sparsity, ARI={results_enhanced[best_enhanced_idx]['ari']:.3f}")
    
    return results_basic, results_enhanced


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Test modular network recovery')
    parser.add_argument('--mode', type=str, default='single',
                       choices=['single', 'sweep'],
                       help='Single test or sweep across modularity levels')
    parser.add_argument('--task', type=str, default='context',
                       help='Task to train on')
    parser.add_argument('--hidden-size', type=int, default=128,
                       help='Total hidden units')
    parser.add_argument('--n-modules', type=int, default=4,
                       help='Number of ground truth modules')
    parser.add_argument('--sparsity', type=float, default=0.9,
                       help='Inter-module sparsity (0.9 = 90%% removed)')
    parser.add_argument('--epochs', type=int, default=1500,
                       help='Training epochs')
    parser.add_argument('--features', type=str, default='enhanced',
                       choices=['basic', 'enhanced'],
                       help='Feature type to use')
    
    args = parser.parse_args()
    
    if args.mode == 'single':
        # Single test
        results = test_modular_recovery(
            task_name=args.task,
            hidden_size=args.hidden_size,
            n_modules=args.n_modules,
            inter_module_sparsity=args.sparsity,
            n_epochs=args.epochs,
            use_enhanced_features=(args.features == 'enhanced')
        )
    
    elif args.mode == 'sweep':
        # Sweep across modularity levels
        results_basic, results_enhanced = compare_modularity_levels(
            task_name=args.task,
            hidden_size=args.hidden_size,
            n_epochs=args.epochs
        )
