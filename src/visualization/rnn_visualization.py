"""
RNN-specific visualization utilities.

Extends src/visualization.py with plots tailored for RNN experiment analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from pathlib import Path


def plot_training_curves(history, save_path, title='Training Progress'):
    """
    Plot training loss and accuracy over epochs.
    
    Args:
        history: Dict with 'loss' and 'accuracy' lists
        save_path: Where to save figure
        title: Plot title
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    epochs = np.arange(1, len(history['loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['loss'], color='#e74c3c', linewidth=2, alpha=0.8)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss (MSE)', fontsize=12)
    ax1.set_title('Training Loss', fontsize=14, fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # Accuracy
    ax2.plot(epochs, np.array(history['accuracy']) * 100, 
            color='#2ecc71', linewidth=2, alpha=0.8)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Accuracy (%)', fontsize=12)
    ax2.set_title('Training Accuracy', fontsize=14, fontweight='bold')
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 105])
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_latent_trajectory_3d(latent_trajectory, deformation_trajectories=None,
                               save_path='latent_trajectory.png', 
                               title='Latent Space Trajectory'):
    """
    Plot 3D trajectory in latent space, optionally colored by deformation mode.
    
    Args:
        latent_trajectory: (n_timesteps, 3) array
        deformation_trajectories: Optional dict with 'rotation', 'contraction', 'expansion'
        save_path: Where to save figure
        title: Plot title
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(14, 5))
    
    # Determine coloring
    if deformation_trajectories is not None:
        rot = deformation_trajectories.get('rotation', None)
        con = deformation_trajectories.get('contraction', None)
        exp = deformation_trajectories.get('expansion', None)
        
        # Create 3 subplots
        for idx, (mode_name, mode_traj, cmap) in enumerate([
            ('Rotation', rot, 'Reds'),
            ('Contraction', con, 'Blues'),
            ('Expansion', exp, 'Greens')
        ]):
            ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
            
            if mode_traj is not None:
                # Normalize for coloring
                colors = mode_traj / (np.max(mode_traj) + 1e-10)
                
                # Plot trajectory
                scatter = ax.scatter(latent_trajectory[:, 0],
                                   latent_trajectory[:, 1],
                                   latent_trajectory[:, 2],
                                   c=colors, cmap=cmap, s=1, alpha=0.6)
                
                plt.colorbar(scatter, ax=ax, label=mode_name, shrink=0.8)
            
            ax.set_xlabel('PC1', fontsize=10)
            ax.set_ylabel('PC2', fontsize=10)
            ax.set_zlabel('PC3', fontsize=10)
            ax.set_title(f'Colored by {mode_name}', fontsize=12, fontweight='bold')
            ax.view_init(elev=20, azim=45)
    
    else:
        # Single plot, time-colored
        ax = fig.add_subplot(111, projection='3d')
        
        time_colors = np.arange(len(latent_trajectory))
        scatter = ax.scatter(latent_trajectory[:, 0],
                           latent_trajectory[:, 1],
                           latent_trajectory[:, 2],
                           c=time_colors, cmap='viridis', s=1, alpha=0.6)
        
        plt.colorbar(scatter, ax=ax, label='Time', shrink=0.8)
        ax.set_xlabel('PC1', fontsize=12)
        ax.set_ylabel('PC2', fontsize=12)
        ax.set_zlabel('PC3', fontsize=12)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_deformation_timeseries(rotation, contraction, expansion, save_path,
                                title='Deformation Signals Over Time'):
    """
    Plot all three deformation signals as time series.
    
    Args:
        rotation, contraction, expansion: (n_timesteps,) arrays
        save_path: Where to save figure
        title: Plot title
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    
    time = np.arange(len(rotation))
    
    # Rotation
    axes[0].plot(time, rotation, color='#e74c3c', linewidth=1.5, alpha=0.8)
    axes[0].fill_between(time, 0, rotation, color='#e74c3c', alpha=0.2)
    axes[0].set_ylabel('Rotation', fontsize=12, fontweight='bold')
    axes[0].grid(alpha=0.3)
    axes[0].set_title('Rotation Magnitude', fontsize=12)
    
    # Contraction
    axes[1].plot(time, contraction, color='#3498db', linewidth=1.5, alpha=0.8)
    axes[1].fill_between(time, 0, contraction, color='#3498db', alpha=0.2)
    axes[1].set_ylabel('Contraction', fontsize=12, fontweight='bold')
    axes[1].grid(alpha=0.3)
    axes[1].set_title('Contraction Magnitude', fontsize=12)
    
    # Expansion
    axes[2].plot(time, expansion, color='#2ecc71', linewidth=1.5, alpha=0.8)
    axes[2].fill_between(time, 0, expansion, color='#2ecc71', alpha=0.2)
    axes[2].set_ylabel('Expansion', fontsize=12, fontweight='bold')
    axes[2].set_xlabel('Time Step', fontsize=12)
    axes[2].grid(alpha=0.3)
    axes[2].set_title('Expansion Magnitude', fontsize=12)
    
    plt.suptitle(title, fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_unit_type_heatmap(hidden_states, labels, interpretation, save_path,
                           downsample=10, title='Unit Activity by Type'):
    """
    Plot heatmap of unit activations grouped by classification.
    
    Args:
        hidden_states: (n_units, n_timesteps) array
        labels: (n_units,) cluster assignments
        interpretation: Dict from interpret_clusters()
        save_path: Where to save figure
        downsample: Factor to downsample timesteps for visualization
        title: Plot title
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    # Downsample for visualization
    hidden_states_ds = hidden_states[:, ::downsample]
    
    # Sort units by cluster
    sort_idx = np.argsort(labels)
    hidden_states_sorted = hidden_states_ds[sort_idx]
    labels_sorted = labels[sort_idx]
    
    # Normalize each unit
    hidden_states_norm = (hidden_states_sorted - np.mean(hidden_states_sorted, axis=1, keepdims=True)) / \
                        (np.std(hidden_states_sorted, axis=1, keepdims=True) + 1e-10)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Plot heatmap
    im = ax.imshow(hidden_states_norm, aspect='auto', cmap='RdBu_r', 
                  vmin=-3, vmax=3, interpolation='nearest')
    
    # Add  cluster boundaries
    cluster_boundaries = []
    current_cluster = labels_sorted[0]
    for i, label in enumerate(labels_sorted):
        if label != current_cluster:
            cluster_boundaries.append(i)
            current_cluster = label
    
    for boundary in cluster_boundaries:
        ax.axhline(boundary - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.7)
    
    # Label clusters on y-axis
    tick_positions = []
    tick_labels = []
    current_pos = 0
    for cluster_id in sorted(np.unique(labels)):
        cluster_units = np.sum(labels_sorted == cluster_id)
        if cluster_units > 0:
            tick_positions.append(current_pos + cluster_units / 2)
            interp = interpretation.get(cluster_id, {})
            tick_labels.append(f"{interp.get('name', f'C{cluster_id}')} ({cluster_units})")
            current_pos += cluster_units
    
    ax.set_yticks(tick_positions)
    ax.set_yticklabels(tick_labels, fontsize=10)
    ax.set_xlabel('Time Step', fontsize=12)
    ax.set_ylabel('Units (grouped by type)', fontsize=12)
    ax.set_title(title, fontsize=14, fontweight='bold')
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Normalized Activity', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_scatter(features, labels, interpretation, save_path,
                         title='Unit Features (3D Deformation Space)'):
    """
    Plot unit features in 3D feature space (rotation, contraction, expansion).
    
    Args:
        features: (n_units, 3) feature array
        labels: (n_units,) cluster assignments
        interpretation: Dict from interpret_clusters()
        save_path: Where to save figure
        title: Plot title
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection='3d')
    
    # Color map for clusters
    colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']
    
    # Plot each cluster
    for cluster_id in sorted(np.unique(labels)):
        mask = labels == cluster_id
        interp = interpretation.get(cluster_id, {})
        name = interp.get('name', f'Cluster {cluster_id}')
        color = colors[cluster_id % len(colors)]
        
        ax.scatter(features[mask, 0],  # Rotation
                  features[mask, 1],  # Contraction
                  features[mask, 2],  # Expansion
                  c=color, label=name, s=50, alpha=0.6, edgecolors='k', linewidth=0.5)
    
    ax.set_xlabel('Rotation Correlation', fontsize=12, fontweight='bold')
    ax.set_ylabel('Contraction Correlation', fontsize=12, fontweight='bold')
    ax.set_zlabel('Expansion Correlation', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper left')
    ax.view_init(elev=20, azim=45)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def create_summary_figure(results, save_path):
    """
    Create comprehensive summary figure for a single RNN experiment.
    
    Args:
        results: Dict from run_single_task_experiment()
        save_path: Where to save figure
    """
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)
    
    # 1. Training curves
    ax1 = fig.add_subplot(gs[0, 0])
    epochs = np.arange(1, len(results['history']['loss']) + 1)
    ax1.plot(epochs, results['history']['loss'], color='#e74c3c', linewidth=2)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training Loss', fontweight='bold')
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # 2. Accuracy
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(epochs, np.array(results['history']['accuracy']) * 100, 
            color='#2ecc71', linewidth=2)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title('Training Accuracy', fontweight='bold')
    ax2.grid(alpha=0.3)
    
    # 3. Unit type distribution
    ax3 = fig.add_subplot(gs[0, 2])
    interp = results['interpretation']
    names = [interp[cid]['name'] for cid in sorted(interp.keys())]
    percentages = [interp[cid]['percentage'] for cid in sorted(interp.keys())]
    colors_bar = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    ax3.bar(names, percentages, color=colors_bar[:len(names)], alpha=0.7, edgecolor='k')
    ax3.set_ylabel('Percentage (%)')
    ax3.set_title('Unit Type Distribution', fontweight='bold')
    ax3.set_ylim([0, 100])
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Deformation timeseries (all 3)
    time = np.arange(len(results['deformation']['rotation']))
    
    ax4 = fig.add_subplot(gs[1, :])
    ax4.plot(time, results['deformation']['rotation'], 
            label='Rotation', color='#e74c3c', linewidth=1.5, alpha=0.7)
    ax4.plot(time, results['deformation']['contraction'], 
            label='Contraction', color='#3498db', linewidth=1.5, alpha=0.7)
    ax4.plot(time, results['deformation']['expansion'], 
            label='Expansion', color='#2ecc71', linewidth=1.5, alpha=0.7)
    ax4.set_xlabel('Time Step')
    ax4.set_ylabel('Magnitude')
    ax4.set_title('Deformation Signals', fontweight='bold')
    ax4.legend()
    ax4.grid(alpha=0.3)
    
    # 5. 3D latent trajectory
    ax5 = fig.add_subplot(gs[2, 0], projection='3d')
    latent = results['latent_trajectory']
    scatter = ax5.scatter(latent[:, 0], latent[:, 1], latent[:, 2],
                         c=np.arange(len(latent)), cmap='viridis', s=1, alpha=0.5)
    ax5.set_xlabel('PC1')
    ax5.set_ylabel('PC2')
    ax5.set_zlabel('PC3')
    ax5.set_title('Latent Trajectory', fontweight='bold')
    
    # 6. Feature scatter (2D projection)
    ax6 = fig.add_subplot(gs[2, 1])
    features = results['features']
    labels = results['labels']
    colors_scatter = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12']
    for cid in sorted(np.unique(labels)):
        mask = labels == cid
        ax6.scatter(features[mask, 0], features[mask, 1],
                   c=colors_scatter[cid % len(colors_scatter)],
                   label=interp[cid]['name'], s=30, alpha=0.6, edgecolors='k', linewidth=0.5)
    ax6.set_xlabel('Rotation Corr.')
    ax6.set_ylabel('Contraction Corr.')
    ax6.set_title('Feature Space (2D)', fontweight='bold')
    ax6.legend(fontsize=8)
    ax6.grid(alpha=0.3)
    
    # 7. Baseline comparison
    ax7 = fig.add_subplot(gs[2, 2])
    comparison = results['baseline_comparison']
    methods = list(comparison.keys())
    scores = list(comparison.values())
    ax7.bar(methods, scores, color=['#2ecc71', '#95a5a6', '#95a5a6'], 
           alpha=0.7, edgecolor='k')
    ax7.set_ylabel('Silhouette Score')
    ax7.set_title('Method Comparison', fontweight='bold')
    ax7.set_ylim([0, max(scores) * 1.2])
    plt.setp(ax7.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # Main title
    task_name = results['task_name'].upper()
    arch_name = results['architecture'].upper()
    accuracy = results['final_accuracy']
    silhouette = results['silhouette']
    
    fig.suptitle(f'{task_name} / {arch_name} | Accuracy: {accuracy:.1%} | Silhouette: {silhouette:.3f}',
                fontsize=16, fontweight='bold', y=0.995)
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
