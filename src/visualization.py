"""
Visualization utilities for robustness testing and plotting results.
"""

import os
import numpy as np
import matplotlib.pyplot as plt


def ensure_dirs():
    """Ensure results/figures directory exists."""
    os.makedirs('results/figures', exist_ok=True)


def plot_bar(labels, stats, path, ylabel):
    """
    Create a bar plot with error bars.
    
    Args:
        labels: List of label strings
        stats: Dictionary mapping labels to {'mean': float, 'ci': float}
        path: Output file path
        ylabel: Label for y-axis
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 5))
    means = [stats[l]['mean'] for l in labels]
    cis = [stats[l]['ci'] for l in labels]
    colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12'][:len(labels)]
    ax.bar(labels, means, yerr=cis, capsize=8, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel(ylabel, fontweight='bold')
    ax.set_ylim([0, max(means) * 1.2])
    ax.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_line(x, stats, path, title):
    """
    Create a line plot with error bands.
    
    Args:
        x: List of x-axis values
        stats: Dictionary mapping x values to {'mean': float, 'ci': float}
        path: Output file path
        title: Title for the plot
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(10, 5))
    x_sorted = sorted(x)
    means = [stats[xi]['mean'] for xi in x_sorted]
    cis = [stats[xi]['ci'] for xi in x_sorted]
    ax.errorbar(x_sorted, means, yerr=cis, marker='o', linewidth=2, capsize=6)
    ax.fill_between(x_sorted, [m-c for m,c in zip(means,cis)], [m+c for m,c in zip(means,cis)], alpha=0.2)
    ax.set_xlabel('Noise Level', fontweight='bold')
    ax.set_ylabel('ARI', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_hist(data, path, title):
    """
    Create a histogram with mean line.
    
    Args:
        data: Array of values
        path: Output file path
        title: Title for the plot
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.hist(data, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    ax.axvline(np.mean(data), color='#e74c3c', linestyle='--', linewidth=2, label=f'Mean: {np.mean(data):.3f}')
    ax.set_xlabel('Adjusted Rand Index', fontweight='bold')
    ax.set_ylabel('Frequency', fontweight='bold')
    ax.set_title(title, fontweight='bold')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_comparison(data1, data2, path):
    """
    Create a comparison bar plot.
    
    Args:
        data1: Array of values for first condition
        data2: Array of values for second condition
        path: Output file path
    """
    ensure_dirs()
    fig, ax = plt.subplots(figsize=(9, 5))
    cats = ['True Types', 'Random Types']
    means = [np.mean(data1), np.mean(data2)]
    cis = [1.96*np.std(data1)/np.sqrt(len(data1)), 1.96*np.std(data2)/np.sqrt(len(data2))]
    colors = ['#2ecc71', '#e74c3c']
    ax.bar(cats, means, yerr=cis, capsize=8, color=colors, alpha=0.8, edgecolor='black')
    ax.set_ylabel('Adjusted Rand Index', fontweight='bold')
    ax.set_ylim([0, 1])
    ax.grid(axis='y', alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
