"""
Visualization utilities for experiments and results.

This module contains all plotting and visualization functions.

Intended contents:
- General plotting utilities
- RNN-specific visualization
- Training curves and diagnostics
- Feature and trajectory plots
"""

from .visualization import (
    ensure_dirs,
    plot_bar,
    plot_line,
    plot_hist,
    plot_comparison
)
from .rnn_visualization import (
    plot_training_curves,
    plot_latent_trajectory_3d,
    plot_deformation_timeseries,
    plot_unit_type_heatmap,
    plot_feature_scatter,
    create_summary_figure
)

__all__ = [
    'ensure_dirs',
    'plot_bar',
    'plot_line',
    'plot_hist',
    'plot_comparison',
    'plot_training_curves',
    'plot_latent_trajectory_3d',
    'plot_deformation_timeseries',
    'plot_unit_type_heatmap',
    'plot_feature_scatter',
    'create_summary_figure',
]
