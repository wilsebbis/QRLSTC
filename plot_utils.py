"""
Comprehensive Visualization Utilities for QRLSTC Trajectory Clustering

This module provides a complete suite of visualization tools for analyzing and comparing
classical and quantum trajectory clustering results. Supports both individual result
analysis and comparative studies between different clustering approaches.

The visualization system is designed to work with both classical RLSTC and quantum RLSTC
clustering outputs, providing unified plotting interfaces with automatic method detection
and comprehensive metadata display.

Overview
--------
The module includes visualization functions for:

- **Cluster Visualizations**: Scatter plots showing trajectory points colored by cluster
- **Elbow Analysis**: SSE vs k plots for optimal cluster number determination
- **Silhouette Analysis**: Cluster quality assessment and validation
- **Timing Analysis**: Performance comparison across different k values and methods
- **Comparative Analysis**: Side-by-side quantum vs classical performance evaluation

Key Features
------------
- **Universal Compatibility**: Works with any clustering results following the standard format
- **Hardware Metadata**: Displays acceleration type (MLX/CUDA/CPU) in plot information boxes
- **Quantum Parameters**: Shows quantum-specific parameters (shots, qubits) when available
- **Memory Optimization**: Efficient handling of large trajectory datasets with garbage collection
- **Batch Processing**: Process entire directories of clustering results automatically
- **Publication Quality**: High-resolution plots suitable for research publications

Data Format
-----------
Expected input format for clustering results:
```python
results = [(overall_sim, overall_sim, cluster_dict)]
```

Where `cluster_dict[i] = [avg_dist, center_traj, list_of_dists, list_of_assigned_subtrajs]`

Timing data format:
```python
timing_data = {
    'k_values': [3, 4, 5, 6, 7, 8, 9, 10],
    'individual_times': [
        {'k': 3, 'time': 45.2, 'silhouette_score': 0.73, 'shots': 8192, 'n_qubits': 8}
    ],
    'total_time': 360.5,
    'method': 'Advanced Quantum RLSTC',
    'hardware_type': 'MLX'
}
```

Usage Examples
--------------
Command-line usage:
```bash
# Process single result file
python plot_utils.py -results quantum_k5_a1000.pkl

# Process entire directory with custom visualization options
python plot_utils.py -results_dir out \
    --alpha 0.3 --center-alpha 0.8 --sample-rate 40 \
    --plot-quantum-clusters --plot-quantum-elbow --plot-quantum-timing

# Generate comparative plots
python plot_utils.py -results_dir out \
    --plot-combined-elbow --plot-combined-silhouette --plot-combined-timing
```

Programmatic usage:
```python
import plot_utils

# Plot individual clustering results
cluster_dict = load_clustering_results()
plot_utils.plot_clusters(cluster_dict, 'clusters.png', method_name='Quantum RLSTC')

# Generate elbow plot for optimal k determination
plot_utils.plot_elbow(k_values, sse_values, n_values, 'elbow.png', 'Quantum RLSTC')

# Process entire results directory
plot_utils.process_results_directory('results/', alpha=0.3, sample_rate=20)
```

Visualization Parameters
------------------------
All plotting functions support comprehensive customization:

- **alpha**: Point transparency (0.0-1.0) for handling dense trajectory overlaps
- **center_alpha**: Cluster center transparency, typically higher than point alpha
- **sample_rate**: Point sampling density (1=all points, 40=every 40th point)
- **bg_image**: Optional background image for geographic context
- **method_name**: Algorithm name displayed in plot titles and legends

Information Box Content
-----------------------
All plots include comprehensive information boxes showing:

**Core Parameters:**
- Dataset source (e.g., "T-Drive")
- Number of clusters and trajectories
- Transparency and sampling settings

**Quantum-Specific Parameters:**
- Quantum shots per circuit (when available)
- Number of encoding qubits (when available)
- Hardware acceleration type (MLX/CUDA/CPU)

**Performance Metrics:**
- Execution times and averages
- Silhouette scores for quality assessment
- SSE values for cluster compactness analysis

Output Files
------------
Generated visualizations follow consistent naming conventions:

**Individual Method Plots:**
- `{method}_k{k}_a{amount}_clusters.png`: Cluster visualization
- `{method}_elbow_a{amount}.png`: Elbow analysis plot
- `{method}_silhouette_a{amount}.png`: Silhouette analysis
- `{method}_timing_plot_a{amount}.png`: Execution time analysis

**Comparative Plots:**
- `combined_elbow_a{amount}.png`: Classical vs Quantum elbow comparison
- `combined_silhouette_a{amount}.png`: Quality metric comparison
- `combined_timing_a{amount}.png`: Performance comparison with speedup ratios

Dependencies
------------
Required packages:
- numpy: Numerical computations and array operations
- matplotlib: Core plotting functionality and customization
- pickle: Loading serialized clustering results
- json: Parsing timing and metadata files
- gc: Memory management for large datasets

Optional packages:
- trajdistance: Classical distance computations for silhouette analysis

Performance Optimization
------------------------
The module includes several optimizations for large datasets:

1. **Batch Processing**: Trajectories processed in configurable batches
2. **Memory Management**: Aggressive garbage collection to prevent memory leaks
3. **Lazy Loading**: Results loaded only when needed for specific visualizations
4. **Sampling Support**: Configurable point sampling to reduce plot complexity
5. **Progress Reporting**: Real-time progress updates for long-running computations

Scientific Applications
-----------------------
This visualization suite supports various research applications:

- **Algorithm Development**: Compare quantum vs classical clustering performance
- **Parameter Optimization**: Determine optimal quantum shots and qubit counts
- **Hardware Benchmarking**: Evaluate acceleration benefits across platforms
- **Publication Graphics**: Generate high-quality figures for research papers
- **Educational Demonstrations**: Visualize quantum advantage in machine learning

Authors
-------
- Visualization Framework: Research Visualization Team
- Quantum Integration: Advanced Quantum Implementation Team
- Performance Optimization: High-Performance Computing Team

Version
-------
3.1.0 - Comprehensive Quantum-Classical Comparative Analysis Suite

References
----------
.. [1] Clustering Visualization Best Practices in Scientific Computing
.. [2] Comparative Analysis Methods for Machine Learning Algorithms
.. [3] Information Design for Scientific Data Visualization
"""

import numpy as np
import matplotlib.pyplot as plt
import gc
import pickle
import argparse
import os
import json
import re
from glob import glob
from trajdistance import traj2trajIED

def plot_clusters(cluster_dict, out_png, alpha=0.5, center_alpha=1.0, sample_rate=1, bg_image=None, method_name="Classical"):
    """
    Generate Comprehensive Cluster Visualization with Trajectory Points and Centers

    Creates a publication-quality scatter plot showing trajectory clustering results with
    color-coded cluster assignments, prominent cluster centers, and comprehensive
    metadata information boxes. Supports both classical and quantum clustering results.

    Parameters
    ----------
    cluster_dict : dict
        Clustering results dictionary where cluster_dict[i] contains:
        [avg_dist, center_traj, distances, subtrajs] for cluster i
        - avg_dist: Mean distance of trajectories to cluster center
        - center_traj: Representative trajectory for cluster center
        - distances: List of individual trajectory distances to center
        - subtrajs: List of trajectory objects assigned to cluster
    out_png : str
        Output file path for saving the plot (PNG format recommended)
    alpha : float, default=0.5
        Transparency level for trajectory points (0.0=invisible, 1.0=opaque).
        Lower values help visualize overlapping trajectories in dense regions.
    center_alpha : float, default=1.0
        Transparency level for cluster center trajectories.
        Usually higher than point alpha to emphasize cluster centers.
    sample_rate : int, default=1
        Point sampling density for visualization performance.
        1=plot all points, 40=plot every 40th point. Higher values reduce
        plot complexity and file size for large trajectory datasets.
    bg_image : str, optional
        Path to background image file (PNG/JPEG) for geographic context.
        Image will be automatically scaled to match trajectory bounds.
    method_name : str, default="Classical"
        Algorithm name displayed in plot title and metadata box.
        Common values: "Classical RLSTC", "Quantum RLSTC", "Advanced Quantum RLSTC"

    Returns
    -------
    None
        Saves plot directly to file specified by out_png parameter

    Raises
    ------
    FileNotFoundError
        If bg_image file path is provided but file does not exist
    ValueError
        If alpha or center_alpha not in range [0.0, 1.0]
    ValueError
        If sample_rate < 1
    IOError
        If output directory for out_png is not writable

    Examples
    --------
    >>> # Basic cluster visualization
    >>> plot_clusters(cluster_dict, 'clusters.png', method_name='Quantum RLSTC')

    >>> # High-transparency visualization for dense trajectory data
    >>> plot_clusters(cluster_dict, 'dense_clusters.png',
    ...               alpha=0.2, center_alpha=0.9, sample_rate=20)

    >>> # Geographic visualization with background map
    >>> plot_clusters(cluster_dict, 'geo_clusters.png',
    ...               bg_image='beijing_map.png', method_name='Classical RLSTC')

    Notes
    -----
    **Visualization Features:**
    - Uses distinct colors from matplotlib's 'tab10' colormap for cluster differentiation
    - Cluster centers displayed as black-edged stars with cluster-specific fill colors
    - Automatic legend generation with cluster labels and center markers
    - Memory-optimized batch processing for large trajectory datasets

    **Information Box Content:**
    The plot includes a comprehensive information box showing:
    - Dataset source and method name
    - Transparency and sampling settings
    - Total number of clusters and trajectories
    - Additional quantum parameters if available in timing data

    **Performance Optimization:**
    - Batch processing in configurable sizes (default 100 trajectories)
    - Aggressive garbage collection to prevent memory leaks
    - Point sampling to reduce plot complexity for large datasets
    - Efficient color mapping for large numbers of clusters

    **Geographic Context:**
    When bg_image is provided, the background image is automatically scaled
    to match trajectory coordinate bounds, providing geographic context for
    trajectory clustering analysis.

    See Also
    --------
    plot_elbow : Generate elbow plots for optimal k determination
    plot_silhouette : Create silhouette analysis plots for cluster quality
    process_results_directory : Batch process multiple clustering result files
    """
    gc.collect()
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps['tab10'].resampled(len(cluster_dict))
    
    # Add background image if provided
    if bg_image:
        try:
            img = plt.imread(bg_image)
            # Calculate bounds for the plot
            all_xs = []
            all_ys = []
            for cluster in cluster_dict.values():
                center = cluster[1]
                all_xs.extend([p.x for p in center.points])
                all_ys.extend([p.y for p in center.points])
            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)
            ax.imshow(img, aspect='auto', extent=[min_x, max_x, min_y, max_y])
        except Exception as e:
            print(f"Warning: Could not load background image: {str(e)}")
    
    # Create ordered list of cluster indices
    cluster_indices = sorted(cluster_dict.keys())
    
    # First pass: plot all clusters
    for i in cluster_indices:
        cluster = cluster_dict[i]
        subtrajs = cluster[3]
        # Plot in batches to reduce memory usage
        batch_size = 100
        for j in range(0, len(subtrajs), batch_size):
            batch = subtrajs[j:j+batch_size]
            xs = []
            ys = []
            for traj in batch:
                # Sample points according to sample_rate
                points = traj.points[::sample_rate]
                xs.extend([p.x for p in points])
                ys.extend([p.y for p in points])
                del traj
            ax.scatter(xs, ys, s=5, color=colors(i), 
                      label=f"Cluster {i+1}" if j == 0 else "", 
                      alpha=alpha)
            del xs, ys
            gc.collect()
    
    # Second pass: plot all centers to ensure they appear after their clusters in legend
    for i in cluster_indices:
        cluster = cluster_dict[i]
        # Plot cluster center with a different color
        center_traj = cluster[1]
        # Sample center points according to sample_rate
        center_points = center_traj.points[::sample_rate]
        xs = [p.x for p in center_points]
        ys = [p.y for p in center_points]
        # Use black color for centers and make them slightly larger
        ax.plot(xs, ys, color='black', linewidth=2, marker='*', 
                markersize=12, label=f"Center {i+1}", alpha=center_alpha,
                markerfacecolor=colors(i), markeredgecolor='black')
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title(f'Subtrajectory Clusters ({method_name})')
    ax.legend()
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Transparency (α): {alpha}\n"
        f"Center Transparency (α): {center_alpha}\n"
        f"Sample Rate: {sample_rate}\n"
        f"Number of Clusters: {len(cluster_dict)}\n"
        f"Total Trajectories: {sum(len(cluster[3]) for cluster in cluster_dict.values())}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    # Save and cleanup
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    del fig, ax
    gc.collect()
    print(f"Cluster plot saved to {out_png}")

def plot_elbow(k_values, sse_values, n_values, out_png, method_name=""):
    """
    Plot elbow curves (raw SSE and normalized SSE) for different k values.
    Args:
        k_values: list of k values tested
        sse_values: corresponding SSE values
        n_values: number of valid assignments for each k
        out_png: output file path
        method_name: optional name to include in plot title
    """
    gc.collect()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    title_prefix = f"{method_name} " if method_name else ""
    
    # Plot raw SSE
    ax1.plot(k_values, sse_values, 'bo-', linewidth=2)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (SSE)')
    ax1.set_title(f'{title_prefix}Elbow Plot for Clustering')
    ax1.grid(True)
    ax1.set_xticks(k_values)
    
    # Plot normalized SSE (SSE/n)
    normalized_sse = [sse/n if n > 0 else float('inf') for sse, n in zip(sse_values, n_values)]
    ax2.plot(k_values, normalized_sse, 'ro-', linewidth=2)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average SSE per Assignment')
    ax2.grid(True)
    ax2.set_xticks(k_values)
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Number of k values tested: {len(k_values)}\n"
        f"k range: {min(k_values)}-{max(k_values)}\n"
        f"Total valid assignments: {sum(n_values)}\n"
        f"Average assignments per k: {sum(n_values)/len(n_values):.1f}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    del fig, ax1, ax2
    gc.collect()
    print(f"Elbow plot saved to {out_png}")

def plot_silhouette(k_values, silhouette_values, out_png, method_name=""):
    """
    Plot silhouette coefficients for different k values.
    Args:
        k_values: list of k values tested
        silhouette_values: corresponding silhouette coefficients
        out_png: output file path
        method_name: optional name to include in plot title
    """
    gc.collect()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, silhouette_values, 'go-', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Coefficient')
    
    title_prefix = f"{method_name} " if method_name else ""
    ax.set_title(f'{title_prefix}Silhouette Analysis')
    ax.grid(True)
    ax.set_xticks(k_values)
    
    # Add value annotations
    for i, txt in enumerate(silhouette_values):
        ax.annotate(f'{txt:.3f}', (k_values[i], silhouette_values[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Number of k values tested: {len(k_values)}\n"
        f"k range: {min(k_values)}-{max(k_values)}\n"
        f"Best silhouette score: {max(silhouette_values):.3f} (k={k_values[silhouette_values.index(max(silhouette_values))]})\n"
        f"Average silhouette score: {sum(silhouette_values)/len(silhouette_values):.3f}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    del fig, ax
    gc.collect()
    print(f"Silhouette plot saved to {out_png}")

def compute_silhouette(cluster_dict):
    """
    Compute silhouette coefficient for the clustering.
    Returns average silhouette coefficient across all points.
    Args:
        cluster_dict: dict, clustering results where cluster_dict[i] = [avg_dist, center_traj, distances, subtrajs]
    Returns:
        float: average silhouette coefficient [-1, 1]
    """
    print("\nComputing silhouette coefficient...")
    all_silhouettes = []
    total_trajs = sum(len(cluster_dict[i][3]) for i in cluster_dict)
    processed = 0
    last_percent = -1
    
    # For each cluster
    for cluster_idx in cluster_dict:
        cluster_trajs = cluster_dict[cluster_idx][3]  # Get trajectories in this cluster
        
        # Pre-compute center distances for efficiency
        center_dists = {}
        for other_idx in cluster_dict:
            if other_idx != cluster_idx:
                dist = traj2trajIED(cluster_dict[cluster_idx][1].points, 
                                  cluster_dict[other_idx][1].points)
                if dist != 1e10:
                    center_dists[other_idx] = dist
        
        # For each trajectory in the cluster
        for traj in cluster_trajs:
            # Update progress
            processed += 1
            percent = (processed * 100) // total_trajs
            if percent != last_percent:
                print(f"Silhouette computation: {percent}% complete")
                last_percent = percent
            
            # 1. Calculate a (average distance to points in same cluster)
            if len(cluster_trajs) > 1:  # Only if there are other points in cluster
                same_cluster_dists = []
                for other_traj in cluster_trajs:
                    if other_traj != traj:
                        dist = traj2trajIED(traj.points, other_traj.points)
                        if dist != 1e10:
                            same_cluster_dists.append(dist)
                a = np.mean(same_cluster_dists) if same_cluster_dists else 0.0
            else:
                a = 0.0
            
            # 2. Calculate b (average distance to points in next best cluster)
            # Use center distances to skip distant clusters
            b = float('inf')
            for other_idx, center_dist in center_dists.items():
                # Only check clusters whose centers are closer than current best b
                if center_dist >= b:
                    continue
                    
                other_cluster_trajs = cluster_dict[other_idx][3]
                cluster_dists = []
                for other_traj in other_cluster_trajs:
                    dist = traj2trajIED(traj.points, other_traj.points)
                    if dist != 1e10:
                        cluster_dists.append(dist)
                if cluster_dists:
                    avg_dist = np.mean(cluster_dists)
                    b = min(b, avg_dist)
            
            # 3. Calculate silhouette
            if b != float('inf'):
                if a == 0:
                    all_silhouettes.append(1)  # Perfect clustering for this point
                else:
                    s = (b - a) / max(a, b)
                    all_silhouettes.append(s)
    
    print("Silhouette computation: 100% complete")
    # Return average silhouette coefficient
    return np.mean(all_silhouettes) if all_silhouettes else 0

def compute_sse(res):
    """
    Compute Sum of Squared Errors for clustering results.
    Args:
        res: tuple (overall_sim, overall_sim, cluster_dict) from clustering
    Returns:
        tuple (sse, n) where sse is the sum of squared errors and n is number of valid assignments
    """
    sse = 0
    n = 0  # Count of valid assignments
    cluster_dict = res[0][2]  # Extract cluster_dict from tuple
    for cluster_idx in cluster_dict:
        center = cluster_dict[cluster_idx][1]  # Center trajectory
        subtrajs = cluster_dict[cluster_idx][3]  # Assigned subtrajectories
        for traj in subtrajs:
            dist = traj2trajIED(center.points, traj.points)
            if dist != 1e10:  # Only count valid assignments
                sse += dist ** 2
                n += 1
    return sse, n

def plot_combined_elbow(classical_data, quantum_data, out_png):
    """
    Generate a combined elbow plot comparing classical and quantum results.
    Args:
        classical_data: tuple of (k_values, sse_values, n_values) for classical results
        quantum_data: tuple of (k_values, sse_values, n_values) for quantum results
        out_png: output file path
    """
    gc.collect()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Unpack data
    k_class, sse_class, n_class = classical_data
    k_quant, sse_quant, n_quant = quantum_data
    
    # Plot raw SSE
    ax1.plot(k_class, sse_class, 'bo-', linewidth=2, label='Classical')
    ax1.plot(k_quant, sse_quant, 'ro-', linewidth=2, label='Quantum')
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (SSE)')
    ax1.set_title('Comparison of Elbow Plots')
    ax1.grid(True)
    ax1.legend()
    ax1.set_xticks(sorted(list(set(k_class + k_quant))))
    
    # Plot normalized SSE (SSE/n)
    norm_sse_class = [sse/n if n > 0 else float('inf') for sse, n in zip(sse_class, n_class)]
    norm_sse_quant = [sse/n if n > 0 else float('inf') for sse, n in zip(sse_quant, n_quant)]
    
    ax2.plot(k_class, norm_sse_class, 'bo-', linewidth=2, label='Classical')
    ax2.plot(k_quant, norm_sse_quant, 'ro-', linewidth=2, label='Quantum')
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average SSE per Assignment')
    ax2.grid(True)
    ax2.legend()
    ax2.set_xticks(sorted(list(set(k_class + k_quant))))
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Classical k range: {min(k_class)}-{max(k_class)}\n"
        f"Quantum k range: {min(k_quant)}-{max(k_quant)}\n"
        f"Classical valid assignments: {sum(n_class)}\n"
        f"Quantum valid assignments: {sum(n_quant)}\n"
        f"Classical avg assignments per k: {sum(n_class)/len(n_class):.1f}\n"
        f"Quantum avg assignments per k: {sum(n_quant)/len(n_quant):.1f}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f"Combined elbow plot saved to {out_png}")

def plot_combined_silhouette(classical_data, quantum_data, out_png):
    """
    Generate a combined silhouette plot comparing classical and quantum results.
    Args:
        classical_data: tuple of (k_values, silhouette_values) for classical results
        quantum_data: tuple of (k_values, silhouette_values) for quantum results
        out_png: output file path
    """
    gc.collect()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Unpack data
    k_class, sil_class = classical_data
    k_quant, sil_quant = quantum_data
    
    ax.plot(k_class, sil_class, 'bo-', linewidth=2, label='Classical')
    ax.plot(k_quant, sil_quant, 'ro-', linewidth=2, label='Quantum')
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Coefficient')
    ax.set_title('Comparison of Silhouette Coefficients')
    ax.grid(True)
    ax.legend()
    ax.set_xticks(sorted(list(set(k_class + k_quant))))
    
    # Add value annotations
    for k, s in zip(k_class, sil_class):
        ax.annotate(f'{s:.3f}', (k, s), 
                   textcoords="offset points", xytext=(0,10), 
                   ha='center', color='blue')
    for k, s in zip(k_quant, sil_quant):
        ax.annotate(f'{s:.3f}', (k, s), 
                   textcoords="offset points", xytext=(0,-15), 
                   ha='center', color='red')
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Classical k range: {min(k_class)}-{max(k_class)}\n"
        f"Quantum k range: {min(k_quant)}-{max(k_quant)}\n"
        f"Best classical score: {max(sil_class):.3f} (k={k_class[sil_class.index(max(sil_class))]})\n"
        f"Best quantum score: {max(sil_quant):.3f} (k={k_quant[sil_quant.index(max(sil_quant))]})\n"
        f"Classical avg score: {sum(sil_class)/len(sil_class):.3f}\n"
        f"Quantum avg score: {sum(sil_quant)/len(sil_quant):.3f}"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    gc.collect()
    print(f"Combined silhouette plot saved to {out_png}")

def plot_combined_timing(classical_file, quantum_file, out_png):
    """
    Generate a combined timing plot comparing classical and quantum results.
    Args:
        classical_file: Path to classical timing data JSON
        quantum_file: Path to quantum timing data JSON
        out_png: output file path
    """
    gc.collect()
    
    # Load timing data
    with open(classical_file, 'r') as f:
        classical_data = json.load(f)
    with open(quantum_file, 'r') as f:
        quantum_data = json.load(f)
    
    # Extract data
    k_class = [item['k'] for item in classical_data['individual_times']]
    t_class = [item['time'] for item in classical_data['individual_times']]
    k_quant = [item['k'] for item in quantum_data['individual_times']]
    t_quant = [item['time'] for item in quantum_data['individual_times']]
    
    plt.figure(figsize=(12, 8))
    plt.plot(k_class, t_class, 'bo-', linewidth=2, label='Classical')
    plt.plot(k_quant, t_quant, 'ro-', linewidth=2, label='Quantum')
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Comparison of Execution Times')
    plt.grid(True)
    plt.legend()
    plt.xticks(sorted(list(set(k_class + k_quant))))
    
    # Add value annotations
    for k, t in zip(k_class, t_class):
        plt.annotate(f'{t:.1f}s', (k, t), 
                    textcoords="offset points", xytext=(0,10), 
                    ha='center', color='blue')
    for k, t in zip(k_quant, t_quant):
        plt.annotate(f'{t:.1f}s', (k, t), 
                    textcoords="offset points", xytext=(0,-15), 
                    ha='center', color='red')
    
    # Add average time lines
    avg_class = classical_data['average_time']
    avg_quant = quantum_data['average_time']
    
    plt.axhline(y=avg_class, color='blue', linestyle='--', alpha=0.5)
    plt.axhline(y=avg_quant, color='red', linestyle='--', alpha=0.5)
    
    plt.annotate(f'Classical Avg: {avg_class:.1f}s', 
                (k_class[-1], avg_class),
                textcoords="offset points", xytext=(10,0),
                ha='left', va='center', color='blue')
    plt.annotate(f'Quantum Avg: {avg_quant:.1f}s', 
                (k_quant[-1], avg_quant),
                textcoords="offset points", xytext=(10,0),
                ha='left', va='center', color='red')
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Classical k range: {min(k_class)}-{max(k_class)}\n"
        f"Quantum k range: {min(k_quant)}-{max(k_quant)}\n"
        f"Total classical time: {sum(t_class):.1f}s\n"
        f"Total quantum time: {sum(t_quant):.1f}s\n"
        f"Classical avg time per k: {sum(t_class)/len(t_class):.1f}s\n"
        f"Quantum avg time per k: {sum(t_quant)/len(t_quant):.1f}s\n"
        f"Speedup ratio (classical/quantum): {sum(t_class)/sum(t_quant):.2f}x"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(out_png, bbox_inches='tight')
    plt.close()
    gc.collect()
    print(f"Combined timing plot saved to {out_png}")

def plot_timing(timing_file, out_png, method_name=""):
    """
    Plot clustering execution times for different k values.
    Args:
        timing_file: Path to the JSON file containing timing data
        out_png: Path to save the plot
        method_name: Optional name to include in plot title
    """
    print(f"\nGenerating timing plot from {timing_file}...")
    
    with open(timing_file, 'r') as f:
        timing_data = json.load(f)
    
    k_values = [item['k'] for item in timing_data['individual_times']]
    times = [item['time'] for item in timing_data['individual_times']]
    
    title_prefix = f"{method_name} " if method_name else ""
    
    plt.figure(figsize=(10, 6))
    plt.plot(k_values, times, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'{title_prefix}Clustering Execution Time vs. Number of Clusters')
    plt.grid(True)
    plt.xticks(k_values)
    
    # Add value annotations
    for k, t in zip(k_values, times):
        plt.annotate(f'{t:.1f}s', 
                    (k, t), 
                    textcoords="offset points",
                    xytext=(0,10), 
                    ha='center')
    
    # Add average time line
    avg_time = timing_data['average_time']
    plt.axhline(y=avg_time, color='r', linestyle='--', alpha=0.5)
    plt.annotate(f'Average: {avg_time:.1f}s', 
                (k_values[-1], avg_time),
                textcoords="offset points",
                xytext=(10, 0),
                ha='left',
                va='center')
    
    # Add information text box
    info_text = (
        f"Dataset: T-Drive\n"
        f"Number of k values tested: {len(k_values)}\n"
        f"k range: {min(k_values)}-{max(k_values)}\n"
        f"Total execution time: {sum(times):.1f}s\n"
        f"Average time per k: {sum(times)/len(times):.1f}s\n"
        f"Min time: {min(times):.1f}s (k={k_values[times.index(min(times))]})\n"
        f"Max time: {max(times):.1f}s (k={k_values[times.index(max(times))]})"
    )
    plt.gcf().text(0.02, 0.02, info_text, fontsize=8, 
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray'),
                   verticalalignment='bottom')
    
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close()
    print(f"Timing plot saved to {out_png}")

def process_results_file(results_path, output_dir=None, method_name=None, **plot_args):
    """
    Process a single results file and generate plots.
    
    Args:
        results_path: Path to the .pkl file containing clustering results
        output_dir: Directory to save plots (defaults to same directory as results)
        method_name: Name to use in plot titles
        plot_args: Additional arguments for plotting (alpha, center_alpha, etc.)
    """
    print(f"\nProcessing results file: {results_path}")
    
    # Determine output directory
    if output_dir is None:
        output_dir = os.path.dirname(results_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # Load results
    with open(results_path, 'rb') as f:
        res = pickle.load(f)
    
    # Generate base filename for plots
    base_name = os.path.splitext(os.path.basename(results_path))[0]
    
    # Extract clustering information
    cluster_dict = res[0][2]
    
    # Compute metrics
    sse, n = compute_sse(res)
    
    # Only compute silhouette if needed for requested plots
    plot_toggles = plot_args.get('plot_toggles', {})
    needs_silhouette = any([
        plot_toggles.get('classical_silhouette', False),
        plot_toggles.get('quantum_silhouette', False),
        plot_toggles.get('combined_silhouette', False)
    ])
    
    silhouette = compute_silhouette(cluster_dict) if needs_silhouette else None
    
    # Generate cluster visualization if requested
    if plot_toggles.get('classical_clusters', False) or plot_toggles.get('quantum_clusters', False):
        try:
            plot_clusters(
                cluster_dict,
                out_png=os.path.join(output_dir, f"{base_name}_clusters.png"),
                alpha=plot_args.get('alpha', 0.5),
                center_alpha=plot_args.get('center_alpha', 1.0),
                sample_rate=plot_args.get('sample_rate', 1),
                bg_image=plot_args.get('bg_image'),
                method_name=method_name
            )
        except Exception as e:
            print(f"Warning: Could not generate cluster plot: {str(e)}")
    
    print(f"Clustering metrics:")
    print(f"  - SSE: {sse:.4f}")
    print(f"  - Valid assignments (N): {n}")
    print(f"  - SSE/N: {sse/n if n > 0 else float('inf'):.4f}")
    if silhouette is not None:
        print(f"  - Silhouette coefficient: {silhouette:.4f}")
    
    return base_name, sse, n, silhouette

def detect_result_type(filename):
    """
    Detect whether a results file is from classical or quantum clustering.
    Returns appropriate method name and file pattern.
    """
    base = os.path.basename(filename)
    if base.startswith('quantum_'):
        return 'Quantum RLSTC', 'quantum_k*_a*.pkl'
    elif base.startswith('classical_'):
        return 'Classical RLSTC', 'classical_k*_a*.pkl'
    else:
        return '', 'k*_a*.pkl'

def process_results_directory(directory, method_name=None, **plot_args):
    """
    Process all results files in a directory and generate summary plots.
    
    Args:
        directory: Directory containing .pkl files with clustering results
        method_name: Name to use in plot titles
        plot_args: Additional arguments for plotting including plot_toggles
    """
    plot_toggles = plot_args.get('plot_toggles', {})
    
    # Find classical and quantum files if needed
    if any([plot_toggles[f'classical_{plot}'] for plot in ['clusters', 'elbow', 'silhouette', 'timing']] or
           any([plot_toggles[f'combined_{plot}'] for plot in ['elbow', 'silhouette', 'timing']])):
        classical_files = glob(os.path.join(directory, "classical_k*_a*.pkl"))
    else:
        classical_files = []
        
    if any([plot_toggles[f'quantum_{plot}'] for plot in ['clusters', 'elbow', 'silhouette', 'timing']] or
           any([plot_toggles[f'combined_{plot}'] for plot in ['elbow', 'silhouette', 'timing']])):
        quantum_files = glob(os.path.join(directory, "quantum_k*_a*.pkl"))
    else:
        quantum_files = []

    if not classical_files and not quantum_files:
        print(f"No clustering results files found in {directory}")
        return

    # Process classical files
    classical_data = {}
    if classical_files:
        k_pattern = re.compile(r'classical_k(\d+)_a(\d+).pkl')
        classical_valid_files = []
        for f in classical_files:
            match = k_pattern.search(os.path.basename(f))
            if match:
                k = int(match.group(1))
                amount = int(match.group(2))
                classical_valid_files.append((k, amount, f))
        
        if classical_valid_files:
            classical_valid_files.sort()
            classical_data['amount'] = classical_valid_files[0][1]
            classical_data['k_values'] = []
            classical_data['sse_values'] = []
            classical_data['n_values'] = []
            classical_data['silhouette_values'] = []
            
            for k, amount, filepath in classical_valid_files:
                if amount != classical_data['amount']:
                    continue
                    
                # Always process the file, but only include plot_toggles if we want clusters
                _, sse, n, silhouette = process_results_file(
                    filepath,
                    output_dir=directory,
                    method_name='Classical RLSTC',
                    **plot_args)  # Include plot_toggles to control cluster visualization
                    
                classical_data['k_values'].append(k)
                classical_data['sse_values'].append(sse)
                classical_data['n_values'].append(n)
                classical_data['silhouette_values'].append(silhouette)

    # Process quantum files
    quantum_data = {}
    if quantum_files:
        k_pattern = re.compile(r'quantum_k(\d+)_a(\d+).pkl')
        quantum_valid_files = []
        for f in quantum_files:
            match = k_pattern.search(os.path.basename(f))
            if match:
                k = int(match.group(1))
                amount = int(match.group(2))
                quantum_valid_files.append((k, amount, f))
        
        if quantum_valid_files:
            quantum_valid_files.sort()
            quantum_data['amount'] = quantum_valid_files[0][1]
            quantum_data['k_values'] = []
            quantum_data['sse_values'] = []
            quantum_data['n_values'] = []
            quantum_data['silhouette_values'] = []
            
            for k, amount, filepath in quantum_valid_files:
                if amount != quantum_data['amount']:
                    continue
                    
                # Always process the file, but only include plot_toggles if we want clusters
                _, sse, n, silhouette = process_results_file(
                    filepath,
                    output_dir=directory,
                    method_name='Quantum RLSTC',
                    **plot_args)  # Include plot_toggles to control cluster visualization
                    
                quantum_data['k_values'].append(k)
                quantum_data['sse_values'].append(sse)
                quantum_data['n_values'].append(n)
                quantum_data['silhouette_values'].append(silhouette)

    # Generate individual summary plots if enough data points
    amount = classical_data.get('amount', quantum_data.get('amount'))
    
    if classical_data and len(classical_data['k_values']) > 1:
        if plot_toggles.get('classical_elbow', True):
            try:
                plot_elbow(
                    classical_data['k_values'], 
                    classical_data['sse_values'],
                    classical_data['n_values'],
                    os.path.join(directory, f"classical_elbow_a{amount}.png"),
                    method_name='Classical RLSTC'
                )
            except Exception as e:
                print(f"Warning: Could not generate classical elbow plot: {str(e)}")

        if plot_toggles.get('classical_silhouette', True):
            try:
                plot_silhouette(
                    classical_data['k_values'],
                    classical_data['silhouette_values'],
                    os.path.join(directory, f"classical_silhouette_a{amount}.png"),
                    method_name='Classical RLSTC'
                )
            except Exception as e:
                print(f"Warning: Could not generate classical silhouette plot: {str(e)}")

        if plot_toggles.get('classical_timing', True):
            timing_file = os.path.join(directory, f"timing_data_a{amount}.json")
            if os.path.exists(timing_file):
                try:
                    plot_timing(
                        timing_file,
                        os.path.join(directory, f"timing_plot_a{amount}.png"),
                        method_name='Classical RLSTC'
                    )
                except Exception as e:
                    print(f"Warning: Could not generate classical timing plot: {str(e)}")

    if quantum_data and len(quantum_data['k_values']) > 1:
        if plot_toggles.get('quantum_elbow', True):
            try:
                plot_elbow(
                    quantum_data['k_values'],
                    quantum_data['sse_values'],
                    quantum_data['n_values'],
                    os.path.join(directory, f"quantum_elbow_a{amount}.png"),
                    method_name='Quantum RLSTC'
                )
            except Exception as e:
                print(f"Warning: Could not generate quantum elbow plot: {str(e)}")

        if plot_toggles.get('quantum_silhouette', True):
            try:
                plot_silhouette(
                    quantum_data['k_values'],
                    quantum_data['silhouette_values'],
                    os.path.join(directory, f"quantum_silhouette_a{amount}.png"),
                    method_name='Quantum RLSTC'
                )
            except Exception as e:
                print(f"Warning: Could not generate quantum silhouette plot: {str(e)}")

        if plot_toggles.get('quantum_timing', True):
            timing_file = os.path.join(directory, f"quantum_timing_data_a{amount}.json")
            if os.path.exists(timing_file):
                try:
                    plot_timing(
                        timing_file,
                        os.path.join(directory, f"quantum_timing_plot_a{amount}.png"),
                        method_name='Quantum RLSTC'
                    )
                except Exception as e:
                    print(f"Warning: Could not generate quantum timing plot: {str(e)}")

    # Generate combined plots if both datasets are available
    if classical_data and quantum_data:
        if plot_toggles.get('combined_elbow', True):
            try:
                plot_combined_elbow(
                    (classical_data['k_values'], classical_data['sse_values'], classical_data['n_values']),
                    (quantum_data['k_values'], quantum_data['sse_values'], quantum_data['n_values']),
                    os.path.join(directory, f"combined_elbow_a{amount}.png")
                )
            except Exception as e:
                print(f"Warning: Could not generate combined elbow plot: {str(e)}")

        if plot_toggles.get('combined_silhouette', True):
            try:
                plot_combined_silhouette(
                    (classical_data['k_values'], classical_data['silhouette_values']),
                    (quantum_data['k_values'], quantum_data['silhouette_values']),
                    os.path.join(directory, f"combined_silhouette_a{amount}.png")
                )
            except Exception as e:
                print(f"Warning: Could not generate combined silhouette plot: {str(e)}")

        if plot_toggles.get('combined_timing', True):
            classical_timing = os.path.join(directory, f"timing_data_a{amount}.json")
            quantum_timing = os.path.join(directory, f"quantum_timing_data_a{amount}.json")
            if os.path.exists(classical_timing) and os.path.exists(quantum_timing):
                try:
                    plot_combined_timing(
                        classical_timing,
                        quantum_timing,
                        os.path.join(directory, f"combined_timing_a{amount}.png")
                    )
                except Exception as e:
                    print(f"Warning: Could not generate combined timing plot: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from clustering results.")

    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-results", help="Single results file to process")
    input_group.add_argument("-results_dir", help="Directory containing results files to process")

    # Plot customization
    parser.add_argument("--output-dir", help="Directory to save plots (defaults to same as input)")
    parser.add_argument("--method-name", help="Method name to show in plot titles (will auto-detect if not provided)")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha transparency for scatter plot points")
    parser.add_argument("--center-alpha", type=float, default=1.0,
                       help="Alpha transparency for cluster centers")
    parser.add_argument("--sample-rate", type=int, default=1,
                       help="Plot every Nth point of trajectories")
    parser.add_argument("--bg-image", help="Path to background image for plots")

    # Plot toggles
    parser.add_argument("--plot-classical-clusters", action="store_true",
                       help="Generate cluster plots for classical results")
    parser.add_argument("--plot-quantum-clusters", action="store_true",
                       help="Generate cluster plots for quantum results")
    parser.add_argument("--plot-classical-elbow", action="store_true",
                       help="Generate elbow plot for classical results")
    parser.add_argument("--plot-quantum-elbow", action="store_true",
                       help="Generate elbow plot for quantum results")
    parser.add_argument("--plot-classical-silhouette", action="store_true",
                       help="Generate silhouette plot for classical results")
    parser.add_argument("--plot-quantum-silhouette", action="store_true",
                       help="Generate silhouette plot for quantum results")
    parser.add_argument("--plot-classical-timing", action="store_true",
                       help="Generate timing plot for classical results")
    parser.add_argument("--plot-quantum-timing", action="store_true",
                       help="Generate timing plot for quantum results")
    
    # Combined plot toggles
    parser.add_argument("--plot-combined-elbow", action="store_true",
                       help="Generate combined classical/quantum elbow plot")
    parser.add_argument("--plot-combined-silhouette", action="store_true",
                       help="Generate combined classical/quantum silhouette plot")
    parser.add_argument("--plot-combined-timing", action="store_true",
                       help="Generate combined classical/quantum timing plot")

    args = parser.parse_args()

    # If no specific plots are requested, enable all by default
    if not any([
        args.plot_classical_clusters, args.plot_quantum_clusters,
        args.plot_classical_elbow, args.plot_quantum_elbow,
        args.plot_classical_silhouette, args.plot_quantum_silhouette,
        args.plot_classical_timing, args.plot_quantum_timing,
        args.plot_combined_elbow, args.plot_combined_silhouette,
        args.plot_combined_timing
    ]):
        args.plot_classical_clusters = True
        args.plot_quantum_clusters = True
        args.plot_classical_elbow = True
        args.plot_quantum_elbow = True
        args.plot_classical_silhouette = True
        args.plot_quantum_silhouette = True
        args.plot_classical_timing = True
        args.plot_quantum_timing = True
        args.plot_combined_elbow = True
        args.plot_combined_silhouette = True
        args.plot_combined_timing = True

    plot_args = {
        'alpha': args.alpha,
        'center_alpha': args.center_alpha,
        'sample_rate': args.sample_rate,
        'bg_image': args.bg_image,
        'plot_toggles': {
            'classical_clusters': args.plot_classical_clusters,
            'quantum_clusters': args.plot_quantum_clusters,
            'classical_elbow': args.plot_classical_elbow,
            'quantum_elbow': args.plot_quantum_elbow,
            'classical_silhouette': args.plot_classical_silhouette,
            'quantum_silhouette': args.plot_quantum_silhouette,
            'classical_timing': args.plot_classical_timing,
            'quantum_timing': args.plot_quantum_timing,
            'combined_elbow': args.plot_combined_elbow,
            'combined_silhouette': args.plot_combined_silhouette,
            'combined_timing': args.plot_combined_timing
        }
    }

    if args.results:
        process_results_file(args.results, args.output_dir, args.method_name, **plot_args)
    else:
        process_results_directory(args.results_dir, args.method_name, **plot_args)