"""
plot_utils.py
------------

Utility functions for plotting clustering results, including:
- Cluster visualization
- Elbow plots
- Silhouette analysis plots
- Execution time analysis

Can be run as a standalone script to visualize clustering results from files:
    python plot_utils.py -results results_file.pkl [options]
    python plot_utils.py -results_dir directory_with_pkl_files [options]

These functions are generic and can work with any clustering results that follow
the expected format of (overall_sim, overall_sim, cluster_dict) tuples.
"""

import numpy as np
import matplotlib.pyplot as plt
from trajdistance import traj2trajIED
import gc
import pickle
import argparse
import os
import json
from glob import glob
import re

def plot_clusters(cluster_dict, out_png, alpha=0.5, center_alpha=1.0, sample_rate=1, bg_image=None):
    """
    Plot each subtrajectory point color-coded by cluster assignment, and cluster centers.
    Args:
        cluster_dict: dict, output from clustering where cluster_dict[i] = [avg_dist, center_traj, distances, subtrajs]
        out_png: save plot to file
        alpha: float, transparency for scatter plot points
        center_alpha: float, transparency for cluster center trajectories
        sample_rate: int, plot every Nth point of trajectories
        bg_image: str, path to background image file
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
    
    # Pre-process data to minimize memory usage during plotting
    for i, cluster in cluster_dict.items():
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
                        label=f"Cluster {i}" if j == 0 else "", 
                        alpha=alpha)
            del xs, ys
            gc.collect()
        
        # Plot cluster center
        center_traj = cluster[1]
        # Sample center points according to sample_rate
        center_points = center_traj.points[::sample_rate]
        xs = [p.x for p in center_points]
        ys = [p.y for p in center_points]
        ax.plot(xs, ys, color=colors(i), linewidth=2, marker='*', 
                markersize=10, label=f"Center {i}", alpha=center_alpha)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Subtrajectory Clusters')
    ax.legend()
    
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
    silhouette = compute_silhouette(cluster_dict)
    
    # Generate cluster visualization
    try:
        plot_clusters(
            cluster_dict,
            out_png=os.path.join(output_dir, f"{base_name}_clusters.png"),
            alpha=plot_args.get('alpha', 0.5),
            center_alpha=plot_args.get('center_alpha', 1.0),
            sample_rate=plot_args.get('sample_rate', 1),
            bg_image=plot_args.get('bg_image')
        )
    except Exception as e:
        print(f"Warning: Could not generate cluster plot: {str(e)}")
    
    print(f"Clustering metrics:")
    print(f"  - SSE: {sse:.4f}")
    print(f"  - Valid assignments (N): {n}")
    print(f"  - SSE/N: {sse/n if n > 0 else float('inf'):.4f}")
    print(f"  - Silhouette coefficient: {silhouette:.4f}")
    
    return base_name, sse, n, silhouette

def process_results_directory(directory, method_name=None, **plot_args):
    """
    Process all results files in a directory and generate summary plots.
    
    Args:
        directory: Directory containing .pkl files with clustering results
        method_name: Name to use in plot titles
        plot_args: Additional arguments for plotting
    """
    # Find all pickle files with pattern classical_k*_a*.pkl
    results_files = glob(os.path.join(directory, "classical_k*_a*.pkl"))
    if not results_files:
        print(f"No clustering results files found in {directory}")
        return
    
    # Extract k values and sort files
    k_pattern = re.compile(r'classical_k(\d+)_a(\d+)\.pkl')
    valid_files = []
    for f in results_files:
        match = k_pattern.search(os.path.basename(f))
        if match:
            k = int(match.group(1))
            amount = int(match.group(2))
            valid_files.append((k, amount, f))
    
    if not valid_files:
        print("No valid clustering results files found")
        return
    
    # Sort by k value and amount
    valid_files.sort()
    k_values = []
    sse_values = []
    n_values = []
    silhouette_values = []
    
    # Process each file
    amount = valid_files[0][1]  # Use amount from first file
    for k, file_amount, filepath in valid_files:
        if file_amount != amount:
            print(f"Warning: Inconsistent amount {file_amount} in {filepath}, expected {amount}")
            continue
            
        _, sse, n, silhouette = process_results_file(
            filepath,
            output_dir=directory,
            method_name=method_name,
            **plot_args
        )
        k_values.append(k)
        sse_values.append(sse)
        n_values.append(n)
        silhouette_values.append(silhouette)
    
    # Generate summary plots
    if len(k_values) > 1:
        try:
            plot_elbow(
                k_values, sse_values, n_values,
                os.path.join(directory, f"classical_elbow_a{amount}.png"),
                method_name=method_name
            )
        except Exception as e:
            print(f"Warning: Could not generate elbow plot: {str(e)}")
            
        try:
            plot_silhouette(
                k_values, silhouette_values,
                os.path.join(directory, f"classical_silhouette_a{amount}.png"),
                method_name=method_name
            )
        except Exception as e:
            print(f"Warning: Could not generate silhouette plot: {str(e)}")
            
        # Generate timing plot if timing data exists
        timing_file = os.path.join(directory, f"timing_data_a{amount}.json")
        if os.path.exists(timing_file):
            try:
                plot_timing(
                    timing_file,
                    os.path.join(directory, f"timing_plot_a{amount}.png"),
                    method_name=method_name
                )
            except Exception as e:
                print(f"Warning: Could not generate timing plot: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate plots from clustering results.")
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("-results", help="Single results file to process")
    input_group.add_argument("-results_dir", help="Directory containing results files to process")
    
    # Plot customization
    parser.add_argument("--output-dir", help="Directory to save plots (defaults to same as input)")
    parser.add_argument("--method-name", default="Classical RLSTC", 
                       help="Method name to show in plot titles")
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha transparency for scatter plot points")
    parser.add_argument("--center-alpha", type=float, default=1.0,
                       help="Alpha transparency for cluster centers")
    parser.add_argument("--sample-rate", type=int, default=1,
                       help="Plot every Nth point of trajectories")
    parser.add_argument("--bg-image", help="Path to background image for plots")
    
    args = parser.parse_args()
    
    plot_args = {
        'alpha': args.alpha,
        'center_alpha': args.center_alpha,
        'sample_rate': args.sample_rate,
        'bg_image': args.bg_image
    }
    
    if args.results:
        process_results_file(
            args.results,
            output_dir=args.output_dir,
            method_name=args.method_name,
            **plot_args
        )
    else:
        process_results_directory(
            args.results_dir,
            method_name=args.method_name,
            **plot_args
        )