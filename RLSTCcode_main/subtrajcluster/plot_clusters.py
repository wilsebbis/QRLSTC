import pickle
import numpy as np
import matplotlib.pyplot as plt
import argparse

from traj import Traj
from point import Point

def plot_clusters(trajfile, cluster_labels_file, out_png=None):
    """
    Plot each trajectory point color-coded by its cluster label.
    Args:
        trajfile: path to pickled list of Traj objects
        cluster_labels_file: path to pickled list of cluster labels (same order as trajectories)
        out_png: if provided, save plot to this file
    """
    # Load trajectories and cluster labels
    trajs = pickle.load(open(trajfile, 'rb'))
    cluster_labels = pickle.load(open(cluster_labels_file, 'rb'))
    assert len(trajs) == len(cluster_labels), "Mismatch between trajectories and cluster labels"

    # Assign a color to each cluster
    unique_labels = np.unique(cluster_labels)
    colors = plt.cm.get_cmap('tab10', len(unique_labels))
    color_map = {label: colors(i) for i, label in enumerate(unique_labels)}

    plt.figure(figsize=(10, 8))
    for traj, label in zip(trajs, cluster_labels):
        xs = [p.x for p in traj.points]
        ys = [p.y for p in traj.points]
        plt.scatter(xs, ys, s=5, color=color_map[label], label=f"Cluster {label}" if label not in plt.gca().get_legend_handles_labels()[1] else "")

    # Optionally plot cluster centers if available
    # Example: cluster_centers = pickle.load(open('cluster_centers.pkl', 'rb'))
    # for i, center in enumerate(cluster_centers):
    #     plt.scatter(center.x, center.y, s=100, marker='*', color=color_map[i], edgecolor='black')

    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Trajectory Points by Cluster')
    plt.legend()
    if out_png:
        plt.savefig(out_png)
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot clustered trajectories")
    parser.add_argument("-trajfile", required=True, help="Pickle file with list of Traj objects")
    parser.add_argument("-labels", required=True, help="Pickle file with cluster labels (list of ints)")
    parser.add_argument("-out", default=None, help="Output PNG file (optional)")
    args = parser.parse_args()
    plot_clusters(args.trajfile, args.labels, args.out)
