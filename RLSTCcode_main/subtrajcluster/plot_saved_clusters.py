import pickle
import matplotlib.pyplot as plt
import argparse

def plot_clusters(cluster_dict, out_png):
    """
    Plot each subtrajectory point color-coded by cluster assignment, and cluster centers.
    Args:
        cluster_dict: dict, output from getbaseclus (or res[0][2])
        out_png: save plot to file
    """
    colors = plt.cm.get_cmap('tab10', len(cluster_dict))
    plt.figure(figsize=(10, 8))
    for i, cluster in cluster_dict.items():
        subtrajs = cluster[3]
        for traj in subtrajs:
            xs = [p.x for p in traj.points]
            ys = [p.y for p in traj.points]
            plt.scatter(xs, ys, s=5, color=colors(i), label=f"Cluster {i}" if i not in plt.gca().get_legend_handles_labels()[1] else "")
        # Plot cluster center trajectory (optional, as a star)
        center_traj = cluster[1]
        xs = [p.x for p in center_traj.points]
        ys = [p.y for p in center_traj.points]
        plt.plot(xs, ys, color=colors(i), linewidth=2, marker='*', markersize=10, label=f"Center {i}")
    plt.xlabel('Longitude')
    plt.ylabel('Latitude')
    plt.title('Subtrajectory Clusters')
    plt.legend()
    plt.savefig(out_png)
    print(f"Cluster plot saved to {out_png}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Plot clusters from saved cluster info file.")
    parser.add_argument("-clusterfile", required=True, help="Pickle file containing cluster info (output from initcenters.py)")
    parser.add_argument("-out", required=True, help="Output PNG file for plot")
    args = parser.parse_args()
    res = pickle.load(open(args.clusterfile, 'rb'))
    cluster_dict = res[0][2]
    plot_clusters(cluster_dict, args.out)
