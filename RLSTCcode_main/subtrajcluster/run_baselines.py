import pickle
import numpy as np
import time
from rl_splitmethod import kMeans_without_dist, sim_affinity, agglomerative_clusteing_without_dist
from initcenters import getbaseclus, saveclus
from traj import Traj

# Optional: If you have a TRACLUS implementation, import it here
# from traclus import traclus_cluster

def run_baselines(subtrajs, trajs, k):
    results = {}

    # RLSTC (initcenters.py)
    start = time.time()
    cluster_dict_rlstc = getbaseclus(trajs, k, subtrajs)
    sse_rlstc = sum([np.sum(cluster_dict_rlstc[i][2]) for i in cluster_dict_rlstc])
    end = time.time()
    results['RLSTC'] = {'SSE': sse_rlstc, 'time': end - start}

    # k-means baseline
    # Use RLSTC centers for fair comparison
    cluster_dict_init = {i: [None, cluster_dict_rlstc[i][1], None, []] for i in cluster_dict_rlstc}
    start = time.time()
    cluster_dict_kmeans = kMeans_without_dist(cluster_dict_init, subtrajs)
    sse_kmeans = sum([np.sum(cluster_dict_kmeans[i][2]) for i in cluster_dict_kmeans])
    end = time.time()
    results['k-means'] = {'SSE': sse_kmeans, 'time': end - start}

    # Agglomerative (TRACLUS-like) baseline
    start = time.time()
    cluster_segment_agg = agglomerative_clusteing_without_dist(subtrajs, k)
    # Compute SSE for agglomerative
    sse_agg = 0
    for i, subtrajs_in_cluster in cluster_segment_agg.items():
        # Use mean trajectory as center (approximate)
        xs = np.concatenate([[p.x for p in traj.points] for traj in subtrajs_in_cluster])
        ys = np.concatenate([[p.y for p in traj.points] for traj in subtrajs_in_cluster])
        center_x, center_y = np.mean(xs), np.mean(ys)
        for traj in subtrajs_in_cluster:
            sse_agg += np.sum((np.array([p.x for p in traj.points]) - center_x) ** 2 + (np.array([p.y for p in traj.points]) - center_y) ** 2)
    end = time.time()
    results['Agglomerative'] = {'SSE': sse_agg, 'time': end - start}

    return results

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run RLSTC, k-means, and Agglomerative baselines on subtrajectories.")
    parser.add_argument("-subtrajsfile", required=True, help="Pickle file containing subtrajectories")
    parser.add_argument("-trajsfile", required=True, help="Pickle file containing full trajectories")
    parser.add_argument("-k", type=int, required=True, help="Number of clusters")
    args = parser.parse_args()

    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))

    results = run_baselines(subtrajs, trajs, args.k)
    for method, stats in results.items():
        print(f"{method}: SSE={stats['SSE']:.4f}, Time={stats['time']:.2f}s")
