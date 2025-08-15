# From https://github.com/llianga/RLSTCcode


"""
initcenters.py
--------------

Cluster initialization and base clustering for trajectory data using trajectory-to-trajectory distance.
Implements a k-means++-like initialization and assignment for sub-trajectory clustering.

Functions:
- initialize_centers: Diversity-maximizing initialization of cluster centers.
- getbaseclus: Assigns subtrajectories to clusters and computes statistics.
- saveclus: Runs clustering and saves results to file.

Usage:
    python initcenters.py -subtrajsfile <subtrajs> -trajsfile <trajs> -k <num_clusters> -amount <num_trajs> -centerfile <output>

Author: Wil Bishop, Date: August 14, 2025
"""

import pickle
import sys
import numpy as np
import random
from point import Point
from segment import Segment
from traj import Traj
from collections import defaultdict
from trajdistance import traj2trajIED
import argparse
import time

def initialize_centers(data, K):
    """
    Initialize cluster centers using a diversity-maximizing (k-means++) strategy.
    Args:
        data: list of Traj objects (trajectories)
        K: int, number of clusters
    Returns:
        List of K selected Traj objects as initial centers
    """
    centers = [random.choice(data)]
    while len(centers) < K:
        # For each trajectory, compute its minimum distance to any current center
        distances = [min([traj2trajIED(center.points, traj.points) for center in centers]) for traj in data]
        # Select the trajectory with the maximum minimum distance as the next center
        new_center = data[distances.index(max(distances))]
        centers.append(new_center)
    return centers

def getbaseclus(trajs, k, subtrajs):
    """
    Assign subtrajectories to clusters based on minimum trajectory-to-trajectory distance.
    Args:
        trajs: list of Traj objects (full trajectories)
        k: int, number of clusters
        subtrajs: list of Traj objects (subtrajectories)
    Returns:
        cluster_dict: dict mapping cluster index to [average distance, center, distances, assigned subtrajectories]
    """
    centers = initialize_centers(trajs, k)
    cluster_dict = defaultdict(list)  # Final output: cluster index -> [avg_dist, center, dists, subtrajs]
    cluster_segments = defaultdict(list)  # cluster index -> list of assigned subtrajectories
    dists_dict = defaultdict(list)  # cluster index -> list of distances for assigned subtrajectories
    # Assign each subtrajectory to the closest center
    for i in range(len(subtrajs)):
        mindist = float("inf")
        minidx = 0
        for j in range(k): 
            dist = traj2trajIED(centers[j].points, subtrajs[i].points)
            if dist == 1e10:
                continue  # Skip if no temporal overlap
            if dist < mindist:
                mindist = dist
                minidx = j
        if mindist == float("inf"):
            continue  # Could not assign this subtrajectory
        else:
            cluster_segments[minidx].append(subtrajs[i])
            dists_dict[minidx].append(mindist)
    # Ensure no empty clusters: if a cluster is empty, assign its center as a dummy member
    for i in range(k):
        if len(cluster_segments[i]) == 0:
            cluster_segments[minidx].append(centers[i])
            dists_dict[i].append(0)
    # Build the final cluster dictionary with statistics
    for i in cluster_segments.keys():
        center = centers[i]
        temp_dist = dists_dict[i]
        aver_dist = np.mean(temp_dist)
        cluster_dict[i].append(aver_dist)  # Average distance to center
        cluster_dict[i].append(center)     # Center trajectory
        cluster_dict[i].append(temp_dist)  # List of distances
        cluster_dict[i].append(cluster_segments[i])  # Assigned subtrajectories
    return cluster_dict

def saveclus(k, subtrajs, trajs, amount):
    """
    Run clustering and compute overall similarity, then package results for saving.
    Args:
        k: int, number of clusters
        subtrajs: list of Traj objects (subtrajectories)
        trajs: list of Traj objects (full trajectories)
        amount: int, number of trajectories to use
    Returns:
        res: list containing a tuple (overall_sim, overall_sim, cluster_dict)
    """
    trajs = trajs[:amount]
    cluster_dict = getbaseclus(trajs, k, subtrajs)
    count_sim, traj_num = 0, 0
    # Compute total similarity and number of assigned subtrajectories
    for i in cluster_dict.keys():
        count_sim += np.sum(cluster_dict[i][2])
        traj_num += len(cluster_dict[i][3])
    if traj_num == 0:
        overall_sim = 1e10
    else:
        overall_sim = count_sim / traj_num
    res = []
    res.append((overall_sim, overall_sim, cluster_dict))
    return res
    
    
if __name__ == "__main__":
    # Command-line interface for running clustering and saving results
    parser = argparse.ArgumentParser(description="Cluster subtrajectories using k-means++-like initialization.")
    parser.add_argument("-subtrajsfile", default='data/traclus_subtrajs', help="Pickle file containing subtrajectories")
    parser.add_argument("-trajsfile", default='data/Tdrive_norm_traj_RLSTC', help="Pickle file containing full trajectories")
    parser.add_argument("-k", type=int, default=10, help="Number of clusters")
    parser.add_argument("-amount", type=int, default=1000, help="Number of trajectories to use")
    parser.add_argument("-centerfile", default='data/tdrive_clustercenter_RLSTC', help="Output file for cluster centers")

    # Parse arguments
    args = parser.parse_args()
    # Load subtrajectories and trajectories from pickle files
    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))

    start_time = time.time()
    res = saveclus(args.k, subtrajs, trajs, args.amount)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Clustering completed in {elapsed_time:.2f} seconds.")

    pickle.dump(res, open(args.centerfile, 'wb'), protocol=2)

    def compute_sse(res):
        sse = 0
        cluster_dict = res[0][2]  # Extract cluster_dict from tuple
        for cluster_idx in cluster_dict:
            center = cluster_dict[cluster_idx][1]  # Center trajectory
            subtrajs = cluster_dict[cluster_idx][3]  # Assigned subtrajectories
            for traj in subtrajs:
                dist = traj2trajIED(center.points, traj.points)
                sse += dist ** 2
        return sse

    sse = compute_sse(res)
    print(f"Goodness of fit (SSE): {sse:.4f}")    