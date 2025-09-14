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
import os
import numpy as np
import random
from point import Point
from segment import Segment
from traj import Traj
from collections import defaultdict
from trajdistance import traj2trajIED
import argparse
import time
import json

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
    Prints percentage and timing progress as clustering proceeds.
    """
    print("Starting center initialization...")
    init_start = time.time()
    centers = initialize_centers(trajs, k)
    init_end = time.time()
    print(f"→ Center initialization done in {init_end - init_start:.2f} seconds.\n")

    cluster_dict = defaultdict(list)
    cluster_segments = defaultdict(list)
    dists_dict = defaultdict(list)

    total = len(subtrajs)
    step = max(1, total // 20)  # Every 5%
    last_time = time.time()

    for i, subtraj in enumerate(subtrajs):
        if i % step == 0 or i == total - 1:
            percent_done = int((i / total) * 100)
            now = time.time()
            delta = now - last_time
            print(f"Clustering {percent_done}% done... (step took {delta:.2f} seconds)")
            last_time = now

        mindist = float("inf")
        minidx = 0
        for j in range(k):
            dist = traj2trajIED(centers[j].points, subtraj.points)
            if dist == 1e10:
                continue
            if dist < mindist:
                mindist = dist
                minidx = j
        if mindist != float("inf"):
            cluster_segments[minidx].append(subtraj)
            dists_dict[minidx].append(mindist)

    # Ensure no empty clusters
    for i in range(k):
        if len(cluster_segments[i]) == 0:
            cluster_segments[i].append(centers[i])
            dists_dict[i].append(0)

    for i in cluster_segments:
        center = centers[i]
        temp_dist = dists_dict[i]
        aver_dist = np.mean(temp_dist)
        cluster_dict[i].append(aver_dist)
        cluster_dict[i].append(center)
        cluster_dict[i].append(temp_dist)
        cluster_dict[i].append(cluster_segments[i])

    print("Clustering 100% done.")
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
    parser.add_argument("-k_values", "-k", nargs='+', type=int, default=[10], help="List of k values to try")
    parser.add_argument("-amount", type=int, default=1000, help="Number of trajectories to use")
    parser.add_argument("-output_dir", default='out', help="Directory to save clustering results")

    # Parse arguments
    args = parser.parse_args()

    # Load subtrajectories and trajectories from pickle files
    print("Loading data files...")
    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("\nStarting clustering process for all k values...")
    overall_start_time = time.time()

    k_values = sorted(args.k_values)
    timing_data = {
        'k_values': k_values,
        'individual_times': [],
        'total_time': None,
        'average_time': None
    }

    for k in k_values:
        print(f"\nClustering with k={k}...")
        start_time = time.time()
        res = saveclus(k, subtrajs, trajs, args.amount)
        end_time = time.time()
        execution_time = end_time - start_time
        print(f"RLSTC Clustering for k={k} completed in {execution_time:.2f} seconds.")

        # Save results for this k value with automatic filename
        out_file = f"{args.output_dir}/classical_k{k}_a{args.amount}"
        pickle.dump(res, open(out_file + '.pkl', 'wb'), protocol=2)
        print(f"Results for k={k} saved to {out_file}.pkl")

        # Record timing data
        timing_data['individual_times'].append({
            'k': k,
            'time': execution_time
        })

    # Calculate and record overall timing information
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    timing_data['total_time'] = overall_elapsed_time
    timing_data['average_time'] = overall_elapsed_time/len(k_values)

    # Save timing data
    timing_file = f"{args.output_dir}/timing_data_a{args.amount}.json"
    with open(timing_file, 'w') as f:
        json.dump(timing_data, f, indent=2)

    print(f"\nEntire clustering process completed in {overall_elapsed_time:.2f} seconds")
    print(f"Average time per k value: {overall_elapsed_time/len(k_values):.2f} seconds")
    print(f"Timing data saved to {timing_file}")
    print(f"\nTo generate plots, run: python plot_utils.py -results_dir {args.output_dir}")

