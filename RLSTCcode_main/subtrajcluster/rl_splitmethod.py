# From https://github.com/llianga/RLSTCcode


"""
rl_splitmethod.py

Implements trajectory clustering methods (Agglomerative, DBSCAN, kMeans) for sub-trajectories.
Provides utilities for distance matrix calculation, cluster center computation, and cluster statistics.
Main entry point allows clustering of sub-trajectories from files using specified method.
"""

import sys
import pickle
import numpy as np
from segment import Segment, compare
from point import Point
from point_xy import Point_xy, _point2line_distance
from traj import Traj
from sklearn.cluster import DBSCAN
from sklearn.cluster import AgglomerativeClustering
from collections import defaultdict
from time import time
from trajdistance import traj2trajIED, makemid
import argparse



def agglomerative_clusteing_with_dist(distance_matrix, split_traj, cluster_num):
    """
    Perform agglomerative clustering using a precomputed distance matrix.
    Args:
        distance_matrix: np.ndarray, pairwise distances between trajectories
        split_traj: list, sub-trajectories to cluster
        cluster_num: int, number of clusters
    Returns:
        cluster_segment: dict, mapping cluster label to list of sub-trajectories
    """
    cluster_segment = defaultdict(list)
    cluster = AgglomerativeClustering(n_clusters=cluster_num, affinity='precomputed', linkage='average').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment


def agglomerative_clusteing_without_dist(split_traj, cluster_num):
    """
    Perform agglomerative clustering by first computing the distance matrix.
    Args:
        split_traj: list, sub-trajectories to cluster
        cluster_num: int, number of clusters
    Returns:
        cluster_segment: dict, mapping cluster label to list of sub-trajectories
    """
    cluster_segment = defaultdict(list)
    distance_matrix = sim_affinity(split_traj)
    cluster = AgglomerativeClustering(n_clusters=cluster_num, affinity='precomputed', linkage='average').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment


def dbscan_with_dist(distance_matrix, split_traj, ep, sample):
    """
    Perform DBSCAN clustering using a precomputed distance matrix.
    Args:
        distance_matrix: np.ndarray, pairwise distances between trajectories
        split_traj: list, sub-trajectories to cluster
        ep: float, DBSCAN epsilon parameter
        sample: int, DBSCAN min_samples parameter
    Returns:
        cluster_segment: dict, mapping cluster label to list of sub-trajectories
    """
    cluster_segment = defaultdict(list)
    remove_cluster = dict()
    count = 0
    for i in range(len(distance_matrix[0])):
        count += distance_matrix[0][i]
    ep = float(ep)
    sample = float(sample)
    cluster = DBSCAN(eps=ep, min_samples=sample, metric='precomputed').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment


def dbscan_without_dist(split_traj, ep, sample):
    """
    Perform DBSCAN clustering by first computing the distance matrix.
    Args:
        split_traj: list, sub-trajectories to cluster
        ep: float, DBSCAN epsilon parameter
        sample: int, DBSCAN min_samples parameter
    Returns:
        cluster_segment: dict, mapping cluster label to list of sub-trajectories
    """
    cluster_segment = defaultdict(list)
    remove_cluster = dict()
    startcal = time()
    distance_matrix = sim_affinity(split_traj)
    endcal = time()
    # Distance matrix calculation time can be printed for profiling
    # print('cal dist matrix time: ', endcal - startcal, 'seconds')
    ep = float(ep)
    sample = float(sample)
    cluster = DBSCAN(eps=ep, min_samples=sample, metric='precomputed').fit(distance_matrix)
    cluster_lables = cluster.labels_
    for i in range(len(cluster_lables)):
        cluster_segment[cluster_lables[i]].append(split_traj[i])
    return cluster_segment


def kMeans_without_dist(cluster_dict, split_traj):
    """
    Perform k-means-like clustering using trajectory distances.
    Args:
        cluster_dict: dict, initial cluster centers and statistics
        split_traj: list, sub-trajectories to cluster
    Returns:
        cluster_dict: dict, updated cluster centers and statistics
    """
    trajsize = len(split_traj)
    clusterAssment = np.mat(np.zeros((trajsize, 2)))
    clusterChanged = True
    count = 0
    while clusterChanged:
        cluster_segment = defaultdict(list)
        count += 1
        clusterChanged = False
        for i in range(trajsize):
            minDist = float("inf")
            minIndex = -1
            # Find closest cluster center for each trajectory
            for j in cluster_dict.keys():
                represent_traj = cluster_dict[j][1]
                dist = traj2trajIED(represent_traj.points, split_traj[i].points)
                if dist < minDist:
                    minDist = dist
                    minIndex = j
            cluster_segment[minIndex].append(split_traj[i])
            if clusterAssment[i, 0] != minIndex:
                clusterChanged = True
            clusterAssment[i, :] = minIndex, minDist ** 2
        # Update cluster centers/statistics
        cluster_dict = compute_statistic(cluster_segment)
        if count == 30:
            break
    return cluster_dict


def sim_affinity(split_traj):
    """
    Compute pairwise trajectory distance matrix for clustering.
    Args:
        split_traj: list, sub-trajectories
    Returns:
        dist_matrix: np.ndarray, pairwise distances
    """
    length = len(split_traj)
    dist_matrix = np.zeros(shape=(length, length), dtype='float32')
    for i in range(length):
        for j in range(i + 1, length):
            temp_dist = traj2trajIED(split_traj[i].points, split_traj[j].points)
            dist_matrix[i][j] = temp_dist
            dist_matrix[j][i] = dist_matrix[i][j]
    return dist_matrix


def compute_center(cluster_segments, threshold, min_dist):
    """
    Compute the center trajectory for a cluster of segments.
    Args:
        cluster_segments: list, segments in the cluster
        threshold: int, minimum number of segments intersecting a time point
        min_dist: float, minimum distance between consecutive center points
    Returns:
        center: list of Point objects representing the center trajectory
    """
    segment_points, timesets, center = [], [], []
    for i in range(len(cluster_segments)):
        segment_points.append(cluster_segments[i].points)
        timesets.append(cluster_segments[i].ts)
        timesets.append(cluster_segments[i].te)
    timesets = sorted(timesets)
    for t in timesets:
        intersect, sum_x, sum_y = 0, 0, 0
        for i in range(len(segment_points)):
            # Check if segment covers time t
            if segment_points[i][0].t > t or segment_points[i][-1].t < t:
                continue
            else:
                intersect += 1
                s_i = 0
                while segment_points[i][s_i].t < t:
                    s_i += 1
                # Interpolate if t is between two points
                if s_i != 0 and segment_points[i][s_i].t != t:
                    x = makemid(segment_points[i][s_i - 1].x, segment_points[i][s_i - 1].t, segment_points[i][s_i].x, segment_points[i][s_i].t, t)
                    y = makemid(segment_points[i][s_i - 1].y, segment_points[i][s_i - 1].t, segment_points[i][s_i].y, segment_points[i][s_i].t, t)
                    sum_x += x
                    sum_y += y
                # Use exact point if available
                if s_i == 0 or segment_points[i][s_i].t == t:
                    sum_x += segment_points[i][s_i].x
                    sum_y += segment_points[i][s_i].y
        if intersect >= threshold:
            new_x, new_y = sum_x / intersect, sum_y / intersect
            newpoint = Point(new_x, new_y, t)
            _size = len(center) - 1
            # Only add if sufficiently far from previous center point
            if _size < 0 or (_size >= 0 and newpoint.distance(center[_size]) > min_dist):
                center.append(newpoint)
    return center


def compute_statistic(cluster_segment, min_lines=20, min_dist=0.005):
    """
    Compute statistics for each cluster: average distance, center trajectory, etc.
    Args:
        cluster_segment: dict, mapping cluster label to list of segments
        min_lines: int, minimum number of lines for center computation
        min_dist: float, minimum distance between center points
    Returns:
        cluster_dict: dict, mapping cluster label to statistics and trajectories
    """
    cluster_dict = defaultdict(list)
    representive_point = defaultdict(list)
    for i in cluster_segment.keys():
        temp_subtrajs = []
        temp = []
        temp_dists = []
        center = compute_center(cluster_segment[i], min_lines, min_dist)
        if len(center) == 0:
            centertraj = cluster_segment[i][0]
            center = cluster_segment[i][0].points
        # Create center trajectory object
        centertraj = Traj(center, len(center), center[0].t, center[-1].t)
        # Compute distance between each subtraj and center
        cluster_size = len(cluster_segment.get(i))
        for j in range(cluster_size):
            dist = traj2trajIED(cluster_segment[i][j].points, center)
            if dist != 1e10:
                temp_dists.append(dist)
                temp_subtrajs.append(cluster_segment[i][j])
            temp.append(dist)
        aver_dist = np.mean(temp)
        cluster_dict[i].append(aver_dist)      # Average distance to center
        cluster_dict[i].append(centertraj)     # Center trajectory
        cluster_dict[i].append(temp)           # All distances
        cluster_dict[i].append(cluster_segment[i]) # All segments in cluster
        cluster_dict[i].append(temp_dists)     # Valid distances
        cluster_dict[i].append(temp_subtrajs)  # Valid subtrajectories
    return cluster_dict


def init_cluster(split_traj, cluster_dict_ori, clustermethod, ep, sample):
    """
    Initialize clusters using specified clustering method and compute statistics.
    Args:
        split_traj: list, sub-trajectories to cluster
        cluster_dict_ori: dict, initial cluster centers/statistics (for kmeans)
        clustermethod: str, clustering method ('AHC', 'dbscan', 'kmeans')
        ep: float, DBSCAN epsilon parameter
        sample: int, DBSCAN min_samples parameter
    Returns:
        cluster_dict: dict, cluster statistics
        overall_sim: float, average similarity for all trajectories
        traj_num: int, number of trajectories
        over_sim: float, average similarity for valid subtrajectories
        less_traj: int, number of valid subtrajectories
    """
    traj_num, count_sim, less_sim, less_traj = 0, 0, 0, 0
    if clustermethod == 'AHC':
        cluster_segment = agglomerative_clusteing_without_dist(split_traj, cluster_num=10)
        cluster_dict = compute_statistic(cluster_segment)
    if clustermethod == 'dbscan':
        cluster_segment = dbscan_without_dist(split_traj, ep, sample)
        cluster_dict = compute_statistic(cluster_segment)
    if clustermethod == 'kmeans':
        cluster_dict = kMeans_without_dist(cluster_dict_ori, split_traj)

    # Aggregate statistics for all clusters
    for i in cluster_dict.keys():
        count_sim += np.sum(cluster_dict[i][2])
        traj_num += len(cluster_dict[i][3])
        less_sim += np.sum(cluster_dict[i][4])
        less_traj += len(cluster_dict[i][5])
    if traj_num == 0:
        overall_sim = 1e10
    else:
        overall_sim = count_sim / traj_num
    if less_traj == 0:
        over_sim = 1e10
    else:
        over_sim = less_sim / less_traj
    return cluster_dict, overall_sim, traj_num, over_sim, less_traj


if __name__ == "__main__":
    """
    Main entry point for trajectory clustering script.
    Loads sub-trajectories and cluster centers from files, performs clustering, and prints results.
    """
    res = []
    parser = argparse.ArgumentParser(description="splitmethod")
    parser.add_argument("-splittrajfile", default='../data/ied_subtrajs_100', help="subtraj file")
    parser.add_argument("-clustermethod", default='dbscan', help="subtraj file")
    parser.add_argument("-baseclusterfile", default='../data/tdrive_clustercenter', help="subtraj file")
    parser.add_argument("-ep", type=float, default=0.005, help="ep")
    parser.add_argument("-sample", type=int, default=70, help="sample")

    args = parser.parse_args()
    # Load cluster centers and sub-trajectories from files
    centers = pickle.load(open(args.baseclusterfile, 'rb'))
    cluster_dict_ori = centers[0][2]
    split_traj = pickle.load(open(args.splittrajfile, 'rb'))
    # Perform clustering and compute statistics
    cluster_dict, overall_sim, traj_num, over_sim, less_traj = init_cluster(split_traj, cluster_dict_ori, args.clustermethod, args.ep, args.sample)

    res.append((overall_sim, over_sim, cluster_dict))
    print('-----over_sim-----', over_sim)