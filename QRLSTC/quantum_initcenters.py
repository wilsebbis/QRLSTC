"""
quantum_initcenters.py
----------------------

Quantum-inspired clustering for trajectory data using quantum swap test and amplitude encoding.

**Purpose and Workflow:**
This module clusters subtrajectories (short trajectory segments) extracted by the Traclus algorithm. Traclus is used to break long trajectories into meaningful segments (subtrajectories) that capture local movement patterns. Clustering these subtrajectories helps discover common movement motifs or representative patterns in the dataset.

**What do clusters and cluster centers represent?**
- Each cluster groups together subtrajectories that are similar in shape, timing, and location, as measured by quantum swap test similarity (inner product of quantum-encoded vectors).
- The cluster center is a trajectory (from the input set) chosen to be maximally diverse from other centers (quantum k-means++ style). It serves as a representative trajectory for its cluster, meaning it is the most central or typical trajectory for the subtrajectories assigned to that cluster.
- Cluster centers are actual encoded trajectories from the data, selected for their representativeness in quantum space.

**Why use subtrajectories from Traclus?**
- Long trajectories often contain multiple movement patterns. Traclus segments these into subtrajectories, allowing the clustering to focus on local patterns rather than entire trips.
- This approach enables the discovery of frequently occurring movement motifs, which can be more informative than clustering whole trajectories.

**How is quantum_initcenters different from initcenters?**
- `initcenters.py` uses classical trajectory-to-trajectory distance (IED) for clustering, while `quantum_initcenters.py` uses quantum swap test similarity between amplitude-encoded trajectories.
- Quantum encoding (via `StateEncoder` and `AmplitudeEncoder`) transforms trajectories into quantum states, allowing the use of quantum circuits to measure similarity.
- Cluster assignment and initialization are performed using quantum-inspired algorithms (QMeans, quantum swap test), leveraging Qiskit and quantum simulation backends.
- The quantum approach can capture more nuanced similarities and is designed to be compatible with future quantum hardware.

**Algorithm Steps:**
1. Subtrajectories are loaded from Traclus output (typically from `data/traclus_subtrajs`).
2. Full trajectories are loaded for use as potential cluster centers.
3. Trajectories are encoded for quantum amplitude encoding (`StateEncoder`, `AmplitudeEncoder`).
4. Cluster centers are initialized using a quantum diversity-maximizing strategy (quantum k-means++), selecting encoded trajectories from the data.
5. Each subtrajectory is assigned to the nearest cluster center using quantum swap test similarity (via quantum circuits).
6. Results are saved for further analysis or visualization.

**Functions:**
- quantum_initialize_centers: Diversity-maximizing initialization of cluster centers using quantum swap test.
- quantum_getbaseclus: Assigns subtrajectories to clusters using QMeans quantum distance.
- quantum_saveclus: Runs quantum clustering and saves results to file.

Usage:
    python quantum_initcenters.py -subtrajsfile <subtrajs> -trajsfile <trajs> -k <num_clusters> -amount <num_trajs> -centerfile <output>
"""

import pickle
import argparse
import numpy as np
from encoder import StateEncoder, AmplitudeEncoder
from qmeans import QMeans
import torch
import time
from trajdistance import traj2trajIED

def quantum_initialize_centers(encoded_data, k, backend=None, shots=1024):
    """
    Quantum-inspired k-means++ initialization using swap-test distances.
    Args:
        encoded_data: np.ndarray of encoded trajectories
        k: int, number of clusters
        backend: Qiskit backend
        shots: Number of shots per quantum circuit
    Returns:
        List of initial center vectors (np.ndarray)
    """
    from qmeans import distance_centroids_parallel
    centers = [encoded_data[np.random.choice(len(encoded_data))]]
    for _ in range(1, k):
        # Compute quantum swap-test distances to current centers
        distances = []
        for traj in encoded_data:
            min_dist = min([
                distance_centroids_parallel(traj, [center], backend=backend, shots=shots)[0]
                for center in centers
            ])
            distances.append(min_dist)
        new_center = encoded_data[np.argmax(distances)]
        centers.append(new_center)
    return np.array(centers)

def quantum_getbaseclus(encoded_data, k, backend=None, shots=1024):
    """
    Assign encoded trajectories to clusters using quantum k-means.
    Args:
        encoded_data: np.ndarray of encoded trajectories
        k: int, number of clusters
        backend: Qiskit backend
        shots: Number of shots per quantum circuit
    Returns:
        qmeans: Fitted QMeans object
    """
    # Quantum-inspired initialization
    centers_ini = quantum_initialize_centers(encoded_data, k, backend=backend, shots=shots)
    qmeans = QMeans(encoded_data, k, centroids_ini=centers_ini, backend=backend, shots=shots)
    qmeans.run()
    return qmeans

def quantum_saveclus(k, subtrajs, trajs, amount, backend=None, shots=1024):
    """
    Run quantum clustering and save results.
    Args:
        k: int, number of clusters
        subtrajs: list of trajectory tensors (subtrajectories)
        trajs: list of trajectory tensors (full trajectories)
        amount: int, number of trajectories to use
        backend: Qiskit backend
        shots: Number of shots per quantum circuit
    Returns:
        qmeans: Fitted QMeans object
    """
    # Encode trajectories
    torch_device = torch.device('cpu')
    num_points = trajs[0].size if len(trajs) > 0 else 10
    # Convert Traj objects to tensors for encoding
    traj_tensors = [torch.tensor([[p.t, p.x, p.y] for p in traj.points], dtype=torch.float32) for traj in trajs[:amount]]
    encoder = StateEncoder(num_points=num_points, torch_device=torch_device)
    encoded_data = encoder(traj_tensors).cpu().numpy()
    qmeans = quantum_getbaseclus(encoded_data, k, backend=backend, shots=shots)
    return qmeans

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantum-inspired cluster subtrajectories using QMeans.")
    parser.add_argument("-subtrajsfile", default='data/traclus_subtrajs', help="Pickle file containing subtrajectories")
    parser.add_argument("-trajsfile", default='data/Tdrive_norm_traj_QRLSTC', help="Pickle file containing full trajectories")
    parser.add_argument("-k", type=int, default=10, help="Number of clusters")
    parser.add_argument("-amount", type=int, default=1000, help="Number of trajectories to use")
    parser.add_argument("-centerfile", default='data/tdrive_clustercenter_QRLSTC', help="Output file for cluster centers")
    parser.add_argument("-shots", type=int, default=512, help="Number of quantum shots")
    # Parse arguments
    args = parser.parse_args()
    # Load subtrajectories and trajectories from pickle files
    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))
    # Run quantum clustering and save results
    start_time = time.time()
    qmeans = quantum_saveclus(args.k, subtrajs, trajs, args.amount, shots=args.shots)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Clustering completed in {elapsed_time:.2f} seconds.")

    pickle.dump(qmeans, open(args.centerfile, 'wb'), protocol=2)

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

    sse = compute_sse(qmeans)
    print(f"Goodness of fit (SSE): {sse:.4f}")