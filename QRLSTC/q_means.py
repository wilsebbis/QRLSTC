# q-means.py
# -------------
# Quantum-inspired k-means clustering for trajectory data using quantum swap test and amplitude encoding.
# Adapted from code taken from q-means/clean_code/q-means.py at https://github.com/Morcu/q-means
# to work for trajectory data.

"""
Example: Amplitude Encoding of a Trajectory
-------------------------------------------
Suppose a trajectory is represented by a vector v = [0.6, 0.8].
Amplitude encoding normalizes this vector so that the sum of squares is 1:
    |v⟩ = 0.6|0⟩ + 0.8|1⟩
This state can be loaded into a quantum register using the initialize operation.

Mathematical Representation of Hadamard Gate
--------------------------------------------
The Hadamard gate H acts on a single qubit as:
    H = (1/√2) * [[1, 1], [1, -1]]
Applied to |0⟩: H|0⟩ = (|0⟩ + |1⟩)/√2
Applied to |1⟩: H|1⟩ = (|0⟩ - |1⟩)/√2

Example: Swap Test Circuit for Inner Product Estimation
------------------------------------------------------
Given two normalized vectors |u⟩ and |v⟩, the swap test estimates their inner product:

    Initial state: |0⟩_ancilla ⊗ |u⟩ ⊗ |v⟩
    1. Apply Hadamard to ancilla:
        |ψ₁⟩ = (|0⟩ + |1⟩)/√2 ⊗ |u⟩ ⊗ |v⟩
    2. Apply controlled SWAP gates between |u⟩ and |v⟩, controlled by ancilla.
    3. Apply Hadamard to ancilla again.
    4. Measure ancilla.

Concrete Example:
-----------------
Let |u⟩ = 0.6|0⟩ + 0.8|1⟩, |v⟩ = 0.8|0⟩ + 0.6|1⟩
Inner product: ⟨u|v⟩ = 0.6*0.8 + 0.8*0.6 = 0.96

After swap test, the probability of measuring ancilla in |0⟩ is:
    p₀ = (1 + |⟨u|v⟩|²)/2 = (1 + 0.96²)/2 = (1 + 0.9216)/2 = 0.9608
The estimated inner product is:
    |⟨u|v⟩| = sqrt(2*p₀ - 1) = sqrt(2*0.9608 - 1) ≈ 0.96

This technique is used in distance_centroids_parallel to compute quantum similarity between trajectories.
"""

#!/usr/bin/env python3
# q-means.py (Qiskit Aer + transpile)

import time
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from q_distance import distance_centroids_parallel

class QMeans:
    def __init__(self, trainig_input, k, centroids_ini=None, threshold=0.04, seed=0, backend=None, shots=1024):
        """
        Args:
            trainig_input: ndarray of shape (n_samples, 2)
            k: number of clusters
            centroids_ini: optional initial centers, shape (k, 2)
            threshold: stopping tolerance on center movement (L2)
            seed: RNG seed for reproducibility
            backend: optional Qiskit backend (defaults to AerSimulator)
            shots: SWAP-test shots
        """
        np.random.seed(seed)
        self.data = np.asarray(trainig_input, dtype=float)
        self.k = int(k)
        self.n = self.data.shape[0]
        self.threshold = float(threshold)
        self.backend = backend if backend is not None else AerSimulator()
        self.shots = int(shots)

        if centroids_ini is None:
            self.centers = np.random.normal(scale=0.6, size=(self.k, 2))
        else:
            self.centers = np.asarray(centroids_ini, dtype=float).copy()

        self.clusters = np.zeros(self.n, dtype=int)

    def run(self, verbose: bool = True):
        """Training loop."""
        centers_old = np.zeros_like(self.centers)
        centers_new = deepcopy(self.centers)

        error = np.inf
        it = 0
        while error > self.threshold:
            it += 1
            # Distances via quantum SWAP-test (returns raw '1' counts)
            # Convert to probabilities by dividing by shots, then argmin.
            dcounts = np.array(
                [distance_centroids_parallel(x, centers_new, backend=self.backend, shots=self.shots) for x in self.data],
                dtype=float,
            )
            distances = dcounts / float(self.shots)

            # Assign to closest center
            self.clusters = np.argmin(distances, axis=1)

            centers_old[:] = centers_new
            # Recompute centers as means; if a cluster is empty, re-sample
            for i in range(self.k):
                mask = (self.clusters == i)
                if np.any(mask):
                    centers_new[i] = self.data[mask].mean(axis=0)
                else:
                    centers_new[i] = np.random.normal(scale=0.6, size=centers_new[i].shape)

            error = np.linalg.norm(centers_new - centers_old)
            if verbose:
                print(f"[iter {it}] center shift L2 = {error:.6f}")

        self.centers = centers_new

    def plot(self):
        """Scatter plot of clustered data and final centers."""
        colors = ['green', 'blue', 'black', 'red', 'orange', 'purple', 'cyan', 'magenta']
        for i in range(self.n):
            plt.scatter(self.data[i, 0], self.data[i, 1], s=7, color=colors[self.clusters[i] % len(colors)])
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*', s=150)
        plt.title("Q-means clustering")
        plt.show()

    def fit(self, point):
        """Return cluster index for a single 2D point."""
        counts = distance_centroids_parallel(point, self.centers, backend=self.backend, shots=self.shots)
        dists = np.array(counts, dtype=float) / float(self.shots)
        return int(np.argmin(dists))


if __name__ == "__main__":
    # Example: toy dataset loader (must return (data, labels))
    data, category = load_dataset("toy")

    k = int(np.max(category) + 1)
    qmeans = QMeans(data, k, threshold=0.02, seed=0, shots=2048)

    t0 = time.time()
    qmeans.run(verbose=True)
    t1 = time.time()
    print(f"Training time: {t1 - t0:.2f}s")

    qmeans.plot()