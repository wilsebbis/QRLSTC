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
import platform
from copy import deepcopy

import numpy as np
import matplotlib.pyplot as plt

from qiskit_aer import AerSimulator
from q_distance import distance_centroids_parallel

# Check platform capabilities
IS_APPLE_SILICON = (
    platform.system() == "Darwin" and 
    platform.machine() == "arm64"
)

HAS_CUDA = False
try:
    import torch
    HAS_CUDA = torch.cuda.is_available()
except ImportError:
    pass

# Set optimal backend based on hardware
if IS_APPLE_SILICON:
    try:
        import mlx.core as mx
        import mlx.nn as nn
        print("Using MLX backend for enhanced performance on Apple Silicon")
        ARRAY_MODULE = mx
        COMPUTE_MODULE = mx
    except ImportError:
        print("MLX not found, falling back to NumPy. Install MLX for better performance on Apple Silicon:")
        print("pip install mlx")
        ARRAY_MODULE = np
        COMPUTE_MODULE = np
elif HAS_CUDA:
    try:
        import torch
        import torch.nn as nn
        print("Using PyTorch CUDA backend for enhanced GPU performance")
        ARRAY_MODULE = torch
        COMPUTE_MODULE = torch
        # Set default tensor type to cuda
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    except ImportError:
        print("PyTorch not found, falling back to NumPy. Install PyTorch for GPU acceleration:")
        print("pip install torch")
        ARRAY_MODULE = np
        COMPUTE_MODULE = np
else:
    ARRAY_MODULE = np
    COMPUTE_MODULE = np

class QMeans:
    def __init__(self, training_input, k, centroids_ini=None, threshold=0.04, seed=0, backend=None, shots=1024):
        """
        Args:
            training_input: ndarray of shape (n_samples, 2)
            k: number of clusters
            centroids_ini: optional initial centers, shape (k, 2)
            threshold: stopping tolerance on center movement (L2)
            seed: RNG seed for reproducibility
            backend: optional Qiskit backend (defaults to AerSimulator)
            shots: SWAP-test shots
        """
        np.random.seed(seed)
        self.data = np.asarray(training_input, dtype=float)
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
        if IS_APPLE_SILICON:
            # Convert data to MLX arrays for faster computation
            self.data = ARRAY_MODULE.array(self.data)
            centers_old = ARRAY_MODULE.zeros_like(self.centers)
            centers_new = ARRAY_MODULE.array(self.centers)
        elif HAS_CUDA:
            # Convert data to CUDA tensors for GPU acceleration
            self.data = ARRAY_MODULE.tensor(self.data, device='cuda')
            centers_old = ARRAY_MODULE.zeros_like(ARRAY_MODULE.tensor(self.centers, device='cuda'))
            centers_new = ARRAY_MODULE.tensor(self.centers, device='cuda')
        else:
            centers_old = np.zeros_like(self.centers)
            centers_new = deepcopy(self.centers)

        error = float('inf')
        it = 0
        while error > self.threshold:
            it += 1
            # Distances via quantum SWAP-test (returns raw '1' counts)
            # Convert to probabilities by dividing by shots, then argmin.
            if IS_APPLE_SILICON or HAS_CUDA:
                # Process in batches for better hardware acceleration
                batch_size = 256  # Adjust based on memory constraints
                all_dcounts = []
                for i in range(0, len(self.data), batch_size):
                    batch = self.data[i:i + batch_size]
                    if HAS_CUDA:
                        # Move batch to CPU for quantum circuit execution
                        batch_cpu = batch.cpu().numpy()
                        centers_cpu = centers_new.cpu().numpy()
                    else:
                        batch_cpu = batch.tolist()
                        centers_cpu = centers_new.tolist()
                    
                    dcounts = ARRAY_MODULE.array([
                        distance_centroids_parallel(x, centers_cpu, 
                                                 backend=self.backend, shots=self.shots)
                        for x in batch_cpu
                    ], dtype=float)
                    
                    if HAS_CUDA:
                        # Move results back to GPU
                        dcounts = ARRAY_MODULE.tensor(dcounts, device='cuda')
                    all_dcounts.append(dcounts)
                
                dcounts = ARRAY_MODULE.cat(all_dcounts) if HAS_CUDA else ARRAY_MODULE.concatenate(all_dcounts)
            else:
                dcounts = np.array([
                    distance_centroids_parallel(x, centers_new, 
                                             backend=self.backend, shots=self.shots)
                    for x in self.data
                ], dtype=float)
            
            distances = dcounts / float(self.shots)

            # Assign to closest center
            self.clusters = (
                ARRAY_MODULE.argmin(distances, axis=1) if IS_APPLE_SILICON 
                else np.argmin(distances, axis=1)
            )

            centers_old[:] = centers_new
            # Recompute centers as means; if a cluster is empty, re-sample
            for i in range(self.k):
                if IS_APPLE_SILICON:
                    mask = (self.clusters == i)
                    if ARRAY_MODULE.any(mask):
                        centers_new[i] = ARRAY_MODULE.mean(self.data[mask], axis=0)
                    else:
                        centers_new[i] = ARRAY_MODULE.random.normal(scale=0.6, size=centers_new[i].shape)
                else:
                    mask = (self.clusters == i)
                    if np.any(mask):
                        centers_new[i] = self.data[mask].mean(axis=0)
                    else:
                        centers_new[i] = np.random.normal(scale=0.6, size=centers_new[i].shape)

            error = (
                float(ARRAY_MODULE.linalg.norm(centers_new - centers_old)) if IS_APPLE_SILICON
                else float(np.linalg.norm(centers_new - centers_old))
            )
            
            if verbose:
                print(f"[iter {it}] center shift L2 = {error:.6f}")

        # Convert back to numpy for compatibility with rest of codebase
        if IS_APPLE_SILICON:
            self.centers = centers_new.numpy()
            self.data = self.data.numpy()
        else:
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
        counts = distance_centroids_parallel(point, 
                                          self.centers.tolist() if IS_APPLE_SILICON else self.centers, 
                                          backend=self.backend, 
                                          shots=self.shots)
        if IS_APPLE_SILICON:
            dists = ARRAY_MODULE.array(counts, dtype=float) / float(self.shots)
            return int(ARRAY_MODULE.argmin(dists))
        else:
            dists = np.array(counts, dtype=float) / float(self.shots)
            return int(np.argmin(dists))


def load_dataset(name: str):
    """
    Example toy dataset loader for demonstration.
    Returns (data points, category labels).
    """
    if name == "toy":
        # Create a simple 2D dataset with 3 clusters
        np.random.seed(42)
        n_samples = 300
        
        # Generate 3 clusters
        cluster1 = np.random.normal(loc=[-2, -2], scale=0.3, size=(n_samples//3, 2))
        cluster2 = np.random.normal(loc=[0, 2], scale=0.3, size=(n_samples//3, 2))
        cluster3 = np.random.normal(loc=[2, -1], scale=0.3, size=(n_samples//3, 2))
        
        # Combine data and create labels
        data = np.vstack([cluster1, cluster2, cluster3])
        labels = np.repeat([0, 1, 2], n_samples//3)
        
        return data, labels
    else:
        raise ValueError(f"Unknown dataset: {name}")

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