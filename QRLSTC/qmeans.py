# qmeans.py
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

import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from qiskit import QuantumRegister, ClassicalRegister
from qiskit import QuantumCircuit
from qiskit import transpile
from qiskit_aer import AerSimulator
import torch
from encoder import StateEncoder, AmplitudeEncoder

def amplitude_encode(vec):
    '''
    Uses AmplitudeEncoder from encoder.py to normalize the vector for quantum encoding.
    '''
    encoder = AmplitudeEncoder()
    vec_torch = torch.tensor(vec, dtype=torch.float32).unsqueeze(0)
    encoded = encoder(vec_torch).squeeze(0).numpy()
    return encoded


def _binary_combinations(n):
    """
    Returns all possible combinations of length n binary numbers as strings
    """
    combinations = []
    for i in range(2**n):
        bin_value = str(bin(i)).split('b')[1]
        while len(bin_value) < n:
            bin_value = "0" + bin_value 

        combinations.append(bin_value)

    return combinations


def _binary_combinations_pos(n, index):
    """
    Returns all possible combinations of binary numbers where bit index=1
    """
    combinations_pos = []
    for bin_number in _binary_combinations(n):
        if bin_number[n - index - 1] == '1':
            combinations_pos.append(bin_number)

    return combinations_pos


def distance_centroids_parallel(point, centroids, backend=None, shots=1024):
    '''
    Estimates distances using quantum computer specified by backend.
    Computes all point-centroid distances in a single batch job for efficiency.
    Args:
        point: Encoded trajectory vector.
        centroids: List of encoded centroid vectors.
        backend: Qiskit backend (default: AerSimulator).
        shots: Number of shots per circuit.
    Returns:
        List of quantum swap test distances.
    '''
    k = len(centroids)
    def pad_to_power_of_2(vec):
        length = len(vec)
        next_pow2 = 1 << (length - 1).bit_length()
        if length < next_pow2:
            vec = np.concatenate([vec, np.zeros(next_pow2 - length)])
        return vec

    point_enc = pad_to_power_of_2(amplitude_encode(point))
    centroids_enc = [pad_to_power_of_2(amplitude_encode(c)) for c in centroids]
    backend = backend or AerSimulator()
    circuits = []
    num_qubits = int(np.log2(len(point_enc)))
    for c_enc in centroids_enc:
        # Quantum registers: 1 ancillary + num_qubits for each vector
        q_anc = QuantumRegister(1, 'anc')
        q_point = QuantumRegister(num_qubits, 'point')
        q_centroid = QuantumRegister(num_qubits, 'centroid')
        c_reg = ClassicalRegister(1, 'c')
        qc = QuantumCircuit(q_anc, q_point, q_centroid, c_reg)

        # Hadamard on ancillary
        qc.h(q_anc[0])
        # Initialize point and centroid registers
        qc.initialize(point_enc/np.linalg.norm(point_enc), q_point)
        qc.initialize(c_enc/np.linalg.norm(c_enc), q_centroid)
        # Controlled swap
        for i in range(num_qubits):
            qc.cswap(q_anc[0], q_point[i], q_centroid[i])
        # Hadamard again
        qc.h(q_anc[0])
        # Measure ancillary
        qc.measure(q_anc[0], c_reg[0])
        circuits.append(qc)

    # Transpile and batch run
    try:
        transpiled = transpile(circuits, backend)
        job = backend.run(transpiled, shots=shots)
        results = job.result()
    except Exception as e:
        print(f"Quantum job failed: {e}")
        return [float('nan')] * k

    distances = []
    for i in range(k):
        counts = results.get_counts(i)
        prob_0 = counts.get('0', 0) / shots
        # Swap test: distance = sqrt(1 - 2*prob_0)
        distance = np.sqrt(max(0, 1 - 2*prob_0))
        distances.append(distance)
    return distances


class QMeans():
    '''
    Quantum-inspired k-means clustering for trajectory data.
    Args:
        trajectory_data: Encoded trajectory vectors.
        k: Number of clusters.
        centroids_ini: Optional initial centroids.
        threshold: Convergence threshold.
        seed: Random seed.
        backend: Qiskit backend for quantum swap test (default: AerSimulator).
        shots: Number of shots per quantum circuit.
    '''
    def __init__(self, trajectory_data, k, centroids_ini=None, threshold=0.04, seed=0, backend=None, shots=1024):
        np.random.seed(seed)
        self.data = trajectory_data
        self.k = k
        self.n = self.data.shape[0]
        self.threshold = threshold
        self.vector_dim = self.data.shape[1]
        self.backend = backend or AerSimulator()
        self.shots = shots
        if centroids_ini is None:
            self.centers = np.random.normal(scale=0.6, size=[k, self.vector_dim])
        else:
            self.centers = centroids_ini

    def run(self):
        centers_old = np.zeros(self.centers.shape)
        centers_new = deepcopy(self.centers)
        clusters = np.zeros(self.n)
        error = np.inf
        while error > self.threshold:
            # Batch quantum job for all data points
            distances = np.array([distance_centroids_parallel(x, centers_new, backend=self.backend, shots=self.shots) for x in self.data])
            clusters = np.argmin(distances, axis=1)
            centers_old = deepcopy(centers_new)
            for i in range(self.k):
                if np.sum(clusters == i) != 0:
                    centers_new[i] = np.mean(self.data[clusters == i], axis=0)
                else:
                    centers_new[i] = np.random.normal(scale=0.6, size=centers_new[i].shape)
            error = np.linalg.norm(centers_new - centers_old)
            print(f"Clustering error: {error}")
        self.centers = centers_new
        self.clusters = clusters

    def plot(self):
        colors=['green', 'blue', 'black', 'red']
        for i in range(self.n):
            plt.scatter(self.data[i, 0], self.data[i, 1], s=7, color=colors[int(self.clusters[i])])
        plt.scatter(self.centers[:, 0], self.centers[:, 1], marker='*', c='g', s=150)
        plt.show()

    def fit(self, point):
        distances = distance_centroids_parallel(point, self.centers, backend=self.backend, shots=self.shots)
        return np.argmin(distances)

if __name__ == "__main__":
    # Example: generate random trajectory data and encode
    torch_device = torch.device('cpu')
    num_points = 5  # number of (timestamp, x, y) triplets per trajectory
    batch_size = 10
    # Generate random trajectories
    trajectories = [torch.rand((num_points, 3)) for _ in range(batch_size)]
    encoder = StateEncoder(num_points=num_points, torch_device=torch_device)
    encoded_data = encoder(trajectories).cpu().numpy()
    k = 3
    backend = AerSimulator()
    shots = 512
    qmeans = QMeans(encoded_data, k, backend=backend, shots=shots)
    qmeans.run()
    qmeans.plot()