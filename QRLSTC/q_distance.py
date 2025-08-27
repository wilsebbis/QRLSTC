# q_distance.py (Qiskit Aer + transpile)
from typing import List, Tuple, Optional

import numpy as np
from numpy import pi
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit, transpile
from qiskit_aer import AerSimulator

def _encode_feature(x: float) -> float:
    """
    Map data feature values to rotation angles in [-pi, pi]:
        phi = (x + 1) * (pi/2)
    We use (theta=encoded_y, phi=encoded_x, lambda=0) in U(θ, φ, λ).
    """
    return (x + 1) * (pi / 2.0)


def _binary_combinations(n: int) -> List[str]:
    """All length-n binary strings."""
    out = []
    for i in range(2 ** n):
        s = bin(i)[2:]
        out.append(s.zfill(n))
    return out


def _binary_combinations_pos(n: int, index: int) -> List[str]:
    """
    All length-n binary strings where bit at position `index` (0-based from LSB/right)
    is '1'. (Matches original code's indexing: bin_number[n - index - 1] == '1')
    """
    wanted = []
    for s in _binary_combinations(n):
        if s[n - index - 1] == "1":
            wanted.append(s)
    return wanted


def _ensure_backend(backend=None):
    """Return a run-capable backend; default AerSimulator."""
    return backend if backend is not None else AerSimulator()


def distance_centroids_parallel(point: Tuple[float, float],
                                centroids: List[Tuple[float, float]],
                                backend=None,
                                shots: int = 1024) -> List[int]:
    """
    SWAP-test distances from a single 2D point to all `centroids` in parallel.
    Returns raw '1' counts per centroid ancilla (length k).
    """
    k = len(centroids)
    if k == 0:
        return []

    backend = _ensure_backend(backend)

    # Encode angles
    phi_list = [_encode_feature(c[0]) for c in centroids]
    theta_list = [_encode_feature(c[1]) for c in centroids]
    phi_input = _encode_feature(point[0])
    theta_input = _encode_feature(point[1])

    # Registers: input, centroid, ancilla — each size k — and k classical bits
    qreg_input = QuantumRegister(k, name="qreg_input")
    qreg_centroid = QuantumRegister(k, name="qreg_centroid")
    qreg_psi = QuantumRegister(k, name="qreg_psi")
    creg = ClassicalRegister(k, name="creg")
    qc = QuantumCircuit(qreg_input, qreg_centroid, qreg_psi, creg, name="qc")

    for i in range(k):
        # Encode centroid[i] and the single point on respective qubits
        qc.u(theta_list[i], phi_list[i], 0.0, qreg_centroid[i])
        qc.u(theta_input,    phi_input,  0.0, qreg_input[i])
        # SWAP test
        qc.h(qreg_psi[i])
        qc.cswap(qreg_psi[i], qreg_input[i], qreg_centroid[i])
        qc.h(qreg_psi[i])
        # Measure ancilla -> classical bit i
        qc.measure(qreg_psi[i], creg[i])

    tqc = transpile(qc, backend)
    result = backend.run(tqc, shots=shots).result()
    counts = result.get_counts(0) if hasattr(result, "get_counts") else {}

    # Sum counts for bitstrings where the i-th ancilla is '1'
    out = [0] * k
    for i in range(k):
        for key in _binary_combinations_pos(k, i):
            out[i] += counts.get(key, 0)
    return out


def distance_centroids(point: Tuple[float, float],
                       centroids: List[Tuple[float, float]],
                       backend=None,
                       shots: int = 1024) -> List[int]:
    """
    SWAP-test distances from a single 2D point to each centroid, run sequentially.
    Returns raw '1' counts (length k).
    """
    k = len(centroids)
    if k == 0:
        return []

    backend = _ensure_backend(backend)

    # Pre-encode
    theta_point = _encode_feature(point[1])
    phi_point = _encode_feature(point[0])

    results: List[int] = []
    for c in centroids:
        theta_cent = _encode_feature(c[1])
        phi_cent = _encode_feature(c[0])

        # Three qubits: |input>, |centroid>, |ancilla|
        qreg = QuantumRegister(3, "q")
        creg = ClassicalRegister(1, "c")
        qc = QuantumCircuit(qreg, creg, name="swaptest")

        # Encode states
        qc.u(theta_point, phi_point, 0.0, qreg[0])
        qc.u(theta_cent,  phi_cent,  0.0, qreg[1])

        # SWAP test
        qc.h(qreg[2])
        qc.cswap(qreg[2], qreg[0], qreg[1])
        qc.h(qreg[2])

        qc.measure(qreg[2], creg[0])

        tqc = transpile(qc, backend)
        result = backend.run(tqc, shots=shots).result()
        counts = result.get_counts(0) if hasattr(result, "get_counts") else {}
        results.append(counts.get("1", 0))

    return results