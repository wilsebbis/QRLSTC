#!/usr/bin/env python3
"""

Drop-in quantum variant for init centers and base clustering that uses
q-means++ seeding and a SWAP-test distance oracle.

- Uses qiskit_aer.AerSimulator (no import of qiskit.Aer).
- Distance calls delegate to q_distance.distance_centroids_parallel,
  which transpiles and runs circuits on the provided backend.
- Same output shape as classical: res = [(overall_sim, overall_sim, cluster_dict)]
  where cluster_dict[i] = [avg_dist, center_traj, list_of_dists, list_of_assigned_subtrajs]
"""







import argparse
import os
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Iterable, List, Tuple, Optional, Union

import numpy as np
import matplotlib.pyplot as plt

# Project types (for compatibility with your repo)
from point import Point       # noqa: F401
from segment import Segment   # noqa: F401
from traj import Traj

# Qiskit imports
from qiskit import transpile
from qiskit.circuit import Parameter, QuantumCircuit
from qiskit.circuit.library import RealAmplitudes
from qiskit.quantum_info import SparsePauliOp
from qiskit_aer import AerSimulator
from qiskit.transpiler import PassManager
from qiskit.transpiler.passes import Optimize1qGates, CommutativeCancellation

# Qiskit ML imports
from qiskit_machine_learning.neural_networks import SamplerQNN
from qiskit.circuit.library import ZZFeatureMap
from qiskit.primitives import Sampler
import qiskit_algorithms.utils.algorithm_globals

# Initialize simulator with automatic method selection
sim = AerSimulator(method='automatic')

# Global flag for circuit diagram saving
save_circuit_enabled = False

# Quantum distance primitive (updated to AerSimulator + transpile)
from q_distance import distance_centroids_parallel

def save_circuit_diagram(qc, filename, style=None):
    """
    Save a quantum circuit diagram as an image.
    Args:
        qc: QuantumCircuit to visualize
        filename: Output filename (will be saved in /out directory)
        style: Optional style dictionary for circuit drawing
    """
    try:
        from qiskit.visualization import circuit_drawer
        import os
        
        # Ensure output directory exists
        os.makedirs('out', exist_ok=True)
        
        # Set default style if none provided
        if style is None:
            style = {
                'backgroundcolor': '#FFFFFF',
                'fontsize': 14,
                'compress': True,
                'fold': 20  # Fold wide circuits
            }
        
        # Draw and save the circuit
        circuit_drawer(qc, output='mpl', style=style, filename=f'out/{filename}')
        print(f"Circuit diagram saved to out/{filename}")
    except Exception as e:
        print(f"Warning: Could not save circuit diagram: {str(e)}")

# Optimized quantum circuit implementations
def create_optimized_swap_test(state1, state2, save_diagram=False, diagram_name=None):
    """
    Create an optimized SWAP test circuit with reduced gate depth.
    Args:
        state1: First quantum state to compare
        state2: Second quantum state to compare
        save_diagram: Whether to save the circuit diagram
        diagram_name: Optional custom name for the diagram file
    """
    global save_circuit_enabled
    save_diagram = save_diagram and save_circuit_enabled  # Only save if globally enabled
    
    # For 2D points, we need 1 qubit per dimension plus ancilla
    n_qubits = 3  # 1 ancilla + 1 qubit per state
    qc = QuantumCircuit(n_qubits)
    
    # Normalize and encode 2D points onto single qubits
    def encode_2d_state(state):
        x, y = state
        # Normalize the vector
        norm = np.sqrt(x*x + y*y)
        if norm == 0:
            return [1.0]  # Default state for zero vector
        
        # Normalize x and y
        x_norm = x / norm
        y_norm = y / norm
        
        # Convert to quantum state [cos(θ/2), sin(θ/2)e^(iφ)]
        theta = 2 * np.arccos(x_norm)  # Maps x to amplitude
        phi = np.arctan2(1, 1) if y_norm >= 0 else np.arctan2(-1, 1)  # Phase based on y sign
        
        # Create normalized state vector
        return [np.cos(theta/2), np.sin(theta/2) * np.exp(1j * phi)]
    
    # Initialize states as single-qubit states
    qc.initialize(encode_2d_state(state1), [1])
    qc.initialize(encode_2d_state(state2), [2])
    
    # Optimized SWAP test
    qc.h(0)  # Hadamard on ancilla
    qc.cswap(0, 1, 2)  # Controlled-SWAP between states
    qc.h(0)  # Final Hadamard
    
    # Measure the ancilla
    qc.measure_all()
    
    # Save circuit diagram if requested
    if save_diagram:
        if diagram_name is None:
            diagram_name = 'swap_test_circuit.png'
        save_circuit_diagram(qc, diagram_name)
    
    return qc

def optimize_quantum_backend(backend_spec=None):
    """
    Configure backend with optimized settings.
    """
    be = _resolve_backend(backend_spec)
    
    # Optimization passes for transpilation
    pm = PassManager([
        Optimize1qGates(),
        CommutativeCancellation(),
    ])
    
    # Backend options for better performance
    # Using current Aer options
    if isinstance(be, AerSimulator):
        be.set_options(
            fusion_enable=True,        # Enable circuit optimization
            fusion_max_qubit=8,        # Maximum qubits for fusion optimization
            fusion_threshold=14,       # Threshold for optimization
            memory=False,              # Disable storing measurement memory
            max_parallel_shots=0,      # Use maximum available threads for parallel shot execution
            max_parallel_experiments=0  # Maximum parallel execution
        )
    
    return be, pm

def batch_quantum_distances(points, centers, backend, shots, batch_size=100, save_first_circuit=False):
    """
    Batch quantum circuit execution for better performance.
    
    Args:
        points: List of point states to compare
        centers: List of center states to compare against
        backend: Quantum backend to use
        shots: Number of shots for each circuit
        batch_size: Size of batches for circuit execution
        save_first_circuit: If True, save a diagram of the first circuit
    
    Note:
        Uses transpile with latest Qiskit optimizations and runs with current
        Aer backend settings for optimal performance.
    """
    all_circuits = []
    results = []
    first_circuit = True  # Track if we've saved the first circuit
    
    # Create circuits in batches
    for i in range(0, len(points), batch_size):
        batch_points = points[i:i+batch_size]
        batch_circuits = []
        
        for point in batch_points:
            for center in centers:
                # Save diagram only for the very first circuit
                save_this_circuit = save_first_circuit and first_circuit
                first_circuit = False  # Clear flag after first circuit
                
                qc = create_optimized_swap_test(point, center, 
                                              save_diagram=save_this_circuit,
                                              diagram_name='first_swap_test_circuit.png')
                batch_circuits.append(qc)
        
        # Get optimized backend
        be, _ = optimize_quantum_backend(backend)
        
        # Transpile with optimization passes
        # Use simpler transpilation settings that don't require timing information
        transpiled_batch = transpile(
            batch_circuits,
            backend=be,
            optimization_level=3,  # Maximum optimization level
            layout_method='sabre'  # Advanced layout method
        )
        all_circuits.extend(transpiled_batch)
    
    # Execute in batches
    for i in range(0, len(all_circuits), batch_size):
        batch = all_circuits[i:i+batch_size]
        job = backend.run(batch, shots=shots)
        # Get counts and ensure they're in the correct format
        batch_results = []
        for circuit_result in job.result().results:
            counts = circuit_result.data.counts
            # Convert counts to dictionary if it's not already
            if isinstance(counts, str):
                try:
                    counts = eval(counts)
                except:
                    counts = {'1': shots}  # Default if parsing fails
            batch_results.append(counts)
        results.extend(batch_results)
    
    return results
# -------------------------
# Clustering (assignment)
# -------------------------

# --- Qiskit ML Quantum Center Initialization Example ---
def qiskit_ml_initialize_centers(trajs, K, seed=42):
    """
    Example: Use Qiskit ML feature maps and SamplerQNN to initialize cluster centers.
    Returns indices of selected centers.
    """
    qiskit_algorithms.utils.algorithm_globals.random_seed = seed
    n_features = 2  # For (x, y) features
    feature_map = ZZFeatureMap(feature_dimension=n_features)
    ansatz = RealAmplitudes(num_qubits=n_features, reps=1)
    qc = QuantumCircuit(n_features)
    qc.compose(feature_map, inplace=True)
    qc.compose(ansatz, inplace=True)
    sampler = Sampler()
    qnn = SamplerQNN(circuit=qc, input_params=feature_map.parameters,
                     output_shape=2**n_features, sampler=sampler)

    # Extract features for each trajectory
    feats = np.array([_traj_feature(tr, mode='mean') for tr in trajs])
    # Normalize features to [0, 1]
    minx, maxx, miny, maxy = _collect_minmax(trajs)
    feats_norm = np.array(_normalize_features(feats, minx, maxx, miny, maxy))

    # Use quantum circuit output as a pseudo-random score for initialization
    scores = []
    for feat in feats_norm:
        # QNN expects a list of dicts for batch input
        input_dict = dict(zip(feature_map.parameters, feat))
        output = qnn.forward(input_dict)
        scores.append(np.sum(output))
    # Select K centers with highest scores
    centers_idx = np.argsort(scores)[-K:]
    return list(centers_idx)

# You can use qiskit_ml_initialize_centers in place of qmeanspp_initialize_centers for quantum seeding.

# -----------------------------
# Utilities: feature extraction
# -----------------------------

def _collect_minmax(trajs: Iterable[Traj]) -> Tuple[float, float, float, float]:
    """Compute global min/max for x and y across all trajectory points."""
    minx = miny = float("inf")
    maxx = maxy = float("-inf")
    for tr in trajs:
        for p in getattr(tr, 'points', []):
            x, y = getattr(p, 'x', None), getattr(p, 'y', None)
            if x is None or y is None:
                continue
            if x < minx: minx = x
            if x > maxx: maxx = x
            if y < miny: miny = y
            if y > maxy: maxy = y
    # Avoid degenerate ranges
    if not np.isfinite(minx) or not np.isfinite(maxx) or minx == maxx:
        minx, maxx = -1.0, 1.0
    if not np.isfinite(miny) or not np.isfinite(maxy) or miny == maxy:
        miny, maxy = -1.0, 1.0
    return minx, maxx, miny, maxy


def _normalize_to_unit_interval(v: float, vmin: float, vmax: float) -> float:
    return 0.5 if vmax - vmin <= 0 else (v - vmin) / (vmax - vmin)


def _to_neg1_pos1(u: float) -> float:
    return 2.0 * u - 1.0


def _traj_feature(tr: Traj, mode: str = "mean") -> Tuple[float, float]:
    """
    Map a trajectory to a 2D feature (x,y) in native units (before global normalization).

    Modes:
      - 'mean' : mean of (x, y)
      - 'start': first point
      - 'end'  : last point
      - 'bbox' : bounding-box center
    """
    pts = getattr(tr, 'points', [])
    if not pts:
        return 0.0, 0.0

    if mode == 'start':
        p = pts[0]
        return float(getattr(p, 'x', 0.0)), float(getattr(p, 'y', 0.0))
    if mode == 'end':
        p = pts[-1]
        return float(getattr(p, 'x', 0.0)), float(getattr(p, 'y', 0.0))
    if mode == 'bbox':
        xs = [float(getattr(p, 'x', 0.0)) for p in pts]
        ys = [float(getattr(p, 'y', 0.0)) for p in pts]
        return 0.5 * (min(xs) + max(xs)), 0.5 * (min(ys) + max(ys))

    # mean
    xs, ys = [], []
    for p in pts:
        x, y = getattr(p, 'x', None), getattr(p, 'y', None)
        if x is None or y is None:
            continue
        xs.append(float(x)); ys.append(float(y))
    if not xs:
        return 0.0, 0.0
    return float(np.mean(xs)), float(np.mean(ys))


def _normalize_features(
    feats: List[Tuple[float, float]],
    minx: float, maxx: float,
    miny: float, maxy: float
) -> List[Tuple[float, float]]:
    out = []
    for x, y in feats:
        u = _normalize_to_unit_interval(x, minx, maxx)
        v = _normalize_to_unit_interval(y, miny, maxy)
        out.append((_to_neg1_pos1(u), _to_neg1_pos1(v)))
    return out


# ----------------------------------------
# Temporal overlap (to mirror classical IED)
# ----------------------------------------

def _has_temporal_overlap(tr_a: Traj, tr_b: Traj) -> bool:
    """Return True if trajectories have overlapping time ranges.
    If no time attribute exists, default to True (no filtering).
    """
    def _minmax_t(tr: Traj) -> Optional[Tuple[float, float]]:
        ts = [getattr(p, 't', None) for p in getattr(tr, 'points', [])]
        ts = [float(t) for t in ts if t is not None]
        return (min(ts), max(ts)) if ts else None

    a = _minmax_t(tr_a); b = _minmax_t(tr_b)
    if a is None or b is None:
        return True
    return not (a[1] < b[0] or b[1] < a[0])


# ------------------------------------------
# Backend resolution (no qiskit.Aer import)
# ------------------------------------------

def _resolve_backend(backend_spec: Optional[Union[str, object]] = None):
    """
    Returns an optimized run-capable backend with improved settings.
    """
    from qiskit_aer import AerSimulator
    
    # Get base backend
    if backend_spec is None:
        backend = AerSimulator()
    elif hasattr(backend_spec, "run"):
        backend = backend_spec
    elif isinstance(backend_spec, str):
        name = backend_spec.lower()
        if name in ("qasm_simulator", "aer_simulator", "qasm", "aer"):
            backend = AerSimulator()
        else:
            try:
                from qiskit_aer import Aer
                backend = Aer.get_backend(backend_spec)
            except Exception:
                backend = AerSimulator()
    else:
        backend = AerSimulator()
    
    # Apply optimized settings
    if isinstance(backend, AerSimulator):
        # Configure for better performance with currently supported options
        backend.set_options(
            fusion_enable=True,        # Enable circuit optimization
            fusion_max_qubit=8,        # Maximum qubits for fusion optimization
            fusion_threshold=14,       # Threshold for optimization
            memory=False,              # Disable storing measurement memory
            max_parallel_shots=0,      # Use maximum available threads for parallel shot execution
            max_parallel_experiments=0  # Maximum parallel execution
        )
    
    return backend


# ------------------------------------------
# Quantum distance wrappers & q-means++ seeding
# ------------------------------------------

def _q_point_to_centers_distances(point_xy: Tuple[float, float],
                                  centers_xy: List[Tuple[float, float]],
                                  backend: Optional[Union[str, object]],
                                  shots: int) -> List[float]:
    """Estimate distances p1 (probabilities) from a point to many centers via optimized SWAP-test."""
    if not centers_xy:
        return []
    
    # Normalize states for quantum encoding
    def normalize_state(xy):
        x, y = xy
        norm = np.sqrt(x*x + y*y)
        return [x/norm, y/norm] if norm > 0 else [1.0, 0.0]
    
    point_state = normalize_state(point_xy)
    center_states = [normalize_state(c) for c in centers_xy]
    
    # Use batched execution with optimized circuits
    be = _resolve_backend(backend)
    # Save circuit diagram only for the first execution of this function
    save_first = not hasattr(_q_point_to_centers_distances, '_circuit_saved')
    _q_point_to_centers_distances._circuit_saved = True  # Mark that we've saved a circuit
    results = batch_quantum_distances([point_state], center_states, be, shots, save_first_circuit=save_first)
    
    # Convert results to distances
    distances = []
    for counts_str in results:
        # Parse the counts dictionary from the result string
        if isinstance(counts_str, str):
            try:
                # Convert string representation to dictionary
                counts = eval(counts_str)
            except:
                # If parsing fails, assume no 0 measurements
                counts = {'1': shots}
        else:
            counts = counts_str
            
        # Count '0' measurements for probability
        p0 = counts.get('0', 0) / shots
        # Convert to distance metric
        dist = 2 * p0 - 1
        distances.append(dist)
    
    return distances


def qmeanspp_initialize_centers(trajs: List[Traj], K: int,
                                feats_trajs_xy: List[Tuple[float, float]],
                                backend: Optional[Union[str, object]] = None,
                                shots: int = 1024,
                                initial: str = 'random',
                                rng: Optional[np.random.Generator] = None) -> List[int]:
    """q-means++ initialization using SWAP-test distances. Returns indices of selected centers."""
    n = len(trajs)
    if K <= 0 or n == 0:
        return []
    if rng is None:
        rng = np.random.default_rng()

    # First center
    if initial == 'far' and n > 1:
        feats = np.asarray(feats_trajs_xy)
        mu = feats.mean(axis=0)
        first = int(np.argmax(np.linalg.norm(feats - mu, axis=1)))
    else:
        first = int(rng.integers(low=0, high=n))

    centers_idx = [first]

    # Add remaining centers with probs ∝ D(x)^2
    while len(centers_idx) < K:
        chosen_feats = [feats_trajs_xy[i] for i in centers_idx]
        D = np.zeros(n, dtype=float)
        for i in range(n):
            if i in centers_idx:
                continue
            p1_list = _q_point_to_centers_distances(feats_trajs_xy[i], chosen_feats, backend, shots)
            D[i] = float(np.min(p1_list)) if p1_list else 1.0

        weights = D ** 2
        s = weights.sum()
        if s <= 0:
            candidates = [i for i in range(n) if i not in centers_idx]
            next_idx = int(rng.choice(candidates))
        else:
            probs = weights / s
            next_idx = int(rng.choice(np.arange(n), p=probs))
        centers_idx.append(next_idx)

    return centers_idx


# -------------------------
# Clustering (assignment)
# -------------------------

def getbaseclus_q(trajs: List[Traj], k: int, subtrajs: List[Traj],
                  feats_trajs_xy: List[Tuple[float, float]],
                  feats_sub_xy: List[Tuple[float, float]],
                  backend: Optional[Union[str, object]] = None,
                  shots: int = 1024,
                  init_mode: str = 'random'):
    """
    Assign subtrajectories to clusters using quantum-estimated distances.
    Returns cluster_dict with the same structure as the classical implementation.
    """
    print("Starting q-means++ center initialization...")
    t0 = time.time()
    centers_idx = qmeanspp_initialize_centers(trajs, k, feats_trajs_xy,
                                              backend=backend, shots=shots,
                                              initial=init_mode)
    centers = [trajs[i] for i in centers_idx]
    centers_xy = [feats_trajs_xy[i] for i in centers_idx]
    t1 = time.time()
    print(f"→ Center initialization done in {t1 - t0:.2f} seconds.\n")

    cluster_dict = defaultdict(list)
    cluster_segments = defaultdict(list)
    dists_dict = defaultdict(list)

    total = len(subtrajs)
    step = max(1, total // 20)  # ~5%
    last_time = time.time()

    for i, subtraj in enumerate(subtrajs):
        if i % step == 0 or i == total - 1:
            percent_done = int((i / max(1, total)) * 100)
            now = time.time()
            delta = now - last_time
            print(f"Clustering {percent_done}% done... (step took {delta:.2f} seconds)")
            last_time = now

        valid_mask = [_has_temporal_overlap(center, subtraj) for center in centers]
        if not any(valid_mask):
            continue

        valid_centers_xy = [centers_xy[j] for j, ok in enumerate(valid_mask) if ok]
        p1_list = _q_point_to_centers_distances(feats_sub_xy[i], valid_centers_xy, backend, shots)
        if not p1_list:
            continue

        full_dists = [float('inf')] * k
        vi = 0
        for j, ok in enumerate(valid_mask):
            if ok:
                full_dists[j] = p1_list[vi]
                vi += 1

        minidx = int(np.argmin(full_dists))
        mindist = full_dists[minidx]
        if not np.isfinite(mindist):
            continue

        cluster_segments[minidx].append(subtraj)
        dists_dict[minidx].append(mindist)

    # Ensure no empty clusters
    for i in range(k):
        if len(cluster_segments[i]) == 0:
            cluster_segments[i].append(centers[i])
            dists_dict[i].append(0.0)

    # Build output
    for i in cluster_segments.keys():
        center = centers[i]
        temp_dist = dists_dict[i]
        aver_dist = float(np.mean(temp_dist)) if temp_dist else float('inf')
        cluster_dict[i].append(aver_dist)
        cluster_dict[i].append(center)
        cluster_dict[i].append(temp_dist)
        cluster_dict[i].append(cluster_segments[i])

    print("Clustering 100% done.")
    return cluster_dict


# -------------------------
# Top-level save wrapper
# -------------------------

def saveclus_q(k: int, subtrajs: List[Traj], trajs: List[Traj], amount: int,
               backend: Optional[Union[str, object]] = None, shots: int = 1024,
               feature_mode: str = 'mean', init_mode: str = 'random'):
    """Run quantum clustering and compute overall similarity metric."""
    trajs = trajs[:amount]

    # Build features + normalization ranges over trajs+subtrajs
    joint_for_range = list(trajs) + list(subtrajs)
    minx, maxx, miny, maxy = _collect_minmax(joint_for_range)

    feats_trajs = [_traj_feature(tr, mode=feature_mode) for tr in trajs]
    feats_sub   = [_traj_feature(tr, mode=feature_mode) for tr in subtrajs]
    feats_trajs_xy = _normalize_features(feats_trajs, minx, maxx, miny, maxy)
    feats_sub_xy   = _normalize_features(feats_sub,   minx, maxx, miny, maxy)

    cluster_dict = getbaseclus_q(trajs, k, subtrajs,
                                 feats_trajs_xy, feats_sub_xy,
                                 backend=backend, shots=shots,
                                 init_mode=init_mode)

    # Mean of assigned distances
    count_sim = 0.0
    traj_num = 0
    for i in cluster_dict.keys():
        dlist = cluster_dict[i][2]
        count_sim += float(np.sum(dlist))
        traj_num += int(len(cluster_dict[i][3]))
    overall_sim = 1e10 if traj_num == 0 else (count_sim / float(traj_num))

    return [(overall_sim, overall_sim, cluster_dict)]


# -------------------------
# SSE using quantum distances
# -------------------------

def plot_clusters(cluster_dict, out_png, alpha=0.5, center_alpha=1.0, sample_rate=1, bg_image=None):
    """
    Plot each subtrajectory point color-coded by cluster assignment, and cluster centers.
    Args:
        cluster_dict: dict, output from getbaseclus (or res[0][2])
        out_png: save plot to file
        alpha: float, transparency for scatter plot points
        center_alpha: float, transparency for cluster center trajectories
        sample_rate: int, plot every Nth point of trajectories
        bg_image: str, path to background image file
    """
    import gc
    gc.collect()
    
    # Create figure and axis objects
    fig, ax = plt.subplots(figsize=(10, 8))
    colors = plt.colormaps['tab10'].resampled(len(cluster_dict))
    
    # Add background image if provided
    if bg_image:
        try:
            img = plt.imread(bg_image)
            # Calculate bounds for the plot
            all_xs = []
            all_ys = []
            for cluster in cluster_dict.values():
                center = cluster[1]
                all_xs.extend([p.x for p in center.points])
                all_ys.extend([p.y for p in center.points])
            min_x, max_x = min(all_xs), max(all_xs)
            min_y, max_y = min(all_ys), max(all_ys)
            ax.imshow(img, aspect='auto', extent=[min_x, max_x, min_y, max_y])
        except Exception as e:
            print(f"Warning: Could not load background image: {str(e)}")
    
    # Pre-process data to minimize memory usage during plotting
    for i, cluster in cluster_dict.items():
        subtrajs = cluster[3]
        # Plot in batches to reduce memory usage
        batch_size = 100
        for j in range(0, len(subtrajs), batch_size):
            batch = subtrajs[j:j+batch_size]
            xs = []
            ys = []
            for traj in batch:
                # Sample points according to sample_rate
                points = traj.points[::sample_rate]
                xs.extend([p.x for p in points])
                ys.extend([p.y for p in points])
                del traj
            ax.scatter(xs, ys, s=5, color=colors(i), 
                      label=f"Cluster {i}" if j == 0 else "", 
                      alpha=alpha)
            del xs, ys
            gc.collect()
        
        # Plot cluster center
        center_traj = cluster[1]
        # Sample center points according to sample_rate
        center_points = center_traj.points[::sample_rate]
        xs = [p.x for p in center_points]
        ys = [p.y for p in center_points]
        ax.plot(xs, ys, color=colors(i), linewidth=2, marker='*', 
                markersize=10, label=f"Center {i}", alpha=center_alpha)
    
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Quantum Subtrajectory Clusters')
    ax.legend()
    
    # Save and cleanup
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    del fig, ax
    gc.collect()
    print(f"Cluster plot saved to {out_png}")

def plot_elbow(k_values, sse_values, n_values, out_png):
    """Plot SSE and normalized SSE vs k values."""
    import gc
    gc.collect()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    
    # Plot raw SSE
    ax1.plot(k_values, sse_values, 'bo-', linewidth=2)
    ax1.set_xlabel('Number of Clusters (k)')
    ax1.set_ylabel('Sum of Squared Errors (SSE)')
    ax1.set_title('Quantum Elbow Plot for Clustering')
    ax1.grid(True)
    ax1.set_xticks(k_values)
    
    # Plot normalized SSE (SSE/n)
    normalized_sse = [sse/n if n > 0 else float('inf') for sse, n in zip(sse_values, n_values)]
    ax2.plot(k_values, normalized_sse, 'ro-', linewidth=2)
    ax2.set_xlabel('Number of Clusters (k)')
    ax2.set_ylabel('Average SSE per Assignment')
    ax2.grid(True)
    ax2.set_xticks(k_values)
    
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    del fig, ax1, ax2
    gc.collect()
    print(f"Elbow plot saved to {out_png}")

def plot_silhouette(k_values, silhouette_values, out_png):
    """Plot silhouette coefficients for different k values."""
    import gc
    gc.collect()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, silhouette_values, 'go-', linewidth=2)
    ax.set_xlabel('Number of Clusters (k)')
    ax.set_ylabel('Silhouette Coefficient')
    ax.set_title('Quantum Silhouette Analysis')
    ax.grid(True)
    ax.set_xticks(k_values)
    
    # Add value annotations
    for i, txt in enumerate(silhouette_values):
        ax.annotate(f'{txt:.3f}', (k_values[i], silhouette_values[i]), 
                   textcoords="offset points", xytext=(0,10), ha='center')
    
    fig.tight_layout()
    fig.savefig(out_png, bbox_inches='tight')
    plt.close(fig)
    del fig, ax
    gc.collect()
    print(f"Silhouette plot saved to {out_png}")

def compute_silhouette_q(cluster_dict, backend=None, shots=1024, feature_mode='mean'):
    """
    Compute quantum silhouette coefficient for the clustering.
    Returns average silhouette coefficient across all points.
    """
    all_silhouettes = []
    
    # Get global min/max for normalization
    all_trajs = []
    for i in cluster_dict:
        all_trajs.append(cluster_dict[i][1])  # center
        all_trajs.extend(cluster_dict[i][3])  # members
    minx, maxx, miny, maxy = _collect_minmax(all_trajs)
    
    # For each cluster
    for cluster_idx in cluster_dict:
        cluster_trajs = cluster_dict[cluster_idx][3]
        
        # For each trajectory in the cluster
        for traj in cluster_trajs:
            # Get normalized features for current trajectory
            p_xy = _normalize_features([_traj_feature(traj, mode=feature_mode)],
                                     minx, maxx, miny, maxy)[0]
            
            # 1. Calculate a (average distance to points in same cluster)
            same_cluster_feats = [_normalize_features([_traj_feature(tr, mode=feature_mode)],
                                                    minx, maxx, miny, maxy)[0] 
                                for tr in cluster_trajs if tr != traj]
            if same_cluster_feats:
                a_dists = _q_point_to_centers_distances(p_xy, same_cluster_feats, backend, shots)
                a = np.mean(a_dists) if a_dists else float('inf')
            else:
                a = 0.0
            
            # 2. Calculate b (average distance to points in next best cluster)
            b = float('inf')
            for other_idx in cluster_dict:
                if other_idx != cluster_idx:
                    other_trajs = cluster_dict[other_idx][3]
                    other_feats = [_normalize_features([_traj_feature(tr, mode=feature_mode)],
                                                     minx, maxx, miny, maxy)[0] 
                                 for tr in other_trajs]
                    if other_feats:
                        b_dists = _q_point_to_centers_distances(p_xy, other_feats, backend, shots)
                        if b_dists:
                            avg_b = np.mean(b_dists)
                            b = min(b, avg_b)
            
            # 3. Calculate silhouette
            if b != float('inf'):
                if a == 0:
                    all_silhouettes.append(1)
                else:
                    s = (b - a) / max(a, b)
                    all_silhouettes.append(s)
    
    return np.mean(all_silhouettes) if all_silhouettes else 0

def compute_sse_q(res,
                  backend: Optional[Union[str, object]] = None,
                  shots: int = 1024,
                  feature_mode: str = 'mean') -> float:
    """Quantum-analogous SSE = sum(p1^2) across assignments."""
    if not res:
        return float('nan')
    cluster_dict = res[0][2]

    centers = []
    members = []
    for idx in cluster_dict:
        centers.append(cluster_dict[idx][1])
        members.extend(cluster_dict[idx][3])

    joint = centers + members
    minx, maxx, miny, maxy = _collect_minmax(joint)

    sse = 0.0
    for idx in cluster_dict:
        center_tr = cluster_dict[idx][1]
        sub_list  = cluster_dict[idx][3]

        c_xy = _normalize_features([_traj_feature(center_tr, mode=feature_mode)],
                                   minx, maxx, miny, maxy)[0]
        c_list = [c_xy]
        for tr in sub_list:
            p_xy = _normalize_features([_traj_feature(tr, mode=feature_mode)],
                                       minx, maxx, miny, maxy)[0]
            p1 = _q_point_to_centers_distances(p_xy, c_list, backend, shots)[0]
            sse += (p1 ** 2)
    return sse


# -------------------------
# Main
# -------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Cluster subtrajectories using q-means++ (quantum SWAP-test distances)."
    )
    parser.add_argument("-subtrajsfile", default='data/traclus_subtrajs',
                       help="Pickle file containing subtrajectories")
    parser.add_argument("-trajsfile", default='data/Tdrive_norm_traj_QRLSTC',
                       help="Pickle file containing full trajectories")
    parser.add_argument("-k_values", "-k", nargs='+', type=int, default=[10],
                       help="List of k values to try (for elbow plot)")
    parser.add_argument("-amount", type=int, default=1000,
                       help="Number of trajectories to use")
    
    # Add plotting parameters
    parser.add_argument("--alpha", type=float, default=0.5,
                       help="Alpha transparency for scatter plot points")
    parser.add_argument("--center-alpha", type=float, default=1.0,
                       help="Alpha transparency for cluster center trajectories")
    parser.add_argument("--sample", type=int, default=1,
                       help="Plot every Nth point of a trajectory")
    parser.add_argument("--bg-image", help="Path to a background image for the plot")
    parser.add_argument("--save-circuit", action="store_true",
                       help="Save quantum circuit diagram for visualization")

    parser.add_argument("--backend", default=None, help="None/qasm_simulator/aer_simulator ⇒ AerSimulator()")
    parser.add_argument("--shots", type=int, default=1024, help="SWAP-test shots")
    parser.add_argument("--init", choices=["random", "far"], default="random",
                        help="q-means++ first-center strategy")
    parser.add_argument("--feature-mode", "-feature-mode",
                        choices=["mean", "start", "end", "bbox"], default="mean",
                        help="2D feature used to encode trajectories")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    # Load data
    subtrajs = pickle.load(open(args.subtrajsfile, 'rb'))
    trajs = pickle.load(open(args.trajsfile, 'rb'))

    # Seeds
    np.random.seed(args.seed)
    random.seed(args.seed)

    # Resolve backend once and pass through
    be = _resolve_backend(args.backend)

    # Configure saving of circuit diagrams
    global save_circuit_enabled
    save_circuit_enabled = args.save_circuit
    if save_circuit_enabled:
        print("Circuit diagram saving is enabled. First circuit will be saved to out/first_swap_test_circuit.png")

    # Create output directory if it doesn't exist
    os.makedirs('out', exist_ok=True)

    print("\nStarting quantum clustering process for all k values...")
    overall_start_time = time.time()

    k_values = sorted(args.k_values)
    sse_values = []
    n_values = []
    silhouette_values = []
    results = []

    for k in k_values:
        print(f"\nProcessing for k={k}...")
        start_time = time.time()
        res = saveclus_q(k, subtrajs, trajs, args.amount,
                        backend=be, shots=args.shots,
                        feature_mode=args.feature_mode, init_mode=args.init)
        end_time = time.time()
        print(f"QRLSTC (quantum) clustering for k={k} completed in {end_time - start_time:.2f} seconds.")

        # Compute metrics
        sse = compute_sse_q(res, backend=be, shots=args.shots,
                           feature_mode=args.feature_mode)
        n = len(res[0][2])  # number of valid assignments
        silhouette = compute_silhouette_q(res[0][2], backend=be, shots=args.shots,
                                        feature_mode=args.feature_mode)
        
        sse_values.append(sse)
        n_values.append(n)
        silhouette_values.append(silhouette)
        
        print(f"QRLSTC Clustering metrics for k={k}:")
        print(f"  - SSE: {sse:.4f}")
        print(f"  - Valid assignments (N): {n}")
        print(f"  - SSE/N: {sse/n if n > 0 else float('inf'):.4f}")
        print(f"  - Silhouette coefficient: {silhouette:.4f}")

        # Save results
        out_file = f"out/quantum_k{k}_a{args.amount}"
        pickle.dump(res, open(out_file + '.pkl', 'wb'), protocol=2)
        print(f"Results for k={k} saved to {out_file}.pkl")
        
        results.append((res, k))
        plt.close('all')
    
    # Generate all plots after clustering
    print("\nGenerating visualization plots...")
    
    # Individual cluster plots
    for res, k in results:
        out_file = f"out/quantum_k{k}_a{args.amount}"
        try:
            plot_clusters(res[0][2], 
                         out_png=f"{out_file}.png",
                         alpha=args.alpha,
                         center_alpha=args.center_alpha,
                         sample_rate=args.sample,
                         bg_image=args.bg_image)
            plt.close('all')
        except Exception as e:
            print(f"Warning: Could not generate plot for k={k}: {str(e)}")
    
    # Elbow plot
    try:
        plot_elbow(k_values, sse_values, n_values, f"out/quantum_elbow_a{args.amount}.png")
        plt.close('all')
    except Exception as e:
        print(f"Warning: Could not generate elbow plot: {str(e)}")
    
    # Silhouette plot
    try:
        plot_silhouette(k_values, silhouette_values, f"out/quantum_silhouette_a{args.amount}.png")
        plt.close('all')
    except Exception as e:
        print(f"Warning: Could not generate silhouette plot: {str(e)}")

    # Print overall timing information
    overall_end_time = time.time()
    overall_elapsed_time = overall_end_time - overall_start_time
    print(f"\nEntire quantum clustering process completed in {overall_elapsed_time:.2f} seconds")
    print(f"Average time per k value: {overall_elapsed_time/len(k_values):.2f} seconds")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)