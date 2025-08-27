#!/usr/bin/env python3
"""
quantum_initcenters.py (q-means++)
----------------------------------

Drop-in quantum variant for init centers and base clustering that uses
q-means++ seeding and a SWAP-test distance oracle.

- Uses qiskit_aer.AerSimulator (no import of qiskit.Aer).
- Distance calls delegate to q_distance.distance_centroids_parallel,
  which transpiles and runs circuits on the provided backend.
- Same output shape as classical: res = [(overall_sim, overall_sim, cluster_dict)]
  where cluster_dict[i] = [avg_dist, center_traj, list_of_dists, list_of_assigned_subtrajs]
"""

import argparse
import pickle
import random
import sys
import time
from collections import defaultdict
from typing import Iterable, List, Tuple, Optional, Union

import numpy as np

# Project types (for compatibility with your repo)
from point import Point       # noqa: F401
from segment import Segment   # noqa: F401
from traj import Traj

# Quantum distance primitive (updated to AerSimulator + transpile)
from q_distance import distance_centroids_parallel

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
    Returns a run-capable backend:
      - None or 'qasm_simulator'/'aer_simulator' => AerSimulator()
      - If `backend_spec` is already a backend, return as-is
      - Else try qiskit_aer.Aer.get_backend(name); fallback to AerSimulator()
    """
    from qiskit_aer import AerSimulator
    if backend_spec is None:
        return AerSimulator()
    if hasattr(backend_spec, "run"):  # looks like a backend object
        return backend_spec
    if isinstance(backend_spec, str):
        name = backend_spec.lower()
        if name in ("qasm_simulator", "aer_simulator", "qasm", "aer"):
            return AerSimulator()
        # Try provider lookup if available
        try:
            from qiskit_aer import Aer
            return Aer.get_backend(backend_spec)
        except Exception:
            return AerSimulator()
    # Fallback
    return AerSimulator()


# ------------------------------------------
# Quantum distance wrappers & q-means++ seeding
# ------------------------------------------

def _q_point_to_centers_distances(point_xy: Tuple[float, float],
                                  centers_xy: List[Tuple[float, float]],
                                  backend: Optional[Union[str, object]],
                                  shots: int) -> List[float]:
    """Estimate distances p1 (probabilities) from a point to many centers via SWAP-test."""
    if not centers_xy:
        return []
    be = _resolve_backend(backend)
    counts = distance_centroids_parallel(point_xy, centers_xy, backend=be, shots=shots)
    return [c / float(shots) for c in counts]


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
    parser.add_argument("--subtrajsfile", "-subtrajsfile",
                        default="data/traclus_subtrajs",
                        help="Pickle file containing subtrajectories")

    parser.add_argument("--trajsfile", "-trajsfile",
                        default="data/Tdrive_norm_traj_QRLSTC",
                        help="Pickle file containing full trajectories")

    parser.add_argument("--k", "-k", type=int, default=10, help="Number of clusters")
    parser.add_argument("--amount", "-amount", type=int, default=1000,
                        help="Number of trajectories to use")

    parser.add_argument("--centerfile", "-centerfile",
                        default="data/tdrive_clustercenter_QRLSTC",
                        help="Output file for cluster centers (pickle)")

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

    start_time = time.time()
    res = saveclus_q(args.k, subtrajs, trajs, args.amount,
                     backend=be, shots=args.shots,
                     feature_mode=args.feature_mode, init_mode=args.init)
    end_time = time.time()
    print(f"QRLSTC (quantum) clustering completed in {end_time - start_time:.2f} seconds.")

    pickle.dump(res, open(args.centerfile, 'wb'), protocol=2)

    sse = compute_sse_q(res, backend=be, shots=args.shots,
                        feature_mode=args.feature_mode)
    print(f"QRLSTC (quantum) Goodness of fit (q-SSE): {sse:.6f}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted.")
        sys.exit(130)