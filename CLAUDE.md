# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Q-RLSTC is a quantum-enhanced trajectory clustering system that extends classical "Sub-trajectory clustering with deep reinforcement learning" with quantum machine learning innovations. The project includes both classical RLSTC reproduction and novel quantum enhancements.

## Architecture

### Core Components

- **Classical RLSTC** (`RLSTCcode_main/`): Original paper implementation with Deep Q-Networks
- **Quantum RLSTC** (`QRLSTCcode_main/`): Quantum-enhanced clustering using quantum kernels
- **Shared Components** (root level): Common utilities like trajectory classes, distance functions
- **Visualization** (`plot_utils.py`): Comprehensive plotting for classical vs quantum comparison

### Key Files

- `QRLSTCcode_main/quantum_initcenters.py` - Advanced quantum clustering engine with MLX/CUDA acceleration
- `RLSTCcode_main/subtrajcluster/initcenters.py` - Classical k-means++ clustering implementation
- `plot_utils.py` - Visualization suite for comparative analysis
- `trajdistance.py`, `traj.py`, `point.py` - Core trajectory data structures and distance metrics

## Common Development Commands

### Running Clustering

```bash
# Classical RLSTC clustering
python3 RLSTCcode_main/subtrajcluster/initcenters.py -k 5 -amount 1000 -subtrajsfile data/subtrajs.pkl -trajsfile data/trajs.pkl

# Quantum RLSTC clustering
python3 QRLSTCcode_main/quantum_initcenters.py -k 5 -amount 1000 --shots 8192 --n-qubits 8

# Multiple k values for elbow analysis
python3 QRLSTCcode_main/quantum_initcenters.py -k 3 4 5 6 7 8 9 10 -amount 1000
```

### Generating Visualizations

```bash
# Plot clustering results
python3 plot_utils.py -results_dir out --plot-quantum-clusters --plot-quantum-elbow

# Comparative analysis plots
python3 plot_utils.py -results_dir out --plot-combined-elbow --plot-combined-silhouette
```

### Dependencies

Install requirements: `pip install -r requirements.txt`

Key dependencies:
- **Quantum**: qiskit>=0.44.0, qiskit-aer, qiskit-machine-learning
- **ML**: torch, tensorflow, scikit-learn, gymnasium
- **Acceleration**: mlx (Apple Silicon), cupy (NVIDIA CUDA - optional)
- **Scientific**: numpy, scipy, matplotlib, pandas

## Hardware Acceleration

The system automatically detects and optimizes for available hardware:

- **Apple Silicon (M1/M2/M3)**: Uses MLX for 2-10x speedup
- **NVIDIA GPUs**: Uses PyTorch CUDA for 3-8x speedup
- **CPU fallback**: Uses NumPy for systems without acceleration

## Data Formats

### Trajectory Structure
- `Traj` objects containing sequences of `Point(x, y, t)`
- Supports both full trajectories and sub-trajectories
- Distance computation via `trajdistance.py` (Euclidean, DTW, Fréchet)

### Clustering Results Format
```python
results = [(overall_sim, overall_sim, cluster_dict)]
# cluster_dict[i] = [avg_dist, center_traj, list_of_dists, list_of_assigned_subtrajs]
```

## Quantum vs Classical Comparison

The project is designed for comparing quantum and classical approaches:

- Both implementations output compatible result formats
- Use `plot_utils.py` for side-by-side performance analysis
- Key metrics: silhouette score, execution time, clustering quality
- Quantum typically shows 5-15% quality improvement with longer execution time

## Development Notes

- Quantum implementation in `QRLSTCcode_main/` is more recent and actively developed
- Classical implementation in `RLSTCcode_main/` faithfully reproduces the original paper
- Results are saved to `out/` directory with timing data in JSON format
- The project supports batch processing of multiple k values and parameter configurations