# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Project Overview

QRLSTC (Quantum-Reinforcement Learning Sub-Trajectory Clustering) is a research project implementing two distinct approaches to trajectory clustering:

1. **Quantum-Inspired Clustering** (`QRLSTC/` directory) - Uses quantum circuits and Qiskit's swap test for similarity computation in k-means clustering
2. **Deep Reinforcement Learning Clustering** (`RLSTCcode_main/` directory) - Uses Deep Q-Networks to learn optimal sub-trajectory splitting strategies

## Key Development Commands

### Environment Setup
```bash
# Install dependencies (requires Python 3.6+ for RL, 3.8+ for quantum)
pip install -r requirements.txt

# Download datasets (for RL clustering)
# Extract to data/ directory: tar -xzvf data.tar.gz
```

### Quantum-Inspired Clustering (QRLSTC/)
```bash
# Run quantum clustering with swap test
python quantum_initcenters.py -subtrajsfile <subtrajs.pkl> -trajsfile <trajs.pkl> -k 10 -amount 1000 -centerfile <output.pkl>

# Run Q-means clustering directly
python q_means.py

# Test quantum distance computation
python q_distance.py
```

### Reinforcement Learning Clustering (QRLSTC/)
```bash
# Preprocess trajectory data
cd subtrajcluster/
python preprocessing.py

# Train RL agent for trajectory clustering
python rl_train.py

# Cross-validation training (5-fold by default)
python crosstrain.py

# Evaluate trained models
python rl_estimate.py -amount 500 -caltime 1

# Effectiveness experiments
python crossvalidate.py -calsse 1

# Test different clustering methods on RL-generated sub-trajectories
python rl_splitmethod.py -clustermethod AHC  # Agglomerative Hierarchical
python rl_splitmethod.py -clustermethod kmeans  # K-means
python rl_splitmethod.py -ep 0.01 -sample 60  # DBSCAN
```

### Running Individual Tests
```bash
# Test specific trajectory distance metrics
python trajdistance.py

# Test MDP environment
python MDP.py

# Test encoder functionality
python encoder.py
```

## Architecture Overview

### Quantum-Inspired Components (`QRLSTC/`)

**Core Quantum Pipeline:**
1. **Trajectory Encoding** (`encoder.py`) - Converts trajectory sequences to fixed-length normalized vectors suitable for quantum amplitude encoding
2. **Quantum Distance** (`q_distance.py`) - Implements SWAP test circuits using Qiskit to compute similarity between encoded trajectories
3. **Quantum K-means** (`q_means.py`) - K-means clustering using quantum swap test as distance oracle
4. **Initialization** (`quantum_initcenters.py`) - Quantum-inspired k-means++ center initialization

**Key Quantum Concepts:**
- **Amplitude Encoding**: Trajectories encoded as quantum states where amplitudes represent trajectory features
- **SWAP Test**: Quantum algorithm using ancillary qubits and controlled SWAP gates to estimate inner product between quantum states
- **Quantum Backend**: Uses Qiskit AerSimulator for quantum circuit execution (configurable shots parameter)

### Reinforcement Learning Components (`QRLSTC/`)

**Core RL Pipeline:**
1. **MDP Environment** (`MDP.py`) - Trajectory clustering formulated as Markov Decision Process
2. **RL Agent** (`rl_nn.py`) - Deep Q-Network implementation for learning splitting policies
3. **Training Loop** (`rl_train.py`) - Training RL agent with experience replay and target network updates
4. **State Representation**: 5-dimensional feature vector capturing trajectory similarity and position information
5. **Actions**: Binary decisions (0=continue trajectory, 1=split at current point)
6. **Rewards**: Based on clustering quality improvement after trajectory splits

### Shared Components

**Trajectory Representation:**
- `traj.py`: Trajectory class containing sequence of Point objects
- `point.py`: Spatio-temporal points (x, y, t coordinates)
- `segment.py`: Geometric operations between trajectory segments
- `trajdistance.py`: Classical distance metrics (Fréchet, DTW, IED)

**Data Processing:**
- `preprocessing.py`: Trajectory filtering, normalization, and sub-trajectory generation
- `cluster.py`: Classical clustering utilities and cluster quality metrics

## Important Implementation Details

### Quantum Circuit Construction
- Uses Qiskit with AerSimulator backend
- Quantum circuits transpiled for optimization before execution  
- SWAP test probability relates to trajectory similarity: P(0) = (1 + |⟨u|v⟩|²)/2
- Lower probabilities indicate higher quantum distances between trajectories

### RL Environment State Space
- **State**: [overall_similarity, current_similarity, scaled_overall_sim, relative_position, remaining_length]
- **Action Space**: Binary (continue vs. split trajectory)
- **Reward**: Clustering quality improvement (based on Inter-cluster/intra-cluster distance ratios)

### Data Flow
1. Raw trajectory data → preprocessing → Traj objects
2. **Quantum Path**: Traj → encoding → quantum circuits → similarity computation → clustering
3. **RL Path**: Traj → MDP states → RL agent actions → sub-trajectory splits → clustering

### Backend Configuration
- Quantum circuits use configurable shot counts (default 1024) for measurement statistics
- RL training uses TensorFlow 2.x with compatibility mode for TF 1.x APIs
- Both approaches support different trajectory distance metrics for comparison

## File Organization Patterns

- `*_initcenter*.py`: Clustering initialization and center selection algorithms
- `q_*.py`: Quantum-specific implementations
- `rl_*.py`: Reinforcement learning implementations  
- `*distance*.py`: Distance/similarity computation methods
- Core data structures: `traj.py`, `point.py`, `segment.py`
- Experiments and evaluation: `cross*.py`, `*estimate*.py`

## Development Notes

- Quantum implementations require careful state normalization for amplitude encoding
- RL training requires adequate episode sampling for convergence  
- Both approaches use classical trajectory preprocessing but differ in clustering methodology
- Cross-validation and hyperparameter tuning essential for both quantum shots and RL hyperparameters
- Quantum circuits can be resource-intensive; consider shot count vs. accuracy tradeoffs