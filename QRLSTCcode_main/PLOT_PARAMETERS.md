# Plot Parameters Guide for QRLSTC Trajectory Clustering

This document explains all parameters displayed in the information boxes of plot_utils.py plots and provides guidance for optimal configuration.

## 📊 Information Box Parameters

### **Core Clustering Parameters**

#### **Number of Clusters (k)**
- **Description**: Number of cluster centers to find in the dataset
- **Range**: Typically 3-15 for trajectory data
- **Impact**: Higher k = more granular clustering but potential overfitting
- **Example**: `k=10` means the algorithm will find 10 distinct trajectory clusters

#### **Amount (Dataset Size)**
- **Description**: Number of trajectories used from the full dataset
- **Range**: 10-50000+ depending on computational resources
- **Impact**: More trajectories = better clustering quality but longer execution time
- **Example**: `Amount: 1000` means 1000 trajectories were analyzed

### **Quantum-Specific Parameters**

#### **Shots**
- **Description**: Number of quantum circuit executions per measurement
- **Range**: 1024-16384 (powers of 2 preferred)
- **Impact**: Higher shots = more accurate quantum measurements but slower execution
- **Quantum Theory**: Each quantum circuit is probabilistic; shots average out quantum noise
- **Example**: `Shots: 8192` means each quantum distance calculation runs 8192 times

#### **n_qubits (Quantum Encoding Qubits)**
- **Description**: Number of qubits used to encode trajectory features
- **Range**: 4-12 qubits (limited by quantum simulator capacity)
- **Impact**: More qubits = richer quantum feature space but exponentially more complex
- **Quantum Theory**: 2^n quantum states possible with n qubits
- **Example**: `n_qubits: 8` provides 2^8 = 256 possible quantum basis states

### **Visualization Parameters**

#### **Alpha (Point Transparency)**
- **Description**: Transparency level for trajectory points in scatter plots
- **Range**: 0.0 (invisible) to 1.0 (opaque)
- **Impact**: Lower alpha helps visualize overlapping dense trajectory regions
- **Example**: `Alpha: 0.3` makes points 30% opaque, allowing overlap visualization

#### **Center Alpha (Center Transparency)**
- **Description**: Transparency level for cluster center trajectories
- **Range**: 0.0 (invisible) to 1.0 (opaque)
- **Impact**: Usually higher than point alpha to emphasize cluster centers
- **Example**: `Center Alpha: 0.8` makes cluster centers highly visible

#### **Sample Rate**
- **Description**: Plotting density - shows every Nth point of trajectories
- **Range**: 1 (all points) to 100+ (sparse sampling)
- **Impact**: Higher sample rate = faster plotting, lower rate = more detail
- **Memory**: Lower sample rates use more memory and plotting time
- **Example**: `Sample Rate: 40` plots every 40th point of each trajectory

### **Performance & Hardware Parameters**

#### **Hardware Acceleration**
- **Description**: Type of computational acceleration being used
- **Options**:
  - `MLX` - Apple Silicon Metal Performance Shaders
  - `CUDA` - NVIDIA GPU acceleration
  - `CPU` - Standard CPU computation
- **Impact**: MLX/CUDA can provide 2-10x speedup over CPU
- **Example**: `Hardware: MLX` indicates Apple Silicon acceleration

#### **Execution Time**
- **Description**: Total time taken for clustering computation
- **Units**: Seconds
- **Factors**: Depends on amount, shots, qubits, and hardware acceleration
- **Example**: `Time: 245.3s` for the clustering computation

## 🎯 Optimal Parameter Configurations

### **Maximum Accuracy & Precision (Full T-Drive Dataset)**

For analyzing the complete T-Drive dataset with maximum fidelity:

```bash
python3 quantum_initcenters.py \
    -k 5 6 7 8 9 10 11 12 \
    -amount 50000 \
    --shots 16384 \
    --n-qubits 10 \
    --output-dir out_full

python3 plot_utils.py -results_dir out_full \
    --alpha 0.1 \
    --center-alpha 1.0 \
    --sample-rate 1 \
    --plot-quantum-clusters \
    --plot-quantum-elbow \
    --plot-quantum-timing
```

**Rationale:**
- **amount=50000**: Uses full T-Drive dataset (17,000+ vehicles, 15M+ points)
- **shots=16384**: Maximum precision for quantum measurements
- **n_qubits=10**: Rich 1024-dimensional quantum feature space
- **sample_rate=1**: Shows every trajectory point for complete visualization
- **alpha=0.1**: Low transparency to handle dense overlapping trajectories
- **k=5-12**: Comprehensive range to find optimal cluster count

**⚠️ Warning**: This configuration requires 8-24 hours on Apple Silicon, 2-7 days on CPU

### **Balanced Performance Configuration**

For good accuracy with reasonable execution time:

```bash
python3 quantum_initcenters.py \
    -k 3 4 5 6 7 8 9 10 \
    -amount 5000 \
    --shots 8192 \
    --n-qubits 8 \
    --output-dir out_balanced

python3 plot_utils.py -results_dir out_balanced \
    --alpha 0.3 \
    --center-alpha 0.8 \
    --sample-rate 20 \
    --plot-quantum-clusters \
    --plot-quantum-elbow \
    --plot-quantum-timing
```

**Execution Time**: 2-6 hours depending on hardware

## 🍎🍊 Fair Quantum vs Classical Comparison (Low Shot Count)

### **Strategy 1: Equivalent Computational Budget**

Match total computational operations rather than shot count:

```bash
# Quantum with low shots but high feature dimensionality
python3 quantum_initcenters.py \
    -k 3 4 5 6 7 8 9 10 \
    -amount 2000 \
    --shots 2048 \
    --n-qubits 10

# Classical with equivalent computational complexity
python3 RLSTCcode_main/subtrajcluster/initcenters.py \
    -k 3 4 5 6 7 8 9 10 \
    -amount 2000 \
    --iterations 50 \
    --feature_dims 10
```

**Rationale**: 2048 shots × 2^10 quantum states ≈ 2M operations vs classical 50 iterations × 2000 trajectories ≈ 100K operations

### **Strategy 2: Time-Matched Comparison**

Run both algorithms for the same wall-clock time:

```bash
# Set time limit using timeout command
timeout 1800s python3 quantum_initcenters.py \
    -k 5 6 7 8 \
    -amount 1500 \
    --shots 1024 \
    --n-qubits 8

timeout 1800s python3 RLSTCcode_main/subtrajcluster/initcenters.py \
    -k 5 6 7 8 \
    -amount 5000 \
    --max_time 1800
```

**Rationale**: Both algorithms get exactly 30 minutes to achieve best results

### **Strategy 3: Quality-First with Noise Modeling**

Add noise to classical algorithm to match quantum uncertainty:

```bash
# Quantum with realistic quantum noise
python3 quantum_initcenters.py \
    -k 3 4 5 6 7 8 \
    -amount 1000 \
    --shots 1024 \
    --n-qubits 8 \
    --enable-noise-simulation

# Classical with added measurement noise
python3 RLSTCcode_main/subtrajcluster/initcenters.py \
    -k 3 4 5 6 7 8 \
    -amount 1000 \
    --add-measurement-noise 0.1 \
    --distance-uncertainty 0.05
```

**Rationale**: Both algorithms work with similar levels of measurement uncertainty

### **Strategy 4: Hardware-Normalized Comparison**

Compare best achievable results on equivalent hardware:

```bash
# Quantum optimized for available hardware
python3 quantum_initcenters.py \
    -k 3 4 5 6 7 8 \
    -amount 3000 \
    --shots 2048 \
    --n-qubits 8 \
    --enable-hardware-optimization

# Classical optimized for same hardware
python3 RLSTCcode_main/subtrajcluster/initcenters.py \
    -k 3 4 5 6 7 8 \
    -amount 10000 \
    --enable-simd \
    --enable-parallel-processing \
    --optimize-for-hardware
```

**Rationale**: Each algorithm uses its optimal configuration for the available hardware

## 📈 Comparison Metrics

When comparing quantum vs classical with low shot counts, focus on these metrics:

### **Quality Metrics**
- **Silhouette Score**: Measure of cluster separation quality
- **Within-Cluster Sum of Squares (WCSS)**: Compactness of clusters
- **Adjusted Rand Index**: Similarity to ground truth (if available)
- **Calinski-Harabasz Score**: Ratio of within/between cluster variance

### **Efficiency Metrics**
- **Operations per Second**: Computational throughput
- **Memory Usage Peak**: Maximum RAM consumption
- **Energy Consumption**: Battery/power usage (for mobile applications)
- **Convergence Rate**: Speed of algorithm convergence

### **Robustness Metrics**
- **Noise Sensitivity**: Performance degradation with noisy data
- **Parameter Stability**: Consistency across different parameter settings
- **Reproducibility**: Variance across multiple runs
- **Scalability**: Performance scaling with dataset size

## 🎛️ Recommended Comparison Configuration

For a fair, comprehensive quantum vs classical comparison:

```bash
# Run quantum clustering
python3 quantum_initcenters.py \
    -k 3 5 7 10 \
    -amount 2000 \
    --shots 2048 \
    --n-qubits 8 \
    --seed 42 \
    --output-dir comparison_quantum

# Run classical clustering
python3 RLSTCcode_main/subtrajcluster/initcenters.py \
    -k 3 5 7 10 \
    -amount 2000 \
    --seed 42 \
    --output-dir comparison_classical

# Generate comparative plots
python3 plot_utils.py -results_dir . \
    --plot-combined-elbow \
    --plot-combined-silhouette \
    --plot-combined-timing \
    --method-name "Quantum vs Classical Comparison"
```

This provides a balanced comparison while keeping quantum shot counts reasonable for practical execution times.