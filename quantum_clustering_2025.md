# Subtrajectory Clustering with Machine Learning on Quantum Computers

**Authors:** Wil Bishop, Eleazar Leal, Le Gruenwald
**Affiliations:** University of Minnesota Duluth, The University of Oklahoma
**Conference:** ACM SIGSPATIAL 2025, Minneapolis, MN

## Abstract

Subtrajectory clustering is vital for real-world applications such as traffic bottleneck detection, public transportation optimization, and play pattern discovery in sports analytics. This problem is NP-hard and computationally intensive for large-scale applications, and many existing classical implementations struggle with scalability and generalizability. While machine learning-based approaches allow for greater generalizability than existing rule-based methods, they still struggle with scalability. Quantum computing offers a compelling promise of superior scalability as hardware matures and more fault-tolerant systems become available. This paper discusses the drawbacks of current subtrajectory clustering approaches, the challenges associated with solving the problem using quantum computing, and a vision for converting a state-of-the-art classical subtrajectory clustering algorithm to a hybrid quantum-classical version.

## 1. Introduction

With pervasive GPS-enabled devices and location-based services, unprecedented volumes of trajectory data are continuously generated. Subtrajectory clustering segments trajectories into smaller, meaningful subtrajectories and groups similar ones together. Unlike full-trajectory clustering, this technique enables discovery of local patterns that may be obscured due to variations in trajectory length, duration, or sampling rate.

### Key Challenges
- **Computational Intensity**: Subtrajectory clustering is NP-hard
- **Scalability Issues**: Classical algorithms like TRACLUS rely on rule-based heuristic methods that scale poorly
- **Parameter Sensitivity**: Methods are sensitive to input parameters that are computationally intensive to calibrate

### Quantum Solution Promise
- **Quantum parallelism, superposition, and entanglement** offer unique scalability advantages
- Quantum methods for combinatorial optimization have shown up to **~6,561× speedup** over classical algorithms
- Hybrid quantum-classical approaches can selectively leverage quantum advantages while retaining classical components where appropriate

## 2. Related Work

### 2.1 Classical Subtrajectory Clustering
- **TRACLUS**: Partition-and-group paradigm with hand-crafted heuristics followed by DBSCAN clustering
- **RLSTC**: Deep Reinforcement Learning approach using DQN to learn segmentation policies directly from data
- **Limitations**:
  - Rule-based methods are parameter-sensitive and scale poorly
  - Distance calculations remain a computational bottleneck
  - DRL approaches are still computationally intensive on classical hardware

### 2.2 Quantum Computing Background
- **Qubits**: Can exist in superposition of 0 and 1 states
- **Quantum Gates**: Basic operations that manipulate qubits
- **Key Properties**: Superposition, entanglement, quantum parallelism
- **Current Limitations**: NISQ devices have limited qubits, short coherence times, high gate error rates

## 3. Research Challenges

### 3.1 Common Challenges (Classical and Quantum)
1. **Data Volume**: Massive datasets from high-sampling-rate sensors
2. **Spatial-Temporal Context**: Ordered data with complex relationships
3. **Size Variability**: Comparing objects of different trajectory lengths
4. **Noise and Data Cleaning**: GPS inaccuracies compounded by quantum noise
5. **Streaming Data**: Real-time processing requirements

### 3.2 Quantum-Specific Challenges
1. **Data Loading and Encoding**: Efficiently encoding classical data into quantum states
   - Angle encoding maps data values to qubit rotation angles
   - Preserving spatial-temporal relationships is crucial
2. **Optimal Circuit Design**: Balancing circuit complexity vs. efficiency
   - Avoid barren plateaus from improper circuit depth
   - Trade-off between representation capability and noise accumulation
3. **Hybrid Algorithm Design**: Optimal division of labor between classical and quantum components
4. **Hardware Constraints**: NISQ limitations (limited qubits, short coherence, high error rates)
5. **Decoding Quantum Outputs**: Translating probabilistic measurements to meaningful classical information

## 4. Proposed Quantum Algorithm: Q-RLSTC

### 4.1 Classical RLSTC Overview

**Key Components:**
1. **Preprocessing**: Trajectory simplification using Minimum Description Length (MDL)
2. **Initial Clustering**: k-means++ for cluster center initialization
3. **MDP Learning**:
   - **States**: 5 features including ODs, ODn, ODb, Lb, Lf
   - **Actions**: Binary segmentation decision at each trajectory point
   - **Rewards**: Based on Overall Distance (OD) improvement
4. **Training**: DQN with ε-greedy strategy and experience replay
5. **Convergence**: Iterative segmentation and clustering until stable cluster centers

### 4.2 Q-RLSTC: Quantum Enhancement Vision

**Hybrid Architecture Components:**

1. **Classical Preprocessing** (retained due to NISQ limitations)
   - MDL trajectory simplification
   - Angle encoding for quantum representation

2. **Quantum Initial Clustering**
   - **q-means++**: Quantum k-means with polylogarithmic scaling vs. linear classical scaling
   - Potential for exponential speedup with large datasets
   - Alternative: Quantum DBSCAN for density-based clustering

3. **Quantum Policy Learning**
   - **Variational Quantum Circuits (VQCs)** for DQN replacement
   - **VQ-DQN**: O(n) parameter complexity vs. classical O(n²)
   - Noise-robust action selection mechanisms

4. **Quantum Components:**
   - **Distance Calculation**: Swap test for quantum state similarity (linear vs. exponential scaling)
   - **Action Selection**: Grover's algorithm replacing ε-greedy (O(√N) vs. O(N) complexity)
   - **Optimization**: Quantum Gradient Descent (O(1) vs. O(N) complexity)

5. **Classical Segmentation and Clustering** (retained for NISQ compatibility)

## 5. Technical Advantages

### Complexity Improvements
- **q-means**: Polylogarithmic scaling vs. linear classical scaling
- **VQ-DQN**: O(n) vs. O(n²) parameter complexity
- **Grover Search**: O(√N) vs. O(N) search complexity
- **Quantum Gradient Descent**: O(1) vs. O(N) parameter complexity
- **Distance Calculation**: Linear vs. exponential scaling with quantum swap test

### Noise Resilience
- Quantum action selection mechanisms don't require exact expectation values
- Only need to identify qubit with largest expectation value
- Robust performance on current NISQ devices

## 6. Implementation Considerations

### Near-term Feasibility
- **Hybrid Approach**: Only quantum-advantaged components implemented in quantum
- **Cost Management**: Quantum computing costs ($96/minute for IBM access) justify selective use
- **Hardware Constraints**: NISQ limitations require careful component selection

### Performance Optimization Strategies
1. **Shot Optimization**: Minimize quantum measurements needed
2. **Circuit Depth Management**: Balance expressivity with coherence times
3. **Ansatz Selection**: Hardware-efficient variational circuits
4. **Error Mitigation**: Techniques to handle current noise levels

## 7. Future Research Directions

### Implementation Goals
- Formalize and implement using Qiskit framework
- Empirical performance evaluation vs. classical counterpart
- Benchmarking across different dataset scales

### Potential Improvements
1. **Advanced Encoding Schemes**: More efficient quantum data representations
2. **Error Correction**: Integration with emerging fault-tolerant systems
3. **Hardware Evolution**: Adaptation to larger, more stable quantum systems
4. **Algorithm Optimization**: Fine-tuning hybrid classical-quantum workflows

### Application Extensions
- Traffic bottleneck detection with real-time quantum processing
- Sports analytics with enhanced pattern discovery
- Urban planning with scalable trajectory analysis
- Public transportation optimization

## 8. Conclusions

Q-RLSTC represents a promising hybrid quantum-classical framework that could significantly improve the scalability of subtrajectory clustering while maintaining the generalizability advantages of machine learning approaches. The selective integration of quantum algorithms for distance estimation, policy learning, and clustering optimization, while retaining classical components where quantum advantages are marginal, offers a practical path toward quantum-enhanced spatial-temporal analysis.

**Key Contributions:**
- First proposed quantum-enhanced subtrajectory clustering algorithm
- Systematic analysis of quantum vs. classical trade-offs
- Practical hybrid architecture addressing current NISQ limitations
- Vision for scalable trajectory analytics with quantum computing

The maturing quantum computing landscape suggests that such hybrid approaches may soon provide practical advantages for large-scale spatial data mining applications.