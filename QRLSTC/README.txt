QRLSTC: Quantum-Inspired Trajectory Clustering
==============================================

This project implements quantum-inspired clustering algorithms for trajectory data, leveraging quantum circuits and Qiskit to compute similarity between trajectories. It includes classical and quantum-inspired k-means clustering, trajectory encoding, and utilities for trajectory analysis.

Project Structure
-----------------

- `qmeans.py`: Quantum-inspired k-means clustering using quantum swap test for trajectory similarity.
- `initcenters.py`: Classical k-means++ initialization and clustering for trajectory data.
- `quantum_initcenters.py`: Quantum-inspired version of clustering using QMeans and quantum circuits.
- `encoder.py`: Encodes trajectory data for quantum amplitude encoding.
- `trajdistance.py`: Functions and classes for computing trajectory distances and similarities (Frechet, DTW, IED, etc.).
- `segment.py`: Segment class for geometric computations between line segments (perpendicular, parallel, angle-based distances).
- `point.py`: Point class for representing spatio-temporal points (x, y, t) and basic operations.
- `point_xy.py`: 2D Point class for geometric calculations, arithmetic, and point-to-line distance utilities.
- `traj.py`: Traj class for representing trajectories as sequences of points, with metadata.
- `read_in_file.py`: Example for reading and encoding trajectory data from text files.

Quantum Concepts & Vocabulary
----------------------------

**Quantum Circuit**: A sequence of quantum gates applied to qubits, representing a quantum algorithm. In this project, quantum circuits are used to implement the swap test for measuring similarity between encoded trajectories.

**Qubit**: The basic unit of quantum information, analogous to a classical bit but can exist in superpositions of 0 and 1.

**Ancilla (Ancillary Qubit)**: An extra qubit used to assist in quantum operations, such as the swap test. It is not part of the main data but helps with computation or measurement.

**Shots**: The number of times a quantum circuit is executed to collect measurement statistics. Quantum measurements are probabilistic, so running the circuit multiple times helps estimate probabilities more accurately.

**Ansatz**: A proposed form or structure for a quantum circuit or wavefunction, often used in variational algorithms. It’s a template circuit with parameters that can be optimized.

**Swap Test**: A quantum algorithm to estimate the similarity (inner product) between two quantum states. It uses an ancillary qubit and controlled swap gates. In this project, the swap test is used to measure the similarity between encoded trajectory vectors.

**Backend**: The quantum simulator or hardware used to run circuits (e.g., Qiskit’s AerSimulator).

**Transpile**: The process of converting a quantum circuit into a form optimized for a specific backend or hardware.

How Quantum Clustering Works Here
---------------------------------
1. **Encoding**: Trajectories are encoded into fixed-length vectors using `StateEncoder` and normalized for quantum amplitude encoding using `AmplitudeEncoder`.
2. **Initialization**: Cluster centers are initialized using a diversity-maximizing strategy (k-means++), but with quantum swap test distances.
3. **Clustering**: The quantum-inspired k-means algorithm (`QMeans`) assigns each trajectory to the nearest center using quantum swap test similarity, running quantum circuits for each comparison.
4. **Iteration**: Cluster centers are updated and the process repeats until convergence.

Usage Example
-------------
1. Prepare trajectory data and encode using `StateEncoder`.
2. Run quantum clustering:
   ```bash
   python quantum_initcenters.py -subtrajsfile <subtrajs.pkl> -trajsfile <trajs.pkl> -k 10 -amount 1000 -centerfile <output.pkl>
   ```
3. The clustering uses Qiskit to run quantum swap test circuits for each trajectory-center pair.

Dependencies
------------
- Python 3.8+
- Qiskit
- torch
- numpy
- matplotlib

References
----------
- Qiskit documentation: https://qiskit.org/documentation/
- Quantum swap test: https://qiskit.org/textbook/ch-algorithms/swap-test.html
- Quantum k-means: https://arxiv.org/abs/1812.03584

Further Reading: Quantum Vocabulary
----------------------------------
- **Quantum Gate**: Basic operation on qubits (e.g., Hadamard, CNOT, SWAP).
- **Measurement**: The process of observing a quantum state, which collapses it to a classical value.
- **Superposition**: A qubit can be in a combination of 0 and 1 states simultaneously.
- **Entanglement**: A quantum phenomenon where qubits become correlated in ways not possible classically.

How Trajectories Are Translated Into Quantum Form and Used in Clustering
------------------------------------------------------------------------

1. **Trajectory Encoding**
   - Each trajectory is a sequence of (timestamp, x, y) triplets.
   - The `StateEncoder` pads or truncates each trajectory to a fixed number of points, then flattens it into a 1D vector.
   - The `AmplitudeEncoder` normalizes these vectors so their L2 norm is 1, making them suitable for quantum amplitude encoding (the sum of squares of amplitudes equals 1).

2. **Quantum Circuit Construction for Swap Test**
   - For each pair of vectors (trajectory and cluster center), a quantum circuit is built:
     - Registers: One ancillary qubit, and two sets of qubits for the two vectors (each set has log2(vector_length) qubits).
     - Gates used:
       - **Initialize**: The two vectors are loaded into quantum registers as quantum states using amplitude encoding.
       - **Hadamard Gate**: Applied to the ancillary qubit before and after the controlled swap.
       - **Controlled SWAP (CSWAP) Gates**: Each qubit in the two registers is swapped, controlled by the ancillary qubit.
       - **Measurement**: The ancillary qubit is measured to estimate the overlap (similarity) between the two states.

3. **Swap Test Output and Clustering**
   - The probability of measuring '0' on the ancillary qubit is related to the inner product (similarity) between the two quantum states.
   - These probabilities are used as quantum distances for clustering (lower probability means higher distance).
   - The quantum k-means algorithm assigns each trajectory to the nearest cluster center based on these quantum distances.

4. **Cluster Centers and Results**
   - Cluster centers are updated as the mean of the assigned trajectories (in classical vector space), then re-encoded for quantum comparison in the next iteration.
   - The final cluster assignments and centers are output as classical vectors (not quantum amplitudes), so you can interpret and use them as regular trajectory features.
   - The quantum circuits are only used for similarity computation; the clustering results are classical.

**Summary:**
- Trajectories are encoded as normalized vectors, loaded into quantum states via amplitude encoding.
- Quantum swap test circuits (using Hadamard and CSWAP gates) compute similarity between trajectories and cluster centers.
- Clustering is performed using these quantum similarities, but the final output (cluster assignments and centers) is classical and interpretable.

Algorithm Pipeline: Step-by-Step Breakdown
------------------------------------------
This section details the full workflow, listing each file and function in the order they are used for quantum-inspired trajectory clustering:

1. **Data Preprocessing (`preprocessing.py`)**
   - `processtrajs(trajs)`: Filters raw trajectory data by geographic bounds.
   - `processlength(trajs, max_length, min_length)`: Truncates or pads trajectories to desired length.
   - `split_traj(traj, max_length, min_length)`: Splits long trajectories into subtrajectories.
   - `normloctrajs(trajs)`: Normalizes longitude/latitude values.
   - `normtimetrajs(trajs)`: Normalizes timestamps.
   - `convert2traj(trajdata)`: Converts lists of points to `Traj` objects (see `traj.py`).
   - `getsimptrajs(trajs)`: Simplifies trajectories for clustering.

2. **Trajectory and Point Representation**
   - `traj.py`: Defines the `Traj` class (sequence of `Point` objects, metadata).
   - `point.py`: Defines the `Point` class (x, y, t coordinates).
   - `segment.py`, `point_xy.py`: Used for geometric calculations and trajectory similarity.

3. **Distance and Similarity Computation (`trajdistance.py`)**
   - Functions for trajectory-to-trajectory distances: `traj2trajIED`, `Frechet`, `DTW`, etc.
   - Used in classical clustering and for analysis.

4. **Encoding for Quantum Clustering (`encoder.py`)**
   - `StateEncoder`: Pads/truncates and flattens each trajectory to a fixed-length vector.
   - `AmplitudeEncoder`: L2-normalizes vectors for quantum amplitude encoding.

5. **Quantum-Inspired Clustering (`quantum_initcenters.py`)**
   - `quantum_saveclus(k, subtrajs, trajs, amount, backend, shots)`: Main entry point; encodes trajectories and runs quantum clustering.
   - `quantum_getbaseclus(encoded_data, k, backend, shots)`: Runs quantum k-means clustering using quantum swap test distances.
   - `quantum_initialize_centers(encoded_data, k, backend, shots)`: Initializes cluster centers using quantum swap test for diversity.

6. **Quantum Swap Test and Clustering (`qmeans.py`)**
   - `distance_centroids_parallel(point, centroids, backend, shots)`: Computes quantum swap test similarity between a trajectory and cluster centers.
   - `QMeans`: Quantum-inspired k-means clustering class; assigns clusters and updates centers using quantum distances.

7. **Saving and Output**
   - Results (cluster assignments, centers) are saved using `pickle` in `quantum_initcenters.py`.

**Summary of Flow:**
Raw data → preprocessing (`preprocessing.py`) → trajectory objects (`traj.py`, `point.py`) → normalization → encoding (`encoder.py`) → quantum clustering (`quantum_initcenters.py`, `qmeans.py`) → results saved.
