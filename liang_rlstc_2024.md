# Sub-trajectory clustering with deep reinforcement learning

**Authors:** Anqi Liang, Bin Yao, Bo Wang, Yinpei Liu, Zhida Chen, Jiong Xie, Feifei Li

**Published:** The VLDB Journal (2024) 33:685–702

## Abstract

Sub-trajectory clustering is a fundamental problem in many trajectory applications. Existing approaches usually divide the clustering procedure into two phases: segmenting trajectories into sub-trajectories and then clustering these sub-trajectories. However, researchers need to develop complex human-crafted segmentation rules for specific applications, making the clustering results sensitive to the segmentation rules and lacking in generality. To solve this problem, we propose a novel algorithm using the clustering results to guide the segmentation, which is based on reinforcement learning (RL). The novelty is that the segmentation and clustering components cooperate closely and improve each other continuously to yield better clustering results. To devise our RL-based algorithm, we model the procedure of trajectory segmentation as a Markov decision process (MDP). We apply Deep-Q-Network (DQN) learning to train an RL model for the segmentation and achieve excellent clustering results. Experimental results on real datasets demonstrate the superior performance of the proposed RL-based approach over state-of-the-art methods.

**Keywords:** Sub-trajectory clustering · Spatio-temporal similarity · Reinforcement learning · Deep learning

## 1. Introduction

With the proliferation of GPS-equipped devices and location-based services, a vast amount of trajectory data is generated at an unprecedented rate. Sub-trajectory clustering has attracted much attention as it segments trajectories into sub-trajectories based on certain principles and then clusters them with respect to a given distance metric.

### Key Problem

Clustering whole trajectories poses challenges in discovering local similarities among trajectories since they vary in length and time range, leading to loss of valuable information. Sub-trajectory clustering helps analyze correlations or common patterns among portions of different trajectories in regions of interest.

### Existing Solutions

Current solutions follow the "first-segment-then-cluster" paradigm and can be classified into two categories:

1. **Independent procedures**: Segment trajectories according to spatial features, then cluster sub-trajectories
2. **Quality-aware segmentation**: Consider clustering quality during segmentation

However, these methods rely on complicated hand-crafted segmentation rules tailored to specific applications, making clustering results sensitive to these rules and limiting generality.

### Our Solution

We propose an efficient and effective reinforcement learning (RL)-based framework that leverages clustering quality to guide segmentation. The segmentation and clustering components cooperate closely, mutually enhancing each other to achieve superior clustering results.

## 2. Methodology

### MDP Formulation

We model trajectory segmentation as a Markov Decision Process (MDP) with:

- **States**: Defined using features that consider both trajectories and clustering quality
- **Actions**: Binary decision (segment at current point or not)
- **Rewards**: Based on clustering quality improvement
- **Transitions**: State changes resulting from segmentation actions

### Key Components

1. **State Representation**:
   - ODs and ODn values (segmentation vs non-segmentation outcomes)
   - Length features (Lb, Lf)
   - Expert knowledge integration (ODb)

2. **Reward Function**:
   - Immediate reward = difference in Overall Distance (OD) values
   - Accumulative reward maximization aligns with OD minimization

3. **DQN Learning**:
   - Main network for Q-value estimation
   - Target network for loss computation
   - Experience replay for training stability

### RLSTC Algorithm

The complete algorithm iteratively:
1. Applies learned policy for trajectory segmentation
2. Assigns sub-trajectories to nearest clusters
3. Updates cluster centers
4. Continues until convergence

## 3. Experimental Results

### Datasets
- **Geolife**: 17,070 GPS trajectories from 182 users
- **T-Drive**: 9,937 taxi trajectories from Beijing

### Performance
- **Effectiveness**: Outperformed baselines by 36% (T-Drive) and 39% (Geolife) in OD metric
- **Efficiency**: 20% faster than competing methods
- **Generality**: Adaptable to various distance measurements and clustering algorithms

### Key Findings
1. RL-based segmentation consistently outperforms hand-crafted rules
2. Expert knowledge integration (ODb) significantly improves performance
3. Algorithm converges within few iterations
4. Works well with different trajectory similarity measurements

## 4. Technical Contributions

1. **Novel RL Framework**: First RL-based solution for sub-trajectory clustering
2. **Creative MDP Modeling**: Defines states and rewards considering clustering characteristics
3. **Data-Driven Segmentation**: Eliminates need for complex hand-crafted rules
4. **Efficiency Optimizations**: Multiple techniques to enhance computational performance

## 5. Applications

The approach is valuable for:
- Hurricane forecasting (analyzing coastal trajectory patterns)
- Transportation planning
- GPS trace analysis
- Location-based services
- Traffic monitoring

## 6. Limitations and Future Work

- Computational cost of trajectory distance remains a bottleneck
- Training requires appropriate k value selection
- Performance degrades when actual k differs significantly from training k

**Future directions:**
- Extending to multi-modal trajectories
- Accelerating training process
- Grid-based indexing for efficiency improvements