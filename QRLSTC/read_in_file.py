import pandas as pd
import torch
from encoder import StateEncoder
from q_means import QMeans

# Read the txt file
with open('1.txt') as f:
    lines = f.readlines()

# If taxi_id is present, group by taxi
trajectories_dict = {}
for line in lines:
    parts = line.strip().split(',')
    if len(parts) == 4:
        taxi_id, timestamp, x, y = parts
    else:
        timestamp, x, y = parts
        taxi_id = 'default'
    if taxi_id not in trajectories_dict:
        trajectories_dict[taxi_id] = []
    trajectories_dict[taxi_id].append([timestamp, float(x), float(y)])

# Convert each trajectory to tensor of shape (num_points, 3)
num_points = 10  # set desired length
trajectories = []
for traj in trajectories_dict.values():
    # Convert timestamps to numeric
    triplets = []
    for t, x, y in traj:
        t_numeric = pd.to_datetime(t).timestamp()
        triplets.append([t_numeric, x, y])
    # Pad/truncate
    if len(triplets) >= num_points:
        triplets = triplets[:num_points]
    else:
        triplets += [[0, 0, 0]] * (num_points - len(triplets))
    trajectories.append(torch.tensor(triplets))

# Encode
encoder = StateEncoder(num_points=num_points, torch_device=torch.device('cpu'))
encoded_data = encoder(trajectories).cpu().numpy()

# Cluster
k = 3  # number of clusters
qmeans = QMeans(encoded_data, k)
qmeans.run()
qmeans.plot()