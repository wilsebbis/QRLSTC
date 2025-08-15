# Adapted from /dina/encoding.py from https://github.com/const-sambird/dina/tree/quantum
# to work for trajectory data

import torch
from torch import nn
import math

class StateEncoder(nn.Module):
    '''
    Encodes a batch of trajectories for quantum encoding. Each trajectory is a sequence of (timestamp, x, y) triplets.
    The encoder pads or truncates each trajectory to a fixed number of points, then flattens it into a 1D vector.
    '''
    def __init__(self, num_points, torch_device):
        super(StateEncoder, self).__init__()
        self.num_points = num_points  # max number of (timestamp, x, y) triplets per trajectory
        self.input_dim = 3  # (timestamp, x, y)
        self.output_size = num_points * self.input_dim
        self.torch_device = torch_device

    def forward(self, batch):
        return self._encode_trajectories(batch)

    def _encode_trajectories(self, batch):
        # batch: list of tensors or a tensor of shape (batch_size, variable_num_points, 3)
        batch_size = len(batch)
        output = torch.zeros((batch_size, self.output_size), device=self.torch_device)
        for i, traj in enumerate(batch):
            # traj: tensor of shape (num_points_i, 3)
            num_points = traj.shape[0]
            # Pad or truncate to self.num_points
            if num_points >= self.num_points:
                padded = traj[:self.num_points, :]
            else:
                pad = torch.zeros((self.num_points - num_points, 3), device=self.torch_device)
                padded = torch.cat([traj, pad], dim=0)
            # Flatten to 1D
            output[i] = padded.flatten()
        return output
    
class AngleEncoder(nn.Module):
    '''
    Transforms an encoded state tensor into one that can
    be used for angle encoding into qubits.

    Essentially a mapping x -> pi / x, which represents
    the angle to be rotated around the x-axis of the Bloch sphere.

    We want to encode into [0, pi] radians and x is in the
    range (0, 1] so this works out nicely
    '''
    def __init__(self):
        super(AngleEncoder, self).__init__()
    
    def forward(self, x):
        return torch.where(x > 0, math.pi / x, 0)

class AmplitudeEncoder(nn.Module):
    '''
    Encodes input vectors for amplitude encoding by L2-normalizing each vector in the batch.
    The output vectors are suitable for quantum amplitude encoding (sum of squares = 1).
    '''
    def __init__(self):
        super(AmplitudeEncoder, self).__init__()

    def forward(self, x):
        # x: (batch_size, vector_dim)
        # L2 normalize each vector
        norm = torch.norm(x, p=2, dim=1, keepdim=True)
        # Avoid division by zero
        norm = torch.where(norm == 0, torch.ones_like(norm), norm)
        return x / norm

class StateDecoder(nn.Module):
    def __init__(self, num_candidates, num_replicas):
        super(StateDecoder, self).__init__()
        self.num_candidates = num_candidates
        self.num_replicas = num_replicas
    
    def forward(self, x):
        return torch.vmap(self._decode_state)(x)

    def _decode_state(self, state):
        '''
        The inverse of StateEncoder._encode_state -- takes a 1-d vector where all state variables are
        encoded as a binary representation of the replicas each candidate they are located in,
        and transforms this into a (num_replicas, num_candidates) state matrix that the environment
        is able to use.
        '''
        result = torch.zeros((self.num_candidates, self.num_replicas))

        for idx, element in enumerate(state):
            result[idx,:] = [1 if element & (1 << i) else 0 for i in range(self.num_replicas)]

        result = torch.where(result > 0, result, 0)

        return result.T