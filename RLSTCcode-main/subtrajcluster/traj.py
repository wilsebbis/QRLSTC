# From https://github.com/llianga/RLSTCcode

"""
traj.py

Defines the Traj class for representing a trajectory as a sequence of points.
Stores metadata such as trajectory size, start/end times, and optional ID.
"""

class Traj(object):
    """
    Traj class represents a trajectory as a sequence of points with metadata.
    Attributes:
        points: list of Point objects representing the trajectory
        size: int, number of points in the trajectory
        ts: start time
        te: end time
        traj_id: optional trajectory identifier
    """
    def __init__(self, points, size, ts, te, traj_id=None):
        """
        Initialize a trajectory object.
        Args:
            points: list of Point objects
            size: int, number of points
            ts: start time
            te: end time
            traj_id: optional trajectory identifier
        """
        self.points = points  # List of Point objects
        self.size = size      # Number of points in trajectory
        self.ts = ts          # Start time
        self.te = te          # End time
        self.traj_id = traj_id  # Optional trajectory ID


