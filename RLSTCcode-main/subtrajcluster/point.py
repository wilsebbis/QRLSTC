
"""
point.py

Defines the Point class for representing a spatial-temporal point in a trajectory.
Includes methods for distance calculation and equality checking.
"""

import math

class Point(object):
    """
    Represents a point with x, y coordinates and time t.
    """
    def __init__(self, x, y, t):
        """
        Initialize a Point instance.
        Args:
            x: float, longitude or x-coordinate
            y: float, latitude or y-coordinate
            t: float, timestamp
        """
        self.x = x
        self.y = y
        self.t = t

    def distance(self, other):
        """
        Compute Euclidean distance to another Point (ignores time).
        Args:
            other: Point instance
        Returns:
            float, Euclidean distance
        """
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def equal(self, other):
        """
        Check if this point is equal to another (x, y, t all match).
        Args:
            other: Point instance
        Returns:
            bool, True if equal
        """
        if self.x == other.x and self.y == other.y and self.t == other.t:
            return True

