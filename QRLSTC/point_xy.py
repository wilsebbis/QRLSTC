"""
2D Point class and utility functions for geometric calculations.

Provides basic arithmetic, distance, and dot product operations for points in 2D space.
Includes a function for computing the distance from a point to a line segment.
"""

import math
import numpy as np

class Point_xy(object):
    """
    Represents a point in 2D space with x and y coordinates.
    Supports arithmetic operations, distance, and dot product.
    """
    def __init__(self, x, y):
        """
        Initialize a 2D point.
        Args:
            x: X coordinate.
            y: Y coordinate.
        """
        self.x = x
        self.y = y

    def get_point(self):
        """
        Return the (x, y) coordinates as a tuple.
        """
        return self.x, self.y

    def __add__(self, other):
        """
        Add two Point_xy objects.
        Args:
            other: Another Point_xy object.
        Returns:
            Point_xy: The vector sum as a new Point_xy.
        """
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point' type.")
        _add_x = self.x + other.x
        _add_y = self.y + other.y
        return Point_xy(_add_x, _add_y)

    def __sub__(self, other):
        """
        Subtract another Point_xy from this one.
        Args:
            other: Another Point_xy object.
        Returns:
            Point_xy: The vector difference as a new Point_xy.
        """
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point' type.")
        _sub_x = self.x - other.x
        _sub_y = self.y - other.y
        return Point_xy(_sub_x, _sub_y)

    def __mul__(self, x):
        """
        Multiply this point by a scalar (float).
        Args:
            x: Scalar value (float).
        Returns:
            Point_xy: Scaled point.
        """
        if isinstance(x, float):
            return Point_xy(self.x * x, self.y * x)
        else:
            raise TypeError("The other object must 'float' type.")

    def __truediv__(self, x):
        """
        Divide this point by a scalar (float).
        Args:
            x: Scalar value (float).
        Returns:
            Point_xy: Scaled point.
        """
        if isinstance(x, float):
            return Point_xy(self.x / x, self.y / x)
        else:
            raise TypeError("The other object must 'float' type.")

    def distance(self, other):
        """
        Compute Euclidean distance to another Point_xy.
        Args:
            other: Another Point_xy object.
        Returns:
            Euclidean distance (float).
        """
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def dot(self, other):
        """
        Compute dot product with another Point_xy.
        Args:
            other: Another Point_xy object.
        Returns:
            Dot product (float).
        """
        return self.x * other.x + self.y * other.y

    def as_array(self):
        """
        Return the point as a NumPy array [x, y].
        """
        return np.array((self.x, self.y))

def _point2line_distance(point, start, end):
    """
    Compute the distance from a point to a line segment in 2D.
    Args:
        point: NumPy array [x, y] for the point.
        start: NumPy array [x, y] for the start of the segment.
        end: NumPy array [x, y] for the end of the segment.
    Returns:
        Distance (float) from the point to the line segment.
    """
    if np.all(np.equal(start, end)):
        # Degenerate segment: treat as a point
        return np.linalg.norm(point - start)
    # Use cross product to compute area of parallelogram, then divide by base length
    return np.divide(np.abs(np.linalg.norm(np.cross(end - start, start - point))),
                     np.linalg.norm(end - start))