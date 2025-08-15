# From https://github.com/llianga/RLSTCcode


"""
point_xy.py

Defines the Point_xy class for representing 2D spatial points and related operations.
Includes arithmetic, distance, dot product, and conversion utilities.
Also provides a function for computing point-to-line distance.
"""

import math
import numpy as np

class Point_xy(object):
    """
    Represents a 2D spatial point with x and y coordinates.
    Provides arithmetic, distance, and conversion operations.
    """
    def __init__(self, x, y):
        """
        Initialize a Point_xy instance.
        Args:
            x: float, x-coordinate
            y: float, y-coordinate
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
        """
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point' type.")
        _add_x = self.x + other.x
        _add_y = self.y + other.y
        return Point_xy(_add_x, _add_y)

    def __sub__(self, other):
        """
        Subtract two Point_xy objects.
        """
        if not isinstance(other, Point_xy):
            raise TypeError("The other type is not 'Point' type.")
        _sub_x = self.x - other.x
        _sub_y = self.y - other.y
        return Point_xy(_sub_x, _sub_y)

    def __mul__(self, x):
        """
        Multiply point by a scalar (float).
        """
        if isinstance(x, float):
            return Point_xy(self.x * x, self.y * x)
        else:
            raise TypeError("The other object must 'float' type.")

    def __truediv__(self, x):
        """
        Divide point by a scalar (float).
        """
        if isinstance(x, float):
            return Point_xy(self.x / x, self.y / x)
        else:
            raise TypeError("The other object must 'float' type.")

    def distance(self, other):
        """
        Compute Euclidean distance to another Point_xy.
        """
        return math.sqrt(math.pow(self.x - other.x, 2) + math.pow(self.y - other.y, 2))

    def dot(self, other):
        """
        Compute dot product with another Point_xy.
        """
        return self.x * other.x + self.y * other.y

    def as_array(self):
        """
        Convert point to numpy array.
        """
        return np.array((self.x, self.y))

def _point2line_distance(point, start, end):
    """
    Compute perpendicular distance from a point to a line segment.
    Args:
        point: np.ndarray, point coordinates
        start: np.ndarray, start of line segment
        end: np.ndarray, end of line segment
    Returns:
        float, distance from point to line
    """
    if np.all(np.equal(start, end)):
        return np.linalg.norm(point - start)
    return np.divide(np.abs(np.linalg.norm(np.cross(end - start, start - point))),
                     np.linalg.norm(end - start))