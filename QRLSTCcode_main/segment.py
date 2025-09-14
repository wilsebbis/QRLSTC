
"""
Segment class for trajectory analysis.

Provides geometric distance computations between line segments, including perpendicular, parallel, and angle-based distances.
Used for trajectory similarity and clustering algorithms.
Depends on point_xy for point-to-line distance calculations.
"""

import math
from point_xy import _point2line_distance

class Segment(object):
    """
    Represents a line segment in 2D space, with optional trajectory ID.
    Provides methods for geometric distance calculations to other segments.
    """
    eps = 1e-12  # Numerical threshold for distance comparisons

    def __init__(self, start_point, end_point, traj_id=None):
        """
        Initialize a segment with start and end points.
        Args:
            start_point: Start point object.
            end_point: End point object.
            traj_id: Optional trajectory identifier.
        """
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id

    @property
    def length(self):
        """
        Returns the Euclidean length of the segment.
        """
        return self.end.distance(self.start)

    def perpendicular_distance(self, other):
        """
        Compute the perpendicular distance from another segment to this segment.
        Args:
            other: Another Segment object.
        Returns:
            Perpendicular distance value.
        """
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)

    def parallel_distance(self, other):
        """
        Compute the parallel distance between this segment and another segment.
        Args:
            other: Another Segment object.
        Returns:
            Parallel distance value.
        """
        l1 = min(self.start.distance(self._projection_point(other, typed='start')), self.end.distance(self._projection_point(other, typed='start')))
        l2 = min(self.end.distance(self._projection_point(other, typed='end')), self.start.distance(self._projection_point(other, typed='end')))
        return min(l1, l2)

    def angle_distance(self, other):
        """
        Compute the angle-based distance between this segment and another segment.
        Args:
            other: Another Segment object.
        Returns:
            Angle distance value.
        """
        self_vector = self.end - self.start
        other_vector = other.end - other.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)

        # Handle degenerate segments (zero length)
        if self_dist == 0:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist == 0:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())

        # If segments are identical
        if self.start == other.start and self.end == other.end:
            return 0
        cos_theta = self_vector.dot(other_vector) / (self_dist * other_dist)
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length

    def _projection_point(self, other, typed="e"):
        """
        Project a point from another segment onto this segment.
        Args:
            other: Another Segment object.
            typed: 'start' or 'end' to select which endpoint to project.
        Returns:
            Projected point on this segment.
        """
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        if math.pow(self.end.distance(self.start), 2) == 0: # start == end
            return self.start
        u = tmp.dot(self.end - self.start) / math.pow(self.end.distance(self.start), 2)
        return self.start + (self.end - self.start) * u

    def get_all_distance(self, seg):
        """
        Compute the sum of angle, parallel, and perpendicular distances to another segment.
        Args:
            seg: Another Segment object.
        Returns:
            Sum of all three distance measures.
        """
        res = self.angle_distance(seg) + self.parallel_distance(seg) + self.perpendicular_distance(seg)
        return res

def compare(segment_a, segment_b):
    """
    Compare two segments and return them ordered by length (longer first).
    Args:
        segment_a: First Segment object.
        segment_b: Second Segment object.
    Returns:
        Tuple (longer_segment, shorter_segment).
    """
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)

