# From https://github.com/llianga/RLSTCcode

"""
segment.py

Defines the Segment class for representing trajectory segments and provides
methods for calculating various distances between segments (perpendicular, parallel, angle).
Also includes a utility function for comparing segment lengths.
"""

import math
from point_xy import _point2line_distance


class Segment(object):
    eps = 1e-12

    def __init__(self, start_point, end_point, traj_id=None):
        """
        Initialize a segment with start and end points and optional trajectory ID.
        """
        self.start = start_point
        self.end = end_point
        self.traj_id = traj_id

    @property
    def length(self):
        """
        Returns the length of the segment.
        """
        return self.end.distance(self.start)


    def perpendicular_distance(self, other):
        """
        Calculates the perpendicular distance between this segment and another.
        Args:
            other: Segment to compare against
        Returns:
            Perpendicular distance (float)
        """
        l1 = other.start.distance(self._projection_point(other, typed="start"))
        l2 = other.end.distance(self._projection_point(other, typed="end"))
        # If both projections are very close, return 0
        if l1 < self.eps and l2 < self.eps:
            return 0
        else:
            # Weighted average of squared distances
            return (math.pow(l1, 2) + math.pow(l2, 2)) / (l1 + l2)


    def parallel_distance(self, other):
        """
        Calculates the parallel distance between this segment and another.
        Args:
            other: Segment to compare against
        Returns:
            Parallel distance (float)
        """
        l1 = min(self.start.distance(self._projection_point(other, typed='start')), self.end.distance(self._projection_point(other, typed='start')))
        l2 = min(self.end.distance(self._projection_point(other, typed='end')), self.start.distance(self._projection_point(other, typed='end')))
        return min(l1, l2)


    def angle_distance(self, other):
        """
        Calculates the angle-based distance between this segment and another.
        Args:
            other: Segment to compare against
        Returns:
            Angle-based distance (float)
        """
        self_vector = self.end - self.start
        other_vector = other.end - other.start
        self_dist, other_dist = self.end.distance(self.start), other.end.distance(other.start)

        # Handle degenerate cases where one segment is a point
        if self_dist == 0:
            return _point2line_distance(self.start.as_array(), other.start.as_array(), other.end.as_array())
        elif other_dist == 0:
            return _point2line_distance(other.start.as_array(), self.start.as_array(), self.end.as_array())

        # If segments are identical, angle distance is zero
        if self.start == other.start and self.end == other.end:
            return 0
        cos_theta = self_vector.dot(other_vector) / (self_dist * other_dist)
        # Clamp cosine to valid range and compute angle distance
        if cos_theta > self.eps:
            if cos_theta >= 1:
                cos_theta = 1.0
            return other.length * math.sqrt(1 - math.pow(cos_theta, 2))
        else:
            return other.length


    def _projection_point(self, other, typed="e"):
        """
        Projects a point from another segment onto this segment.
        Args:
            other: Segment to project from
            typed: 'start' or 'end' to select which endpoint to project
        Returns:
            Projected point (Point_xy)
        """
        if typed == 's' or typed == 'start':
            tmp = other.start - self.start
        else:
            tmp = other.end - self.start
        # If segment is degenerate, return start
        if math.pow(self.end.distance(self.start), 2) == 0: #start=end
            return self.start
        u = tmp.dot(self.end-self.start) / math.pow(self.end.distance(self.start), 2)
        return self.start + (self.end-self.start) * u


    def get_all_distance(self, seg):  
        """
        Returns the sum of angle, parallel, and perpendicular distances to another segment.
        Args:
            seg: Segment to compare against
        Returns:
            Sum of all distances (float)
        """
        res = self.angle_distance(seg) + self.parallel_distance(seg) + self.perpendicular_distance(seg)
        return res


def compare(segment_a, segment_b):
    """
    Compares two segments by length and returns them in descending order.
    Args:
        segment_a: First segment
        segment_b: Second segment
    Returns:
        Tuple of segments (longer, shorter)
    """
    return (segment_a, segment_b) if segment_a.length > segment_b.length else (segment_b, segment_a)

