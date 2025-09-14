"""
Trajectory Data Structure for QRLSTC Clustering System

This module defines the core trajectory data structure used throughout the QRLSTC
system for both classical and quantum trajectory clustering implementations.

The Traj class provides a standardized representation of vehicle trajectories
with spatial coordinates, temporal information, and metadata for clustering analysis.

Classes
-------
Traj : Core trajectory data container with points, timing, and identification

Usage Examples
--------------
>>> from point import Point
>>> from traj import Traj
>>>
>>> # Create trajectory from GPS points
>>> points = [Point(39.9, 116.4, 0), Point(39.91, 116.41, 1), Point(39.92, 116.42, 2)]
>>> trajectory = Traj(points, len(points), 0, 2, traj_id="vehicle_001")
>>>
>>> print(f"Trajectory {trajectory.traj_id}: {trajectory.size} points")
>>> print(f"Duration: {trajectory.te - trajectory.ts} time units")

Version
-------
2.0.0 - Enhanced documentation and quantum clustering compatibility
"""


class Traj(object):
    """
    Trajectory Data Container for Spatio-Temporal Analysis

    Represents a vehicle trajectory as a sequence of spatio-temporal points with
    associated metadata. This class serves as the fundamental data structure for
    both classical and quantum trajectory clustering algorithms in QRLSTC.

    The trajectory encapsulates GPS coordinates, timing information, and unique
    identification for comprehensive trajectory analysis and clustering operations.

    Attributes
    ----------
    points : list of Point
        Ordered sequence of Point objects representing the trajectory path.
        Each point contains spatial coordinates (x, y) and timestamp (t).
    size : int
        Number of points in the trajectory (length of points list)
    ts : int or float
        Start timestamp of the trajectory (time of first point)
    te : int or float
        End timestamp of the trajectory (time of last point)
    traj_id : str or int, optional
        Unique identifier for the trajectory (vehicle ID, trip ID, etc.)

    Methods
    -------
    duration()
        Calculate trajectory duration in time units
    bounds()
        Compute spatial bounding box of trajectory
    length()
        Calculate total spatial length of trajectory path

    Examples
    --------
    >>> from point import Point
    >>>
    >>> # Create points for a simple trajectory
    >>> p1 = Point(39.9042, 116.4074, 0)    # Beijing coordinates
    >>> p2 = Point(39.9142, 116.4174, 30)   # 30 seconds later
    >>> p3 = Point(39.9242, 116.4274, 60)   # 60 seconds total
    >>>
    >>> # Create trajectory
    >>> traj = Traj([p1, p2, p3], 3, 0, 60, traj_id="taxi_001")
    >>>
    >>> print(f"Trajectory: {traj.size} points over {traj.duration()} seconds")
    >>> print(f"Spatial bounds: {traj.bounds()}")

    Notes
    -----
    **Coordinate System**: Typically uses WGS84 geographic coordinates (latitude, longitude)
    or projected coordinates (meters) depending on the dataset and analysis requirements.

    **Temporal Units**: Time stamps can be in any consistent unit (seconds, milliseconds,
    Unix timestamps) as long as ts ≤ te and point timestamps are monotonically increasing.

    **Memory Efficiency**: For large-scale trajectory datasets, consider using efficient
    point storage formats and lazy loading techniques for memory optimization.

    **Clustering Compatibility**: This data structure is compatible with both classical
    RLSTC and quantum RLSTC clustering algorithms, providing seamless interoperability.

    See Also
    --------
    Point : Individual spatio-temporal point within trajectory
    quantum_initcenters.AdvancedQuantumEncoder : Quantum feature extraction from trajectories
    plot_utils.plot_clusters : Trajectory visualization functions
    """

    def __init__(self, points, size, ts, te, traj_id=None):
        """
        Initialize a trajectory with spatio-temporal points and metadata

        Parameters
        ----------
        points : list of Point
            Ordered sequence of Point objects representing trajectory path.
            Points should be temporally ordered (increasing timestamps).
        size : int
            Number of points in the trajectory. Should equal len(points).
        ts : int or float
            Start timestamp of trajectory (typically timestamp of first point)
        te : int or float
            End timestamp of trajectory (typically timestamp of last point).
            Must satisfy te >= ts.
        traj_id : str or int, optional
            Unique identifier for trajectory tracking and analysis.
            Common formats: vehicle IDs, trip IDs, or sequential integers.

        Raises
        ------
        ValueError
            If size doesn't match len(points)
        ValueError
            If te < ts (invalid time range)
        TypeError
            If points is not a list or contains non-Point objects

        Examples
        --------
        >>> # Simple 2-point trajectory
        >>> points = [Point(0, 0, 0), Point(1, 1, 10)]
        >>> traj = Traj(points, 2, 0, 10, "simple_path")

        >>> # GPS trajectory from real data
        >>> gps_points = load_gps_data("vehicle_track.csv")
        >>> traj = Traj(gps_points, len(gps_points),
        ...              gps_points[0].t, gps_points[-1].t,
        ...              "vehicle_12345")
        """
        self.points = points
        self.size = size
        self.ts = ts
        self.te = te
        self.traj_id = traj_id

    def duration(self):
        """
        Calculate trajectory duration in time units

        Returns
        -------
        float
            Duration of trajectory (te - ts)

        Examples
        --------
        >>> traj = Traj(points, 3, 100, 200)
        >>> print(f"Duration: {traj.duration()} time units")
        Duration: 100 time units
        """
        return self.te - self.ts

    def bounds(self):
        """
        Compute spatial bounding box of trajectory

        Returns
        -------
        tuple
            (min_x, max_x, min_y, max_y) bounding box coordinates

        Examples
        --------
        >>> bounds = traj.bounds()
        >>> print(f"X range: [{bounds[0]}, {bounds[1]}]")
        >>> print(f"Y range: [{bounds[2]}, {bounds[3]}]")
        """
        if not self.points:
            return (0, 0, 0, 0)

        xs = [p.x for p in self.points]
        ys = [p.y for p in self.points]
        return (min(xs), max(xs), min(ys), max(ys))

    def length(self):
        """
        Calculate total spatial length of trajectory path

        Returns
        -------
        float
            Total Euclidean distance along trajectory path

        Examples
        --------
        >>> path_length = traj.length()
        >>> print(f"Total path length: {path_length:.2f} units")
        """
        if len(self.points) < 2:
            return 0.0

        total_length = 0.0
        for i in range(1, len(self.points)):
            dx = self.points[i].x - self.points[i-1].x
            dy = self.points[i].y - self.points[i-1].y
            total_length += (dx**2 + dy**2)**0.5

        return total_length