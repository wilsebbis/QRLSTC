"""
Trajectory distance and similarity computation utilities.

This module provides functions and classes for computing various trajectory distances and similarities,
including Frechet, Dynamic Time Warping (DTW), and custom integrated error distances (IED).
It is designed for use with trajectory data represented as lists of Point objects, and depends on
segment, point, traj, and point_xy modules for geometric and temporal calculations.

Main features:
- Model complexity (MDL) for trajectory segments
- Sub-trajectory extraction by time interval
- Integrated error distance (IED) between trajectories
- Frechet and DTW distance classes
- Wasserstein-like distance for trajectory segments
"""

import math
from segment import Segment, compare
from point import Point
from traj import Traj
import numpy as np
from point_xy import Point_xy

eps = 1e-12   # Threshold for segment length; if length < eps then l_h=0

def makemid(x1, t1, x2, t2, t):
    """
    Linearly interpolate the value at time t between (x1, t1) and (x2, t2).
    Used for reconstructing intermediate points in a trajectory.
    Args:
        x1: Value at time t1.
        t1: Start time.
        x2: Value at time t2.
        t2: End time.
        t: Target time for interpolation.
    Returns:
        Interpolated value at time t.
    """
    if (t2 - t1) * (x2 - x1) == 0:
        return x1 + (t - t1)
    return x1 + (t - t1) / (t2 - t1) * (x2 - x1)

def traj_mdl_comp(points, start_index, curr_index, typed):
    """
    Compute the model complexity (MDL) for a trajectory segment.
    Args:
        points: List of Point objects representing the trajectory.
        start_index: Start index of the segment.
        curr_index: End index of the segment.
        typed: 'simp' for simplified, 'orign' for original computation.
    Returns:
        h: Model complexity value for the segment.
    """
    # Create a segment from start to current index
    seg = Segment(points[start_index], points[curr_index])
    h = 0  # Model complexity value
    lh = 0 # Accumulated error for 'simp' type
    # For 'simp' type, use segment length and time difference for initial complexity
    if typed == 'simp':
        if seg.length > eps:
            h = 0.5 * math.log2(seg.length) + 0.5 * (abs(points[start_index].t - points[curr_index].t))
    t1, t2 = points[start_index].t, points[curr_index].t
    x1, x2 = points[start_index].x, points[curr_index].x
    y1, y2 = points[start_index].y, points[curr_index].y

    # Iterate through points in the segment
    for i in range(start_index, curr_index, 1):
        if typed == 'simp':
            # For 'simp', compute error to interpolated point
            t = points[i].t
            new_x = makemid(x1, t1, x2, t2, t)
            new_y = makemid(y1, t1, y2, t2, t)
            new_p = Point(new_x, new_y, t)
            lh += points[i].distance(new_p)
        elif typed == 'orign':
            # For 'orign', use distance and time difference to next point
            d = 0.5 * (points[i].distance(points[i + 1])) + 0.5 * (abs(points[i].t - points[i + 1].t))
            if d > eps:
                h += math.log2(d)
    # Finalize complexity for 'simp' type
    if typed == 'simp':
        if lh > eps:
            h += math.log2(lh)
        return h
    else:
        return h

def timedTraj(points, ts, te):
    """
    Extract a sub-trajectory between time ts and te.
    Args:
        points: List of Point objects.
        ts: Start time.
        te: End time.
    Returns:
        Traj object representing the sub-trajectory.
    """
    # Return None if interval is degenerate or out of bounds
    if ts == te:
        return
    if ts > points[-1].t or te < points[0].t:
        return
    s_i = 0
    e_i = len(points) - 1
    new_points = []
    # Find start index for ts
    while points[s_i].t < ts:
        s_i += 1
    # Find end index for te
    while points[e_i].t > te:
        e_i -= 1
    # Interpolate and add start point if needed
    if s_i != 0 and points[s_i].t != ts:
        x = makemid(points[s_i - 1].x, points[s_i - 1].t, points[s_i].x, points[s_i].t, ts)
        y = makemid(points[s_i - 1].y, points[s_i - 1].t, points[s_i].y, points[s_i].t, ts)
        new_p = Point(x, y, ts)
        new_points.append(new_p)
    # Add all points in interval
    for i in range(s_i, e_i + 1):
        new_points.append(points[i])
    # Interpolate and add end point if needed
    if e_i != len(points) - 1 and points[e_i].t != te:
        x = makemid(points[e_i].x, points[e_i].t, points[e_i + 1].x, points[e_i + 1].t, te)
        y = makemid(points[e_i].y, points[e_i].t, points[e_i + 1].y, points[e_i + 1].t, te)
        new_p = Point(x, y, te)
        new_points.append(new_p)
    # Build new trajectory object
    new_ts = new_points[0].t
    new_te = new_points[-1].t
    new_size = len(new_points)
    new_traj = Traj(new_points, new_size, new_ts, new_te)
    return new_traj
 
def line2lineIDE(p1s, p1e, p2s, p2e):
    """
    Compute the integrated distance error (IDE) between two line segments.
    Args:
        p1s, p1e: Start and end Point objects for segment 1.
        p2s, p2e: Start and end Point objects for segment 2.
    Returns:
        d: Integrated distance error value.
    """
    # Compute distance between start and end points of two segments
    d1 = p1s.distance(p2s)
    d2 = p1e.distance(p2e)
    # Average the distances and multiply by time interval
    d = 0.5 * (d1 + d2) * (p1e.t - p1s.t)
    return d

def getstaticIED(points, x, y, t1, t2):
    """
    Compute the integrated error distance (IED) for a static point over a time interval.
    Args:
        points: List of Point objects.
        x, y: Coordinates of the static point.
        t1, t2: Start and end times (t1 < t2).
    Returns:
        sum: Total integrated error distance.
    """
    # Clamp time interval to trajectory bounds
    s_t = max(points[0].t, t1)
    e_t = min(points[-1].t, t2)
    sum = 0
    if s_t >= e_t:
        return 1e10
    # Create static points at (x, y) for each time
    ps, pe = Point(x, y, 0), Point(x, y, 0)
    timedpoints = timedTraj(points, s_t, e_t)
    # Accumulate error over each segment
    for i in range(timedpoints.size - 1):
        ps.t = timedpoints.points[i].t
        pe.t = timedpoints.points[i + 1].t
        pd = line2lineIDE(timedpoints.points[i], timedpoints.points[i + 1], ps, pe)
        sum += pd
    return sum
 
def traj2trajIED(traj_points1, traj_points2):
    """
    Compute the integrated error distance (IED) between two trajectories.
    Args:
        traj_points1: List of Point objects for trajectory 1.
        traj_points2: List of Point objects for trajectory 2.
    Returns:
        sum: Total integrated error distance between the two trajectories.
    """
    # Get time intervals for both trajectories
    t1s, t1e = traj_points1[0].t, traj_points1[-1].t
    t2s, t2e = traj_points2[0].t, traj_points2[-1].t
    # If intervals do not overlap, return large error
    if t1s >= t2e or t1e <= t2s:
        return 1e10
    sum = 0
    # Extract overlapping sub-trajectories
    timedtraj = timedTraj(traj_points2, t1s, t1e)
    cut1 = timedtraj.ts
    cut2 = timedtraj.te

    commontraj = timedTraj(traj_points1, cut1, cut2)
    # Handle non-overlapping intervals at start/end
    if t1s < cut1:
        pd = getstaticIED(traj_points1, timedtraj.points[0].x, timedtraj.points[0].y, t1s, cut1)
        sum += pd
    if t2s < t1s:
        pd = getstaticIED(traj_points2, traj_points1[0].x, traj_points1[0].y, t2s, t1s)
        sum += pd
    if t1e > cut2:
        pd = getstaticIED(traj_points1, timedtraj.points[-1].x, timedtraj.points[-1].y, cut2, t1e)
        sum += pd
    if t1e < t2e:
        pd = getstaticIED(traj_points2, traj_points1[-1].x, traj_points1[-1].y, t1e, t2e)
        sum += pd

    # Accumulate error over common interval
    if commontraj is not None and commontraj.size != 0:
        newtime, lasttime = commontraj.ts, commontraj.ts
        iter1, iter2 = 0, 0
        lastp1, lastp2 = commontraj.points[0], timedtraj.points[0]

        while lasttime != timedtraj.te:
            # Synchronize time steps between trajectories
            if timedtraj.points[iter2 + 1].t == commontraj.points[iter1 + 1].t:
                newtime = timedtraj.points[iter2 + 1].t
                newp1 = commontraj.points[iter1 + 1]
                newp2 = timedtraj.points[iter2 + 1]
                iter1 += 1
                iter2 += 1
            elif timedtraj.points[iter2 + 1].t < commontraj.points[iter1 + 1].t:
                t = timedtraj.points[iter2 + 1].t
                x = makemid(commontraj.points[iter1].x, commontraj.points[iter1].t, commontraj.points[iter1 + 1].x, commontraj.points[iter1 + 1].t, t)
                y = makemid(commontraj.points[iter1].y, commontraj.points[iter1].t, commontraj.points[iter1 + 1].y, commontraj.points[iter1 + 1].t, t)
                newp1 = Point(x, y, t)
                newp2 = timedtraj.points[iter2 + 1]
                iter2 += 1
            else:
                t = commontraj.points[iter1 + 1].t
                x = makemid(timedtraj.points[iter2].x, timedtraj.points[iter2].t, timedtraj.points[iter2 + 1].x,
                            timedtraj.points[iter2 + 1].t, t)
                y = makemid(timedtraj.points[iter2].y, timedtraj.points[iter2].t, timedtraj.points[iter2 + 1].y,
                            timedtraj.points[iter2 + 1].t, t)
                newp2 = Point(x, y, t)
                newp1 = commontraj.points[iter1 + 1]
                iter1 += 1
            lasttime = newtime
            pd = line2lineIDE(lastp1, newp1, lastp2, newp2)
            sum += pd
            lastp1 = newp1
            lastp2 = newp2
    return sum

class Distance:
    """
    Frechet distance computation between two trajectories.
    """
    def __init__(self, N, M):
        """
        Create a Frechet distance matrix for two trajectories of lengths N and M.
        The matrix is used to store intermediate results for dynamic programming.
        """
        """
        Initialize Frechet distance matrix.
        Args:
            N: Length of trajectory C.
            M: Length of trajectory Q.
        """
        self.D0 = np.zeros((N + 1, M + 1))
        self.flag = np.zeros((N, M))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D = self.D0[1:, 1:]

    def FRECHET(self, traj_C, traj_Q, skip=[]):
        """
        Compute the discrete Frechet distance between two trajectories.
        Args:
            traj_C: List of Point objects for trajectory C.
            traj_Q: List of Point objects for trajectory Q.
            skip: Optional indices to skip (unused).
        Returns:
            Frechet distance value.
        """
        n = len(traj_C)
        m = len(traj_Q)
        for i in range(n):
            for j in range(m):
                if self.flag[i, j] == 0:
                    cost = traj_C[i].distance(traj_Q[j])
                    self.D[i, j] = max(cost, min(self.D0[i, j], self.D0[i, j + 1], self.D0[i + 1, j]))
                    self.flag[i, j] = 1
        return self.D[n - 1, m - 1]
    

class Dtwdistance:
    """
    Dynamic Time Warping (DTW) distance computation between two trajectories.
    """
    def __init__(self, N, M):
        """
        Create a DTW distance matrix for two trajectories of lengths N and M.
        The matrix is used to store intermediate results for dynamic programming.
        """
        """
        Initialize DTW distance matrix.
        Args:
            N: Length of trajectory C.
            M: Length of trajectory Q.
        """
        self.D0 = np.zeros((N + 1, M + 1))
        self.flag = np.zeros((N, M))
        self.D0[0, 1:] = np.inf
        self.D0[1:, 0] = np.inf
        self.D = self.D0[1:, 1:]

    def DTW(self, traj_C, traj_Q, skip=[]):
        """
        Compute the DTW distance between two trajectories.
        Args:
            traj_C: List of Point objects for trajectory C.
            traj_Q: List of Point objects for trajectory Q.
            skip: Optional indices to skip (unused).
        Returns:
            DTW distance value.
        """
        n = len(traj_C)
        m = len(traj_Q)
        for i in range(n):
            for j in range(m):
                if self.flag[i, j] == 0:
                    temp_res = []
                    temp_x = traj_C[i].x - traj_Q[j].x
                    temp_y = traj_C[i].y - traj_Q[j].y
                    temp_res.append(temp_x)
                    temp_res.append(temp_y)
                    cost = np.linalg.norm(temp_res)
                    self.D[i, j] = cost + min(self.D0[i, j], self.D0[i, j + 1], self.D0[i + 1, j])
                    self.flag[i, j] = 1
        return self.D[n - 1, m - 1]
    
def wd_dist(t1, t2):
    """
    Compute the Wasserstein-like distance between two trajectory segments.
    Args:
        t1: List of Point objects for trajectory 1.
        t2: List of Point objects for trajectory 2.
    Returns:
        dist: Wasserstein-like distance value.
    """
    # Convert endpoints to Point_xy for segment comparison
    sp1, ep1 = Point_xy(t1[0].x, t1[0].y), Point_xy(t1[-1].x, t1[-1].y)
    sp2, ep2 = Point_xy(t2[0].x, t2[0].y), Point_xy(t2[-1].x, t2[-1].y)
    seg1 = Segment(sp1, ep1)
    seg2 = Segment(sp2, ep2)
    # Compare segments and compute Wasserstein-like distance
    seg_long, seg_short = compare(seg1, seg2)
    dist = seg_long.get_all_distance(seg_short)
    return dist
