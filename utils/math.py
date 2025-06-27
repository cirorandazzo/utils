# math.py
# 2025.03.20 CDR
#

import numpy as np


def add_angle_vectors_2d(magnitudes, angles):

    x = np.sum(np.multiply(magnitudes, np.cos(angles)))
    y = np.sum(np.multiply(magnitudes, np.sin(angles)))

    mag = np.sqrt(x**2 + y**2)
    ang = np.arctan(y / x)

    return mag, ang


def is_point_in_range(point, ranges):
    """
    Check if a point lies within the specified ranges in n-dimensional space.

    Parameters:
        point (list or tuple): The n-dimensional point (e.g., [1, 5]).
        ranges (list of [lower, upper] pairs): Each pair corresponds to the allowed range for a dimension.

    Returns:
        bool: True if the point lies within the given range in all dimensions, False otherwise.
    """
    if len(point) != len(ranges):
        raise ValueError("Point and ranges must have the same number of dimensions")

    return all(l <= coord <= r for coord, (l, r) in zip(point, ranges))
