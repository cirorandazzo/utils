# math.py
# 2025.03.20 CDR
#

import numpy as np

def add_angle_vectors_2d(magnitudes, angles):

    x = np.sum(np.multiply(magnitudes, np.cos(angles)))
    y = np.sum(np.multiply(magnitudes, np.sin(angles)))

    mag = np.sqrt(x**2 + y**2)
    ang = np.arctan(y/x)

    return mag, ang