# distributions.py
#
# functions for dealing with distributions, spline fitting, etc.

import numpy as np
from scipy.stats import gaussian_kde


def get_kde_distribution(data, xlim=None, xsteps=100, x_dist=None, **kwargs):
    """
    TODO: docstring

    """

    data = np.array(data)

    # Perform Kernel Density Estimation (KDE) to create a smooth distribution
    kde = gaussian_kde(data, **kwargs)

    # Sample from kde distribution
    if x_dist is None:
        if xlim is None:
            xlim = (data.min(), data.max())

        # Generate evenly spaced x-values covering the range of the breath data
        x_dist = np.linspace(*xlim, xsteps)

    # return sampled distr
    y_dist = kde(x_dist)

    return kde, x_dist, y_dist
