# utils.breath.segment
#
# Utils related to breath segmentation. The functions you're probably after are, in order:
#
# (1) get_kde_threshold, which returns information on the location of the trough between inspiratory and expiratory amplitude modes.
# (2) segment_breaths, which takes a trace, (optionally) filters, and returns exp/insp crossings.
#


import numpy as np

from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde


def segment_breaths(
    breathing_unfilt,
    fs,
    do_filter=True,
    b=None,
    a=None,
    threshold=lambda x: np.median(x),
    threshold_exp=None,
    threshold_insp=None,
):
    """
    TODO: accept numerical value as threshold instead of function.

    Segments a raw breathing signal into inspiration and expiration phases.

    The function filters the input signal and identifies the onsets of inspiration and expiration
    phases based on a threshold function applied to the filtered signal. Alternatively, threshold
    function can be set separately for inspiration and expiration. The function returns the frame
    indices of the detected transitions (where 0 is first audio frame).

    Args:
        breathing_unfilt (numpy.ndarray): A 1D array representing the raw (unfiltered) breathing signal.
        fs (float): The sampling frequency (Hz) of the breathing signal.
        do_filter (boolean): Whether to filter before thresholding. Default: True (filters inputed data)
        b, a (numpy.ndarray, optional): Numerator & denominator coefficients of a filter. If either or both are None, uses a 50Hz lowpass butterworth filter. Default is None.
        threshold (function, optional): A threshold function used to determine the boundary for inspiration/expiration. Default is `lambda x: np.median(x)`.
        threshold_exp (function, optional): A custom threshold function for expiration phases. If None, the `threshold` function is used.
        threshold_insp (function, optional): A custom threshold function for inspiration phases. If None, the `threshold` function is used.

    Returns:
        tuple: A tuple containing:
            - exps (numpy.ndarray): The indices where expiration phases start.
            - insps (numpy.ndarray): The indices where inspiration phases start.

    Example:
        >>> breathing_signal = np.random.randn(1000)  # Example raw breathing signal
        >>> fs = 1000  # Sampling frequency of 1 kHz
        >>> exps, insps = segment_breaths(breathing_signal, fs)
        >>> print("Expiration onsets:", exps)
        >>> print("Inspiration onsets:", insps)
    """
    from scipy.signal import butter, filtfilt

    if do_filter:
        # Default: 50Hz low-pass filter
        if b is None or a is None:
            b, a = butter(N=2, Wn=50, btype="low", fs=fs)
        breathing_filt = filtfilt(b, a, breathing_unfilt)
    else:
        breathing_filt = breathing_unfilt

    # Set default thresholds
    if threshold_exp is None:
        threshold_exp = threshold
    if threshold_insp is None:
        threshold_insp = threshold

    # Exps
    br_thr = breathing_filt >= threshold_exp(breathing_filt)
    previous = np.append([1], br_thr[:-1])
    exps = np.flatnonzero(br_thr & ~previous)

    # Insps
    br_thr = breathing_filt <= threshold_insp(breathing_filt)
    previous = np.append([1], br_thr[:-1])
    insps = np.flatnonzero(br_thr & ~previous)

    return (exps, insps)


# ===== KDE ===== #


def fit_breath_distribution(breath, **kde_kwargs):
    """
    For 1d data, constructs a smoothed distribution using a Kernel Density Estimate (KDE), extracts the inspiratory and expiratory peaks, then takes zero point threshold as the trough between those peaks.

    The function identifies the two most prominent peaks in the pressure amplitude distribution of a breath waveform, and the minimum (trough) value between those peaks. Additionally, it returns a spline fit of the pressure distribution.

    Parameters:
    ----------
    breath : array-like
        The breath waveform data (pressure amplitude distribution).


    **kde_kwargs: kwargs for computing sampled kde distribution (see get_kde_distribution)

    Returns:
    -------
    x_dist : ndarray
        The x-values of the KDE distribution, representing the pressure values over the
        given range of the breath waveform.

    dist_kde : ndarray
        The smoothed distribution (KDE) of the pressure amplitude over `x_dist`.

    trough_ii : int
        The index of the minimum value (trough) between the two most prominent peaks in the
        pressure distribution.

    amplitude_ii : list of int
        The indices of the two most prominent peaks in the pressure distribution (inspiratory
        and expiratory peaks).
    """

    kde, x_dist, dist_kde = get_kde_distribution(breath, **kde_kwargs)

    # Identify the indices of all peaks in the KDE distribution
    peak_indices = find_peaks(dist_kde)[0]

    # Compute the prominence of each peak
    prominences = peak_prominences(dist_kde, peak_indices)[0]

    # Select the two most prominent peaks
    amplitude_ii = sorted(peak_indices[np.argsort(prominences)][-2:])

    # Find the trough (minimum) between the two peaks
    trough_ii = amplitude_ii[0] + np.argmin(dist_kde[np.arange(*amplitude_ii)])

    return x_dist, dist_kde, trough_ii, amplitude_ii


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


def get_kde_threshold(breath, **fit_kwargs):
    """
    Wrapper of fit_breath_distribution which directly returns spline-fit threshold given a breath waveform. E.g., useful for passing as `centering` function into segment_breaths. 
    """

    x_dist, dist_kde, trough_ii, amplitude_ii = fit_breath_distribution(
        breath, **fit_kwargs
    )

    threshold = x_dist[trough_ii]

    return threshold
