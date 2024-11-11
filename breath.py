# utils/breath.py


# normalize to pre-stim
# roughly recenter by subtracting median
# roughly segment (w/ insp/exp threshold)
# recenter based on mean(pre-stim breaths)

import numpy as np


def segment_breaths(
    breathing_unfilt,
    fs,
    b=None,
    a=None,
    threshold=lambda x: np.median(x),
    threshold_exp=None,
    threshold_insp=None,
):
    """
    Segments a raw breathing signal into inspiration and expiration phases.

    The function filters the input signal and identifies the onsets of inspiration and expiration
    phases based on a threshold function applied to the filtered signal. The threshold function can
    be customized for inspiration and expiration. The function returns the indices of the detected
    transitions.

    Args:
        breathing_unfilt (numpy.ndarray): A 1D array representing the raw (unfiltered) breathing signal.
        fs (float): The sampling frequency (Hz) of the breathing signal.
        b (numpy.ndarray, optional): Numerator coefficients of the low-pass filter. Default is None,
                                      which results in a filter with a cutoff of 50 Hz.
        a (numpy.ndarray, optional): Denominator coefficients of the low-pass filter. Default is None,
                                      which results in a filter with a cutoff of 50 Hz.
        threshold (function, optional): A threshold function used to determine the boundary for inspiration/expiration.
                                         Default is `lambda x: np.median(x)`.
        threshold_exp (function, optional): A custom threshold function for expiration phases. If None,
                                             the `threshold` function is used.
        threshold_insp (function, optional): A custom threshold function for inspiration phases. If None,
                                              the `threshold` function is used.

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

    # Default: 20Hz low-pass filter
    if b is None or a is None:
        b, a = butter(N=2, Wn=50, btype="low", fs=fs)
    breathing_filt = filtfilt(b, a, breathing_unfilt)

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


def make_notmat_vars(
    exps,
    insps,
    last_offset,
    exp_label="e",
    insp_label="i",
):
    """
    Generates the onset, offset, and label information for inspiration and expiration phases.

    Given the onsets of inspiration and expiration, this function combines them into a sorted list of
    events, and generates corresponding offset times. It also assigns a label ("e" for expiration, "i" for
    inspiration, by defaukt) to each event.

    Args:
        exps (numpy.ndarray): A 1D array of indices where expiration phases start.
        insps (numpy.ndarray): A 1D array of indices where inspiration phases start.
        last_offset (int): The last offset time (in samples) to complete the offset list.
        exp_label, insp_label (string): string/character with which to label exps/insps in `labels`

    Returns:
        tuple: A tuple containing:
            - onsets (numpy.ndarray): A sorted array of onset times (inspiration and expiration).
            - offsets (numpy.ndarray): A corresponding array of offset times.
            - labels (numpy.ndarray): An array of labels ("e" for expiration, "i" for inspiration)
                                      corresponding to the onsets.

    Example:
        >>> exps = np.array([10, 50, 90])
        >>> insps = np.array([30, 70, 110])
        >>> last_offset = 120
        >>> onsets, offsets, labels = make_notmat_vars(exps, insps, last_offset)
        >>> print("Onsets:", onsets)
        >>> print("Offsets:", offsets)
        >>> print("Labels:", labels)
    """

    # Combine and sort the onsets
    onsets = np.concatenate([exps, insps])
    labels = np.array([exp_label] * len(exps) + [insp_label] * len(insps))

    # Sort by onset time
    ii_sort = onsets.argsort()
    onsets = onsets[ii_sort]
    labels = labels[ii_sort]

    offsets = np.append(onsets[1:] - 1, last_offset)

    return (onsets, offsets, labels)
