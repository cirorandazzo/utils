# utils/breath.py


# normalize to pre-stim
# roughly recenter by subtracting median
# roughly segment (w/ insp/exp threshold)
# recenter based on mean(pre-stim breaths)

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.signal import find_peaks, peak_prominences
from scipy.stats import gaussian_kde

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .umap import loc_relative as umap__loc_relative


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


def plot_breath_callback_trial(
    breath,
    fs,
    stim_trial,
    y_breath_labels,
    pre_time_s,
    post_time_s,
    ylims,
    st_s,
    en_s,
    ax=None,
    color_dict={"exp": "r", "insp": "b"},
):

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 5))

    # indices of waveform segment
    ii_audio = (np.array([st_s - pre_time_s, st_s + post_time_s]) * fs).astype(int)

    if ii_audio[1] >= len(breath):
        ii_audio[1] = len(breath)

    # plot waveform
    y = breath[np.arange(*ii_audio)]
    x = (np.arange(len(y)) / 44100) - pre_time_s
    ax.plot(x, y, color="k", linewidth=0.5, label="breath")

    # plot trial onset/offset (start of this stim & next stim)
    ax.vlines(
        x=[0, en_s - st_s],
        ymin=ylims[0],
        ymax=ylims[1],
        color="g",
        linewidth=3,
        label="stimulus",
    )

    # plot breath overlay
    ii_breaths = [c in ["exp", "insp"] for c in stim_trial["call_types"]]

    if len(ii_breaths) > 0:
        if y_breath_labels == "infer":
            br2fr = lambda t_br: min(
                (fs * (t_br + st_s)).astype(int), len(breath) - 1
            )  # rounding might take slightly out of range

            y_func = lambda br: breath[br2fr(br)]

        else:
            y_func = lambda br: y_breath_labels

        arcs = [
            np.array([[br_st, y_func(br_st)], [br_en, y_func(br_en)]])
            for br_st, br_en in np.array(stim_trial["call_times_stim_aligned"])[
                ii_breaths
            ]
        ]
        colors = [color_dict[t] for t in np.array(stim_trial["call_types"])[ii_breaths]]

        lc = LineCollection(
            arcs,
            colors=colors,
            linewidths=4,
            alpha=0.5,
        )
        ax.add_collection(lc)

    ax.set(
        xlim=[-1 * pre_time_s, post_time_s],
        xlabel="Time, stim-aligned (s)",
        ylabel="Breath pressure (raw)",
        ylim=ylims,
    )

    return ax


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


def get_kde_threshold(breath, **fit_kwargs):
    """
    Wrapper of fit_breath_distribution which directly returns spline-fit threshold given a breath waveform. E.g., useful for passing as `centering` function into segment_breaths.
    """

    x_dist, dist_kde, trough_ii, amplitude_ii = fit_breath_distribution(
        breath, **fit_kwargs
    )

    threshold = x_dist[trough_ii]

    return threshold


def plot_amplitude_dist(
    breath,
    ax=None,
    binwidth=100,
    leftmost=None,
    rightmost=None,
    percentiles=(25, 75),
    median_multiples=(1, 1.5, 2),
):
    """
    Plots a histogram of the amplitude distribution from the provided data and overlays statistical lines.

    Parameters:
    -----------
    breath : array-like
        A 1D array or list of numerical values representing the breath data (or any other data representing amplitude values) to be plotted.

    ax : matplotlib.axes.Axes, optional
        An optional `matplotlib` Axes object to plot the histogram. If not provided, a new `matplotlib` figure and axis are created.

    binwidth : int, optional
        The width of each bin for the histogram. Default is 100.

    leftmost : int or float, optional
        The leftmost boundary for the histogram bins. If not provided, it is set to two times the `binwidth` smaller than the minimum value of `breath`.

    rightmost : int or float, optional
        The rightmost boundary for the histogram bins. If not provided, it is set to two times the `binwidth` larger than the maximum value of `breath`.

    Percentiles : iterable of numeric, optional.
        Plots percentiles of data as vertical black lines on distribution. Does not plot if median_multiples is None or an empty list. Default: None.

    median_multiples : iterable of numeric, optional.
        Plots multiples of median as vertical red lines on distribution. Does not plot if median_multiples is None or an empty list. Default: None.

    Returns:
    --------
    ax : matplotlib.axes.Axes
        The `matplotlib` Axes object containing the histogram plot, including statistical lines for percentiles and multiples of the median.

    Description:
    ------------
    This function generates a histogram of the distribution of `breath` data using a specified bin width. It also overlays the following additional information:

    - Percentiles: Vertical dashed lines representing the 25th and 75th percentiles of the `breath` data.
    - Median multiples: Vertical dotted lines representing multiples of the median value of `breath` (1x, 1.5x, and 2x).

    The histogram is normalized (`density=True`) to show a probability density rather than raw counts. The additional statistical lines help to visualize the distribution of the data in relation to its central tendency and spread.

    Example Usage:
    --------------
    ```python
    import matplotlib.pyplot as plt
    import numpy as np

    # Example data
    breath_data = np.random.normal(0, 1, 1000)

    # Create plot
    fig, ax = plt.subplots()
    plot_amplitude_dist(breath_data, ax=ax)

    # Show plot
    plt.show()
    ```

    Notes:
    ------
    - The function automatically determines the histogram boundaries unless explicitly provided via `leftmost` and `rightmost`.
    - The median lines are plotted at multiples of the median of the `breath` data, specifically at 1, 1.5, and 2 times the median value.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if leftmost is None:
        leftmost = min(breath) - 2 * binwidth

    if rightmost is None:
        rightmost = max(breath) + 2 * binwidth

    hist, edges = np.histogram(
        breath, bins=np.arange(leftmost, rightmost, binwidth), density=True
    )

    ax.stairs(hist, edges, fill=True)

    if percentiles is not None and len(percentiles) > 0:
        ax.vlines(
            x=[np.percentile(breath, p) for p in percentiles],
            ymin=0,
            ymax=max(hist),
            color="k",
            linestyles="--",
            alpha=0.5,
            zorder=3,
            label=f"percentile(s): {percentiles}",
        )

    if median_multiples is not None and len(median_multiples) > 0:
        # median & multiples: red lines
        ax.vlines(
            x=[q * np.median(breath) for q in median_multiples],
            ymin=0,
            ymax=max(hist),
            color="r",
            linestyles=":",
            alpha=0.5,
            zorder=3,
            label=f"median * {median_multiples}",
        )

    return ax


def get_first_breath_segment(
    trial,
    breath_type,
    fs,
    earliest_allowed_onset=None,
    buffer_s=0,
    return_stim_aligned=True,
    return_unit="samples",
):
    """
    Returns the onset and offset of the first instance of a specified 'breath_type' call after a given
    earliest allowed onset, optionally adding a buffer. The result can be returned in different units
    (samples, frames, seconds, or milliseconds) and optionally adjusted for the stimulus-aligned time.

    Parameters:
    -----------
    trial : dict
        The trial data containing 'call_types' and 'call_times_stim_aligned'.
    breath_type : str
        The type of breath (or call) to search for.
    fs : float
        The sampling frequency (samples per second) of the audio data.
    earliest_allowed_onset : float, optional
        The earliest allowed onset time (in seconds) after stimulus presentation for the breath type to occur. Default is None.
    buffer_s : float, optional
        The amount of time (in seconds) to extend both the onset and offset of the detected breath. Default is 0.
    return_stim_aligned : bool, optional
        Whether to return the time in stimulus-aligned units. If False, the time is adjusted by the trial start time. Default is True.
    return_unit : str, optional
        The unit to return the onset and offset times in. Options are "samples", "frames", "seconds", or "milliseconds". Default is "samples".

    Returns:
    --------
    numpy.ndarray
        A 2-element array containing the onset and offset of the first matching breath type call,
        adjusted for the buffer and in the requested unit. Returns np.nan if no such call is found.

    Raises:
    -------
    ValueError
        If an invalid unit is provided in the `return_unit` parameter.

    Notes:
    ------
    - If no breath of the specified type is found after the `earliest_allowed_onset`, a warning is raised and np.nan is returned.
    - The resulting onset and offset times are either in stimulus-aligned or trial time, depending on `return_stim_aligned`.
    - The time is adjusted by the specified `buffer_s` and converted to the requested unit (`samples`, `frames`, `seconds`, or `ms`).

    Example:
    --------
    get_first_breath_segment(trial, 'inhalation', fs=1000, earliest_allowed_onset=1.0, buffer_s=0.5, return_unit='seconds')
    """

    # select call type of interest
    ii_breath_type = np.array(trial["call_types"]) == breath_type

    # get onsets & offsets
    breath_times = trial["call_times_stim_aligned"][ii_breath_type, :]

    # reject calls that are too early
    if earliest_allowed_onset is not None:
        breath_times = breath_times[breath_times[:, 0] >= earliest_allowed_onset]

    if len(breath_times) == 0:
        warnings.warn(
            f"No calls of type `{breath_type}` found after `{earliest_allowed_onset}s post-stimulus`."
        )
        return np.nan

    # get earliest matching call in samples
    i_earliest_onset = breath_times[:, 0].argmin()

    earliest_call = breath_times[i_earliest_onset, :]

    if not return_stim_aligned:
        earliest_call += trial["trial_start_s"]

    # add buffer
    earliest_call[0] -= buffer_s
    earliest_call[1] += buffer_s

    # convert to requested unit
    if return_unit in ["samples", "frames"]:
        earliest_call = (earliest_call * fs).astype(int)
    elif return_unit in ["seconds"]:
        pass
    elif return_unit in ["milliseconds", "ms"]:
        earliest_call = earliest_call / 1000
    else:
        raise ValueError(f"Unknown unit: {return_unit}")

    return earliest_call


def get_phase(
    breathDur,
    avgExpDur,
    avgInspDur,
    wrap=True,
):
    """
    python implementation of ziggy phase computation code ("phase2.m")

    Note: algorithm errors on breathDur > 2 normal breath lengths (for consistency with ZK code)

    breathDur: time between preceding expiration & event for which to compute phase - usually onset of call expiration. (formerly t_nMin1Exp_to_Call)
    avgExpDur: duration of expiration for comparison (in same unit as breathDur). Usually, this is the mean duration of non-call expirations. breathDur in [0, avgExpDur] (time) is mapped to [0, pi].  breathDur in [0, avgExpDur] (time) is mapped to [0, pi].
    avgInspDur: duration of inspiration for comparison (in same unit as breathDur). Usually, this is the mean duration of non-call inspirations. breathDur in [0, avgExpDur] (time) is mapped to [0, pi].  breathDur in [avgExpDur, avgExpDur+avgInspDur] (time) is mapped to [pi, 2pi].
    wrap: Behavior for breaths longer than average breath. If True, computes based on (breathDur MOD avgDur) - ie, constrains to 2pi. If False, breathDur longer than avgBreathDur return phase greater than 2pi. Default True (as in ZK implementation).
    """

    phase = None

    avgBreathDur = avgExpDur + avgInspDur
    dur_ratio = breathDur / avgBreathDur  # get it? duration ratio?

    assert (
        dur_ratio <= 2
    ), "algorithm only implmented for callT within 2 normal breath lengths!"

    if wrap:
        n = 0
    else:
        n = 2 * np.pi * np.floor(dur_ratio)

    breathDur = breathDur % avgBreathDur

    # call happens before the expiration before the call... (ie, oops)
    if breathDur < 0:
        phase = 0.1

    # call happens during this expiration
    elif breathDur < avgExpDur:
        # expiration is [0, pi]
        phase = np.pi * (breathDur / avgExpDur)

    # call happens during inspiration after that
    elif breathDur >= avgExpDur and breathDur < avgBreathDur:
        # inspiration is [pi, 2pi]
        phase = np.pi * (1 + (breathDur - avgExpDur) / avgInspDur)

    else:
        ValueError("this really shouldn't happen...")

    return n + phase


def loc_relative(*args, **kwargs):
    """
    alias for utils.umap > loc_relative()
    """

    return umap__loc_relative(*args, **kwargs)


def get_segment_duration(trial, df, rel_index=[-2, -1]):
    """
    Get the total duration of breath segments in rel_index (where
    current == 0).

    Eg, for an expiration (n=0), default [-2, -1] will return the
    duration of the previous expiration and previous inspiration.
    (-2 = prev exp, -1 = prev insp).
    """

    # duration for: [prev exp, prev insp]
    durations = np.array(
        [
            loc_relative(*trial.name, df, field="duration_s", i=i, default=np.nan)
            for i in rel_index
        ]
    )

    return durations.sum()


def plot_duration_distribution(
    all_breaths,
    hist_kwargs=None,
    kde_kwargs=None,
    mean_kwargs=None,
):
    """
    Plot duration distributions for each type of breath in input df.

    Useful for getting a sense of whether mean represents a good estimate for phase calculations.

    In that case, it's recommended to reject putative call breaths before running.
    """
    fig, axs = plt.subplots(nrows=len(all_breaths["type"].unique()), sharex=True)

    # plotting parameters from kwargs
    default_hist_kwargs = dict(density=True)
    default_kde_kwargs = dict()
    default_mean_kwargs = dict(c="r", linewidth=0.5, linestyle="--")

    if hist_kwargs is None:
        hist_kwargs = {}
    hist_kwargs = {**default_hist_kwargs, **hist_kwargs}

    if kde_kwargs is None:
        kde_kwargs = {}
    kde_kwargs = {**default_kde_kwargs, **kde_kwargs}

    if mean_kwargs is None:
        mean_kwargs = {}
    mean_kwargs = {**default_mean_kwargs, **mean_kwargs}

    # plot each type in separate ax
    for (type, breaths), ax in zip(all_breaths.groupby("type"), axs):
        data = breaths["duration_s"]
        title = f"{type} (n={len(data)})"

        # plot histogram
        ax.hist(data, bins=np.linspace(0, 1.6, 200), label="data", **hist_kwargs)
        ax.set(title=title, ylabel="density")

        # plot spline fit distribution
        kde, x_kde, y_kde = get_kde_distribution(data, xlim=(0, 1.6), xsteps=200)
        ax.plot(x_kde, y_kde, label="kde", **kde_kwargs)

        # plot mean (vertical line)
        ax.axvline(x=np.mean(data), label="mean", **mean_kwargs)

    fig.tight_layout()

    axs[0].legend()
    ax.set(xlim=[-0.01, 0.6], xlabel="duration_s")

    return fig, axs


def plot_traces_by_cluster_and_phase(
    df_breaths,
    fs,
    window_s,
    trace_folder,
    phase_bins,
    cluster_col_name="cluster",
    ncols=4,
    figsize=(11, 8.5),
    plot_axis_lines=True,
    trace_kwargs=None,
    axline_kwarg=None,
):
    """
    Plot waveform traces grouped by cluster and phase bins.

    Parameters
    ----------
    df_breaths : pd.DataFrame
        DataFrame containing metadata for each trace. Must include a 'phase' column and a cluster identifier column.
    fs : int or float
        Sampling frequency (Hz).
    window_s : tuple of float
        Time window around the event to extract (start_time, end_time) in seconds.
    trace_folder : str
        Path to the folder containing trace data files.
    phase_bins : list or array-like
        Sequence of bin edges for phase segmentation. Should include both start and end.
    cluster_col_name : str, optional
        Column name for cluster labels in `df_breaths`. Default is "cluster".
    ncols : int, optional
        Number of columns in the subplot grid. Default is 4.
    figsize : tuple, optional
        Size of each figure in inches. Default is (11, 8.5).
    plot_axis_lines : bool, optional
        Whether to include horizontal and vertical axis lines in each subplot. Default is True.
    trace_kwargs : dict, optional
        Keyword arguments passed to `matplotlib.pyplot.plot` for the traces.
    axline_kwarg : dict, optional
        Keyword arguments passed to `axhline` and `axvline` for drawing axis lines.

    Returns
    -------
    figs : dict
        Dictionary mapping cluster labels to their corresponding matplotlib Figure objects.
    """

    figs = {}

    n_bins = len(phase_bins) - 1
    window_fr = (fs * window_s).astype(int)
    x = np.linspace(*window_s, np.ptp(window_fr))

    # default plot kwargs
    default_trace_kwargs = dict(
        linewidth=0.1,
        color="k",
        alpha=0.4,
    )
    if trace_kwargs is None:
        trace_kwargs = {}
    trace_kwargs = {**default_trace_kwargs, **trace_kwargs}

    default_axline_kwarg = dict(
        linewidth=1,
        color="tab:blue",
    )
    if axline_kwarg is None:
        axline_kwarg = {}
    axline_kwarg = {**default_axline_kwarg, **axline_kwarg}

    # plot by cluster
    for cluster, df_cluster_breaths in df_breaths.groupby(cluster_col_name):
        phases = df_cluster_breaths["phase"]

        fig, axs = plt.subplots(
            figsize=figsize,
            ncols=ncols,
            nrows=np.ceil(n_bins / ncols).astype(int),
            sharex=True,
            sharey=True,
        )

        for st_ph, en_ph, ax in zip(
            phase_bins[:-1],
            phase_bins[1:],
            axs.ravel()[:n_bins],
        ):
            calls_in_phase = df_cluster_breaths.loc[
                (phases > st_ph) & (phases <= en_ph)
            ]
            traces = calls_in_phase.apply(
                get_wav_snippet_from_numpy, axis=1, args=[window_fr, fs, trace_folder]
            )

            if plot_axis_lines:
                ax.axhline(**axline_kwarg)
                ax.axvline(**axline_kwarg)

            if len(traces) != 0:
                traces = np.vstack(traces.loc[traces.notnull()])

                ax.plot(x, traces.T, **trace_kwargs)
                ax.plot(x, traces.mean(axis=0), color="r")

            ax.set(
                title=f"({st_ph:.2f},{en_ph:.2f}], n={traces.shape[0]}",
                xlim=window_s,
            )

        fig.suptitle(f"Cluster: {cluster} (n={len(df_cluster_breaths)} call exps)")

        figs[cluster] = fig
    return figs


def get_wav_snippet_from_numpy(
    breath_trial,
    window_fr,
    fs,
    trace_folder,
    error_value=pd.NA,
):
    """
    Given the breath in "breath_trial", load a requested window from a preprocessed numpy file. Window is centered on breath onset, with "post" time including breath length.

    Presumes numpy file is in trace_folder, which contain .npy copies of the .wav file in breath_trial.name sans folder structure.
    """

    # get file
    wav_file = breath_trial.name[0]
    np_file = trace_folder.joinpath(Path(wav_file).stem + ".npy")
    breath = np.load(np_file)

    # get indices
    onset = int(fs * breath_trial["start_s"])
    ii = np.arange(*window_fr) + onset

    try:
        return breath[ii]
    except IndexError:
        return error_value
