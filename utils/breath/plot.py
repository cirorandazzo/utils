# utils.breath.plot
#
# utils related to plotting

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection

from .breath import get_wav_snippet_from_numpy
from .segment import get_kde_distribution


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
    npy_breath_channel=0,
    cluster_col_name="cluster",
    alignment_col_name="start_s",
    ncols=4,
    figsize=(11, 8.5),
    plot_axis_lines=True,
    max_traces=500,
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
    max_traces : int, optional
        Cluster/phase combo with more traces than this will simply plot mean trace +/- std for memory reasons.
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
                get_wav_snippet_from_numpy,
                axis=1,
                args=[
                    window_fr,
                    fs,
                    trace_folder,
                    npy_breath_channel,
                    alignment_col_name,
                ],
            )

            if plot_axis_lines:
                ax.axhline(**axline_kwarg)
                ax.axvline(**axline_kwarg)

            if len(traces) != 0 and not all(traces.isnull()):
                traces = traces.loc[traces.notnull()]
                traces = np.vstack(traces)
                mean = traces.mean(axis=0)

                # if too many traces, plot mean +/- std
                if traces.shape[0] > max_traces:
                    std = traces.std(axis=0)

                    ax.fill_between(
                        x,
                        mean - std,
                        mean + std,
                        alpha=0.2,
                        color="k",
                        label="$\pm$ std",
                    )
                else:
                    ax.plot(x, traces.T, **trace_kwargs)

                ax.plot(x, mean, color="r")

            ax.set(
                title=f"({st_ph:.2f},{en_ph:.2f}], n={traces.shape[0]}",
                xlim=window_s,
            )

        fig.suptitle(f"Cluster: {cluster} (n={len(df_cluster_breaths)} call exps)")

        figs[cluster] = fig
    return figs
