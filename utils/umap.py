# utils/umap.py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, to_rgba

import hdbscan

from .file import parse_birdname
from .plot import remap_cmap


def get_time_since_stim(x, all_trials):
    """
    Calculate the time since the previous stimulus for a breath trial x.

    Parameters:
    - x: Row in DataFrame that contains the breath info.
    - all_trials: DataFrame containing the trial information.

    Returns:
    - Time since the previous stimulus if present, else pd.NA.
    """
    if pd.isna(x.stims_index):
        return pd.NA
    else:
        return x.start_s - all_trials.loc[(x.name[0], x.stims_index), "trial_start_s"]


def loc_relative(wav_filename, calls_index, df, field="index", i=1, default=None):
    """
    Fetch the index or field of a breath segment relative to the inputted one.

    Parameters:
    - wav_filename: Filename of the audio file.
    - calls_index: Index of the current call.
    - df: DataFrame containing the breath segment data.
    - field: The specific field to retrieve from the segment (default is "index").
    - i: Relative index to retrieve (positive for future, negative for past).
    - default: Default value if the field is not found.

    Returns:
    - The index or field value relative to the inputted one.
    """
    v = default

    try:
        # get trial & check its existence
        i_next = (wav_filename, calls_index + i)
        trial = df.loc[i_next]

        if field == "index":
            v = i_next
        else:
            v = trial[field]

    except KeyError as e:
        pass

    return v


def plot_embedding_data(
    embedding,
    embedding_name,
    df=None,
    ax=None,
    plot_type="putative_call",
    show_colorbar=True,
    cmap_name=None,
    scatter_kwargs=None,
    set_kwargs=None,
    set_bad=None,
    **kwargs,
):
    """
    Plot embeddings colored according to specified plot type.

    Parameters:
    - embedding: 2d data to plot (e.g., UMAP embedding).
    - embedding_name: Name of the embedding.
    - df: DataFrame containing metadata for color mapping. Order should match embedding. Can be all_breaths or all_trials.
    - ax: The axis object to plot on. If None, a new figure is created.
    - plot_type: Type of plot to generate (e.g., "putative_call", "amplitude", etc.). Deals with selecting relevant field from df & particulars of colormap.
    - show_colorbar: Whether to display a colorbar.
    - cmap_name: Colormap to use. Default is None, which uses a predefined mapping. You can alternatively pass an actual cmap to kwargs, maybe.
    - scatter_kwargs: Additional arguments for the scatter plot.
    - set_kwargs: Settings for plot appearance (e.g., axis labels).
    - set_bad: Settings for "bad" values in the colormap (default is None). Also applies to masked clusters for cluster plot type.
    - **kwargs: Additional keyword arguments for specific plot types.

    Returns:
    - The axis object containing the plot.
    """
    x, y = [embedding[:, i] for i in [0, 1]]

    if ax is None:
        fig, ax = plt.subplots()

    if scatter_kwargs is None:
        scatter_kwargs = dict(s=0.2, alpha=0.5)

    if set_kwargs is None:
        set_kwargs = dict(xlabel="UMAP1", ylabel="UMAP2")

    if set_bad is None:
        set_bad = dict(c="k", alpha=0.2)

    if isinstance(set_bad, dict):
        set_bad = to_rgba(**set_bad)

    set_kwargs = {**set_kwargs}  # Copy to avoid modifying input

    if "title" not in set_kwargs.keys():
        set_kwargs["title"] = f"{embedding_name}: {plot_type.upper()}"

    if cmap_name is None:
        # Default colormaps based on plot type
        default_cmaps = {
            "putative_call": "Dark2_r",
            "amplitude": "magma_r",
            "duration": "cool",
            "breaths_since_stim": "viridis_r",
            "bird_id": "Set2",
            "insp_onset": "RdYlGn",
            "insp_offset": "magma_r",
            "clusters": "jet",
        }
        cmap_name = default_cmaps.get(plot_type, "viridis")

    # Plot specific handling based on plot_type
    if "c" in kwargs.keys():
        # Manually defined color labels
        plot_type_kwargs = dict(c=kwargs.pop("c"), cmap=plt.get_cmap(cmap_name))

    elif plot_type == "putative_call":
        plot_type_kwargs = dict(
            c=np.array(df["putative_call"]).astype(int),
            cmap=plt.get_cmap(cmap_name, 2),
            vmin=0,
            vmax=1,
        )
        cbar_ticks = [0.25, 0.75]
        cbar_tick_labels = ["no_call", "call"]
        cbar_label = None

    elif plot_type == "amplitude":
        plot_type_kwargs = dict(c=df.amplitude, cmap=plt.get_cmap(cmap_name))
        cbar_label = "amplitude (normalized)"

    elif plot_type == "duration":
        # Calculate duration from either 'duration_s' or 'ii_first_insp' columns
        if "duration_s" in df.columns:
            c = df.duration_s * 1000  # Convert to ms
        elif "ii_first_insp" in df.columns:
            try:
                all_insps = np.vstack(df["ii_first_insp"]).T / kwargs.pop("fs") * 1000
            except KeyError:
                raise KeyError(
                    "Pass fs as a kwarg to compute duration from column ii_first_insp"
                )
            c = all_insps[1, :] - all_insps[0, :]
        else:
            raise ValueError(
                "Input df needs 'duration_s' or 'ii_first_insp' column to plot duration."
            )

        plot_type_kwargs = dict(
            c=c,
            cmap=plt.get_cmap(cmap_name),
            vmax=1000,
        )
        cbar_label = "duration (ms)"

    elif plot_type == "breaths_since_stim":
        # Handle "breaths since stimulus" plot type
        # by default 8+ breaths are merged on cmap
        n_breaths = kwargs.pop("n_breaths", 8)
        n_since_stim = df["trial_index"].fillna(-1)
        cmap = plt.get_cmap(cmap_name, n_breaths + 1)
        cmap.set_bad(set_bad)

        plot_type_kwargs = dict(
            c=np.ma.masked_equal(n_since_stim, -1),
            cmap=cmap,
            vmin=0,
            vmax=n_breaths + 1,
        )
        cbar_label = "# breaths segs since last stim"
        cbar_ticks = np.arange(n_breaths + 1) + 0.5
        cbar_tick_labels = [str(int(t)) for t in cbar_ticks]
        cbar_tick_labels[-1] = f"{cbar_tick_labels[-1]}+"

    elif plot_type == "bird_id":
        birdnames = pd.Categorical(
            df.apply(lambda x: parse_birdname(x.name[0]), axis=1)
        )
        n_birds = len(birdnames.unique())
        cmap = plt.get_cmap(cmap_name, n_birds)

        plot_type_kwargs = dict(c=birdnames.codes, cmap=cmap)
        cbar_label = "bird_id"
        cbar_ticks = np.arange(n_birds)
        cbar_tick_labels = birdnames.unique()
        cbar_label = None

    elif plot_type in ["insp_onset", "insp_offset"]:
        assert (
            "fs" in kwargs.keys()
        ), f"fs must be provided as kwarg for {plot_type} embedding plot."
        all_insps = np.vstack(df["ii_first_insp"]).T / kwargs.pop("fs") * 1000

        if plot_type == "insp_onset":
            onsets = all_insps[0, :]
            cmap_zero = abs(onsets.min()) / np.ptp(onsets)
            plot_type_kwargs = dict(
                c=onsets,
                vmin=onsets.min(),
                vmax=onsets.max(),
                cmap=remap_cmap(cmap_name, midpoint=cmap_zero),
            )

        elif plot_type == "insp_offset":
            plot_type_kwargs = dict(c=all_insps[1, :], cmap=plt.get_cmap(cmap_name))

        cbar_label = f"{plot_type} (ms, stim_aligned)"

    elif plot_type in ["clusters", "cluster", "clusterer"]:
        assert (
            "clusterer" in kwargs.keys()
        ), "clusterer must be provided as kwarg for 'clusters' plot type."
        clusterer = kwargs.pop("clusterer")

        cmap = plt.get_cmap(cmap_name, np.unique(clusterer.labels_).size)
        vmin, vmax = min(clusterer.labels_), max(clusterer.labels_) + 1
        n = vmax - vmin

        hl = "highlighted_clusters" in kwargs.keys()
        msk = "masked_clusters" in kwargs.keys()

        if hl and msk:
            raise ValueError(
                "Provide either masked_clusters or highlighted_clusters, not both."
            )
        elif hl:
            masked_clusters = np.setdiff1d(
                np.arange(vmin, vmax), kwargs.pop("highlighted_clusters")
            )
        elif msk:
            masked_clusters = kwargs.pop("masked_clusters")
        else:
            masked_clusters = [-1]  # Default: mask hdbscan noise cluster

        ii_masked = np.isin(np.arange(vmin, vmax), masked_clusters)
        cmap = plt.get_cmap(cmap_name, n)
        colors = cmap(np.linspace(0, 1, n))
        colors[ii_masked] = set_bad

        plot_type_kwargs = dict(c=clusterer.labels_, cmap=ListedColormap(colors))
        cbar_label = "cluster"

    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    # Overwrite default kwargs with user input
    plot_type_kwargs = {**plot_type_kwargs, **kwargs}
    plot_type_kwargs["cmap"].set_bad(set_bad)

    # Plot the embedding data
    sc = ax.scatter(x, y, **plot_type_kwargs, **scatter_kwargs)
    ax.set(**set_kwargs)

    # colorbar
    if show_colorbar and "cbar_label" in vars():
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    if "cbar_ticks" in vars():
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_tick_labels)

    return ax


def plot_cluster_traces_pipeline(
    trace_type,
    df,
    fs,
    cluster_labels,
    padding_kwargs=None,
    aligned_to=None,
    select="all",
    cluster_cmap=None,
    set_kwargs=None,
    axs=None,
    **plot_kwargs,
):
    """
    Pipeline function to plot cluster traces with appropriate preprocessing.

    Parameters:
    - trace_type: Type of traces to plot (e.g., "breath" or "insp").
    - df: DataFrame containing trace data.
    - fs: Sampling frequency.
    - clusterer: Clusterer object to label clusters.
    - padding_kwargs: Optional parameters for trace padding or alignment.
    - aligned_to: The reference point to which traces are aligned.
    - select: Selection criteria for traces (default is "all").
    - cluster_cmap: Colormap to use for clusters.
    - set_kwargs: Plot settings (e.g., axis limits, labels).
    - **plot_kwargs: Additional arguments for trace plotting.

    Returns:
    - The axis object containing the plot.
    """
    # =========GET TRACES=========#
    # Stack traces in a np array, padding or cutting as necessary
    if padding_kwargs is None:
        padding_kwargs = {}

    try:
        raw_traces = df[trace_type]
    except KeyError:
        raw_traces = df

    traces, aligned_at_f = stack_traces(traces=raw_traces, **padding_kwargs)

    all_cluster_labels = np.unique(cluster_labels)

    # Select based on 'putative_call' column
    traces, cluster_labels = select_traces(df, cluster_labels, traces, select)

    # Use full clusterer.labels_ to still plot empty clusters
    cluster_data = {
        i_cluster: traces[cluster_labels == i_cluster, :]
        for i_cluster in all_cluster_labels
    }

    # Generate x-axis for the plot
    x, xlabel = get_trace_x(trace_type, traces.shape[1], fs, aligned_at_f, aligned_to)

    # Set default plot settings
    default_set_kwargs = dict(
        xlabel=xlabel,
        ylabel="amplitude",
        xlim=[x.min(), x.max()],
        ylim=[traces.min(), traces.max()],
    )

    if set_kwargs is None:
        set_kwargs = default_set_kwargs
    else:
        set_kwargs = {**default_set_kwargs, **set_kwargs}

    # =========PLOT TRACES=========#
    axs_cluster_traces = plot_cluster_traces(
        cluster_data,
        x,
        select,
        clusters_to_plot="all",
        cmap=cluster_cmap,
        all_labels=cluster_labels,
        set_kwargs=set_kwargs,
        axs=axs,
        **plot_kwargs,
    )

    return axs_cluster_traces


def stack_traces(
    traces,
    pad_method=None,
    max_length=None,
    i_align=None,
    aligned_at=0,
):
    """
    Stack and align traces in a DataFrame, padding or cutting them as needed.

    Parameters:
    - traces: DataFrame or Series containing the traces to stack.
    - pad_method: Method for padding/cutting ("end", "beginning", "offset", "aligned", "index").
    - max_length: Maximum length for padding or cutting. If None, the longest trace is used.
    - i_align: The index to align traces to if padding/cutting method requires it.
    - aligned_at: The index position at which traces are aligned.

    Returns:
    - A tuple (stacked_traces, aligned_at) where stacked_traces is the combined numpy array
      and aligned_at is the index used for alignment.
    """

    # === use traces as-is (requires same length)
    if pad_method is None:
        assert (
            len(set(traces.apply(len))) == 1
        ), "All traces must be the same length to run with `pad_method=None`!"

    # === pad/cut at end (keep default alignment)
    elif pad_method == "end":
        if max_length is None:
            max_length = max(traces.apply(len))

        traces = traces.apply(_pad_cut_end, max_length=max_length)

        aligned_at = 0

    # === pad/cut at beginning
    elif pad_method in ["beginning", "start"]:
        if max_length is None:
            max_length = max(traces.apply(len))

        traces = traces.apply(_pad_cut_beginning, max_length=max_length)

        aligned_at = max_length

    # === align to a certain point in each trace
    elif pad_method in ["offset", "aligned", "index"]:

        assert (
            i_align is not None
        ), f"i_align (index in each trace used for alignment) must be provided for `pad_method={pad_method}`"

        if max_length is None:
            max_pre = max(i_align)
            max_post = max(traces.apply(lambda x: len(x) - max_pre))
        else:
            assert len(max_length) == 2

        max_pre, max_post = max_length

        new_traces = pd.Series(index=traces.index, dtype=object)

        # Loop through traces to apply alignment based on max_pre and max_post
        for n, i in enumerate(traces.index):
            trace = traces.loc[i]

            adl_len_start = max_pre - i_align[n]

            trace = _pad_cut_beginning(
                trace,
                len(trace) + adl_len_start,
            )
            trace = _pad_cut_end(trace, max_pre + max_post)

            new_traces.loc[i] = trace

        traces = new_traces
        aligned_at = max_pre

    else:
        raise ValueError(f"pad_method={pad_method} not recognized")

    return (np.vstack(traces), aligned_at)


def _pad_cut_beginning(x, max_length):
    """
    Pad or cut the beginning of a trace to ensure it has a specific length.

    Parameters:
    - x: The trace to modify.
    - max_length: The desired length for the trace.

    Returns:
    - A padded or cut version of the input trace.
    """
    if len(x) > max_length:
        return x[-max_length:]
    else:
        return np.pad(x, [max_length - len(x), 0])


def _pad_cut_end(x, max_length):
    """
    Pad or cut the end of a trace to ensure it has a specific length.

    Parameters:
    - x: The trace to modify.
    - max_length: The desired length for the trace.

    Returns:
    - A padded or cut version of the input trace.
    """
    if len(x) > max_length:
        return x[:max_length]
    else:
        return np.pad(x, [0, max_length - len(x)])


def select_traces(all_breaths, cluster_labels, traces, select):
    """
    Select traces based on the 'select' criteria (e.g., "all", "call", "no call").

    Parameters:
    - all_breaths: DataFrame containing information about each breath (e.g., putative call).
    - clusterer: Clusterer object that assigns labels to the traces.
    - traces: The traces to filter based on the selection criteria.
    - select: The criterion for selecting traces ("all", "call", "no call").

    Returns:
    - A tuple (selected_traces, cluster_labels) where selected_traces are the filtered traces
      and cluster_labels are the labels from the clusterer.
    """
    if select == "all":
        ii_select = np.ones(len(all_breaths)).astype(bool)
    elif select == "call":
        ii_select = all_breaths["putative_call"]
    elif select == "no call":
        ii_select = ~all_breaths["putative_call"]
    else:
        raise ValueError(f"select={select} not recognized")

    traces = traces[ii_select]
    cluster_labels = cluster_labels[ii_select]
    
    return traces, cluster_labels


def get_trace_x(trace_type, trace_len_f, fs, aligned_at_f, aligned_to=None):
    """
    Get the x-axis values and label for the trace plot based on trace type.

    Parameters:
    - trace_type: The type of the trace ("breath_interpolated", "insps_interpolated", etc.).
    - trace_len_f: The length of the trace in frames.
    - fs: The sampling frequency.
    - aligned_at_f: The frame at which the trace is aligned.
    - aligned_to: Optional alignment reference.

    Returns:
    - A tuple (x, xlabel) where x is the x-axis values and xlabel is the label for the axis.
    """
    interpolated_types = ["breath_interpolated", "insps_interpolated"]

    # === interpolated types: [0, 1]
    if trace_type in interpolated_types:
        x = np.linspace(0, 1, trace_len_f)
        xlabel = "normalized duration"

        if aligned_at_f is not None:
            x = x + aligned_at_f
    # === time
    else:
        x = (np.arange(trace_len_f) - aligned_at_f) / fs * 1000
        xlabel = "time (ms)"

        if aligned_to is not None:
            xlabel = f"{xlabel}, aligned to {aligned_to}"

    return x, xlabel


def plot_cluster_traces(
    cluster_data,
    x,
    select,
    clusters_to_plot="all",
    set_kwargs=None,
    cmap=None,
    all_labels=None,
    axs=None,
    **plot_kwargs,
):
    """
    Plot traces for each cluster, optionally with various plot settings.

    Parameters:
    - cluster_data: Dictionary where keys are cluster labels and values are the traces for that cluster.
    - x: The x-axis values for the plot.
    - select: The selection criteria ("all", "call", "no call").
    - clusters_to_plot: The clusters to plot (can be "all" or specific labels).
    - set_kwargs: Plot settings (e.g., labels, limits).
    - cmap: Colormap to use for coloring the clusters.
    - all_labels: All cluster labels for plotting (if available).
    - plot_kwargs: Additional keyword arguments for plotting (e.g., line style, color).

    Returns:
    - A dictionary of axes for each cluster.
    """
    # Overwrite default plot_kwargs with user input
    default_plot_kwargs = dict(
        color="k",
        alpha=0.2,
        linewidth=0.5,
    )
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    # Default settings for plot if not provided
    if set_kwargs is None:
        set_kwargs = dict(
            xlabel="time",
            ylabel="amplitude",
        )

    # If clusters_to_plot is set to "all", use all available clusters
    if clusters_to_plot == "all":
        clusters_to_plot = cluster_data.keys()

    # Predefine axes for each cluster
    if axs is None:
        axs = {k: plt.subplots()[1] for k in clusters_to_plot}

    # Loop through clusters to plot their traces
    for i_cluster in clusters_to_plot:
        traces = cluster_data[i_cluster]
        ax = axs[i_cluster]

        # Plot the traces for the cluster
        ax.plot(
            x,
            traces.T,
            **plot_kwargs,
        )

        # Plot the mean trace in red
        ax.plot(x, traces.T.mean(axis=1), color="r", linewidth=1)

        # Set the title with the number of traces in the cluster
        if select == "all" or all_labels is None:
            n = f"{traces.shape[0]}"
        else:
            n = f"{traces.shape[0]}/{sum(all_labels == i_cluster)}"

        # Set the title color based on the colormap or default
        if cmap is not None:
            title_color = cmap(i_cluster)
        else:
            title_color = "k"

        # Set title and other settings for the plot
        ax.set_title(
            f"cluster {i_cluster} traces {select} (n={n})",
            color=title_color,
        )

        ax.set(**set_kwargs)

    return axs


def plot_violin_by_cluster(
    data,
    cluster_labels,
    set_kwargs=None,
    cluster_cmap=None,
    **plot_kwargs,
):
    """
    Plot a violin plot for each cluster based on the input data.

    Parameters:
    - data: The data to plot (e.g., duration, amplitude).
    - cluster_labels: cluster label for each point in data (eg `clusterer.labels_` for hdbscan obj)
    - set_kwargs: Settings for the plot appearance (e.g., labels, limits).
    - cluster_cmap: Colormap to use for coloring the clusters (according to cluster label). If none, uses default colormap (all blue).
    - **plot_kwargs: Additional keyword arguments for violin plot.

    Returns:
    - The axis object containing the plot.
    """
    # Overwrite default plot_kwargs with user input
    default_plot_kwargs = dict(showextrema=False)
    plot_kwargs = {**default_plot_kwargs, **plot_kwargs}

    # Overwrite default set_kwargs with user input
    default_set_kwargs = dict(xlabel="cluster", ylabel="data")
    set_kwargs = {**default_set_kwargs, **set_kwargs}

    cluster_data = {
        i_cluster: data[(cluster_labels == i_cluster) & ~np.isnan(data)]
        for i_cluster in np.unique(cluster_labels)
    }
    labels, data = list(cluster_data.keys()), cluster_data.values()

    # Create a violin plot for each cluster
    fig, ax = plt.subplots()
    parts = ax.violinplot(data, **plot_kwargs)

    if cluster_cmap is not None:
        for i, pc in enumerate(parts["bodies"]):
            pc.set_facecolor(cluster_cmap(labels[i]))

    ax.set_xticks(ticks=range(1, 1 + len(labels)), labels=labels)

    # Set the plot appearance
    ax.set(**set_kwargs)

    return ax, parts
