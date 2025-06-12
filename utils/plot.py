# ./utils/plot.py
# 2024.05.14 CDR
#
# Plotting functions
#

import itertools

import numpy as np
import pandas as pd

from sklearn.metrics import ConfusionMatrixDisplay

import matplotlib.pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Rectangle


day_colors = {1: "#a2cffe", 2: "#840000"}


def plot_callback_heatmap(
    df,
    field_name,
    fig=None,
    norm=None,
    cmap_name="viridis",
    bad_color="black",
    day_labels=None,
    block_labels=None,
    **imshow_kwargs,
):
    """
    Given df_trials, plot a heatmap of a given field by day/block.
    """

    if fig is None:
        fig, ax = plt.subplots()
    else:
        ax = fig.axes
        assert len(ax) == 1, "Can only plot on fig with 1 Axes."
        ax = ax[0]

    days, blocks = [np.unique(df.index.get_level_values(l)) for l in ("day", "block")]

    # default to -1 to populate unfilled w diff color
    X = -1 * np.ones([len(blocks), len(days)])

    # construct heatmap

    # for each day/block combo
    for day, block in itertools.product(days, blocks):
        ii_day = days == day
        ii_block = blocks == block

        row = df.xs(key=(day, block), level=("day", "block"))

        if len(row) == 1:
            X[ii_block, ii_day] = row.iloc[0][field_name]
        elif len(row) == 0:
            pass
        else:
            raise KeyError(
                f"More than one entry found for d{day}/bl{block}. (Must be 0 or 1, was {len(row)}.)"
            )

    # prepare cmap. values not found will default to "bad_color"
    cmap = plt.get_cmap(cmap_name)
    cmap.set_bad(bad_color)

    # norm to vmin, vmax in measure_range if passed.
    if not (norm is None or isinstance(norm, (Normalize, str))):
        assert len(norm) == 2, "`measure range` must be length 2! (vmin, vmax)"

        norm = Normalize(*norm)

    im = ax.imshow(
        np.ma.masked_equal(X, -1),
        cmap=cmap,
        norm=norm,
        origin="lower",
        **imshow_kwargs,
    )

    cbar = fig.colorbar(im)
    cbar.set_label(field_name)

    block_labels = _parse_labels(blocks, block_labels)
    day_labels = _parse_labels(days, day_labels)

    ax.set_yticks(np.arange(len(blocks)), labels=block_labels)
    ax.set_xticks(np.arange(len(days)), labels=day_labels)

    return fig, ax, im, cbar


def plot_callback_raster(
    data,
    ax=None,
    plot_stim_blocks=True,
    show_legend=True,
    y_offset=0,
    call_types_to_plot="all",
    call_type_plot_kwargs=None,
    default_plot_kwargs=None,
    force_yticks_int=True,
    **common_plot_kwargs,
):
    """
    Plots a raster plot of callback responses for stimulus-aligned data.

    Parameters:
    - data: pandas DataFrame containing callback data. Each row should correspond to a single stimulus trial and provide columns: 'stim_duration_s', 'call_types', 'call_times_stim_aligned'.
    - ax: matplotlib Axes object (optional). If None, a new figure and axes will be created.
    - plot_stim_blocks: bool, whether to plot stimulus blocks (default: True).
    - show_legend: bool, whether to display the legend. (default: True).
    - y_offset: float, vertical offset for positioning calls along the y-axis. Mostly useful when plotting multiple rasters stacked on the same Axes - eg, see `plot_callback_raster_multiblock` (default: 0).
    - call_types_to_plot: list or "all", specifies which call types to include (default: "all").
    - call_type_plot_kwargs: dict, contains plot style parameters for each call type (default: None).
    - default_plot_kwargs: dict, default plot style parameters (for call types not in `call_type_plot_kwargs`) (default: None).
    - force_yticks_int: bool, whether to force integer tick marks on the y-axis (default: True).
    - **common_plot_kwargs: default kwargs for PatchCollection (and Rectangle proxy artists for legend) common across all call types (superceded by keywords in call_type_plot_kwargs). By default, sets alpha=0.7 and edgecolor=None.

    Returns:
    - ax: matplotlib Axes object with the plotted raster.
    """

    # Use default plotting kwargs if not provided
    if call_type_plot_kwargs is None:
        call_type_plot_kwargs = {
            "Stimulus": dict(facecolor="red"),
            "Call": dict(facecolor="black"),
            "Song": dict(facecolor="blue"),
        }

    if default_plot_kwargs is None:
        default_plot_kwargs = dict(facecolor="green")

    # default kwargs common across all call types (unless superceded)
    common_plot_kwargs = {**{"alpha": 0.7, "edgecolor": None}, **common_plot_kwargs}

    # Create figure and axes if not provided
    if ax is None:
        fig, ax = plt.subplots()

    # Initialize boxes dictionary with call types to plot
    if call_types_to_plot != "all":
        boxes = {call_type: [] for call_type in call_types_to_plot}
    else:
        boxes = {}

    boxes["Stimulus"] = []  # stimulus will be deleted later if not plotting.

    # Iterate through trials and create rectangles for each call and stimulus
    for i, (_, row) in enumerate(data.iterrows()):
        height = i + y_offset

        # Create rectangle for stimulus block
        stim_duration = row["stim_duration_s"]
        boxes["Stimulus"].append(Rectangle((0, height), stim_duration, 1))

        call_types = row["call_types"]
        call_times = row["call_times_stim_aligned"]

        # Add rectangles for each call
        for call_type, (st, en) in zip(call_types, call_times):

            # add type to boxes dict if necessary
            if call_types_to_plot == "all" and call_type not in boxes:
                boxes[call_type] = []

            # add box for this call
            if call_type in boxes:
                boxes[call_type].append(Rectangle((st, height), en - st, 1))

    # Optionally remove stimulus blocks
    if not plot_stim_blocks:
        boxes.pop("Stimulus", None)

    # Plot PatchCollection objects & create legend proxies
    legend_proxies = {}
    for call_type, type_boxes in boxes.items():
        style = {
            **common_plot_kwargs,
            **call_type_plot_kwargs.get(call_type, default_plot_kwargs),
        }

        call_patches = PatchCollection(type_boxes, **style)
        ax.add_collection(call_patches)

        # Create proxy artists for the legend
        legend_proxies[call_type] = Rectangle([0, 0], 0, 0, **style)

    # Add legend if enabled
    if show_legend:
        ax.legend(legend_proxies.values(), legend_proxies.keys())

    # Adjust view (not automatic with add_collection)
    ax.autoscale_view()

    # Force integer tick marks on y-axis if requested
    if force_yticks_int:
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    # Set labels
    ax.set(
        xlabel="Time since stimulus onset (s)",
        ylabel="Trial #",
    )

    return ax


def plot_callback_raster_multiblock(
    data,
    ax=None,
    plot_hlines=True,
    hline_xlim=[-0.1, 3],
    hline_kwargs=None,
    show_block_axis=True,
    y_offset_initial=0,
    block_level_name="block",
    **raster_plot_kwargs,
):
    """
    Plot rasters for multiple callback blocks on the same axis, separated by horizontal lines.

    This function takes a DataFrame with data indexed by blocks and plots callback raster plots for
    each block, separating them with horizontal lines. Optionally, a secondary y-axis can be added
    to show block labels.

    Parameters:
    - data: pandas DataFrame.
      A DataFrame indexed by blocks, where each block contains data for trials or calls to be plotted. Each block is assumed to have its own trial data.
    - ax: matplotlib Axes object (optional) (default: None).
      The Axes to plot on. If None, a new figure and axes will be created.
    - plot_hlines: bool, optional (default: True).
      If True, horizontal lines will be drawn to separate the blocks.
    - hline_xlim: list, optional (default: [-0.1, 3]).
      The x-axis limits for the horizontal lines that separate the blocks. This will also set the
      `xlim` for the entire plot.
    - hline_kwargs: dict, optional (default: None).
      Additional keyword arguments to customize the appearance of the horizontal lines (e.g., color,
      line style). If None, default line properties will be used.
    - show_block_axis: bool, optional (default: True).
      If True, a secondary y-axis will be added on the right side of the plot to label the blocks.
    - y_offset_initial: int, optional (default: 0).
      The initial vertical offset for plotting the first block. Subsequent blocks will be positioned
      below this offset.
    - **raster_plot_kwargs: additional keyword arguments.
      These are passed directly to the `plot_callback_raster` function for customizing the plot
      of individual callback rasters (e.g., plot colors, markers, etc.).

    Returns:
    - ax: matplotlib Axes object.
      The Axes object with the plot. The plot includes multiple callback raster blocks,
      optional horizontal lines, and a secondary y-axis for block labels if enabled.

    Notes:
    - `xlim` sets both `ax.xlim` and the `xmin`, `xmax` for the horizontal block lines.
    - The plot automatically adjusts to the data provided, scaling the view as needed.

    Example:
    ```python
    ax = plot_callback_raster_multiblock(data, plot_hlines=True, show_block_axis=True)
    ```

    """

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots()

    # Extract and order blocks
    blocks = list(
        set(data.index.get_level_values(block_level_name))
    )  # Get unique blocks
    blocks.sort()

    # Initialize vertical offset for plotting
    y_offset = y_offset_initial
    block_locs = []  # List to store y-locations for each block

    # Plot data for each block
    for block in blocks:
        block_locs.append(y_offset)  # Store y-position of the current block
        data_block = data.xs(
            key=block, level=block_level_name
        )  # Get data for the current block

        # Plot raster for this block using the `plot_callback_raster` function
        plot_callback_raster(
            data_block,
            ax=ax,
            y_offset=y_offset,
            **raster_plot_kwargs,  # Pass additional plotting arguments
        )

        # Increment the y_offset for the next block
        y_offset += len(data_block)

    # Plot horizontal lines to separate blocks if requested
    if plot_hlines:
        # Set default horizontal line arguments if none are provided
        if hline_kwargs is None:
            hline_kwargs = dict(
                colors="k",
                linestyles="solid",
                linewidths=0.5,
            )

        # Draw the horizontal lines on the plot
        ax.hlines(
            y=block_locs,
            xmin=hline_xlim[0],
            xmax=hline_xlim[1],
            **hline_kwargs,
        )

    # Add a secondary y-axis for block labels if requested
    if show_block_axis:
        block_axis = ax.secondary_yaxis(location="right")  # Create secondary y-axis
        block_axis.set(
            ylabel="Block",  # Label for the secondary y-axis
            yticks=block_locs,  # Set the y-ticks to match block locations
            yticklabels=blocks,  # Label blocks according to their identifiers
        )

    # Automatically adjust the view to fit the data
    ax.autoscale_view()

    # Set xlim for the entire plot (also affects horizontal lines)
    ax.set(xlim=hline_xlim)

    return ax


def plot_callback_raster_multiday(
    data,
    ax=None,
    hline_xlim=[0.1, 3],
    hline_day_kwargs=None,
    hline_block_kwargs=None,
    show_day_axis=True,
    y_offset_initial=0,
    day_labels=None,
    subday_level="block",
    **raster_plot_kwargs,
):
    """
    Plot callback raster plots for multiple days, with horizontal lines separating the days and blocks.

    This function takes a DataFrame indexed by days and blocks, and plots a callback raster. Every
    day of callbacks is stacked, with horizontal lines separating the days and blocks. Optionally, a
    secondary y-axis can be added to label the days.

    Parameters:
    - data (pandas DataFrame, required):
        A DataFrame indexed by days and blocks, where each day contains trial data with multiple blocks.
        Each block within a day is assumed to have its own callback raster plot data.

    - ax (matplotlib Axes object, optional):
        The Axes to plot on. If None, a new figure and axes will be created. (default: None)

    - hline_xlim (list of float, optional):
        The x-axis limits for the horizontal lines that separate the days. This also sets the `xlim` for
        the entire plot. (default: [0.1, 3])

    - hline_day_kwargs (dict, optional):
        Additional keyword arguments to customize the appearance of the horizontal lines separating the days
        (e.g., color, line style). If None, default line properties will be used (solid black lines).
        (default: None)

    - hline_block_kwargs (dict, optional):
        Additional keyword arguments to customize the appearance of the horizontal lines separating the blocks
        within each day (e.g., color, line style). If None, default dashed black lines will be used.
        (default: None)

    - show_day_axis (bool, optional):
        If True, a secondary y-axis will be added on the right side of the plot to label the days.
        (default: True)

    - y_offset_initial (int, optional):
        The initial vertical offset for plotting the first day. Subsequent days will be positioned below this offset.
        (default: 0)

    - day_labels (list or dict, optional):
        Labels for each day. If None, the days will be labeled with their identifiers (e.g., Day 1, Day 2, etc.).
        If a dictionary is provided, it should map days to custom labels. (default: None)

    - **raster_plot_kwargs (additional keyword arguments):
        These are passed directly to the `plot_callback_raster` function for customizing the plot of individual
        callback raster blocks (e.g., plot colors, etc.).

    Returns:
    - ax (matplotlib Axes object):
        The Axes object with the plot. The plot includes multiple callback raster blocks for each day, horizontal
        lines, and a secondary y-axis for day labels if enabled.

    Notes:
    - The `xlim` parameter sets both `ax.xlim` and the `xmin`, `xmax` for the horizontal day lines.
    - The plot automatically adjusts to the data provided, scaling the view as needed.

    Example:
    ```python
    ax = plot_callback_raster_multiday(data, show_day_axis=True)
    ```
    """

    # Create a new figure and axis if none are provided
    if ax is None:
        fig, ax = plt.subplots()

    # Set default horizontal line properties if none are provided
    if hline_day_kwargs is None:
        hline_day_kwargs = dict(
            colors="k",
            linestyles="solid",
            linewidths=2,
        )

    if hline_block_kwargs is None:
        hline_block_kwargs = dict(
            colors="k",
            linestyles="dashed",
            linewidths=0.5,
        )

    # Extract and order days
    days = list(set(data.index.get_level_values("day")))  # Get unique days
    days.sort()

    # Initialize vertical offset for plotting
    y_offset = y_offset_initial
    day_locs = []  # List to store y-locations for each day

    # Plot data for each day
    for day in days:
        day_locs.append(y_offset)  # Store y-position of the current day
        data_day = data.xs(key=day, level="day")  # Get data for the current day

        # Plot multi-block raster for this day
        plot_callback_raster_multiblock(
            data_day,
            ax=ax,
            block_level_name=subday_level,
            y_offset_initial=y_offset,
            hline_kwargs=hline_block_kwargs,
            show_block_axis=False,
            **raster_plot_kwargs,  # Pass additional plotting arguments
        )

        # Increment the y_offset for the next day
        y_offset += len(data_day)

    # Draw the horizontal lines on the plot to separate days
    ax.hlines(
        y=day_locs,
        xmin=hline_xlim[0],
        xmax=hline_xlim[1],
        **hline_day_kwargs,  # Apply line properties for day separators
    )

    day_labels = _parse_labels(days, day_labels)

    # Optionally add a secondary y-axis for day labels
    if show_day_axis:
        day_axis = ax.secondary_yaxis(location="right")  # Create secondary y-axis
        day_axis.set(
            ylabel="Day",  # Label for the secondary y-axis
            yticks=day_locs,  # Set the y-ticks to match day locations
            yticklabels=day_labels,  # Label days according to their identifiers
        )

    # Automatically adjust the view to fit the data
    ax.autoscale_view()

    # Set xlim for the entire plot (also affects horizontal lines)
    ax.set(xlim=hline_xlim)

    return ax


def plot_group_hist(
    df,
    field,
    grouping_level,
    groups_to_plot=None,
    group_colors=None,
    density=False,
    ax=None,
    ignore_nan="non/total",
    alphabetize_legend=False,
    alt_labels=None,
    binwidth=None,
    histogram_kwargs=None,
    stair_kwargs=None,
):
    """
    TODO: add new parameters to documentation

    Plot overlaid histograms for 1 or more groups, given a DataFrame, a fieldname to plot, and an multi-index level by which to group.

    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame containing the data.
    field : str
        The name of the column in `df` for which the histogram is plotted.
    grouping_level : int or str
        Multi-Index level by which to group. Can be int or string (see pandas.Index.unique)
    groups_to_plot : iterable, optional
        List of groups in grouping_level to plot. If None, plots all groups found. Default: None.
    group_colors : dict or iterable, optional
        A dictionary mapping group names to colors or an iterable of colors for each group.
        If not provided, matplotlib default colors will be used.
    density : bool, optional
        If True, plot density (ie, normalized to count) instead of raw count for each group.
    ax : matplotlib AxesSubplot object, optional
        The axes on which to draw the plot. If not provided, a new figure and axes will be created.
    ignore_nan: bool or string, optional
        If True, cut nan values out of plotted data. If false, np.histogram works, but legend value includes nan value count. If "non/total", legend shows non-nan value count divided by total value count. Default: "non/total".

    Returns:
    --------
    ax : matplotlib Axes
        The axes on which the plot is drawn.
    """

    if ax is None:
        fig, ax = plt.subplots()

    if groups_to_plot is None:
        df.index.unique(level=grouping_level)

    if histogram_kwargs is None:
        histogram_kwargs = {}

    if stair_kwargs is None:
        stair_kwargs = {}

    # use same binning for all groups
    if binwidth is not None:
        assert (
            "bins" not in histogram_kwargs.keys()
        ), "Can't provide both `histogram_kwargs['bins']` and `binwidth`."

        histogram_kwargs["bins"] = np.arange(
            np.min(df[field]),
            np.max(df[field]) + binwidth,
            binwidth,
        )

    elif "range" not in histogram_kwargs.keys():
        histogram_kwargs["range"] = (np.min(df[field]), np.max(df[field]))

    for i_grp, groupname in enumerate(groups_to_plot):
        group_data = df.xs(key=groupname, level=grouping_level)[field]

        if ignore_nan == True:
            group_data = group_data[~np.isnan(group_data)]
            length_string = f"({len(group_data)})"
        elif ignore_nan == "non/total":  # show non-nan div by total
            n_with_nan = len(group_data)
            group_data = group_data[~np.isnan(group_data)]
            length_string = f"({len(group_data)}/{n_with_nan})"
        elif ignore_nan == False:
            length_string = f"({len(group_data)})"
        else:
            KeyError(f"Unknown input for ignore_nan: {ignore_nan}")

        hist, edges = np.histogram(group_data, **histogram_kwargs)

        # histogram: plot density vs count
        if density:
            hist = hist / len(group_data)
            ax.set(ylabel="density")
        else:
            ax.set(ylabel="count")

        # get group colors
        if isinstance(group_colors, dict):
            color = group_colors[groupname]
        elif isinstance(group_colors, (list, tuple)):
            color = group_colors[i_grp]
        else:
            color = f"C{i_grp}"

        if alt_labels is None or groupname not in alt_labels.keys():
            label = f"{grouping_level} {groupname} {length_string}"
        else:
            label = f"{alt_labels[groupname]}{length_string}"

        ax.stairs(hist, edges, label=label, color=color, **stair_kwargs)

    ax.legend()
    if alphabetize_legend:
        handles, labels = plt.gca().get_legend_handles_labels()

        sort_ii = np.argsort(labels)

        plt.legend([handles[i] for i in sort_ii], [labels[i] for i in sort_ii])

    return ax


def plot_violins_by_block(
    bird_data,
    field,
    days,
    day_colors,
    ax=None,
    width=0.75,
    dropna=False,
):
    """
    TODO: document utils.plot.plot_violins_by_block
    """

    for day in days:
        data = bird_data.loc[day]

        if dropna:
            data.dropna(inplace=True)

        blocks = list(set(data.index.get_level_values(0)))  # get blocks

        values_by_block = [list(data.loc[bl, field]) for bl in blocks]

        parts = ax.violinplot(
            values_by_block,
            blocks,
            widths=width,
            showmedians=True,
            showextrema=False,
            # side='low',  # TODO: use matplotlib 3.9 for 'side' parameter
        )

        # customize violin bodies
        for pc in parts["bodies"]:
            pc.set(
                facecolor=day_colors[day],
                edgecolor=day_colors[day],
                alpha=0.5,
            )

            # clip half of polygon
            m = np.mean(pc.get_paths()[0].vertices[:, 0])

            if day == 1:
                lims = [-np.inf, m]
            elif day == 2:
                lims = [m, np.inf]
            else:
                raise Exception("Unknown day.")

            pc.get_paths()[0].vertices[:, 0] = np.clip(
                pc.get_paths()[0].vertices[:, 0], lims[0], lims[1]
            )

        # customize median bars
        parts["cmedians"].set(
            edgecolor=day_colors[day],
        )

    return ax


def plot_pre_post(
    df_day,
    fieldname,
    ax=None,
    color="k",
    bird_id_fieldname="birdname",
    plot_kwargs={},
    add_bird_label=False,
):
    """
    TODO: documentation

    Given `df_day` which has field `fieldname`, plot pre/post type line plot. `color` can be a pd.DataFrame with bird names as index containing color info for every bird in column "color", or a single matplotlib color
    """

    if ax is None:
        fig, ax = plt.subplots()

    all_birds = np.unique(df_day.index.get_level_values(bird_id_fieldname))

    for bird in all_birds:
        bird_data = df_day.loc[bird]

        if isinstance(color, pd.DataFrame):
            c = color.loc[bird, "color"]
        else:
            c = color  # any matplotlib color formats should work here.

        ax.plot(
            bird_data.index,  # days
            bird_data[fieldname],
            color=c,
            **plot_kwargs,
        )

        if add_bird_label:
            i = np.argmax(bird_data.index)

            x_t = bird_data.index[i] + 0.05
            y_t = bird_data[fieldname].iloc[i]

            # TODO: deal with nan
            # print(f'{bird}: ({x_t}, {y_t})')

            ax.text(
                x_t,
                y_t,
                bird,
                fontsize="xx-small",
                verticalalignment="center",
                horizontalalignment="left",
            )

    return ax


def make_graph(transition_counts):
    import networkx as nx

    G = nx.DiGraph()

    for syl, nextSyl in transition_counts.index:
        count = transition_counts.loc[(syl, nextSyl)]
        G.add_edge(syl, nextSyl, weight=count)

    return G


def draw_graph(
    G,
    ax=None,
    graphviz_layout="neato",
    node_kwargs={
        "node_color": "w",  # [[0.3, 0.5, 0.8]],
        "node_size": 400,
        "edgecolors": "k",
    },
    edge_kwargs={
        "arrows": True,
        "arrowsize": 12,
        "width": 0.5,
    },
    font_kwargs={},
):
    import networkx as nx

    if ax is None:
        fig, ax = plt.subplots()
    else:
        plt.sca(ax)

    seed = 9  # Seed random number generators for reproducibility
    pos = nx.nx_agraph.graphviz_layout(G, graphviz_layout)
    # pos = nx.circular_layout(G)
    # pos = nx.spring_layout(G, seed=seed)
    # pos = nx.shell_layout(G)
    # pos = nx.arf_layout(G)

    nodes = nx.draw_networkx_nodes(G, pos, **node_kwargs)
    labels = nx.draw_networkx_labels(G, pos)

    edges = nx.draw_networkx_edges(G, pos, **edge_kwargs)

    weights = nx.get_edge_attributes(G, "weight")
    if isinstance(list(weights.values())[0], float):  # make float labels 2 dec
        weights = {k: "%.2f" % v for k, v in weights.items()}

    edge_labels = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=weights,
        **font_kwargs,
    )

    return ax


def confusion_matrix_plot(
    cm,
    labels=None,
    prob=True,
    ax=None,
    cmap="magma",
    text_kw={"size": "x-small"},
    values_format=".1e",
    **plot_kwarg,
):

    if ax is None:
        fig, ax = plt.subplots()

    if prob:
        cm = (cm.T / cm.sum(1)).T
        cm[np.isnan(cm)] = 0  # columns with no predictions

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=labels,
    )
    disp.plot(
        ax=ax,
        cmap=cmap,
        text_kw=text_kw,
        values_format=values_format,
        **plot_kwarg,
    )

    return ax


def get_custom_tab25_cmap():
    """
    Returns a length 25 colormap originally generated by distinctipy.
    """
    cmap = (
        (0.4497487, 0.4718767, 0.5942480),
        (0.6290267, 0.8856539, 0.4503266),
        (0.6904194, 0.6914681, 0.9815206),
        (0.9863279, 0.4796556, 0.5154868),
        (0.4517669, 0.4705035, 0.9665921),
        (0.5431298, 0.9495135, 0.8116775),
        (0.9817885, 0.9248574, 0.5941716),
        (0.4480328, 0.7592994, 0.7388683),
        (0.6290131, 0.5237998, 0.8076570),
        (0.9629470, 0.7563068, 0.9631502),
        (0.4490579, 0.6493701, 0.4551497),
        (0.9952212, 0.7182117, 0.6317027),
        (0.9788578, 0.4793364, 0.7165476),
        (0.7880826, 0.4522087, 0.6080134),
        (0.5289738, 0.7714431, 0.5404403),
        (0.9610336, 0.9060630, 0.8381917),
        (0.8896936, 0.6847108, 0.7907137),
        (0.6347932, 0.9817782, 0.9932554),
        (0.4563075, 0.8928227, 0.9615504),
        (0.5119891, 0.5842180, 0.9950240),
        (0.5451765, 0.9827373, 0.6166428),
        (0.9645324, 0.6281462, 0.4450814),
        (0.9728047, 0.4467564, 0.8979948),
        (0.4522874, 0.5088940, 0.4451807),
        (0.5029561, 0.4517384, 0.7365014),
    )

    return cmap


def _parse_labels(values, labels):
    """
    Parse and format labels based on the input values and optional custom labels.

    This function processes the `labels` argument based on the provided `values`. If no custom labels
    are provided, it defaults to using the values themselves. If a dictionary of labels is provided,
    it maps each value to its corresponding label using the dictionary. If a list of labels is provided,
    it ensures that the length matches the number of values.

    Parameters:
    - values (list):
        A list of values for which labels are to be assigned. These are typically days or trial identifiers.

    - labels (list, dict, or None):
        Custom labels to assign to the values. If `None`, the function defaults to using the `values` themselves.
        If a dictionary, it should map each value to its corresponding label. If a list, it should have the same
        length as `values`.

    Returns:
    - labels (list):
        A list of labels corresponding to each value in `values`. The labels will be either the input `values`
        themselves, mapped from a dictionary, or directly provided as a list.

    Example:
    ```python
    values = [1, 2, 3]
    labels = {1: 'Day 1', 2: 'Day 2', 3: 'Day 3'}
    result = _parse_labels(values, labels)
    # result will be ['Day 1', 'Day 2', 'Day 3']
    ```

    Notes:
    - If `labels` is provided as a list, it must have the same length as `values`. Otherwise, an assertion error will be raised.
    - If `labels` is a dictionary, it should map each item in `values` to a corresponding label.

    """

    if labels is None:  # Default to values as labels
        labels = values
    elif isinstance(labels, dict):  # Dict lookup for labels
        labels = [labels[d] for d in values]
    else:  # Use labels as provided, ensuring the lengths match
        assert len(labels) == len(values)

    return labels


def remap_cmap(
    cmap,
    start=0,
    midpoint=0.5,
    stop=1.0,
    name="shiftedcmap",
):
    '''
    Function to offset the median value of a colormap, and scale the
    remaining color range. Useful for data with a negative minimum and
    positive maximum where you want the middle of the colormap's dynamic
    range to be at zero.

    Source: https://github.com/TheChymera/chr-helpers/blob/d05eec9e42ab8c91ceb4b4dcc9405d38b7aed675/chr_matplotlib.py

    Input
    -----
      cmap : The matplotlib colormap to be altered
      start : Offset from lowest point in the colormap's range.
          Defaults to 0.0 (no lower ofset). Should be between
          0.0 and 0.5; if your dataset mean is negative you should leave 
          this at 0.0, otherwise to (vmax-abs(vmin))/(2*vmax) 
      midpoint : The new center of the colormap. Defaults to 
          0.5 (no shift). Should be between 0.0 and 1.0; usually the
          optimal value is abs(vmin)/(vmax+abs(vmin)) 
      stop : Offset from highets point in the colormap's range.
          Defaults to 1.0 (no upper ofset). Should be between
          0.5 and 1.0; if your dataset mean is positive you should leave 
          this at 1.0, otherwise to (abs(vmin)-vmax)/(2*abs(vmin)) 
    '''
    cdict = {
        'red': [],
        'green': [],
        'blue': [],
        'alpha': []
    }

    # regular index to compute the colors
    reg_index = np.hstack([
        np.linspace(start, 0.5, 128, endpoint=False), 
        np.linspace(0.5, stop, 129)
    ])

    # shifted index to match the data
    shift_index = np.hstack([
        np.linspace(0.0, midpoint, 128, endpoint=False), 
        np.linspace(midpoint, 1.0, 129)
    ])

    if isinstance(cmap, str):
        cmap = plt.get_cmap(cmap)

    for ri, si in zip(reg_index, shift_index):
        r, g, b, a = cmap(ri)

        cdict['red'].append((si, r, r))
        cdict['green'].append((si, g, g))
        cdict['blue'].append((si, b, b))
        cdict['alpha'].append((si, a, a))

    newcmap = LinearSegmentedColormap(name, cdict)

    return newcmap
