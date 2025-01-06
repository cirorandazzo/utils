# ./utils/plot.py
# 2024.05.14 CDR
#
# Plotting functions

callback_raster_stim_kwargs = dict(color="red", alpha=0.5)
callback_raster_call_kwargs = dict(color="black", alpha=0.5)
callback_raster_song_kwargs = dict(color="blue", alpha=0.5)
callback_raster_default_kwargs = dict(color="green", alpha=0.5)

day_colors = {1: "#a2cffe", 2: "#840000"}


def plot_callback_raster(
    data,
    ax=None,
    plot_stim_blocks=True,
    show_legend=True,
    y_offset=0,
    call_types_to_plot="all",
    call_type_plot_kwargs={
        "Stimulus": callback_raster_stim_kwargs,
        "Call": callback_raster_call_kwargs,
        "Song": callback_raster_song_kwargs
    },
    default_plot_kwargs=callback_raster_default_kwargs,
    force_yticks_int=True,
):
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Rectangle

    if ax is None:
        fig = plt.figure()
        ax = fig.subplots()

    # Construct patch collections for call boxes. Separate by call type.
    boxes = {k: [] for k in call_types_to_plot}
    boxes["Stimulus"] = [] # ensure stimulus is present; may be removed later

    for i, trial_i in enumerate(data.index):
        height = i + y_offset

        boxes["Stimulus"].append(
            Rectangle((0, height), data.loc[trial_i, "stim_duration_s"], 1)
        )

        call_types = data.loc[trial_i, "call_types"]
        call_times = data.loc[trial_i, "call_times_stim_aligned"]

        for type, time in zip(call_types, call_times):
            st, en = time

            # dealing with "all" keyword
            if call_types_to_plot=="all" and type not in boxes.keys():
                boxes[type] = []

            if type in boxes.keys():
                boxes[type].append(Rectangle((st, height), en - st, 1))

    if not plot_stim_blocks:
        del boxes['Stimulus']

    legend = {}

    for type, type_boxes in boxes.items():
        style = call_type_plot_kwargs.get(type, default_plot_kwargs)
        call_patches = PatchCollection(type_boxes, **style)
        ax.add_collection(call_patches)

        # construct proxy artists for legend
        legend[type] = Rectangle([0, 0], 0, 0, **style)

    if show_legend:
        ax.legend(legend.values(), legend.keys())

    ax.autoscale_view()

    if force_yticks_int:
        from matplotlib.ticker import MaxNLocator

        ax.yaxis.set_major_locator(MaxNLocator(integer=True))

    ax.set(
        xlabel="Time since stimulus onset (s)",
        ylabel="Trial #",
    )

    return ax


def plot_callback_raster_multiblock(
    data,
    ax=None,
    plot_hlines=True,
    show_block_axis=True,
    show_legend=False,
    xlim=[-0.1, 3],
    call_kwargs=callback_raster_call_kwargs,
    stim_kwargs=callback_raster_stim_kwargs,
    title=None,
):
    """
    Plot multiple blocks on the same axis with horizontal lines separating.

    Notes: xlim sets ax.xlim, but also xmin and xmax for horizontal block lines.
    """

    blocks = list(set(data.index.get_level_values(0)))  # get & order blocks
    blocks.sort()

    y_offset = 0
    block_locs = []

    for block in blocks:
        block_locs.append(y_offset)
        data_block = data.loc[block]

        plot_callback_raster(
            data_block,
            ax=ax,
            y_offset=y_offset,
            plot_stim_blocks=True,
            show_legend=show_legend,
            call_kwargs=call_kwargs,
            stim_kwargs=stim_kwargs,
        )

        y_offset += len(data_block)

    if plot_hlines:
        ax.hlines(
            y=block_locs,
            xmin=xlim[0],
            xmax=xlim[1],
            colors="k",
            linestyles="solid",
            linewidths=0.5,
        )

    if show_block_axis:
        block_axis = ax.secondary_yaxis(location="right")
        block_axis.set(
            ylabel="Block",
            yticks=block_locs,
            yticklabels=blocks,
        )

    ax.set(
        title=title,
        xlim=xlim,
        ylim=[0, len(data)],
    )

    return ax


def plot_group_hist(
    df,
    field,
    grouping_level,
    group_colors=None,
    density=False,
    ax=None,
    ignore_nan=False,
    alphabetize_legend=False,
    alt_labels=None,
    histogram_kwargs={},
    stair_kwargs={},
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
    group_colors : dict or iterable, optional
        A dictionary mapping group names to colors or an iterable of colors for each group.
        If not provided, matplotlib default colors will be used.
    density : bool, optional
        If True, plot density (ie, normalized to count) instead of raw count for each group.
    ax : matplotlib AxesSubplot object, optional
        The axes on which to draw the plot. If not provided, a new figure and axes will be created.
    ignore_nan: bool, optional
        If True, cut nan values out of plotted data. Else, np.histogram throws ValueError if nan values are in data to plot.

    Returns:
    --------
    ax : matplotlib Axes
        The axes on which the plot is drawn.
    """
    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    if ax is None:
        fig, ax = plt.subplots()

    # use same binning for all groups
    if "range" not in histogram_kwargs.keys():
        histogram_kwargs["range"] = (np.min(df[field]), np.max(df[field]))

    for i_grp, groupname in enumerate(df.index.unique(level=grouping_level)):
        group_data = np.array(
            df.loc[df.index.get_level_values(level=grouping_level) == groupname, field]
        )

        if ignore_nan:
            group_data = group_data[~np.isnan(group_data)]

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
            label = f"{grouping_level} {groupname} ({len(group_data)})"
        else:
            label = f"{alt_labels[groupname]} ({len(group_data)})"

        ax.stairs(
            hist, edges, label=label, color=color, **stair_kwargs.get(groupname, {})
        )

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
    import pandas as pd
    import numpy as np
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd

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
    import numpy as np
    import pandas as pd

    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt

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
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.metrics import ConfusionMatrixDisplay

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
