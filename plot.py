# ./utils/plot.py
# 2024.05.14 CDR
# 
# Plotting functions
# 


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
    if 'range' not in histogram_kwargs.keys():
        histogram_kwargs['range'] = (np.min(df[field]), np.max(df[field]))

    for i_grp, groupname in enumerate(df.index.unique(level=grouping_level)):
        group_data = np.array(df.loc[
            df.index.get_level_values(level=grouping_level) == groupname,
            field
        ])

        if ignore_nan:
            group_data = group_data[~np.isnan(group_data)]

        hist, edges = np.histogram(group_data, **histogram_kwargs)

        # histogram: plot density vs count
        if density:
            hist = hist/len(group_data)
            ax.set(ylabel='density')
        else:
            ax.set(ylabel='count')

        # get group colors
        if isinstance(group_colors, dict):
            color = group_colors[groupname]
        elif isinstance(group_colors, (list, tuple)):
            color = group_colors[i_grp]
        else:
            color = f'C{i_grp}'

        if alt_labels is None or groupname not in alt_labels.keys():
            label = f'{grouping_level} {groupname} ({len(group_data)})'
        else:
            label = f'{alt_labels[groupname]} ({len(group_data)})'

        ax.stairs(hist, edges, label=label, color=color, **stair_kwargs.get(groupname, {}))

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
    '''
    TODO: document utils.plot.plot_violins_by_block
    '''
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
        for pc in parts['bodies']:
            pc.set(
                facecolor=day_colors[day],
                edgecolor=day_colors[day],
                alpha=0.5,
            )

            # clip half of polygon
            m = np.mean(pc.get_paths()[0].vertices[:, 0])

            if day==1:
                lims = [-np.inf, m]
            elif day==2:
                lims = [m, np.inf]
            else:
                raise Exception('Unknown day.')
            
            pc.get_paths()[0].vertices[:, 0] = np.clip(pc.get_paths()[0].vertices[:, 0], lims[0], lims[1])

        # customize median bars
        parts['cmedians'].set(
            edgecolor=day_colors[day],
        )

    return ax


def plot_pre_post(
    df_day,
    fieldname,
    ax=None,
    color='k',
    bird_id_fieldname="birdname",
    plot_kwargs={},
    add_bird_label=False
):
    '''
    TODO: documentation

    Given `df_day` which has field `fieldname`, plot pre/post type line plot. `color` can be a pd.DataFrame with bird names as index containing color info for every bird in column "color", or a single matplotlib color
    '''
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

            x_t = bird_data.index[i] + .05
            y_t = bird_data[fieldname].iloc[i]

            # TODO: deal with nan
            # print(f'{bird}: ({x_t}, {y_t})')

            ax.text(
                x_t, 
                y_t, 
                bird, 
                fontsize='xx-small', 
                verticalalignment='center', 
                horizontalalignment='left',
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

    nodes = nx.draw_networkx_nodes(
        G,
        pos,
        node_color="w",  # [[0.3, 0.5, 0.8]],
        node_size=400,
        edgecolors="k",
    )
    labels = nx.draw_networkx_labels(G, pos)

    edges = nx.draw_networkx_edges(
        G,
        pos,
        arrows=True,
        arrowsize=12,
        width=0.5,
    )

    weights = nx.get_edge_attributes(G, "weight")
    if isinstance(list(weights.values())[0], float):  # make float labels 2 dec
        weights = {k: "%.2f" % v for k, v in weights.items()}

    edge_labels = nx.draw_networkx_edge_labels(
        G,
        pos,
        edge_labels=weights,
    )

    return ax


def confusion_matrix_plot(
    cm,
    labels=None,
    prob=True,
    ax=None,
    plot_kw={
        "cmap": "magma",
        "text_kw": {"size": "x-small"},
    },
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
    disp.plot(ax=ax, **plot_kw)

    return ax
