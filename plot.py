# ./utils/plot.py
# 2024.05.14 CDR
#
# Plotting functions
#


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


