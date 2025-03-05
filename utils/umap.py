# utils/umap.py

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

import hdbscan

from .file import parse_birdname
from .plot import remap_cmap

def get_time_since_stim(x, all_trials):
    """
    For each breath in all_trials, returns time since previous stimulus
    """
    if pd.isna(x.stims_index):
        return pd.NA
    else:
        return x.start_s - all_trials.loc[(x.name[0], x.stims_index), "trial_start_s"]


def loc_relative(wav_filename, calls_index, df, field="index", i=1, default=None):
    """
    Fetch index or field of a breath segment relative to the inputted one.
    """
    # return value
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
    df,
    ax=None,
    plot_type="putative_call",
    show_colorbar=True,
    cmap_name=None,
    scatter_kwargs=None,
    set_kwargs=None,
    **kwargs,
):
    """
    Function to plot different types of embeddings based on plot_type.

    Parameters:
    - embedding_name: Name of the embedding.
    - df: DataFrame containing the metadata for cmap. Order should match embedding. Can be all_breaths or all_trials.
    - scatter_kwargs: Keyword arguments for the scatter plot.
    - set_kwargs: Settings for the plot (e.g., axis limits, labels).
    - plot_type: Type of plot to generate ("putative_call", "amplitude", "duration", "time_since_stim").
    - n_breaths: Number of breaths for the "time_since_stim" plot (default is 8).
    """

    x, y = [embedding[:, i] for i in [0, 1]]

    if ax is None:
        fig, ax = plt.subplots()

    if scatter_kwargs is None:
        scatter_kwargs = dict(
            s=.2,
            alpha=0.5,
        )

    if set_kwargs is None:
        set_kwargs = dict(
            xlabel="UMAP1",
            ylabel="UMAP2",
        )

    set_kwargs = {**set_kwargs}  # copy set_kwargs; don't modify input

    if "title" not in set_kwargs.keys():
        set_kwargs["title"] = f"{embedding_name}: {plot_type.upper()}"

    if cmap_name is None:
        default_cmaps = {
            "putative_call": "Dark2_r",
            "amplitude": "magma_r",
            "duration": "cool",
            "breaths_since_stim": "viridis_r",
            "bird_id": "Set2",
            "insp_onset": "RdYlGn",
            "insp_offset": "magma_r",
        }

        cmap_name = default_cmaps.get(plot_type, "viridis")

    # GET PLOT TYPE SPEIFICS
    if "c" in kwargs.keys():
        # manually defined color labels
        plot_type_kwargs = dict(
            c=kwargs.pop("c"),
            cmap=cmap_name,
        )

    elif plot_type == "putative_call":
        plot_type_kwargs = dict(
            c=np.array(df["putative_call"]).astype(int),
            cmap=plt.get_cmap(cmap_name, 2),
            vmin=0,
            vmax=1,
        )

        cbar_ticks = [.25, .75]
        cbar_tick_labels = (["no_call", "call"])
        cbar_label = None

    elif plot_type == "amplitude":
        plot_type_kwargs = dict(
            c=df.amplitude,
            cmap=cmap_name,
        )
        cbar_label = "amplitude (normalized)"

    elif plot_type == "duration":
        if "duration_s" in df.columns:
            c = df.duration_s * 1000
        elif "ii_first_insp" in df.columns:
            try:
                all_insps = np.vstack(df["ii_first_insp"]).T / kwargs.pop("fs") * 1000
            except KeyError as e:
                raise KeyError("Pass fs as a kwarg to compute duration from column ii_first_insp")
            c = all_insps[1, :] - all_insps[0, :]
        else:
            raise ValueError("Inputted df needs one column of [duration_s, ii_first_insp] to plot duration.")

        plot_type_kwargs = dict(
            c=c,
            cmap=cmap_name,
            vmax=1000,
        )
        cbar_label = "duration (ms)"

    elif plot_type == "breaths_since_stim":
        # by default 8+ breaths are merged on cmap
        n_breaths = kwargs.pop("n_breaths", 8)

        n_since_stim = df["trial_index"].fillna(-1)
        cmap = plt.get_cmap(cmap_name, n_breaths+1)
        cmap.set_bad("k")

        plot_type_kwargs = dict(
            c=np.ma.masked_equal(n_since_stim, -1),
            cmap=cmap,
            vmin=0,
            vmax=n_breaths+1,
        )
        cbar_label = "# breaths segs since last stim"

        cbar_ticks = np.arange(n_breaths+1) + 0.5
        cbar_tick_labels = [str(int(t)) for t in cbar_ticks]
        cbar_tick_labels[-1] = f"{cbar_tick_labels[-1]}+"

    elif plot_type == "bird_id":
        birdnames = pd.Categorical(df.apply(lambda x: parse_birdname(x.name[0]), axis=1))
        n_birds = len(birdnames.unique())

        cmap = plt.get_cmap(cmap_name, n_birds)

        plot_type_kwargs = dict(
            c=birdnames.codes,
            cmap=cmap,
        )
        cbar_label = "bird_id"

        cbar_ticks = np.arange(n_birds)
        cbar_tick_labels = birdnames.unique()
        cbar_label = None

    elif plot_type in ["insp_onset", "insp_offset"]:
        assert "fs" in kwargs.keys(), f"fs must be provided as kwarg for {plot_type} embedding plot."
        all_insps = np.vstack(df["ii_first_insp"]).T / kwargs.pop("fs") * 1000

        if plot_type == "insp_onset":
            onsets = all_insps[0, :]
            # map time 0 --> center of cmap
            cmap_zero = abs(onsets.min()) / (np.ptp(onsets))

            plot_type_kwargs = dict(
                c=onsets,
                vmin=onsets.min(),
                vmax=onsets.max(),
                cmap=remap_cmap(cmap_name, midpoint=cmap_zero),
            )

        elif plot_type == "insp_offset":
            plot_type_kwargs = dict(c=all_insps[1, :], cmap=cmap_name)

        else:
            raise KeyError("Invalid plot type. How'd you even get here?")

        cbar_label = f"{plot_type} (ms, stim_aligned)"

    else:
        raise ValueError(f"Unsupported plot type: {plot_type}")

    # overwrite default kwargs with user input
    plot_type_kwargs = {**plot_type_kwargs, **kwargs}

    # DO PLOTTING
    sc = ax.scatter(
        x,
        y,
        **plot_type_kwargs,
        **scatter_kwargs,
    )
    ax.set(**set_kwargs)

    # COLORBAR
    if show_colorbar and "cbar_label" in vars():
        cbar = plt.colorbar(sc, ax=ax)
        cbar.set_label(cbar_label)

    if "cbar_ticks" in vars():
        cbar.set_ticks(cbar_ticks)
        cbar.set_ticklabels(cbar_tick_labels)

    return ax
