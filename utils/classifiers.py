# classifiers.py
#
# utils for calcium classifiers


def cut_calcium_data(
    row,
    pre_onset=3,
    post_onset=1,
):
    import numpy as np

    x = row["x"]
    first = np.flatnonzero((x[1:] >= 0) & (x[:-1] < 0))[
        0
    ]  # first Ca frame post-syl onset

    for col in ["dff", "deriv", "x"]:
        row.loc[col] = row.loc[col][..., first - pre_onset : first + post_onset]

    return row


def make_scores_df(
    path,
    classifier_filename="classifiers.pickle",
    max_syls=15,
    syl_folder_prefix="n",
    neg_syls=True,
):
    """
    Generate a DataFrame containing performance scores of classifiers across syllables.

    This function searches for classifier performance data stored in pickle files.
    It compiles the performance of various models across cross-validation folds into a single
    DataFrame, indexed by syllable number, model name, and fold index.

    Parameters:
    ----------
    path : str
        The directory path where syllable folders are located. Each folder should contain
        a classifier pickle file.

    classifier_filename : str, optional
        The filename of the classifier pickle file within each syllable folder.
        Default is "classifiers.pickle".

    max_syls : int, optional
        The maximum number of syllables to process. The function will stop if a folder
        corresponding to a syllable number does not exist. Default is 15.

    syl_folder_prefix : str, optional
        A prefix that precedes the syllable number in folder names.
        Default is "n", but can also be set to "nPl", etc.

    neg_syls : bool, optional
        If True, returns syllable indices as negative (ie, -4 for 4 syls before divergent syl). Default is True.

    Returns:
    -------
    pandas.DataFrame
        A DataFrame containing the performance scores, indexed by syllable number, model name,
        and cross-validation fold index. The DataFrame includes columns for syllable number
        (`syl`), model name (`model`), and cross-validation fold index (`xfold`).

    Notes:
    -----
    - The function loads performance data from each classifier's pickle file, which is expected
      to contain a dictionary with a "performance" key.
    - If the specified folder for a syllable does not exist, the function will terminate early.
    - The resulting DataFrame will be empty if no valid classifier files are found.

    Example:
    --------
    >>> scores_df = make_scores_dict("/path/to/syllables")
    """

    import os
    import pickle

    import numpy as np
    import pandas as pd

    all_scores = pd.DataFrame()  # Initialize an empty DataFrame to store all scores

    for i_syl in range(max_syls):
        # Load classifier.pickle details; stop if syllable folder not found
        file = os.path.join(path, f"{syl_folder_prefix}{i_syl}", classifier_filename)

        if not os.path.exists(file):
            break

        with open(file, "rb") as f:
            data = pickle.load(f)
            performance = data["performance"]
            del data

        # Make a DataFrame containing all models & all crossfolds for this syllable
        this_syl = pd.DataFrame()
        for model, model_performance in performance.items():
            this_model = pd.DataFrame(model_performance)
            this_model["model"] = model
            this_model["xfold"] = this_model.index

            this_syl = pd.concat([this_syl, this_model])

        if neg_syls:
            this_syl["syl"] = -1 * i_syl
        else:
            this_syl["syl"] = i_syl

        all_scores = pd.concat([all_scores, this_syl])

    all_scores.set_index(["syl", "model", "xfold"], inplace=True)

    return all_scores


def plot_all_cms(
    cms,
    cm_folder,
    nrows,
    ncols,
    y=None,
    cm_labels=None,
    figure_extension="svg",
    skip_crossfolds=True,
):
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    from .plot import confusion_matrix_plot

    if not ((y is None) ^ (cm_labels is None)):
        raise TypeError("Exactly one of `y` or `labels` must be defined.")

    if y is not None:
        cm_labels = (sorted(np.unique(y)),)

    multi_fig, multi_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

    for i, (clf_name, all_cms) in enumerate(cms.items()):
        cm = np.array(all_cms).sum(0)  # add all CV cms together

        # plot cross-val avg on multifig
        ax = np.ravel(multi_ax)[i]

        confusion_matrix_plot(
            cm,
            labels=cm_labels,
            prob=True,
            ax=ax,
            colorbar=False,
            values_format=".2f",
        )

        ax.set(title=clf_name)

        # plot cross-val avg separately
        ax = confusion_matrix_plot(
            cm,
            labels=cm_labels,
            prob=True,
            values_format=".2f",
        )

        tstr = f"ALL-{clf_name}"
        ax.set(title=tstr)

        fig = ax.get_figure()
        fig.savefig(os.path.join(cm_folder, f"{tstr}.{figure_extension}"))
        plt.close(fig)

        # plot each crossfold
        if not skip_crossfolds:
            for i, cm in enumerate(all_cms):
                ax = confusion_matrix_plot(
                    cm,
                    labels=cm_labels,
                    prob=True,
                    values_format=".2f",
                )

                tstr = f"{clf_name} ({i})"
                ax.set(title=tstr)

                fig = ax.get_figure()
                fig.savefig(os.path.join(cm_folder, f"{tstr}.{figure_extension}"))
                plt.close(fig)

    for empty_ax in np.ravel(multi_ax)[len(cms) - np.size(multi_ax) :]:
        empty_ax.axis("off")

    multi_fig.tight_layout()
    multi_fig.savefig(
        os.path.join(cm_folder, f"!cm_all_classifiers.{figure_extension}")
    )
    plt.close(multi_fig)


def plot_all_tr_cms(
    data,
    X,
    y,
    fitted_classifiers,
    cv,
    pth,
    trans_map,
    fmt="g",
    sort_key=None,
    figure_extension="svg",
    skip_crossfolds=True,
    **heatmap_kwargs,
):
    """
    unused_kw: eats unused kwargs. enables using 'to_save' dicts without deleting params
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    all_trans = data.reset_index().apply(trans_map, axis=1)

    unique_trans = list(np.unique(all_trans))
    unique_trans.sort(key=sort_key)
    # unique_trans.sort(key=lambda seq: seq[-1])  # to sort by last syl
    y_pred_unique = np.unique(y)

    tr_cms = {}

    # fitted classifiers: dict with keys being model names. each value is array of 5 models (from 5fcv)
    for clf_name, clfs in fitted_classifiers.items():  # classifier type
        tr_cms[clf_name] = []

        for i_cv, (ii_train, ii_test) in enumerate(cv.split(X, y)):
            clf = clfs[i_cv]  # classifier trained on this crossfold

            # cm labels
            y_pred = clf.predict(X[ii_test])
            y_true_mapped = data.iloc[ii_test].reset_index().apply(trans_map, axis=1)
            # y_true = y[ii_test]  # useful for debugging, make df with these 3 as cols

            # make & store
            cm = make_subcondition_confusion_matrix(
                y_true_mapped,
                y_pred,
                prob=False,
                y_pred_unique_labels=y_pred_unique,
                y_true_unique_labels=unique_trans,
            )

            tr_cms[clf_name].append(cm)

            # plot individual crossfolds
            if not skip_crossfolds:
                fig, ax = plt.subplots()

                ax = subcondition_confusion_matrix_plot(
                    cm,
                    ax=ax,
                    y_pred_unique_labels=y_pred_unique,
                    y_true_unique_labels=unique_trans,
                    fmt=fmt,
                    **heatmap_kwargs,
                )

                tstr = f"{clf_name} ({i_cv})"
                ax.set(ylabel="True label in context", title=tstr)

                fig.savefig(os.path.join(pth, f"{tstr}.{figure_extension}"))
                plt.close()

        cm = np.array(tr_cms[clf_name]).sum(0)  # add all cms together

        fig, ax = plt.subplots()
        ax = subcondition_confusion_matrix_plot(
            cm,
            ax=ax,
            y_pred_unique_labels=y_pred_unique,
            y_true_unique_labels=unique_trans,
            fmt=fmt,
            **heatmap_kwargs,
        )

        tstr = f"!ALL-{clf_name}"
        ax.set(ylabel="True label in context", title=tstr)

        fig.savefig(os.path.join(pth, f"{tstr}.{figure_extension}"))

        plt.close()

    return tr_cms


def plot_all_tr_cms_diffs(
    tr_cms,
    data,
    y,
    trans_map,
    pth,
    fmt="g",
    diff_cmap_extremes=[(0, 0, 0), (1, 0, 0)],
    sort_key=None,
    figure_extension="svg",
):
    import os

    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap

    # make true transition confusion matrix (tr_cm)
    y_trans = data.reset_index().apply(trans_map, axis=1)

    unique_trans = list(np.unique(y_trans))
    unique_trans.sort(key=sort_key)

    y_unique = np.unique(y)
    y_trans_unique = np.unique(y_trans)

    tr_cm_true = make_subcondition_confusion_matrix(
        true_labels_mapped=y_trans,
        predicted_labels=y,
        prob=False,
        y_pred_unique_labels=y_unique,
        y_true_unique_labels=y_trans_unique,
    )

    # plot & save true tr_cm
    fig, ax = plt.subplots()

    ax = subcondition_confusion_matrix_plot(
        tr_cm_true,
        ax=ax,
        y_pred_unique_labels=y_unique,
        y_true_unique_labels=y_trans_unique,
        fmt=fmt,
    )

    ax.set(title="HYPOTHETICAL: Perfect performance")
    plt.savefig(os.path.join(pth, f"!example-perfect_tm.{figure_extension}"))
    plt.close()

    # make cmap
    diff_cmap = LinearSegmentedColormap.from_list(
        "Custom", diff_cmap_extremes, N=np.max(tr_cm_true)
    )

    # go thru tr_cms & save/plot diffs
    tr_cm_diffs = {}

    for clf_name, clf_cms in tr_cms.items():
        this_cm = np.array(clf_cms).sum(0)  # add all cms together

        tr_cm_diffs[clf_name] = np.abs(tr_cm_true - this_cm)

        fig, ax = plt.subplots()

        # vmin/vmax standardize colormap across plots.
        ax = subcondition_confusion_matrix_plot(
            tr_cm_diffs[clf_name],
            ax=ax,
            y_pred_unique_labels=y_unique,
            y_true_unique_labels=y_trans_unique,
            fmt=fmt,
            cmap=diff_cmap,
            vmin=0,
            vmax=np.max(tr_cm_true) / 2,
        )

        tstr = f"!tr_cm_diff-{clf_name}"
        ax.set(ylabel="True label in context", title=tstr)

        fig.savefig(os.path.join(pth, f"{tstr}.{figure_extension}"))
        plt.close()

    return tr_cm_true, tr_cm_diffs


def plot_scores(
    mean_scores,
    ax=None,
    horizontal_reference_line=0.5,
    condition_plot_kwargs={},
    default_plot_kwargs={},
    alpha=0.8,
    **all_plot_kwargs,
):
    """
    Plot the mean scores of models across different conditions or folds.

    This function generates a line plot of mean scores, with an option to
    include a horizontal reference line. It can create a new plot or utilize
    an existing axes object for customization.

    Note: it's often useful to set Axes cycler in advance, eg with
    >>> ax.set_prop_cycle(color=plt.cm.Set2.colors)

    Parameters:
    ----------
    mean_scores : pandas.DataFrame
        A DataFrame where each row represents the mean scores of a different model
        or condition, and each column corresponds to a different evaluation metric
        or cross-validation fold.

    ax : matplotlib.axes.Axes, optional
        An existing matplotlib Axes object to plot on. If not provided, a new figure
        and axes will be created. Default is None.

    horizontal_reference_line : float, optional
        A y-value for the horizontal reference line across the plot. If set to None,
        no line will be plotted. Default is 0.5.

    alpha : float, optional
        The transparency level of the plotted lines. Default is 0.8.

    condition_plot_kwargs : dict, optional
        A dictionary of keyword arguments for customizing the plot for each condition.
        These will override default_plot_kwargs for the respective model.

    default_plot_kwargs : dict, optional
        A dictionary of default keyword arguments to customize the plot, such as line style,
        color, etc. These will be applied to all models unless overridden by condition_plot_kwargs.

    **all_plot_kwargs : keyword arguments
        Additional keyword arguments to customize the Axes object, such as labels,
        limits, etc.

    Returns:
    -------
    matplotlib.axes.Axes
        The Axes object with the plotted scores.

    Notes:
    -----
    - The function expects `mean_scores` to be a DataFrame where the index represents
      different models or conditions, and the columns represent different score metrics.
    - The plotted lines will have an alpha transparency of 0.8 for better visualization.
    - If an Axes object is created, it will have a default size of (12, 6) inches.

    Example:
    --------
    Plot Model A in red, every other model in black, and all models with dashed linestyle

    >>> import pandas as pd
    >>> import numpy as np
    >>> import matplotlib.pyplot as plt

    >>> mean_scores = pd.DataFrame({
    >>>     '1': [0.6, 0.7, 0.8],
    >>>     '2': [0.65, 0.75, 0.85]
    >>>     }, index=['Model A', 'Model B', 'Model C'])
    >>> ax = plot_scores(
    >>>     mean_scores,
    >>>     condition_plot_kwargs = {
    >>>         "Model A": {"color": "red"},
    >>>     },
    >>>     default_plot_kwargs = {"color": "black"},
    >>>     linestyle="--",)
    >>> plt.show()
    """

    import matplotlib.pyplot as plt
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 6))

    # draw a horizontal line, by default at y=0.5
    if horizontal_reference_line is not None:
        ax.plot(
            [min(mean_scores.columns), max(mean_scores.columns)],
            horizontal_reference_line * np.array([1, 1]),
            c="k",
            linestyle="--",
            linewidth=0.5,
            alpha=0.5,
        )

    # plot all rows of mean_scores

    mean_scores.apply(
        lambda x: ax.plot(
            x,
            label=x.name,
            alpha=alpha,
            # use condition kwargs if given, else use default.
            **(condition_plot_kwargs.get(x.name, default_plot_kwargs)),
            **all_plot_kwargs,
        ),
        axis=1,
        result_type="reduce",
    )

    ax.legend(loc="best")
    ax.set(
        xticks=np.arange(min(mean_scores.columns), max(mean_scores.columns) + 1),
    )

    return ax


def train_models(
    names,
    classifiers,
    cv,
    X,
    y,
    preprocessing_steps=[],
    return_cm_labels=False,
):
    import numpy as np

    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import (
        balanced_accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
    )
    from copy import deepcopy

    fitted_classifiers = {}
    performance = {}
    cms = {}

    y_unique = np.unique(y)

    for name, clf in zip(names, classifiers):
        performance[name] = {}
        fitted_classifiers[name] = list(
            range(cv.get_n_splits())
        )  # preallocate to enable assignment
        cms[name] = list(range(cv.get_n_splits()))

        empty_clf = make_pipeline(
            *preprocessing_steps,
            clf,
        )

        for k, (ii_train, ii_test) in enumerate(cv.split(X, y)):

            this_clf = deepcopy(empty_clf)

            this_clf.fit(X[ii_train], y[ii_train])

            y_pred = this_clf.predict(X[ii_test])
            y_test = y[ii_test]

            scores = {
                "score": balanced_accuracy_score(y_test, y_pred),
                "f1": f1_score(y_test, y_pred, average="weighted"),
                "precision": precision_score(y_test, y_pred, average="weighted"),
                "recall": recall_score(y_test, y_pred, average="weighted"),
            }

            fitted_classifiers[name][k] = this_clf

            for score_name, score in scores.items():
                if score_name not in performance[name].keys():
                    performance[name][score_name] = [score]
                else:
                    performance[name][score_name].append(score)

            cm = confusion_matrix(y_test, y_pred, labels=y_unique)
            cms[name][k] = cm

            print(f"{name:10} ({k}): {scores['score']:.4f}")

    if return_cm_labels:
        return fitted_classifiers, performance, cms, y_unique
    else:
        return fitted_classifiers, performance, cms


def make_subcondition_confusion_matrix(
    true_labels_mapped,
    predicted_labels,
    y_true_unique_labels=None,
    y_pred_unique_labels=None,
    prob=True,
):
    import numpy as np
    from sklearn.metrics import confusion_matrix

    # define unique labels if not provided
    def _check_unique_labels(unique_labels, all_labels):
        if unique_labels is None:
            unique_labels = np.unique(all_labels)
        else:
            assert len(unique_labels) == len(
                set(unique_labels)
            ), "unique labels inputs must be unique!"

        return unique_labels

    y_true_unique_labels = _check_unique_labels(
        y_true_unique_labels, true_labels_mapped
    )
    y_pred_unique_labels = _check_unique_labels(y_pred_unique_labels, predicted_labels)

    assert len(set(y_true_unique_labels).intersection(set(y_pred_unique_labels))) == 0
    "Can't have overlap between context & predicted labels! Sorry, it breaks stuff."

    all_labels = list(y_true_unique_labels) + list(y_pred_unique_labels)

    # Compute confusion matrix
    cm = confusion_matrix(true_labels_mapped, predicted_labels, labels=all_labels)

    cm = cm[
        : -1 * len(y_pred_unique_labels), -1 * len(y_pred_unique_labels) :
    ]  # remove unused other labels. this probably returns an error if any true keys in pred keys

    if prob:
        cm = (cm.T / cm.sum(1)).T
        cm[np.isnan(cm)] = 0  # columns with no predictions

    return cm


def subcondition_confusion_matrix_plot(
    cm,
    y_pred_unique_labels,
    y_true_unique_labels,
    ax=None,
    cmap="magma",
    **heatmap_kwarg,
):
    import numpy as np
    import matplotlib.pyplot as plt
    from seaborn import heatmap

    # Plotting the confusion matrix
    if ax is None:
        fig, ax = plt.subplots()

    heatmap(
        cm,
        ax=ax,
        annot=True,
        xticklabels=y_pred_unique_labels,
        yticklabels=y_true_unique_labels,
        cmap=cmap,
        **heatmap_kwarg,
    )
    plt.yticks(rotation=0)

    ax.set(
        ylabel="True label",
        xlabel="Predicted label",
    )

    return ax


def print_performance(performance):
    import numpy as np
    import pandas as pd

    if isinstance(list(performance.values())[0], dict):
        df = pd.DataFrame.from_dict(performance, orient="index").applymap(np.mean)
        df.index.name = "Model Name"
        print(df)

    else:  # old, only stored balanced accuracy score
        print(f"{'Classifier':18}| Score")

        for k, v in sorted(
            performance.items(),
            key=lambda item: np.mean(item[1]),
            reverse=True,
        ):
            print(f"{k:18}| {np.mean(v):.4f}")
