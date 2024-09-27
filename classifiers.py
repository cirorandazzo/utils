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
    first = np.flatnonzero((x[1:] > 0) & (x[:-1] < 0))[
        0
    ]  # first Ca frame post-syl onset

    for col in ["dff", "deriv", "x"]:
        row.loc[col] = row.loc[col][..., first - pre_onset : first + post_onset]

    return row


def plot_all_cms(cms, cm_folder, y, nrows, ncols):
    import os

    import matplotlib.pyplot as plt
    import numpy as np

    from .plot import confusion_matrix_plot

    multi_fig, multi_ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(16, 8))

    for i, (clf_name, all_cms) in enumerate(cms.items()):
        cm = np.array(all_cms).sum(0)  # add all CV cms together

        # plot cross-val avg on multifig
        ax = np.ravel(multi_ax)[i]

        confusion_matrix_plot(
            cm,
            labels=sorted(np.unique(y)),
            prob=True,
            ax=ax,
            colorbar=False,
            values_format=".2f",
        )

        ax.set(title=clf_name)

        # plot cross-val avg separately
        ax = confusion_matrix_plot(
            cm,
            labels=sorted(np.unique(y)),
            prob=True,
        )

        tstr = f"ALL-{clf_name}"
        ax.set(title=tstr)

        fig = ax.get_figure()
        fig.savefig(os.path.join(cm_folder, f"{tstr}.png"))
        plt.close(fig)

        # plot each crossfold
        for i, cm in enumerate(all_cms):
            ax = confusion_matrix_plot(
                cm,
                labels=sorted(np.unique(y)),
                prob=True,
                values_format=".2f",
            )

            tstr = f"{clf_name} ({i})"
            ax.set(title=tstr)

            fig = ax.get_figure()
            fig.savefig(os.path.join(cm_folder, f"{tstr}.png"))
            plt.close(fig)

    for empty_ax in np.ravel(multi_ax)[len(cms) - np.size(multi_ax) :]:
        empty_ax.axis("off")

    multi_fig.tight_layout()
    multi_fig.savefig(os.path.join(cm_folder, f"!cm_all_classifiers.png"))
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
    **unused_kw,
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

            # plot
            fig, ax = plt.subplots()

            ax = subcondition_confusion_matrix_plot(
                cm,
                ax=ax,
                y_pred_unique_labels=y_pred_unique,
                y_true_unique_labels=unique_trans,
                fmt=fmt,
            )

            tstr = f"{clf_name} ({i_cv})"
            ax.set(ylabel="True label in context", title=tstr)

            fig.savefig(os.path.join(pth, f"{tstr}.png"))
            plt.close()

        cm = np.array(tr_cms[clf_name]).sum(0)  # add all cms together

        fig, ax = plt.subplots()
        ax = subcondition_confusion_matrix_plot(
            cm,
            ax=ax,
            y_pred_unique_labels=y_pred_unique,
            y_true_unique_labels=unique_trans,
            fmt=fmt,
        )

        tstr = f"ALL-{clf_name}"
        ax.set(ylabel="True label in context", title=tstr)

        fig.savefig(os.path.join(pth, f"{tstr}.png"))

        plt.close()

    return tr_cms


def plot_all_tr_cms_diffs(
        tr_cms,
        data,
        y,
        trans_map,
        pth,
        fmt="g",
        diff_cmap_extremes=[(0,0,0),(1,0,0)],
        sort_key=None,
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
    plt.savefig(os.path.join(pth, "!example-perfect_tm.png"))
    plt.close()

    # make cmap
    diff_cmap = LinearSegmentedColormap.from_list("Custom", diff_cmap_extremes, N= np.max(tr_cm_true))


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
            vmax=np.max(tr_cm_true)/2,
        )

        tstr = f"!tr_cm_diff-{clf_name}"
        ax.set(ylabel="True label in context", title=tstr)

        fig.savefig(os.path.join(pth, f"{tstr}.png"))
        plt.close()


    return tr_cm_true, tr_cm_diffs
    

def train_models(names, classifiers, cv, X, y):
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
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

    for name, clf in zip(names, classifiers):
        performance[name] = {}
        fitted_classifiers[name] = list(range(cv.get_n_splits()))  # preallocate to enable assignment
        cms[name] = list(range(cv.get_n_splits()))

        empty_clf = make_pipeline(
            StandardScaler(),
            # TODO: try PCA
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

            cm = confusion_matrix(y_test, y_pred)
            cms[name][k] = cm

            print(f"{name:10} ({k}): {scores['score']:.4f}")

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
