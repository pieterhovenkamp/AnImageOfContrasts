import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
import sklearn.metrics

from cnn_evaluate.calc_classifier_metrics import calc_metric_per_group_for_evaluated_df


# Modified 20260420
def heatmap(data, text_labels, ax=None, show_cbar=None, contains_prec_recall=False, colour_sum_axis='x',
            cbar_kw={}, cbarlabel="", hide_zeros=True, hide_numbers_diagonal=False, text_kw={},
            hide_colour_precision_recall=False, # ADDED
            **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels. If contains_prec_recall=True,
    then a separate color mask is calculated for the 'group-values' and the precision/recall-values.

    From: https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py
    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    text_labels
        Either:
        - list or array of length N with the labels for the rows and columns
        - tuple of list or array of length N, with the labels for the x (1st) and y axis (2nd)
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """
    if text_kw == None:
        text_kw = {}

    if not ax:
        ax = plt.gca()

    colour_scale_separate = False
    if contains_prec_recall:
        # if colour_scale_separate, a colour mapping is used based on all the numbers from precision/recall only
        # (ranging from 0 to 100), and from the numbers of the separate predictions (which can range higher than 100
        # depending on the numbers of labels.
        if colour_scale_separate:
            # Make a mask of the values of the groups
            groups_mask = np.zeros_like(data).astype(bool)
            groups_mask[:-1, :-1] = True

            # Precision/recall mask (except the corner). By default, corner value is plotted in white
            prec_recall_mask = ~groups_mask
            prec_recall_mask[-1, -1] = False

            for mask in [groups_mask, prec_recall_mask]:
                data_copy = data.copy()
                data_copy[~mask] = np.nan

                im = ax.imshow(data_copy, **kwargs, alpha=mask.astype(float))
                annotate_heatmap(im, use_mask=mask.astype(bool), **text_kw)

        # if False, then we create a single colour mapping for the whole figure by rescaling the separate predictions
        # divided by the total number of predictions ('y') or the total number of true labels ('x') in a group. Note
        # that this ONLY applies to the colour mapping, the numbers (text) is not rescaled.
        else:
            # We sum along the specified axis and divide all entries by this sum. This gives a copy of the
            # entries scaled to the number of predictions in each label, which we want for the heatmap colours.
            # Naturally, we do not modify the precision and recall values as these are percentages already
            data_scaled = data.copy()

            # We want to replace zero-sums along the axis with nans - this occurs when no true labels or predictions
            # exist for a group
            sum_axis = np.sum(data[:-1, :-1], axis=1 if colour_sum_axis == 'x' else 0)
            sum_axis = np.where(sum_axis == 0, np.nan, sum_axis)  # First we set nans

            for i in range(data.shape[0] - 1):
                if colour_sum_axis == 'y':
                    data_scaled[i, :-1] = (data_scaled[i, :-1] / sum_axis) * 100
                elif colour_sum_axis == 'x':
                    data_scaled[:-1, i] = (data_scaled[:-1, i] / sum_axis) * 100

            data_scaled = np.nan_to_num(data_scaled)  # We replace the nans by zero values

            # We mask the corner value (otherwise a zero is plotted there)
            corner_mask = (np.zeros_like(data) + 1).astype(bool)
            corner_mask[-1, -1] = False

            if hide_zeros:
                # We create a boolean mask with False at zero-counts and True at all precision/recall values
                zero_mask = data != 0
                zero_mask[-1, :] = True
                zero_mask[:, -1] = True
                zero_mask[-1, -1] = False
                mask = zero_mask
            else:
                mask = corner_mask

            # ADDED
            if hide_colour_precision_recall:
                prec_recall_mask = (np.zeros_like(data) + 1).astype(bool)
                prec_recall_mask[-1, :] = False
                prec_recall_mask[:, -1] = False

                corner_mask = prec_recall_mask

            # ADDED
            if hide_numbers_diagonal:
                for i in range(data.shape[1]):
                    mask[i, i] = False

            im = ax.imshow(data_scaled, **kwargs, alpha=corner_mask.astype(float))
            annotate_heatmap(im, data=data, use_mask=mask,
                             # threshold=90,
                             threshold=90 if not hide_colour_precision_recall else 101,  # ADDED
                             textcolors=('black', 'white'), **text_kw)
    else:
        if hide_zeros:
            # We create a boolean mask with False at zero-counts
            zero_mask = data != 0
            im = ax.imshow(data, alpha=zero_mask.astype(float), **kwargs)
        else:
            im = ax.imshow(data, **kwargs)
            zero_mask = None

        if 'use_mask' in text_kw.keys():
            annotate_heatmap(im, data=data, **text_kw)
        else:
            annotate_heatmap(im, data=data, use_mask=zero_mask, **text_kw)

    # Create colorbar
    if show_cbar:
        cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
        cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    else:
        cbar = None

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    if type(text_labels) == tuple:
        ax.set_xticklabels(text_labels[0])
        ax.set_yticklabels(text_labels[1])
    else:
        ax.set_xticklabels(text_labels)
        ax.set_yticklabels(text_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1] + 1) - .5, minor=True)
    ax.set_yticks(np.arange(data.shape[0] + 1) - .5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im, cbar


def annotate_heatmap(im, data=None, valfmt="{x:.0f}",
                     textcolors=("black", "white"),
                     threshold=None, use_mask=None, **textkw):
    """
    A function to annotate a heatmap.
    Adapted from:
    https://matplotlib.org/stable/gallery/images_contours_and_fields/image_annotated_heatmap.html#sphx-glr-gallery-images-contours-and-fields-image-annotated-heatmap-py

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A pair of colors.  The first is used for values below a threshold,
        the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max()) / 2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            # Based on use_mask we skip the annotation of data that is marked as False
            if (use_mask is not None) and (not use_mask[i, j]):
                continue

            # kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            kw.update(color=textcolors[int(im.norm(im.get_array()[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts


def calc_confusion_matrix_for_val_df(val_df, true_name_label='true_name', pred_name_label='pred_name',
                                     add_precision_recall=False, return_names=True, names=None):
    """
    Calculate the confusion matrix by comparing the true names versus the predictions in val_df.
    Confusion matrix is calculated for each group that appears in either true_name_label or
    in pred_name_label and the output is sorted by the name of the group.

    If a group has zero predictions, precision for this group is not defined and is set to 0
    without warning whereas recall is 0 by definition. If on the other hand, a group has zero
    true names, then the recall is not defined and is set to 0 without warning, whereas
    precision is 0 by definition.

    :param val_df: DataFrame - needs to contain columns true_name_label and pred_name_label
    :param true_name_label: str
    :param pred_name_label: str
    :param add_precision_recall: bool
    :param return_names: bool
    :param names: list - if defined, the confusion matrix is only calculated for this set of labels. If None, all labels
                         that appear in either true_name_label or pred_name_label are taken.
    :return: (confusion matrix, labels) - np.array (float), np.array (str)
    """

    # We make a sorted array of all labels that appear in either true_name_label or pred_name_label
    names_unique = np.union1d(val_df[true_name_label].unique(), val_df[pred_name_label].unique())
    names_unique = np.sort(names_unique)

    if not names:
        names = names_unique
    else:
        names_not_in_val_df = [name for name in names if name not in names_unique]
        if len(names_not_in_val_df):
            raise ValueError("The following labels are in argument 'names' but not present in val_df",
                             *names_not_in_val_df)

    # By setting the argument 'labels', we guarantee that the order of the arrays matches
    # the order of the names.
    conf_matrix = sklearn.metrics.confusion_matrix(val_df[true_name_label],
                                                   val_df[pred_name_label],
                                                   labels=names)

    if add_precision_recall:
        df_metrics = calc_metric_per_group_for_evaluated_df(val_df,
                                                            true_name_label=true_name_label,
                                                            pred_name_label=pred_name_label)
        df_metrics.index = df_metrics['name']

        # This way we guarantee that the order of precision and recall is the same as in names
        precision = np.array([df_metrics.loc[name, 'precision'] for name in names]) * 100
        recall = np.array([df_metrics.loc[name, 'recall'] for name in names]) * 100

        conf_matrix = np.append(conf_matrix,
                                precision.reshape(1, len(precision)),
                                axis=0)
        conf_matrix = np.append(conf_matrix,
                                np.append(recall, 0).reshape(len(recall) + 1, 1),
                                axis=1)

    if return_names:
        return conf_matrix, names
    else:
        return conf_matrix


# Modified 20260420
def plot_confusion_matrix_for_val_df(val_df, true_name_label='true_name', pred_name_label='pred_name',
                                     add_precision_recall=False, hide_colour_precision_recall=False,
                                     hide_zeros=True, hide_numbers_diagonal=False,
                                     colour_sum_axis='x',
                                     names=None, line_before_names=None, line_colour='gray', line_alpha=0.5,
                                     linestyle='--', text_kw=None, y_label_right=False,
                                     return_ax=False, **fig_kwargs):
    """
    Calculate the confusion matrix of val_df using calc_confusion_matrix_for_val_df and
    plot as a table with a heatmap. If add_precision_recall = True, include an extra
    row/column for recall/precision (in %).

    :param val_df: DataFrame - DataFrame with the ground truth labels and the predicted labels per image. Needs to
                               contain columns true_name_label and pred_name_label
    :param true_name_label: str - name of the column in val_df with the ground truth labels
    :param pred_name_label: str - name of the column in val_df with the predicted labels
    :param add_precision_recall: bool
    :param names: list - if defined, the confusion matrix is only calculated for this set of labels. If None, all labels
                         that appear in either true_name_label or pred_name_label are taken. Can also be used to
                         to specify the order of the groups (otherwise groups are shown in alphabetical order)
    :param hide_zeros: boolean - if True, we do not annotate cells with zeros (recommended)
    :param colour_sum_axis: 'x' or 'y' - heatmap colours of cells are scaled per class with respect to total number of
                                         true labels ('x') or total number of predicted labels ('y')
    :param line_before_names: list - if specified, lines are added to indicate a grouping among classes which can help
                                     when the number of classes is large. Lines are added before the specified names.
    :param linestyle: passed to matplotlib.pyplot.<h/v>lines object
    :param line_alpha: passed to matplotlib.pyplot.<h/v>lines object
    :param line_colour: passed to matplotlib.pyplot.<h/v>lines object
    :param figsize:
    :return: None
    """
    conf_matrix, labels = calc_confusion_matrix_for_val_df(val_df, true_name_label=true_name_label,
                                                           pred_name_label=pred_name_label,
                                                           add_precision_recall=add_precision_recall,
                                                           names=names,
                                                           return_names=True)

    # We need to update the text labels with precision and recall
    if add_precision_recall:
        labels = list(labels)
        labels_x, labels_y = labels.copy(), labels.copy()
        labels_x.append('Recall (%)')
        labels_y.append('Precision (%)')
        labels = (labels_x, labels_y)
    else:
        labels_x, labels_y = labels, labels

    fig, ax = plt.subplots(**fig_kwargs)
    heatmap(conf_matrix, text_labels=labels, ax=ax, cmap='Oranges', colour_sum_axis=colour_sum_axis,
            contains_prec_recall=add_precision_recall, hide_colour_precision_recall=hide_colour_precision_recall, hide_zeros=hide_zeros, hide_numbers_diagonal=hide_numbers_diagonal, text_kw=text_kw)

    # If specified, we plot a line before these groups in order to visualise structure within the groups
    if line_before_names:
        # We determine the coordinates of the specified labels and take the position halfway between the previous label
        line_locs_x = [(i + (i - 1)) / 2 for i, label in enumerate(labels_y) if label in line_before_names]
        line_locs_y = [(i + (i - 1)) / 2 for i, label in enumerate(labels_y) if label in line_before_names]

        ax.hlines(line_locs_y, ax.get_xlim()[0], ax.get_xlim()[1], color=line_colour, alpha=line_alpha, ls=linestyle)
        ax.vlines(line_locs_x, ax.get_ylim()[0], ax.get_ylim()[1], color=line_colour, alpha=line_alpha, ls=linestyle)

    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")

    if y_label_right:
        ax.yaxis.set_label_position("right")

    if return_ax:
        return fig, ax
    else:
        fig.tight_layout()
        plt.show()
        return

