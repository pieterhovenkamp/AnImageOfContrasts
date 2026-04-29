import numpy as np
import pandas as pd
import sklearn.metrics


def calc_metric_for_evaluated_df(df, true_name_label='true_name', pred_name_label='pred_name'):
    """

    :param df: DataFrame - needs to contain the columns with names of true_name_label and pred_name_label
    :param true_name_label: str - name of column in df containing the true names
    :param pred_name_label: str - name of column in df containing the predicted names
    :return: tuple (float, float, float, float, float) - metrics in [0, 1]
    """
    # Calculate accuracy & balanced accuracy
    acc = sklearn.metrics.accuracy_score(df[true_name_label], df[pred_name_label])
    bal_acc = sklearn.metrics.balanced_accuracy_score(df[true_name_label], df[pred_name_label], sample_weight=None,
                                                      adjusted=False)
    prec = sklearn.metrics.precision_score(df[true_name_label], df[pred_name_label], average='weighted',
                                           zero_division=0)
    bal_prec = sklearn.metrics.precision_score(df[true_name_label], df[pred_name_label], average='macro',
                                               zero_division=0)
    f1 = sklearn.metrics.f1_score(df[true_name_label], df[pred_name_label], average='macro',
                                  zero_division=0)

    return acc, bal_acc, prec, bal_prec, f1


def calc_metric_per_group_for_evaluated_df(df, true_name_label='true_name', pred_name_label='pred_name'):
    """
    Calculate a new DataFrame with the precision and recall of the group for each group that
    appears in either true_name_label or in pred_name_label.

    If a group has zero predictions, precision for this group is not defined and is set to 0
    without warning whereas recall is 0 by definition. If on the other hand, a group has zero
    true names, then the recall is not defined and is set to 0 without warning, whereas
    precision is 0 by definition.

    :param df: DataFrame - needs to contain the columns with names of true_name_label and pred_name_label
    :param true_name_label: str - name of column in df containing the true names
    :param pred_name_label: str - name of column in df containing the predicted names
    :return: DataFrame - with columns:
                        - 'name': because a name in this column can be either from true_name_label
                                  or pred_name_label, we omit the wording 'true' and 'predicted in
                                  this column name
                        - 'recall' in [0, 1]
                        - 'precision' in [0, 1]
                        - 'f1-score' in [0, 1]
                        - n_true_name_label: number of occurrences of name in true_name_label
                        - n_pred_name_label: number of occurrences of name in pred_name_label
    """
    # We make a sorted array of all labels that appear in either true_name_label or pred_name_label
    names = np.union1d(df[true_name_label].unique(), df[pred_name_label].unique())
    names = np.sort(names)

    # We also add the number of occurrences per group for both the true names and predictions
    n_true_names = np.array([np.sum(df[true_name_label] == name) for name in names])
    n_pred_names = np.array([np.sum(df[pred_name_label] == name) for name in names])

    # By setting the argument 'labels', we guarantee that the order of the arrays matches
    # the order of the names.
    recall = sklearn.metrics.recall_score(df[true_name_label], df[pred_name_label],
                                          labels=names, average=None, zero_division=0)
    precision = sklearn.metrics.precision_score(df[true_name_label], df[pred_name_label],
                                                labels=names, average=None, zero_division=0)
    f1 = sklearn.metrics.f1_score(df[true_name_label], df[pred_name_label],
                                  labels=names, average=None, zero_division=0)

    return pd.DataFrame({'name': names,
                         'recall': recall,
                         'precision': precision,
                         'f1_score': f1,
                         f"n_{true_name_label}": n_true_names,
                         f"n_{pred_name_label}": n_pred_names})
