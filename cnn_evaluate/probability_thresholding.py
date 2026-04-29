#!/usr/bin/env python3

import pandas as pd
import numpy as np

from general_utils.pickle_functions import load_dict


def apply_probability_filters_to_df(df_labels, default_threshold=0.,
                                    threshold_dict_file=None, df_thresholds=None,
                                    label_col='label'):
    """
    We apply the probability thresholds to the labels in df_labels. Probability thresholds are loaded
    from the file threshold_dict_file, which should be a saved dictionary with as keys the labels and
    as values the thresholds for each label. Labels should match exactly with the labels in df_labels.
    If a label is not in threshold_dict_file, the threshold will be set to default_threshold.

    :param df_labels: pd.DataFrame - requires column label_col
    :param df_threshold: pd.DataFrame - DataFrame with an index entry for each label and a column 'threshold'.
    :param threshold_dict_file: str - path to saved dictionary with as keys the labels and as values the thresholds.
                                      Ignored if threshold_dict is specified.
    :param default_threshold: float - default threshold value between 0 and 1
    :return pd.DataFrame - with new column 'threshold' (float), 'above_threshold' (boolean)
    """
    if df_thresholds is None:
        if threshold_dict_file is not None:
            threshold_dict = load_dict(threshold_dict_file, args_as_float=True)

            df_thresholds = pd.DataFrame({'label': threshold_dict.keys(), 'threshold': threshold_dict.values()})
        else:
            raise ValueError('Either threshold_dict or threshold_dict_file must be specified')
    else:
        df_thresholds = df_thresholds.reset_index(names='label')

    # We add the thresholds to the DataFrame
    df_labels_filter = pd.merge(df_labels, df_thresholds, on=label_col, how='left')

    # If no threshold is defined, we set the default threshold
    df_labels_filter.fillna({'threshold': default_threshold}, inplace=True)

    # We indicate which ROIs are above or below the threshold
    df_labels_filter['above_threshold'] = ~ (df_labels_filter['softmax'] <= df_labels_filter['threshold'])

    return df_labels_filter


def calc_threshold_dict(val_df, threshold_reqs, default_threshold=0.3):
    """
    We calculate a threshold per group based on a set of precision thresholds with respect to other specified groups.
    There can be multiple threshold requirements per group, then the final threshold is the maximum (per label) of these.

    Requirements should be specified in a list as follows:
    threshold_reqs = [
        # Predicted groups                                     # True groups                   # Required precision for this combination
        [['living_group1', 'living_group2', 'living_group3'], ['non_living1', 'non_living2'], 'max'],
        [['living_group1', 'living_group4', 'living_group5'], ['living_group1', 'living_group4'], 0.9],
        [['group6', 'group7'], ['non_living1'], 0.9],
        ]

    In this example, thresholds for will be calculated for groups 'living_group1', 'living_group2', 'living_group3', 'living_group4', 'living_group5', 'group6', 'group7' such that:
    - false predictions from 'non_living1' and 'non_living2' into 'living_group1', 'living_group2' and 'living_group3' are zero AND
    - precision of each of 'living_group1', 'living_group4' and 'living_group5' with respect to 'living_group1' and 'living_group4' is > 90% AND
    - precision of each of 'group6' and 'group7' with respect to 'non_living1' is > 90%

    :param val_df: pd.DataFrame - validation set with one entry per image. Requires columns 'true_name', 'label', 'softmax'
    :param threshold_reqs:
    :param default_threshold: float between 0.0 and 1.0 - if for a group no threshold can be determined based on the above, this default threshold will be used.
    :return: pd.DataFrame - a DataFrame with an index entry for each group specified as predicted group in threshold_reqs and a column 'threshold'.
    """
    # We construct an empty DataFrame of the labels and softmax threshold
    df_threshold = pd.DataFrame(index=val_df['true_name'].unique(), columns=['threshold'])

    # We loop over the requirements - note that there can be multiple requirements per label
    for item in threshold_reqs:
        pred_groups, true_groups, req = item

        threshold_dict = {}
        for group, df_sub in val_df.loc[val_df['label'].isin(pred_groups)].groupby('label'):
            if req == 'max':
                # If 'max', we consider the subset where the true name is in one of these groups, and take the maximum softmax of this subset
                threshold = df_sub.loc[df_sub['true_name'].isin(true_groups), 'softmax'].max()
            else:
                if req < 0. or req > 1.:
                    raise ValueError(f"The precision threshold should be between 0 and 1, but current value is: {req}")

                df_sub = df_sub.loc[df_sub['true_name'].isin([group] + true_groups)].copy()

                if sum(mask_incorrect := df_sub['true_name'] != group):
                    df_sub = df_sub.sort_values(by='softmax').copy()
                    df_sub['correct'] = ~mask_incorrect
                    df_sub['incorrect'] = mask_incorrect

                    df_sub['correct_count_cum'] = df_sub['correct'].sum() - df_sub['correct'].cumsum()
                    df_sub['incorrect_count_cum'] = df_sub['incorrect'].sum() - df_sub['incorrect'].cumsum()
                    df_sub['precision_per_softmax'] = df_sub['correct_count_cum'] / (df_sub['correct_count_cum'] + df_sub['incorrect_count_cum'])

                    if sum(df_sub['precision_per_softmax'] > req):
                        threshold = np.round(df_sub.loc[df_sub['precision_per_softmax'] > req, 'softmax'].iat[0], 3) + 0.001
                    else:
                        print(f"No softmax threshold exists such that precision is above {req} for {group} with true groups {true_groups}")
                        threshold = default_threshold
                else:
                    threshold = default_threshold

            threshold_dict[group] = threshold

        # We combine the requirements per label by taking the pair-wise maximum. We use a fill-value of -1, so that the maximum
        # value always exists and then replace the -1 values (which remain when no threshold was defined by the default threshold
        df_threshold = df_threshold.combine(
            pd.DataFrame.from_dict(threshold_dict, orient='index', columns=['threshold']),
            np.maximum, fill_value=-1,
        ).replace(-1, default_threshold)

    return df_threshold