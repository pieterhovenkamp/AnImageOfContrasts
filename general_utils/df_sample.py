#!/usr/bin/env python3
import numpy as np

from general_utils.pickle_functions import *


def take_balanced_subsample(df_sub, group_col, target_num, high_or_low_prob):
    """
    We take a subsample of target_num unique entries from df_sub with the highest or lowest softmax probability,
    and with balanced contributions per group_col, provided that enough entries are present for each group.
    If not, entries will be filled with the more abundant groups, so that we always end up with target_num unique entries,
    unless the total number of entries is lower than target_num.

    :param df_sub: pd.DataFrame - with columns group_col, softmax
    :param group_col: str - name of column based on which subsamples are taken, e.g. cast_id, campaign, station
    :param target_num: int - total number of rows that are sampled
    :param high_or_low_prob: str - 'high' or 'low'. Choose whether the highest or lowest softmax scores will be sampled
    :return: pd.DataFrame - a subsample of the input DataFrame with the same columns
    """
    target_num = int(target_num)

    if high_or_low_prob not in ['high', 'low']:
        raise ValueError("Parameter high_ or_low_prob must be 'high' or 'low'")

    if len(df_sub) < target_num:
        return df_sub
    else:
        campaign_counts = df_sub[group_col].value_counts()
        df_sub_sample = pd.DataFrame()

        while len(df_sub_sample) < target_num:
            target_num_it = max(1, int(np.floor((target_num - len(df_sub_sample)) / len(campaign_counts))))
            df_remaining = df_sub.drop(df_sub_sample.index)

            for campaign, df_sub_campaign in df_remaining.groupby(group_col):
                if len(df_sub_campaign) < target_num_it:
                    df_sub_sample = pd.concat([df_sub_sample, df_sub_campaign])
                else:
                    if high_or_low_prob == 'high':
                        df_sub_sample = pd.concat([df_sub_sample, df_sub_campaign.sort_values(by='softmax').tail(target_num_it)])
                    else:
                        df_sub_sample = pd.concat([df_sub_sample, df_sub_campaign.sort_values(by='softmax').head(target_num_it)])

                if len(df_sub_sample) >= target_num:
                    break

        return df_sub_sample


def take_balanced_subsample_per_label(df, label_col='label', group_col='campaign', duplicate_check_col='image_name', target_num=500):
    """
    We take a subsample of target_num unique entries from df_sub of both the highest and lowest softmax probability,
    and with balanced contributions per group_col, provided that enough entries are present for each group.
    If not, entries will be filled with the more abundant groups, so that we always end up with target_num unique entries
    (unless the total number of entries is lower than target_num).

    :param df: pd.DataFrame - with columns group_col, softmax, label_col
    :param label_col: str - name of column with the labels
    :param group_col: str - name of column based on which subsamples are taken, e.g. cast_id, campaign, station
    :param target_num: int - total number of rows that are sampled
    :param duplicate_check_col: str or None - if str, name of column that is checked for duplicate entries in the input DataFrame.
    We do not want duplicate entries, because it would mean that the same image can end up multiple times in the subsample.
    :return: pd.DataFrame - a subsample of the input DataFrame with the same columns
    """
    if duplicate_check_col is not None:
        # We check if the DataFrame contains duplicate entries
        duplicates = len(df.groupby(duplicate_check_col).filter(lambda x: len(x) > 1))
        if duplicates > 0:
            raise ValueError("Duplicate entries were found in the input DataFrame")

    # We take a subsample of the highest softmax scores per group per group_col
    df_high_prob = df.groupby(label_col, group_keys=False).apply(take_balanced_subsample, high_or_low_prob='high',
                                                                 group_col=group_col, target_num=target_num)
    df_high_prob['subsample'] = 'high_prob'

    # We take a subsample of the (remaining) lowest softmax scores per group per group_col
    df_low_prob = df.drop(df_high_prob.index).groupby(label_col, group_keys=False).apply(take_balanced_subsample, high_or_low_prob='low',
                                                                                         group_col=group_col, target_num=target_num)
    df_low_prob['subsample'] = 'low_prob'

    return pd.concat([df_high_prob, df_low_prob], ignore_index=True).reset_index(drop=True)


def sample_from_df(df, col, n=None, **sample_kwargs):
    """
    Convenience function around df.sample that takes a random sample from a DataFrame per entry of column 'col'. If
    the number of sampled items is smaller than the total number of entries, then all entries are taken.

    :param df: pd.DataFrame - needs to contain the column specified in ol
    :param col: str - name of column, or list of multiple columns
    :param n: int - number of items per entry to return
    :param sample_kwargs: keyword arguments of df.sample()
    :return: pd.DataFrame - with same columns as input dataframe
    """
    if not len(df):
        raise ValueError('df is empty')

    df_grouped = df.groupby(by=col)

    def test_func(df, n, **sample_kwargs):
        if len(df) < n:
            return df
        else:
            return df.sample(n=n, **sample_kwargs)

    return df_grouped.apply(test_func, n=n, **sample_kwargs).reset_index(drop=True)


def split_list_in_chunks(input_list, chunksize):
    """
    Split the input list in chunks of the specified size

    :param input_list:
    :param chunksize:
    :return: list
    """
    chunks = range(0, len(input_list), chunksize)
    input_list = list(input_list)
    list_chunks = []
    for i in range(len(chunks) + 1):
        if i == 0:
            continue
        elif i == len(chunks):
            list_chunk = input_list[chunks[i - 1]:]
        else:
            list_chunk = input_list[chunks[i - 1]:chunks[i]]

        list_chunks.append(list_chunk)

    return list_chunks


def split_df_in_chunks(df, chunksize):
    """
    Split a pd.DataFrame into smaller chunks of length 'chunksize'. Return the a list with the subsets of 'df' with the exact same columns as 'df'. If 'chunksize' is smaller
    than or equal to len(df), a list with only 'df' is returned.

    :param df:
    :param chunksize: int
    :return: list with pd.DataFrames, the subsets of the input dataframe. Note that the last element of the list can be smaller than 'chunksize' if len(df) is not a multiple
    of 'chunksize'.
    """
    df_list = []
    chunks = range(0, len(df), int(chunksize))
    for i in range(len(chunks) + 1):
        if i == 0:
            continue
        elif i == len(chunks):
            df_sub = df.iloc[chunks[i - 1]:].copy()
        else:
            df_sub = df.iloc[chunks[i - 1]:chunks[i]].copy()

        df_list.append(df_sub)
    return df_list
