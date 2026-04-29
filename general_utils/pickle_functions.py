#!/usr/bin/env python3

import os
import pandas as pd
from pathlib import Path
import sys
import warnings

pd.set_option("display.max_columns", None)

"""
Import this local script at the start of all files within this project 
"""

def read_pickle(pkl_name, dir, verbose=0):
    if dir is None:
        raise ValueError("A directory should be specified")

    df = pd.read_pickle(pkl_name := f"{dir}/{pkl_name}")

    if verbose == 1:
        print(f"DataFrame was loaded from {pkl_name}")

    return df


def to_pickle(df, pkl_name, dir, verbose=1, overwrite=True):
    """

    :param df:
    :param pkl_name:
    :param dir: str - if None, pkl_name should be absolute
    :param verbose:
    :param overwrite: bool - if True, overwrite any existing pkl at the specified location without warning. If False,
                             an exception is raised in that case.
    :return:
    """
    if dir is None:
        raise ValueError("A directory should be specified")

    pkl_name = f"{dir}/{pkl_name}"

    if os.path.isfile(pkl_name) and not overwrite:
        raise FileExistsError(f"A pkl at {pkl_name} already exists. Specify overwrite=True if you want to overwrite")
    else:
        df.to_pickle(pkl_name)

    if verbose == 1:
        print(f"DataFrame saved to {pkl_name}")


def save_dict(dictionary, path):
    """
    Save a dictionary to a txt.file with for each key in a separate line, followed by
    a double dot and then the value, e.g:
    key1: value1
    key2: value2
    ...

    Use load_args_dict to load back the original dictionary

    :param dictionary:
    :param path:
    :return:
    """
    with open(path, "w") as file:
        for key, val in dictionary.items():
            file.write(f"{key}: {str(val)} \n")


def load_dict(filename, args_as_float=False):
    with open(filename, 'r') as file:
        args_dict = {}
        for line in file:
            sep = line.find(': ')
            arg, val = line[:sep], line[sep + 1:]
            val = val.strip()

            if args_as_float:
                val = float(val)

            args_dict[arg.strip()] = val

    return args_dict