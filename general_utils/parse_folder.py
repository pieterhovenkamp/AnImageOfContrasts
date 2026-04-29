#!/usr/bin/env python3

import glob
import os
from pathlib import Path

import pandas as pd


def parse_folder_to_df(dir, extension=None, verbose=True, add_dot_to_extension=True):
    """
    Searches for all files in a directory.

    :param dir: str - path to dir to be parsed (without slash in the end)
    :param extension: str - file extension with or without the dot prepended. if None, all files will be parsed
                            regardless of the extension
    :param add_dot_to_extension: bool - if True, a dot is prepended to the extension if it is not present already
    :param verbose: bool

    :return: DataFrame - with columns:
        - 'filename'
        - 'filepath' - includes the path of dir (i.e. dir/<image_path>)
        - 'dirpath' of image
    DataFrame is sorted by image_name
    """
    dir = str(dir)

    if not extension:
        extension = ''
    elif not extension.startswith('.') and add_dot_to_extension:
        extension = '.' + extension

    # To check if the dir exists, we need to cut-off any search patterns of asteriks
    if (ind_find := dir.find('*')) != -1:
        dir = dir[:ind_find]

    if not os.path.isdir(dir):
        raise NotADirectoryError(f"No directory was found with name {dir}")

    filepaths = glob.glob((search_pattern := f"{dir}/**/*{extension}"), recursive=True)

    if not len(filepaths):
        raise FileNotFoundError(f"No files were found with search_pattern: {search_pattern}")

    if verbose:
        print(f"{len(filepaths)} files were loaded from {search_pattern}")

    filepath_objects = list(map(lambda x: Path(x), filepaths))
    # dirpaths = list(map(lambda x: x.replace(dir + '/', '').split(sep='/')[0], filepaths))
    files_df = pd.DataFrame({
        'filename': list(map(lambda x: x.name, filepath_objects)),
        'filepath': filepaths,
        'dirpath': list(map(lambda x: x.parent.name, filepath_objects))})
    files_df.sort_values(by=['filename'], ignore_index=True, inplace=True)

    return files_df


def get_filenames_in_dir(dir):
    search_pattern = f"{dir}/*"
    filepaths = glob.glob(search_pattern, recursive=True)
    filenames = list(map(lambda x: os.path.split(x)[-1], filepaths))
    print(filenames)

    df = pd.DataFrame({'dirnames': filenames})
    df.to_excel("tb_logs.xlsx")


def get_all_images(dir, extension, subdir=None, verbose=True):
    """
    Searches for image files in a directory. If parsing the CPICS-archive and subdir=None, then all
    subdirs with a subdir 'rois' are parsed. If a subdir is specified, then only this dir is parsed.
    If data is loaded from an external drive, then only the image path without PATH_TO_EXTERNAL_STORAGE is saved.

    :param dir: str or Path - it is converted it to a string anyway
    :param extension: str - e.g. 'png', 'tiff' or 'jpg'. Disabled: if None, defaults to 'png' if
                            dir = CPICS_ARCHIVE_PATH and to 'tiff' if archive_path = ISIIS_ARCHIVE_PATH
    :param subdir: str
    :param verbose:

    :return: DataFrame - with columns:
        - 'image_name'
        - 'image_path' relative to SMILE_code
        - 'dirpath' of image
    DataFrame is sorted by image_name
    """
    dir = str(dir)

    if subdir:
        search_pattern = f"{dir}/{subdir}"
    else:
        search_pattern = dir

    df = parse_folder_to_df(dir=search_pattern, extension=extension, verbose=verbose)
    df.rename(columns={'filename': 'image_name', 'filepath': 'image_path'}, inplace=True)

    return df
