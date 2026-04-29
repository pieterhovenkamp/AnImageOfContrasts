#!/usr/bin/env python3
import os
from pathlib import Path
import sys

import cv2
import numpy as np
from skimage import util

from general_utils.pickle_functions import *
from general_utils.parse_folder import parse_folder_to_df

"""
This script contains functions to load avi-files and convert all or a subset to individual images

Convention: the ff_counter within an avi-file starts at 1 (not 0)
"""


def read_ff_from_avi_file(avi_path, ff_counter=None, as_float=False, as_float32=False):
    """
    Load a single or multiple fullframe images from an avi-file, for selected ff_counters. The first fullframe image
    in an avi-file has an ff_counter of 1 (not 0).

    Regardless of ff_counter, all images are temporarily loaded from the disk, so in case all images need to
    be parsed, it is strongly recommended to loop over avi-files and use ff_counter = None, instead of
    looping over combinations of avi_path and ff_counter.

    !Note that images are NOT returned in the order specified in ff_counter, but in the order in which they appear in
    the avi!

    :param avi_path: str - path to avi-file
    :param ff_counter: int or list of ints or None - if None, all frames in the avi-file are returned
    :param as_float: bool - if True, convert image(s) to float (in [0, 1]) instead of the default uint8 (in [0, 255])
    :param as_float32: bool - if True, convert image(s) to float32 (in [0, 1]) instead of the default uint8. Note that
    as_float32 is only used if as_float is False!
    :return: list - list of images (as np.array) - if ff_counter is a single int, a list of 1 element is returned
    """
    if not os.path.exists(avi_path):
        raise FileNotFoundError(f"File {avi_path} was not found")

    video = cv2.VideoCapture(str(avi_path))

    # If single ff_counter is given, convert to a list, so we can use 'is in' later on
    if type(ff_counter) == int:
        ff_counter = [ff_counter]

    if ff_counter is not None:
        # We sort the ff_counter list, so that we can break the loop if the maximal counter is encountered
        ff_counter.sort()
        ff_counter_max = max(ff_counter)

    # We loop over all frames and select only the frames that we requested based on the counter
    i_frame = 1
    ff_list = []
    while True:
        ret, frame = video.read()
        if ret:
            # In this case, all images are returned
            if ff_counter is None:
                # frame = color.rgb2gray(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                # frame = color.rgb2gray(frame)
                if as_float:
                    frame = util.img_as_float(frame)
                elif as_float32:
                    frame = util.img_as_float32(frame)
                ff_list.append(frame)
            # In this case, only selected images are returned
            elif i_frame in ff_counter:
                # frame = color.rgb2gray(frame)
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                if as_float:
                    frame = util.img_as_float(frame)
                elif as_float32:
                    frame = util.img_as_float32(frame)
                ff_list.append(frame)
                # If the maximum counter has been loaded, we break the loop to save (a lot of) time
                if i_frame == ff_counter_max:
                    i_frame += 1
                    break
            i_frame += 1
        else:
            break
    video.release()

    if ff_counter is not None:
        for i in ff_counter:
            # minus 1, because i_frame after the loop is n_frames + 1
            if i > i_frame - 1:
                print(f"ff_counter {i} is larger than the number of frames ({i_frame - 1}) in {avi_path}, "
                      f"so this one is skipped")

    return ff_list


def read_avi_meta_to_df(*avi_paths):
    """
    Make a DataFrame with metadata per avi-file, with columns 'avi_path', 'ff_counter', 'timestamp_avi'

    :param avi_paths: str or *list - filepath or list of filepaths
    :return: pd.DataFrame
    """
    df_list = []
    for avi_path in avi_paths:
        video = cv2.VideoCapture(str(avi_path))
        n_frames = count_avi_frames(avi_path)
        video.release()

        # timestamp = pd.to_datetime(avi_path, format='%Y%m%dT%H%M%S.%f')
        df_sub = pd.DataFrame({'avi_path': avi_path, 'ff_counter': range(1, n_frames + 1), 'timestamp_avi': 'dummy'})
        df_list.append(df_sub)

    return pd.concat(df_list, ignore_index=True)


def parse_fullframes_from_avi_df(avi_df, count_all_avis=True):
    """
    From a pd.DataFrame with the avi-files, read the timestamps of the individual images and create a new
    dataframe for these. Input dataframe needs to have columns 'avi_name', 'dirpath', 'avi_path'

    This costs 7s for a file of 162 fullframes, but the advantage
    is that in the rest of the pipeline, one can be sure that all files can be opened.

    Assumes avi_name is of form <counter>_YYYYMMDD_hhmmss.f.avi. Names of the fullframe images (column 'image_name')
    are created as: <YYYYMMDD_hhmmss.fff>_img<ff_counter>.avi

    Column 'image_path' is created as <avi_path>/<image_name>. This is a bit artificial since these are not
    existing paths that can be loaded directly, but the avi-paths are easily obtained from this by dropping
    the <image_name>. And the rationale behind this is that it is possible throughout the pipeline to load
    an image using only the image_path, instead of requiring both the avi_path and the image_name.

    !Currently, if no fullframes could be read from an avi, a single row is added to the DataFrame with all NaNs except
    for 'avi_name'. This can cause issues later on, so reconsider this.

    :param avi_df: pd.DataFrame
    :param count_all_avis: bool - if True, the number of fullframes for all avi-files is counted separately.
                                  if False, only the first and last avi-files in the folder are counted and
                                  the rest of the files is inferred from the first file.
    :return: pd.DataFrame - with columns 'avi_name', 'image_name', 'Time', 'image_path', 'ff_counter', 'dirpath'
    """
    # Get the start time from the avi-file
    avi_df['avi_time'] = avi_df['avi_name'].apply(lambda x: Path('_'.join(x.split(sep='_')[1:])).stem)
    avi_df['avi_time'] = pd.to_datetime(avi_df['avi_time'],
                                        format='%Y%m%d_%H%M%S.%f')

    df_list = []
    for dirpath, df_grouped in avi_df.groupby('dirpath'):
        df_grouped.sort_values(by='avi_time', inplace=True, ascending=True)

        if len(df_grouped) <= 1:
            print(f"Directory {dirpath} contains only 1 avi, therefore timestamp inference is not implemented yet.\n"
                  f"So this one is skipped.")
            continue

        # Calculate the time increment between fullframe images within the avi, based on the timestamp between
        # consecutive avi files and the number of fullframes in an avi file
        df_grouped['avi_time_next'] = df_grouped['avi_time'].shift(-1)
        df_grouped['Time_delta'] = df_grouped['avi_time_next'] - df_grouped['avi_time']

        if count_all_avis:
            df_grouped['n_fullframes'] = df_grouped['avi_path'].apply(lambda x: count_avi_frames(x, verbose=True))
        else:
            # We count the number of fullframes per avi of the first and last file in the folder (because the dataframe
            # is sorted by time). For each deployment, the number of images per avi's is constant, only the last file
            # can be less since the deployment is aborted there. Therefore, we extrapolate the number of files of the
            # first file to all except for the last file.
            df_grouped['n_fullframes'] = int(count_avi_frames(df_grouped['avi_path'].iat[0]))
            df_grouped['n_fullframes'].iat[-1] = int(count_avi_frames(df_grouped['avi_path'].iat[-1]))

        # Calculate the time-delta for each ff as the time difference between the two avi-files over
        # the number of fullframes
        df_grouped['Time_delta_per_ff'] = df_grouped['Time_delta'] / df_grouped['n_fullframes']

        # For the last avi file we cannot infer the time-delta from the difference with the next file, so we copy
        # time-delta from the previous file
        df_grouped['Time_delta_per_ff'].iat[-1] = df_grouped['Time_delta_per_ff'].iat[-2]

        # Make a DataFrame with the individual fullframes per avi
        for i, row in df_grouped.iterrows():
            if row['n_fullframes']:
                df_sub = pd.DataFrame({'image_path': row['avi_path'],
                                       'avi_name': row['avi_name'],
                                       'ff_counter': range(1, row['n_fullframes'] + 1),
                                       'dirpath': row['dirpath']
                                       })

                # Calculate the timestamps
                df_sub['Time'] = row['avi_time'] + row['Time_delta_per_ff'] * (df_sub['ff_counter'] - 1)

                # Create a name for each fullframe - format: YYYMMDD_hhmmss.f_<counter>.avi
                df_sub['image_name'] = df_sub.apply(
                    lambda df: f"{df['Time'].strftime('%Y%m%d_%H%M%S.%f'):.19}_img{df['ff_counter']:03}.avi", axis=1)

                # We (artificially) name the image paths as '<avi_path>/<image_name>'
                df_sub['image_path'] = df_sub.apply(
                    lambda df: f"{df['image_path']}/{df['image_name']}", axis=1)
            else:
                # If no fullframes could be loaded from this avi
                df_sub = pd.DataFrame({
                    'avi_name': row['avi_name'],
                    'ff_counter': np.nan,
                    'dirpath': row['dirpath'],
                    'Time': np.nan,
                    'image_name': np.nan,
                    'image_path': np.nan,
                }, index=[0])

            df_list.append(df_sub)

    if len(df_list):
        return pd.concat(df_list, ignore_index=True)
    else:
        return pd.DataFrame({
                    'avi_name': [],
                    'ff_counter': [],
                    'dirpath': [],
                    'Time': [],
                    'image_name': [],
                    'image_path': []})


def parse_fullframes_from_avi_folder(dir, count_all_avis=True):
    """
    Parse all avi-files in folder and its subdirs and per avi-file parse the images.

    :param dir: str
    :return: pd.Dataframe - with columns 'image_name', 'Time', 'image_path', 'ff_counter', 'dirpath' for each fullframe image
    """
    avi_df = parse_folder_to_df(dir=dir, extension='avi')
    avi_df.rename(columns={'filepath': 'avi_path', 'filename': 'avi_name'}, inplace=True)
    return parse_fullframes_from_avi_df(avi_df, count_all_avis=count_all_avis)


def count_avi_frames(avi_path, verbose=False):
    """
    Count number of frames in an avi file, either using cv2.CAP_PROP_FRAME_COUNT (faster) or manually by looping
    over all frames (slower). Manual method is applied if automatic method throws any errors or if override=True

    :param avi_path: str
    :param override: bool
    :return: int
    """
    video = cv2.VideoCapture(str(avi_path))

    # The 'fast way' turned out to be inaccurate, so it is disabled
    # if the override flag is passed in, revert to the manual method of counting frames
    override = True
    if override:
        total = count_avi_frames_manual(video)
    # otherwise, let's try the fast way first
    # else:
        # We try to determine the number of frames in a video via video properties; this method can be very buggy
        # and might throw an error based on your OpenCV version or may fail entirely based on your which video codecs
        # you have installed. In case we got an error, we revert to counting manually
        # try:
        #     total = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        #     print(video.get(cv2.CAP_PROP_FRAME_COUNT))
        # except:
        #     total = count_avi_frames_manual(video)
    video.release()

    if verbose:
        print(f"Counted {total} frames in {avi_path}")

    return total


def count_avi_frames_manual(avi_video):
    """
    Count the number of frames in an avi-video loaded as with cv2.VideoCapture. Looping over the frames as done
    in this function is less efficient, but more fool-proof.

    :param avi_video:
    :return: int - total number of frames present in avi_video
    """
    total = 0

    # Loop over the frames of the video
    while True:
        grabbed, frame = avi_video.read()
        # Check to see if we have reached the end of the video
        if not grabbed:
            break
        total += 1

    return total


def add_avi_paths_from_image_paths(df, image_path_col='image_path'):
    """
    :param df: pd.DataFrame - needs to contain the column named as image_path_col
    :return: df - copy of the input dataframe with an additional column 'avi_path'
    """
    df['avi_path'] = df[image_path_col].apply(lambda x: Path(x).parent if Path(x).suffix == '.avi' else np.nan)
    return df.copy()


def add_ff_counter_from_image_names(df, image_name_col='image_name'):
    """
    :param df: pd.DataFrame - needs to contain the column named as image_name_col
    :return: df - copy of the input dataframe with an additional column 'ff_counter'
    """
    df['ff_counter'] = df[image_name_col].apply(lambda x: int(Path(x).stem[-3:]) if Path(x).suffix == '.avi' else np.nan)
    return df.copy()
