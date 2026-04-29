#!/usr/bin/env python3

import datetime as dt
import multiprocessing
import os

from general_utils.pickle_functions import *
from general_utils.parse_folder import get_all_images, parse_folder_to_df
from plankton_image_ops.ISIIS_segmentation import run_flatfielding_and_segmentation_on_df, \
    parse_processed_fullframes_from_folder


# 20260215
def run_segmentation_procedure(df, roi_dest_folder, num_proc, segment_kwargs, from_avis, isiis_background_path, reprocess=False,
                               sub_sample=None, test_run=False, processed_folders=None):
    """

    :param df: pd.DataFrame - needs columns 'image_path', 'cast_id', 'image_name'
    :param roi_dest_folder: str
    :param num_proc: int
    :param segment_kwargs:
    :param sub_sample: int or None - number of fullframes or avis that are subsampled from df (useful for testing)
    :param test_run: bool
    :param processed_folders: list
    :param from_avis: bool - whether the fullframes are in avis or in already extracted tiffs. This affects how the
                             split of all files among the parallel processes is made.
    :return:
    """
    print(f"{len(df)} fullframes are present in the input DataFrame")

    if not os.path.isdir(roi_dest_folder):
        os.mkdir(roi_dest_folder)

    if not reprocess:
        if not processed_folders:
            processed_folders = [roi_dest_folder]
        else:
            processed_folders = processed_folders + [roi_dest_folder]

        for processed_folder in processed_folders:
            if os.path.isdir(processed_folder):
                # Parse already processed files
                print(f"Parsing {processed_folder} for already processed files..")
                try:
                    df_processed = parse_folder_to_df(processed_folder, extension='.csv', verbose=False)
                except FileNotFoundError as error:
                    print(error)
                else:
                    df_processed['ff_name'] = df_processed['filename'].apply(lambda x: Path(x).name.replace("_rois.csv", ""))
                    print(f"{len(df_processed)} images were already processed")

                    # Only keep files that have not been processed yet
                    df = df.loc[~df['image_name'].apply(lambda x: Path(x).stem).isin(df_processed['ff_name'])].copy()
                    del df_processed
            else:
                print(f"No folder was found at {processed_folder}")

    if not len(df):
        print("DataFrame is empty - we terminate run_segmentation_procedure")
        return

    if from_avis:
        df['avi_path'] = df['image_path'].apply(lambda x: str(Path(x).parent))

    if sub_sample:
        if from_avis:
            avis_list = df['avi_path'].unique()
            avis_sample = np.random.default_rng(seed=1234).choice(avis_list, size=sub_sample, replace=False,
                                                                  shuffle=False)
            df = df.loc[df['avi_path'].isin(avis_sample)].copy()
        else:
            df = df.sample(n=sub_sample, random_state=1234)

    print(f"Start processing {len(df)} fullframes")
    if test_run:
        print("Stopped run_segmentation_procedure because the parameter test_run = True")
        return

    t1 = dt.datetime.now()
    print(f"{t1}: started segmentation procedure!")
    if num_proc > 1:
        if from_avis:
            # If the images are in avi's, the split of images is made such that all files for an avi end up in the
            # same parallel process. So we need to split the list of avis.
            avis_list = df['avi_path'].unique()
            n_sample = len(avis_list)
        else:
            n_sample = len(df)

        jobs = []
        for i in range(num_proc):
            # Divide the DataFrame in batches for each parallel process, the remainder of this division (in case the
            # size of the DataFrame is not a multiple of num_proc is handled below
            if from_avis:
                # We modified this split such that the split is based on the avi-files, not on the fullframes
                avis_sub = avis_list[i * int(np.floor(n_sample / num_proc)): (i + 1) * int(np.floor(n_sample / num_proc))]
                df_sub = df.loc[df['avi_path'].isin(avis_sub)].copy()
            else:
                df_sub = df[i * int(np.floor(n_sample / num_proc)): (i + 1) * int(np.floor(n_sample / num_proc))].copy()

            process = multiprocessing.Process(target=run_flatfielding_and_segmentation_on_df,
                                              args=(df_sub, f"{roi_dest_folder}/mp{i}", isiis_background_path, from_avis),
                                              kwargs=segment_kwargs)
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        # The remainder in case the size of the DataFrame is not a multiple of num_proc
        if from_avis:
            avis_sub = avis_list[num_proc * int(np.floor(n_sample / num_proc)):]
            df_sub = df.loc[df['avi_path'].isin(avis_sub)].copy()
        else:
            df_sub = df[num_proc * int(np.floor(n_sample / num_proc)):].copy()

        run_flatfielding_and_segmentation_on_df(df_sub, f"{roi_dest_folder}/mp0", isiis_background_path, from_avis, **segment_kwargs)
    else:
        run_flatfielding_and_segmentation_on_df(df, roi_dest_folder, isiis_background_path, from_avis, **segment_kwargs)

    t_elapsed = (dt.datetime.now() - t1).seconds / (len(df))
    print("t_elapsed per fullframe image: ", t_elapsed)

    return t_elapsed



def run_segmentation_on_tiffs_from_disk(source_dir, dest_dir, sub_sample,
                                        num_proc, isiis_background_path, segment_kwargs, test_run=False):
    """
    Assumes tiffs are stored in folders per cast and stores the resulting ROIs in folders per cast as well

    :param source_dir: str
    :param sub_sample: None or int -
    :param num_proc: int - number of parallel processes
    :param isiis_background_path: str
    :param segment_kwargs: dict - with following keys (see run_flatfielding_and_segmentation for explanation):
                                    - snr_threshold
                                    - min_roi_area
                                    - max_roi_frac,
                                    - min_bbox_area
                                    - roi_contour_pad
                                    - d
                                    - sigma_color
                                    - sigma_space
                                    - plot_prob
                                    - verbose
                                    - verbose_num
                                    - save_rois_with_background
    :param test_run: bool - if True, select and print the data, then continue without processing
    :return: None
    """
    df_full = get_all_images(dir=source_dir, extension='tiff')
    df_full.rename(columns={'dirpath': 'cast_id'}, inplace=True)

    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    for cast_id, df in df_full.groupby('cast_id'):
        print(f"\nProcessing {cast_id} to {dest_dir}")
        dest_dir_cast = f"{dest_dir}/{cast_id}"

        run_segmentation_procedure(df, isiis_background_path=isiis_background_path, roi_dest_folder=dest_dir_cast,
                                   num_proc=num_proc, segment_kwargs=segment_kwargs,
                                   sub_sample=sub_sample, test_run=test_run, from_avis=False)

    return


def remove_most_recent_files_from_roi_dest_folder(roi_dest_folder, test=True):
    """
    Based on the timestamp of the file modification, we remove the most recent file from the folders 'roi_data' and
     'rois' within each mp-folder within roi_dest_folder. This is useful if the segmentation was interrupted,
     because we want to avoid that incomplete zip-archives are stored.

    :param roi_dest_folder:
    :param test: bool - if True, the detected most recent files are printed. Recommended to first run the function
    with test=True to check if everything works.
    :return: None
    """
    for extension in ['.csv', '.zip']:
        df_processed = parse_folder_to_df(roi_dest_folder, extension=extension, verbose=False)
        df_processed['filepath'] = df_processed['filepath'].apply(lambda x: Path(x))
        df_processed['mp_dir'] = df_processed['filepath'].apply(lambda x: x.parent.parent.parent.name)

        # We select the most recent file based on the timestamp of modification
        df_processed['timestamp'] = df_processed['filepath'].apply(lambda x: x.stat().st_mtime)
        print(df_processed)

        for mp_dir, df_sub in df_processed.groupby('mp_dir'):
            print(mp_dir)
            max_timestamp = df_sub['timestamp'].max()
            max_filepath = df_sub.loc[df_sub['timestamp'] == max_timestamp, 'filepath'].iat[0]
            print(max_timestamp, max_filepath)

            if not test:
                # Here we actually remove the file
                max_filepath.unlink()
