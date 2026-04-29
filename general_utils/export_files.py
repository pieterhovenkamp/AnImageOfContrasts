from collections import Counter
import os
from pathlib import Path
import shutil

import numpy as np
import pandas as pd
import zipfile

from general_utils.parse_folder import parse_folder_to_df


def export_files_to_zip(filepaths, archive_path, overwrite=True):
    """
    Create a zip-file with the specified files. Folder structure within the filepaths is ignored, i.e. all images
    appear in the archive with their filenames only.

    :param filepaths: list - containing paths to the files that need to be zipped
    :param archive_path: str - path to zip-file that is created
    :param overwrite: bool - If True, archive is overwritten if it already exists.
    :return: None
    """
    if not archive_path.endswith('.zip'):
        archive_path = archive_path + '.zip'

    if zipfile.is_zipfile(archive_path) and not overwrite:
        raise ValueError(f"Zip-file {archive_path} already exists. Choose overwrite=True if you want to overwrite "
                         f"this file.")

    with zipfile.ZipFile(archive_path, mode="w") as archive:
        print(f"Added files to {archive_path}...")
        counter = 1
        for filepath in filepaths:
            archive.write(filepath, arcname=filepath.split(sep='/')[-1])

            if counter % 10000 == 0:
                print(f"added {counter} files to {archive_path}...")
            counter += 1

    print(f"{len(filepaths)} files were saved to {archive_path}")
    
    
def copy_rois_between_zip_files(zip_path_source, zip_path_target, roi_names, verbose=False):
    """
    We make a copy of a zip-archive that contains only the filenames specified in roi_names.
    Folder structure of the source archive is preserved.

    :param zip_path_source:
    :param zip_path_target:
    :param roi_names: list-like object of filenames that should be copied
    :return:
    """
    with zipfile.ZipFile(zip_path_target, mode="w") as archive_target:
        with zipfile.ZipFile(zip_path_source, mode="r") as archive_source:
            namelist = [name for name in archive_source.namelist() if name.split(sep='/')[-1] in list(roi_names)]
            
            for roi_name in roi_names:
                if roi_name not in [name.split(sep='/')[-1] for name in namelist]:
                    print(f"Item {roi_name} was not found in {zip_path_source}")

            if verbose:
                print(f"Adding {len(namelist)} files to {zip_path_target}...")

            for roi in namelist:
                archive_target.writestr(roi, archive_source.read(roi))



def copy_selected_rois_from_zips_from_df(df, dest_dir, fill_first=False):
    """
    We make a copy of all zip-archives in the DataFrame, where each copied archive only contains the ROIs that are present in the DataFrame.

    :param df: pd.DataFrame - needs to contain columns 'zip_path', 'ff_name' and 'roi_name'
    :param dest_dir: str - destination folder
    :return:
    """
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)

    if len(df_multiple := df.groupby('ff_name').filter(lambda df: len(df['zip_path'].unique()) > 1).sort_values('image_name')[['ff_name', 'roi_name', 'zip_path']]):
        print("Multiple zip-paths were found for the same ff_name! This leads to incomplete copies of the zip-files.")
        print("Check this, or set fill_first=True to automatically select the first zip-path for each ff_name")
        for ff_name, df_sub in df_multiple.groupby('ff_name'):
            print(ff_name, *df_sub['zip_path'].unique(), sep='\n')
        
        if fill_first:
            print("We select the first zip-path for each ff_name")
            df['zip_path'] = df.groupby('ff_name')['zip_path'].transform('first')
        else:
            print("Check this, or set fill_first=True to automatically select the first zip-path for each ff_name")

    counter = 0
    subfolder_prev = None
    for zip_path, df_sub_per_zip in df.groupby('zip_path'):
        subfolder = f"sub{int(np.floor(counter / 1000))}"

        if subfolder != subfolder_prev and not os.path.isdir(dest_dir / subfolder):
            os.mkdir(dest_dir / subfolder)
            print("Created subfolder: ", dest_dir / subfolder)

        zip_path_source = Path(zip_path)
        zip_path_target = dest_dir / subfolder/ zip_path_source.name
        copy_rois_between_zip_files(zip_path_source, zip_path_target, roi_names=df_sub_per_zip['roi_name'])

        counter += 1
        subfolder_prev = subfolder


def export_rois_per_label_from_zip_files(df, dest_folder, path_to_training_data, with_background,
                                         add_size_to_roi_name=False,
                                         label_col='label'):
    """
    Export the ROIs in a DataFrame from the zip-files to dest_folder, with a separate subfolder per label

    :param df: pd.DataFrame - needs to contain columns named 'zip_path', 'roi_name', label_col, 'diag_mm'
    :param dest_folder: str - path to folder with path_to_training_data in which the ROIs are saved
    :param path_to_training_data: str - path to the parent directory of dest_folder
    :param with_background: bool - whether to extract the ROIs with background or not
    :param label_col: str - name of the column in df with the labels
    :return: None
    """
    dest_folder = Path(path_to_training_data) / dest_folder
    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    if with_background:
        roi_folder = Path('rois_bg')
    else:
        roi_folder = Path('rois')

    if add_size_to_roi_name and 'bbox_diag_mm' not in df.columns:
        raise KeyError("if add_roi_size_to_roi_name is True, column 'diag_mm' must be present in the DataFrame")

    # For each entry in the DataFrame, we extract the ROI from the zip-archive
    for i, row in df.iterrows():
        archive_path = Path(row['zip_path'])
        filename = roi_folder / row['roi_name']
        dest_path = dest_folder / row[label_col]

        with zipfile.ZipFile(archive_path, mode="r") as archive:

            try:
                archive.extract(str(filename), dest_path)
            except KeyError as error:
                print(f"An error occurred while reading {filename} from {archive_path}")
                print(error)
            else:
                # The extracted ROI inherits the folder structure from the zip-file, so it is saved as <label>/<roi_folder>/<roi_counter>.png. Here, we change the ROI path to <label>/<ff_name>_<roi_counter>.png
                roi_path_extracted = dest_path / filename

                if add_size_to_roi_name:
                    roi_path_dest = dest_path / f"{np.round(row['bbox_diag_mm'], 2)}mm_{archive_path.stem}_{row['roi_name']}"
                else:
                    roi_path_dest = dest_path / f"{archive_path.stem}_{row['roi_name']}"

                roi_path_extracted.replace(roi_path_dest)

    # For each label, we need to remove the empty directory <roi_folder> that is still present.
    for label in df['label'].unique():
        dir_to_remove = dest_folder / label / roi_folder
        if os.path.isdir(dir_to_remove):
            shutil.rmtree(dir_to_remove)


def extract_rois_from_zip_files(zip_path, dest_folder, with_background):
    """
    Export the ROIs from a zip-archive to dest_folder. One can choose to take the ROIs either with or without background

    :param zip_path: str - path to the zip-archive
    :param dest_folder: str - path to the folder in which the ROIs are saved. Will be created if it does not exist yet.
    :param with_background: bool - whether to extract the ROIs with background or not
    :return: None
    """

    if not os.path.isdir(dest_folder):
        os.makedirs(dest_folder)

    if with_background:
        roi_folder = 'rois_bg/'  # The '/' is important to use .startswith is criterion later on
    else:
        roi_folder = 'rois/'

    # For each entry in the DataFrame, we extract the ROI from the zip-archive
    try:
        with zipfile.ZipFile(zip_path, mode="r") as archive:
            namelist = [name for name in archive.namelist() if name.startswith(roi_folder)]
            num_rois = len(namelist)
            if not num_rois:
                print(f"Archive {zip_path} is empty")
            for roi_path in namelist:
                archive.extract(str(roi_path), dest_folder)
    except FileNotFoundError as error:
        print(f"Skipped {zip_path} because it was not found")
        num_rois = 0

    return num_rois


def unpack_zip(zip_path, dest_folder=None, extension=None, max_num=None, verbose=None):
    """
    Unpack a zip-file and keep the original folder structure in the zip-archive.

    :param zip_path: str -
    :param dest_folder: str - if None, a folder will be created at the location of the zip-archive with the same name
    :param extension: str - if specified, only these file extension inside the zip-archive will be unpacked. If None, all files will be unpacked.
    :return: None
    """

    if dest_folder is None:
        dest_folder = Path(zip_path).parent / Path(zip_path).stem

    # We parse already unpacked files so we can skip these
    unpacked_files = parse_folder_to_df(dest_folder, extension=extension, verbose=False)
    unpacked_files = unpacked_files.loc[unpacked_files['filepath'].apply(lambda x: os.path.isfile(x))]

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if extension is None:
            namelist = [name for name in archive.namelist()]
        else:
            namelist = [name for name in archive.namelist() if Path(name).suffix == extension]

        if not len(namelist):
            print(f"\nArchive {zip_path} is empty")
        else:
            namelist = [name for name in namelist if Path(name).name not in list(unpacked_files['filename'])]
            print(f"\nUnpacking {len(namelist)} files from {zip_path}, {len(unpacked_files)} were already unpacked")

        if max_num:
            max_num = int(max_num)
            namelist = namelist[:max_num]

        t0 = dt.datetime.now()
        for i, roi_path in enumerate(namelist):
            try:
                archive.extract(str(roi_path), dest_folder)
            except Exception as e:
                print("Failed to extract {roi_path}: {error}".format(roi_path=roi_path, error=e))

            if verbose is not None:
                if (i + 1 ) % verbose == 0:
                    print(f"{i + 1} files were extracted - time elapsed: {dt.datetime.now() - t0}")
                    t0 = dt.datetime.now()
        print("Finished!")


def export_learning_set_to_folder_from_zips(df_train, df_val, df_test, learning_dir, path_to_training_data,
                                            with_background, label_col='label'):
    """
    Export the learning set defined in df_train, df_val and df_test to learning_dir, with a subfolder for each of the DataFrames.

    :param df_train: pd.DataFrame - needs to contain columns named 'zip_path', 'roi_name', label_col
    :param df_val:
    :param df_test:
    :param learning_dir:
    :param path_to_training_data:
    :param with_background: bool
    :return: None
    """
    export_rois_per_label_from_zip_files(df_train, Path(learning_dir) / 'training',
                                         path_to_training_data=path_to_training_data, with_background=with_background,
                                         label_col=label_col)
    export_rois_per_label_from_zip_files(df_val, Path(learning_dir) / 'validation',
                                         path_to_training_data=path_to_training_data, with_background=with_background,
                                         label_col=label_col)
    export_rois_per_label_from_zip_files(df_test, Path(learning_dir) / 'test',
                                         path_to_training_data=path_to_training_data, with_background=with_background,
                                         label_col=label_col)


def export_learning_set_to_folder(df_train, df_val, learning_dir, path_to_training_data, df_test=None):
    """

    :param df_train: pd.DataFrame - needs to contain column named 'image_path', 'image_name', 'label'
    :param df_val:
    :param df_test:
    :param learning_dir:
    :param path_to_training_data:
    :return: None
    """
    dest = Path(path_to_training_data) / learning_dir
    if not os.path.isdir(dest):
        os.mkdir(dest)

    copy_images_per_label(df_train, dest=str(dest / 'training'))
    copy_images_per_label(df_val, dest=str(dest / 'validation'))

    if df_test is not None:
        copy_images_per_label(df_test, dest=str(dest / 'test'))


def count_files_in_zip_archive(zip_path, extension):
    """
    For a zip-archive that can contain subfolders, count the number of files with the specified extension.

    :param zip_path: str
    :param extension: str - extension of the files to be counted. Can be denoted starting with or without '.', e.g as '.png' or 'png;
    :return: int
    """
    if not extension.startswith('.'):
        extension = f".{extension}"

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        roi_paths = [roi_path for roi_path in archive.namelist() if Path(roi_path).suffix == extension]
    return len(roi_paths)


def show_files_zip_file(archive_path):
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        archive.printdir()


def extract_zip_file(archive_path, dest_folder, verbose=1):
    """

    :param archive_path:
    :param dest_folder:
    :param verbose:
    :return:
    """
    with zipfile.ZipFile(archive_path, mode="r") as archive:
        for filename in (num := archive.namelist()):
            archive.extract(filename, dest_folder)

    if verbose:
        print(f"Extracted {num} files from {archive_path} to {dest_folder}")


def copy_images_per_label(df, label_name='label', dest=''):
    """
    Export images in the DataFrame to folder 'dest', organised in subfolders
    corresponding to their label. The label-column can be specified with label_name.

    :param df: DataFrame - needs columns 'image_path', 'image_name', label_name
    :return: None
    """
    if dest != '' and not os.path.isdir(dest):
        os.mkdir(dest)

    for index, row in df.iterrows():
        image_path, image_name, label = row['image_path'], row['image_name'], row[label_name]
        dest_folder = dest + f"/{label}"

        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        shutil.copy(image_path, f"{dest_folder}/{image_name}")
    return


def export_images(db, dest_folder):
    """Exports images in db to dest_folder. Dest_folder is created if it doesn't exist
    yet. db needs to have columns named 'image_path' and 'image_name'
    """
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    for file, name in zip(db['image_path'], db['image_name']):
        shutil.copyfile(file, f"{dest_folder}/{name}")
    print(f"{len(db['image_path'])} files copied to {dest_folder}")


def export_images_per_label(df):
    """
    Export ROIs in df to subfolders, corresponding to their edge_bool-label,
    that are created in the source locations of the ROIs.

    :param df: DataFrame - needs columns 'image_path', 'image_name', 'edge_bool'
    :return: None
    """
    for index, row in df.iterrows():
        image_path, image_name = row['image_path'], row['image_name']

        if row['edge_bool']:
            sharpness = 'sharp'
        else:
            sharpness = 'unsharp'

        dest_folder = os.path.dirname(image_path) + f"/{sharpness}"

        if not os.path.isdir(dest_folder):
            os.mkdir(dest_folder)

        # Move files (not copy, be aware when copying this segment for other purposes!)
        shutil.move(image_path, f"{dest_folder}/{image_name}")

    return
