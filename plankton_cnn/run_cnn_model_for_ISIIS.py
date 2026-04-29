#!/usr/bin/env python3
import datetime
import shutil
import numpy as np
import multiprocessing

from general_utils.pickle_functions import *
from general_utils.parse_folder import parse_folder_to_df
from general_utils.df_sample import split_list_in_chunks
from general_utils.export_files import extract_rois_from_zip_files

from plankton_cnn import pvnp_save_and_load_utils, pvnp_models, pvnp_build
from plankton_cnn.pvnp_use import apply_model_to_folder, apply_model_to_val_df

from cnn_evaluate.calc_classifier_metrics import calc_metric_for_evaluated_df


def run_cnn_model_on_img_folder(img_folder, model_name, train_prefix, batch_size, get_2nd_choice, path_to_label_folder,
                                path_to_ml_data, extension,
                                **import_kwargs):
    """
    Run a trained CNN on all images in a folder and store the output in a csv-file.
    
    :param img_folder: 
    :param model_name: 
    :param train_prefix: 
    :param batch_size: 
    :param get_2nd_choice: 
    :param path_to_label_folder: 
    :param path_to_ml_data: 
    :param extension: 
    :param import_kwargs: 
    :return: 
    """
    print(f"Start processing at {(t1 := datetime.datetime.now())}.")

    # Load input parameters for model and train_prefix
    model_run = f"{model_name}_{train_prefix}"
    image_size = pvnp_models.model_dict[model_name]['img_size']

    if not os.path.isdir(path_to_ml_data / model_run):
        raise ValueError(f"A model directory was not found at {path_to_ml_data / model_run}")

    # We create the folder where the labels are saved if it does not exist already
    if not os.path.isdir(path_to_label_folder):
        os.makedirs(path_to_label_folder)

    try:
        label_dict = pvnp_save_and_load_utils.load_label_dict(model_dir=model_run, path_to_models=path_to_ml_data)
    except FileNotFoundError as error:
        print(error)
        raise ValueError(f"An existing label_dict for {model_run} was not found. Use"
                         "pvnp_save_and_load_utils.save_label_dict to save a label_dict.")

    # Import the model
    model = pvnp_build.load_model_from_disk(model_name, train_prefix, compile=False,
                                            path_to_ml_data=path_to_ml_data, verbose=True)

    # Parse folder and apply the model
    df = apply_model_to_folder(img_folder, model, label_dict, image_size, batch_size, get_2nd_choice,
                               extension, **import_kwargs)

    # We save this DataFrame to a csv
    df.drop(columns='image_path', inplace=True)
    df.to_csv(path_to_label_folder / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

    del df

    print(f"Finished processing at {datetime.datetime.now()}."
          f"Time elapsed: ", datetime.datetime.now() - t1, "\n")


def apply_model_to_zips_multiproc(model_name, train_prefixes, df_zips, chunk_size,
                                  path_to_label_folder,
                                  num_proc=1,
                                  batch_size=32, as_grayscale=False, pad_value=0., adjust_contrast=True, prefetch=False,
                                  get_2nd_choice=False, path_to_ml_data=None,
                                  temp_unpack_base_folder=None,
                                  stop_at_chunk=None):
    """
    Apply a saved CNN model to the ROIs in the zip-archives in zip_list. We load and evaluate the images per chunk as
    specified with 'chunk_size'. Labels are stored in a csv-file per fullframe image in a
    file <path_to_ISIIS_labels>/<dest_folder_labels>/<ff_name>_roilabels.csv.

    The last chunk may be smaller than specified if the remainder of len(zip_list) over chunk_size is non-zero.

    :param model_name: str
    :param train_prefix: str
    :param df_zips: pd.DataFrame - with column 'zip_path' (with paths (as str) to the zip-archives)
    :param chunk_size: int or None - if not None, zip_list will be split among chunks of the specified size, and only
    the images per chunk will be loaded into memory at once and fed to the model. if None, all files in zip-files
    will be processed at the same time. Note that this may fail if the zip_list is too large to load all images into
    memory.
    :param path_to_label_folder: str or Path - e.g., 'volumes/<external_storage>/ISIIS_labels/<campaign>
    :param batch_size: int
    :param as_grayscale: bool
    :param pad_value: value in [0, 1] or 'mean_img'
    :param get_2nd_choice: bool
    :param path_to_ml_data: str or Path
    :param temp_unpack_base_folder: str or Path - Parent folder of a temporary folder where we will unpack the
    zip-archives. This temporary folder will be removed again after the images have been loaded into memory
    :return: None
    """
    # Filter zip-files that were already processed
    try:
        df_zips = filter_processed_zips(path_to_label_folder, df_zips)
    except NotADirectoryError:
        pass

    # df_zips = filter_processed_zips(path_to_label_folder, df_zips)
    df_zips = df_zips['zip_path'].copy()

    # Load input parameters for model and train_prefix
    if type(train_prefixes) is str:
        train_prefixes = [train_prefixes]

    model_runs = [f"{model_name}_{train_prefix}" for train_prefix in train_prefixes]
    image_size = pvnp_models.model_dict[model_name]['img_size']

    # We create the folder where the labels are saved if it does not exist already
    if not os.path.isdir(Path(path_to_label_folder).parent):
        os.mkdir(Path(path_to_label_folder).parent)

    if not os.path.isdir(path_to_label_folder):
        os.mkdir(path_to_label_folder)

    label_dest_folders = [Path(path_to_label_folder) / model_run for model_run in model_runs]
    for label_dest_folder in label_dest_folders:
        if not os.path.isdir(label_dest_folder):
            os.makedirs(label_dest_folder)

    # We define a temporary folder where we will unpack the zip-archives. This folder will be removed again after the
    # images have been loaded into memory. In case it already exists, we delete it now.
    dest_folder = Path(temp_unpack_base_folder) / 'temp'
    if os.path.isdir(dest_folder):
        shutil.rmtree(dest_folder)

    # Some checks
    if not len(df_zips):
        raise ValueError("List of zip-files is empty")

    for model_run in model_runs:
        if not os.path.isdir(path_to_ml_data / model_run):
            raise ValueError(f"A model directory was not found at {path_to_ml_data / model_run}")

    # Modify
    label_dicts = []
    for model_run in model_runs:
        try:
            label_dict = pvnp_save_and_load_utils.load_label_dict(model_dir=model_run, path_to_models=path_to_ml_data)
        except FileNotFoundError as error:
            print(error)
            raise ValueError(f"An existing label_dict for {model_run} was not found. Use"
                             "pvnp_save_and_load_utils.save_label_dict to save a label_dict.")
        label_dicts.append(label_dict)

    # Import the model
    models = [pvnp_build.load_model_from_disk(model_name, train_prefix, compile=False,
                                            path_to_ml_data=path_to_ml_data, verbose=True) for train_prefix in train_prefixes]

    if chunk_size:
        # Split the zip-list in chunks of the specified size
        zip_list_chunks = split_list_in_chunks(df_zips, chunksize=chunk_size)
    else:
        zip_list_chunks = [df_zips]

    import_kwargs = {
        # 'image_size': image_size,
        # 'batch_size': batch_size,
        'as_grayscale': as_grayscale,
        'adjust_contrast': adjust_contrast,
        'pad_value': pad_value,
        'prefetch': prefetch
    }

    print(f"\nStarting model {model_run}..")
    if num_proc > 1:
        n_sample = len(zip_list_chunks)

        jobs = []
        for i in range(num_proc):
            # Divide the list in batches for each parallel process, the remainder of this division (in case the
            # size is not a multiple of num_proc is handled below
            zip_list_chunk = zip_list_chunks[
                             i * int(np.floor(n_sample / num_proc)): (i + 1) * int(np.floor(n_sample / num_proc))]

            label_dest_folders_mp = [label_dest_folder / f"mp{i}" for label_dest_folder in label_dest_folders]
            process = multiprocessing.Process(target=apply_model_to_zip_list_chunks,
                                              args=(zip_list_chunk, models, label_dicts, image_size, batch_size,
                                                    get_2nd_choice,
                                                    dest_folder / f"mp{i}", label_dest_folders_mp, 100),
                                              kwargs=import_kwargs)
            jobs.append(process)

        for j in jobs:
            j.start()

        for j in jobs:
            j.join()

        # The remainder in case the size of the list is not a multiple of num_proc
        zip_list_chunk = zip_list_chunks[num_proc * int(np.floor(n_sample / num_proc)):]

        label_dest_folders_mp = [label_dest_folder / f"mp0" for label_dest_folder in label_dest_folders]
        apply_model_to_zip_list_chunks(zip_list_chunk, models, label_dicts, image_size, batch_size, get_2nd_choice,
                                       dest_folder / f"mp0", label_dest_folders_mp, 100, **import_kwargs)

    else:
        apply_model_to_zip_list_chunks(zip_list_chunks, models=models, label_dicts=label_dicts,
                                       get_2nd_choice=get_2nd_choice,
                                       dest_folder=dest_folder, path_to_label_folders=label_dest_folders,
                                       stop_at_chunk=stop_at_chunk,
                                       image_size=image_size, batch_size=batch_size, as_grayscale=as_grayscale,
                                       adjust_contrast=adjust_contrast,
                                       pad_value=pad_value, prefetch=prefetch)


# We also keep using the more general name of this function
apply_model_to_zips = apply_model_to_zips_multiproc


def apply_model_to_zip_list_chunks(zip_list_chunks, models, label_dicts, image_size, batch_size, get_2nd_choice,
                                   dest_folder, path_to_label_folders, stop_at_chunk, **import_kwargs):
    """

    :param zip_list_chunks:
    :param models: keras.model or list of keras.models
    :param label_dict:
    :param image_size:
    :param batch_size:
    :param get_2nd_choice:
    :param dest_folder:
    :param path_to_label_folder: str or list of str - the shape must match the shape of 'models'
    :param stop_at_chunk:
    :param import_kwargs:
    :return:
    """
    if type(models) is not list:
        models = [models]

    if type(path_to_label_folders) is not list:
        path_to_label_folders = [path_to_label_folders]

    if type(label_dicts) is not list:
        label_dicts = [label_dicts]

    if (len(models) != len(path_to_label_folders)) and (len(models) != len(label_dicts)):
        raise ValueError("The number of models, label_dict and path_to_label_folder must match.")

    # Loop over the chunks
    for i, zip_list_chunk in enumerate(zip_list_chunks):
        print(f"Start processing chunk {i + 1} with {len(zip_list_chunk)} zip-files")
        t1 = datetime.datetime.now()

        for zip_path in zip_list_chunk:
            extract_rois_from_zip_files(zip_path, Path(dest_folder) / Path(zip_path).stem, with_background=False)

        # Parse folder and apply the model
        for model, path_to_label_folder, label_dict in zip(models, path_to_label_folders, label_dicts):
            df = apply_model_to_folder(dest_folder, model, label_dict, image_size, batch_size, get_2nd_choice,
                                       extension='png',
                                       **import_kwargs)

            df['ff_name'] = df['image_path'].apply(lambda x: Path(x).parent.parent.name)
            df.rename(columns={'image_name': 'roi_name'}, inplace=True)
            df.drop(columns='image_path', inplace=True)

            # We save this DataFrame to a csv
            df.to_csv(path_to_label_folder / f"{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", index=False)

            del df

        # We remove the folder in which we unpacked the current set of zips
        shutil.rmtree(dest_folder)

        print(f"Finished processing this chunk at {datetime.datetime.now()}."
              f"Time elapsed: ", datetime.datetime.now() - t1, "\n")

        if stop_at_chunk and i == stop_at_chunk:
            print(f"We stop the function at chunk {i} so we can restart")
            return


def filter_processed_zips(path_to_label_folder, df_zips):
    """

    :param path_to_label_folder - str or Path
    :param df_zips: pd.DataFrame - with column 'zip_path', 'ff_name' (without suffix), other columns are unaffected
    :return: pd.DataFrame - copy of df_zips with only the zip-files that are not already in a csv-file
                            in path_to_label_folder
    """
    try:
        df_csv = parse_folder_to_df(dir=path_to_label_folder, extension='.csv')
    except FileNotFoundError:
        print(f"No files were found that are already processed in {path_to_label_folder}")
        return df_zips
    else:
        df_processed_list = []
        for filepath in df_csv['filepath']:
            df_processed = pd.read_csv(filepath, usecols=['ff_name'])
            df_processed_list.append(df_processed)

        df_processed = pd.concat(df_processed_list, ignore_index=True)
        df_processed.drop_duplicates(['ff_name'], inplace=True)

        print(f"{len(df_processed)} files were found that are already processed")
        df_zips = df_zips.loc[~df_zips['ff_name'].isin(df_processed['ff_name'])].drop(columns=['ff_name']).copy()
        print(f"Continuing with {len(df_zips)} zip-files")

        return df_zips


def check_model_performance(model_name, train_prefix, learning_dir, path_to_ml_data, path_to_training_data,
                            **import_kwargs):
    """
    Before running a model inference, we should always check if the model gives the same accuracy on the validation set on a
    new inference as during training (on the validation set). If not, the model weights were not properly copied.
    :return:
    """
    val_df = apply_model_to_val_df(model_name, learning_dir, train_prefix, batch_size=128, prefetch=True,
                                   as_grayscale=False, pad_value=1,
                                   adjust_contrast=False, get_2nd_choice=False, verbose=1.,
                                   path_to_ml_data=path_to_ml_data, path_to_training_data=path_to_training_data)

    val_acc_now = calc_metric_for_evaluated_df(val_df, true_name_label='true_name', pred_name_label='label')[0]
    val_act_training = pvnp_save_and_load_utils.load_tuner_args(model_dir=f"{model_name}_{train_prefix}",
                                             path_to_model=path_to_ml_data)['accuracy']
    
    print("\nValidation accuracy during current evaluation: ", val_acc_now)
    print("Validation accuracy during training: ", val_act_training)
    print("Difference: ", abs(val_acc_now - val_act_training))
