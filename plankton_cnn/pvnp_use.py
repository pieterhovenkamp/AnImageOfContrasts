#!/usr/bin/env python3

import sys

import numpy as np
import scipy

from general_utils.parse_folder import get_all_images
from general_utils.pickle_functions import *

from plankton_cnn import pvnp_models, pvnp_save_and_load_utils, pvnp_import, pvnp_build


def apply_model_to_df(model_name, train_prefix, df,
                      path_to_ml_data,
                      batch_size=32,
                      as_grayscale=False, pad_value=0., adjust_contrast=True,
                      apply_softmax=False, get_2nd_choice=False, verbose=1, prefetch=True,
                      ):
    """
    Apply model_name, trained in training round train_prefix to the images in DataFrame df.

    :param model_name: str
    :param train_prefix: str
    :param df: DataFrame - has to contain a column named 'image_path'
    :param as_grayscale: bool
    :param batch_size: int
    :param apply_softmax: bool
    :param get_2nd_choice: bool - if True, also the softmax-value and predicted label of the second best
                                  predictions is added to the DataFrame
    :param path_to_ml_data: str - path to the folder 'saved_models' of the model prefix
    :return: DataFrame df with new columns:
        - 'label' with model predictions
        - 'softmax' with softmax output in [0, 1] of predicted label
    """
    # Load input parameters for model and train_prefix
    model_run = f"{model_name}_{train_prefix}"
    img_size = pvnp_models.model_dict[model_name]['img_size']

    # Some checks
    if not len(df):
        raise ValueError("DataFrame is empty")

    if not os.path.isdir(path_to_ml_data / model_run):
        raise ValueError(f"A model directory was not found at {path_to_ml_data / model_run}")

    try:
        label_dict = pvnp_save_and_load_utils.load_label_dict(model_dir=model_run, path_to_models=path_to_ml_data)
    except FileNotFoundError as error:
        print(error)
        raise ValueError(f"An existing label_dict for {model_run} was not found. Use"
                         f"pvnp_save_and_load_utils.save_label_dict to save a label_dict.")

    # Import data as Tensorflow dataset
    ds = pvnp_import.import_images_from_df(df, img_size, batch_size, as_grayscale=as_grayscale, pad_value=pad_value,
                                           adjust_contrast=adjust_contrast, prefetch=prefetch)
    print(f"\nSetting up model {model_run} to run on {len(df)} images")
    # model = keras.models.load_model(f"{model_path}/{model_run}", compile=False)
    model = pvnp_build.load_model_from_disk(model_name, train_prefix, compile=False,
                                            path_to_ml_data=path_to_ml_data, verbose=True)
    # Remark: if in load_model, compile=False then tfa might not be required in order to load a saved model

    # Predict
    print("Starting model inference..")
    predictions = model.predict(ds, verbose=verbose)
    print("Model inference finished!")

    if apply_softmax:
        # In the more recent models that are imported, a final softmax-layer is added to the
        # model. For older models where this is not the case, a separate softmax needs
        # to be applied afterwards.
        predictions = scipy.special.softmax(predictions, axis=1)

    df = add_model_predictions_to_df(df, predictions, label_dict, get_2nd_choice)

    return df


def add_model_predictions_to_df(df, predictions, label_dict, get_2nd_choice):
    """
    Add the predicted labels and softmax output to a DataFrame. The DataFrame should have the same length as predictions and the order of the predictions and the DataFrame should match exactly.

    :param df: pd.DataFrame
    :param predictions: as output from model.predict()
    :param label_dict: dict
    :param get_2nd_choice: bool
    :return: pd.DataFrame with new columns 'label', 'softmax', and 'label_2nd', 'softmax_2nd' if get_2nd_choice = True
    """
    # Find maximum prediction and add label and softmax number to the dataframe
    predictions_args = np.argsort(predictions, axis=1)
    predictions_val = np.sort(predictions, axis=1)

    df['label'] = predictions_args[:, -1]
    df['softmax'] = predictions_val[:, -1]

    # Add string-labels instead of numbers
    df = df.replace({'label': label_dict})

    # Find 2nd maximum prediction and add the label and softmax number to the dataframe
    if get_2nd_choice:
        df['label_2nd'] = predictions_args[:, -2]
        df['softmax_2nd'] = predictions_val[:, -2]

        # Add string-labels instead of numbers
        df = df.replace({'label_2nd': label_dict})

    return df


def apply_model_to_val_df(model_name, learning_dir, train_or_tune_prefix,
                          path_to_ml_data, path_to_training_data,
                          subset='validation', as_grayscale=False, pad_value=0.,
                          adjust_contrast=True, get_2nd_choice=False, verbose=1, prefetch=True, batch_size=32,
                          ):
    """

    :param model_name: str
    :param learning_dir:  str - path to training data set
    :param train_or_tune_prefix:
    :param as_grayscale:
    :return: DataFrame - with columns 'image_name', 'image_path',
                         true_label' as int, 'true_name' (str) and the predictions
                         'label' (str)
    """
    # Load the validation set as a DataFrame
    val_df = pvnp_save_and_load_utils.load_learning_df_from_dir(learning_dir=learning_dir, subset=subset,
                                       path_to_training_data=path_to_training_data)

    # Rename columns, because 'label' and 'name' are overwritten in apply_model_to_df
    val_df.rename(columns={'label': 'true_label', 'name': 'true_name'}, inplace=True)
    val_df = apply_model_to_df(model_name, train_or_tune_prefix, val_df, batch_size=batch_size, prefetch=prefetch,
                               as_grayscale=as_grayscale, pad_value=pad_value, adjust_contrast=adjust_contrast,
                               get_2nd_choice=get_2nd_choice, verbose=verbose, path_to_ml_data=path_to_ml_data)

    return val_df


def apply_model_to_folder(img_folder, model, label_dict, image_size, batch_size, get_2nd_choice, extension,
                          **import_kwargs):
    """
    Parse all images with <extension> from img_folder and feed them into the model.

    Parameters
    ----------
    img_folder
    model
    label_dict
    get_2nd_choice

    Returns
    -------
    DataFrame - with columns 'image_name', 'image_path', 'label', 'softmax', and 'label_2nd', 'softmax_2nd' if get_2nd_choice = True
    """
    # Parse the images from the folder
    df = get_all_images(str(img_folder), extension=extension, verbose=False)
    df.drop(columns=['dirpath'], inplace=True)

    # Import images from the DataFrame as Tensorflow dataset
    img_ds = pvnp_import.import_images_from_df(df, image_size=image_size, batch_size=batch_size, **import_kwargs)

    # Predict
    print(f"\nStarting model inference on {len(df)} images...")
    predictions = model.predict(img_ds)
    df = add_model_predictions_to_df(df, predictions, label_dict, get_2nd_choice)
    print("Model inference finished!")

    return df
