#!/usr/bin/env python3

from keras.losses import CategoricalCrossentropy
import numpy as np
import tensorflow as tf

from plankton_cnn.pvnp_save_and_load_utils import load_label_dict
from general_utils.pickle_functions import *


def get_conversion_matrix(conversion_list):
    """
    Example:
    A list [[0, 1, 2], [3, 5, 7, 9], [4, 6, 8]] means we want to make 3 supergroups of 1: [0, 1, 2], 2: [3, 5, 7, 9] and 3: [4, 6, 8].
    The following conversion matrix will be returned:

    [[1. 0. 0.]
     [1. 0. 0.]
     [1. 0. 0.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 1. 0.]
     [0. 0. 1.]
     [0. 1. 0.]], shape=(10, 3), dtype=float32)

    :param conversion_list: a list of lists with elements the groups that will be grouped
    :return: tf.Tensor - conversion matrix of shape (len(unique groups), len(unique supergroups))
    """
    groups_orig = [k for i in conversion_list for k in i]
    groups_unique = np.unique(groups_orig)

    if len(groups_orig) != len(groups_unique):
        raise ValueError("Each group should be assigned to maximum one supergroup")

    conversion_array = np.zeros(len(groups_unique))
    for i, el in enumerate(conversion_list):
        conversion_array[el] = i

    return tf.constant(tf.keras.utils.to_categorical(conversion_array, num_classes=len(conversion_list)))


def hierarchical_loss(y_true, y_pred, weight_factor, conversion_matrix):
    """
    y_pred needs to be softmax probabilities before we sum over the predictions to calculate y_pred_sg.

    :param y_true:
    :param y_pred:
    :return:
    """

    # When we did not use one-hot encoding, the type conversion was necessary
    y_pred = tf.convert_to_tensor(y_pred)
    y_true = tf.cast(y_true, y_pred.dtype)
    conversion_matrix = tf.cast(conversion_matrix, y_pred.dtype)

    # If the inputs are single elements we need to expand the dimensions but model.fit does not needs this currently
    # y_true = tf.expand_dims(y_true, axis=0)

    # In linear algebra we trust
    y_true_sg = tf.linalg.matmul(y_true, conversion_matrix)
    y_pred_sg = tf.linalg.matmul(y_pred, conversion_matrix)

    # We calculate the loss based on the original grouping and on the higher-lever grouping
    loss_fn_base = CategoricalCrossentropy(from_logits=False)
    return weight_factor * loss_fn_base(y_true_sg, y_pred_sg) + (1. - weight_factor) * loss_fn_base(y_true, y_pred)


def get_conversion_matrix_from_labels(conversion_list, learning_dir, path_to_training_data, path_to_ml_data):
    """
    Construct the conversion matrix as needed for hierarchical_loss, based on the group labels. For this we need the label dict.

    Groups that are in the label_dict but not in conversion_list are added to a single rest group. Note that the specific one-hot
    representation of the supergroups depends on the order of appearance in conversion_list. But currently this does not matter
    as we do not use the supergroup predictions.

    :param conversion_list:
    :param learning_dir:
    :param path_to_training_data:
    :return: tf.Tensor
    """

    # We need the label dict and invert it
    label_dict = load_label_dict(learning_data_dir=learning_dir, path_to_training_data=path_to_training_data,
                                 path_to_models=path_to_ml_data)
    invert_label_dict = {val: key for key, val in label_dict.items()}

    # We check which groups are not in conversion_list and append these as a single group to the list
    conversion_list_flat = [k for i in conversion_list for k in i]
    conversion_list_missing = [group for group in label_dict.values() if group not in conversion_list_flat]

    if len(conversion_list_missing):
        print("\nThe following groups are not in specified within a hierarchical group:", *conversion_list_missing,
              sep='\n')
        print("A separate group was created for these groups together.")
        conversion_list.append(conversion_list_missing)

    return get_conversion_matrix([[invert_label_dict[x] for x in el] for el in conversion_list])




