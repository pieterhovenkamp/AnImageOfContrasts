#!/usr/bin/env python3

import glob

from sklearn import model_selection
from keras.utils import dataset_utils

from general_utils.pickle_functions import *
from plankton_cnn import pvnp_import, pvnp_models
from plankton_cnn.pvnp_import import AUGMENT_DICT
from plankton_cnn.pvnp_models import model_dict


def get_saved_model_path(model_dir, path_to_ml_data):
    return Path(path_to_ml_data) / model_dir / 'saved_model'


def get_model_checkpoint_path(model_dir, path_to_ml_data):
    return Path(path_to_ml_data) / model_dir / 'checkpoints'


def get_model_tuner_path(model_dir, path_to_ml_data):
    return Path(path_to_ml_data) / model_dir / 'keras_tuner'


def get_tb_logs_path(model_dir, path_to_ml_data):
    return Path(path_to_ml_data) / model_dir / 'tb_logs'


def get_best_model_path(model_dir, path_to_ml_data):
    """
    This functions assesses whether a folder 'keras_tuner/best_model' exists in model_dir. If so, the path to that
    folder is returned. Else, 'saved_model' is returned.

    We use this function to load the saved model for both the case of hyperparameter tuning and when a single training
    run was done.

    :param model_dir:
    :param path_to_ml_data:
    :return:
    """

    tuner_path = get_model_tuner_path(model_dir, path_to_ml_data=path_to_ml_data)
    if os.path.isdir(tuner_path / 'best_model'):
        best_model_path = tuner_path / 'best_model'
    else:
        best_model_path = get_saved_model_path(model_dir, path_to_ml_data=path_to_ml_data)

    return best_model_path


def get_train_val_test_split(df, train_split=0.7, val_split=0.15, test_split=0.15,
                             label_col='label',
                             random_state=1234):
    """
    Convenience function to create a train, test and val split in one function. train_split, val_split and test_split
    should sum to 1.

    :param df: pd.DataFrame - needs to have a column named 'label'
    :param train_split: float - in [0, 1]
    :param val_split: float - in [0, 1]
    :param test_split: float - in [0, 1]
    :param label_col: str - name of the label column in df that needs to be stratified (see docs of train_test_split)
    :param random_state: int - random seed for the split functions
    :return: pd.DataFrames with same columns as input df - if val_split and test_split are non-zero, then 3 dataframes
             are returned (df_train, df_val, df_test). If one of these is zero, then (df_train, df_val or df_test) are
             returned.
    """
    assert (train_split + test_split + val_split) == 1

    df_train, df_val_test = model_selection.train_test_split(df,
                                                             random_state=random_state,
                                                             train_size=train_split,
                                                             stratify=df[label_col])

    if (val_split > 0.) and (test_split > 0.):
        df_val, df_test = model_selection.train_test_split(df_val_test,
                                                           random_state=random_state,
                                                           train_size=(val_split / (val_split + test_split)),
                                                           stratify=df_val_test[label_col])
        return df_train, df_val, df_test
    else:
        return df_train, df_val_test


def load_learning_df_from_dir(learning_dir, subset, path_to_training_data):
    """
    Load the training, validation or test set from its folder

    :param learning_dir: str
    :param subset: str - one of 'validation', 'training', 'test'
    :param path_to_training_data: str
    :return: pd.DataFrame - with columns 'image_name', 'image_path', 'label' (as int), 'name' (as str)
    """

    if subset not in ['validation', 'training', 'test']:
        raise ValueError(f"Value for subset {subset} not recognized. Options are: "
                         "'validation', 'training', 'test'")

    image_paths, labels, _ = dataset_utils.index_directory(Path(path_to_training_data) / learning_dir / subset,
                                                           labels='inferred', formats=(".png", ".jpg"),
                                                           class_names=None)

    df = pd.DataFrame({'image_name': map(lambda x: Path(x).name, image_paths),
                       'image_path': image_paths,
                       'label': labels,
                       'name': map(lambda x: Path(x).parent.name, image_paths)})
    return df


def get_numbers_per_group_in_train_data(learning_dir, path_to_training_data):
    """
    Acces the sizes per group same as for a dict: with the names as keys. e.g. train_group_sizes['trochophore']

    :param learning_dir: str
    :param path_to_training_data: str
    :return: pd.Series - with index the names and values the numbers per group
    """
    df_train = load_learning_df_from_dir(learning_dir, subset='training', path_to_training_data=path_to_training_data)
    return df_train.groupby(by='name').size()


def save_model(model, model_dir, path_to_ml_data):
    """

    :param model:
    :param model_dir:
    :param path_to_ml_data:
    :return:
    """
    save_path = get_saved_model_path(model_dir, path_to_ml_data=path_to_ml_data)

    try:
        model.save(save_path)
    except ValueError as error:
        print("Saving the model generated the following exception:\n", error, "\nTherefore we skipped this model")
    else:
        print(f"model was saved to {save_path}")


def load_label_dict(model_dir=None, learning_data_dir=None, path_to_models=None,
                    path_to_training_data=None):
    """
    Load label dictionary with keys the integer labels (as integers) and as values the
    string names of the labels of the training data directory.

    Either loads the dictionary from a txt-file using load_saved_label_dict or reloads the dictionary by parsing the
    files in training_data_dir in the same way as is done in pvnp.import.import_plankton_images. So one of train_prefix
    or training_data_dir should be specified.

    :param model_dir: str - name of the model directory
    :param learning_data_dir: str
    :param path_to_models: str - path to the folder with the model directory
    :return: dict - dictionary with keys the integer labels (as integers) and as values the
    string names of the labels
    """

    try:
        label_dict = load_saved_label_dict(model_dir, path_to_models=path_to_models)
        label_dict = dict(zip([int(x) for x in label_dict.keys()], label_dict.values()))
    except FileNotFoundError:
        if learning_data_dir:
            label_dict = pvnp_import.get_labeL_dict_for_training_directory(learning_data_dir,
                                                                           path_to_folder=path_to_training_data)
        else:
            raise ValueError(f"An existing label_dict for {model_dir} in path_to_ml_data {path_to_models} was not found,"
                             f" so function arguments learning_data_dir should be specified")
    return label_dict


def save_history(history, model_dir, path_to_models, round_prefix=None):
    """
    Saves db with metrics as columns to the path
    train_prefix/prefix_hist.pkl

    :param history: history-object
    :param model_dir: str - name of the model directory
    :param round_prefix: str
    :param path_to_models: str
    :return: None
    """
    df = pd.DataFrame(history.history)

    if round_prefix:
        filename = Path(model_dir) / f"{round_prefix}_hist.pkl"
    else:
        filename = Path(model_dir) / "hist.pkl"

    to_pickle(df, filename, dir=path_to_models, verbose=0)
    print(f"History of {round_prefix} was saved to {filename}")


def load_history_single(model_dir, path_to_models, round_prefix=None):
    if round_prefix:
        filename = Path(model_dir) / f"{round_prefix}_hist.pkl"
    else:
        filename = Path(model_dir) / "hist.pkl"

    return read_pickle(filename, dir=path_to_models)


def load_history(model_dir, path_to_model):
    """
    For a training round, finds the history files that were saved via
    save_history()

    :param model_dir: str - name of the model directory
    :param path_to_model: str
    :return: dict - with for each training prefix, a pd.DataFrame
    with the saved metrics as columns and the metric values as entries.
    """
    search_pattern = Path(path_to_model) / model_dir / "*_hist.pkl"
    history_paths = glob.glob(str(search_pattern))

    if not len(history_paths):
        raise FileNotFoundError(f"No hist.pkl-file was found for {search_pattern}")

    hist_dict = {}
    for history_path in history_paths:
        prefix = Path(history_path).name.replace('_hist.pkl', '')
        hist_dict[prefix] = read_pickle(history_path, dir=Path(history_path).parent)

    return hist_dict


def save_tuner_hist(history_dict, model_dir, path_to_ml_data, verbose=1):
    """
    Save the history from a tuner as returned by search_return_history.

    Note that we assume here that there is only one execution (= round of epochs per trial). If there would be
    more, only the first execution is saved here.

    :param history_dict: dict - dictionary with as keys the trial and as items a list of history-objects
    :param model_dir: str
    :param path_to_ml_data: str
    :param verbose:
    :return: None
    """
    df_list = []
    for trial, history in history_dict.items():
        # history is a list of History objects for each execution. We normally
        # do 1 execution (= 1 round of epochs per trial), so we can just select the first item
        df_sub = pd.DataFrame(history[0].history)
        df_sub['trial'] = trial
        df_sub['epoch'] = df_sub.index + 1

        df_list.append(df_sub)
    df_hist = pd.concat(df_list, ignore_index=True)

    to_pickle(df_hist, Path(model_dir) / f"hist.pkl", dir=path_to_ml_data, verbose=verbose)


def load_tuner_hist(model_dir, path_to_ml_data):
    return read_pickle(Path(model_dir) / f"hist.pkl", dir=path_to_ml_data)


def save_training_args(args_dict, model_dir, path_to_model):
    """

    :param args_dict:
    :param model_dir:
    :param path_to_model:
    :return:
    """
    save_dict(args_dict, path=(dest := Path(path_to_model) / model_dir / "training_args.txt"))
    print(f"Trainings arguments saved to {dest}")


def load_training_args(model_dir, path_to_model):
    """

    :param model_dir:
    :param path_to_model:
    :return:
    """
    return load_dict(Path(path_to_model) / model_dir / "training_args.txt")


def save_tuner_args(args_dict, model_dir, path_to_model):
    dest = Path(path_to_model) / model_dir / "keras_tuner" / "best_hyperparameters.txt"
    save_dict(args_dict, path=dest)
    print(f"Best hyperparameters {args_dict} \n saved to {dest}")


def load_tuner_args(model_dir, path_to_model):
    args_dict = load_dict(Path(path_to_model) / model_dir / "keras_tuner" / "best_hyperparameters.txt")
    for key, item in args_dict.items():
        args_dict[key] = float(item)
    return args_dict


def save_label_dict(model_dir, learning_dir, path_to_model,
                    path_to_training_data):
    """

    :param model_dir: str
    :param learning_dir: str
    :param path_to_model: str
    :param path_to_training_data: str
    :return: None
    """
    label_dict = pvnp_import.get_labeL_dict_for_training_directory(learning_dir=learning_dir,
                                                                   path_to_folder=path_to_training_data)

    dest = Path(path_to_model) / model_dir / "labels_dict.txt"
    save_dict(label_dict, dest)
    print(f"Label dict {model_dir} was saved to {dest}")


def get_best_checkpoint_path(model_dir, path_to_ml_data):
    """
    This functions assesses whether a folder 'keras_tuner' exists in model_dir. If so, the checkpointpath corresponding to
    the best trial is returned. Else, the checkpoint path to 'checkpoints/best_cp.ckpt' is returned.

    We use this function to load the best checkpoint for both the case of hyperparameter tuning and when a single training
    run was done.

    :param model_dir:
    :param path_to_ml_data:
    :return: str
    """
    if os.path.isdir(tuner_path := get_model_tuner_path(model_dir, path_to_ml_data=path_to_ml_data)):
        best_trial = int(load_tuner_args(model_dir, path_to_model=path_to_ml_data)['best_trial'])
        if os.path.isdir(best_trial_path := tuner_path / f"trial_{best_trial}"):
            return best_trial_path / 'checkpoint'
        # If 10 or more trials are done, trials are numbered as 'trial_0X' instead of 'trial_X'
        elif os.path.isdir(best_trial_path := tuner_path / f"trial_0{best_trial}"):
            return best_trial_path / 'checkpoint'
        else:
            raise FileNotFoundError(f"No folder of best trial number {best_trial} was found in {tuner_path}")
    else:
        return get_model_checkpoint_path(model_dir, path_to_ml_data=path_to_ml_data) / 'best_cp.ckpt'


def load_saved_label_dict(model_dir, path_to_models):
    if model_dir is None:
        raise FileNotFoundError
    else:
        return load_dict(Path(path_to_models) / model_dir / "labels_dict.txt")


def load_and_stitch_history(model_name, *train_prefixes, train_round='best', path_to_model):
    """

    :param model_name:
    :param train_prefixes:
    :param train_round:
    :param path_to_model:
    :return:
    """
    stitch_points = []
    for i, prefix in enumerate(train_prefixes):
        model_dir = f"{model_name}_{prefix}"
        if not i:
            df_history_stitched = load_history(model_dir, path_to_model=path_to_model)[train_round]
            continue

        df_history = load_history(model_dir, path_to_model=path_to_model)[train_round]
        stitch_points.append(len(df_history_stitched))
        df_history_stitched = pd.concat([df_history_stitched, df_history], ignore_index=True)

    return df_history_stitched, stitch_points


def get_best_acc_from_saved_model(model_name, train_prefix, path_to_model, train_round='best'):
    """
    Accuracy in [0, 1]

    :param model_name: str
    :param train_prefix: str
    :param train_round: str
    :return: float
    """
    df_history = load_history_single(f"{model_name}_{train_prefix}",
                                     round_prefix=train_round, path_to_models=path_to_model)

    return max(df_history['val_accuracy'])


def get_total_length_with_augment(learning_dir, augmentation, path_to_training_data):
    """

    :param dataset: str - path to training data
    :param path_to_training_data:
    :param augmentation:
    :return:
    """
    total_length = len(load_learning_df_from_dir(learning_dir, subset='training',
                                                 path_to_training_data=path_to_training_data))
    if augmentation:
        total_length = total_length * AUGMENT_DICT[augmentation]

    return total_length


if __name__ == '__main__':
    import sys

    print(get_saved_model_path('EfficientNetV2S_run18-bo_gamma_spring2022',
                               path_to_ml_data='databases/ml_data_new_structure'))

    print(get_model_checkpoint_path('EfficientNetV2S_run18-bo_gamma_spring2022',
                                    path_to_ml_data='databases/ml_data_new_structure'))

    print(get_model_tuner_path('EfficientNetV2S_run18-bo_gamma_spring2022',
                               path_to_ml_data='databases/ml_data_new_structure'))
    print(get_model_tuner_path('EfficientNetV2S_run18-bo_gamma_spring2022',
                               path_to_ml_data='databases/ml_data_new_structure').parent)
    print(get_model_tuner_path('EfficientNetV2S_run18-bo_gamma_spring2022',
                               path_to_ml_data='databases/ml_data_new_structure').name)

    print(get_tb_logs_path('EfficientNetV2S_run18-bo_gamma_spring2022',
                           path_to_ml_data='databases/ml_data_new_structure'))
    sys.exit()

    print("get_numbers_per_group_in_train_data:\n",
          get_numbers_per_group_in_train_data('gamma_spring2022',
                                              path_to_training_data='databases/training_data'))

    df = load_learning_df_from_dir('gamma_spring2022', subset='training',
                                   path_to_training_data='databases/training_data')

    df = load_learning_df_from_dir('gamma_spring2022', subset='validation',
                                   path_to_training_data='databases/training_data')
    print(df)

    df = load_learning_df_from_dir('gamma_spring2022', subset='test',
                                   path_to_training_data='databases/training_data')

    df = load_learning_df_from_dir('datasets_alpha', subset='test',
                                   path_to_training_data='databases/training_data')
    sys.exit()

    path_to_model = 'ml_data/ml_data_new_structure'
    model_dir = 'EfficientNetV2S_run18-bo_gamma_spring2022'
    save_label_dict(model_dir, training_dir='gamma_spring2022', path_to_model=path_to_model,
                    path_to_training_data='databases/training_data')
    print("load_saved_label_dict:\n", load_saved_label_dict(model_dir, path_to_models=path_to_model))

    model_dir = 'EfficientNetV2S_gamma_spring2022_tl2'
    print("load_history:\n", load_history(model_dir, path_to_model=path_to_model))
    print("load_and_stitch_history:\n",
          load_and_stitch_history('EfficientNetV2S', *['gamma_spring2022_tl2', 'gamma_spring2022_tl2'],
                                  train_round='best', path_to_model=path_to_model))
    print("load_and_stitch_history:\n", load_and_stitch_history('EfficientNetV2S', 'gamma_spring2022_tl2',
                                                                train_round='best', path_to_model=path_to_model))

    model_dir = 'EfficientNetV2S_run18-bo_gamma_spring2022'
    test_args_dict = {'learning_rate': '1', 'optimiser': 'an_optimiser_name'}
    save_training_args(test_args_dict, model_dir, path_to_model=path_to_model)
    print("load_training_args:\n", load_training_args(model_dir, path_to_model=path_to_model))

    model_dir = 'EfficientNetV2S_gamma_spring2022_tl2'
    test_args_dict = {'learning_rate': '1', 'weight_decay': '98'}
    save_tuner_args(test_args_dict, model_dir, path_to_model=path_to_model)
    print("load_tuner_args:\n", load_tuner_args(model_dir, path_to_model=path_to_model))

    print("get_best_acc_from_saved_model:\n",
          get_best_acc_from_saved_model('EfficientNetV2S', 'gamma_spring2022_tl2',
                                        train_round='best', path_to_model=path_to_model))

    sys.exit()

    train_dir, val_dir, val_split = import_train_or_val_dir('alpha_spring2022', val_split=0.2)
    train_dir, val_dir, val_split = import_train_or_val_dir('delta_grayscale_spring2022', val_split=0.2)
    print(train_dir)
    print(val_dir)
    print(val_split)
