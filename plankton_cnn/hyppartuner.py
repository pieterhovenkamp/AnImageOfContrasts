#!/usr/bin/env python3

from functools import partial
import math
import shutil
import sys
import warnings

import pandas as pd
import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow import keras

import keras_tuner
from keras_tuner.engine import trial as trial_module
from keras_tuner.engine import tuner_utils

from plankton_cnn import pvnp_import, pvnp_build
from plankton_cnn.pvnp_save_and_load_utils import *
from plankton_cnn.pvnp_models import model_dict


def build_model_new(hp, model_name, num_classes,
                    lr_min, lr_max, opt_par_min, opt_par_max, loss, optimizer, freeze_base_model,
                    path_to_ml_data,
                    batch_size=32,
                    model_prev=None,
                    decay_function=None,
                    decay_steps=None,
                    total_length=None,
                    use_imagenet_weights=True):
    """
    Function that serves as an argument for the keras_tuner. keras_tuner requires
    this to have hp as argument and to return the model-object

    :param hp:
    :param model_name: str
    :param num_classes:
    :param lr_min:
    :param lr_max:
    :param opt_par_min:
    :param opt_par_max:
    :param loss:
    :param optimizer:
    :param freeze_base_model:
    :param batch_size:
    :param model_prev: str - if specified, load model from this prefix of previous training round as start of training
    :param decay_function:
    :param decay_steps:
    :param total_length:
    :param use_imagenet_weights:
    :return:
    """
    # Specify the range of hyperparameters
    lr_start = hp.Float("lr_start", min_value=lr_min, max_value=lr_max, sampling="log")

    if optimizer.func == tf.keras.optimizers.Adam:
        opt_par_start = hp.Float("eps", min_value=opt_par_min, max_value=opt_par_max, sampling='log')
        optimizer_name = 'Adam'
    elif optimizer.func == tfa.optimizers.AdamW:
        opt_par_start = lr_start * hp.Float("wd_start", min_value=opt_par_min, max_value=opt_par_max, sampling='log')
        optimizer_name = 'AdamW'
    elif optimizer.func == tfa.optimizers.weight_decay_optimizers.SGDW:
        opt_par_start = lr_start * hp.Float("wd_start", min_value=opt_par_min, max_value=opt_par_max, sampling='log')
        optimizer_name = 'SDGW'
    else:
        raise ValueError(
            "optimizer should be either Adam, AdamW or SDGW, or else modify the build_model function first")

    if type(batch_size) == list:
        hp.Choice("batch_size", values=batch_size, ordered=True)
    elif type(batch_size) == int:
        hp.Fixed("batch_size", value=batch_size)
    else:
        raise ValueError("batch_size should be either a list or an integer")

    if decay_function:
        # Calculate the decay schedules and accordingly, the optimizers' learning rate
        decay_fn = partial(decay_function,
                           decay_steps=math.ceil(decay_steps * total_length / hp.get("batch_size")))
        lr = decay_fn(initial_learning_rate=lr_start)

        # If SDGW, we want the opt_par (= weight decay) to decay with the learning rate
        if optimizer_name == 'SDGW':
            opt_par = decay_fn(initial_learning_rate=opt_par_start)
        else:
            opt_par = opt_par_start
    else:
        lr = lr_start
        opt_par = opt_par_start

    if optimizer_name == 'Adam':
        opt_kwargs = {'learning_rate': lr, 'epsilon': opt_par}
    elif optimizer_name == 'AdamW':
        opt_kwargs = {'learning_rate': lr, 'weight_decay': opt_par}
    elif optimizer_name == 'SDGW':
        opt_kwargs = {'learning_rate': lr, 'weight_decay': opt_par}
    else:
        raise ValueError("First define the optimiser's keyword arguments for this optimiser in build_model_new")

    # Call the functions that build and compile the model
    if model_prev:
        # model = keras.models.load_model(model_prev)
        model = pvnp_build.load_model_from_disk(model_name, model_prev, path_to_ml_data=path_to_ml_data)
    else:
        model = pvnp_build.build_model(
            model_name, num_classes,
            classifier_activation='softmax',
            path_to_ml_data=path_to_ml_data,
            use_imagenet_weights=use_imagenet_weights)

    if freeze_base_model:
        for layer in model.layers[:-1]:
            layer.trainable = False
    else:
        for layer in model.layers:
            layer.trainable = True

    # model.summary()

    model.compile(optimizer=optimizer(**opt_kwargs),
                  loss=loss, metrics=['accuracy'])

    return model


class MyHyperModel(keras_tuner.HyperModel):
    def __init__(self, model_name, num_classes, build_model_func,
                 decay_steps, total_length, path_to_ml_data):
        self.model_name = model_name
        self.num_classes = num_classes
        self.build_model_func = build_model_func
        self.decay_steps = decay_steps
        self.total_length = total_length
        self.path_to_ml_data = path_to_ml_data

    def build(self, hp):
        """
        :param hp:
        :return:
        """
        return self.build_model_func(
            hp, self.model_name, self.num_classes,
            decay_steps=self.decay_steps,
            total_length=self.total_length, path_to_ml_data=self.path_to_ml_data)

    def fit(self, hp, model, train_ds, val_ds, **kwargs):
        train_ds, val_ds = train_ds.unbatch(), val_ds.unbatch()
        train_ds = train_ds.batch(hp.get("batch_size"))
        val_ds = val_ds.batch(hp.get("batch_size"))

        return model.fit(train_ds, validation_data=val_ds, **kwargs)


class BayesianOptimization(keras_tuner.BayesianOptimization):
    """
    We added a method search_return_history to keras_tuner.BayesianOptimization so that we can save the training
    history of each epoch.
    """

    # This method was modified from Tuner.search in base_tuner.py
    def search_return_history(self, *fit_args, **fit_kwargs):
        """Performs a search for best hyperparameter configuations.

        Args:
            *fit_args: Positional arguments that should be passed to
              `run_trial`, for example the training and validation data.
            **fit_kwargs: Keyword arguments that should be passed to
              `run_trial`, for example the training and validation data.
        """
        if "verbose" in fit_kwargs:
            self._display.verbose = fit_kwargs.get("verbose")
        self.on_search_begin()
        history_dict = {}  # Added by PH
        while True:
            self.pre_create_trial()
            trial = self.oracle.create_trial(self.tuner_id)
            if trial.status == trial_module.TrialStatus.STOPPED:
                # Oracle triggered exit.
                tf.get_logger().info("Oracle triggered exit")
                break
            if trial.status == trial_module.TrialStatus.IDLE:
                # Oracle is calculating, resend request.
                continue

            self.on_trial_begin(trial)
            results = self.run_trial(trial, *fit_args, **fit_kwargs)
            # `results` is None indicates user updated oracle in `run_trial()`.
            if results is None:
                warnings.warn(
                    "`Tuner.run_trial()` returned None. It should return one of "
                    "float, dict, keras.callbacks.History, or a list of one "
                    "of these types. The use case of calling "
                    "`Tuner.oracle.update_trial()` in `Tuner.run_trial()` is "
                    "deprecated, and will be removed in the future.",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                tuner_utils.validate_trial_results(
                    results, self.oracle.objective, "Tuner.run_trial()"
                ),
                self.oracle.update_trial(
                    trial.trial_id,
                    # Convert to dictionary before calling `update_trial()`
                    # to pass it from gRPC.
                    tuner_utils.convert_to_metrics_dict(
                        results,
                        self.oracle.objective,
                    ),
                    step=tuner_utils.get_best_step(results, self.oracle.objective),
                )
            trial_number = int(trial.trial_id) + 1  # Added by PH
            history_dict[trial_number] = results  # Added by PH
            self.on_trial_end(trial)
        self.on_search_end()
        return history_dict  # Added by PH


def save_best_values_from_tuner(tuner, val_ds, model_dir, path_to_ml_data, save=True):
    """
    We save the best tuner arguments in a txt-file and we save the trained model corresponding to these hyperparameters.

    :param path_to_ml_data:
    :param tuner:
    :param val_ds:
    :param tuner_data_dir: str - of the form <model_name>_<train_prefix>
    :param save:
    :return:
    """
    best_hyperparameters = tuner.get_best_hyperparameters()[0].values
    best_hyperparameters['score'] = tuner.oracle.get_best_trials()[0].score
    best_hyperparameters['best_trial'] = tuner.oracle.get_best_trials()[0].trial_id

    best_model = tuner.get_best_models(num_models=1)[0]
    best_hyperparameters['accuracy'] = best_model.evaluate(val_ds, verbose=0)[1]

    if save:
        save_tuner_args(best_hyperparameters, model_dir=model_dir, path_to_model=path_to_ml_data)
        save_model(best_model, model_dir=model_dir, path_to_ml_data=path_to_ml_data)

    return


def run_tuner(model_name, learning_data_dir, train_prefix,
              num_trials, epochs_per_exc, num_initial_points, build_model_func,
              path_to_ml_data,
              path_to_training_data,
              preproc_prefix=None,
              decay_steps=None,
              es=None,
              es_baseline=None,
              immediate_baseline=None,
              augment='simple',
              batch_size=32,
              overwrite=False,
              print_search_space_only=False,
              return_best_values=False,
              as_grayscale=False,
              pad_value=0.,
              adjust_contrast=True,
              one_hot_labels=False,
              subset_fraction=None,
              prefetch=True,
              fit_verbose=1,
              objective='val_accuracy',
              shuffle_seed=1234,
              ):
    """

    :param model_name: str - see function 'build_model'
    :param learning_data_dir: str - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param train_prefix: str - name of the training procedure that will be used in the resulting saved files
    :param num_trials: int - number of different trials (i.e. different combination of hyperparameters) that are tried
    :param epochs_per_exc: int - number of training epochs per trial
    :param num_initial_points: int - number of initial hp-combinations that are chosen randomly before the remaining
                                     trials (=epochs_per_exc - num_initial_points) with Bayesian Optimisation start
    :param build_model_func: function - complicated. Use 'build_model_new' inside 'partial()', check a working example
    :param preproc_prefix: str - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param decay_steps: int - number of training epochs after which the learning rate is halved, if a decay function is
                              specified in build_model_func
    :param es: int - maximum number of epochs that can be passed without improvement before early stopping
    :param es_baseline: float - if the 'objective' after the number of epochs defined in 'es' is below this baseline,
                                training will be stopped.
    :param immediate_baseline: float - in [0, 1] or None. Defines a baseline for accuracy below which training
                                       is stopped immediately (the difference with es_baseline is 1: that this is
                                       only evaluated after certain number of epochs as defined in es 2: that
                                       immediat_baseline always evaluates accuracy, regardless of 'objective). Note also
                                       that this option operates independent of es (immediate_baseline can be applied
                                       while es=None.
    :param augment: str - either 'simple' or 'all_rotations' or None. For more information see
                          function 'import_learning_set_from_dir_as_ds_optimised'
    :param batch_size: int - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param overwrite: bool - overwrite any existing tuner-files with the same savename (=<model_name>_<train_prefix>)
    :param print_search_space_only: bool - print the specified search space of hyperparameters without starting the
                                           actual tuning
    :param return_best_values: bool - if a savename already exists, save the best found values.
    :param as_grayscale: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param pad_value: float in [0, 1] or 'mean_img': - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param adjust_contrast: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param one_hot_labels: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param subset_fraction: - currently not in use.
    :param prefetch: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param fit_verbose: int - Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param objective: str - should be one of ['loss', 'val_loss', 'accuracy', 'val_accuracy']. Metric that is used
                               to assess model performance
    :param shuffle_seed: int - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param path_to_ml_data: str - path to the folder where the models and files related to the model will be saved.
                                  These will be stored in a newly created folder as:
                                   <path_to_ml_data>/<model_name>_<train_prefix>
    :param path_to_training_data: str - see function 'import_learning_set_from_dir_as_ds_optimised'
    :return:
    """

    if return_best_values:
        overwrite = False

    save_name = f"{model_name}_{train_prefix}"
    tuner_path = get_model_tuner_path(save_name, path_to_ml_data=path_to_ml_data)
    save_path_tb = get_tb_logs_path(save_name, path_to_ml_data=path_to_ml_data)

    print(f"\nStarting run_tuner() on {save_name}\n")

    train_ds, train_dict = pvnp_import.import_learning_set_from_dir_as_ds_optimised(learning_dir=learning_data_dir,
                                                                                    subset='training',
                                                                                    image_size=model_dict[model_name][
                                                                                        'img_size'],
                                                                                    batch_size=batch_size, seed=shuffle_seed,
                                                                                    preproc_prefix=preproc_prefix,
                                                                                    augment=augment,
                                                                                    as_grayscale=as_grayscale,
                                                                                    adjust_contrast=adjust_contrast,
                                                                                    pad_value=pad_value,
                                                                                    one_hot_labels=one_hot_labels,
                                                                                    prefetch=prefetch,
                                                                                    path_to_training_data=path_to_training_data)
    val_ds, _ = pvnp_import.import_learning_set_from_dir_as_ds_optimised(learning_dir=learning_data_dir,
                                                                         subset='validation',
                                                                         image_size=model_dict[model_name]['img_size'],
                                                                         batch_size=batch_size, seed=shuffle_seed,
                                                                         preproc_prefix=preproc_prefix,
                                                                         as_grayscale=as_grayscale,
                                                                         adjust_contrast=adjust_contrast,
                                                                         pad_value=pad_value,
                                                                         prefetch=prefetch,
                                                                         one_hot_labels=one_hot_labels,
                                                                         path_to_training_data=path_to_training_data)

    class_weights = pvnp_build.calc_class_weights(train_dict)
    num_classes = len(train_dict)
    total_length = sum([dic['length'] for dic in train_dict.values()])

    tuner = BayesianOptimization(
        MyHyperModel(model_name, num_classes, build_model_func,
                     decay_steps, total_length, path_to_ml_data=path_to_ml_data),
        objective=objective,
        max_trials=num_trials,
        num_initial_points=num_initial_points,
        seed=1234,
        overwrite=overwrite,
        directory=str(tuner_path.parent),
        project_name=str(tuner_path.name))

    print()
    tuner.search_space_summary()
    print()

    if print_search_space_only:
        return

    if return_best_values:
        save_best_values_from_tuner(tuner, val_ds, save_name, save=True, path_to_ml_data=path_to_ml_data)
        return

    if overwrite:
        callbacks = [
            keras.callbacks.TensorBoard(save_path_tb),
            keras.callbacks.TerminateOnNaN(),
        ]

        if immediate_baseline:
            callbacks.append(pvnp_build.TerminateFinetuning(baseline=immediate_baseline))

        if es:
            callbacks.append(keras.callbacks.EarlyStopping(
                monitor=objective, min_delta=0, patience=es, baseline=es_baseline,
                verbose=1, mode="auto"))

        # We manually remove the existing folders of tensorboard and keras-tuner, since
        # I do not trust how this is handled automatically
        if os.path.isdir(tuner_path):
            shutil.rmtree(tuner_path)
            print(f"Existing folder {tuner_path} was removed")

        if os.path.isdir(save_path_tb):
            shutil.rmtree(save_path_tb)
            print(f"Existing folder {save_path_tb} was removed")

        history_dict = tuner.search_return_history(train_ds, val_ds,
                                                   epochs=epochs_per_exc,
                                                   class_weight=class_weights,
                                                   callbacks=callbacks,
                                                   verbose=fit_verbose)

        # Save the training history of all epochs
        save_tuner_hist(history_dict, model_dir=save_name, path_to_ml_data=path_to_ml_data, verbose=1)

        # Get the best hyperparameters from this search and save
        save_best_values_from_tuner(tuner, val_ds, save_name, save=True,
                                    path_to_ml_data=path_to_ml_data)

        # We also save the label_dict
        save_label_dict(save_name, learning_dir=learning_data_dir, path_to_training_data=path_to_training_data,
                        path_to_model=path_to_ml_data)



if __name__ == '__main__':
    # Notes of experience:
    # I suspect that continuing existing tuners does not work properly, so I always recommend starting a new tuner
    # instead of continuing an existing one

    from general_utils.pickle_functions import *

    # For developing - a minimum working example

    path_to_ml_data = PATH_TO_ML_DATA
    model_name = 'MobileNetV2'
    train_prefix = '20240628_testing'
    learning_data_dir = 'actnow_20240224_test_subset'
    path_to_training_data = PATH_TO_TRAINING_DATA
    shuffle_seed = 1234
    augment = False
    as_grayscale = False
    adjust_contrast = False
    pad_value = 1.
    fit_verbose = 1

    build_model_func = partial(build_model_new,
                               path_to_ml_data=path_to_ml_data,
                               loss=keras.losses.SparseCategoricalCrossentropy(from_logits=False),
                               optimizer=partial(tfa.optimizers.SGDW, nesterov=True, momentum=0.9, name="SGDW"),
                               freeze_base_model=True,
                               use_imagenet_weights=True,
                               lr_min=1e-4,
                               lr_max=1e-2,
                               opt_par_min=1e-6,
                               opt_par_max=1e-3,
                               batch_size=32,
                               decay_function=partial(keras.optimizers.schedules.ExponentialDecay,
                                                      decay_rate=0.5))

    decay_steps = 25
    num_trials = 2
    num_initial_points = 2
    overwrite = True
    epochs_per_exc = 3

    save_name = f"{model_name}_{train_prefix}"
    tuner_path = get_model_tuner_path(save_name, path_to_ml_data=path_to_ml_data)
    save_path_tb = get_tb_logs_path(save_name, path_to_ml_data=path_to_ml_data)

    print(f"\nStarting run_tuner() on {save_name}\n")

    train_ds, train_dict = pvnp_import.import_learning_set_from_dir_as_ds(learning_dir=learning_data_dir,
                                                                          subset='training',
                                                                          image_size=model_dict[model_name]['img_size'],
                                                                          batch_size=32, seed=shuffle_seed,
                                                                          augment=augment, as_grayscale=as_grayscale,
                                                                          adjust_contrast=adjust_contrast,
                                                                          pad_value=pad_value,
                                                                          path_to_training_data=path_to_training_data)
    val_ds, _ = pvnp_import.import_learning_set_from_dir_as_ds(learning_dir=learning_data_dir,
                                                               subset='validation',
                                                               image_size=model_dict[model_name]['img_size'],
                                                               batch_size=32, seed=shuffle_seed,
                                                               as_grayscale=as_grayscale,
                                                               adjust_contrast=adjust_contrast,
                                                               pad_value=pad_value,
                                                               path_to_training_data=path_to_training_data)

    class_weights = pvnp_build.calc_class_weights(train_dict)
    num_classes = len(train_dict)
    total_length = get_total_length_with_augment(learning_data_dir, augment,
                                                 path_to_training_data=path_to_training_data)

    tuner = BayesianOptimization(
        MyHyperModel(model_name, num_classes, build_model_func,
                     decay_steps, total_length),
        objective="val_accuracy",
        max_trials=num_trials,
        num_initial_points=num_initial_points,
        seed=1234,
        overwrite=overwrite,
        directory=str(tuner_path.parent),
        project_name=str(tuner_path.name))

    history_dict = tuner.search_return_history(train_ds, val_ds,
                                               epochs=epochs_per_exc,
                                               class_weight=class_weights,
                                               # callbacks=callbacks,
                                               verbose=fit_verbose)

    save_tuner_hist(history_dict, model_dir=save_name, path_to_ml_data=path_to_ml_data, verbose=1)



