#!/usr/bin/env python3
import urllib.error

import sys
import shutil
import datetime
import time
import statistics
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import layers
from keras.models import Model
from keras.utils import io_utils, tf_utils
import tensorflow_hub as tf_hub

from plankton_cnn.pvnp_save_and_load_utils import *
from plankton_cnn import pvnp_import, pvnp_models
from plankton_cnn.pvnp_models import model_dict, N_COLOR_CHANNELS


class TerminateFinetuning(keras.callbacks.Callback):
    """Callback that terminates training when the training accuracy decreases below the baseline. Useful for
    hyperparameter tuning during finetuning, where continuing is pointless when all accuracy from the previous
    training is lost already."""

    def __init__(self, baseline):
        super().__init__()
        self._supports_tf_logs = True
        self.baseline = baseline

    def on_batch_end(self, batch, logs=None):
        logs = logs or {}
        acc = logs.get("accuracy")
        if acc is not None:
            acc = tf_utils.sync_to_numpy_or_python_type(acc)
            if acc < self.baseline:
                io_utils.print_msg(
                    f"Batch {batch}: Accuracy decreased below baseline {self.baseline}, terminating training"
                )
                self.model.stop_training = True


# Source: https://www.kaggle.com/code/kerneler/fine-tune-deit-model
class WarmUpCosine(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(
        self, learning_rate_base, total_steps, warmup_learning_rate, warmup_steps
    ):
        super(WarmUpCosine, self).__init__()

        self.learning_rate_base = learning_rate_base
        self.total_steps = total_steps
        self.warmup_learning_rate = warmup_learning_rate
        self.warmup_steps = warmup_steps
        self.pi = tf.constant(np.pi)

    def __call__(self, step):
        if self.total_steps < self.warmup_steps:
            raise ValueError("Total_steps must be larger or equal to warmup_steps.")
        learning_rate = (
            0.5
            * self.learning_rate_base
            * (
                1
                + tf.cos(
                    self.pi
                    * (tf.cast(step, tf.float32) - self.warmup_steps)
                    / float(self.total_steps - self.warmup_steps)
                )
            )
        )

        if self.warmup_steps > 0:
            if self.learning_rate_base < self.warmup_learning_rate:
                raise ValueError(
                    "Learning_rate_base must be larger or equal to "
                    "warmup_learning_rate."
                )
            slope = (
                self.learning_rate_base - self.warmup_learning_rate
            ) / self.warmup_steps
            warmup_rate = slope * tf.cast(step, tf.float32) + self.warmup_learning_rate
            learning_rate = tf.where(
                step < self.warmup_steps, warmup_rate, learning_rate
            )
        return tf.where(
            step > self.total_steps, 0.0, learning_rate, name="learning_rate"
        )


def import_and_save_model(model_name, path_to_ml_data, overwrite=False):
    """
    Function to store models from tensorflow-hub or locally constructed models, so they can be loaded from the local
    disk afterwards .Models need to be defined in model_dict in pvnp_models.py either as a string
    (a link to tensorflow-hub) or as a function (if locally defined). Function needs to be executed only once for
    every model.

    :param model_name: str - key of the model in model_dict
    :return: None
    """
    model_path = model_dict[model_name]["link"]
    img_size = model_dict[model_name]["img_size"]

    # We make a model_dir consisting of just the model_name, so without the train_prefix
    model_save_path = get_saved_model_path(model_name, path_to_ml_data=path_to_ml_data)

    if os.path.isdir(model_save_path):
        if overwrite:
            shutil.rmtree(model_save_path, ignore_errors=True)
            print(f"Existing folder {model_save_path} was removed")
        else:
            print(f"A folder already exists at {model_save_path} - choose overwrite=True to remove")

    if isinstance(model_path, str):
        # Loading external model
        try:
            layer = tf_hub.KerasLayer(model_path, trainable=False)
        except urllib.error.HTTPError as error:
            print(error)
            print(f"{model_name} skipped")
        else:
            # Load the model from tf_hub and create the most simple model architecture - without the dense layer
            # (which depends on the number of classes)
            if model_name.startswith('deit'):
                inputs = tf.keras.Input((img_size, img_size, N_COLOR_CHANNELS))
                hub_module = tf_hub.KerasLayer(model_path, trainable=True)
                outputs, _ = hub_module(inputs)  # Second output in the tuple is a dictionary containing attention scores.

                model = tf.keras.Model(inputs, outputs)
                model.save(model_save_path)
                print(f"Model from {model_path} was saved to {model_save_path}")
            else:
                model = tf.keras.Sequential(layer)
                model.build([None, img_size, img_size, N_COLOR_CHANNELS])
                model.save(model_save_path)
                print(f"Model from {model_path} was saved to {model_save_path}")
    else:
        # Loading own model
        model = model_path
        model.build([None, img_size, img_size, N_COLOR_CHANNELS])
        model.save(model_save_path)
        print(f"Model from {model_path} was saved to {model_save_path}")


def load_model_from_disk(model_name, train_prefix, path_to_ml_data, compile=False, verbose=True):
    """
    First we try to import <model_name>, stored as a SavedModel from local storage. This requires a model to be stored
    as a folder named 'model_name'.

    If a model was not found or could not be loaded due to presence of custom objects, we try and load the model from
    its checkppoint. For this, we first need to determine the path to the best checkpoint, which is different for models
    originating from run_tuner and models originating from run_training_procedure (single training)

    :param train_prefix:
    :param model_name: str - name of model
    :param path_to_ml_data:
    :param compile: bool
    :param verbose: bool
    :return: TensorFlow model
    """

    model_dir = f"{model_name}_{train_prefix}"
    model_local_path = get_best_model_path(model_dir, path_to_ml_data=path_to_ml_data)

    try:
        model = keras.models.load_model(model_local_path, compile=compile)

    except (OSError, ValueError) as error:
        print(f"\nLoading model from {model_local_path} generated an error")
        # print(error)
        print("We proceed by loading the model from its' checkpoint path")

        checkpoint_path = get_best_checkpoint_path(model_dir, path_to_ml_data=path_to_ml_data)
        label_dict = load_saved_label_dict(model_dir, path_to_models=path_to_ml_data)

        model = build_model(model_name, len(label_dict), classifier_activation='softmax',
                            path_to_ml_data=path_to_ml_data)
        model.compile(metrics=['accuracy'])
        model.load_weights(checkpoint_path)

        if verbose:
            print(f"Loaded model from {checkpoint_path} at local storage")
    else:
        if verbose:
            print(f"Located SavedModel {model_dir} at local storage")

    return model


def build_model(model_name, num_classes, path_to_ml_data, classifier_activation='softmax', dropout=None,
                use_imagenet_weights=True):
    """
    TO DO: check how new, locally defined models are handled

    Load a generic model and construct it based on the specific requirements of the training data. Among others, a final
    dense layer is appended based on the number of classes. For the existing models, specific features such as
    preprocessing layers, final layers and kernel initializers are defined in accordance with the fully trained models
    as they exist in keras.applications.

    :param model_name: str - name of the model. Only models listed in pvnp_models.model_dict are allowed.
    :param num_classes: int - number of classes in the training data
    :param classifier_activation: name of the activation in the final dense layer. Any argument that can be passed
                                  to layers.Dense is accepted. No reason to use anything else than softmax - note that
                                  some apply-model-functions may assume models have a softmax layer.
    :param dropout: bool - if True, a dropout-layer is added before the final layers. Only applies to tf-hub models!
    :param use_imagenet_weights: bool - if True, weights of the model is trained on ImageNet are used. If False,
                                        the model weights are newly initialized.
    :param path_to_ml_data: str - if applicable, the folder where the base models downloaded from tf_hub are located. Do
                                  not modify this path to the project-specific subdirectories.
    :return: tf.model
    """

    model_path = model_dict[model_name]["link"]
    img_size = model_dict[model_name]["img_size"]

    if isinstance(model_path, str):
        model_save_path = get_best_model_path(model_name, path_to_ml_data=path_to_ml_data)
        if not os.path.isdir(model_save_path):
            import_and_save_model(model_name, path_to_ml_data=path_to_ml_data)

        # if hub-model at local storage
        model = keras.models.load_model(model_save_path, compile=False)

        x = model(model.input)
        outputs = layers.Dense(num_classes, activation=classifier_activation)(x)
        model = tf.keras.Model(model.input, outputs, name=model_name)

        return model

    else:
        # if from Keras applications
        if use_imagenet_weights:
            weights = 'imagenet'
        else:
            weights = None

        base_model = model_path(include_top=False, weights=weights,
                                input_shape=(img_size, img_size, N_COLOR_CHANNELS))

        # check what 'training parameter' does in batch normalization layers.

        kernel_initializer = 'glorot_uniform'
        bias_initializer = 'zeros'

        if model_name == 'Xception':
            # Done
            preprocess = tf.keras.applications.xception.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['VGG16', 'VGG19']:
            # Done
            if model_name == 'VGG16':
                preprocess = tf.keras.applications.vgg16.preprocess_input
            else:
                preprocess = tf.keras.applications.vgg19.preprocess_input

            preds = layers.Flatten(name='flatten')(base_model.output)
            preds = layers.Dense(4096, activation='relu', name='fc1')(preds)
            preds = layers.Dense(4096, activation='relu', name='fc2')(preds)

        elif model_name in ['ResNet50V2', 'ResNet101V2', 'ResNet152V2']:
            # Done
            preprocess = tf.keras.applications.resnet_v2.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name == 'InceptionV3':
            # Done
            preprocess = tf.keras.applications.inception_v3.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name == 'InceptionResNetV2':
            # Done
            preprocess = tf.keras.applications.inception_resnet_v2.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name == 'MobileNetV2':
            # Done
            preprocess = tf.keras.applications.mobilenet_v2.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['DenseNet121', 'DenseNet169', 'DenseNet201']:
            # Done
            preprocess = tf.keras.applications.densenet.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['NASNetMobile', 'NASNetLarge']:
            # Done
            preprocess = tf.keras.applications.nasnet.preprocess_input
            preds = layers.GlobalAveragePooling2D(name='avg_pool')(base_model.output)

        elif model_name in ['EfficientNetV2B0', 'EfficientNetV2B1',
                            'EfficientNetV2B2', 'EfficientNetV2B3',
                            'EfficientNetV2S']:
            preprocess = layers.Rescaling(1.)
            preds = layers.GlobalAveragePooling2D()(base_model.output)

            kernel_initializer = {"class_name": "VarianceScaling",
                                  "config": {"scale": 1. / 3.,
                                             "mode": "fan_out",
                                             "distribution": "uniform"}}
            bias_initializer = tf.constant_initializer(0)

        # Build the preprocessing model
        # The model-specific preprocessing layers require the model input to be in [0, 255]. Here we
        # assume the model input is in [0, 1] conform our import-function, so we add a rescaling layer.
        x = layers.Rescaling(255)(base_model.input)
        x = preprocess(x)
        input_model = Model(inputs=base_model.input, outputs=x,
                            name=f"Preprocessing-{model_name}")

        # Build the final classifier model
        preds = layers.Dense(num_classes,
                             activation=classifier_activation,
                             kernel_initializer=kernel_initializer,
                             bias_initializer=bias_initializer,
                             name='predictions')(preds)

        classifier_model = Model(inputs=base_model.output,
                                 outputs=preds,
                                 name='Classifier')

        # Build the complete model, using the preprocessing, base and classifier model
        y = input_model(input_model.input)
        y = base_model(y, training=False)
        # Setting training to False has to do with the batch normalization layers
        # see https://keras.io/guides/transfer_learning/#freezing-layers-understanding-the-trainable-attribute
        y = classifier_model(y)
        model = Model(inputs=input_model.input, outputs=y,
                      name=f"{model_name}-imnet")

        return model


def calc_class_weights(train_dict):
    """
    Calculate the weights of the groups in train_dict.

    The weights are such that they balance the length of the groups in order to
    obtain equal contribution to the network's loss function. I.e., by passing these weights
    to the training procedure, errors in rare groups get a larger contribution to the loss function
    whereas the contribution from errors in abundant groups becomes smaller. This prevents the network
    from fitting too much to the most abundant groups only.

    :param train_dict: dict - as returned by pvnp_import.import_plankton_images
    with the groups' integer labels as keys, as items dictionaries with key 'length' and 'name'
    :return: dict - with integer labels as keys, weights per group as items
    """
    mean_length = statistics.mean(map(
        lambda key: train_dict[key]['length'], train_dict.keys()))
    weights = list(map(
        lambda key: mean_length / train_dict[key]['length'], train_dict.keys()))
    return dict(zip(train_dict.keys(), weights))


def run_training_procedure(model_name, train_prefix_new, train_prefix_prev,
                           learning_dir, batch_size, augment, seed, as_grayscale, pad_value,
                           finetuning, optimizer, loss,
                           epochs, perf_monitor, es_min_delta, es_patience, es_verbose,
                           path_to_ml_data, path_to_training_data,
                           fit_verbose,
                           preproc_prefix=None, adjust_contrast=True, nr_rounds=None, one_hot_labels=False,
                           use_imagenet_weights=True, classifier_activation=None, dropout=None, save_model=True,
                           prefetch=True,
                           ):
    """
    Run the full training procedure: import the training data from the local storage, import a locally stored model,
    run the training and save both the best model result and the training history both.
    In case files already exist with the same train_prefix_new, these will be removed prior to the new training.

    :param model_name: str - see function 'build_model'
    :param train_prefix_new: str - name of the training procedure that will be used in the resulting saved files
    :param train_prefix_prev: str - if specified, load an already trained model to continue training. Only the model is
                                    loaded, any training parameters need to be specified again.
    :param learning_dir: str - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param batch_size: int - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param augment: str - either 'simple' or 'all_rotations' or None. For more information see
                          function 'import_learning_set_from_dir_as_ds_optimised'
    :param seed: int - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param as_grayscale: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param pad_value: float in [0, 1] or 'mean_img': - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param finetuning: bool - if False, only the final dense layer(s) are trained, all others are 'frozen'
    :param optimizer: keras.optimizers-function
    :param loss: keras.losses-function
    :param epochs: int - number of training epochs
    :param perf_monitor: str - should be one of ['loss', 'val_loss', 'accuracy', 'val_accuracy']. Metric that is used
                               to assess model performance
    :param es_min_delta: float - minimum improvement in perf_monitor in order to prevent early stopping
    :param es_patience: int - maximum number of epochs that can be passed without improvement before early stopping
    :param es_verbose: 0 or 1 - verbosity mode of the early stopping. Mode 0 is silent, and mode 1 displays messages
                                when the callback takes action.
    :param fit_verbose: int - Verbosity mode. 0 = silent, 1 = progress bar, 2 = one line per epoch.
    :param preproc_prefix: str - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param adjust_contrast: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param nr_rounds: int - number of rounds. If > 1, multiple rounds with the same parameters are executed and the best
                            results are selected
    :param one_hot_labels: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param use_imagenet_weights: bool - see function 'build_model'
    :param classifier_activation: str - see function 'build_model'
    :param dropout: bool - see function 'build_model'
    :param save_model: bool
    :param prefetch: bool - see function 'import_learning_set_from_dir_as_ds_optimised'
    :param path_to_ml_data: str - path to the folder where the models and files related to the model will be saved.
                                  These will be stored in a newly created folder as:
                                   <path_to_ml_data>/<model_name>_<train_prefix_new>
    :param path_to_training_data: str - see function 'import_learning_set_from_dir_as_ds_optimised'

    :return: float - elapsed time during the fitting procedure

    """

    # Define the save names
    model_dir = f"{model_name}_{train_prefix_new}"
    print(f"\nStarting run_training_procedure on {model_dir}\n")

    save_name_cp = get_model_checkpoint_path(model_dir, path_to_ml_data=path_to_ml_data)
    checkpoint_path = f"{save_name_cp}/best_cp.ckpt"
    tensorboard_path = get_tb_logs_path(model_dir, path_to_ml_data=path_to_ml_data)

    image_size = model_dict[model_name]["img_size"]

    if not os.path.isdir(dest_folder := path_to_ml_data / model_dir):
        os.mkdir(dest_folder)

    # Remove existing save files
    if os.path.isdir(save_name_cp):
        shutil.rmtree(save_name_cp, ignore_errors=True)
        print(f"Existing folder {save_name_cp} was removed")

    if os.path.isdir(tensorboard_path):
        shutil.rmtree(tensorboard_path, ignore_errors=True)
        print(f"Existing folder {tensorboard_path} was removed")

    # Import the learning set
    train_ds, train_dict = pvnp_import.import_learning_set_from_dir_as_ds_optimised(subset='training',
                                                                                    learning_dir=learning_dir,
                                                                                    image_size=image_size,
                                                                                    augment=augment,
                                                                                    preproc_prefix=preproc_prefix,
                                                                                    batch_size=batch_size,
                                                                                    seed=seed,
                                                                                    as_grayscale=as_grayscale,
                                                                                    adjust_contrast=adjust_contrast,
                                                                                    pad_value=pad_value,
                                                                                    one_hot_labels=one_hot_labels,
                                                                                    prefetch=prefetch,
                                                                                    path_to_training_data=path_to_training_data)
    val_ds, _ = pvnp_import.import_learning_set_from_dir_as_ds_optimised(subset='validation', learning_dir=learning_dir,
                                                                         image_size=image_size, augment=augment,
                                                                         preproc_prefix=preproc_prefix,
                                                                         batch_size=batch_size,
                                                                         seed=seed, as_grayscale=as_grayscale,
                                                                         adjust_contrast=adjust_contrast,
                                                                         pad_value=pad_value,
                                                                         prefetch=prefetch,
                                                                         one_hot_labels=one_hot_labels,
                                                                         path_to_training_data=path_to_training_data)

    class_weights = calc_class_weights(train_dict)

    print("For training:")
    for dic in train_dict.values():
        print(f" - using {dic['length']} files with label {dic['name']}")

    # Save label_dict
    save_label_dict(model_dir, learning_dir,
                                             path_to_training_data=path_to_training_data, path_to_model=path_to_ml_data)

    # Import the model
    if train_prefix_prev:
        # Resume training from saved model
        model = load_model_from_disk(model_name=model_name, train_prefix=train_prefix_prev,
                                     path_to_ml_data=path_to_ml_data, verbose=True)
    else:
        # Load new model
        model = build_model(model_name, len(train_dict),
                            classifier_activation=classifier_activation,
                            dropout=dropout,
                            path_to_ml_data=path_to_ml_data,
                            use_imagenet_weights=use_imagenet_weights)

    if finetuning:
        for layer in model.layers:  # Could also be the last xx layers
            layer.trainable = True
    else:
        for layer in model.layers[:-1]:
            layer.trainable = False

    model.summary()
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])

    # Early stopping - fit procedure stops if es_monitor hasn't increased
    # over last xx epochs
    es_callback = tf.keras.callbacks.EarlyStopping(
        monitor=perf_monitor, min_delta=es_min_delta, patience=es_patience,
        verbose=es_verbose, mode="auto", restore_best_weights=True)

    cp_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path, save_weights_only=True,
        save_best_only=True, monitor=perf_monitor, verbose=1)

    # Here the actual training starts
    t_before = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        callbacks=[cp_callback,
                                   es_callback,
                                   keras.callbacks.TensorBoard(tensorboard_path),
                                   keras.callbacks.TerminateOnNaN()],
                        class_weight=class_weights,
                        verbose=fit_verbose)
    t_elapsed = time.time() - t_before

    # We save the training history
    save_history(history, model_dir, path_to_models=path_to_ml_data)

    if save_model:
        # We save the complete model, and remove the existing file if it exists already
        model_save_path = get_saved_model_path(model_dir, path_to_ml_data=path_to_ml_data)
        model.load_weights(checkpoint_path)

        if os.path.isdir(model_save_path):
            shutil.rmtree(model_save_path, ignore_errors=True)
            print(f"Existing folder {model_save_path} was removed")
        model.save(model_save_path)
        print(f"model was saved to {model_save_path}")

    return t_elapsed


if __name__ == '__main__':
    
    from general_utils.pickle_functions import *

    from tensorflow import keras
    from keras import layers

    model_name = 'deit_tiny_patch16_224_fe'
    model_path = model_dict[model_name]["link"]
    img_size = model_dict[model_name]["img_size"]



    # # Load the model from tf_hub and create the model architecture
    # inputs = tf.keras.Input((img_size, img_size, N_COLOR_CHANNELS))
    # hub_module = tf_hub.KerasLayer(model_path, trainable=True)
    #
    # outputs, _ = hub_module(inputs)  # Second output in the tuple is a dictionary containing attention scores.
    # # outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    #
    # model = tf.keras.Model(inputs, outputs)
    # model.summary()

    # import_and_save_model(model_name, path_to_ml_data=PATH_TO_ML_DATA)

    # model_local_path = get_best_model_path(model_name, path_to_ml_data=PATH_TO_ML_DATA)
    #
    # model = keras.models.load_model(model_local_path, compile=compile)

    model = build_model(model_name, num_classes=4, classifier_activation='softmax', path_to_ml_data=PATH_TO_ML_DATA)
    model.summary()

    # import_and_save_model(model_name, path_to_ml_data=PATH_TO_ML_DATA)
    sys.exit()

    name = "EfficientNetV2B0"
    model = build_model(name, 3, use_imagenet_weights=True, path_to_ml_data=PATH_TO_ML_DATA)
    model.summary()

    submodel = model.get_layer('efficientnetv2-b0')
    submodel.trainable = False
    submodel.summary()
    # sys.exit()

    submodel = model.get_layer(name='Classifier')
    submodel.summary()
    sys.exit()

    # model = build_model("EfficientNetV2B0", 28, use_imagenet_weights=True)
    # sys.exit()

    # save_new_models()
    # # load_training_args(f"{PATH_TO_CHECKPOINTS}/mobilenet_v2_050_160_test/training_args.txt")
    # sys.exit()

    model = load_model("efficientnet_v2_240")
    model.summary()
    # save_label_dict("alpha")
    # print(load_label_dict("alpha"))

