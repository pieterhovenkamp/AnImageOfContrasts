#!/usr/bin/env python3

import sys
from pathlib import Path

import numpy as np
import skimage.exposure
from skimage import io, util
import tensorflow as tf

from tensorflow.python.ops import image_ops, io_ops
from tensorflow.python.data.ops import dataset_ops
from keras.utils import dataset_utils

from general_utils.pickle_functions import *

if __name__ == '__main__':
    print(tf.__version__)


def get_labeL_dict_for_training_directory(learning_dir, path_to_folder):
    """
    Make a dictionary of the integer labels and the class names (as specified in the names
    of the directories inside training_dir).

    The correct order is guaranteed: in dataset_utils.index_directory, with class_names=None,
    class names are returned in alphanumerical order. With labels='inferred',
    integer labels are also assigned in alphanumerical order of the class names. Since np.unique()
    also sorts in alphanumerical order, it is guaranteed that the order of class_names
    and unique_labels is the same.

    :param learning_dir: str - nmae of the training directory
    :param path_to_folder: str - path to the training directory (the parent directory of learning_dir)
    :return: dict - with keys the integer labels, values the names as strings
    """
    image_paths, labels, class_names = dataset_utils.index_directory(
        Path(path_to_folder) / learning_dir / "training", labels="inferred", formats=(".png", ".jpg"), class_names=None)

    return dict(zip(np.unique(labels), class_names))


def import_learning_set_from_dir_as_ds(learning_dir, subset, image_size, batch_size,
                                       path_to_training_data,
                                       shuffle_buffer=10000, seed=1234,
                                       augment='simple', as_grayscale=False, pad_value=0., adjust_contrast=True,
                                       one_hot_labels=False,
                                       ):
    """

    :param one_hot_labels: bool - whether to convert the integer labels of the training and validation set to
                                  one-hot vectors.
    :param learning_dir: str - should contain subdirectory named subset ('training', 'validation' or 'test')
    :param image_size: int
    :param batch_size: int
    :param seed: int
    :param subset: str - should be 'training' or 'validation'
    :param shuffle_buffer: int - size of buffer during shuffling, see
                              https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
                              If smaller than size of dataset, randomness during shuffling is compromised. If too large,
                              leads to memory errors.
    :param augment: str - either 'simple' or 'all_rotations' or None
    :param as_grayscale: bool -
    :param pad_value: float in [0, 1] or 'mean_img':
    :param path_to_training_data: str
    :return:
    """
    num_channels = 3

    if subset not in ['validation', 'training']:
        raise ValueError(f"Value for subset {subset} not recognized. Options are: "
                         "'validation', 'training'")

    image_paths, labels, class_names = dataset_utils.index_directory(
        Path(path_to_training_data) / learning_dir / subset,
        labels='inferred', formats=(".png", ".jpg"), class_names=None, seed=seed)

    # Make a dictionary of the numbers of images per group
    class_dict = {}
    for label, name in zip(np.unique(labels), class_names):
        label_dict = {'length': np.sum(labels == label),
                      'name': name}
        class_dict[label] = label_dict

    if one_hot_labels:
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))

    # Convert the image paths to images using load_image()
    images_ds = convert_paths_to_ds_of_images_and_labels(image_paths=image_paths, image_size=image_size,
                                                         num_channels=num_channels, labels=labels,
                                                         as_grayscale=as_grayscale, adjust_contrast=adjust_contrast,
                                                         pad_value=pad_value)

    if subset == 'training':
        if augment == 'simple':
            print(f"\nAugmentation '{augment}' was applied to training data")
            images_ds = augm_image_ds(images_ds, all_rotations=False)
        elif augment == 'all_rotations':
            print(f"\nAugmentation '{augment}' was applied to training data")
            images_ds = augm_image_ds(images_ds, all_rotations=True)
        else:
            print("\nArgument for data augmentation was not recognised. No augmentation applied")

        if shuffle_buffer:
            # We shuffle only train_ds, so val_ds can be evaluated manually.
            # With 10.000, the buffer size can be lower than the size of the dataset but this is
            # necessary since for large datasets the complete set might not fit into the memory
            # for shuffling. This compromises the randomness though
            # (see https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle)
            images_ds = images_ds.shuffle(buffer_size=shuffle_buffer, seed=seed)
    images_ds = images_ds.batch(batch_size)

    print(f"Using {sum([dic['length'] for dic in class_dict.values()])} images for {subset}\n")
    if as_grayscale:
        print("Imported images as grayscale")

    return images_ds, class_dict


def export_preprocessed_images(learning_dir, preproc_prefix, image_size, num_channels,
                               as_grayscale, adjust_contrast, pad_value, augment,
                               path_to_training_data):
    """
     A copy of the training, validation and test set with the specified pre-processing (i.e. augment, as_grayscale, pad
     and adjust_contrast) are saved locally in a folder <learning_dir>/<preproc_prefix>.

     See function 'import_learning_set_from_dir_as_ds_optimised' for parameter descriptions.

    :param learning_dir:
    :param preproc_prefix:
    :param image_size:
    :param num_channels:
    :param as_grayscale:
    :param adjust_contrast:
    :param pad_value:
    :param augment:
    :param path_to_training_data:
    :return:
    """

    print(f"Running export_preprocessed_images to {preproc_prefix} - eager execution? ", tf.executing_eagerly())

    def save_tf_img(img, path):
        img = img.numpy()
        img = skimage.exposure.rescale_intensity(img, out_range=(img.min(), 1))
        try:
            img = util.img_as_ubyte(img)
        except ValueError as e:
            print(f"Error while saving {path}")
            print(e)
        else:
            io.imsave(path, img, check_contrast=False)

    dest_folder_augm = path_to_training_data / learning_dir / preproc_prefix
    if not os.path.isdir(dest_folder_augm):
        os.mkdir(dest_folder_augm)

    for subset in ['training', 'validation', 'test']:

        subset_path = Path(path_to_training_data) / learning_dir / subset
        if os.path.isdir(subset_path):
            print(f"\nExport {subset} set")
            image_paths, _, groups = dataset_utils.index_directory(
                subset_path,
                labels='inferred', formats=(".png", ".jpg"), class_names=None, shuffle=False)
            labels = map(lambda x: Path(x).parent.name, image_paths)

            dest_folder = dest_folder_augm / subset
            if not os.path.isdir(dest_folder):
                os.mkdir(dest_folder)

            for group in groups:
                if not os.path.isdir(dest_folder / group):
                    os.mkdir(dest_folder / group)

            for img_path, label in zip(image_paths, labels):
                tf_img = load_image(img_path, num_channels, image_size=image_size,
                                    as_grayscale=as_grayscale,
                                    adjust_contrast=adjust_contrast,
                                    pad_value=pad_value)
                label_folder = dest_folder / label

                # Save the original image
                save_tf_img(tf_img, label_folder / Path(img_path).name)

                if subset == 'training':
                    if augment == 'all_rotations':
                        for num in [1, 2, 3]:
                            tf_img_augm = augm_image(tf_img, rotate=False)
                            tf_img_augm = tf.image.rot90(tf_img_augm, k=num)

                            # Save the augmented image
                            save_tf_img(tf_img_augm, label_folder / f"{Path(img_path).stem}_augm{num}.png")

                    elif augment == 'simple':
                        # Create a duplicate for each image with randomly varying brightness/contrast/rotation
                        tf_img_augm = augm_image(tf_img, rotate=True)

                        # Save the augmented image
                        save_tf_img(tf_img_augm, label_folder / f"{Path(img_path).stem}_augm1.png")
                    elif not augment:
                        print("\nNo augmentation was applied")
                    else:
                        print("\nArgument for data augmentation was not recognised. No augmentation was applied")

    print("\nexport_preprocessed_images finished!\n")


def import_learning_set_from_dir_as_ds_optimised(learning_dir, subset, image_size, batch_size,
                                                 path_to_training_data,
                                                 shuffle_buffer=None, seed=1234, preproc_prefix=None,
                                                 augment=None, as_grayscale=False, pad_value=0.,
                                                 adjust_contrast=True,
                                                 one_hot_labels=False,
                                                 prefetch=True,
                                                 ):
    """

    :param learning_dir: str - name of the folder that contains the training data. Images need to be organised in
                               folders per class in separate folders for training, validation and optionally test data.
                               I.e.: learning_dir
                                        training
                                            class1
                                                <image1>
                                                <image2>
                                                ...
                                            class2
                                                <image1>
                                                <image2>
                                                ...
                                        validation
                                            ...
                                        test
                                            ...
    :param subset: str - should be 'training' or 'validation'
    :param batch_size: int - Images enter the training procedure in batches, i.e. model weights are updated per batch.
                             Too small batches may make model training unstable, but too large may cause memory issues.
    :param image_size: int - image size in pixels to which images are resized (using padding)
    :param shuffle_buffer: int - size of buffer during shuffling, see
                              https://stackoverflow.com/questions/46444018/meaning-of-buffer-size-in-dataset-map-dataset-prefetch-and-dataset-shuffle
                              If smaller than size of dataset, randomness during shuffling is compromised. If too large,
                              leads to memory errors.
    :param seed: int - random seed that is used when shuffling the data (for reproducibility). Only applies to initial
                       shuffling when import images, not to the shuffling during training.
    :param preproc_prefix: str - if specified (which is necessary to use augment), a copy of the training, validation
                                 and test set with the specified pre-processing (i.e. augment, as_grayscale, pad and
                                 adjust_contrast) are saved locally in a folder <learning_dir>/<preproc_prefix>.
    :param augment: str - either 'simple' or 'all_rotations' or None. Defines which pre-defined augmentation is used to
                          artificially increase the number of training images. Additional options need to be defined
                          explicitly in function 'export_preprocessed_images' and in AUGMENT_DICT.
    :param as_grayscale: bool - whether to convert colour images to grayscale before training. This is relevant only for
                                colour images.
    :param pad_value: float in [0, 1] or 'mean_img': - value of the padded pixels. Choose 1 (white) for white background
                                                       and 0 (black) for black background.
    :param adjust_contrast: bool - whether to increase contrast of images before training
    :param one_hot_labels: bool - whether to convert the integer labels of the training and validation set to
                                  one-hot vectors.
    :param prefetch: bool - whether prefetch is used when importing the images (see Tensorflow docs)
    :param path_to_training_data: str - path to the folder in which the folder <learning_dir> is located.
    :return:
    """

    num_channels = 3
    path_to_training_data = Path(path_to_training_data)

    if subset not in ['validation', 'training']:
        raise ValueError(f"Value for subset {subset} not recognized. Options are: "
                         "'validation', 'training'")

    if not preproc_prefix and augment:
        raise ValueError("If you want augmentations to be applied, a preproc_prefix should be specified.")

    if preproc_prefix:
        preproc_dir = Path(learning_dir) / preproc_prefix
        if not os.path.isdir(path_to_training_data / preproc_dir):
            export_preprocessed_images(learning_dir, preproc_prefix, image_size, num_channels, as_grayscale,
                                       adjust_contrast, pad_value, augment, path_to_training_data=path_to_training_data)
        else:
            print(f"\nExisting folder {preproc_dir} was found")

        # Load images from this dir using code below (but adjust load_image)
        learning_dir = preproc_dir

        # This was done already in export_preprocessed_images so we don't have to do it again
        as_grayscale = False
        adjust_contrast = False

    image_paths, labels, class_names = dataset_utils.index_directory(path_to_training_data / learning_dir / subset,
                                                                     labels='inferred', formats=(".png", ".jpg"),
                                                                     class_names=None, seed=seed)

    # Make a dictionary of the numbers of images per group
    class_dict = {}
    for label, name in zip(np.unique(labels), class_names):
        label_dict = {'length': np.sum(labels == label),
                      'name': name}
        class_dict[label] = label_dict

    if one_hot_labels:
        labels = tf.keras.utils.to_categorical(labels, num_classes=len(class_names))

    # Convert paths to dataset and combine with the labels to a single dataset (before we shuffle)
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    label_ds = dataset_ops.Dataset.from_tensor_slices(labels)
    zipped_ds = dataset_ops.Dataset.zip((path_ds, label_ds))

    # Shuffle the training set before we load the images - this saves a lot of memory
    if subset == 'training':
        zipped_ds = zipped_ds.shuffle(buffer_size=len(image_paths) if not shuffle_buffer else shuffle_buffer, seed=seed)

    # Convert to a dataset containing the actual images using load_image
    img_ds = zipped_ds.map(lambda x, y: (load_image(x, num_channels, image_size=image_size,
                                                    as_grayscale=as_grayscale,
                                                    adjust_contrast=adjust_contrast,
                                                    pad_value=pad_value), y),
                           num_parallel_calls=tf.data.AUTOTUNE)

    # Consider to remove batch to training procedure instead of import function
    img_ds = img_ds.batch(batch_size)

    if prefetch:
        img_ds = img_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    print(f"Using {sum([dic['length'] for dic in class_dict.values()])} images for {subset}\n")

    # if as_grayscale:
    #     print("Imported images as grayscale")

    return img_ds, class_dict


def convert_paths_to_ds_of_images_and_labels(image_paths, image_size, num_channels, labels,
                                             as_grayscale=False, adjust_contrast=True, pad_value=0.):
    """

    :param image_paths:
    :param image_size:
    :param num_channels:
    :param labels:
    :param pad_value:
    :param as_grayscale:
    :return:
    """
    # Convert image_paths and labels to tf.datasets
    path_ds = dataset_ops.Dataset.from_tensor_slices(image_paths)
    label_ds = dataset_ops.Dataset.from_tensor_slices(labels)

    # Convert path_ds to a dataset containing the actual images using load_image and combine with the labels
    # to a single dataset
    img_ds = path_ds.map(lambda x: load_image(x, num_channels, image_size=image_size,
                                              as_grayscale=as_grayscale,
                                              adjust_contrast=adjust_contrast,
                                              pad_value=pad_value))
    return dataset_ops.Dataset.zip((img_ds, label_ds))


def prepare_tf_image(img, num_channels, image_size, as_grayscale=False, adjust_contrast=True, pad_value=0.):
    """

    :param img: tf.Tensor with dtype float32
    :param num_channels:
    :param image_size: int - shape of output image will be (image_size, image_size)
    :param num_channels: int - desired number of output channels. If 3 and input image is in grayscale,
                               then the output image (shape (image_size, image_size, 3)) will have
                               3 identical channels
    :param as_grayscale: bool - if True, then an RGB image will be converted to grayscale with 3 identical
                                channels (shape (image_size, image_size, 3)). No effect if input image on storage
                                is already in grayscale.
    :param adjust_contrast: bool
    :param pad_value: float in [0, 1] or 'mean_img':
    :return: tf.Tensor - output image
    """
    if adjust_contrast:
        img = tf.image.adjust_contrast(img, 1.3)

    if as_grayscale:
        img = to_grayscale(img)

    # Rescale before padding, because padding could add zeros. After resize_with_pad, pixel values might
    # not fully extend to 0 and 1 due to bilinear resizing.
    img = rescale_image_values(img)

    if pad_value == 'mean_img':
        pad_value = tf.math.reduce_mean(img)
    elif (pad_value > 1.) or (pad_value < 0.):
        raise ValueError("pad_value should be either 'mean_img' or in [0., 1.]")

    # img = tf.image.resize_with_pad(img, image_size, image_size)
    img = resize_image_with_pad_homemade(img, image_size, image_size, pad_value=pad_value)
    return tf.reshape(img, [image_size, image_size, num_channels])


# @tf.function
def load_image(path, num_channels, **img_kwargs):
    """
    Load an image from a path, convert pixel values to float in range [0, 1)
    and resize it with padding.

    :param path: str
    :param num_channels: int - desired number of output channels. If 3 and input image is in grayscale,
                               then the output image (shape (image_size, image_size, 3)) will have
                               3 identical channels
    :param img_kwargs: keyword arguments of prepare_tf_image. See docs of that function.
    :return: tf.Tensor - output image
    """
    if not tf.executing_eagerly():
        print("Running load_image in graph execution")

    img = io_ops.read_file(path)
    img = image_ops.decode_image(img, channels=num_channels, expand_animations=False, dtype=tf.float32)

    return prepare_tf_image(img, num_channels=num_channels, **img_kwargs)


def resize_image_with_pad_homemade(image,
                                   target_height,
                                   target_width,
                                   method=tf.image.ResizeMethod.BILINEAR,
                                   antialias=False,
                                   pad_value=0.):
    # Modified from tf.image.resize_with_pad - all necessary functions and import statements were
    # copied from image_ops_impl.py
    """Resizes and pads an image to a target width and height.

    Resizes an image to a target width and height by keeping
    the aspect ratio the same without distortion. If the target
    dimensions don't match the image dimensions, the image
    is resized and then padded with zeroes to match requested
    dimensions.

    Args:
    image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
      of shape `[height, width, channels]`.
    target_height: Target height.
    target_width: Target width.
    method: Method to use for resizing image. See `image.resize()`
    antialias: Whether to use anti-aliasing when resizing. See 'image.resize()'.

    Raises:
    ValueError: if `target_height` or `target_width` are zero or negative.

    Returns:
    Resized and padded image.
    If `images` was 4-D, a 4-D float Tensor of shape
    `[batch, new_height, new_width, channels]`.
    If `images` was 3-D, a 3-D float Tensor of shape
    `[new_height, new_width, channels]`.
    """
    from tensorflow.python.framework import ops
    from tensorflow.python.ops import array_ops
    from tensorflow.python.ops import control_flow_ops
    from tensorflow.python.ops import math_ops
    from tensorflow.python.framework import dtypes
    from tensorflow.python.ops import variables
    from tensorflow.python.ops import check_ops

    def _assert(cond, ex_type, msg):
        """A polymorphic assert, works with tensors and boolean expressions.

        If `cond` is not a tensor, behave like an ordinary assert statement, except
        that a empty list is returned. If `cond` is a tensor, return a list
        containing a single TensorFlow assert op.

        Args:
          cond: Something evaluates to a boolean value. May be a tensor.
          ex_type: The exception class to use.
          msg: The error message.

        Returns:
          A list, containing at most one assert op.
        """
        if _is_tensor(cond):
            return [control_flow_ops.Assert(cond, [msg])]
        else:
            if not cond:
                raise ex_type(msg)
            else:
                return []

    def _is_tensor(x):
        """Returns `True` if `x` is a symbolic tensor-like object.

        Args:
          x: A python object to check.

        Returns:
          `True` if `x` is a `tf.Tensor` or `tf.Variable`, otherwise `False`.
        """
        return isinstance(x, (ops.Tensor, variables.Variable))

    def _ImageDimensions(image, rank):
        """Returns the dimensions of an image tensor.

        Args:
          image: A rank-D Tensor. For 3-D  of shape: `[height, width, channels]`.
          rank: The expected rank of the image

        Returns:
          A list of corresponding to the dimensions of the
          input image.  Dimensions that are statically known are python integers,
          otherwise, they are integer scalar tensors.
        """
        if image.get_shape().is_fully_defined():
            return image.get_shape().as_list()
        else:
            static_shape = image.get_shape().with_rank(rank).as_list()
            dynamic_shape = array_ops.unstack(array_ops.shape(image), rank)
            return [
                s if s is not None else d for s, d in zip(static_shape, dynamic_shape)
            ]

    def _CheckAtLeast3DImage(image, require_static=True):
        """Assert that we are working with a properly shaped image.

            Args:
              image: >= 3-D Tensor of size [*, height, width, depth]
              require_static: If `True`, requires that all dimensions of `image` are known
                and non-zero.

            Raises:
              ValueError: if image.shape is not a [>= 3] vector.

            Returns:
              An empty list, if `image` has fully defined dimensions. Otherwise, a list
              containing an assert op is returned.
            """
        try:
            if image.get_shape().ndims is None:
                image_shape = image.get_shape().with_rank(3)
            else:
                image_shape = image.get_shape().with_rank_at_least(3)
        except ValueError:
            raise ValueError("'image' (shape %s) must be at least three-dimensional." %
                             image.shape)
        if require_static and not image_shape.is_fully_defined():
            raise ValueError('\'image\' must be fully defined.')
        if any(x == 0 for x in image_shape[-3:]):
            raise ValueError('inner 3 dims of \'image.shape\' must be > 0: %s' %
                             image_shape)
        if not image_shape[-3:].is_fully_defined():
            return [
                check_ops.assert_positive(
                    array_ops.shape(image)[-3:],
                    ["inner 3 dims of 'image.shape' "
                     'must be > 0.']),
                check_ops.assert_greater_equal(
                    array_ops.rank(image),
                    3,
                    message="'image' must be at least three-dimensional.")
            ]
        else:
            return []

    # In this function we added the parameter pad_value
    def pad_to_bounding_box_internal(image, offset_height, offset_width,
                                     target_height, target_width, pad_value, check_dims):
        """Pad `image` with zeros to the specified `height` and `width`.

        Adds `offset_height` rows of zeros on top, `offset_width` columns of
        zeros on the left, and then pads the image on the bottom and right
        with zeros until it has dimensions `target_height`, `target_width`.

        This op does nothing if `offset_*` is zero and the image already has size
        `target_height` by `target_width`.

        Args:
          image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
            of shape `[height, width, channels]`.
          offset_height: Number of rows of zeros to add on top.
          offset_width: Number of columns of zeros to add on the left.
          target_height: Height of output image.
          target_width: Width of output image.
          check_dims: If True, assert that dimensions are non-negative and in range.
            In multi-GPU distributed settings, assertions can cause program slowdown.
            Setting this parameter to `False` avoids this, resulting in faster speed
            in some situations, with the tradeoff being that some error checking is
            not happening.

        Returns:
          If `image` was 4-D, a 4-D float Tensor of shape
          `[batch, target_height, target_width, channels]`
          If `image` was 3-D, a 3-D float Tensor of shape
          `[target_height, target_width, channels]`

        Raises:
          ValueError: If the shape of `image` is incompatible with the `offset_*` or
            `target_*` arguments, or either `offset_height` or `offset_width` is
            negative. Not raised if `check_dims` is `False`.
        """
        with ops.name_scope(None, 'pad_to_bounding_box', [image]):
            image = ops.convert_to_tensor(image, name='image')

            is_batch = True
            image_shape = image.get_shape()
            if image_shape.ndims == 3:
                is_batch = False
                image = array_ops.expand_dims(image, 0)
            elif image_shape.ndims is None:
                is_batch = False
                image = array_ops.expand_dims(image, 0)
                image.set_shape([None] * 4)
            elif image_shape.ndims != 4:
                raise ValueError(
                    '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
                    image_shape)

            batch, height, width, depth = _ImageDimensions(image, rank=4)

            after_padding_width = target_width - offset_width - width

            after_padding_height = target_height - offset_height - height

            if check_dims:
                assert_ops = _CheckAtLeast3DImage(image, require_static=False)
                assert_ops += _assert(offset_height >= 0, ValueError,
                                      'offset_height must be >= 0')
                assert_ops += _assert(offset_width >= 0, ValueError,
                                      'offset_width must be >= 0')
                assert_ops += _assert(after_padding_width >= 0, ValueError,
                                      'width must be <= target - offset')
                assert_ops += _assert(after_padding_height >= 0, ValueError,
                                      'height must be <= target - offset')
                image = control_flow_ops.with_dependencies(assert_ops, image)

            # Do not pad on the depth dimensions.
            paddings = array_ops.reshape(
                array_ops.stack([
                    0, 0, offset_height, after_padding_height, offset_width,
                    after_padding_width, 0, 0
                ]), [4, 2])
            padded = array_ops.pad(image, paddings, constant_values=pad_value)

            padded_shape = [
                None if _is_tensor(i) else i
                for i in [batch, target_height, target_width, depth]
            ]
            padded.set_shape(padded_shape)

            if not is_batch:
                padded = array_ops.squeeze(padded, axis=[0])

            return padded

    # In this function we added the parameter pad_value
    def pad_to_bounding_box(image, offset_height, offset_width, target_height,
                            target_width, pad_value):
        """Pad `image` with zeros to the specified `height` and `width`.

        Adds `offset_height` rows of zeros on top, `offset_width` columns of
        zeros on the left, and then pads the image on the bottom and right
        with zeros until it has dimensions `target_height`, `target_width`.

        This op does nothing if `offset_*` is zero and the image already has size
        `target_height` by `target_width`.

        Usage Example:

        # >>> x = [[[1., 2., 3.],
        # ...       [4., 5., 6.]],
        # ...       [[7., 8., 9.],
        # ...       [10., 11., 12.]]]
        # >>> padded_image = tf.image.pad_to_bounding_box(x, 1, 1, 4, 4)
        # >>> padded_image
        <tf.Tensor: shape=(4, 4, 3), dtype=float32, numpy=
        array([[[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]],
        [[ 0.,  0.,  0.],
        [ 1.,  2.,  3.],
        [ 4.,  5.,  6.],
        [ 0.,  0.,  0.]],
        [[ 0.,  0.,  0.],
        [ 7.,  8.,  9.],
        [10., 11., 12.],
        [ 0.,  0.,  0.]],
        [[ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.],
        [ 0.,  0.,  0.]]], dtype=float32)>

        Args:
          image: 4-D Tensor of shape `[batch, height, width, channels]` or 3-D Tensor
            of shape `[height, width, channels]`.
          offset_height: Number of rows of zeros to add on top.
          offset_width: Number of columns of zeros to add on the left.
          target_height: Height of output image.
          target_width: Width of output image.

        Returns:
          If `image` was 4-D, a 4-D float Tensor of shape
          `[batch, target_height, target_width, channels]`
          If `image` was 3-D, a 3-D float Tensor of shape
          `[target_height, target_width, channels]`

        Raises:
          ValueError: If the shape of `image` is incompatible with the `offset_*` or
            `target_*` arguments, or either `offset_height` or `offset_width` is
            negative.
        """
        return pad_to_bounding_box_internal(
            image,
            offset_height,
            offset_width,
            target_height,
            target_width,
            pad_value,
            check_dims=True)

    def _resize_fn(im, new_size):
        return tf.image.resize(im, new_size, method, antialias=antialias)

    def _resize_image_with_pad_common(image, target_height, target_width,
                                      resize_fn, pad_value=0.):
        """Core functionality for v1 and v2 resize_image_with_pad functions."""
        with ops.name_scope(None, 'resize_image_with_pad', [image]):
            image = ops.convert_to_tensor(image, name='image')
            image_shape = image.get_shape()
            is_batch = True
            if image_shape.ndims == 3:
                is_batch = False
                image = array_ops.expand_dims(image, 0)
            elif image_shape.ndims is None:
                is_batch = False
                image = array_ops.expand_dims(image, 0)
                image.set_shape([None] * 4)
            elif image_shape.ndims != 4:
                raise ValueError(
                    '\'image\' (shape %s) must have either 3 or 4 dimensions.' %
                    image_shape)

            assert_ops = _CheckAtLeast3DImage(image, require_static=False)
            assert_ops += _assert(target_width > 0, ValueError,
                                  'target_width must be > 0.')
            assert_ops += _assert(target_height > 0, ValueError,
                                  'target_height must be > 0.')

            image = control_flow_ops.with_dependencies(assert_ops, image)

            def max_(x, y):
                if _is_tensor(x) or _is_tensor(y):
                    return math_ops.maximum(x, y)
                else:
                    return max(x, y)

            _, height, width, _ = _ImageDimensions(image, rank=4)

            # convert values to float, to ease divisions
            f_height = math_ops.cast(height, dtype=dtypes.float32)
            f_width = math_ops.cast(width, dtype=dtypes.float32)
            f_target_height = math_ops.cast(target_height, dtype=dtypes.float32)
            f_target_width = math_ops.cast(target_width, dtype=dtypes.float32)

            # Find the ratio by which the image must be adjusted
            # to fit within the target
            ratio = max_(f_width / f_target_width, f_height / f_target_height)
            resized_height_float = f_height / ratio
            resized_width_float = f_width / ratio
            resized_height = math_ops.cast(
                math_ops.floor(resized_height_float), dtype=dtypes.int32)
            resized_width = math_ops.cast(
                math_ops.floor(resized_width_float), dtype=dtypes.int32)

            padding_height = (f_target_height - resized_height_float) / 2
            padding_width = (f_target_width - resized_width_float) / 2
            f_padding_height = math_ops.floor(padding_height)
            f_padding_width = math_ops.floor(padding_width)
            p_height = max_(0, math_ops.cast(f_padding_height, dtype=dtypes.int32))
            p_width = max_(0, math_ops.cast(f_padding_width, dtype=dtypes.int32))

            # Resize first, then pad to meet requested dimensions
            resized = resize_fn(image, [resized_height, resized_width])

            padded = pad_to_bounding_box(resized, p_height, p_width, target_height,
                                         target_width, pad_value)

            if padded.get_shape().ndims is None:
                raise ValueError('padded contains no shape.')

            _ImageDimensions(padded, rank=4)

            if not is_batch:
                padded = array_ops.squeeze(padded, axis=[0])

            return padded

    return _resize_image_with_pad_common(image, target_height, target_width,
                                         _resize_fn, pad_value=pad_value)


def rescale_image_values(img):
    """
    Rescale the pixel values such that the mininmum, maximum pixel
    value equals 0, 1 respectively.

    :param img: tf.image
    :return: tf.image of same shape as original image
    """
    img_min, img_max = tf.reduce_min(img), tf.reduce_max(img)
    numerator = tf.subtract(img, img_min)

    # The denominator cannot be smaller than (an arbitrary choice) 0.001 in order
    # to prevent zero-divison. In practice, this means that for images where the difference
    # between the original minimum and maximum value is small, the maximum pixel value after
    # rescaling remains < 1
    denom = tf.reduce_max(
        tf.stack(
            [tf.subtract(img_max, img_min), tf.constant(0.001)]
        ))

    # Due to some floating point effect, the result is sometimes slightly larger than 1. (e.g. 1.000001). Therefore we
    # can set a maximum value, but we decide not to do it right now. The line below does not work because the shapes do
    # not match
    # img_rescaled = tf.reduce_min(tf.stack([tf.divide(numerator, denom), tf.constant(1.)]))
    return tf.divide(numerator, denom)


def augm_image(image, rotate=True):
    if rotate:
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_flip_up_down(image)
    image = tf.image.random_brightness(image, 0.05)
    image = tf.image.random_contrast(image, 0.5, 2.0)
    return rescale_image_values(image)


def augm_image_ds(ds, all_rotations=False):
    ''''
    input: tf.dataset of (image, label)
    applies augm_image to all images in dataset and concatenates augmented images
    to original dataset
    return: tf.dataset of (image, label) of 2* the input length
    '''
    if all_rotations:
        # Add all 90degree-rotations of each image with randomly varying brightness/contrast
        ds_new = ds
        for num in [1, 2, 3]:
            ds_ext = ds.map(lambda x, y: (augm_image(x, rotate=False), y))
            ds_rot = ds_ext.map(lambda x, y: (tf.image.rot90(x, k=num), y))
            ds_new = ds_new.concatenate(ds_rot)
    else:
        # Add a duplicate for each image with randomly varying brightness/contrast/rotation
        ds_ext = ds.map(lambda x, y: (augm_image(x, rotate=True), y))
        ds_new = ds.concatenate(ds_ext)

    return ds_new

# tf.clip_by_value

# Not super elegant, but in order to access the multiplier of the applied augmentations, we put these values in
# a dictionary
AUGMENT_DICT = {'simple': 2, 'all_rotations': 4}


def to_grayscale(image):
    '''
    input: tf.Tensor - image as tensor with 3 channels
    returns: tf.Tensor - image with original shape with 3 identical color channels,
    which are result of conversion of input image to grayscale
    '''
    image = tf.image.rgb_to_grayscale(image)

    # In order to properly stack the image back to its original shape,
    # we need to extract the first 2 dimensions
    image = image[:, :, 0]
    return tf.stack([image, image, image], axis=2)


def import_images_from_df(df, image_size, batch_size, as_grayscale=False, adjust_contrast=True, pad_value=0.,
                          prefetch=False):
    """
    df has to contain a column named 'image_path'
    """
    num_channels = 3

    path_ds = dataset_ops.Dataset.from_tensor_slices(df['image_path'])
    img_ds = path_ds.map(lambda x: load_image(x, num_channels, image_size=image_size,
                                              as_grayscale=as_grayscale, adjust_contrast=adjust_contrast,
                                              pad_value=pad_value))
    img_ds = img_ds.batch(batch_size)

    if prefetch:
        img_ds = img_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return img_ds


def import_images_from_avi_list(avi_list, image_size, batch_size, as_grayscale=False, adjust_contrast=True, pad_value=0.,
                          prefetch=False):
    """
    df has to contain a column named 'image_path'
    """
    num_channels = 3

    path_ds = dataset_ops.Dataset.from_tensor_slices(avi_list)
    img_ds = path_ds.map(lambda x: load_image(x, num_channels, image_size=image_size,
                                              as_grayscale=as_grayscale, adjust_contrast=adjust_contrast,
                                              pad_value=pad_value))
    img_ds = img_ds.batch(batch_size)

    if prefetch:
        img_ds = img_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    return img_ds


def convert_img_to_tf(img, num_channels=3, **img_kwargs):
    img = tf.convert_to_tensor(img, dtype=tf.float32)
    img = tf.stack([img, img, img], axis=2)
    return prepare_tf_image(img, num_channels, **img_kwargs)

