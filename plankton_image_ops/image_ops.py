#!/usr/bin/env python3
import sys
from pathlib import Path

import imagesize
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from skimage import io, feature, filters, color
import zipfile

from plankton_image_ops.read_avi_img import *


def read_img(img_path, force_dtype='float', plugin=None, as_gray=False):
    """
    Reads image using io.imread() from scikit-image. We did several test on which plugin is fasted for CPICS and ISIIS
    images with timeit. See test_image_ops.ipynb.

    Results ISIIS (in seconds):
    tifffile:       0.0047778204916539835
    pil:            0.01494338819168964
    imageio:        0.0043137208333064335
    matplotlib:     0.008620551391656288
    cv-imread (not available as plugin, it's a different function):  0.03443533784165993

    Results CPICS (in seconds):
    pil:            0.0017746965333268843
    imageio:        0.0022458034750040194
    matplotlib:     0.0016366246583250663
    cv-imread:      0.0018974649333358684

    If img_path points to an avi-fullframe, it is assumed that the parent is the path to the actual avi (which contains
    many fullframe images, and the 'name' of the path indicates the unique fullframe and that this name ends (before the\
    file extension) with a 3-digit counter identifies the fullframe.

    :param img_path: str - path to image OR pseudo-path to the avi-fullframe
    :param force_dtype: str - should be False, 'float' or 'uint8'. If 'float', image is returned as float in range [0, 1], and if
                        'uint8', image is returned as np.uint8 in range [0, 255]. If False, image is returned as default dtype
                        of the plugin, which is float for 'matplotlib' and uint8 for 'imageio'.
    :param plugin: str - one of the available plugins used in skimage.io.imread OR 'CPICS' or 'ISIIS' to use the best
                         tested plugin per filetype.
    :param as_gray: bool - if True, image is converted from RGB to grayscale if applicable. No effect if input image is
                           already grayscale
    :return: np.array of image with shape (image height, image width) if grayscale and (image height, image width, 3) if RGB.
    """
    if force_dtype and (force_dtype not in ['uint8', 'float']):
        raise ValueError(f"force_dtype should be False, 'uint8' or 'float', while current value is {force_dtype}")

    img_path = Path(img_path)
    if img_path.suffix == '.avi':
        avi_path = img_path.parent
        ff_counter = int(img_path.stem[-3:])

        if force_dtype == 'float':
            as_float = True
        else:
            as_float = False

        img = read_ff_from_avi_file(avi_path, ff_counter, as_float=as_float)[0]
    else:
        # If not avi, it is assumed that it is a standard image format, such as .png or .tiff
        if plugin == 'CPICS':
            plugin = 'matplotlib'
        elif plugin == 'ISIIS':
            plugin = 'imageio'

        img = io.imread(img_path, plugin=plugin, as_gray=as_gray)

        if force_dtype == 'float':
            img = util.img_as_float(img)
        elif force_dtype == 'uint8':
            img = util.img_as_ubyte(img)

    return img


def read_img_from_zip_archive(zip_path, img_name, with_background=False, force_dtype=None):
    """

    :param zip_path:
    :param img_name:
    :param with_background:
    :param force_dtype: str - should be 'float' or 'uint8' or None
    :return:
    """
    if with_background:
        roi_folder = 'rois_bg'
    else:
        roi_folder = 'rois'

    with zipfile.ZipFile(zip_path, mode="r") as archive:
        if not len(archive.namelist()):
            print(f"Archive {zip_path} is empty")
        else:
            img_file = archive.open(f"{roi_folder}/{img_name}")
            img = io.imread(img_file, plugin='matplotlib')

    if force_dtype == 'float':
        img = util.img_as_float(img)
    elif force_dtype == 'uint8':
        img = util.img_as_ubyte(img)

    return img


def load_all_rois_from_zip(zip_path, with_background=False):
    if with_background:
        roi_folder = 'rois_bg'
    else:
        roi_folder = 'rois'

    img_list = []
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        namelist = archive.namelist()
        if not len(namelist):
            print(f"Archive {zip_path} is empty")
        for roi_path in namelist:
            if Path(roi_path).parent.name == roi_folder:
                img_file = archive.open(roi_path)
                img = io.imread(img_file, plugin='matplotlib')
                # img = util.img_as_float(img)
                img_list.append(img)

    return img_list


def is_image_corrupt(image_path):
    """
    Check if an image file can be opened. Sometimes corrupt images are created at end of
    sampling sessions. Returns True if images is corrupt and False if image is readable.

    :param image_path: str
    :return: boolean
    """
    try:
        read_img(image_path)
    except ValueError:
        print(f"image {image_path} could not be opened")
        return True
    else:
        return False


def find_edge(img_path, sigma=3., low_thres=None, high_thres=None, enhance_contrast=False, read_plugin='CPICS'):
    """
    Applies canny edge operator from sci-kit image, using feature.canny()

    :param img_path: str - path to image
    :param sigma: float- standard deviations of Gaussian denoising
    :param low_thres: float - lower threshold for stitching edges
    :param high_thres: float - upper threshold for stitching edges
    :param enhance_contrast: boolean - if True, contrast is enhanced using PIL's ImageOps.autocontrast
    :param read_plugin: str - plugin that is used in read_img. Defaults to 'CPICS' since this is what the sharp filter
                              is designed for.
    :return: np.array - binary image of edges
    """
    # The Canny algorithm only accepts grayscale images
    if enhance_contrast:
        img = Image.open(img_path)
        img = ImageOps.autocontrast(img)
        img = util.img_as_float(np.array(img))
        if len(img.shape) == 3:
            img = color.rgb2gray(img)
    else:
        img = read_img(img_path, plugin=read_plugin, as_gray=True)
        # try:
        #     img = read_img(img_path, plugin=read_plugin, as_gray=True)
        # except Exception as error:
        #     print(error)
        #     print(f"Could not read file {img_path}, so this one was skipped")

    return feature.canny(img, sigma=sigma, low_threshold=low_thres, high_threshold=high_thres)


def find_edge_ratio(img_path, denominator='roi_area', **kwargs):
    """
    Applies canny edge operator from sci-kit image and calculates the ratio between
    edge pixels and the total ROI. If denominator = 'otsu_area', the Otsu thresholding
    algorithm is used in order to find the area of the particle instead of using the full image.

    :param img_path: str - path to image
    :param kwargs: keyword arguments of find_edge
    :param denominator: str - either 'roi_area' or 'otsu_area'
    :return: float
    """
    img = find_edge(img_path, **kwargs)

    if denominator == 'otsu_area':
        img_orig = read_img(img_path, as_gray=True)
        global_thresh = filters.threshold_otsu(img_orig)
        img_binary = img_orig > global_thresh
        return np.sum(img) / np.sum(img_binary)
    elif denominator == 'roi_area':
        return np.sum(img) / len(img.flatten())
    else:
        raise Exception("value for denominator was not recognised")


def find_edge_bool(img_path, **kwargs):
    """
    Applies canny edge operator from sci-kit image, using feature.canny().

    :param img_path: str - path to image
    :param kwargs: keyword arguments of find_edge
    :return: boolean - True when output image from canny filter contains >= 1
    edge pixel, else False
    """
    return np.any(find_edge(img_path, **kwargs))


def calc_img_size_default(img_path):
    """
    Calculates image height, width and area in pixels

    :param img_path: str - path to image
    :return: tuple (image height, image width, height * width)
    """
    img = read_img(img_path)
    height, width = img.shape[:2]
    return height, width, height * width


def calc_img_size_with_pil(img_path, func):
    """
    Calculates image height, width and area in pixels

    :param img_path: str - path to image
    :return: tuple (image height, image width, height * width)
    """
    width, height = func(img_path).size
    return height, width, height * width


def calc_img_size_with_imagesize(img_path, func):
    """
    Calculates image height, width and area in pixels

    :param img_path: str - path to image
    :param func: function
    :return: tuple (image height, image width, height * width)
    """
    width, height = func(img_path)
    return height, width, height * width


def calc_img_size_for_avi(avi_path, ff_counter=1):
    """
    Calculate the image size and area for a single image from an avi-file. By default the first frame is taken.

    :param avi_path: str
    :param ff_counter: int
    :return: int, int, int
    """
    img = read_ff_from_avi_file(avi_path, ff_counter=ff_counter, as_float=False)[0]
    height, width = np.shape(img)[0], np.shape(img)[1]
    return height, width, height * width


def calc_img_size_for_df(df, tool='imagesize'):
    """
    Apply function calc_img_size to df and create columns for height, width and
    area

    :param df: DataFrame - needs to have column named 'image_path'
    :return: DataFrame - with columns 'image_height', 'image_width', 'image_area'
    """
    if tool == 'PIL':
        df['result'] = df['image_path'].map(lambda x: calc_img_size_with_pil(x, func=Image.open))
    elif tool == 'imagesize':
        df['result'] = df['image_path'].map(lambda x: calc_img_size_with_imagesize(x, func=imagesize.get))
    elif tool == 'default':
        df['result'] = df['image_path'].map(lambda x: calc_img_size_default(x))

    df['image_height'] = df['result'].apply(lambda x: x[0])
    df['image_width'] = df['result'].apply(lambda x: x[1])
    df['image_area'] = df['result'].apply(lambda x: x[2])
    df.drop(columns='result', inplace=True)

    return df


def calc_signal_to_noise(img):
    """
    Calculate signal-to-noise ratio based on Panaïotis et al. 2022

    'SNR can be used to determine the relative noise level in an image and was computed as:
    SNR = 20 log(S/N)
    where S is the signal, defined as the mean of the input data,
    N is the noise, computed as the standard deviation around that mean.'

    High SNR means low noise and vice versa. Within our scripts, SNR is meant to be used on
    flat-fielded and contrast enhanced ISIIS images.

    :param img: np.array - can be both as float in [0, 1] or as np.uint8 in [0, 255]
                           (this does not change the SNR)
    :return: float
    """
    mean = np.mean(img)
    std = np.std(img, ddof=1)

    return 20. * np.log(mean / std)


def save_img(img, path, extension='.png', verbose=True):
    """

    """
    if type(path) != str:
        path = str(path)

    if not extension.startswith('.'):
        extension = '.' + extension

    if (suffix := Path(path).suffix) in ['.tiff', '.png']:
        extension = suffix

    if not path.endswith(extension):
        path = path + extension

    img = util.img_as_ubyte(img)
    io.imsave(path, img)

    if verbose:
        print(f"image was saved to {path}")


if __name__ == '__main__':
    pass

    """
    Speed comparison of:
    - calc_img_size_with_pil
    - calc_img_size_with_imagesize
    - calc_img_size_default
    
    Takes, for num=10.000:
    - 42s for PIL
    - 0.4s for imagesize
    - 4s for default
    _speed_comparison_test(num=10000)
    """
