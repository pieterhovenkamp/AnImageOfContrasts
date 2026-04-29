#!/usr/bin/env python3
import sys
from pathlib import Path

from skimage import exposure

from plankton_image_ops.image_ops import *


def take_stacked_median_per_pixel(df, num=25, sub_quadrant=False):
    """
    Take a subsample of df, then stack the images into a single array of shape (num, img_height, img_width)
    and take the median along the first axis.

    :param df: pd.DataFrame - needs to contain a column named 'image_path'
    :param sub_quadrant: boolean - if True, take a subsample of a quarter of each images to speed up testing
    :param num: int - number of images in random subsample of the total dataframe
    :return: np.array - array of shape (img_height, img_width) with per pixel the median of the pixel intensities
    """
    if num > len(df):
        raise Exception("Sample size cannot be larger than the length of the dataframe")

    df = df.sample(n=num, random_state=123).copy()

    img_list = []
    for img_path in df['image_path']:
        img = read_img(img_path, plugin='ISIIS', force_dtype=False)
        # img = util.img_as_ubyte(img)

        if sub_quadrant:
            img = img[int(img.shape[0] / 2):, int(img.shape[1] / 2):]

        img_list.append(img)

    # Slower
    # time1 = time.perf_counter()
    # img_array = np.empty((100, img.shape[0], img.shape[1]))
    # for i in range(num):
    #     img_array[i] = img_list[i]
    # print("time elapsed (B1):" , time.perf_counter() - time1)
    # print(img_array)

    # Stack the images into a single array of shape (num, img_height, img_width)
    img_array = np.stack(img_list, axis=0)

    # Take the median along the first axis
    median_array = np.median(img_array, axis=0, overwrite_input=True)

    return np.ceil(median_array).astype(np.uint8)


def find_background_with_stacked_median(df, blur_kernel_size=51, **kwargs):
    """

    An older version of this function used skimage.filters.median, but replacing this with OpenCV sped up this function
    from 460s for an ISIIS image to 0.3s.

    # footprint = morphology.square(blur_kernel_size)
    # median_array_blur = filters.median(median_array, footprint)

    :param df:
    :param kwargs:
    :return:
    """
    median_array = take_stacked_median_per_pixel(df, **kwargs)
    return cv2.medianBlur(median_array, blur_kernel_size)


def subtract_median_background_from_image(img, median_array):
    """

    :param img: np.array
    :param median_array: np.array
    :return: np.array
    """
    # Check if shapes are the same
    if img.shape != median_array.shape:
        raise ValueError("Shapes of image and median_array should match")

    if img.dtype != median_array.dtype:
        raise ValueError("Check that image and median_array should have the same dtype")

    # In a previous version, we used subtraction instead of division
    # return img - median_array
    img = img / median_array * np.mean(median_array)
    # return util.img_as_ubyte(img)
    return img


def subtract_median_background_from_image_path(image_path, median_array):
    """

    :param image_path:
    :param median_array:
    :return:
    """
    img = read_img(image_path, plugin='ISIIS', force_dtype=False)
    return subtract_median_background_from_image(img, median_array)


def stretch_contrast(img, p_low=2, p_up=98):
    """
    Contrast enhancement by clipping the percentiles to the min/max values of the input dtype. If
    the input image is float, then the output image is float in range [0, 1] and the image values
    span this whole interval.

    See https://scikit-image.org/docs/stable/auto_examples/color_exposure/plot_equalize.html#sphx-glr-auto-examples-color-exposure-plot-equalize-py

    :param img:
    :param p_up:
    :param p_low:
    :return:
    """
    p_low, p_up = np.percentile(img, (p_low, p_up))
    return exposure.rescale_intensity(img, in_range=(p_low, p_up))


def get_flatfield_background(cast_id, isiis_background_path, df=None, recalc=False, min_depth=0.5, sample_num=50):
    """
    Get the background flat-fielding images for ISIIS fullframes by either recalculating the background (which takes a
    few seconds) or by loaded it from a saved .npy-file. If a saved file does not exist or if recalc=True,
    the background image is calculated.

    :param cast_id: str
    :param df: DataFrame, optional - if provided, then the images will be loaded from this dataframe. If None, the
                                     dataframe will be loaded from the database at each function call (taking ~3s).
                                     Needs columns 'image_path', 'cast_id', 'depth'
    :param recalc: bool
    :param sample_num: int - number of images that are used in calculating the mean background
    :return: np.array - the stacked median array necessary to flat-field individual images
    """
    save_name = Path(isiis_background_path) / f"ISIIS_background_{cast_id}.npy"

    if recalc:
        # Select only images from this cast
        if df is None:
            raise ValueError(f"In order to (re)calculate the flatfield background for {cast_id}, a DataFrame should be "
                             "specified")
        else:
            df = df.loc[df['cast_id'] == cast_id]

        # Filter for images with a real depth value (in principle this should filter most of the deck images)
        df = df.loc[df['depth'] > min_depth].copy()
        
        if not len(df):
            raise ValueError("DataFrame is empty")

        median_array_blur = find_background_with_stacked_median(df, num=sample_num)
        np.save(save_name, median_array_blur)
        print(f"A flat-fielding background array for cast {cast_id} was saved to {save_name}")

    else:
        if os.path.isfile(save_name):
            median_array_blur = np.load(save_name)
        else:
            raise FileNotFoundError(f"No flatfield background was found at {save_name}")

    return median_array_blur


def apply_flatfielding(img, cast_id, isiis_background_path, stretch_contrast_bool=True, p_low=2., p_up=98.,
                       **flatfield_kwargs):
    """

    :param img:
    :param cast_id:
    :param isiis_background_path:
    :param p_low:
    :param p_up:
    :return: np.array - output image
    """
    try:
        median_array_blur = get_flatfield_background(cast_id, isiis_background_path, **flatfield_kwargs)
    except FileNotFoundError as error:
        raise error
    else:
        img = subtract_median_background_from_image(img, median_array_blur)
        if stretch_contrast_bool:
            img = stretch_contrast(img, p_low=p_low, p_up=p_up)
        else:
            img = exposure.rescale_intensity(img, in_range='image')
            img = exposure.adjust_log(img, 1)
        return img


def calc_signal_to_noise_for_img_path(img_path, cast_id, isiis_background_path, return_image=False):
    median_array_blur = get_flatfield_background(cast_id, isiis_background_path)
    img = subtract_median_background_from_image_path(img_path, median_array_blur)
    img = stretch_contrast(img, p_low=2, p_up=98)

    if return_image:
        return calc_signal_to_noise(img), img
    else:
        return calc_signal_to_noise(img)
