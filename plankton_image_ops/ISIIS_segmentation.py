#!/usr/bin/env python3
import warnings
from collections import Counter
import datetime
import glob
from pathlib import Path
import sys

# Import the letters from the alphabet so we can iterate to label plots
from string import ascii_lowercase as alc

from scipy import ndimage
from skimage import filters, measure, morphology, util

from plankton_image_ops.plot_image_ops import quick_plot_of_single_img, plot_img_and_result, plot_bbox_in_img
from plankton_image_ops.denoise_ISIIS import *


# 202060215
def run_segmentation(img, ff_name, dest_folder, img_to_segment=None, min_roi_area=800, max_roi_frac=0.5, min_bbox_area=1200,
                     roi_contour_pad=5,
                     d=20, sigma_color=50, sigma_space=50, save=True,
                     save_rois_with_background=False, verbose_saved_rois=False, plot_steps=False, plot_result=False):
    """
    For an ISIIS fullframe image, apply the segmentation and save the output ROIs to dest_folder.

    Recommend usage is to apply the flatfielding procedure before running this function (see
    denoise_ISIIS and apply_image_ops_to_db for combining the flatfielding and segmentation)

    :param img: np.array
    :param ff_name: str - name of fullframe image, which serves as prefix for the ROI names
    :param dest_folder: str - will be created if it does not exist yet. If exists, then the ROIs will be added.
    :param img_to_segment: np.array - image from which the detected ROIs are actually cut. The difference with parameter
                                      'img' is that 'img' is used to detect the foreground (ROI) pixels. The recommended
                                       usage is to apply contrast enhancement to 'img' and to cut the segments from the
                                       non-contrast-enhanced 'img_to_segment'. Default to img if img_to_segment = None
    :param min_roi_area: int
    :param max_roi_frac: float
    :param min_bbox_area: float - minimum pixel area of the selected ROIs
    :param roi_contour_pad: int
    :param d: int - see apply_bilateral_filter
    :param sigma_color: int - see apply_bilateral_filter
    :param sigma_space: int - see apply_bilateral_filter
    :param save_rois_with_background: bool - if True, also save a version of each ROI with the background in place,
                                             instead of a gray background outside the segments
    :param verbose_saved_rois: bool - if True, print notice of saved ROIs for each fullframe image. Silent if False.
    :return: list, list - list of filenames of the saved rois
                          list of coordinates of the ROI bounding boxes with each element a tuple (y1, x1, y2, x2)
    """
    # Get the ROIs
    img_mask, mean_background, img_steps = get_segmentation_mask(img, min_area=min_roi_area, d=d,
                                                      sigma_color=sigma_color, sigma_space=sigma_space,
                                                      plot_steps=plot_steps, plot_result=plot_result)
    if img_to_segment is None:
        img_to_segment = img

    if save_rois_with_background:
        roi_list, coord_list, roi_background_list = extract_rois_from_binary_mask(img_to_segment, img_mask, mean_background,
                                                                                  max_roi_frac=max_roi_frac,
                                                                                  min_bbox_area=min_bbox_area,
                                                                                  pad=roi_contour_pad,
                                                                                  get_rois_with_background=True)
        if save:
            roi_names = save_rois_double_from_fullframe_zip(ff_name, roi_list, roi_background_list,
                                                            dest_folder=dest_folder, verbose=verbose_saved_rois)
        else:
            roi_names = []
    else:
        roi_list, coord_list = extract_rois_from_binary_mask(img_to_segment, img_mask, mean_background,
                                                             max_roi_frac=max_roi_frac,
                                                             min_bbox_area=min_bbox_area,
                                                             pad=roi_contour_pad,
                                                             get_rois_with_background=False)

        if save:
            roi_names = save_rois_from_fullframe_zip(ff_name, roi_list,
                                                     dest_folder=dest_folder, verbose=verbose_saved_rois)
        else:
            roi_names = []

    return roi_names, coord_list, img_steps


# 20260215
def run_flatfielding_and_segmentation(img, cast_id, dest_folder, isiis_background_path, ff_name=None,
                                      snr_threshold=None, flatfield_min_depth=0.5, recalc_flatfield=False,
                                      plot_steps=False, plot_result=False, stretch_contrast_p_low=2,
                                      stretch_contrast_p_up=98, **segment_kwargs):
    """
    For an ISIIS fullframe image path, apply the segmentation and save the output ROIs to dest_folder.

    :param img_path: str
    :param cast_id: str
    :param dest_folder: str - will be created if it does not exist yet. If exists, then the ROIs will be added.
    :param ff_name: str - ff-name without extension. If None, ff-name (that is used in name of saved ROIs) of filepath
                          is deduced. Not recommended! Since for ISIIS, the ff-names are not unique among different
                          deployments
    :param snr_threshold: float - lower limit on allowed signal-to-noise ratio of a fullframe image. Recommend values
                                  are 15 - 20, because the segmentation often fails if the SNR is lower than that.
                                  If None, no SNR threshold is enforced.
    :param segment_kwargs: keyword arguments of function ISIIS_segmentation.run_segmentation (see docstrings)
                           of that function.
    :return: float, list, np.array - signal-to-noise ratio, list of filenames of rois,
                                     array of bbox coordinates (y1, x1, y2, x2) (see docs of run_segmentation),

    Parameters
    ----------
    isiis_background_path
    """
    # Import fullframe image - if a string is provided the image is imported, else it is assumed that the image
    # is given as input
    if type(img) == str:
        img_path = img
        img = read_img(img_path, plugin='ISIIS', force_dtype=False)

    # Apply flatfielding
    img_orig = img
    img_ff = apply_flatfielding(img_orig, cast_id, isiis_background_path, stretch_contrast_bool=False,
                                min_depth=flatfield_min_depth, recalc=recalc_flatfield)
    img = stretch_contrast(img_ff, p_low=stretch_contrast_p_low, p_up=stretch_contrast_p_up)
    img = util.img_as_ubyte(img)
    img_ff = util.img_as_ubyte(img_ff)

    # Calc SNR and check threshold
    snr = calc_signal_to_noise(img)

    if snr_threshold is not None:
        if snr < snr_threshold:
            if plot_result:
                fig, ax1 = plt.subplots()
                ax1.imshow(img, cmap='gray', vmin=0, vmax=255)
                ax1.axis('off')
                ax1.set_title(f"Original - SNR {snr} below threshold of {snr_threshold}")

            return snr, [], np.array([])
        else:
            if plot_result:
                print("SNR: ", snr)

    # Extract and save the ROIs
    if ff_name is None:
        ff_name = Path(img_path).stem

    roi_names, bbox_coord_list, img_steps = run_segmentation(img, ff_name, dest_folder,
                                                  img_to_segment=img_ff,
                                                  plot_steps=plot_steps,
                                                  plot_result=plot_result, **segment_kwargs)

    if plot_steps or plot_result:
        img_bf, binary, binary_op, img_mask_large, img_mask_dilated, contours_mask, mean_background, yen_threshold = img_steps

        # We apply the mask to the original image
        img_masked = np.zeros_like(img) + mean_background
        img_masked[contours_mask] = img_ff[contours_mask]

        print(f"Yen threshold value: {yen_threshold:.0f}")

        if plot_steps:
            axes_layout = [
            ['ax1', 'ax2', 'ax3'],
            ['ax4', 'ax5', 'ax6'],
            ['ax7', 'ax8', 'ax9'],
            ]

            ax_width, ax_height = 3, 3
            fig = plt.figure(figsize=(ax_width * 3, ax_height * 3), dpi=300)
            axes = fig.subplot_mosaic(
                axes_layout,
                gridspec_kw={'wspace': 0.05, 'hspace': 0.05},
            )
            for ax_name, plot_img, title in zip(axes.keys(),
                                                [img_orig, img_ff, img, img_bf, binary, binary_op, img_mask_large, img_mask_dilated, img_masked],
                                                ['Raw image', 'Flat-fielded', 'Contrast enhanced', 'Bilateral filter', f"Yen thresholding", 'Morphological opening', 'Remove small objects, fill holes', 'Dilation', 'Result']
                                                ):
                ax = axes[ax_name]

                if title == 'Result':
                    for coords in bbox_coord_list:
                        plot_bbox_in_img(ax, coords[1], coords[3], coords[0], coords[2], color='red', linewidths=0.5)

                ax.imshow(plot_img, cmap='gray', vmin=None, vmax=None)
                ax.set_title(title)
                ax.axis('off')

            for ax, title in zip(axes.values(), alc):
                ax.set_title(title, loc='left')
                
            plt.show()
        else:
            ax_width, ax_height = 3, 3
            fig = plt.figure(figsize=(ax_width * 3, ax_height * 3), dpi=300)
            axes = fig.subplot_mosaic(
                [
                ['ax1', 'ax2'],
                ],
                gridspec_kw={'wspace': 0.1, 'hspace': 0.1},
            )

            for ax_name, plot_img, title in zip(axes.keys(),
                                                [img_orig, img_masked],
                                                ['Raw image', 'Segmentation result']
                                                ):
                ax = axes[ax_name]

                if title == 'Segmentation result':
                    for coords in bbox_coord_list:
                        plot_bbox_in_img(ax, coords[1], coords[3], coords[0], coords[2], color='red', linewidths=0.5)

                ax.imshow(plot_img, cmap='gray', vmin=None, vmax=None)
                ax.set_title(title)
                ax.axis('off')

            # add_alphabet_plot_titles(axes.values())

            for ax, title in zip(axes.values(), alc):
                ax.set_title(title, loc='left')
            
            plt.show()

    return snr, roi_names, np.array(bbox_coord_list)


# 20260215
def run_flatfielding_and_segmentation_on_df(df, dest_folder, isiis_background_path, from_avis,
                                            flatfield_min_depth=0.5, recalc_flatfield=False, verbose_num=10000,
                                            stop_at_num_images=20000, save=True,
                                            **kwargs):
    """
    Run the integrated workflow for flatfielding ISIIS images followed by segmentation for all fullframes in a
    dataframe.

    The resulting ROI-data is stored in <dest_folder>/roi_data, containing a csv-file per fullframe image, and in
    <dest_folder>/rois containing a zip-archive per fullframe image. The csv-file contains the SNR and
    file counters of all ROIs ('<counter>.png') and ROI coordinates and is named <fullframe_image>_rois.csv. The zip-archive contains a folder 'rois' and,
    if applicable 'rois_bg'. These folders contain the ROI images as '<counter>.png' with and without background.

    If no ROIs were found for a fullframe image, then a csv-file is created with a single line, where the ROI columns
    are NaN, and no zip-archive is created.

    See run_flatfielding_and_segmentation for all keyword arguments.

    :param df: pd.DataFrame - needs to have columns 'image_path', 'cast_id', 'image_name'
    :param isiis_background_path: str
    :param dest_folder: str - will be created if it does not exist, else new data is appended to the directory
    :param kwargs: keyword arguments of run_flatfielding_and_segmentation
    :return: None

    """
    if not os.path.isdir(dest_folder) and save:
        os.mkdir(dest_folder)

    dest_folder_rois = f"{dest_folder}/rois"
    if not os.path.isdir(dest_folder_rois) and save:
        os.mkdir(dest_folder_rois)

    dest_folder_csv = f"{dest_folder}/roi_data"
    if not os.path.isdir(dest_folder_csv) and save:
        os.mkdir(dest_folder_csv)

    # We create folders per verbose_num within dest_folder_rois and dest_folder_csv
    t_now = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    dest_folder_csv_now = f"{dest_folder_csv}/{t_now}"
    if save:
        os.mkdir(dest_folder_csv_now)

    dest_folder_rois_now = f"{dest_folder_rois}/{t_now}"
    if save:
        os.mkdir(dest_folder_rois_now)
    # print(f"Stored output of {len(df_sub)} images in {output_file} - time elapsed: {dt.datetime.now() - t0}")

    if from_avis:
        # Loop over the avi files - we can obtain these from the image paths
        print("Running segmentation loop for avi-files")

        df['avi_path'] = df['image_path'].apply(lambda x: Path(x).parent)
        counter, t0 = 1, datetime.datetime.now()
        for avi_path, df_sub in df.groupby('avi_path'):
            # Per avi-file:
            #   read all images
            #   take subset of df for this avi and sort by counter
            #   loop over this dataframe together with the images

            # We select the ff_counters in df_sub for this avi (some might miss due to e.g. the depth filtering)
            ff_counters = list(df_sub['image_name'].apply(lambda x: int(Path(x).stem[-3:])))

            # Read all images for this avi
            img_list = read_ff_from_avi_file(avi_path, ff_counter=ff_counters)

            # Sort the dataframe by image name, so that the order is the same as for img_list
            df_sub.sort_values(by='image_name', inplace=True)

            # print(df_sub)
            for img, (_, row) in zip(img_list, df_sub.iterrows()):
                ff_name = Path(row['image_name']).stem
                # img = read_img(row['image_path'], force_dtype=None)

                try:
                    snr, roi_names, bbox_coord_list = run_flatfielding_and_segmentation(img, row['cast_id'],
                                                                                        dest_folder=dest_folder_rois_now,
                                                                                        isiis_background_path=isiis_background_path,
                                                                                        ff_name=ff_name,
                                                                                        flatfield_min_depth=flatfield_min_depth,
                                                                                        recalc_flatfield=recalc_flatfield,
                                                                                        save=save,
                                                                                        **kwargs)
                except ValueError as error:
                    print(error)
                    print(f"An error occurred while processing file {row['image_path']}, \nso this one was skipped")
                else:
                    if len(roi_names):
                        df_new_sub = pd.DataFrame({
                            'snr': snr,
                            'roi_name': roi_names,
                            'bbox_x1': bbox_coord_list[:, 1],
                            'bbox_x2': bbox_coord_list[:, 3],
                            'bbox_y1': bbox_coord_list[:, 0],
                            'bbox_y2': bbox_coord_list[:, 2]})

                    else:
                        df_new_sub = pd.DataFrame({
                            'snr': snr,
                            'roi_name': None,
                            'bbox_x1': None,
                            'bbox_x2': None,
                            'bbox_y1': None,
                            'bbox_y2': None}, index=[0])

                    if save:
                        # Create separate folder with csv-files per fullframe of the above DataFrame
                        df_new_sub.to_csv(f"{dest_folder_csv_now}/{ff_name}_rois.csv", index=False)

                    if counter % verbose_num == 0:
                        print(f"{datetime.datetime.now()}: {counter} fullframe images processed! (in this parallel"
                              f"execution - t_elapsed: {datetime.datetime.now() - t0})")
                        t0 = datetime.datetime.now()

                        if counter != stop_at_num_images:
                            # We update the folders per verbose_num within dest_folder_rois and dest_folder_csv
                            t_now = t0.strftime('%Y%m%d_%H%M%S')
                            dest_folder_csv_now = f"{dest_folder_csv}/{t_now}"
                            if save:
                                os.mkdir(dest_folder_csv_now)

                            dest_folder_rois_now = f"{dest_folder_rois}/{t_now}"
                            if save:
                                os.mkdir(dest_folder_rois_now)

                    if counter == stop_at_num_images:
                        print(f"We stop the function at {stop_at_num_images} images so we can restart")
                        return

                    counter += 1

    else:
        # Else we assume that each images is stored as a separate file
        print("Running segmentation loop for separate image files")

        counter, t0 = 1, datetime.datetime.now()
        for i, row in df.iterrows():
            ff_name = Path(row['image_name']).stem
            try:
                snr, roi_names, bbox_coord_list = run_flatfielding_and_segmentation(row['image_path'], row['cast_id'],
                                                                                    dest_folder=dest_folder_rois_now,
                                                                                    isiis_background_path=isiis_background_path,
                                                                                    ff_name=ff_name,
                                                                                    flatfield_min_depth=flatfield_min_depth,
                                                                                    recalc_flatfield=recalc_flatfield,
                                                                                    save=save,
                                                                                    **kwargs)
            except ValueError as error:
                print(error)
                print(f"An error occurred while processing file {row['image_path']}, \nso this one was skipped")
            else:
                if len(roi_names):
                    df_new_sub = pd.DataFrame({
                        'snr': snr,
                        'roi_name': roi_names,
                        'bbox_x1': bbox_coord_list[:, 1],
                        'bbox_x2': bbox_coord_list[:, 3],
                        'bbox_y1': bbox_coord_list[:, 0],
                        'bbox_y2': bbox_coord_list[:, 2]})

                else:
                    df_new_sub = pd.DataFrame({
                        'snr': snr,
                        'roi_name': None,
                        'bbox_x1': None,
                        'bbox_x2': None,
                        'bbox_y1': None,
                        'bbox_y2': None}, index=[0])

                # Create separate folder with csv-files per fullframe of the above DataFrame
                if save:
                    df_new_sub.to_csv(f"{dest_folder_csv_now}/{ff_name}_rois.csv", index=False)

                if counter % verbose_num == 0:
                    print(f"{datetime.datetime.now()}: {counter} fullframe images processed! (in this parallel"
                          f"execution - t_elapsed: {datetime.datetime.now() - t0})")
                    t0 = datetime.datetime.now()

                    if counter == stop_at_num_images:
                        # We update the folders per verbose_num within dest_folder_rois and dest_folder_csv
                        t_now = t0.strftime('%Y%m%d_%H%M%S')
                        dest_folder_csv_now = f"{dest_folder_csv}/{t_now}"
                        if save:
                            os.mkdir(dest_folder_csv_now)

                        dest_folder_rois_now = f"{dest_folder_rois}/{t_now}"
                        if save:
                            os.mkdir(dest_folder_rois_now)

                if counter == stop_at_num_images:
                    print(f"We stop the function at {stop_at_num_images} images so we can restart")
                    return

                counter += 1


def apply_bilateral_filter(img, d, sigma_color, sigma_space):
    """
    see https://machinelearningknowledge.ai/bilateral-filtering-in-python-opencv-with-cv2-bilateralfilter/

    :param img: np.array - input image as np.uint8
    :param d: int - Diameter of each pixel neighborhood that is used during filtering. If it is non-positive, it is
                    computed from sigmaSpace.
    :param sigma_color: float - Filter sigma in the color space. A larger value of the parameter means that farther
                                colors within the pixel neighborhood will be mixed together, resulting in larger areas
                                of semi-equal color.
    :param sigma_space: float - Filter sigma in the coordinate space. A larger value of the parameter means that farther
                                pixels will influence each other as long as their colors are close enough. When d>0, it
                                specifies the neighborhood size regardless of sigmaSpace. Otherwise, d is proportional
                                to sigmaSpace.
    :return: np.array - resulting image as np.uint8
    """
    return cv2.bilateralFilter(img, d=d, sigmaColor=sigma_color, sigmaSpace=sigma_space)


# Modified 20260114
def apply_yen_thresholding(img):
    """
    Make a binary mask via yen thresholding. In the resulting image background and foreground
    pixels are marked as False and True, respectively.

    :param img: np.array
    :return: np.array - resulting binary image as np.uint8 with values {True, False}
    """
    # In the yen_thresholding, zero-division sometimes occur in an array of the cumulative histogram. For the result
    # this does not matter, since the maximum value of the resulting array is selected anyway and we tested that for
    # images that printed this warning, the result of the segmentation was still fine. Therefore, we want to ignore
    # the warning that is printed each time this occurs.
    with np.errstate(divide='ignore'):
        threshold = filters.threshold_yen(img)

    return img < threshold, threshold


def remove_small_areas_from_binary(img_binary, min_area):
    """
    Remove the regions from a binary mask that have an area smaller than min_area (in pixels).
    In the resulting image background pixels are marked as False and foreground pixels are marked as True. Note that
    for an image that is either completely background or completely foreground, an image with only background pixels is
    returned.

    Note that findContours uses 8-connectivity.

    :param img_binary: np.array - a binary image. 0-valued pixels (or False, equivalently) are assumed to be
                                  the background
    :param min_area: int - in pixels, all regions that are smaller or equal to min_area are marked as background pixels
    :return: np.array - resulting binary image as boolean with values {False, True}
             np.array, optional - image of integers that identify the foreground regions as created by
                                  skimage.measure.label. Contains all the regions in the input image, including
                                  the regions that are smaller than min_area.
    """
    # Check if image indeed is binary
    img_binary = img_binary.astype(np.uint8)

    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    img_mask_large = np.zeros_like(img_binary)
    if hierarchy is not None:
        contours_large = [contour for contour, hier in zip(contours, hierarchy[0]) if
                          (hier[-1] == -1) & (cv2.contourArea(contour) > min_area)]
        cv2.drawContours(img_mask_large, contours_large, -1, color=(255, 255, 255), thickness=-1)
    img_mask_large = img_mask_large.astype(bool)

    return img_mask_large


def fill_holes_in_binary_image(img_binary, return_num=False):
    """
    Draw contours in a binary image and remove all the child contours, then fill all parent contours
    and return as a new binary image. For cv2.findContours, the image needs to be integer-valued.
    See https://pythonexamples.org/python-opencv-cv2-find-contours-in-image/

    :param img_binary: np.array - a binary image as np.uint8
    :param return_num: bool - if True, return number of remaining contours
    :return: np.array - resulting binary image as boolean with values {False, True}
             int, optional - number of remaining distinct regions in image
    """
    img_binary = img_binary.astype(np.uint8)
    # Check if image indeed is binary

    # We draw contours and remove all the child contours (hierarchy -1 means parent contour)
    contours, hierarchy = cv2.findContours(img_binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # We fill the contours and make a mask of regions inside the contours
    img_binary_new = np.zeros_like(img_binary)
    if hierarchy is not None:
        contours_parent = [contour for contour, hier in zip(contours, hierarchy[0]) if hier[-1] == -1]
        cv2.drawContours(img_binary_new, contours_parent, -1, color=(255, 255, 255), thickness=-1)
    else:
        contours_parent = []
    img_binary_new = img_binary_new.astype(bool)

    if return_num:
        return img_binary_new, len(contours_parent)
    else:
        return img_binary_new


def find_and_mark_contours(img_binary):
    """
    Draft. Mark contours as single lines on a binary image.

    :param img_binary:
    :return:
    """
    # We mark the contour edges on the original image
    contours, hierarchy = cv2.findContours(img_binary.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)
    contours_parent = [contour for contour, hier in zip(contours, hierarchy[0]) if hier[-1] == -1]

    img_contours = np.zeros_like(img)
    for contour in contours_parent:
        img_contours[contour[:, :, 1].astype(int), contour[:, :, 0].astype(int)] = True

    return img_contours


# 20260215
def get_segmentation_mask(img, min_area=100, d=20, sigma_color=50, sigma_space=50, plot_steps=False, plot_result=False):
    """

    :param img: np.array - image as integer in [0, 255]
    :param min_area: int - minimum selected area in square pixels
    :param d: int - parameter of bilater filter, see apply_bilateral_filter
    :param sigma_color: int - parameter of bilater filter, see apply_bilateral_filter
    :param sigma_space: int - parameter of bilater filter, see apply_bilateral_filter
    :param plot_steps:
    :return: np.array - resulting binary image as boolean with values {False, True}
             int -
    """
    img_bf = apply_bilateral_filter(img, d, sigma_color, sigma_space)

    binary, threshold = apply_yen_thresholding(img_bf)
    binary_op = cv2.morphologyEx(binary.astype(np.uint8),
                                 cv2.MORPH_OPEN,
                                 kernel=ndimage.generate_binary_structure(2, 1).astype(np.uint8))
    img_mask_large = remove_small_areas_from_binary(binary_op, min_area=min_area)
    img_mask_dilated = cv2.dilate(img_mask_large.astype(np.uint8), kernel=morphology.disk(radius=2),
                                  iterations=1)
    contours_mask = fill_holes_in_binary_image(img_mask_dilated, return_num=False)

    # We calculate the mean of the background value of the image
    mean_background = int(np.mean(img[~binary_op.astype(bool)]))

    if plot_steps or plot_result:
        return contours_mask, mean_background, [img_bf, binary, binary_op, img_mask_large, img_mask_dilated, contours_mask, mean_background, threshold]
    else:
        return contours_mask, mean_background, None


def get_padded_bbox_corners(bbox, img_max_x, img_max_y, pad=0):
    """
    # Get the bounding box corners, taking into account padding and the edges of the full images

    :param bbox: tuple - bounding box as returned by skimage.measure.regionsprops
    :param img_max_x: int - maximal dimension of total image in x-direction
    :param img_max_y: int - maximal dimension of total image in y-direction
    :param pad: int - add extra pixels at each side (in pixels)
    :return: (int, int, int, int) - dimensions of new bounding box as (y_min, x_min, y_max, x_max)
    """
    x1, x2 = max(bbox[1] - pad, 0), min(bbox[3] + pad, img_max_x)
    y1, y2 = max(bbox[0] - pad, 0), min(bbox[2] + pad, img_max_y)
    return y1, x1, y2, x2


def extract_rois_from_binary_mask(img, img_mask, mean_background, max_roi_frac=1., min_bbox_area=1200, pad=5,
                                  get_rois_with_background=False):
    """

    If an empty img is entered (background pixels only), an empty roi_list is returned.

    For an explanation of the stats returned by cv2.connectedComponentsWithStats, see:
    https://docs.opencv.org/3.4/d3/dc0/group__imgproc__shape.html#gac7099124c0390051c6970a987e7dc5c5

    :param img: np.array - image as integer in [0, 255]
    :param img_mask: np.array - binary image as boolean with values {False, True}
    :param mean_background: int - in [0, 255]
    :param max_roi_frac: float - in (0, 1]. Size fraction of total fullframe image that an ROI is allowed
                                 to be. If 1., all ROI sizes are allowed. Size is measured in pixel area of the
                                 mask, not of the total bounding box
    :param min_bbox_area: float - minimum pixel area of the selected ROIs
    :param pad: int - in pixels
    :param get_rois_with_background: bool - if True, also save a version of each ROI with the background in place,
                                             instead of a gray background outside the segments
    :return: list, list - list elements the ROIs with each element an array of np.uint8,
                          list of coordinates of the ROI bounding boxes with each element a tuple (y1, x1, y2, x2)
    """
    if max_roi_frac <= 0. or max_roi_frac > 1.:
        raise ValueError("max_roi_frac should be in interval (0, 1]")

    # We label all connected regions in order to iterate over separate regions (i.e. ROIs)
    num_contours, img_mask_labelled, stats, _ = cv2.connectedComponentsWithStats(img_mask.astype(np.uint8),
                                                                                 connectivity=8)

    img_max_x, img_max_y = img.shape[1], img.shape[0]
    roi_list, coord_list, roi_background_list = [], [], []

    # Loop through each connected region
    for i in range(1, num_contours):
        if (stats[i, cv2.CC_STAT_AREA] <= max_roi_frac * img_max_x * img_max_y) and \
                (stats[i, cv2.CC_STAT_AREA] > min_bbox_area):
            # We get the coordinates of the bounding boxes as output from cv2.connectedComponentsWithStats
            x1, y1 = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
            x2, y2 = x1 + stats[i, cv2.CC_STAT_WIDTH], y1 + stats[i, cv2.CC_STAT_HEIGHT]
            y1, x1, y2, x2 = get_padded_bbox_corners((y1, x1, y2, x2), img_max_x, img_max_y, pad=pad)
            # print((x2 - x1) * (y2 - y1))

            # First we select the bounding box within the full image in order to speed up all further calculations
            img_roi = img[y1:y2, x1:x2]

            # Within each bounding box, we make a mask of only the current region...
            img_roi_mask = img_mask_labelled[y1:y2, x1:x2] == i

            # ...and apply this mask to the image, and set the background to the mean of the background value
            img_roi_masked = np.zeros_like(img_roi) + mean_background
            img_roi_masked[img_roi_mask] = img_roi[img_roi_mask]

            if get_rois_with_background:
                roi_background_list.append(img_roi)

            roi_list.append(img_roi_masked)
            coord_list.append((y1, x1, y2, x2))

    if get_rois_with_background:
        return roi_list, coord_list, roi_background_list
    else:
        return roi_list, coord_list


def save_rois_from_fullframe_zip(ff_name, roi_list, dest_folder, verbose=True):
    """
    Save the rois and the rois with background to a single zip-file for each fullframe image, provided that
    ROIs were found for this image. Otherwise, no zip-file will be created.

    Zip-file contains subdirs named 'rois' and 'rois_bg'.

    :param ff_name:
    :param roi_list:
    :param dest_folder:
    :param verbose: bool - if True, print notice of saved ROIs for each fullframe image. Silent if False.
    :return:
    """
    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    roi_names = []
    # archive_name = f"{dest_folder}/{ff_name.replace('.tiff', '')}.zip"
    archive_name = f"{dest_folder}/{ff_name}.zip"
    if len(roi_list):
        with zipfile.ZipFile(archive_name, mode="a") as archive:
            for i, roi, roi_bg in zip(range(1, len(roi_list) + 1), roi_list):
                if i < 10:
                    roi_counter = f"000{i}"
                elif i < 100:
                    roi_counter = f"00{i}"
                elif i < 1000:
                    roi_counter = f"0{i}"
                else:
                    roi_counter = f"{i}"

                roi_name = f"{roi_counter}.png"
                roi_names.append(roi_name)

                _, buf = cv2.imencode('.png', roi)

                # By default, files within archive are not compressed
                archive.writestr(f"rois/{roi_name}", buf)

        if verbose:
            print(f"{len(roi_list)} ROIs were copied to {archive_name}")
    else:
        print(f"No ROIs were found to for {ff_name}, so no archive was created")

    return roi_names

def save_rois_double_from_fullframe_zip(ff_name, roi_list, roi_background_list, dest_folder, verbose=True):
    """
    Save the rois and the rois with background to a single zip-file for each fullframe image. Zip-file contains subdirs
    named 'rois' and 'rois_bg'. If roi_list is empty, no archive is created.

    :param ff_name:
    :param roi_list:
    :param roi_background_list:
    :param dest_folder:
    :param verbose: bool - if True, print notice of saved ROIs for each fullframe image. Silent if False.
    :return:
    """
    if len(roi_list) != len(roi_background_list):
        raise ValueError("Length of roi_list and roi_background_list should be the same")

    if not os.path.isdir(dest_folder):
        os.mkdir(dest_folder)

    roi_names = []
    archive_name = f"{dest_folder}/{ff_name}.zip"
    if len(roi_list):
        with zipfile.ZipFile(archive_name, mode="a") as archive:
            for i, roi, roi_bg in zip(range(1, len(roi_list) + 1), roi_list, roi_background_list):
                if i < 10:
                    roi_counter = f"000{i}"
                elif i < 100:
                    roi_counter = f"00{i}"
                elif i < 1000:
                    roi_counter = f"0{i}"
                else:
                    roi_counter = f"{i}"

                roi_name = f"{roi_counter}.png"
                roi_names.append(roi_name)

                _, buf = cv2.imencode('.png', roi)
                _, buf_bg = cv2.imencode('.png', roi_bg)

                # By default, files within archive are not compressed
                archive.writestr(f"rois/{roi_name}", buf)
                archive.writestr(f"rois_bg/{roi_name}", buf_bg)

    if verbose:
        print(f"{len(roi_list)} ROIs were copied with and without background to {archive_name}")

    return roi_names


                
# Updated 20250710
def parse_roi_data_from_csv_files(segment_folder):
    """
    Parse all the csv-files with ROI data from segment folder and make a single DataFrame of these. Note that
    if for a fullframe image no ROIs were found, this image is present in the DataFrame with a single row where
    the columns 'roi_path', 'bbox_x1', 'bbox_x2', 'bbox_y1', 'bbox_y2', 'zip_path' are NaN.

    :param segment_folder: str
    :return: pd.DataFrame - with columns:
                             - 'ff_name' (without extension)
                             - 'snr'
                             - 'roi_name'
                             - 'bbox_x1'
                             - 'bbox_x2'
                             - 'bbox_y1'
                             - 'bbox_y2'
                             - 'zip_path'
    """
    # Per run, parse all csv-files
    df_files = parse_folder_to_df(segment_folder, extension='.csv')

    # Check for duplicate fullframe images
    if (df_duplicate := df_files.value_counts('filename') > 1).sum():
        df_value_counts = df_files.value_counts('filename')
        print(df_value_counts.loc[df_value_counts > 1])
        raise ValueError("Multiple csv-files for the same fullframe image were found. Check this before continuing.")

    # Iterate over the csv-files
    df_list = []
    counter = 1
    t1 = datetime.datetime.now()
    for csv_path in df_files['filepath']:
        # Import csv-file as df
        try:
            df_sub = pd.read_csv(csv_path)
        except Exception as e:
            print(e)
            print("Error while reading csv file, so skipping this one:", csv_path)
            continue

        # We infer the ff_name from the filepath of the csv
        df_sub['ff_name'] = Path(csv_path).name.replace('_rois.csv', '')

        if np.all(df_sub['roi_name'].notnull()):
            # If ROIs were detected, we get the zip-path by replacing '_rois.csv' with '.zip'
            df_sub['zip_path'] = csv_path.replace('_rois.csv', '.zip')
            df_sub['zip_path'] = df_sub['zip_path'].str.replace('/roi_data/', '/rois/')
        else:
            # If no ROIs were detected, a zip-path does not exist for this fullframe
            df_sub['zip_path'] = np.nan

        df_sub = np.round(add_bbox_size_to_df(df_sub), 0)
        df_list.append(df_sub)

        if counter % 20000 == 0:
            print(f"ROIS from {counter} fullframes were parsed to DataFrame")
            print(f"t_elapsed last 20k images is: ", datetime.datetime.now() - t1)
            t1 = datetime.datetime.now()
        counter += 1

    # Concatenate all df's
    t1 = datetime.datetime.now()
    print("Finished parsing the images to DataFrame, now concatenating all..")
    df_all = pd.concat(df_list, ignore_index=True)
    print(f"Finished concatenating in ", datetime.datetime.now() - t1, " seconds")

    return df_all


def check_number_of_rois_versus_csv_files(df_csv, segment_folder):
    """
    For segment folder, run a number of checks to ensure that the data from the ROIs in the csv-files and in the
    zip-archives match. Specifically, that:
        - for each zip-archive a csv-file exists
        - for each csv-file a zip-archive exists
        - for each zip-archive, the number of rois in 'rois' and 'rois_bg' (if exists) match
        - for each zip-archive and csv-file, the number of rois in 'rois' and in the csv-file match

    Reasons one of the above could go wrong is for example if the segmentation procedure was aborted halfway.
    Note that df_csv is modified in place to speed up the processing of large DataFrames.

    :param df_csv: pd.DataFrame - needs to have columns 'roi_name', 'ff_name'
    :param segment_folder: str - folder is accessed during this function execution
    :return: pd.DataFrame - with columns 'issue' and 'filepath' with, respectively:
                                        - "no_csv" - zip-archive for which no csv-file was found
                                        - "no_zip" - csv-file for which no zip-archive was found
                                        - "not_matching_bg" - zip-archive for which rois and rois_bg do not match
                                        - "not_matching_csv" - zip-archive for which ROIs in archive and csv-file do not match
    """
    # We drop the csv-files for ff-images for which no ROIs were saved, since for these no zip-archive was created
    df_csv.dropna(subset=['roi_name'], inplace=True)
    # print(df_csv)
    # print(len(np.unique(df_csv['ff_name'])))

    # We parse the zip-archives
    filepaths_zip = glob.glob(f"{segment_folder}/**/*.zip", recursive=True)
    df_zip = pd.DataFrame({'zip_path': filepaths_zip,
                           'ff_name': list(map(lambda x: Path(x).stem, filepaths_zip))})
    df_merge = pd.merge(df_zip, df_csv, on='ff_name', how='outer')

    # Check that for every zip-file, there is a csv-file
    df_no_csv = df_merge.loc[df_merge['roi_name'].isnull()]
    if len(df_no_csv):
        print(f"For the following {len(df_no_csv)} files a zip-file exists but no csv-file:\n",
              df_no_csv['zip_path'])
        no_csv = df_no_csv['zip_path']
    else:
        no_csv = []

    # Check that for every csv-file, there is a zip-file
    df_csv_unique = df_merge.loc[df_merge['zip_path'].isnull()].copy()
    if len(df_csv_unique):
        df_csv_unique['csv_path'] = df_csv_unique.apply(
            lambda df: f"{df['campaign_dir']}/{df['sub_dir']}/roi_data/{df['ff_name']}_rois.csv",
            axis=1)
        csv_unique = np.unique(df_csv_unique['csv_path'])
        print(f"For the following {len(csv_unique)} files a csv-file exists but no zip-file:")
        print('\n', csv_unique)
    else:
        csv_unique = []

    # Look inside the zip-archives to check the number of ROIs there
    t1 = datetime.datetime.now()
    df_merge.dropna(subset=['zip_path', 'roi_name'], how='any', inplace=True)
    df_csv_size = df_merge.groupby('ff_name').size()
    bg_list, csv_list, zip_empty_list = [], [], []
    for i, row in df_merge.drop_duplicates(subset='ff_name', keep='first').iterrows():
        zip_path, ff_name = row['zip_path'], row['ff_name']
        counts = count_rois_in_zip_archive(zip_path)

        # Check that the zip-files are non-empty
        if not counts['rois']:
            print(f"\n The following zip-file is empty: \n{zip_path}")
            zip_empty_list.append(zip_path)

        # Check that the roi-count of 'rois' and 'rois_bg' (if present) match
        if ('rois_bg' in counts.keys()) and (counts['rois'] != counts['rois_bg']):
            print(f"\nFor the following zip-file, the number of rois and rois_bg does not match: \n{zip_path}\n"
                  f"files in rois is {counts['rois']}, whereas in rois_bg this is {counts['rois_bg']}")
            bg_list.append(zip_path)

        # Check that the roi-count of the zip-files and the csv-files match
        if counts['rois'] != df_csv_size[ff_name]:
            print(f"\nFor the following zip-file, the number of files in rois and in csv-file does not match:"
                  f"\n{zip_path}\n"
                  f"files in rois is {counts['rois']}, whereas in csv-file this is {df_csv_size[ff_name]}")
            csv_list.append(zip_path)

    print(f"t_elapsed for {len(filepaths_zip)} fullframe files: ", datetime.datetime.now() - t1)

    df_issues = pd.concat([pd.DataFrame({'filepath': no_csv, 'issue': 'no_csv'}),
                           pd.DataFrame({'filepath': csv_unique, 'issue': 'no_zip'}),
                           pd.DataFrame({'filepath': zip_empty_list, 'issue': 'zip_empty'}),
                           pd.DataFrame({'filepath': bg_list, 'issue': 'not_matching_bg'}),
                           pd.DataFrame({'filepath': csv_list, 'issue': 'not_matching_csv'})])
    return df_issues


def add_bbox_size_to_df(df):
    df['bbox_diag'] = np.sqrt((df['bbox_x2'] - df['bbox_x1']) ** 2. + (df['bbox_y2'] - df['bbox_y1']) ** 2.)
    return df


def parse_processed_fullframes_from_folder(segment_folder, ff_extension):
    """
    Parses the csv-files that contain the ROI-data in a folder, and extracts the name of the corresponding
    fullframe images from this (in the segmentation, a csv-file is made for each fullframe image).

    :param segment_folder: str
    :param ff_extension: str - desired extension of the fullframes in the returned list, in form '.avi' or '.tiff'
    :return: list - If no existing files are found, an empty list is returned.
    """
    filepaths = glob.glob(f"{segment_folder}/**/*.csv", recursive=True)
    print(f"{len(filepaths)} csv-files were found in {segment_folder}")
    return list(map(lambda x: Path(x).name.replace("_rois.csv", ff_extension), filepaths))


def count_rois_in_zip_archive(zip_path):
    """
    For a zip-archive that contains two folders, count the number of .png-files per folder.

    :param zip_path: str
    :return: Counter - counts (as int) are accessible using counts[<sub_dir>]
    """
    with zipfile.ZipFile(zip_path, mode="r") as archive:
        parent_dirs = [Path(roi_path).parent.name for roi_path in archive.namelist()
                       if Path(roi_path).suffix == '.png']
        counts = Counter(parent_dirs)
    return counts
