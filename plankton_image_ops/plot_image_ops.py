#!/usr/bin/env python3
import sys
from pathlib import Path

from matplotlib import pyplot as plt

from general_utils.pickle_functions import *


def plot_orig_image_with_bboxes(img, coord_list, figsize=(12, 12), color='yellow'):
    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')

    for coords in coord_list:
        plot_bbox_in_img(ax, coords[1], coords[3], coords[0], coords[2], color=color)

    fig.tight_layout()
    plt.show()


def quick_plot_of_single_img(img, title=None, figsize=None, grayscale=True, im_vmin=None, im_vmax=None):
    if figsize:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig, ax = plt.subplots()

    if grayscale:
        cmap = 'gray'
    else:
        cmap = None

    ax.imshow(img, cmap=cmap, vmin=im_vmin, vmax=im_vmax)

    if title:
        ax.set_title(title)

    ax.axis('off')
    fig.tight_layout()
    plt.show()


def plot_img_and_result(img, img_result, savefig_path=None, figtitle=None, title_left=None, title_right=None, im_vmin=None, im_vmax=None, **fig_kwargs):
    fig, (ax1, ax2) = plt.subplots(ncols=2, **fig_kwargs)
    ax1.imshow(img, cmap='gray', vmin=im_vmin, vmax=im_vmax)
    ax1.axis('off')
    if title_left:
        ax1.set_title(title_left)

    ax2.imshow(img_result, cmap='gray', vmin=im_vmin, vmax=im_vmax)
    ax2.axis('off')
    if title_right:
        ax2.set_title(title_right)

    if figtitle:
        fig.suptitle(figtitle)

    fig.tight_layout()

    if savefig_path:
        plt.savefig(savefig_path)
    else:
        plt.show()


def plot_img_with_bbox(img, y1, x1, y2, x2):
    """
    # Plot the full image with a bounding box at the specified coordinates

    :param img:
    :param y1:
    :param x1:
    :param y2:
    :param x2:
    :return:
    """
    fig, ax = plt.subplots()
    ax.imshow(img, cmap='gray')
    ax.hlines([y1, y2], x1, x2, color='yellow')
    ax.vlines([x1, x2], y1, y2, color='yellow')
    ax.set_axis_off()
    plt.show()


def plot_bbox_in_img(ax, x1, x2, y1, y2, **kwargs):
    """
    Plot a bounding box with corner coordinates x1, x2, y1, y2

    :param ax:
    :param x1:
    :param x2:
    :param y1:
    :param y2:
    :param color:
    :return:
    """
    ax.hlines([y1, y2], x1, x2, **kwargs)
    ax.vlines([x1, x2], y1, y2, **kwargs)
    return ax


def plot_full_image_with_bboxes(img, img_mask, coord_list, mean_background=0, figsize=(12, 12), color='yellow',
                                plot_area=False):
    """
    Plot an image with bounding boxes around the found ROIs. ROI bounding boxes are specified in coord_list. If img_mask
    is specified, the masked image is plotted on top of img, whereas if img_mask = None, the bounding boxes are plotted
    on img.

    :param img:
    :param img_mask:
    :param coord_list:
    :param mean_background:
    :param figsize:
    :param color:
    :param plot_area: bool - if True, plot the area of each bounding box next to the bounding box
    :return:
    """
    # For plotting, we apply this mask to whole original image
    if img_mask is not None:
        img_masked = np.zeros_like(img) + mean_background
        img_masked[img_mask] = img[img_mask]
    else:
        img_masked = img

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_masked, cmap='gray', vmin=0, vmax=255)

    for coords in coord_list:
        plot_bbox_in_img(ax, coords[1], coords[3], coords[0], coords[2], color=color)

        if plot_area:
            area = abs((coords[3] - coords[1]) * (coords[2] - coords[0]))
            ax.text(coords[1] + 5, coords[0] + 5, str(area), color=color)

    ax.set_xlim(0, img_masked.shape[1])
    ax.set_ylim(img_masked.shape[0], 0)
    ax.axis('off')

    fig.tight_layout()
    plt.show()


def plot_residual_image(img, img_mask, coord_list, mean_background=0, figsize=(12, 12)):
    # For plotting, we apply the negative of this mask to whole original image,
    # so we can see what is being missed.
    img_masked = np.zeros_like(img) + mean_background
    img_masked[~img_mask] = img[~img_mask]

    fig, ax = plt.subplots(figsize=figsize)
    ax.imshow(img_masked, cmap='gray', vmin=0, vmax=255)
    ax.axis('off')
    #
    # for coords in coord_list:
    #     plot_bbox_in_img(ax, coords[1], coords[3], coords[0], coords[2])

    fig.tight_layout()
    plt.show()
