#!/usr/bin/env python3

import numpy as np
from matplotlib import pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.lines as mlines
import matplotlib.patches as mpatches

from plankton_cnn.pvnp_save_and_load_utils import *


def show_subsample(batched_ds):
    """

    :param batched_ds:
    :return:
    """
    img_list = []
    for images, _ in batched_ds.take(1):
        for i in range(9):
            img_list.append(images[i])

    fig = plt.figure(figsize=(10, 10), tight_layout=True)
    grid = ImageGrid(fig, 111, nrows_ncols=(3, 3), axes_pad=0.3, share_all=True)

    for ax, im in zip(grid, img_list):
        # ax.imshow(im.numpy().astype("uint8"))
        ax.imshow(im.numpy())
        ax.set_axis_off()
    plt.show()


def plot_learning_rate():

    def decayed_learning_rate(step, initial_learning_rate, decay_rate, decay_steps):
        return initial_learning_rate * decay_rate ** (step / decay_steps)

    steps = np.arange(0, 50)
    lr = [decayed_learning_rate(step, 0.001, 0.5, 50) for step in steps]

    fig, ax = plt.subplots()
    ax.scatter(steps, lr)
    plt.show()


def plot_training_history(model_dir, path_to_models, figsize=None, fig_suptitle=None, objective=None,
                          loss_ylim=None, round_prefix=None, print_best_epoch=False):

    df_hist = load_history_single(model_dir, path_to_models=path_to_models, round_prefix=round_prefix)

    if 'epoch' not in df_hist.columns:
        df_hist['epoch'] = df_hist.index + 1

    fig, (ax1, ax2) = plt.subplots(ncols=2, figsize=figsize)
    ls_best_val_kwargs = {'ls': ':', 'color': 'tab:grey', 'alpha': 0.5}
    best_trial_kwargs = {'linewidth': None, 'alpha': 1.}
    other_trials_kwargs = {'linewidth': None, 'alpha': 0.2}
    ls_train, ls_val = '--', '-'

    if objective == 'val_loss':
        best_index = df_hist['val_loss'].argmin()
    elif objective == 'val_accuracy':
        best_index = df_hist['val_accuracy'].argmax()
    else:
        best_index = None

    if best_index is not None and ('trial' in df_hist.columns):
        best_trial = df_hist.loc[best_index, 'trial']
    else:
        best_trial = None

    # On the left axis we plot the training and validation loss
    ax = ax1
    if 'trial' in df_hist.columns:
        legend_markers = []
        # We loop over the trial
        for trial, df_sub in df_hist.groupby('trial'):
            if objective:
                if trial == best_trial:
                    trial_kwargs = best_trial_kwargs
                else:
                    trial_kwargs = other_trials_kwargs
            else:
                trial_kwargs = best_trial_kwargs

            lines = ax.plot(df_sub['epoch'], df_sub['loss'], ls=ls_train, **trial_kwargs)
            colour = lines[0].get_color()
            ax.plot(df_sub['epoch'], df_sub['val_loss'], ls=ls_val, color=colour, **trial_kwargs)

            # Plot a marker at the best epoch of the best trial
            if trial == best_trial:
                ax.scatter(df_sub.loc[best_index, 'epoch'], df_sub.loc[best_index, 'val_loss'],
                           color='black', marker='x', s=50.)

                if print_best_epoch:
                    print("Best epoch: ", df_sub.loc[best_index, 'epoch'])

            # Explicitly define a legend for each trial
            legend_marker = mlines.Line2D([], [], color=colour,
                                          markersize=10, label=f"Trial {trial}", linestyle='-', **trial_kwargs)
            legend_markers.append(legend_marker)

        # Explicitly define a legend for training/validation
        for label, ls in zip(['Training', 'Validation'], [ls_train, ls_val]):
            legend_marker = mlines.Line2D([], [], color='tab:grey',
                                          markersize=10, label=label, linestyle=ls)
            legend_markers.append(legend_marker)

        ax.legend(handles=legend_markers)
    else:
        lines = ax.plot(df_hist['epoch'], df_hist['loss'], label='Training', ls=ls_train)
        ax.plot(df_hist['epoch'], df_hist['val_loss'], label='Validation', color=lines[0].get_color(), ls=ls_val)
        ax.scatter(df_hist.loc[best_index, 'epoch'], df_hist.loc[best_index, 'val_loss'],
                   color='black', marker='x', s=50.)

        if print_best_epoch:
            print("Best epoch: ", df_hist.loc[best_index, 'epoch'])

        ax.legend()

    # Plot the minimum value of the loss
    xlims = ax.get_xlim()
    for metric in ['loss', 'val_loss']:
        # ax.hlines(val := df_hist[metric].min(), xlims[0], xlims[1], **ls_best_val_kwargs)
        ax.hlines(val := df_hist.loc[best_index, metric], xlims[0], xlims[1], **ls_best_val_kwargs)
        ax.text(xlims[1], val, np.round(val, 3), horizontalalignment='right', verticalalignment='bottom',
                color='tab:grey')

    if loss_ylim:
        ax.set_ylim(loss_ylim)
    else:
        # The loss can have extremely high outlier values - we want to ignore these when determining the y-lims
        y_loss_max = df_hist.loc[
            (df_hist['loss'] <= df_hist['loss'].mean() + (4. * df_hist['loss'].std())), 'loss'].max()
        y_val_loss_max = df_hist.loc[
            (df_hist['val_loss'] <= df_hist['val_loss'].mean() + (4. * df_hist['val_loss'].std())), 'val_loss'].max()
        y_max = max(y_loss_max, y_val_loss_max)

        if ax.get_ylim()[1] > 2 * y_max:
            ax.set_ylim(0, 2 * y_max)

    ax.set_xlabel('Epoch')
    ax.set_ylabel("Loss")
    ax.set_xlim(xlims)

    # On the right axis we plot the training and validation accuracy
    ax = ax2
    if 'trial' in df_hist.columns:
        legend_markers = []
        # We loop over the trial
        for trial, df_sub in df_hist.groupby('trial'):
            if objective:
                if trial == best_trial:
                    trial_kwargs = best_trial_kwargs
                else:
                    trial_kwargs = other_trials_kwargs
            else:
                trial_kwargs = best_trial_kwargs

            lines = ax.plot(df_sub['epoch'], df_sub['accuracy'], ls=ls_train, **trial_kwargs)
            colour = lines[0].get_color()
            ax.plot(df_sub['epoch'], df_sub['val_accuracy'], ls=ls_val, color=colour, **trial_kwargs)

            # Plot a marker at the best epoch of the best trial
            if trial == best_trial:
                ax.scatter(df_sub.loc[best_index, 'epoch'], df_sub.loc[best_index, 'val_accuracy'],
                           color='black', marker='x', s=50.)

                if print_best_epoch:
                    print("Best epoch: ", df_sub.loc[best_index, 'epoch'])

            # Explicitly define a legend for each trial
            legend_marker = mlines.Line2D([], [], color=colour,
                                          markersize=10, label=f"Trial {trial}", linestyle='-', **trial_kwargs)
            legend_markers.append(legend_marker)

        # Explicitly define a legend for training/validation
        for label, ls in zip(['Training', 'Validation'], [ls_train, ls_val]):
            legend_marker = mlines.Line2D([], [], color='tab:grey',
                                          markersize=10, label=label, linestyle=ls)
            legend_markers.append(legend_marker)

        ax.legend(handles=legend_markers)
    else:
        lines = ax.plot(df_hist['epoch'], df_hist['accuracy'], label='Training', ls=ls_train)
        ax.plot(df_hist['epoch'], df_hist['val_accuracy'], label='Validation', color=lines[0].get_color(), ls=ls_val)
        ax.scatter(df_hist.loc[best_index, 'epoch'], df_hist.loc[best_index, 'val_accuracy'],
                   color='black', marker='x', s=50.)

        if print_best_epoch:
            print("Best epoch: ", df_hist.loc[best_index, 'epoch'])

        ax.legend()

    # Plot the maximum value of the accuracy
    xlims = ax.get_xlim()

    for metric in ['accuracy', 'val_accuracy']:
        ax.hlines(val := df_hist.loc[best_index, metric], xlims[0], xlims[1], **ls_best_val_kwargs)
        # ax.hlines(val := df_hist[metric].max(), xlims[0], xlims[1], **ls_best_val_kwargs)
        ax.text(xlims[1], val, np.round(val, 3), horizontalalignment='right', verticalalignment='bottom',
                color='tab:grey')

    ax.set_xlabel('Epoch')
    ax.set_ylabel("Accuracy")
    ax.set_xlim(xlims)

    if fig_suptitle:
        fig.suptitle(fig_suptitle)

    fig.tight_layout()
    plt.show()
