#!/usr/bin/env python3
import datetime as dt
from pathlib import Path
from string import ascii_lowercase as alc

import cmocean
import cmcrameri

from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib import cm
import matplotlib.lines as mlines
import matplotlib.patches as mpatches
import matplotlib.transforms as transforms
import matplotlib.dates as mdates
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib.patches import BoxStyle
import matplotlib.ticker as mticker
from matplotlib.ticker import FuncFormatter, AutoMinorLocator, MultipleLocator, LogLocator

import numpy as np
import pandas as pd

from scipy.interpolate import griddata
from skimage import morphology


plot_colours_dict = {'detritus': 'tab:gray',
                     'sali': 'tab:blue',
                     'dosat': 'tab:orange',
                     'chla': 'tab:green',
                     'turb': 'tab:red',
                     'temp': 'tab:purple'}

ENV_COLOUR_MAPS_DICT = {
    'sali': cmocean.cm.haline,
    'bin_sali': cmocean.cm.haline,
    'temp': cmocean.cm.thermal,
    'bin_temp': cmocean.cm.thermal,
    'temp2': cmocean.cm.thermal,
    'bin_temp2': cmocean.cm.thermal,
    'dosat': cmocean.cm.oxy,
    'bin_dosat': cmocean.cm.oxy,
    'chla': cmocean.cm.algae,
    'bin_chla': cmocean.cm.algae,
    'turb': cmocean.cm.turbid,
    'bin_turb': cmocean.cm.turbid,
    'density': cmocean.cm.dense,
    'bin_density': cmocean.cm.dense,
    'nitrate': cmocean.cm.matter}


# CTD related
unit_n_per_m3 = "ind. m$^{-3}$"

plot_label_quant_dict_no_unit = {
    'dosat': 'Dissolved oxygen saturation',
    'temp': r'Temperature',
    'temp2': r'Temperature',
    'chla': 'Chlorophyll-a',
    'turb': 'Turbidity',
    'density': r'Density anomaly',
    'sali': 'Salinity',
    'depth': 'Depth',
    'abundance': "Abundance",
    'biovolume': "Biovolume",
    'biomass': "Carbon weight",
}

unit_dict = {
    'temp': r'$\degree$C',
    'temp2': r'$\degree$C',
    'chla': '$\mu$g L$^{-1}$',
    'sali': 'PSU',
    'dosat': '%',
    'turb': 'NTU',
    'depth': 'm',
    'density': r'kg m$^{-3}$',
    'abundance': unit_n_per_m3,
    'biovolume': r'mm$^3$ m$^{-3}$',
    'biomass': r'mg C m$^{-3}$',
    'vol_per_bin_m3': 'm$^3$',
}

plot_label_quant_dict = {'dosat': f"Dissolved oxygen saturation (%)",
                         'temp': f"Temperature ({unit_dict['temp']})",
                         'temp2': f"Temperature ({unit_dict['temp2']})",
                         'chla': f"Chlorophyll-a ({unit_dict['chla']})",
                         'turb': f"Turbidity ({unit_dict['turb']})",
                         'density': f"Density anomaly ({unit_dict['density']})",
                         'sali': f"Salinity ({unit_dict['sali']})",
                         'depth': f"Depth ({unit_dict['depth']})",
                         'do': 'Dissolved Oxygen',
                         'cond': 'Conductivity',
                         'abundance': fr'Abundance ({unit_n_per_m3})',
                         'biomass': f"Carbon weight ({unit_dict['biomass']})",
                         'biovolume': fr'Biovolume (mm$^3$ {unit_n_per_m3})',
                         }

quant_title_dict = plot_label_quant_dict.copy()


def format_datetime_axis(ax, axis='x', format='%H:%M', rotation=30):
    """
    Format an axis with a timestamp to a more readible format.

    :param ax: matplotlib Axes
    :param axis: str - 'x' or 'y'
    :param format: str - a datetime format
    :param rotation: int - rotation of the timestamp in degrees
    """
    ax.xaxis.set_major_formatter(mdates.DateFormatter(format))
    ax.tick_params(axis=axis, labelrotation=rotation)
    return ax


def add_hline_to_plot(ax, y, x_min=None, x_max=None, **kwargs):
    """

    """
    # We keep the original xlims of the plot
    xlims = ax.get_xlim()

    # If not specified, the line spans the entire plot in x-direction
    if x_min is None:
        x_min=xlims[0]
    if x_max is None:
        x_max=xlims[1]

    ax.hlines(y=y, xmin=x_min, xmax=x_max, **kwargs)
    ax.set_xlim(xlims)

    return ax


def get_relative_fraction_in_log_scale(y_tot, y_frac, y0):
    """
    The aim is to plot a relative contribution to some quantity y on a log scale, where
    y_tot are the total values, y_frac are the fractional values of y_tot (between 0 and 1),
    and y0 is the position of the x-axis. Note that this function can equally well be used
    for x-values in the exact same way.

    See: general_code/dev/develop_relative_contributions_on_log_scale

    """
    return np.power(y0, 1 - y_frac) * np.power(y_tot, y_frac)


def add_counts_to_barplot(ax, bar, num_digits=0, **text_kwargs):
    """
    Add counts to horizontal barplot (generated with ax.barh)
    :param ax:
    :param bar: output of ax.barh, i.e. bar = ax.barh(..)
    :param num_digits:
    :param text_kwargs:
    :return:
    """
    # Add counts above the two bar graphs
    for rect in bar:
        width, height = rect.get_width(), rect.get_height()
        ax.text(width, rect.get_y() + height / 2, f'{width:,.{int(num_digits)}f}', va='center', **text_kwargs)


def make_bar_plot_with_perc(ax, names, values, colours):
    bar1 = ax.barh(names, values, color=colours)

    # Add percentages next to the bars
    for rect in bar1:
        width, height = rect.get_width(), rect.get_height()
        ax.text(width, rect.get_y() + height / 2, f'{width:.1f}%', ha='left', va='center')

    return ax


# Updated 20251121
def make_contour_plot_of_env(fig, ax, df, quant, x_quant='distance', ngridx=1000, ngridy=1000,
                             rolling_mean=None, rolling_mean_kwargs=None,
                             interp_method='linear',
                             contour_colour_levels=50, contour_line_levels=None, plot_clabels=True,
                             colour_map=None, vmin=None, vmax=None,
                             draw_colourbar=True, cmap_norm=None, colourbar_kwargs=None, contour_label_kwargs=None,
                             cbar_labels_kwargs=None,
                             invert_x_axis=False, max_plot_depth=None, print_clabels=False,
                             show_actual_data=True,
                             mask_bottom=True, find_depth_extrema='continuous',
                             cbar_ticks=None):
    """

    :param fig:
    :param ax:
    :param df: pd.DataFrame - needs columns 'Time' (if x_quant='Time'), 'depth', quant, 'distance' (only necessary if x_quant='distance')
    :param quant: str - the quantity to plot as colour map. This quantity should be a column in df.
    :param x_quant:  str - should be 'Time' or 'distance'. Determines the quantity on the x-axis. A column with this
                           name should be in df.
    :param ngridx:
    :param ngridy:
    :param rolling_mean: int - number of data points over which a rolling mean is calculated to reduce effect of outliers. If None, no rolling mean is calculated.
    :param interp_method: str - interpolation method used in scipy.interpolate.griddate
    :param contour_colour_levels: int - number of colour levels in the colour map. If None, no filled contours are drawn
    :param contour_line_levels:
    :param colour_map: str
    :param vmin: float - minimum value of quant in colour map
    :param vmax: float - maximum value of quant in colour map
    :param draw_colourbar: bool - whether to draw a colour bar
    :param invert_x_axis: bool
    :param max_plot_depth:
    :param show_actual_data:
    :param mask_bottom: bool - if True, we mask the bottom depth to prevent interpolation across the sea floor between deeper stations
    :param find_depth_extrema: str - should be 'continuous' (this option requires the column 'Time' in df) or 'discrete'. Ignored if mask_bottom=False.
    The bottom depth is detected from the depth-profile in df. Choose 'continuous' when the DataFrame contains towed data that is continuous along the x-axis,
     and choose 'discrete' the data consists of single x-values for each station. In the latter case, we can simply take the maximum depth per station as the bottom depth, whereas
     with continuous x-data, we need to use the function find_local_extrema().
    :param cbar_ticks:
    :return:
    """
    # If we do not plot a colour map, we do not need the colour bar
    if contour_colour_levels is None:
        draw_colourbar = False

    if cbar_labels_kwargs is None:
        cbar_labels_kwargs = {}

    if colourbar_kwargs is None:
        colourbar_kwargs = {}

    if contour_label_kwargs is None:
        contour_label_kwargs = {}

    if rolling_mean is not None:
        if rolling_mean_kwargs is None:
            rolling_mean_kwargs = {}

        df[quant] = df[quant].copy().rolling(rolling_mean, **rolling_mean_kwargs).mean()

    if x_quant == 'Time':
        # We change the timestamps to the time difference in seconds with the first timestamp of the cast, because this makes the interpolation easier than with timestamps
        t0 = df['Time'].iat[0]
        df['time_delta'] = (df['Time'] - t0).dt.seconds
        x_quant = 'time_delta'

    # We choose easier names for the x, y and z arrays
    z = df[quant].copy()
    x, y = df[x_quant].copy(), df['depth'].copy()

    # Remove the nans (based on z) from all 3 arrays, because tricontour cannot handle these
    z_notna = z.loc[z.notna()]
    x = x.loc[z.notna()]
    y = y.loc[z.notna()]
    z = z_notna

    if not len(x):
        # ax.set_title(quant)
        ax.set_axis_off()
        print(f"No non-nan values are present for {quant}")
        return

    xi = np.linspace(x.min(), x.max(), ngridx)
    yi = np.linspace(y.min(), y.max(), ngridy)
    zi = griddata((x, y), z, (xi[None, :], yi[:, None]), method=interp_method)

    if mask_bottom:
        if find_depth_extrema == 'continuous':
            # For the case where data is continuous in x (as for towed transects), we need the following function to detect the depth extreme.

            # This requires the column 'Time'
            if 'Time' not in df.columns:
                raise ValueError(
                    "Detecting depth extreme on continuous data using get_local_extrema requires the column 'Time'")

            df_max = get_local_extrema(df, 'depth', min_or_max='max', include_boundaries=False)
            depth_quant = 'depth'

        elif find_depth_extrema == 'discrete':
            # For the case where data of each station has the same x-value, we use the column 'bottom_depth' if it is present already
            depth_quant = 'bottom_depth'

            if not 'bottom_depth' in df.columns:
                # Or we can simply get the bottom depth per station (no 'Time' required)
                df['bottom_depth'] = df.groupby('station')['depth'].transform('max')

            df_max = df.drop_duplicates(subset='bottom_depth', keep='first').sort_values(x_quant)
        else:
            raise ValueError(
                "Parameter find_depth_extreme should be either 'continuous' or 'discrete' when mask_bottom=True. See the docstrings for an explanation.")

        zi_masked = np.zeros_like(zi).astype(bool)
        for i in range(len(df_max) - 1):
            t1, t2 = df_max[x_quant].iat[i], df_max[x_quant].iat[i + 1]
            d1, d2 = df_max[depth_quant].iat[i], df_max[depth_quant].iat[i + 1]

            # We need the indices of the two corresponding extrema points on the zi-grid. We want the depth of these points > d2, d3 and we choose to take the first time > t2, t3
            # We could equally well have chosen the last time < t2, t3 but since we treat the consecutive pairs anyway, this does not make a difference. The boundary effect of this choice
            # is then covered because the first and last extrema are covered by the convex hull approach of the interpolation anyway.

            # We added 'or equal to' in order to work with pre-specified bottom depths. Modify if this gives errors with using get_local_extreme (which requires time-bins)
            idx_t2, idx_t3 = np.where(xi >= t1)[0][0], np.where(xi >= t2)[0][0]
            # idx_t2, idx_t3 = np.where(xi > t1)[0][0], np.where(xi > t2)[0][0]

            # If the detected bottom depth is deeper than the depth present in the DataFrame (this may occurs if data for 'quant' is missing for certain depths), we set the bottom depth to the maximum depth that is present
            # idx_d2, idx_d3 = np.where(yi >= d1)[0][0], np.where(yi >= d2)[0][0]
            if len(np.where(yi >= d1)[0]):
                idx_d2 = np.where(yi >= d1)[0][0]
            else:
                idx_d2 = np.where(yi == np.max(yi))[0][0]

            if len(np.where(yi >= d2)[0]):
                idx_d3 = np.where(yi >= d2)[0][0]
            else:
                idx_d3 = np.where(yi == np.max(yi))[0][0]

            # The upper boundary points of the masked region are the extrema that we detected. The lower bounds are the same time-coordinate but at the maximum depth
            zi_idx_point1, zi_idx_point2 = (idx_d2, idx_t2), (idx_d3, idx_t3)
            zi_idx_point3, zi_idx_point4 = (-1, idx_t2), (-1, idx_t3)

            zi_masked_sub = np.zeros_like(zi)
            for zi_idx_point in [zi_idx_point1, zi_idx_point2, zi_idx_point3, zi_idx_point4]:
                zi_masked_sub[zi_idx_point] = 1

            zi_masked += morphology.convex_hull_image(zi_masked_sub)

        # We set the z-grid to NaN at the masked region that we determined using the extrema
        zi[zi_masked] = np.nan

    if x_quant == 'time_delta':
        # Convert the timedelta in seconds back to timestamps
        xi = df['Time'].iat[0] + np.array([dt.timedelta(seconds=time_delta) for time_delta in xi])
        x_quant = 'Time'

    # Plot a colour map
    if contour_colour_levels:
        cntr1 = ax.contourf(xi, yi, zi, levels=contour_colour_levels, cmap=colour_map, vmin=vmin, vmax=vmax,
                            norm=cmap_norm)

    # Plot the contour lines
    if contour_line_levels:
        cntr2 = ax.contour(xi, yi, zi, levels=contour_line_levels, linewidths=0.5, colors='black')

        if plot_clabels:
            if 'colors' not in contour_label_kwargs:
                contour_label_kwargs['colors'] = 'black'
            clabels = ax.clabel(cntr2, **contour_label_kwargs)

        if print_clabels:
            print([clabel.get_position() for clabel in clabels])

        if draw_colourbar:
            cbar = fig.colorbar(cntr1,
                                # ax=ax,
                                ticks=contour_line_levels if cbar_ticks is None else cbar_ticks,
                                **colourbar_kwargs)
            cbar.add_lines(cntr2)
            cbar.set_ticklabels(contour_line_levels if cbar_ticks is None else cbar_ticks,
                                **cbar_labels_kwargs
                                )

    else:
        if draw_colourbar:
            if cbar_ticks:
                cbar = fig.colorbar(cntr1, ax=ax, ticks=cbar_ticks, **colourbar_kwargs)
            else:
                cbar = fig.colorbar(cntr1, ax=ax, **colourbar_kwargs)

    if show_actual_data:
        # Also plot the actual data points
        ax.plot(df[x_quant], df['depth'], color='tab:grey', ls='--')

    # x-axis
    ax.set_xlim(df[x_quant].min(), ax.get_xlim()[1])

    if invert_x_axis:
        ax.invert_xaxis()

    if x_quant == 'Time':
        format_datetime_axis(ax, axis='x', format='%H:%M', rotation=30)
    elif x_quant == 'distance':
        ax.set_xlabel('Distance from start (km)')

    # y-axis
    ax.set_ylabel('depth (m)')
    ax.invert_yaxis()

    if max_plot_depth:
        ax.set_ylim(max_plot_depth, 0)
        

def make_contour_plot_of_env_multiple(cast_id, ctd_list, df_env, figsize_per_row=3, figsize_per_col=6, figtitle='',
                                      vmin_dosat=0, vmax_dosat=120, colour_maps_dict=ENV_COLOUR_MAPS_DICT,
                                      contour_line_levels_dict=None, rolling_mean_dict=None,
                                      **contour_plot_kwargs):
    """
    Function around make_contour_plot_of_env to plot multiple CTD quantities in a single figure with nice colour
    maps, colour scaling etcetera.

    :param cast_id:
    :param ctd_list:
    :param df_env: pd.DataFrame - with columns 'cast_id', 'Time', 'depth', quant for every quant in ctd_list,
        'distance' (only necessary if x_quant='distance')
    :param figsize_per_row:
    :param figsize_per_col:
    :param figtitle:
    :param vmin_dosat:
    :param vmax_dosat:
    :param colour_maps_dict:
    :param contour_line_levels_dict:
    :param contour_plot_kwargs:
    :return:
    """
    max_depth = df_env['depth'].max()

    df_env_sub = df_env.loc[df_env['cast_id'] == cast_id].copy()

    if not len(df_env_sub):
        raise ValueError("DataFrame is empty")

    # Make the CTD plots
    ncols = 2
    nrows = int(np.ceil(len(ctd_list) / ncols))
    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * figsize_per_col, nrows * figsize_per_row))
    [ax.set_axis_off() for ax in axes.flatten()]

    for quant, ax in zip(ctd_list, axes.flatten()):
        ax.set_axis_on()

        if rolling_mean_dict:
            rolling_mean = rolling_mean_dict[quant]
        else:
            rolling_mean = None

        if quant == 'dosat':
            vmin_quant = vmin_dosat
            vmax_quant = vmax_dosat
        else:
            # Find the min/max of each quantity over all included casts, so that the colour maps have the same scaling. We need to do this for the rolling mean if applicable
            if rolling_mean:
                vmin_quant = df_env[quant].rolling(rolling_mean).mean().min()
                vmax_quant = df_env[quant].rolling(rolling_mean).mean().max()
            else:
                vmin_quant = df_env[quant].min()
                vmax_quant = df_env[quant].max()

        make_contour_plot_of_env(fig, ax, df_env_sub, quant,
                                 vmin=vmin_quant, vmax=vmax_quant, rolling_mean=rolling_mean,
                                 colour_map=colour_maps_dict[quant],
                                 contour_line_levels=contour_line_levels_dict[quant], **contour_plot_kwargs)

        ax.set_title(plot_label_quant_dict.get(quant, quant))
        ax.set_ylim(max_depth, 0)

    fig.suptitle(figtitle)
    fig.tight_layout()
    plt.show()


def get_log10_bins(bin_min, bin_max, n_bins, return_bin_widths=False):
    """
    Calculate a bin-range with logarithmically scaled bin sizes, based on log10.

    :param bin_min: float - lowest bin
    :param bin_max: float - maximal bin
    :param n_bins: int - number of bins
    :return: np.array - the boundaries of the bins, with length n_bins + 1.
    """
    # We calculate the exponents of the bin limits. We need n_bins + 1 values to get a
    # bin number of n_bins
    exp_range = np.linspace(np.log10(bin_min), np.log10(bin_max), n_bins + 1)
    bins = 10.**exp_range
    bin_widths = bins[1:] - bins[:-1]

    if return_bin_widths:
        return bins, bin_widths
    else:
        return bins


def get_bins_size_spectrum_rois(df, n_bins, col_roi_size='bbox_diag', normalise=False, return_bin_weights=False):
    """
    Please note that this is not a true NBSS, since the data are binned by counts, not by summed size per bin.
    """
    # Calculate logarithmically spaced bins and add weights corresponding to the length per bin
    bins, bin_lengths = get_log10_bins(df[col_roi_size].min() * 0.99, df[col_roi_size].max() * 1.01, n_bins,
                                       return_bin_widths=True)
    bin_weights = np.mean(bin_lengths) / bin_lengths
    
    binned, _ = np.histogram(df[col_roi_size], bins, density=normalise)
    binned = binned * bin_weights
    
    if return_bin_weights:
        return binned, bins, bin_weights
    else:
        return binned, bins
    
    
def get_size_spectrum_per_group(df, n_bins, col_roi_size, group_cols):
    """
    Calculate the size spectrum of the summed ROI sizes within each size bin, divided by bin widths, which is the standard procedure for calulcating
    a normalised biomass or biovolume size spectrum (NBSS).

    :param df: pd.DataFrame - DataFrame with a row per ROI and columns <col_roi_size> and <group_cols>
    :param n_bins: int
    :param col_roi_size: str - the column with the ROI sizes
    :param group_cols: str - ROIs are binned according to the specified columns (e.g. labels, sampling areas etc)
    :return: pd.DataFrame, np.array, np.array - with index the size bin and columns per group_cols of the summed sizes divided by the bin_widths,
                                                array of the bounds of each bin, array of the bin widths
    """
    # We define logarithmically spaced bins over the observed size range over all data
    bins, bin_widths = get_log10_bins(df[col_roi_size].min() * 0.99, df[col_roi_size].max() * 1.01, n_bins,
                                      return_bin_widths=True)

    # We assign each entry in the DataFrame to a bin with pd.cut
    df['size_bin'] = pd.cut(df[col_roi_size], bins=bins)
    df_pivot = pd.pivot_table(df, values=col_roi_size, index='size_bin', columns=group_cols,
                              aggfunc='sum', margins=False, observed=False, sort=True)

    # We add the width of each bin - this works because df_pivot and bin_widths are both sorted
    df_pivot = df_pivot.div(bin_widths, axis=0) * np.mean(bin_widths)

    return df_pivot, bins, bin_widths


def plot_size_spectrum_rois(df_rois, n_bins, col_roi_size='bbox_diag', pixel_res_mu=None, normalise=False):
    """
    Plot a ROI size spectrum based on counts

    :param df_rois: pd.DataFrame - with a row per ROI. Needs to contain column 'bbox_diag' (assumed to be in pixels)
    :param n_bins:
    :param col_roi_size:
    :param pixel_res_mu:
    :param normalise:
    :return: None
    """
    df_rois = df_rois.dropna(subset=[col_roi_size]).copy()

    binned, bins = get_bins_size_spectrum_rois(df_rois, n_bins, col_roi_size=col_roi_size, normalise=normalise)

    x_label = "Bbox diagonal"
    if pixel_res_mu is not None:
        bins = bins * pixel_res_mu / 1000
        x_label += " (mm)"

    fig, ax = plt.subplots()
    ax.stairs(binned, bins)

    ax.set_xscale('log')
    ax.set_xlabel(x_label)

    ax.set_yscale('log')
    ax.set_ylabel("ROI counts / bin size")

    ax.set_title("Size spectrum of ROIs, # total: " + f"{len(df_rois):,}".replace(',', '.'))
    plt.show()


# Updated 20250516
def plotter_quant_vs_time(ax, df, quant, cast=None, time_col='Time', plot_nans_at_zero=True,
                          include_legend=False, y_label=None, invert_y_axis=False, prof_type_col='prof_type',
                          show_title=True):
    """
    db needs to contain columns:
    - 'Time' (or else, specified by time_col)
    - 'depth'
    - 'prof_type',
    - 'datetime_start'
    - 'cast_id'
    """
    if quant not in df.columns:
        raise ValueError(f"Column {quant} was not found in df")

    if cast:
        df_cast = df.loc[df['cast_id' == cast]].copy()
    else:
        df_cast = df.copy()

    if not len(df_cast):
        raise ValueError("DataFrame is empty")

    if quant == 'depth':
        if prof_type_col in df_cast.columns:
            for prof_type, colour in zip(['UP', 'DOWN'], ['tab:blue', 'tab:orange']):
                ax.scatter(
                    df_cast.loc[df_cast[prof_type_col] == prof_type, time_col],
                    df_cast.loc[df_cast[prof_type_col] == prof_type, quant],
                    s=2, c=colour, label=prof_type.lower())

            ax.scatter(
                df_cast.loc[~df_cast[prof_type_col].isin(['UP', 'DOWN']), time_col],
                df_cast.loc[~df_cast[prof_type_col].isin(['UP', 'DOWN']), quant],
                s=2, c='tab:grey', label='n/a')
        else:
            ax.scatter(df_cast[time_col], df_cast[quant],
                       s=2, c='tab:grey')
    else:
        ax.scatter(df_cast[time_col], df_cast[quant],
           s=2, c=plot_colours_dict.get(quant, 'tab:blue'))

    # If True, we plot NaN-values at y=0, if False these won't be shown
    if plot_nans_at_zero:
        na_mask = df_cast[quant].isna()
        df_cast.loc[df_cast[quant].isna(), quant] = 0.
        ax.scatter(df_cast.loc[na_mask, time_col], [0] * sum(na_mask),
                   s=2, c='black', label='CTD NaN')

    # X-axis
    format_datetime_axis(ax, axis='x', format='%H:%M', rotation=30)

    # Y-axis
    if y_label:
        ax.set_ylabel(y_label)
    else:
        ax.set_ylabel(quant)

    if quant == 'depth' or invert_y_axis:
        ax.invert_yaxis()

    if include_legend:
        ax.legend()

    if show_title:
        ax.set_title(
            f"{df_cast['datetime_start'].iat[0]}, {cast}, {df_cast['station_name'].iat[0]}")
    return


# Updated 20250516
def plot_quant_per_cast(df_env, quant_list, ncols=4, plot_label_quant_dict=None, prof_type_col='prof_type'):
    """
    Simple function to plot the selected quantities (y-axis) versus Time (x-axis), with separate figures per cast in df_env.

    :param df_env: pd.DataFrame - needs columns 'Time', 'depth', 'prof_type', 'cast_id', and columns for the specified quants in quant_list
    :param quant_list: list
    :param ncols: int
    :param plot_label_quant_dict: None or dict - with plottable names of (part) of the quantities
    :return: None
    """
    if plot_label_quant_dict is None:
        plot_label_quant_dict = {}

    for cast, df_sub in df_env.groupby('cast_id'):
        nrows = int(np.ceil(len(quant_list) / ncols))
        fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=(ncols * 3, nrows * 3))
        
        if ncols == 1 and nrows == 1:
            axes = np.array([axes])
            
        [ax.set_axis_off() for ax in axes.flatten()]

        for quant, ax in zip(quant_list, axes.flatten()):
            ax.set_axis_on()

            # Plot the full data
            plotter_quant_vs_time(ax, df_sub, quant, cast=None, time_col='Time',
                                  plot_nans_at_zero=False,
                                  # plot_nans_at_zero=True,
                                  include_legend=True if quant == 'depth' else False, y_label='',
                                  show_title=False,
                                  prof_type_col=prof_type_col,
                                  )

            ax.set_title(plot_label_quant_dict.get(quant, quant))

        fig.suptitle(f"{cast}, {df_sub['station_name'].iat[0]}, {df_sub['ctd_instrument'].iat[0]}")
        fig.tight_layout()
        plt.show()


def plotter_depth_vs_time(df, cast, include_legend=False, figsize=(6, 6), ax=None):
    """
    df needs to contain columns:
    - 'Time'
    - 'depth'
    - 'prof_type',
    - 'datetime_start'
    - 'cast_id'
    """
    df_cast = df.query("cast_id == @cast").copy()
    # print(df_cast)
    if not len(df_cast):
        raise Exception("df_cast is empty")

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)

    if 'prof_type' in df.columns:
        df_cast_up = df_cast.query("prof_type == 'UP'")
        df_cast_down = df_cast.query("prof_type == 'DOWN'")
        df_cast_grey = df_cast.query("prof_type not in ['DOWN', 'UP']")

        ax.scatter(df_cast_down['Time'], df_cast_down['depth'], s=2,
                   c='tab:orange', label='down')
        ax.scatter(df_cast_up['Time'], df_cast_up['depth'], s=2,
                   c='tab:blue', label='up')

        if len(df_cast_grey):
            ax.scatter(df_cast_grey['Time'], df_cast_grey['depth'], s=2,
                       c='grey', label='n/a')
    else:
        ax.scatter(df_cast['Time'], df_cast['depth'], s=2,
                   c='grey', label='n/a')

    if any(df_cast.isna()):
        na_mask = df_cast['depth'].isna()
        df_cast.loc[na_mask, 'depth'] = 0.
        ax.scatter(df_cast.loc[na_mask, 'Time'], df_cast.loc[na_mask, 'depth'], s=2,
                   c='Black', label='CTD NaN')

    # X-axis
    format_datetime_axis(ax, axis='x')

    # Y-axis
    ax.set_ylabel("depth (m)")
    ax.set_ylim(ax.get_ylim()[1], ax.get_ylim()[0])

    if include_legend:
        ax.legend()
    ax.set_title(f"{df_cast['datetime_start'].iat[0]}, {cast}")

    if ax is None:
        plt.show()


def plot_cast_with_fps(df_img, rolling_mean=None, figsize=(12, 6), include_legend=True):
    """
    We plot the cast profiles (depth vs Time) in df_img together with the frame rate (fps vs Time).
    The full fps is plotted and if specified, the rolling mean is applied in addition.

    :param df_img: pd.DataFrame - with columns 'Time', 'ff_name', 'cast_id', 'fps'
    :param rolling_mean: int or None
    :return: None
    """
    df_img = df_img.copy()

    # df_fps = add_fps_to_img_df(df_img, rolling_mean=None)

    if rolling_mean:
        df_img['fps_rolling'] = df_img['fps'].rolling(rolling_mean, center=True, min_periods=1).mean()
        # df_fps_rolling_mean = add_fps_to_img_df(df_img, rolling_mean=1000)
        # df_fps = pd.merge(df_fps, df_fps_rolling_mean[['ff_name', 'fps']], suffixes=('', '_rolling'), on='ff_name')

    for cast_id, df_sub in df_img.groupby('cast_id'):

        fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=figsize)

        ax = ax1
        plotter_depth_vs_time(df_img, cast_id, include_legend=include_legend, ax=ax)

        ax = ax2
        ax.plot(df_sub['Time'], df_sub['fps'], ls='--', lw=1., color='grey')
        if rolling_mean:
            ax.plot(df_sub['Time'], df_sub['fps_rolling'], color='tab:blue')

        format_datetime_axis(ax, axis='x')
        ax.set_ylabel('fps')

        plt.show()


def get_local_extrema(df, quant, min_or_max, time_bin_width=60, include_boundaries=False):
    """

    :param df: pd.DataFrame - needs columns 'Time', quant
    :param quant: str - column of df for which extrema are detected
    :param min_or_max: str - one of ['min', 'max'], specifies which type of local extreme is considered
    :param time_bin_width: float - Width of the time bins in seconds. Any extrema that are apart less than this distance are smoothened out (meaning the extremer one will remain), and
                                   any extremes more than 3 time this distance will always be detected (provided there is another extreme in between with minimal width of time_bin).
    :param include_boundaries: bool - Add the start/end points of (pieces of) the cast to the extrema
    :return: pd.DataFrame - subset of df with only the rows of the detected extreme, with the original columns.
    """
    df = df.copy()

    # We define bins per unit of time and find the extreme per bin
    if min_or_max == 'min':
        # We explicitly need to handle the empty bins (these can arise from interruptions within the cast)
        # df_idx_extr = df.resample(f"{time_bin_width}S", on='Time')[quant].idxmin()
        df_idx_extr = df.resample(f"{time_bin_width}S", on='Time')[quant].apply(lambda x: x.idxmin() if len(x) else None)
    elif min_or_max == 'max':
        # df_idx_extr = df.resample(f"{time_bin_width}S", on='Time')[quant].idxmax()
        df_idx_extr = df.resample(f"{time_bin_width}S", on='Time')[quant].apply(lambda x: x.idxmax() if len(x) else None)
    else:
        raise ValueError(f"Parameter min_or_max should be 'min' or 'max'")

    # Make a new dataframe with only the maximums per timebin and use the depth gradient to find the local minima/maxima
    df_idx_extr.dropna(inplace=True)
    df_extr_sparse = df.loc[df_idx_extr].copy()
    df_extr_sparse['gradient'] = df_extr_sparse[quant] - df_extr_sparse[quant].shift(1)

    if min_or_max == 'min':
        df_extr = df_extr_sparse.loc[(df_extr_sparse['gradient'].shift(-1) > 0.) & (df_extr_sparse['gradient'] < 0.)].copy()
    else:
        df_extr = df_extr_sparse.loc[(df_extr_sparse['gradient'].shift(-1) < 0.) & (df_extr_sparse['gradient'] > 0.)].copy()
    df_extr.drop(columns='gradient', inplace=True)

    if include_boundaries:
        # We first etect jumps in time to define separate pieces of the cast
        df['Time_prev'] = df['Time'].shift(1)
        df['Time_gap'] = df['Time'] - df['Time_prev'] > dt.timedelta(seconds=time_bin_width)
        df['Time_gap_sum'] = df['Time_gap'].cumsum()

        # We loop over the pieces and get the indices of the first and last entries
        list_idx_boundaries = []
        for i, df_group in df.groupby('Time_gap_sum'):
            list_idx_boundaries += list(df_group.iloc[[0, -1]].index)

        # We make a DataFrame of these entries and append it to the DataFrame witht the extrema
        df_boundaries = df.loc[list_idx_boundaries].copy()
        df_extr = pd.concat([df_extr, df_boundaries])

    df_extr.sort_values('Time')

    return df_extr


def view_current_labeling(df, label_name='label', image_name_col='image_name', figsize=(12, 12), x_log=False,
                          num_digits=0, title=None, **text_kwargs):
    """

    :param df: DataFrame that needs to contain a column named 'image_name' and
    'label' (or else, specified by label_name)
    :param label_name: str - name of column containing the labels
    :return:
    """
    
    if np.any(na_mask := df[label_name].isna()):
        print(f"{na_mask.sum()} entries of {label_name} are NaN")
        df = df.dropna(subset=[label_name]).copy()

    grouped = df.groupby(label_name).count()

    fig, ax = plt.subplots(tight_layout=True, figsize=figsize)
    bar1 = ax.barh(grouped.index, grouped[image_name_col])
    ax.invert_yaxis()

    if x_log:
        ax.set_xscale('log')

    if title is not None:
        ax.set_title(title)
    else:
        ax.set_title(f"# images per label. Total # in selection: {len(df):,}")

    # Add counts above the two bar graphs
    add_counts_to_barplot(ax, bar1, num_digits=num_digits, **text_kwargs)

    plt.show()



def add_alphabet_plot_titles(axes_list, start_at_letter=None, as_capital=False, left_parentheses=False, right_parentheses=False, **title_kwargs):
    """
    Give plot titles to each subplot in axes_list with letters of the alphabet.

    :param axes_list: - list of axes.objects
    :param start_at_letter: str - if None, start at 'a'. If a letter of the alphabet, start there and continue alphabetically.
    :param title_kwargs: keyword arguments of ax.set_title()
    :return: None
    """
    if start_at_letter and start_at_letter in alc:
        alphabet_string = alc[alc.find(start_at_letter):]
    else:
        alphabet_string = alc


    for ax, title in zip(axes_list, alphabet_string):
        if as_capital:
            title = title.capitalize()

        if left_parentheses:
            title = '(' + title

        if right_parentheses:
            title = title + ')'

        ax.set_title(title, loc='left', **title_kwargs)


if __name__ == '__main__':
    pass

