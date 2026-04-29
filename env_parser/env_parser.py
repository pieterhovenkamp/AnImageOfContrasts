#!/usr/bin/env python3

import datetime as dt
import numpy as np
import pandas as pd
from pathlib import Path
from pyproj import Transformer
import gpxpy.gpx
import gsw

from general_utils.parse_folder import parse_folder_to_df


# Updated 20230516
def import_env_files_from_folder_multiple_instruments(folders, ctd_instruments, use_temp_oxygen_sensor=False):
    """
    Import files with environmental data from different instruments simultaneously, and return all files in a
    single DataFrame. Note that different covered instruments return different data columns. NaNs are added in missing
    columns, and possible additional columns that are not explicitly included in the local parameter return_columns
    are dropped, such that the output DataFrame of this function always contains the same columns with the same names.

    For documentation on the different instruments that are included, see the docs
    of import_env_files_from_folder_per_instrument.
    
    Make sure that different file types are not mixed within the same folder, because the correct import function may
    not be applied then.

    :param folders: list - list of paths to folders to parse
    :param ctd_instruments: list - list of instrument names to parse. Order of the instruments has to correspond to the
     order of the folders. Options are: ['RBR', 'RBRConcerto',  'CPICS', 'ISIIS', 'LISST']
    :param use_temp_oxygen_sensor: bool - see import_env_files_from_folder_per_instrument
    :return: pd.DataFrame - with columns:
        'source_filename',
        'Time',
        'ctd_time',
        'temp',
        'pres',
        'chla',
        'do',
        'turb',
        'depth',
        'sali',
        'dosat',
        'cond',
        'prof_type',
        'ctd_instrument',
    """
    return_columns = [
        'source_filename',
        'Time',
        'ctd_time',
        'temp',
        'pres',
        'chla',
        'do',
        'turb',
        'depth',
        'sali',
        'dosat',
        'cond',
        'prof_type',
        'ctd_instrument',
    ]

    if len(folders) != len(ctd_instruments):
        raise ValueError('Number of folders does not match number of instruments')

    df_list = []
    for ctd_instrument, folder in zip(ctd_instruments, folders):
        df = import_env_files_from_folder_per_instrument(folder=folder, ctd_instrument=ctd_instrument, use_temp_oxygen_sensor=use_temp_oxygen_sensor)

        # Harmonize the columns of the different instruments in line with return_columns
        print_cols = []
        for col in return_columns:
            if col not in df.columns:
                print_cols.append(col)
                if col in ['ctd_time', 'prof_start', 'prof_end']:
                    df[col] = pd.NaT
                else:
                    df[col] = np.nan
        if len(print_cols) > 0:
            print(f"Columns {print_cols} were not found in the DataFrame of {ctd_instrument}, so they were filled with NaNs")

        print_cols = []
        for col in df.columns:
            if col not in return_columns:
                print_cols.append(col)
                df.drop(columns=[col], inplace=True)
        if len(print_cols) > 0:
            print(f"Columns {print_cols} were found in the DataFrame of {ctd_instrument} but not in return_columns, so they were dropped")

        df_list.append(df)

    return pd.concat(df_list, ignore_index=True).sort_values(['Time'])


# Updated 20230516
def import_env_files_from_folder_per_instrument(folder, ctd_instrument, use_temp_oxygen_sensor=False):
    """
    Import files with environmental data from <folder> for various options of instruments and return all files in a
    single DataFrame. Possible instruments are:
    - 'RBR'
        - Data recorded by the RBRMaestro and converted using the Ruskin software.
        - Function used to import files: read_rbr_env_files. Expected file extension: _data.txt
        - return columns are:
            'source_filename'
            'Time'
            'temp'
            'pres'
            'chla'
            'do'
            'turb'
            'depth'
            'sali'
            'dosat'
            'cond'
            'prof_type'
            'ctd_instrument'

            not present:
            'ctd_time'
    - 'RBRConcerto'
        - Data recorded by the RBRConcerto and converted using the Ruskin software.
        - Function used to import files: read_rbr_env_files. Expected file extension: _data.txt
        - CHECK return columns are:
            'source_filename'
            'Time'
            'temp'
            'pres'
            'chla'
            'do'
            'turb'
            'depth'
            'sali'
            'dosat'
            'cond'
            'prof_type'
            'ctd_instrument'

            not present:
            'ctd_time'
    - CPICS
        - Data recorded by the CPICS (via an external CTD instruments, current functionality was based on this being the RBRMaestro)
        - Function used to import files: read_cpic_env_files. Expected file extension: .aux.dat
        - return columns are same as for 'RBR'
    - ISIIS
        - Data recorded by the ISIIS (via an external CTD instrument, current functionality was based on this being the RBRMaestro)
        - Function used to import files: import_isiis_ctd_file. Expected file extension: .csv
        - return columns are:
            'source_filename'
            'Time' (as measured in the Sidekick),
            'ctd_time' (the corresponding timestamp on the RBR - use this for matching to any other instrument),
            'temp',
            'pres',
            'chla',
            'do',
            'turb',
            'depth',
            'sali',
            'dosat'
            'cond'
            'ctd_instrument'

            Not present:
            'prof_type'
    - LISST
        - Data recorded by the LISST
        - Function used to import files: import_lisst_file. Expected file extension: .csv
        - return columns are:
            'source_filename'
            'Time'
            'depth',
            'temp',
            'ctd_instrument'

            Not present:
            'ctd_time' (the corresponding timestamp on the RBR - use this for matching to any other instrument),
            'pres',
            'chla',
            'do',
            'turb',
            'sali',
            'dosat'
            'cond'
            'prof_type'
    - ship_ctd_pelagia
        - Data recorded by Pelagia's ship CTD
        - Function used to import files: read_ship_ctd_pelagia. Expected file extension: .cnv
        - return columns are:
            'source_filename'
            'Time' (as measured by the ship CTD),
            'temp',
            'pres',
            'chla',
            'do',
            'turb',
            'depth',
            'sali',
            'dosat' (calculated using GSW package),
            'cond',
            'ctd_instrument'
            
            Not present:
            'ctd_time'
            'prof_type'

    :param folder: str - path to the folder containing the data files to be imported
    :param ctd_instrument: str - name of the CTD instrument (see possible options below)
    :param use_temp_oxygen_sensor: bool - The current RBRMaestro contains two temperature sensors. By default we only
    import the temperature from the marine sensor, but there can be reasons to import the temperature sensor from the
    oxygen instead. Note that this parameter has no effect if ctd_instrument='LISST'
    :return:
    """
    if ctd_instrument == 'RBR':
        df_env = import_env_files_from_folder(folder, '_data.txt', import_func=read_rbr_env_file, add_dot_to_extension=False,
                                              use_temp_oxygen_sensor=use_temp_oxygen_sensor, type='maestro')
    elif ctd_instrument == 'RBRConcerto':
        df_env = import_env_files_from_folder(folder, '_data.txt', import_func=read_rbr_env_file, add_dot_to_extension=False,
                                              use_temp_oxygen_sensor=False, type='concerto')
    elif ctd_instrument == 'CPICS':
        df_env = import_env_files_from_folder(folder, extension='.aux.dat', import_func=read_cpics_env_file,
                                              use_temp_oxygen_sensor=use_temp_oxygen_sensor)
    elif ctd_instrument == 'ISIIS':
        df_env = import_env_files_from_folder(folder, extension='.csv', import_func=import_isiis_ctd_file,
                                              use_temp_oxygen_sensor=use_temp_oxygen_sensor)
    elif ctd_instrument == 'LISST':
        df_env = import_env_files_from_folder(folder, extension='.csv', import_func=import_lisst_file)
        
    elif ctd_instrument == 'ship_ctd_pelagia':
        df_env = import_env_files_from_folder(folder, extension='.cnv', import_func=read_ship_ctd_pelagia)

    else:
        raise ValueError(f'Instrument {ctd_instrument} was not recognized')

    df_env['ctd_instrument'] = ctd_instrument

    return df_env


# Updated 20230322
def import_env_files(*filepaths, import_func, drop_duplicates_col='Time', **import_func_kwargs):
    """
    Generic function to import multiple env or gps file and concatenate them in a single DataFrame.
    The function import_func should return a DataFrame with at least a column named 'Time'.

    :param filepaths: str or list of str
    :param import_func: function - function to import a single env file. Should return a DataFrame with column 'Time'
    :return: pd.DataFrame
    """
    df_sub_list = []
    for file in filepaths:
        try:
            df_sub = import_func(file, **import_func_kwargs)
            df_sub_list.append(df_sub)
        except Exception as error:
            print(f"Loading {file} with specified import_func generated the following error: ")
            print(error)

    if not len(df_sub_list):
        raise ValueError(f"No data was found")

    return pd.concat(df_sub_list, ignore_index=True).drop_duplicates(subset=[drop_duplicates_col], ignore_index=True).sort_values([drop_duplicates_col], ignore_index=True)


# Updated 20230322
def import_env_files_from_folder(folder, extension, import_func, drop_duplicates_col='Time', add_dot_to_extension=True, **import_func_kwargs):
    """
    Generic function to parse env or gps files from a folder and import all found files in a single DataFrame.

    :param folder: str - path to the folder with the env or gps files
    :param extension:
    :param import_func:
    :return:
    """
    df_files = parse_folder_to_df(folder, extension, add_dot_to_extension=add_dot_to_extension)['filepath']
    return import_env_files(*df_files, import_func=import_func, drop_duplicates_col=drop_duplicates_col, **import_func_kwargs)


# Updated 20230322
def read_rbr_env_file(filepath, apply_utc_diff=True, use_temp_oxygen_sensor=False, type='maestro'):
    """

    filenames: single filename or list of filenames, i.e. read_envfiles(filename) reads a single file,
        whereas read_envfiles(*filename_list) reads multiple files and concatenates the resulting dataframes.
    :param apply_utc_diff:
    :param folder: str - path to folder containing the files. Used to import utc-diff. data and cast (down/up) data.
    :return: pd.DataFrame: with columns:
    - 'source_filename'
    - 'Time'
    - 'temp'
    - 'pres'
    - 'chla'
    - 'do'
    - 'turb'
    - 'depth'
    - 'sali'
    - 'dosat'
    - 'cond'
    - 'prof_type'
    """
    if type == 'maestro':
        cols_dict = {
            'Time': 'Time',
            'Temperature': 'temp',
            'Temperature.1': 'temp2',
            'Pressure': 'pres',
            'Chlorophyll a': 'chla',
            'Dissolved O2 concentration': 'do',
            'Turbidity': 'turb',
            'Depth': 'depth',
            'Salinity': 'sali',
            'Dissolved O2 saturation': 'dosat',
            'Conductivity': 'cond',
        }
    elif type == 'concerto':
        cols_dict = {
            'Time': 'Time',
            'Temperature': 'temp',
            'Pressure': 'pres',
            'Chlorophyll': 'chla',
            'Depth': 'depth',
            'Salinity': 'sali',
            'Conductivity': 'cond',
        }
    else:
        raise ValueError(f"'Instrument RBR with type '{type}' was not recognized'")

    env_df = pd.read_csv(filepath, usecols=cols_dict.keys())
    env_df['source_filename'] = filepath

    if not len(env_df):
        raise ValueError(f"env_df is empty")

    env_df['Time'] = pd.to_datetime(env_df['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    env_df.iloc[:, 1:-1] = env_df.iloc[:, 1:-1].astype("float64")
    env_df.rename(columns=cols_dict, inplace=True)

    # We add the inferred cast_type as saved in *annotations_profile.txt
    annotations_filepath = filepath.replace('_data.txt', '_annotations_profile.txt')
    try:
        df_cast = _read_rbr_down_up_cast_data(annotations_filepath)
    except FileNotFoundError:
        print(f"{annotations_filepath}: no annotations file was found, so the annotations for this deployment are skipped")
        env_df['prof_start'], env_df['prof_end'], env_df['prof_type'] = pd.NaT, pd.NaT, pd.NaT
    else:
        df_cast.query("prof_type != 'PROFILE'", inplace=True)

        # Merge the cast data to env_df
        env_df = pd.merge_asof(env_df, df_cast, left_on='Time', right_on='prof_start',
                               direction='backward')
        env_df.loc[env_df['Time'] > env_df['prof_end'], df_cast.columns] = np.nan

    if apply_utc_diff:
        env_df = _apply_utc_diff_to_rbr_env_data(env_df, filepath.replace('_data.txt', '_metadata.txt'))

    if type == 'maestro':
        if use_temp_oxygen_sensor:
                env_df.drop(columns=['temp'], inplace=True)
                env_df.rename(columns={'temp2': 'temp'}, inplace=True)
        else:
            env_df.drop(columns=['temp2'], inplace=True)
    else:
        if use_temp_oxygen_sensor == True:
            print("No oxygen sensor was available, so the argument 'use_temp_oxygen_sensor' is ignored")
        
    env_df.drop(columns=['prof_start', 'prof_end'], inplace=True)

    return env_df


# Updated 20230322
def _read_rbr_env_metadata_file(filepath):
    """
    Read the metadata file <filename>_metadata.txt as saved by the RBR. We specifically read the lines stating the 'starttime' and 'endtime'.
    The resulting output are the corresponding values for these (one per file). We only need this data in the function apply_utc_diff_to_rbr_env_data

    :param filepath: str - path to metadata file (expected to end with _metadata.txt)
    :return: start_time, end_time of deployment
    """
    lines_list = []
    with open(filepath) as file:
        for line in file:
            if line.find("starttime") > 0 or line.find("endtime") > 0:
                line = line.strip('\n').strip().strip(',')
                for line_i in line.split(sep=' : '):
                    lines_list.append(line_i.strip('"'))

    start_time, end_time = pd.to_datetime(lines_list[1], format='%Y-%m-%d %H:%M:%S.%f'), pd.to_datetime(lines_list[3], format='%Y-%m-%d %H:%M:%S.%f')
    return start_time, end_time


# Updated 20250322
def _read_rbr_down_up_cast_data(filepath):
    """
    Parse all txt-files ending with annotations_profile.txt. Every identified cast
    contains 3 entries in the table: the first with the whole cast (Type = 'Profile'),
    the second with Type = 'DOWN', the third with Type = 'UP'.

    We only need this function in read_rbr_env_file

    :return: pd.DataFrame - with columns 'prof_start', 'prof_end', 'prof_type'
    """
    df = pd.read_csv(filepath, usecols=['Time 1', 'Time 2', 'Type'])
    for time in ['Time 1', 'Time 2']:
        df[time] = pd.to_datetime(df[time], format='%Y-%m-%d %H:%M:%S.%f')

    df.sort_values(by=['Time 1', 'Time 2'], ascending=[True, False], inplace=True)
    df.drop_duplicates(subset=['Time 1', 'Time 2', 'Type'],
                       inplace=True, ignore_index=True)
    # check_overlap_in_down_up_casts(df)
    return df.rename(columns={'Time 1': 'prof_start', 'Time 2': 'prof_end', 'Type': 'prof_type'})


# Updated 20230322
def _apply_utc_diff_to_rbr_env_data(env_df, filepath):
    """
    In some of the (older) RBRMaestro files, a timeshift was present between the timestamps from the _data.txt file,
    and the indicated start/end time in the '_metadata.txt' file of the deployment, which caused a time lag with the
    imaging data. From metadata file, we infer a possible timeshift that we apply to the timestamps from the
     _data.txt file.

    :param env_df: pd.DataFrame - with column 'Time', 'prof_start', 'prof_end'
    :param filepath: str - path to metadata file (expected to end with _metadata.txt)
    :return: pd.DataFrame - original DataFrame with, if applicable, a time shift applied to
    the columns 'Time', 'prof_start', 'prof_end'
    """
    # Check if a metafile is present
    try:
        start_time, end_time = _read_rbr_env_metadata_file(filepath)
    except FileNotFoundError:
        print(f"{filepath}: no RBR metadata file was found, therefore we did not check for possible UTC corrections in the RBR-timestamps for this file.")
        return env_df
    else:
        # Check if diff is consistent
        diff_start, diff_end = env_df['Time'].min() - start_time, env_df['Time'].max() - end_time
        if abs(diff_end - diff_start) > dt.timedelta(seconds=3):
            print(f"For file {filepath}, the detected time difference is not consistent between the start and end of the deployment: {diff_end} - {diff_start}")
            return env_df

        # Check if a time correction is necessary
        if diff_start > dt.timedelta(seconds=3):
            print(f"{filepath}: based on the RBR metadata file, a time correction of {diff_start} was applied to the RBR timestamps")
        else:
            return env_df

        # Apply the correction
        for col in ['Time', 'prof_start', 'prof_end']:
            env_df[col] = env_df[col] - diff_start

        return env_df


# Updated 20230322
def read_cpics_env_file(filename, use_temp_oxygen_sensor=False):
    """
    Import an aux.dat-file named 'filename' as created by CPICS. A DataFrame
    is returned with the same columns as read_envfile_single_or_list, all other columns
    are dropped.

    The columns 'prof_start, 'prof_end', 'prof_type' are NaN since this
    data is not saved in the aux.dat-file, but added for consistency.

    :param filename: str
    :return: DataFrame - with columns 'Time', 'temp', 'pres, 'chla', 'do', 'turb',
    'depth', 'sali', 'dosat', 'cond', 'source_filename',
    'prof_start', 'prof_end', 'prof_type'

    """
    usecols = ['Time',
               'temp',
               'temp2',
               'pres',
               'chla',
               'do',
               'turb',
               'depth',
               'sali',
               'dosat',
               'cond',
               ]

    df = pd.read_csv(filename, on_bad_lines='skip',
                     names=['Time',
                            'cond',
                            'temp',
                            'pres',
                            'chla',
                            'temp2',
                            'do',
                            'turb',
                            'Sea pressure',
                            'depth',
                            'sali',
                            'Speed of sound',
                            'Specific conductivity',
                            'dosat'],
                     dtype=str,
                     encoding_errors='ignore')

    df.drop(columns=[col for col in df.columns if col not in usecols],
            inplace=True)

    for col in [col for col in df.columns if col != 'Time']:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # Convert the column 'Time'. First, we need to remove double timestamps
    df['Time'] = df['Time'].apply(
        lambda x: ' '.join(x.split(sep=' ')[:2]) if type(x) == str else x)
    df['Time'] = pd.to_datetime(df['Time'],
                                format='%Y/%m/%d %H:%M:%S.%f',
                                errors='coerce')
    df.dropna(subset=['Time'], how='any', inplace=True)
    
    if use_temp_oxygen_sensor:
        df.drop(columns=['temp'], inplace=True)
        df.rename(columns={'temp2': 'temp'}, inplace=True)
    else:
        df.drop(columns=['temp2'], inplace=True)

    # For consistency of the output, we add the following columns
    df['prof_type'] = np.nan

    df['source_filename'] = filename

    return df


def import_lisst_file(path):
    """
    Parse LISST file - see manual p.80 for content of the columns

    :param path: str - path to file
    :return: pd.DataFrame - with columns 'source_filename', 'depth', 'temp', 'Time'
    """
    df = pd.read_csv(path, header=None,
                     usecols=[40, 41, 42, 43, 44, 45, 46, 47],
                     names=['depth', 'temp', 'year', 'month', 'day', 'hour', 'minute', 'second'])
    df['Time'] = pd.to_datetime(df[['year', 'month', 'day', 'hour', 'minute', 'second']])
    df.drop(columns=['year', 'month', 'day', 'hour', 'minute', 'second'], inplace=True)
    df['source_filename'] = path

    return df


# Updated 20250602
def import_suna_files_from_folder(folder, add_col_clean=True):
    df_suna_paths = parse_folder_to_df(folder, extension='.CSV')
    df_suna = import_env_files(*df_suna_paths['filepath'], import_func=import_suna_file)

    if add_col_clean:
        df_suna['nitrate_clean'] = df_suna['nitrate']

    return df_suna


def import_suna_file(path):
    """
    Nitrate concentration is given from the instrument in microMolar. This is converted to microgram per liter as:
    1 muM = 0.014007 microgram / liter
    :param path: str - path to file
    :return: pd.DataFrame - with columns 'Time', 'nitrate' (in micro molar)
    """
    df_suna = pd.read_csv(path, skiprows=14, usecols=[1, 2, 3], names=['year_day_julian', 'hour_julian', 'nitrate'])

    # Convert the day number to a date using a timedelta with respect to January 1st,
    # and convert the hour fraction to a timestamp
    df_suna['date'] = df_suna['year_day_julian'].apply(
        lambda x: dt.datetime(int(str(x)[:4]), 1, 1) + dt.timedelta(days=(int(str(x)[4:]) - 1)))
    df_suna['Time'] = df_suna['date'] + df_suna['hour_julian'].apply(lambda x: dt.timedelta(hours=x))
    df_suna.drop(columns=['year_day_julian', 'hour_julian', 'date'], inplace=True)

    # Convert to microgram per liter based on conversion in SUNA manual p.8
    # df_suna['nitrate'] = df_suna['nitrate'] * 0.014007

    return df_suna


def import_isiis_ctd_file(filepath, use_temp_oxygen_sensor=False):
    """
    Function to read the ctd files as created on the ISIIS Sidekick. By default these files contain many error entries
    because the Sidekick starts recording before the CTD does, so we filter these error messages here.

    The column Log Date in the csv-files is in unix time since 1904/1/1 UTC (see jade.docs.sixclear.com), so we
    need to translate this to a readable datetime.

    :param filepath:
    :return: pd.DataFrame - with columns:
        'source_filename'
        'Time' (as measured in the Sidekick),
        'ctd_time' (the corresponding timestamp on the RBR - use this for matching to any other instrument),
        'temp',
        'pres',
        'chla',
        'do',
        'turb',
        'depth',
        'sali',
        'dosat'
        'cond'
    """

    isiis_ctd_cols_dict = {
        'Log Date': 'Time',
        'Sensor Date': 'ctd_time',
        'Conductivity marine sensor[mS/cm]': 'cond',
        'Temperature marine sensor [degree C]': 'temp',
        'Temperature oxygen sensor [degrees Celcius]': 'temp2',
        'Pressure [dbar]': 'pres',
        'Chlorophyll a [ug/l]': 'chla',
        'Dissolved oxygen concentration [umol/l]': 'do',
        'Turbidity [NTU]': 'turb',
        'Depth [m]': 'depth',
        'Salinity [PSU]': 'sali',
        'Dissolved oxygen saturation [%]': 'dosat',
    }

    df = pd.read_csv(filepath, usecols=isiis_ctd_cols_dict.keys(), low_memory=False)

    # Drop al the entries where an error message was printed (these start with '{')
    df['error_entries'] = df['Log Date'].apply(lambda x: str(x).startswith('{'))
    df = df.loc[~df['error_entries']].copy()
    df.drop(columns=['error_entries'], inplace=True)

    # df = df.loc[~df['Log Date'].str.startswith('{')].copy()
    df.rename(columns=isiis_ctd_cols_dict, inplace=True)

    unix_offset = (dt.datetime(1970, 1, 1) - dt.datetime(1904, 1, 1)).days * 86400  # (number of days between 1904 and 1970) * 86400
    df['Time'] = df['Time'].apply(lambda x: dt.datetime.utcfromtimestamp(float(x) - unix_offset))

    df['ctd_time'] = pd.to_datetime(df['ctd_time'], format='%Y-%m-%d %H:%M:%S.%f')

    # df.iloc[:, 2:-1] = df.iloc[:, 2:-1].astype("float64")
    # In this way of converting the dtypes, error entries are filled with NaNs
    for col in df.columns:
        if col not in ['Time', 'ctd_time']:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    df['source_filename'] = filepath

    if use_temp_oxygen_sensor:
        df.drop(columns=['temp'], inplace=True)
        df.rename(columns={'temp2': 'temp'}, inplace=True)
    else:
        df.drop(columns=['temp2'], inplace=True)

    return df


def read_ship_ctd_pelagia(filepath):
    """
    Function to read the CTD files from Pelagia's ship CTD 

    :param filepath:
    :return: pd.DataFrame - with columns:
        'source_filename'
        'Time' (as measured by the ship CTD),
        'temp',
        'pres',
        'chla',
        'do',
        'turb',
        'depth',
        'sali',
        'dosat' (calculated using GSW package),
        'cond',
        'ctd_instrument',
    """
    # number refers to same number as in header of file (first column is '0' = first entry of each line)
    col_ind_list = [5, 1, 2, 4, 9, 15, 10, 12, 14, 3, 13]

    # These are the names we assign the columns, in the same order as the index list above
    # Note that in earlier versions we tried using the descent rate to estimate the down casts, but this was not accurate enough
    col_name_list = ['Time', 'pres', 'temp', 'do', 'chla', 'depth', 'turb', 'sali', 'density', 'cond', 'potential_temp']

    header_list = []
    col_list_list = [[] for i in range(len(col_name_list))]
    with open(filepath, 'r', errors='replace') as file:
        line_number = 1
        for line in file:
            if line_number < 300:
                header_list.append(line)
                line_number += 1
                continue

            line = line.split()

            for col_name, col_ind, col_list in zip(col_name_list, col_ind_list, col_list_list):
                # print(col_name, col_ind, line[col_ind])
                col_list.append(line[col_ind])
            line_number += 1

    df_ctd = pd.DataFrame(dict(zip(col_name_list, col_list_list)))
    # df_ctd['station'] = Path(filepath).name.replace('64PE518-', 'station').replace('CTD01.cnv', '')

    # Find the start time from the file header
    start_time_found = False
    for line in header_list:
        if line.startswith('# start_time = '):
            start_time = line.replace('# start_time = ', '')
            start_time = start_time[:start_time.find('[')].strip()
            start_time = dt.datetime.strptime(start_time, '%b %d %Y %H:%M:%S')
            start_time_found = True

    if not start_time_found:
        print("start time was not found")

    # Add the start time to the Time-bin (which is in seconds)
    df_ctd['Time'] = pd.to_timedelta(df_ctd['Time'].astype(float), unit='S')
    df_ctd['Time'] = df_ctd['Time'] + start_time

    # Convert all columns to floats except 'Time'
    for col in col_name_list:
        if col != 'Time':
            df_ctd[col] = df_ctd[col].astype(float)

    df_ctd['dosat'] = df_ctd['do'] / gsw.O2sol_SP_pt(df_ctd['sali'], df_ctd['potential_temp']) * 100
    df_ctd['source_filename'] = Path(filepath).name
    df_ctd['ctd_instrument'] = 'ship_ctd_pelagia'

    return df_ctd


def parse_gps_archive(path=None, extensions=['gpx', 'csv', 'plt']):
    """

    :param path:
    :return:
    """
    if path is None:
        warnings.warn("No argument for path is given, so the default from class DefaultArguments is "
                      "used. This behaviour will be deprecated. You can suppress this warning by specifying the "
                      "argument in the function call.")
        path = DefaultArguments.path_to_gps_files


    list_all = []
    for ext in extensions:
        filepaths = glob.glob(f"{path}/**/*.{ext}", recursive=True)
        list_all += filepaths
    return list_all


# Updated 20251020
def import_gps_files_from_folder(folder, filetype, **import_kwargs):
    """

    :param folder: str
    :param filetype: str - file type, which determines which import function is used
    :return: pd.DataFrame - with columns 'Time', 'lon', 'lat'
    """
    # Select suitable read function for the file extension
    if filetype == 'ozi_plt':
        df_gps = import_env_files_from_folder(folder, extension='.plt', import_func=read_ozi_file, **import_kwargs)
    elif (filetype == 'csv') or (filetype == '.csv'):
        df_gps = import_env_files_from_folder(folder, extension='.csv', import_func=read_gps_csv_file,  **import_kwargs)
    elif (filetype == 'gpx') or (filetype == '.gpx'):
        df_gps = import_env_files_from_folder(folder, extension='.gpx', import_func=read_gpx_file,  **import_kwargs)
    elif filetype == 'station_xlsx':
        df_gps = import_env_files_from_folder(folder, extension='.xlsx', import_func=read_gps_station_file,
                                              drop_duplicates_col='station_name',  **import_kwargs)
    else:
        raise ValueError("The current function was written for filetypes ozi_plt, csv and gpx, not for current"
                         f"\ntype: {filetype}")

    return df_gps


def read_gps_csv_file(filepath, date_col='Date', lat_col='latitude', lon_col='longitude',
                      lat_dir_col=None, lon_dir_col=None, extra_cols=None,
                      header=True, datetime_format=None, **read_csv_kwargs):
    """
    Suitable for reading shiplog-csv files. Assumes columns Date, latitude, longitude are present and
    any other columns is ignored.

    :param filenames: single filename or list of filenames, i.e. read_gps_file(filename) reads a single file,
        whereas read_gps_file(*filename_list) reads multiple files and concatenates the resulting dataframes.
    :paeam header: bool - if True, use the header of the columns as column names of the DataFrame. If False, column
    names '0', '1', '2' are assigned. If so, specify which are the datetime, lon and lat columns with date_col, lat_col,
    and lon_col respectively.
    :return: pd.DataFrame - with column 'Time' (dtype=datetime64), 'lat' (dtype=float64), 'lon' (dtype=float64)
    """
    if extra_cols is None:
        extra_cols = []
        
    if date_col is None and lat_col is None and lon_col is None:
        df_gps = pd.read_csv(filepath, **read_csv_kwargs)
    elif header == False:
        df_gps = pd.read_csv(filepath, header=None, **read_csv_kwargs)
    elif lat_dir_col is not None and lon_dir_col is not None:
        df_gps = pd.read_csv(filepath, usecols=[date_col, lat_col, lon_col, lat_dir_col, lon_dir_col] + extra_cols,
                             **read_csv_kwargs)
        if np.any(~df_gps[lat_dir_col].isin(['N', 'S'])) or np.any(~df_gps[lon_dir_col].isin(['E', 'W'])):
            raise ValueError(f"Entries for {lat_dir_col} or {lon_dir_col} in datafile were not recognized")
        else:
            df_gps.loc[df_gps[lat_dir_col] == 'S', lat_col] *= -1
            df_gps.loc[df_gps[lon_dir_col] == 'W', lon_col] *= -1
    else:
        df_gps = pd.read_csv(filepath, usecols=[date_col, lat_col, lon_col] + extra_cols, **read_csv_kwargs)

    df_gps.dropna(how='any', inplace=True)

    df_gps.rename(columns={date_col: 'Time', lat_col: 'lat', lon_col: 'lon'}, inplace=True)

    if datetime_format is None:
        df_gps['Time'] = pd.to_datetime(df_gps['Time'], format='%Y-%m-%d %H:%M:%S.%f')
    else:
        df_gps['Time'] = pd.to_datetime(df_gps['Time'], format=datetime_format)

    return df_gps


def read_gpx_file(filepath):
    """
    Suitable for reading a single gpx-file. Tested only for files with a '.gpx'-extension.

    :param filepath: str - path to a .gpx-file.
    :return: pd.DataFrame - with column 'Time' (dtype=datetime64), 'lat' (dtype=float64), 'lon' (dtype=float64)
    """
    with open(filepath, 'r') as file:
        gpx = gpxpy.parse(file)

    lat_list, lon_list, time_list = [], [], []
    for track in gpx.tracks:
        for segment in track.segments[-4:]:
            # print('\n', segment.points)
            for point in segment.points:
                # print(f"lat: {point.latitude}, lon: {point.longitude}, time: {point.time}")
                lat_list.append(point.latitude)
                lon_list.append(point.longitude)
                time_list.append(point.time)

    df_gps = pd.DataFrame({'lat': np.array(lat_list), 'lon': np.array(lon_list), 'Time': np.array(time_list)})
    df_gps['Time'] = df_gps['Time'].dt.tz_localize(None)
    df_gps.dropna(how='any', inplace=True)

    return df_gps


def read_ozi_file(filepath, n_header_rows=6):
    """
    Suitable for reading a gps file with extention .plt from ozi. Tested on ReViFES 2022 data, if applying on new
    files test whether same number of lines is present in header.

    This function was written specifically to parse .plt-files - a GPS related format.
    It is not guaranteed at all to give correct results on any other type of files!

    :param filenames: str - path to single file
    :param n_header_rows: int - defaults to 6, which was the correct number for ReViFES 2022 files.
    :return: pd.DataFrame - with column 'Time' (dtype=datetime64), 'lat' (dtype=float64), 'lon' (dtype=float64)
    """
    lat_list, lon_list, date_list, time_list = [], [], [], []
    with open(filepath, 'r') as reader:
        counter = 1
        for line in reader:
            if counter > n_header_rows:
                line_mod = line.replace(" ", "").replace("\n", "").split(sep=',')
                lat, lon, date, time = line_mod[0], line_mod[1], line_mod[5], line_mod[6]
                lat_list.append(lat)
                lon_list.append(lon)
                date_list.append(date + '_')
                time_list.append(time)
            counter += 1

        df_gps = pd.DataFrame({
            'Time': np.char.add(np.array(date_list).astype(str), np.array(time_list).astype(str)),
            'lat': np.array(lat_list).astype(float),
            'lon': np.array(lon_list).astype(float),
        })
        df_gps['Time'] = pd.to_datetime(df_gps['Time'], format='%d-%m-%Y_%H:%M:%S')

    return df_gps


def read_gps_station_file(filepath):
    """
    Read a hand-made Excel-file with gps-data per station. Excel-file should contain the columns 'station_name', 'lon', 'lat'.
    Any other columns are ignored.

    :param filepath - str
    :return: pd.DataFrame - with column 'station_name', 'lat' (dtype=float64), 'lon' (dtype=float64)
    """
    if '$' in filepath:
        print(f"Filepath {filepath} containing a $ was probably not meant to be parsed, so we skip this one")
        return
    
    df_gps = pd.read_excel(filepath)
    df_gps.dropna(how='all', inplace=True)

    # Check if station_locations matches assumed pattern
    station_columns = ['station_name', 'lon', 'lat']
    if len(missing_cols := [keep_col for keep_col in station_columns if keep_col not in df_gps.columns]):
        raise ValueError(f"sheet station_locations in {filepath} should contains the following columns: "
                         f"{station_columns}, but the following columns are missing: {missing_cols}")

    # We tidy the station names
    df_gps['station_name'] = df_gps['station_name'].str.strip().str.replace(' ', '_').str.lower()

    # If present, we convert the RD_x and RD_y columns to lon/lat coordinates when lon/lat are missing
    if ('RD_x' in df_gps.columns) and ('RD_y' in df_gps.columns):
        # Only select entries where RD_X and RD_y are present and where lon or lat are missing
        df_gps_sub = df_gps.dropna(subset=['RD_x', 'RD_y'], how='any').copy()
        df_gps_sub = df_gps_sub.loc[df_gps_sub['lon'].isnull() | df_gps_sub['lat'].isnull()].copy()

        # Define the RD (Rijksdriehoek) and WGS84 coordinate systems
        rd_to_wgs84 = Transformer.from_crs(
            "EPSG:28992",  # RD (Rijksdriehoekstelsel)
            "EPSG:4326",   # WGS84 (Lon/Lat)
        )

        df_gps_sub[['lat', 'lon']] = pd.DataFrame(
            df_gps_sub.apply(lambda df: rd_to_wgs84.transform(df['RD_x'], df['RD_y']), axis=1).tolist(),
            index=df_gps_sub.index)

        # We overwrite the current entries of df_gps - we need to set existing values of lon/lat to NaN
        df_gps.loc[df_gps['lon'].isnull() | df_gps['lat'].isnull(), ['lon', 'lat']] = np.nan
        df_gps = df_gps.combine_first(df_gps_sub)

    df_gps.drop(columns=[col for col in df_gps.columns if col not in station_columns + ['RD_x', 'RD_y']], inplace=True)
    return df_gps



if __name__ == '__main__':
    sample_path_lisst = f"/users/Pieter/Documents/SMILE_code/databases/LISST_examples/L2771547.csv"
    df = import_lisst_file(sample_path_lisst)
    print(df)

    sample_path_suna = f"/users/Pieter/Documents/SMILE_code/databases/SUNA_examples/A0000155_20231004.CSV"
    df = import_suna_file(sample_path_suna)
    print(df)
