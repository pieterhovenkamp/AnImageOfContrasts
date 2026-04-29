#!/usr/bin/env python3

import numpy as np
import pandas as pd
import scipy
from skimage import util, measure

from general_utils.df_sample import split_df_in_chunks
from plankton_image_ops.image_ops import read_img_from_zip_archive


def calc_volume_from_esd(esd):
    return 4. / 3. * np.pi * np.power(esd / 2., 3.)


def calc_esd_of_zip_img(zip_path, roi_name):
    img = read_img_from_zip_archive(zip_path, roi_name, with_background=False)
    binary = util.img_as_uint(img != scipy.stats.mode(img.flatten(), keepdims=False)[0])

    return measure.regionprops(binary)[0].equivalent_diameter_area


def calc_esd_for_df(df, save_dir, chunksize=1e4):
    # Split the DataFrame into chunks
    df_chunks = split_df_in_chunks(df, chunksize=chunksize)

    chunk_counter = 1
    for df_sub in df_chunks:
        t1 = dt.datetime.now()
        name_list, esd_list = [], []
        for i, row in df_sub.iterrows():
            esd = calc_esd_of_zip_img(row['filepath'], row['roi_name'])
            img_name = f"{row['ff_name']}_{row['roi_name']}"
            name_list.append(img_name)
            esd_list.append(esd)

        print("chunk counter: ", chunk_counter)
        chunk_counter += 1
        print("t_elapsed: ", dt.datetime.now() - t1)
        print()

        df_esd = pd.DataFrame({'image_name': name_list, 'esd': esd_list})

        save_file = dt.datetime.now().strftime("%Y%m%d_%H%M%S.%f") + ".csv"
        df_esd.to_csv(Path(save_dir) / save_file)