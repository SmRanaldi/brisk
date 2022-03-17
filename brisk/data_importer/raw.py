import pandas as pd

import os
import json

from brisk import config_dir, out_dir
from brisk.utils import path

# ----- Auxiliary functions -----

# Parse the conditions file
def _parse_conditions(dir_in):

    names = []
    with open(os.path.join(dir_in,'conditions.txt'),'r') as f:
        for line in f:
            out_str = line.strip().lower().replace(' ', '_')
            names.append(out_str)

    return names

# Save imu data
def _save_imu_df(path_in, schema_in, path_out):
    data = pd.read_csv(path_in)
    for (k,v) in schema_in.items():
        data.columns = [x.replace(v,k) for x in data.columns]
    data.iloc[:,:-1].to_csv(path_out)


# ----- Main functions -----

# Import data and save to the directory
# in the config files
def import_imu_data():

    base_dir = path.get_folder()
    if not base_dir:
        print('No directory selected, aborting.')
    else:
        files = path.get_imu_filename(base_dir)
        sname = base_dir.split(os.sep)[-1]
        with open(os.path.join(config_dir,'imu_std.json'), 'r') as f:
            imus = json.load(f)

        s_dir = os.path.join(out_dir, sname)
        path.make_directory(s_dir)

        conditions = _parse_conditions(base_dir)
        if len(conditions) == len(files):
            for condition, file in zip(conditions, files):
                cond_dir = os.path.join(s_dir, condition)
                path.make_directory(cond_dir)
                rawdata_dir = os.path.join(cond_dir, 'rawdata')
                path.make_directory(rawdata_dir)
                _save_imu_df(os.path.join(base_dir,file), imus, os.path.join(rawdata_dir,'imu.csv'))
        else:
            print('Different number of conditions and data files, aborting.')

    return
