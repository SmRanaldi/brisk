import pandas as pd
import matplotlib.pyplot as plt

import os
import json
import shutil

from brisk import config_dir, out_dir
from brisk.utils import path
from brisk.utils.cl import print_error

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
def _save_imu_data(path_in, schema_in, path_out):
    data = pd.read_csv(path_in)
    for (k,v) in schema_in.items():
        data.columns = [x.replace(v,k) for x in data.columns]
    data.iloc[:,:-1].to_csv(path_out, index=False)
    events = _ask_events(data)
    abs_path = os.path.split(path_out)[0]

    if not os.path.exists(os.path.join(abs_path, 'events.json')):
        dict_out = {'events': {'imu': events}}
    else:
        with open(os.path.join(abs_path, 'events.json'), 'r') as f:
            dict_out = json.load(f)
        dict_out['events']['imu'] = events

    with open(os.path.join(abs_path, 'events.json'), 'w') as f:
        f.write(json.dumps(dict_out, indent=4))


# Ask for events
def _ask_events(data_in):
    sig = data_in['forearm_acc_y']
    plt.figure()
    plt.plot(sig)
    plt.title('Select start and stop points')
    (x0,_), (x1,_) = plt.ginput(2)
    plt.close()
    points = sorted([int(x0), int(x1)])
    return points


# ----- Main functions -----

# Get IMU config
def get_imu_config(base_dir):

    if os.path.exists(path.join_path([base_dir,'imu_config.json'])):
        print('\nUsing custom IMU configuration...\n')
        with open(path.join_path([base_dir,'imu_config.json']), 'r') as f:
            imus = json.load(f)
    else:
        print('\nUsing standard IMU configuration...\n')
        with open(path.join_path([config_dir,'imu_std.json']), 'r') as f:
            imus = json.load(f)

    return imus


# Import data and save to the directory
# in the config files
def import_imu_data(db_path=None):

    if not db_path:

        print('Select data folder:')

        base_dir = path.get_folder()

    if not base_dir:
        print_error('No directory selected, aborting.')
    else:
        files = path.get_imu_filename(base_dir)
        sname = base_dir.split(os.sep)[-1]

        imus = get_imu_config(base_dir)

        s_dir = os.path.join(out_dir, sname)
        arch_s_dir = os.path.join(out_dir, '_archive', sname)
        path.make_directory(s_dir)

        if os.path.exists(arch_s_dir):
            shutil.rmtree(arch_s_dir)
        shutil.copytree(base_dir,arch_s_dir)

        conditions = _parse_conditions(base_dir)
        if len(conditions) == len(files):
            for condition, file in zip(conditions, files):
                cond_dir = os.path.join(s_dir, condition)
                path.make_directory(cond_dir)
                rawdata_dir = os.path.join(cond_dir, 'rawdata')
                path.make_directory(rawdata_dir)
                _save_imu_data(os.path.join(base_dir,file), imus, os.path.join(rawdata_dir,'imu.csv'))
        else:
            print_error('Different number of conditions and data files, aborting.')

    return

# Load raw data
def load_raw_data(base_dir = None):

    if not base_dir:

        print('Select data folder:')

        base_dir = path.get_folder()

    if not base_dir:
        print_error('No directory selected, aborting.')
    else:
        if not os.path.exists(base_dir):
            print_error('Subject not in archive')
            return
        files = path.get_imu_filename(base_dir)
        sname = base_dir.split(os.sep)[-1]

        imus = get_imu_config(base_dir)

        conditions = _parse_conditions(base_dir)

        out_dict = {}
        for c,file in zip(conditions, files):
            data = pd.read_csv(os.path.join(base_dir,file))
            for (k,v) in imus.items():
                data.columns = [x.replace(v,k) for x in data.columns]
            out_dict[c] = data.iloc[:,:-1]

        return out_dict
