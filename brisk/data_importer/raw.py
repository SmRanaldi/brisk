import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

import os
import json
import shutil

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
    (x0,y0), (x1,y1) = plt.ginput(2)
    plt.close()
    points = sorted([int(x0), int(x1)])
    return points


# ----- Main functions -----

# Import data and save to the directory
# in the config files
def import_imu_data():

    print('Select data folder:')

    base_dir = path.get_folder()
    if not base_dir:
        print('No directory selected, aborting.')
    else:
        files = path.get_imu_filename(base_dir)
        sname = base_dir.split(os.sep)[-1]
        with open(os.path.join(config_dir,'imu_std.json'), 'r') as f:
            imus = json.load(f)

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
            print('Different number of conditions and data files, aborting.')

    return
