import os
import json

import brisk
from brisk.utils import settings

def hello():
    print('Hello, BRISK user!')
    return

def show_folders():
    print(f'\nConfig folder: \n\t{brisk.config_dir}')
    print(f'\nOutput folder: \n\t{brisk.out_dir}\n')
    return

def show_imu_config():
    with open(os.path.join(brisk.config_dir, 'imu_std.json'), 'r') as f:
        imus = json.load(f)
    print(json.dumps(imus, indent=4))
    return

def set_out_dir():
    print('\nSelect directory for output files.')
    settings.set_directory()
    with open(brisk.config_file_path, 'r') as f:
        print('\nNew output folder set to:')
        print(f"\t {json.load(f)['output_dir']}\n")