import os
import json

import brisk

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