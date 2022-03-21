import os
import sys
import json

import brisk
from brisk.utils import settings, path

# General hello function
def hello():
    if len(sys.argv)>1:
        print(f'Hello, {sys.argv[1].capitalize()}!')
    else:
        print('Hello, BRISK user!')
    return

# Show the working folders
def show_folders():
    print(f'\nConfig folder: \n\t{brisk.config_dir}')
    print(f'\nOutput folder: \n\t{brisk.out_dir}\n')
    return

# Show active imu configuration
def show_imu_config():
    with open(os.path.join(brisk.config_dir, 'imu_std.json'), 'r') as f:
        imus = json.load(f)
    print(json.dumps(imus, indent=4))

# Set output directory
def set_out_dir():
    print('\nSelect directory for output files.')
    settings.set_directory()
    with open(brisk.config_file_path, 'r') as f:
        print('\nNew output folder set to:')
        print(f"\t {json.load(f)['output_dir']}\n")

# Print all the subjects that are available
# in the database
def print_subjects():
    subjects = [x for x in os.listdir(brisk.out_dir) 
        if os.path.isdir(os.path.join(brisk.out_dir,x))]
    print('\nDatabase directory:')
    print(f'\t{brisk.out_dir}\n')
    print('Trials found:')
    for s in subjects:
        print(f'\t{s}')
        for cond in path.get_trials(s):
            print(f'\t  {cond}')
    print('')