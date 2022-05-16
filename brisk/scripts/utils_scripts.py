import argparse
from termcolor import colored

import os
import sys
import json

import brisk
from brisk.utils import settings, path

# General hello function
def hello():
    parser = argparse.ArgumentParser(description='Just an hello script, useful for visualizing folders and subjects')
    parser.add_argument('-n', '--name', default='BRISK user', help='Name of the user (Not needed in this version)')
    parser.add_argument('-s', '--subjects', help='Show subjects list', action='store_true')
    parser.add_argument('-f', '--folders', help='Show folders', action='store_true')
    parser.add_argument('-c', '--show-config', help='Show IMU config', action='store_true')
    args = parser.parse_args()
    print(f'Hello, {args.name.capitalize()}!')
    if args.subjects:
        print_subjects()
    if args.folders:
        show_folders()
    if args.show_config:
        show_imu_config()
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
    print()
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
        if os.path.isdir(os.path.join(brisk.out_dir,x)) and not x.startswith('_')]
    print('\nDatabase directory:')
    print(f'\t{brisk.out_dir}\n')
    print('Trials found:')
    for s in subjects:
        print(f'\t{s.title()}')
        for cond in path.get_trials(s):
            print(f'\t  {cond.replace("_"," ").title()}')
    print('')