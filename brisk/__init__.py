import os
import json

root_dir = os.path.normpath(os.path.dirname(__file__))
config_dir = os.path.join(root_dir,'config')
config_file_path = os.path.join(config_dir,'directories.json')

with open(config_file_path, 'r') as f:
    out_dir = json.load(f)['output_dir']

fs_marker = 250
fs_imu = 102.4
fs_emg = 1000