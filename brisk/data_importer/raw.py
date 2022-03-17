import pandas as pd

from brisk.utils import path

# -----

# Just a wrapper around read_csv
def import_imu_data(filename):

    return pd.read_csv(filename)

# Import data and save to the directory
# in the config files
def save_imu_data():

    base_dir = path.get_folder()
    files = path.get_imu_filename(base_dir)

    return