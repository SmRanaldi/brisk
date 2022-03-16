import pandas as pd

from brisk.utils import path

# -----

# Just a wrapper around read_csv
def import_imu_data(filename):

    return pd.read_csv(filename)
