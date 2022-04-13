import pandas as pd

import os
import json

from brisk import out_dir
from brisk.utils import path

# Load raw data from a trial
def load_raw_trial(subject, trial):
    if path.search_subject(subject) and path.search_trial(subject,trial):
        data_imu = pd.read_csv(os.path.join(out_dir,subject,trial,'rawdata','imu.csv'))
        return data_imu
    else:
        exit()

# Load start and stop indexes
def load_index(subject, trial):
    if path.search_subject(subject) and path.search_trial(subject,trial):
        with open(os.path.join(out_dir,subject,trial,'rawdata','events.json'),'r') as f:
            events = json.load(f)
            return events['events']['imu']
    else:
        print('Subject or trial not found.')
