import numpy as np

import os
import sys

from brisk.analysis import segmentation
from brisk.utils import path
from brisk import out_dir

def run_subject():
    if len(sys.argv)>1:
        subject = sys.argv[1]
    else:
        print('Specify subject.')
        return
    
    trials_list = path.get_trials(subject)

    for t in trials_list:
        print(f'\nTrial {t}, extracting indexes and average profiles...\n')
        segmentation.get_average_profiles(subject, t)
