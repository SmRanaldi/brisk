import numpy as np
from scipy.stats import iqr

import os

from brisk.analysis import segmentation

# ----- Functions -----

# Get all the time parameters from all the trials
def get_time_parameters(subject):
    
    frequencies = segmentation.get_frequencies(subject)

    time_param = {k: 
        {
            'avg_frequency': np.median(v),
            'all_frequency': v,
            'avg_duration': np.median(1/v),
            'std_duration': iqr(1/v)
        }
        for k, v in frequencies.items()}

    return time_param