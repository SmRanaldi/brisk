import numpy as np
from scipy.stats import iqr
from scipy.signal import find_peaks

import os

from brisk.analysis import segmentation
from brisk.utils.stats import remove_outliers
from brisk.utils.signal import norm_autocorrelation

# ----- Parameters -----
fs = 102.4
dt = 1/fs

# ----- Auxiliary -----

# Get smoothness
def _smoothness(signal_in):
    v = np.max(np.cumsum(signal_in*dt))
    jerk = np.diff(signal_in)
    log_dless_jerk = -np.log(jerk.size*(dt/v)*np.cumsum((jerk**2)*dt))

    return log_dless_jerk

# Get range
def _range(signal_in):

    return np.ptp(signal_in)

# Get rms
def _rms(signal_in):

    return np.sqrt(np.mean(signal_in**2))

# Get regularity
def _regularity(signal_in):

    acorr = norm_autocorrelation(signal_in)
    pk = find_peaks(acorr, height=0.1, distance=2*fs)

    return pk[0]

# ----- Functions -----

# Get all the time parameters from all the trials
def get_time_parameters(subject):
    
    frequencies = segmentation.get_frequencies(subject)

    frequencies = {k: remove_outliers(v,5) for k,v in frequencies.items()}

    time_param = {k: 
        {
            'avg_frequency': np.median(v),
            'all_frequency': v,
            'avg_duration': np.median(60/v),
            'std_duration': iqr(60/v)
        }
        for k, v in frequencies.items()}

    return time_param