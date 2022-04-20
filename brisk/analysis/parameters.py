import numpy as np
import pandas as pd
from scipy.stats import iqr
from scipy.signal import find_peaks

import os

from brisk.analysis import segmentation
from brisk.utils.stats import remove_outliers
from brisk.utils.signal import trim_norm_autocorrelation
from brisk.utils.path import get_trials

# ----- Parameters -----
fs = 102.4
dt = 1/fs

# ----- Auxiliary -----

# Get smoothness
def _smoothness(signal_in):
    signal_in -= np.mean(signal_in)
    v = np.max(np.abs(np.cumsum(signal_in*dt)))
    jerk = np.diff(signal_in)
    log_dless_jerk = -np.log(jerk.size*(dt/v)*np.sum((jerk**2)*dt))

    return log_dless_jerk

# Get range
def _range(signal_in):

    return np.ptp(signal_in)

# Get rms
def _rms(signal_in):

    signal_in -= np.mean(signal_in)

    return np.sqrt(np.mean(signal_in**2))

# Get all single cycle parameters
def _get_cycle_param(signal_in):
    
    parameters = (_smoothness(signal_in), _range(signal_in), _rms(signal_in))
    labels = ('smoothness', 'range', 'rms')

    return parameters, labels

# Get regularity
def _regularity(signal_in):

    acorr = trim_norm_autocorrelation(signal_in, 20*fs)
    pk = find_peaks(acorr, height=0.1, distance=2*fs)
    if len(pk[0]):
        out = acorr[pk[0][0]]
    else:
        out = 0

    return out

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

# Get all parameters for each cycle
def cycle_parameters(subject):

    trials = get_trials(subject)

    out = []

    for t in trials:

        events = segmentation.load_indexes(subject, t)
        data_in = segmentation.get_filtered_data(subject, t)

        acc_col = [x for x in data_in.columns if 'acc' in x]
        gyr_col = [x for x in data_in.columns if 'gyr' in x]
        segments = np.unique([x.split('_')[0] for x in acc_col])

        for e in range(events.shape[0]-1):
            data_segment = data_in.iloc[events[e]:events[e+1],:]
            for c,cg in zip(acc_col,gyr_col):
                segment_body, _, dimension = c.split("_")
                parameters, labels = _get_cycle_param(data_segment[c].values)
                parameters_gyr, labels = _get_cycle_param(data_segment[cg].values)
                out.append([
                    t,
                    e,
                    segment_body,
                    dimension,
                    *parameters,
                    *parameters_gyr,
                ])
            for s in segments:
                s_col = [x for x in acc_col  if s in x]
                global_signal = np.sqrt(np.sum((data_segment[s_col].values)**2, axis=1))
                s_col = [x for x in gyr_col  if s in x]
                global_signal_gyr = np.sqrt(np.sum((data_segment[s_col].values)**2, axis=1))
                parameters_acc, labels = _get_cycle_param(global_signal)
                parameters_gyr, labels = _get_cycle_param(global_signal_gyr)
                out.append([
                    t,
                    e,
                    s,
                    'global',
                    *parameters_acc,
                    *parameters_gyr
                ])
        column_names = ['trial', 'event', 'segment', 'dimension',
            *[x+'_acc' for x in labels],
            *[x+'_gyr' for x in labels]
        ]

    return pd.DataFrame(out, columns=column_names)

# Get global and cycle parameters
def global_parameters(subject):
    param = cycle_parameters(subject)
    param_mean = param.groupby(['trial', 'segment','dimension']).mean()
    n_events = param.groupby(['trial', 'segment','dimension']).max()['event']+1
    param_mean['n_events'] = n_events
    param_mean.drop(columns=['event'], inplace=True)
    param_mean.reset_index(inplace=True)
    
    trials = get_trials(subject)
    out = []
    for t in trials:
        data = segmentation.get_filtered_data(subject, t)
        acc_col = [x for x in data.columns if 'acc' in x]
        gyr_col = [x for x in data.columns if 'gyr' in x]
        segments = np.unique([x.split('_')[0] for x in acc_col])
        for c, cg in zip(acc_col, gyr_col):
            segment_body, _, dimension = c.split("_")
            reg_acc = _regularity(data[c].values)
            reg_gyr = _regularity(data[cg].values)
            out.append([
                t,
                segment_body,
                dimension,
                reg_acc,
                reg_gyr
            ])
        for s in segments:
            s_col = [x for x in acc_col  if s in x]
            global_signal = np.sqrt(np.sum((data[s_col].values)**2, axis=1))
            s_col = [x for x in gyr_col  if s in x]
            global_signal_gyr = np.sqrt(np.sum((data[s_col].values)**2, axis=1))
            reg_acc = _regularity(global_signal)
            reg_gyr = _regularity(global_signal_gyr)
            out.append([
                t,
                s,
                'global',
                reg_acc,
                reg_gyr
            ])
    column_names = ['trial', 'segment', 'dimension','reg_acc','reg_gyr']
    out_reg = pd.DataFrame(out, columns=column_names)
        
    return param_mean.merge(right=out_reg, how='inner', on=['trial','segment','dimension'])