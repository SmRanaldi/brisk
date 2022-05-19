from calendar import c
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sgn
from scipy.interpolate import interp1d

import os

from brisk.data_loader import trials
from brisk.utils.cl import *
from brisk.utils.signal import norm_autocorrelation, norm_templatematching
from brisk.utils import path
from brisk import out_dir

# ----- Constants -----

fs = 102.4
n_points = 200

# ----- Auxiliary methods -----

# Get indexes for segmentation
def _get_segmentation_indexes(subject, trial):

    data = trials.load_raw_trial(subject, trial)
    events = trials.load_index(subject, trial)

    columns_segment = [x for x in data.columns if 'forearm' in x]
    segment = data[columns_segment].values[events[0]:events[1],:]

    b, a = sgn.butter(3, 0.5/(fs/2), btype='low')
    segment_filt = sgn.filtfilt(b, a, segment[:,0]).squeeze()

    c_temp = norm_autocorrelation(segment_filt)
    
    indexes_temp = sgn.find_peaks(
        c_temp,
        height = 0.05,
        distance = fs
    )[0]
    template = segment_filt[:indexes_temp[0]]
    
    c = norm_templatematching(segment_filt, template)
    indexes = sgn.find_peaks(
        c,
        height=0.5,
        distance=fs*np.floor(template.shape[0]/fs)
    )[0]

    return indexes, pd.DataFrame(data.values[events[0]:events[1],:],columns=data.columns), pd.DataFrame(template, columns=['template'])

# Filter the IMU data and save results
def _filter_data(subject, trial):
    data_file = os.path.join(out_dir, subject, trial, 'segmented_imu.csv')
    if not os.path.exists(data_file):
        update_indexes(subject, trial)
    data_in = pd.read_csv(data_file)
    b, a = sgn.butter(3, 5/(fs/2), btype='low')
    data_out = pd.DataFrame(
        sgn.filtfilt(b, a, data_in.values, axis=0),
        columns = data_in.columns,
    )
    data_out.to_csv(os.path.join(out_dir, subject, trial, 'filtered_imu.csv'), index=None)

# Calculate average profiles
def _calculate_average(subject, trial, update=False):
    data_file = os.path.join(out_dir, subject, trial, 'filtered_imu.csv')
    if (not os.path.exists(data_file)) or (update):
        print_ongoing('\nSegmentation not found, calculating...\n')
        _filter_data(subject, trial)
    movements = []
    indexes = load_indexes(subject, trial)
    data = get_filtered_data(subject, trial)
    for i in range(len(indexes)-1):
        segm = data.values[indexes[i]:indexes[i+1],:]
        segm_interp = interp1d(np.linspace(0,segm.shape[0],segm.shape[0]),segm, axis=0)(np.linspace(0,segm.shape[0],n_points))
        movements.append(segm_interp)
    movements = np.asarray(movements)
    out = pd.DataFrame(np.mean(movements, axis=0).squeeze(), columns=data.columns)
    out.to_csv(os.path.join(out_dir, subject, trial, 'average_imu.csv'), index=None)

# ----- Functions -----

# Update and save the indexes
def update_indexes(subject, trial):

    trial_dir = os.path.join(out_dir,subject,trial)
    if not os.path.exists(trial_dir):
        print_error('Subject or trial not found')
        return
    
    indexes, data, template = _get_segmentation_indexes(subject,trial)

    pd.DataFrame(indexes, columns=['indexes']).to_csv(
        os.path.join(trial_dir,'indexes_cycles.csv'),
        index=None
    )
    data.to_csv(os.path.join(out_dir,subject,trial,'segmented_imu.csv'),index=None)
    template.to_csv(os.path.join(out_dir,subject,trial,'template_imu.csv'),index=None)

# Load segmentation indexes. If not found, calculate those
def load_indexes(subject, trial):
    index_file = os.path.join(out_dir,subject,trial,'indexes_cycles.csv')
    data_file = os.path.join(out_dir,subject,trial,'segmented_imu.csv')

    if not os.path.exists(index_file):
        print_ongoing('\nSegmentation not found, calculating...\n')
        update_indexes(subject, trial)
    
    indexes = pd.read_csv(index_file)

    return indexes.values.squeeze()

# Get the filtered data from memory
def get_filtered_data(subject, trial):
    data_file = os.path.join(out_dir, subject, trial, 'filtered_imu.csv')
    if not os.path.exists(data_file):
        print_ongoing('\nNo filtered data found, calculating...\n')
        _filter_data(subject, trial)

    return pd.read_csv(data_file)

# Get average profiles
def get_average_profiles(subject, trial, update=False):
    data_file = os.path.join(out_dir, subject, trial, 'average_imu.csv')
    if (not os.path.exists(data_file)) or (update):
        print_ongoing('\nNo filtered data found, calculating...\n')
        _calculate_average(subject, trial, update)

    return pd.read_csv(data_file)

# Get template
def get_template(subject, trial):
    data_file = os.path.join(out_dir, subject, trial, 'template_imu.csv')
    if not os.path.exists(data_file):
        print_ongoing('\nNo template found, calculating...\n')
        update_indexes(subject, trial)
    return pd.read_csv(data_file).values.squeeze()

# Get frequencies
def get_frequencies(subject):
    trials_list = path.get_trials(subject)
    idx = {k: load_indexes(subject,k) for k in trials_list}
    frequencies = {k: 60/(np.diff(idx[k])/fs) for k in trials_list}
    return frequencies
