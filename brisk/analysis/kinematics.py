from distutils.log import Log
import scipy.signal as sgn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# from matplotlib.colors import LogNorm

import warnings

from brisk import fs_imu
from brisk.utils.cl import print_error

# --- Normalize imu data
def normalize_imu_data(data_in):

    b, a = sgn.butter(3, 5/fs_imu, btype='lowpass')
    data_out = sgn.filtfilt(b, a, data_in, axis=0)

    b, a = sgn.butter(3, 0.1/fs_imu, btype='highpass')
    data_out = sgn.filtfilt(b, a, data_out, axis=0)

    return data_out

# --- Get 2D activity zones
def get_zones(dict_in, labels):

    data_all = np.concatenate([
        normalize_imu_data(v[labels].values) for v in dict_in.values()
    ], axis=0)
    x_limits = [np.percentile(data_all[:,0], 25), np.percentile(data_all[:,0], 75)]
    y_limits = [np.percentile(data_all[:,1], 25), np.percentile(data_all[:,1], 75)]
    limits = np.concatenate([np.asarray(x_limits).reshape(1,-1), np.asarray(y_limits).reshape(1,-1)], axis=0)
    return limits # Limits: first row x-limits, second row y-limits

# --- Count activity in the phases
def phase_count(data_in_raw, limits):
    data_in = normalize_imu_data(data_in_raw)
    out_phases = np.zeros((3,3))
    idx_phase = [] # Top left = 0. Bottom right = 8
    for d in range(data_in.shape[0]):
        point = data_in[d,:]
        if (point[0] <= limits[0,0]) and (point[1] > limits[1,1]):
            out_phases[0,0] += 1
            idx_phase.append(0)
        elif (point[0] > limits[0,0]) and (point[0] <= limits[0,1]) and (point[1] > limits[1,1]):
            out_phases[0,1] += 1
            idx_phase.append(1)
        elif (point[0] > limits[0,1]) and (point[1] > limits[1,1]):
            out_phases[0,2] += 1
            idx_phase.append(2)
        elif (point[0] <= limits[0,0]) and (point[1] <= limits[1,1]) and (point[1] > limits[1,0]):
            out_phases[1,0] += 1
            idx_phase.append(3)
        elif (point[0] <= limits[0,1]) and (point[0] > limits[0,0]) and (point[1] <= limits[1,1]) and (point[1] > limits[1,0]):
            out_phases[1,1] += 1
            idx_phase.append(4)
        elif (point[0] > limits[0,1]) and (point[1] <= limits[1,1]) and (point[1] > limits[1,0]):
            out_phases[1,2] += 1
            idx_phase.append(5)
        elif (point[0] <= limits[0,0]) and (point[1] <= limits[1,1]):
            out_phases[2,0] += 1
            idx_phase.append(6)
        elif (point[0] > limits[0,0]) and (point[0] <= limits[0,1]) and (point[1] <= limits[1,0]):
            out_phases[2,1] += 1
            idx_phase.append(7)
        else:
            out_phases[2,2] += 1
            idx_phase.append(8)
    out_phases = np.asarray(out_phases)
    idx_phase = np.asarray(idx_phase)
    if not sum(out_phases.flatten())==data_in.shape[0]:
        print(np.sum(out_phases))
        print(data_in.shape[0])
    
    return out_phases, idx_phase

# --- Average a signal by phase
def average_by_phase(data_in, idx_phase, method='mean'):
    out = np.zeros((3,3))
    for i in range(9):
        col = i%3
        row = int(i/3)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            if method=='mean':
                out[row,col] = np.mean(data_in[np.where(idx_phase==i)[0]])
            elif method=='median':
                out[row,col] = np.median(data_in[np.where(idx_phase==i)[0]])
            elif method=='max':
                out[row,col] = np.max(data_in[np.where(idx_phase==i)[0]])
            elif method=='min':
                out[row,col] = np.min(data_in[np.where(idx_phase==i)[0]])
            elif method=='sum':
                out[row,col] = np.sum(data_in[np.where(idx_phase==i)[0]])
            else:
                print_error('Wrong averaging method selected, using mean.')
                out[row,col] = np.mean(data_in[np.where(idx_phase==i)[0]])
    return np.nan_to_num(out)

# --- Plot phases heatmap
def plot_phases(matrix_in, ax, vmin=0, vmax=1):
    heatmap_args = {
        'cbar': False,
        'annot': True,
        'xticklabels': False,
        'yticklabels': False,
        'annot_kws':{
            'fontsize': 16
        },
        'cmap': sns.color_palette("YlOrRd", as_cmap=True),
        'vmin': vmin,
        'vmax': vmax,
    }
    sns.heatmap(matrix_in, **heatmap_args, ax=ax)