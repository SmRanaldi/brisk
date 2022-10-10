import numpy as np
from scipy.spatial import distance_matrix
import scipy.signal as sgn

def recurrence_distance_matrix(signal_in, ds_step):
    signal_in = signal_in[::ds_step,:]
    return distance_matrix(signal_in, signal_in)
    
def embedded_recurrence_distance_matrix(signal_in, ds_step, tau=0, m=0, n_tau=None):
    if n_tau is not None:
        if n_tau > m:
            signal_tmp = signal_in[:tau*n_tau,:]
        else:
            signal_tmp = signal_in[:tau*m,:]
    else:
        signal_tmp = signal_in
    tau_ds = int(tau/ds_step)
    signal_tmp = signal_tmp[::ds_step,:]
    signal_embedded = signal_tmp[:-tau_ds*m]
    if m>0:
        for i in range(m):
            signal_embedded = np.hstack([signal_embedded, signal_tmp[tau_ds*m:tau_ds*m+signal_embedded.shape[0],:]])
    return distance_matrix(signal_embedded, signal_embedded)

def recurrence_matrix(data_in, ds_step):
    matrix_in = recurrence_distance_matrix(data_in, ds_step)
    # threshold = np.std(data_in, axis=(0,1))
    threshold = 0.2*np.mean(matrix_in.flatten()) 
    return (matrix_in<threshold).astype('float64', copy=True)

def embedded_recurrence_matrix(data_in, ds_step, tau=0, m=0, n_tau=None):
    matrix_in = embedded_recurrence_distance_matrix(data_in, ds_step, tau, m, n_tau)
    # threshold = np.std(data_in, axis=(0,1))
    threshold = 0.2*np.mean(matrix_in.flatten()) 
    return (matrix_in<threshold).astype('float64', copy=True)

def recurrence_rate(map_in):
    N = map_in.shape[0]
    map_out = map_in.copy()
    for i in range(N):
        map_out[i,i] = 0
    return np.sum(map_out)/(N**2-N)

def normalize_trunk_data(data_in):

    b, a = sgn.butter(3, 5/102.4, btype='lowpass')
    data_in = sgn.filtfilt(b, a, data_in, axis=0)

    b, a = sgn.butter(3, 0.1/102.4, btype='highpass')
    data_in = sgn.filtfilt(b, a, data_in, axis=0)

    return data_in