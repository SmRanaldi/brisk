import numpy as np
import scipy.signal as sgn

from brisk import fs_emg

# --- Filtering EMG
def filter_EMG(signal_in):
    b, a = sgn.butter(3, np.array([25, 450])/fs_emg, btype='bandpass')
    signal_out = sgn.filtfilt(b, a, signal_in, axis=0)
    b, a = sgn.butter(3, np.array([49.5, 50.5])/fs_emg, btype='bandstop')
    signal_out = sgn.filtfilt(b, a, signal_out, axis=0)
    return signal_out

# --- Envelope extraction
def envelope_EMG(signal_in):
    b, a = sgn.butter(3, 15/fs_emg, btype='lowpass')
    return sgn.filtfilt(b, a, np.abs(signal_in), axis=0)

# --- Coactivation
def coactivation_EMG(signal_in):
    env = envelope_EMG(filter_EMG(signal_in))
    n_muscles = env.shape[1]
    for i in range(n_muscles):
        env[:,i] /= np.mean(env[:,i])

    d = 0
    for i in range(n_muscles):
        for j in range(i+1, n_muscles):
            d += np.abs(signal_in[:,i] - signal_in[:,j])
    d /= 100 * np.math.factorial(n_muscles)/(np.math.factorial(2)*np.math.factorial(n_muscles-2))
    c = 1 - 1 / (1 + np.exp(-12 * (d - 0.5)))
    m = np.sum(signal_in, axis=1) / (100*n_muscles)
    cc  = (1/(100*n_muscles)) * np.sum(signal_in, axis=1) / np.max(signal_in, axis=1)

    return c*m*cc
    