import numpy as np
import scipy.signal as sgn

from brisk import fs_emg

# --- Filtering EMG
def filter_EMG(signal_in):
    b, a = sgn.butter(4, np.array([25, 350])/fs_emg, btype='bandpass')
    signal_out = sgn.filtfilt(b, a, signal_in, axis=0)
    b, a = sgn.butter(3, np.array([49.5, 50.5])/fs_emg, btype='bandstop')
    signal_out = sgn.filtfilt(b, a, signal_out, axis=0)
    return signal_out

# --- Envelope extraction
def envelope_EMG(signal_in):
    b, a = sgn.butter(3, 5/fs_emg, btype='lowpass')
    env = sgn.filtfilt(b, a, np.abs(signal_in), axis=0)
    env[np.where(env<1e-4*np.max(env))] = 1e-4*np.max(env)
    return env

# --- Coactivation
def coactivation_EMG(signal_in, events):
    env = envelope_EMG(filter_EMG(signal_in))
    n_muscles = env.shape[1]
    env = normalize_EMG(env, events)

    d = 0
    for i in range(n_muscles-1):
        for j in range(i+1, n_muscles):
            d += np.abs(env[:,i] - env[:,j])
    d /=  np.math.factorial(n_muscles)/(np.math.factorial(2)*np.math.factorial(n_muscles-2))
    c = 1 - 1 / (1 + np.exp(-12 * (d - 0.5)))
    m = (np.sum(env, axis=1) / (n_muscles))**2
    cc  = 1 / np.max(env, axis=1)

    return c*m*cc

# --- Normalization via events
def normalize_EMG(signal_in, events_in):
    norm_factor = []
    for evt in zip(events_in[:-1], events_in[1:]):
        e1 = int(evt[0]*fs_emg)
        e2 = int(evt[1]*fs_emg)
        norm_factor.append(np.max(signal_in[e1:e2,:], axis=0))
    norm_factor = np.asarray(norm_factor)
    norm_factor = np.median(norm_factor, axis=0)
    signal_out = signal_in
    for i in range(signal_out.shape[1]):
        signal_out[:,i] /= norm_factor[i]
    return signal_out
    