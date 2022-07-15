import numpy as np
import scipy.signal as sgn
from sklearn.decomposition import PCA

from brisk import fs_emg
from brisk.utils.signal import norm_templatematching, norm_autocorrelation

# --- Filtering EMG
def filter_EMG(signal_in):
    b, a = sgn.butter(4, np.array([25, 400])/(fs_emg/2), btype='bandpass')
    signal_out = sgn.filtfilt(b, a, signal_in, axis=0)
    b, a = sgn.butter(3, np.array([49.5, 50.5])/(fs_emg/2), btype='bandstop')
    signal_out = sgn.filtfilt(b, a, signal_out, axis=0)
    return signal_out

# --- Envelope extraction
def envelope_EMG(signal_in):
    b, a = sgn.butter(3, 5/(fs_emg/2), btype='lowpass')
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

# --- Remove ECG
def remove_ECG(signal_in, mode='pca'):
    b, a = sgn.butter(4, 450/(fs_emg/2), btype='low')
    bb, aa = sgn.butter(4, 5/(fs_emg/2), btype='low')
    b_bp, a_bp = sgn.butter(4, [100/(fs_emg/2), 250/(fs_emg/2)], btype='bandpass')
    half_l_template = 100
    half_l = 400
    signal_lp = sgn.filtfilt(b, a, signal_in)
    signal_out = filter_EMG(signal_in)
    xcorr = sgn.filtfilt(bb, aa, np.abs(norm_autocorrelation(signal_lp[7000:14000])))
    # pk_loc_template = sgn.find_peaks(xcorr[500:].squeeze(), distance=500, height=np.mean(xcorr[500:])+2*np.std(xcorr[500:]))[0] + 500 + 7000
    # template = signal_lp[pk_loc_template[1]-350:pk_loc_template[1]+350].copy()
    pk_loc_template = np.argmax(xcorr[1500:].squeeze()) + 1500 + 7000
    template = signal_lp[pk_loc_template-half_l:pk_loc_template+half_l].copy()
    pk_loc_template = np.argmax(template[half_l_template:-half_l_template]) + half_l_template
    template = template[pk_loc_template-half_l_template:pk_loc_template+half_l_template]
    tm = norm_templatematching(signal_out, template)
    pk_loc = sgn.find_peaks(tm, distance=500, height=np.mean(tm)+np.std(tm))[0]
    pk_loc = pk_loc[pk_loc < signal_out.shape[0] - 2*half_l]
    pk_loc = pk_loc[pk_loc > 2*half_l]
    if (mode == 'multi') :
        half_l  = 500
        pk_loc = pk_loc[pk_loc < signal_out.shape[0] - half_l]
        pk_loc = pk_loc[pk_loc > half_l]
        qrs_all = []
        for p in pk_loc:
            qrs_all.append(signal_in[p-half_l:p+half_l])
        template = np.mean(np.asarray(qrs_all),axis=0).squeeze()
        pk_loc_template = np.argmax(template[half_l_template:-half_l_template]) + half_l_template
        template = template[pk_loc_template-half_l_template:pk_loc_template+half_l_template]
        tm = norm_templatematching(signal_lp, template)
        pk_loc = sgn.find_peaks(tm, distance=500, height=np.mean(tm)+np.abs(tm))[0]
        pk_loc = pk_loc[pk_loc < signal_out.shape[0] - 2*half_l]
        pk_loc = pk_loc[pk_loc > 2*half_l]
    for p in pk_loc:
        signal_out[p:p+2*half_l_template] -= (template/np.max(np.abs(template)))*np.max(np.abs(signal_out[p:p+2*half_l_template]))
        signal_out[p-2*half_l:p+2*half_l] = sgn.filtfilt(b_bp, a_bp, signal_out[p-2*half_l:p+2*half_l])
    if mode == 'pca':
        pca = PCA(n_components=2) 
        ref_signal = pca.fit_transform(
            np.concatenate([
                signal_out.reshape(-1,1),
                (signal_in - signal_out).reshape(-1,1)
            ], axis=1)
        )[:,0].squeeze()
        for p in pk_loc:
            signal_out[p:p+2*half_l_template] -= (ref_signal[p:p+2*half_l_template]/np.max(ref_signal[p:p+2*half_l_template]))*np.max(signal_out[p:p+2*half_l_template])
            signal_out[p-2*half_l:p+2*half_l] = sgn.filtfilt(b_bp, a_bp, signal_out[p-2*half_l:p+2*half_l])
    return signal_out, template