from sklearn.decomposition import NMF
import numpy as np

from brisk.analysis import emg
from brisk.utils.cl import print_error

NMF_OPTIONS = {
    'solver': 'mu',
    'max_iter': 1000,
    'init': 'random'
}

# --- VAF Function
def VAF(true_data, rec_data):

    return 1 - np.sum((true_data.flatten() - rec_data.flatten())**2)/np.sum(rec_data.flatten()**2)

# --- Remove negative values
def remove_negative(data_in):
    data_out = data_in.copy()
    for i in range(data_in.shape[1]):
        ths = 1e-6*np.max(data_in[:,i])
        data_out[data_in[:,i]<ths,i] = ths
    return data_out

# --- Downsample EMG
def downsample_EMG(data_in, ds_factor):
    data_out = data_in[::ds_factor,:]
    return data_out

# --- Synergy extractor
def extract_synergies(emg_in, events_in=None, ds_factor=10):

    env = emg.envelope_EMG(signal_in=emg.filter_EMG(emg_in))
    if events_in is not None:
        env = emg.normalize_EMG(signal_in=env, events_in=events_in)
    if env.shape[0] < env.shape[1]:
        env = env.transpose()
    env = downsample_EMG(env, ds_factor)
    env = remove_negative(env)
    n_muscles = env.shape[1]

    H_tot = []
    W_tot = []
    VAF_curve = []
    VAF_muscles = []

    for i in range(n_muscles):
        nmf = NMF(n_components=i+1, **NMF_OPTIONS)
        H_tot.append(nmf.fit_transform(env).transpose())
        W_tot.append(nmf.components_.transpose())
        rec = (W_tot[-1]@H_tot[-1]).transpose()
        VAF_curve.append(VAF(env, rec))
        VAF_muscles.append([VAF(env[:,j], rec[:,j]) for j in range(n_muscles)])
    
    return np.asarray(VAF_curve), W_tot, H_tot, np.asarray(VAF_muscles)
    
# --- Reconstructor
def nnr(data_in, w_in, max_iter, tol):
    if data_in.shape[1]<data_in.shape[0]:
        data_in = data_in.transpose()
    if w_in.shape[0]<w_in.shape[1]:
        w_in = w_in.transpose()
    c = 0
    convergence = False
    err = []
    h = np.random.rand(w_in.shape[1], data_in.shape[1])
    while c<max_iter:
        num = w_in.transpose()@data_in
        den = w_in.transpose()@w_in@h
        h *= num/den
        err.append(np.sqrt(np.sum((data_in.flatten() - (w_in@h).flatten())**2)))

        if c>10:
            if np.abs(err[-10] - err[-1]) < tol:
                c=max_iter
                convergence = True
        c += 1

    if not convergence:
        print_error('Algorithm did not converge')
        
    return h