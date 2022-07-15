import numpy as np

from dtaidistance import dtw
import array

# ----- Functions -----

# Custom autocorrelation
def norm_autocorrelation(a):

    a = a.squeeze()

    corr_temp = [np.sum(a[:-t]*a[t:])/(a.size-t)
        for t in range(
                1, 
                a.shape[0]-1
            )
        ]

    return np.asarray(corr_temp)

# Custom trimmed autocorrelation
def trim_norm_autocorrelation(a, n_samples):

    n = int(np.round(n_samples))

    a = a.squeeze()

    corr_temp = [np.sum(a[:-t]*a[t:])/(a.size-t) 
        for t in range(
                1,
                n
            )
        ]

    return np.asarray(corr_temp)

# Custom template matching
def norm_templatematching(a,template_in):

    a = a.squeeze()
    template = template_in.copy().squeeze()
    template /= np.linalg.norm(template)
    l = template.shape[0]

    corr_temp = [np.sum(template*(a[t:t+l]/np.linalg.norm(a[t:t+l])))
        for t in range(
                0, 
                a.shape[0]-l
            )
        ]

    return np.asarray(corr_temp)

# RMSE template matching
def rms_templatematching(a,template):

    a = a.squeeze()
    template = template.squeeze()

    corr_temp = [np.sqrt(np.sum((template - a[t:t+template.shape[0]])**2)) 
        for t in range(
                0, 
                a.shape[0]-template.shape[0]
            )
        ]

    return np.asarray(corr_temp)

# Custom template matching
def dtw_templatematching(a,template):

    a = a.squeeze()
    template = template.squeeze()

    corr_temp = [
        dtw.distance_fast(
            array.array('d', template), 
            array.array('d', a[t:t+template.shape[0]]),
            use_pruning=True,
            window=5
        ) 
        for t in range(
                0, 
                a.shape[0]-template.shape[0]
            )
        ]

    return np.asarray(corr_temp)