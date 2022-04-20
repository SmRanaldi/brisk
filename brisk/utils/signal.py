import numpy as np

# ----- Functions -----

# Custom autocorrelation
def norm_autocorrelation(a):

    a = a.squeeze()

    corr_temp = [np.corrcoef(a[:-t], a[t:])[0,1] 
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

    corr_temp = [np.corrcoef(a[:-t], a[t:])[0,1] 
        for t in range(
                1,
                n
            )
        ]

    return np.asarray(corr_temp)

# Custom template matching
def norm_templatematching(a,template):

    a = a.squeeze()
    template = template.squeeze()

    corr_temp = [np.corrcoef(template, a[t:t+template.shape[0]])[0,1] 
        for t in range(
                0, 
                a.shape[0]-template.shape[0]
            )
        ]

    return np.asarray(corr_temp)