import numpy as np

# ----- Functions -----

def norm_autocorrelation(a):

    a = a.squeeze()

    corr_temp = [np.corrcoef(a[:-t], a[t:])[0,1] 
        for t in range(
                1, 
                a.shape[0]-1
            )
        ]

    return np.asarray(corr_temp)


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