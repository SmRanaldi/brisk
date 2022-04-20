import numpy as np
from scipy.stats import iqr

# ----- Function -----

# Remove outliers
def remove_outliers(data_in, ths):

    ub = np.percentile(data_in, 100-ths)
    lb = np.percentile(data_in, ths)

    idx_good = (data_in<ub) & (data_in>lb)

    return data_in[idx_good]