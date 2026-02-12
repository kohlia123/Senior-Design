import numpy as np
import scipy.stats as sp_stats

def get_ptp_amplitude(epochs):
    return np.ptp(epochs, axis=1)
