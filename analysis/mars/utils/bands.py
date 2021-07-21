"""
Frequency band information for different types of data processing.
"""
import os

from scipy.io import loadmat
import numpy as np

class DataFormat(object):
    def write_preprocessed(self):
        raise NotImplementedError
    def read_preprocessed(self):
        raise NotImplementedError

def log_spaced_cfs(fmin, fmax, nbin=6):
    """
    Center frequencies that are uniform in log space
    """
    noct = np.ceil(np.log2(fmax/fmin))
    return fmin * 2**(np.arange(noct*nbin)/nbin)

def const_Q_sds(cfs, Q=8):
    return cfs/Q

def scaled_sqrt_sds(cfs, scale=0.39):
    # equivalent to:
    # return scale * np.sqrt(cfs)
    return 10 ** ( np.log10(scale) + .5 * (np.log10(cfs))) * np.sqrt(2.)


# Chang lab frequencies
fq_min = 4.0749286538265
fq_max = 200.
scale = 7.
cfs = 2 ** (np.arange(np.log2(fq_min) * scale, np.log2(fq_max) * scale) / scale)
sds = scaled_sqrt_sds(cfs)
chang_lab = {'fq_min': fq_min,
             'fq_max': fq_max,
             'scale': scale,
             'cfs': cfs,
             'sds': sds,
             'block_path': '{}_Hilb.h5'}

# Standard neuro bands
bands = ['theta', 'alpha', 'beta', 'high beta', 'gamma', 'high gamma', 'ultra high gamma', 'multiunit activity range']
abrev = ['T','A','B','HB','G','HG','UHG','MUAR']
min_freqs = [4., 9., 15., 21., 30., 70.,180.,500]
max_freqs = [8., 14., 20., 29., 59., 170.,450.,1200]
HG_freq = 200.
neuro = {'bands': bands,
         'abrev': abrev,
         'min_freqs': min_freqs,
         'max_freqs': max_freqs,
         'HG_freq': HG_freq,
         'block_path': '{}_neuro_Hilb.h5'}

def frequency_range(abrev):
    frq_ind = neuro['abrev'].index(abrev)
    return [neuro['min_freqs'][frq_ind],neuro['max_freqs'][frq_ind]]

# Wavelet 4-1200hz 54 bands
# which actually start at 2.6308 hz
wavelet_cfs = log_spaced_cfs(2.6308, 1200.0)
wavelet_sds = const_Q_sds(wavelet_cfs)
wavelet = {'cfs': wavelet_cfs, 'sds': wavelet_sds}

if __name__ == '__main__':
    # with open(os.path.join(os.path.dirname(__file__), 'cfs.4_1200.54Wvl.mat'), 'r') as matfile:
    #     mat = loadmat(matfile)
    #     cfs = np.squeeze(mat['cfs'])
    #     sds = 10 ** ( np.log10(.39) + .5 * (np.log10(cfs)))
    #     sds = np.array(sds)
    #     print cfs
    pass
