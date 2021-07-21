import glob
import os
import json

import numpy as np
from scipy.signal import butter, lfilter

def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def bandpass(data, fs, lowcut=20, highcut=5000, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_highpass(lowcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    b, a = butter(order, low, btype='highpass')
    return b, a

def highpass(data, fs, lowcut, order=5):
    b, a = butter_highpass(lowcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def log_spaced_cfs(fmin, fmax, nbin=6):
    """
    Center frequencies that are uniform in log space
    """
    noct = np.ceil(np.log2(fmax/fmin))
    return fmin * 2**(np.arange(noct*nbin)/nbin)

wavelet_cfs = log_spaced_cfs(2.6308, 1200.0)

CCROOT = '/Users/vbaratham/src/cortical-column'

def find_layer_ei_ecp_file(jobnum):
    output_dir = os.path.join(CCROOT, 'runs', jobnum, '1', 'output')
    ecp_files = glob.glob(os.path.join(output_dir, 'ecp*layer_ei*.nwb'))
    if len(ecp_files) == 0:
        raise ValueError('No layer_ei ECP file found')
    elif len(ecp_files) == 1:
        return ecp_files[0]
    else:
        log.info(
            'Found multiple layer_ei ECP files: \n{}\n'.format('\n'.join(ecp_files)) + 
            '\nUsing {}\n'.format(ecp_files[-1])
        )
        return ecp_files[-1]

def find_slice_ecp_file(jobnum, thickness=100):
    output_dir = os.path.join(CCROOT, 'runs', jobnum, '1', 'output')
    ecp_files = glob.glob(os.path.join(output_dir, 'ecp*{}um*.nwb'.format(thickness)))
    if len(ecp_files) == 0:
        raise ValueError('No 100um slice ECP file found')
    elif len(ecp_files) == 1:
        return ecp_files[0]
    else:
        log.info(
            'Found multiple 100um slice ECP files: \n{}\n'.format('\n'.join(ecp_files)) + 
            '\nUsing {}\n'.format(ecp_files[-1])
        )
        return ecp_files[-1]

def get_layer_slice_counts(jobnum, thickness=100):
    fn = os.path.join(CCROOT, 'runs', jobnum, '1', 'output', 'layer_slice_counts.json')
    with open(fn, 'r') as infile:
        orig = json.load(infile)
        counts = {
            int(layer): {int(slice_i): count for slice_i, count in slice_counts.items()}
            for layer, slice_counts in orig.items()
        }
        if thickness == 100:
            return counts
        elif thickness == 200:
            def convert(layercounts):
                return {slice_i: layercounts[slice_i*2] + layercounts.get(slice_i*2 + 1, 0)
                        for slice_i in range(11)}
            for layer in counts.keys():
                counts[layer] = convert(counts[layer])
            return counts
        else:
            raise ValueError("Can only do 100 or 200um slices")

numerals = {1: 'I', 2: 'II', 3: 'III', 4: 'IV', 5: 'V', 6: 'VI'}
