# Taken from https://scipy-cookbook.readthedocs.io/items/ButterworthBandpass.html

from __future__ import print_function

from scipy.signal import butter, filtfilt, sosfilt, sosfiltfilt

__all__ = ['butter_bandpass']



def butter_bandpass(data, fs, lowcut, highcut, order=8, filter_fcn=sosfiltfilt):
    nyq = 0.5 * fs
    low = lowcut / nyq

    if nyq > highcut:
        high = highcut / nyq
        sos = butter(order, [low, high], btype='bandpass', output='sos')
    else:
        print("WARNING: Requested filter abovve nyquist frequency")
        sos = butter(order, [low], btype='highpass', output='sos')
    
    y = filter_fcn(sos, data)
    return y