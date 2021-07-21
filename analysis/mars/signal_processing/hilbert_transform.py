from __future__ import division
import numpy as np
from numpy.fft import fftfreq

from .fft import fft, ifft

__authors__ = "Alex Bujan, Jesse Livezey"


__all__ = ['gaussian', 'hamming', 'hilbert_transform']


def gaussian(X, rate, center, sd):
    n_channels, time = X.shape
    freq = fftfreq(time, 1./rate)

    k = np.exp((-(np.abs(freq) - center)**2)/(2 * (sd**2)))

    return k / k.sum()


def hamming(X, rate, min_freq, max_freq):
    n_channels, time = X.shape
    freq = fftfreq(time, 1./rate)

    pos_in_window = np.logical_and(freq >= min_freq, freq <= max_freq)
    neg_in_window = np.logical_and(freq <= -min_freq, freq >= -max_freq)

    k = np.zeros(len(freq))
    window_size = np.count_nonzero(pos_in_window)
    window = np.hamming(window_size)
    k[pos_in_window] = window
    window_size = np.count_nonzero(neg_in_window)
    window = np.hamming(window_size)
    k[neg_in_window] = window

    return k / k.sum()


def hilbert_transform(X, rate, filters=None, normalize_filters=True):
    """
    Apply bandpass filtering with Hilbert transform using
    a prespecified set of filters.

    Parameters
    ----------
    X : ndarray (n_channels, n_time)
        Input data, dimensions
    rate : float
        Number of samples per second.
    filters : filter or list of filters (optional)
        One or more bandpass filters
    normalize_filters : bool
        If true, normalize each filter so that its entries sum to 1

    Returns
    -------
    Xc : array
        Bandpassed analytical signal (dtype: complex)
    """
    if not isinstance(filters, list):
        filters = [filters]
    time = X.shape[-1]
    freq = fftfreq(time, 1. / rate)

    # Heavyside filter
    h = np.zeros(len(freq))
    h[freq > 0] = 2.
    h[0] = 1.
    h = h[np.newaxis, :]

    Xh = np.zeros((len(filters),) + X.shape, dtype=np.complex)
    X_fft_h = fft(X) * h
    for ii, f in enumerate(filters):
        if f is None:
            Xh[ii] = ifft(X_fft_h)
        else:
            if normalize_filters:
                f = f / f.sum()
            Xh[ii] = ifft(X_fft_h * f)
    if Xh.shape[0] == 1:
        return Xh[0]

    return Xh
