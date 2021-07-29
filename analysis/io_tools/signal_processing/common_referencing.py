from __future__ import division
import numpy as np


__all__ = ['subtract_CAR',
           'subtract_common_median_reference']

def subtract_CAR(X, mean_frac=0.95, round_fcn=np.ceil):
    """
    Compute and subtract common average reference
    mean_frac - average is calculated over the middle X percent. This is X.
    """
    timepts, channels = X.shape
    nchs_excl = int(round_fcn(channels*(1-mean_frac)/2.0))
    avg = np.mean(np.sort(X)[:, nchs_excl:-nchs_excl], axis=1)

    return X - np.tile(avg, (channels, 1)).T


def subtract_common_median_reference(X, channel_axis=-2):
    """
    Compute and subtract common median reference
    for the entire grid.

    Parameters
    ----------
    X : ndarray (..., n_channels, n_time)
        Data to common median reference.

    Returns
    -------
    Xp : ndarray (..., n_channels, n_time)
        Common median referenced data.
    """

    median = np.nanmedian(X, axis=channel_axis, keepdims=True)
    Xp = X - median

    return Xp
