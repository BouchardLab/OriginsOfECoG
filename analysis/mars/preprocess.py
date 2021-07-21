#!/usr/bin/env python
from __future__ import print_function

import argparse
import h5py
import time
import sys
import os
import logging

import numpy as np

from hdmf.data_utils import AbstractDataChunkIterator, DataChunk

try:
    from tqdm import tqdm
except:
    def tqdm(x, *args, **kwargs):
        return x

from mars.signal_processing import resample
from mars.signal_processing import subtract_CAR
from mars.signal_processing import linenoise_notch
from mars.signal_processing import hilbert_transform
from mars.signal_processing import gaussian
from mars.utils import bands
from mars.wn import mua_signal, mua_rate
from mars.io import NSENWB

log = logging.getLogger('mars_preprocess')
log.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
ch = logging.StreamHandler(stream=sys.stdout)
ch.setFormatter(formatter)
ch.setLevel(logging.DEBUG)
log.addHandler(ch)


def _get_cfs(_cfs):
    # Default: use precomputed wavelet cfs
    if _cfs is None:
        return bands.wavelet['cfs']
        
    # Case 1: use precomputed cfs for Chang Lab or wavelet
    if _cfs[0].lower() in ('chang', 'changlab'):
        return bands.chang_lab['cfs']
    elif _cfs[0].lower() in ('wavelet', 'wave', 'wvlt'):
        return bands.wavelet['cfs']

    # Case 2: call to a function in bands.py
    elif _cfs[0].lower() in ('log', 'logspace', 'logspaced'):
        return bands.log_spaced_cfs(*[float(arg) for arg in _cfs[1:]])

    # Case 3: list of numbers
    else:
        return np.array([float(cf) for cf in _cfs])


def _get_sds(cfs, _sds):
    # Default: use precomputed wavelet cfs
    if _sds is None:
        return bands.wavelet['sds']
    
    # Case 1: use precomputed sds for Chang Lab or wavelet
    if _sds[0].lower() in ('chang', 'changlab'):
        return bands.chang_lab['sds']
    elif _sds[0].lower() in ('wavelet', 'wave', 'wvlt'):
        return bands.wavelet['sds']

    # Case 2: Call to a function in bands.py
    elif _sds[0] in ('q', 'constq', 'cqt'):
        return bands.const_Q_sds(cfs, *[float(arg) for arg in _sds[1:]])
    elif _sds[0] in ('sqrt', 'scaledsqrt'):
        return bands.scaled_sqrt_sds(cfs, *[float(arg) for arg in _sds[1:]])

    # Case 3: list of numbers
    else:
        return np.array([float(sd) for sd in _sds])


def __resample(X, new_freq, old_freq, axis=-1):
    assert new_freq < old_freq
    n_timepts, n_ch = X.shape
    if not np.allclose(new_freq, old_freq):
        for ch in range(n_ch):
            ch_X = X[:, ch]
            yield resample(ch_X, new_freq, old_freq, axis=axis)
            log.info("resampled channel {} of {}".format(ch+1, n_ch))

def _resample(X, new_freq, old_freq, axis=-1):
    return np.stack(__resample(X, new_freq, old_freq, axis=-1)).T

# def _resample_iterator(X, new_freq, old_freq, axis=-1):
#     assert new_freq < old_freq
#     n_timepts, n_ch = X.shape
#     if not np.allclose(new_freq, old_freq):
#         for ch in range(n_ch):
#             ch_X = X[:, ch]
#             yield DataChunk(data=resample(ch_x, new_freq, old_freq, axis=axis),
#                             selection=np.s_[:, ch])

class MyDataChunkIterator(AbstractDataChunkIterator):
    def __init__(self, it, dtype, n_ch, n_bands, approx_timepts=200000):
        self.it = it
        self._dtype = dtype
        self._maxshape = (None, n_ch, n_bands)
        self._approx_timepts = approx_timepts
        self._chunkshape = self._approx_timepts, 1, 1
        self._n_bands = n_bands
        self._i = 0

    def __iter__(self):
        return self

    def __next__(self):
        data = next(self.it)
        ch = self._i / self._n_bands
        band = self._i % self._n_bands
        self._i += 1
        
        return DataChunk(data=data, selection=np.s_[:data.shape[0], ch, band])

    next = __next__

    @property
    def dtype(self):
        return self._dtype

    @property
    def maxshape(self):
        return self._maxshape

    def recommended_chunk_shape(self):
        return self._chunkshape

    def recommended_data_shape(self):
        return (self._approx_timepts, self.maxshape[1], self.maxshape[2])

def _notch_filter(X, rate):
    return linenoise_notch(X, rate)

def _subtract_CAR(X):
    subtract_CAR(X)
    return X

def _hilbert_onech(ch_X, rate, cfs, sds, final_resample):
    """
    Hilbert transform one channel, band by band
    First resample already performed. Rate == first_resample
    """
    for i, (cf, sd) in enumerate(zip(cfs, sds)):
        kernel = gaussian(ch_X, rate, cf, sd)
        transform = np.abs(hilbert_transform(ch_X, rate, kernel))
        final_data = resample(transform, final_resample, rate)
        log.info("done band {}".format(i))
        yield np.squeeze(final_data)
        # yield DataChunk(data=final_data, selection=np.s_[:, ch, i])

def _hilbert_iterator(X, rate, cfs, sds, first_resample, final_resample):
    n_timepts, n_ch = X.shape
    for ch in range(n_ch):
        # ch_X = resample(np.atleast_2d(X[:, ch]).T, first_resample, rate).T
        ch_X = np.atleast_2d(resample(np.squeeze(X[:, ch]), first_resample, rate)) # HACK
        yield np.stack(_hilbert_onech(ch_X, first_resample, cfs, sds, final_resample), axis=-1)
        log.info("done Hilbert on channel {} of {}".format(ch+1, n_ch))

def _hilbert_one_by_one(X, rate, cfs, sds, first_resample, final_resample):
    n_timepts, n_ch = X.shape
    for ch in range(n_ch):
        ch_X = _resample(np.atleast_2d(X[:, ch]).T, first_resample, rate).T
        for one_band_done in _hilbert_onech(ch_X, first_resample, cfs, sds, final_resample):
            yield one_band_done
        log.info("done Hilbert on channel {} of {}".format(ch+1, n_ch))

def _hilbert_transform(X, rate, cfs, sds, first_resample, final_resample):
    n_timepts, n_ch = X.shape
    approx_timepts_final = float(n_timepts) * final_resample / rate
    # final = None #np.zeros(shape=(n_timepts, n_ch, len(cfs)), dtype=np.float32)
    # for datachunk in _hilbert_iterator(X, rate, cfs, sds, final_resample=final_resample):
    #     import ipdb; ipdb.set_trace()
    #     if final is None:
    #         pass
    #     final[datachunk.selection] = datachunk.data


    # return np.stack(_hilbert_iterator(X, rate, cfs, sds, first_resample, final_resample), axis=1)
    it = _hilbert_one_by_one(X, rate, cfs, sds, first_resample, final_resample)
    return MyDataChunkIterator(it, X.dtype, n_ch, 54, approx_timepts=approx_timepts_final)
    # return DataChunkIterator(data=it, maxshape=(None, n_ch, 54), dtype=np.dtype(float))

    # x = np.stack(_hilbert_iterator(X, rate, cfs, sds, first_resample, final_resample), axis=-1)
    # return x

def _mua(X_raw, fs, lowcut, highcut, order):
    mua = mua_signal(X_raw[:], fs, lowcut, highcut, order)
    return mua, mua_rate(mua, fs)

def _write_data(nsenwb, outfile, device, rate, raw_rate, X, Y, mua, mua_rate, decomp_type, cfs, sds, postfix):
    def postfixed(s):
        return '{}_{}'.format(s, postfix) if postfix else s

    nsenwb.add_proc(X, device, postfixed(device), rate)
    nsenwb.add_proc(Y, device, postfixed('Hilb_54bands'), rate, cfs=cfs, sds=sds)
    nsenwb.add_proc(mua, device, postfixed('tMUA'), raw_rate)
    nsenwb.add_proc(mua_rate, device, postfixed('tMUA_rate'), rate)

    if outfile and os.path.exists(outfile):
        os.remove(outfile)
    nsenwb.write(save_path=outfile)
    nsenwb.close()



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing ecog data from nwb.')
    parser.add_argument('datafile', type=str, help="Input .nwb file")
    parser.add_argument('--outfile', type=str, default=None,
                        help="Output file. Default = write to input file")
    parser.add_argument('--block', type=str, required=True)
    parser.add_argument('--device', '--device-name', type=str, default='ECoG')
    parser.add_argument('--acquisition', '--acq', type=str, default='Raw')
    parser.add_argument('--first-resample', type=float, default=None,
                        help='Resample data to this rate before processing. ' +
                        'Omit to skip resampling before processing.')
    parser.add_argument('--final-resample', type=float, default=400.,
                        help='Resample data to this rate after processing. ' +
                        'Omit to skip resampling after processing.')
    parser.add_argument('--cfs', type=str, nargs='+', default=None,
                        help="Center frequency of the Gaussian filter. " +
"""
Must be one of the following:
1.) The name of a precomputed set of filters (choices: 'changlab', 'wavelet')
2.) The name of a function (choices: 'logspaced') followed by
    args to that function (usually fmin, fmax, but see bands.py)
3.) A list of floats specifying the center frequencies
Default = precomputed wavelet 4-1200hz cfs
eg. to use the precomputed Chang lab filters, use `--cfs changlab`
eg. to use log spaced frequencies from 10-200hz, use `--cfs logspaced 10 200`
eg. to use your own list of center frequencies, use `--cfs 2 4 8 16 [...]`
"""
    )
    parser.add_argument('--sds', type=str, nargs='+', default=None,
                        help="Standard deviation of the Gaussian filter. " +
"""
Must be one of the following:
1.) The name of a precomputed set of filters (choices: 'changlab', 'wavelet')
2.) The name of a function (choices: 'constq', 'scaledsqrt') followed by
    args to that function (q-factor, or scale, etc. See bands.py)
3.) A list of floats specifying the center frequencies
Default = precomputed wavelet 4-1200hz sds
eg. to use the precomputed Chang lab filters, use `--sds changlab`
eg. to use constant Q filters with Q=4, use `--sds constq 4`
eg. to use constant filter widths of 10hz, use `--sds 10 10 10 10 [...]`
"""
    )
    parser.add_argument('--no-notch', default=False, action='store_true',
                        help="Do not perform notch filtering")
    parser.add_argument('--no-car', default=False, action='store_true',
                        help="Do not perform common avg reference subtraction")
    parser.add_argument('--decomp-type', type=str, default='hilbert',
                        choices=['hilbert', 'hil'],
                        help="frequency decomposition method")
    parser.add_argument('--no-magnitude', default=False, action='store_true',
                        help="Do not take the magnitude of the frequency decomp")
    parser.add_argument('--no-mua', default=False, action='store_true',
                        help="Do not compute MUA")
    parser.add_argument('--mua-range', type=float, nargs=2, default=(500, 5000),
                        help="critical frequencies for MUA bandpass filter")
    parser.add_argument('--mua-order', type=int, default=8,
                        help="order for butterworth bandpass filter for MUA")
    parser.add_argument('--dset-postfix', default=None, required=False,
                        help="String to append to nwb dset names")
    # parser.add_argument('--luigi', action='store_true', required=False, default=False,
    #                     help="use luigi logger, which doesn't go to console")
    parser.add_argument('--logfile', type=str, default=None, required=False)

    args = parser.parse_args()

    if args.logfile:
        fh = logging.FileHandler(args.logfile)
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        fh.setFormatter(formatter)
        fh.setLevel(logging.DEBUG)
        log.addHandler(fh)

    # PARSE ARGS
    if args.decomp_type in ('hilbert', 'hil'):
        decomp_type = 'hilbert'
    else:
        raise NotImplementedError()

    cfs = _get_cfs(args.cfs)
    sds = _get_sds(cfs, args.sds)

    if args.outfile and (args.outfile != args.infile):
        raise NotImplementedError("Cannot write to different outfile until pynwb issue #668 is addressed")

    # LOAD DATA
    start = time.time()
    nsenwb = NSENWB.from_existing_nwb(args.block, args.datafile)
    raw_dset = nsenwb.read_raw(args.device, acq_name=args.acquisition)
    raw_freq = raw_dset.rate

    X = raw_dset.data

    log.info("Time to load: {} sec".format(time.time()-start))

    # TODO: remove bad electrodes. Or maybe keep them in the file but mark them bad?

    # CAR REMOVAL
    if not args.no_car:
        start = time.time()
        X = _subtract_CAR(X)
        log.info("Time to subtract CAR: {} sec".format(time.time()-start))

    # NOTCH FILTER
    if not args.no_notch:
        start = time.time()
        X = _notch_filter(X, raw_dset.rate)
        log.info("Time to notch filter: {} sec".format(time.time()-start))

    # MUA RATE
    if not args.no_mua:
        start = time.time()
        mua, mua_rate = _mua(X, raw_freq, args.mua_range[0], args.mua_range[1], args.mua_order)
        log.info("Time to compute MUA rate: {} sec".format(time.time()-start))
    else:
        mua, mua_rate = None, None

    # FREQUENCY DECOMPOSITION
    if decomp_type == 'hilbert':
        start = time.time()
        Y = _hilbert_transform(X, raw_freq, cfs, sds, args.first_resample, args.final_resample)
        log.info("Time to Hilbert transform: {} sec".format(time.time()-start))
    else:
        raise NotImplementedError()

    # FINAL RESAMPLE
    if args.final_resample:
        start = time.time()
        # Y = _resample(Y, args.final_resample, rate, axis=0) # Done in Hilbert
        # X = _resample(X, args.final_resample, rate, axis=0) # TODO: uncomment
        if mua_rate is not None:
            mua_rate = _resample(mua_rate, args.final_resample, raw_freq, axis=0)
        log.info("Time to resample: {} sec".format(time.time()-start))
 
    # TOKENIZE
    # TODO: store tokenizer class in block directory and load it here.
    # For now, just assume white noise tokenizer, which may become some sort of default
    import mars.tokenizers
    try:
        tokenizer_name = nsenwb.stim['tokenizer']
        tokenize = getattr(mars.tokenizers, tokenizer_name)
    except KeyError:
        log.error('no tokenizer specified in block directory')
    except AttributeError:
        log.error('tokenizer {} not found'.format(tokenizer_name))
    else:
        tokenize(nsenwb)

    # WRITE DATA
    start = time.time()
    _write_data(nsenwb, args.outfile, args.device, args.final_resample, raw_freq, X, Y, mua, mua_rate,
                decomp_type, cfs, sds, args.dset_postfix)
    log.info("Time to write {}: {} sec".format(args.datafile, time.time()-start))
