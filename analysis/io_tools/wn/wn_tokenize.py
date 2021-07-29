"""
Tokenize white noise stimulus data
"""

__author__ = 'Vyassa Baratham <vbaratham@lbl.gov>'

import numpy as np
from mars.io import NSENWB
from mars.signal_processing import smooth

def get_stim_onsets(nsenwb, mark_name):
    if 'Simulation' in nsenwb.block_name:
        raw_dset = nsenwb.read_raw('ECoG')
        end_time = raw_dset.data.shape[0] / raw_dset.rate
        return np.arange(0.5, end_time, 0.3)
    
    mark_dset = nsenwb.read_mark(mark_name)
    mark_fs = mark_dset.rate
    mark_offset = nsenwb.stim['mark_offset']
    stim_dur = nsenwb.stim['duration']
    stim_dur_samp = stim_dur*mark_fs

    mark_threshold = 0.25 if nsenwb.stim.get('mark_is_stim') else nsenwb.stim['mark_threshold']
    thresh_crossings = np.diff( (mark_dset.data[:] > mark_threshold).astype('int'), axis=0 )
    stim_onsets = np.where(thresh_crossings > 0.5)[0] + 1 # +1 b/c diff gets rid of 1st datapoint

    real_stim_onsets = [stim_onsets[0]]
    for stim_onset in stim_onsets[1:]:
        # Check that each stim onset is more than 2x the stimulus duration since the previous
        if stim_onset > real_stim_onsets[-1] + 2*stim_dur_samp:
            real_stim_onsets.append(stim_onset)

    if len(real_stim_onsets) != nsenwb.stim['nsamples']:
        print("WARNING: found {} stim onsets in block {}, but supposed to have {} samples".format(
            len(real_stim_onsets), nsenwb.block_name, nsenwb.stim['nsamples']))
        
    return (real_stim_onsets / mark_fs) + mark_offset

def get_end_time(nsenwb, mark_name):
    mark_dset = nsenwb.read_mark(mark_name)
    end_time = mark_dset.num_samples/mark_dset.rate
    return end_time

def already_tokenized(nsenwb):
    return nsenwb.nwb.trials and 'sb' in nsenwb.nwb.trials.colnames

def tokenize(nsenwb, mark_name='recorded_mark'):
    """
    Required: mark track

    Output: stim on/off as "wn"
            baseline as "baseline"
    """
    if already_tokenized(nsenwb):
        return
    
    stim_onsets = get_stim_onsets(nsenwb, mark_name)
    stim_dur = nsenwb.stim['duration']
    bl_start = nsenwb.stim['baseline_start']
    bl_end = nsenwb.stim['baseline_end']

    nsenwb.add_trial_column('sb', 'Stimulus (s) or baseline (b) period')

    # Add the pre-stimulus period to baseline
    # nsenwb.add_trial(start_time=0.0, stop_time=stim_onsets[0]-stim_dur, sb='b')

    for onset in stim_onsets:
        nsenwb.add_trial(start_time=onset, stop_time=onset+stim_dur, sb='s')
        if bl_start==bl_end:
            continue
        nsenwb.add_trial(start_time=onset+bl_start, stop_time=onset+bl_end, sb='b')

    # Add the period after the last stimulus to  baseline
    # rec_end_time = get_end_time(nsenwb,mark_name)
    # nsenwb.add_trial(start_time=stim_onsets[-1]+bl_end, stop_time=rec_end_time, sb='b')


if __name__ == '__main__':
    fn = '/data/ECoGData/R32_B6_tokenizetest.nwb'

    nsenwb = NSENWB.from_existing_nwb('R32_B6', fn)

    tokenize(nsenwb)
