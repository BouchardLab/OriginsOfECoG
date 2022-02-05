""" This script adds the trial table to a given NWB file by using
the recorded mark to find the start trial times
"""
from json import load
import h5py
import os
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO

FILE = '/path/to/R6_B10.nwb' #file to add trials table to
DEFAULT_FILE = '/path/to/R32_B7.nwb' #assume the same labels in this file's trials table
NUMBER_OF_TRIALS = 4800


def get_start_samples(x, thres=1):
    # take d/dt
    dx = np.diff(x, axis=0)
    dx = np.insert(dx, 0, 0)
    params = {'distance': 1000,
              'height': 0.5}
    # find peaks
    samples, _ = find_peaks(dx, **params)
    return samples


def get_trials_table(file):
    with NWBHDF5IO(file, 'r') as f:
        nwb = f.read()
        return nwb.trials.to_dataframe()


def convert_mark_to_trial_times(file):
    with NWBHDF5IO(file, 'r') as f:
        nwb = f.read()
        mark = nwb.stimulus['recorded_mark'].data
        rate = nwb.stimulus['recorded_mark'].rate
        samples = get_start_samples(mark)
        if len(samples) != NUMBER_OF_TRIALS:
            raise Exception(f'The number of trials {len(samples)} '
                            + 'is not what is expected')
        times = samples/rate
        end_time = len(mark)/rate
        return times, end_time


def make_trials_table(start_times, end_time, table):
    ntrials = len(table)

    # first baseline trial
    table['stop_time'][0] = start_times[0] - 0.05

    for tidx in range(1, ntrials-1):
        sidx = tidx // 2
        if tidx % 2:
            # stimulus trial
            table['start_time'][tidx] = start_times[sidx]
            table['stop_time'][tidx] = start_times[sidx] + 0.05
        else:
            # baseline trial
            table['start_time'][tidx] = table['stop_time'][tidx-1] + 0.05
            table['stop_time'][tidx] = table['start_time'][tidx] + 0.08

    # last baseline trial
    table['start_time'][-1] = table['stop_time'][tidx]
    table['stop_time'][-1] = end_time
    return table


def add_table(file, table):
    with NWBHDF5IO(file, 'a') as f:
        nwb = f.read()
        nwb.add_trial_column(
            name='sb', description='Stimulus (s) or baseline (b) period')
        nwb.add_trial_column(name='frq', description='Stimulus Frequency')
        nwb.add_trial_column(name='amp', description='Stimulus Amplitude')
        for i, row in table.iterrows():
            nwb.add_trial(**row)
        f.write(nwb)


def run():
    start_times, end_time = convert_mark_to_trial_times(FILE)
    table = get_trials_table(DEFAULT_FILE)
    new_table = make_trials_table(start_times, end_time, table)
    add_table(FILE, new_table)


if __name__ == '__main__':
    run()
