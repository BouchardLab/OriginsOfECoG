"""
Stores the stimuli used in the simulation. Names should agree with block_directory.py in mars
"""

import itertools
import csv
from collections import defaultdict

from scipy.signal import hanning
import numpy as np

# What this script refers to as "probabilities" or "prob"s seems like it's really spike rates


# GENERAL SPIKE-RELATED UTILS

def spiketimes(probs, tmin=0.0, fs=0.001):
    """
    Generate spike times according to Poisson            
    Put spikes at random position in the timebin they occur

    probs = 1d array whose entries are the probability of an external
            cell firing at each timestep
    """
    t = np.arange(tmin, len(probs)*fs, fs)
    num_spikes = np.random.poisson(lam=probs*fs)
    # TODO: this assumes only one spike per timebin
    spike_times = t[np.where(num_spikes > 0)]
    spike_times += np.random.uniform(high=fs, size=len(spike_times)).astype(np.float16)
    spike_times /= fs

    # if no spikes, produce one spike after the simulation ends b/c bmtk chokes on no spikes
    return spike_times if len(spike_times) != 0 else [len(t)/fs]
    
    

def save_csv(probs, gids, outfile_name):
    with open(outfile_name, 'w') as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=' ')
        csv_writer.writerow(['gid', 'spike-times'])
        for gid in gids:
            spikes = map(str, spiketimes(probs))
            csv_writer.writerow([gid, ','.join(spikes)])


# UTILS RELATED TO TIMESERIES GENERATION
# These functions return tuples of 1D numpy arrays that can be stitched
# together with the '+' operator

def hanning_ramped(length, min_rate, max_rate, ramp=15):
    h = (max_rate - min_rate) * hanning(2*ramp)
    const = np.ones(length - 2*ramp) * (max_rate - min_rate)
    return min_rate + np.concatenate([h[:ramp], const, h[ramp:]])
    
def wn_stim(stim_prob, quiet_prob, stim_dur=100, quiet_dur=900):
    stim = hanning_ramped(stim_dur, quiet_prob, stim_prob)
    quiet = np.ones(quiet_dur)*quiet_prob
    return stim, quiet

def full_wn(stim_prob, quiet_prob, n_stims, intro_len=500, stim_dur=100, quiet_dur=900):
    intro = quiet_prob * np.ones(intro_len)
    return np.concatenate(
        (intro,) + n_stims * wn_stim(stim_prob, quiet_prob, stim_dur=stim_dur, quiet_dur=quiet_dur)
    )

# I think "length" is milliseconds... sampling freq fs is specified as argument to spiketimes()
def baseline(prob, length):
    return prob * np.ones(length)

# TIMESERIES OF FIRING PROBABILITIES

stimulus = {}
stimulus['wn_simulation_v0'] = {
    'bkg': full_wn(15.0, 3.0, 4),
    'thalamus': full_wn(15.0, 1.0, 4),
}

wn_mark_track = np.zeros(shape=(45000,), dtype=np.float32)
wn_mark_track[5000:6000] = 1.0
wn_mark_track[15000:16000] = 1.0
wn_mark_track[25000:26000] = 1.0
wn_mark_track[35000:36000] = 1.0

# TODO: When there are different mark tracks, this needs to become a real dict
mark = defaultdict(lambda: wn_mark_track)

# stimulus['wn_simulation_ampl_v0_0'] = {
#     'bkg': full_wn(5.0, 3.0, 4),
#     'thalamus': full_wn(5.0, 1.0, 4),
# }
# stimulus['wn_simulation_ampl_v0_1'] = {
#     'bkg': full_wn(10.0, 3.0, 4),
#     'thalamus': full_wn(10.0, 1.0, 4),
# }
# stimulus['wn_simulation_ampl_v0_2'] = {
#     'bkg': full_wn(12.5, 3.0, 4),
#     'thalamus': full_wn(12.5, 1.0, 4),
# }
# stimulus['wn_simulation_ampl_v0_3'] = {
#     'bkg': full_wn(15.0, 3.0, 4),
#     'thalamus': full_wn(15.0, 1.0, 4),
# }
# stimulus['wn_simulation_ampl_v0_4'] = {
#     'bkg': full_wn(17.5, 3.0, 4),
#     'thalamus': full_wn(17.5, 1.0, 4),
# }
# stimulus['wn_simulation_ampl_v0_5'] = {
#     'bkg': full_wn(20.0, 3.0, 4),
#     'thalamus': full_wn(20.0, 1.0, 4),
# }

stimulus['wn_simulation_v1'] = {
    'bkg': baseline(5.0, 10000),
    'thalamus': full_wn(15.0, 1.0, 5),
}

stimulus['wn_simulation_v2'] = {
    'bkg': baseline(5.0, 60000),
    'thalamus': full_wn(25.0, 1.0, 60),
}

stimulus['wn_simulation_v3'] = {
    'bkg': baseline(5.0, 60000),
    'thalamus': full_wn(35.0, 1.0, 60),
}

stimulus['wn_simulation_v4'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(35.0, 1.0, 60),
}

stimulus['short_wn_simulation_v4'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(35.0, 1.0, 60, stim_dur=50, quiet_dur=950),
}

stimulus['wn_simulation_v5'] = {
    'bkg': baseline(7.0, 60000),
    'thalamus': full_wn(35.0, 1.0, 60),
}

stimulus['wn_simulation_v6'] = {
    'bkg': baseline(7.0, 60000),
    'thalamus': full_wn(25.0, 1.0, 60),
}

stimulus['long_wn_simulation_v6'] = {
    'bkg': baseline(7.0, 60000),
    'thalamus': full_wn(25.0, 1.0, 60, stim_dur=400, quiet_dur=600),
}

stimulus['long_wn_simulation_v5'] = {
    'bkg': baseline(7.0, 60000),
    'thalamus': full_wn(35.0, 1.0, 60, stim_dur=400, quiet_dur=600),
}

# Ampl vary v1
stimulus['wn_simulation_ampl_v1_05'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(5.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_04'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(10.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_03'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(15.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_02'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(20.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_01'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(23.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_0'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(26.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_1'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(29.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_2'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(32.0, 1.0, 60),
}
# stimulus['wn_simulation_ampl_v1_3'] = { # NB don't rly need to run this one b/c already have 60s 
#     'bkg': baseline(10.0, 60000),
#     'thalamus': full_wn(35.0, 1.0, 60),
# }
stimulus['wn_simulation_ampl_v1_4'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(38.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_5'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(41.0, 1.0, 60),
}
stimulus['wn_simulation_ampl_v1_6'] = {
    'bkg': baseline(10.0, 60000),
    'thalamus': full_wn(44.0, 1.0, 60),
}




                           
if __name__ == '__main__':
    NTHAL = 10000
    # save_csv(stimulus['wn_simulation_v3']['bkg'], range(1, 5000), 'test_spikes_bkg.csv')
    save_csv(stimulus['wn_simulation_v4']['thalamus'], range(1, NTHAL), 'test_spikes_thal.csv')
    import matplotlib.pyplot as plt
    import pandas as pd

    # plot the stim
    tstart = 1000
    tstop = 2000
    def plot_spike_rate(axs, spikes, num_nodes, name_for_label, color=None, binsize=1.0):
        # binsize in ms
        bins = np.arange(tstart, tstop, binsize)
        num_spikes, _ = np.histogram(spikes, bins=bins)
        p_spike = np.float64(num_spikes)/num_nodes
        axs.plot(bins[:-1], p_spike, color=color, label=name_for_label)
        axs.set_xlabel('time (s)')
        axs.set_xlim([tstart, tstop])
        # TODO: Set max appropriately for multi-plots
        bin_100ms = int(100. / binsize)
        # ignore first 100ms for y range
        axs.set_ylim([0, max(p_spike[bin_100ms:]) * 1.1])
        # axs.set_title("{} spike prob. ({}ms bins)".format(name_for_label, int(binsize*1000)))

    plt.figure(figsize=(6, 1))
    bkg_rate_ax = plt.gca()
    thal_spikes_df = pd.read_csv('test_spikes_thal.csv', sep=' ')
    thal_spike_times = [
        float(t) for spike_times_str in thal_spikes_df['spike-times']
        for t in spike_times_str.split(',')
    ]
    num_thal_nodes = len(thal_spikes_df['spike-times'])
    plot_spike_rate(bkg_rate_ax, thal_spike_times, num_thal_nodes, "Thalamus")
    plt.show()
    

    # plot the stim on periods and mark track
    # plt.plot(stimulus['wn_simulation_v1']['thalamus'], label='stim')
    # plt.plot(mark['wn_simulation_v1'], label='mark')
    # plt.legend()
    # plt.show()
    
