"""
Plot the avg raw response to the BF on a given electrode
"""

import numpy as np
import matplotlib.pyplot as plt
import h5py

from analysis import BasePlotter, PlotterArgParser
from utils import bandpass, highpass

class ToneAvgECP(BasePlotter):
    def plot(self, channel):
        rate = self.raw_dset.rate
        bf = self.get_bfs()[channel]
        trials = self.nwb.trials
        trial_idxs = np.logical_and(np.logical_and(trials['sb'][:] == 's', trials['frq'][:] == str(bf)), trials['amp'][:] == '7')
        # trial_idxs = trials['sb'][:] == 's'
        start_times = trials['start_time'][trial_idxs]-0.05
        window_len_samp = int(0.15 * rate)
        stim_periods = [(int(t*rate), int(t*rate) + window_len_samp) for t in start_times]
        ch_data = self.raw_dset.data[:, channel] * 1000
        # ch_data = bandpass(ch_data, rate, 2, 3000)
        # ch_data = highpass(ch_data, rate, 800)
        all_stim_data = [ch_data[istart:istop] for istart, istop in stim_periods]
        stim_data = np.stack(all_stim_data)
        avg_waveform = np.average(stim_data, axis=0)
        std_waveform = np.std(stim_data, axis=0)
        t = np.linspace(-50, 100, len(avg_waveform))

        # DEBUG
        # for i in range(len(start_times)):
        #     plt.plot(t, stim_data[i, :], color='k', linewidth=0.3, alpha=0.3)
        # END DEBUG
        plt.fill_between(t, avg_waveform+std_waveform, avg_waveform-std_waveform, color='grey')
        plt.plot(t, avg_waveform, color='black')

        # Draw stim bars
        ymin, ymax = plt.ylim()
        plt.plot([0, 0], [ymin, ymax], linestyle='--', linewidth=0.5, color='k')
        plt.plot([50, 50], [ymin, ymax], linestyle='--', linewidth=0.5, color='k')
        
        # Draw peak bars
        center_samp = 10
        center_time = center_samp / self.proc_dset.rate
        t1, t2 = (center_time - .005) * 1000, (center_time + .005) * 1000
        plt.plot([t1, t1], [ymin, ymax], linewidth=0.3, color='red')
        plt.plot([t2, t2], [ymin, ymax], linewidth=0.3, color='red')

        plt.xlim([-50, 100])
        plt.ylim([ymin, ymax])
        
        plt.xlabel("Time (ms)")
        plt.ylabel("Voltage (mV)")
        plt.tight_layout()
        

if __name__ == '__main__':
    # TONE150 (not used)
    # rat = 'R72'
    # block = 'R72_B6'
    # rat = 'R73'
    # block = 'R73_B2'
    rat = 'R70'
    block = 'R70_B8'
    # rat = 'R75'
    # block = 'R75_B8'
    my_preproc = ['R70', 'R67']

    rat = 'R32'
    block = 'R32_B7'
    
    nwbfile = '/data/{}/{}.nwb'.format(rat, block)
    auxfile = '/data/{}/{}_aux.h5'.format(rat, block)

    # Not used - all tone blocks are preprocessed by me 
    # proc_dset_name = 'Hilb_54bands' if rat in my_preproc else 'Wvlt_4to1200_54band_CAR0'

    for channel in range(128):
        plotter = ToneAvgECP(nwbfile, '.', no_baseline_stats=True, auxfile=auxfile)
        plt.figure(figsize=(4.2, 4))
        plotter.plot(channel)
        plt.savefig('plots/tone_raw_{}_ch{}.pdf'.format(block, channel))
        plt.close()
        print("done channel {}".format(channel))
    
