import os
import numpy as np
import matplotlib.pyplot as plt

from analysis import BasePlotter, PlotterArgParser

NORM = 'MEAN' # if mean does mean normalization, otherwise does ZSCORE

class ToneSpectrogram(BasePlotter):

    def get_t_extent(self):
        t = np.arange(-100, 150, 1/self.proc_dset.rate)
        extent = [-100, 150, 0, 1]
        return t, extent

    def draw_stim_bars(self):
        ymax, ymin = plt.ylim()
        plt.plot([0, 0], [ymin, ymax], linestyle='--', linewidth=0.5, color='k')
        plt.plot([50, 50], [ymin, ymax], linestyle='--', linewidth=0.5, color='k')
        plt.ylim([ymax, ymin])

    def draw_peak_bars(self):
        ymax, ymin = plt.ylim()
        center_samp = 8
        center_time = center_samp / self.proc_dset.rate
        t1, t2 = (center_time - .005) * 1000, (center_time + .005) * 1000
        plt.plot([t1, t1], [ymin, ymax], linestyle='--', linewidth=0.3, color='red')
        plt.plot([t2, t2], [ymin, ymax], linestyle='--', linewidth=0.3, color='red')
        plt.ylim([ymax, ymin])
    
    def plot_one(self, channel, **plot_kwargs):
        """
        Make one spectrogram and save it to file
        """
        ch_data = self.proc_dset.data[:, channel, :]
        rate = self.proc_dset.rate

        # Grab stim-on data, trial average if requested
        bfs = self.get_bfs()
        trials = self.nwb.trials
        if bfs is None:
            trial_idxs = trials['sb'][:] == 's'
        else:
            bf = bfs[channel]
            trial_idxs = np.logical_and(np.logical_and(trials['sb'][:] == 's', trials['frq'][:] == str(bf)), trials['amp'][:] == '7')
        times = zip(trials['start_time'][trial_idxs]-.1, trials['start_time'][trial_idxs]+.15)
        stim_periods = [(int(t[0]*self.proc_dset.rate), int(t[1]*self.proc_dset.rate)) for t in times] # start and stop times in samples
        if self.stim_i == 'avg':
            print("doing stim avg")
            n_stim_timepts = stim_periods[0][1] - stim_periods[0][0]
            stim_data = np.average(
                np.stack([ch_data[t[0]:t[0]+n_stim_timepts] for t in stim_periods]),
                axis=0
            )
        elif self.tstart is not None and self.tstop is not None:
            print("using tstart, tstop")
            istart, istop = int(self.tstart*rate), int(self.tstop*rate)
            stim_data = ch_data[istart:istop, :]
        else: # self.stim_i is an integer index
            print("doing stim {}".format(self.stim_i))
            tstart, tstop = stim_periods[self.stim_i]
            stim_data = ch_data[tstart:tstop, :]

        # Rescale to baseline mean/stdev
        bl_mean, bl_std = self.proc_bl_stats
        bl_mean, bl_std = bl_mean[channel, :], bl_std[channel, :]
        
        if NORM == 'MEAN':
            stim_data = stim_data / bl_mean # mean normalize
        else:
            stim_data = (stim_data - bl_mean) / bl_std # zscore    

        # Get band info for axis labels
        bands = self.proc_dset.bands['band_mean'][:]
        
        # Make plot
        t, extent = self.get_t_extent()
        ax = plt.gca()
        fig = plt.gcf()
        im = ax.imshow(stim_data.T, origin='lower', cmap='Greys', aspect='auto',
                       extent=extent, **plot_kwargs) # , vmin=0, vmax=5)

        
        plt.xlabel('Time (ms)')
        plt.ylabel("Frequency (Hz)")
        plt.xlim([-51, 100])
        ticks, ticklabels = [], []
        for i in range(0, len(bands), 8):
            ticks.append(float(i)/len(bands))
            ticklabels.append(int(bands[i]))
        ax.set_yticks(ticks)
        ax.set_yticklabels(ticklabels)
        # plt.colorbar(label="Stim/baseline ratio")
        fig.colorbar(im).set_label(label="Z-score Amplitude", size=8)
        fig.tight_layout()

        # Draw stim bars
        self.draw_stim_bars()

        # Draw peak bars
        self.draw_peak_bars()
        
        if not self.nosave:
            fn = 'spectrogram_{}_ch{:02d}_{}.{}'.format(
                self.device, channel, self.identifier, self.filetype
            )
            full_fn = os.path.join(self.outdir, fn)
            plt.savefig(full_fn)

        if self.show:
            plt.show()

        return im

    
if __name__ == '__main__':
    # TONE150 (not used)
    # rat = 'R72'
    # block = 'R72_B6'
    # rat = 'R73'
    # block = 'R73_B2'
    # rat = 'R70'
    # block = 'R70_B8'
    # rat = 'R75'
    # block = 'R75_B8'
    my_preproc = ['R70', 'R67']

    rat = 'simulation_ecp_layers'
    block = 'B01'
    
    nwbfile = '/path/to/simulation_ecp_layers.nwb'
    auxfile = None

    # Not used - all tone blocks are preprocessed by me 
    # proc_dset_name = 'Hilb_54bands' if rat in my_preproc else 'Wvlt_4to1200_54band_CAR0'

    plotter = ToneSpectrogram(nwbfile, '.',auxfile=auxfile)
    num_channels = 1
    for channel in range(num_channels):
        plt.figure(figsize=(5, 4))
        plotter.plot_one(channel, vmin=0, vmax=10)
        plt.savefig('plots/tone_spect_{}_ch{}.pdf'.format(block, channel))
        plt.savefig('plots/tone_spect_{}_ch{}.png'.format(block, channel))
        plt.close()
        print("done channel {}".format(channel))
    
