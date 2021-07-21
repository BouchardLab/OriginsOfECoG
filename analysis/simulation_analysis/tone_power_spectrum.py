"""Generate power spectrum from tone150 experimental block. Overlays
channels. Uses best frequency at each channel only.  """
import os

import numpy as np
import h5py
import matplotlib.pyplot as plt

from power_spectrum import PowerSpectrum
from utils import wavelet_cfs

def bf_tone(plotter, auxfile, hg_min=65, hg_max=170):
    """
    Compute and store the baseline stats and best frequencies on each channel
    """
    nwb = plotter.nwb
    bl_mu, bl_std = plotter.proc_bl_stats

    # Grab list of all frequencies presented
    all_stim_freq = [int(x) for x in np.unique(nwb.trials['frq'][:])]
    n_stim_freq = len(all_stim_freq)

    # Grab Z-scored high gamma data for each stim freq individually,
    # compute max within each trial, and average across trials
    proc_dset = plotter.proc_dset
    _, n_ch, _ = proc_dset.data.shape
    trials = plotter.nwb.trials
    f_idx = np.logical_and(wavelet_cfs > hg_min, wavelet_cfs < hg_max)
    
    freq_maxes = np.empty(shape=(n_stim_freq, n_ch)) # trial-avg of max Hg amplitude per ch
    for stim_freq_i, stim_freq in enumerate(all_stim_freq):
        trial_idxs = np.logical_and(trials['sb'][:] == 's', trials['frq'][:] == str(stim_freq))
        times = zip(trials['start_time'][trial_idxs], trials['stop_time'][trial_idxs])
        time_idxs = [(int(t[0]*proc_dset.rate), int(t[1]*proc_dset.rate)) for t in times]
        ch_maxes = np.empty(shape=(len(time_idxs), n_ch))
        for trial_i, (istart, istop) in enumerate(time_idxs):
            trial_data = proc_dset.data[istart:istop, :, f_idx]
            trial_data = (trial_data - bl_mu[:, f_idx]) / bl_std[:, f_idx]
            trial_data = np.average(trial_data, axis=-1)
            ch_maxes[trial_i, :] = np.max(trial_data, axis=0)
        freq_maxes[stim_freq_i, :] = np.average(ch_maxes, axis=0)
    bf_idxs = np.argmax(freq_maxes, axis=0)
    bf = np.array([all_stim_freq[bf_i] for bf_i in bf_idxs])

    with h5py.File(auxfile) as h5file:
        h5file.create_dataset('/bl_mu', data=bl_mu)
        h5file.create_dataset('/bl_std', data=bl_std)
        h5file.create_dataset('/freq_maxes', data=freq_maxes)
        h5file.create_dataset('/bf', data=bf)
    

class TonePowerSpectrum(PowerSpectrum):
    def get_bfs(self):
        with h5py.File(self.auxfile) as infile:
            return infile['/bf'][:]
        

    def get_spectrum(self, channel):
        ch_data = self.proc_dset.data[:, channel, :]
        n_timepts, n_bands = ch_data.shape
        rate = self.proc_dset.rate

        hw = int(self.half_width * rate)

        # Rescale to baseline mean/stdev
        bl_mean, bl_std = self.proc_bl_stats
        bl_mean, bl_std = bl_mean[channel, :], bl_std[channel, :]
        ch_data = (ch_data - bl_mean) / bl_std

        # Grab center frequencies from bands table
        # def log_spaced_cfs(fmin, fmax, nbin=6):
        #     noct = np.ceil(np.log2(fmax/fmin))
        #     return fmin * 2**(np.arange(noct*nbin)/nbin)
        # band_means = log_spaced_cfs(2.6308, 1200.0)
        # band_means = self.proc_dset.bands['band_mean'][:]
        band_means = wavelet_cfs
        hg_band_idx = np.logical_and(band_means > 65, band_means < 170)

        # Grab stim-on data for best freq
        bf = self.get_bfs()[channel]
        trials = self.nwb.trials
        trial_idxs = np.logical_and(np.logical_and(trials['sb'][:] == 's', trials['frq'][:] == str(bf)), trials['amp'][:] == '7')
        times = zip(trials['start_time'][trial_idxs], trials['stop_time'][trial_idxs])
        stim_periods = [(int(t[0]*self.proc_dset.rate), int(t[1]*self.proc_dset.rate)) for t in times]

        n_stim_timepts = stim_periods[0][1] - stim_periods[0][0]
        stim_data = np.zeros(shape=(len(stim_periods), n_stim_timepts, n_bands))
        for i, (t1, t2) in enumerate(stim_periods):
            stim_data[i, :, :] = ch_data[t1:t1+n_stim_timepts, :]

        # Calculate max of average high gamma response
        # Average over stims, bands in hg range: time axis remains
        hg_data = np.average(stim_data[:, :, hg_band_idx], axis=(0,2))
        max_i = np.argmax(hg_data)
        self.max_i = max_i
        print(channel, max_i)
        max_i += self.time_shift_samp
        if max_i - hw <= 0:
            spectrum = np.zeros(shape=(54,))
            errors = np.zeros(shape=(54,))
        else:
            # Average over stims, time: freq (bands) axis remainds
            spectrum = np.average(stim_data[:, max_i-hw:max_i+hw, :], axis=(0,1))
            errors = np.std(stim_data[:, max_i-hw:max_i+hw, :], axis=(0,1))

        return band_means, spectrum, errors

    def plot_all_and_avg(self, specfile=None):
        if specfile and os.path.exists(specfile):
            print("using saved spectra")
            with h5py.File(specfile) as infile:
                all_spectra = infile['power_spectra'][:]
        else:
            print("Computing spectra")
            all_spectra = [self.get_spectrum(ch)[1] for ch in range(self.n_ch)]

        # if specfile and os.path.exists(specfile):
        #     os.remove(specfile)
        if specfile and not os.path.exists(specfile):
            with h5py.File(specfile) as outfile:
                outfile.create_dataset('f', data=wavelet_cfs)
                outfile.create_dataset('power_spectra', data=np.stack(all_spectra))
                
        ch_spectra = []
        for ch in range(self.n_ch):
            # f, spectrum, errors = self.get_spectrum(ch)
            spectrum = all_spectra[ch]
            if np.any(spectrum > 3.0):
                ch_spectra.append(spectrum)
                plt.plot(wavelet_cfs, spectrum, color='k', alpha=0.3, linewidth=0.3)
        avg_spectrum = np.average(np.stack(ch_spectra), axis=0)
        plt.plot(wavelet_cfs, avg_spectrum, color='red', alpha=1, linewidth=2)
        print("plotted {} spectra".format(len(ch_spectra)))

        

if __name__ == '__main__':
    # TONE150:
    # rat = 'R72'
    # block = 'R72_B6'
    # rat = 'R73'
    # block = 'R73_B2'
    # rat = 'R75'
    # block = 'R75_B8'
    # rat = 'R70'
    # block = 'R70_B8'

    # rat = 'R32'
    # block = 'R32_B7'
    rat = 'R18'
    block = 'R18_B12'
    
    nwbfile = '/data/{}/{}.nwb'.format(rat, block)
    auxfile = '/data/{}/{}_aux.h5'.format(rat, block)
    specfile = '/data/{}/{}_spectra.h5'.format(rat, block)
    
    plotter = TonePowerSpectrum(nwbfile, '.',
                                   # proc_dset_name='Wvlt_4to1200_54band_CAR0',
                                   auxfile=auxfile, half_width=0.005)

    if not os.path.exists(auxfile):
        bf_tone(plotter, auxfile)

    plt.figure(figsize=(4, 4))
    plotter.prepare_axes()
    plotter.plot_all_and_avg(specfile=specfile)

    plt.savefig("plots/tone_ps_{}.pdf".format(block))
    
