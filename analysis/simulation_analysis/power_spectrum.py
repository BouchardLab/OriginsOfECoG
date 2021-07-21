"""
z-scored power spectrum during peak Hg response
"""

import os
import numpy as np
import matplotlib.pyplot as plt

from analysis import BasePlotter, PlotterArgParser

def find_peak_signal(stim_data, rate, search_window=.005, avg_window=1, time_shift_samp=0):
    """
    Average the signal on each channel/band around the peak,
    where peak is defined as the maximum within some time (search_window)
    of the peak response (across ALL times) of the high gamma signal

    stim_data: data during the stimulus period
        shape (n_stims, n_timepts, n_channels, n_bands)
    search_window: number of SECONDS to search around the high gamma peak
    avg_window: number of SAMPLES around the high gamma peak to avg over
    """
    n_stims, n_timepts, n_ch, n_bands = stim_data.shape
    
    # Average across stimulus presentations
    stim_spectra = np.mean(stim_data, axis=0)

    # Compute the max in the high gamma range across all timepoints
    search = int(search_window * rate) # samples
    hg_signal = np.mean(stim_spectra[search+avg_window:-search-avg_window, :, 30:37], axis=-1) # 29:36?
    hg_maxes = np.argmax(hg_signal, axis=0) + search + avg_window # Max of the hg signal on each electrode
    print("hg maxes", hg_maxes)

    # Compute the average within the search window of the hg peaks
    avg_ampl_during_peak = np.zeros(shape=stim_spectra.shape[1:])
    for i in range(n_ch):
        hg_max = hg_maxes[i] + time_shift_samp
        for j in range(n_bands):
            maxidx = np.argmax(stim_spectra[hg_max-search:hg_max+search+1, i, j]) + hg_max - search
            x = np.mean(stim_spectra[maxidx-avg_window:maxidx+avg_window+1, i, j])
            avg_ampl_during_peak[i, j] = x

    return avg_ampl_during_peak


class PowerSpectrum(BasePlotter):
    ylabel = 'Z-score'

    def __init__(self, nwbfile, outdir, **kwargs):
        self.half_width = kwargs.pop('half_width', .0025)
        self.normalize = kwargs.pop('normalize', False)
        self.time_shift_samp = kwargs.pop('time_shift_samp', 0)
        self.errors = kwargs.pop('errors', False)
        self.elinewidth = kwargs.pop('elinewidth', 0.5)

        super(PowerSpectrum, self).__init__(nwbfile, outdir, **kwargs)

    def set_normalize(self, normalize):
        self.normalize = normalize

    def transform_data(self, data, bl_mean, bl_std):
        """Z-score"""
        return (data - bl_mean) / bl_std

    def get_spectrum(self, channel):
        """
        Return the z-scored power spectrum
        """
        ch_data = self.proc_dset.data[:, channel, :]
        n_timepts, n_bands = ch_data.shape
        rate = self.proc_dset.rate

        hw = int(self.half_width * rate)

        # Rescale to baseline mean/stdev
        bl_mean, bl_std = self.proc_bl_stats
        bl_mean, bl_std = bl_mean[channel, :], bl_std[channel, :]
        ch_data = self.transform_data(ch_data, bl_mean, bl_std)

        # Grab center frequencies from bands table
        # def log_spaced_cfs(fmin, fmax, nbin=6):
        #     noct = np.ceil(np.log2(fmax/fmin))
        #     return fmin * 2**(np.arange(noct*nbin)/nbin)
        # band_means = log_spaced_cfs(2.6308, 1200.0)
        band_means = self.proc_dset.bands['band_mean'][:]
        hg_band_idx = np.logical_and(band_means > 65, band_means < 170)

        # Grab stim-on data, trial average if requested
        stim_periods = self.get_stim_periods(rate=rate, pre_dur=0, post_dur=0)
        if self.stim_i == 'avg':
            n_stim_timepts = stim_periods[0][1] - stim_periods[0][0]
            stim_data = np.zeros(shape=(len(stim_periods), n_stim_timepts, n_bands))
            for i, (t1, t2) in enumerate(stim_periods):
                stim_data[i, :, :] = ch_data[t1:t1+n_stim_timepts, :]
                
            # Calculate max of average high gamma response
            # Average over stims, bands in hg range: time axis remains
            hg_data = np.average(stim_data[:, :, hg_band_idx], axis=(0,2))
            max_i = np.argmax(hg_data)
            self.max_i = max_i
            print(max_i)
            max_i += self.time_shift_samp
            assert max_i - hw > 0

            # Average over stims, time: freq (bands) axis remainds
            spectrum = np.average(stim_data[:, max_i-hw:max_i+hw, :], axis=(0,1))
            errors = np.std(stim_data[:, max_i-hw:max_i+hw, :], axis=(0,1))
        else: # self.stim_i is an integer index
            tstart, tstop = stim_periods[self.stim_i]
            stim_data = ch_data[tstart:tstop, :]
            hg_data = np.average(ch_data[tstart:tstop, hg_band_idx], axis=1)
            max_i = np.argmax(hg_data)
            self.max_i = max_i
            spectrum = np.average(stim_data[max_i-hw:max_i+hw+1, :], axis=0)
            errors = np.zeros(len(spectrum))

        return band_means, spectrum, errors

    def get_data(self):
        """
        Return all data during the stimulus period
        shape (n_stims, n_timepts, n_channels, n_bands)
        """
        n_timepts, n_ch, n_bands = self.proc_dset.data.shape
        rate = self.proc_dset.rate
        stim_periods = self.get_stim_periods(rate=rate, pre_dur=0.0, post_dur=0.0)
        n_stim_timepts = stim_periods[0][1] - stim_periods[0][0]
        stim_data = np.zeros(shape=(len(stim_periods), n_stim_timepts, n_ch, n_bands))
        for i, (t1, t2) in enumerate(stim_periods):
            stim_data[i, ...] = self.proc_dset.data[t1:t1+n_stim_timepts, ...]
        return stim_data

    def save_spectra(self):
        all_ch_spectra = np.stack([self.get_spectrum(ch)[1] for ch in range(n_ch)])
        self.nwb.add_scratch(all_ch_spectra, name='power_spectrum', notes='power spectrum')

    def get_avg_spectrum(self):
        """
        Channel average z-score.
        This should really only be used for experimental blocks.
        Does not support errorbars
        """
        bl_mean, bl_std = self.proc_bl_stats
        def log_spaced_cfs(fmin, fmax, nbin=6):
            noct = np.ceil(np.log2(fmax/fmin))
            return fmin * 2**(np.arange(noct*nbin)/nbin)
        band_means = log_spaced_cfs(2.6308, 1200.0) if self.is_expt else self.proc_dset.bands['band_mean'][:]
        stim_data = self.get_data()
        n_stims, n_timepts, n_ch, n_bands = stim_data.shape
        bl_mean = np.tile(bl_mean, (n_stims, n_timepts, 1, 1))
        bl_std = np.tile(bl_std, (n_stims, n_timepts, 1, 1))
        stim_data = self.transform_data(stim_data, bl_mean, bl_std)
        stim_peak_spectra = find_peak_signal(stim_data, self.proc_dset.rate, time_shift_samp=self.time_shift_samp)
        avg_spectrum = np.average(stim_peak_spectra, axis=0)
        return band_means, avg_spectrum

    def prepare_axes(self):
        plt.gca().set_xscale('log')
        plt.xlim([10, 1200])
        plt.xscale('log')
        plt.xlabel('Frequency (Hz)')
        ylabel = self.ylabel + '/max' if self.normalize else self.ylabel
        plt.ylabel(ylabel)
        
    def plot_avg(self, **plot_args):
        """channel average"""
        f, avg_spectrum = self.get_avg_spectrum()
        if self.normalize:
            avg_spectrum = avg_spectrum / np.max(avg_spectrum)
        self.prepare_axes()
        plt.gca().plot(f, avg_spectrum, color=self.color, **plot_args)

    
    def plot_one(self, channel, **plot_args):
        """
        Make one channel's power spectrum and save it to file
        """

        band_means, spectrum, errors = self.get_spectrum(channel)

        if self.normalize:
            errors = errors / max(spectrum)
            spectrum = spectrum / max(spectrum)

        final_plot_args = {
            'label': self.label,
            'color': self.color,
            'linewidth': self.linewidth,
            'elinewidth': self.elinewidth,
            'capsize': 1,
        }
        final_plot_args.update(plot_args)
            
        self.prepare_axes()
        plt.errorbar(
            band_means, spectrum, yerr=(errors if self.errors else None),
            **final_plot_args
        )
        self.label = None # Prevent the label from being applied multiple times
        plt.tight_layout()

        if not self.nosave:
            fn = 'power_spectrum_{}_ch{:02d}_{}.{}'.format(
                self.device, channel, self.identifier, self.filetype
            )
            full_fn = os.path.join(self.outdir, fn)
            plt.savefig(full_fn)

        if self.show:
            plt.show()

        return band_means, spectrum, errors


    def plot_one_layer_ei(self, layer, ei, contrib_or_lesion, **plot_args):
        """Only valid for simulation, so channel must be 0"""
        old_proc_dset_name = self.proc_dset_name # "Hilb_54bands"
        old_proc_bl_stats = self.proc_bl_stats
        lesion = 'l' if contrib_or_lesion in ('lesion', 'removal') else ''
        self.proc_dset_name = '{}_{}{}{}'.format(old_proc_dset_name, lesion, layer, ei)
        self.proc_bl_stats = self._compute_proc_baseline_stats()
        
        self.plot_one(0, **plot_args)
        
        self.proc_dset_name = old_proc_dset_name
        self.proc_bl_stats = old_proc_bl_stats

    def plot_one_slice(self, slice_i, contrib_or_lesion, **plot_args):
        old_proc_dset_name = self.proc_dset_name # "Hilb_54bands"
        old_proc_bl_stats = self.proc_bl_stats
        lesion = 'l' if contrib_or_lesion in ('lesion', 'removal') else ''
        self.proc_dset_name = '{}_{}{}'.format(old_proc_dset_name, lesion, slice_i)
        self.proc_bl_stats = self._compute_proc_baseline_stats()
        
        self.plot_one(0, **plot_args)
        
        self.proc_dset_name = old_proc_dset_name
        self.proc_bl_stats = old_proc_bl_stats
        
    def plot_diff_layer_ei(self, layer, ei, contrib_or_lesion, **plot_args):
        """
        Plot the difference between layer contribution/lesion and full power spectrum
        """
        f, full_spectrum, full_errors = self.get_spectrum(0)
        
        old_proc_dset_name = self.proc_dset_name # "Hilb_54bands"
        old_proc_bl_stats = self.proc_bl_stats
        lesion = 'l' if contrib_or_lesion in ('lesion', 'removal') else ''
        self.proc_dset_name = '{}_{}{}{}'.format(old_proc_dset_name, lesion, layer, ei)
        self.proc_bl_stats = self._compute_proc_baseline_stats()
        
        f, layer_spectrum, layer_errors = self.get_spectrum(0)
        
        self.proc_dset_name = old_proc_dset_name
        self.proc_bl_stats = old_proc_bl_stats

        diff = full_spectrum - layer_spectrum

        self.prepare_axes()
        final_plot_args = {
            'label': self.label,
            'color': self.color,
            'linewidth': self.linewidth,
        }
        final_plot_args.update(plot_args)
        plt.plot(f, diff, **final_plot_args)
        
        

class PowerSpectrumRatio(PowerSpectrum):
    ylabel = 'Stim/baseline ratio'
    
    def transform_data(self, data, bl_mean, bl_std):
        """Ratio"""
        return data / bl_mean
    
    def plot_one_layer_ei(self, layer, ei, contrib_or_lesion, **plot_args):
        """Only valid for simulation, so channel must be 0"""
        old_proc_dset_name = self.proc_dset_name # "Hilb_54bands"
        lesion = 'l' if contrib_or_lesion in ('lesion', 'removal') else ''
        self.proc_dset_name = '{}_{}{}{}'.format(old_proc_dset_name, lesion, layer, ei)
        self.plot_one(0, **plot_args)
        self.proc_dset_name = old_proc_dset_name

    def plot_one_slice(self, slice_i, contrib_or_lesion, **plot_args):
        old_proc_dset_name = self.proc_dset_name # "Hilb_54bands"
        lesion = 'l' if contrib_or_lesion in ('lesion', 'removal') else ''
        self.proc_dset_name = '{}_{}{}'.format(old_proc_dset_name, lesion, slice_i)
        self.plot_one(0, **plot_args)
        self.proc_dset_name = old_proc_dset_name

    def plot_diff_layer_ei(self, layer, ei, contrib_or_lesion, **plot_args):
        f, full_spectrum, full_errors = self.get_spectrum(0)
        
        old_proc_dset_name = self.proc_dset_name # "Hilb_54bands"
        lesion = 'l' if contrib_or_lesion in ('lesion', 'removal') else ''
        self.proc_dset_name = '{}_{}{}{}'.format(old_proc_dset_name, lesion, layer, ei)
        f, layer_spectrum, layer_errors = self.get_spectrum(0)
        self.proc_dset_name = old_proc_dset_name

        diff = full_spectrum - layer_spectrum

        self.prepare_axes()
        final_plot_args = {
            'label': self.label,
            'color': self.color,
            'linewidth': self.linewidth,
        }
        final_plot_args.update(plot_args)
        plt.plot(f, diff, **final_plot_args)
        
if __name__ == '__main__':
    parser = PlotterArgParser()
    args = parser.parse_args()

    analysis = PowerSpectrumRatio(args.nwbfile, args.outdir, **parser.kwargs)
    analysis.run()
