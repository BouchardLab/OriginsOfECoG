"""
Combined figure of spike raster, spike rates, raw ECP, spectrogram, and z-scored power spectrum
"""
import os
import glob
import argparse
import logging as log

import h5py
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['Arial']
rcParams['font.family'] = 'sans-serif'
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
from pynwb import NWBHDF5IO

import utils
from myplotters import get_cmap

from spectrogram import Spectrogram, SpectrogramRatio
from power_spectrum import PowerSpectrum, PowerSpectrumRatio
from power_spectrum_expt_avg import plot_expt_avg, plot_expt_avg_ratio
from raw_power_spectrum import RawPowerSpectrum

plt.rc('axes', labelsize=10)
plt.rc('xtick', labelsize=8)
plt.rc('ytick', labelsize=8)

# activate latex text rendering
# rc('text', usetex=True)

# WN:
# TSTART, TSTOP = 2400, 2700
# TSTIM = 2500

# TONE:
# TSTART, TSTOP = 2250, 2400
# TSTIM = 2300
TSTART, TSTOP = 2450, 2600
TSTIM = 2500
TSTIMS = [i*1000+500 for i in range(1, 60)]
STIM_DUR = 50

TYPE = 'zscore'
SPECT_CLS = Spectrogram if TYPE == 'zscore' else SpectrogramRatio
PS_CLS = PowerSpectrum if TYPE == 'zscore' else PowerSpectrumRatio
PS_FCN = plot_expt_avg if TYPE == 'zscore' else plot_expt_avg_ratio

def iter_spike_data(nodes_df, spikes_file, groupby=('layer', 'ei'), stim_avg=False):
    """
    Return a dict mapping layer or layer_ei to a dict containing 'gids' and 'spike_times'
    Also return all spike times as 2nd retval for convenience
    If stim_avg = True, returns spike times % 1000 ms
    """
    # Grab all gid's and spike times
    with h5py.File(spikes_file, 'r') as spikes_h5:
        spike_gids = np.array(spikes_h5['/spikes/gids'], dtype=np.uint)
        spike_times = np.array(spikes_h5['/spikes/timestamps'], dtype=np.float)

    if stim_avg:
        # TODO: TEST (this isn't working)
        spike_times = spike_times % 1000

    # Group by layer or layer/ei
    groupings = nodes_df.groupby(list(groupby))
    def strify(args):
        if type(args) == int:
            return str(args)
        if len(args) == 1:
            return str(args[0])
        else:
            return str(args[0]) + args[1]

    grouped = {strify(name): grp for name, grp in groupings}

    # Put the spike times and gid's into a dict
    spike_dfs = {}
    for groupname, group_df in grouped.items():
        indexes = np.in1d(spike_gids, group_df.index)
        spike_dfs[groupname] = {'gids': spike_gids[indexes], 'spike_times': spike_times[indexes]}

    return spike_dfs, spike_times
    
def moving_average(a, n=20):
    """
    https://stackoverflow.com/a/14314054
    """
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    padding = np.zeros(shape=((n)//2,))
    return np.hstack([padding, ret[n - 1:] / n, padding])


def get_time_to_first_spike_and_peaks(jobnum, peaks_only=False):
    network_dir = utils.get_network_dir(jobnum)
    output_dir = utils.get_output_dir(jobnum)
    cells_file = os.path.join(network_dir, 'cortical_column_nodes.h5')
    cell_models_file = os.path.join(network_dir, 'cortical_column_node_types.csv')
    spikes_file = os.path.join(output_dir, 'spikes.h5')

    nodes_df = utils.get_nodes_df(cells_file, cell_models_file)
    spike_dfs, spike_times = iter_spike_data(nodes_df, spikes_file, groupby=('layer',))
    all_bins = np.arange(0, 60000, 1.0)
    time_to_first_spike = {layer: [] for layer in [1, 2, 3, 4, 5, 6]}
    peaks = {layer: [] for layer in [1, 2, 3, 4, 5, 6]}
    for layer in [1, 2, 3, 4, 5, 6]:
        popn = str(layer)
        spikedata = spike_dfs[popn]

        # Compute layer-avg spike rates
        avg_spikerate = np.mean(np.vstack(
            moving_average(
                np.histogram(
                    spikedata['spike_times'][spikedata['gids'] == gid],
                    all_bins
                )[0],
                n=5 # # bins (ms) to average for each point
            ) * 1000 # 1/ms --> 1/s = hz
            for gid in set(spikedata['gids'])
        ), axis=0)

        # Compute times to first spike in each neuron for each stim
        gids = spikedata['gids']
        for tstim in TSTIMS:
            if not peaks_only:
                for gid in np.unique(gids):
                    this_stim_idx = np.logical_and(spikedata['spike_times'] > tstim,
                                                   spikedata['spike_times'] < tstim + STIM_DUR)
                    this_stim_this_gid_idx = np.logical_and(gids == gid, this_stim_idx)
                    if np.any(this_stim_this_gid_idx):
                        first_spike = np.min(spikedata['spike_times'][this_stim_this_gid_idx]) - tstim
                        # plt.plot(first_spike, layer, color=cmap[popn], marker='.')
                        time_to_first_spike[layer].append(first_spike)

            # Obtain time to peak rate within layer
            peaks[layer].append(np.argmax(avg_spikerate[tstim:tstim+STIM_DUR]))

    return time_to_first_spike, peaks

def plot_layer_timings(jobnum, ax):
    _, peaks = get_time_to_first_spike_and_peaks(jobnum, peaks_only=True)
    cmap = get_cmap(real=True)
    for layer in [1, 2, 3, 4, 5, 6]:
        popn = str(layer)
        peak = np.average(peaks[layer])
        std = np.std(peaks[layer])
        ax.errorbar(peak, layer, xerr=std, capsize=3, color=cmap[popn])

    ax.set_xlabel("Time to peak spike rate (ms)")
    ax.set_ylabel("Layer")
    ax.set_yticks([1, 2, 3, 4, 5, 6])
    ax.set_yticklabels(['I', 'II', 'III', 'IV', 'V', 'VI'])

    ax.set_xlim([0, 30])
    ax.set_ylim(reversed(ax.get_ylim()))


def plot_popn_spike_rate(ax, spikes, num_nodes, label, color):
    binsize = 1.0
    bins = np.arange(0, 60000, binsize)
    num_spikes, _ = np.histogram(spikes, bins=bins)
    p_spike = np.float64(num_spikes) / np.float64(num_nodes) / 60.0
    window_len = 150
    p_spike_trial = np.zeros(shape=(window_len, len(TSTIMS)))
    for i, tstim in enumerate(TSTIMS):
        tstart = tstim - 50
        p_spike_trial[:, i] = p_spike[tstart:tstart+window_len]
    plot_bins = np.arange(-50, 100, 1.0)
    avg = np.average(p_spike_trial, axis=-1)
    std = np.std(p_spike_trial, axis=-1)
    ax.fill_between(plot_bins, avg-std, avg+std, color=color, alpha=0.2)
    ax.plot(plot_bins, avg, color=color, label=label)
    ax.set_xlim([-50, 100])
    ax.set_ylim([0, max(avg) * 1.1])

def plot_ecp(ecp_file, ax):
    with NWBHDF5IO(ecp_file, 'r') as io:
        nwb = io.read()
        dset = nwb.acquisition['Raw'].electrical_series['ECoG']
        rate = dset.rate
        window_len_samp = int(.150 * rate)
        ecp = np.zeros(shape=(window_len_samp, len(TSTIMS)))
        for i, tstim in enumerate(TSTIMS):
            tstart = tstim - 50
            istart = int(round(tstart/1000.0*rate))
            ecp[:, i] = dset.data[istart:istart+window_len_samp, 0]
        avg = np.average(ecp, axis=-1)
        std = np.std(ecp, axis=1)
        t = np.linspace(TSTART, TSTOP, len(ecp))
        import ipdb; ipdb.set_trace()
        ax.fill_between(t, avg-std, avg+std, color='grey')
        ax.plot(t, avg, linewidth=0.5, color='black')

def plot_spikes(jobnum, raster_ax, ctx_rate_ax, bkg_rate_ax, stim_avg=False):
    network_dir = utils.get_network_dir(args.jobnum)
    output_dir = utils.get_output_dir(args.jobnum)
    cells_file = os.path.join(network_dir, 'cortical_column_nodes.h5')
    cell_models_file = os.path.join(network_dir, 'cortical_column_node_types.csv')
    thal_spikes_file = os.path.join(network_dir, 'thalamus_spikes.csv')
    bkg_spikes_file = os.path.join(network_dir, 'bkg_spikes.csv')
    spikes_file = os.path.join(output_dir, 'spikes.h5')

    cmap = get_cmap()

    nodes_df = utils.get_nodes_df(cells_file, cell_models_file, population=None)
    spike_dfs, spike_times = iter_spike_data(nodes_df, spikes_file, stim_avg=stim_avg)

    for layer_ei, spikedata in spike_dfs.items():
        spike_idxs = np.logical_and(spikedata['spike_times'] > TSTART,
                                    spikedata['spike_times'] < TSTOP)
        depths = nodes_df['depth'][spikedata['gids']][spike_idxs] - 2082
        raster_ax.scatter(
            spikedata['spike_times'][spike_idxs], depths, marker='.', facecolors=cmap[layer_ei],
            label=layer_ei, lw=0, s=5, alpha=0.1 if stim_avg else 0.3
        )
    ymin, ymax = min(nodes_df['depth']), max(nodes_df['depth'])
    raster_ax.set_ylim([ymin-2082, ymax-2082])
    raster_ax.set_xlim([TSTART, TSTOP])

    log.info("Done w/ raster")

    # PLOT SPIKE RATES
    plot_popn_spike_rate(ctx_rate_ax, spike_times, len(nodes_df), "Cortex", "red")
    log.info("Done w/ ctx pop'n spike rate")
    
    bkg_spikes_df = pd.read_csv(bkg_spikes_file, sep=' ')
    bkg_spike_times = [
        float(t) for spike_times_str in bkg_spikes_df['spike-times']
        for t in spike_times_str.split(',')
    ]
    num_bkg_nodes = len(bkg_spikes_df['spike-times'])
    plot_popn_spike_rate(bkg_rate_ax, bkg_spike_times, num_bkg_nodes, "Bkg", color='black')
    log.info("Done w/ bkg pop'n spike rate")

    thal_spikes_df = pd.read_csv(thal_spikes_file, sep=' ')
    thal_spike_times = [
        float(t) for spike_times_str in thal_spikes_df['spike-times']
        for t in spike_times_str.split(',')
    ]
    num_thal_nodes = len(thal_spikes_df['spike-times'])
    plot_popn_spike_rate(bkg_rate_ax, thal_spike_times, num_thal_nodes, "Thalamus", color='red')
    log.info("Done w/ thal pop'n spike rate")



def plot_layer_spikerates(jobnum, ax):
    network_dir = utils.get_network_dir(args.jobnum)
    output_dir = utils.get_output_dir(args.jobnum)
    cells_file = os.path.join(network_dir, 'cortical_column_nodes.h5')
    cell_models_file = os.path.join(network_dir, 'cortical_column_node_types.csv')
    spikes_file = os.path.join(output_dir, 'spikes.h5')

    cmap = get_cmap(real=True)

    nodes_df = utils.get_nodes_df(cells_file, cell_models_file)
    spike_dfs, spike_times = iter_spike_data(nodes_df, spikes_file, groupby=('layer',))
    bins = np.arange(TSTART, TSTOP, 1.0)
    # for popn in ['1i', '2i', '3i', '4i', '5i', '6i', '2e', '3e', '4e', '5e', '6e']:
    for popn, numeral in zip(['1', '2', '3', '4', '5', '6'], ['I', 'II', 'III', 'IV', 'V', 'VI']):
        spikedata = spike_dfs[popn]
        avg_spikerate = np.mean(np.vstack(
            moving_average(
                np.histogram(
                    spikedata['spike_times'][spikedata['gids'] == gid],
                    bins
                )[0],
                n=5 # # bins (ms) to average for each point
            ) * 1000 # 1/ms --> 1/s = hz
            for gid in set(spikedata['gids'])
        ), axis=0)
        offset = len(bins) - len(avg_spikerate)
        if offset > 0:
            avg_spikerate = np.append(avg_spikerate, [0]*offset)
        max_spikerate = max(avg_spikerate)
        ax.plot(bins-TSTIM, avg_spikerate/max_spikerate*100,
                color=cmap[popn], label=numeral)
        ymax = max(max_spikerate, ax.get_ylim()[1])

power_spectrum_rats = ['R32'] # ['R32', 'R18', 'R6', 'R6']
power_spectrum_blocks = ['R32_B7'] # ['R32_B7', 'R18_B12', 'R6_B10', 'R6_B16']
def plot_spectrogram_power_spectrum(ecp_file, spect_ax, ps_ax):
    # SIM PWR SPECTRUM
    ps_plotter = PS_CLS(ecp_file, '', device='ECoG', stim_i='avg', half_width=0.005,
    # ps_plotter = PowerSpectrumRatio(ecp_file, '', device='ECoG', stim_i='avg', half_width=0.005,
                               # nosave=True, color='red', label=r'\textit{In silico}')
                               nosave=True, color='red', label=r'In silico')
    # ps_plotter = RawPowerSpectrum(ecp_file, '', device='ECoG', stim_i='avg',
    #                                   nosave=True, color='blue', label='Raw', half_width=.1)
    plt.sca(ps_ax)
    ps_plotter.set_normalize(True)
    bands, _, _ = ps_plotter.plot_one(0)

    # EXPT PWR SPECTRUM
    # all_spectra = []
    # for _rat, _block in zip(power_spectrum_rats, power_spectrum_blocks):
    #     _specfile = '/data/{}/{}_spectra.h5'.format(_rat, _block)
    #     with h5py.File(_specfile) as infile:
    #         spectra = infile['power_spectra'][:]
    #         n_ch = spectra.shape[1]
    #         for i in range(n_ch):
    #             spectrum = spectra[i]
    #             if np.any(spectrum > 3.0):
    #                 # plt.plot(bands, spectrum/max(spectrum), color='k', alpha=0.3, linewidth=0.3)
    #                 all_spectra.append(spectrum)
    # avg_spectrum = np.average(np.stack(all_spectra), axis=0)
    # std = np.std(np.stack(all_spectra), axis=0)
    # plt.plot(bands, avg_spectrum/max(avg_spectrum), color='k', alpha=1, linewidth=2, label='In vivo')
    PS_FCN(label='In vivo')

    ps_ax.legend(prop={'style': 'italic', 'size': 'x-small'})

    
    # SPECTROGRAM
    PEAK_TIME = float(ps_plotter.max_i)/400.0
    # hw = 1.0/400 # half-width
    hw = .005
    spect_plotter = SPECT_CLS(ecp_file, '', device='ECoG', nosave=True, stim_i='avg')
    plt.sca(spect_ax) # hack to get my plotter to work on a given axis
    spect_plotter.plot_one(0, vmax=10)
    spect_ax.plot([PEAK_TIME-hw, PEAK_TIME-hw], spect_ax.get_ylim(),
            linestyle='--', linewidth=0.5, color='red')
    spect_ax.plot([PEAK_TIME+hw, PEAK_TIME+hw], spect_ax.get_ylim(),
            linestyle='--', linewidth=0.5, color='red')

def find_ecp_file(jobnum):
    output_dir = utils.get_output_dir(args.jobnum)
    ecp_files = glob.glob(os.path.join(output_dir, 'ecp*.nwb'))
    if len(ecp_files) == 0:
        raise ValueError('No ECP file found, specify with --ecp-file')
    elif len(ecp_files) == 1:
        return ecp_files[0]
    else:
        basic_ecp_files = [x for x in ecp_files if 'layer' not in x]
        log.info(
            'Found multiple ECP files: \n{}\n'.format('\n'.join(ecp_files)) + 
            'Using {}\n'.format(basic_ecp_files[0]) +
            'To choose another, use --ecp-file'
        )
        return basic_ecp_files[0]


def make_figure(jobnum, ecp_file, render_file, outfile):

    ecp_file = ecp_file or find_ecp_file(jobnum)
    
    # Figure and gridspec
    plt.figure(figsize=(10, 10))
    gs = GridSpec(100, 105)

    ###############
    # FIRST COLUMN
    ###############

    # LAYER TIMINGS
    TIMING_WD = 15
    TIMING_HSTART = 44
    TIMING_HT = 18

    timing_ax = plt.subplot(gs[TIMING_HSTART:TIMING_HSTART+TIMING_HT, :TIMING_WD])
    plot_layer_timings(jobnum, timing_ax)
    log.info("Done with layer timings")

    #################
    # MIDDLE COLUMN
    #################

    MIDSTART = 28

    ECP_HT = 10
    ECP_WD = 40

    ecp_ax = plt.subplot(gs[:ECP_HT, MIDSTART:MIDSTART+ECP_WD])
    plot_ecp(ecp_file, ecp_ax)
    ecp_ax.axes.get_xaxis().set_visible(False)
    ecp_ax.axes.get_yaxis().set_visible(False)
    ecp_ax.axis('off')


    # SPIKE RASTER
    RASTER_HT = 30
    RASTER_WD = ECP_WD
    POPN_SPKRT_HT = 6
    
    raster_ax = plt.subplot(gs[ECP_HT:ECP_HT+RASTER_HT, MIDSTART:MIDSTART+RASTER_WD])
    raster_ax.axes.set_ylabel('Depth (um)')
    raster_ax.axes.get_xaxis().set_visible(False)
    ctx_rate_ax = plt.subplot(gs[ECP_HT+RASTER_HT:ECP_HT+RASTER_HT+POPN_SPKRT_HT, MIDSTART:MIDSTART+RASTER_WD])
    ctx_rate_ax.axes.get_xaxis().set_visible(False)
    # ctx_rate_ax.set_ylabel('Population\nspike rate')
    bkg_rate_ax = plt.subplot(gs[ECP_HT+RASTER_HT+POPN_SPKRT_HT:ECP_HT+RASTER_HT+2*POPN_SPKRT_HT, MIDSTART:MIDSTART+RASTER_WD])
    bkg_rate_ax.axes.get_xaxis().set_visible(False)
    # bkg_rate_ax.set_ylabel('Population\nspike rate')

    plot_spikes(jobnum, raster_ax, ctx_rate_ax, bkg_rate_ax)

    ctx_rate_ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='x-small')
    bkg_rate_ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='x-small')

    RASTER_BOTTOM = ECP_HT + RASTER_HT + 2*POPN_SPKRT_HT

    log.info("Done w/ raster + pop'n rate plots")

    
    
    # LAYER SPIKE RATES
    LAYER_SPKRT_HT = 10
    
    layer_spkrate_ax = plt.subplot(gs[RASTER_BOTTOM:RASTER_BOTTOM+LAYER_SPKRT_HT, MIDSTART:MIDSTART+RASTER_WD])
    layer_spkrate_ax.set_ylim([0, 115])
    layer_spkrate_ax.set_ylabel('% of max\nspike rate')
    layer_spkrate_ax.set_xlim([TSTART-TSTIM, TSTOP-TSTIM])
    layer_spkrate_ax.set_xlabel('Time (ms)')
    
    plot_layer_spikerates(jobnum, layer_spkrate_ax)

    layer_spkrate_ax.legend(loc='upper right', bbox_to_anchor=(1.0, 1.0), fontsize='xx-small',
                            ncol=2)

    log.info('Done w/ layer spike rates')


    ###############
    # RIGHT COLUMN
    ###############
    # TODO: right column as sub-gridspec?
    
    RIGHTSTART = MIDSTART + RASTER_WD + 10

    # SPECTROGRAM
    SPECT_TOPGAP = ECP_HT
    SPECT_WD = 27 # will be square with colorbar
    SPECT_HT = 22

    # Arrow from ECP to spectrogram
    ecp_spect_arrow_ax = plt.subplot(gs[:SPECT_TOPGAP, MIDSTART+RASTER_WD:MIDSTART+RASTER_WD+SPECT_WD])
    ecp_spect_arrow_ax.axis('off')
    ecp_spect_arrow_ax.set_xlim([0, 1])
    ecp_spect_arrow_ax.set_ylim([0, 1])
    leftright = 0.75
    updown = 0.7
    ecp_spect_arrow_ax.plot([0, leftright], [updown, updown], color='black')
    ecp_spect_arrow_ax.arrow(leftright, updown, 0, -updown, # x, y, dx, dy
                             head_width=0.07, length_includes_head=True, 
                             color='black')
    ecp_spect_arrow_ax.text(leftright/2, 0.75, "Freq. Decomp.",
                            horizontalalignment='center',
                            verticalalignment='bottom')
    
    spect_ax = plt.subplot(gs[SPECT_TOPGAP:SPECT_TOPGAP+SPECT_HT, RIGHTSTART:RIGHTSTART+SPECT_WD])
    # SIM/EXP POWER SPECTRUM
    PWRSP_TOP = SPECT_TOPGAP + SPECT_HT + 8
    PWRSP_WD = 22
    PWRSP_HT = PWRSP_WD

    pwrsp_ax = plt.subplot(gs[PWRSP_TOP:PWRSP_TOP+PWRSP_HT, RIGHTSTART:RIGHTSTART+PWRSP_WD])
    plot_spectrogram_power_spectrum(ecp_file, spect_ax, pwrsp_ax)
    spect_ax.set_xticks([-.05, 0, .05, .10])
    spect_ax.set_xticklabels([-50, 0, 50, 100])
    spect_ax.set_xlabel("Time (ms)")
    # pwrsp_ax.set_ylabel("Ratio (norm)")
    pwrsp_ax.set_ylabel("Z-score (norm)" if TYPE == 'zscore' else "Ratio (norm)")
    pwrsp_ax.set_ylim([0, 1.15])

    log.info('Done w/ spectrogram+power spectrum')
    

    ###############
    # Save or show
    ###############
    if outfile:
        plt.savefig(outfile)
    else:
        plt.show()
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--jobnum', type=str, required=True)
    parser.add_argument('--ecp-file', type=str, required=False, default=None)
    parser.add_argument('--render-file', type=str, required=False, default='renders/render_100um_10pct_somadot_noshading_excfirst_alpha0.2_0.5.png')
    parser.add_argument('--outfile', type=str, required=False, default=None)
    parser.add_argument('--debug', action='store_true', required=False, default=False)
    args = parser.parse_args()

    log.basicConfig(format='%(asctime)s %(message)s', level=log.DEBUG if args.debug else log.INFO)
    
    make_figure(args.jobnum, args.ecp_file, args.render_file, args.outfile)
