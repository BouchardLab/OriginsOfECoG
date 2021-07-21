# -*- coding: utf-8 -*-
"""
Figure 1 of the High gamma ECoG paper
"""
import os, sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.patches as patches
import matplotlib.cm as cmx
import matplotlib.colors as colors

from tone_avg_ecp import ToneAvgECP
from tone_spectrogram import ToneSpectrogram
from tone_power_spectrum import TonePowerSpectrum
from tone_spectrogram import ToneSpectrogram

rat = 'R18'
block = 'R18_B12'
tstart, tstop = 40, 42.5
# channel = 109 # for single channel plots
# channels = [10, 20, 30, 40, 50, 60, 70, 80, 124, 100, 109] # for Hg plot
if len(sys.argv) > 1:
    channel = int(sys.argv[-1])
    channels = []
else:
    channel = np.random.randint(128)
    channels = list(np.random.randint(128, size=10))
channels.append(channel)
print("Channel = {}".format(channel))
print("Channels = {}".format(channels))
if os.path.exists("fig1_ch{}.pdf".format(channel)):
    exit("Already done channel")

nwbfile = '/data/{}/{}.nwb'.format(rat, block)
auxfile = '/data/{}/{}_aux.h5'.format(rat, block)
specfile = '/data/{}/{}_spectra.h5'.format(rat, block)

fig = plt.figure(figsize=(7, 7))
# gs = GridSpec(4, 3, height_ratios=(1, 2, 2, 3.8))

########
# AXES
########

# ECoG micrograph (not produced here)

CBAR_WD = .015
CBAR_GAP = .005
LEN = .5

# Stimulus
# stim_ax = plt.subplot(gs[0, 1:])
stim_ax = fig.add_axes([.4, 1-.08, LEN, .08])
stim_ax.get_xaxis().set_visible(False)
stim_ax.get_yaxis().set_visible(False)

# Z-scored High gamma response
# hg_ax = plt.subplot(gs[1, 1:], sharex=stim_ax)
hg_ax = fig.add_axes([.4, 1-.08-.16, LEN, .16], sharex=stim_ax)
hg_ax.get_xaxis().set_visible(False)
hg_ax.set_ylabel("Hγ (Z-score)")

freq_colorbar_ax = fig.add_axes([.4+LEN+CBAR_GAP, 1-.08-.16+.005, CBAR_WD, .23])

# Spectrogram
# spect_ax = plt.subplot(gs[2, 1:], sharex=stim_ax)
bottom = 1-.08-.16-.16
spect_ax = fig.add_axes([.4, bottom, LEN, .16], sharex=stim_ax)
spect_colorbar_ax = fig.add_axes([.4+LEN+CBAR_GAP, bottom, CBAR_WD, .16])

SQ = .25

# Trial-avg raw trace
# raw_ax = plt.subplot(gs[3, 0])
raw_ax = fig.add_axes([.08, 1-.08-.16-.16-SQ, SQ, SQ])
raw_ax.get_xaxis().set_visible(False)

# Trial-avg spectrogram
# avg_spect_ax = plt.subplot(gs[3, 1])
bottom = 1-.08-.16-.16-SQ-.02-SQ
avg_spect_ax = fig.add_axes([.08, bottom, SQ, SQ])
avg_colorbar_ax = fig.add_axes([.08+SQ+CBAR_GAP, bottom, CBAR_WD, SQ])

# Power spectrum
# ps_ax = plt.subplot(gs[3, 2])
ps_ax = fig.add_axes([.5, .08, .45, .45])


########
# PLOTS
########


# Trial-avg raw trace
plt.sca(raw_ax)
plotter = ToneAvgECP(nwbfile, '.', no_baseline_stats=True, auxfile=auxfile, nosave=True)
plotter.plot(channel)

# Trial-avg spectrogram
plt.sca(avg_spect_ax)
plotter = ToneSpectrogram(nwbfile, '.', auxfile=auxfile, nosave=True)
im = plotter.plot_one(channel)
cbar = fig.colorbar(im, cax=avg_colorbar_ax)
avg_colorbar_ax.tick_params(labelsize=6)
cbar.set_label("Z-score", size=8)

# Power spectrum
plt.sca(ps_ax)
plotter = TonePowerSpectrum(nwbfile, '.', auxfile=auxfile, half_width=.005, nosave=True)
plotter.prepare_axes()
plotter.plot_all_and_avg(specfile=specfile)

# Stimulus
nwb = plotter.nwb
bfs = plotter.get_bfs()
trial_idxs = np.logical_and(nwb.trials['start_time'][:] > tstart,
                            nwb.trials['stop_time'][:] < tstop)
trial_idxs = np.logical_and(trial_idxs, nwb.trials['sb'][:] == 's')
start_times = nwb.trials['start_time'][trial_idxs]
freqs = np.array([float(f) for f in nwb.trials['frq'][trial_idxs]])
ampls = np.array([float(f) for f in nwb.trials['amp'][trial_idxs]])
all_freqs = np.array([float(f) for f in nwb.trials['frq'][::2]])
fmin, fmax = np.min(all_freqs), np.max(all_freqs)
nfreqs = len(np.unique(all_freqs))
print(bfs)
print([bfs[ch] for ch in channels])
print(freqs)
print(ampls)

color_norm = colors.LogNorm(vmin=fmin, vmax=fmax)
cmap = cmx.ScalarMappable(norm=color_norm, cmap='jet')
# use scalar_map.to_rgba(freq) for each freq's color
cbar = fig.colorbar(cmap, cax=freq_colorbar_ax)
cmin, cmax = freq_colorbar_ax.get_xlim()
cbar_mid = np.exp((np.log(cmin) + np.log(cmax)) / 2.0)
freq_colorbar_ax.plot(cbar_mid, bfs[channel], marker='.', color='k')
freq_colorbar_ax.tick_params(labelsize=6)
cbar.set_label("Freq (Hz)", size=8)

for start_time, freq, ampl in zip(start_times, freqs, ampls):
    stim_ax.add_patch(patches.Rectangle((start_time, 0), .05, ampl, color=cmap.to_rgba(freq)))

stim_ax.set_xlim([tstart, tstop])
stim_ax.set_ylim([0, 8])


# Hg
bands = plotter.proc_dset.bands['band_mean'][:]
f_idx = np.logical_and(bands > 65, bands < 170)
rate = plotter.proc_dset.rate
istart, istop = int(tstart*rate), int(tstop*rate)
t = np.linspace(tstart, tstop, istop-istart)
bl_mu, bl_std = plotter.proc_bl_stats
for ch in channels:
    mu = np.average(bl_mu[ch, f_idx])
    std = np.average(bl_std[ch, f_idx])
    ch_hg = plotter.proc_dset.data[istart:istop, ch, f_idx]
    ch_hg = np.average(ch_hg, axis=-1)
    ch_hg = (ch_hg - mu) / std
    hg_ax.plot(t, ch_hg, linewidth=0.5, color=cmap.to_rgba(bfs[ch]))
    
ymin, ymax = hg_ax.get_ylim()
for start_time in start_times:
    hg_ax.plot([start_time, start_time], (ymin, ymax),
               linestyle='--', color='grey', linewidth=0.5)
    hg_ax.plot([start_time+.05, start_time+.05], (ymin, ymax),
               linestyle='--', color='grey', linewidth=0.5)
hg_ax.set_ylim([ymin, ymax])
hg_ax.set_xlim([tstart, tstop])


# Spectrogram
class ToneSpectrogramLong(ToneSpectrogram):
    def get_t_extent(self):
        t = np.arange(tstart, tstop, 1.0/self.proc_dset.rate)
        extent = [tstart, tstop, 0, 1]
        return t, extent
    
    def draw_stim_bars(self):
        pass
    def draw_peak_bars(self):
        pass

plt.sca(spect_ax)
plotter = ToneSpectrogramLong(nwbfile, '.', tstart=tstart, tstop=tstop, stim_i='', auxfile=auxfile)
im = plotter.plot_one(channel, vmin=0, vmax=7)
cbar = fig.colorbar(im, cax=spect_colorbar_ax)
spect_colorbar_ax.tick_params(labelsize=6)
cbar.set_label("Z-score", size=8)

ymin, ymax = spect_ax.get_ylim()
for start_time in start_times:
    spect_ax.plot([start_time, start_time], (ymin, ymax),
                  linestyle='--', color='grey', linewidth=0.5)
    spect_ax.plot([start_time+.05, start_time+.05], (ymin, ymax),
                  linestyle='--', color='grey', linewidth=0.5)
spect_ax.set_ylim([ymin, ymax])
spect_ax.set_xlim([tstart, tstop])




# plt.show()
plt.savefig("fig1_ch{}.pdf".format(channel))
