# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import matplotlib.colors as colors
import matplotlib.cm as cmx

from layer_reader import LayerReader
from power_spectrum import PowerSpectrum, PowerSpectrumRatio
from utils import find_layer_ei_ecp_file

TSTART, TSTOP, TSTIM = 2400, 2700, 2500
JOBNUM = '29812500'
TYPE = 'ratio'
nwbfile = find_layer_ei_ecp_file(JOBNUM)
expt_nwbfile = "/Users/vbaratham/src/simulation_analysis/R32_B6_notch_filtered.nwb"

def get_colors(n=6):
    color_norm_e = colors.Normalize(vmin=0, vmax=n+1)
    scalar_map_e = cmx.ScalarMappable(norm=color_norm_e, cmap='Greys')
    cmap = [scalar_map_e.to_rgba(i+1) for i in range(0, n+1)][1:]

    return cmap
    
def plot_ecp(reader, ax, layer, offset=0, color='red'):
    data = reader.raw_contrib(layer=layer)
    rate = reader.raw_rate()
    istart, istop = int(round(TSTART/1000.0*rate)), int(round(TSTOP/1000.0*rate))
    ecp = data[istart:istop]
    t = np.linspace(TSTART, TSTOP, len(ecp))
    ax.plot(t-TSTIM, ecp-offset, linewidth=0.5, color=color)

def plot_contribs(nwb, ax):
    layers = [1, 2, 3, 4, 5, 6]
    colors = get_colors()
    offset_per_layer = .15
    reader = LayerReader(nwb=nwb)
    for layer, color in zip(layers, colors):
        offset = layer * offset_per_layer
        plot_ecp(reader, ax, layer, offset=offset, color=color)

def plot_power_spectra_contrib(plotter, ax, _type='contrib'):
    layers = [1, 2, 3, 4, 5, 6]
    colors = get_colors()
    plt.sca(ax)
    y = .95
    label = 'Contribution' if _type == 'contrib' else 'Lesion'
    for layer, color in zip(layers, colors):
        plotter.plot_one_layer_ei(layer, '', _type, color=color)
        ax.text(.95, y, "L{} {}".format(layer, label), fontsize=7, color=color,
                horizontalalignment='right', verticalalignment='top', transform=ax.transAxes)
        y -= 0.07
        print("done layer {}".format(layer))

fig = plt.figure(figsize=(9, 3))
gs = GridSpec(1, 3, figure=fig)

PS_cls = PowerSpectrum if TYPE == 'zscore' else PowerSpectrumRatio
sim_plotter = PS_cls(nwbfile, '', device='ECoG', stim_i='avg', nosave=True,
                            color='red', label='In silico')

# Raw contribs
ax = plt.subplot(gs[0, 0])
plot_contribs(sim_plotter.nwb, ax)
ax.get_yaxis().set_visible(False)
ax.set_xlabel("Time (ms)")
ax.set_title("a", loc='left')

ax = plt.subplot(gs[0, 1])
plot_power_spectra_contrib(sim_plotter, ax, _type='contrib')
ax.set_title("b", loc='left')

ax = plt.subplot(gs[0, 2])
plot_power_spectra_contrib(sim_plotter, ax, _type='lesion')
ax.set_title("c", loc='left')

plt.tight_layout()
plt.savefig('power_spectrum_layers_{}_{}.png'.format(TYPE, JOBNUM))
plt.savefig('power_spectrum_layers_{}_latest.pdf'.format(TYPE))
