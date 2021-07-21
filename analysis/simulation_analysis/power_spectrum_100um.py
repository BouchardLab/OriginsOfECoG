"""Plot of power spectrum in each 100um slice, along with 6 graphs
showing the # of segments in layer i in each slice

"""
import numpy as np
import json
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
import matplotlib.colors as colors
import matplotlib.cm as cmx

from layer_reader import LayerReader
from power_spectrum import PowerSpectrum, PowerSpectrumRatio
from utils import find_slice_ecp_file, get_layer_slice_counts, numerals

TSTART, TSTOP, TSTIM = 2400, 2700, 2500
JOBNUM = '29812500'
TYPE = 'zscore'
# COLOR = 'Greys'
# COLOR = 'gist_rainbow'
COLOR = 'gist_heat'
THICKNESS = 200
NSLICES = 11
nwbfile = find_slice_ecp_file(JOBNUM, thickness=THICKNESS)

if TYPE == 'zscore':
    PS_cls = PowerSpectrum
else:
    PS_cls = PowerSpectrumRatio

def plot_contribs(reader, ax):
    rate = reader.raw_rate()
    istart, istop = int(round(TSTART/1000.0*rate)), int(round(TSTOP/1000.0*rate))
    offset = 0
    for slice_i, color in zip(range(NSLICES), get_colors()):
        offset += .1
        ecp = reader.raw_slice(slice_i, thickness=THICKNESS)[istart:istop]
        t = np.linspace(TSTART, TSTOP, len(ecp))
        ax.plot(t-TSTIM, ecp-offset, linewidth=0.5, color=color)
    
def get_colors(cmap=COLOR, n=NSLICES, skipfirst=4 if COLOR=='Greys' else 1):
    color_norm = colors.Normalize(vmin=0, vmax=n+skipfirst)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
    cmap = [scalar_map.to_rgba(n-(i+skipfirst)) for i in range(n)]
    return cmap

def plot_counts(ax, slice_counts, halfgap=8):
    for (slice_i, count), color in zip(slice_counts.items(), get_colors()):
        top = slice_i * THICKNESS + halfgap
        bottom = (slice_i + 1) * THICKNESS - halfgap
        mid = (top + bottom) / 2.0
        ax.plot([count, count], [top, bottom], color=color)
        # if count > 0:
        #     ax.text(count, mid, str(count), rotation=90, fontsize=4,
        #             horizontalalignment='right', verticalalignment='center')
        
fig = plt.figure(figsize=(11, 4))
gs = GridSpec(1, 20, figure=fig)

# Raw traces
ax = plt.subplot(gs[0, 0:3])
ax.get_yaxis().set_visible(False)
ax.set_xticks([-100, 0, 100, 200])
ax.set_xlabel("Time (ms)")
reader = LayerReader(nwbfile=nwbfile)
plot_contribs(reader, ax)

# Power spectra
plotter = PS_cls(nwbfile, '', device='ECoG', nosave=True)
ax = plt.subplot(gs[0, 4:10])
plt.sca(ax)
# plt.title("Power spectra, {}um slices".format(THICKNESS))
for slice_i, color in zip(range(NSLICES), get_colors()):
    plotter.plot_one_slice(slice_i, 'contrib', color=color)

layer_slice_counts = get_layer_slice_counts(JOBNUM, thickness=THICKNESS)
print(json.dumps(layer_slice_counts, indent=4, sort_keys=True))
layers = [1, 2, 3, 4, 5, 6]

# Num layer segments in each slice
for layer, slice_counts in layer_slice_counts.items():
    ax = plt.subplot(gs[0, 11+layer-1:11+layer])
    ax.set_title(numerals[layer])
    plt.sca(ax)
    ax.set_ylim([2100, 0])
    if layer in (1, 2):
        xlim = 14000 if THICKNESS == 100 else 25000
        ax.set_xlim([0, xlim])
        plt.xticks([xlim], rotation='vertical')
    else:
        xlim = 135000 if THICKNESS == 100 else 265000
        ax.set_xlim([0, xlim])
        plt.xticks([xlim], rotation='vertical')
        
    if layer == 1:
        ax.set_ylabel("Depth (um)")
    else:
        ax.get_yaxis().set_visible(False)
    plot_counts(ax, slice_counts)
# depth axis on L1

# Num total segments in each slice
ax = plt.subplot(gs[0, 17])
plt.sca(ax)
ax.set_title("Total")
ax.set_ylim([NSLICES-0.5, -0.5])
ax.get_yaxis().set_visible(False)
# ax.get_xaxis().set_visible(False)
xlim = 200000 if THICKNESS == 100 else 350000
ax.set_xlim([0, xlim])
plt.xticks([xlim], rotation='vertical')
# for spine in ['left', 'right', 'top', 'bottom']:
#     ax.spines[spine].set_visible(False)
total_counts = [sum(layer_slice_counts[layer][slice_i] for layer in layers) for slice_i in range(NSLICES)]
print(total_counts)
ax.barh(
    range(NSLICES),
    total_counts,
    color=get_colors(),
    height=0.8
)

# Fraction of segments in each layer
subgs = GridSpecFromSubplotSpec(NSLICES, 1, subplot_spec=gs[0, 18:20])
for slice_i, color in zip(range(NSLICES), get_colors()):
    ax = plt.subplot(subgs[slice_i, 0])
    ax.axis('off')
    plt.bar(layers, [layer_slice_counts[l][slice_i] for l in layers], width=0.7, color=color)
# Label layers on bottom subplot only
ax = plt.subplot(subgs[-1, 0])
ax.axis('on')
ax.get_yaxis().set_visible(False)
for spine in ['left', 'right', 'top', 'bottom']:
    ax.spines[spine].set_visible(False)
ax.set_xticks(layers)
ax.set_xticklabels([numerals[l] for l in layers])

col = '' if COLOR == 'Greys' else ('_'+COLOR)
plt.tight_layout()
plt.savefig('power_spectrum_{}um_{}_{}{}.pdf'.format(THICKNESS, JOBNUM, TYPE, col))
plt.savefig('power_spectrum_100um_latest.pdf'.format(THICKNESS, JOBNUM, TYPE, col))
