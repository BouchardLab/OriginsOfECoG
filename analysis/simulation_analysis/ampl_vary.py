"""
Plots of thalamic amplitude variations
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import matplotlib.cm as cmx

from utils import find_layer_ei_ecp_file
from power_spectrum import PowerSpectrum, PowerSpectrumRatio

JOBNUMS = (
    (20, '30103194'),
    (23, '30103187'),
    (26, '30034900'),
    (29, '30034918'),
    (32, '30034921'),
    (35, '29812500'),
    (38, '30034922'),
    (44, '30034925'),
)

def get_colors(cmap='Reds', n=len(JOBNUMS)):
    color_norm = colors.Normalize(vmin=0, vmax=n+1)
    scalar_map = cmx.ScalarMappable(norm=color_norm, cmap=cmap)
    cmap = [scalar_map.to_rgba(i+1) for i in range(n)]
    return cmap

def plot(ps_ax, peak_ax, PS_cls=PowerSpectrum):
    for color, (thal_freq, jobnum) in zip(get_colors(), JOBNUMS):
        nwbfile = find_layer_ei_ecp_file(jobnum)
        plt.sca(ps_ax)
        plotter = PS_cls(nwbfile, '', nosave=True, color=color)
        f, spectrum, errs = plotter.plot_one(0)
        max_f = f[np.argmax(spectrum)]
        max_resp = np.max(spectrum)
        alpha = (thal_freq-18.0)/26.0
        peak_ax.scatter(max_resp, max_f, c=color, #alpha=alpha,
                        marker='s', s=48, edgecolors='none')
        if thal_freq == 35:
            peak_ax.plot(max_resp, max_f, 'ro', fillstyle='none', markersize=16, alpha=alpha)

if __name__ == '__main__':
    fig, axs = plt.subplots(2, 2, figsize=(6, 6))
    
    axs[0, 0].set_xlabel('Neural freq (Hz)')
    axs[0, 0].set_ylabel('Z-score')
    axs[0, 1].set_xlabel('Resp. magnitude (Z-score)')
    axs[0, 1].set_ylabel('Resp. peak freq (Hz)')
    plot(axs[0, 0], axs[0, 1], PS_cls=PowerSpectrum)

    axs[1, 0].set_xlabel('Neural freq (Hz)')
    axs[1, 0].set_ylabel('Stim/bl ratio')
    axs[1, 1].set_xlabel('Resp. magnitude (ratio)')
    axs[1, 1].set_ylabel('Resp. peak freq (Hz)')
    plot(axs[1, 0], axs[1, 1], PS_cls=PowerSpectrumRatio)

    plt.tight_layout()
    plt.savefig("ampl_vary.pdf")
